"""Mixture-of-Agents runtime helpers for /moa turns.

The slash command is deliberately not a model tool. It marks one user turn as
MoA-enabled; the normal Hermes agent loop still owns tool calling and turn
termination, while this module gathers reference-model context before each model
iteration.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agent.auxiliary_client import call_llm
from agent.transports import get_transport

logger = logging.getLogger(__name__)

# Upper bound on concurrent reference-model calls. References are independent
# advisory calls (no tools, no inter-dependence), so we fan them out the same
# way delegate_task runs a batch: all in flight at once, results collected when
# every reference finishes. Presets rarely list more than a handful of
# references; this cap just protects against a pathologically large preset
# opening dozens of sockets at once.
_MAX_REFERENCE_WORKERS = 8


class _RefAccounting:
    """Per-reference token usage + estimated cost + full trace, carried as the
    third slot of a reference-output tuple.

    Kept as a tiny object (not a bare CanonicalUsage) because an advisor may
    run on a different model/provider than the aggregator, so its cost MUST be
    priced at its OWN model's rate — folding advisor tokens into the
    aggregator's usage and pricing the sum at the aggregator's rate would
    misprice every advisor. ``usage`` feeds accurate token counts;
    ``cost_usd`` feeds accurate cost.

    ``messages`` / ``output`` / ``model`` / ``provider`` / ``temperature``
    carry the FULL reference input and output for trace persistence (the
    display ``text`` is a truncated preview and is not enough to audit what an
    advisor actually saw). They are only populated when tracing is on; they add
    negligible cost otherwise.
    """

    __slots__ = (
        "usage",
        "cost_usd",
        "cost_status",
        "cost_source",
        "messages",
        "output",
        "model",
        "provider",
        "temperature",
    )

    def __init__(
        self,
        usage: Any,
        cost_usd: Any = None,
        cost_status: str | None = None,
        cost_source: str | None = None,
        *,
        messages: Any = None,
        output: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        temperature: Any = None,
    ):
        self.usage = usage
        self.cost_usd = cost_usd
        self.cost_status = cost_status
        self.cost_source = cost_source
        self.messages = messages
        self.output = output
        self.model = model
        self.provider = provider
        self.temperature = temperature

# Per-tool-result character budget for the advisory reference view. Tool
# results can be huge (a full diff, a 5000-line file dump); replaying them
# verbatim per reference per tool-loop step would blow the reference model's
# context window and cost. We keep the agent's *actions* (tool calls) in full —
# they are cheap, high-signal, and tell the reference what the agent did — but
# preview each tool *result* head+tail so the reference still sees what came
# back without replaying megabytes. The acting aggregator always gets the full,
# untrimmed transcript; this budget only shapes the advisory copy.
_REFERENCE_TOOL_RESULT_BUDGET = 4000

# 系统提示词会被附加到每一次参考模型调用的最前面。参考模型是
# 顾问性质的 —— 它们**不**执行操作、不调用工具，也不承担该任务。如果没有这种
# 框架设定，参考模型接收到裸露的、裁剪后的对话后，会误以为自己是实际执行操作的智能体：
# 随后它会拒绝执行（“我无法从这里访问代码仓库 / URL”）或试图调用它并不拥有的工具。
# 该提示词将模型重新定义为一个分析师，其职责是对呈现出来的状态进行推理，
# 并将其最佳的思考结果提交给将要真正执行操作的聚合器/编排器（aggregator/orchestrator）。
#
# --------------------------------------
#
# "你是智能体混合架构（Mixture of Agents，MoA）流程中的参考顾问。你
# # **不是**实际执行操作的智能体，你**不**执行任何操作：你无法调用工具、运行命令、
# # 浏览网页，也无法访问文件、代码仓库或 URL，你不应该尝试这样做，也不需要为无法做到而道歉。
# # 一个独立的聚合器/编排器（aggregator/orchestrator）模型拥有这些能力，并将采取实际行动。\n\n"
# # "下方的对话是该执行智能体当前处理任务的状态。你的职责是对该状态给出你最智能的分析：
# # 理解目标、对问题进行推理，并对下一步该做什么提出建议。指出最佳方法、具体的下一步行动
# # 和工具使用策略、可能存在的陷阱和风险，以及执行智能体可能遗漏或弄错的任何内容。
# # 假设任何被引用的文件、URL 或系统都是存在的，并根据给定的上下文对其进行推理，而不是
# # 请求访问权限。\n\n"
# # "请直接返回你的建议 —— 无需开场白，无需对工具或访问权限进行免责声明。你的响应是
# # 交付给聚合器的私有引导，而不是展示给用户的最终回答。"
_REFERENCE_SYSTEM_PROMPT = (
    "You are a reference advisor in a Mixture of Agents (MoA) process. You are "
    "NOT the acting agent and you do NOT execute anything: you cannot call "
    "tools, run commands, browse, or access files, repositories, or URLs, and "
    "you should not try to or apologize for being unable to. A separate "
    "aggregator/orchestrator model holds those capabilities and will take the "
    "actual actions.\n\n"
    "The conversation below is the current state of a task handled by that "
    "acting agent. Your job is to give your most intelligent analysis of that "
    "state: understand the goal, reason about the problem, and advise on what "
    "to do next. Surface the best approach, concrete next steps and tool-use "
    "strategy, likely pitfalls and risks, and anything the acting agent may "
    "have missed or gotten wrong. Assume any referenced files, URLs, or "
    "systems exist and reason about them from the context given rather than "
    "asking for access.\n\n"
    "Respond with your advice directly — no preamble, no disclaimers about "
    "tools or access. Your response is private guidance handed to the "
    "aggregator, not an answer shown to the user."
)



def _slot_label(slot: dict[str, str]) -> str:
    return f"{(slot.get('provider') or '').strip()}:{(slot.get('model') or '').strip()}"


def _slot_runtime(slot: dict[str, str]) -> dict[str, Any]:
    """将参考/聚合器插槽解析为实际的运行时调用关键字参数（kwargs）。

    一个 MoA 插槽仅仅是一个模型的选择 —— 它必须以与其他地方调用任何模型相同的
    方式被调用，而不是通过一个光秃秃的 ``call_llm(provider=..., model=...)``
    来进行，那会导致 base_url/api_key/api_mode 处于未解析状态，并让辅助自动检测器
    去瞎猜。我们通过 ``resolve_runtime_provider``（CLI、网关和 delegate_task
    都在使用的规范 provider→api_mode/base_url/api_key 解析器）来路由该插槽的
    服务商，从而让该插槽获得其服务商真正的 API 表现面 —— 例如 MiniMax → anthropic_messages，
    GPT-5/o 系列 → max_completion_tokens，自定义端点 → 它们的 base_url。

    返回要透传给 ``call_llm`` 的关键字参数（kwargs）（包含 provider/model，以及
    解析出的 base_url/api_key，如果可用的话）。在发生任何解析错误时，会回退到
    仅使用 provider/model 的基础参数，以便配置错误的插槽仍能尝试进行调用，而不是
    中止整个 MoA 回合。
    """
    provider = str(slot.get("provider") or "").strip()
    model = str(slot.get("model") or "").strip()
    out: dict[str, Any] = {"provider": provider, "model": model}
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        rt = resolve_runtime_provider(requested=provider, target_model=model)
        # 无条件地将解析后的端点（endpoint）透传给 call_llm。
        # call_llm 的 _resolve_task_provider_model() 是决定显式 base_url 是将调用
        # 降级（collapse）为通用的 ``custom`` 路由，还是保留服务商真实身份的唯一关卡：
        # 它会为任何一流（first-class）服务商保留身份（通过
        # _preserve_provider_with_base_url，这是一种服务商目录的能力检查），
        # 这样一来，那些添加了认证刷新 / 请求元数据 / 请求形状适配器的服务商分支 ——
        # 如 Anthropic OAuth (Bearer + anthropic-beta)、OpenAI-Codex 响应包裹 + Cloudflare
        # 请求头、XAI-OAuth、Bedrock SigV4 签名、Nous Portal 标签 —— 依然可以正常触发。
        # 这些分支会按名称重新解析它们自己的凭证，并忽略透传过来的 base_url/api_key，
        # 因此即使透传的是一个占位符密钥（如 Bedrock 的 "aws-sdk"），透传操作也是安全的。
        # 我们以前在这里也维护了一个名称保留集；但这造成了关卡的重复，并且逐渐变得不同步，
        # 所以现在该单一事实来源（single source of truth）已移至 call_llm 中。
        if rt.get("base_url"):
            out["base_url"] = rt["base_url"]
        if rt.get("api_key"):
            out["api_key"] = rt["api_key"]
        if rt.get("api_mode"):
            out["api_mode"] = rt["api_mode"]
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("MoA slot runtime resolution failed for %s: %s", _slot_label(slot), exc)
    return out


def _maybe_apply_moa_cache_control(
    messages: list[dict[str, Any]],
    runtime: dict[str, Any],
) -> list[dict[str, Any]]:
    """当顾问模型或聚合器模型的请求路由支持时，用 cache_control 对其进行装饰。

    该函数复用了与主智能体循环相同的策略函数（``anthropic_prompt_cache_policy``），
    并根据插槽自身的 provider/base_url/api_mode/model 进行解析，
    同时也采用了相同的断点布局（``apply_anthropic_cache_control``，即 system_and_3 策略）。
    这使得顾问模型和聚合器模型的调用装饰方式，与该服务商上实际执行的智能体完全一致 ——
    从而避免了因引入特定于 MoA 的缓存逻辑而导致设计偏离。

    在发生任何解析错误，或者当策略判定该路由不支持缓存标记时，
    将原样返回未做修改的消息列表。
    """
    try:
        from types import SimpleNamespace

        from agent.agent_runtime_helpers import anthropic_prompt_cache_policy
        from agent.prompt_caching import apply_anthropic_cache_control

        # The policy function reads agent.* only as fallbacks for kwargs we
        # don't pass; provide a stub so the slot is judged purely on its own
        # resolved runtime.
        stub = SimpleNamespace(provider="", base_url="", api_mode="", model="")
        should_cache, native_layout = anthropic_prompt_cache_policy(
            stub,
            provider=runtime.get("provider") or "",
            base_url=runtime.get("base_url") or "",
            api_mode=runtime.get("api_mode") or "",
            model=runtime.get("model") or "",
        )
        if not should_cache:
            return messages
        return apply_anthropic_cache_control(
            messages, native_anthropic=native_layout
        )
    except Exception as exc:  # pragma: no cover - decoration must never break a call
        logger.debug("MoA cache_control decoration skipped: %s", exc)
        return messages


def _run_reference(
    slot: dict[str, str],
    ref_messages: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, str, Any]:
    """调用一个参考模型并返回 ``(label, text, usage)``。

    该插槽将被解析为其服务商的实际运行环境（通过 ``_slot_runtime``），
    并通过与任何模型都使用的相同的 ``call_llm`` 请求构建路径进行调用，
    因此，特定于模型的网线格式处理（anthropic_messages、max_completion_tokens、
    固定/禁用的温度参数）将完全同样地适用于参考模型，就像该模型是实际执行模型时一样。
    MoA 本身不施加任何限制（``max_tokens`` 默认为 ``None`` → 忽略 → 模型自身的
    实际最大值）；``temperature`` 仅为用户配置的预设值，call_llm 仍可能针对每个模型
    对其进行覆盖。

    参考模型的 Token 使用量会根据该插槽自身解析出的服务商/api_mode 进行规范化
    （顾问模型可能会在与聚合器不同的服务商上运行，并具有不同的使用量网线格式），
    并作为一个 ``CanonicalUsage`` 返回，以便调用者可以将顾问模型的开销合并到会话账目中。
    如果不进行此处理，整个参考模型分发（通常占据了 MoA 回合中 Token 开销的大部分）
    对于成本追踪来说是不可见的，因为成本追踪以往只能看到聚合器的使用量。

    绝不抛出异常：失败的参考模型会变成一条带标签的笔记，以便聚合器依然可以利用部分
    上下文进行操作。该函数设计在线程池内部运行 —— 由于 ``call_llm`` 是同步/阻塞的，
    因此线程（而非 asyncio）是正确的并发原语，这与 ``delegate_task`` 的批量分发相呼应。
    """
    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost, normalize_usage

    label = _slot_label(slot)
    runtime = _slot_runtime(slot)
    try:
        # 在最前面添加顾问角色（advisory-role）系统提示词，以便参考模型明白
        # 它正在为聚合器分析状态，而不是在执行该任务。裁剪后的视图
        # （_reference_messages）已经剥离了智能体自身的系统提示词，
        # 因此这是参考模型看到的唯一系统消息。
        messages = [{"role": "system", "content": _REFERENCE_SYSTEM_PROMPT}, *ref_messages]
        # 应用与主智能体循环相同的 Anthropic 风格的提示词缓存装饰（system_and_3 断点）。
        # 顾问视图在多次迭代之间是“仅追加”的（新回合会被追加在结尾的合成标记之前），
        # 因此在遵循缓存规则的路由上（如通过 OpenRouter/原生的 Claude、MiniMax、通义千问/DashScope），
        # 第 N+1 次迭代的前缀会重放第 N 次迭代已缓存的前缀。如果没有这个处理，
        # 整个基准测试运行下来，Claude 顾问模型的缓存读取命中次数为零（经测算：在 1227 次调用中
        # 命中 0 次，导致 1150 万个输入 Token 被重复计费），因为 Anthropic 的缓存机制在
        # 每次请求中是需要主动选择开启（opt-in）的。OpenAI 家族的顾问模型则不受影响
        # （它们的缓存是自动进行的；这些标记会被无害地忽略，但我们只有在策略判定该路由
        # 遵循这些标记时才会进行装饰）。
        messages = _maybe_apply_moa_cache_control(messages, runtime)
        response = call_llm(
            task="moa_reference",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **runtime,
        )
        usage = CanonicalUsage()
        raw_usage = getattr(response, "usage", None)
        if raw_usage:
            try:
                usage = normalize_usage(
                    raw_usage,
                    provider=runtime.get("provider"),
                    api_mode=runtime.get("api_mode"),
                )
            except Exception:  # pragma: no cover - defensive
                usage = CanonicalUsage()
        # Price this advisor at ITS OWN model/provider rate (with correct
        # cache-read/cache-write split), not the aggregator's. This is why
        # advisor cost is summed as dollars rather than by folding tokens into
        # the aggregator's usage.
        cost_usd = None
        cost_status = None
        cost_source = None
        try:
            cost = estimate_usage_cost(
                slot.get("model") or "",
                usage,
                provider=runtime.get("provider"),
                base_url=runtime.get("base_url"),
                api_key=runtime.get("api_key"),
            )
            cost_usd = cost.amount_usd
            cost_status = cost.status
            cost_source = cost.source
        except Exception:  # pragma: no cover - defensive
            pass
        _output_text = _extract_text(response) or "(empty response)"
        acct = _RefAccounting(
            usage,
            cost_usd,
            cost_status,
            cost_source,
            messages=messages,
            output=_output_text,
            model=slot.get("model"),
            provider=runtime.get("provider") or slot.get("provider"),
            temperature=temperature,
        )
        return label, _output_text, acct
    except Exception as exc:
        logger.warning("MoA reference model %s failed: %s", label, exc)
        return label, f"[failed: {exc}]", _RefAccounting(
            CanonicalUsage(),
            messages=[{"role": "system", "content": _REFERENCE_SYSTEM_PROMPT}, *ref_messages],
            output=f"[failed: {exc}]",
            model=slot.get("model"),
            provider=runtime.get("provider") or slot.get("provider"),
            temperature=temperature,
        )


def _run_references_parallel(
    reference_models: list[dict[str, str]],
    ref_messages: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> list[tuple[str, str, Any]]:
    """并行分发（Fan out）所有参考模型，并按顺序返回输出结果。

    类似于 ``delegate_task`` 的批处理模式，所有参考模型都会被同时派遣，
    我们会一直阻塞，直到它们全部运行结束，然后将拼接好的结果交给聚合器。
    输出顺序与 ``reference_models`` 保持一致，以确保 ``Reference {idx}``
    的标签保持稳定。在此处会跳过引用了另一个 MoA 预设的 MoA 预设（递归防御），
    并附带一个标明原因的笔记。

    每个元素均为 ``(label, text, usage)`` 结构，其中 usage 是一个
    ``CanonicalUsage``（对于被跳过或失败的参考模型，该值清零）。
    """
    from agent.usage_pricing import CanonicalUsage

    if not reference_models:
        return []

    results: list[tuple[str, str, Any] | None] = [None] * len(reference_models)
    futures = {}
    workers = min(_MAX_REFERENCE_WORKERS, len(reference_models))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idx, slot in enumerate(reference_models):
            if slot.get("provider") == "moa":
                results[idx] = (
                    _slot_label(slot),
                    "[skipped: MoA presets cannot recursively reference MoA]",
                    _RefAccounting(CanonicalUsage()),
                )
                continue
            futures[
                executor.submit(
                    _run_reference,
                    slot,
                    ref_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            ] = idx
        # Collect every reference before returning — the aggregator needs the
        # complete set, so there is no early-exit / first-completed path here.
        for future, idx in futures.items():
            results[idx] = future.result()

    return [r for r in results if r is not None]


def _truncate_tool_result(text: str, budget: int = _REFERENCE_TOOL_RESULT_BUDGET) -> str:
    """用于顾问视图的工具结果头部+尾部预览。

    保留预算内前半部分和后半部分的内容，并在它们之间放置一个 ``[... 漏掉了 N 个字符 ...]``
    的标记，以便参考模型既能看到结果是如何开始的，也能看到它是如何结束的，而无需重放整个有载荷。
    """
    if not text or len(text) <= budget:
        return text
    half = budget // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n[... {omitted} chars omitted ...]\n{text[-half:]}"


def _render_tool_calls(tool_calls: Any) -> str:
    """将助手回合的 tool_calls 渲染为可读的文本行。

    顾问视图无法携带真实的 ``tool_calls`` 有载荷（严格的服务商会拒绝参考模型从未生成过的
    tool_calls），因此智能体的操作会被扁平化为参考模型可以阅读并进行推理的文本。
    """
    lines: list[str] = []
    for tc in tool_calls or []:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        name = fn.get("name") or (tc.get("name") if isinstance(tc, dict) else "") or "tool"
        args = fn.get("arguments")
        if isinstance(args, str):
            args_text = args
        elif args is not None:
            try:
                import json

                args_text = json.dumps(args, ensure_ascii=False)
            except Exception:
                args_text = str(args)
        else:
            args_text = ""
        lines.append(f"[called tool: {name}({args_text})]" if args_text else f"[called tool: {name}]")
    return "\n".join(lines)


_ADVISORY_INSTRUCTION = (
    "[The conversation above is the current state of the task. Give your "
    "most intelligent judgement: what is going on, what should happen next, "
    "what risks or mistakes you see, and how the acting agent should "
    "proceed.]"
)


def _reference_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # """为参考模型构建对话的顾问视图。
    #
    # 参考模型需要对当前状态做出“知情”的判断，因此它必须看到智能体实际上做了什么 ——
    # 它的工具调用以及返回的工具结果 —— 而不仅仅是智能体的自述。因此，我们保留了整个
    # 对话流程，但将其扁平化（flatten）为干净的用户/助手*文本*回合：
    #
    #   - 系统提示词：丢弃（8K 字节的 Hermes 模版，并非顾问信号）。
    #   - 助手回合：保留；任何 ``tool_calls`` 都会作为内联的 ``[called tool: name(args)]``
    #     文本行附加到该回合的文本末尾。
    #   - ``tool`` 角色的结果：不丢弃。每一个结果都会被折叠（头部+尾部预览，
    #     参见 ``_truncate_tool_result``）并作为 ``[tool result: ...]`` 块合并到*前一个*
    #     助手回合中，以便参考模型能够看到返回的内容。
    #
    # 这样只会输出纯文本的用户/助手回合，而包含零个 ``tool`` 角色的消息和零个 ``tool_calls``
    # 数组 —— 从而让那些会因孤立的工具消息 / 未生成的 tool_calls 而返回 400 错误的
    # 严格服务商（如 Mistral、Fireworks）不会报错，同时参考模型依然能掌握全局。
    #
    # 该视图必须以 ``user`` 回合结束。Anthropic（以及 OpenRouter→Anthropic）会将结尾的
    # 助手回合解释为助手用于继续输出的*预填（prefill）*内容，而有些不支持预填的模型
    # （例如 Claude Opus 4.8）会因 ``400 ... must end with a user message`` 拒绝请求。
    # 与其为了满足该要求而删除智能体最新的上下文（这会导致参考模型对当前状态一无所知），
    # 我们选择追加一个合成的用户回合，请求参考模型对上述状态做出评判。这样既满足了
    # “以用户回合结束”的要求，又没有丢失任何上下文。
    #
    # 实际执行的聚合器（acting aggregator）总是接收完整、未裁剪的转录记录；本函数仅对
    # 临时可丢弃的顾问副本进行格式整形。
    # """
    rendered: list[dict[str, Any]] = []
    last_user_content: str | None = None
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        text = content if isinstance(content, str) else ""

        if role == "system":
            continue
        if role == "user":
            if text.strip():
                last_user_content = text
            rendered.append({"role": "user", "content": text})
        elif role == "assistant":
            parts: list[str] = []
            if text.strip():
                parts.append(text.strip())
            calls_text = _render_tool_calls(msg.get("tool_calls"))
            if calls_text:
                parts.append(calls_text)
            # Empty assistant turns (no text, no calls) carry nothing advisory.
            if parts:
                rendered.append({"role": "assistant", "content": "\n".join(parts)})
        elif role == "tool":
            # 将工具结果作为文本折叠进前一个助手回合中，以便参考模型能够看到返回的内容，
            # 同时避免触发一个参考模型从未生成过的工具角色（tool-role）消息。
            result_text = _truncate_tool_result(text)
            block = f"[tool result: {result_text}]"
            if rendered and rendered[-1].get("role") == "assistant":
                rendered[-1]["content"] = rendered[-1]["content"] + "\n" + block
            else:
                # No assistant turn to attach to (e.g. a leading tool result);
                # keep it as advisory context on its own assistant-role line.
                rendered.append({"role": "assistant", "content": block})
        # Any other role is ignored.

    # End on a user turn: append a synthetic advisory request rather than
    # deleting the agent's latest assistant context. This satisfies Anthropic's
    # no-trailing-assistant-prefill rule while preserving full state.
    if rendered and rendered[-1].get("role") == "assistant":
        rendered.append({"role": "user", "content": _ADVISORY_INSTRUCTION})
    elif rendered and rendered[-1].get("role") == "user":
        # Already ends on a user turn (fresh user prompt, no agent action yet).
        # Leave it — the reference answers that prompt directly.
        pass

    if not rendered:
        # Degenerate case: nothing rendered. Fall back to the latest user turn.
        if last_user_content is not None:
            return [{"role": "user", "content": last_user_content}]
        for msg in reversed(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return [{"role": "user", "content": msg["content"]}]
    return rendered



def _extract_text(response: Any) -> str:
    try:
        transport = get_transport("chat_completions")
        if transport is None:
            raise RuntimeError("chat_completions transport unavailable")
        normalized = transport.normalize_response(response)
        text = (normalized.content or "").strip()
        if text:
            return text
    except Exception:
        pass
    try:
        message = response.choices[0].message
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", message)
        if not isinstance(content, str):
            content = str(content) if content else ""
        return content.strip()
    except Exception:
        return ""


def _preset_temperature(preset: dict[str, Any], key: str) -> float | None:
    """Read an optional temperature from a preset.

    Returns None when the key is absent, empty, or explicitly null — meaning
    "don't send temperature; let the provider default apply", exactly like a
    single-model Hermes agent (which never sends temperature unless
    configured). The old coercion ``float(preset.get(key, 0.6) or 0.6)``
    made unset impossible: absent, null, and even 0 all collapsed to the
    hardcoded default, so MoA advisors/aggregator always ran at 0.6/0.4
    while the same model running solo used the provider default.
    """
    value = preset.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("ignoring non-numeric %s=%r in MoA preset", key, value)
        return None


def aggregate_moa_context(
    *,
    user_prompt: str,
    api_messages: list[dict[str, Any]],
    reference_models: list[dict[str, str]],
    aggregator: dict[str, str],
    temperature: float | None = None,
    aggregator_temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    # 运行配置好的参考模型并合成它们的建议。
    #
    # 失败信息会以特定于模型的笔记形式返回，而不是中止正常的智能体循环；
    # 主模型仍然可以利用部分上下文进行操作。
    #
    # ``max_tokens`` 默认值为 ``None``：混合模型架构（MoA）不会限制参考模型
    # 或聚合模型的输出，因此每个模型都使用其自身的上限。当 ``max_tokens`` 为
    # ``None`` 时，``call_llm`` 会完全忽略该参数（参见其文档字符串），这同时也
    # 规避了那些完全拒绝 ``max_tokens`` 参数的服务商。此前在此处硬编码的上限
    # 会截断较长的聚合模型合成结果。
    #
    # ``temperature`` / ``aggregator_temperature`` 默认值为 ``None``：
    # 类似于 max_tokens，当其为 None 时，``call_llm`` 会忽略温度参数，从而应用
    # 服务商的默认设置 —— 这与单模型智能体的行为保持一致。预设（Presets）可能
    # 仍会固定显式的值。
    reference_outputs: list[tuple[str, str, Any]] = []
    ref_messages = _reference_messages(api_messages)
    reference_outputs = _run_references_parallel(
        reference_models,
        ref_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    joined = "\n\n".join(
        f"Reference {idx} — {label}:\n{text}"
        for idx, (label, text, _usage) in enumerate(reference_outputs, start=1)
    )
    # "你是智能体混合架构（Mixture of Agents）流程中的聚合器（aggregator）。请将
    # 参考响应合成简明、可操作的指南，供 Hermes 主智能体使用。重点关注下一步行动、
    # 工具使用策略、风险以及任何分歧。除非只需要这样做，否则不要直接回答用户；
    # 请生成主智能体在其正常循环中应使用的上下文。\n\n"
    # f"原始用户提示词：\n{user_prompt}\n\n"
    # f"参考响应：\n{joined}"
    synth_prompt = (
        "You are the aggregator in a Mixture of Agents process. Synthesize the "
        "reference responses into concise, actionable guidance for the main "
        "Hermes agent. Focus on next steps, tool-use strategy, risks, and any "
        "disagreements. Do not answer the user directly unless that is all that "
        "is needed; produce context the main agent should use in its normal loop.\n\n"
        f"Original user prompt:\n{user_prompt}\n\n"
        f"Reference responses:\n{joined}"
    )

    agg_label = _slot_label(aggregator)
    agg_runtime = _slot_runtime(aggregator)
    try:
        # 采用与 _run_reference 的顾问（advisor）调用相同的 cache_control 装饰
        # （参见 _maybe_apply_moa_cache_control） —— 此合成调用是
        # 第三个独立的 MoA 调用路径，提交 22c5048d9 并未涵盖该路径（该提交
        # 仅恢复了持久化 `provider: moa` 模型中执行聚合器回合以及顾问分发的缓存）。
        # 如果没有它，单次（one-shot） `/moa <prompt>` 命令的合成调用在每次调用时
        # 都会对完整的输入（包含每个拼接好的参考输出的无系统提示词）重新计费，
        # 且没有任何 cache_control 断点，即使解析出的聚合器插槽是一个
        # 遵循缓存规则的路由（例如 OpenRouter 上的 Claude 或原生 Anthropic）。
        agg_messages = _maybe_apply_moa_cache_control(
            [{"role": "user", "content": synth_prompt}], agg_runtime
        )
        response = call_llm(
            task="moa_aggregator",
            messages=agg_messages,
            temperature=aggregator_temperature,
            max_tokens=max_tokens,
            **agg_runtime,
        )
        synthesis = _extract_text(response)
    except Exception as exc:
        logger.warning("MoA aggregator model %s failed: %s", agg_label, exc)
        synthesis = ""

    if not synthesis:
        synthesis = joined

        # "[智能体混合架构（Mixture of Agents）上下文 — 请将此作为 Hermes 智能体
        # 正常循环的私有引导。你可以调用工具、继续推理，或正常结束。]\n"
        # f"聚合器: {agg_label}\n"
        # f"参考模型: {', '.join(_slot_label(slot) for slot in reference_models)}\n\n"
        # f"{synthesis.strip()}"
    return (
        "[Mixture of Agents context — use this as private guidance for the "
        "normal Hermes agent loop. You may call tools, continue reasoning, or "
        "finish normally.]\n"
        f"Aggregator: {agg_label}\n"
        f"References: {', '.join(_slot_label(slot) for slot in reference_models)}\n\n"
        f"{synthesis.strip()}"
    )


def _attach_reference_guidance(agg_messages: list[dict[str, Any]], guidance: str) -> None:
    """Attach the per-turn reference block at the END of the aggregator prompt.

    The reference text differs on every tool-loop iteration. In an agentic loop
    the most recent ``user`` message is the *original task* sitting near the TOP
    of the context (everything after it is assistant/tool turns), so merging the
    turn-varying reference block into it diverges the prompt prefix early — the
    server's KV cache cannot be reused and the entire conversation re-prefills on
    every step (full prefill each tool call, dominating latency on long contexts).

    Appending at the very end keeps the ``[system][task][tool-history]`` prefix
    stable and cache-reusable (only the new block re-prefills), and gives the
    aggregator the references with recency. Merge into the last message only when
    it is already a trailing string ``user`` turn (plain chat — still at the end).
    """
    last = agg_messages[-1] if agg_messages else None
    if last is not None and last.get("role") == "user" and isinstance(last.get("content"), str):
        last["content"] = last["content"] + "\n\n" + guidance
    else:
        agg_messages.append({"role": "user", "content": guidance})


class MoAChatCompletions:
    """OpenAI-chat-compatible facade where the aggregator is the acting model."""

    def __init__(self, preset_name: str, reference_callback: Any = None):
        self.preset_name = preset_name or "default"
        # Optional display hook. Called as reference outputs become available so
        # frontends can show each reference model's answer as a labelled block
        # before the aggregator acts. Signature:
        #   reference_callback(event, **kwargs)
        # where event is one of:
        #   "moa.reference"   kwargs: index, count, label, text
        #   "moa.aggregating" kwargs: aggregator (label), ref_count
        # Never raises into the model call — display is best-effort.
        self.reference_callback = reference_callback
        # State-scoped reference cache. The agent loop calls create() once per
        # tool-loop iteration; references should re-run whenever the task STATE
        # advances — i.e. on every new user message AND every new tool result —
        # so each reference judges the latest state. The advisory view
        # (_reference_messages) now renders tool calls + results as text, so its
        # signature changes on every new tool response; the cache key is that
        # signature, so a new tool result is a cache MISS (references re-run)
        # while a redundant create() call with identical state is a HIT (no
        # re-run, no re-emit). This gives "fire on every user/tool response"
        # for free, without re-firing on a pure no-op re-call.
        self._ref_cache_key: tuple | None = None
        self._ref_cache_outputs: list[tuple[str, str, Any]] = []
        # Token usage + estimated cost of the reference fan-out from the most
        # recent cache-MISS create() call, awaiting consumption by session
        # accounting. Set on every create() (zeroed on a cache HIT so per-turn
        # advisor spend is counted exactly once). Consumed via
        # ``consume_reference_usage``.
        from agent.usage_pricing import CanonicalUsage

        self._pending_reference_usage: Any = CanonicalUsage()
        self._pending_reference_cost: Any = None
        # Resolved aggregator slot ({provider, model, ...}) from the most recent
        # create(); read by session cost accounting to price the aggregator's
        # acting turn at its real model instead of the virtual preset name.
        self.last_aggregator_slot: Any = None
        # Full-turn trace parts stashed on a cache-MISS create(), awaiting the
        # caller to stitch in the live session_id + resolved aggregator output
        # and flush to the trace file (only when moa.save_traces is on).
        self._pending_trace: Any = None

    def consume_reference_usage(self) -> tuple[Any, Any]:
        """Pop pending reference-fan-out usage + cost, resetting both to empty.

        Returns ``(CanonicalUsage, cost_usd_or_None)`` for the most recent
        ``create()`` and clears the pending values, so a subsequent read (e.g.
        a streaming retry re-entering accounting) cannot double-count. Usage is
        always a ``CanonicalUsage`` (zeroed if none); cost is a summed-dollars
        float or ``None`` when no advisor could be priced.
        """
        from agent.usage_pricing import CanonicalUsage

        usage = self._pending_reference_usage or CanonicalUsage()
        cost = self._pending_reference_cost
        self._pending_reference_usage = CanonicalUsage()
        self._pending_reference_cost = None
        return usage, cost

    def consume_and_save_trace(
        self, session_id: Any = None, aggregator_output_fallback: Any = None
    ) -> None:
        """Flush the pending full-turn trace to disk, if one is pending.

        No-op when tracing is off (``save_moa_turn`` checks the config), when
        there is no pending trace (a cache-HIT iteration ran no references), or
        when the aggregator input was never recorded. Clears the pending trace
        so a repeat consume cannot double-write. Best-effort — never raises.

        ``aggregator_output_fallback`` is the aggregator's resolved acting text
        as the caller already holds it in memory (the streamed assistant text).
        On the streaming path the aggregator's output could not be captured
        inline at ``create()`` time (the raw token stream was handed to the live
        consumer), so ``pending["aggregator_output"]`` is None; we fold the
        caller's resolved text in here so the trace is self-contained in BOTH
        streaming and non-streaming modes. Non-streaming already has the inline
        output and ignores the fallback.
        """
        pending = self._pending_trace
        self._pending_trace = None
        if not pending or "aggregator_input_messages" not in pending:
            return
        try:
            from agent.moa_trace import save_moa_turn

            agg_slot = pending.get("aggregator_slot") or {}
            # Prefer the inline capture (non-streaming); fall back to the
            # caller's resolved streamed text when streaming left it None.
            agg_output = pending.get("aggregator_output")
            if agg_output is None and aggregator_output_fallback:
                agg_output = aggregator_output_fallback
            save_moa_turn(
                session_id=session_id,
                preset_name=pending.get("preset", ""),
                reference_outputs=pending.get("reference_outputs", []),
                aggregator_label=pending.get("aggregator_label", ""),
                aggregator_model=agg_slot.get("model"),
                aggregator_provider=agg_slot.get("provider"),
                aggregator_temperature=pending.get("aggregator_temperature"),
                aggregator_input_messages=pending.get("aggregator_input_messages"),
                aggregator_output=agg_output,
                aggregator_streamed=bool(pending.get("aggregator_streamed")),
            )
        except Exception as exc:  # pragma: no cover - tracing must never break a turn
            logger.debug("MoA trace flush failed: %s", exc)

    def _emit(self, event: str, **kwargs: Any) -> None:
        cb = self.reference_callback
        if cb is None:
            return
        try:
            cb(event, **kwargs)
        except Exception as exc:  # pragma: no cover - display must never break the turn
            logger.debug("MoA reference_callback failed for %s: %s", event, exc)

    def create(self, **api_kwargs: Any) -> Any:
        from hermes_cli.config import load_config
        from hermes_cli.moa_config import resolve_moa_preset

        preset = resolve_moa_preset(load_config().get("moa") or {}, self.preset_name)
        messages = list(api_kwargs.get("messages") or [])
        reference_models = preset.get("reference_models") or []
        aggregator = preset.get("aggregator") or {}
        # Expose the resolved aggregator slot so session cost accounting can
        # price the aggregator's acting turn at its REAL model/provider. The
        # agent's model/provider on the MoA path are the virtual preset name
        # ("closed") and "moa", which have no pricing entry — without this the
        # aggregator's spend (often the bulk of the turn) is silently dropped
        # and the session cost reflects advisor fan-out only.
        self.last_aggregator_slot = dict(aggregator) if aggregator else None
        # By default MoA does not cap reference or aggregator output: each model
        # uses its own maximum (max_tokens=None → call_llm omits the parameter,
        # so a long aggregator synthesis is never truncated and providers that
        # reject max_tokens don't 400). A preset MAY set reference_max_tokens to
        # cap ADVISOR output only — advisor generation is the dominant MoA
        # latency (turn latency correlates ~0.88 with output tokens), and the
        # aggregator only needs the gist of each advisor's judgement, so a cap
        # (e.g. 600) measurably cuts per-turn wall time (~44% on a sample task).
        # The acting aggregator is never capped here (its output is the
        # user-visible answer).
        reference_max_tokens = preset.get("reference_max_tokens")
        # None (the default) = don't send temperature; provider default
        # applies, matching single-model agent behavior. Presets may pin
        # explicit values. See _preset_temperature.
        temperature = _preset_temperature(preset, "reference_temperature")
        aggregator_temperature = _preset_temperature(preset, "aggregator_temperature")
        if aggregator_temperature is None and api_kwargs.get("temperature") is not None:
            # The acting agent's own configured temperature (if any) still
            # applies to the aggregator, which IS the acting model.
            aggregator_temperature = api_kwargs.get("temperature")

        # When the preset is disabled, skip the reference fan-out and let the
        # configured aggregator act alone — it is the preset's acting model, so
        # a disabled MoA preset is simply "use the aggregator directly."
        if not preset.get("enabled", True):
            reference_models = []

        from agent.usage_pricing import CanonicalUsage

        reference_outputs: list[tuple[str, str, Any]] = []
        ref_messages = _reference_messages(messages)

        # Fan-out cadence. "per_iteration" (default): advisors re-run whenever
        # the advisory view changes — i.e. every tool iteration, since the
        # view grows with each tool result. "user_turn": advisors run ONCE per
        # user turn; subsequent tool iterations reuse that turn's advice and
        # the aggregator acts alone (the original MoA shape: synthesize at the
        # start, then let the acting model work). Implemented by hashing only
        # the prefix up to the LAST USER message so mid-turn growth doesn't
        # change the signature — iteration 2+ becomes a cache HIT.
        fanout_mode = str(preset.get("fanout") or "per_iteration").strip().lower()
        sig_messages = ref_messages
        if fanout_mode == "user_turn":
            # Find the last REAL user message. The advisory view appends a
            # synthetic user marker (_ADVISORY_INSTRUCTION) when it ends on an
            # assistant turn — i.e. on every tool iteration after the first —
            # so that marker must not count as a user turn or the prefix
            # would include the grown mid-turn context and the signature
            # would change every iteration (defeating the once-per-turn
            # cadence entirely).
            last_user_idx = None
            for _i in range(len(ref_messages) - 1, -1, -1):
                _m = ref_messages[_i]
                if _m.get("role") == "user" and _m.get("content") != _ADVISORY_INSTRUCTION:
                    last_user_idx = _i
                    break
            if last_user_idx is not None:
                sig_messages = ref_messages[: last_user_idx + 1]

        # Turn-scoped cache: only run + display references when the advisory
        # view changed (i.e. a new user turn). Within one turn the agent loop
        # calls create() once per tool iteration; in user_turn mode the
        # signature is stable across those iterations (prefix hash above), so
        # the fan-out runs once per user turn and iterations reuse the advice.
        _sig = hashlib.sha256(
            "\u0000".join(
                f"{m.get('role')}:{m.get('content')}" for m in sig_messages
            ).encode("utf-8", "replace")
        ).hexdigest()
        _cache_key = (self.preset_name, _sig, tuple(_slot_label(s) for s in reference_models))
        _refs_from_cache = _cache_key == self._ref_cache_key and bool(self._ref_cache_outputs)

        if _refs_from_cache:
            reference_outputs = list(self._ref_cache_outputs)
            # References already ran (and were accounted) earlier this turn;
            # this create() is a repeat tool-iteration reusing the cached
            # advice. Charging their tokens/cost again here would multiply
            # advisor spend by the tool-iteration count, so pending is zero.
            self._pending_reference_usage = CanonicalUsage()
            self._pending_reference_cost = None
            # Likewise no trace on a cache HIT — the full turn was already
            # traced on the MISS that ran the references. A repeat iteration is
            # not a new MoA turn.
            self._pending_trace = None
        else:
            reference_outputs = _run_references_parallel(
                reference_models,
                ref_messages,
                temperature=temperature,
                max_tokens=reference_max_tokens,
            )
            self._ref_cache_key = _cache_key
            self._ref_cache_outputs = list(reference_outputs)
            # Sum the advisor fan-out's token usage AND cost so the caller can
            # fold advisor spend into session accounting exactly once per turn.
            # Only the freshly run references (cache MISS) contribute; a cache
            # HIT above zeroes this. Token counts sum directly (each already
            # normalized per-advisor provider/api_mode); cost sums in dollars
            # because each advisor was priced at its OWN model rate — advisors
            # may be cheaper/pricier than the aggregator, so their tokens must
            # NOT be repriced at the aggregator's rate.
            _ref_usage = CanonicalUsage()
            _ref_cost: Any = None
            for _lbl, _txt, _acct in reference_outputs:
                if isinstance(_acct, _RefAccounting):
                    if isinstance(_acct.usage, CanonicalUsage):
                        _ref_usage = _ref_usage + _acct.usage
                    if _acct.cost_usd is not None:
                        _ref_cost = (_ref_cost or 0) + _acct.cost_usd
            self._pending_reference_usage = _ref_usage
            self._pending_reference_cost = _ref_cost
            # Stash the full reference fan-out for trace persistence. The
            # aggregator input/label are filled in below once agg_messages is
            # built; the aggregator OUTPUT is stitched in by the caller
            # (consume_and_save_trace) once the response resolves — the caller
            # holds the live session_id and the resolved aggregator response.
            self._pending_trace = {
                "preset": self.preset_name,
                "reference_outputs": list(reference_outputs),
                "aggregator_slot": aggregator,
                "aggregator_temperature": aggregator_temperature,
            }

            # Surface each reference model's answer to the display BEFORE the
            # aggregator acts — once per turn (only on the iteration that
            # actually ran them). The user sees one labelled block per
            # reference (rendered like a thinking block) so the MoA process is
            # visible rather than a silent pause. Best-effort: never blocks the
            # turn.
            _ref_count = len(reference_outputs)
            for _idx, (_label, _text, _usage) in enumerate(reference_outputs, start=1):
                self._emit(
                    "moa.reference",
                    index=_idx,
                    count=_ref_count,
                    label=_label,
                    text=_text,
                )
            if _ref_count:
                self._emit(
                    "moa.aggregating",
                    aggregator=_slot_label(aggregator),
                    ref_count=_ref_count,
                )

        agg_messages = [dict(m) for m in messages]
        if reference_outputs:
            joined = "\n\n".join(
                f"Reference {idx} — {label}:\n{text}"
                for idx, (label, text, _usage) in enumerate(reference_outputs, start=1)
            )
            guidance = (
                "[Mixture of Agents reference context]\n"
                f"Preset: {self.preset_name}\n"
                f"Aggregator/acting model: {_slot_label(aggregator)}\n"
                f"References: {', '.join(label for label, _, _ in reference_outputs)}\n\n"
                "Use the reference responses below as private context. You are the aggregator and acting model: "
                "answer the user directly or call tools as needed.\n\n"
                f"{joined}"
            )
            _attach_reference_guidance(agg_messages, guidance)

        if aggregator.get("provider") == "moa":
            raise RuntimeError("MoA aggregator cannot be another MoA preset")
        agg_kwargs = dict(api_kwargs)
        agg_kwargs["messages"] = agg_messages
        # Record the exact aggregator INPUT (incl. the injected reference
        # context) into the pending trace so a trace captures what the
        # aggregator actually saw, not a reconstruction.
        if self._pending_trace is not None:
            self._pending_trace["aggregator_input_messages"] = agg_messages
            self._pending_trace["aggregator_label"] = _slot_label(aggregator)
        # The aggregator is the acting model. Resolve its slot to the provider's
        # real runtime (base_url/api_key/api_mode) and call it through the same
        # request-building path any model uses — so per-model wire-format
        # handling (anthropic_messages, max_completion_tokens, fixed/forbidden
        # temperature) applies identically to it. MoA imposes no output cap:
        # max_tokens is passed through from the caller (normally None → omitted
        # → the model's real maximum). The preset's old hardcoded 4096 default
        # is gone — it truncated long syntheses.
        # When the agent's streaming consumer calls us with stream=True, run the
        # references first (above) and then return the aggregator's RAW token
        # stream so the acting model's output reaches the user live. The consumer
        # reassembles chunks + tool_calls, runs stale-stream detection, and falls
        # back to a non-streaming retry on error. The non-streaming path
        # (stream=False) is unchanged — no stream/stream_options/timeout are
        # forwarded, so its behavior is byte-for-byte identical to before.
        stream = bool(api_kwargs.get("stream"))
        stream_kwargs: dict[str, Any] = {}
        if stream:
            stream_kwargs["stream"] = True
            stream_kwargs["stream_options"] = (
                api_kwargs.get("stream_options") or {"include_usage": True}
            )
            # Forward the consumer's per-request (stream read) timeout so it
            # actually governs the aggregator stream, not just call_llm's default.
            if api_kwargs.get("timeout") is not None:
                stream_kwargs["timeout"] = api_kwargs["timeout"]
        _agg_response = call_llm(
            task="moa_aggregator",
            messages=agg_messages,
            temperature=aggregator_temperature,
            max_tokens=agg_kwargs.get("max_tokens"),
            tools=agg_kwargs.get("tools"),
            extra_body=agg_kwargs.get("extra_body"),
            **stream_kwargs,
            **_slot_runtime(aggregator),
        )
        # Non-streaming path (quiet mode / eval / subagents): the aggregator
        # output is available inline, so capture it into the pending trace now.
        # Streaming path: the aggregator's raw token stream is returned to the
        # consumer live and its acting output lands as the turn's assistant
        # message; the trace marks it streamed and points there.
        if self._pending_trace is not None:
            if stream:
                self._pending_trace["aggregator_streamed"] = True
                self._pending_trace["aggregator_output"] = None
            else:
                self._pending_trace["aggregator_streamed"] = False
                try:
                    self._pending_trace["aggregator_output"] = _extract_text(_agg_response)
                except Exception:  # pragma: no cover - defensive
                    self._pending_trace["aggregator_output"] = None
        return _agg_response


class MoAClient:
    def __init__(self, preset_name: str, reference_callback: Any = None):
        self.chat = type("_MoAChat", (), {})()
        self.chat.completions = MoAChatCompletions(preset_name, reference_callback=reference_callback)

    def consume_reference_usage(self) -> Any:
        """Pop the pending reference-fan-out usage from the completions facade.

        Lets session accounting fold the MoA advisor tokens into the turn's
        usage without reaching into ``.chat.completions`` internals.
        """
        return self.chat.completions.consume_reference_usage()

    @property
    def last_aggregator_slot(self) -> Any:
        """Resolved aggregator slot ({provider, model, ...}) from the most
        recent create(), or None. Read by session cost accounting to price the
        aggregator's acting turn at its real model instead of the virtual
        preset name."""
        return getattr(self.chat.completions, "last_aggregator_slot", None)

    def consume_and_save_trace(
        self, session_id: Any = None, aggregator_output_fallback: Any = None
    ) -> None:
        """Flush the pending full-turn MoA trace via the completions facade.

        No-op unless ``moa.save_traces`` is enabled and a turn is pending.
        ``aggregator_output_fallback`` supplies the resolved acting text so the
        streaming path's trace is self-contained (see the facade docstring).
        """
        return self.chat.completions.consume_and_save_trace(
            session_id, aggregator_output_fallback=aggregator_output_fallback
        )
