"""Mixture-of-Agents runtime helpers for /moa turns.

The slash command is deliberately not a model tool. It marks one user turn as
MoA-enabled; the normal Hermes agent loop still owns tool calling and turn
termination, while this module gathers reference-model context before each model
iteration.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait as _futures_wait
from types import SimpleNamespace
from typing import Any

from agent.auxiliary_client import call_llm
from agent.message_content import flatten_message_text
from agent.transports import get_transport

logger = logging.getLogger(__name__)

# --- MoA privacy filter (config: moa.privacy_filter — '' | display | full) ---
#
# Advisor (reference) outputs can echo PII from the conversation — emails,
# phone numbers, credentials pasted by the user — into surfaces the user may
# not expect: the labelled reference blocks rendered in the UI, saved MoA
# trace files, and (in `full` mode) the guidance block injected into the
# aggregator prompt (issue #59959). Secret/credential shapes (API-key
# prefixes, JWTs, private keys, DB connection strings, E.164 phone numbers)
# are handled by the repo's central redactor, ``agent.redact
# .redact_sensitive_text`` — the MoA filter never re-implements those. The
# two patterns below cover the PII classes the central redactor deliberately
# leaves alone for log/tool output (emails and formatted phone numbers).
#
# Pattern safety: advisory text is frequently code-review-shaped — line
# numbers, timestamps, git SHAs, IDs, IP addresses. A bare 10-digit match
# would mangle all of those, so the phone pattern requires clearly delimited
# formatting: a parenthesized area code and/or explicit `-`/`.` separators
# between groups ((555) 123-4567, 555-123-4567, 555.123.4567, +1 555-123-4567).
# Undelimited digit runs (5551234567), dates (2026-07-12), times (12:34:56),
# hex IDs, and dotted quads never match. International numbers in E.164 form
# (+14155551234) are already masked by the central redactor.
_MOA_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_MOA_PHONE_RE = re.compile(
    r"(?<![\w.+-])"                    # no leading word char / dot / + / - (kills IPs, IDs, versions)
    r"(?:\+?1[ .-])?"                  # optional NA country code
    r"(?:\(\d{3}\)[ .-]?|\d{3}[.-])"   # delimited area code: (555) or 555- / 555.
    r"\d{3}[.-]\d{4}"                  # exchange-subscriber with explicit separator
    r"(?![\w-])"                       # no trailing word char / hyphen
)


def _redact_reference_text(text: Any) -> Any:
    """Redact secrets + PII from one advisor/reference text surface.

    Centralized secret shapes first (force=True: the MoA privacy filter is
    its own explicit opt-in, independent of the global log-redaction toggle;
    code_file=True: advisory text is prose/code, so the ENV/JSON assignment
    heuristics that mangle source snippets stay off), then the MoA-specific
    email/formatted-phone patterns. Non-string inputs pass through unchanged.
    """
    if not isinstance(text, str) or not text:
        return text
    from agent.redact import redact_sensitive_text

    text = redact_sensitive_text(text, force=True, code_file=True)
    text = _MOA_EMAIL_RE.sub("[redacted email]", text)
    text = _MOA_PHONE_RE.sub("[redacted phone]", text)
    return text


def _moa_privacy_mode(moa_raw: Any) -> str:
    """Resolve the normalized privacy-filter mode from a raw ``moa`` config."""
    from hermes_cli.moa_config import coerce_privacy_filter

    raw = moa_raw if isinstance(moa_raw, dict) else {}
    return coerce_privacy_filter(raw.get("privacy_filter"))


def _redact_reference_outputs(
    reference_outputs: list[tuple[str, str, Any]],
) -> list[tuple[str, str, Any]]:
    """Return reference-output tuples with their advisor text redacted.

    The ``_RefAccounting`` third slot is left as-is — accounting fields carry
    no advisor text; the full-output/input trace fields are redacted
    separately at trace-stash time (see create()) so the LIVE cache keeps raw
    accounting objects untouched.
    """
    return [
        (label, _redact_reference_text(text), acct)
        for label, text, acct in reference_outputs
    ]


def _redact_trace_messages(messages: Any) -> Any:
    """Redact message copies destined for trace persistence.

    Handles both string content and structured content-part lists (e.g.
    cache_control-decorated text parts). Unknown shapes pass through.
    """
    if not isinstance(messages, list):
        return messages
    out: list[Any] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        content = m.get("content")
        if isinstance(content, str):
            out.append({**m, "content": _redact_reference_text(content)})
        elif isinstance(content, list):
            out.append(
                {
                    **m,
                    "content": [
                        {**p, "text": _redact_reference_text(p.get("text"))}
                        if isinstance(p, dict) and isinstance(p.get("text"), str)
                        else p
                        for p in content
                    ],
                }
            )
        else:
            out.append(m)
    return out


def _redact_trace_accounting(acct: Any) -> Any:
    """Return a copy of a ``_RefAccounting`` with its trace text redacted.

    Traces persist the advisor's FULL input messages and output to disk, so a
    privacy-filtered run must not write raw PII there. Usage/cost fields are
    copied verbatim (numbers, no text). Non-accounting objects pass through.
    """
    if not isinstance(acct, _RefAccounting):
        return acct
    return _RefAccounting(
        acct.usage,
        acct.cost_usd,
        acct.cost_status,
        acct.cost_source,
        messages=_redact_trace_messages(acct.messages),
        output=_redact_reference_text(acct.output),
        model=acct.model,
        provider=acct.provider,
        temperature=acct.temperature,
    )


# Cold-start caches. A MoA preset switch used to re-resolve the full
# config + preset + every slot's provider runtime on EACH create() call
# (once per tool-loop iteration), serially before the parallel fan-out could
# start — adding 5-30s of "frozen" latency on complex presets
# (#66793). The preset structure is immutable for the life of a turn, so
# cache both the resolved preset and each (provider, model) runtime.
_preset_cache_lock = threading.Lock()
_preset_cache: dict[tuple, Any] = {}

_runtime_cache_lock = threading.Lock()
_runtime_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}

# Runtime entries go stale when providers/credentials change (key rotation,
# base_url edits). Deliberately short-lived: 300s collapses the per-iteration
# re-resolution inside a turn while bounding credential staleness between
# turns — the non-MoA path picks up rotated keys immediately, this path
# within 5 minutes.
_RUNTIME_CACHE_TTL_SECONDS = 300.0

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
# _REFERENCE_SYSTEM_PROMPT = (
#     "你是 Agent 混合架构（Mixture of Agents, MoA）流程中的参考顾问（reference advisor）。\n"
#     "你**不是**执行 Agent，也不执行任何操作：\n"
#     "你无法调用工具、运行命令、浏览网页，或访问文件、仓库及 URL，\n"
#     "你不应该尝试这样做，也不必为无法做到而道歉。\n"
#     "一个独立的聚合器/编排器（aggregator/orchestrator）模型具备这些能力，并会采取实际行动。\n\n"
#     "绝对关键：你绝不能声称或暗示自己执行了命令、下载了文件、访问了 URL 或执行了任何操作。\n"
#     "你只能基于对话上下文进行分析并提供建议。以下是需要避免和推荐使用的表达示例：\n"
#     "- 错误示例：\"我运行了 curl 得到了 404。\"\n"
#     "- 错误示例：\"我已成功下载该文件。\"\n"
#     "- 错误示例：\"我检查了仓库，发现……\"\n"
#     "- 正确示例：\"根据错误模式，向该 URL 发起 curl 请求很可能会返回 404。\"\n"
#     "- 正确示例：\"对话记录表明，下载此文件可能会有所帮助。\"\n"
#     "- 正确示例：\"根据上下文，检查仓库将会揭示……\"\n\n"
#     "下方的对话是该执行 Agent 当前处理任务的最新状态。\n"
#     "你的职责是对该状态提供你最明智的分析：理解目标、推演问题，并就下一步操作提供建议。\n"
#     "指出最佳方法、具体的下一步行动与工具使用策略、潜在的陷阱与风险，\n"
#     "以及执行 Agent 可能遗漏或弄错的任何事项。\n"
#     "请假定提及的所有文件、URL 或系统均已存在，并基于给定的上下文进行推演，而不是请求获取访问权限。\n\n"
#     "请直接输出你的建议 —— 无需开场白，无需声明关于工具或访问权限的免责声明。\n"
#     "你的回复是提交给聚合器的内部指导意见，不会展示给最终用户。\n"
#     "切记：绝不能声称自己执行了任何操作。"
# )
_REFERENCE_SYSTEM_PROMPT = (
    "You are a reference advisor in a Mixture of Agents (MoA) process. You are "
    "NOT the acting agent and you do NOT execute anything: you cannot call "
    "tools, run commands, browse, or access files, repositories, or URLs, and "
    "you should not try to or apologize for being unable to. A separate "
    "aggregator/orchestrator model holds those capabilities and will take the "
    "actual actions.\n\n"
    "CRITICAL: You must NEVER claim or imply that you have executed a command, "
    "downloaded a file, accessed a URL, or performed any action. You can only "
    "analyze and advise based on the conversation context. Examples of what to "
    "avoid:\n"
    "- Bad: \"I ran curl and got 404.\"\n"
    "- Bad: \"I downloaded the file successfully.\"\n"
    "- Bad: \"I checked the repository and found...\"\n"
    "- Good: \"Based on the error pattern, a curl request to that URL would likely return 404.\"\n"
    "- Good: \"The conversation suggests downloading this file may help.\"\n"
    "- Good: \"From the context, checking the repository would reveal...\"\n\n"
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
    "aggregator, not an answer shown to the user. NEVER claim to have executed "
    "anything."
)



def _slot_label(slot: dict[str, Any]) -> str:
    label = f"{(slot.get('provider') or '').strip()}:{(slot.get('model') or '').strip()}"
    effort = str(slot.get("reasoning_effort") or "").strip()
    return f"{label}[reasoning={effort}]" if effort else label


def _slot_reasoning_config(slot: dict[str, Any]) -> dict[str, Any] | None:
    """Translate optional per-MoA-slot reasoning_effort into runtime config."""
    effort = slot.get("reasoning_effort")
    try:
        from hermes_constants import parse_reasoning_effort

        return parse_reasoning_effort(effort)
    except Exception:  # pragma: no cover - defensive; bad config must not break MoA
        return None


def _aggregator_reasoning_config(aggregator: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve the aggregator's reasoning config: slot > per-model > global.

    The aggregator is MoA's ACTING model, so when its slot doesn't pin a
    reasoning_effort it must resolve exactly like any other acting model:
    through the shared chokepoint (``resolve_reasoning_config``), which
    applies ``agent.reasoning_overrides`` for the slot's model first, then
    the global ``agent.reasoning_effort``. Without this the main loop's
    reasoning gates (keyed to the virtual ``moa://local`` identity) never
    fire, so the aggregator silently ran at the backend default (#64187).

    Reference advisors intentionally do NOT get this fallback: they are side
    calls (like auxiliary tasks), and inheriting a global ``xhigh`` into every
    advisor fan-out would silently multiply cost. Their depth is slot-or-
    provider-default only.
    """
    cfg = _slot_reasoning_config(aggregator)
    if cfg is not None:
        return cfg
    try:
        from hermes_cli.config import load_config
        from hermes_constants import resolve_reasoning_config

        return resolve_reasoning_config(
            load_config() or {}, str(aggregator.get("model") or "")
        )
    except Exception:  # pragma: no cover - defensive; bad config must not break MoA
        return None


def _slot_runtime(slot: dict[str, Any]) -> dict[str, Any]:
    """Resolve a reference/aggregator slot to real runtime call kwargs.

    A MoA slot is just a model selection — it must be called the same way any
    model is called elsewhere, not through a bare ``call_llm(provider=...,
    model=...)`` that leaves base_url/api_key/api_mode unresolved and lets the
    auxiliary auto-detector guess. We route the slot's provider through
    ``resolve_runtime_provider`` (the canonical provider→api_mode/base_url/
    api_key resolver the CLI, gateway, and delegate_task all use), so the slot
    gets its provider's real API surface — e.g. MiniMax → anthropic_messages,
    GPT-5/o-series → max_completion_tokens, custom endpoints → their base_url.
    Returns the kwargs to pass through to ``call_llm`` (provider/model plus the
    resolved base_url/api_key when available). Falls back to the bare
    provider/model on any resolution error so a misconfigured slot still
    attempts the call rather than aborting the whole MoA turn.

    The resolved runtime is cached per (provider, model) with a short TTL
    (``_RUNTIME_CACHE_TTL_SECONDS``): the resolution does real I/O (catalog
    query + config read) that used to run serially per create() call before
    the parallel fan-out could start — the dominant source of MoA cold-start
    latency (#66793). The TTL bounds credential staleness (key rotation,
    base_url edits) instead of caching for the process lifetime.
    """
    provider = str(slot.get("provider") or "").strip()
    model = str(slot.get("model") or "").strip()
    cache_key = (provider, model)
    now = time.monotonic()
    with _runtime_cache_lock:
        entry = _runtime_cache.get(cache_key)
    if entry is not None:
        stamped_at, cached = entry
        if now - stamped_at < _RUNTIME_CACHE_TTL_SECONDS:
            return cached
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
        request_overrides = rt.get("request_overrides")
        if isinstance(request_overrides, dict):
            extra_body = request_overrides.get("extra_body")
            if isinstance(extra_body, dict) and extra_body:
                out["extra_body"] = dict(extra_body)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("MoA slot runtime resolution failed for %s: %s",
                     _slot_label(slot), exc)
        # Never cache a fallback-shaped result: a transient resolution error
        # (config mid-write, catalog hiccup) would otherwise pin the bare
        # provider/model kwargs for a full TTL.
        return out
    with _runtime_cache_lock:
        _runtime_cache[cache_key] = (now, out)
    return out


def _merge_slot_extra_body(
    slot_extra_body: Any,
    caller_extra_body: Any,
) -> Any:
    """Merge slot defaults with a caller override for ``call_llm``."""
    if isinstance(slot_extra_body, dict) and slot_extra_body:
        if isinstance(caller_extra_body, dict):
            return {**slot_extra_body, **caller_extra_body}
        if caller_extra_body:
            return caller_extra_body
        return dict(slot_extra_body)
    return caller_extra_body


def _maybe_apply_moa_cache_control(
    messages: list[dict[str, Any]],
    runtime: dict[str, Any],
    *,
    cache_disabled: bool | None = None,
) -> list[dict[str, Any]]:
    """当顾问（advisor）或聚合器（aggregator）请求的路由支持缓存时，
    为其添加 cache_control 装饰。

    复用与主 Agent 循环相同的策略函数
    （``anthropic_prompt_cache_policy``），
    该函数针对插槽（slot）自身的 provider/base_url/api_mode/model
    以及共享的标记辅助函数（``apply_anthropic_cache_control``）进行解析。
    MoA 没有针对单次会话的静态前缀，
    因此它使用辅助函数的“传统系统提示词加前 3 条消息（legacy system-and-3）”回退方案，
    而无需维护单独的缓存策略。

    当发生任何解析错误，或者策略表明该路由不支持标记时，
    将原封不动地返回消息。

    将 ``cache_disabled``（省略时则使用实时配置）标记到策略存根（policy stub）上，
    这样 ``prompt_caching.cache_ttl: off`` 就不会被空 Agent 模式（blank-agent pattern）所绕过（#76085）。
    """
    try:
        from agent.agent_runtime_helpers import (
            anthropic_prompt_cache_policy,
            blank_cache_policy_stub,
        )
        from agent.prompt_caching import apply_anthropic_cache_control

        # Prefer an explicit kwarg, then a snapshot on the runtime dict
        # (threaded from the live agent), else config via the stub factory.
        if cache_disabled is None and "_cache_disabled" in runtime:
            cache_disabled = runtime.get("_cache_disabled")

        # The policy function reads agent.* only as fallbacks for kwargs we
        # don't pass; blank_cache_policy_stub is the only sanctioned stub
        # so _cache_disabled cannot be left off again (#76085).
        stub = blank_cache_policy_stub(cache_disabled)
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
    slot: dict[str, Any],
    ref_messages: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reference_timeout: float | None = None,
    context_length_cache: Any = None,
    cache_disabled: bool | None = None,
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
        # 裁剪内容以适应“当前参考模型”的上下文窗口。
        #
        # 参考模型拥有的窗口可能比聚合器更小
        # （例如，一个 262K 上下文的 kimi-k2.7-code 正在为 1M 上下文的 glm-5.2 提供建议）；
        # 如果不进行裁剪，提供商将直接返回 HTTP 400 错误，
        # 而下方的 except 语句会将其静默转换为 [failed: …] 提示（问题 #60345）。
        #
        # 估算将在预置顾问系统提示词（advisory system prompt）之后进行，
        # 因此系统提示词的 Token 也会计入预算指标中。
        messages = _trim_messages_for_reference(
            messages,
            slot,
            runtime,
            reserve_output_tokens=max_tokens,
            context_length_cache=context_length_cache,
        )
        # 应用主 Agent 循环所使用的 Anthropic 风格提示词缓存装饰（prompt-caching decoration）。
        #
        # 该固定的参考提示词（reference prompt）没有特定于会话的前缀划分，
        # 因此辅助函数会使用其传统的“系统提示词加前 3 条消息”回退方案。
        #
        # 在多次迭代过程中，顾问视图（advisory view）仅进行追加操作
        # （新轮次会追加在末尾的合成标记之前），
        # 因此在支持缓存的路由（如通过 OpenRouter/原生调用的 Claude、MiniMax、通义千问/DashScope）上，
        # 迭代 N+1 的前缀会复用迭代 N 已缓存的前缀。
        #
        # 如果不进行此设置，由于 Anthropic 的缓存机制是针对每个请求单点选择开启（opt-in）的，
        # 导致在整个基准测试运行期间，Claude 顾问的缓存读取次数为零
        # （实际测量：0/1227 次调用，重复计费了 1150 万输入 Token）。
        #
        # OpenAI 系列的顾问保持不变
        # （它们的缓存是自动进行的；虽然标记会被安全地忽略，但我们仅在策略明确指定该路由支持时才进行装饰）。
        #
        # 将实时的 Agent 禁用设置绑定（Pin）到运行时上，
        # 以便顾问装饰可以追踪对话状态，而不是重新读取一份全新的配置（#76085）。
        cache_runtime = runtime
        if cache_disabled is not None:
            cache_runtime = {**runtime, "_cache_disabled": cache_disabled}
        messages = _maybe_apply_moa_cache_control(messages, cache_runtime)
        # Per-slot max_tokens takes precedence over the preset-level
        # reference_max_tokens passed in by the caller. This lets each
        # reference model have its own output cap independently.
        _slot_max_tokens: int | None = slot.get("max_tokens")
        _effective_max_tokens = _slot_max_tokens if _slot_max_tokens is not None else max_tokens
        extra_headers = None
        # Normalize provider aliases (github, github-copilot, github-models,
        # ...) through the auxiliary client's canonical alias table so slot
        # configs that spell Copilot differently still get the header.
        from agent.auxiliary_client import _normalize_aux_provider

        if _normalize_aux_provider(str(runtime.get("provider") or "")) in (
            "copilot",
            "copilot-acp",
        ):
            # Copilot Pro/Pro+ gates some premium chat models on request
            # attribution. The main agent marks the first API request of a
            # user turn as ``x-initiator: user``; MoA reference fan-out is also
            # directly serving the user's current turn, not a background agent
            # task, so mirror that header here. Without it, Claude/Gemini
            # Copilot advisors can be rejected as unavailable to the
            # ``copilot-language-server`` integrator even though standalone
            # Copilot calls work.
            extra_headers = {"x-initiator": "user"}
        response = call_llm(
            task="moa_reference",
            messages=messages,
            temperature=temperature,
            max_tokens=_effective_max_tokens,
            timeout=reference_timeout,
            reasoning_config=_slot_reasoning_config(slot),
            extra_headers=extra_headers,
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


# Output-token headroom reserved inside the reference's context window when
# the preset does not cap advisor output (reference_max_tokens=None). Roughly
# one long-form advisory answer; generous enough for thinking models' visible
# output without starving the input budget.
_REFERENCE_DEFAULT_OUTPUT_RESERVE = 8192

# Additional estimation slack: estimate_messages_tokens_rough is a rough
# chars/4 heuristic and providers tokenize less favorably on code/JSON-heavy
# transcripts, so keep a safety fraction of the window unbudgeted.
_REFERENCE_TRIM_SAFETY_FRACTION = 0.10


def _trim_messages_for_reference(
    messages: list[dict[str, Any]],
    slot: dict[str, str],
    runtime: dict[str, Any],
    *,
    reserve_output_tokens: int | None = None,
    context_length_cache: Any = None,
) -> list[dict[str, Any]]:
    """Trim an advisory request to fit within a reference model's context window.

    Reference models may have a smaller context window than the aggregator or
    the main conversation. Without this trim, a reference whose window is
    exceeded gets a hard HTTP 400 from the provider, which ``_run_reference``'s
    try/except silently converts to a ``[failed: …]`` note — the MoA turn
    silently degrades to fewer references (issue #60345).

    ``messages`` is the FULL request as it will be sent — the advisory system
    prompt already prepended — so the estimate covers everything the provider
    will count. The budget reserves ``reserve_output_tokens`` (the preset's
    ``reference_max_tokens`` when set, else a sane constant) for the model's
    response plus a safety fraction for estimator error.

    Trimming drops the OLDEST conversation frames (right after the system
    prompt) and preserves two invariants of the advisory view, which is
    text-only user/assistant turns (``_reference_messages`` renders tool
    calls/results inline, so there are no tool-result frames to orphan):

      - the system prompt (index 0) is always kept;
      - the first non-system message stays ``user``-first — after each pop,
        any now-leading assistant turns are popped too, so no provider ever
        sees an assistant-first conversation;
      - the trailing user turn (the synthetic judge-the-state marker) and at
        least one preceding turn are always kept, even if still over budget —
        a too-long-but-recent view beats an empty request.

    ``context_length_cache`` is an optional per-turn dict keyed by
    ``(provider, model)`` so one fan-out (and every iteration reusing the
    cache) resolves each model's window at most once instead of re-probing
    metadata sources per-reference-per-iteration. When the window cannot be
    resolved, messages are returned unchanged.
    """
    if not messages:
        return messages

    from agent.model_metadata import (
        estimate_messages_tokens_rough,
        get_model_context_length,
    )

    model = str(slot.get("model") or "")
    provider = str(runtime.get("provider") or slot.get("provider") or "")
    if not model:
        return messages

    cache_key = (provider, model)
    context_length: int | None = None
    if isinstance(context_length_cache, dict) and cache_key in context_length_cache:
        context_length = context_length_cache[cache_key]
    else:
        try:
            context_length = get_model_context_length(
                model=model,
                base_url=str(runtime.get("base_url") or ""),
                api_key=str(runtime.get("api_key") or ""),
                provider=provider,
            )
        except Exception:
            logger.debug(
                "MoA reference context-length resolution failed for %s",
                _slot_label(slot),
            )
            context_length = None
        if isinstance(context_length_cache, dict):
            # Cache failures too (as None) — a flaky metadata source should
            # not be re-probed for every reference of every iteration.
            context_length_cache[cache_key] = context_length

    if not isinstance(context_length, int) or context_length <= 0:
        return messages

    reserve = (
        int(reserve_output_tokens)
        if isinstance(reserve_output_tokens, int) and reserve_output_tokens > 0
        else _REFERENCE_DEFAULT_OUTPUT_RESERVE
    )
    budget = int(context_length * (1.0 - _REFERENCE_TRIM_SAFETY_FRACTION)) - reserve
    if budget <= 0:
        return messages

    estimated = estimate_messages_tokens_rough(messages)
    if estimated <= budget:
        return messages

    has_system = bool(messages) and messages[0].get("role") == "system"
    head = [messages[0]] if has_system else []
    body = list(messages[1:] if has_system else messages)

    # Keep the trailing user turn plus at least one preceding turn.
    while len(body) > 2 and estimate_messages_tokens_rough(head + body) > budget:
        body.pop(0)
        # Preserve the user-first invariant: never leave the advisory
        # conversation starting on an assistant turn after a pop.
        while len(body) > 2 and body[0].get("role") == "assistant":
            body.pop(0)
    # The loop can stop with two frames left where the first is an
    # assistant turn — enforce user-first even then (a lone trailing user
    # turn is a valid request; an assistant-first one is not).
    while len(body) > 1 and body[0].get("role") == "assistant":
        body.pop(0)

    trimmed = head + body
    dropped = len(messages) - len(trimmed)
    if dropped:
        logger.info(
            "MoA reference %s: estimated %d tokens exceeds budget %d "
            "(window %d, output reserve %d); dropped %d oldest message(s).",
            _slot_label(slot),
            estimated,
            budget,
            context_length,
            reserve,
            dropped,
        )
    return trimmed


_REFERENCE_POLL_INTERVAL_S = 5.0

# Sentinel text for a reference slot whose wait was aborted by a user
# interrupt. Shared by _run_references_parallel (which writes it) and the
# facade cache logic (which must never cache it as real advice).
_INTERRUPTED_REFERENCE_NOTE = "[skipped: interrupted by user]"


def _run_references_parallel(
    reference_models: list[dict[str, Any]],
    ref_messages: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    progress_callback: Any = None,
    reference_timeout: float | None = None,
    agent: Any = None,
    late_accounting_sink: Any = None,
) -> list[tuple[str, str, Any]]:
    """并行分发（Fan out）所有参考模型，并按顺序返回输出。

    类似于 ``delegate_task`` 的批处理模式，所有参考模型都会被同时分发，
    且系统会阻塞等待，直到它们全部完成，才将聚合后的结果交给聚合器。
    输出顺序与 ``reference_models`` 保持一致，以确保 ``Reference {idx}`` 标签的稳定性。
    在此处，引用了其他 MoA 预设的 MoA 预设会被跳过（递归防护），并附带一条带标签的说明。

    如果提供了 ``progress_callback``，则每当一个参考模型完成时就会被调用：
    ``progress_callback(refs_done, refs_total, label)``。
    总数与 ``len(reference_models)`` 一致，因此监听器可以渲染类似于
    ``MOA: 2/3 refs done`` 的状态栏进度。
    该回调采用尽力而为（Best-effort）策略 —— 发生的失败会被记录，
    但绝不会中断分发流程（界面展示绝不能阻塞一次对话轮次）。

    每个元素均为 ``(label, text, accounting)``，其中 accounting 是一个
    ``_RefAccounting`` 对象（对于被跳过/失败/被中断的参考模型，该对象值全清为零）。

    当传入 *agent* 参数时，并行分发是可中断的：
    对批处理的等待会被拆分为间隔为 ``_REFERENCE_POLL_INTERVAL_S`` 秒的轮询
    （而不是对每个参考模型进行单次阻塞式的 ``future.result()`` 调用），
    以便用户在轮次中途发起的中断能够中止等待 ——
    这镜像复用了 ``agent.tool_executor`` 应用于其自身并发工具批处理的同一中断检查。
    这不会添加或改变任何针对单个参考模型的*超时设置*
    （超时时间由 ``reference_timeout`` / ``auxiliary.moa_reference.timeout`` 控制，并在其他地方解析）
    —— 它仅仅是允许调用方提前停止等待。
    已经在进行中的参考模型无法被强制终止
    （``call_llm`` 是一个阻塞式的 HTTP 调用，自身没有中断钩子，这与 tool_executor 对不具备中断检查功能的工具所面临的限制相同）；
    被中断的参考模型自身的超时机制仍会独立回收其线程。
    *agent* 为可选参数，默认为 ``None``，可为任何未传入该参数的调用方保留不可中断的阻塞行为。
    """
    from agent.usage_pricing import CanonicalUsage

    if not reference_models:
        return []

    results: list[tuple[str, str, Any] | None] = [None] * len(reference_models)
    futures: dict[Any, int] = {}
    workers = min(_MAX_REFERENCE_WORKERS, len(reference_models))
    # 参考插槽（Reference slots）在裸执行器线程（bare executor threads）上运行，
    # 这些线程启动时包含的是一个空的 contextvars.Context ——
    # 请将父轮次（parent turn）的上下文（包括审批回调以及 Nous Portal 会话标签）
    # 传递（propagate）到每个工作线程中，
    # 以便顾问调用（advisor calls）能够归属到与执行轮次（acting turn）相同的会话中。
    from tools.thread_context import propagate_context_to_thread

    total = len(reference_models)
    completed = 0
    executor = ThreadPoolExecutor(max_workers=workers)
    interrupted = False
    # 由所有参考工作线程（reference worker）共享的单次并行分发（per-fan-out）上下文长度缓存，
    # 从而使重复的 (provider, model) 插槽在每次轮次中只需解析一次窗口，
    # 而不必为每个参考模型重新探测元数据源
    # （字典的 get/set 操作具有 GIL 原子性；在首次调用的竞态条件下极少发生的重复探测也是无害的）。
    _ctx_len_cache: dict[tuple[str, str], int | None] = {}
    cache_disabled = (
        getattr(agent, "_cache_disabled", None) if agent is not None else None
    )
    try:
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
                    propagate_context_to_thread(_run_reference),
                    slot,
                    ref_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reference_timeout=reference_timeout,
                    context_length_cache=_ctx_len_cache,
                    cache_disabled=cache_disabled,
                )
            ] = idx

        # 在返回之前收集所有的参考结果 ——
        # 聚合器（aggregator）需要完整的数据集，
        # 因此除了用户主动中断之外，此处不存在提前退出或“首个完成即返回”的路径。
        #
        # 每当一个参考模型完成时，进度回调（progress callback）就会被触发，
        # 以便前端能够渲染类似于 "MOA: k/n refs done" 的状态。
        pending = set(futures)
        while pending:
            done, pending = _futures_wait(pending, timeout=_REFERENCE_POLL_INTERVAL_S)
            for future in done:
                idx = futures[future]
                results[idx] = future.result()
                completed += 1
                if progress_callback is not None:
                    try:
                        label = _slot_label(reference_models[idx])
                        progress_callback(completed, total, label)
                    except Exception as exc:  # pragma: no cover - display must never break
                        logger.debug("MoA progress_callback failed: %s", exc)
            if not pending:
                break
            if agent is not None and getattr(agent, "_interrupt_requested", False):
                interrupted = True
                break

        if interrupted:
            for future, idx in futures.items():
                if results[idx] is not None:
                    continue
                if future.cancel():
                    # Never dispatched — genuinely nothing was billed.
                    results[idx] = (
                        _slot_label(reference_models[idx]),
                        _INTERRUPTED_REFERENCE_NOTE,
                        _RefAccounting(CanonicalUsage()),
                    )
                elif future.done():
                    # 在中断检查和当前时间点之间已执行完毕 ——
                    # 该调用已成功完成并产生了计费，
                    # 因此保留其真实的输出和核算数据（accounting），
                    # 而不是用占位符将其全清为零。
                    results[idx] = future.result()
                else:
                    # 已经在运行中 —— 无法被强制终止（参见 docstring）；
                    # 保持其现状即可，以免阻塞调用方，
                    # 并记录其输出已被弃用。
                    #
                    # 提供商调用仍然在进行中，且在完成时**必定会**产生计费，
                    # 因此，请将其最终的核算数据（accounting）移交给调用方的接收器（sink），
                    # 而不是静默丢弃。
                    label = _slot_label(reference_models[idx])
                    results[idx] = (
                        label,
                        _INTERRUPTED_REFERENCE_NOTE,
                        _RefAccounting(CanonicalUsage()),
                    )
                    if late_accounting_sink is not None:
                        def _record_late(f: Any, _label: str = label) -> None:
                            try:
                                _lbl, _txt, _acct = f.result()
                            except Exception:  # pragma: no cover - defensive
                                return
                            try:
                                late_accounting_sink(_label, _acct)
                            except Exception:  # pragma: no cover - defensive
                                logger.debug(
                                    "MoA: late accounting sink failed for %s",
                                    _label,
                                )
                        future.add_done_callback(_record_late)
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

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
    """将助手轮次（assistant turn）的 tool_calls 渲染为易读的文本行。

    顾问视图（advisory view）无法包含真实的 ``tool_calls`` 有效载荷
    （严格的提供商会拒绝参考模型从未产生的 tool_calls），
    因此 Agent 的操作会被展平（flattened）为参考模型可以阅读和推演的纯文本。

    同时兼容字典（dict）形状和 ``SimpleNamespace`` 形状的条目
    （且嵌套的 ``function`` 也可以是这两种形状中的任意一种），
    从而使该辅助函数能够统一适用于 OpenAI 风格的传输格式以及 SDK 风格的流拼装响应。

    如果没有这种形状兼容性，源自 SimpleNamespace 的条目会被渲染为
    ``[called tool: tool]``，并静默丢失函数名称。
    """
    lines: list[str] = []
    for tc in tool_calls or []:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            fn_name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            fn_args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
            top_name = tc.get("name")
        else:
            fn = getattr(tc, "function", None)
            fn_name = getattr(fn, "name", None) if fn is not None else None
            fn_args = getattr(fn, "arguments", None) if fn is not None else None
            top_name = getattr(tc, "name", None)
        name = fn_name or top_name or "tool"
        if isinstance(fn_args, str):
            args_text = fn_args
        elif fn_args is not None:
            try:
                import json

                args_text = json.dumps(fn_args, ensure_ascii=False)
            except Exception:
                args_text = str(fn_args)
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
        # 将结构化内容（parts 列表）展平（Flatten）为可见文本。
        # 在以下两种常见情况下，内容会以列表（而不是字符串）的形式传入：
        #   1. Anthropic 提示词缓存装饰：conversation_loop 会在 MoA 门面（facade）之前
        #      运行 apply_anthropic_cache_control，
        #      将字符串内容转换为 [{"type": "text", "text": ..., "cache_control": ...}]。
        #      此处如果仅按字符串读取，会把用户的“整个”提示词展平为空字符串 "" ——
        #      进而导致 Claude 参考模型抛出 HTTP 400 错误（"messages: at least one message is required"），
        #      而容错性较高的模型则会回复 "no user request is present"。
        #   2. 多模态轮次（粘贴图片 → 文本 + image_url parts）以及
        #      多模态工具结果（截图）。
        #
        # flatten_message_text 会提取文本 parts 并跳过图片 parts，
        # 对于字符串则原封不动地返回 ——
        # 这样无论记录（transcript）是否经过装饰，都能生成字节级完全一致的顾问视图
        # （从而保持顾问前缀在多次迭代中的稳定性，以便进行顾问提示词缓存）。
        text = flatten_message_text(content)

        if role == "system":
            continue
        if role == "user":
            if not text.strip() and isinstance(content, list) and content:
                # 结构化内容中没有可提取的文本（例如仅包含图片的轮次）。
                # 如果输出一条空的 user 消息，会被严格的提供商丢弃/拒绝
                # （Anthropic 会因文本块为空而抛出 400 错误 —— 这正是最初“closed”预设失败的原因）；
                # 而如果静默跳过该轮次，又会破坏顾问视图（advisory view）中 user/assistant 的交替结构。
                # 因此使用占位符进行替代，以便参考模型（reference）知道发生了一次非文本的交互轮次。
                # 只有结构化内容才符合此条件 ——
                # 空的或仅包含空白字符的“字符串”轮次不带任何有效信息，会在下方被直接丢弃。
                text = "[user sent non-text content (e.g. an image attachment)]"
            if not text.strip():
                # 真正的空 user 轮次（content="" 或 None）。
                # 它不包含任何咨询价值，且严格的提供商（如 Kimi/Moonshot、ZAI
                # 以及其他强制要求 user 内容非空的提供商）会拒绝该消息并抛出 400 错误：
                # "message ... with role 'user' must not be empty" ——
                # 这与下方 assistant 分支丢弃不含任何 parts 的轮次方式是一致的。
                # 容错性较高的提供商（如 DeepSeek）会接受空轮次，
                # 这也是为什么在渲染视图完全相同的情况下，MoA 的并行分发（fan-out）会在某一个参考模型上失败，
                # 却在另一个参考模型上成功的原因。
                # 顾问视图本身就不是严格交替的（在每个工具循环中都会出现连续的 assistant 轮次），
                # 因此丢弃不含内容的轮次是安全的。
                continue
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
            if msg.get("role") == "user":
                fallback_text = flatten_message_text(msg.get("content"))
                if fallback_text.strip():
                    return [{"role": "user", "content": fallback_text}]
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


def _is_failed_reference(text: str) -> bool:
    """Return whether a reference output is an internal failure/skip sentinel.

    Covers both the ``[failed: …]`` notes produced when a reference call
    raises (which may embed raw provider error text) and the
    ``[skipped: …]`` recursion-guard notes — neither is real advice, so
    neither belongs in the aggregator prompt.
    """
    sentinel = text.lstrip().lower()
    return sentinel.startswith("[failed:") or sentinel.startswith("[skipped:")


def _successful_references(
    reference_outputs: list[tuple[str, str, Any]],
) -> list[tuple[str, str, Any]]:
    """Filter failed advice while preserving each accounting payload."""
    return [output for output in reference_outputs if not _is_failed_reference(output[1])]


def _failed_reference_labels(
    reference_outputs: list[tuple[str, str, Any]],
) -> list[str]:
    return [label for label, text, _accounting in reference_outputs if _is_failed_reference(text)]


def _degraded_notice(failed_labels: list[str], policy: str) -> str:
    if not failed_labels or policy.strip().lower() == "silent":
        return ""
    return f"[Reference models unavailable: {', '.join(failed_labels)}]"


def aggregate_moa_context(
    *,
    user_prompt: str,
    api_messages: list[dict[str, Any]],
    reference_models: list[dict[str, Any]],
    aggregator: dict[str, Any],
    temperature: float | None = None,
    aggregator_temperature: float | None = None,
    reference_max_tokens: int | None = None,
    reference_timeout: float | None = None,
    degraded_reference_policy: str = "loud",
    agent: Any = None,
) -> str:
    """运行已配置的参考模型并对其建议进行综合汇总（synthesize）。

    发生的失败会以特定于模型的提示信息形式返回，而不会中断正常的 Agent 循环；
    主模型仍可以基于部分上下文继续执行操作。

    ``reference_max_tokens`` “仅”适用于参考模型的并行分发（fan-out）过程 ——
    聚合器（aggregator）自身的合成调用绝不会受到限制，因此它始终使用其模型自身的最大 Token 数。
    当该参数为 ``None`` 时，``call_llm`` 会完全忽略该参数（参见其 docstring），
    这同时也避开了那些直接拒绝 ``max_tokens`` 参数的提供商。
    此前，在聚合器调用上硬编码上限曾导致长文本合成被截断（#53580）——
    在此处将 ``reference_max_tokens`` 同时传递给这两个调用，会导致该问题再次静默复现。

    ``temperature`` / ``aggregator_temperature`` 默认均为 ``None``：
    与 ``reference_max_tokens`` 类似，当其为 None 时，``call_llm`` 会忽略温度参数，
    从而采用提供商的默认值 —— 这与单模型 Agent 的行为保持一致。
    预设（Presets）仍可以显式固定具体数值。

    当传入 ``agent`` 参数时，允许在用户发起中断时提前终止参考模型的并行分发过程
    —— 详情参见 ``_run_references_parallel`` 的 docstring。
    """
    reference_models = [slot for slot in reference_models if slot.get("enabled", True)]
    reference_outputs: list[tuple[str, str, Any]] = []
    ref_messages = _reference_messages(api_messages)
    reference_outputs = _run_references_parallel(
        reference_models,
        ref_messages,
        temperature=temperature,
        max_tokens=reference_max_tokens,
        reference_timeout=reference_timeout,
        agent=agent,
    )

    successful_outputs = _successful_references(reference_outputs)
    failed_labels = _failed_reference_labels(reference_outputs)

    # 'full' privacy mode (moa.privacy_filter) also covers this one-shot /moa
    # synthesis path: advisor text is redacted before it reaches the
    # synthesizing aggregator. 'display' does not apply here — this path has
    # no user-visible reference blocks or trace records of its own. Redaction
    # runs on the successful outputs only (failed refs are already filtered
    # into the degraded notice).
    try:
        from hermes_cli.config import load_config as _load_config

        if _moa_privacy_mode((_load_config() or {}).get("moa")) == "full":
            successful_outputs = _redact_reference_outputs(successful_outputs)
    except Exception:  # pragma: no cover - privacy filter must never break a turn
        logger.debug("MoA privacy filter check failed", exc_info=True)

    joined = "\n\n".join(
        f"Reference {idx} — {label}:\n{text}"
        for idx, (label, text, _accounting) in enumerate(successful_outputs, start=1)
    )
    degraded = _degraded_notice(failed_labels, degraded_reference_policy)
    if degraded:
        joined = f"{joined}\n\n{degraded}" if joined else degraded

    # 当所有参考调用均失败或被跳过时，直接跳过聚合器（aggregator）的调用——
    # 对零条有效建议进行合成不仅浪费 Token，
    # 还可能在超时前一直阻塞（例如：在 SenseNova 上观察到长达 ~6 分钟的阻塞），
    # 最终返回一个不可重试的错误，导致整个会话挂起。
    #
    # 此处的提前返回仅包含脱敏后的“不可用通知”
    # （绝不包含原始服务商错误文本），
    # 以保证主 Agent 循环仍能在单模型模式下正常运行。
    if reference_outputs and not successful_outputs:
        logger.warning(
            "MoA: all %d reference(s) failed — skipping aggregator synthesis",
            len(reference_outputs),
        )
        notice = degraded or "[Reference models unavailable]"
        return (
            "[Mixture of Agents context — all reference models failed. "
            "Proceeding without aggregated guidance.]\n"
            f"References: {', '.join(_slot_label(slot) for slot in reference_models)}\n\n"
            f"{notice}"
        )

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
    # Pin the live agent disable onto synthesis decoration so mid-session
    # config flips cannot re-enable markers on this path alone (#76085).
    # Same not-None guard as _run_reference: stamping None would be a no-op
    # (present-None falls through to the config fallback anyway).
    agg_cache_runtime = agg_runtime
    _agg_cache_disabled = (
        getattr(agent, "_cache_disabled", None) if agent is not None else None
    )
    if _agg_cache_disabled is not None:
        agg_cache_runtime = {
            **agg_runtime,
            "_cache_disabled": _agg_cache_disabled,
        }
    try:
        # 与 _run_reference 中顾问（advisor）调用的 cache_control 装饰逻辑相同
        # （参见 _maybe_apply_moa_cache_control）——
        # 这里的合成（synthesis）调用是第三条独立的 MoA 调用路径，
        # 此前的提交 22c5048d9 并未涵盖该路径（当时仅恢复了持久化 `provider: moa` 模型中
        # 担当聚合器（acting-aggregator）轮次的缓存，以及顾问扇出（advisor fan-out）的缓存）。
        #
        # 如果没有此配置，单次执行的 `/moa <prompt>` 命令中的合成调用
        # 将在每次执行时全额计费其输入（包含所有已合并参考输出的无系统提示词），
        # 且没有任何 cache_control 断点；
        # 即便解析出的聚合器槽位属于支持缓存的路由（例如 OpenRouter 上的 Claude 或 native Anthropic），
        # 也会面临这一问题。
        agg_messages = _maybe_apply_moa_cache_control(
            [{"role": "user", "content": synth_prompt}], agg_cache_runtime
        )
        response = call_llm(
            task="moa_aggregator",
            messages=agg_messages,
            temperature=aggregator_temperature,
            reasoning_config=_aggregator_reasoning_config(aggregator),
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


def _completed_response_as_stream_chunk(response: Any) -> Any:
    """Convert a completed Chat Completions response into one delta stream chunk.

    MoA's outer streaming consumer expects ``choices[0].delta`` chunks. A
    completed aggregator response carries ``choices[0].message`` instead; adapt
    it here, at the MoA facade boundary, so provider-specific Relay behavior and
    other transports remain untouched.
    """

    choices = getattr(response, "choices", None)
    first_choice = choices[0] if isinstance(choices, (list, tuple)) and choices else None
    message = getattr(first_choice, "message", None)
    raw_tool_calls = getattr(message, "tool_calls", None)
    tool_call_deltas = None
    if isinstance(raw_tool_calls, (list, tuple)) and raw_tool_calls:
        tool_call_deltas = []
        for index, tc in enumerate(raw_tool_calls):
            function = getattr(tc, "function", None)
            tool_call_deltas.append(SimpleNamespace(
                index=getattr(tc, "index", index),
                id=getattr(tc, "id", None),
                type=getattr(tc, "type", None) or "function",
                function=SimpleNamespace(
                    name=getattr(function, "name", None),
                    arguments=getattr(function, "arguments", None),
                ),
            ))
    delta = SimpleNamespace(
        content=getattr(message, "content", None),
        tool_calls=tool_call_deltas,
        reasoning_content=getattr(message, "reasoning_content", None),
        reasoning=getattr(message, "reasoning", None),
        reasoning_details=getattr(message, "reasoning_details", None),
    )
    choice = SimpleNamespace(
        index=getattr(first_choice, "index", 0),
        delta=delta,
        finish_reason=getattr(first_choice, "finish_reason", None) or "stop",
    )
    return SimpleNamespace(
        id=getattr(response, "id", None),
        model=getattr(response, "model", None),
        choices=[choice],
        usage=getattr(response, "usage", None),
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
    it is already a trailing ``user`` turn (plain chat — still at the end).

    A trailing user turn's content may be a STRING or a LIST of content parts —
    Anthropic prompt-cache decoration (which runs before the MoA facade)
    converts string content to ``[{"type": "text", ..., "cache_control": ...}]``,
    and multimodal turns are lists natively. Both shapes are merged in place:
    appending a new text part AFTER the cache_control-marked part keeps the
    cached prefix byte-stable (the marker still terminates it) while the
    turn-varying guidance rides outside the cached span. Appending a SEPARATE
    user message here instead would produce two consecutive user turns —
    strict providers reject that.
    """
    last = agg_messages[-1] if agg_messages else None
    if last is not None and last.get("role") == "user":
        last_content = last.get("content")
        if isinstance(last_content, str):
            last["content"] = last_content + "\n\n" + guidance
            return
        if isinstance(last_content, list):
            last["content"] = [*last_content, {"type": "text", "text": "\n\n" + guidance}]
            return
    agg_messages.append({"role": "user", "content": guidance})


def peel_reference_guidance(
    messages: list[dict[str, Any]],
    guidance: Any,
) -> list[dict[str, Any]]:
    """Remove reference guidance previously attached by ``_attach_reference_guidance``.

    Exact inverse of the three attach shapes above (string merge, trailing
    text part, appended user message) — kept adjacent so the two evolve
    together; a drifting separator or shape would make the peel silently
    no-op and let a cache breakpoint land on the turn-varying guidance
    block (the bug class #72626 fixes).

    Used by the failover redecoration chokepoint: redecoration must run on
    the base transcript so the last cache breakpoint does not land on the
    guidance; callers then rebase via ``rebase_prepared_request``.

    Returns a new list (input list and its messages are not mutated).
    """
    if not guidance or not messages:
        return messages
    guidance_text = str(guidance)
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return messages
    content = last.get("content")
    if content == guidance_text:
        # Attach shape (c): guidance was appended as its own user message.
        return list(messages[:-1])
    suffix = "\n\n" + guidance_text
    if isinstance(content, str) and content.endswith(suffix):
        # Attach shape (a): merged into a trailing string user turn.
        peeled = dict(last)
        peeled["content"] = content[: -len(suffix)]
        return [*messages[:-1], peeled]
    if isinstance(content, list) and content:
        last_part = content[-1]
        if isinstance(last_part, dict) and last_part.get("type", "text") == "text":
            text = last_part.get("text") or ""
            if text == suffix or text == guidance_text:
                # Attach shape (b): guidance rode as its own trailing part.
                peeled = dict(last)
                peeled["content"] = list(content[:-1])
                if not peeled["content"]:
                    # The guidance part was the only content — mirror the
                    # string shape (c) and drop the whole message rather
                    # than leaving an empty-content user turn behind.
                    return list(messages[:-1])
                return [*messages[:-1], peeled]
            if text.endswith(suffix):
                new_part = dict(last_part)
                new_part["text"] = text[: -len(suffix)]
                peeled = dict(last)
                peeled["content"] = [*content[:-1], new_part]
                return [*messages[:-1], peeled]
    return messages


class MoAChatCompletions:
    """OpenAI-chat-compatible facade where the aggregator is the acting model."""

    def __init__(self, preset_name: str, reference_callback: Any = None, agent: Any = None):
        self.preset_name = preset_name or "default"
        # Optional display hook. Called as reference outputs become available so
        # frontends can show each reference model's answer as a labelled block
        # before the aggregator acts. Signature:
        #   reference_callback(event, **kwargs)
        # where event is one of:
        #   "moa.reference"   kwargs: index, count, label, text
        #   "moa.progress"    kwargs: refs_done, refs_total, label
        #                       (fired once per reference completion — drives
        #                        status-bar progress like ``MOA: 2/3 refs done``)
        #   "moa.phase"       kwargs: phase, refs_done, refs_total, aggregator
        #                       (fired on phase transitions, currently
        #                        phase="aggregator" right before the aggregator
        #                        acts; phase="reference" mirrors ``moa.progress``
        #                        so listeners can rely on a single event family)
        #   "moa.aggregating" kwargs: aggregator (label), ref_count
        # Never raises into the model call — display is best-effort.
        self.reference_callback = reference_callback
        # Back-reference to the owning AIAgent, so the reference fan-out can
        # check agent._interrupt_requested (see _run_references_parallel).
        # Optional — a caller that doesn't pass it just keeps the fan-out
        # uninterruptible, as it was before.
        self._agent = agent
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
        # Guards pending usage/cost against concurrent late-accounting
        # callbacks (see _record_late_reference_accounting), which fire on
        # executor worker threads after an interrupted fan-out returns.
        self._accounting_lock = threading.Lock()
        # Resolved aggregator slot ({provider, model, ...}) from the most recent
        # create(); read by session cost accounting to price the aggregator's
        # acting turn at its real model instead of the virtual preset name.
        self.last_aggregator_slot: Any = None
        # Full-turn trace parts stashed on a cache-MISS create(), awaiting the
        # caller to stitch in the live session_id + resolved aggregator output
        # and flush to the trace file (only when moa.save_traces is on).
        self._pending_trace: Any = None
        # Per-advisor metrics for observability hooks. Unlike _pending_trace
        # this is NOT consumed — post_api_request fires on a different branch
        # than consume_and_save_trace, so a consuming read would race it. Holds
        # until the next fan-out replaces it.
        self._last_reference_metrics: Any = None
        # every_n fan-out cadence state. The iteration counter is scoped to a
        # single USER TURN (not the facade lifetime): it counts create() calls
        # since the last new user message and resets whenever the user-turn
        # signature changes, so cadence position never leaks across turns —
        # iteration 1 of every turn is always on-cadence (fresh advice for a
        # fresh request). See the fanout handling in create().
        self._fanout_iteration_count = 0
        self._fanout_turn_sig: str | None = None
        self._fanout_last_state_sig: str | None = None
        # Normalized moa.privacy_filter mode for the current turn ('' |
        # 'display' | 'full'), refreshed from config on every create().
        self._privacy_mode: str = ""

    def consume_reference_usage(self) -> tuple[Any, Any]:
        """Pop pending reference-fan-out usage + cost, resetting both to empty.

        Returns ``(CanonicalUsage, cost_usd_or_None)`` for the most recent
        ``create()`` and clears the pending values, so a subsequent read (e.g.
        a streaming retry re-entering accounting) cannot double-count. Usage is
        always a ``CanonicalUsage`` (zeroed if none); cost is a summed-dollars
        float or ``None`` when no advisor could be priced.
        """
        from agent.usage_pricing import CanonicalUsage

        with self._accounting_lock:
            usage = self._pending_reference_usage or CanonicalUsage()
            cost = self._pending_reference_cost
            self._pending_reference_usage = CanonicalUsage()
            self._pending_reference_cost = None
        return usage, cost

    def last_reference_metrics(self) -> Any:
        """Per-advisor metrics from the most recent fan-out, or None.

        Read-only: a MoA turn's post_api_request hook must not disturb the
        accounting that consume_reference_usage and consume_and_save_trace own.
        """
        return self._last_reference_metrics

    def _record_late_reference_accounting(self, label: str, accounting: Any) -> None:
        """Fold a late-completing interrupted reference's real spend in.

        When a user interrupt aborts the fan-out wait, references already in
        flight keep running (they cannot be force-killed) and DO bill when
        they complete. Their placeholder results carry zeroed accounting, so
        without this hook that spend would vanish from session accounting.
        The fan-out registers this as a done-callback on abandoned futures;
        it folds the eventual real usage/cost into the pending totals, where
        the next ``consume_reference_usage`` pick-up records it. Thread-safe:
        done-callbacks fire on executor worker threads.
        """
        from agent.usage_pricing import CanonicalUsage

        if not isinstance(accounting, _RefAccounting):
            return
        with self._accounting_lock:
            if isinstance(accounting.usage, CanonicalUsage):
                self._pending_reference_usage = (
                    self._pending_reference_usage or CanonicalUsage()
                ) + accounting.usage
            if accounting.cost_usd is not None:
                self._pending_reference_cost = (
                    self._pending_reference_cost or 0
                ) + accounting.cost_usd
        logger.debug(
            "MoA: recorded late accounting for interrupted reference %s", label
        )

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

    def prepare(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Run the advisor fan-out and return the exact aggregator request.

        The normal agent loop needs to measure this augmented prompt before its
        compression gate.  ``create()`` also uses this method for direct callers;
        when the loop supplies the returned private object back to ``create()``,
        the advisor fan-out is not repeated.
        """
        return self.create(messages=messages, _moa_prepare_only=True)

    def rebase_prepared_request(
        self, prepared: dict[str, Any], messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Apply already-generated advisor guidance to a rebuilt API transcript.

        Context compression changes the persisted transcript but not the
        ephemeral advisor result.  Reusing the guidance avoids a second costly
        fan-out while keeping the aggregator request aligned with the compacted
        history.
        """
        guidance = prepared.get("guidance")
        agg_messages = [dict(message) for message in messages]
        if guidance:
            _attach_reference_guidance(agg_messages, str(guidance))
        return {**prepared, "messages": agg_messages}

    def _call_prepared_aggregator(
        self, prepared: dict[str, Any], api_kwargs: dict[str, Any]
    ) -> Any:
        """Send an already prepared MoA aggregator request exactly once."""
        agg_messages = prepared["messages"]
        aggregator = prepared["aggregator"]
        aggregator_temperature = prepared["aggregator_temperature"]
        if aggregator.get("provider") == "moa":
            raise RuntimeError("MoA aggregator cannot be another MoA preset")
        agg_kwargs = dict(api_kwargs)
        max_tokens: Any = agg_kwargs.get("max_tokens")
        tools: Any = agg_kwargs.get("tools")
        extra_body: Any = agg_kwargs.get("extra_body")
        agg_runtime = _slot_runtime(aggregator)
        try:
            from agent.agent_runtime_helpers import (
                plan_cache_sections_for_destination,
            )

            guidance = prepared.get("guidance")
            planning_messages = agg_messages
            if guidance:
                planning_messages = peel_reference_guidance(
                    agg_messages,
                    str(guidance),
                )
            # plan_cache_sections_for_destination never mutates its inputs
            # and always returns request-local copies, so the prepared
            # state stays canonical.
            # Tri-state: only pass a bool when a live agent snapshot exists.
            # Prepared-aggregator facades built via __new__ have no _agent;
            # getattr(self._agent, ...) raises and bool(None-agent) would
            # force False and suppress the planner's config fallback (#76085).
            _agent = getattr(self, "_agent", None)
            _cache_disabled = (
                getattr(_agent, "_cache_disabled", None)
                if _agent is not None
                else None
            )
            agg_messages, tools = plan_cache_sections_for_destination(
                planning_messages,
                tools,
                provider=agg_runtime.get("provider") or "",
                base_url=agg_runtime.get("base_url") or "",
                api_mode=agg_runtime.get("api_mode") or "",
                model=agg_runtime.get("model") or "",
                cache_disabled=_cache_disabled,
            )
            if guidance:
                _attach_reference_guidance(agg_messages, str(guidance))
        except Exception as exc:  # pragma: no cover - cache planning must not block MoA
            # Warning, not debug: since the call-block site skips MoA, this
            # block is the aggregator's ONLY decoration path — a silent
            # failure here ships an undecorated request and regresses the
            # exact 0%-cache MoA failure the planning exists to prevent.
            logger.warning(
                "MoA aggregator cache plan failed — sending undecorated "
                "request (cache misses expected): %s", exc,
            )
        # Record the exact aggregator INPUT (incl. the injected reference
        # context) into the pending trace so a trace captures what the
        # aggregator actually saw, not a reconstruction. Traces are a
        # persisted surface: when the privacy filter is active, the stored
        # COPY is redacted ('display' mode's live aggregator input stays raw —
        # only the on-disk record is filtered; 'full' mode's input is already
        # redacted upstream, so this is a near no-op there).
        if self._pending_trace is not None:
            self._pending_trace["aggregator_input_messages"] = (
                _redact_trace_messages([dict(m) for m in agg_messages])
                if getattr(self, "_privacy_mode", "")
                else agg_messages
            )
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
        # _slot_runtime may carry the provider's request_overrides.extra_body;
        # pop it and merge with the caller's extra_body (caller wins) so the
        # explicit kwarg below never collides with **agg_runtime.
        agg_extra_body = _merge_slot_extra_body(
            agg_runtime.pop("extra_body", None),
            extra_body,
        )
        _agg_response = call_llm(
            task="moa_aggregator",
            messages=agg_messages,
            temperature=aggregator_temperature,
            max_tokens=max_tokens,
            tools=tools,
            extra_body=agg_extra_body,
            # Prepared requests must retain the acting aggregator's reasoning
            # policy exactly as the direct create() path does (#64187).
            reasoning_config=_aggregator_reasoning_config(aggregator),
            **stream_kwargs,
            **agg_runtime,
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
        if stream and hasattr(_agg_response, "choices"):
            # Some aggregator adapters (notably openai-codex Responses) consume
            # their provider stream internally and return a completed response
            # object even when the acting consumer requested token streaming.
            # The outer chat-completions streaming loop expects delta chunks;
            # hand it a one-chunk iterator instead of letting it iterate the
            # SimpleNamespace response itself (#55933).
            return iter((_completed_response_as_stream_chunk(_agg_response),))
        return _agg_response

    def create(self, **api_kwargs: Any) -> Any:
        prepared_request = api_kwargs.pop("_moa_prepared_request", None)
        if prepared_request is not None:
            if not isinstance(prepared_request, dict):
                raise TypeError("_moa_prepared_request must be a dict")
            return self._call_prepared_aggregator(prepared_request, api_kwargs)

        from hermes_cli.config import get_config_path, load_config
        from hermes_cli.moa_config import resolve_moa_preset

        # Resolve the preset once per (config st_mtime_ns, preset_name).
        # resolve_moa_preset re-normalizes + re-validates the whole moa
        # config block on every call, and create() runs once per tool-loop
        # iteration — a serial cold-start cost before the parallel fan-out
        # can begin (#66793). Keyed on the config FILE's mtime_ns (not a
        # config-object attribute, which load_config()'s dicts don't carry),
        # so a config edit invalidates on the next call.
        try:
            _cfg_stamp = get_config_path().stat().st_mtime_ns
        except OSError:
            _cfg_stamp = None
        # load_config() is itself (mtime_ns, size)-cached upstream, so this
        # read is cheap; the expensive part this cache skips is
        # resolve_moa_preset's re-normalization + re-validation.
        _moa_raw = load_config().get("moa") or {}
        preset_cache_key = (_cfg_stamp, self.preset_name)
        preset = None
        if _cfg_stamp is not None:
            with _preset_cache_lock:
                preset = _preset_cache.get(preset_cache_key)
        if preset is None:
            preset = resolve_moa_preset(_moa_raw, self.preset_name)
            if _cfg_stamp is not None:
                with _preset_cache_lock:
                    _preset_cache.clear()  # one live config stamp at a time
                    _preset_cache[preset_cache_key] = preset
        # Privacy filter mode: '' (off, default) | 'display' | 'full'. See
        # coerce_privacy_filter / the pattern block at the top of this module.
        # Remembered on self so _call_prepared_aggregator (which may run on a
        # later prepared-request call without re-reading config) redacts the
        # trace's aggregator input consistently with this turn's fan-out.
        privacy_mode = _moa_privacy_mode(_moa_raw)
        self._privacy_mode = privacy_mode
        messages = list(api_kwargs.get("messages") or [])
        reference_models = [
            slot for slot in (preset.get("reference_models") or [])
            if slot.get("enabled", True)
        ]
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
        # None (the default) = no per-preset override; the fan-out inherits
        # auxiliary.moa_reference.timeout (900s default) via call_llm's own
        # per-task timeout resolution. Explicit per-preset values are honored.
        raw_reference_timeout = preset.get("reference_timeout")
        reference_timeout = (
            float(raw_reference_timeout) if raw_reference_timeout else None
        )
        degraded_reference_policy = str(
            preset.get("degraded_reference_policy") or "loud"
        )
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

        # Fan-out cadence. "user_turn" (default — cheapest cadence, #67199):
        # advisors run ONCE per user turn; subsequent tool iterations reuse
        # that turn's advice and the aggregator acts alone (the original MoA
        # shape: synthesize at the start, then let the acting model work).
        # Implemented by hashing only the prefix up to the LAST USER message
        # so mid-turn growth doesn't change the signature — iteration 2+
        # becomes a cache HIT. "per_iteration": advisors re-run whenever the
        # advisory view changes — i.e. every tool iteration, since the view
        # grows with each tool result; advice tracks live task state at the
        # cost of multiplying advisor latency/spend by tool-loop depth.
        # "every_n:<N>" (N >= 2): the middle ground (issue #63393 — advisor
        # fan-out multiplies latency/cost by the tool-iteration count).
        # Advisors run on iteration 1 of a user turn and then every Nth tool
        # iteration; the iterations in between REUSE the cached guidance from
        # the last on-cadence run (same mechanism as user_turn's cache HIT —
        # the aggregator still gets advice every iteration, it's just not
        # refreshed against the very latest tool results). The iteration
        # counter is scoped per user turn and resets on a new user message,
        # so every turn starts with fresh advice.
        fanout_mode = str(preset.get("fanout") or "user_turn").strip().lower()
        every_n = 0
        if fanout_mode.startswith("every_n:"):
            try:
                every_n = int(fanout_mode.split(":", 1)[1])
            except (TypeError, ValueError):
                every_n = 0
            if every_n < 2:
                # every_n:1 semantically IS per-iteration; degrade there,
                # mirroring _coerce_fanout's collapse of degenerate N.
                fanout_mode = "per_iteration"
        sig_messages = ref_messages
        turn_prefix = ref_messages
        if fanout_mode in ("user_turn",) or every_n >= 2:
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
                turn_prefix = ref_messages[: last_user_idx + 1]
            if fanout_mode == "user_turn":
                sig_messages = turn_prefix

        def _hash_messages(msgs: list[dict[str, Any]]) -> str:
            return hashlib.sha256(
                "\u0000".join(
                    f"{m.get('role')}:{m.get('content')}" for m in msgs
                ).encode("utf-8", "replace")
            ).hexdigest()

        # every_n cadence bookkeeping: advance the per-turn iteration counter
        # only when the advisory STATE actually advanced (a redundant create()
        # with identical state — e.g. a streaming retry — must not consume a
        # cadence slot), and reset it whenever the user-turn prefix changes.
        _every_n_reuse = False
        if every_n >= 2:
            _turn_sig = _hash_messages(turn_prefix)
            if _turn_sig != self._fanout_turn_sig:
                self._fanout_turn_sig = _turn_sig
                self._fanout_iteration_count = 0
                self._fanout_last_state_sig = None
            _state_sig = _hash_messages(ref_messages)
            if _state_sig != self._fanout_last_state_sig:
                self._fanout_last_state_sig = _state_sig
                self._fanout_iteration_count += 1
            # Iteration 1 is on-cadence; then every Nth iteration after it.
            _on_cadence = (self._fanout_iteration_count - 1) % every_n == 0
            _every_n_reuse = not _on_cadence and bool(self._ref_cache_outputs)

        # Turn-scoped cache: only run + display references when the advisory
        # view changed (i.e. a new user turn). Within one turn the agent loop
        # calls create() once per tool iteration; in user_turn mode the
        # signature is stable across those iterations (prefix hash above), so
        # the fan-out runs once per user turn and iterations reuse the advice.
        _sig = _hash_messages(sig_messages)
        _cache_key = (self.preset_name, _sig, tuple(_slot_label(s) for s in reference_models))
        if _every_n_reuse:
            # Off-cadence every_n iteration: pin the key to the last
            # on-cadence run so the lookup below is a HIT and its guidance is
            # reused (no advisor calls, no double accounting, no re-emit) —
            # exactly the user_turn cache-HIT path. When the cache is empty
            # (defensive; a new turn resets the counter to on-cadence) the
            # flag above stays False and the references run normally.
            _cache_key = self._ref_cache_key
        _refs_from_cache = _cache_key == self._ref_cache_key and bool(self._ref_cache_outputs)

        if _refs_from_cache:
            reference_outputs = list(self._ref_cache_outputs)
            # References already ran (and were accounted) earlier this turn;
            # this create() is a repeat tool-iteration reusing the cached
            # advice. Charging their tokens/cost again here would multiply
            # advisor spend by the tool-iteration count, so nothing new is
            # deposited — but do NOT zero the pending totals: a
            # late-completing interrupted reference may have deposited its
            # real spend since the last consume(), and that must survive
            # until the next consume_reference_usage() pick-up.
            # Likewise no trace on a cache HIT — the full turn was already
            # traced on the MISS that ran the references. A repeat iteration is
            # not a new MoA turn.
            self._pending_trace = None
        else:
            # Per-reference progress callback: emits ``moa.progress`` so
            # listeners can render ``MOA: N/M refs done`` in the status bar as
            # each reference completes. The callback is bound to self so it
            # goes through the same display hook as the existing
            # ``moa.reference`` / ``moa.aggregating`` events.
            def _progress(done: int, total: int, label: str) -> None:
                self._emit(
                    "moa.progress",
                    refs_done=done,
                    refs_total=total,
                    label=label,
                )

            reference_outputs = _run_references_parallel(
                reference_models,
                ref_messages,
                temperature=temperature,
                max_tokens=reference_max_tokens,
                progress_callback=_progress,
                reference_timeout=reference_timeout,
                agent=self._agent,
                late_accounting_sink=self._record_late_reference_accounting,
            )
            interrupted_any = any(
                text == _INTERRUPTED_REFERENCE_NOTE
                for _lbl, text, _acct in reference_outputs
            )
            if interrupted_any:
                # An interrupted fan-out is a partial snapshot, not real
                # advice for this state. Caching it would replay the
                # placeholder notes on every subsequent iteration of the
                # turn (a cache HIT never re-runs the references), so leave
                # the cache empty and let the next create() re-run them.
                self._ref_cache_key = None
                self._ref_cache_outputs = []
            else:
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
            with self._accounting_lock:
                # Fold (don't overwrite): a late-completing interrupted
                # reference from a PREVIOUS turn may have deposited its real
                # spend here between consume() calls — keep it.
                self._pending_reference_usage = (
                    self._pending_reference_usage or CanonicalUsage()
                ) + _ref_usage
                if _ref_cost is not None:
                    self._pending_reference_cost = (
                        self._pending_reference_cost or 0
                    ) + _ref_cost
            # Stash the full reference fan-out for trace persistence. The
            # aggregator input/label are filled in below once agg_messages is
            # built; the aggregator OUTPUT is stitched in by the caller
            # (consume_and_save_trace) once the response resolves — the caller
            # holds the live session_id and the resolved aggregator response.
            # Traces are a persisted, user-readable surface, so ANY active
            # privacy mode ('display' or 'full') redacts the advisor text and
            # the full per-advisor input/output carried by _RefAccounting.
            if privacy_mode:
                _trace_refs = [
                    (label, _redact_reference_text(text), _redact_trace_accounting(acct))
                    for label, text, acct in reference_outputs
                ]
            else:
                _trace_refs = list(reference_outputs)
            self._pending_trace = {
                "preset": self.preset_name,
                "reference_outputs": _trace_refs,
                "aggregator_slot": aggregator,
                "aggregator_temperature": aggregator_temperature,
            }
            # Derived from the same privacy-redacted _trace_refs, so an active
            # privacy mode redacts the observability payload too.
            try:
                from agent.moa_trace import slot_metrics

                self._last_reference_metrics = [
                    slot_metrics(acct, label, output=text)
                    for label, text, acct in _trace_refs
                ]
            except Exception as exc:  # pragma: no cover - never break a turn
                logger.debug("MoA reference metrics render failed: %s", exc)
                self._last_reference_metrics = None

            # Surface each reference model's answer to the display BEFORE the
            # aggregator acts — once per turn (only on the iteration that
            # actually ran them). The user sees one labelled block per
            # reference (rendered like a thinking block) so the MoA process is
            # visible rather than a silent pause. Best-effort: never blocks the
            # turn. Reference blocks are a user-visible surface: both privacy
            # modes redact them (the cache keeps the RAW text — redaction
            # always happens at the consuming surface, so a mid-session mode
            # change never leaks or double-redacts).
            _ref_count = len(reference_outputs)
            for _idx, (_label, _text, _accounting) in enumerate(reference_outputs, start=1):
                self._emit(
                    "moa.reference",
                    index=_idx,
                    count=_ref_count,
                    label=_label,
                    text=_redact_reference_text(_text) if privacy_mode else _text,
                )
            if _ref_count:
                # Phase transition: reference fan-out is complete, the
                # aggregator is about to act. Listeners that prefer a single
                # event family for phase tracking can switch on ``phase``
                # instead of subscribing to ``moa.aggregating`` separately.
                self._emit(
                    "moa.phase",
                    phase="aggregator",
                    refs_done=_ref_count,
                    refs_total=_ref_count,
                    aggregator=_slot_label(aggregator),
                )
                self._emit(
                    "moa.aggregating",
                    aggregator=_slot_label(aggregator),
                    ref_count=_ref_count,
                )

        guidance: str | None = None
        agg_messages = [dict(m) for m in messages]
        successful_outputs = _successful_references(reference_outputs)
        failed_labels = _failed_reference_labels(reference_outputs)
        joined = ""
        _agg_refs: list = []
        if successful_outputs:
            # 'full' privacy mode: redact the advisor text that reaches the
            # AGGREGATOR too (issue #59959's literal ask). 'display' leaves
            # the aggregator input raw so synthesis quality is unaffected.
            # The redaction is applied to a per-call copy — the cache always
            # holds raw advisor text (see the emit comment above). Failed
            # refs are already filtered out; only successful advisor text is
            # joined (and redacted when requested).
            _agg_refs = (
                _redact_reference_outputs(successful_outputs)
                if privacy_mode == "full"
                else successful_outputs
            )
            joined = "\n\n".join(
                f"Reference {idx} — {label}:\n{text}"
                for idx, (label, text, _usage) in enumerate(_agg_refs, start=1)
            )
        degraded = _degraded_notice(failed_labels, degraded_reference_policy)
        if reference_outputs and not successful_outputs:
            # Every reference failed or was skipped: don't wrap a wall of
            # failure sentinels in "use the reference responses below"
            # guidance — the aggregator IS the acting model, so it simply
            # acts alone this turn. Under the loud policy it still gets the
            # sanitized unavailability notice so it can disclose degraded
            # mode; under silent it gets nothing.
            logger.warning(
                "MoA: all %d reference(s) failed — acting aggregator-alone "
                "without reference guidance",
                len(reference_outputs),
            )
            if degraded:
                guidance = (
                    "[Mixture of Agents reference context]\n"
                    f"Preset: {self.preset_name}\n"
                    f"Aggregator/acting model: {_slot_label(aggregator)}\n\n"
                    "All reference models failed this turn — no advisory "
                    "guidance is available. Act on your own judgment.\n\n"
                    f"{degraded}"
                )
                _attach_reference_guidance(agg_messages, guidance)
        elif joined or degraded:
            if degraded:
                joined = f"{joined}\n\n{degraded}" if joined else degraded
            guidance = (
                "[Mixture of Agents reference context]\n"
                f"Preset: {self.preset_name}\n"
                f"Aggregator/acting model: {_slot_label(aggregator)}\n"
                f"References: {', '.join(label for label, _, _ in _agg_refs)}\n\n"
                "Use the reference responses below as private context. You are the aggregator and acting model: "
                "answer the user directly or call tools as needed.\n\n"
                f"{joined}"
            )
            _attach_reference_guidance(agg_messages, guidance)

        prepared_request = {
            "messages": agg_messages,
            "guidance": guidance,
            "aggregator": aggregator,
            "aggregator_temperature": aggregator_temperature,
        }
        if api_kwargs.pop("_moa_prepare_only", False):
            return prepared_request
        return self._call_prepared_aggregator(prepared_request, api_kwargs)


class MoAClient:
    def __init__(self, preset_name: str, reference_callback: Any = None, agent: Any = None):
        self.chat = type("_MoAChat", (), {})()
        self.chat.completions = MoAChatCompletions(
            preset_name, reference_callback=reference_callback, agent=agent,
        )

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

    def last_reference_metrics(self) -> Any:
        """Per-advisor metrics from the most recent fan-out, or None.

        Read-only, unlike the two consume_* methods above: the observability
        hook fires on a different branch than the accounting they own.
        """
        return self.chat.completions.last_reference_metrics()


def build_moa_facade(agent, preset_name: Any = None) -> MoAClient:
    """Build the MoA facade client for ``agent``, wiring the reference relay.

    Single construction point for ``MoAClient`` wherever the agent's shared
    client is (re)built: initial setup (``agent_init``), turn-start fallback
    restore (``restore_primary_runtime``), transient transport recovery
    (``try_recover_primary_transport``), and mid-session model switches
    (``switch_model``).

    Constructing a bare ``MoAClient(preset)`` at any of those sites silently
    drops the ``reference_callback`` relay that ``agent_init`` wires to
    ``agent.tool_progress_callback`` — after a fallback+restore cycle the
    facade would still work, but every frontend (CLI spinner, TUI, desktop,
    gateway) would stop receiving ``moa.reference`` / ``moa.aggregating``
    display events for the rest of the session (#53802).

    The relay reads ``agent.tool_progress_callback`` at *emit* time, so a
    callback attached after client construction is picked up automatically.
    Best-effort and display-only — it never raises into the model call.
    """
    def _moa_reference_relay(event: str, **kwargs: Any) -> None:
        cb = getattr(agent, "tool_progress_callback", None)
        if cb is None:
            return
        try:
            if event == "moa.reference":
                label = str(kwargs.get("label") or "")
                text = str(kwargs.get("text") or "")
                idx = kwargs.get("index")
                count = kwargs.get("count")
                cb(
                    "moa.reference",
                    label,
                    text,
                    None,
                    moa_index=idx,
                    moa_count=count,
                )
            elif event == "moa.progress":
                # Per-reference completion. Frontends render this as a
                # status-bar progress indicator like ``MOA: N/M refs done``.
                cb(
                    "moa.progress",
                    str(kwargs.get("label") or ""),
                    None,
                    None,
                    moa_refs_done=kwargs.get("refs_done"),
                    moa_refs_total=kwargs.get("refs_total"),
                )
            elif event == "moa.phase":
                # Phase transition (currently only ``phase="aggregator"``
                # fires once the fan-out is done). Subscribers can switch
                # on ``moa_phase`` to know which phase is active.
                cb(
                    "moa.phase",
                    str(kwargs.get("aggregator") or ""),
                    None,
                    None,
                    moa_phase=kwargs.get("phase"),
                    moa_refs_done=kwargs.get("refs_done"),
                    moa_refs_total=kwargs.get("refs_total"),
                )
            elif event == "moa.aggregating":
                cb(
                    "moa.aggregating",
                    str(kwargs.get("aggregator") or ""),
                    None,
                    None,
                    moa_ref_count=kwargs.get("ref_count"),
                )
        except Exception:
            pass

    resolved_preset = preset_name
    if resolved_preset is None and getattr(agent, "provider", None) == "moa":
        resolved_preset = getattr(agent, "model", None)

    resolved_preset = str(resolved_preset or "default")
    try:
        from hermes_cli.config import load_config
        from hermes_cli.moa_config import normalize_moa_config

        moa_cfg = normalize_moa_config(load_config().get("moa") or {})
        presets = moa_cfg.get("presets") or {}
        if resolved_preset not in presets:
            resolved_preset = moa_cfg.get("default_preset") or "default"
    except Exception:
        resolved_preset = "default"

    return MoAClient(
        resolved_preset,
        reference_callback=_moa_reference_relay,
        # Thread the agent through so the reference fan-out wait can be
        # aborted on a user interrupt (see _run_references_parallel).
        agent=agent,
    )
