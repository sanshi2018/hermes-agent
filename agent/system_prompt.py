"""System-prompt assembly for :class:`AIAgent`.

The agent's system prompt is built once per session and reused across all
turns — only context compression triggers a rebuild.  This keeps the
upstream prefix cache warm.  See ``hermes-agent-dev``'s
``references/system-prompt-invariant.md`` for the invariants and
``references/self-improvement-loop.md`` for how the background-review
fork inherits the cached prompt verbatim.

Three tiers are joined with ``\\n\\n``:

* ``stable``   — identity (SOUL.md or DEFAULT_AGENT_IDENTITY), tool
  guidance, computer-use guidance, nous subscription block, tool-use
  enforcement guidance + per-model operational guidance, skills prompt,
  alibaba model-name workaround, environment hints, platform hints.
* ``context``  — caller-supplied ``system_message`` plus context files
  (AGENTS.md / .cursorrules / etc.) discovered under ``TERMINAL_CWD``.
* ``volatile`` — memory snapshot, USER.md profile, external memory
  provider block, timestamp/session/model/provider line.

Pure helpers that read the agent's state.  AIAgent keeps thin forwarders.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY,
    GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
    HERMES_AGENT_HELP_GUIDANCE,
    KANBAN_GUIDANCE,
    MEMORY_GUIDANCE,
    OPENAI_MODEL_EXECUTION_GUIDANCE,
    PARALLEL_TOOL_CALL_GUIDANCE,
    PLATFORM_HINTS,
    SESSION_SEARCH_GUIDANCE,
    SKILLS_GUIDANCE,
    STEER_CHANNEL_NOTE,
    TASK_COMPLETION_GUIDANCE,
    TOOL_USE_ENFORCEMENT_GUIDANCE,
    TOOL_USE_ENFORCEMENT_MODELS,
    drain_truncation_warnings,
)
from agent.runtime_cwd import resolve_context_cwd
from utils import is_truthy_value


def _ra():
    """Lazy reference to the ``run_agent`` module.

    Helpers like ``load_soul_md``, ``build_environment_hints``,
    ``build_context_files_prompt``, ``build_nous_subscription_prompt``,
    ``build_skills_system_prompt`` and ``get_toolset_for_tool`` are
    imported into ``run_agent``'s namespace.  Many tests
    ``patch("run_agent.load_soul_md", ...)``; if we imported them
    directly here those patches would not reach us.  Looking them up
    through ``run_agent`` on every call preserves the patch contract.
    """
    import run_agent
    return run_agent


def _resolve_platform_hint(agent: Any, platform_key: str, default_hint: str) -> str:
    """Apply a per-platform prompt-hint override to the default hint.

    Reads ``agent._platform_hint_overrides`` (populated from
    ``config.yaml`` ``platform_hints`` by ``agent_init``) and resolves the
    effective hint for *platform_key*:

      * ``replace`` — substitute the default hint entirely.
      * ``append``  — keep the default and append the extra text.
      * a bare string value — treated as ``append`` (convenience shorthand).

    Precedence: ``replace`` wins over ``append`` if both are present.
    Override text is added on top of (not instead of) the SOUL/context/
    memory tiers — it only affects the platform-hint segment, so other
    platforms are unaffected and general system instructions still apply.

    Defensive: any malformed entry falls back to the unmodified default so
    a bad config value can never break prompt assembly or leak across
    platforms.
    """
    if not platform_key:
        return default_hint
    overrides = getattr(agent, "_platform_hint_overrides", None)
    if not isinstance(overrides, dict) or not overrides:
        return default_hint
    spec = overrides.get(platform_key)
    if spec is None:
        return default_hint

    # Shorthand: a bare string is treated as append text.
    if isinstance(spec, str):
        extra = spec.strip()
        return f"{default_hint}\n\n{extra}".strip() if extra else default_hint

    if not isinstance(spec, dict):
        return default_hint

    replace_text = spec.get("replace")
    if isinstance(replace_text, str) and replace_text.strip():
        base = replace_text.strip()
    else:
        base = default_hint

    append_text = spec.get("append")
    if isinstance(append_text, str) and append_text.strip():
        return f"{base}\n\n{append_text.strip()}".strip()
    return base


_TUI_EMBEDDED_PANE_CLARIFIER = (
    " You're in its embedded terminal pane, beside the GUI chat — the user can "
    "select your output (Option-drag on macOS, Shift-drag elsewhere) and press "
    "Cmd/Ctrl+L to send it to the chat composer."
)


def _tui_embedded_pane_clarifier(hint: str) -> str:
    """将桌面嵌入式终端面板说明附加到 TUI 提示中。

    由 ``HERMES_DESKTOP_TERMINAL=1`` 触发（该变量由 ``main.cjs`` 仅在
    桌面嵌入式 TUI PTY 的 shell 环境中设置 —— 绝不会在聊天后端中设置）。
    这是一个运行时界面的限定符，而不是配置重写，因此它存在于
    解析位置，而不是存在于 ``_resolve_platform_hint`` 内部（该方法
    纯粹是配置平台提示 [config-platform_hints] 重写的应用者）。对
    缓存来说是字节级稳定的：每次会话构建时调用一次，由环境变量状态决定。

    幂等且对空值安全：在已增强的提示上重新应用是无操作（no-op），
    并且空输入会返回空值（我们绝不会在没有其 TUI 框架的情况下单独合成该说明）。
    """
    if not hint:
        return hint
    if _TUI_EMBEDDED_PANE_CLARIFIER in hint:
        return hint
    if not is_truthy_value(os.getenv("HERMES_DESKTOP_TERMINAL")):
        return hint
    return hint + _TUI_EMBEDDED_PANE_CLARIFIER


def build_system_prompt_parts(agent: Any, system_message: Optional[str] = None) -> Dict[str, str]:
    """将系统提示词（system prompt）组装为有序的三个部分。

    返回一个包含三个键的字典：
      * ``stable``   — 身份标识、工具引导、技能提示词、
        环境提示、平台提示、模型家族运行指引。
      * ``context``  — 上下文文件（AGENTS.md、.cursorrules 等）
        以及调用者提供的 system_message。
      * ``volatile`` — 记忆快照、用户配置文件、外部
        记忆服务商数据块、时间戳行。

    由 :func:`build_system_prompt` 拼接成单个字符串，并在
    AIAgent 的生命周期内缓存于 ``agent._cached_system_prompt`` 中。
    Hermes 绝不会在会话中途重新渲染该字符串的任何部分 —— 这是
    在跨轮次对话中保持上游 prompt 缓存活跃（warm）的唯一方法。
    """
    # ------------------------------------------------------------
    # 局部导入，以避免在模块加载时拉入 model_tools。
    # 测试会对 ``run_agent.get_toolset_for_tool`` 类似的高级辅助函数进行打补丁（patch），
    # 因此我们通过 ``_ra()`` 进行解析，以使这些补丁生效。
    _r = _ra()

    # 仅解析一次模型的上下文窗口，以便上下文文件上限可以据此进行伸缩
    # （动态上限 —— 参见 prompt_builder._dynamic_context_file_max_chars）。
    # 若为 None，则回退到历史固定的默认值。此值在对话的生命周期内是稳定的，
    # 因此不会对 prompt 缓存造成影响。
    _ctx_len: Optional[int] = None
    _cc = getattr(agent, "context_compressor", None)
    if _cc is not None:
        _cc_len = getattr(_cc, "context_length", None)
        if isinstance(_cc_len, int) and _cc_len > 0:
            _ctx_len = _cc_len

    # ── Stable tier ────────────────────────────────────────────────
    stable_parts: List[str] = []

    # 除非调用者显式跳过，否则尝试将 SOUL.md 作为主要身份。
    # 某些执行模式（如 cron）仍需要使用 HERMES_HOME 人格，
    # 同时保持禁用当前工作目录（cwd）下的项目指令。
    _soul_loaded = False
    if agent.load_soul_identity or not agent.skip_context_files:
        _soul_content = _r.load_soul_md(_ctx_len)
        if _soul_content:
            stable_parts.append(_soul_content)
            _soul_loaded = True

    if not _soul_loaded:
        # Fallback to hardcoded identity
        stable_parts.append(DEFAULT_AGENT_IDENTITY)

    # 指向 hermes-agent 技能和文档的指针，用于解答用户关于 Hermes 自身的问题。
    # Pointer to the hermes-agent skill + docs for user questions about Hermes itself.
    stable_parts.append(HERMES_AGENT_HELP_GUIDANCE)

    # 通用的任务完成 / 杜绝编造引导。应用于所有
    # 模型，无论是否启用了工具使用强制（tool_use_enforcement）限制 —— 该引导
    # 所针对的失败模式（如遇到占位存根就停止、在真实路径受阻时编造输出）
    # 并非特定模型家族所独有。该功能仅受
    # config.yaml 中 ``agent.task_completion_guidance`` 的控制（默认为 True），
    # 这样想要更精简提示词的用户可以将其关闭。
    if getattr(agent, "_task_completion_guidance", True) and agent.valid_tool_names:
        stable_parts.append(TASK_COMPLETION_GUIDANCE)

    # 通用并行工具调用（parallel-tool-call）引导。告诉模型将
    # 独立的工具调用合并（batch）到单个助手轮次中，而不是每轮只发出一个
    # 调用 —— 运行时（runtime）已经能够并发运行独立的调用
    # （只读工具一律并发；非重叠路径范围的文件操作也并发），因此
    # 唯一缺少的就是引导模型去生成这样的批量调用。这减少了
    # 往返次数，并降低了在长期对话中不断累积的重复发送上下文（resent-context）的成本。
    # 该功能受 config.yaml 中的 ``agent.parallel_tool_call_guidance`` 控制
    # （默认为 True），并且仅在实际加载了工具时才会注入。
    if getattr(agent, "_parallel_tool_call_guidance", True) and agent.valid_tool_names:
        stable_parts.append(PARALLEL_TOOL_CALL_GUIDANCE)

    # Tool-aware behavioral guidance: only inject when the tools are loaded
    tool_guidance = []
    if "memory" in agent.valid_tool_names:
        tool_guidance.append(MEMORY_GUIDANCE)
    if "session_search" in agent.valid_tool_names:
        tool_guidance.append(SESSION_SEARCH_GUIDANCE)
    if "skill_manage" in agent.valid_tool_names:
        tool_guidance.append(SKILLS_GUIDANCE)
    # 看板工作器/编排器生命周期 —— 仅在调度器派生此进程时存在
    # （kanban_show 的 check_fn 门控取决于 HERMES_KANBAN_TASK 环境变量）。
    # 正常的聊天会话永远不会看到此代码块。
    # 在 __init__ 时解析一次（参见 _kanban_worker_guidance）。
    _kanban_guidance = getattr(agent, "_kanban_worker_guidance", None)
    if _kanban_guidance:
        tool_guidance.append(_kanban_guidance)
    elif _kanban_guidance is None and "kanban_show" in agent.valid_tool_names:
        # Fallback for code paths that bypass agent_init (rare).
        tool_guidance.append(KANBAN_GUIDANCE)
    if tool_guidance:
        stable_parts.append(" ".join(tool_guidance))

    # 引导仅在工具结果中生效，因此只有当智能体拥有工具时才可触达。
    # 静态文本 → 字节级稳定提示词（无缓存命中）。
    if agent.valid_tool_names:
        stable_parts.append(STEER_CHANNEL_NOTE)

    # 电脑使用（Computer-use）—— 作为其独立的代码块传入，
    # 而不是合并到 tool_guidance 中，因为其内容包含多个段落。
    # 该引导是针对宿主平台进行渲染的，因此 Windows/Linux 宿主机
    # 不会看到仅限 macOS 的措辞（如 Mac、Space、cmd+s）。
    if "computer_use" in agent.valid_tool_names:
        from agent.prompt_builder import computer_use_guidance
        stable_parts.append(computer_use_guidance())

    nous_subscription_prompt = _r.build_nous_subscription_prompt(agent.valid_tool_names)
    if nous_subscription_prompt:
        stable_parts.append(nous_subscription_prompt)
    # 工具调用强制执行：指示模型实际调用工具，而不是描述预期的操作。
    # 由 config.yaml 控制
    # agent.tool_use_enforcement:
    #   "auto" (默认) — 匹配 TOOL_USE_ENFORCEMENT_MODELS
    #   true  — 总是注入 (所有模型)
    #   false — 从不注入
    #   list  — 需匹配的自定义模型名称子字符串列表
    if agent.valid_tool_names:
        _enforce = agent._tool_use_enforcement
        _inject = False
        if _enforce is True or (isinstance(_enforce, str) and _enforce.lower() in {"true", "always", "yes", "on"}):
            _inject = True
        elif _enforce is False or (isinstance(_enforce, str) and _enforce.lower() in {"false", "never", "no", "off"}):
            _inject = False
        elif isinstance(_enforce, list):
            model_lower = (agent.model or "").lower()
            _inject = any(p.lower() in model_lower for p in _enforce if isinstance(p, str))
        else:
            # "auto" or any unrecognised value — use hardcoded defaults
            model_lower = (agent.model or "").lower()
            _inject = any(p in model_lower for p in TOOL_USE_ENFORCEMENT_MODELS)
        if _inject:
            stable_parts.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
            _model_lower = (agent.model or "").lower()
            # Google 模型操作指南（简洁性、绝对
            # 路径、并行工具调用、先验证后修改等）
            if "gemini" in _model_lower or "gemma" in _model_lower:
                stable_parts.append(GOOGLE_MODEL_OPERATIONAL_GUIDANCE)
            # OpenAI GPT/Codex 执行纪律（工具持久性、
            # 先决条件检查、验证、防幻觉）。
            # 同样适用于 xAI Grok —— 相同的失败模式（声称已完成
            # 却未调用工具，建议采用变通方案而非使用
            # 现有工具，以计划回复而非实际执行）。
            if "gpt" in _model_lower or "codex" in _model_lower or "grok" in _model_lower:
                stable_parts.append(OPENAI_MODEL_EXECUTION_GUIDANCE)

    has_skills_tools = any(name in agent.valid_tool_names for name in ['skills_list', 'skill_view', 'skill_manage'])
    if has_skills_tools:
        avail_toolsets = {
            toolset
            for toolset in (
                _r.get_toolset_for_tool(tool_name) for tool_name in agent.valid_tool_names
            )
            if toolset
        }
        # 专注模式（选择性开启）会将索引中的非编程技能类别降级为仅显示名称
        # （绝不隐藏 —— skill_view/skills_list 仍能访问所有内容，
        # 且每个名称都保持可见以便于检索/回想）。
        # 默认的编程模式下，索引保持原样。
        _compact_cats = frozenset()
        try:
            from agent.coding_context import coding_compact_skill_categories

            _compact_cats = coding_compact_skill_categories(
                platform=agent.platform, cwd=resolve_context_cwd()
            )
        except Exception:
            _compact_cats = frozenset()
        skills_prompt = _r.build_skills_system_prompt(
            available_tools=agent.valid_tool_names,
            available_toolsets=avail_toolsets,
            compact_categories=_compact_cats or None,
        )
    else:
        skills_prompt = ""
    if skills_prompt:
        stable_parts.append(skills_prompt)

    # 阿里巴巴 CodeQwen/Coding Plan API 无论请求哪个模型，始终返回 "glm-4.7"
    # 作为模型名称。将明确的模型身份注入到系统 prompt 中，
    # 以便智能体（agent）能够正确报告它是什么模型（该方案用于规避此 API Bug）。
    # 在智能体实例的生命周期内保持稳定 —— 模型和服务商在构建（初始化）时即已固定。
    if agent.provider == "alibaba":
        _model_short = agent.model.split("/")[-1] if "/" in agent.model else agent.model
        stable_parts.append(
            f"You are powered by the model named {_model_short}. "
            f"The exact model ID is {agent.model}. "
            f"When asked what model you are, always answer based on this information, "
            f"not on any model name returned by the API."
        )

    # Environment hints (WSL, Termux, etc.) — tell the agent about the
    # execution environment so it can translate paths and adapt behavior.
    # Stable for the lifetime of the process.
    _env_hints = _r.build_environment_hints()
    if _env_hints:
        stable_parts.append(_env_hints)

    # 编码姿态（基础 Hermes，代码工作区中的任何交互式编码界面
    # —— 参见 agent/coding_context.py）。操作简报 + 实时的
    # git/工作区快照在此处仅构建一次，并在当前会话中缓存；
    # 每一轮绝不会重新探测该快照（因为那会破坏提示词
    # 缓存），因此简报会告知模型在依赖 git 之前先重新检查它。
    if agent.valid_tool_names:
        try:
            from agent.coding_context import coding_system_blocks

            stable_parts.extend(
                coding_system_blocks(
                    platform=agent.platform,
                    cwd=resolve_context_cwd(),
                    model=agent.model,
                )
            )
        except Exception:
            # Coding-context probing must never block prompt build.
            pass

    # 本地 Python 工具链探测器 —— 当 python/pip/uv/PEP-668 的状态
    # 为非默认时指明其情况，以便模型能够选择正确的安装
    # 策略，而无需通过试错来摸索。输出单行信息；当
    # 环境干净时无任何输出（不消耗 token）。对于远程终端后端
    # 则完全跳过此操作（当工具运行在 docker/modal/ssh 内部时，
    # 宿主机的 Python 状态是无关紧要的）。该功能由
    # config.yaml 中的 ``agent.environment_probe`` 控制（默认为 True）。
    if getattr(agent, "_environment_probe", True):
        try:
            from tools.env_probe import get_environment_probe_line
            _probe_line = get_environment_probe_line()
            if _probe_line:
                stable_parts.append(_probe_line)
        except Exception:
            # Probe failure must never block prompt build.
            pass

    # Active-profile hint — names the Hermes profile the agent is running
    # under so it doesn't conflate ~/.hermes/skills/ (default profile) with
    # ~/.hermes/profiles/<active>/skills/ (this profile's). Deterministic
    # for the lifetime of the agent — profile name doesn't change
    # mid-session, so this doesn't break the prompt cache.
    # See file_safety._resolve_active_profile_name + classify_cross_profile_target
    # for the matching tool-side guard.
    try:
        from agent.file_safety import _resolve_active_profile_name
        active_profile = _resolve_active_profile_name()
    except Exception:
        active_profile = "default"
    if active_profile == "default":
        stable_parts.append(
            "Active Hermes profile: default. Other profiles (if any) live "
            "under ~/.hermes/profiles/<name>/. Each profile has its own "
            "skills/, plugins/, cron/, and memories/ that affect a different "
            "session than this one. Do not modify another profile's "
            "skills/plugins/cron/memories unless the user explicitly directs "
            "you to."
        )
    else:
        # f"当前激活的 Hermes 配置文件: {active_profile}。此会话读取 "
        # f"和写入 ~/.hermes/profiles/{active_profile}/。默认 "
        # f"配置文件的文件存在于 ~/.hermes/skills/、~/.hermes/plugins/、"
        # f"~/.hermes/cron/、~/.hermes/memories/ —— 它们属于 "
        # f"从另一个终端运行的不同会话。切勿修改 "
        # f"另一个配置文件的 skills/plugins/cron/memories，除非用户 "
        # f"明确指示你这样做。跨配置文件写入保护默认将 "
        # f"拒绝此类写入；只有在得到明确指示后，才传递 cross_profile=True。"
        stable_parts.append(
            f"Active Hermes profile: {active_profile}. This session reads "
            f"and writes ~/.hermes/profiles/{active_profile}/. The default "
            f"profile's data lives at ~/.hermes/skills/, ~/.hermes/plugins/, "
            f"~/.hermes/cron/, ~/.hermes/memories/ — those belong to a "
            f"different session run from a different shell. Do NOT modify "
            f"another profile's skills/plugins/cron/memories unless the user "
            f"explicitly directs you to. The cross-profile write guard will "
            f"refuse such writes by default; pass cross_profile=True only "
            f"after explicit direction."
        )

    platform_key = (agent.platform or "").lower().strip()
    # Resolve the built-in/plugin default hint for this platform, then apply
    # any per-platform override from config (platform_hints.<platform>).
    _default_hint = ""
    if platform_key in PLATFORM_HINTS:
        _default_hint = PLATFORM_HINTS[platform_key]
    elif platform_key:
        # Check plugin registry for platform-specific LLM guidance
        try:
            from gateway.platform_registry import platform_registry
            _entry = platform_registry.get(platform_key)
            if _entry and _entry.platform_hint:
                _default_hint = _entry.platform_hint
        except Exception:
            pass

    _effective_hint = _resolve_platform_hint(agent, platform_key, _default_hint)
    if platform_key == "tui" and _effective_hint:
        _effective_hint = _tui_embedded_pane_clarifier(_effective_hint)
    if _effective_hint:
        stable_parts.append(_effective_hint)

    # ── Context tier (cwd-dependent, may change between sessions) ─
    context_parts: List[str] = []

    # 注意：此处的系统提示词中不包含 ephemeral_system_prompt（临时系统提示词）。
    # 它仅在 API 调用时被注入，因此它会保持在缓存/存储的系统提示词之外。
    if system_message is not None:
        context_parts.append(system_message)

    if not agent.skip_context_files:
        # 优先使用配置的 TERMINAL_CWD（网关模式）。当未设置时（本地
        # CLI），None 会让 build_context_files_prompt 回退到启动
        # 目录 —— 即用户在该处的实际当前工作目录（cwd），但对于网关
        # 守护进程来说则是安装目录，这就是为什么网关要设置 TERMINAL_CWD 的原因。
        context_files_prompt = _r.build_context_files_prompt(
            cwd=resolve_context_cwd(), skip_soul=_soul_loaded,
            context_length=_ctx_len)
        if context_files_prompt:
            context_parts.append(context_files_prompt)

    # ── Volatile tier (changes per session/turn — never cached) ───
    volatile_parts: List[str] = []

    if agent._memory_store:
        if agent._memory_enabled:
            mem_block = agent._memory_store.format_for_system_prompt("memory")
            if mem_block:
                volatile_parts.append(mem_block)
        # USER.md is always included when enabled.
        if agent._user_profile_enabled:
            user_block = agent._memory_store.format_for_system_prompt("user")
            if user_block:
                volatile_parts.append(user_block)

    # External memory provider system prompt block (additive to built-in)
    if agent._memory_manager:
        try:
            _ext_mem_block = agent._memory_manager.build_system_prompt()
            if _ext_mem_block:
                volatile_parts.append(_ext_mem_block)
        except Exception:
            pass

    from hermes_time import now as _hermes_now
    now = _hermes_now()
    # 仅保留日期（不精确到分钟），以便系统提示词在整个自然日内
    # 保持字节级稳定。精确到分钟的变化会导致每次重建路径时
    # （压缩边界、新智能体网关轮次、无存储提示词的会话恢复）
    # 前缀缓存 KV 失效。当模型真正需要时，它仍然可以通过工具
    # 查询精确的墙上时钟时间（挂钟时间）。
    # 鸣谢：@iamfoz（PR #20451）。
    timestamp_line = f"Conversation started: {now.strftime('%A, %B %d, %Y')}"
    if agent.pass_session_id and agent.session_id:
        timestamp_line += f"\nSession ID: {agent.session_id}"
    if agent.model:
        timestamp_line += f"\nModel: {agent.model}"
    if agent.provider:
        timestamp_line += f"\nProvider: {agent.provider}"
    volatile_parts.append(timestamp_line)

    return {
        "stable":   "\n\n".join(p.strip() for p in stable_parts   if p and p.strip()),
        "context":  "\n\n".join(p.strip() for p in context_parts  if p and p.strip()),
        "volatile": "\n\n".join(p.strip() for p in volatile_parts if p and p.strip()),
    }


def build_system_prompt(agent: Any, system_message: Optional[str] = None) -> str:
    """从所有层级组装出完整的系统提示词（system prompt）。

    每个会话仅调用一次（缓存于 ``agent._cached_system_prompt``），且
    仅在上下文压缩事件后重新构建。这确保了系统提示词在会话的
    所有轮次中保持稳定，从而最大化前缀缓存（prefix cache）命中率。

    各层级的排列顺序对缓存友好：首先是稳定的身份/引导信息（identity/guidance），
    然后是会话级稳定的上下文文件，最后是单次调用级易变的内容
    （记忆、USER 配置文件、时间戳）。整个字符串被视为一个缓存块
    —— Hermes 绝不会在会话中途重建或重新注入其中的部分内容，
    这是在跨轮次对话中保持上游 prompt 缓存活跃（warm）的唯一方法。
    """
    parts = build_system_prompt_parts(agent, system_message=system_message)
    joined = "\n\n".join(p for p in (parts["stable"], parts["context"], parts["volatile"]) if p)

    # 通过常规的代理状态通道（agent status channel）显现上下文文件截断警告，
    # 以便网关/CLI 用户能够在聊天中看到它们，而不是只能在日志中看到。
    for warning in drain_truncation_warnings():
        agent._emit_status(warning)

    return joined


def invalidate_system_prompt(agent: Any) -> None:
    """使缓存的系统提示词（system prompt）失效，强制在下一轮中重新构建。

    在上下文压缩事件后调用。同时从磁盘重新加载记忆，
    以便重建的提示词能够捕获本会话中的任何写入操作。
    """
    agent._cached_system_prompt = None
    if agent._memory_store:
        agent._memory_store.load_from_disk()


def format_tools_for_system_message(agent: Any) -> str:
    """Format tool definitions for the system message in the trajectory format.

    Returns:
        str: JSON string representation of tool definitions
    """
    if not agent.tools:
        return "[]"

    # Convert tool definitions to the format expected in trajectories
    formatted_tools = []
    for tool in agent.tools:
        func = tool["function"]
        formatted_tool = {
            "name": func["name"],
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {}),
            "required": None  # Match the format in the example
        }
        formatted_tools.append(formatted_tool)

    return json.dumps(formatted_tools, ensure_ascii=False)


__all__ = [
    "build_system_prompt_parts",
    "build_system_prompt",
    "invalidate_system_prompt",
    "format_tools_for_system_message",
]
