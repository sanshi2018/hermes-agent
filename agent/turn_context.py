"""
``run_conversation`` 的每轮次设置（即轮次序言）。

在工具调用循环真正开始之前，``run_conversation`` 原本会先执行
约 470 行线性的准备逻辑：stdio 保护、runtime-main 接线、
重试计数器重置、用户消息清洗、todo / nudge 计数器水合、
系统提示词恢复或构建、崩溃恢复持久化、预检上下文压缩、
``pre_llm_call`` 插件钩子，以及外部记忆预取。

所有这些都属于“序言”——
它们每轮只运行一次，不会反向引用循环内部逻辑，
并且会产出一组固定的值，供后续循环消费。

``TurnContext`` 会捕获这些产出的值；
``build_turn_context`` 则负责执行设置工作并返回一个上下文对象。
这样，``run_conversation`` 只需解包该上下文并运行循环，
从而把完整的序言部分从编排器中瘦身出去。

该构建器仍会像原来的内联代码一样，
大量修改 ``agent``（计数器、线程 ID、缓存提示词、会话数据库等）；
这些副作用正是这段设置逻辑的目的。

它返回的 ``TurnContext`` 只携带循环后续会读回的局部变量。

行为与原来的内联序言完全一致；
这只是一次纯粹的“移动并命名”式重构，
不包含任何语义变化。
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.conversation_compression import conversation_history_after_compression
from agent.iteration_budget import IterationBudget
from agent.model_metadata import (
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
)

logger = logging.getLogger(__name__)


def _compression_made_progress(
    orig_len: int, new_len: int, orig_tokens: int, new_tokens: int
) -> bool:
    """Return ``True`` if a compression pass materially reduced the request.

    Compression can succeed by summarising message contents — reducing the
    estimated request token count — without reducing the message row
    count.  Treating row count as the sole progress signal false-positives
    on size-only wins and surfaces a misleading "Cannot compress further"
    failure even when post-compression tokens are well below the model
    context window.  See issue #39548 for an observed case: 220 → 220
    messages, ~288k → ~183k tokens on a 1M-context model still triggered
    auto-reset.

    The token reduction must be *material* (>5%) to count as progress — the
    same floor the overflow-handler retry path uses (conversation_loop.py,
    #39550) — so a sub-5% wobble doesn't keep the multi-pass loop spinning.
    """
    if new_len < orig_len:
        return True
    return orig_tokens > 0 and new_tokens < orig_tokens * 0.95


def _should_run_preflight_estimate(
    messages: List[Dict[str, Any]],
    protect_first_n: int,
    protect_last_n: int,
    threshold_tokens: int,
) -> bool:
    """Cheap gate for the (expensive) full preflight token estimate.

    Returns ``True`` when either:
      (a) message count exceeds the protected ranges (the historical gate), or
      (b) a cheap char-based estimate already crosses the configured threshold
          — the few-but-huge case from issue #27405 that the count-only gate
          would silently skip (a handful of very large messages never trips
          the count condition, so compression was never attempted and the
          turn hit a hard context-overflow error).

    Branch (b) uses ``estimate_messages_tokens_rough`` (the shared char-based
    estimator) so a single large base64 image isn't mistaken for ~250K tokens.
    It intentionally undercounts vs. the full request estimate — it omits the
    system prompt and tool schemas — because it is only a *hint* deciding
    whether to pay for the authoritative ``estimate_request_tokens_rough``,
    which (together with ``should_compress``) makes the real decision.
    """
    if len(messages) > protect_first_n + protect_last_n + 1:
        return True
    return estimate_messages_tokens_rough(messages) >= threshold_tokens


@dataclass
class TurnContext:
    """由轮次序言生成，并由轮次循环消费的值。"""

    # 已清洗的入站消息（已移除代理项字符）。
    user_message: str
    # 为转录记录 / 记忆查询保留的干净消息（未注入 nudge）。
    original_user_message: Any
    # 当前轮次使用的工作消息列表（循环会向其中追加内容）。
    messages: List[Dict[str, Any]]
    # 可能会被预检压缩重置为 None（表示已创建新会话）。
    conversation_history: Optional[List[Dict[str, Any]]]
    # 当前轮次生效的缓存系统提示词（可能会被压缩流程重建）。
    active_system_prompt: Optional[str]
    # 任务 / 轮次标识符。
    effective_task_id: str
    turn_id: str
    # 当前用户轮次在 ``messages`` 中的索引。
    current_turn_user_idx: int
    # 是否应触发轮次后的记忆审查。
    should_review_memory: bool = False
    # 由 ``pre_llm_call`` 插件提供的上下文（会追加到用户消息中）。
    plugin_user_context: str = ""
    # 外部记忆预取结果，会在多次循环迭代之间复用。
    ext_prefetch_cache: str = ""


def build_turn_context(
    agent,
    user_message: str,
    system_message: Optional[str],
    conversation_history: Optional[List[Dict[str, Any]]],
    task_id: Optional[str],
    stream_callback,
    persist_user_message: Optional[str],
    persist_user_timestamp: Optional[float] = None,
    *,
    restore_or_build_system_prompt,
    install_safe_stdio,
    sanitize_surrogates,
    summarize_user_message_for_log,
    set_session_context,
    set_current_write_origin,
    ra,
) -> TurnContext:
    """
    执行每轮一次的设置，并返回供循环使用的输入上下文。

    原始序言中引用自 ``conversation_loop`` 模块的可调用对象 / 辅助函数
    会被显式传入，
    以避免本模块与 ``agent.conversation_loop`` 之间形成导入环。
    """

    # 保护 stdio，避免因管道断开而抛出 OSError
    # （systemd / 无头环境 / 守护进程场景）。
    install_safe_stdio()

    # 注意：DB 会话行会在稍后创建，
    # 也就是在系统提示词被恢复 / 构建之后
    # （见系统提示词代码块下方的 _ensure_db_session()）。
    #
    # 如果在这里创建——也就是在 _cached_system_prompt 填充之前创建——
    # 那么对于携带客户端托管历史的全新 API / 网关 agent，
    # 会插入一行 system_prompt=NULL 的记录。
    # 这随后会触发“已存储的系统提示词为空；从头重建”的警告，
    # 并导致第一次轮次出现不必要的前缀缓存未命中。
    # （Issue #45499。）
    #
    # 告诉 auxiliary_client 当前轮次实际使用的主提供方 / 模型。
    try:
        from agent.auxiliary_client import set_runtime_main
        set_runtime_main(
            getattr(agent, "provider", "") or "",
            getattr(agent, "model", "") or "",
            base_url=getattr(agent, "base_url", "") or "",
            api_key=getattr(agent, "api_key", "") or "",
            api_mode=getattr(agent, "api_mode", "") or "",
        )
    except Exception:
        pass

    # Tag log records on this thread with the session ID for ``hermes logs``.
    set_session_context(agent.session_id)

    # Bind the skill write-origin ContextVar for this thread.
    set_current_write_origin(getattr(agent, "_memory_write_origin", "assistant_tool"))

    # Restore the primary runtime if the previous turn activated fallback.
    agent._restore_primary_runtime()

    # 轮次之间的 MCP 刷新：
    # 如果某个 MCP 服务器在上一轮之后才完成连接
    # （慢速 HTTP / OAuth 服务器在冷启动连接时通常需要 2–6 秒，
    #  因而可能错过有界的启动等待），
    # 那么它会进入“当前这一轮”的工具快照。
    #
    # 该设计天然是缓存安全的：
    # 它运行在每轮序言中，
    # 位于当前轮次首次 API 调用组装 ``tools=`` 之前；
    # 因此它只会扩展一个新的请求前缀，
    # 绝不会修改正在进行的轮次所使用的缓存前缀。
    #
    # 在没有注册 MCP 服务器时不执行任何操作
    # （这是常见情况，由廉价的 ``has_registered_mcp_tools`` 检查保护）；
    # 或者当工具集未变化时也不执行任何操作
    # （``refresh_agent_mcp_tools`` 会按名称进行差异比较，
    #  并在没有变化时保持快照不变）。
    try:
        if not getattr(agent, "_skip_mcp_refresh", False):
            # 导入成本门控：
            # ``tools.mcp_tool`` 会拉入整个 ``mcp`` 包
            # （实测约 0.4 秒），
            # 即使用户没有配置任何 MCP 服务器也是如此。
            #
            # MCP 工具只能由已经导入 ``tools.mcp_tool`` 的代码注册
            # （例如发现流程、/reload-mcp、后期绑定刷新）；
            # 因此，如果它尚未出现在 sys.modules 中，
            # 就说明没有任何内容需要刷新，
            # 也就可以直接跳过这次导入。
            #
            # 这样可以让未使用 MCP 的首次轮次
            # 避开沉重的导入路径，
            # 同时不改变 MCP 用户的行为。
            import sys as _sys
            if "tools.mcp_tool" in _sys.modules:
                from tools.mcp_tool import has_registered_mcp_tools, refresh_agent_mcp_tools
                if has_registered_mcp_tools():
                    refresh_agent_mcp_tools(agent, quiet_mode=True)
    except Exception:
        logger.debug("between-turns MCP tool refresh skipped", exc_info=True)

    # Sanitize surrogate characters from user input.
    if isinstance(user_message, str):
        user_message = sanitize_surrogates(user_message)
    if isinstance(persist_user_message, str):
        persist_user_message = sanitize_surrogates(persist_user_message)

    # Store stream callback for _interruptible_api_call to pick up.
    agent._stream_callback = stream_callback
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = persist_user_message
    agent._persist_user_message_timestamp = persist_user_timestamp
    # Generate unique task_id if not provided to isolate VMs between tasks.
    effective_task_id = task_id or str(uuid.uuid4())
    agent._current_task_id = effective_task_id
    turn_id = f"{agent.session_id or 'session'}:{effective_task_id}:{uuid.uuid4().hex[:8]}"
    agent._current_turn_id = turn_id
    agent._current_api_request_id = ""

    # Reset retry counters and iteration budget at the start of each turn.
    agent._invalid_tool_retries = 0
    agent._invalid_json_retries = 0
    agent._empty_content_retries = 0
    agent._incomplete_scratchpad_retries = 0
    agent._codex_incomplete_retries = 0
    agent._thinking_prefill_retries = 0
    agent._post_tool_empty_retried = False
    agent._last_content_with_tools = None
    agent._last_content_tools_all_housekeeping = False
    agent._mute_post_response = False
    agent._unicode_sanitization_passes = 0
    agent._tool_guardrails.reset_for_turn()
    agent._tool_guardrail_halt_decision = None
    _reset_consol = getattr(agent._memory_store, "reset_consolidation_failures", None)
    if callable(_reset_consol):
        _reset_consol()
    agent._vision_supported = True

    # Pre-turn connection health check: clean up dead TCP connections.
    if agent.api_mode != "anthropic_messages":
        try:
            if agent._cleanup_dead_connections():
                agent._emit_status(
                    "🔌 Detected stale connections from a previous provider "
                    "issue — cleaned up automatically. Proceeding with fresh "
                    "connection."
                )
        except Exception:
            pass
    # Replay compression warning through status_callback for gateway platforms.
    if agent._compression_warning:
        agent._replay_compression_warning()
        agent._compression_warning = None  # send once

    # NOTE: _turns_since_memory and _iters_since_skill are NOT reset here.
    agent.iteration_budget = IterationBudget(agent.max_iterations)

    # Log conversation turn start for debugging/observability.
    _preview_text = summarize_user_message_for_log(user_message)
    _msg_preview = (_preview_text[:80] + "...") if len(_preview_text) > 80 else _preview_text
    _msg_preview = _msg_preview.replace("\n", " ")
    logger.info(
        "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
        agent.session_id or "none", agent.model, agent.provider or "unknown",
        agent.platform or "unknown", len(conversation_history or []),
        _msg_preview,
    )

    # Initialize conversation (copy to avoid mutating the caller's list).
    messages = list(conversation_history) if conversation_history else []

    # 从对话历史记录中
    # 填充待办事项存储。
    if conversation_history and not agent._todo_store.has_items():
        agent._hydrate_todo_store(conversation_history)

    # 从持久化的历史记录中
    # 填充每个会话的微调/提示计数器（Issue #22357）。
    if conversation_history and agent._user_turn_count == 0:
        prior_user_turns = sum(
            1 for m in conversation_history if m.get("role") == "user"
        )
        if prior_user_turns > 0:
            agent._user_turn_count = prior_user_turns
            if agent._memory_nudge_interval > 0 and agent._turns_since_memory == 0:
                agent._turns_since_memory = prior_user_turns % agent._memory_nudge_interval

    # Track user turns for memory flush and periodic nudge logic.
    agent._user_turn_count += 1
    # Copilot x-initiator：用户本轮对话的首次 API 调用
    # 由用户发起；工具循环中的后续调用则重置为 "agent"（#3040）。
    agent._is_user_initiated_turn = True

    # Reset the streaming context scrubber at the top of each turn.
    scrubber = getattr(agent, "_stream_context_scrubber", None)
    if scrubber is not None:
        scrubber.reset()
    # Reset the think scrubber for the same reason.
    think_scrubber = getattr(agent, "_stream_think_scrubber", None)
    if think_scrubber is not None:
        think_scrubber.reset()

    # Preserve the original user message (no nudge injection).
    original_user_message = persist_user_message if persist_user_message is not None else user_message

    # Track memory nudge trigger (turn-based, checked here).
    should_review_memory = False
    if (agent._memory_nudge_interval > 0
            and "memory" in agent.valid_tool_names
            and agent._memory_store):
        agent._turns_since_memory += 1
        if agent._turns_since_memory >= agent._memory_nudge_interval:
            should_review_memory = True
            agent._turns_since_memory = 0

    # Add user message.
    user_msg = {"role": "user", "content": user_message}
    messages.append(user_msg)
    current_turn_user_idx = len(messages) - 1
    agent._persist_user_message_idx = current_turn_user_idx

    # 装饰性的辅助信号：检测表达喜爱的“回应”（ily / <3 / good bot），
    # 并通知宿主，以便播放爱心动画。
    # 该机制不消耗令牌、从不影响对话，也绝不会导致致命错误——
    # 仅仅是一个完全可选的 UI 小效果。
    reaction_callback = getattr(agent, "reaction_callback", None)
    if reaction_callback is not None:
        try:
            from agent.reactions import detect_reaction

            kind = detect_reaction(original_user_message)
            if kind:
                reaction_callback(kind)
        except Exception:
            pass

    if not agent.quiet_mode:
        _print_preview = summarize_user_message_for_log(user_message)
        agent._safe_print(
            f"💬 Starting conversation: '{_print_preview[:60]}"
            f"{'...' if len(_print_preview) > 60 else ''}'"
        )

    # ── System prompt (cached per session for prefix caching) ──
    if agent._cached_system_prompt is None:
        restore_or_build_system_prompt(agent, system_message, conversation_history)

    active_system_prompt = agent._cached_system_prompt

    # 现在创建数据库会话记录，因为此时 _cached_system_prompt 已完成填充，
    # 从而确保首次对话写入的持久化快照不是 NULL（Issue #45499）。
    #
    # 该操作具备幂等性：一旦记录已存在，
    # _ensure_db_session() 将不执行任何操作。
    agent._ensure_db_session()

    # 崩溃恢复机制：会话记录一旦存在，便立即持久化写入本轮收到的用户消息。
    try:
        agent._persist_session(messages, conversation_history)
    except Exception:
        logger.warning(
            "Early turn-start session persistence failed for session=%s",
            agent.session_id or "none",
            exc_info=True,
        )

    # ── 上下文压缩预检 ──
    # 先通过低开销的预检查进行筛选，再执行成本较高的完整令牌数估算。
    # 有关修复问题 #27405 的“或”逻辑语义，请参阅
    # ``_should_run_preflight_estimate``。
    # 该问题会导致少量超大消息绕过消息数量检查。
    if agent.compression_enabled and _should_run_preflight_estimate(
        messages,
        agent.context_compressor.protect_first_n,
        agent.context_compressor.protect_last_n,
        agent.context_compressor.threshold_tokens,
    ):
        _preflight_tokens = estimate_request_tokens_rough(
            messages,
            system_prompt=active_system_prompt or "",
            tools=agent.tools or None,
        )
        _compressor = agent.context_compressor
        _defer_preflight = getattr(
            _compressor,
            "should_defer_preflight_to_real_usage",
            lambda _tokens: False,
        )
        _preflight_deferred = _defer_preflight(_preflight_tokens)
        # Codex app-server threads are compacted by the codex agent itself;
        # Hermes only initiates compaction in "hermes" mode (#36801).
        _codex_native_auto = (
            getattr(agent, "api_mode", None) == "codex_app_server"
            and str(
                getattr(
                    agent,
                    "codex_app_server_auto_compaction",
                    "native",
                )
                or "native"
            ).lower()
            in {"native", "off"}
        )

        if not _preflight_deferred:
            _last = _compressor.last_prompt_tokens
            # Do NOT overwrite the -1 sentinel (#36718).
            if _last >= 0 and _preflight_tokens > _last:
                _compressor.last_prompt_tokens = _preflight_tokens

        _compression_cooldown = getattr(
            _compressor,
            "get_active_compression_failure_cooldown",
            lambda: None,
        )()

        if _preflight_deferred:
            logger.info(
                "Skipping preflight compression: rough estimate ~%s >= %s, "
                "but last real provider prompt was %s after compression",
                f"{_preflight_tokens:,}",
                f"{_compressor.threshold_tokens:,}",
                f"{_compressor.last_real_prompt_tokens:,}",
            )
        elif _compression_cooldown:
            logger.info(
                "Skipping preflight compression: same-session cooldown active "
                "(~%s seconds remaining, session %s)",
                int(_compression_cooldown.get("remaining_seconds", 0.0)),
                agent.session_id or "none",
            )
        elif _codex_native_auto:
            logger.info(
                "Skipping Hermes preflight compression for codex app-server "
                "(mode=%s); Hermes will not start thread compaction here.",
                getattr(agent, "codex_app_server_auto_compaction", "native"),
            )
        elif _compressor.should_compress(_preflight_tokens):
            logger.info(
                "Preflight compression: ~%s tokens >= %s threshold (model %s, ctx %s)",
                f"{_preflight_tokens:,}",
                f"{_compressor.threshold_tokens:,}",
                agent.model,
                f"{_compressor.context_length:,}",
            )
            agent._emit_status(
                f"📦 Preflight compression: ~{_preflight_tokens:,} tokens "
                f">= {_compressor.threshold_tokens:,} threshold. "
                "This may take a moment."
            )
            for _pass in range(3):
                _orig_len = len(messages)
                _orig_tokens = _preflight_tokens
                messages, active_system_prompt = agent._compress_context(
                    messages, system_message, approx_tokens=_preflight_tokens,
                    task_id=effective_task_id,
                )
                # Re-estimate now so size-only compression (same row count,
                # lower token count — e.g. summarising tool outputs) is
                # recognised as progress instead of being misread as
                # "Cannot compress further". Fixes #39548.
                _preflight_tokens = estimate_request_tokens_rough(
                    messages,
                    system_prompt=active_system_prompt or "",
                    tools=agent.tools or None,
                )
                if not _compression_made_progress(
                    _orig_len, len(messages), _orig_tokens, _preflight_tokens
                ):
                    break  # Cannot compress further: neither rows nor tokens moved
                conversation_history = conversation_history_after_compression(
                    agent, messages
                )
                agent._empty_content_retries = 0
                agent._thinking_prefill_retries = 0
                agent._last_content_with_tools = None
                agent._last_content_tools_all_housekeeping = False
                agent._mute_post_response = False
                if not _compressor.should_compress(_preflight_tokens):
                    break

    # TODO KEY 通过hook从插件中召回记忆
    # Plugin hook: pre_llm_call (context injected into user message, not system prompt).
    plugin_user_context = ""
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        _pre_results = _invoke_hook(
            "pre_llm_call",
            session_id=agent.session_id,
            task_id=effective_task_id,
            turn_id=turn_id,
            user_message=original_user_message,
            conversation_history=list(messages),
            is_first_turn=(not bool(conversation_history)),
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
            sender_id=getattr(agent, "_user_id", None) or "",
        )
        _ctx_parts: list[str] = []
        # 将过大的单 Hook 上下文转储至磁盘，
        # 避免失控的插件撑大后续每一轮对话的 Prompt。
        # 移植自 openai/codex PR #21069（“从上下文中转储大型 Hook 输出”）。
        try:
            from tools.hook_output_spill import (
                get_spill_config as _spill_cfg,
                spill_if_oversized as _spill_if_oversized,
            )
            _spill_config_cached = _spill_cfg()
        except Exception:
            _spill_if_oversized = None  # type: ignore[assignment]
            _spill_config_cached = None
        for r in _pre_results:
            _piece: str = ""
            if isinstance(r, dict) and r.get("context"):
                _piece = str(r["context"])
            elif isinstance(r, str) and r.strip():
                _piece = r
            else:
                continue
            if _spill_if_oversized is not None:
                try:
                    _piece = _spill_if_oversized(
                        _piece,
                        session_id=agent.session_id,
                        source="plugin hook",
                        config=_spill_config_cached,
                    )
                except Exception as _spill_exc:
                    logger.warning("hook context spill failed: %s", _spill_exc)
            _ctx_parts.append(_piece)
        if _ctx_parts:
            plugin_user_context = "\n\n".join(_ctx_parts)
    except Exception as exc:
        logger.warning("pre_llm_call hook failed: %s", exc)

    # Per-turn file-mutation verifier state.
    agent._turn_failed_file_mutations = {}
    agent._turn_file_mutation_paths = set()
    agent._verification_stop_nudges = 0
    agent._pre_verify_nudges = 0

    # Record the execution thread so interrupt()/clear_interrupt() can scope
    # the tool-level interrupt signal to THIS agent's thread only.
    agent._execution_thread_id = threading.current_thread().ident

    # Clear stale per-thread interrupt state, preserving a pending interrupt.
    ra()._set_interrupt(False, agent._execution_thread_id)
    if agent._interrupt_requested:
        ra()._set_interrupt(True, agent._execution_thread_id)
        agent._interrupt_thread_signal_pending = False
    else:
        agent._interrupt_message = None
        agent._interrupt_thread_signal_pending = False

    # Notify memory providers of the new turn (BEFORE prefetch_all).
    if agent._memory_manager:
        try:
            _turn_msg = original_user_message if isinstance(original_user_message, str) else ""
            agent._memory_manager.on_turn_start(agent._user_turn_count, _turn_msg)
        except Exception:
            pass

    # External memory provider: prefetch once before the tool loop.
    ext_prefetch_cache = ""
    if agent._memory_manager:
        try:
            _query = original_user_message if isinstance(original_user_message, str) else ""
            ext_prefetch_cache = agent._memory_manager.prefetch_all(_query) or ""
        except Exception:
            pass

    return TurnContext(
        user_message=user_message,
        original_user_message=original_user_message,
        messages=messages,
        conversation_history=conversation_history,
        active_system_prompt=active_system_prompt,
        effective_task_id=effective_task_id,
        turn_id=turn_id,
        current_turn_user_idx=current_turn_user_idx,
        should_review_memory=should_review_memory,
        plugin_user_context=plugin_user_context,
        ext_prefetch_cache=ext_prefetch_cache,
    )
