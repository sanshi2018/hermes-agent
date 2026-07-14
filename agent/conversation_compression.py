"""Context compression — extract the AIAgent methods that drive summarisation.

Three concerns live here:

* :func:`check_compression_model_feasibility` — startup probe of the
  configured auxiliary compression model.  Warns when the aux context
  window can't fit the main model's compression threshold; auto-lowers
  the session threshold when possible; hard-rejects auxes below
  ``MINIMUM_CONTEXT_LENGTH``.

* :func:`replay_compression_warning` — re-emit a stored warning through
  the gateway ``status_callback`` once it's wired up (the callback is
  set after :class:`AIAgent` construction).

* :func:`compress_context` — the actual compression call.  Runs the
  configured compressor, splits the SQLite session, rotates the
  session_id, notifies plugin context engines / memory providers, and
  returns the compressed message list and active system prompt.

* :func:`try_shrink_image_parts_in_messages` — image-too-large recovery
  helper that re-encodes ``data:image/...;base64,...`` parts at a smaller
  size so retries can fit under provider ceilings (Anthropic's 5 MB).

``run_agent`` keeps thin wrappers for each so existing call sites
(``self._compress_context(...)``) keep working.  Tests that exercise
these paths see no behavioural change.
"""

from __future__ import annotations

import inspect
import logging
import os
import tempfile
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

from agent.context_engine import sanitize_memory_context
from agent.model_metadata import estimate_request_tokens_rough

logger = logging.getLogger(__name__)

# Stable marker the gateway matches on to re-tag the auto-compaction lifecycle
# status as ``kind="compacting"`` (tui_gateway/server.py::_status_update), so
# drivers like the desktop app can show an explicit "Summarizing…" indicator
# instead of the transcript appearing to silently reset. Keep the marker phrase
# intact if you reword COMPACTION_STATUS.
COMPACTION_STATUS_MARKER = "Compacting context"
COMPACTION_STATUS = (
    f"🗜️ {COMPACTION_STATUS_MARKER} — summarizing earlier conversation so I can continue..."
)


def _builtin_memory_prompt_snapshot(agent: Any) -> Optional[Tuple[str, str]]:
    """Return the built-in memory text that can affect a system prompt.

    ``MemoryStore`` freezes this text until ``load_from_disk()``.  Rendering
    the frozen blocks after that reload lets compression retain the exact
    cached system prompt when it already embeds the current memory (see
    :func:`_cached_prompt_reflects_builtin_memory`).  An unreadable snapshot
    returns ``None`` so callers take the conservative rebuild path.
    """
    store = getattr(agent, "_memory_store", None)
    if store is None:
        return "", ""
    try:
        memory = (
            store.format_for_system_prompt("memory") or ""
            if getattr(agent, "_memory_enabled", False)
            else ""
        )
        user = (
            store.format_for_system_prompt("user") or ""
            if getattr(agent, "_user_profile_enabled", False)
            else ""
        )
    except Exception:
        return None
    return memory, user


def _cached_prompt_reflects_builtin_memory(agent: Any, cached_prompt: str) -> bool:
    """Whether the cached system prompt already embeds current built-in memory.

    The retention fast path must NOT compare the memory snapshot before vs
    after the disk reload: on fresh-agent surfaces (gateway, TUI) the cached
    prompt is restored from the session DB and can predate mid-session memory
    writes that the fresh ``MemoryStore`` already picked up at init — the
    snapshot is then identical on both sides of the reload while the prompt
    itself is stale, and retaining it would latch old memory for the life of
    the session (and re-persist it via ``update_system_prompt``).

    Instead, verify the CURRENT (post-reload) rendered blocks appear verbatim
    in the cached prompt, and that no leftover block header remains for a
    target whose entries have since been emptied or disabled.
    """
    snapshot = _builtin_memory_prompt_snapshot(agent)
    if snapshot is None:
        return False
    try:
        from tools.memory_tool import MEMORY_BLOCK_HEADERS
    except Exception:
        return False
    for target, block in zip(("memory", "user"), snapshot):
        block = block.strip()
        if block:
            # build_system_prompt_parts embeds the stripped block verbatim;
            # the rendered text includes the usage header, so any entry
            # change (or char-count change) breaks containment → rebuild.
            if block not in cached_prompt:
                return False
        elif MEMORY_BLOCK_HEADERS[target] in cached_prompt:
            # The prompt still carries a block for a target that is now
            # empty/disabled — stale; rebuild.
            return False
    return True


def _lock_api_is_absent_on_session_db(lock_db: Any) -> bool:
    """Whether the live in-memory SessionDB class structurally predates locks.

    In the supported hot-reload skew, this module is new while the already
    imported ``hermes_state.SessionDB`` class (and its live instances) is old.
    Only that exact class identity may fail open. Proxies, nominal lookalikes,
    non-callables, and descriptor failures must fail closed. Static lookup
    avoids invoking a present-but-broken descriptor.
    """
    try:
        from hermes_state import SessionDB

        missing = object()
        return (
            type(lock_db) is SessionDB
            and inspect.getattr_static(
                SessionDB, "try_acquire_compression_lock", missing
            ) is missing
        )
    except Exception:
        return False


def _refresh_persisted_compression_guards(compressor: Any) -> None:
    """Refresh durable automatic-compression guards on a built-in compressor."""
    method_calls = (
        ("get_active_compression_failure_cooldown", {"refresh": True}),
        ("_load_fallback_compression_streak", {}),
    )
    for method_name, kwargs in method_calls:
        method = getattr(type(compressor), method_name, None)
        if not callable(method):
            continue
        try:
            method(compressor, **kwargs)
        except Exception as exc:
            logger.debug("compression guard refresh failed (%s): %s", method_name, exc)


def _session_was_rotated_by_compression(session_db: Any, session_id: str) -> bool:
    """Return whether another path already rotated this compression parent."""
    getter = getattr(type(session_db), "get_session", None)
    if not callable(getter):
        return False
    session = getter(session_db, session_id)
    return bool(
        session
        and session.get("ended_at") is not None
        and session.get("end_reason") == "compression"
    )


def _compression_lock_holder(agent: Any) -> str:
    """Build a unique holder id for the lock: pid:tid:agent-instance:uuid.

    The pid+tid prefix lets ops tell crashed/abandoned holders apart from
    live ones (expiry-based recovery uses the timestamp, but ``holder``
    is what shows up in diagnostics + log lines). The agent instance id
    and a per-acquire uuid disambiguate two co-resident agents on the
    same thread (background_review forks run on a worker thread, but
    on machines where compression itself dispatches to a thread pool
    we want each acquire to be unique).
    """
    import threading
    return (
        f"pid={os.getpid()}"
        f":tid={threading.get_ident()}"
        f":agent={id(agent):x}"
        f":nonce={uuid.uuid4().hex[:8]}"
    )


def _supported_compression_kwargs(
    compress_fn: Any,
    *,
    current_tokens: Optional[int],
    focus_topic: Optional[str],
    force: bool,
    memory_context: str,
) -> dict:
    """Return only compression kwargs accepted by an engine callable.

    Context-engine plugins can outlive additions to the optional host contract.
    Inspecting the callable before invoking it keeps those older signatures
    compatible without catching an internal ``TypeError`` and executing a
    stateful compressor twice.
    """
    candidates = {
        "current_tokens": current_tokens,
        "focus_topic": focus_topic,
        "force": force,
    }
    if memory_context:
        candidates["memory_context"] = memory_context
    try:
        parameters = inspect.signature(compress_fn).parameters
    except (TypeError, ValueError):
        # ``current_tokens`` has been part of the ContextEngine ABC since its
        # introduction. Keep the oldest documented call shape when a C-backed
        # or otherwise opaque callable has no inspectable signature.
        return {"current_tokens": current_tokens}

    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_kwargs:
        return candidates
    return {name: value for name, value in candidates.items() if name in parameters}


class _CompressionLockLeaseRefresher:
    def __init__(
        self,
        db: Any,
        session_id: str,
        holder: str,
        ttl_seconds: float,
        refresh_interval_seconds: float | None = None,
    ) -> None:
        self._db = db
        self._session_id = session_id
        self._holder = holder
        self._ttl_seconds = ttl_seconds
        if refresh_interval_seconds is None:
            refresh_interval_seconds = max(1.0, min(60.0, ttl_seconds / 2.0))
        self._refresh_interval_seconds = max(0.1, float(refresh_interval_seconds))
        # Tolerate transient refresh failures for at most one lease's worth of
        # time, so the give-up window is genuinely bounded by the TTL the
        # acquirer set (a single blip recovers on the next tick; a persistent
        # failure stops before the lease could outlive its TTL). Floor of 1 so a
        # degenerate interval >= ttl still tolerates one blip.
        self._max_consecutive_failures = max(
            1, int(self._ttl_seconds / self._refresh_interval_seconds)
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="compression-lock-refresh",
            daemon=True,
        )

    def start(self) -> "_CompressionLockLeaseRefresher":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        # join() may time out while the refresher is mid-UPDATE; that's safe —
        # it's a daemon thread, and a late refresh on an already-released lock
        # matches rowcount 0 (a no-op). stop() returning does not guarantee the
        # thread has fully quiesced, only that we've signalled it and waited
        # briefly.
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        # A single falsy refresh must NOT permanently kill the lease: a
        # transient DB blip (write contention escaping _execute_write's retry
        # budget, a momentary "database is locked") returns False just like a
        # genuine lost-ownership, but only the latter should stop the loop.
        # Tolerate consecutive failures for at most one lease's worth of time
        # (_max_consecutive_failures = ttl / interval), so a one-off blip
        # recovers on the next tick while the total give-up window stays bounded
        # by the TTL the acquirer set — the lock can never be held past its TTL
        # by a stuck refresher.
        consecutive_failures = 0
        while not self._stop.wait(self._refresh_interval_seconds):
            try:
                refreshed = self._db.refresh_compression_lock(
                    self._session_id,
                    self._holder,
                    ttl_seconds=self._ttl_seconds,
                )
            except Exception as exc:
                logger.debug("compression lock refresh raised: %s", exc)
                refreshed = False
            if refreshed:
                consecutive_failures = 0
                continue
            consecutive_failures += 1
            if consecutive_failures >= self._max_consecutive_failures:
                logger.debug(
                    "compression lock refresh failed %d times in a row; "
                    "stopping lease refresher for session %s",
                    consecutive_failures, self._session_id,
                )
                break


def check_compression_model_feasibility(agent: Any) -> None:
    """如果在会话开始时，辅助压缩模型的上下文窗口小于主模型的压缩阈值，则发出警告。

    当辅助模型无法容纳需要被总结的内容时，压缩操作要么会直接失败（LLM 调用报错），
    要么会生成一个被严重截断的总结。

    该函数在 ``AIAgent.__init__`` 期间被调用，以便 CLI 用户能够立即（通过 ``_vprint``）
    看到该警告。网关（gateway）是在构造函数执行 *之后* 才设置 ``status_callback`` 的，
    因此 :func:`replay_compression_warning` 会在第一次调用 ``run_conversation()`` 时，
    通过回调函数重新发送存储的警告。
    """
    if not agent.compression_enabled:
        return
    try:
        from agent.auxiliary_client import (
            _resolve_task_provider_model,
            _try_configured_fallback_for_unavailable_client,
            get_text_auxiliary_client,
        )
        from agent.model_metadata import (
            MINIMUM_CONTEXT_LENGTH,
            get_model_context_length,
        )

        # Best-effort aux provider label for the warning message. The
        # configured provider may be "auto", in which case we fall back
        # to the client's base_url hostname so the user can still tell
        # where the compression model is actually being called.
        try:
            _aux_cfg_provider, _, _, _, _ = _resolve_task_provider_model("compression")
        except Exception:
            _aux_cfg_provider = ""
        client, aux_model = get_text_auxiliary_client(
            "compression",
            main_runtime=agent._current_main_runtime(),
        )
        if client is None or not aux_model:
            fb_client, fb_model, fb_label = _try_configured_fallback_for_unavailable_client(
                "compression",
                _aux_cfg_provider,
            )
            if fb_client is not None and fb_model:
                client, aux_model = fb_client, fb_model
                if "(" in fb_label and fb_label.endswith(")"):
                    _aux_cfg_provider = fb_label.rsplit("(", 1)[1][:-1]
        if client is None or not aux_model:
            if _aux_cfg_provider and _aux_cfg_provider != "auto":
                msg = (
                    "⚠ Configured auxiliary compression provider "
                    f"'{_aux_cfg_provider}' is unavailable — context "
                    "compression will drop middle turns without a summary. "
                    "Check auxiliary.compression in config.yaml and "
                    "reauthenticate that provider."
                )
            else:
                msg = (
                    "⚠ No auxiliary LLM provider configured — context "
                    "compression will drop middle turns without a summary. "
                    "Run `hermes setup` or set OPENROUTER_API_KEY."
                )
            agent._compression_warning = msg
            agent._emit_status(msg)
            logger.warning(
                "No auxiliary LLM provider for compression — "
                "summaries will be unavailable."
            )
            return

        aux_base_url = str(getattr(client, "base_url", ""))
        # ``client.api_key`` may be a callable (Azure Foundry Entra ID
        # bearer provider). The context-length resolver chain expects a
        # string, but it only needs a key for live catalogue probes
        # (provider model lists). For Entra clients the model-metadata
        # chain still resolves via models.dev + hardcoded family
        # fallbacks, which don't require auth — pass empty string rather
        # than minting a bearer JWT just to look up a context length.
        _raw_aux_key = getattr(client, "api_key", "")
        aux_api_key = "" if (callable(_raw_aux_key) and not isinstance(_raw_aux_key, str)) else str(_raw_aux_key or "")

        aux_context = get_model_context_length(
            aux_model,
            base_url=aux_base_url,
            api_key=aux_api_key,
            config_context_length=getattr(agent, "_aux_compression_context_length_config", None),
            # Each model must be resolved with its own provider so that
            # provider-specific paths (e.g. Bedrock static table, OpenRouter API)
            # are invoked for the correct client, not inherited from the main model.
            provider=(_aux_cfg_provider if _aux_cfg_provider and _aux_cfg_provider != "auto" else getattr(agent, "provider", "")),
            custom_providers=agent._custom_providers,
        )

        # Hard floor: the auxiliary compression model must have at least
        # MINIMUM_CONTEXT_LENGTH (64K) tokens of context.  The main model
        # is already required to meet this floor (checked earlier in
        # __init__), so the compression model must too — otherwise it
        # cannot summarise a full threshold-sized window of main-model
        # content.  Mirrors the main-model rejection pattern.
        if aux_context and aux_context < MINIMUM_CONTEXT_LENGTH:
            raise ValueError(
                f"Auxiliary compression model {aux_model} has a context "
                f"window of {aux_context:,} tokens, which is below the "
                f"minimum {MINIMUM_CONTEXT_LENGTH:,} required by Hermes "
                f"Agent.  Choose a compression model with at least "
                f"{MINIMUM_CONTEXT_LENGTH // 1000}K context (set "
                f"auxiliary.compression.model in config.yaml), or set "
                f"auxiliary.compression.context_length to override the "
                f"detected value if it is wrong."
            )

        threshold = agent.context_compressor.threshold_tokens
        if aux_context < threshold:
            # Auto-correct: lower the live session threshold so
            # compression actually works this session.  The hard floor
            # above guarantees aux_context >= MINIMUM_CONTEXT_LENGTH,
            # so the new threshold is always >= 64K.
            #
            # The compression summariser sends a single user-role
            # prompt (no system prompt, no tools) to the aux model, so
            # new_threshold == aux_context is safe: the request is
            # the raw messages plus a small summarisation instruction.
            old_threshold = threshold
            new_threshold = aux_context
            agent.context_compressor.threshold_tokens = new_threshold
            # Keep threshold_percent in sync so future main-model
            # context_length changes (update_model) re-derive from a
            # sensible number rather than the original too-high value.
            main_ctx = agent.context_compressor.context_length
            if main_ctx:
                agent.context_compressor.threshold_percent = (
                    new_threshold / main_ctx
                )
            safe_pct = int((aux_context / main_ctx) * 100) if main_ctx else 50
            # Build human-readable "model (provider)" labels for both
            # the main model and the compression model so users can
            # tell at a glance which provider each side is actually
            # using. When the configured provider is empty or "auto",
            # fall back to the client's base_url hostname.
            _main_model = getattr(agent, "model", "") or "?"
            _main_provider = getattr(agent, "provider", "") or ""
            _aux_provider_label = (
                _aux_cfg_provider
                if _aux_cfg_provider and _aux_cfg_provider != "auto"
                else ""
            )
            if not _aux_provider_label:
                try:
                    from urllib.parse import urlparse
                    _aux_provider_label = (
                        urlparse(aux_base_url).hostname or aux_base_url
                    )
                except Exception:
                    _aux_provider_label = aux_base_url or "auto"
            _main_label = (
                f"{_main_model} ({_main_provider})"
                if _main_provider
                else _main_model
            )
            _aux_label = f"{aux_model} ({_aux_provider_label})"
            msg = (
                f"⚠ Compression model {_aux_label} context is "
                f"{aux_context:,} tokens, but the main model "
                f"{_main_label}'s compression threshold was "
                f"{old_threshold:,} tokens. "
                f"Auto-lowered this session's threshold to "
                f"{new_threshold:,} tokens so compression can run.\n"
                f"  To make this permanent, edit config.yaml — either:\n"
                f"  1. Use a larger compression model:\n"
                f"       auxiliary:\n"
                f"         compression:\n"
                f"           model: <model-with-{old_threshold:,}+-context>\n"
                f"  2. Lower the compression threshold:\n"
                f"       compression:\n"
                f"         threshold: 0.{safe_pct:02d}"
            )
            agent._compression_warning = msg
            agent._emit_status(msg)
            logger.warning(
                "Auxiliary compression model %s has %d token context, "
                "below the main model's compression threshold of %d "
                "tokens — auto-lowered session threshold to %d to "
                "keep compression working.",
                aux_model,
                aux_context,
                old_threshold,
                new_threshold,
            )
    except ValueError:
        # Hard rejections (aux below minimum context) must propagate
        # so the session refuses to start.
        raise
    except Exception as exc:
        logger.debug(
            "Compression feasibility check failed (non-fatal): %s", exc
        )


def replay_compression_warning(agent: Any) -> None:
    """Re-send the compression warning through ``status_callback``.

    During ``__init__`` the gateway's ``status_callback`` is not yet
    wired, so ``_emit_status`` only reaches ``_vprint`` (CLI).  This
    method is called once at the start of the first
    ``run_conversation()`` — by then the gateway has set the callback,
    so every platform (Telegram, Discord, Slack, etc.) receives the
    warning.
    """
    msg = getattr(agent, "_compression_warning", None)
    if msg and agent.status_callback:
        try:
            agent.status_callback("lifecycle", msg)
        except Exception:
            pass


def conversation_history_after_compression(agent: Any, messages: list) -> Optional[list]:
    """返回压缩边界（compression boundary）后正确的刷新基准线（flush baseline）。

    传统压缩方式（Legacy compression）会轮转到一个全新的子会话（child session）。
    由于该子会话尚未通过正常的“同轮刷新路径”（same-turn flush path）看到已压缩的对话
    记录（transcript），因此调用方必须将 ``conversation_history`` 清空为 ``None``，
    并让下一次持久化（persistence）调用写入整个已压缩的列表。

    原地压缩（In-place compaction）则不同：``archive_and_compact()`` 已经对之前
    的活跃行进行了软归档（soft-archived），并在同一个会话 ID 下插入了 ``messages``
    作为新的活跃实时对话记录。如果同一个智能体轮次（agent turn）在
    ``conversation_history=None`` 的情况下继续，基于标识的刷新路径（identity-based
    flush path）会将这些已经持久化的压缩字典视为新内容并进行二次追加，从而导致活跃
    上下文翻倍并再次触发压缩。

    这里故意使用浅拷贝（shallow copy）：它将当前已压缩的字典标识捕获为历史记录，
    同时允许后续的同轮追加保持为新内容。
    """
    if bool(getattr(agent, "_last_compaction_in_place", False)):
        return list(messages)
    return None


_SYNTHETIC_USER_PREFIXES = (
    "[System: Your previous response was truncated",
    "[System: The previous response was cut off",
    "[System: Your previous tool call",
    "[Your active task list was preserved across context compression]",
    "[IMPORTANT: Background process ",
)


def _message_text(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or part.get("content") or "")
            for part in content
            if isinstance(part, dict)
        )
    return ""


_SYNTHETIC_USER_FLAGS = (
    "_todo_snapshot_synthetic",
    "_empty_recovery_synthetic",
    "_verification_stop_synthetic",
    "_pre_verify_synthetic",
)


def _is_real_user_message(message: Any) -> bool:
    """Distinguish human intent from user-role runtime scaffolding.

    A compaction summary pinned to ``role="user"`` (the compressor flips the
    summary role to preserve alternation when the tail starts with an
    assistant message) is scaffolding too: treating it as human intent would
    short-circuit anchor restoration with a message the model is explicitly
    told NOT to act on.
    """
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    if any(message.get(flag) for flag in _SYNTHETIC_USER_FLAGS):
        return False
    text = _message_text(message).strip()
    if not text:
        return False
    if text.startswith(_SYNTHETIC_USER_PREFIXES):
        return False
    from agent.context_compressor import ContextCompressor

    return not ContextCompressor._is_context_summary_content(text)


def _merge_anchor_into_user_message(target: dict, anchor: dict) -> None:
    """Fold the human anchor into an existing user-role scaffolding turn.

    Used only when every insertion slot would create two consecutive
    user-role messages. The anchor text leads (it is the active task), the
    scaffolding content is preserved after it, and the synthetic flags are
    cleared because the merged turn now carries real human intent.
    """
    anchor_content = anchor.get("content")
    target_content = target.get("content")
    if isinstance(anchor_content, list) or isinstance(target_content, list):
        anchor_parts = (
            list(anchor_content)
            if isinstance(anchor_content, list)
            else [{"type": "text", "text": str(anchor_content or "")}]
        )
        target_parts = (
            list(target_content)
            if isinstance(target_content, list)
            else [{"type": "text", "text": str(target_content or "")}]
        )
        target["content"] = anchor_parts + target_parts
    else:
        merged = f"{anchor_content or ''}\n\n{target_content or ''}".strip()
        target["content"] = merged
    for flag in _SYNTHETIC_USER_FLAGS:
        target.pop(flag, None)


def _insert_real_user_anchor(messages: list, anchor: dict) -> None:
    """Insert the latest human turn without breaking role alternation."""

    def _role(msg: Any) -> Optional[str]:
        return msg.get("role") if isinstance(msg, dict) else None

    # Preferred: the summary boundary — before the first assistant message
    # not already preceded by a user turn. The left neighbour is then
    # non-user by construction and the right neighbour is an assistant.
    for index, message in enumerate(messages):
        if _role(message) != "assistant":
            continue
        previous_role = _role(messages[index - 1]) if index > 0 else None
        if previous_role != "user":
            messages.insert(index, anchor)
            return
    # Every assistant is user-preceded (or there are none). Appending is
    # safe whenever the transcript does not already end with a user turn.
    if not messages or _role(messages[-1]) != "user":
        messages.append(anchor)
        return
    # The transcript ends with a user-role message and no slot avoids
    # user/user adjacency.
    from agent.context_compressor import ContextCompressor

    if ContextCompressor._is_context_summary_content(
        _message_text(messages[-1])
    ):
        # Never merge into a compaction summary: the summary prefix must
        # stay at the start of its message for downstream summary detection.
        # Appending after it makes the anchor "the latest user message after
        # the summary" — exactly what the handoff prefix instructs — and the
        # adjacent user turns are merged summary-first by
        # repair_message_sequence before the next API call.
        messages.append(anchor)
        return
    # Trailing user-role scaffolding (e.g. the todo snapshot): merge instead
    # of inserting a consecutive same-role message (#55677 strict templates).
    _merge_anchor_into_user_message(messages[-1], anchor)


def _ensure_compressed_has_user_turn(original_messages: list, compressed: list) -> None:
    """当压缩器仅返回 assistant/tool（助手/工具）上下文时，保留一个真实的用户轮次。

    在反复压缩过程中，受保护的 head 会衰减到仅剩系统提示词（system prompt），
    中间的摘要可能会作为 ``role="assistant"`` 存入，而包含大量工具调用的 tail
    可能全是 assistant/tool —— 因此压缩后的脚本中确实可能出现零个用户消息的情况。
    严格的聊天模板（如 LM Studio / llama.cpp 的 Jinja 模板）随后会因
    "No user query found in messages"（消息中未找到用户查询）而失败（#55677）。

    恢复的轮次会被追加到最末尾（END）：该保护机制仅在 ``compressed`` 当前以
    assistant/tool 消息结尾时才会运行（任何已存在的用户轮次 —— 包括追加的
    todo 镜像快照 —— 都会直接触发 ``any()`` 检查的短路退出），因此追加用户消息
    绝对不会导致产生连续的同角色消息。``_fresh_compaction_message_copy`` 会复制
    该消息并清除 ``_db_persisted`` 标记，以便轮转/原地刷新（rotation/in-place flush）
    仍能将恢复的行持久化到新会话中（#57491）。

    如果压缩前的脚本本身就完全不包含任何用户轮次（这几乎是不可能的 —— 每一个真实的
    对话都以用户请求开始 —— 但此处作为防御性兜底保留），则会追加一个极简的
    延续标记（continuation marker），以确保严格的模板仍能看到一条用户消息。
    """
    if any(isinstance(msg, dict) and msg.get("role") == "user" for msg in compressed):
        return
    from agent.context_compressor import _fresh_compaction_message_copy

    for message in reversed(original_messages):
        if _is_real_user_message(message):
            _insert_real_user_anchor(
                compressed,
                _fresh_compaction_message_copy(message),
            )
            return
    compressed.append({
        "role": "user",
        "content": (
            "Continue from the compressed conversation context above. "
            "This marker exists because no human user turn was available."
        ),
    })


def compress_context(
    agent: Any,
    messages: list,
    system_message: str,
    *,
    approx_tokens: Optional[int] = None,
    task_id: str = "default",
    focus_topic: Optional[str] = None,
    force: bool = False,
) -> Tuple[list, str]:
    """压缩对话上下文并在 SQLite 中拆分会话。

    参数：
        agent: 所属的 :class:`AIAgent` 实例。
        messages: 当前的消息历史记录（将被总结）。
        system_message: 当前的系统提示词；在压缩后会被重新构建。
        approx_tokens: 压缩前的 Token 估算值，用于运维日志记录。
        task_id: 工具任务的作用域（用于清除文件读取去重状态）。
        focus_topic: 可选的焦点字符串，用于引导定向压缩 —— 总结器
            会优先保留与该主题相关的信息。灵感来源于 Claude Code
            的 ``/compact <focus>`` 命令。
        force: 若为 True，则绕过当前处于激活状态的“总结失败冷却时间”。
            由手动的 ``/compress`` 斜杠命令设置，以便用户在自动压缩
            中止后可以立即重试。自动压缩的调用者使用默认值 ``False``。

    返回：
        ``(compressed_messages, new_system_prompt)`` 元组。当
        压缩中止（辅助 LLM 未能生成可用的总结）时，返回未更改的
        原始消息和现有的系统提示词 —— 会话【不会】被轮转。调用者
        应通过 ``len(returned) == len(input)`` 检测此空操作（no-op）
        并停止重试循环。
    """
    # Codex app-server会话：Codex 智能体（agent）拥有真实的线程上下文；
    # Hermes 的总结器只会重写本地镜像，而无法缩小实际的线程（参见 #36801）。
    # 将压缩操作路由到应用服务器自身的线程/压缩机制（thread/compact mechanism）。
    # 该行为由 ``compression.codex_app_server_auto`` 控制（可选值为 native|hermes|off）。
    if getattr(agent, "api_mode", None) == "codex_app_server":
        return _compress_context_via_codex_app_server(
            agent,
            messages,
            system_message,
            approx_tokens=approx_tokens,
            task_id=task_id,
            force=force,
        )

    # 每一个自动入口点（automatic entrypoint）都必须遵循压缩器（compressor）自身的冷却与熔断器状态。
    #
    # 由于网关清理机制（Gateway hygiene）会新建一个 AIAgent 实例，
    # 因此在此之前，已持久化的备用连续失败次数（fallback streak）
    # 会由 bind_session_state() 加载完成。
    if not force:
        _refresh_persisted_compression_guards(agent.context_compressor)
        blocked = getattr(
            type(agent.context_compressor),
            "_automatic_compression_blocked",
            None,
        )
        if callable(blocked) and blocked(agent.context_compressor):
            existing_prompt = getattr(agent, "_cached_system_prompt", None)
            if not existing_prompt:
                existing_prompt = agent._build_system_prompt(system_message)
            return messages, existing_prompt

    # 延迟（惰性）可行性检查 ——
    # 在首次尝试压缩时，才即时（just-in-time）运行辅助提供商探测与上下文长度查询，
    # 而不是在 `AIAgent.__init__` 初始化时执行。
    #
    # 对于从未达到触发阈值的短会话（绝大多数的 ``chat -q`` 运行均属于此类），
    # 这样能为每次冷启动节省约 400 毫秒的时间。
    #
    # 该检查操作本身会设置 ``agent._compression_warning``，
    # 因此，当警告首次产生实际影响时，
    # 状态回调重播机制（status-callback replay machinery）依然能正常向用户发出该警告。
    if not getattr(agent, "_compression_feasibility_checked", False):
        # 只有在探测（probe）完成后才标记为已检查（checked）。如果检查过程中
        # 抛出异常（例如导致会话中止的致命辅助上下文 ValueError），保持该标志（flag）
        # 为未设置状态是无害的；而非致命的瞬态故障（transient failure）会在函数内部
        # 被吞掉（swallowed），因此在下一次成功通过时，该标志仍会被正常设置。
        check_compression_model_feasibility(agent)
        agent._compression_feasibility_checked = True

    _pre_msg_count = len(messages)
    # 原地压缩（配置项：compression.in_place，参见 #38763）。当该值为 True 时，
    # 此压缩操作会重写消息列表并重新构建系统提示词，但会保持【完全相同】的 session_id
    # —— 也就是说，没有 end_session，没有 parent_session_id 子会话，没有
    # `name #N` 的重新编号，没有 contextvar/环境变量/日志记录的重新同步，
    # 也没有记忆/上下文引擎的会话切换。整个对话在生命周期内始终保持一个持久的 ID，
    # 从而彻底消除了因会话轮转（session-rotation）导致的一连串 Bug。在逐步推广期间默认值为 False。
    in_place = bool(getattr(agent, "compression_in_place", False))
    # 一旦原地数据库写入实际完成，即设置为 True（数据库代码块可能会抛出异常并跳过它）。
    # 通过 agent._last_compaction_in_place 暴露给网关（gateway）。
    compacted_in_place = False
    logger.info(
        "context compression started: session=%s messages=%d tokens=~%s model=%s focus=%r",
        agent.session_id or "none", _pre_msg_count,
        f"{approx_tokens:,}" if approx_tokens else "unknown", agent.model,
        focus_topic,
    )
    agent._emit_status(COMPACTION_STATUS)

    # ── 压缩锁 ────────────────────────────────────────────────
    # 每个 session_id 专属的、由 state.db 支持的原子锁。如果没有这个锁，
    # 共享相同 session_id 的两个 AIAgent 实例（最常见的是父轮次智能体
    # 及其后台审查分支 —— 参见 ``agent/background_review.py``：
    # ``review_agent.session_id = agent.session_id``）可能会各自对
    # 同一对话的重叠快照（overlapping snapshots）调用 compress()。
    # 结果是两者都会成功，两者都会将 ``agent.session_id`` 轮转为一个全新的 ID，
    # 并且两者都会在 state.db 中创建以同一个旧 ID 为父级的子会话。
    # 网关的 SessionEntry 只能捕获到其中一次轮转，因此另一个子会话就会变成
    # 孤儿会话（orphan），在后台默默地累积写入数据 —— 这正是 Damien 重现出的 Bug 形态。
    #
    # 获取锁时需要以 旧的 session_id（即轮转目标的父级）作为键（key），
    # 因为这是竞争路径在各自尝试压缩之初，从 SessionEntry 中看到并读取的 ID。
    #
    # 如果我们无法获取该锁，说明另一个路径正在对该会话进行中途压缩。
    # 此时中止（Aborting）是正确的做法：消息保持不变，另一个路径的轮转
    # 将生成规范的新 session_id，并且我们调用者的自动压缩循环会检测到
    # ``len(returned) == len(input)`` 并停止本轮周期的重试。
    # 会话【不会】遭到损坏 —— 我们只是退出这一轮竞争，让胜出者完成操作。
    _lock_db = getattr(agent, "_session_db", None)
    _lock_sid = agent.session_id or ""
    _lock_holder: Optional[str] = None
    # 探测当前 SessionDB 实例上锁子系统（lock subsystem）是否真正可用。
    #
    # 如果进程运行了版本不匹配的模块，
    # 可能会在长寿命 SessionDB 实例尚未支持锁 API（lock API）时调用此处的代码。
    # 只有在这种“结构性缺失”的情况下，选择“故障开放（fail open）”才是安全的：
    # 因为在更新之后，压缩任务必须向前推进，而不是陷入无限循环。
    #
    # 一旦该方法解析成功，其具体实现抛出的任何异常都必须“故障关闭（fail closed）”，
    # 因为在没有锁的情况下继续执行，可能会导致会话谱系（session lineage）发生分叉。
    _try_acquire_lock = None
    _lock_lookup_error: Optional[Exception] = None
    _legacy_session_db_without_lock_api = False
    if _lock_db is not None:
        try:
            _legacy_session_db_without_lock_api = _lock_api_is_absent_on_session_db(
                _lock_db
            )
        except Exception as exc:
            _lock_lookup_error = exc
        if _lock_lookup_error is None and not _legacy_session_db_without_lock_api:
            try:
                _try_acquire_lock = _lock_db.try_acquire_compression_lock
                if not callable(_try_acquire_lock):
                    _lock_lookup_error = TypeError(
                        "compression lock API is present but not callable"
                    )
            except Exception as exc:
                _lock_lookup_error = exc
    try:
        _lock_ttl = float(getattr(agent, "_compression_lock_ttl_seconds", 300.0) or 300.0)
    except (TypeError, ValueError):
        _lock_ttl = 300.0
    _lock_refresh_interval = getattr(agent, "_compression_lock_refresh_interval", None)
    _lock_refresher: Optional[_CompressionLockLeaseRefresher] = None
    if _lock_db is not None and _lock_sid:
        _lock_holder = _compression_lock_holder(agent)
        if _lock_lookup_error is not None:
            # Attribute lookup itself failed for a reason other than a missing
            # lock API. It is unsafe to proceed without a lock in that case.
            _lock_holder = None
            logger.warning(
                "compression lock lookup raised unexpectedly for session=%s "
                "(%s: %s) — skipping compression this cycle",
                _lock_sid, type(_lock_lookup_error).__name__, _lock_lookup_error,
            )
            _lock_acquired = False
        elif _try_acquire_lock is None:
            # The lock API itself is absent on this in-memory instance. Log once
            # and proceed unlocked so an update-version skew cannot leave the
            # outer auto-compression loop making no progress forever.
            _lock_holder = None
            if getattr(agent, "_last_compression_lock_error_sid", None) != _lock_sid:
                agent._last_compression_lock_error_sid = _lock_sid
                logger.warning(
                    "compression lock subsystem unavailable for session=%s "
                    "— proceeding without lock. This usually means a stale "
                    "in-memory module after an update; restart the process "
                    "(or `hermes update`) to resync.",
                    _lock_sid,
                )
            _lock_acquired = True  # acquired-but-unlocked compatibility path
        else:
            try:
                _lock_acquired = _try_acquire_lock(
                    _lock_sid, _lock_holder, ttl_seconds=_lock_ttl
                )
            except Exception as _lock_err:
                # The method exists and entered its implementation but failed.
                # Do not mistake an internal AttributeError or TypeError for
                # version skew: fail closed and preserve session lineage. A
                # failure after SQLite committed the acquire can leave our
                # holder row behind, so release it best-effort before returning
                # unchanged messages; release is holder-qualified and safe when
                # acquisition never succeeded.
                try:
                    _lock_db.release_compression_lock(_lock_sid, _lock_holder)
                except Exception as _release_err:
                    logger.debug(
                        "compression lock cleanup after failed acquire failed: %s",
                        _release_err,
                    )
                _lock_holder = None
                logger.warning(
                    "compression lock acquisition raised unexpectedly for "
                    "session=%s (%s: %s) — skipping compression this cycle",
                    _lock_sid, type(_lock_err).__name__, _lock_err,
                )
                _lock_acquired = False
        if not _lock_acquired:
            try:
                existing = _lock_db.get_compression_lock_holder(_lock_sid)
            except Exception:
                existing = None
            logger.warning(
                "compression skipped: another path is compressing session=%s "
                "(holder=%s) — returning messages unchanged to avoid session fork",
                _lock_sid, existing,
            )
            _lock_holder = None  # don't release a lock we don't own
            # Surface to the user once — quiet for downstream auto-compress loops
            if getattr(agent, "_last_compression_lock_warning_sid", None) != _lock_sid:
                agent._last_compression_lock_warning_sid = _lock_sid
                try:
                    agent._emit_warning(
                        "⚠ Skipping concurrent compression — another path "
                        "is already compressing this session. Will retry "
                        "after it finishes."
                    )
                except Exception:
                    pass
            _existing_sp = getattr(agent, "_cached_system_prompt", None)
            if not _existing_sp:
                _existing_sp = agent._build_system_prompt(system_message)
            return messages, _existing_sp
        if _lock_holder is not None:
            _lock_refresher = _CompressionLockLeaseRefresher(
                _lock_db,
                _lock_sid,
                _lock_holder,
                _lock_ttl,
                _lock_refresh_interval,
            ).start()

    def _release_lock() -> None:
        """Release the lock keyed on the OLD session_id (before rotation)."""
        if _lock_refresher is not None:
            _lock_refresher.stop()
        if _lock_db is not None and _lock_sid and _lock_holder:
            try:
                _lock_db.release_compression_lock(_lock_sid, _lock_holder)
            except Exception as _rel_err:
                logger.debug("compression lock release failed: %s", _rel_err)

    # A delayed contender can acquire the parent lock after the winning path
    # has released it and completed rotation. The lock serializes work but does
    # not by itself prove that this stale agent still owns a live parent.
    if _lock_db is not None and _lock_sid:
        try:
            _parent_already_rotated = _session_was_rotated_by_compression(
                _lock_db, _lock_sid
            )
        except Exception as _session_err:
            logger.warning(
                "compression session ownership lookup failed for session=%s "
                "(%s: %s) - skipping compression this cycle",
                _lock_sid,
                type(_session_err).__name__,
                _session_err,
            )
            _release_lock()
            _existing_sp = getattr(agent, "_cached_system_prompt", None)
            if not _existing_sp:
                _existing_sp = agent._build_system_prompt(system_message)
            return messages, _existing_sp
        if _parent_already_rotated:
            logger.info(
                "compression skipped: session=%s was already rotated by "
                "another compression path",
                _lock_sid,
            )
            _release_lock()
            _existing_sp = getattr(agent, "_cached_system_prompt", None)
            if not _existing_sp:
                _existing_sp = agent._build_system_prompt(system_message)
            return messages, _existing_sp

    # The agent may have been constructed before another path completed an
    # in-place compaction on the same session. Re-read durable breaker state
    # after acquiring the session lock so this final gate cannot act on the
    # stale snapshot loaded by bind_session_state().
    if not force:
        compressor = agent.context_compressor
        _refresh_persisted_compression_guards(compressor)
        blocked = getattr(
            type(compressor),
            "_automatic_compression_blocked",
            None,
        )
        if callable(blocked) and blocked(compressor):
            _release_lock()
            existing_prompt = getattr(agent, "_cached_system_prompt", None)
            if not existing_prompt:
                existing_prompt = agent._build_system_prompt(system_message)
            return messages, existing_prompt

    try:
        # Notify external memory provider before compression discards context.
        # The provider's on_pre_compress() may return a string of insights it
        # wants surfaced inside the compression summary; capture and forward it
        # instead of silently discarding the provider's return value.
        memory_context = ""
        if agent._memory_manager:
            try:
                _maybe_ctx = agent._memory_manager.on_pre_compress(messages)
                if isinstance(_maybe_ctx, str):
                    memory_context = sanitize_memory_context(_maybe_ctx)
            except Exception:
                pass

        compress_fn = agent.context_compressor.compress
        compress_kwargs = _supported_compression_kwargs(
            compress_fn,
            current_tokens=approx_tokens,
            focus_topic=focus_topic,
            force=force,
            memory_context=memory_context,
        )
        if memory_context.strip() and "memory_context" not in compress_kwargs:
            engine_name = getattr(
                agent.context_compressor,
                "name",
                type(agent.context_compressor).__name__,
            )
            if (
                getattr(agent, "_last_memory_context_unsupported_engine", None)
                != engine_name
            ):
                agent._last_memory_context_unsupported_engine = engine_name
                logger.warning(
                    "context engine %s does not accept memory_context; continuing "
                    "without provider-supplied summary context",
                    engine_name,
                )

        compressed = compress_fn(messages, **compress_kwargs)
    except BaseException:
        # ANY exception after lock acquisition — memory hook, capability
        # inspection, engine lookup, or compress() — must release the lock so
        # the session isn't permanently blocked from future compression.
        _release_lock()
        raise

    # 在会话轮换（session-rotation）回调运行之前捕获边界质量。
    # 内置和插件的生命周期钩子（lifecycle hooks）可能会在重新绑定到子 ID 时，
    # 重置单次会话的压缩器字段；
    #
    # 已完成尝试的判定结果（verdict）必须在重新绑定后留存下来，
    # 并且仅在整个边界提交完成后才被记录。
    _compression_made_progress = bool(
        getattr(agent.context_compressor, "_last_compression_made_progress", False)
    )
    _compression_used_fallback = bool(
        getattr(agent.context_compressor, "_last_summary_fallback_used", False)
    )

    # 如果压缩中止（辅助 LLM 未能生成可用的摘要），
    # 压缩器将原封不动地返回输入消息。
    #
    # 向用户展示该错误，完全跳过会话轮换（session-rotation）工作
    # （因为逻辑上没有会话结束），
    # 并让自动压缩的调用方通过 len(returned) == len(input) 检测到此无操作（no-op）。
    if getattr(agent.context_compressor, "_last_compress_aborted", False):
        try:
            _err = getattr(agent.context_compressor, "_last_summary_error", None) or "unknown error"
            if getattr(agent, "_last_compression_summary_warning", None) != _err:
                agent._last_compression_summary_warning = _err
                agent._emit_warning(
                    f"⚠ Compression aborted: {_err}. "
                    "No messages were dropped — conversation continues unchanged. "
                    "Run /compress to retry, or /new to start a fresh session."
                )
            _existing_sp = getattr(agent, "_cached_system_prompt", None)
            if not _existing_sp:
                _existing_sp = agent._build_system_prompt(system_message)
            return messages, _existing_sp
        finally:
            _release_lock()

    # 压缩器如果返回了完全相同的输入对象，则说明没有取得任何结构性进展。
    # 在这种情况下，不要轮转/重写会话，也不要启用压缩后延迟（post-compression deferral）；
    # 压缩器自身的防抖动计数器会记录下这次无操作（no-op）。
    if compressed is messages:
        logger.info(
            "Compression made no progress (session=%s) — skipping boundary rewrite.",
            agent.session_id or "none",
        )
        _existing_sp = getattr(agent, "_cached_system_prompt", None)
        if not _existing_sp:
            _existing_sp = agent._build_system_prompt(system_message)
        _release_lock()
        return messages, _existing_sp

    if not compressed:
        logger.error(
            "context compression returned an empty transcript; refusing to "
            "rotate session=%s so the parent remains resumable",
            agent.session_id or "none",
        )
        try:
            agent._emit_warning(
                "⚠ Compression returned an empty transcript. "
                "No session split was performed; conversation continues unchanged."
            )
        except Exception:
            pass
        _existing_sp = getattr(agent, "_cached_system_prompt", None)
        if not _existing_sp:
            _existing_sp = agent._build_system_prompt(system_message)
        _release_lock()
        return messages, _existing_sp

    try:
        summary_error = getattr(agent.context_compressor, "_last_summary_error", None)
        if summary_error:
            if getattr(agent, "_last_compression_summary_warning", None) != summary_error:
                agent._last_compression_summary_warning = summary_error
                agent._emit_warning(
                    f"⚠ Compression summary failed: {summary_error}. "
                    "Inserted a fallback context marker."
                )
        else:
            # 不是硬性失败 —— 但配置的辅助模型是否报错，
            # 并通过在主模型上重试得以恢复？
            # 将此情况呈现给用户，以便他们知道即使压缩成功，
            # 他们的 auxiliary.compression.model 设置也是损坏的。
            _aux_fail_model = getattr(agent.context_compressor, "_last_aux_model_failure_model", None)
            _aux_fail_err = getattr(agent.context_compressor, "_last_aux_model_failure_error", None)
            if _aux_fail_model:
                # Dedup on (model, error) so we don't spam on every compaction
                _aux_key = (_aux_fail_model, _aux_fail_err)
                if getattr(agent, "_last_aux_fallback_warning_key", None) != _aux_key:
                    agent._last_aux_fallback_warning_key = _aux_key
                    agent._emit_warning(
                        f"ℹ Configured compression model '{_aux_fail_model}' failed "
                        f"({_aux_fail_err or 'unknown error'}). Recovered using main model — "
                        "check auxiliary.compression.model in config.yaml."
                    )

        todo_snapshot = agent._todo_store.format_for_injection()
        if todo_snapshot:
            compressed.append({
                "role": "user",
                "content": todo_snapshot,
                "_todo_snapshot_synthetic": True,
            })
        _ensure_compressed_has_user_turn(messages, compressed)

        cached_system_prompt = agent._cached_system_prompt
        agent._invalidate_system_prompt()

        # 内置内存（Built-in memory）是常规压缩唯一会重新加载的系统提示词（system-prompt）输入。
        #
        # 当缓存的提示词中已经逐字包含了新鲜重新加载的内存块时，
        # 保持完全一致的缓存提示词，以便本地后端能够保留其 KV 缓存前缀。
        #
        # 此处要求的是包含关系（Containment），而不是变更前/后的快照相等（snapshot equality）：
        # 新建的 Agent 交互界面（fresh-agent surfaces）会从会话数据库（session DB）中恢复缓存的提示词，
        # 而该提示词的产生时间可能早于内存快照（in-memory snapshot）中已吸收的会话中途内存写入。
        #
        # 外部提供商（External providers）可以在 on_pre_compress() 执行期间修改它们自己的提示词块，
        # 因此它们保留了重新构建的路径。
        if (
            cached_system_prompt is not None
            and getattr(agent, "_memory_manager", None) is None
            and _cached_prompt_reflects_builtin_memory(agent, cached_system_prompt)
        ):
            new_system_prompt = cached_system_prompt
            agent._cached_system_prompt = cached_system_prompt
        else:
            new_system_prompt = agent._build_system_prompt(system_message)
            agent._cached_system_prompt = new_system_prompt

        if agent._session_db:
            try:
                # 在对话文字记录被重写之前，触发当前会话的记忆提取
                # （在两种模式下均会运行 — 无论 ID 是否轮换，逻辑对话
                # 中压缩前的轮次都即将被摘要精简掉）。
                agent.commit_memory_session(messages)

                if in_place:
                    # ── 原地压缩（In-place compaction）：保持相同的 session_id ───────
                    # 不结束会话（no end_session）、不生成新行、不改变 parent_session_id、不重新生成标题、
                    # 不对轮次重新编号，也不进行 contextvar/环境变量/日志的重新同步。该会话的
                    # id、标题、当前工作目录（cwd）、/goal 以及网关路由都保持原样。
                    #
                    # 持久化且非破坏性的替换操作：
                    # 软归档（soft-archive）压缩前的轮次消息（ active=0，保留在磁盘上，支持 FTS 全文检索且可恢复），
                    # 并以原子方式插入 `compressed` 作为新的活动消息集（ active=1 ）。
                    #
                    # 由于 `compressed` 已经包含了保留的尾部消息
                    # （即压缩器通过 protect_last_n 参数保留的当前轮次消息），
                    # 因此我们在此处不需要提前刷盘（pre-flush）——
                    # 提前刷盘会插入当前轮次的行数据，
                    # 而随后 archive_and_compact 会将这些行与其他消息一起归档（虽然无害，但会导致不必要的写入开销）。
                    #
                    # 活动上下文的加载操作只过滤 active=1 的数据，
                    # 因此恢复会话时仅重新加载压缩后的数据集；
                    # 原始轮次消息仍保留在同一个 ID 下，以便进行搜索和恢复
                    # （遵循 Teknium 审查意见 —— 保留同一个持久化 ID 且不销毁历史记录，这与硬替换 replace_messages 不同）。
                    # 详情参见 #38763。
                    agent._session_db.archive_and_compact(agent.session_id, compressed)
                    # 重置刷新标识集（flush identity set），以便下一轮的追加操作
                    # 能够与【压缩后】的对话记录进行差异对比（diff）：压缩后的字典
                    # 将在下一轮作为 conversation_history 传入，并会通过标识进行跳过，
                    # 从而确保只有真正属于新一轮的消息才会被追加
                    # （既不会重复添加摘要，也不会使已被丢弃的轮次“起死回生”）。
                    agent._flushed_db_message_ids = set()
                    # 轮转无关信号（Rotation-independent signal）：对话已被原地压缩
                    # （ID 未发生改变）。网关会读取此信号（这【不是】一个 ID 变更的
                    # diff 对比），用以重新基准化（re-baseline）对话记录的处理。
                    compacted_in_place = True
                else:
                    # ── Rotation (legacy): end this session, fork a continuation ─
                    # Flush any un-persisted current-turn messages to the OLD
                    # session before ending it, so they survive in the preserved
                    # parent transcript (#47202). (In-place skips this — see above.)
                    try:
                        agent._flush_messages_to_session_db(messages)
                    except Exception:
                        pass  # best-effort — don't block compression on a flush error
                    # Propagate title to the new session with auto-numbering
                    old_title = agent._session_db.get_session_title(agent.session_id)
                    agent._session_db.end_session(agent.session_id, "compression")
                    old_session_id = agent.session_id
                    agent.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                    # Ordering contract: the agent thread updates the contextvar here;
                    # the gateway propagates to SessionEntry after run_in_executor returns.
                    try:
                        from gateway.session_context import set_current_session_id

                        set_current_session_id(agent.session_id)
                    except Exception:
                        os.environ["HERMES_SESSION_ID"] = agent.session_id
                    # The gateway/tools session context (ContextVar + env) and the
                    # logging session context are SEPARATE mechanisms. The call above
                    # moves the former; the ``[session_id]`` tag on log lines comes
                    # from ``hermes_logging._session_context`` (set once per turn in
                    # conversation_loop.py). Without this, post-rotation log lines in
                    # the same turn keep the STALE old id while the message/DB/gateway
                    # state carry the new one — breaking log correlation exactly at the
                    # compaction boundary (see #34089). Guarded separately so a logging
                    # failure can never regress the routing update above.
                    try:
                        from hermes_logging import set_session_context

                        set_session_context(agent.session_id)
                    except Exception:
                        pass
                    agent._session_db_created = False
                    try:
                        agent._session_db.create_session(
                            session_id=agent.session_id,
                            source=agent.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                            model=agent.model,
                            model_config=agent._session_init_model_config,
                            parent_session_id=old_session_id,
                        )
                    except Exception as _cs_err:
                        # The child row could not be created (e.g. FK constraint,
                        # contended write). Previously the outer handler simply
                        # warned and let the agent continue on the NEW id — which
                        # has no row in state.db, producing an orphan: the parent
                        # is ended, the child is never indexed, and every
                        # subsequent message is attributed to a session that
                        # doesn't exist (#33906/#33907). Roll the live id back to
                        # the parent so the conversation stays attached to a real,
                        # indexed session instead of a phantom.
                        logger.warning(
                            "Compression child session create failed (%s) — "
                            "rolling back to parent session %s to avoid an orphan.",
                            _cs_err, old_session_id,
                        )
                        agent.session_id = old_session_id
                        try:
                            from gateway.session_context import set_current_session_id
                            set_current_session_id(agent.session_id)
                        except Exception:
                            os.environ["HERMES_SESSION_ID"] = agent.session_id
                        try:
                            from hermes_logging import set_session_context
                            set_session_context(agent.session_id)
                        except Exception:
                            pass
                        # Re-open the parent: it was ended above, but we're
                        # continuing on it, so it must not stay closed.
                        try:
                            agent._session_db.reopen_session(old_session_id)
                        except Exception:
                            pass
                        old_session_id = None  # no rotation happened
                        # The parent row already exists in state.db, so mark the
                        # session as created — _ensure_db_session would otherwise
                        # retry a (harmless INSERT OR IGNORE) create next turn.
                        agent._session_db_created = True
                        raise
                    agent._session_db_created = True
                    # Carry a persistent /goal onto the continuation session.
                    # Compression mints a fresh child id; load_goal does a flat
                    # per-session lookup with no parent walk, so without this an
                    # active goal silently dies at the boundary (#33618).
                    try:
                        from hermes_cli.goals import migrate_goal_to_session
                        migrate_goal_to_session(old_session_id, agent.session_id, reason="compression")
                    except Exception as _goal_err:
                        logger.debug("Could not migrate goal on compression: %s", _goal_err)
                    # Auto-number the title for the continuation session
                    if old_title:
                        try:
                            new_title = agent._session_db.get_next_title_in_lineage(old_title)
                            agent._session_db.set_session_title(agent.session_id, new_title)
                        except (ValueError, Exception) as e:
                            logger.debug("Could not propagate title on compression: %s", e)

                # 共享的写后步骤（两种模式都以 agent.session_id 为目标，
                # 原地压缩会保留该 ID，而会话轮转则已经将其重新分配给了新的 ID）：
                # 刷新存储的系统提示词并重置刷新游标（flush cursor），
                # 以便下一轮次重新构建其追加差量（append diff）的基准。
                agent._session_db.update_system_prompt(agent.session_id, new_system_prompt)
                if in_place:
                    agent._last_flushed_db_idx = 0
                else:
                    # A headless turn can be killed before its finalizer. Persist
                    # the rotated child's compacted handoff at the boundary so
                    # the new session is immediately resumable.
                    agent._session_db.replace_messages(agent.session_id, compressed)
                    agent._last_flushed_db_idx = len(compressed)
                    agent._flushed_db_message_session_id = agent.session_id
                    agent._flushed_db_message_ids = {
                        id(message)
                        for message in compressed
                        if isinstance(message, dict)
                    }
            except Exception as e:
                # If the rotation rolled back to the parent (orphan-avoidance
                # above), agent.session_id is the still-indexed parent and
                # old_session_id was cleared — so this is recovery, not an
                # un-indexed orphan. Otherwise an earlier step failed before the
                # child was created and the warning's original meaning holds.
                if locals().get("old_session_id") is None and not in_place:
                    logger.warning(
                        "Compression rotation aborted and rolled back to the "
                        "parent session (%s): %s", agent.session_id or "?", e,
                    )
                else:
                    logger.warning("Session DB compression split failed — new session will NOT be indexed: %s", e)

        # 压缩边界记账（Compaction-boundary bookkeeping），仅计算一次。
        # `old_session_id` 仅在轮转（rotation）分支中进行绑定；原地压缩（in-place）时
        # 则保持未设置状态。`_boundary_parent` 是边界通知将先前状态归属到的 ID：
        # 在轮转时为旧 ID，在原地压缩时则为（未发生改变的）当前 ID。
        _old_sid = locals().get("old_session_id")
        _is_boundary = bool(_old_sid) or in_place
        _boundary_parent = _old_sid or agent.session_id or ""

        # 通知上下文引擎（context engine）发生了一个压缩边界。插件
        # 引擎（例如 hermes-lcm）使用 boundary_reason="compression" 来保留
        # DAG 谱系（lineage），以便跨越边界时对每个会话的状态进行检查点（checkpoint）记录，
        # 而不是重新初始化一个全新的会话。参见 hermes-lcm#68。内置的 ContextCompressor
        # 会忽略 kwargs。在【两种】模式下都会触发：轮转模式会传递 旧→新 ID；原地模式
        # 则会传递【相同的】ID（尽管 ID 没有改变，但该边界是真实存在的）。
        try:
            if _is_boundary and hasattr(agent.context_compressor, "on_session_start"):
                agent.context_compressor.on_session_start(
                    agent.session_id or "",
                    boundary_reason="compression",
                    old_session_id=_boundary_parent,
                    platform=getattr(agent, "platform", None) or "cli",
                    conversation_id=getattr(agent, "_gateway_session_key", None),
                )
        except Exception as _ce_err:
            logger.debug("context engine on_session_start (compression): %s", _ce_err)

        # 通知内存提供者（memory providers）已发生压缩边界，以便提供者缓存的
        # 特定会话状态（如 Hindsight 的 _document_id、累积的轮次缓冲区、
        # 计数器）进行刷新。因为逻辑上的对话仍在继续，所以设置 reset=False。
        # 参见 #6672。在【两种】模式下都会触发：原地模式使用与父会话相同的 id
        # （对话并没有分叉，但仍必须告知缓冲区对话记录已被压缩，以避免对已丢弃的轮次进行重复计数）。
        try:
            if _is_boundary and agent._memory_manager:
                agent._memory_manager.on_session_switch(
                    agent.session_id or "",
                    parent_session_id=_boundary_parent,
                    reset=False,
                    reason="compression",
                )
        except Exception as _me_err:
            logger.debug("memory manager on_session_switch (compression): %s", _me_err)

        # 针对重复压缩发出警告（每压缩一次，质量都会随之下降）。
        # 通过 _emit_status 进行路由（就像上面其他压缩警告一样），
        # 以便警告能通过 status_callback 送达 TUI / Telegram / Discord，
        # 而不仅仅是 CLI 的标准输出（stdout）。_emit_status 仍然会为 CLI 进行 _vprint 打印，
        # 并且将其存储在 _compression_warning 中，可以让 replay_compression_warning
        # 在后期绑定的网关 status_callback 接通后，重新投递该警告（#36908）。
        _cc = agent.context_compressor.compression_count
        if _cc >= 2:
            _cc_msg = (
                f"{agent.log_prefix}⚠️  Session compressed {_cc} times — "
                f"accuracy may degrade. Consider /new to start fresh."
            )
            agent._compression_warning = _cc_msg
            agent._emit_status(_cc_msg)

        # 触发 session:compress 事件，以便钩子函数（例如 MemPalace 同步）能够
        # 在已完成的旧会话细节丢失之前对其进行摄取。在原地（in-place）模式下
        # 不存在旧 ID（会话相同）；``in_place=True`` 会告知钩子函数，
        # 对话记录是在同一个 ID 上被压缩的，而不是进行了轮转。
        if getattr(agent, "event_callback", None):
            try:
                agent.event_callback("session:compress", {
                    "platform": agent.platform or "",
                    "session_id": agent.session_id,
                    "old_session_id": _old_sid or "",
                    "in_place": in_place,
                    "compression_count": agent.context_compressor.compression_count,
                })
            except Exception as e:
                logger.debug("event_callback error on session:compress: %s", e)

        # 通过一个与轮转无关的标志（rotation-independent flag），将压缩模式显式提供给
        # 调用者（run_conversation / gateway）。当原地发生压缩时，网关会利用此标志
        # — 而【不是】通过 ID 变更的 diff 对比 — 来重新基准化（re-baseline）对话记录的
        # 处理（即在同一个 ID 上将 history_offset 设为 0 并进行重写）。参见 #38763。
        agent._last_compaction_in_place = compacted_in_place

        # 保留压缩后的粗略预估值用于诊断，但不要
        # 将其视为服务商报告的 prompt 使用量。即使在下一次真实的 API 请求
        # 能够容纳之后，包含大量 Schema 的粗略预估值可能仍会高于阈值。
        _compressed_est = estimate_request_tokens_rough(
            compressed,
            system_prompt=new_system_prompt or "",
            tools=agent.tools or None,
        )
        agent.context_compressor.last_compression_rough_tokens = _compressed_est
        agent.context_compressor.last_prompt_tokens = -1
        agent.context_compressor.last_completion_tokens = 0
        agent.context_compressor.awaiting_real_usage_after_compression = True
        # 仅当完成的重写跨越整个压缩边界（compaction boundary）后，
        # 才触发效果判定（effectiveness verdict）。
        #
        # 异常、中止以及无操作（no-op）的尝试都会使该状态保持为 false，
        # 从而确保后续无关的使用不会被归咎于一次从未修改过转录文本（transcript）的尝试。
        if _compression_made_progress:
            record_boundary = getattr(
                type(agent.context_compressor),
                "record_completed_compaction",
                None,
            )
            if callable(record_boundary):
                record_boundary(
                    agent.context_compressor,
                    used_fallback=_compression_used_fallback,
                )
            else:
                agent.context_compressor._verify_compaction_cleared_threshold = True

        # 清除文件读取的去重缓存。压缩之后，原始的
        # 读取内容已被摘要精简 — 如果模型重新读取相同的
        # 文件，它需要的是完整的内容，而不是一个“文件未更改”的存根。
        try:
            from tools.file_tools import reset_file_dedup
            reset_file_dedup(task_id)
        except Exception:
            pass

        logger.info(
            "context compression done: session=%s messages=%d->%d rough_tokens=~%s awaiting_real_usage=true",
            agent.session_id or "none", _pre_msg_count, len(compressed),
            f"{_compressed_est:,}",
        )
        return compressed, new_system_prompt
    finally:
        # Release the lock on the OLD session_id only AFTER rotation completed
        # and all post-rotation bookkeeping (memory manager, context engine,
        # file dedup) ran. A concurrent path that wakes up the moment we
        # release will see the NEW session_id in state.db / SessionEntry and
        # acquire on that — no race against our just-finished work.
        _release_lock()


def _compress_context_via_codex_app_server(
    agent: Any,
    messages: list,
    system_message: Optional[str],
    *,
    approx_tokens: Optional[int] = None,
    task_id: str = "default",
    force: bool = False,
) -> Tuple[list, str]:
    """对于 Codex 拥有的线程，将压缩（compaction）操作路由至 Codex 应用服务器。

    Hermes 正常的压缩器会重写本地的 OpenAI 风格的对话记录（transcript）。
    但这并不能缩小实际的 Codex 应用服务器线程上下文。对于这种运行时环境，
    应让 Codex 去压缩它自己的线程，并保持 Hermes 的对话记录不发生改变。
    """
    auto_mode = str(
        getattr(agent, "codex_app_server_auto_compaction", "native") or "native"
    ).lower()
    if auto_mode not in {"native", "hermes", "off"}:
        auto_mode = "native"
    if not force and auto_mode != "hermes":
        logger.info(
            "codex app-server compaction skipped: mode=%s force=false "
            "(session=%s messages=%d tokens=~%s)",
            auto_mode,
            getattr(agent, "session_id", None) or "none",
            len(messages),
            f"{approx_tokens:,}" if approx_tokens else "unknown",
        )
        existing_prompt = getattr(agent, "_cached_system_prompt", None)
        if not existing_prompt:
            existing_prompt = agent._build_system_prompt(system_message)
        return messages, existing_prompt

    codex_session = getattr(agent, "_codex_session", None)
    if codex_session is None:
        logger.info(
            "codex app-server compaction skipped: no active codex thread "
            "(session=%s messages=%d tokens=~%s)",
            getattr(agent, "session_id", None) or "none",
            len(messages),
            f"{approx_tokens:,}" if approx_tokens else "unknown",
        )
        existing_prompt = getattr(agent, "_cached_system_prompt", None)
        if not existing_prompt:
            existing_prompt = agent._build_system_prompt(system_message)
        return messages, existing_prompt

    logger.info(
        "codex app-server compaction started: session=%s messages=%d tokens=~%s",
        getattr(agent, "session_id", None) or "none",
        len(messages),
        f"{approx_tokens:,}" if approx_tokens else "unknown",
    )
    try:
        agent._emit_status(COMPACTION_STATUS)
    except Exception:
        pass

    result = codex_session.compact_thread()
    if getattr(result, "should_retire", False):
        try:
            codex_session.close()
        except Exception:
            pass
        agent._codex_session = None

    if getattr(result, "interrupted", False) or getattr(result, "error", None):
        try:
            agent._emit_warning(
                f"⚠ Codex app-server compaction failed: {result.error}"
            )
        except Exception:
            pass
        existing_prompt = getattr(agent, "_cached_system_prompt", None)
        if not existing_prompt:
            existing_prompt = agent._build_system_prompt(system_message)
        return messages, existing_prompt

    try:
        from agent.codex_runtime import (
            _record_codex_app_server_compaction,
            _record_codex_app_server_usage,
        )

        _record_codex_app_server_compaction(
            agent,
            result,
            approx_tokens=approx_tokens,
            force=True,
        )
        # An empty usage report must consume the pending post-compaction verdict
        # rather than leaving preflight deferral armed until some unrelated later
        # Codex turn supplies usage. Minimal external test engines may not expose
        # the ContextEngine update hook; preserve their existing bookkeeping.
        if hasattr(agent.context_compressor, "update_from_response"):
            _record_codex_app_server_usage(agent, result)
    except Exception:
        logger.debug("codex compaction bookkeeping failed", exc_info=True)

    try:
        from tools.file_tools import reset_file_dedup

        reset_file_dedup(task_id)
    except Exception:
        pass

    logger.info(
        "codex app-server compaction done: session=%s thread=%s turn=%s",
        getattr(agent, "session_id", None) or "none",
        getattr(result, "thread_id", None) or "",
        getattr(result, "turn_id", None) or "",
    )
    existing_prompt = getattr(agent, "_cached_system_prompt", None)
    if not existing_prompt:
        existing_prompt = agent._build_system_prompt(system_message)
    return messages, existing_prompt


def try_shrink_image_parts_in_messages(
    api_messages: list,
    *,
    max_dimension: int = 8000,
) -> bool:
    """Re-encode all native image parts at a smaller size to recover from
    image-too-large errors (Anthropic 5 MB, unknown other providers).

    Mutates ``api_messages`` in place. Returns True if any image part was
    actually replaced, False if there were no image parts to shrink or
    Pillow couldn't help (caller should surface the original error).

    Strategy: look for ``image_url`` / ``input_image`` parts carrying a
    ``data:image/...;base64,...`` payload, plus Anthropic-native
    ``{"type": "image", "source": {"type": "base64", ...}}`` blocks.
    For each one whose encoded size exceeds 4 MB (a safe target that slides
    under Anthropic's 5 MB ceiling with header overhead) or whose longest side
    exceeds ``max_dimension``, write the base64 to a tempfile, call
    ``vision_tools._resize_image_for_vision`` to produce a smaller data
    URL, and substitute it in place.

    Non-data-URL images (http/https URLs) are not touched — the provider
    fetches those itself and the size limit is different.
    """
    if not api_messages:
        return False

    try:
        from tools.vision_tools import _resize_image_for_vision
    except Exception as exc:
        logger.warning("image-shrink recovery: vision_tools unavailable — %s", exc)
        return False

    # 4 MB target leaves comfortable headroom under Anthropic's 5 MB.
    # Non-Anthropic providers we haven't observed rejecting are fine with
    # much larger; shrinking to 4 MB here loses quality but only fires
    # after a confirmed provider rejection, so the alternative is failure.
    target_bytes = 4 * 1024 * 1024
    # Anthropic enforces an 8000px per-side dimension cap independently of
    # the 5 MB byte cap.  In many-image requests, the provider can report a
    # lower cap (observed: 2000px).  The caller passes that parsed ceiling
    # when the rejection includes it.
    changed_count = 0
    # Track parts that are over the target but could NOT be shrunk under it.
    # If any survive, retrying is pointless — the same oversized payload will
    # be re-sent and rejected again, wasting the single retry budget.  We only
    # report success (caller retries) when every over-threshold image was
    # actually brought under the target.
    unshrinkable_oversized = 0

    def _decode_pixels(data_url: str) -> Optional[tuple]:
        """Return ``(width, height)`` of a base64 data URL, or None on failure.

        Soft-depends on Pillow; returns None (caller falls back to a
        bytes-only check) if Pillow is missing or the payload is corrupt.
        """
        try:
            import base64 as _b64_dim
            import io as _io_dim
            header_d, _, data_d = data_url.partition(",")
            if not data_d or not data_url.startswith("data:"):
                return None
            from PIL import Image as _PILImage
            with _PILImage.open(_io_dim.BytesIO(_b64_dim.b64decode(data_d))) as _img:
                return _img.size
        except Exception:
            return None

    def _shrink_data_url(url: str) -> tuple:
        """Return ``(resized_url, unshrinkable)`` for a data URL.

        ``resized_url`` is a smaller/dimension-correct data URL, or None when
        no rewrite was applied.  ``unshrinkable`` is True only when the image
        exceeded a constraint (byte-size or dimensions) and the resize failed
        to satisfy *that same* constraint — so the caller knows retrying is
        pointless even if a different image in the request shrank.
        """
        if not isinstance(url, str) or not url.startswith("data:"):
            return None, False

        # Determine which constraint is binding.  The accept/reject gate below
        # MUST be checked against the same axis that triggered the shrink: a
        # downscaled screenshot PNG routinely re-encodes to *more* bytes than
        # the original (PNG compression is non-monotonic in image size — a
        # smaller raster with LANCZOS resampling noise compresses worse than a
        # larger smooth one).  Rejecting a pixel-correct downscale purely
        # because its bytes grew permanently wedges sessions on the Anthropic
        # many-image 2000px path (#48013).
        needs_shrink = len(url) > target_bytes  # over byte budget
        triggered_by = "bytes" if needs_shrink else None
        if not needs_shrink:
            # Bytes are fine — check pixel dimensions against the provider's
            # reported per-side cap.  A screenshot can be tiny in bytes yet
            # too large in pixels.
            dims = _decode_pixels(url)
            if dims is None:
                # Pillow missing or corrupt data — fall back to byte-only.
                return None, False
            if max(dims) <= max_dimension:
                return None, False  # both bytes and pixels are within limits
            needs_shrink = True
            triggered_by = "dimension"

        try:
            header, _, data = url.partition(",")
            mime = "image/jpeg"
            if header.startswith("data:"):
                mime_part = header[len("data:"):].split(";", 1)[0].strip()
                if mime_part.startswith("image/"):
                    mime = mime_part
            import base64 as _b64
            raw = _b64.b64decode(data)
            suffix = {
                "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp",
                "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/bmp": ".bmp",
            }.get(mime, ".jpg")
            tmp = tempfile.NamedTemporaryFile(
                prefix="hermes_shrink_", suffix=suffix, delete=False,
            )
            try:
                tmp.write(raw)
                tmp.close()
                resized = _resize_image_for_vision(
                    Path(tmp.name),
                    mime_type=mime,
                    max_base64_bytes=target_bytes,
                    max_dimension=max_dimension,
                )
            finally:
                try:
                    Path(tmp.name).unlink(missing_ok=True)
                except Exception:
                    pass
            if not resized:
                # Resize returned nothing — Pillow couldn't help.
                return None, True
            if triggered_by == "bytes":
                # Byte budget is the binding constraint — bytes must shrink.
                if len(resized) >= len(url):
                    return None, True  # re-encode made it bigger
                # The per-side dimension cap is ALSO an active provider
                # constraint on this request (the caller passes the parsed cap
                # to both this helper and the resizer).  _resize_image_for_vision
                # returns a best-effort, possibly-over-cap blob when it
                # exhausts its halving budget — it freezes the long side once
                # the short side hits its 64px floor, so a very-high-aspect
                # image can stay over the cap even after bytes shrank.  If the
                # output is still over the cap, retrying would re-400 on
                # dimensions; treat it as unshrinkable.  (Skip when dims can't
                # be decoded — preserves historical byte-only behaviour.)
                new_dims = _decode_pixels(resized)
                if new_dims is not None and max(new_dims) > max_dimension:
                    return None, True
                return resized, False
            # triggered_by == "dimension": the per-side cap is binding.  The
            # re-encode may have grown in bytes; accept it as long as it is now
            # within the dimension cap.  Verify the new dimensions when we can.
            new_dims = _decode_pixels(resized)
            if new_dims is not None:
                if max(new_dims) <= max_dimension:
                    return resized, False
                # Still over the per-side cap — the resize didn't satisfy it.
                return None, True
            # Couldn't verify the re-encode's dimensions (corrupt output or
            # Pillow gone mid-call).  Fall back to the historical "bytes must
            # shrink" gate so we never accept an unverifiable, byte-larger blob.
            if len(resized) >= len(url):
                return None, True
            return resized, False
        except Exception as exc:
            logger.warning("image-shrink recovery: re-encode failed — %s", exc)
            return None, triggered_by is not None

    def _source_to_data_url(source: Any) -> Optional[str]:
        if not isinstance(source, dict) or source.get("type") != "base64":
            return None
        data = source.get("data")
        if not isinstance(data, str) or not data:
            return None
        media_type = str(source.get("media_type") or "image/jpeg").strip()
        if not media_type.startswith("image/"):
            media_type = "image/jpeg"
        return f"data:{media_type};base64,{data}"

    def _write_data_url_to_source(source: dict, data_url: str) -> None:
        header, _, data = data_url.partition(",")
        media_type = "image/jpeg"
        if header.startswith("data:"):
            candidate = header[len("data:"):].split(";", 1)[0].strip()
            if candidate.startswith("image/"):
                media_type = candidate
        source["type"] = "base64"
        source["media_type"] = media_type
        source["data"] = data

    for msg in api_messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "image":
                source = part.get("source")
                url = _source_to_data_url(source)
                resized, unshrinkable = _shrink_data_url(url or "")
                if resized and isinstance(source, dict):
                    _write_data_url_to_source(source, resized)
                    changed_count += 1
                elif unshrinkable:
                    unshrinkable_oversized += 1
                continue
            if ptype not in {"image_url", "input_image"}:
                continue
            image_value = part.get("image_url")
            # OpenAI chat.completions: {"image_url": {"url": "data:..."}}
            # OpenAI Responses: {"image_url": "data:..."}
            if isinstance(image_value, dict):
                url = image_value.get("url", "")
                resized, unshrinkable = _shrink_data_url(url)
                if resized:
                    image_value["url"] = resized
                    changed_count += 1
                elif unshrinkable:
                    unshrinkable_oversized += 1
            elif isinstance(image_value, str):
                resized, unshrinkable = _shrink_data_url(image_value)
                if resized:
                    part["image_url"] = resized
                    changed_count += 1
                elif unshrinkable:
                    unshrinkable_oversized += 1

    if changed_count:
        logger.info(
            "image-shrink recovery: re-encoded %d image part(s) to fit under %.0f MB",
            changed_count, target_bytes / (1024 * 1024),
        )
    if unshrinkable_oversized:
        # At least one oversized image could not be shrunk under the target.
        # Retrying would re-send it and fail identically, so signal "no
        # progress" even if other parts shrank — the caller will surface the
        # original error rather than burning its single retry on a no-op.
        logger.warning(
            "image-shrink recovery: %d oversized image part(s) could not be "
            "shrunk under %.0f MB — not retrying (would re-send rejected payload)",
            unshrinkable_oversized, target_bytes / (1024 * 1024),
        )
        return False
    return changed_count > 0


__all__ = [
    "COMPACTION_STATUS",
    "COMPACTION_STATUS_MARKER",
    "check_compression_model_feasibility",
    "replay_compression_warning",
    "compress_context",
    "try_shrink_image_parts_in_messages",
]
