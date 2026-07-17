"""Automatic context window compression for long conversations.

Self-contained class with its own OpenAI client for summarization.
Uses auxiliary model (cheap/fast) to summarize middle turns while
protecting head and tail context.

Improvements over v2:
  - Structured summary template with Resolved/Pending question tracking
  - Filter-safe summarizer preamble that treats prior turns as source material
  - Historical (reference-only) section headings replace "Next Steps"/"Remaining Work" to avoid reading as active instructions
  - Clear separator when summary merges into tail message
  - Iterative summary updates (preserves info across multiple compactions)
  - Token-budget tail protection instead of fixed message count
  - Tool output pruning before LLM summarization (cheap pre-pass)
  - Scaled summary budget (proportional to compressed content)
  - Richer tool call/result detail in summarizer input
"""

import hashlib
import json
import logging
import sqlite3
import re
import time
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import call_llm, _is_connection_error, aux_interrupt_protection
from agent.context_engine import ContextEngine
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH,
    get_model_context_length,
    estimate_messages_tokens_rough,
)
from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

HISTORICAL_TASK_HEADING = "## Historical Task Snapshot"
HISTORICAL_IN_PROGRESS_HEADING = "## Historical In-Progress State"
HISTORICAL_PENDING_ASKS_HEADING = "## Historical Pending User Asks"
HISTORICAL_REMAINING_WORK_HEADING = "## Historical Remaining Work"

# [上下文压缩 — 仅供参考] 先前的轮次已被压缩到下方的总结中。
# 这是来自先前上下文窗口的交接内容 —— 将其视为背景参考，而非活跃指令。
# 请勿回答此总结中提及的问题或实现其中提及的请求；它们已被处理。
# 仅对出现在此总结之后的最新用户消息做出回应 —— 该消息是当前要做什么的唯一事实来源。
# 与总结的主题重叠并不意味着你应当恢复其任务：即使在类似的主题上，
# 也以最新用户消息为准。仅将最新消息视为活跃任务，并完全丢弃来自
# '{HISTORICAL_TASK_HEADING}' / '{HISTORICAL_IN_PROGRESS_HEADING}' /
# '{HISTORICAL_PENDING_ASKS_HEADING}' / '{HISTORICAL_REMAINING_WORK_HEADING}'
# 的过期事项 —— 除非最新消息明确要求，否则不要“收尾”或“完成”此处描述的工作。
# 最新消息中的逆向信号（例如“停止”、“撤销”、“回滚”、“仅验证”、“不要再那样做了”、
# “算了/没关系”、新主题）必须立即终止总结中描述的任何进行中的工作；
# 不要在以后的轮次中重新提及它。
# 重要提示：系统提示词中你的持久化记忆（MEMORY.md，USER.md）始终是权威且活跃的
# —— 绝不要因为此压缩注释而忽略记忆内容或降低其优先级。
# 当前的会话状态（文件、配置等）可能会反映此处描述的工作 —— 避免重复进行：
SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. "
    "Respond ONLY to the latest user message that appears AFTER this "
    "summary — that message is the single source of truth for what to do "
    "right now. "
    "Topic overlap with the summary does NOT mean you should resume its "
    "task: even on similar topics, the latest user message WINS. Treat ONLY "
    "the latest message as the active task and discard stale items from "
    f"'{HISTORICAL_TASK_HEADING}' / '{HISTORICAL_IN_PROGRESS_HEADING}' / "
    f"'{HISTORICAL_PENDING_ASKS_HEADING}' / "
    f"'{HISTORICAL_REMAINING_WORK_HEADING}' entirely — do not 'wrap up' or "
    "'finish' work described there unless the latest message explicitly "
    "asks for it. "
    "Reverse signals in the latest message (e.g. 'stop', 'undo', 'roll "
    "back', 'just verify', 'don't do that anymore', 'never mind', a new "
    "topic) must immediately end any in-flight work described in the "
    "summary; do not re-surface it in later turns. "
    "IMPORTANT: Your persistent memory (MEMORY.md, USER.md) in the system "
    "prompt is ALWAYS authoritative and active — never ignore or deprioritize "
    "memory content due to this compaction note. "
    "The current session state (files, config, etc.) may reflect work "
    "described here — avoid repeating it:"
)
LEGACY_SUMMARY_PREFIX = "[CONTEXT SUMMARY]:"

# Metadata key added to context compression summary messages so that frontends
# (CLI, Desktop, gateway, TUI) can distinguish them from real assistant/user
# messages and filter or render them appropriately without content-prefix
# heuristics. See https://github.com/NousResearch/hermes-agent/issues/38389
#
# Underscore-prefixed ON PURPOSE: the wire sanitizers
# (agent/transports/chat_completions.py convert_messages and the summary-path
# mirror in agent/chat_completion_helpers.py) strip every top-level message
# key starting with "_" before the request leaves the process. Strict
# OpenAI-compatible gateways (Fireworks, Mistral, Moonshot/Kimi, opencode-go)
# reject payloads carrying unknown keys with "Extra inputs are not permitted",
# poisoning every subsequent request in the session — a bare key like
# "is_compressed_summary" would reach the wire and trip exactly that.
COMPRESSED_SUMMARY_METADATA_KEY = "_compressed_summary"
_DB_PERSISTED_MARKER = "_db_persisted"


def _fresh_compaction_message_copy(msg: Dict[str, Any]) -> Dict[str, Any]:
    """复制用于压缩组装的消息，不包含持久化标记。

    实时缓存网关的转录记录（transcripts）会在增量刷新期间标记上 ``_db_persisted``。
    浅拷贝 ``.copy()`` 会将该标记传播到轮转后的压缩列表中，导致
    ``_flush_messages_to_session_db`` 在写入新的子会话时跳过每一行（#57491）。

    这会在拷贝处（copy site）剥离该标记（此处的意图最清晰，且开销极低），
    但最权威的保障是 ``compress()`` 中单次的终结扫除（``_strip_persistence_markers``）：
    无论未来的重构会增加多少个中间拷贝处，任何消息在离开 ``compress()`` 时
    都绝不能携带 ``_db_persisted``。
    """
    fresh = msg.copy()
    # 删除缓存标记
    fresh.pop(_DB_PERSISTED_MARKER, None)
    return fresh


def _strip_persistence_markers(messages: List[Dict[str, Any]]) -> None:
    """强制执行压缩不变性：任何组装好的消息都不得携带
    会话存储持久化标记（session-store persistence marker）。

    ``compress()`` 会从处于活动状态的缓存网关脚本中复制受保护的 head/tail 消息，
    在会话的生命周期内，该脚本会在每条消息上盖上 ``_db_persisted`` 戳记。
    如果任何被复制的字典保留了该标记，那么向子会话的轮转刷新（rotation flush）
    将会跳过它，导致压缩后的脚本在 ``state.db`` 中丢失（#57491）。
    在每个复制位置进行清除是必要的，但这只是“基于位置”的——如果在组装循环
    之后添加了新的复制位置，就会再次发生泄漏。
    而这次统一的终端清扫则转为从“结构上”提供保证：在完全组装好的列表上
    运行一次，这样无论复制发生在何处，该不变性都能保持成立。
    此操作为原地修改（这些字典是压缩本地的副本）。
    """
    for msg in messages:
        if isinstance(msg, dict):
            msg.pop(_DB_PERSISTED_MARKER, None)


# Appended to every standalone summary message (and to the merged-into-tail
# prefix) so the model has an unambiguous "summary ends here" boundary.
# Without it, weak models read the verbatim "## Active Task" quote as fresh
# user input (#11475, #14521) or regurgitate an assistant-role summary as
# their own output (#33256).
_SUMMARY_END_MARKER = (
    "--- END OF CONTEXT SUMMARY — "
    "respond to the message below, not the summary above ---"
)

# When the summary must be merged into the first tail message (the alternation
# corner case where a standalone summary role would collide with both head and
# tail), the tail message's own prior content is preserved BEFORE the summary,
# wrapped in these delimiters so the model doesn't read it as a fresh message.
# The summary prefix therefore lands AFTER _MERGED_SUMMARY_DELIMITER rather than
# at the start of the message, so _is_context_summary_content must look past it.
_MERGED_PRIOR_CONTEXT_HEADER = "[PRIOR CONTEXT — for reference only; not a new message]"
_MERGED_SUMMARY_DELIMITER = "[END OF PRIOR CONTEXT — COMPACTION SUMMARY BELOW]"

# Handoff prefixes that shipped in earlier releases. A summary persisted under
# one of these can be inherited into a resumed lineage (#35344); when it is
# re-normalized on re-compaction we must strip the OLD prefix too, otherwise the
# stale directive it carried (e.g. "resume exactly from Active Task") survives
# embedded in the body and keeps hijacking replies. Keep newest-first; entries
# are matched literally. Add a frozen copy here whenever SUMMARY_PREFIX changes.
_HISTORICAL_SUMMARY_PREFIXES = (
    # Carveout era (#41607/#38364/#42812): "consistent → use as background"
    # licensed stale-task resumption on topic overlap.
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. "
    "Respond ONLY to the latest user message that appears AFTER this "
    "summary — that message is the single source of truth for what to do "
    "right now. "
    "If the latest user message is consistent with the '## Active Task' "
    "section, you may use the summary as background. If the latest user "
    "message contradicts, supersedes, changes topic from, or in any way "
    "diverges from '## Active Task' / '## In Progress' / '## Pending User "
    "Asks' / '## Remaining Work', the latest message WINS — discard those "
    "stale items entirely and do not 'wrap up the old task first'. "
    "Reverse signals in the latest message (e.g. 'stop', 'undo', 'roll "
    "back', 'just verify', 'don't do that anymore', 'never mind', a new "
    "topic) must immediately end any in-flight work described in the "
    "summary; do not re-surface it in later turns. "
    "IMPORTANT: Your persistent memory (MEMORY.md, USER.md) in the system "
    "prompt is ALWAYS authoritative and active — never ignore or deprioritize "
    "memory content due to this compaction note. "
    "The current session state (files, config, etc.) may reflect work "
    "described here — avoid repeating it:",
    # Pre-#35344: contained the self-contradicting "resume exactly" directive.
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. "
    "Your current task is identified in the '## Active Task' section of the "
    "summary — resume exactly from there. "
    "Respond ONLY to the latest user message "
    "that appears AFTER this summary. The current session state (files, "
    "config, etc.) may reflect work described here — avoid repeating it:",
)

# Minimum tokens for the summary output
_MIN_SUMMARY_TOKENS = 2000
# Proportion of compressed content to allocate for summary
_SUMMARY_RATIO = 0.20
# Absolute ceiling for summary tokens (even on very large context windows).
# Summaries must stay within a 1K-10K token envelope — anything larger is
# itself a context-pressure source and slows every compaction.
_SUMMARY_TOKENS_CEILING = 10_000

# Placeholder used when pruning old tool results
_PRUNED_TOOL_PLACEHOLDER = "[Old tool output cleared to save context space]"

# Chars per token rough estimate
_CHARS_PER_TOKEN = 4
# Flat token cost per attached image part.  Real cost varies by provider and
# dimensions (Anthropic ≈ width×height/750, GPT-4o up to ~1700 for
# high-detail 2048×2048, Gemini 258/tile), but 1600 is a realistic ceiling
# that keeps compression budgeting honest for multi-image conversations.
# Matches Claude Code's IMAGE_TOKEN_ESTIMATE constant.
_IMAGE_TOKEN_ESTIMATE = 1600
# Same figure expressed in the char-budget currency the rest of the
# compressor speaks in.  Used when accumulating message "content length"
# for tail-cut decisions.
_IMAGE_CHAR_EQUIVALENT = _IMAGE_TOKEN_ESTIMATE * _CHARS_PER_TOKEN
_SUMMARY_FAILURE_COOLDOWN_SECONDS = 600

# Hard ceiling for the deterministic summary-failure handoff.  The fallback is
# only meant to preserve continuity anchors from the dropped window, not to
# become another unbounded transcript copy after the LLM summarizer failed.
_FALLBACK_SUMMARY_MAX_CHARS = 8_000
_FALLBACK_TURN_MAX_CHARS = 700
_AUTO_FOCUS_MAX_TURNS = 3
_AUTO_FOCUS_TURN_MAX_CHARS = 260
_AUTO_FOCUS_MAX_CHARS = 700
# Keep a short run of recent messages verbatim even when the token budget is
# already exhausted.  The public ``protect_last_n`` default is intentionally
# high for small/light tails, but using all 20 as a hard floor here would bring
# back the old large-tool-output case where nothing can be compacted.
_MAX_TAIL_MESSAGE_FLOOR = 8

# Models with context windows below this get their compression threshold
# floored at ``_SMALL_CTX_THRESHOLD_PERCENT`` (raise-only — an explicitly
# higher user/model threshold always wins).  At the default 50% trigger a
# 128K-262K model compacts with only ~64-131K consumed; the incompressible
# floor (system prompt + tool schemas + protected tail + rolling summary)
# eats most of the reclaimed headroom, so compaction re-fires every 1-2
# turns and the session spends most of its wall-clock summarizing.
_SMALL_CTX_WINDOW_LIMIT = 512_000
_SMALL_CTX_THRESHOLD_PERCENT = 0.75


_PATH_MENTION_RE = re.compile(r"(?:/|~/?|[A-Za-z]:\\)[^\s`'\")\]}<>]+")

# MEDIA delivery directives must not reach the summarizer — if one leaks into
# the summary, the downstream model may re-emit it as an active directive on
# the next turn, triggering bogus attachment sends (#14665).
_MEDIA_DIRECTIVE_RE = re.compile(r"MEDIA:\S+")


def _dedupe_append(items: list[str], value: str, *, limit: int) -> None:
    value = value.strip()
    if value and value not in items and len(items) < limit:
        items.append(value)


def _extract_tool_call_name_and_args(tool_call: Any) -> tuple[str, str]:
    """Return a best-effort ``(name, arguments)`` pair for dict/object tool calls."""
    if isinstance(tool_call, dict):
        fn = tool_call.get("function") or {}
        return str(fn.get("name") or "unknown"), str(fn.get("arguments") or "")

    fn = getattr(tool_call, "function", None)
    if fn is None:
        return "unknown", ""
    return str(getattr(fn, "name", None) or "unknown"), str(getattr(fn, "arguments", None) or "")


def _extract_tool_call_id(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("id") or "")
    return str(getattr(tool_call, "id", "") or "")


def _collect_path_mentions(text: str, relevant_files: list[str], *, limit: int = 12) -> None:
    for match in _PATH_MENTION_RE.findall(text):
        _dedupe_append(relevant_files, match.rstrip(".,:;"), limit=limit)


def _content_length_for_budget(raw_content: Any) -> int:
    """Return the effective char-length of a message's content for token budgeting.

    Plain strings: ``len(content)``. Multimodal lists: sum of text-part
    ``len(text)`` plus a flat ``_IMAGE_CHAR_EQUIVALENT`` per image part
    (``image_url`` / ``input_image`` / Anthropic-style ``image``). This
    keeps the compressor from treating a turn with 5 attached images as
    near-zero tokens just because the text part is empty.
    """
    if isinstance(raw_content, str):
        return len(raw_content)
    if not isinstance(raw_content, list):
        return len(str(raw_content or ""))

    total = 0
    for p in raw_content:
        if isinstance(p, str):
            total += len(p)
            continue
        if not isinstance(p, dict):
            total += len(str(p))
            continue
        ptype = p.get("type")
        if ptype in {"image_url", "input_image", "image"}:
            total += _IMAGE_CHAR_EQUIVALENT
        else:
            # text / input_text / tool_result-with-text / anything else with
            # a text field.  Ignore the raw base64 payload inside image_url
            # dicts — dimensions don't matter, only whether it's an image.
            total += len(p.get("text", "") or "")
    return total


def _serialized_length_for_budget(value: Any) -> int:
    """Return a stable char-length for non-content replay/metadata fields."""
    if value is None or value == "":
        return 0
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return len(str(value))


# Provider replay/metadata fields that ride the wire on every request but are
# invisible to ``msg["content"]``/``msg["tool_calls"]`` accounting.  Codex
# Responses sessions in particular carry ``codex_reasoning_items`` blobs of
# ``encrypted_content`` that can dominate the serialized session (a measured
# 214-turn session held ~115K tokens / 27% of its payload there — #55572).
_REPLAY_BUDGET_KEYS = (
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
)


def _estimate_msg_budget_tokens(msg: dict) -> int:
    """Token estimate for one message in the tail-protection budget walks.

    Counts the message content plus the **full** ``tool_call`` envelope —
    ``id``, ``type``, ``function.name`` and JSON structure — not just
    ``function.arguments``.  Counting only the arguments string undercounted
    assistant turns that fan out into parallel tool calls by 2-15x (a
    4-tool-call turn measures ~73 vs ~1,090 real tokens), so the protected
    tail overshot ``tail_token_budget`` and compression became ineffective.
    See issue #28053.

    Also counts provider replay fields (``codex_reasoning_items`` etc. —
    see ``_REPLAY_BUDGET_KEYS``).  The preflight "should I compress?"
    estimator sees the full message shape, so the tail walk must use the
    same size class; otherwise an assistant message with tiny visible
    content but large hidden replay blobs is protected as if it were small,
    the post-compression session stays near the context limit, and
    compaction re-fires continuously (#55572).  Accounting-only: replay
    fields are never mutated or pruned here.
    """
    content_len = _content_length_for_budget(msg.get("content") or "")
    tokens = content_len // _CHARS_PER_TOKEN + 10  # +10 for role/key overhead
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            tokens += len(str(tc)) // _CHARS_PER_TOKEN
    for key in _REPLAY_BUDGET_KEYS:
        tokens += _serialized_length_for_budget(msg.get(key)) // _CHARS_PER_TOKEN
    return tokens


def _content_text_for_contains(content: Any) -> str:
    """Return a best-effort text view of message content.

    Used only for substring checks when we need to know whether we've already
    appended a note to a message. Keeps multimodal lists intact elsewhere.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content)


def _append_text_to_content(content: Any, text: str, *, prepend: bool = False) -> Any:
    """安全地将纯文本追加（append）或前置（prepend）到消息内容中。

    压缩有时需要向现有消息中添加注释或合并总结。
    消息内容可能是纯文本，也可能是多模态的块列表（multimodal list of blocks），
    因此直接进行字符串拼接并不总是安全的。
    """
    if content is None:
        return text
    if isinstance(content, str):
        return text + content if prepend else content + text
    if isinstance(content, list):
        text_block = {"type": "text", "text": text}
        return [text_block, *content] if prepend else [*content, text_block]
    rendered = str(content)
    return text + rendered if prepend else rendered + text


def _strip_image_parts_from_parts(parts: Any) -> Any:
    """Strip image parts from an OpenAI-style content-parts list.

    Returns a new list with image_url / image / input_image parts replaced
    by a text placeholder, or None if the list had no images (callers
    skip the replacement in that case). Used by the compressor to prune
    old computer_use screenshots.
    """
    if not isinstance(parts, list):
        return None
    had_image = False
    out = []
    for part in parts:
        if not isinstance(part, dict):
            out.append(part)
            continue
        ptype = part.get("type")
        if ptype in {"image", "image_url", "input_image"}:
            had_image = True
            out.append({"type": "text", "text": "[screenshot removed to save context]"})
        else:
            out.append(part)
    return out if had_image else None


def _truncate_tool_call_args_json(args: str, head_chars: int = 200) -> str:
    """在保持 JSON 有效性的同时，缩减工具调用（tool-call）参数 JSON 块内部的过长字符串值。

    工具调用中的 ``function.arguments`` 字段是一个传递给 LLM 服务商的 JSON 编码字符串；
    下游服务商会对其进行严格校验，当其格式不正确（not well-formed）时，会返回一个不可重试的
     400 错误。早期的实现是在固定的字节偏移量处直接对原始 JSON 进行切片，
    并追加 ``...[truncated]`` —— 这经常会产生如下形式的字符串：:

        {"path": "/foo/bar", "content": "# long markdown
        ...[truncated]

    也就是说，产生了一个未闭合的字符串且缺失了结尾的右花括号。例如，MiniMax
    会以 ``invalid function arguments json string`` 为由拒绝此类请求，
    从而导致会话卡死，在每一轮中都不断重复发送这段损坏的历史记录。
    观察到的死循环具体参见 issue #11762。

    该辅助函数会解析这些参数，缩减解析后结构内部的过长字符串叶子节点（string leaves），
    然后重新进行序列化。非字符串值（路径、整数、布尔值）将保持原样。如果参数本身
    一开始就不是有效的 JSON —— 某些模型后端会使用非 JSON 格式的工具参数 —— 则会
    原样返回原始字符串，而不是将其替换为我们和后端都无法解析的内容。
    """
    try:
        parsed = json.loads(args)
    except (ValueError, TypeError):
        return args

    def _shrink(obj: Any) -> Any:
        if isinstance(obj, str):
            if len(obj) > head_chars:
                return obj[:head_chars] + "...[truncated]"
            return obj
        if isinstance(obj, dict):
            return {k: _shrink(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_shrink(v) for v in obj]
        return obj

    shrunken = _shrink(parsed)
    # ensure_ascii=False preserves CJK/emoji instead of bloating with \uXXXX
    return json.dumps(shrunken, ensure_ascii=False)


_IMAGE_PART_TYPES = frozenset({"image_url", "input_image", "image"})


def _is_image_part(part: Any) -> bool:
    """True if ``part`` is a multimodal image content block.

    Recognizes all three shapes the agent handles:
      - OpenAI chat.completions: ``{"type": "image_url", "image_url": ...}``
      - OpenAI Responses API:    ``{"type": "input_image", "image_url": "..."}``
      - Anthropic native:        ``{"type": "image", "source": {...}}``
    """
    if not isinstance(part, dict):
        return False
    return part.get("type") in _IMAGE_PART_TYPES


def _content_has_images(content: Any) -> bool:
    """True if a message's ``content`` is a multimodal list with image parts."""
    if not isinstance(content, list):
        return False
    return any(_is_image_part(p) for p in content)


def _strip_images_from_content(content: Any) -> Any:
    """Return a copy of ``content`` with every image part replaced by a
    short text placeholder.

    - String content is returned unchanged.
    - Non-list, non-string content is returned unchanged.
    - List content: image parts become ``{"type": "text", "text": "[Attached
      image — stripped after compression]"}``; other parts are preserved as-is.

    Input is never mutated.
    """
    if not isinstance(content, list):
        return content
    if not any(_is_image_part(p) for p in content):
        return content

    new_parts: List[Any] = []
    for p in content:
        if _is_image_part(p):
            new_parts.append({
                "type": "text",
                "text": "[Attached image — stripped after compression]",
            })
        else:
            new_parts.append(p)
    return new_parts


def _strip_historical_media(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """用占位符文本替换旧消息中的图片部分。

    锚点是 *最后一条* 包含任何图片内容的用户消息。在该锚点之前的
    每一条消息，其图片部分都会被替换为一个简短的占位符，从而使发出的
    请求停止在每一轮对话中重复发送相同的数 MB 大小的 base-64 图片数据块。

    如果没有用户消息包含图片，则原样返回列表。
    如果唯一包含图片的用户消息就是最开始的第一条（前面没有可清除的内容），
    则原样返回列表。

    仅对受影响的消息进行浅拷贝；绝对不会修改输入数据（input is never mutated）。
    移植自 Kilo-Org/kilocode#9434（并针对 hermes 压缩器输出的
    OpenAI 风格消息格式进行了适配）。
    """
    if not messages:
        return messages

    # Find the newest user message that carries at least one image part.
    # We anchor on image-bearing user messages (not all user messages) so
    # a plain text follow-up after a big-image turn still strips the old
    # image — matching the problem kilocode#9434 set out to solve.
    anchor = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        if _content_has_images(msg.get("content")):
            anchor = i
            break

    if anchor <= 0:
        # No image-bearing user message, or it's the very first message —
        # nothing before it to strip.
        return messages

    changed = False
    result: List[Dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if i >= anchor or not isinstance(msg, dict):
            result.append(msg)
            continue
        content = msg.get("content")
        if not _content_has_images(content):
            result.append(msg)
            continue
        new_msg = msg.copy()
        new_msg["content"] = _strip_images_from_content(content)
        result.append(new_msg)
        changed = True

    return result if changed else messages


def _summarize_tool_result(tool_name: str, tool_args: str, tool_content: str) -> str:
    """生成包含有效信息的工具调用及结果的单行总结。

    在压缩前的修剪阶段使用，用关于工具行为的简短且有用的描述来替换庞大的
    工具输出，而不是使用不包含任何信息的通用占位符。

    返回类似如下的字符串：:

        [terminal] ran `npm test` -> exit 0, 47 lines output
        [read_file] read config.py from line 1 (1,200 chars)
        [search_files] content search for 'compress' in agent/ -> 12 matches
    """
    try:
        args = json.loads(tool_args) if tool_args else {}
    except (json.JSONDecodeError, TypeError):
        args = {}

    content = tool_content or ""
    content_len = len(content)
    line_count = content.count("\n") + 1 if content.strip() else 0

    if tool_name == "terminal":
        cmd = args.get("command", "")
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        exit_match = re.search(r'"exit_code"\s*:\s*(-?\d+)', content)
        exit_code = exit_match.group(1) if exit_match else "?"
        return f"[terminal] ran `{cmd}` -> exit {exit_code}, {line_count} lines output"

    if tool_name == "read_file":
        path = args.get("path", "?")
        offset = args.get("offset", 1)
        return f"[read_file] read {path} from line {offset} ({content_len:,} chars)"

    if tool_name == "write_file":
        path = args.get("path", "?")
        written_lines = args.get("content", "").count("\n") + 1 if args.get("content") else "?"
        return f"[write_file] wrote to {path} ({written_lines} lines)"

    if tool_name == "search_files":
        pattern = args.get("pattern", "?")
        path = args.get("path", ".")
        target = args.get("target", "content")
        match_count = re.search(r'"total_count"\s*:\s*(\d+)', content)
        count = match_count.group(1) if match_count else "?"
        return f"[search_files] {target} search for '{pattern}' in {path} -> {count} matches"

    if tool_name == "patch":
        path = args.get("path", "?")
        mode = args.get("mode", "replace")
        return f"[patch] {mode} in {path} ({content_len:,} chars result)"

    if tool_name in {"browser_navigate", "browser_click", "browser_snapshot",
                     "browser_type", "browser_scroll", "browser_vision"}:
        url = args.get("url", "")
        ref = args.get("ref", "")
        detail = f" {url}" if url else (f" ref={ref}" if ref else "")
        return f"[{tool_name}]{detail} ({content_len:,} chars)"

    if tool_name == "web_search":
        query = args.get("query", "?")
        return f"[web_search] query='{query}' ({content_len:,} chars result)"

    if tool_name == "web_extract":
        urls = args.get("urls", [])
        url_desc = urls[0] if isinstance(urls, list) and urls else "?"
        if isinstance(urls, list) and len(urls) > 1:
            url_desc += f" (+{len(urls) - 1} more)"
        return f"[web_extract] {url_desc} ({content_len:,} chars)"

    if tool_name == "delegate_task":
        goal = args.get("goal", "")
        if len(goal) > 60:
            goal = goal[:57] + "..."
        return f"[delegate_task] '{goal}' ({content_len:,} chars result)"

    if tool_name == "execute_code":
        code_preview = (args.get("code") or "")[:60].replace("\n", " ")
        if len(args.get("code", "")) > 60:
            code_preview += "..."
        return f"[execute_code] `{code_preview}` ({line_count} lines output)"

    if tool_name in {"skill_view", "skills_list", "skill_manage"}:
        name = args.get("name", "?")
        return f"[{tool_name}] name={name} ({content_len:,} chars)"

    if tool_name == "vision_analyze":
        question = args.get("question", "")[:50]
        return f"[vision_analyze] '{question}' ({content_len:,} chars)"

    if tool_name == "memory":
        action = args.get("action", "?")
        target = args.get("target", "?")
        return f"[memory] {action} on {target}"

    if tool_name == "todo":
        return "[todo] updated task list"

    if tool_name == "clarify":
        return "[clarify] asked user a question"

    if tool_name == "text_to_speech":
        return f"[text_to_speech] generated audio ({content_len:,} chars)"

    if tool_name == "cronjob":
        action = args.get("action", "?")
        return f"[cronjob] {action}"

    if tool_name == "process":
        action = args.get("action", "?")
        sid = args.get("session_id", "?")
        return f"[process] {action} session={sid}"

    # Generic fallback
    first_arg = ""
    for k, v in list(args.items())[:2]:
        sv = str(v)[:40]
        first_arg += f" {k}={sv}"
    return f"[{tool_name}]{first_arg} ({content_len:,} chars result)"


class ContextCompressor(ContextEngine):
    """Default context engine — compresses conversation context via lossy summarization.

    Algorithm:
      1. Prune old tool results (cheap, no LLM call)
      2. Protect head messages (system prompt + first exchange)
      3. Protect tail messages by token budget (most recent ~20K tokens)
      4. Summarize middle turns with structured LLM prompt
      5. On subsequent compactions, iteratively update the previous summary
    """

    @property
    def name(self) -> str:
        return "compressor"

    def on_session_reset(self) -> None:
        """Reset all per-session state for /new or /reset."""
        super().on_session_reset()
        self._context_probed = False
        self._context_probe_persistable = False
        self._previous_summary = None
        self._last_summary_error = None
        self._last_summary_dropped_count = 0
        self._last_summary_fallback_used = False
        self._last_aux_model_failure_error = None
        self._last_aux_model_failure_model = None
        self._last_compression_savings_pct = 100.0
        self._ineffective_compression_count = 0
        self._verify_compaction_cleared_threshold = False
        self._last_compression_made_progress = False
        self._summary_failure_cooldown_until = 0.0  # transient errors must not block a fresh session
        self._last_summary_error = None
        self._last_compress_aborted = False
        self.last_real_prompt_tokens = 0
        self.last_compression_rough_tokens = 0
        self.last_rough_tokens_when_real_prompt_fit = 0
        self.awaiting_real_usage_after_compression = False

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """在一个真正的会话边界处，清除所有针对该会话的压缩状态。

        会话结束（CLI 退出、网关过期、会话 ID 轮转）会走这个方法，
        而不是走 ``on_session_reset()``（针对 /new、/reset）。
        最初的修复方案（#38788）只清除了 ``_previous_summary``，但跨会话污染的
        风险同样适用于 ``on_session_reset()`` 所清除的每一个特定会话变量：
        过期的 ``_ineffective_compression_count`` 可能会抑制后续活动会话中的压缩；
        ``_summary_failure_cooldown_until`` 可能会阻止摘要的生成；
        ``_last_compress_aborted`` 可能会让调用者误以为压缩仍处于中止状态；
        ``_last_aux_model_failure_*`` 可能会抛出过期的错误警告；
        ``_last_summary_dropped_count`` / ``_last_summary_fallback_used``
        则可能会产生误导性的用户警告。

        ``compress()`` 已经在具体使用的地方对 ``_previous_summary`` 的泄漏做了防范；
        而本方法则是一种深度防御机制，在所属会话结束的瞬间，重置整个会话的所有状态面。
        """
        self._previous_summary = None
        self._last_summary_error = None
        self._last_summary_dropped_count = 0
        self._last_summary_fallback_used = False
        self._last_aux_model_failure_error = None
        self._last_aux_model_failure_model = None
        self._last_compression_savings_pct = 100.0
        self._ineffective_compression_count = 0
        self._verify_compaction_cleared_threshold = False
        self._last_compression_made_progress = False
        self._summary_failure_cooldown_until = 0.0
        self._last_compress_aborted = False
        self._context_probed = False
        self._context_probe_persistable = False
        self.last_real_prompt_tokens = 0
        self.last_compression_rough_tokens = 0
        self.last_rough_tokens_when_real_prompt_fit = 0
        self.awaiting_real_usage_after_compression = False

    def bind_session_state(self, session_db: Any = None, session_id: str = "") -> None:
        """Bind the current session row so durable cooldowns can round-trip."""
        self._session_db = session_db
        self._session_id = session_id or ""
        self._summary_failure_cooldown_until = 0.0
        self._last_summary_error = None
        self.get_active_compression_failure_cooldown()

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Bind session-scoped compression state for a new or resumed session."""
        super().on_session_start(session_id, **kwargs)
        self.bind_session_state(kwargs.get("session_db", getattr(self, "_session_db", None)), session_id)

    def get_active_compression_failure_cooldown(self) -> Optional[Dict[str, Any]]:
        """Return the live compression-failure cooldown for the bound session."""
        now_mono = time.monotonic()
        if self._summary_failure_cooldown_until > now_mono:
            return {
                "cooldown_until": time.time() + (
                    self._summary_failure_cooldown_until - now_mono
                ),
                "remaining_seconds": self._summary_failure_cooldown_until - now_mono,
                "error": self._last_summary_error,
            }

        session_db = getattr(self, "_session_db", None)
        session_id = getattr(self, "_session_id", "")
        if not session_db or not session_id:
            return None

        getter = getattr(session_db, "get_compression_failure_cooldown", None)
        if getter is None:
            return None
        try:
            state = getter(session_id)
        except sqlite3.Error as exc:
            logger.debug("compression failure cooldown lookup failed: %s", exc)
            return None
        except Exception:
            return None
        if not state:
            return None

        remaining_seconds = float(state.get("remaining_seconds") or 0.0)
        if remaining_seconds <= 0:
            return None

        self._summary_failure_cooldown_until = now_mono + remaining_seconds
        self._last_summary_error = state.get("error")
        return {
            "cooldown_until": float(state.get("cooldown_until") or 0.0),
            "remaining_seconds": remaining_seconds,
            "error": self._last_summary_error,
        }

    def _record_compression_failure_cooldown(
        self,
        cooldown_seconds: float,
        error: Optional[str],
    ) -> None:
        cooldown_until = time.time() + cooldown_seconds
        self._summary_failure_cooldown_until = time.monotonic() + cooldown_seconds
        self._last_summary_error = error

        session_db = getattr(self, "_session_db", None)
        session_id = getattr(self, "_session_id", "")
        if not session_db or not session_id:
            return

        recorder = getattr(session_db, "record_compression_failure_cooldown", None)
        if recorder is None:
            return
        try:
            recorder(session_id, cooldown_until, error)
        except sqlite3.Error as exc:
            logger.debug("compression failure cooldown persist failed: %s", exc)
        except Exception as exc:
            logger.debug("compression failure cooldown persist failed (non-sqlite): %s", exc)

    def _clear_compression_failure_cooldown(self) -> None:
        self._summary_failure_cooldown_until = 0.0
        self._last_summary_error = None

        session_db = getattr(self, "_session_db", None)
        session_id = getattr(self, "_session_id", "")
        if not session_db or not session_id:
            return

        clearer = getattr(session_db, "clear_compression_failure_cooldown", None)
        if clearer is None:
            return
        try:
            clearer(session_id)
        except sqlite3.Error as exc:
            logger.debug("compression failure cooldown clear failed: %s", exc)
        except Exception as exc:
            logger.debug("compression failure cooldown clear failed (non-sqlite): %s", exc)

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: Any = "",
        provider: str = "",
        api_mode: str = "",
        max_tokens: int | None = None,
    ) -> None:
        """Update model info after a model switch or fallback activation."""
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode
        self.context_length = context_length
        # Re-apply the small-context threshold floor for the NEW window,
        # starting from the originally-configured percent (not the possibly
        # floored live value) so a small -> large switch drops back to the
        # configured threshold and a large -> small switch gains the floor.
        # Guard with getattr: compressors unpickled/constructed before this
        # attribute existed fall back to the live value.
        _configured_pct = getattr(
            self, "_configured_threshold_percent", self.threshold_percent,
        )
        self.threshold_percent = self._effective_threshold_percent(
            context_length, _configured_pct,
        )
        # max_tokens=None here means "caller didn't specify" → keep the existing
        # output reservation. A switch that genuinely changes the output budget
        # passes the new value explicitly. (#43547)
        if max_tokens is not None:
            self.max_tokens = self._coerce_max_tokens(max_tokens)
        self.threshold_tokens = self._compute_threshold_tokens(
            context_length, self.threshold_percent, self.max_tokens,
        )
        # Recalculate token budgets for the new context length so the
        # compressor stays calibrated after a model switch (e.g. 200K → 32K).
        target_tokens = int(self.threshold_tokens * self.summary_target_ratio)
        self.tail_token_budget = target_tokens
        self.max_summary_tokens = min(
            int(context_length * 0.05), _SUMMARY_TOKENS_CEILING,
        )

        # Reset cross-call calibration state captured under the PREVIOUS model.
        # These fields encode "the provider proved this prompt fit" / "preflight
        # can be deferred" decisions that are only valid for the model that
        # produced them. Carrying them across a switch to a smaller-context
        # model would let should_defer_preflight_to_real_usage() suppress a
        # preflight compression the new model actually needs — the exact
        # oversized-send-after-switch failure in #23767. The new model's first
        # response repopulates them via update_from_response(). Setting
        # last_prompt_tokens to 0 (NOT -1) is deliberate: 0 is the documented
        # "no real usage yet -> use the rough estimate" state, so the post-
        # response should_compress path falls back to estimate_request_tokens_rough
        # rather than skipping compression. -1 is a different sentinel
        # (#36718, "compression just ran, await real usage") and must not be set here.
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.last_real_prompt_tokens = 0
        self.last_rough_tokens_when_real_prompt_fit = 0
        self.last_compression_rough_tokens = 0
        self.awaiting_real_usage_after_compression = False
        self._ineffective_compression_count = 0
        self._verify_compaction_cleared_threshold = False
        self._last_compression_made_progress = False

    # When the MINIMUM_CONTEXT_LENGTH floor meets/exceeds a small context
    # window, compacting at the percentage (50% → 32K of a 64K window) wastes
    # half the usable context. Trigger near the top of the window instead so a
    # minimum-context model uses most of its budget before compacting — same
    # rationale as the gpt-5.5/Codex 85% autoraise.
    _MIN_CTX_TRIGGER_RATIO = 0.85

    @staticmethod
    def _coerce_max_tokens(value: Any) -> int | None:
        """Normalize a max_tokens value to a positive int or None.

        Only a positive integer is a real output reservation. None (provider
        default), non-numeric values, or <= 0 all mean "no reservation" — this
        keeps the threshold arithmetic safe from non-int inputs (e.g. a test
        MagicMock reaching ContextCompressor via a mocked parent agent).
        """
        if value is None:
            return None
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            return None
        return ivalue if ivalue > 0 else None

    @staticmethod
    def _effective_threshold_percent(
        context_length: int, threshold_percent: float,
    ) -> float:
        """Apply the small-context threshold floor (raise-only).

        Models under ``_SMALL_CTX_WINDOW_LIMIT`` (512K) trigger at no less
        than ``_SMALL_CTX_THRESHOLD_PERCENT`` (75%) of the window.  An
        explicitly higher threshold (user config or per-model autoraise,
        e.g. Codex gpt-5.5's 85%) always wins; only lower values are raised.
        Large-context models keep the configured value — at 512K+ the default
        50% trigger already leaves ample post-compaction headroom.
        """
        if context_length and context_length < _SMALL_CTX_WINDOW_LIMIT:
            return max(threshold_percent, _SMALL_CTX_THRESHOLD_PERCENT)
        return threshold_percent

    @staticmethod
    def _compute_threshold_tokens(
        context_length: int, threshold_percent: float, max_tokens: int | None = None,
    ) -> int:
        """Compute the compaction trigger threshold in tokens.

        The base value is ``effective_input_budget * threshold_percent``, floored
        at ``MINIMUM_CONTEXT_LENGTH`` so large-context models don't compress
        prematurely at 50%. BUT that floor degenerates at small windows: for a
        model whose ``context_length`` is at/below the minimum (e.g. a 64K
        local model), ``max(0.5*64000, 64000) == 64000`` makes the threshold
        equal the ENTIRE window — auto-compression can never fire because the
        provider rejects the request before usage reaches 100% (#14690).

        When the floor would meet or exceed the context window, trigger at
        ``_MIN_CTX_TRIGGER_RATIO`` (85%) of the window — high enough that a
        small model uses most of its context before compacting, but below
        100% so compaction fires before the provider rejects the request.

        The provider reserves ``max_tokens`` of output space out of the same
        window, so the usable INPUT budget is ``context_length - max_tokens``.
        With a large ``max_tokens`` (e.g. 65536 on a custom provider) the input
        budget is materially smaller than the raw window, and a threshold based
        on the full window lets the session hit a provider 400 before compaction
        fires (#43547). The percentage and the degenerate-window check below both
        operate on the effective input budget. ``max_tokens=None`` (provider
        default) conservatively assumes no reservation (full window).
        """
        effective_window = context_length - (max_tokens or 0)
        if effective_window <= 0:
            effective_window = context_length
        pct_value = int(effective_window * threshold_percent)
        floored = max(pct_value, MINIMUM_CONTEXT_LENGTH)
        # If flooring pushed the threshold to/over the effective window it can
        # never be reached. Trigger at 85% of the effective input budget so a
        # minimum-context model rides most of its budget before compacting
        # instead of wasting half.
        if effective_window > 0 and floored >= effective_window:
            return max(1, min(int(effective_window * ContextCompressor._MIN_CTX_TRIGGER_RATIO),
                              effective_window - 1))
        return floored

    def __init__(
        self,
        model: str,
        threshold_percent: float = 0.50,
        protect_first_n: int = 3,
        protect_last_n: int = 20,
        summary_target_ratio: float = 0.20,
        quiet_mode: bool = False,
        summary_model_override: str = None,
        base_url: str = "",
        api_key: str = "",
        config_context_length: int | None = None,
        provider: str = "",
        api_mode: str = "",
        abort_on_summary_failure: bool = False,
        max_tokens: int | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode
        self.threshold_percent = threshold_percent
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.summary_target_ratio = max(0.10, min(summary_target_ratio, 0.80))
        self.quiet_mode = quiet_mode
        # Output-token reservation: the provider carves max_tokens out of the
        # context window, so the usable input budget is context_length -
        # max_tokens. None = provider default => assume no reservation. (#43547)
        # Coerce defensively: only a positive int is a real reservation; any
        # other value (None, non-numeric, <=0) means "no reservation" so the
        # threshold arithmetic never sees a non-int (e.g. a test MagicMock).
        self.max_tokens = self._coerce_max_tokens(max_tokens)
        # When True, summary-generation failure aborts compression entirely
        # (returns messages unchanged, sets _last_compress_aborted=True).
        # When False (default = historical behavior), insert a
        # deterministic "summary unavailable" handoff and drop the middle window.
        self.abort_on_summary_failure = abort_on_summary_failure

        self.context_length = get_model_context_length(
            model, base_url=base_url, api_key=api_key,
            config_context_length=config_context_length,
            provider=provider,
        )
        # Small-context threshold floor: models under 512K trigger at >=75%
        # so compaction doesn't fire with half the window still free (the
        # incompressible floor makes 50%-triggered compaction thrash on
        # 128K-262K models). Raise-only; must run AFTER context_length is
        # resolved and BEFORE threshold_tokens is derived. The pre-floor
        # value is kept so update_model() can re-derive for a new window
        # (switching small -> large must drop back to the configured value).
        self._configured_threshold_percent = self.threshold_percent
        self.threshold_percent = self._effective_threshold_percent(
            self.context_length, self.threshold_percent,
        )
        threshold_percent = self.threshold_percent
        # Floor: never compress below MINIMUM_CONTEXT_LENGTH tokens even if
        # the percentage would suggest a lower value.  This prevents premature
        # compression on large-context models at 50% while keeping the % sane
        # for models right at the minimum. _compute_threshold_tokens also
        # guards the degenerate case where the floor would equal/exceed the
        # window (small models), so auto-compression can still fire (#14690).
        self.threshold_tokens = self._compute_threshold_tokens(
            self.context_length, threshold_percent, self.max_tokens,
        )
        self.compression_count = 0

        # Derive token budgets: ratio is relative to the threshold, not total context
        target_tokens = int(self.threshold_tokens * self.summary_target_ratio)
        self.tail_token_budget = target_tokens
        self.max_summary_tokens = min(
            int(self.context_length * 0.05), _SUMMARY_TOKENS_CEILING,
        )

        if not quiet_mode:
            logger.info(
                "Context compressor initialized: model=%s context_length=%d "
                "threshold=%d (%.0f%%) target_ratio=%.0f%% tail_budget=%d "
                "provider=%s base_url=%s",
                model, self.context_length, self.threshold_tokens,
                threshold_percent * 100, self.summary_target_ratio * 100,
                self.tail_token_budget,
                provider or "none", base_url or "none",
            )
        self._context_probed = False  # True after a step-down from context error

        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_real_prompt_tokens = 0
        self.last_compression_rough_tokens = 0
        self.last_rough_tokens_when_real_prompt_fit = 0
        self.awaiting_real_usage_after_compression = False

        self.summary_model = summary_model_override or ""
        self._session_db: Any = None
        self._session_id: str = ""

        # Stores the previous compaction summary for iterative updates
        self._previous_summary: Optional[str] = None
        # Anti-thrashing: track whether last compression was effective
        self._last_compression_savings_pct: float = 100.0
        self._ineffective_compression_count: int = 0
        # Set after a completed compression boundary; consumed by the next
        # provider-reported prompt count in update_from_response().
        self._verify_compaction_cleared_threshold: bool = False
        # Lets the boundary wrapper distinguish a completed rewrite from a
        # no-op/abort without inferring progress from message-list length.
        self._last_compression_made_progress: bool = False
        self._summary_failure_cooldown_until: float = 0.0
        self._last_summary_error: Optional[str] = None
        # When summary generation fails and a static fallback is inserted,
        # record how many turns were unrecoverably dropped so callers
        # (gateway hygiene, /compress) can surface a visible warning.
        self._last_summary_dropped_count: int = 0
        self._last_summary_fallback_used: bool = False
        # When summary generation fails we now ABORT compression entirely
        # and return the original messages unchanged instead of dropping
        # the middle window with a static placeholder.  Callers inspect
        # this flag to know "compression was attempted but aborted, freeze
        # the chat until the user manually retries via /compress".
        self._last_compress_aborted: bool = False
        # Set True when the summary call failed with an authentication /
        # permission error (HTTP 401/403). Auth failures are non-recoverable
        # at the request level — the credential or endpoint is broken — so
        # compress() must ABORT (preserve the session unchanged) rather than
        # rotate into a degraded child session with a placeholder summary.
        # This is independent of the abort_on_summary_failure config flag:
        # rotating on a broken credential is never the right behavior.
        self._last_summary_auth_failure: bool = False
        # Set when summary generation ultimately fails due to a transient
        # network/connection error (httpx/httpcore connection drop, premature
        # stream close, etc.) — distinct from auth failures but treated the
        # same way by compress(): ABORT and preserve the session unchanged
        # rather than destroy the middle window for a deterministic
        # "summary unavailable" marker. Retrying once the network recovers is
        # strictly better than discarding context for a transient blip
        # (#29559, #25585). Independent of abort_on_summary_failure.
        self._last_summary_network_failure: bool = False
        # retrying on the main model, record the failure so gateway /
        # CLI callers can still warn the user even though compression
        # succeeded.  Silent recovery would hide the broken config.
        self._last_aux_model_failure_error: Optional[str] = None
        self._last_aux_model_failure_model: Optional[str] = None

    def update_from_response(self, usage: Dict[str, Any]):
        """Update tracked token usage from API response."""
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", self.last_prompt_tokens + self.last_completion_tokens)
        if self.last_prompt_tokens > 0:
            self.last_real_prompt_tokens = self.last_prompt_tokens
            if self.last_prompt_tokens < self.threshold_tokens:
                if self.awaiting_real_usage_after_compression and self.last_compression_rough_tokens > 0:
                    self.last_rough_tokens_when_real_prompt_fit = self.last_compression_rough_tokens
                # Any real provider reading below the trigger proves the prompt
                # fits again. Clear the episode latch even when this response was
                # not the one immediately following compaction.
                self._ineffective_compression_count = 0
            else:
                self.last_rough_tokens_when_real_prompt_fit = 0

            # Anti-thrashing verdict, judged HERE because this is the only place
            # that sees the provider's real prompt count for the just-compacted
            # conversation. Effectiveness is "did the prompt get under the
            # threshold?", not "did the message list shrink?": compaction can
            # only shrink messages, while the system prompt and tool schemas are
            # an incompressible floor (with 50+ tools, 20-30K tokens — see
            # #14695). When that floor alone meets the threshold, every pass
            # shrinks messages by a healthy margin yet leaves the prompt over the
            # line, so the next turn compacts again, forever.
            #
            # It must NOT live in should_compress(): that runs twice per turn
            # with two different measures (a rough preflight estimate and the
            # real post-response count, #36718), and the rough one can dip below
            # the threshold and reset the strike every turn, re-opening the loop.
            # Keying on real usage compares like with like and fires exactly once
            # per compaction.
            if self._verify_compaction_cleared_threshold:
                if self.last_prompt_tokens >= self.threshold_tokens:
                    self._ineffective_compression_count += 1
                    if not self.quiet_mode:
                        logger.warning(
                            "Compaction did not clear the threshold: %d real "
                            "tokens still >= %d. The incompressible prompt "
                            "(system prompt + tool schemas) may already exceed "
                            "it, in which case shrinking messages cannot help. "
                            "ineffective_compression_count=%d",
                            self.last_prompt_tokens, self.threshold_tokens,
                            self._ineffective_compression_count,
                        )
                else:
                    self._ineffective_compression_count = 0
        # Consume the pending-verification flag once real usage arrives, whether
        # or not prompt_tokens was reported, so a usage-less response can't leave
        # it armed for a later, unrelated reading.
        self._verify_compaction_cleared_threshold = False
        self.awaiting_real_usage_after_compression = False

    def should_defer_preflight_to_real_usage(self, rough_tokens: int) -> bool:
        """Return True when a high rough preflight estimate is known-noisy.

        ``estimate_request_tokens_rough(..., tools=...)`` intentionally
        overestimates schema-heavy requests so Hermes compresses before a
        provider rejects the payload. After a successful compressed API call,
        though, provider ``prompt_tokens`` are a better signal than repeating
        compaction from the same rough schema overhead. Defer only while the
        rough estimate has grown modestly since a request the provider proved
        fit under the threshold.
        """
        if rough_tokens < self.threshold_tokens:
            return False
        # Immediately after a compaction the post-compression path sets
        # ``awaiting_real_usage_after_compression`` and parks
        # ``last_prompt_tokens = -1``, but ``last_real_prompt_tokens`` still
        # holds the STALE pre-compression value (above threshold — that's why
        # compaction fired).  Without this guard that stale value defeats the
        # ``last_real_prompt_tokens >= threshold_tokens`` check below, so
        # preflight fires a SECOND compaction before the provider has reported
        # real token usage for the now-shorter conversation.  Defer for exactly
        # one turn; update_from_response() clears the flag when real usage
        # arrives.  (#36718)
        if self.awaiting_real_usage_after_compression:
            return True
        if self.last_real_prompt_tokens <= 0:
            return False
        if self.last_real_prompt_tokens >= self.threshold_tokens:
            return False

        baseline = self.last_rough_tokens_when_real_prompt_fit or self.last_compression_rough_tokens
        if baseline <= 0:
            return False

        growth = max(0, rough_tokens - baseline)
        tolerated_growth = max(4096, int(self.threshold_tokens * 0.05))
        if growth > tolerated_growth:
            return False

        self.last_rough_tokens_when_real_prompt_fit = max(baseline, rough_tokens)
        return True

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Check if context exceeds the compression threshold.

        Includes anti-thrashing protection: if the last two compressions
        each saved less than 10%, skip compression to avoid infinite loops
        where each pass removes only 1-2 messages.
        """
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if tokens < self.threshold_tokens:
            return False
        # Do not trigger compression while the summary LLM is in cooldown.
        # On a 429/transient failure _generate_summary() sets a cooldown and
        # returns None; compress() then inserts a static fallback marker and
        # returns. Tokens stay above threshold, so without this guard every
        # subsequent turn re-fires _compress_context() — re-inserting the
        # marker and re-entering the loop, making the CLI appear frozen until
        # the cooldown expires (issue #11529). Manual /compress passes
        # force=True, which clears this cooldown in compress() before running,
        # so it still retries immediately.
        _cooldown_remaining = self._summary_failure_cooldown_until - time.monotonic()
        if _cooldown_remaining > 0:
            if not self.quiet_mode:
                logger.debug(
                    "Compression deferred — summary LLM in cooldown for %.0fs more",
                    _cooldown_remaining,
                )
            return False
        # Anti-thrashing: back off if recent compressions were ineffective
        if self._ineffective_compression_count >= 2:
            if not self.quiet_mode:
                logger.warning(
                    "Compression skipped — last %d compaction attempts did not "
                    "restore enough context headroom. Consider /new to start a "
                    "fresh session, or /compress <topic> for focused compression.",
                    self._ineffective_compression_count,
                )
            return False
        return True

    # ------------------------------------------------------------------
    # Tool output pruning (cheap pre-pass, no LLM call)
    # ------------------------------------------------------------------

    def _prune_old_tool_results(
        self, messages: List[Dict[str, Any]], protect_tail_count: int,
        protect_tail_tokens: int | None = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        """将旧的工具结果内容替换为包含有效信息的单行总结。

        它不会使用通用的占位符，而是生成类似如下的总结：:

            [terminal] 运行了 `npm test` -> exit 0，47 行输出
            [read_file] 从第 1 行读取了 config.py（3,400 字符）

        此外，还会对完全相同的工具结果进行去重（例如：读取同一个文件 5 次，将仅保留
        最新的一份完整副本），并截断受保护尾部（protected tail）之外的 assistant 消息中
        庞大的 tool_call 参数。

        从末尾向后（反向）遍历，保护落入 ``protect_tail_tokens``（若提供）范围内的最新消息，
        或者保护最后的 ``protect_tail_count`` 条消息（向后兼容的默认设置）。
        当两者都提供时，Token 预算具有更高优先级，而消息数量则作为硬性的最低下限。

        返回：
            (pruned_messages, pruned_count) 元组。
        """
        if not messages:
            return messages, 0

        result = [m.copy() for m in messages]
        pruned = 0

        # Build index: tool_call_id -> (tool_name, arguments_json)
        call_id_to_tool: Dict[str, tuple] = {}
        for msg in result:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        cid = tc.get("id", "")
                        fn = tc.get("function", {})
                        call_id_to_tool[cid] = (fn.get("name", "unknown"), fn.get("arguments", ""))
                    else:
                        cid = getattr(tc, "id", "") or ""
                        fn = getattr(tc, "function", None)
                        name = getattr(fn, "name", "unknown") if fn else "unknown"
                        args_str = getattr(fn, "arguments", "") if fn else ""
                        call_id_to_tool[cid] = (name, args_str)

        # Determine the prune boundary
        if protect_tail_tokens is not None and protect_tail_tokens > 0:
            # Token-budget approach: walk backward accumulating tokens
            accumulated = 0
            boundary = len(result)
            min_protect = min(protect_tail_count, len(result))
            for i in range(len(result) - 1, -1, -1):
                msg = result[i]
                msg_tokens = _estimate_msg_budget_tokens(msg)
                if accumulated + msg_tokens > protect_tail_tokens and (len(result) - i) >= min_protect:
                    boundary = i
                    break
                accumulated += msg_tokens
                boundary = i
            # 将预算遍历（budget walk）转换为“受保护数量”，并在数量空间（count-space）中
            # 应用下限（在这里使用 `max` 读起来很自然：至少保护 `min_protect` 条消息，
            # 或者预算所保留的消息，两者之中取较大值），然后再转换回修剪边界（prune boundary）。
            # 如果在索引空间（index-space）中通过 `max` 来做这件事，会使方向发生反转
            # （索引越小 = 受保护的内容越多），这样一来，一个宽裕的预算反而会
            # 被默默地截断回 `min_protect` 的水平。
            budget_protect_count = len(result) - boundary
            protected_count = max(budget_protect_count, min_protect)
            prune_boundary = len(result) - protected_count
        else:
            prune_boundary = len(result) - protect_tail_count

        # 第一阶段：对完全相同的工具结果进行去重。
        # 当同一个文件被读取多次时，仅保留最新的一份完整副本，
        # 并将较旧的重复项替换为反向引用（back-reference）。
        content_hashes: dict = {}  # hash -> (index, tool_call_id)
        for i in range(len(result) - 1, -1, -1):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content") or ""
            # Multimodal content — dedupe by the text summary if available.
            if isinstance(content, list):
                continue
            if not isinstance(content, str):
                # Multimodal dict envelopes ({_multimodal: True, content: [...]}) and
                # other non-string tool-result shapes can't be hashed/deduped by text.
                continue
            if len(content) < 200:
                continue
            h = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()[:12]
            if h in content_hashes:
                # This is an older duplicate — replace with back-reference
                result[i] = {**msg, "content": "[Duplicate tool output — same content as a more recent call]"}
                pruned += 1
            else:
                content_hashes[h] = (i, msg.get("tool_call_id", "?"))

        # Pass 2: Replace old tool results with informative summaries
        for i in range(prune_boundary):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            # 多模态内容（base64 截图等）：剥离图片负载 —— 在其原本位置保留一个轻量级的
            # 文本占位符。如果不这样做，一个旧的 `computer_use` 截图（约 1MB base64 数据 +
            # 约 1500 个真实 Token）将在每一次压缩处理中永远留存下来。
            if isinstance(content, list):
                stripped = _strip_image_parts_from_parts(content)
                if stripped is not None:
                    result[i] = {**msg, "content": stripped}
                    pruned += 1
                continue
            if isinstance(content, dict) and content.get("_multimodal"):
                summary = content.get("text_summary") or "[screenshot removed to save context]"
                result[i] = {**msg, "content": f"[screenshot removed] {summary[:200]}"}
                pruned += 1
                continue
            if not isinstance(content, str):
                continue
            if not content or content == _PRUNED_TOOL_PLACEHOLDER:
                continue
            # Skip already-deduplicated or previously-summarized results
            if content.startswith("[Duplicate tool output"):
                continue
            # Only prune if the content is substantial (>200 chars)
            if len(content) > 200:
                call_id = msg.get("tool_call_id", "")
                tool_name, tool_args = call_id_to_tool.get(call_id, ("unknown", ""))
                summary = _summarize_tool_result(tool_name, tool_args, content)
                result[i] = {**msg, "content": summary}
                pruned += 1

        # 第三阶段：截断受保护尾部（protected tail）之外的 assistant 消息中庞大的 tool_call 参数。
        # 例如，如果不进行此处理，包含 50KB 内容的 write_file 调用将完全免于被修剪。
        #
        # 这种缩减（shrinking）是在解析后的 JSON 结构内部进行的，从而确保
        # 结果依然是有效的 JSON —— 否则下游服务商会在随后的每一个轮次中返回 400 错误，
        # 直到这个损坏的调用移出上下文窗口。参见 ``_truncate_tool_call_args_json`` 的 docstring。
        for i in range(prune_boundary):
            msg = result[i]
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            new_tcs = []
            modified = False
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict):
                    args = tc.get("function", {}).get("arguments", "")
                    if len(args) > 500:
                        new_args = _truncate_tool_call_args_json(args)
                        if new_args != args:
                            tc = {**tc, "function": {**tc["function"], "arguments": new_args}}
                            modified = True
                new_tcs.append(tc)
            if modified:
                result[i] = {**msg, "tool_calls": new_tcs}

        return result, pruned

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def _compute_summary_budget(self, turns_to_summarize: List[Dict[str, Any]]) -> int:
        """Scale summary token budget with the amount of content being compressed.

        The maximum scales with the model's context window (5% of context,
        capped at ``_SUMMARY_TOKENS_CEILING``) so large-context models get
        richer summaries instead of being hard-capped at 8K tokens.
        """
        content_tokens = estimate_messages_tokens_rough(turns_to_summarize)
        budget = int(content_tokens * _SUMMARY_RATIO)
        return max(_MIN_SUMMARY_TOKENS, min(budget, self.max_summary_tokens))

    # Truncation limits for the summarizer input.  These bound how much of
    # each message the summary model sees — the budget is the *summary*
    # model's context window, not the main model's.
    _CONTENT_MAX = 6000       # total chars per message body
    _CONTENT_HEAD = 4000      # chars kept from the start
    _CONTENT_TAIL = 1500      # chars kept from the end
    _TOOL_ARGS_MAX = 1500     # tool call argument chars
    _TOOL_ARGS_HEAD = 1200    # kept from the start of tool args

    def _serialize_for_summary(self, turns: List[Dict[str, Any]]) -> str:
        """将对话轮次序列化为带有标签的文本，以供总结器使用。

        包含工具调用参数和结果内容（每条消息最多 ``_CONTENT_MAX`` 个字符），
        以便总结器能够保留文件路径、命令和输出等特定细节。

        所有内容在序列化之前都会进行脱敏（redacted），以防止敏感凭据（API 密钥、
        Token、密码）泄露到发送给辅助模型并在多次压缩（compactions）中持久化的总结中。
        """
        # 延迟导入（与 title_generator.py 保持一致） —— agent_runtime_helpers
        # 会引入沉重的传递性导入（transitive imports），我们不希望在模块加载时引入它们。
        from agent.agent_runtime_helpers import strip_think_blocks

        parts = []
        for msg in turns:
            role = msg.get("role", "unknown")
            content = redact_sensitive_text(msg.get("content") or "")
            content = _MEDIA_DIRECTIVE_RE.sub("[media attachment]", content)
            # 在 assistant 内容进入总结器之前，从中剥离内联推理块（`<think>`、`<reasoning>` 等）。
            # 推理轨迹（Reasoning traces）是临时的草稿（scratch work）—— 将它们发送给辅助
            # 模型会浪费总结器的上下文，并且存在将草稿中的结论作为事实保留在总结中的风险。
            # 原生的 ``reasoning`` 消息字段已被排除（只有 ``content`` 会被序列化）；
            # 这封堵了当原生思考被禁用，或者服务商将推理轨迹内联到 content 中时所使用的内联标签路径。
            if role == "assistant" and content:
                content = strip_think_blocks(None, content)

            # Tool results: keep enough content for the summarizer
            if role == "tool":
                tool_id = msg.get("tool_call_id", "")
                if len(content) > self._CONTENT_MAX:
                    content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
                parts.append(f"[TOOL RESULT {tool_id}]: {content}")
                continue

            # Assistant messages: include tool call names AND arguments
            if role == "assistant":
                if len(content) > self._CONTENT_MAX:
                    content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tc_parts = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            args = redact_sensitive_text(fn.get("arguments", ""))
                            # Truncate long arguments but keep enough for context
                            if len(args) > self._TOOL_ARGS_MAX:
                                args = args[:self._TOOL_ARGS_HEAD] + "..."
                            tc_parts.append(f"  {name}({args})")
                        else:
                            fn = getattr(tc, "function", None)
                            name = getattr(fn, "name", "?") if fn else "?"
                            tc_parts.append(f"  {name}(...)")
                    content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
                parts.append(f"[ASSISTANT]: {content}")
                continue

            # User and other roles
            if len(content) > self._CONTENT_MAX:
                content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
            parts.append(f"[{role.upper()}]: {content}")

        return "\n\n".join(parts)

    def _build_static_fallback_summary(
        self,
        turns_to_summarize: List[Dict[str, Any]],
        reason: str | None = None,
    ) -> str:
        """Build a deterministic handoff when the LLM summarizer is unavailable.

        This is intentionally much less rich than an LLM-written summary, but it
        is still better than a bare "N messages were removed" marker.  It keeps
        the most useful continuity anchors that can be extracted locally:
        recent user asks, assistant/tool actions, files/commands mentioned in
        tool calls, and any error text.  The result uses the normal summary
        structure so downstream prompts can recover gracefully after a provider
        outage or summary-model failure.
        """
        user_asks: list[str] = []
        assistant_actions: list[str] = []
        tool_actions: list[str] = []
        relevant_files: list[str] = []
        blockers: list[str] = []
        last_dropped_turns: list[str] = []

        def _compact_fallback_turn(value: Any) -> str:
            text = redact_sensitive_text(_content_text_for_contains(value))
            text = re.sub(r"\bgh[pousr]_[A-Za-z0-9_]{8,}\b", "[REDACTED]", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > _FALLBACK_TURN_MAX_CHARS:
                text = text[: _FALLBACK_TURN_MAX_CHARS - 15].rstrip() + " ...[truncated]"
            return re.sub(r"\bgh[pousr]_[A-Za-z0-9_.-]+", "[REDACTED]", text)

        def _remember_dropped_turn(label: str, text: str, *, limit: int = 8) -> None:
            text = text.strip()
            if not text:
                return
            last_dropped_turns.append(f"{label}: {text}")
            if len(last_dropped_turns) > limit:
                del last_dropped_turns[0]

        def _collect_paths_from_jsonish(obj: Any) -> None:
            if isinstance(obj, dict):
                for key, val in obj.items():
                    if key in {"path", "workdir", "file_path", "output_path"} and isinstance(val, str):
                        _dedupe_append(relevant_files, val, limit=12)
                    _collect_paths_from_jsonish(val)
            elif isinstance(obj, list):
                for val in obj:
                    _collect_paths_from_jsonish(val)
            elif isinstance(obj, str):
                _collect_path_mentions(obj, relevant_files)

        call_id_to_tool: dict[str, tuple[str, str]] = {}
        for msg in turns_to_summarize:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls") or []:
                    name, raw_args = _extract_tool_call_name_and_args(tc)
                    args = redact_sensitive_text(raw_args)
                    call_id = _extract_tool_call_id(tc)
                    if call_id:
                        call_id_to_tool[call_id] = (name, args)
                    if args:
                        try:
                            parsed = json.loads(args)
                        except Exception:
                            parsed = args
                        _collect_paths_from_jsonish(parsed)

        for msg in turns_to_summarize:
            role = msg.get("role", "unknown")
            text = _compact_fallback_turn(msg.get("content"))
            _collect_path_mentions(text, relevant_files)

            turn_text = text
            turn_tool_names: list[str] = []
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls") or []:
                    name, _args = _extract_tool_call_name_and_args(tc)
                    turn_tool_names.append(name)
                if turn_tool_names:
                    prefix = "tool calls: " + ", ".join(turn_tool_names[:6])
                    turn_text = f"{prefix}; {turn_text}" if turn_text else prefix
            _remember_dropped_turn(str(role).upper(), turn_text)

            if len(text) > 600:
                text = text[:420].rstrip() + " ... " + text[-160:].lstrip()

            if role == "user" and text:
                user_asks.append(text)
            elif role == "assistant":
                tool_names: list[str] = []
                for tc in msg.get("tool_calls") or []:
                    name, _args = _extract_tool_call_name_and_args(tc)
                    tool_names.append(name)
                if tool_names:
                    assistant_actions.append(
                        "Called tool(s): " + ", ".join(tool_names[:6])
                    )
                elif text:
                    assistant_actions.append(text)
            elif role == "tool":
                call_id = str(msg.get("tool_call_id") or "")
                tool_name, tool_args = call_id_to_tool.get(call_id, ("unknown", ""))
                tool_actions.append(
                    _summarize_tool_result(tool_name, tool_args, text or "")
                )
                if re.search(
                    r"\b(error|failed|exception|traceback|timeout|timed out|fatal)\b",
                    text,
                    re.I,
                ):
                    blockers.append(text[:500])

        def _bullets(items: list[str], limit: int = 8) -> str:
            unique: list[str] = []
            seen: set[str] = set()
            for item in items:
                item = item.strip()
                if not item or item in seen:
                    continue
                seen.add(item)
                unique.append(item)
                if len(unique) >= limit:
                    break
            return "\n".join(f"- {item}" for item in unique) if unique else "None."

        completed: list[str] = []
        for idx, item in enumerate((assistant_actions + tool_actions)[:12], start=1):
            completed.append(f"{idx}. {item}")

        active_task = (
            f"User asked: {user_asks[-1]!r}"
            if user_asks
            else "Unknown from deterministic fallback."
        )
        previous_summary_note = ""
        if self._previous_summary:
            previous_summary_note = (
                "\n\nPrevious compaction summary was present and should still be treated as "
                "background continuity context, but the latest LLM summary update failed."
            )

        reason_text = f" Summary failure reason: {reason}." if reason else ""
        body = f"""{HISTORICAL_TASK_HEADING}
{active_task}

## Goal
Recovered from a deterministic fallback because the LLM context summarizer was unavailable. Continue from the protected recent messages after this summary and use current file/system state for exact details.{previous_summary_note}

## Constraints & Preferences
- This fallback was generated locally without an LLM summary call.
- Secrets and credentials were redacted before preservation.
- The summary may be incomplete; prefer verifying current files, git state, processes, and test results instead of assuming omitted details.

## Completed Actions
{chr(10).join(completed) if completed else "None recoverable from compacted turns."}

## Active State
Unknown from deterministic fallback. Inspect current repository/session state if needed.

{HISTORICAL_IN_PROGRESS_HEADING}
Unknown from deterministic fallback — the latest user ask is recorded once under
"{HISTORICAL_TASK_HEADING}" above as historical context only. Do NOT treat it as an
unfulfilled instruction to re-answer; verify current state and continue from the
protected recent messages after this summary.

## Blocked
{_bullets(blockers, limit=5)}

## Key Decisions
None recoverable from deterministic fallback.

## Resolved Questions
None recoverable from deterministic fallback.

{HISTORICAL_PENDING_ASKS_HEADING}
None recoverable from deterministic fallback. (The latest user ask is preserved once
under "{HISTORICAL_TASK_HEADING}" as historical context — it is NOT necessarily
outstanding.)

## Relevant Files
{_bullets(relevant_files, limit=12)}

{HISTORICAL_REMAINING_WORK_HEADING}
Continue from the most recent unfulfilled user ask and protected tail messages. Verify state with tools before making claims.

## Last Dropped Turns
{_bullets(last_dropped_turns, limit=8)}

## Critical Context
Summary generation was unavailable, so this is a best-effort deterministic fallback for {len(turns_to_summarize)} compacted message(s).{reason_text}"""
        summary = self._with_summary_prefix(redact_sensitive_text(body.strip()))
        if len(summary) > _FALLBACK_SUMMARY_MAX_CHARS:
            summary = summary[: _FALLBACK_SUMMARY_MAX_CHARS - 42].rstrip() + "\n...[fallback summary truncated]"
        return summary

    def _fallback_to_main_for_compression(self, e: Exception, reason: str) -> None:
        """Switch from a separate ``summary_model`` back to the main model.

        Centralises the bookkeeping shared by every fallback branch in
        :meth:`_generate_summary` (model-not-found, timeout, JSON decode,
        unknown error): record the aux-model failure for ``/usage``-style
        callers, clear the summary model so the next call uses the main one,
        and clear the cooldown so the immediate retry can run.

        ``reason`` is a short human-readable phrase ("unavailable",
        "timed out", "returned invalid JSON", "failed") that is interpolated
        into the warning log.
        """
        self._summary_model_fallen_back = True
        logger.warning(
            "Summary model '%s' %s (%s). "
            "Falling back to main model '%s' for compression.",
            self.summary_model, reason, e, self.model,
        )
        _err_text = str(e).strip() or e.__class__.__name__
        if len(_err_text) > 220:
            _err_text = _err_text[:217].rstrip() + "..."
        self._last_aux_model_failure_error = _err_text
        self._last_aux_model_failure_model = self.summary_model
        self.summary_model = ""  # empty = use main model
        self._clear_compression_failure_cooldown()  # no cooldown — retry immediately

    def _generate_summary(
        self,
        turns_to_summarize: List[Dict[str, Any]],
        focus_topic: Optional[str] = None,
    ) -> Optional[str]:
        """生成对话轮次的结构化总结。

        使用结构化模板（目标、进度、决策、已解决/待解决的问题、文件、剩余工作），
        并带有明确的前导语（preamble）以指示总结器不要回答问题。当存在先前的
        总结时，生成迭代更新，而不是从头开始总结。

        参数：
            focus_topic: 用于引导式压缩的可选核心主题字符串。当提供时，
                总结器会优先保留与该主题相关的信息，并会更积极地压缩
                其他所有内容。灵感源自 Claude Code 的 ``/compact``。

        如果所有尝试均失败，则返回 None —— 调用方应直接丢弃中间的轮次（不进行总结），
        而不是注入一个无用的占位符。
        """
        now = time.monotonic()
        if now < self._summary_failure_cooldown_until:
            logger.debug(
                "Skipping context summary during cooldown (%.0fs remaining)",
                self._summary_failure_cooldown_until - now,
            )
            return None

        summary_budget = self._compute_summary_budget(turns_to_summarize)
        content_to_summarize = self._serialize_for_summary(turns_to_summarize)

        # 用于时间锚定（temporal anchoring）的当前日期（参见下文的 ## Temporal Anchoring）。
        # 仅限日期的粒度与 system_prompt.py:337 (PR #20451) 以及通过
        # hermes_time.now() 获取的用户配置的时区相匹配。压缩总结（compaction summary）
        # 是一个对话中途的消息，它【不】属于缓存前缀（cached prefix）的一部分，因此这里的
        # 日期绝不会影响提示词缓存（prompt-cache）的稳定性。采用防御性处理 ——
        # 时钟故障绝不能阻塞压缩。
        try:
            from hermes_time import now as _hermes_now

            _today_str = _hermes_now().strftime("%Y-%m-%d")
        except Exception:  # pragma: no cover - clock resolution is best-effort
            _today_str = ""

        # 首次压缩和迭代更新提示词所共享的前导语。
        # 故意保持措辞平实：兼容 Azure/OpenAI 的内容过滤器曾对更强烈的
        # “注入”/“不要回应”话术构建做出过标记。
        # ---------------------------------
        # 你是一个正在创建上下文检查点的总结智能体。
        # 将下方的对话轮次视为先前工作紧凑记录的源材料。
        # 仅生成结构化总结；请勿添加问候语、前导语或前缀。
        # 用用户在对话中使用的相同语言来撰写总结 —— 请勿翻译或切换为英语。
        # 绝不要在总结中包含 API 密钥、Token、密码、机密信息、凭据或连接字符串
        # —— 将出现的任何此类内容替换为 [REDACTED]。请注意，虽然用户提供了凭据，
        # 但不要保留它们的值。
        _summarizer_preamble = (
            "You are a summarization agent creating a context checkpoint. "
            "Treat the conversation turns below as source material for a "
            "compact record of prior work. "
            "Produce only the structured summary; do not add a greeting, "
            "preamble, or prefix. "
            "Write the summary in the same language the user was using in the "
            "conversation — do not translate or switch to English. "
            "NEVER include API keys, tokens, passwords, secrets, credentials, "
            "or connection strings in the summary — replace any that appear "
            "with [REDACTED]. Note that the user had credentials present, but "
            "do not preserve their values."
        )

        # 时间锚定指令。将相对的、听起来仍悬而未决的描述重写为绝对的、带有日期的、
        # 过去时态的事实，以便恢复后的对话不会重新执行已完成的操作。
        # 仅在当前日期成功解析时才会发出；否则将忽略该规则，
        # 从而使总结器永远不会接收到空的日期占位符。
        # ------------------------------
        # 时间锚定：当前日期是 {_today_str}。当某个操作已经执行完毕时，
        # 将其表述为一个已完成的、带有日期的、过去时态的事实，而不是一条未决的指令。
        # 例如，将 “就提议给 John 发邮件” 重写为 “已于 {_today_str} 向 John 发送了提议邮件。”
        # 绝不要将已完成的操作表述得像仍然需要去做一样，也绝不要为尚未发生的工作捏造日期。
        if _today_str:
            _temporal_anchoring_rule = (
                f"\nTEMPORAL ANCHORING: The current date is {_today_str}. When an "
                "action has already been carried out, phrase it as a completed, "
                "dated, past-tense fact rather than an open instruction. For "
                'example, rewrite "email John about the proposal" as "Sent the '
                f'proposal email to John on {_today_str}." Never leave a finished '
                "action worded as if it still needs doing, and never invent a date "
                "for work that has not happened yet.\n"
            )
        else:
            _temporal_anchoring_rule = ""

        # [唯一最重要的字段。逐字捕获用户最近未实现的输入 ——
        # 即他们使用的原话。这包括：
        # - 明确的任务指派（例如“重构 auth 模块”）
        # - 等待回答的问题（例如“waarom staat X op Y?”、“wat zijn de volgende stappen?”）
        # - 等待输入的决策（例如“optie A of B?”）
        # - 正在进行的讨论，其中 assistant 需给出下一个实质性的回复
        # 用户刚刚提出问题的对话也是一个处于活跃状态的任务 ——
        # 该任务即“结合完整上下文回答该问题”。
        # 请勿仅因为用户未发出祈使命令就填写“None”；
        # 仅在极少数情况下（即最后的交流已完全解决，且用户说了类似
        # “thanks, that's all”的话时）才保留使用“None”。
        # 如果有多项待办事项，仅列出尚未完成的事项。
        # 后续工作应当紧接此处开始。示例：
        # “User asked: 'Now refactor the auth module to use JWT instead of sessions'”
        # “User asked: 'Waarom stond provider ineens op openrouter?' —— 需要排查 + 解答”
        # “User chose option A; awaiting implementation of step 2”
        # 如果用户最近的消息是一个覆盖了先前工作的逆向信号（停止、撤销、回滚、
        # 算了/没关系、仅验证、更换主题），请逐字写下该逆向信号，并且
        # 【不要】结转已取消的任务。示例：“User asked: 'Stop the i18n refactor
        # and just verify the current diff' —— 先前进行中的 i18n 工作已取消。”
        # 如果不存在未决的任务，写“None”。]
        # ------------
        # Shared structured template (used by both paths).
        _template_sections = f"""{HISTORICAL_TASK_HEADING}
[THE SINGLE MOST IMPORTANT FIELD. Capture the user's most recent unfulfilled
input verbatim — the exact words they used. This includes:
- Explicit task assignments ("refactor the auth module")
- Questions awaiting an answer ("waarom staat X op Y?", "wat zijn de volgende stappen?")
- Decisions awaiting input ("optie A of B?")
- Ongoing discussions where the assistant owes the next substantive reply
A conversation where the user just asked a question IS an active task — the
task is "answer that question with full context". Do NOT write "None" merely
because the user did not issue an imperative command; reserve "None" for the
rare case where the last exchange was fully resolved and the user said
something like "thanks, that's all".
If multiple items are outstanding, list only the ones NOT yet completed.
Continuation should pick up exactly here. Examples:
"User asked: 'Now refactor the auth module to use JWT instead of sessions'"
"User asked: 'Waarom stond provider ineens op openrouter?' — needs investigation + answer"
"User chose option A; awaiting implementation of step 2"
If the user's most recent message was a reverse signal (stop, undo, roll
back, never mind, just verify, change of topic) that supersedes earlier
work, write the reverse signal verbatim and DO NOT carry forward the
cancelled task. Example: "User asked: 'Stop the i18n refactor and just
verify the current diff' — earlier i18n in-flight work is cancelled."
If no outstanding task exists, write "None."]
## Goal（目标）
[用户整体上试图完成的事情]

## Constraints & Preferences（约束与偏好）
[用户的偏好、编码风格、约束条件、重要决策]

## Completed Actions（已完成的操作）
[已采取的具体操作的编号列表 —— 包含使用的工具、目标以及结果。
每一项的格式为：N. 操作 目标 — 结果 [工具：名称]
示例：
1. READ config.py:45 — 发现 `==` 应当为 `!=` [工具: read_file]
2. PATCH config.py:45 — 将 `==` 修改为 `!=` [工具: patch]
3. TEST `pytest tests/` — 3/50 失败：test_parse, test_validate, test_edge [工具: terminal]
请具体写明文件路径、命令、行号和结果。]

## Active State（活动状态）
[当前的工作状态 —— 包含：
- 工作目录和分支（如果适用）
- 已修改/已创建的文件，并对每个文件进行简要说明
- 测试状态（通过 X/Y 个）
- 任何正在运行的进程或服务器
- 关键的环境细节]
## Goal
[What the user is trying to accomplish overall]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions]

## Completed Actions
[Numbered list of concrete actions taken — include tool used, target, and outcome.
Format each as: N. ACTION target — outcome [tool: name]
Example:
1. READ config.py:45 — found `==` should be `!=` [tool: read_file]
2. PATCH config.py:45 — changed `==` to `!=` [tool: patch]
3. TEST `pytest tests/` — 3/50 failed: test_parse, test_validate, test_edge [tool: terminal]
Be specific with file paths, commands, line numbers, and results.]

## Active State
[Current working state — include:
- Working directory and branch (if applicable)
- Modified/created files with brief note on each
- Test status (X/Y passing)
- Any running processes or servers
- Environment details that matter]

[当前正在进行的工作 —— 触发压缩时正在做的事情]

## Blocked（受阻）
[任何尚未解决的阻碍、错误或问题。包含准确的错误信息。]

## Key Decisions（关键决策）
[重要的技术决策以及做出这些决策的原因]

## Resolved Questions（已解决的问题）
[用户提出且【已经】得到解答的问题 —— 包含相应的解答以避免重复回答]
{HISTORICAL_IN_PROGRESS_HEADING}
[Work currently underway — what was being done when compaction fired]

## Blocked
[Any blockers, errors, or issues not yet resolved. Include exact error messages.]

## Key Decisions
[Important technical decisions and WHY they were made]

## Resolved Questions
[Questions the user asked that were ALREADY answered — include the answer so it is not repeated]

# [用户提出但【尚未】解答或实现的提问或请求。这些是过期的（STALE）
# —— 它们来自已被压缩的轮次。在此处写下它们仅供参考。
# 除非最新的用户消息有明确要求，否则智能体【决不能】针对它们采取行动。
# 若无，写“None”。]
# 
# ## Relevant Files（相关文件）
# [已读取、修改或创建的文件 —— 以及对每个文件的简要说明]
{HISTORICAL_PENDING_ASKS_HEADING}
[Questions or requests from the user that have NOT yet been answered or fulfilled. These are STALE — they were from the compacted turns. Write them here for reference only. The agent must NOT act on them unless the latest user message explicitly requests it. If none, write "None."]

## Relevant Files
[Files read, modified, or created — with brief note on each]

# {HISTORICAL_REMAINING_WORK_HEADING}
# [尚需完成的工作 —— 作为过期的（STALE）上下文，仅供参考。除非最新的用户消息
# 有明确要求，否则智能体【决不能】恢复此工作。]
# 
# ## Critical Context（关键上下文）
# [任何若不明确保留就会丢失的特定值、错误信息、配置细节或数据。
# 绝不要包含 API 密钥、Token、密码或凭据 —— 请改写为 [REDACTED]。]
{HISTORICAL_REMAINING_WORK_HEADING}
[What remains to be done — framed as STALE context for reference only. The agent must NOT resume this work unless the latest user message explicitly asks for it.]

## Critical Context
[Any specific values, error messages, configuration details, or data that would be lost without explicit preservation. NEVER include API keys, tokens, passwords, or credentials — write [REDACTED] instead.]

# 目标大约为 {summary_budget} 个 Token。要具体（CONCRETE） —— 包含文件路径、命令输出、错误信息、行号和特定值。避免使用类似“做了一些更改”这样模糊的描述 —— 准确说出更改了什么。
# {_temporal_anchoring_rule}
# 只写总结正文。不要包含任何前导语或前缀。"""

        if self._previous_summary:
            # 迭代更新：保留现有信息，添加新进度
            prompt = f"""{_summarizer_preamble}

你正在更新一份上下文压缩总结。先前的压缩生成了下方的总结。此后发生了新的对话轮次，需要将其合并进来。

先前的总结：
{self._previous_summary}

要合并的新轮次：
{content_to_summarize}

Target ~{summary_budget} tokens. Be CONCRETE — include file paths, command outputs, error messages, line numbers, and specific values. Avoid vague descriptions like "made some changes" — say exactly what changed.
{_temporal_anchoring_rule}
Write only the summary body. Do not include any preamble or prefix."""

        if self._previous_summary:
            # Iterative update: preserve existing info, add new progress
            prompt = f"""{_summarizer_preamble}

You are updating a context compaction summary. A previous compaction produced the summary below. New conversation turns have occurred since then and need to be incorporated.

PREVIOUS SUMMARY:
{self._previous_summary}

NEW TURNS TO INCORPORATE:
{content_to_summarize}

# 使用这种精确的结构更新总结。保留所有仍然相关的现有信息。
# 将新完成的操作添加到编号列表中（继续顺延编号）。
# 完成后，将事项从“进行中（In Progress）”移至“已完成的操作（Completed Actions）”。
# 将已回答的问题移至“已解决的问题（Resolved Questions）”。
# 更新“活动状态（Active State）”以反映当前状态。
# 仅在信息明显过时的情况下才将其移除。
# 至关重要：更新“## 活跃任务（Active Task）”以反映用户最近未实现的输入 ——
# 这包括智能体（assistant）尚未回答的任何提问、决策请求或讨论轮次。
# 仅当最后的交流已完全解决时，才填写“None”。
Update the summary using this exact structure. PRESERVE all existing information that is still relevant. ADD new completed actions to the numbered list (continue numbering). Move items from "In Progress" to "Completed Actions" when done. Move answered questions to "Resolved Questions". Update "Active State" to reflect current state. Remove information only if it is clearly obsolete. CRITICAL: Update "## Active Task" to reflect the user's most recent unfulfilled input — this includes any question, decision request, or discussion turn that the assistant has not yet answered. Only write "None" if the last exchange was fully resolved.

{_template_sections}"""
        else:
            # First compaction: summarize from scratch
            prompt = f"""{_summarizer_preamble}
# 在先前的轮次被压缩后，为对话创建一个结构化的检查点总结。
# 该总结应当保留足够的细节以保持连贯性，而无需重新阅读原始轮次。
Create a structured checkpoint summary for the conversation after earlier turns are compacted. The summary should preserve enough detail for continuity without re-reading the original turns.

TURNS TO SUMMARIZE:
{content_to_summarize}

Use this exact structure:

{_template_sections}"""

        # Inject focus topic guidance when the user provides one via /compress <focus>.
        # This goes at the end of the prompt so it takes precedence.
        if focus_topic:
            prompt += f"""

FOCUS TOPIC: "{focus_topic}"
# 此次压缩应当【优先】保留与上方核心主题相关的所有信息。
# 对于与 "{focus_topic}" 相关的内容，应包含完整的细节 —— 确切的值、文件路径、命令输出、错误信息以及决策。
# 对于与核心主题【无关】的内容，要更积极地进行总结（简短的单行说明，或者如果确实无关则予以忽略）。
# 核心主题部分应当分配大约 60-70% 的总结 Token 预算。
# 即使是针对核心主题，也【绝不要】保留 API 密钥、Token、密码或凭据 —— 请使用 [REDACTED]。
This compaction should PRIORITISE preserving all information related to the focus topic above. For content related to "{focus_topic}", include full detail — exact values, file paths, command outputs, error messages, and decisions. For content NOT related to the focus topic, summarise more aggressively (brief one-liners or omit if truly irrelevant). The focus topic sections should receive roughly 60-70% of the summary token budget. Even for the focus topic, NEVER preserve API keys, tokens, passwords, or credentials — use [REDACTED].
"""

        try:
            call_kwargs = {
                "task": "compression",
                "main_runtime": {
                    "model": self.model,
                    "provider": self.provider,
                    "base_url": self.base_url,
                    "api_key": self.api_key,
                    "api_mode": self.api_mode,
                },
                "messages": [{"role": "user", "content": prompt}],
                # 不设置 max_tokens：输出上限决不能截断总结。
                # ``summary_budget`` 仅为提示词级别的引导（即上方的 "Target ~N tokens"）。
                # 大多数兼容 OpenAI 的接口层（wires）已经省略了该参数（参见 _build_call_kwargs），
                # 但 Anthropic Messages 接口层和 NVIDIA NIM 仍会转发它 ——
                # 在这些接口中设置硬上限会导致总结在中途被切断（推理模型会首先在推理上耗尽该额度），
                # 从而产生被截断的或仅包含推理的总结，并引发压缩死循环。
                # 省略该参数可以让适配器（adapter）回退到模型原生的输出上限。
                # timeout 由 call_llm 从 auxiliary.compression.timeout 配置中解析得出。
            }
            if self.summary_model:
                call_kwargs["model"] = self.summary_model
            # 压缩是原子性的：保护进行中的总结调用免受轮次中途网关中断的影响。
            # 否则，传入的用户消息会中止总结，导致压缩回退到降级的静态标记，
            # 从而丢失真正的交接内容（参见 #23975）。可重入：主模型的重试
            # （`_generate_summary` 递归）可以无害地重入。
            with aux_interrupt_protection():
                response = call_llm(**call_kwargs)
            # `_validate_llm_response` 仅保证 `choices[0].message` 存在，
            # 而不保证它是一个带有 `.content` 属性的对象。某些
            # 兼容 OpenAI 的代理或本地后端会返回字典（dict）或字符串（str）
            # 结构的消息；应进行防御性类型转换，而不是直接崩溃。
            message = response.choices[0].message
            if isinstance(message, dict):
                content = message.get("content")
            else:
                content = getattr(message, "content", message)
            # Handle cases where content is not a string (e.g., dict from llama.cpp)
            if not isinstance(content, str):
                content = str(content) if content else ""
            # 某些兼容 OpenAI 的代理（例如 cmkey.cn、one-api 渠道）
            # 会返回一个格式良好的 HTTP 200 响应，但其 ``content`` 为空或仅包含空白字符，
            # 而不是返回错误或空的 ``choices``。该有效载荷（payload）可以通过
            # ``_validate_llm_response`` 的验证（因为 ``message`` 存在），
            # 因此它会到达这里，如果不加处理，就会被存储为一个只有前缀而没有正文的
            # 总结 —— 从而静默地擦除被压缩的轮次，并导致模型忘记进行中的任务（#11978，#11914）。
            # 将空内容视为失败，以便它像传输错误一样，通过相同的主模型回退 + 冷却机制
            # 进行路由，而不是用一个空总结来替换真实的上下文。
            if not content.strip():
                raise RuntimeError(
                    "Context compression LLM returned empty content "
                    f"(provider={self.provider or 'auto'} "
                    f"model={self.summary_model or self.model})"
                )
            # 剥离总结器模型可能发出的推理块（来自 MiniMax、DeepSeek、QwQ
            # 等推理模型的 `<think>...</think>` 等）。如果不这样做，推理轨迹
            # （trace）就会被存储在 `_previous_summary` 中，被注入到对话中，
            # 并且会被反馈到后续的每一个迭代更新提示词中 —— 从而在多次压缩中
            # 加剧 Token 膨胀。镜像了 title_generator.py 的做法。
            from agent.agent_runtime_helpers import strip_think_blocks
            stripped = strip_think_blocks(None, content).strip()
            if stripped:
                content = stripped
            # Redact the summary output as well — the summarizer LLM may
            # ignore prompt instructions and echo back secrets verbatim.
            summary = redact_sensitive_text(content.strip())
            # Store for iterative updates on next compaction
            self._previous_summary = summary
            self._clear_compression_failure_cooldown()
            self._summary_model_fallen_back = False
            self._last_summary_error = None
            self._last_summary_auth_failure = False
            self._last_summary_network_failure = False
            return self._with_summary_prefix(summary)
        except Exception as e:
            # ``call_llm`` 会在两种截然不同的情况下抛出 ``RuntimeError``：
            #   1. 未配置服务商（"No LLM provider configured ..."）——
            #      这属于永久性的配置错误，应用较长的冷却时间是正确的。
            #   2. 来自已配置服务商的空/无效响应
            #      （``_validate_llm_response`` 发生空 ``choices``/返回 ``None``，或触发我们上文的
            #      空 ``content`` 防御逻辑）—— 这属于临时/代理故障，
            #      应当首先回退到主模型，完全就像下文处理的传输错误一样。
            # 只有情况 (1) 属于“未配置服务商”的长冷却时间；情况 (2) 以及其他所有
            # 异常都会流向通用的回退逻辑，以便在进入任何冷却时间之前先进行主模型重试。（#11978，#11914）
            if isinstance(e, RuntimeError) and "no llm provider configured" in str(e).lower():
                # No provider configured — long cooldown, unlikely to self-resolve
                self._record_compression_failure_cooldown(
                    _SUMMARY_FAILURE_COOLDOWN_SECONDS,
                    "no auxiliary LLM provider configured",
                )
                self._last_summary_error = "no auxiliary LLM provider configured"
                logger.warning("Context compression: no provider available for "
                                "summary. Middle turns will be dropped without summary "
                                "for %d seconds.",
                                _SUMMARY_FAILURE_COOLDOWN_SECONDS)
                return None
            # If the summary model is different from the main model and the
            # error looks permanent (model not found, 503, 404), fall back to
            # using the main model instead of entering cooldown that leaves
            # context growing unbounded.  (#8620 sub-issue 4)
            _status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            _err_str = str(e).lower()
            _is_model_not_found = (
                _status in {404, 503}
                or "model_not_found" in _err_str
                or "does not exist" in _err_str
                or "no available channel" in _err_str
            )
            _is_timeout = (
                _status in {408, 429, 502, 504}
                or "timeout" in _err_str
            )
            # Non-JSON / malformed-body responses from misconfigured providers
            # or proxies (e.g. an HTML 502 page returned with
            # ``Content-Type: application/json``) bubble up as
            # ``json.JSONDecodeError`` from the OpenAI SDK's ``response.json()``,
            # or as a wrapping ``APIResponseValidationError`` whose message
            # carries the substring "expecting value".  Treat these like a
            # transient provider failure: one retry on the main model, then a
            # short cooldown.  Issue #22244.
            _is_json_decode = (
                isinstance(e, json.JSONDecodeError)
                or "expecting value" in _err_str
            )
            # httpcore / httpx streaming premature-close errors surface as
            # ConnectionError subclasses or plain Exception with characteristic
            # substrings ("incomplete chunked read", "peer closed connection",
            # "response ended prematurely", "unexpected eof").  These are
            # transient network events; treat them like a timeout so we fall
            # back to the main model instead of entering a 60-second cooldown.
            # See issue #18458.
            _is_streaming_closed = _is_connection_error(e)
            # Authentication / permission failures (401/403) are NOT transient
            # and NOT fixable by retrying the same request: the credential is
            # invalid/blocked/expired or the endpoint is wrong (e.g. a prod
            # token sent to a staging inference URL). Flag them so compress()
            # aborts and preserves the session instead of rotating into a
            # degraded child with a placeholder summary. We still allow the
            # one-shot fallback to the MAIN model below when the failure came
            # from a distinct auxiliary summary_model (its dedicated creds may
            # be the only broken thing); only a failure on the main model — or
            # a fallback that also auth-fails — makes the abort stick.
            _is_auth_error = (
                _status in {401, 403}
                or "invalid api key" in _err_str
                or "invalid x-api-key" in _err_str
                or ("api key" in _err_str and ("invalid" in _err_str or "blocked" in _err_str))
                or "unauthorized" in _err_str
                or "authentication" in _err_str
            )
            if _is_auth_error:
                self._last_summary_auth_failure = True
            if _is_json_decode and not _is_model_not_found and not _is_timeout:
                logger.error(
                    "Context compression failed: auxiliary LLM returned a "
                    "non-JSON response. provider=%s summary_model=%s "
                    "main_model=%s base_url=%s err=%s",
                    self.provider or "auto",
                    self.summary_model or "(main)",
                    self.model,
                    self.base_url or "default",
                    e,
                )
            if (
                (_is_model_not_found or _is_timeout or _is_json_decode or _is_streaming_closed)
                and self.summary_model
                and self.summary_model != self.model
                and not getattr(self, "_summary_model_fallen_back", False)
            ):
                if _is_json_decode:
                    _reason = "returned invalid JSON"
                elif _is_model_not_found:
                    _reason = "unavailable"
                elif _is_streaming_closed:
                    _reason = "closed stream prematurely"
                else:
                    _reason = "timed out"
                self._fallback_to_main_for_compression(e, _reason)
                return self._generate_summary(turns_to_summarize, focus_topic=focus_topic)  # retry immediately

            # Unknown-error best-effort retry on main model.  Losing N turns of
            # context is almost always worse than one extra summary attempt, so
            # if we haven't already fallen back and the summary model differs
            # from the main model, try once more on main before entering
            # cooldown.  Errors that DID match _is_model_not_found above are
            # already handled by the fast-path retry; this branch catches
            # everything else (400s, provider-specific "no route" strings,
            # aggregator rejections, etc.) where auto-retry is still safer
            # than dropping the turns.
            if (
                self.summary_model
                and self.summary_model != self.model
                and not getattr(self, "_summary_model_fallen_back", False)
            ):
                self._fallback_to_main_for_compression(e, "failed")
                return self._generate_summary(turns_to_summarize, focus_topic=focus_topic)

            # Transient errors (timeout, rate limit, network, JSON decode,
            # streaming premature-close) — shorter cooldown for JSON decode and
            # streaming-closed since those conditions can self-resolve quickly.
            _transient_cooldown = 30 if (_is_json_decode or _is_streaming_closed) else 60
            err_text = str(e).strip() or e.__class__.__name__
            if len(err_text) > 220:
                err_text = err_text[:217].rstrip() + "..."
            self._record_compression_failure_cooldown(_transient_cooldown, err_text)
            self._last_summary_error = err_text
            # A terminal connection/network failure (we reach this branch only
            # after any main-model fallback has already been tried or is
            # unavailable). Flag it so compress() ABORTS and preserves the
            # session unchanged instead of destroying the middle window for a
            # placeholder marker — retrying once the network recovers is
            # strictly better than dropping context (#29559, #25585). Mirrors
            # the auth-failure carve-out; independent of abort_on_summary_failure.
            if _is_streaming_closed:
                self._last_summary_network_failure = True
            logger.warning(
                "Failed to generate context summary: %s. "
                "Further summary attempts paused for %d seconds.",
                e,
                _transient_cooldown,
            )
            return None

    @staticmethod
    def _strip_summary_prefix(summary: str) -> str:
        """返回不包含当前、遗留或任何历史交接前缀的总结正文。

        历史前缀也必须被剥离：在较旧的前缀下持久化的交接内容可能会被继承到恢复的谱系（lineage）中（参见 #35344）。
        如果我们只是重新前置（re-prepend）当前的前缀而不移除旧的前缀，
        那么它所携带的陈旧指令就会一直内嵌在正文之中。
        """
        text = (summary or "").strip()
        # 尾部合并式总结（Merge-into-tail summaries）会在总结正文之前包裹先前的尾部内容。
        # 丢弃分隔符及其之前的全部内容，以便在二次压缩（re-compaction）时，仅将真正的
        # 总结正文结转（carried forward）下去 —— 否则，`[PRIOR CONTEXT]` 头部和陈旧的
        # 尾部内容将会泄露到下一个总结器提示词（summarizer prompt）中。
        if _MERGED_SUMMARY_DELIMITER in text:
            text = text.split(_MERGED_SUMMARY_DELIMITER, 1)[1].strip()
        for prefix in (SUMMARY_PREFIX, LEGACY_SUMMARY_PREFIX, *_HISTORICAL_SUMMARY_PREFIXES):
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip()
                break
        # 同时也剥离尾部的结束标记 —— 如果重新恢复的交接正文（rehydrated handoff body）
        # 保留了该标记，会将边界指令（boundary directive）泄露到迭代更新的总结器提示词中
        # （况且该标记在插入时反正也会被重新追加）。
        if text.endswith(_SUMMARY_END_MARKER):
            text = text[: -len(_SUMMARY_END_MARKER)].rstrip()
        return text

    @classmethod
    def _with_summary_prefix(cls, summary: str) -> str:
        """Normalize summary text to the current compaction handoff format."""
        text = cls._strip_summary_prefix(summary)
        return f"{SUMMARY_PREFIX}\n{text}" if text else SUMMARY_PREFIX

    @staticmethod
    def _is_context_summary_content(content: Any) -> bool:
        text = _content_text_for_contains(content).lstrip()
        # 尾部合并式总结（Merge-into-tail summaries）会在总结内容之前包裹先前的尾部内容，
        # 因此交接前缀（handoff prefix）会落在 `_MERGED_SUMMARY_DELIMITER` 之后，而不是位于开头。
        # 在该区域内也需要对总结进行检测，否则调用方（如自动焦点跳过、结转总结查找、
        # 最后一个真实用户锚点）会将合并后的总结消息误认为是真实的用户轮次。
        if _MERGED_SUMMARY_DELIMITER in text:
            text = text.split(_MERGED_SUMMARY_DELIMITER, 1)[1].lstrip()
        if text.startswith(SUMMARY_PREFIX) or text.startswith(LEGACY_SUMMARY_PREFIX):
            return True
        return any(text.startswith(p) for p in _HISTORICAL_SUMMARY_PREFIXES)

    @staticmethod
    def _has_compressed_summary_metadata(message: Any) -> bool:
        """Return True if *message* carries the compressed-summary flag.

        Callers (frontends, CLI, gateway) can use this to distinguish context
        compaction summaries from real assistant or user messages without
        relying on content-prefix heuristics.  The flag is in-process only —
        the wire sanitizers strip underscore-prefixed keys before API calls.
        """
        if not isinstance(message, dict):
            return False
        return bool(message.get(COMPRESSED_SUMMARY_METADATA_KEY))

    @classmethod
    def _derive_auto_focus_topic(
        cls,
        messages: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Infer a compact focus hint from the most recent real user turns."""
        candidates: list[str] = []
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if cls._is_context_summary_content(content):
                continue
            text = redact_sensitive_text(_content_text_for_contains(content).strip())
            if not text:
                continue
            text = " ".join(text.split())
            if len(text) > _AUTO_FOCUS_TURN_MAX_CHARS:
                text = text[: _AUTO_FOCUS_TURN_MAX_CHARS - 1].rstrip() + "…"
            candidates.append(text)
            if len(candidates) >= _AUTO_FOCUS_MAX_TURNS:
                break

        if not candidates:
            return None

        candidates.reverse()
        focus = "Recent user focus:\n" + "\n".join(f"- {item}" for item in candidates)
        if len(focus) > _AUTO_FOCUS_MAX_CHARS:
            focus = focus[: _AUTO_FOCUS_MAX_CHARS - 1].rstrip() + "…"
        return focus

    @classmethod
    def _find_latest_context_summary(
        cls,
        messages: List[Dict[str, Any]],
        start: int,
        end: int,
    ) -> tuple[Optional[int], str]:
        """Find the newest handoff summary inside a compression window."""
        for idx in range(end - 1, start - 1, -1):
            content = messages[idx].get("content")
            if cls._is_context_summary_content(content):
                return idx, cls._strip_summary_prefix(_content_text_for_contains(content))
        return None, ""

    # ------------------------------------------------------------------
    # Tool-call / tool-result pair integrity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_tool_call_id(tc) -> str:
        """Extract the call ID from a tool_call entry (dict or SimpleNamespace)."""
        if isinstance(tc, dict):
            return tc.get("call_id", "") or tc.get("id", "") or ""
        return getattr(tc, "call_id", "") or getattr(tc, "id", "") or ""

    def _sanitize_tool_pairs(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """修复压缩后孤立的 tool_call / tool_result 对。

        两种失败模式：
        1. 工具的 *结果*（result）引用了一个其 assistant 的 tool_call 已被
           移除（被摘要/截断）的 call_id。API 会对此予以拒绝，并报错
           "No tool call found for function call output with call_id ..."。
        2. assistant 消息中包含 tool_calls，但其对应的结果已被丢弃。
           API 会对此予以拒绝，因为每个 tool_call 后面都必须紧跟一个
           具有匹配 call_id 的工具结果（tool result）。

        此方法会移除孤立的结果，并从 assistant 消息中清除孤立的 tool_calls，
        以确保消息列表始终保持良构（well-formed）。

        以前的方法是为孤立的 tool_calls 插入桩（stub）``role="tool"`` 结果。
        但这引起了次级失败：API 前置的 ``repair_message_sequence()`` 使用
        ``tc.get("id")`` 来跟踪已知的 call ID，而此净化器（sanitizer）使用的是
        ``call_id || id``。当两者不一致时（Codex 响应 API 格式下：``id != call_id``），
        桩结果会在修复步骤中被默默丢弃，从而重新暴露原本的孤立问题。
        从源头上进行清除（Stripping）则避免了这一整类不匹配问题。
        """
        surviving_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = self._get_tool_call_id(tc)
                    if cid:
                        surviving_call_ids.add(cid)

        result_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_call_ids.add(cid)

        # 1. Remove tool results whose call_id has no matching assistant tool_call
        orphaned_results = result_call_ids - surviving_call_ids
        if orphaned_results:
            messages = [
                m for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
            ]
            if not self.quiet_mode:
                logger.info("Compression sanitizer: removed %d orphaned tool result(s)", len(orphaned_results))

        # 2. Strip orphaned tool_calls from assistant messages whose results
        #    were dropped.  Stripping is preferred over inserting stub results
        #    because stubs can be dropped by downstream repair_message_sequence
        #    when call_id != id (Codex Responses API format), re-exposing orphans.
        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            for msg in messages:
                if msg.get("role") != "assistant":
                    continue
                tcs = msg.get("tool_calls")
                if not tcs:
                    continue
                kept = [tc for tc in tcs if self._get_tool_call_id(tc) not in missing_results]
                if len(kept) != len(tcs):
                    if kept:
                        msg["tool_calls"] = kept
                    else:
                        msg.pop("tool_calls", None)
                        # Ensure the assistant message still has visible
                        # content so the API does not reject an empty turn.
                        content = msg.get("content")
                        if not content or (isinstance(content, str) and not content.strip()):
                            msg["content"] = "(tool call removed)"
            if not self.quiet_mode:
                logger.info(
                    "Compression sanitizer: stripped %d orphaned tool_call(s) from assistant messages",
                    len(missing_results),
                )

        return messages

    def _align_boundary_forward(self, messages: List[Dict[str, Any]], idx: int) -> int:
        """Push a compress-start boundary forward past any orphan tool results.

        If ``messages[idx]`` is a tool result, slide forward until we hit a
        non-tool message so we don't start the summarised region mid-group.
        """
        while idx < len(messages) and messages[idx].get("role") == "tool":
            idx += 1
        return idx

    def _effective_protect_first_n(self) -> int:
        """``protect_first_n`` decayed across compression cycles.

        ``protect_first_n`` keeps the first N non-system messages verbatim so
        the original task framing survives the FIRST compaction. But applying
        it on every subsequent pass fossilizes those early turns — they're
        re-copied into each child session and never summarized away, so old
        user messages become immortal and grow the head unboundedly across a
        long session (#11996). Once the session has been compressed at least
        once, the early turns are already captured in the handoff summary, so
        there's no need to keep re-protecting them: decay to 0 (the system
        prompt is still always protected separately by _protect_head_size).
        """
        if self.compression_count >= 1 or self._previous_summary:
            return 0
        return self.protect_first_n

    def _protect_head_size(self, messages: List[Dict[str, Any]]) -> int:
        """需要保护的头部消息的总数。

        ``protect_first_n`` 被定义为在系统提示词（system prompt）之外，*额外*需要保护
        的消息数量。系统提示词（如果存在于索引 0 处）总是被隐式保护的 —— 它是承重级
        的核心上下文（load-bearing context），绝对不能被总结掉。这确保了在不同调用路径
        下的语义稳定性，因为在这些路径中，系统提示词可能包含在、也可能不包含在 ``messages``
        列表中（例如，网关的 ``/compress`` 处理程序在调用 compress() 之前会先将其剥离）。

        首次压缩之后，``protect_first_n`` 部分会**发生衰减**（参见 _effective_protect_first_n），
        这样早期的用户轮次就不会在反复的压缩中被“固化/化石化”（fossilize）（参见 #11996）。

        示例（首次压缩）：
          protect_first_n=0 → 仅保护系统提示词（如果没有系统消息则什么都不保护）
          protect_first_n=3 → 系统提示词 + 前 3 条非系统消息
        首次压缩之后：
          仅保护系统提示词。
        """
        head = 0
        if messages and messages[0].get("role") == "system":
            head = 1
        return head + self._effective_protect_first_n()

    def _align_boundary_backward(self, messages: List[Dict[str, Any]], idx: int) -> int:
        """Pull a compress-end boundary backward to avoid splitting a
        tool_call / result group.

        If the boundary falls in the middle of a tool-result group (i.e.
        there are consecutive tool messages before ``idx``), walk backward
        past all of them to find the parent assistant message.  If found,
        move the boundary before the assistant so the entire
        assistant + tool_results group is included in the summarised region
        rather than being split (which causes silent data loss when
        ``_sanitize_tool_pairs`` removes the orphaned tail results).
        """
        if idx <= 0 or idx >= len(messages):
            return idx
        # Walk backward past consecutive tool results
        check = idx - 1
        while check >= 0 and messages[check].get("role") == "tool":
            check -= 1
        # If we landed on the parent assistant with tool_calls, pull the
        # boundary before it so the whole group gets summarised together.
        if check >= 0 and messages[check].get("role") == "assistant" and messages[check].get("tool_calls"):
            idx = check
        return idx

    # ------------------------------------------------------------------
    # Tail protection by token budget
    # ------------------------------------------------------------------

    def _find_last_user_message_idx(
        self, messages: List[Dict[str, Any]], head_end: int
    ) -> int:
        """Return the index of the last user-role message at or after *head_end*, or -1.

        A context-compaction handoff banner can be inserted as a ``role="user"``
        message (see the summary-role selection in ``compress``). It is internal
        continuity state, not a real user turn, so it must not be picked as the
        tail anchor — otherwise ``_ensure_last_user_message_in_tail`` protects
        the summary and rolls the genuine last user message into the next
        compaction, re-triggering the active-task loss the anchor exists to
        prevent.
        """
        for i in range(len(messages) - 1, head_end - 1, -1):
            msg = messages[i]
            if msg.get("role") == "user" and not self._is_context_summary_content(
                msg.get("content")
            ):
                return i
        return -1

    def _find_last_assistant_message_idx(
        self, messages: List[Dict[str, Any]], head_end: int
    ) -> int:
        """Return the index of the last user-visible assistant reply at or
        after *head_end*, or -1.

        A "user-visible reply" is an assistant message with non-empty
        textual content — i.e. one that the WebUI / TUI / SessionsPage
        rendered as a bubble the operator could read. We deliberately
        skip assistant messages that contain only ``tool_calls`` (and
        no text), because those render as small "calling tool X"
        indicators and aren't what the reporter means by "the output
        of the last message you sent" (#29824).

        Falling back to the most recent assistant message of ANY kind
        only kicks in when no content-bearing assistant message exists
        in the compressible region — typically a fresh session that
        just started a multi-step tool sequence with no prior reply
        to anchor. In that case the agent fix is a no-op and the
        existing user-message anchor carries the load.
        """
        last_any = -1
        for i in range(len(messages) - 1, head_end - 1, -1):
            msg = messages[i]
            if msg.get("role") != "assistant":
                continue
            if last_any < 0:
                last_any = i
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return i
            if isinstance(content, list):
                # Multimodal / Anthropic-style content: look for any
                # text block with non-empty text.
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text") or part.get("content")
                        if isinstance(text, str) and text.strip():
                            return i
        return last_any

    def _ensure_last_assistant_message_in_tail(
        self,
        messages: List[Dict[str, Any]],
        cut_idx: int,
        head_end: int,
    ) -> int:
        """Guarantee the most recent assistant message is in the protected tail.

        WebUI / TUI / SessionsPage bug (#29824). Without this anchor,
        ``_find_tail_cut_by_tokens`` can leave the user's most recent
        visible assistant response inside the compressed middle region —
        especially when the conversation has a single oversized tool
        result or a long stretch of tool-call/result pairs after the
        last assistant reply. The summariser then rolls that reply up
        into the single ``[CONTEXT COMPACTION — REFERENCE ONLY]`` block
        persisted as ``role="user"`` or ``role="assistant"``. From the
        operator's perspective the WebUI session viewer
        (``web/src/pages/SessionsPage.tsx``) and the TUI chat panel
        both suddenly show the opaque "Context compaction" block in the
        slot where they were just reading the assistant's actual reply:

            User:       "i cant see the output of the last message you
                         sent, i did see it previously, however now see
                         'context compaction'"

        Mirror of ``_ensure_last_user_message_in_tail`` but anchors on
        the last assistant-role message. Re-runs the tool-group
        alignment so we don't split a ``tool_call`` / ``tool_result``
        group that immediately precedes the anchored message — orphaned
        tool messages would otherwise be removed by
        ``_sanitize_tool_pairs`` and trigger the same data-loss symptom
        we're trying to prevent.
        """
        last_asst_idx = self._find_last_assistant_message_idx(messages, head_end)
        if last_asst_idx < 0:
            # No assistant message in the compressible region — nothing
            # to anchor (single-turn pre-reply state, etc.).
            return cut_idx
        if last_asst_idx >= cut_idx:
            # Already in the tail — the token-budget walk did the right
            # thing on its own.
            return cut_idx
        # Pull cut_idx back to the assistant message, then re-align so
        # we don't split a tool group that immediately precedes it
        # (e.g. an ``assistant(tool_calls)`` → ``tool(result)`` →
        # ``assistant(final reply)`` sequence would otherwise leave the
        # ``tool`` orphan when cut lands at the final reply).
        new_cut = self._align_boundary_backward(messages, last_asst_idx)
        if not self.quiet_mode:
            logger.debug(
                "Anchoring tail cut to last assistant message at index %d "
                "(was %d, aligned to %d) to keep the previously-visible "
                "reply out of the compaction summary (#29824)",
                last_asst_idx, cut_idx, new_cut,
            )
        # Safety: never go back into the head region.
        return max(new_cut, head_end + 1)

    def _ensure_last_user_message_in_tail(
        self,
        messages: List[Dict[str, Any]],
        cut_idx: int,
        head_end: int,
    ) -> int:
        """Guarantee the most recent user message is in the protected tail.

        Context compressor bug (#10896): ``_align_boundary_backward`` can pull
        ``cut_idx`` past a user message when it tries to keep tool_call/result
        groups together.  If the last user message ends up in the *compressed*
        middle region the LLM summariser writes it into "Historical Pending User Asks",
        but ``SUMMARY_PREFIX`` tells the next model to respond only to user
        messages *after* the summary — so the task effectively disappears from
        the active context, causing the agent to stall, repeat completed work,
        or silently drop the user's latest request.

        Fix: if the last user-role message is not already in the tail
        (``messages[cut_idx:]``), walk ``cut_idx`` back to include it.  We
        then re-align backward one more time to avoid splitting any
        tool_call/result group that immediately precedes the user message.

        Causal Coupling guard (#22523): the final ``max(last_user_idx,
        head_end + 1)`` clamp can push the cut *past* the user message when
        the user sits at ``head_end`` (the first compressible index) — the
        only case where ``head_end + 1 > last_user_idx``.  That splits the
        turn-pair: the user lands in the compressed region without its
        assistant reply, so the summariser records it as a pending ask and
        the next session re-executes the already-completed task.  When this
        split is unavoidable, push the cut *forward* to ``pair_end`` so the
        full pair (user + reply + tool results) is summarised together and
        correctly marked as completed.
        """
        last_user_idx = self._find_last_user_message_idx(messages, head_end)
        if last_user_idx < 0:
            # No user message found beyond head — nothing to anchor.
            return cut_idx

        if last_user_idx >= cut_idx:
            # Already in the tail; nothing to do.
            return cut_idx

        # The last user message is in the middle (compressed) region.
        # Pull cut_idx back to it directly — a user message is already a
        # clean boundary (no tool_call/result splitting risk), so there is no
        # need to call _align_boundary_backward here; doing so would
        # unnecessarily pull the cut further back into the preceding
        # assistant + tool_calls group.
        if not self.quiet_mode:
            logger.debug(
                "Anchoring tail cut to last user message at index %d "
                "(was %d) to prevent active-task loss after compression",
                last_user_idx,
                cut_idx,
            )
        # Safety: never go back into the head region.
        adjusted = max(last_user_idx, head_end + 1)
        if adjusted > last_user_idx:
            # The clamp would leave the user in the compressed region without
            # its reply.  Keep the pair intact by pushing the cut forward past
            # the whole (user + assistant + tool results) turn-pair so it is
            # summarised as a completed unit rather than a dangling ask.
            pair_end = self._find_turn_pair_end(messages, last_user_idx)
            if not self.quiet_mode:
                logger.debug(
                    "Causal Coupling: cut would split turn-pair at user %d; "
                    "pushing cut forward to pair_end %d so the completed pair "
                    "is summarised together (#22523)",
                    last_user_idx,
                    pair_end,
                )
            return max(pair_end, head_end + 1)
        return adjusted

    def _find_turn_pair_end(
        self,
        messages: List[Dict[str, Any]],
        user_idx: int,
    ) -> int:
        """Return the index *after* the complete turn-pair starting at *user_idx*.

        A turn-pair is: ``user`` -> ``assistant`` [-> zero-or-more ``tool``
        results].  Returns the index of the first message that does *not*
        belong to the pair, i.e. the natural cut point that keeps the pair
        intact on one side of the boundary.

        If *user_idx* is the last message (no assistant reply yet), returns
        ``user_idx + 1`` so the user message itself is minimally covered.
        """
        n = len(messages)
        idx = user_idx + 1
        if idx >= n:
            return idx  # user is the very last message — no reply yet
        if messages[idx].get("role") != "assistant":
            return idx  # no assistant reply immediately following
        idx += 1
        # Include any tool results that belong to this assistant turn.
        while idx < n and messages[idx].get("role") == "tool":
            idx += 1
        return idx

    def _find_tail_cut_by_tokens(
        self, messages: List[Dict[str, Any]], head_end: int,
        token_budget: int | None = None,
    ) -> int:
        """从消息末尾反向遍历，累加 Token 直至达到预算上限。返回尾部（tail）开始处的索引。

            ``token_budget`` 默认值为 ``self.tail_token_budget``，该值衍生自
            ``summary_target_ratio * context_length``，因此它会随着模型的上下文窗口
            自动缩放。

            Token 预算是首要基准。即使预算耗尽，有界的最低消息数量下限也会将近期的一小段
            轮次逐字原样（verbatim）保留，但预算允许最多超额 1.5 倍，以避免从一条
            超大消息（如工具输出、文件读取等）内部进行截断。如果连该下限也超过了预算的
            1.5 倍，则截断位置将被直接放置在头部（head）之后，以便压缩依然能够运行。

            绝不从 tool_call/result（工具调用/结果）组的内部进行截断。始终确保最晚（最新）
            的一条用户消息包含在尾部之中（参见 ``_ensure_last_user_message_in_tail``）。
            """
        if token_budget is None:
            token_budget = self.tail_token_budget
        n = len(messages)
        # Hard minimum: always keep a bounded recent-message floor in the tail.
        # ``protect_last_n`` remains a minimum up to the cap; the cap avoids
        # preserving a whole run of bulky tool outputs on every compaction.
        available_tail = max(0, n - head_end - 1)
        min_tail_floor = max(3, min(self.protect_last_n, _MAX_TAIL_MESSAGE_FLOOR))
        # Leave at least two non-head messages available to summarize on short
        # transcripts; otherwise compression can replace a tiny middle with a
        # summary and save no messages at all.
        compressible_tail_cap = max(3, available_tail - 2)
        min_tail = (
            min(min_tail_floor, compressible_tail_cap, available_tail)
            if available_tail > 1 else 0
        )
        soft_ceiling = int(token_budget * 1.5)
        accumulated = 0
        cut_idx = n  # start from beyond the end

        for i in range(n - 1, head_end - 1, -1):
            msg = messages[i]
            msg_tokens = _estimate_msg_budget_tokens(msg)
            # Stop once we exceed the soft ceiling (unless we haven't hit min_tail yet)
            if accumulated + msg_tokens > soft_ceiling and (n - i) >= min_tail:
                break
            accumulated += msg_tokens
            cut_idx = i

        # If the backward walk never broke early because the entire transcript
        # fits within soft_ceiling, accumulated now holds the total transcript
        # size.  Without intervention _ensure_last_user_message_in_tail pushes
        # cut_idx forward to include the last user message, and the caller's
        # compress_start >= compress_end guard either returns unchanged (no-op)
        # or compresses a single message — both of which trigger the infinite
        # compaction loop described in #40803.
        #
        # Fix: when the whole transcript fits in soft_ceiling, compute a
        # meaningful cut point using the raw (non-inflated) budget so that
        # compression actually summarizes a worthwhile middle section.
        if cut_idx <= head_end and accumulated <= soft_ceiling and accumulated > 0:
            # The entire compressable region fits in the soft ceiling.
            # Re-walk with the raw budget (no 1.5x multiplier) to find a
            # split that gives the summarizer something useful.
            raw_budget = token_budget
            raw_accumulated = 0
            for j in range(n - 1, head_end - 1, -1):
                raw_msg = messages[j]
                raw_tok = _estimate_msg_budget_tokens(raw_msg)
                if raw_accumulated + raw_tok > raw_budget and (n - j) >= min_tail:
                    cut_idx = j
                    break
                raw_accumulated += raw_tok
                cut_idx = j
            # If the raw-budget walk also consumed everything (very small
            # transcript), fall through — the existing fallback logic below
            # will still force a minimal cut after head_end.

        # Ensure we protect at least min_tail messages
        fallback_cut = n - min_tail
        cut_idx = min(cut_idx, fallback_cut)

        # If the token budget would protect everything (small conversations),
        # force a cut after the head so compression can still remove middle turns.
        if cut_idx <= head_end:
            cut_idx = max(fallback_cut, head_end + 1)

        # Align to avoid splitting tool groups
        cut_idx = self._align_boundary_backward(messages, cut_idx)

        # Ensure the most recent user message is always in the tail so the
        # active task is never lost to compression (fixes #10896).
        cut_idx = self._ensure_last_user_message_in_tail(messages, cut_idx, head_end)

        # Ensure the most recent assistant message is always in the tail
        # so the previously-visible reply isn't silently rolled into the
        # ``[CONTEXT COMPACTION — REFERENCE ONLY]`` block (fixes #29824).
        # Each anchor only walks ``cut_idx`` backward, so chaining them is
        # monotonic — the tail can only grow, never shrink.
        cut_idx = self._ensure_last_assistant_message_in_tail(messages, cut_idx, head_end)

        return max(cut_idx, head_end + 1)

    # ------------------------------------------------------------------
    # ContextEngine: manual /compress preflight
    # ------------------------------------------------------------------

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        """Return True if there is a non-empty middle region to compact.

        Overrides the ABC default so the gateway ``/compress`` guard can
        skip the LLM call when the transcript is still entirely inside
        the protected head/tail.
        """
        compress_start = self._align_boundary_forward(messages, self._protect_head_size(messages))
        compress_end = self._find_tail_cut_by_tokens(messages, compress_start)
        return compress_start < compress_end

    # ------------------------------------------------------------------
    # Main compression entry point
    # ------------------------------------------------------------------

    def compress(self, messages: List[Dict[str, Any]], current_tokens: int = None, focus_topic: str = None, force: bool = False) -> List[Dict[str, Any]]:
        """通过总结中间轮次来压缩对话消息。

        算法步骤：
          1. 修剪旧的工具结果（开销低廉的预处理，无需调用 LLM）
          2. 保护头部消息（系统提示词 + 第一次交谈）
          3. 根据 Token 预算寻找尾部边界（保留约 20K Token 的近期上下文）
          4. 使用结构化的 LLM 提示词总结中间轮次
          5. 在二次压缩时，迭代更新上一次的总结

        压缩完成后，孤立的 tool_call / tool_result（工具调用/工具结果）对会被清理干净，
        从而确保 API 绝不会接收到不匹配的 ID。

        参数：
            focus_topic: 用于引导定向压缩的可选焦点字符串。当
                提供该参数时，总结器会优先保留与该主题相关的信息，
                并更具侵略性地（更大力度地）压缩其他所有内容。
                灵感来源于 Claude Code 的 ``/compact``。
            force: 若为 True，则在运行前清除当前处于激活状态的“总结失败冷却时间”，
                以便在自动压缩中止后，手动的 ``/compress`` 可以立即重试。
                自动压缩的调用者则传入 False。
        """
        # 重置每次调用的总结失败状态 —— 调用者在 compress() 返回后检查这些字段，
        # 以决定是否向用户展示警告。
        self._last_summary_dropped_count = 0
        self._last_summary_fallback_used = False
        self._last_summary_error = None
        self._last_aux_model_failure_error = None
        self._last_aux_model_failure_model = None
        self._last_compress_aborted = False
        self._last_compression_made_progress = False
        # 注意：【不要】在此处重置 _last_summary_auth_failure 或
        # _last_summary_network_failure。这些标志是由 _generate_summary()
        # 在发生终结性故障（terminal failure）时设置的，并且在成功生成总结时
        # 已经被清除了。过早（急切地）重置它们会使冷却时间保护机制失效：
        # _generate_summary() 会从冷却时间逻辑中提前返回并返回 None，而不会重新
        # 设置这些标志，从而导致下方的中止保护（abort guard）检测到 False，并向下
        # 进入具有破坏性的静态兜底方案（static-fallback） —— 这正是 #29559
        # 中所描述的导致数据丢失的确切情况。让它们在多次 compress() 调用之间
        # 保持持久存在是安全的，因为一次成功的总结总会清除这两个标志。

        # 手动的 /compress（force=True）会绕过失败冷却时间，以便用户在
        # 自动压缩中止后可以立即重试。如果没有此处理，/compress 在失败后
        # 的 30-60 秒内将会默默地执行空操作（no-op）。
        if force:
            self._clear_compression_failure_cooldown()
        n_messages = len(messages)
        # Only need head + 3 tail messages minimum (token budget decides the real tail size)
        _min_for_compress = self._protect_head_size(messages) + 3 + 1
        if n_messages <= _min_for_compress:
            # 记录该空操作（no-op），正如其下方同级的“无可压缩窗口”分支
            # 所做的那样（参见 #40803）。如果在返回时没有更新
            # 防抖动计数器（anti-thrashing counter），会导致 `should_compress()` 在面对
            # 一个永远无法缩小的对话记录时始终返回 True：当提示词由于无法压缩的
            # 基础下限（系统提示词 + 工具 schema）而超过阈值时，随后的每一个轮次
            # 都会重新触发一次在这里原样返回的压缩操作，从而导致 CLI 看起来像卡死了一样。
            self._ineffective_compression_count += 1
            self._last_compression_savings_pct = 0.0
            if not self.quiet_mode:
                logger.warning(
                    "Cannot compress: only %d messages (need > %d). "
                    "ineffective_compression_count=%d",
                    n_messages, _min_for_compress,
                    self._ineffective_compression_count,
                )
            return messages

        display_tokens = current_tokens if current_tokens else self.last_prompt_tokens or estimate_messages_tokens_rough(messages)

        # Phase 1: Prune old tool results (cheap, no LLM call)
        messages, pruned_count = self._prune_old_tool_results(
            messages, protect_tail_count=self.protect_last_n,
            protect_tail_tokens=self.tail_token_budget,
        )
        if pruned_count and not self.quiet_mode:
            logger.info("Pre-compression: pruned %d old tool result(s)", pruned_count)

        # Phase 2: Determine boundaries
        compress_start = self._protect_head_size(messages)
        compress_start = self._align_boundary_forward(messages, compress_start)

        # Use token-budget tail protection instead of fixed message count
        compress_end = self._find_tail_cut_by_tokens(messages, compress_start)

        if compress_start >= compress_end:
            # 无可压缩窗口 —— 整个对话记录完全容纳在尾部预算（软上限/soft_ceiling）之内。
            # 如果不将其记录为一次无效压缩，`should_compress()` 中的防抖动保护机制
            # （anti-thrashing guard）就永远不会触发，从而导致随后的每一个轮次都会
            # 重新触发一个空操作（no-op）的压缩死循环。（参见 #40803）
            self._ineffective_compression_count += 1
            self._last_compression_savings_pct = 0.0
            if not self.quiet_mode:
                logger.warning(
                    "Compression skipped: compress_start (%d) >= compress_end (%d) "
                    "— transcript fits within tail budget, nothing to compress. "
                    "ineffective_compression_count=%d",
                    compress_start, compress_end,
                    self._ineffective_compression_count,
                )
            return messages
        # 祛除上一轮总结之后的raw msg
        turns_to_summarize = messages[compress_start:compress_end]
        # 在会话恢复（resume）后，一个持久化的交接总结（handoff summary）可能会位于受保护的头部
        # （通常紧跟在系统提示词之后）。从第一条非系统消息开始，在整个压缩窗口内进行搜索，
        # 以便我们能够恢复迭代总结的状态，而无需将该交接总结序列化为一个新的轮次。
        # 交接总结之后的受保护消息仍属于活动上下文（live context），因此仅总结那些
        # 既在交接总结之后、又在当前压缩窗口之内的消息。
        summary_search_start = 1 if messages and messages[0].get("role") == "system" else 0
        summary_idx, summary_body = self._find_latest_context_summary(
            messages,
            summary_search_start,
            compress_end,
        )
        if summary_idx is not None:
            if summary_body and not self._previous_summary:
                self._previous_summary = summary_body
            turns_to_summarize = messages[max(compress_start, summary_idx + 1):compress_end]
        elif self._previous_summary:
            # 当前消息中未找到交接总结，但 `_previous_summary` 不为空 ——
            # 它是由另一个（现已结束的）会话设置的（例如定时任务/cron job、先前的 `/new`）。
            # 将其丢弃，以防止 `_generate_summary()` 通过迭代更新路径将跨会话的内容
            # 注入到总结器提示词（summarizer prompt）中。
            self._previous_summary = None

        if not self.quiet_mode:
            logger.info(
                "Context compression triggered (%d tokens >= %d threshold)",
                display_tokens,
                self.threshold_tokens,
            )
            logger.info(
                "Model context limit: %d tokens (%.0f%% = %d)",
                self.context_length,
                self.threshold_percent * 100,
                self.threshold_tokens,
            )
            tail_msgs = n_messages - compress_end
            logger.info(
                "Summarizing turns %d-%d (%d turns), protecting %d head + %d tail messages",
                compress_start + 1,
                compress_end,
                len(turns_to_summarize),
                compress_start,
                tail_msgs,
            )

        # Phase 3: Generate structured summary
        # 数据脱敏后，各个msg的简要总结（机器切分，限制最大长度）
        summary_focus_topic = focus_topic or self._derive_auto_focus_topic(messages)
        summary = self._generate_summary(turns_to_summarize, focus_topic=summary_focus_topic)

        # 如果总结生成失败，其行为取决于
        # ``abort_on_summary_failure`` (配置: compression.abort_on_summary_failure):
        #   True  → 完全中止（ABORT）压缩。返回未修改的消息，
        #           并设置 _last_compress_aborted=True，以便调用方可以警告
        #           用户并停止自动压缩重试循环。
        #   False → 顺延到下方的默认回退路径：插入
        #           一个确定性的“总结不可用（summary unavailable）”交接并丢弃
        #           中间窗口。记录 _last_summary_fallback_used /
        #           _last_summary_dropped_count，以便网关健康检查（gateway hygiene）
        #           显现警告。
        # 默认值为 False（历史行为）。
        #
        # 异常情况 —— 身份验证（auth）与临时网络故障总是会中止。
        # 总结调用返回的 401/403 意味着凭据或终结点
        # 已损坏（无效/被封禁的密钥，或者指向了错误
        # 推理主机的 Token）。连接/流关闭错误意味着网络
        # 在压缩瞬间发生了波动（#29559）。在这两种情况下，
        # 在凭据损坏时轮转到带有占位总结的子会话，
        # 不仅毫无益处，还会让用户滞留在功能降级的会话中 ——
        # 随后的每一次调用都会以相同的方式失败。因此，当失败属于
        # 身份验证错误时，无论 abort_on_summary_failure 的值为什么，
        # 我们都会中止压缩，在凭据修复前保持对话原样不变。
        if not summary and (
            self.abort_on_summary_failure
            or self._last_summary_auth_failure
            or self._last_summary_network_failure
        ):
            n_skipped = compress_end - compress_start
            self._last_summary_dropped_count = 0  # nothing actually dropped
            self._last_summary_fallback_used = False
            self._last_compress_aborted = True
            if not self.quiet_mode:
                if self._last_summary_auth_failure:
                    logger.warning(
                        "Summary generation failed with an authentication "
                        "error — aborting compression. %d message(s) preserved "
                        "unchanged; the session was NOT rotated. Check your "
                        "provider credential / inference endpoint, then retry "
                        "with /compress or start fresh with /new.",
                        n_skipped,
                    )
                elif self._last_summary_network_failure:
                    logger.warning(
                        "Summary generation failed with a network/connection "
                        "error — aborting compression. %d message(s) preserved "
                        "unchanged; the session was NOT rotated. This is "
                        "transient: retry with /compress once connectivity "
                        "recovers, or continue the conversation as-is.",
                        n_skipped,
                    )
                else:
                    logger.warning(
                        "Summary generation failed — aborting compression "
                        "(compression.abort_on_summary_failure=true). "
                        "%d message(s) preserved unchanged. Conversation is "
                        "frozen until the next /compress or /new.",
                        n_skipped,
                    )
            return messages

        # Phase 4: Assemble compressed message list
        compressed = []
        for i in range(compress_start):
            msg = _fresh_compaction_message_copy(messages[i])
            if i == 0 and msg.get("role") == "system":
                existing = msg.get("content")
                # [注意：为了节省上下文空间，先前的部分对话轮次已被压缩为一份交接总结。
                # 当前的会话状态可能仍会反映先前的成果，
                # 因此请在这份总结和状态的基础上继续开展工作，而不是重复劳动。
                # 无论是否进行过压缩，你的持久化记忆（MEMORY.md、USER.md）
                # 都始终具有完全的权威性。]
                _compression_note = "[Note: Some earlier conversation turns have been compacted into a handoff summary to preserve context space. The current session state may still reflect earlier work, so build on that summary and state rather than re-doing work. Your persistent memory (MEMORY.md, USER.md) remains fully authoritative regardless of compaction.]"
                if _compression_note not in _content_text_for_contains(existing):
                    msg["content"] = _append_text_to_content(
                        existing,
                        "\n\n" + _compression_note if isinstance(existing, str) and existing else _compression_note,
                    )
            compressed.append(msg)

        # 如果 LLM 总结失败，插入一个确定性的回退内容，以便模型
        # 至少能获得局部可恢复的连贯性锚点，而不是一个
        # 没有实质内容的“移除了 N 条消息”的标记。
        if not summary:
            if not self.quiet_mode:
                logger.warning("Summary generation failed — inserting deterministic fallback context summary")
            n_dropped = compress_end - compress_start
            self._last_summary_dropped_count = n_dropped
            self._last_summary_fallback_used = True
            summary = self._build_static_fallback_summary(
                turns_to_summarize,
                reason=self._last_summary_error,
            )

        _merge_summary_into_tail = False
        last_head_role = messages[compress_start - 1].get("role", "user") if compress_start > 0 else "user"
        first_tail_role = messages[compress_end].get("role", "user") if compress_end < n_messages else "user"
        # 当唯一受保护的头部消息（head message）是系统提示词时，
        # 总结将成为 API 请求中第一个 *可见* 的消息
        # （大多数适配器 —— Anthropic、Bedrock —— 会将系统提示词作为
        # 一个独立的 ``system`` 参数发送，而不是放在 ``messages[]`` 内部）。
        # Anthropic 会无条件拒绝第一个消息不是 role=user 的请求，
        # 因此我们必须将总结固定为 "user" 角色，
        # 并防止下方的翻转逻辑将其还原（#52160）。
        _force_user_leading = last_head_role == "system"
        # 零用户轮次保护（#58753）。上方的 #52160 保护仅在系统提示词
        # 位于 ``messages`` *内部* 时（即网关的 ``/compress`` 路径）才会触发。
        # 主自动压缩路径传递的转录记录中不包含系统提示词（它是在
        # 构建请求时被前置追加的），因此 ``last_head_role`` 默认为 "user"，
        # 总结会以 role="assistant" 的形式发出。在某些会话中，唯一真正的
        # 用户轮次恰好落在了被压缩的中间区域 —— 例如，一个 ``hermes kanban``
        # 工作流在启动时仅被塞入了一个简短的 ``"work kanban task <id>"`` 提示词，
        # 随后全是 assistant/tool 轮次 —— 这会导致压缩后的转录记录中包含
        # 零个用户角色的消息。兼容 OpenAI 的后端（如 vLLM/Qwen）会拒绝
        # 此类请求，并返回不可重试的 ``400 No user query found in messages`` 错误，
        # 从而导致工作流崩溃且无法恢复（每次恢复运行都会重放同一段被污染的
        # 历史记录）。如果受保护的头部和保留的尾部中都没有用户角色的消息幸存，
        # 则总结【必须】携带 role="user"，以确保请求中始终至少包含一个用户轮次。
        if not _force_user_leading:
            _user_survives = any(
                messages[i].get("role") == "user"
                for i in range(0, compress_start)
            ) or any(
                messages[i].get("role") == "user"
                for i in range(compress_end, n_messages)
            )
            if not _user_survives:
                _force_user_leading = True
        # 选择一个能避免与两侧邻居连续出现相同角色的角色。
        # 优先级：首先避免与头部（已提交）冲突，其次是尾部。
        if last_head_role in {"assistant", "tool"} or _force_user_leading:
            summary_role = "user"
        else:
            summary_role = "assistant"
        # 如果选中的角色与尾部相撞，并且翻转后不会
        # 与头部相撞，则进行翻转。
        if summary_role == first_tail_role:
            flipped = "assistant" if summary_role == "user" else "user"
            if flipped != last_head_role and not _force_user_leading:
                summary_role = flipped
            else:
                # 两个角色都会导致连续出现相同角色的消息
                # （例如：head=assistant，tail=user —— 哪种角色都不行）。
                # 将摘要合并到第一条 tail 消息中，
                # 而不是插入一条会破坏交替的独立消息。
                _merge_summary_into_tail = True

        # 当摘要作为一条独立的 role="user" 消息时，
        # 弱模型会将过去用户请求的“## 活跃任务”字面引用误读为新的输入（#11475，#14521）。
        # 当它作为 role="assistant" 时，模型可能会将摘要文本
        # 机械地重复为自己的输出（#33256）。在这两种情况下，都要追加
        # 显式的结束标记，以便模型拥有清晰的“摘要在此结束，
        # 请响应下方消息”的信号。
        if not _merge_summary_into_tail:
            summary = summary + "\n\n" + _SUMMARY_END_MARKER

        if not _merge_summary_into_tail:
            compressed.append({
                "role": summary_role,
                "content": summary,
                COMPRESSED_SUMMARY_METADATA_KEY: True,
            })

        for i in range(compress_end, n_messages):
            msg = _fresh_compaction_message_copy(messages[i])
            if _merge_summary_into_tail and i == compress_end:
                # 将摘要合并到第一条 tail 消息中，但将
                # 结束标记（END MARKER）放在最末尾，以便模型看到一个
                # 明确无误的边界。旧的 tail 内容保留在摘要之前
                # 作为参考资料，并清晰地划分开，
                # 以免被误认为是需要响应的新消息。
                # 使用 _append_text_to_content 来安全地处理
                # 字符串和多模态列表（multimodal-list）两种内容类型。
                # 修复了跨压缩边界的幽灵消息（ghost-message）泄漏问题，
                # 之前旧的 head 消息会原封不动地幸存下来并出现在
                # 摘要之前。
                old_content = msg.get("content", "")
                suffix = (
                    "\n\n" + _MERGED_SUMMARY_DELIMITER + "\n\n"
                    + summary + "\n\n"
                    + _SUMMARY_END_MARKER
                )
                msg["content"] = _append_text_to_content(
                    _append_text_to_content(old_content, suffix, prepend=False),
                    _MERGED_PRIOR_CONTEXT_HEADER + "\n",
                    prepend=True,
                )
                # Mark the merged message so frontends can identify it as
                # containing a compression summary prefix.
                msg[COMPRESSED_SUMMARY_METADATA_KEY] = True
                _merge_summary_into_tail = False
            compressed.append(msg)

        self.compression_count += 1

        compressed = self._sanitize_tool_pairs(compressed)

        # 将最新一个带有图片的 user 轮次之前的所有被压缩消息中的图片部分
        # 替换为简短的文本占位符。如果不进行此处理，tail 消息将永远保留
        # 它们原本数 MB 大小的 base-64 图片载荷，这可能会使后续的每一次
        # API 请求都超出服务商的请求体大小限制，从而导致会话卡死（wedge）。
        # 移植自 Kilo-Org/kilocode#9434。
        compressed = _strip_historical_media(compressed)

        new_estimate = estimate_messages_tokens_rough(compressed)

        # 防抖动/防抖机制：在同等基础上衡量有效性。
        #
        # ``display_tokens`` 通常是 ``current_tokens`` —— 也就是服务商实际的
        # prompt 算力占用（Token 计数），其中包含了系统提示词（system prompt）和工具模式（tool schemas）。
        # 而 ``new_estimate`` 仅涵盖消息（messages）部分。将两者进行对比，会使一个
        # 几乎没有释放任何空间的消息压缩过程，看起来像是节省了大约 96% 的空间，
        # 从而导致下方的计数器在每次传递时都会重置，使防抖机制形同虚设。
        # 消息压缩只能缩小消息本身，因此应根据它所接收的消息来对其效果进行评分。
        pre_estimate = estimate_messages_tokens_rough(messages)
        saved_estimate = pre_estimate - new_estimate
        savings_pct = (saved_estimate / pre_estimate * 100) if pre_estimate > 0 else 0
        self._last_compression_savings_pct = savings_pct

        # 仅针对消息的节省量仅用于诊断。防抖机制的最终判定
        # 由下一次服务商报告的 prompt 计数决定，它回答了
        # 实际的问题：这个已完成的边界是否降到了阈值以下？
        # 如果在此处也把低的消息节省估算值计算在内，那么在实际读取值
        # 仍然超出阈值的情况下，会让一次压缩被判定两次失败（两击出局）。

        if not self.quiet_mode:
            logger.info(
                "Compressed: %d -> %d messages (~%d tokens saved, %.0f%%)",
                n_messages,
                len(compressed),
                saved_estimate,
                savings_pct,
            )
            logger.info("Compression #%d complete", self.compression_count)

        # 强制不变性（#57491）：任何被压缩的消息在离开 compress() 时，
        # 都不得携带会话存储持久化标记（session-store persistence marker）。
        # 上面针对每个位置的清除操作是基于位置的；而这次统一的终端清扫
        # 从结构上保证了这一点，从而使未来的复制位置无法将该标记
        # 再次泄漏到子会话的刷新中。
        _strip_persistence_markers(compressed)
        self._last_compression_made_progress = True

        return compressed
