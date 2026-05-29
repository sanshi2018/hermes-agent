"""Assorted AIAgent runtime helpers — moved out of run_agent.py for clarity.

Each function takes the parent ``AIAgent`` as its first argument
(``agent``) except for the static helpers (``sanitize_tool_call_arguments``,
``drop_thinking_only_and_merge_users``) which are stateless.  AIAgent
keeps thin forwarders for backward compatibility.

Methods covered:
* ``convert_to_trajectory_format`` — internal -> trajectory-file format
* ``sanitize_tool_call_arguments`` — repair corrupted JSON in tool_calls
* ``repair_message_sequence`` — enforce alternation invariants
* ``strip_think_blocks`` — remove inline reasoning from stored content
* ``recover_with_credential_pool`` — rotate pool entries on 429
* ``try_recover_primary_transport`` — re-create OpenAI client after rate-limit
* ``drop_thinking_only_and_merge_users`` — Anthropic-style cleanup
* ``restore_primary_runtime`` — un-do fallback activation
* ``extract_reasoning`` — pull reasoning fields out of API responses
* ``dump_api_request_debug`` — write request body for post-mortem
* ``anthropic_prompt_cache_policy`` — compute cache_control breakpoints
* ``create_openai_client`` — build the per-agent OpenAI SDK client
"""

from __future__ import annotations

import copy
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_cli.timeouts import get_provider_request_timeout
from agent.tool_dispatch_helpers import _trajectory_normalize_msg, make_tool_result_message
from agent.trajectory import convert_scratchpad_to_think
from agent.credential_pool import STATUS_EXHAUSTED
from agent.error_classifier import FailoverReason
from utils import base_url_host_matches, base_url_hostname, env_var_enabled, atomic_json_write

logger = logging.getLogger(__name__)


def _ra():
    """Lazy ``run_agent`` reference for test-patch routing."""
    import run_agent
    return run_agent



def convert_to_trajectory_format(agent, messages: List[Dict[str, Any]], user_query: str, completed: bool) -> List[Dict[str, Any]]:
    """将内部消息格式转换为用于保存的轨迹（trajectory）格式。

    参数：
        messages (List[Dict]): 内部消息历史记录
        user_query (str): 用户原始查询
        completed (bool): 对话是否已成功完成

    返回：
        List[Dict]: 轨迹格式的消息列表
    """
    # 规范化多模态工具结果 —— 轨迹仅包含纯文本，
    # 因此将带有图像的工具消息替换为其 text_summary（文本摘要），
    # 以避免在每个保存的轨迹中嵌入约 1MB 的 base64 数据块。
    messages = [_trajectory_normalize_msg(m) for m in messages]
    trajectory = []
    
    # Add system message with tool definitions
    # system_msg = (
    #     "你是一个支持函数调用的 AI 模型。"
    #     "在 <tools> </tools> XML 标签内为你提供了函数签名。"
    #     "你可以调用一个或多个函数来协助处理用户的查询。"
    #     "如果可用工具与协助处理用户查询无关，只需以自然对话语言进行回复。"
    #     "请勿对传入函数的值做任何主观假设。"
    #     "在调用并执行函数后，函数结果将通过 <tool_response> </tool_response> XML 标签提供给你。"
    #     "以下是可用工具：\n"
    #     f"<tools>\n{agent._format_tools_for_system_message()}\n</tools>\n"
    #     "对于每次函数调用，需返回一个 JSON 对象，每个对象应符合以下 Pydantic 模型的 JSON Schema：\n"
    #     "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
    #     "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\n"
    #     "每次函数调用都必须包含在 <tool_call> </tool_call> XML 标签内。\n"
    #     "示例：\n<tool_call>\n{'name': <函数名>,'arguments': <参数字典>}\n</tool_call>"
    # )
    system_msg = (
        "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
        "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
        "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
        "into functions. After calling & executing the functions, you will be provided with function results within "
        "<tool_response> </tool_response> XML tags. Here are the available tools:\n"
        f"<tools>\n{agent._format_tools_for_system_message()}\n</tools>\n"
        "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
        "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
        "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\n"
        "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
        "Example:\n<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>"
    )
    
    trajectory.append({
        "from": "system",
        "value": system_msg
    })
    
    # Add the actual user prompt (from the dataset) as the first human message
    trajectory.append({
        "from": "human",
        "value": user_query
    })

    # 跳过第一条消息（即用户查询），因为我们上面已经添加过了。
    # 预填（prefill）消息仅在发起 API 调用时注入
    # （并不在 messages 列表内），因此此处无需调整偏移量。
    i = 1
    
    while i < len(messages):
        msg = messages[i]
        
        if msg["role"] == "assistant":
            # Check if this message has tool calls
            if "tool_calls" in msg and msg["tool_calls"]:
                # Format assistant message with tool calls
                # Add <think> tags around reasoning for trajectory storage
                content = ""
                
                # Prepend reasoning in <think> tags if available (native thinking tokens)
                if msg.get("reasoning") and msg["reasoning"].strip():
                    content = f"<think>\n{msg['reasoning']}\n</think>\n"
                
                if msg.get("content") and msg["content"].strip():
                    # Convert any <REASONING_SCRATCHPAD> tags to <think> tags
                    # (used when native thinking is disabled and model reasons via XML)
                    content += convert_scratchpad_to_think(msg["content"]) + "\n"
                
                # Add tool calls wrapped in XML tags
                for tool_call in msg["tool_calls"]:
                    if not tool_call or not isinstance(tool_call, dict): continue
                    # Parse arguments - should always succeed since we validate during conversation
                    # but keep try-except as safety net
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                    except json.JSONDecodeError:
                        # This shouldn't happen since we validate and retry during conversation,
                        # but if it does, log warning and use empty dict
                        logger.warning(f"Unexpected invalid JSON in trajectory conversion: {tool_call['function']['arguments'][:100]}")
                        arguments = {}
                    
                    tool_call_json = {
                        "name": tool_call["function"]["name"],
                        "arguments": arguments
                    }
                    content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"
                
                # Ensure every gpt turn has a <think> block (empty if no reasoning)
                # so the format is consistent for training data
                if "<think>" not in content:
                    content = "<think>\n</think>\n" + content
                
                trajectory.append({
                    "from": "gpt",
                    "value": content.rstrip()
                })
                
                # Collect all subsequent tool responses
                tool_responses = []
                j = i + 1
                while j < len(messages) and messages[j]["role"] == "tool":
                    tool_msg = messages[j]
                    # Format tool response with XML tags
                    tool_response = "<tool_response>\n"
                    
                    # Try to parse tool content as JSON if it looks like JSON
                    tool_content = tool_msg["content"]
                    try:
                        if tool_content.strip().startswith(("{", "[")):
                            tool_content = json.loads(tool_content)
                    except (json.JSONDecodeError, AttributeError):
                        pass  # Keep as string if not valid JSON
                    
                    tool_index = len(tool_responses)
                    tool_name = (
                        msg["tool_calls"][tool_index]["function"]["name"]
                        if tool_index < len(msg["tool_calls"])
                        else "unknown"
                    )
                    tool_response += json.dumps({
                        "tool_call_id": tool_msg.get("tool_call_id", ""),
                        "name": tool_name,
                        "content": tool_content
                    }, ensure_ascii=False)
                    tool_response += "\n</tool_response>"
                    tool_responses.append(tool_response)
                    j += 1
                
                # Add all tool responses as a single message
                if tool_responses:
                    trajectory.append({
                        "from": "tool",
                        "value": "\n".join(tool_responses)
                    })
                    i = j - 1  # Skip the tool messages we just processed
            
            else:
                # Regular assistant message without tool calls
                # Add <think> tags around reasoning for trajectory storage
                content = ""
                
                # Prepend reasoning in <think> tags if available (native thinking tokens)
                if msg.get("reasoning") and msg["reasoning"].strip():
                    content = f"<think>\n{msg['reasoning']}\n</think>\n"
                
                # Convert any <REASONING_SCRATCHPAD> tags to <think> tags
                # (used when native thinking is disabled and model reasons via XML)
                raw_content = msg["content"] or ""
                content += convert_scratchpad_to_think(raw_content)
                
                # Ensure every gpt turn has a <think> block (empty if no reasoning)
                if "<think>" not in content:
                    content = "<think>\n</think>\n" + content
                
                trajectory.append({
                    "from": "gpt",
                    "value": content.strip()
                })
        
        elif msg["role"] == "user":
            trajectory.append({
                "from": "human",
                "value": msg["content"]
            })
        
        i += 1
    
    return trajectory


# 它保证发给 API 的历史里每个 tool_call 的参数都是合法 JSON——坏的替换成 {}，
# 同时给对应的 tool 结果打上"参数已损坏"的标记（必要时补插占位结果），
# 既让请求能通过 provider 的校验，又让模型知道那次调用不可信。
# 它和后面的 repair_message_sequence（修复角色交替错乱）
# 同属请求前的"消息history消毒"防线。
def sanitize_tool_call_arguments(
    messages: list,
    *,
    logger=None,
    session_id: str = None,
) -> int:
    """Repair corrupted assistant tool-call argument JSON in-place."""
    log = logger or logging.getLogger(__name__)
    if not isinstance(messages, list):
        return 0

    repaired = 0
    marker = _ra().AIAgent._TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER

    def _prepend_marker(tool_msg: dict) -> None:
        existing = tool_msg.get("content")
        if isinstance(existing, str):
            if not existing:
                tool_msg["content"] = marker
            elif not existing.startswith(marker):
                tool_msg["content"] = f"{marker}\n{existing}"
            return
        if existing is None:
            tool_msg["content"] = marker
            return
        try:
            existing_text = json.dumps(existing)
        except TypeError:
            existing_text = str(existing)
        tool_msg["content"] = f"{marker}\n{existing_text}"

    message_index = 0
    while message_index < len(messages):
        msg = messages[message_index]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            message_index += 1
            continue

        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            message_index += 1
            continue

        insert_at = message_index + 1
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue

            arguments = function.get("arguments")
            if arguments is None or arguments == "":
                function["arguments"] = "{}"
                continue
            if isinstance(arguments, str) and not arguments.strip():
                function["arguments"] = "{}"
                continue
            if not isinstance(arguments, str):
                continue

            try:
                json.loads(arguments)
            except json.JSONDecodeError:
                tool_call_id = tool_call.get("id")
                function_name = function.get("name", "?")
                preview = arguments[:80]
                log.warning(
                    "Corrupted tool_call arguments repaired before request "
                    "(session=%s, message_index=%s, tool_call_id=%s, function=%s, preview=%r)",
                    session_id or "-",
                    message_index,
                    tool_call_id or "-",
                    function_name,
                    preview,
                )
                function["arguments"] = "{}"

                existing_tool_msg = None
                scan_index = message_index + 1
                while scan_index < len(messages):
                    candidate = messages[scan_index]
                    if not isinstance(candidate, dict) or candidate.get("role") != "tool":
                        break
                    if candidate.get("tool_call_id") == tool_call_id:
                        existing_tool_msg = candidate
                        break
                    scan_index += 1

                if existing_tool_msg is None:
                    messages.insert(
                        insert_at,
                        make_tool_result_message(
                            function_name if function_name != "?" else "",
                            marker,
                            tool_call_id,
                        ),
                    )
                    insert_at += 1
                else:
                    _prepend_marker(existing_tool_msg)

                repaired += 1

        message_index += 1

    return repaired



def repair_message_sequence(agent, messages: List[Dict]) -> int:
    """Collapse malformed role-alternation left in the live history.

    Providers (OpenAI, OpenRouter, Anthropic) expect strict alternation:
    after the system message, user/tool alternates with assistant, with
    no two consecutive user messages and no tool-result that doesn't
    follow an assistant-with-tool_calls. Violations cause silent empty
    responses on most providers, which triggers the empty-retry loop.

    This runs right before the API call as a defensive belt — by the
    time it fires, the scaffolding strip should already have prevented
    most shapes, but external callers (gateway multi-queue replay,
    session resume, cron, explicit conversation_history passed in by
    host code) can feed in already-broken histories.

    Repairs applied:
      1. Stray ``tool`` messages whose ``tool_call_id`` doesn't match
         any preceding assistant tool_call — dropped.
      2. Consecutive ``user`` messages — merged with newline separator
         so no user input is lost.

    Deliberately does NOT rewind orphan ``assistant(tool_calls)+tool``
    pairs that precede a user message — that pattern IS valid when the
    previous turn completed normally and the user jumped in to redirect
    before the model got a continuation turn (the ongoing dialog
    pattern). The empty-response scaffolding stripper handles the
    genuinely-broken variant via its flag-gated rewind.

    Returns the number of repairs made (for logging/telemetry).
    """
    if not messages:
        return 0

    repairs = 0

    # Pass 1: drop stray tool messages that don't follow a known
    # assistant tool_call_id. Uses a rolling set of known ids refreshed
    # on each assistant message.
    known_tool_ids: set = set()
    filtered: List[Dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            filtered.append(msg)
            continue
        role = msg.get("role")
        if role == "assistant":
            known_tool_ids = set()
            for tc in (msg.get("tool_calls") or []):
                tc_id = tc.get("id") if isinstance(tc, dict) else None
                if tc_id:
                    known_tool_ids.add(tc_id)
            filtered.append(msg)
        elif role == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id and tc_id in known_tool_ids:
                filtered.append(msg)
            else:
                repairs += 1
        else:
            if role == "user":
                # A user turn closes the tool-result run; subsequent
                # tool messages without a fresh assistant tool_call
                # are orphans.
                known_tool_ids = set()
            filtered.append(msg)

    # Pass 2: merge consecutive user messages. Preserves all user input
    # so nothing the user typed is lost.
    merged: List[Dict] = []
    for msg in filtered:
        if (
            merged
            and isinstance(msg, dict)
            and msg.get("role") == "user"
            and isinstance(merged[-1], dict)
            and merged[-1].get("role") == "user"
        ):
            prev = merged[-1]
            prev_content = prev.get("content", "")
            new_content = msg.get("content", "")
            # Only merge plain-text content; leave multimodal (list)
            # content alone — collapsing image/audio blocks risks
            # mangling the attachment structure.
            if isinstance(prev_content, str) and isinstance(new_content, str):
                prev["content"] = (
                    (prev_content + "\n\n" + new_content)
                    if prev_content and new_content
                    else (prev_content or new_content)
                )
                repairs += 1
                continue
        merged.append(msg)

    if repairs > 0:
        # Rewrite in place so downstream paths (persistence, return
        # value, session DB flush) see the repaired sequence.
        messages[:] = merged

    return repairs



def strip_think_blocks(agent, content: str) -> str:
    """从内容中移除推理/思考块，仅返回可见文本。

    处理以下四种情况：
      1. 闭合标签对（``<think>…</think>``）—— 当服务商发出完整的推理块时的常见路径。
      2. 块边界处（文本开头或换行符之后）未闭合的开放标签 —— 例如 MiniMax M2.7 / NIM
         端点，这些情况下结束标签会被丢弃。从开放标签到字符串结尾的所有内容都会被剥离。
         块边界检查镜像了 ``gateway/stream_consumer.py`` 的过滤器，从而避免模型在正文中
         提及 ``<think>`` 时被过度剥离。
      3. 漏掉的孤立开放/闭合标签。
      4. 标签变体：``<think>``、``<thinking>``、``<reasoning>``、
         ``<REASONING_SCRATCHPAD>``、``<thought>``（Gemma 4），均不区分大小写。

    此外，还会剥离某些开源模型（尤其是 OpenRouter 上的 Gemma 变体）在 assistant
    内容内部发出的独立工具调用 XML 块，而不是通过结构化的 ``tool_calls`` 字段发出的块：
      * ``<tool_call>…</tool_call>``
      * ``<tool_calls>…</tool_calls>``
      * ``<tool_result>…</tool_result>``
      * ``<function_call>…</function_call>``
      * ``<function_calls>…</function_calls>``
      * ``<function name="…">…</function>``（Gemma 风格）

    移植自 openclaw/openclaw#67318。对 ``<function>`` 变体进行了边界把关（仅当标签
    位于行首或标点符号之后，且带有 ``name="..."`` 属性时才进行剥离），以便保留正文中
    诸如“在 JavaScript 中使用 <function>”之类的提及。
    """
    if not content:
        return ""
    # 1. Closed tag pairs — case-insensitive for all variants so
    #    mixed-case tags (<THINK>, <Thinking>) don't slip through to
    #    the unterminated-tag pass and take trailing content with them.
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<REASONING_SCRATCHPAD>.*?</REASONING_SCRATCHPAD>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL | re.IGNORECASE)
    # 1b. Tool-call XML blocks (openclaw/openclaw#67318). Handle the
    #     generic tag names first — they have no attribute gating since
    #     a literal <tool_call> in prose is already vanishingly rare.
    for _tc_name in ("tool_call", "tool_calls", "tool_result",
                      "function_call", "function_calls"):
        content = re.sub(
            rf'<{_tc_name}\b[^>]*>.*?</{_tc_name}>',
            '',
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
    # 1c. <function name="...">...</function> — Gemma-style standalone
    #     tool call. Only strip when the tag sits at a block boundary
    #     (start of text, after a newline, or after sentence-ending
    #     punctuation) AND carries a name="..." attribute. This keeps
    #     prose mentions like "Use <function> to declare" safe.
    content = re.sub(
        r'(?:(?<=^)|(?<=[\n\r.!?:]))[ \t]*'
        r'<function\b[^>]*\bname\s*=[^>]*>'
        r'(?:(?:(?!</function>).)*)</function>',
        '',
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # 2. Unterminated reasoning block — open tag at a block boundary
    #    (start of text, or after a newline) with no matching close.
    #    Strip from the tag to end of string.  Fixes #8878 / #9568
    #    (MiniMax M2.7 leaking raw reasoning into assistant content).
    content = re.sub(
        r'(?:^|\n)[ \t]*<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)\b[^>]*>.*$',
        '',
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # 3. Stray orphan open/close tags that slipped through.
    content = re.sub(
        r'</?(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>\s*',
        '',
        content,
        flags=re.IGNORECASE,
    )
    # 3b. Stray tool-call closers. (We do NOT strip bare <function> or
    #     unterminated <function name="..."> because a truncated tail
    #     during streaming may still be valuable to the user; matches
    #     OpenClaw's intentional asymmetry.)
    content = re.sub(
        r'</(?:tool_call|tool_calls|tool_result|function_call|function_calls|function)>\s*',
        '',
        content,
        flags=re.IGNORECASE,
    )
    return content



def recover_with_credential_pool(
    agent,
    *,
    status_code: Optional[int],
    has_retried_429: bool,
    classified_reason: Optional[FailoverReason] = None,
    error_context: Optional[Dict[str, Any]] = None,
) -> tuple[bool, bool]:
    """Attempt credential recovery via pool rotation.

    Returns (recovered, has_retried_429).
    On rate limits: first occurrence retries same credential (sets flag True).
                    second consecutive failure rotates to next credential.
    On billing exhaustion: immediately rotates.
    On auth failures: attempts token refresh before rotating.

    `classified_reason` lets the recovery path honor the structured error
    classifier instead of relying only on raw HTTP codes. This matters for
    providers that surface billing/rate-limit/auth conditions under a
    different status code, such as Anthropic returning HTTP 400 for
    "out of extra usage".
    """
    pool = agent._credential_pool
    if pool is None:
        return False, has_retried_429

    # Defensive guard: if a fallback provider is active and its provider name
    # doesn't match the pool's provider, the pool belongs to the PRIMARY
    # provider.  Mutating it based on fallback errors would corrupt the
    # primary's credential state (see #33088) and, via _swap_credential,
    # overwrite the agent's base_url back to the primary's endpoint — every
    # subsequent request then goes to the wrong host and 404s (see #33163).
    # The pool should only act when the agent is still on the same provider
    # that seeded the pool.
    current_provider = (getattr(agent, "provider", "") or "").strip().lower()
    pool_provider = (getattr(pool, "provider", "") or "").strip().lower()
    if current_provider and pool_provider and current_provider != pool_provider:
        _ra().logger.warning(
            "Credential pool provider mismatch: pool=%s, agent=%s — "
            "skipping pool mutation to avoid cross-provider contamination",
            pool_provider, current_provider,
        )
        return False, has_retried_429

    effective_reason = classified_reason
    if effective_reason is None:
        if status_code == 402:
            effective_reason = FailoverReason.billing
        elif status_code == 429:
            effective_reason = FailoverReason.rate_limit
        elif status_code in {401, 403}:
            effective_reason = FailoverReason.auth

    if effective_reason == FailoverReason.billing:
        rotate_status = status_code if status_code is not None else 402
        next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
        if next_entry is not None:
            _ra().logger.info(
                "Credential %s (billing) — rotated to pool entry %s",
                rotate_status,
                getattr(next_entry, "id", "?"),
            )
            agent._swap_credential(next_entry)
            return True, False
        return False, has_retried_429

    if effective_reason == FailoverReason.rate_limit:
        # If current credential is already marked exhausted, skip retry and
        # rotate immediately. This prevents the "cancel-between-429s" trap
        # where has_retried_429 (a local var) gets reset on each new prompt,
        # causing the pool to retry the same exhausted credential forever.
        current_entry = pool.current()
        current_last_status = getattr(current_entry, "last_status", None) if current_entry else None
        if current_last_status == STATUS_EXHAUSTED:
            _ra().logger.info(
                "Credential already exhausted (last_status=%s) — rotating immediately instead of retrying",
                current_last_status,
            )
            rotate_status = status_code if status_code is not None else 429
            next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
            if next_entry is not None:
                _ra().logger.info(
                    "Credential %s (rate limit, pre-exhausted) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                agent._swap_credential(next_entry)
                return True, False
            return False, True

        usage_limit_reached = False
        if error_context:
            context_reason = str(error_context.get("reason") or "").lower()
            context_message = str(error_context.get("message") or "").lower()
            usage_limit_reached = (
                "usage_limit_reached" in context_reason
                or "gousagelimit" in context_reason
                or "usage limit reached" in context_message
                or "usage limit has been reached" in context_message
            )
        if not has_retried_429 and not usage_limit_reached:
            return False, True
        rotate_status = status_code if status_code is not None else 429
        next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
        if next_entry is not None:
            _ra().logger.info(
                "Credential %s (rate limit) — rotated to pool entry %s",
                rotate_status,
                getattr(next_entry, "id", "?"),
            )
            agent._swap_credential(next_entry)
            return True, False
        return False, True

    if effective_reason == FailoverReason.auth:
        # Subscription/entitlement 403s look like auth failures on the wire
        # but refresh cannot fix them — the OAuth token is already valid,
        # the account simply lacks the entitlement.  Without this guard,
        # ``try_refresh_current()`` keeps minting fresh tokens against the
        # same unsubscribed account and the main agent loop spins re-issuing
        # the same 403 until the user Ctrl+C's.
        #
        # Defense-in-depth for #26847: xAI's backend has been seen to 403
        # standard SuperGrok subscribers with bodies that don't match the
        # existing entitlement keyword set in ``_is_entitlement_failure``.
        # Any 403 against ``xai-oauth`` is treated as entitlement here so
        # the refresh loop can't spin in those cases either.
        #
        # Exception (#29344): xAI's ``[WKE=unauthenticated:...]`` suffix and
        # the ``OAuth2 access token could not be validated`` phrasing are
        # xAI's authoritative "this is a stale token, not entitlement"
        # signal.  When either fires we must NOT apply the catch-all
        # override — refresh is the recoverable path for these bodies, and
        # blanket-classifying them as entitlement was the bug that left
        # long-running TUI sessions stuck on stale tokens until the user
        # exited and reopened.
        is_entitlement = agent._is_entitlement_failure(error_context, status_code)
        if not is_entitlement and status_code == 403 and (agent.provider or "") == "xai-oauth":
            _disambiguator_haystack = " ".join(
                str(error_context.get(k) or "").lower()
                for k in ("message", "reason", "code", "error")
                if isinstance(error_context, dict)
            )
            _is_xai_auth_failure = (
                "[wke=unauthenticated:" in _disambiguator_haystack
                or "oauth2 access token could not be validated" in _disambiguator_haystack
            )
            if not _is_xai_auth_failure:
                is_entitlement = True
        if is_entitlement:
            _ra().logger.info(
                "Credential %s — entitlement-shaped 403 from %s; "
                "skipping pool refresh (account lacks subscription, "
                "not a transient auth failure).",
                status_code if status_code is not None else "auth",
                agent.provider or "provider",
            )
            return False, has_retried_429
        refreshed = pool.try_refresh_current()
        if refreshed is not None:
            _ra().logger.info(f"Credential auth failure — refreshed pool entry {getattr(refreshed, 'id', '?')}")
            agent._swap_credential(refreshed)
            return True, has_retried_429
        # Refresh failed — rotate to next credential instead of giving up.
        # The failed entry is already marked exhausted by try_refresh_current().
        rotate_status = status_code if status_code is not None else 401
        next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
        if next_entry is not None:
            _ra().logger.info(
                "Credential %s (auth refresh failed) — rotated to pool entry %s",
                rotate_status,
                getattr(next_entry, "id", "?"),
            )
            agent._swap_credential(next_entry)
            return True, False

    return False, has_retried_429



def try_recover_primary_transport(
    agent, api_error: Exception, *, retry_count: int, max_retries: int,
) -> bool:
    """Attempt one extra primary-provider recovery cycle for transient transport failures.

    After ``max_retries`` exhaust, rebuild the primary client (clearing
    stale connection pools) and give it one more attempt before falling
    back.  This is most useful for direct endpoints (custom, Z.AI,
    Anthropic, OpenAI, local models) where a TCP-level hiccup does not
    mean the provider is down.

    Skipped for proxy/aggregator providers (OpenRouter, Nous) which
    already manage connection pools and retries server-side — if our
    retries through them are exhausted, one more rebuilt client won't help.
    """
    if agent._fallback_activated:
        return False

    # Only for transient transport errors
    error_type = type(api_error).__name__
    if error_type not in _TRANSIENT_TRANSPORT_ERRORS:
        return False

    # Skip for aggregator providers — they manage their own retry infra
    if agent._is_openrouter_url():
        return False
    provider_lower = (agent.provider or "").strip().lower()
    if provider_lower in {"nous", "nous-research"}:
        return False

    try:
        # Close existing client to release stale connections
        if getattr(agent, "client", None) is not None:
            try:
                agent._close_openai_client(
                    agent.client, reason="primary_recovery", shared=True,
                )
            except Exception:
                pass

        # Rebuild from primary snapshot
        rt = agent._primary_runtime
        agent._client_kwargs = dict(rt["client_kwargs"])
        agent.model = rt["model"]
        agent.provider = rt["provider"]
        agent.base_url = rt["base_url"]
        agent.api_mode = rt["api_mode"]
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        agent.api_key = rt["api_key"]

        if agent.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client
            agent._anthropic_api_key = rt["anthropic_api_key"]
            agent._anthropic_base_url = rt["anthropic_base_url"]
            agent._anthropic_client = build_anthropic_client(
                rt["anthropic_api_key"], rt["anthropic_base_url"],
                timeout=get_provider_request_timeout(agent.provider, agent.model),
            )
            agent._is_anthropic_oauth = rt["is_anthropic_oauth"]
            agent.client = None
        else:
            agent.client = agent._create_openai_client(
                dict(rt["client_kwargs"]),
                reason="primary_recovery",
                shared=True,
            )

        wait_time = min(3 + retry_count, 8)
        agent._vprint(
            f"{agent.log_prefix}🔁 Transient {error_type} on {agent.provider} — "
            f"rebuilt client, waiting {wait_time}s before one last primary attempt.",
            force=True,
        )
        time.sleep(wait_time)
        return True
    except Exception as e:
        logger.warning("Primary transport recovery failed: %s", e)
        return False

# ── End provider fallback ──────────────────────────────────────────────



def drop_thinking_only_and_merge_users(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """丢弃仅包含思考的助手轮次；合并由此遗留下的任何相邻的用户消息。

    该操作仅在每次调用的 ``api_messages`` 副本上运行。存储的对话历史
    （``agent.messages``）绝不会被修改，因此用户在 CLI/网关的输出记录中
    仍然可以看到思考块，且会话持久化也会保留完整的追踪轨迹。只有发送给
    服务商的传输层副本会被清理。

    为什么选择“丢弃并合并”而不是“注入存根（占位）文本”：
    - 伪造 ``"."`` / ``"(continued)"`` 文本会留在历史记录中，并导致未来的轮次
      看到模型实际上并未生成的输出。
    - 丢弃该轮次能保持数据的真实性；合并相邻的用户消息则能满足服务商对
      角色交替（User/Assistant 交替出现）的绝对约束。
    - 这是 Claude Code 中的 ``normalizeMessagesForAPI`` 所采用的模式
      （包含 filterOrphanedThinkingOnlyMessages + mergeAdjacentUserMessages）。
    """
    if not messages:
        return messages

    # Pass 1: drop thinking-only assistant turns.
    kept = [m for m in messages if not _ra().AIAgent._is_thinking_only_assistant(m)]
    dropped = len(messages) - len(kept)
    if dropped == 0:
        return messages

    # Pass 2: merge any newly-adjacent user messages.
    merged: List[Dict[str, Any]] = []
    merges = 0
    for m in kept:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.get("role") == "user"
            and m.get("role") == "user"
        ):
            prev_content = prev.get("content", "")
            cur_content = m.get("content", "")
            # Work on a copy of ``prev`` so the caller's input dicts are
            # never mutated. ``_sanitize_api_messages`` upstream already
            # hands us per-call copies, but staying pure here means we
            # can be called safely from anywhere (tests, other loops).
            prev_copy = dict(prev)
            # Only string-content merge is meaningful for role-alternation
            # purposes. If either side is a list (multimodal), append as a
            # separate block rather than collapsing.
            if isinstance(prev_content, str) and isinstance(cur_content, str):
                sep = "\n\n" if prev_content and cur_content else ""
                prev_copy["content"] = prev_content + sep + cur_content
            elif isinstance(prev_content, list) and isinstance(cur_content, list):
                prev_copy["content"] = list(prev_content) + list(cur_content)
            elif isinstance(prev_content, list) and isinstance(cur_content, str):
                if cur_content:
                    prev_copy["content"] = list(prev_content) + [
                        {"type": "text", "text": cur_content}
                    ]
                else:
                    prev_copy["content"] = list(prev_content)
            elif isinstance(prev_content, str) and isinstance(cur_content, list):
                new_blocks: List[Dict[str, Any]] = []
                if prev_content:
                    new_blocks.append({"type": "text", "text": prev_content})
                new_blocks.extend(cur_content)
                prev_copy["content"] = new_blocks
            else:
                # Unknown content shape — fall back to appending separately
                # (violates alternation, but safer than raising in a hot path).
                merged.append(m)
                continue
            merged[-1] = prev_copy
            merges += 1
        else:
            merged.append(m)

    _ra().logger.debug(
        "Pre-call sanitizer: dropped %d thinking-only assistant turn(s), "
        "merged %d adjacent user message(s)",
        dropped,
        merges,
    )
    return merged



def restore_primary_runtime(agent) -> bool:
    """Restore the primary runtime at the start of a new turn.

    In long-lived CLI sessions a single AIAgent instance spans multiple
    turns.  Without restoration, one transient failure pins the session
    to the fallback provider for every subsequent turn.  Calling this at
    the top of ``run_conversation()`` makes fallback turn-scoped.

    The gateway caches agents across messages (``_agent_cache`` in
    ``gateway/run.py``), so this restoration IS needed there too.
    """
    if not agent._fallback_activated:
        # Reset the chain index even when no fallback was activated this
        # turn.  Without this, a turn where _try_activate_fallback() was
        # called but returned False (chain exhausted or provider not
        # configured) leaves _fallback_index >= len(_fallback_chain) while
        # _fallback_activated stays False.  The next turn skips this block
        # entirely, stranding the index and silently blocking all future
        # fallback attempts for the session.  Fixes #20465.
        agent._fallback_index = 0
        return False

    if getattr(agent, "_rate_limited_until", 0) > time.monotonic():
        return False  # primary still in rate-limit cooldown, stay on fallback

    rt = agent._primary_runtime
    try:
        # ── Core runtime state ──
        agent.model = rt["model"]
        agent.provider = rt["provider"]
        agent.base_url = rt["base_url"]           # setter updates _base_url_lower
        agent.api_mode = rt["api_mode"]
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        agent.api_key = rt["api_key"]
        agent._client_kwargs = dict(rt["client_kwargs"])
        agent._use_prompt_caching = rt["use_prompt_caching"]
        # Default to native layout when the restored snapshot predates the
        # native-vs-proxy split (older sessions saved before this PR).
        agent._use_native_cache_layout = rt.get(
            "use_native_cache_layout",
            agent.api_mode == "anthropic_messages" and agent.provider == "anthropic",
        )

        # ── Rebuild client for the primary provider ──
        if agent.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client
            agent._anthropic_api_key = rt["anthropic_api_key"]
            agent._anthropic_base_url = rt["anthropic_base_url"]
            agent._anthropic_client = build_anthropic_client(
                rt["anthropic_api_key"], rt["anthropic_base_url"],
                timeout=get_provider_request_timeout(agent.provider, agent.model),
            )
            agent._is_anthropic_oauth = rt["is_anthropic_oauth"]
            agent.client = None
        else:
            agent.client = agent._create_openai_client(
                dict(rt["client_kwargs"]),
                reason="restore_primary",
                shared=True,
            )

        # ── Restore context engine state ──
        cc = agent.context_compressor
        cc.update_model(
            model=rt["compressor_model"],
            context_length=rt["compressor_context_length"],
            base_url=rt["compressor_base_url"],
            api_key=rt["compressor_api_key"],
            provider=rt["compressor_provider"],
            api_mode=rt.get("compressor_api_mode", ""),
        )

        # ── Reset fallback chain for the new turn ──
        agent._fallback_activated = False
        agent._fallback_index = 0

        logger.info(
            "Primary runtime restored for new turn: %s (%s)",
            agent.model, agent.provider,
        )
        return True
    except Exception as e:
        logger.warning("Failed to restore primary runtime: %s", e)
        return False

# Which error types indicate a transient transport failure worth
# one more attempt with a rebuilt client / connection pool.
_TRANSIENT_TRANSPORT_ERRORS = frozenset({
    "ReadTimeout", "ConnectTimeout", "PoolTimeout",
    "ConnectError", "RemoteProtocolError",
    "APIConnectionError", "APITimeoutError",
})



def extract_reasoning(agent, assistant_message) -> Optional[str]:
    """从助手的消息中提取推理/思考内容。

    OpenRouter 以及各种服务商可能会以多种格式返回推理内容：
    1. message.reasoning - 直接推理字段（DeepSeek、Qwen 等）
    2. message.reasoning_content - 替代字段（Moonshot AI、Novita 等）
    3. message.reasoning_details - 由 {type, summary, ...} 对象组成的数组（OpenRouter 统一格式）

    参数:
        assistant_message: 来自 API 响应的助手消息对象

    返回:
        合并后的推理文本，若未找到推理内容则返回 None
    """
    reasoning_parts = []
    
    # Check direct reasoning field
    if hasattr(assistant_message, 'reasoning') and assistant_message.reasoning:
        reasoning_parts.append(assistant_message.reasoning)
    
    # Check reasoning_content field (alternative name used by some providers)
    if hasattr(assistant_message, 'reasoning_content') and assistant_message.reasoning_content:
        # Don't duplicate if same as reasoning
        if assistant_message.reasoning_content not in reasoning_parts:
            reasoning_parts.append(assistant_message.reasoning_content)
    
    # Check reasoning_details array (OpenRouter unified format)
    # Format: [{"type": "reasoning.summary", "summary": "...", ...}, ...]
    if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
        for detail in assistant_message.reasoning_details:
            if isinstance(detail, dict):
                # Extract summary from reasoning detail object
                summary = (
                    detail.get('summary')
                    or detail.get('thinking')
                    or detail.get('content')
                    or detail.get('text')
                )
                if summary and summary not in reasoning_parts:
                    reasoning_parts.append(summary)

    # Some providers embed reasoning directly inside assistant content
    # instead of returning structured reasoning fields.  Only fall back
    # to inline extraction when no structured reasoning was found.
    content = getattr(assistant_message, "content", None)
    if not reasoning_parts and isinstance(content, list):
        # DeepSeek V4 Pro (and compatible providers) return content as a
        # list of typed blocks, e.g.:
        #   [{"type": "thinking", "thinking": "..."}, {"type": "output", ...}]
        # Without this branch the thinking text is silently dropped and the
        # next turn fails with HTTP 400 ("thinking must be passed back").
        # Refs #21944.
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                thinking_text = block.get("thinking") or block.get("text") or ""
                thinking_text = thinking_text.strip()
                if thinking_text and thinking_text not in reasoning_parts:
                    reasoning_parts.append(thinking_text)
    if not reasoning_parts and isinstance(content, str) and content:
        inline_patterns = (
            r"<think>(.*?)</think>",
            r"<thinking>(.*?)</thinking>",
            r"<thought>(.*?)</thought>",
            r"<reasoning>(.*?)</reasoning>",
            r"<REASONING_SCRATCHPAD>(.*?)</REASONING_SCRATCHPAD>",
        )
        for pattern in inline_patterns:
            flags = re.DOTALL | re.IGNORECASE
            for block in re.findall(pattern, content, flags=flags):
                cleaned = block.strip()
                if cleaned and cleaned not in reasoning_parts:
                    reasoning_parts.append(cleaned)
    
    # Combine all reasoning parts
    if reasoning_parts:
        return "\n\n".join(reasoning_parts)
    
    return None



def dump_api_request_debug(
    agent,
    api_kwargs: Dict[str, Any],
    *,
    reason: str,
    error: Optional[Exception] = None,
) -> Optional[Path]:
    """
    为当前激活的推理 API 导出（Dump）一份便于调试的 HTTP 请求记录。

    从 api_kwargs 中捕获请求体（排除类似 timeout 等仅用于传输的键）。
    旨在用于调试服务商端那些重试也无济于事的 4xx 错误。
    """
    try:
        body = copy.deepcopy(api_kwargs)
        body.pop("timeout", None)
        body = {k: v for k, v in body.items() if v is not None}

        api_key = None
        try:
            api_key = getattr(agent.client, "api_key", None)
        except Exception as e:
            _ra().logger.debug("Could not extract API key for debug dump: %s", e)

        dump_payload: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "session_id": agent.session_id,
            "reason": reason,
            "request": {
                "method": "POST",
                "url": f"{agent.base_url.rstrip('/')}{'/responses' if agent.api_mode == 'codex_responses' else '/chat/completions'}",
                "headers": {
                    "Authorization": f"Bearer {agent._mask_api_key_for_logs(api_key)}",
                    "Content-Type": "application/json",
                },
                "body": body,
            },
        }

        if error is not None:
            error_info: Dict[str, Any] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            for attr_name in ("status_code", "request_id", "code", "param", "type"):
                attr_value = getattr(error, attr_name, None)
                if attr_value is not None:
                    error_info[attr_name] = attr_value

            body_attr = getattr(error, "body", None)
            if body_attr is not None:
                error_info["body"] = body_attr

            response_obj = getattr(error, "response", None)
            if response_obj is not None:
                try:
                    error_info["response_status"] = getattr(response_obj, "status_code", None)
                    error_info["response_text"] = response_obj.text
                except Exception as e:
                    _ra().logger.debug("Could not extract error response details: %s", e)

            dump_payload["error"] = error_info

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dump_file = agent.logs_dir / f"request_dump_{agent.session_id}_{timestamp}.json"
        atomic_json_write(dump_file, dump_payload, default=str)

        agent._vprint(f"{agent.log_prefix}🧾 Request debug dump written to: {dump_file}")

        if env_var_enabled("HERMES_DUMP_REQUEST_STDOUT"):
            print(json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str))

        return dump_file
    except Exception as dump_error:
        if agent.verbose_logging:
            logger.warning(f"Failed to dump API request debug payload: {dump_error}")
        return None



def anthropic_prompt_cache_policy(
    agent,
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[bool, bool]:
    """Decide whether to apply Anthropic prompt caching and which layout to use.

    Returns ``(should_cache, use_native_layout)``:
      * ``should_cache`` — inject ``cache_control`` breakpoints for this
        request (applies to OpenRouter Claude, native Anthropic, and
        third-party gateways that speak the native Anthropic protocol).
      * ``use_native_layout`` — place markers on the *inner* content
        blocks (native Anthropic accepts and requires this layout);
        when False markers go on the message envelope (OpenRouter and
        OpenAI-wire proxies expect the looser layout).

    Third-party providers using the native Anthropic transport
    (``api_mode == 'anthropic_messages'`` + Claude-named model) get
    caching with the native layout so they benefit from the same
    cost reduction as direct Anthropic callers, provided their
    gateway implements the Anthropic cache_control contract
    (MiniMax, Zhipu GLM, LiteLLM's Anthropic proxy mode all do).

    Qwen / Alibaba-family models on OpenCode, OpenCode Go, and direct
    Alibaba (DashScope) also honour Anthropic-style ``cache_control``
    markers on OpenAI-wire chat completions. Upstream pi-mono #3392 /
    pi #3393 documented this for opencode-go Qwen. Without markers
    these providers serve zero cache hits, re-billing the full prompt
    on every turn.
    """
    eff_provider = (provider if provider is not None else agent.provider) or ""
    eff_base_url = base_url if base_url is not None else (agent.base_url or "")
    eff_api_mode = api_mode if api_mode is not None else (agent.api_mode or "")
    eff_model = (model if model is not None else agent.model) or ""

    model_lower = eff_model.lower()
    provider_lower = eff_provider.lower()
    is_claude = "claude" in model_lower
    is_openrouter = base_url_host_matches(eff_base_url, "openrouter.ai")
    # Nous Portal proxies to OpenRouter behind the scenes — identical
    # OpenAI-wire envelope cache_control semantics. Treat it as an
    # OpenRouter-equivalent endpoint for caching layout purposes.
    is_nous_portal = "nousresearch" in eff_base_url.lower()
    is_anthropic_wire = eff_api_mode == "anthropic_messages"
    is_native_anthropic = (
        is_anthropic_wire
        and (eff_provider == "anthropic" or base_url_hostname(eff_base_url) == "api.anthropic.com")
    )

    if is_native_anthropic:
        return True, True
    if (is_openrouter or is_nous_portal) and is_claude:
        return True, False
    # Nous Portal Qwen (e.g. qwen3.6-plus) takes the same envelope-layout
    # cache_control path as Portal Claude. Portal proxies to OpenRouter
    # and the upstream Qwen route accepts cache_control markers; without
    # this branch the alibaba-family check below only matches
    # provider=opencode/alibaba and Portal traffic falls through to
    # (False, False), serving 0% cache hits and re-billing the full
    # prompt on every turn.
    if is_nous_portal and "qwen" in model_lower:
        return True, False
    if is_anthropic_wire and is_claude:
        # Third-party Anthropic-compatible gateway.
        return True, True

    # MiniMax on its Anthropic-compatible endpoint serves its own
    # model family (MiniMax-M2.7, M2.5, M2.1, M2) with documented
    # cache_control support (0.1× read pricing, 5-minute TTL).  The
    # blanket is_claude gate above excludes these — opt them in
    # explicitly via provider id or host match so users on
    # provider=minimax / minimax-cn (or custom endpoints pointing at
    # api.minimax.io/anthropic / api.minimaxi.com/anthropic) get the
    # same cost reduction as Claude traffic.
    # Docs: https://platform.minimax.io/docs/api-reference/anthropic-api-compatible-cache
    if is_anthropic_wire:
        is_minimax_provider = provider_lower in {"minimax", "minimax-cn"}
        is_minimax_host = (
            base_url_host_matches(eff_base_url, "api.minimax.io")
            or base_url_host_matches(eff_base_url, "api.minimaxi.com")
        )
        if is_minimax_provider or is_minimax_host:
            return True, True

    # Qwen/Alibaba on OpenCode (Zen/Go) and native DashScope: OpenAI-wire
    # transport that accepts Anthropic-style cache_control markers and
    # rewards them with real cache hits.  Without this branch
    # qwen3.6-plus on opencode-go reports 0% cached tokens and burns
    # through the subscription on every turn.
    model_is_qwen = "qwen" in model_lower
    provider_is_alibaba_family = provider_lower in {
        "opencode", "opencode-zen", "opencode-go", "alibaba",
    }
    if provider_is_alibaba_family and model_is_qwen:
        # Envelope layout (native_anthropic=False): markers on inner
        # content parts, not top-level tool messages.  Matches
        # pi-mono's "alibaba" cacheControlFormat.
        return True, False

    return False, False



def create_openai_client(agent, client_kwargs: dict, *, reason: str, shared: bool) -> Any:
    from agent.auxiliary_client import _validate_base_url, _validate_proxy_env_urls
    # 将 client_kwargs 视为只读参数。
    # 调用方会传入 agent._client_kwargs（或其浅拷贝）；
    # 任何原地修改（in-place mutation）都会污染保存的字典，
    # 并影响后续的请求。
    #
    # Issue #10933 就曾因此出过问题：
    # 当时注入了一个在首次请求后即被销毁的 httpx.Client 传输层（transport），
    # 导致下一次请求继续包裹这个已被关闭的传输层，
    # 从而在每次重试时都抛出 "Cannot send a request, as the client has been closed" 异常。
    #
    # 之前的代码撤回修复了那条特定的执行路径；
    # 此处的拷贝则锁定了规范，
    # 以确保未来的 transport/keepalive 相关开发不会再次引入同类 Bug。
    client_kwargs = dict(client_kwargs)
    _validate_proxy_env_urls()
    _validate_base_url(client_kwargs.get("base_url"))
    if agent.provider == "copilot-acp" or str(client_kwargs.get("base_url", "")).startswith("acp://copilot"):
        from agent.copilot_acp_client import CopilotACPClient

        client = CopilotACPClient(**client_kwargs)
        _ra().logger.info(
            "Copilot ACP client created (%s, shared=%s) %s",
            reason,
            shared,
            agent._client_log_context(),
        )
        return client
    if agent.provider == "google-gemini-cli" or str(client_kwargs.get("base_url", "")).startswith("cloudcode-pa://"):
        from agent.gemini_cloudcode_adapter import GeminiCloudCodeClient

        # Strip OpenAI-specific kwargs the Gemini client doesn't accept
        safe_kwargs = {
            k: v for k, v in client_kwargs.items()
            if k in {"api_key", "base_url", "default_headers", "project_id", "timeout"}
        }
        client = GeminiCloudCodeClient(**safe_kwargs)
        _ra().logger.info(
            "Gemini Cloud Code Assist client created (%s, shared=%s) %s",
            reason,
            shared,
            agent._client_log_context(),
        )
        return client
    if agent.provider == "gemini":
        from agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url

        base_url = str(client_kwargs.get("base_url", "") or "")
        if is_native_gemini_base_url(base_url):
            safe_kwargs = {
                k: v for k, v in client_kwargs.items()
                if k in {"api_key", "base_url", "default_headers", "timeout", "http_client"}
            }
            if "http_client" not in safe_kwargs:
                keepalive_http = agent._build_keepalive_http_client(base_url)
                if keepalive_http is not None:
                    safe_kwargs["http_client"] = keepalive_http
            client = GeminiNativeClient(**safe_kwargs)
            _ra().logger.info(
                "Gemini native client created (%s, shared=%s) %s",
                reason,
                shared,
                agent._client_log_context(),
            )
            return client
    # 注入 TCP 保活机制（keepalives），
    # 以便内核能够检测出已失效的服务商连接，
    # 而不是让它们默默地停留在 CLOSE-WAIT 状态（#10324）。
    # 如果没有这项配置，当对端在传输流途中断开时，
    # 套接字会处于 epoll_wait 永远不会触发的状态，
    # ``httpx`` 的读取超时也可能无法生效，
    # 最终导致 Agent 挂起，直至被手动杀死。
    # 设置为闲置 30 秒后开始探测，每 10 秒重试一次，尝试 3 次后放弃
    # → 即可在约 60 秒内检测到失效的对端。
    #
    # 针对 #10933 的安全防护：
    # 上文的 ``client_kwargs = dict(client_kwargs)``
    # 意味着此处注入的配置仅作用于单次调用的局部副本，
    # 绝不会写回 ``agent._client_kwargs`` 中。
    # 因此，每次调用 ``_create_openai_client`` 都会获得一个全新的、独立的 ``httpx.Client``，
    # 其生命周期与接收它的 OpenAI 客户端紧密绑定。
    # 当 OpenAI 客户端被关闭（如重新构建、销毁、凭证轮换）时，
    # 配对的 ``httpx.Client`` 也会随之关闭，
    # 随后的下一次调用则会创建一个全新的实例 — 绝不会复用已关闭的陈旧传输层。
    # ``tests/run_agent/test_create_openai_client_reuse.py`` 和
    # ``tests/run_agent/test_sequential_chats_live.py`` 中的测试固化了这一不变性要求。
    if "http_client" not in client_kwargs:
        keepalive_http = agent._build_keepalive_http_client(client_kwargs.get("base_url", ""))
        if keepalive_http is not None:
            client_kwargs["http_client"] = keepalive_http
    # Uses the module-level `OpenAI` name, resolved lazily on first
    # access via __getattr__ below. Tests patch via `run_agent.OpenAI`.
    client = _ra().OpenAI(**client_kwargs)
    _ra().logger.info(
        "OpenAI client created (%s, shared=%s) %s",
        reason,
        shared,
        agent._client_log_context(),
    )
    return client


def switch_model(agent, new_model, new_provider, api_key='', base_url='', api_mode=''):
    """Switch the model/provider in-place for a live agent.

    Called by the /model command handlers (CLI and gateway) after
    ``model_switch.switch_model()`` has resolved credentials and
    validated the model.  This method performs the actual runtime
    swap: rebuilding clients, updating caching flags, and refreshing
    the context compressor.

    The implementation mirrors ``_try_activate_fallback()`` for the
    client-swap logic but also updates ``_primary_runtime`` so the
    change persists across turns (unlike fallback which is
    turn-scoped).
    """
    from hermes_cli.providers import determine_api_mode

    # ── Determine api_mode if not provided ──
    if not api_mode:
        api_mode = determine_api_mode(new_provider, base_url)

    # Defense-in-depth: ensure OpenCode base_url doesn't carry a trailing
    # /v1 into the anthropic_messages client, which would cause the SDK to
    # hit /v1/v1/messages.  `model_switch.switch_model()` already strips
    # this, but we guard here so any direct callers (future code paths,
    # tests) can't reintroduce the double-/v1 404 bug.
    if (
        api_mode == "anthropic_messages"
        and new_provider in {"opencode-zen", "opencode-go"}
        and isinstance(base_url, str)
        and base_url
    ):
        base_url = re.sub(r"/v1/?$", "", base_url)

    old_model = agent.model
    old_provider = agent.provider

    # ── Snapshot all fields the swap+rebuild can mutate ──
    # If the rebuild raises (bad API key, network error, build_anthropic_client
    # failure, etc.) we restore these atomically so the agent isn't left with a
    # new model/provider name paired with the OLD client — that mismatch causes
    # HTTP 400s like "claude-sonnet-4-6 is not supported on openai-codex" on the
    # next turn.  Callers in cli.py / gateway/run.py / tui_gateway/server.py
    # catch the re-raised exception and show the user a warning; without this
    # rollback the warning is misleading because the swap partially succeeded.
    # Use a sentinel so we can distinguish "attribute was unset" from
    # "attribute was None" and skip the restore for genuinely-missing
    # attributes (tests construct bare agents via __new__ without all fields).
    _MISSING = object()
    _snapshot = {
        name: getattr(agent, name, _MISSING)
        for name in (
            "model",
            "provider",
            "base_url",
            "api_mode",
            "api_key",
            "client",
            "_anthropic_client",
            "_anthropic_api_key",
            "_anthropic_base_url",
            "_is_anthropic_oauth",
            "_config_context_length",
        )
    }
    # _client_kwargs is a dict — snapshot a shallow copy so mutating the
    # live dict doesn't poison the rollback target.
    _snapshot["_client_kwargs"] = dict(getattr(agent, "_client_kwargs", {}) or {})

    try:
        # Clear the per-config context_length override so the new model's
        # actual context window is resolved via get_model_context_length()
        # instead of inheriting the stale value from the previous model.
        agent._config_context_length = None

        # ── Swap core runtime fields ──
        agent.model = new_model
        agent.provider = new_provider
        # Use new base_url when provided; only fall back to current when the
        # new provider genuinely has no endpoint (e.g. native SDK providers).
        # Without this guard the old provider's URL (e.g. Ollama's localhost
        # address) would persist silently after switching to a cloud provider
        # that returns an empty base_url string.
        if base_url:
            agent.base_url = base_url
        agent.api_mode = api_mode
        # Invalidate transport cache — new api_mode may need a different transport
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        if api_key:
            agent.api_key = api_key

        # ── Build new client ──
        if api_mode == "anthropic_messages":
            from agent.anthropic_adapter import (
                build_anthropic_client,
                resolve_anthropic_token,
                _is_oauth_token,
            )
            # Only fall back to ANTHROPIC_TOKEN when the provider is actually Anthropic.
            # Other anthropic_messages providers (MiniMax, Alibaba, etc.) must use their own
            # API key — falling back would send Anthropic credentials to third-party endpoints.
            _is_native_anthropic = new_provider == "anthropic"
            effective_key = (api_key or agent.api_key or resolve_anthropic_token() or "") if _is_native_anthropic else (api_key or agent.api_key or "")

            # MiniMax OAuth: swap static string for a per-request callable token
            # provider so the rebuilt client survives 15-min token expiry. See
            # the matching block in agent_init.py for the full rationale.
            if new_provider == "minimax-oauth" and isinstance(effective_key, str) and effective_key:
                try:
                    from hermes_cli.auth import build_minimax_oauth_token_provider
                    effective_key = build_minimax_oauth_token_provider()
                except Exception as _mm_exc:  # noqa: BLE001
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "MiniMax OAuth: failed to install per-request token provider "
                        "on switch (%s); using static bearer.",
                        _mm_exc,
                    )

            agent.api_key = effective_key
            agent._anthropic_api_key = effective_key
            agent._anthropic_base_url = base_url or getattr(agent, "_anthropic_base_url", None)
            agent._anthropic_client = build_anthropic_client(
                effective_key, agent._anthropic_base_url,
                timeout=get_provider_request_timeout(agent.provider, agent.model),
            )
            agent._is_anthropic_oauth = _is_oauth_token(effective_key) if (_is_native_anthropic and isinstance(effective_key, str)) else False
            agent.client = None
            agent._client_kwargs = {}
        else:
            effective_key = api_key or agent.api_key
            effective_base = base_url or agent.base_url
            agent._client_kwargs = {
                "api_key": effective_key,
                "base_url": effective_base,
            }
            _sm_timeout = get_provider_request_timeout(agent.provider, agent.model)
            if _sm_timeout is not None:
                agent._client_kwargs["timeout"] = _sm_timeout
            agent.client = agent._create_openai_client(
                dict(agent._client_kwargs),
                reason="switch_model",
                shared=True,
            )
    except Exception:
        # Rollback every mutated field to the pre-swap snapshot so the agent
        # is left consistent (old model + old provider + old client) and the
        # caller's exception handler can surface a meaningful warning.  The
        # exception is re-raised; cli.py / gateway/run.py / tui_gateway catch
        # it and print "Agent swap failed; change applied to next session".
        for _name, _value in _snapshot.items():
            if _value is _MISSING:
                # Attribute did not exist before the swap — don't fabricate it.
                continue
            try:
                setattr(agent, _name, _value)
            except Exception:  # noqa: BLE001
                pass
        raise

    # ── Re-evaluate prompt caching ──
    agent._use_prompt_caching, agent._use_native_cache_layout = (
        agent._anthropic_prompt_cache_policy(
            provider=new_provider,
            base_url=agent.base_url,
            api_mode=api_mode,
            model=new_model,
        )
    )

    # ── LM Studio: preload before probing context length ──
    agent._ensure_lmstudio_runtime_loaded()

    # ── Update context compressor ──
    if hasattr(agent, "context_compressor") and agent.context_compressor:
        from agent.model_metadata import get_model_context_length
        # Re-read custom_providers from live config so per-model
        # context_length overrides are honored when switching to a
        # custom provider mid-session (closes #15779).
        _sm_custom_providers = None
        try:
            from hermes_cli.config import load_config, get_compatible_custom_providers
            _sm_cfg = load_config()
            _sm_custom_providers = get_compatible_custom_providers(_sm_cfg)
        except Exception:
            _sm_custom_providers = None
        # ``agent.api_key`` may be a callable (Azure Foundry Entra ID
        # token provider). ``get_model_context_length`` expects a
        # string for its live-probe paths; for Foundry the context
        # length normally resolves via config or static catalogs and
        # never hits a probe, but coerce to empty string defensively.
        _ctx_api_key = agent.api_key if isinstance(agent.api_key, str) else ""
        new_context_length = get_model_context_length(
            agent.model,
            base_url=agent.base_url,
            api_key=_ctx_api_key,
            provider=agent.provider,
            config_context_length=getattr(agent, "_config_context_length", None),
            custom_providers=_sm_custom_providers,
        )
        agent.context_compressor.update_model(
            model=agent.model,
            context_length=new_context_length,
            base_url=agent.base_url,
            api_key=agent.api_key,  # context_compressor forwards to call_llm; callable preserved
            provider=agent.provider,
            api_mode=agent.api_mode,
        )

    # ── Invalidate cached system prompt so it rebuilds next turn ──
    agent._cached_system_prompt = None

    # ── Update _primary_runtime so the change persists across turns ──
    _cc = agent.context_compressor if hasattr(agent, "context_compressor") and agent.context_compressor else None
    agent._primary_runtime = {
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        "api_mode": agent.api_mode,
        "api_key": getattr(agent, "api_key", ""),
        "client_kwargs": dict(agent._client_kwargs),
        "use_prompt_caching": agent._use_prompt_caching,
        "use_native_cache_layout": agent._use_native_cache_layout,
        "compressor_model": getattr(_cc, "model", agent.model) if _cc else agent.model,
        "compressor_base_url": getattr(_cc, "base_url", agent.base_url) if _cc else agent.base_url,
        "compressor_api_key": getattr(_cc, "api_key", "") if _cc else "",
        "compressor_provider": getattr(_cc, "provider", agent.provider) if _cc else agent.provider,
        "compressor_context_length": _cc.context_length if _cc else 0,
        "compressor_api_mode": getattr(_cc, "api_mode", agent.api_mode) if _cc else agent.api_mode,
        "compressor_threshold_tokens": _cc.threshold_tokens if _cc else 0,
    }
    if api_mode == "anthropic_messages":
        agent._primary_runtime.update({
            "anthropic_api_key": agent._anthropic_api_key,
            "anthropic_base_url": agent._anthropic_base_url,
            "is_anthropic_oauth": agent._is_anthropic_oauth,
        })

    # ── Reset fallback state ──
    agent._fallback_activated = False
    agent._fallback_index = 0

    # When the user deliberately swaps primary providers (e.g. openrouter
    # → anthropic), drop any fallback entries that target the OLD primary
    # or the NEW one.  The chain was seeded from config at agent init for
    # the original provider — without pruning, a failed turn on the new
    # primary silently re-activates the provider the user just rejected,
    # which is exactly what was reported during TUI v2 blitz testing
    # ("switched to anthropic, tui keeps trying openrouter").
    old_norm = (old_provider or "").strip().lower()
    new_norm = (new_provider or "").strip().lower()
    fallback_chain = list(getattr(agent, "_fallback_chain", []) or [])
    if old_norm and new_norm and old_norm != new_norm:
        fallback_chain = [
            entry for entry in fallback_chain
            if (entry.get("provider") or "").strip().lower() not in {old_norm, new_norm}
        ]
    agent._fallback_chain = fallback_chain
    agent._fallback_model = fallback_chain[0] if fallback_chain else None

    logger.info(
        "Model switched in-place: %s (%s) -> %s (%s)",
        old_model, old_provider, new_model, new_provider,
    )



def invoke_tool(agent, function_name: str, function_args: dict, effective_task_id: str,
                 tool_call_id: Optional[str] = None, messages: list = None,
                 pre_tool_block_checked: bool = False) -> str:
    """调用单个工具并返回结果字符串。不包含任何显示逻辑。

    同时处理 Agent 层级的工具（如 todo、memory 等）以及注册表分发的
    工具。由并发执行路径使用；顺序路径保留
    其现有的内联调用，以实现向下兼容的显示处理。
    """
    # Check plugin hooks for a block directive before executing anything.
    block_message: Optional[str] = None
    if not pre_tool_block_checked:
        try:
            from hermes_cli.plugins import get_pre_tool_call_block_message
            block_message = get_pre_tool_call_block_message(
                function_name, function_args, task_id=effective_task_id or "",
            )
        except Exception:
            pass
    if block_message is not None:
        return json.dumps({"error": block_message}, ensure_ascii=False)

    if function_name == "todo":
        from tools.todo_tool import todo_tool as _todo_tool
        return _todo_tool(
            todos=function_args.get("todos"),
            merge=function_args.get("merge", False),
            store=agent._todo_store,
        )
    elif function_name == "session_search":
        session_db = agent._get_session_db_for_recall()
        if not session_db:
            from hermes_state import format_session_db_unavailable
            return json.dumps({"success": False, "error": format_session_db_unavailable()})
        from tools.session_search_tool import session_search as _session_search
        return _session_search(
            query=function_args.get("query", ""),
            role_filter=function_args.get("role_filter"),
            limit=function_args.get("limit", 3),
            session_id=function_args.get("session_id"),
            around_message_id=function_args.get("around_message_id"),
            window=function_args.get("window", 5),
            sort=function_args.get("sort"),
            db=session_db,
            current_session_id=agent.session_id,
        )
    elif function_name == "memory":
        target = function_args.get("target", "memory")
        from tools.memory_tool import memory_tool as _memory_tool
        result = _memory_tool(
            action=function_args.get("action"),
            target=target,
            content=function_args.get("content"),
            old_text=function_args.get("old_text"),
            store=agent._memory_store,
        )
        # Bridge: notify external memory provider of built-in memory writes
        if agent._memory_manager and function_args.get("action") in {"add", "replace"}:
            try:
                agent._memory_manager.on_memory_write(
                    function_args.get("action", ""),
                    target,
                    function_args.get("content", ""),
                    metadata=agent._build_memory_write_metadata(
                        task_id=effective_task_id,
                        tool_call_id=tool_call_id,
                    ),
                )
            except Exception:
                pass
        return result
    elif agent._memory_manager and agent._memory_manager.has_tool(function_name):
        return agent._memory_manager.handle_tool_call(function_name, function_args)
    elif function_name == "clarify":
        from tools.clarify_tool import clarify_tool as _clarify_tool
        return _clarify_tool(
            question=function_args.get("question", ""),
            choices=function_args.get("choices"),
            callback=agent.clarify_callback,
        )
    elif function_name == "delegate_task":
        return agent._dispatch_delegate_task(function_args)
    else:
        return _ra().handle_function_call(
            function_name, function_args, effective_task_id,
            tool_call_id=tool_call_id,
            session_id=agent.session_id or "",
            enabled_tools=list(agent.valid_tool_names) if agent.valid_tool_names else None,
            skip_pre_tool_call_hook=True,
            enabled_toolsets=getattr(agent, "enabled_toolsets", None),
            disabled_toolsets=getattr(agent, "disabled_toolsets", None),
        )



def repair_tool_call(agent, tool_name: str) -> str | None:
    """在放弃之前，尝试修复不匹配的工具名称。

    模型有时会输出与工具名称仅在大小写、分隔符或类风格
    后缀上有所不同的变体。在回退到模糊匹配之前进行
    激进的标准化：

    1. 转小写直接匹配。
    2. 转小写 + 将连字符/空格转换为下划线。
    3. 驼峰命名（CamelCase） -> 蛇形命名（snake_case）(TodoTool -> todo_tool)。
    4. 剥离 Claude 风格模型有时会附加的末尾 ``_tool`` / ``-tool`` / ``tool``
       后缀 (TodoTool_tool -> TodoTool -> Todo -> todo)。此操作应用两次，
       以便像 ``TodoTool_tool`` 这样双重附加的后缀可以彻底简化。
    5. 模糊匹配 (difflib, 相似度阈值 cutoff=0.7)。

    参见 #14784 了解原始报告（在此之前，TodoTool_tool, Patch_tool,
    BrowserClick_tool 都会返回 "Unknown tool" 错误）。

    如果在 valid_tool_names 中找到，则返回修复后的名称，否则返回 None。
    """
    import re
    from difflib import get_close_matches

    if not tool_name:
        return None

    # VolcEngine api/plan 变通解决方案 (issue #33007)：该端点的
    # 协议转换层偶尔会将原始 XML 属性片段泄露到 tool_use.name 中，
    # 例如：
    #   `terminal" parameter="command" string="true`
    #   `execute_code" parameter="code" string="true`
    #   `session_search" parameter="session_id" string="true`
    # 我们在第一个明确的 XML/引号字符处进行截断，以便后续
    # 的修复流水线（小写化 / 蛇形命名 / 模糊匹配）能够
    # 将清洗后的名称解析为真实的工具。
    #
    # 至关重要的一点是，我们【绝对不能】按空格进行切分：像 "write file"
    # 这样合法的输入必须保持继续流经 ``_norm`` -> ``write_file``
    # （该逻辑由 tests/run_agent/test_repair_tool_call_name.py 中的
    # test_space_to_underscore 测试所覆盖）。


    def _norm(s: str) -> str:
        return s.lower().replace("-", "_").replace(" ", "_")

    def _camel_snake(s: str) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()

    def _strip_tool_suffix(s: str) -> str | None:
        lc = s.lower()
        for suffix in ("_tool", "-tool", "tool"):
            if lc.endswith(suffix):
                return s[: -len(suffix)].rstrip("_-")
        return None

    # Cheap fast-paths first — these cover the common case.
    lowered = tool_name.lower()
    if lowered in agent.valid_tool_names:
        return lowered
    normalized = _norm(tool_name)
    if normalized in agent.valid_tool_names:
        return normalized

    # Build the full candidate set for class-like emissions.
    cands: set[str] = {tool_name, lowered, normalized, _camel_snake(tool_name)}
    # Strip trailing tool-suffix up to twice — TodoTool_tool needs it.
    for _ in range(2):
        extra: set[str] = set()
        for c in cands:
            stripped = _strip_tool_suffix(c)
            if stripped:
                extra.add(stripped)
                extra.add(_norm(stripped))
                extra.add(_camel_snake(stripped))
        cands |= extra

    for c in cands:
        if c and c in agent.valid_tool_names:
            return c

    # Fuzzy match as last resort.
    matches = get_close_matches(lowered, agent.valid_tool_names, n=1, cutoff=0.7)
    if matches:
        return matches[0]

    return None



def sanitize_api_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix orphaned tool_call / tool_result pairs before every LLM call.

    Runs unconditionally — not gated on whether the context compressor
    is present — so orphans from session loading or manual message
    manipulation are always caught.
    """
    # --- Role allowlist: drop messages with roles the API won't accept ---
    filtered = []
    for msg in messages:
        role = msg.get("role")
        if role not in _ra().AIAgent._VALID_API_ROLES:
            _ra().logger.debug(
                "Pre-call sanitizer: dropping message with invalid role %r",
                role,
            )
            continue
        filtered.append(msg)
    messages = filtered

    surviving_call_ids: set = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                cid = _ra().AIAgent._get_tool_call_id_static(tc)
                if cid:
                    surviving_call_ids.add(cid)

    result_call_ids: set = set()
    for msg in messages:
        if msg.get("role") == "tool":
            cid = msg.get("tool_call_id")
            if cid:
                result_call_ids.add(cid)

    # 1. Drop tool results with no matching assistant call
    orphaned_results = result_call_ids - surviving_call_ids
    if orphaned_results:
        messages = [
            m for m in messages
            if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
        ]
        _ra().logger.debug(
            "Pre-call sanitizer: removed %d orphaned tool result(s)",
            len(orphaned_results),
        )

    # 2. Inject stub results for calls whose result was dropped
    missing_results = surviving_call_ids - result_call_ids
    if missing_results:
        patched: List[Dict[str, Any]] = []
        for msg in messages:
            patched.append(msg)
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = _ra().AIAgent._get_tool_call_id_static(tc)
                    if cid in missing_results:
                        patched.append({
                            "role": "tool",
                            "name": _ra().AIAgent._get_tool_call_name_static(tc),
                            "content": "[Result unavailable — see context summary above]",
                            "tool_call_id": cid,
                        })
        messages = patched
        _ra().logger.debug(
            "Pre-call sanitizer: added %d stub tool result(s)",
            len(missing_results),
        )
    return messages



def looks_like_codex_intermediate_ack(
    agent,
    user_message: str,
    assistant_content: str,
    messages: List[Dict[str, Any]],
) -> bool:
    """Detect a planning/ack message that should continue instead of ending the turn."""
    if any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in messages):
        return False

    assistant_text = agent._strip_think_blocks(assistant_content or "").strip().lower()
    if not assistant_text:
        return False
    if len(assistant_text) > 1200:
        return False

    has_future_ack = bool(
        re.search(r"\b(i['’]ll|i will|let me|i can do that|i can help with that)\b", assistant_text)
    )
    if not has_future_ack:
        return False

    action_markers = (
        "look into",
        "look at",
        "inspect",
        "scan",
        "check",
        "analyz",
        "review",
        "explore",
        "read",
        "open",
        "run",
        "test",
        "fix",
        "debug",
        "search",
        "find",
        "walkthrough",
        "report back",
        "summarize",
    )
    workspace_markers = (
        "directory",
        "current directory",
        "current dir",
        "cwd",
        "repo",
        "repository",
        "codebase",
        "project",
        "folder",
        "filesystem",
        "file tree",
        "files",
        "path",
    )

    user_text = (user_message or "").strip().lower()
    user_targets_workspace = (
        any(marker in user_text for marker in workspace_markers)
        or "~/" in user_text
        or "/" in user_text
    )
    assistant_mentions_action = any(marker in assistant_text for marker in action_markers)
    assistant_targets_workspace = any(
        marker in assistant_text for marker in workspace_markers
    )
    return (user_targets_workspace or assistant_targets_workspace) and assistant_mentions_action




def copy_reasoning_content_for_api(agent, source_msg: dict, api_msg: dict) -> None:
    """Copy provider-facing reasoning fields onto an API replay message."""
    if source_msg.get("role") != "assistant":
        return

    # 1. Explicit reasoning_content already set — preserve it verbatim
    # (includes DeepSeek/Kimi's own space-placeholder written at creation
    # time, and any valid reasoning content from the same provider).
    #
    # Exception: sessions persisted BEFORE #17341 have empty-string
    # placeholders pinned at creation time. DeepSeek V4 Pro rejects
    # those with HTTP 400. When the active provider enforces the
    # thinking-mode echo, upgrade "" → " " on replay so stale history
    # doesn't 400 the user on the next turn.
    existing = source_msg.get("reasoning_content")
    if isinstance(existing, str):
        if existing == "" and agent._needs_thinking_reasoning_pad():
            api_msg["reasoning_content"] = " "
        else:
            api_msg["reasoning_content"] = existing
        return
    needs_thinking_pad = agent._needs_thinking_reasoning_pad()
    # 2. 跨服务商污染的历史记录 (#15748)：在 DeepSeek/Kimi 上，
    # 如果源回合同时包含 tool_calls 和 'reasoning' 字段，但没有
    # 'reasoning_content' 键，则说明该 'reasoning' 文本是由之前的
    # 服务商（例如 MiniMax）写入的 —— 在此修复之后，DeepSeek 自身的
    # _build_assistant_message 会在创建时为工具调用回合固定填充
    # reasoning_content，因此在同服务商的 DeepSeek 历史记录中，是无法到达
    # 该形态的（即：设置了 reasoning、缺失 reasoning_content、存在 tool_calls）。
    # 此处注入一个单个空格以满足 API 的要求，同时避免将另一个服务商的思维链
    # 泄露给 DeepSeek/Kimi。使用空格（而非 ""）是因为 DeepSeek V4 Pro
    # 在思考模式下会拒绝空字符串形式的 reasoning_content（参见 #17341）。
    normalized_reasoning = source_msg.get("reasoning")
    if (
        needs_thinking_pad
        and source_msg.get("tool_calls")
        and isinstance(normalized_reasoning, str)
        and normalized_reasoning
    ):
        api_msg["reasoning_content"] = " "
        return

    # 3. 正常会话：针对使用内部 'reasoning' 键的服务商，
    # 将 'reasoning' 字段提升（promote）为 'reasoning_content'。
    # 此操作必须在无条件空字符串回退处理之前执行，以防止真实的推理内容
    # 被覆盖（PR #15478 中引入的 #15812 回归问题）。仅针对强制要求
    # 回显的服务商进行提升 —— 严格的服务商会拒绝该字段（参见 #45655）。
    if isinstance(normalized_reasoning, str) and normalized_reasoning:
        api_msg["reasoning_content"] = normalized_reasoning
        return

    # 4. DeepSeek / Kimi 思考模式：所有助手（assistant）消息都需要
    # reasoning_content。当没有显式的推理内容存在时，注入一个单个空格以满足
    # 服务商的要求。这同时涵盖了工具调用回合（完全没有推理内容的已被污染的历史记录）
    # 和纯文本回合。使用空格（而非 ""）是因为 DeepSeek V4 Pro 缩紧了校验规则，
    # 会拒绝空字符串并返回 HTTP 400 错误（“思考模式下的推理内容必须传递回
    # API”）。参见 #17341。
    if needs_thinking_pad:
        api_msg["reasoning_content"] = " "
        return

    # 5. reasoning_content was present but not a string (e.g. None after
    # context compaction).  Don't pass null to the API.
    api_msg.pop("reasoning_content", None)


def reapply_reasoning_echo_for_provider(agent, api_messages: list) -> int:
    """针对当前激活的服务商，重新填充（或清除）助手轮次（assistant turns）的 reasoning_content。

    ``api_messages`` 是在此重试循环之前、*主*服务商激活时构建一次的。
    对话中途的降级容灾（fallback）随后可能会切换服务商，因此固化在
    ``api_messages`` 中的推理字段是针对*前一个*服务商定制的，必须与*当前*服务商进行协调调和：

    * 切换到“有强制要求”的服务商（如 DeepSeek / Kimi / MiMo 的思考模式）：
      在前一个服务商不需要回显（echo-back）时构建的助手轮次，在发送时会缺少
      ``reasoning_content``，而新服务商会以 HTTP 400 拒绝它们（“思考模式下的
      reasoning_content 必须传回”）。此时需要重新应用填充。

    * 切换到“严格拒绝该字段”的服务商（如 Mistral、Cerebras、Groq、SambaNova 等）：
      在支持推理的主服务商下构建的助手轮次会携带 ``reasoning_content`` 填充（通常是
      一个空格 ``" "``），而严格的服务商会以 HTTP 400/422 拒绝它（“不允许有额外的输入”）。
      此时需要清除该字段。这正是 #45655 中出现的跨服务商降级 Bug —— DeepSeek 作为
      主服务商时用 ``" "`` 填充了历史记录，请求降级到 Mistral 后，Mistral 因这个过时的
      填充返回了 422 错误。

    在构建请求关键字参数（kwargs）之前立即调用此函数，可以使这些字段与*当前*服务商保持一致。
    该操作是幂等的，在每次循环迭代时调用都很安全；它涵盖了所有的降级路径。

    返回被添加或移除 reasoning_content 的助手轮次的总数。
    """
    if not agent._needs_thinking_reasoning_pad():
        return 0
    padded = 0
    for api_msg in api_messages:
        if api_msg.get("role") != "assistant":
            continue
        if api_msg.get("reasoning_content"):
            continue
        copy_reasoning_content_for_api(agent, api_msg, api_msg)
        if api_msg.get("reasoning_content"):
            padded += 1
    return padded


def _iter_pool_sockets(client: Any):
    """
    产出可从 OpenAI/httpx 客户端连接池中访问到的原始 socket。

    httpcore 1.x 会将具体的 HTTP11/HTTP2 连接存放在
    ``conn._connection`` 下；而更早的版本则直接在连接池条目上
    暴露 stream 属性。

    由于这些都属于私有的传输层内部实现，
    并且会随 httpx/httpcore 版本变化，
    因此遍历时需要保持防御性。
    """
    try:
        http_client = getattr(client, "_client", None)
        if http_client is None:
            return
        transport = getattr(http_client, "_transport", None)
        if transport is None:
            return
        pool = getattr(transport, "_pool", None)
        if pool is None:
            return
        connections = (
            getattr(pool, "_connections", None)
            or getattr(pool, "_pool", None)
            or []
        )
    except Exception:
        return

    seen: set[int] = set()
    for conn in list(connections):
        candidates = [conn]
        inner = getattr(conn, "_connection", None)
        if inner is not None:
            candidates.append(inner)
        for candidate in candidates:
            stream = (
                getattr(candidate, "_network_stream", None)
                or getattr(candidate, "_stream", None)
            )
            if stream is None:
                continue
            sock = getattr(stream, "_sock", None)
            if sock is None:
                get_extra_info = getattr(stream, "get_extra_info", None)
                if callable(get_extra_info):
                    try:
                        sock = get_extra_info("socket")
                    except Exception:
                        sock = None
            if sock is None:
                wrapped = getattr(stream, "stream", None)
                if wrapped is not None:
                    sock = getattr(wrapped, "_sock", None)
            if sock is None:
                # anyio-backed streams expose the raw socket through
                # SocketAttribute.raw_socket when available.
                wrapped = getattr(stream, "_stream", None)
                extra = getattr(wrapped, "extra", None)
                if callable(extra):
                    try:
                        from anyio.abc import SocketAttribute
                        sock = extra(SocketAttribute.raw_socket)
                    except Exception:
                        sock = None
            if sock is None:
                continue
            marker = id(sock)
            if marker in seen:
                continue
            seen.add(marker)
            yield sock


def cleanup_dead_connections(agent) -> bool:
    """Detect and clean up dead TCP connections on the primary client.

    Inspects the httpx connection pool for sockets in unhealthy states
    (CLOSE-WAIT, errors).  If any are found, force-closes all sockets
    and rebuilds the primary client from scratch.

    Returns True if dead connections were found and cleaned up.
    """
    client = getattr(agent, "client", None)
    if client is None:
        return False
    try:
        dead_count = 0
        for sock in _iter_pool_sockets(client):
            # Probe socket health with a non-blocking recv peek
            import socket as _socket
            try:
                sock.setblocking(False)
                data = sock.recv(1, _socket.MSG_PEEK | _socket.MSG_DONTWAIT)
                if data == b"":
                    dead_count += 1
            except BlockingIOError:
                pass  # No data available — socket is healthy
            except OSError:
                dead_count += 1
            finally:
                try:
                    sock.setblocking(True)
                except OSError:
                    pass
        if dead_count > 0:
            _ra().logger.warning(
                "Found %d dead connection(s) in client pool — rebuilding client",
                dead_count,
            )
            agent._replace_primary_openai_client(reason="dead_connection_cleanup")
            return True
    except Exception as exc:
        _ra().logger.debug("Dead connection check error: %s", exc)
    return False



def extract_api_error_context(error: Exception) -> Dict[str, Any]:
    """Extract structured rate-limit details from provider errors."""
    context: Dict[str, Any] = {}

    body = getattr(error, "body", None)
    payload = None
    if isinstance(body, dict):
        payload = body.get("error") if isinstance(body.get("error"), dict) else body
    if isinstance(payload, dict):
        reason = payload.get("code") or payload.get("type") or payload.get("error")
        if isinstance(reason, str) and reason.strip():
            context["reason"] = reason.strip()
        message = payload.get("message") or payload.get("error_description")
        if isinstance(message, str) and message.strip():
            context["message"] = message.strip()
        for key in ("resets_at", "reset_at"):
            value = payload.get(key)
            if value not in {None, ""}:
                context["reset_at"] = value
                break
        retry_after = payload.get("retry_after")
        if retry_after not in {None, ""} and "reset_at" not in context:
            try:
                context["reset_at"] = time.time() + float(retry_after)
            except (TypeError, ValueError):
                pass

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after and "reset_at" not in context:
            try:
                context["reset_at"] = time.time() + float(retry_after)
            except (TypeError, ValueError):
                pass
        ratelimit_reset = headers.get("x-ratelimit-reset")
        if ratelimit_reset and "reset_at" not in context:
            context["reset_at"] = ratelimit_reset

    if "message" not in context:
        raw_message = str(error).strip()
        if raw_message:
            context["message"] = raw_message[:500]

    if "reset_at" not in context:
        message = context.get("message") or ""
        if isinstance(message, str):
            delay_match = re.search(r"quotaResetDelay[:\s\"]+(\d+(?:\.\d+)?)(ms|s)", message, re.IGNORECASE)
            if delay_match:
                value = float(delay_match.group(1))
                seconds = value / 1000.0 if delay_match.group(2).lower() == "ms" else value
                context["reset_at"] = time.time() + seconds
            else:
                resets_in_match = re.search(
                    r"resets?\s+in\s+"
                    r"(?:(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b\s*)?"
                    r"(?:(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)\b\s*)?"
                    r"(?:(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b)?",
                    message,
                    re.IGNORECASE,
                )
                if resets_in_match and any(resets_in_match.groups()):
                    hours = float(resets_in_match.group(1) or 0)
                    minutes = float(resets_in_match.group(2) or 0)
                    seconds = float(resets_in_match.group(3) or 0)
                    context["reset_at"] = time.time() + (hours * 3600) + (minutes * 60) + seconds
                else:
                    sec_match = re.search(
                        r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)",
                        message,
                        re.IGNORECASE,
                    )
                    if sec_match:
                        context["reset_at"] = time.time() + float(sec_match.group(1))

    return context



def apply_pending_steer_to_tool_results(agent, messages: list, num_tool_msgs: int) -> None:
    """将任何待处理的 /steer 文本追加到本轮次的最后一个工具结果中。

    在工具调用批次结束时、下一次 API 调用之前被调用。
    Steer 文本会被追加到最后一个 ``role:"tool"`` 消息的内容中，
    并带有清晰的标记，以便模型明白它来自用户，
    而不是来自工具本身。角色交替关系（Role alternation）得以保留 —
    不会插入任何新消息，我们仅修改现有内容。

    参数：
        messages：正在运行的消息列表。
        num_tool_msgs：本批次中追加的工具结果数量；
            用于安全地定位尾部切片（tail slice）。
    """
    if num_tool_msgs <= 0 or not messages:
        return
    steer_text = agent._drain_pending_steer()
    if not steer_text:
        return
    # Find the last tool-role message in the recent tail. Skipping
    # non-tool messages defends against future code appending
    # something else at the boundary.
    target_idx = None
    for j in range(len(messages) - 1, max(len(messages) - num_tool_msgs - 1, -1), -1):
        msg = messages[j]
        if isinstance(msg, dict) and msg.get("role") == "tool":
            target_idx = j
            break
    if target_idx is None:
        # No tool result in this batch (e.g. all skipped by interrupt);
        # put the steer back so the caller's fallback path can deliver
        # it as a normal next-turn user message.
        _lock = getattr(agent, "_pending_steer_lock", None)
        if _lock is not None:
            with _lock:
                if agent._pending_steer:
                    agent._pending_steer = agent._pending_steer + "\n" + steer_text
                else:
                    agent._pending_steer = steer_text
        else:
            existing = getattr(agent, "_pending_steer", None)
            agent._pending_steer = (existing + "\n" + steer_text) if existing else steer_text
        return
    marker = f"\n\nUser guidance: {steer_text}"
    existing_content = messages[target_idx].get("content", "")
    if not isinstance(existing_content, str):
        # Anthropic multimodal content blocks — preserve them and append
        # a text block at the end.
        try:
            blocks = list(existing_content) if existing_content else []
            blocks.append({"type": "text", "text": marker.lstrip()})
            messages[target_idx]["content"] = blocks
        except Exception:
            # Fall back to string replacement if content shape is unexpected.
            messages[target_idx]["content"] = f"{existing_content}{marker}"
    else:
        messages[target_idx]["content"] = existing_content + marker
    _ra().logger.info(
        "Delivered /steer to agent after tool batch (%d chars): %s",
        len(steer_text),
        steer_text[:120] + ("..." if len(steer_text) > 120 else ""),
    )



def force_close_tcp_sockets(client: Any) -> int:
    """
    通过关闭 socket 的读写方向来中止正在进行的 TCP I/O，
    但不关闭文件描述符。

    当提供方在流式传输中途断开连接，或用户发出中断时，
    我们希望立即解除 httpx 读写器的阻塞，
    而不是等待内核的单连接超时。

    ``shutdown(SHUT_RDWR)`` 可以实现这一点：
    它会发送 FIN，使任何挂起的 ``recv`` / ``send``
    以 EOF 或 ``EPIPE`` 退出，但不会释放文件描述符。

    历史上，这个辅助函数也会调用 ``socket.close()``，
    因此 FD 会被立即释放。
    但当该辅助函数运行在驱动请求的线程之外时，
    这样做并不安全；中断中止路径和过期调用终止路径都属于这种情况：

      * 我们在这里关闭的 Python ``socket.socket``，
        与 httpx 连接池持有的是同一个对象；
        因此通过 Python 关闭它会把它的 ``_fd`` 置为 -1，
        后续在该 Python 对象上的操作会安全失败。

      * 但是 SSL 包装层
        （``ssl.SSLSocket`` 底层的 OpenSSL ``BIO``）
        会缓存原始的整数 FD。
        一旦执行 ``os.close(fd)``，
        内核就可能立刻把这个整数复用给下一次 ``open()`` 调用，
        例如 kanban 分发器打开 ``kanban.db``。

      * 随后，拥有该连接的工作线程开始展开 httpx；
        SSL 层会刷新一条挂起的 TLS 记录，
        加密后的字节就会被写入错误的文件。
        这就是 #29507 问题：
        一条 24 字节的 TLS 应用数据记录覆盖了
        SQLite 文件头的第 5..28 字节。

    修复方式是让拥有连接的线程负责关闭。
    从任意线程调用 ``shutdown()`` 都是 FD 安全的；
    ``close()`` 则不是。

    httpx 连接自身的关闭路径会在工作线程展开时运行，
    并通过同一个 ``socket.socket`` 对象释放 FD。
    由于 Python 的 socket close 会在发出 ``os.close`` 之前，
    先原子地把 ``_fd`` 交换为 -1，
    因此只要只有一个线程执行 close，
    就不会出现 FD 别名窗口。

    返回被 shutdown 的 socket 数量。
    日志行中该字段仍保留为 ``tcp_force_closed=N``，
    以兼容旧的解析逻辑。
    """
    import socket as _socket

    shutdown_count = 0
    try:
        for sock in _iter_pool_sockets(client):
            try:
                sock.shutdown(_socket.SHUT_RDWR)
            except OSError:
                # Already shut down / not connected / FD invalid — all benign.
                pass
            # IMPORTANT (#29507): do NOT call sock.close() here. See docstring.
            shutdown_count += 1
    except Exception as exc:
        _ra().logger.debug("Force-close TCP sockets sweep error: %s", exc)
    return shutdown_count



__all__ = [
    "convert_to_trajectory_format",
    "sanitize_tool_call_arguments",
    "repair_message_sequence",
    "strip_think_blocks",
    "recover_with_credential_pool",
    "try_recover_primary_transport",
    "drop_thinking_only_and_merge_users",
    "restore_primary_runtime",
    "extract_reasoning",
    "dump_api_request_debug",
    "anthropic_prompt_cache_policy",
    "create_openai_client",
    "switch_model",
    "invoke_tool",
    "repair_tool_call",
    "sanitize_api_messages",
    "looks_like_codex_intermediate_ack",
    "copy_reasoning_content_for_api",
    "cleanup_dead_connections",
    "extract_api_error_context",
    "apply_pending_steer_to_tool_results",
    "_iter_pool_sockets",
    "force_close_tcp_sockets",
]
