"""Anthropic prompt caching strategy.

Single layout: ``system_and_3``. 4 cache_control breakpoints — system
prompt + last 3 non-system messages, all at the same TTL (5m or 1h).
Reduces input token costs by ~75% on multi-turn conversations within a
single session.

Pure functions -- no class state, no AIAgent dependency.
"""

import copy
from typing import Any, Dict, List


def _apply_cache_marker(msg: dict, cache_marker: dict, native_anthropic: bool = False) -> None:
    """Add cache_control to a single message, handling all format variations."""
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool" and native_anthropic:
        # Native Anthropic layout: top-level marker; the adapter moves it
        # inside the tool_result block.
        msg["cache_control"] = cache_marker
        return

    if content is None or content == "":
        if role == "tool" and not native_anthropic:
            # OpenRouter 会拒绝在 role:tool 消息的顶层使用 cache_control（会导致静默挂起），
            # 且空消息没有可以携带该标记的 content 部分 —— 因此跳过。
            # 非空的工具内容会落入下方的处理逻辑，并将标记放置在 content 部分中，
            # 这是 OpenRouter 所支持的。
            return
        if role == "assistant" and not native_anthropic:
            # Empty assistant turns are pure tool_calls. A top-level marker
            # here is ignored on the envelope layout, so skip.
            return
        msg["cache_control"] = cache_marker
        return

    if isinstance(content, str):
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": cache_marker}
        ]
        return

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = cache_marker


def _can_carry_marker(msg: dict, native_anthropic: bool) -> bool:
    """如果该消息上的标记确实能被服务商识别并遵循，则返回 True。

    在原生的 Anthropic 布局中，每条消息都有效（顶层标记会被适配器重新定位）。
    但在信封（envelope）布局中（如 OpenRouter 等），只有放置在 content 部分内部的
    标记才会被识别：空内容的消息（例如纯 tool_calls 的助手回合）以及空的工具消息
    会接收到一个被服务商忽略的顶层标记 —— 从而白白浪费了四个断点中的一个。
    跳过这些消息，以便让断点落在真正起作用的消息上。
    """
    if native_anthropic:
        return True
    content = msg.get("content")
    if content is None or content == "":
        return False
    if isinstance(content, list):
        # _apply_cache_marker 仅对“最后一个” content 部分进行标记，所以载体
        # 断言（predicate）必须与其保持一致：如果一个列表的最后一个元素不是字典（dict），
        # 它实际上就无法接收到标记，从而会浪费一个断点。此处需要镜像复现
        # _apply_cache_marker 中针对 `content` 真值以及“最后一个元素是否为字典”的检查逻辑。
        return bool(content) and isinstance(content[-1], dict)
    return isinstance(content, str)


def _build_marker(ttl: str) -> Dict[str, str]:
    """Build a cache_control marker dict for the given TTL ('5m' or '1h')."""
    marker: Dict[str, str] = {"type": "ephemeral"}
    if ttl == "1h":
        marker["ttl"] = "1h"
    return marker


def apply_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
) -> List[Dict[str, Any]]:
    """针对 Anthropic 模型将 system_and_3 缓存策略应用于消息列表。

    最多放置 4 个 cache_control 断点：系统提示词 + 最后的 3 条非系统消息，
    它们全部具有相同的生存时间（TTL）。

    返回值:
        注入了 cache_control 断点后的消息列表的深拷贝（Deep copy）。
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    marker = _build_marker(cache_ttl)

    breakpoints_used = 0

    if messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
        breakpoints_used += 1

    remaining = 4 - breakpoints_used
    non_sys = [
        i
        for i in range(len(messages))
        if messages[i].get("role") != "system"
        and _can_carry_marker(messages[i], native_anthropic=native_anthropic)
    ]
    for idx in non_sys[-remaining:]:
        _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages
