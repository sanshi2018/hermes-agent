"""Anthropic Messages API transport.

Delegates to the existing adapter functions in agent/anthropic_adapter.py.
This transport owns format conversion and normalization — NOT client lifecycle.
"""

from typing import Any, Dict, List, Optional

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse


class AnthropicTransport(ProviderTransport):
    """Transport for api_mode='anthropic_messages'.

    Wraps the existing functions in anthropic_adapter.py behind the
    ProviderTransport ABC.  Each method delegates — no logic is duplicated.
    """

    @property
    def api_mode(self) -> str:
        return "anthropic_messages"

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Convert OpenAI messages to Anthropic (system, messages) tuple.

        kwargs:
            base_url: Optional[str] — affects thinking signature handling.
        """
        from agent.anthropic_adapter import convert_messages_to_anthropic

        base_url = kwargs.get("base_url")
        return convert_messages_to_anthropic(messages, base_url=base_url)

    def convert_tools(self, tools: List[Dict[str, Any]]) -> Any:
        """Convert OpenAI tool schemas to Anthropic input_schema format."""
        from agent.anthropic_adapter import convert_tools_to_anthropic

        return convert_tools_to_anthropic(tools)

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params,
    ) -> Dict[str, Any]:
        """Build Anthropic messages.create() kwargs.

        Calls convert_messages and convert_tools internally.

        params (all optional):
            max_tokens: int
            reasoning_config: dict | None
            tool_choice: str | None
            is_oauth: bool
            preserve_dots: bool
            context_length: int | None
            base_url: str | None
            fast_mode: bool
            drop_context_1m_beta: bool
        """
        from agent.anthropic_adapter import build_anthropic_kwargs

        return build_anthropic_kwargs(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=params.get("max_tokens", 16384),
            reasoning_config=params.get("reasoning_config"),
            tool_choice=params.get("tool_choice"),
            is_oauth=params.get("is_oauth", False),
            preserve_dots=params.get("preserve_dots", False),
            context_length=params.get("context_length"),
            base_url=params.get("base_url"),
            fast_mode=params.get("fast_mode", False),
            drop_context_1m_beta=params.get("drop_context_1m_beta", False),
        )

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """将 Anthropic 响应规范化为 NormalizedResponse。

        解析内容块（text、thinking、tool_use），将 stop_reason
        映射为 OpenAI 的 finish_reason，并在 provider_data 中收集 reasoning_details。
        """
        import json
        from agent.anthropic_adapter import _to_plain_data, _sanitize_replay_block
        from agent.transports.types import ToolCall

        strip_tool_prefix = kwargs.get("strip_tool_prefix", False)
        _MCP_PREFIX = "mcp__"

        text_parts = []
        reasoning_parts = []
        reasoning_details = []
        tool_calls = []
        # 逐字记录、保留顺序的每轮对话中所有内容块的副本。
        # Anthropic 会根据在其位置*之前*的每轮对话内容对每个思考块（thinking block）进行签名；
        # 当一轮对话交织了思考和工具调用时（自适应/交织思考，Claude 4.6+），下面并行的
        # reasoning_details + tool_calls 列表会丢失这种跨类型的顺序。以错误的顺序回放
        # 最新助手消息（assistant message）会导致签名失效 -> 触发 HTTP 400 “最新助手
        # 消息中的 thinking ... 块不能被修改”。在此处保留准确的块序列，以便适配器能够
        # 原封不动地进行回放。参见 tests/agent/test_anthropic_thinking_block_order.py。
        # --------
        # Claude模型（特别是4.6
        # 及以上版本）在回复时，会把“思考过程”和“调用工具”的操作穿插在一起，并且系统会根据这些内容的先后顺序给思考过程“加密盖章”（签名）。
        # 如果我们图省事，把思考过程和工具调用拆开存到两个不同的列表里，就会打乱它们原本的排版顺序。
        # 顺序一旦乱了，下次我们再把这段历史对话发给官方服务器时，
        # 服务器会发现“印章”和内容顺序对不上，判定我们篡改了数据，然后直接报错拦截（报HTTP400错误）。
        # 所以，为了防止报错，代码里必须原汁原味、一字不落、按绝对的先后顺序把内容存下来，保证下次能按原样发回去。
        ordered_blocks = []

        for block in response.content:
            block_dict = _to_plain_data(block)
            clean_block = None
            if isinstance(block_dict, dict):
                # 在捕获时进行脱敏处理，确保仅作为输出的 SDK 字段（如 parsed_output、
                # caller、citations=None 等）永远不会被持久化到 state.db 中，并在回放时
                # 泄露回传为请求输入 → 从而导致 HTTP 400 "Extra inputs are not permitted"
                # （不允许有额外的输入）。这与回放端的脱敏处理共同构成了深度防御。
                clean_block = _sanitize_replay_block(block_dict)
                if clean_block is not None:
                    ordered_blocks.append(clean_block)
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type in ("thinking", "redacted_thinking"):
                if block.type == "thinking":
                    reasoning_parts.append(block.thinking)
                # Use the sanitized block (clean_block) for reasoning_details too,
                # since _extract_preserved_thinking_blocks replays these on the
                # non-ordered path. Falls back to raw only if sanitize dropped it.
                if isinstance(clean_block, dict):
                    reasoning_details.append(clean_block)
                elif isinstance(block_dict, dict):
                    reasoning_details.append(block_dict)
            elif block.type == "tool_use":
                name = block.name
                if strip_tool_prefix and name.startswith(_MCP_PREFIX):
                    # 在 OAuth 传输线上，每个工具都带有双下划线 ``mcp__`` 前缀
                    # （该前缀在 build_anthropic_kwargs 中被添加，以避开
                    # Anthropic 的单下划线第三方分类器）。
                    # 将其逆向恢复为注册表/调度器所知晓的工具名称。
                    # 有两种原始形式会被映射到同一个 ``mcp__`` 传输线名称上：
                    #   ``mcp__read_file``       <- 纯原生工具 ``read_file``
                    #   ``mcp__linear_get_issue`` <- MCP 服务器工具
                    #                                ``mcp_linear_get_issue``
                    # 通过注册表查找来解决冲突，优先选择实际已注册的那个原始名称；
                    # 绝不要重写大语言模型（LLM）使用的、且本身已经能够原生解析的名称。
                    # 参见 GH-25255。
                    from tools.registry import registry as _tool_registry
                    if not _tool_registry.get_entry(name):
                        bare = name[len(_MCP_PREFIX):]            # read_file
                        single = "mcp_" + bare                    # mcp_read_file / mcp_linear_get_issue
                        if _tool_registry.get_entry(single):
                            name = single
                        elif _tool_registry.get_entry(bare):
                            name = bare
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=name,
                        arguments=json.dumps(block.input),
                    )
                )

        finish_reason = self._STOP_REASON_MAP.get(response.stop_reason, "stop")

        provider_data = {}
        if reasoning_details:
            provider_data["reasoning_details"] = reasoning_details
        # Only worth carrying the ordered-blocks channel when the turn
        # actually interleaves signed thinking with tool_use — that's the
        # only shape the parallel lists reconstruct incorrectly. A turn that
        # is purely text, or thinking-then-tools with a single leading
        # thinking block, replays correctly without it.
        _has_signed_thinking = any(
            isinstance(b, dict)
            and b.get("type") in ("thinking", "redacted_thinking")
            and (b.get("signature") or b.get("data"))
            for b in ordered_blocks
        )
        _has_tool_use = any(
            isinstance(b, dict) and b.get("type") == "tool_use"
            for b in ordered_blocks
        )
        if _has_signed_thinking and _has_tool_use:
            provider_data["anthropic_content_blocks"] = ordered_blocks

        return NormalizedResponse(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            reasoning="\n\n".join(reasoning_parts) if reasoning_parts else None,
            usage=None,
            provider_data=provider_data or None,
        )

    def validate_response(self, response: Any) -> bool:
        """Check Anthropic response structure is valid.

        An empty content list is legitimate for terminal stop reasons that
        carry no text payload:

        - ``end_turn`` — the model's canonical "nothing more to add" after a
          tool turn that already delivered the user-facing text.
        - ``refusal`` — the model declined to respond (Claude 4.5+). The
          Messages API returns an empty ``content`` list with this stop
          reason. Treating it as invalid sends a deterministic refusal into
          the invalid-response retry loop, which reproduces the refusal on
          every attempt and surfaces a misleading "rate limited / invalid
          response" error instead of the refusal. ``normalize_response`` maps
          ``refusal`` → ``content_filter`` so the agent loop's refusal handler
          can surface it.

        Treating either as invalid falsely retries a completed response.
        """
        if response is None:
            return False
        content_blocks = getattr(response, "content", None)
        if not isinstance(content_blocks, list):
            return False
        if not content_blocks:
            return getattr(response, "stop_reason", None) in {"end_turn", "refusal"}
        return True

    def extract_cache_stats(self, response: Any) -> Optional[Dict[str, int]]:
        """Extract Anthropic cache_read and cache_creation token counts."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        written = getattr(usage, "cache_creation_input_tokens", 0) or 0
        if cached or written:
            return {"cached_tokens": cached, "creation_tokens": written}
        return None

    # Promote the adapter's canonical mapping to module level so it's shared
    _STOP_REASON_MAP = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "refusal": "content_filter",
        "model_context_window_exceeded": "length",
    }

    def map_finish_reason(self, raw_reason: str) -> str:
        """Map Anthropic stop_reason to OpenAI finish_reason."""
        return self._STOP_REASON_MAP.get(raw_reason, "stop")


# Auto-register on import
from agent.transports import register_transport  # noqa: E402

register_transport("anthropic_messages", AnthropicTransport)
