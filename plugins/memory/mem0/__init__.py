"""Mem0 内存插件 — MemoryProvider 接口。

通过 Mem0 Platform API（云端）或基于 Memory 的 OSS（自托管），
在服务端实现大模型事实提取、语义搜索以及自动去重功能。

原 PR #2933 由 kartik-mem0 提交，现已适配为 MemoryProvider 抽象基类（ABC）。

配置项
-------------
密钥（存放在 $HERMES_HOME/.env 或系统环境变量中）：
  MEM0_API_KEY       — Mem0 平台 API 密钥（平台模式必填）
  MEM0_HOST          — 自托管 Mem0 服务器的基础 URL。设置后，插件将直接
                       通过 HTTP（使用 X-API-Key 鉴权）与该服务器通信，
                       而非调用云端 API。

行为设置（存放在 $HERMES_HOME/mem0.json 中，通过 `hermes memory setup` 进行配置）：
  mode               — 后端模式："platform"（默认）或 "oss"
  host               — 自托管 Mem0 服务器 URL（备选方案：MEM0_HOST 环境变量）。
                       设置后，请求将路由至自托管的 HTTP 后端。
  user_id            — 规范化的用户标识符。设置后，该标识符将统一应用于
                       每一个网关（CLI、Telegram、Slack、Discord 等），
                       从而使同一个用户共享一套合并后的内存库。
                       若未设置，则默认使用网关原生 ID（如 Telegram 的数字 ID、
                       Discord 的 Snowflake ID）。
  agent_id           — Agent 标识符（默认值：hermes）

仍会读取对应的 MEM0_MODE / MEM0_USER_ID / MEM0_AGENT_ID 环境变量
作为向后兼容的备用方案，但 mem0.json 才是这些非密钥设置的标准存储位置。
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# 熔断器：在连续失败达到此次数后，
# 将暂停 API 调用 _BREAKER_COOLDOWN_SECS 秒，
# 以避免对已宕机的服务器进行频繁请求（导致雪崩）。
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120
_PREFETCH_WAIT_SECS = 3

_CLIENT_ERROR_TYPES = ("MemoryNotFoundError", "ValidationError")

# 当 MEM0_USER_ID 和网关原生 ID 都不可用时返回的哨兵值（Sentinel）。
# initialize() 会将其视为“操作员未配置 user_id”，
# 从而使设置向导生成的旧版 mem0.json 文件（历史上曾写入该特定占位符）
# 仍能允许网关原生 ID 通过，
# 而不是静默地用该占位符覆盖它们。
_DEFAULT_USER_ID = "hermes-user"


def _is_client_error(exc: Exception) -> bool:
    """True for user-caused errors (bad ID, not found) that should NOT trip circuit breaker."""
    etype = type(exc).__name__
    if etype in _CLIENT_ERROR_TYPES:
        return True
    err_str = str(exc).lower()
    return "404" in err_str or "not found" in err_str or "valid uuid" in err_str


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """从环境变量加载配置，并使用 $HERMES_HOME/mem0.json 中的值进行覆盖。

    环境变量提供默认值；若存在 mem0.json，其中的配置会覆盖单个对应的键。
    这可以避免当 JSON 文件存在，但缺少用户在 ``.env`` 中设置的字段（如 ``api_key``）时
    发生静默失败。
    """
    from hermes_constants import get_hermes_home

    config = {
        "mode": os.environ.get("MEM0_MODE", "platform"),
        "api_key": os.environ.get("MEM0_API_KEY", ""),
        "host": os.environ.get("MEM0_HOST", ""),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        "oss": {},
    }
    # 仅在操作员显式配置了 user_id（通过环境变量或 mem0.json）时才进行携带。
    # 缺少该键值会告知 initialize() 退而使用来自 kwargs 的网关原生 ID，
    # 而不是用占位符将其覆盖。
    env_user_id = os.environ.get("MEM0_USER_ID")
    if env_user_id:
        config["user_id"] = env_user_id

    config_path = get_hermes_home() / "mem0.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
# SEARCH_SCHEMA = {
#     "name": "mem0_search",
#     "description": (
#         "根据语义搜索用户的记忆；返回按相关性排序的事实。"
#         "在回答任何可能依赖于你对用户了解（偏好、事实、历史、人员、"
#         "项目、过去的决策）的问题之前，请使用此工具。"
#         "对于多部分或多跳（multi-hop）问题，请调用多次 —— "
#         "变换不同的措辞，并根据先前搜索到的结果展开进一步追问搜索；"
#         "仅进行一次搜索往往是不够的。"
#     ),
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "query": {"type": "string", "description": "搜索的内容。"},
#             "top_k": {"type": "integer", "description": "最大结果数量（默认值：10，最大值：50）。"},
#             "rerank": {"type": "boolean", "description": "对结果重新按相关性排序（默认值：false，仅限平台模式）。"},
#         },
#         "required": ["query"],
#     },
# }
#
# ADD_SCHEMA = {
#     "name": "mem0_add",
#     "description": (
#         "逐字保存关于用户的持久事实（不经过大模型额外提取）。"
#         "一旦用户说出了值得在未来的对话轮次中复用的持久偏好、更正、"
#         "决策或个人细节，请立即调用此工具 —— 不要等到被要求记住时才记录。"
#         "请跳过短暂的闲聊以及你已经保存过的事实。"
#     ),
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "content": {"type": "string", "description": "要保存的事实。"},
#         },
#         "required": ["content"],
#     },
# }
#
# UPDATE_SCHEMA = {
#     "name": "mem0_update",
#     "description": (
#         "通过 ID 替换已存在的记忆文本（ID 需从 mem0_search 的搜索结果中获取）。"
#         "当已存储的事实发生变更或存在错误时使用 —— "
#         "请在原位直接更正，而不是添加一条重复的记忆。"
#     ),
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "memory_id": {"type": "string", "description": "要更新的记忆 UUID。"},
#             "text": {"type": "string", "description": "新的文本内容。"},
#         },
#         "required": ["memory_id", "text"],
#     },
# }
#
# DELETE_SCHEMA = {
#     "name": "mem0_delete",
#     "description": (
#         "通过 ID 删除一条记忆（ID 需从 mem0_search 的搜索结果中获取）。"
#         "当存储的事实已过时或用户明确要求你忘记它时使用；"
#         "如果事实仅仅是发生了变更，请优先使用 mem0_update。"
#     ),
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "memory_id": {"type": "string", "description": "要删除的记忆 UUID。"},
#         },
#         "required": ["memory_id"],
#     },
# }
SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search the user's memories by meaning; returns facts ranked by "
        "relevance. Use this before answering any question that may depend on "
        "what you know about the user (preferences, facts, history, people, "
        "projects, past decisions). For multi-part or multi-hop questions, "
        "call it several times — vary the wording and run follow-up searches "
        "on what earlier results reveal; one search is rarely enough."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
            "rerank": {"type": "boolean", "description": "Rerank results for relevance (default: false, platform mode only)."},
        },
        "required": ["query"],
    },
}

ADD_SCHEMA = {
    "name": "mem0_add",
    "description": (
        "Store a durable fact about the user, verbatim (no LLM extraction). "
        "Call this the moment the user states a lasting preference, correction, "
        "decision, or personal detail worth recalling on future turns — don't "
        "wait to be asked to remember. Skip transient chit-chat and facts you've "
        "already stored."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to store."},
        },
        "required": ["content"],
    },
}

UPDATE_SCHEMA = {
    "name": "mem0_update",
    "description": (
        "Replace the text of an existing memory by its ID (take the ID from a "
        "mem0_search result). Use when a stored fact has changed "
        "or was wrong — correct it in place instead of adding a duplicate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory UUID to update."},
            "text": {"type": "string", "description": "New text content."},
        },
        "required": ["memory_id", "text"],
    },
}

DELETE_SCHEMA = {
    "name": "mem0_delete",
    "description": (
        "Delete a memory by its ID (take the ID from a mem0_search "
        "result). Use when a stored fact is obsolete or the user asks you to "
        "forget it; prefer mem0_update if the fact merely changed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory UUID to delete."},
        },
        "required": ["memory_id"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem0MemoryProvider(MemoryProvider):
    """Mem0 memory with server-side extraction and semantic search.

    Supports Platform API (cloud) and OSS (self-hosted) modes via MEM0_MODE.
    """

    def __init__(self):
        self._config = None
        self._backend = None
        self._mode = "platform"
        self._api_key = ""
        self._host = ""
        self._user_id = _DEFAULT_USER_ID
        self._agent_id = "hermes"
        self._rerank_default = False
        self._channel = "cli"  # gateway channel name (cli/telegram/discord/...)
        self._sync_thread = None
        self._prefetch_thread = None
        self._prefetch_query = ""
        self._prefetch_result = ""
        self._prefetch_done = False
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        self._breaker_lock = threading.Lock()
        self._sync_lock = threading.Lock()
        self._prefetch_lock = threading.Lock()
        self._atexit_registered = False

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        mode = cfg.get("mode", "platform")
        if mode == "oss":
            return bool(cfg.get("oss", {}).get("vector_store"))
        # Platform needs an api_key; self-hosted needs a host (api_key optional
        # when the server runs with AUTH_DISABLED).
        return bool(cfg.get("api_key") or cfg.get("host"))

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/mem0.json."""
        import json
        from pathlib import Path
        config_path = Path(hermes_home) / "mem0.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        from utils import atomic_json_write
        atomic_json_write(config_path, existing, mode=0o600)

    def get_config_schema(self):
        cfg = _load_config()
        mode = cfg.get("mode", "platform")
        api_key_required = mode != "oss"
        return [
            {"key": "api_key", "description": "Mem0 Platform API key", "secret": True, "required": api_key_required, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "host", "description": "Self-hosted Mem0 server URL (leave blank for cloud)", "required": False, "env_var": "MEM0_HOST"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "false", "choices": ["true", "false"]},
        ]

    def post_setup(self, hermes_home: str, config: dict) -> None:
        from ._setup import post_setup
        post_setup(hermes_home, config)

    def _create_backend(self):
        # 在任何后端导入 mem0 SDK 之前，根据需要对其进行延迟安装（Lazy-install）。
        # ensure() 会遵循 security.allow_lazy_installs（默认为 true）；
        # 在受限的 Docker 虚拟环境中，会将安装重定向至持久化目标位置。
        # 如果安装失败，程序将向下投射（fall through），
        # 使得后端内部的导入操作产生规范的错误，并在下方被捕获。
        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("memory.mem0", prompt=False)
        except ImportError:
            pass
        except Exception:
            pass
        try:
            if self._mode == "oss":
                from ._backend import OSSBackend
                return OSSBackend(self._config.get("oss", {}))
            if self._host:
                from ._backend import SelfHostedBackend
                return SelfHostedBackend(self._api_key, self._host)
            from ._backend import PlatformBackend
            return PlatformBackend(self._api_key)
        except Exception as e:
            logger.error("Mem0 backend failed to initialize (%s mode): %s", self._mode, e)
            self._init_error = str(e)
            return None

    def _is_breaker_open(self) -> bool:
        """Return True if the circuit breaker is tripped (too many failures)."""
        with self._breaker_lock:
            if self._consecutive_failures < _BREAKER_THRESHOLD:
                return False
            if time.monotonic() >= self._breaker_open_until:
                self._consecutive_failures = 0
                return False
            return True

    def _format_error(self, prefix: str, exc: Exception) -> str:
        msg = f"{prefix}: {exc}"
        if self._mode == "oss":
            err_str = str(exc).lower()
            if "connection" in err_str or "refused" in err_str or "timeout" in err_str:
                vs = self._config.get("oss", {}).get("vector_store", {})
                msg += f" (check that {vs.get('provider', 'vector store')} is running)"
        return msg

    def _record_success(self):
        with self._breaker_lock:
            self._consecutive_failures = 0

    def _record_failure(self):
        with self._breaker_lock:
            self._consecutive_failures += 1
            count = self._consecutive_failures
            if count >= _BREAKER_THRESHOLD:
                self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            else:
                count = 0
        if count >= _BREAKER_THRESHOLD:
            hint = ""
            if self._mode == "oss":
                vs = self._config.get("oss", {}).get("vector_store", {})
                provider = vs.get("provider", "unknown")
                hint = f" Check that your {provider} vector store is running and reachable."
            logger.warning(
                "Mem0 circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.%s",
                count, _BREAKER_COOLDOWN_SECS, hint,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._mode = self._config.get("mode", "platform")
        self._api_key = self._config.get("api_key", "")
        self._host = self._config.get("host", "")
        # user_id 的解析顺序：
        #   1. 操作员配置的 MEM0_USER_ID（通过环境变量或 $HERMES_HOME/mem0.json）——
        #      规范的主体（principal），应用于每一个网关，
        #      使得同一个用户共享一套合并后的内存库。
        #   2. 来自 kwargs 的网关原生 ID（Telegram 数字 ID、Discord Snowflake 等）——
        #      在未配置重写参数时，保留各平台间的数据隔离。
        #   3. 硬编码的备用值 _DEFAULT_USER_ID（无鉴权信息的 CLI）。
        # 文本字面量 _DEFAULT_USER_ID 会被视为未设置，
        # 因此使用推荐默认值运行过设置向导的用户，
        # 仍能获取网关原生 ID，而不会被静默地归类合并在一起。
        configured = self._config.get("user_id")
        if configured == _DEFAULT_USER_ID:
            configured = None
        self._user_id = configured or kwargs.get("user_id") or _DEFAULT_USER_ID
        self._agent_id = self._config.get("agent_id", "hermes")
        # 持久化的重排序（rerank）偏好设置（来自设置向导 / mem0.json）。
        # 当模型未明确传递 ``rerank`` 参数时，
        # 用作 mem0_search 的默认值（DEFAULT）；
        # 每次调用的参数仍具有最高优先级。
        # 此功能仅限平台（Platform）模式支持 ——
        # 其他后端会接收但忽略该标志。
        _rr = self._config.get("rerank", False)
        self._rerank_default = (
            _rr.lower() in ("true", "1", "yes") if isinstance(_rr, str) else bool(_rr)
        )
        self._channel = kwargs.get("platform") or "cli"
        self._backend = self._create_backend()
        if self._backend and not self._atexit_registered:
            atexit.register(self._shutdown_backend)
            self._atexit_registered = True

    def _read_filters(self) -> Dict[str, Any]:
        # 按照设计，仅作用于 user_id ——
        # 这样便能召回在该主体下，来自任何网关/Agent 的记忆。
        # 写入操作会附带 agent_id（以及 metadata.channel），
        # 以便在需要时，仍可在查询阶段按 Agent / Channel 进行过滤；
        # 读取操作则默认采用跨 Agent 的更大范围召回。
        return {"user_id": self._user_id}

    def _write_metadata(self) -> Dict[str, Any]:
        # 为每次写入添加网关渠道（channel）标签，
        # 这样仪表盘就能提供按渠道过滤的视图，
        # 同时又不会将身份标识与特定渠道强行绑定。
        return {"channel": self._channel} if self._channel else {}

    def system_prompt_block(self) -> str:
        # 镜像 _create_backend 中的优先级关系（oss > host > platform），
        # 从而确保标签始终能够准确命名实际运行的后端。
        # 如果在这里优先检查 ``host``，
        # 会将包含 ``oss``+``host`` 的配置误标记为自托管的 HTTP 模式，
        # 即使在实际路由中 OSS 拥有更高的优先级。
        if self._mode == "oss":
            mode_label = "OSS (self-hosted)"
        elif self._host:
            mode_label = "self-hosted (HTTP API)"
        else:
            mode_label = "platform (cloud API)"
        # Rerank is a Mem0 Platform feature only.
        rerank_note = " Rerank is available on search." if (self._mode == "platform" and not self._host) else ""
        # # Mem0 记忆机制
        # f"已激活。模式: {mode_label}。用户: {self._user_id}。\n"
        # "你对该用户在过去的对话中留有持久记忆。"
        # "在回答任何可能依赖于先前上下文（用户的偏好、事实、历史、人物、"
        # "项目或早先的决定）的问题之前，你应该调用 mem0_search —— 不要仅依赖于"
        # "当前的聊天窗口，也不要假设你没有任何记忆。\n"
        # "对于包含多个部分或多跳的问题，请使用不同的措辞/角度进行多次搜索，"
        # "并针对初次搜索结果中浮现的内容进行跟进搜索；仅靠一次搜索很少是足够的。"
        # "在回答之前，请持续搜索，直到你掌握了该问题所需的每一个事实。\n"
        # "工具：mem0_search 用于查找记忆，mem0_add 用于存储事实，"
        # f"mem0_update 和 mem0_delete 用于通过 ID 进行管理。{rerank_note}"
        return (
            "# Mem0 Memory\n"
            f"Active. Mode: {mode_label}. User: {self._user_id}.\n"
            "You have persistent memory of this user from past conversations. "
            "You should call mem0_search before answering anything that could depend "
            "on prior context (the user's preferences, facts, history, people, "
            "projects, or earlier decisions) — do not rely on the chat window "
            "alone, and do not assume you have no memory.\n"
            "For multi-part or multi-hop questions, run several searches with "
            "different wording/angles and follow-up searches on what the first "
            "results surface; one search is rarely enough. Keep searching until "
            "you have every fact the question needs before you answer.\n"
            "Tools: mem0_search to find memories, mem0_add to store facts, "
            f"mem0_update and mem0_delete to manage by ID.{rerank_note}"
        )

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._start_prefetch(message)

    def _consume_prefetch_result(self, query: str) -> str | None:
        with self._prefetch_lock:
            if self._prefetch_query != query or not self._prefetch_done:
                return None
            result = self._prefetch_result
            self._prefetch_result = ""
            self._prefetch_done = False
            return result

    def _start_prefetch(self, query: str) -> None:
        if not query or self._backend is None or self._is_breaker_open():
            return
        backend = self._backend
        with self._prefetch_lock:
            if self._prefetch_query == query:
                if self._prefetch_done:
                    return
                if self._prefetch_thread and self._prefetch_thread.is_alive():
                    return
            self._prefetch_query = query
            self._prefetch_result = ""
            self._prefetch_done = False

        def _run():
            body = ""
            try:
                results = backend.search(
                    query, filters=self._read_filters(), top_k=10, rerank=False,
                )
                lines = [r.get("memory", "") for r in (results or []) if r.get("memory")]
                if lines:
                    body = "## Mem0 Memory\n" + "\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 prefetch failed: %s", e)
            with self._prefetch_lock:
                if self._prefetch_query == query:
                    self._prefetch_result = body
                    self._prefetch_done = True

        t = threading.Thread(target=_run, daemon=True, name="mem0-prefetch")
        with self._prefetch_lock:
            self._prefetch_thread = t
        t.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall memories for the CURRENT question with a short hot-path wait."""
        cached = self._consume_prefetch_result(query)
        if cached is not None:
            return cached
        self._start_prefetch(query)
        with self._prefetch_lock:
            thread = self._prefetch_thread if self._prefetch_query == query else None
        if thread:
            thread.join(timeout=_PREFETCH_WAIT_SECS)
        cached = self._consume_prefetch_result(query)
        if cached is not None:
            return cached
        # Slow backend: skip injection; mem0_search tool remains the backstop.
        return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Send the turn to Mem0 for server-side fact extraction (non-blocking)."""
        if self._backend is None or self._is_breaker_open():
            return

        def _sync():
            backend = self._backend
            if backend is None:
                return
            try:
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                backend.add(
                    messages,
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    infer=True,
                    metadata=self._write_metadata(),
                )
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Mem0 sync failed: %s", e)

        with self._sync_lock:
            if self._sync_thread and self._sync_thread.is_alive():
                self._sync_thread.join(timeout=5.0)
            # If still alive after timeout, skip to avoid duplicate ingestion.
            if self._sync_thread and self._sync_thread.is_alive():
                return
            self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0-sync")
            self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, ADD_SCHEMA, UPDATE_SCHEMA, DELETE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._backend is None:
            err = getattr(self, "_init_error", "unknown error")
            hint = ""
            if self._mode == "oss":
                vs = self._config.get("oss", {}).get("vector_store", {})
                provider = vs.get("provider", "vector store")
                hint = f" Check that {provider} is running and reachable."
            return json.dumps({"error": f"Mem0 backend not initialized: {err}.{hint}"})

        if self._is_breaker_open():
            msg = "Mem0 temporarily unavailable (multiple consecutive failures). Will retry automatically."
            if self._mode == "oss":
                vs = self._config.get("oss", {}).get("vector_store", {})
                msg += f" Check that your {vs.get('provider', 'vector store')} is running."
            return json.dumps({"error": msg})

        if tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            try:
                top_k = max(1, min(int(args.get("top_k", 10)), 50))
                rerank_raw = args.get("rerank", getattr(self, "_rerank_default", False))
                if isinstance(rerank_raw, str):
                    rerank = rerank_raw.lower() not in ("false", "0", "no")
                else:
                    rerank = bool(rerank_raw)
                results = self._backend.search(query, filters=self._read_filters(), top_k=top_k, rerank=rerank)
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [{"id": r.get("id"), "memory": r.get("memory", ""),
                          "score": r.get("score", 0)} for r in results]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                if not _is_client_error(e):
                    self._record_failure()
                return tool_error(self._format_error("Search failed", e))

        elif tool_name == "mem0_add":
            content = args.get("content", "")
            if not content:
                return tool_error("Missing required parameter: content")
            try:
                result = self._backend.add(
                    [{"role": "user", "content": content}],
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    infer=False,
                    metadata=self._write_metadata(),
                )
                self._record_success()
                event_id = result.get("event_id") if isinstance(result, dict) else None
                # Cloud add is async (server-side extraction); OSS and self-hosted store synchronously.
                msg = "Fact stored." if (self._mode == "oss" or self._host) else "Fact queued for storage."
                return json.dumps({"result": msg, "event_id": event_id})
            except Exception as e:
                self._record_failure()
                return tool_error(self._format_error("Failed to store", e))

        elif tool_name == "mem0_update":
            memory_id = args.get("memory_id", "")
            text = args.get("text", "")
            if not memory_id:
                return tool_error("Missing required parameter: memory_id")
            if not text:
                return tool_error("Missing required parameter: text")
            try:
                result = self._backend.update(memory_id, text)
                self._record_success()
                return json.dumps(result)
            except Exception as e:
                if _is_client_error(e):
                    return tool_error(f"Memory not found: {memory_id}")
                self._record_failure()
                return tool_error(self._format_error("Update failed", e))

        elif tool_name == "mem0_delete":
            memory_id = args.get("memory_id", "")
            if not memory_id:
                return tool_error("Missing required parameter: memory_id")
            try:
                result = self._backend.delete(memory_id)
                self._record_success()
                return json.dumps(result)
            except Exception as e:
                if _is_client_error(e):
                    return tool_error(f"Memory not found: {memory_id}")
                self._record_failure()
                return tool_error(self._format_error("Delete failed", e))

        return tool_error(f"Unknown tool: {tool_name}")

    def _shutdown_backend(self):
        try:
            if self._backend:
                self._backend.close()
                self._backend = None
        except Exception:
            pass

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        self._shutdown_backend()


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())
