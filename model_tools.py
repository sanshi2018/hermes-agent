#!/usr/bin/env python3
"""
Model Tools Module

Thin orchestration layer over the tool registry. Each tool file in tools/
self-registers its schema, handler, and metadata via tools.registry.register().
This module triggers discovery (by importing all tool modules), then provides
the public API that run_agent.py, cli.py, batch_runner.py, and the RL
environments consume.

Public API (signatures preserved from the original 2,400-line version):
    get_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode) -> list
    handle_function_call(function_name, function_args, task_id, user_task) -> str
    TOOL_TO_TOOLSET_MAP: dict          (for batch_runner.py)
    TOOLSET_REQUIREMENTS: dict         (for cli.py, doctor.py)
    get_all_tool_names() -> list
    get_toolset_for_tool(name) -> str
    get_available_toolsets() -> dict
    check_toolset_requirements() -> dict
    check_tool_availability(quiet) -> tuple
"""

import os
import json
import re
import asyncio
import logging
import threading
import time
from typing import Dict, Any, List, Optional, Tuple

from tools.registry import discover_builtin_tools, registry
from toolsets import resolve_toolset, validate_toolset

logger = logging.getLogger(__name__)

# Tracks platform-bundle names already flagged in disabled_toolsets so the
# advisory (#33924) is logged once per name, not on every tool recompute.
_WARNED_DISABLED_BUNDLES: set = set()


# =============================================================================
# Async Bridging  (single source of truth -- used by registry.dispatch too)
# =============================================================================

_tool_loop = None          # persistent loop for the main (CLI) thread
_tool_loop_lock = threading.Lock()
_worker_thread_local = threading.local()  # per-worker-thread persistent loops


def _get_tool_loop():
    """Return a long-lived event loop for running async tool handlers.

    Using a persistent loop (instead of asyncio.run() which creates and
    *closes* a fresh loop every time) prevents "Event loop is closed"
    errors that occur when cached httpx/AsyncOpenAI clients attempt to
    close their transport on a dead loop during garbage collection.
    """
    global _tool_loop
    with _tool_loop_lock:
        if _tool_loop is None or _tool_loop.is_closed():
            _tool_loop = asyncio.new_event_loop()
        return _tool_loop


def _get_worker_loop():
    """Return a persistent event loop for the current worker thread.

    Each worker thread (e.g., delegate_task's ThreadPoolExecutor threads)
    gets its own long-lived loop stored in thread-local storage.  This
    prevents the "Event loop is closed" errors that occurred when
    asyncio.run() was used per-call: asyncio.run() creates a loop, runs
    the coroutine, then *closes* the loop — but cached httpx/AsyncOpenAI
    clients remain bound to that now-dead loop and raise RuntimeError
    during garbage collection or subsequent use.

    By keeping the loop alive for the thread's lifetime, cached clients
    stay valid and their cleanup runs on a live loop.
    """
    loop = getattr(_worker_thread_local, 'loop', None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _worker_thread_local.loop = loop
    return loop


def _run_async(coro):
    """Run an async coroutine from a sync context.

    If the current thread already has a running event loop (e.g., inside
    the gateway's async stack or Atropos's event loop), we spin up a
    disposable thread so asyncio.run() can create its own loop without
    conflicting.

    For the common CLI path (no running loop), we use a persistent event
    loop so that cached async clients (httpx / AsyncOpenAI) remain bound
    to a live loop and don't trigger "Event loop is closed" on GC.

    When called from a worker thread (parallel tool execution), we use a
    per-thread persistent loop to avoid both contention with the main
    thread's shared loop AND the "Event loop is closed" errors caused by
    asyncio.run()'s create-and-destroy lifecycle.

    This is the single source of truth for sync->async bridging in tool
    handlers. Each handler is self-protecting via this function.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Inside an async context (gateway, RL env) — run in a fresh thread
        # with its own event loop we own a reference to, so on timeout we
        # can cancel the task inside that loop (ThreadPoolExecutor.cancel()
        # only works on not-yet-started futures — it's a no-op on a running
        # worker, which previously leaked the thread on every 300 s timeout).
        import concurrent.futures

        worker_loop: Optional[asyncio.AbstractEventLoop] = None
        loop_ready = threading.Event()

        def _run_in_worker():
            nonlocal worker_loop
            worker_loop = asyncio.new_event_loop()
            loop_ready.set()
            try:
                asyncio.set_event_loop(worker_loop)
                return worker_loop.run_until_complete(coro)
            finally:
                try:
                    # Cancel anything still pending (e.g. task cancelled
                    # externally via call_soon_threadsafe on timeout).
                    pending = asyncio.all_tasks(worker_loop)
                    for t in pending:
                        t.cancel()
                    if pending:
                        worker_loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass
                worker_loop.close()

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # Carry the active profile + approval/sudo callbacks into the worker so
        # async tools resolve get_hermes_home() under the active profile.
        from tools.thread_context import propagate_context_to_thread

        future = pool.submit(propagate_context_to_thread(_run_in_worker))
        try:
            return future.result(timeout=300)
        except concurrent.futures.TimeoutError:
            # Cancel the coroutine inside its own loop so the worker thread
            # can wind down instead of running forever.
            if loop_ready.wait(timeout=1.0) and worker_loop is not None:
                try:
                    for t in asyncio.all_tasks(worker_loop):
                        worker_loop.call_soon_threadsafe(t.cancel)
                except RuntimeError:
                    # Loop already closed — nothing to cancel.
                    pass
            raise
        finally:
            # wait=False: don't block the caller on a stuck coroutine. We've
            # already requested cancellation above; the worker will exit
            # once the coroutine observes it (usually at the next await).
            pool.shutdown(wait=False)

    # If we're on a worker thread (e.g., parallel tool execution in
    # delegate_task), use a per-thread persistent loop.  This avoids
    # contention with the main thread's shared loop while keeping cached
    # httpx/AsyncOpenAI clients bound to a live loop for the thread's
    # lifetime — preventing "Event loop is closed" on GC cleanup.
    if threading.current_thread() is not threading.main_thread():
        worker_loop = _get_worker_loop()
        return worker_loop.run_until_complete(coro)

    tool_loop = _get_tool_loop()
    return tool_loop.run_until_complete(coro)


# =============================================================================
# Tool Discovery  (importing each module triggers its registry.register calls)
# =============================================================================

discover_builtin_tools()

# MCP tool discovery (external MCP servers from config) used to run here as
# a module-level side effect.  It was removed because discover_mcp_tools()
# internally uses a blocking future.result(timeout=120) wait, and the
# gateway lazy-imports this module from inside the asyncio event loop on
# the first user message — freezing Discord/Telegram heartbeats for up to
# 120s whenever any configured MCP server was slow or unreachable (#16856).
#
# Each entry point now runs discovery explicitly at its own startup:
#   - gateway/run.py            -> start_gateway() uses run_in_executor
#   - cli.py, hermes_cli/*      -> inline on startup (no event loop)
#   - tui_gateway/server.py     -> inline on startup (no event loop)
#   - acp_adapter/server.py     -> asyncio.to_thread on session init

# Plugin tool discovery (user/project/pip plugins)
try:
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
except Exception as e:
    logger.debug("Plugin discovery failed: %s", e)


# =============================================================================
# Backward-compat constants  (built once after discovery)
# =============================================================================

TOOL_TO_TOOLSET_MAP: Dict[str, str] = registry.get_tool_to_toolset_map()

TOOLSET_REQUIREMENTS: Dict[str, dict] = registry.get_toolset_requirements()

# Resolved tool names from the last get_tool_definitions() call.
# Used by code_execution_tool to know which tools are available in this session.
_last_resolved_tool_names: List[str] = []


# =============================================================================
# Legacy toolset name mapping  (old _tools-suffixed names -> tool name lists)
# =============================================================================

_LEGACY_TOOLSET_MAP = {
    "web_tools": ["web_search", "web_extract"],
    "terminal_tools": ["terminal"],
    "vision_tools": ["vision_analyze"],
    "image_tools": ["image_generate"],
    "skills_tools": ["skills_list", "skill_view", "skill_manage"],
    "browser_tools": [
        "browser_navigate", "browser_snapshot", "browser_click",
        "browser_type", "browser_scroll", "browser_back",
        "browser_press", "browser_get_images",
        "browser_vision", "browser_console"
    ],
    "cronjob_tools": ["cronjob"],
    "file_tools": ["read_file", "write_file", "patch", "search_files"],
    "tts_tools": ["text_to_speech"],
}


# =============================================================================
# get_tool_definitions  (the main schema provider)
# =============================================================================

# Module-level memoization for get_tool_definitions(). Keyed on
# (frozenset(enabled_toolsets), frozenset(disabled_toolsets), registry._generation).
# Hot callers (gateway runner, AIAgent.__init__) invoke this on every turn
# with quiet_mode=True; caching avoids ~7 ms of registry walking + schema
# filtering + check_fn probing per call. Only active when quiet_mode=True
# because quiet_mode=False has stdout side effects (tool-selection prints).
#
# Invalidation happens transparently via the registry's _generation counter,
# which bumps on register() / deregister() / register_toolset_alias(). The
# inner check_fn TTL cache in registry.py handles environment drift (Docker
# daemon start/stop, env var changes, etc.) on a 30 s horizon.
_tool_defs_cache: Dict[tuple, List[Dict[str, Any]]] = {}

# Hard cap on memoized get_tool_definitions() results. A long-lived Gateway
# process sees many distinct toolset/config fingerprints over its lifetime
# (per-session toolset sets, config edits, kanban-task toggles); without a
# bound the cache grows unboundedly. 8 comfortably covers the warm working
# set (the handful of distinct platform/toolset combos a gateway actually
# serves) while keeping the cap small. (#19251)
_TOOL_DEFS_CACHE_MAX = 8


def _clear_tool_defs_cache() -> None:
    """Drop memoized get_tool_definitions() results. Called when dynamic
    schema dependencies change (e.g. discord capability cache reset,
    execute_code sandbox reconfigured)."""
    _tool_defs_cache.clear()


def get_tool_definitions(
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
    quiet_mode: bool = False,
    skip_tool_search_assembly: bool = False,
) -> List[Dict[str, Any]]:
    """
    获取用于模型 API 调用的工具定义，并基于工具集（toolset）进行过滤。

    所有工具必须属于某个工具集才能被访问。

    参数：
        enabled_toolsets:仅包含来自这些工具集的工具。
        disabled_toolsets: 排除来自这些工具集的工具（在 enabled_toolsets 为 None 时生效）。
        quiet_mode: 静默模式，抑制状态打印。
        skip_tool_search_assembly: 为 True 时，返回组装前的原始工具列表
            （即每个已启用工具的原始 schema）。由
            tool_search / tool_describe 桥接句柄在内部使用，以便它们可以读取
            真实的目录，而不是已被折叠（collapsed）的目录。外部/公开调用者应
            保持其为 False。

    返回：
        过滤后的 OpenAI 格式工具定义列表。
    """
    # 快速路径：当调用方不需要 stdout 打印时的备忘化（memoized）结果。
    # 缓存键捕获了每个参数级别的输入；注册表的
    # 代际（generation）捕获了注册表的变更（MCP 刷新、插件加载）。
    # check_fn 的结果在下一层（即 registry.get_definitions 内部）按 TTL 进行缓存。
    # 下方的 config-mtime 指纹捕获了
    # 影响动态 schema 的用户可见配置修改（如 execute_code
    # 模式、discord 动作白名单等），而无需在每个配置写入方上附加显式的无效化钩子（invalidate hook）。
    if quiet_mode:
        try:
            from hermes_cli.config import get_config_path
            cfg_path = get_config_path()
            cfg_stat = cfg_path.stat()
            cfg_fp = (cfg_stat.st_mtime_ns, cfg_stat.st_size)
        except (FileNotFoundError, OSError, ImportError):
            cfg_fp = None
        cache_key = (
            frozenset(enabled_toolsets) if enabled_toolsets is not None else None,
            frozenset(disabled_toolsets) if disabled_toolsets else None,
            registry._generation,
            cfg_fp,
            bool(os.environ.get("HERMES_KANBAN_TASK")),
            bool(skip_tool_search_assembly),
        )
        cached = _tool_defs_cache.get(cache_key)
        if cached is not None:
            # Update _last_resolved_tool_names so downstream callers see
            # consistent state even on a cache hit.
            global _last_resolved_tool_names
            _last_resolved_tool_names = [t["function"]["name"] for t in cached]
            # Return a shallow copy of the list but share the dict references —
            # schemas are treated as read-only by all known callers.
            return list(cached)

    result = _compute_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode,
                                       skip_tool_search_assembly=skip_tool_search_assembly)
    if quiet_mode:
        # 缓存最新计算出来的列表，但向调用方提供一份浅拷贝（shallow copy），以使
        # 下游的修改（例如 run_agent 将 memory/LCM 工具的
        # schema 追加到 self.tools）不会污染缓存。如果不这样做，
        # 长期运行的 Gateway 进程会在多次 Agent 初始化过程中积累重复的工具名称，
        # 导致对工具名称唯一性有强制要求的提供商
        # （DeepSeek、小米 MiMo、月之暗面 Kimi）拒绝请求并返回
        # HTTP 400。此处逻辑与上方的缓存命中（cache-hit）路径保持一致。(issue #17335)
        # 通过 LRU 淘汰机制对缓存设定上限，防止长期运行的 Gateway 进程
        # 在其生命周期内遇到众多不同的
        # 工具集/配置指纹时，无节制地积累缓存条目 (#19251)。
        if len(_tool_defs_cache) >= _TOOL_DEFS_CACHE_MAX:
            _tool_defs_cache.pop(next(iter(_tool_defs_cache)))  # evict oldest
        _tool_defs_cache[cache_key] = result
        return list(result)
    return result


def _compute_tool_definitions(
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
    quiet_mode: bool = False,
    skip_tool_search_assembly: bool = False,
) -> List[Dict[str, Any]]:
    """:func:`get_tool_definitions` 的未缓存实现。"""
    # 确定调用方需要的工具名称
    tools_to_include: set = set()

    if enabled_toolsets is not None:
        effective_enabled_toolsets = list(enabled_toolsets)
        if os.environ.get("HERMES_KANBAN_TASK") and "kanban" not in effective_enabled_toolsets:
            # Dispatcher-spawned workers are scoped by HERMES_KANBAN_TASK and
            # must always receive the lifecycle handoff tools. Assignee
            # profiles may intentionally restrict their normal chat toolsets
            # (for token/cost reasons), but that should not strip the kanban
            # worker's completion/block/heartbeat surface.
            effective_enabled_toolsets.append("kanban")
        for toolset_name in effective_enabled_toolsets:
            if validate_toolset(toolset_name):
                resolved = resolve_toolset(toolset_name)
                tools_to_include.update(resolved)
                if not quiet_mode:
                    print(f"✅ Enabled toolset '{toolset_name}': {', '.join(resolved) if resolved else 'no tools'}")
            elif toolset_name in _LEGACY_TOOLSET_MAP:
                legacy_tools = _LEGACY_TOOLSET_MAP[toolset_name]
                tools_to_include.update(legacy_tools)
                if not quiet_mode:
                    print(f"✅ Enabled legacy toolset '{toolset_name}': {', '.join(legacy_tools)}")
            elif not quiet_mode:
                print(f"⚠️  Unknown toolset: {toolset_name}")
    else:
        # Default: start with everything
        from toolsets import get_all_toolsets
        for ts_name in get_all_toolsets():
            tools_to_include.update(resolve_toolset(ts_name))

    # 始终将禁用工具集（disabled toolsets）作为最后的扣除步骤来应用。
    # 这确保了即使启用了复合工具集（例如 hermes-cli），
    # 属于禁用工具集的任何工具也都会被严格剥离掉。
    # 参见 issue #17309。
    if disabled_toolsets:
        for toolset_name in disabled_toolsets:
            if validate_toolset(toolset_name):
                from toolsets import bundle_non_core_tools, get_toolset
                if toolset_name.startswith("hermes-") or (get_toolset(toolset_name) or {}).get("posture"):
                    # 平台包（hermes-*）包含 _HERMES_CORE_TOOLS，且
                    # 姿态工具集（`posture: True`，例如 `coding`）重新列出了
                    # 这些相同的核心工具但并不拥有它们，因此直接减去
                    # 整个工具集会剥离掉其他已启用工具集所共享的核心工具，
                    # 从而导致工具列表变空（#33924、#57315）。
                    # 仅减去非核心的增量部分（non-core delta）；保留核心工具。
                    to_remove = bundle_non_core_tools(toolset_name)
                    tools_to_include.difference_update(to_remove)
                    resolved = sorted(to_remove)
                    if (not quiet_mode and toolset_name.startswith("hermes-")
                            and toolset_name not in _WARNED_DISABLED_BUNDLES):
                        _WARNED_DISABLED_BUNDLES.add(toolset_name)
                        logger.info(
                            "agent.disabled_toolsets contains platform-bundle "
                            "name '%s'; core tools are preserved and only its "
                            "platform-specific tools (%s) are removed. Bundle "
                            "names usually belong in `toolsets:`, not "
                            "`disabled_toolsets` (#33924).",
                            toolset_name,
                            ", ".join(resolved) if resolved else "none",
                        )
                else:
                    resolved = resolve_toolset(toolset_name)
                    tools_to_include.difference_update(resolved)
                if not quiet_mode:
                    print(f"🚫 Disabled toolset '{toolset_name}': {', '.join(resolved) if resolved else 'no tools'}")
            elif toolset_name in _LEGACY_TOOLSET_MAP:
                legacy_tools = _LEGACY_TOOLSET_MAP[toolset_name]
                tools_to_include.difference_update(legacy_tools)
                if not quiet_mode:
                    print(f"🚫 Disabled legacy toolset '{toolset_name}': {', '.join(legacy_tools)}")
            elif not quiet_mode:
                print(f"⚠️  Unknown toolset: {toolset_name}")

    # 由插件注册的工具现在统一通过常规工具集
    # 路径进行解析 —— validate_toolset() / resolve_toolset() / get_all_toolsets()
    # 均会检查工具注册表中由插件提供的工具集。不再需要旁路（bypass）
    # 处理；插件会与其他工具集一样，严格遵循
    # enabled_toolsets / disabled_toolsets 的配置。

    # 向注册表请求 schema（仅返回通过 check_fn 检查的工具）
    filtered_tools = registry.get_definitions(tools_to_include, quiet=quiet_mode)

    # 实际通过 check_fn 过滤的工具名称集合。
    # 对于任何按名称引用其他工具的下游 schema，请使用此集合
    # （而非 tools_to_include）—— 否则模型会在描述中看到
    # 实际上并不存在的工具，并对其产生幻觉调用。
    available_tool_names = {t["function"]["name"] for t in filtered_tools}

    # 重构 execute_code schema，仅列出实际可用的
    # 沙箱工具。如果不这样做，即使 API 密钥未配置或工具集已被
    # 禁用，模型仍会看到“web_search 在 execute_code 中可用”
    # (#560-discord)。
    if "execute_code" in available_tool_names:
        from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS, build_execute_code_schema, _get_execution_mode
        sandbox_enabled = SANDBOX_ALLOWED_TOOLS & available_tool_names
        dynamic_schema = build_execute_code_schema(sandbox_enabled, mode=_get_execution_mode())
        for i, td in enumerate(filtered_tools):
            if td.get("function", {}).get("name") == "execute_code":
                filtered_tools[i] = {"type": "function", "function": dynamic_schema}
                break

    # 基于 Bot 的特权 Intent（通过 GET /applications/@me 检测）
    # 和配置中用户的动作白名单（action allowlist），重构 discord / discord_admin
    # 的 schema。隐藏 Bot Intent 不支持的动作，从而使模型
    # 永远不会尝试它们；并在缺少 MESSAGE_CONTENT Intent 时
    # 为 fetch_messages 添加标注。
    _discord_schema_fns = {
        "discord": "get_dynamic_schema_core",
        "discord_admin": "get_dynamic_schema_admin",
    }
    for discord_tool_name in _discord_schema_fns:
        if discord_tool_name in available_tool_names:
            try:
                from tools import discord_tool as _dt
                schema_fn = getattr(_dt, _discord_schema_fns[discord_tool_name])
                dynamic = schema_fn()
            except Exception:
                dynamic = None
            if dynamic is None:
                filtered_tools = [
                    t for t in filtered_tools
                    if t.get("function", {}).get("name") != discord_tool_name
                ]
                available_tool_names.discard(discord_tool_name)
            else:
                for i, td in enumerate(filtered_tools):
                    if td.get("function", {}).get("name") == discord_tool_name:
                        filtered_tools[i] = {"type": "function", "function": dynamic}
                        break

    # 当 web_search / web_extract 不可用时，从 browser_navigate
    # 描述中移除对 Web 工具的交叉引用（cross-references）。静态
    # schema 中写有“优先使用 web_search 或 web_extract”，这在这些工具
    # 缺失时会导致模型对它们产生幻觉。
    if "browser_navigate" in available_tool_names:
        web_tools_available = {"web_search", "web_extract"} & available_tool_names
        if not web_tools_available:
            for i, td in enumerate(filtered_tools):
                if td.get("function", {}).get("name") == "browser_navigate":
                    desc = td["function"].get("description", "")
                    desc = desc.replace(
                        " For simple information retrieval, prefer web_search or web_extract (faster, cheaper).",
                        "",
                    )
                    filtered_tools[i] = {
                        "type": "function",
                        "function": {**td["function"], "description": desc},
                    }
                    break

    if not quiet_mode:
        if filtered_tools:
            tool_names = [t["function"]["name"] for t in filtered_tools]
            print(f"🛠️  Final tool selection ({len(filtered_tools)} tools): {', '.join(tool_names)}")
        else:
            print("🛠️  No tools selected (all filtered out or unavailable)")

    global _last_resolved_tool_names
    _last_resolved_tool_names = [t["function"]["name"] for t in filtered_tools]

    # 清理 schema 以实现广泛的后端兼容性。llama.cpp 的
    # json-schema-to-grammar 转换器（其 OAI 服务用它来构建
    # GBNF 工具调用解析器）会拒绝某些云服务商会隐式接受的
    # 格式 —— 例如不带 properties 的裸 "type": "object"、
    # 来自异常 MCP 服务器的字符串值 schema 节点等。对于
    # 本身已经规范的 schema，此操作无任何副作用（no-op）。
    try:
        from tools.schema_sanitizer import sanitize_tool_schemas
        filtered_tools = sanitize_tool_schemas(filtered_tools)
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("Schema sanitization skipped: %s", e)

    # ── 工具搜索（渐进式披露 / progressive disclosure） ────────────────────────────
    # 当可延迟加载（deferrable）的工具规模超过配置的阈值（默认为上下文窗口的
    # 10%）时，有条件地将 MCP + 插件（非核心）工具替换为三个桥接
    # 工具（tool_search / tool_describe / tool_call）。Hermes 核心工具
    # （toolsets._HERMES_CORE_TOOLS）绝不会被延迟加载。完整设计说明
    # 请参见 tools/tool_search.py。
    #
    # 这特意安排在返回之前的最后一步 —— 模式清理（sanitization）
    # 已经对 schema 进行了规范化处理，且该组装过程具备幂等性（idempotent），
    # 以防某些调用方多次调用 get_tool_definitions。
    try:
        from tools.tool_search import assemble_tool_defs, load_config as _load_ts_config
        ts_cfg = _load_ts_config()
        if not skip_tool_search_assembly and ts_cfg.enabled != "off":
            context_length = _resolve_active_context_length()
            assembly = assemble_tool_defs(
                filtered_tools,
                context_length=context_length,
                config=ts_cfg,
            )
            if assembly.activated and not quiet_mode:
                print(
                    f"🔎 Tool Search: {assembly.deferred_count} MCP/plugin tools deferred "
                    f"(~{assembly.deferred_tokens} tokens) behind tool_search/describe/call. "
                    f"Threshold ~{assembly.threshold_tokens} tokens."
                )
            filtered_tools = assembly.tool_defs
    except Exception as e:  # pragma: no cover — never break tool loading
        logger.warning("Tool search assembly skipped: %s", e)

    return filtered_tools


def _resolve_active_context_length() -> int:
    """查找用于工具搜索门槛（tool-search gate）的当前模型上下文长度。

    当无法解析模型时返回 0 —— 在这种情况下，``should_activate``
    会回退到一个固定的 Token 截断值。
    """
    try:
        from hermes_cli.config import load_config as _load
        cfg = _load() or {}
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        if not isinstance(model_cfg, dict):
            model_cfg = {}
        model_id = (model_cfg.get("model") or model_cfg.get("default") or "").strip()
        if not model_id:
            return 0
        from agent.model_metadata import get_model_context_length
        # 优先使用 config.yaml 中显式指定的 `model.context_length` —
        # 在 get_model_context_length 的第 0 步直接短路跳过 OpenRouter /models 探针，
        # 这样非 OpenRouter 提供商就不会在每次 CLI 启动时白白耗费
        # 约 2-3 秒的 OpenRouter 获取时间。参见 issue #46620。
        raw_ctx = model_cfg.get("context_length")
        config_ctx = raw_ctx if isinstance(raw_ctx, int) and raw_ctx > 0 else None
        return int(get_model_context_length(model_id, config_context_length=config_ctx) or 0)
    except Exception as e:
        logger.debug("Could not resolve active context length: %s", e)
        return 0


# =============================================================================
# handle_function_call  (the main dispatcher)
# =============================================================================

# Tools whose execution is intercepted by the agent loop (run_agent.py)
# because they need agent-level state (TodoStore, MemoryStore, etc.).
# The registry still holds their schemas; dispatch just returns a stub error
# so if something slips through, the LLM sees a sensible message.
_AGENT_LOOP_TOOLS = {"todo", "memory", "session_search", "delegate_task"}
_READ_SEARCH_TOOLS = {"read_file", "search_files"}


# =========================================================================
# Tool error sanitization
# =========================================================================
#
# Tool exceptions can carry arbitrary text into the model's context as the
# `tool` message content. json.dumps() handles quote/backslash escaping so a
# raw injection of `</tool_call>` won't break message framing, but the model
# still *reads* those tokens and they can confuse downstream tool-call
# parsing or, in adversarial cases, nudge it toward role-confusion framing.
#
# This helper strips structural framing tokens (XML role tags, CDATA,
# markdown code fences) and caps the message at a sane upper bound before it
# becomes part of the conversation. It's defense-in-depth — the json layer
# already prevents framing escape — but cheap and worth having.
#
# Ported from ironclaw#1639.
_TOOL_ERROR_ROLE_TAG_RE = re.compile(
    r'</?(?:tool_call|function_call|result|response|output|input|system|assistant|user)>',
    re.IGNORECASE,
)
_TOOL_ERROR_FENCE_OPEN_RE = re.compile(r'^\s*```(?:json|xml|html|markdown)?\s*', re.MULTILINE)
_TOOL_ERROR_FENCE_CLOSE_RE = re.compile(r'\s*```\s*$', re.MULTILINE)
_TOOL_ERROR_CDATA_RE = re.compile(r'<!\[CDATA\[.*?\]\]>', re.DOTALL)
_TOOL_ERROR_MAX_LEN = 2000


def _sanitize_tool_error(error_msg: str) -> str:
    """Strip structural framing tokens from a tool error before showing it to the model.

    See _TOOL_ERROR_ROLE_TAG_RE docstring above for rationale.
    """
    if not error_msg:
        return "[TOOL_ERROR] "
    sanitized = _TOOL_ERROR_ROLE_TAG_RE.sub("", error_msg)
    sanitized = _TOOL_ERROR_FENCE_OPEN_RE.sub("", sanitized)
    sanitized = _TOOL_ERROR_FENCE_CLOSE_RE.sub("", sanitized)
    sanitized = _TOOL_ERROR_CDATA_RE.sub("", sanitized)
    if len(sanitized) > _TOOL_ERROR_MAX_LEN:
        sanitized = sanitized[:_TOOL_ERROR_MAX_LEN - 3] + "..."
    return f"[TOOL_ERROR] {sanitized}"


# =========================================================================
# Tool argument type coercion
# =========================================================================

def coerce_tool_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """强制转换工具调用参数以符合其 JSON 模式（schema）类型。

    大语言模型经常将数字作为字符串返回（如用 ``"42"`` 代替 ``42``），
    将布尔值作为字符串返回（如用 ``"true"`` 代替 ``true``）。本函数会将
    每个参数值与工具已注册的 JSON 模式进行比对，并在值为字符串但模式
    期望其他类型时尝试进行安全类型转换。转换失败时保留原始值。

    支持处理 ``"type": "integer"``、``"type": "number"``、``"type": "boolean"``
    以及联合类型（``"type": ["integer", "string"]``）。

    此外，当模式声明为 ``"type": "array"`` 时，本函数还会将单独的标量值
    包裹进单元素列表中。开源权重模型（DeepSeek、Qwen、GLM）
    在工具期望 ``{"urls": ["https://a.com"]}`` 时，有时会输出
    ``{"urls": "https://a.com"}``；在此处进行包裹处理可避免调用的格式
    原本符合预期却引发令人困惑的工具失败问题。
    """
    if not args or not isinstance(args, dict):
        return args

    schema = registry.get_schema(tool_name)
    if not schema:
        return args

    properties = (schema.get("parameters") or {}).get("properties")
    if not properties:
        return args

    for key, value in list(args.items()):
        prop_schema = properties.get(key)
        if not prop_schema:
            continue
        expected = prop_schema.get("type")

        # 当模式（schema）声明为 ``array`` 时包裹非列表的单独值。
        # 字符串仍会先经过 _coerce_value 处理，这样 JSON 编码的
        # 数组（``'["a","b"]'``）会被解析，且可空的 ``"null"``
        # 会变成 ``None`` 而不是 ``["null"]``。
        # ``None`` 本身会被保留 — 我们无法确定模型是想表达
        # “省略”还是“空列表”，且具有合理默认值的工具
        # （例如 read_file 的 normalize_read_pagination）自身已经处理了该情况。
        if expected == "array" and value is not None and not isinstance(value, (list, tuple)):
            if isinstance(value, str):
                coerced = _coerce_value(value, expected, schema=prop_schema)
                if coerced is not value:
                    # _coerce_value handled it (JSON-parsed list or
                    # nullable "null" → None).
                    args[key] = coerced
                    continue
                # If the string looks like a JSON array but _coerce_value
                # failed to parse it, warn clearly instead of silently wrapping.
                if value.strip().startswith("["):
                    logger.warning(
                        "coerce_tool_args: %s.%s looks like a JSON array string "
                        "but could not be parsed — model may have emitted a "
                        "JSON-encoded string instead of a native array. "
                        "Falling back to single-element list.",
                        tool_name, key,
                    )
                args[key] = [value]
                logger.info(
                    "coerce_tool_args: wrapped bare string in list for %s.%s",
                    tool_name, key,
                )
                continue
            args[key] = [value]
            logger.info(
                "coerce_tool_args: wrapped bare %s in list for %s.%s",
                type(value).__name__, tool_name, key,
            )
            continue

        if not isinstance(value, str):
            # 递归进入已经是原生容器的结构中，以便对 JSON 编码的
            # *元素*（数组项）和 *子字段*（嵌套对象属性）也进行规范化处理
            # — 例如当某个元素被作为 JSON 字符串输出时的 ``todos: ['{"id":...}']``
            # 或 ``tasks: [{"goal": "..."}]``。
            # 上方的顶层强制转换仅修复最外层的值。
            if expected == "array" and isinstance(value, (list, tuple)):
                args[key] = _normalize_json_strings_for_schema(value, prop_schema)
            elif expected == "object" and isinstance(value, dict):
                args[key] = _normalize_json_strings_for_schema(value, prop_schema)
            continue
        if not expected and not _schema_allows_null(prop_schema):
            continue
        coerced = _coerce_value(value, expected, schema=prop_schema)
        if coerced is not value:
            args[key] = coerced
            # If we just JSON-parsed a string into a container, recurse so
            # nested JSON-encoded elements/fields get normalized as well.
            if isinstance(coerced, (list, tuple, dict)):
                args[key] = _normalize_json_strings_for_schema(coerced, prop_schema)

    return args


def _schema_accepts_kind(schema: Any, kind: str) -> bool:
    """Return True when *schema* permits a value of JSON type *kind*.

    Looks at ``type`` (string or list) and recurses through
    ``anyOf``/``oneOf``/``allOf`` branches — matching the JSON-Schema shapes
    open-weight models emit against. ``kind`` is ``"array"`` or ``"object"``.
    """
    if not isinstance(schema, dict):
        return False
    t = schema.get("type")
    if t == kind or (isinstance(t, list) and kind in t):
        return True
    for union_key in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(union_key)
        if isinstance(branches, list) and any(
            _schema_accepts_kind(b, kind) for b in branches
        ):
            return True
    return False


def _normalize_json_strings_for_schema(value: Any, schema: Any) -> Any:
    """递归解析模式（schema）期望为数组或对象的 JSON 编码字符串值，
    包括嵌套的数组项和对象属性。

    开源权重模型（DeepSeek、Qwen、GLM 等）有时会把结构化字段 —
    或者结构化字段中的某个*元素* — 输出为 JSON 编码的字符串，
    而不是原生值。顶层的 :func:`coerce_tool_args` 流程只修复了最外层的值；
    本辅助函数则遍历树的其余部分，因此像::

        {"todos": ["{\\"id\\": \\"1\\", \\"content\\": \\"x\\"}"]}

    （其元素为 JSON 字符串的列表）以及嵌套对象的子字段等情况也能得到修复。
    解析是由模式引导的：只有当对应的模式位置实际上期望数组或对象时，
    字符串才会被解析，因此合法的、看起来像 JSON 的字符串字段（``type: string``）
    会被保留。

    移植自 cline/cline#11803，并适配了 hermes-agent 的强制转换层。
    当没有任何改变时返回原始值对象（保留标识一致性，
    以便调用方可以低成本地检测出空操作）。
    """
    if not isinstance(schema, dict):
        return value

    # Parse a JSON-encoded string into the container the schema expects.
    if isinstance(value, str):
        trimmed = value.strip()
        expects_array = _schema_accepts_kind(schema, "array")
        expects_object = _schema_accepts_kind(schema, "object")
        if (expects_array and trimmed.startswith("[")) or (
            expects_object and trimmed.startswith("{")
        ):
            try:
                parsed = json.loads(trimmed)
            except (ValueError, TypeError):
                return value
            if isinstance(parsed, list) and expects_array:
                value = parsed
            elif isinstance(parsed, dict) and expects_object:
                value = parsed
            else:
                return value
        else:
            return value

    # Recurse into list items using the ``items`` schema.
    if isinstance(value, list):
        items_schema = schema.get("items")
        if not isinstance(items_schema, dict):
            return value
        changed = False
        out = []
        for item in value:
            nxt = _normalize_json_strings_for_schema(item, items_schema)
            changed = changed or (nxt is not item)
            out.append(nxt)
        return out if changed else value

    # Recurse into object properties using each property's schema.
    if isinstance(value, dict):
        props = schema.get("properties")
        if not isinstance(props, dict):
            return value
        changed = False
        out = dict(value)
        for k, prop_schema in props.items():
            if k not in value or not isinstance(prop_schema, dict):
                continue
            nxt = _normalize_json_strings_for_schema(value[k], prop_schema)
            if nxt is not value[k]:
                out[k] = nxt
                changed = True
        return out if changed else value

    return value


def _coerce_value(value: str, expected_type, schema: dict | None = None):
    """Attempt to coerce a string *value* to *expected_type*.

    Returns the original string when coercion is not applicable or fails.
    """
    if _schema_allows_null(schema) and value.strip().lower() == "null":
        return None

    if isinstance(expected_type, list):
        # Union type — try each in order, return first successful coercion
        for t in expected_type:
            result = _coerce_value(value, t, schema=schema)
            if result is not value:
                return result
        return value

    if expected_type in {"integer", "number"}:
        return _coerce_number(value, integer_only=(expected_type == "integer"))
    if expected_type == "boolean":
        return _coerce_boolean(value)
    if expected_type == "array":
        return _coerce_json(value, list)
    if expected_type == "object":
        return _coerce_json(value, dict)
    if expected_type == "null" and value.strip().lower() == "null":
        return None
    return value


def _schema_allows_null(schema: dict | None) -> bool:
    """Return True when a JSON Schema fragment explicitly permits null."""
    if not isinstance(schema, dict):
        return False

    schema_type = schema.get("type")
    if schema_type == "null":
        return True
    if isinstance(schema_type, list) and "null" in schema_type:
        return True
    if schema.get("nullable") is True:
        return True

    for union_key in ("anyOf", "oneOf"):
        variants = schema.get(union_key)
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if isinstance(variant, dict) and variant.get("type") == "null":
                return True

    return False


def _coerce_json(value: str, expected_python_type: type):
    """Parse *value* as JSON when the schema expects an array or object.

    Handles model output drift where a complex oneOf/discriminated-union schema
    causes the LLM to emit the array/object as a JSON string instead of a native
    structure.  Returns the original string if parsing fails or yields the wrong
    Python type.
    """
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError) as exc:
        logger.warning(
            "coerce_tool_args: failed to parse string as JSON for expected type %s: %s",
            expected_python_type.__name__,
            exc,
        )
        return value
    if isinstance(parsed, expected_python_type):
        logger.debug(
            "coerce_tool_args: coerced string to %s via json.loads",
            expected_python_type.__name__,
        )
        return parsed
    logger.warning(
        "coerce_tool_args: JSON-parsed value is %s, expected %s — skipping coercion",
        type(parsed).__name__,
        expected_python_type.__name__,
    )
    return value


def _coerce_number(value: str, integer_only: bool = False):
    """Try to parse *value* as a number.  Returns original string on failure."""
    try:
        f = float(value)
    except (ValueError, OverflowError):
        return value
    # Guard against inf/nan — not JSON-serializable, keep original string
    if f != f or f == float("inf") or f == float("-inf"):
        return value
    # If it looks like an integer (no fractional part), return int
    if f == int(f):
        return int(f)
    if integer_only:
        # Schema wants an integer but value has decimals — keep as string
        return value
    return f


def _coerce_boolean(value: str):
    """Try to parse *value* as a boolean.  Returns original string on failure."""
    low = value.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    return value


def _tool_result_observer_fields(result: Any) -> tuple[str, Optional[str], Optional[str]]:
    try:
        parsed_result = json.loads(result) if isinstance(result, str) else result
        if isinstance(parsed_result, dict) and parsed_result.get("error"):
            return "error", "tool_error", str(parsed_result.get("error"))
    except Exception:
        pass
    return "ok", None, None


def _emit_post_tool_call_hook(
    *,
    function_name: str,
    function_args: Dict[str, Any],
    result: Any,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    api_request_id: Optional[str] = None,
    duration_ms: int = 0,
    status: Optional[str] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Emit the ``post_tool_call`` observer hook.

    No-ops cheaply when no plugin has registered for ``post_tool_call`` —
    the ``has_hook`` gate skips both the result-field derivation and the
    payload dispatch so the no-listener path costs one dict lookup.  When
    ``status`` is not supplied, the ok/error fields are derived from the
    result *after* the gate (parsing the result is only worth it when a
    listener will actually consume it).
    """
    try:
        from hermes_cli.plugins import has_hook, invoke_hook
        if not has_hook("post_tool_call"):
            return
        if status is None:
            status, error_type, error_message = _tool_result_observer_fields(result)
        invoke_hook(
            "post_tool_call",
            tool_name=function_name,
            args=function_args,
            result=result,
            task_id=task_id or "",
            session_id=session_id or "",
            tool_call_id=tool_call_id or "",
            turn_id=turn_id or "",
            api_request_id=api_request_id or "",
            duration_ms=duration_ms,
            status=status,
            error_type=error_type,
            error_message=error_message,
            middleware_trace=list(middleware_trace or []),
        )
    except Exception as _hook_err:
        logger.debug("post_tool_call hook error: %s", _hook_err)


def handle_function_call(
    function_name: str,
    function_args: Dict[str, Any],
    task_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    session_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    api_request_id: Optional[str] = None,
    user_task: Optional[str] = None,
    enabled_tools: Optional[List[str]] = None,
    skip_pre_tool_call_hook: bool = False,
    skip_tool_request_middleware: bool = False,
    tool_request_middleware_trace: Optional[List[Dict[str, Any]]] = None,
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
) -> str:
    """
    将调用路由至工具注册表的主函数调用调度器。

    参数：
        function_name: 要调用的函数名称。
        function_args: 函数的参数。
        task_id: 用于终端/浏览器会话隔离的唯一标识符。
        user_task: 用户的原始任务（用于 browser_snapshot 上下文）。
        enabled_tools: 为此会话启用的工具名称列表。当提供时，
                       execute_code 会使用此列表来决定要生成哪些沙盒
                       工具。为了向下兼容，可退回使用进程全局的
                       ``_last_resolved_tool_names``。
        enabled_toolsets: 会话启用的工具集。用于限定工具搜索（Tool Search）
                       桥接目录的作用域，使得 ``tool_search`` /
                       ``tool_describe`` / ``tool_call`` 仅能看到和调用
                       该会话实际被授予的工具。``None`` 表示
                       “无限制”（调用方限定为所有工具集），
                       与 ``get_tool_definitions`` 的语义保持一致。
        disabled_toolsets: 会话禁用的工具集，在限定桥接目录作用域时
                       作为减项应用。

    返回：
        JSON 字符串格式的函数结果。
    """
    # 强制将字符串参数转换为其模式（schema）声明的类型（例如 "42"→42）
    function_args = coerce_tool_args(function_name, function_args)
    if not isinstance(function_args, dict):
        function_args = {}
    _tool_middleware_trace = list(tool_request_middleware_trace or [])

    # ── 工具搜索（Tool Search）桥接调度 ──────────────────────────────
    # tool_search 和 tool_describe 是纯粹的目录读取操作 — 在内联中
    # 进行处理。tool_call 会解包为底层的实际工具，以便所有
    # 下游钩子（pre/post、编辑批准、安全护栏）看到的都是真实的
    # 工具名称，而非桥接工具。
    _ts_mod = None
    try:
        from tools import tool_search as _ts_mod  # noqa: F401
    except Exception:
        _ts_mod = None

    if _ts_mod is not None and _ts_mod.is_bridge_tool(function_name):
        try:
            # 使用 skip_tool_search_assembly=True，以便我们看到真实的目录，
            # 而非已经被收拢且仅含桥接工具的列表（否则桥接工具将只能搜索其自身）。
            #
            # 将目录作用域限定为会话的工具集，以便桥接工具只能露出并调用该会话实际被授予的工具。
            # 如果不这样做，被限制工具集的会话（子 agent、看板工作线程、精选网关会话）将能通过桥接工具看到并调用
            # 整个进程注册表中的所有工具。传入与组装该会话时完全相同的
            # enabled/disabled 工具集，可使延迟目录与该会话自身工具列表的
            # 可延迟子集保持一致，并能避免将作用域之外的工具
            # 污染到进程全局的 _last_resolved_tool_names 中。
            current_defs = get_tool_definitions(
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
                quiet_mode=True, skip_tool_search_assembly=True,
            ) or []
        except Exception:
            current_defs = []
        if function_name == _ts_mod.TOOL_SEARCH_NAME:
            return _ts_mod.dispatch_tool_search(function_args or {},
                                                current_tool_defs=current_defs)
        if function_name == _ts_mod.TOOL_DESCRIBE_NAME:
            return _ts_mod.dispatch_tool_describe(function_args or {},
                                                  current_tool_defs=current_defs)
        if function_name == _ts_mod.TOOL_CALL_NAME:
            underlying_name, underlying_args, err = _ts_mod.resolve_underlying_call(function_args or {})
            if err or not underlying_name:
                return json.dumps({"error": err or "tool_call could not be resolved"},
                                  ensure_ascii=False)
            # Defense in depth: the underlying tool MUST be in the session's
            # scoped deferrable catalog. resolve_underlying_call() only checks
            # that the name is deferrable in the global registry; this gate
            # additionally rejects any tool the session was not granted, so a
            # restricted session can never invoke an out-of-scope tool through
            # the bridge even if the catalog scoping above regressed.
            _scoped_deferrable = _ts_mod.scoped_deferrable_names(current_defs)
            if underlying_name not in _scoped_deferrable:
                return json.dumps({
                    "error": (
                        f"'{underlying_name}' is not available in this session. "
                        "Use tool_search to find tools you can call."
                    ),
                }, ensure_ascii=False)
            # Recurse with the underlying tool. All hooks fire against the
            # real tool name. The bridge is invisible to hooks by design.
            return handle_function_call(
                function_name=underlying_name,
                function_args=underlying_args,
                task_id=task_id,
                tool_call_id=tool_call_id,
                session_id=session_id,
                user_task=user_task,
                enabled_tools=enabled_tools,
                skip_pre_tool_call_hook=skip_pre_tool_call_hook,
                skip_tool_request_middleware=skip_tool_request_middleware,
                tool_request_middleware_trace=list(_tool_middleware_trace),
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
            )

    _tool_original_args = dict(function_args)
    if not skip_tool_request_middleware:
        try:
            from hermes_cli.middleware import apply_tool_request_middleware

            _tool_request_mw = apply_tool_request_middleware(
                function_name,
                function_args,
                task_id=task_id or "",
                session_id=session_id or "",
                tool_call_id=tool_call_id or "",
                turn_id=turn_id or "",
                api_request_id=api_request_id or "",
            )
            function_args = _tool_request_mw.payload
            _tool_original_args = _tool_request_mw.original_payload
            _tool_middleware_trace = _tool_request_mw.trace
        except Exception as _mw_err:
            logger.debug("tool_request middleware error: %s", _mw_err)

    try:
        if function_name in _AGENT_LOOP_TOOLS:
            return json.dumps({"error": f"{function_name} must be handled by the agent loop"})

        # Check plugin hooks for a block/approve directive (unless caller
        # already checked — e.g. run_agent._invoke_tool passes skip=True to
        # avoid double-firing the hook).
        #
        # Single-fire contract: pre_tool_call fires exactly once per tool
        # execution. resolve_pre_tool_block() internally calls
        # invoke_hook("pre_tool_call", ...) once and returns the block message
        # for a `block` directive OR for an `approve` directive whose human
        # gate denied/timed-out/errored (fail-closed). Observer plugins see
        # the hook on that same pass. When skip=True, the caller already
        # fired it — do nothing here.
        if not skip_pre_tool_call_hook:
            block_message: Optional[str] = None
            try:
                from hermes_cli.plugins import resolve_pre_tool_block
                block_message = resolve_pre_tool_block(
                    function_name,
                    function_args,
                    task_id=task_id or "",
                    session_id=session_id or "",
                    tool_call_id=tool_call_id or "",
                    turn_id=turn_id or "",
                    api_request_id=api_request_id or "",
                    middleware_trace=list(_tool_middleware_trace),
                )
            except Exception as _hook_err:
                logger.debug("pre_tool_call hook error: %s", _hook_err)

            if block_message is not None:
                result = json.dumps({"error": block_message}, ensure_ascii=False)
                _emit_post_tool_call_hook(
                    function_name=function_name,
                    function_args=function_args,
                    result=result,
                    task_id=task_id,
                    session_id=session_id,
                    tool_call_id=tool_call_id,
                    turn_id=turn_id,
                    api_request_id=api_request_id,
                    status="blocked",
                    error_type="plugin_block",
                    error_message=block_message,
                    middleware_trace=list(_tool_middleware_trace),
                )
                return result

        # ACP/Zed edit approval runs before any file mutation.  The requester
        # is bound via ContextVar only for ACP sessions, so CLI/gateway paths
        # are unaffected when it is unset.
        try:
            from acp_adapter.edit_approval import maybe_require_edit_approval

            edit_block_message = maybe_require_edit_approval(function_name, function_args)
            if edit_block_message is not None:
                return edit_block_message
        except Exception as _edit_approval_err:
            logger.debug("ACP edit approval guard error: %s", _edit_approval_err)
            if function_name in {"write_file", "patch"}:
                return json.dumps({"error": "Edit approval denied: approval guard failed"}, ensure_ascii=False)

        # Notify the read-loop tracker when a non-read/search tool runs,
        # so the *consecutive* counter resets (reads after other work are fine).
        if function_name not in _READ_SEARCH_TOOLS:
            try:
                from tools.file_tools import notify_other_tool_call
                notify_other_tool_call(task_id or "default")
            except Exception:
                pass  # file_tools may not be loaded yet

        # Measure tool dispatch latency so post_tool_call and
        # transform_tool_result hooks can observe per-tool duration.
        # Inspired by Claude Code 2.1.119, which added ``duration_ms`` to
        # PostToolUse hook inputs so plugin authors can build latency
        # dashboards, budget alerts, and regression canaries without having
        # to wrap every tool manually.  We use monotonic() so the value is
        # unaffected by wall-clock adjustments during the call.
        _dispatch_start = time.monotonic()
        _approval_tokens = None
        try:
            from tools.approval import (
                reset_current_observability_context,
                set_current_observability_context,
            )
            _approval_tokens = set_current_observability_context(
                turn_id=turn_id or "",
                tool_call_id=tool_call_id or "",
            )
        except Exception:
            reset_current_observability_context = None
        try:
            if function_name == "execute_code":
                # Prefer the caller-provided list so subagents can't overwrite
                # the parent's tool set via the process-global.
                sandbox_enabled = enabled_tools if enabled_tools is not None else _last_resolved_tool_names
                def _dispatch(next_args: Dict[str, Any]) -> Any:
                    return registry.dispatch(
                        function_name, next_args,
                        task_id=task_id,
                        session_id=session_id,
                        enabled_tools=sandbox_enabled,
                    )
            else:
                def _dispatch(next_args: Dict[str, Any]) -> Any:
                    return registry.dispatch(
                        function_name, next_args,
                        task_id=task_id,
                        session_id=session_id,
                        user_task=user_task,
                    )
            from hermes_cli.middleware import run_tool_execution_middleware

            result = run_tool_execution_middleware(
                function_name,
                function_args,
                _dispatch,
                original_args=_tool_original_args,
                task_id=task_id or "",
                session_id=session_id or "",
                tool_call_id=tool_call_id or "",
                turn_id=turn_id or "",
                api_request_id=api_request_id or "",
            )
        finally:
            if _approval_tokens is not None and reset_current_observability_context is not None:
                try:
                    reset_current_observability_context(_approval_tokens)
                except Exception:
                    pass
        duration_ms = int((time.monotonic() - _dispatch_start) * 1000)

        _emit_post_tool_call_hook(
            function_name=function_name,
            function_args=function_args,
            result=result,
            task_id=task_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
            duration_ms=duration_ms,
            middleware_trace=list(_tool_middleware_trace),
        )

        # Generic tool-result canonicalization seam: plugins receive the
        # final result string (JSON, usually) and may replace it by
        # returning a string from transform_tool_result. Runs after
        # post_tool_call (which stays observational) and before the result
        # is appended back into conversation context. Fail-open; the first
        # valid string return wins; non-string returns are ignored.
        # Gated on has_hook so the no-listener path skips both the result
        # field derivation and the payload dispatch.
        try:
            from hermes_cli.plugins import has_hook, invoke_hook
            if has_hook("transform_tool_result"):
                status, error_type, error_message = _tool_result_observer_fields(result)
                hook_results = invoke_hook(
                    "transform_tool_result",
                    tool_name=function_name,
                    args=function_args,
                    result=result,
                    task_id=task_id or "",
                    session_id=session_id or "",
                    tool_call_id=tool_call_id or "",
                    turn_id=turn_id or "",
                    api_request_id=api_request_id or "",
                    duration_ms=duration_ms,
                    status=status,
                    error_type=error_type,
                    error_message=error_message,
                )
                for hook_result in hook_results:
                    if isinstance(hook_result, str):
                        result = hook_result
                        break
        except Exception as _hook_err:
            logger.debug("transform_tool_result hook error: %s", _hook_err)

        return result

    except Exception as e:
        error_msg = f"Error executing {function_name}: {str(e)}"
        logger.exception(error_msg)
        return json.dumps({"error": _sanitize_tool_error(error_msg)}, ensure_ascii=False)


# =============================================================================
# Backward-compat wrapper functions
# =============================================================================

def get_all_tool_names() -> List[str]:
    """Return all registered tool names."""
    return registry.get_all_tool_names()


def get_toolset_for_tool(tool_name: str) -> Optional[str]:
    """Return the toolset a tool belongs to."""
    return registry.get_toolset_for_tool(tool_name)


def get_available_toolsets() -> Dict[str, dict]:
    """Return toolset availability info for UI display."""
    return registry.get_available_toolsets()


def check_toolset_requirements() -> Dict[str, bool]:
    """Return {toolset: available_bool} for every registered toolset."""
    return registry.check_toolset_requirements()


def check_tool_availability(quiet: bool = False) -> Tuple[List[str], List[dict]]:
    """Return (available_toolsets, unavailable_info)."""
    return registry.check_tool_availability(quiet=quiet)
