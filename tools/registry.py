"""hermes-agent 所有工具的中央注册表。

每个工具文件都会在模块层级调用 ``registry.register()``，
以声明其 Schema、处理程序（handler）、所属工具集以及可用性检查机制。
``model_tools.py`` 会查询此注册表，
而不是去维护一套自己平行的独立数据结构。

导入链（无循环导入风险）：
    tools/registry.py  （不导入 model_tools 或任何工具文件）
           ^
    tools/*.py  （在模块层级从 tools.registry 导入）
           ^
    model_tools.py  （导入 tools.registry 及所有工具模块）
           ^
    run_agent.py, cli.py, batch_runner.py 等
"""

import ast
import functools
import importlib
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)

# Cap on a tool error body; only trims runaway interpolated exceptions (static msgs are ~115 chars).
_MAX_TOOL_ERROR_CHARS = 2048
_TOOL_ERROR_TRUNCATION_MARKER = "… [truncated]"
# Logs keep more of the body than the model sees, but still a bounded amount.
_MAX_LOGGED_ERROR_CHARS = 8192


def _bound_error_text(text: str) -> str:
    """Bound an error body destined for model context; logs keep a longer prefix."""
    if len(text) <= _MAX_TOOL_ERROR_CHARS:
        return text
    logger.debug(
        "tool error body truncated for context (%d chars): %s",
        len(text),
        text[:_MAX_LOGGED_ERROR_CHARS],
    )
    return text[:_MAX_TOOL_ERROR_CHARS] + _TOOL_ERROR_TRUNCATION_MARKER


def _bound_json_error_result(result: str) -> str:
    """Trim an oversized ``error`` field in a JSON string result.

    Handlers that serialize exceptions directly — ``json.dumps({"error":
    str(exc), ...})`` instead of ``tool_error()`` — bypass the cap in
    ``tool_error``. Applied at the dispatch boundary so no registered tool
    can return an unbounded error body that stacks across retries.
    """
    if len(result) <= _MAX_TOOL_ERROR_CHARS or '"error"' not in result:
        return result
    try:
        payload = json.loads(result)
    except ValueError:
        return result
    if not isinstance(payload, dict):
        return result
    error = payload.get("error")
    if not isinstance(error, str) or len(error) <= _MAX_TOOL_ERROR_CHARS:
        return result
    payload["error"] = _bound_error_text(error)
    return json.dumps(payload, ensure_ascii=False)


def _is_registry_register_call(node: ast.AST) -> bool:
    """Return True when *node* is a ``registry.register(...)`` call expression."""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "register"
        and isinstance(func.value, ast.Name)
        and func.value.id == "registry"
    )


def _module_registers_tools(module_path: Path) -> bool:
    """Return True when the module contains a top-level ``registry.register(...)`` call.

    Only inspects module-body statements so that helper modules which happen
    to call ``registry.register()`` inside a function are not picked up.

    A cheap text prefilter avoids the ``ast.parse`` cost for files that do not
    mention both ``registry`` and ``register`` — a necessary condition for a
    top-level ``registry.register()`` call to exist.
    """
    try:
        source = module_path.read_text(encoding="utf-8")
    except OSError:
        return False
    if "registry" not in source or "register" not in source:
        return False
    try:
        tree = ast.parse(source, filename=str(module_path))
    except SyntaxError:
        return False

    return any(_is_registry_register_call(stmt) for stmt in tree.body)


def discover_builtin_tools(tools_dir: Optional[Path] = None) -> List[str]:
    """Import built-in self-registering tool modules and return their module names.

    The per-file AST scan (:func:`_module_registers_tools`) costs ~145 ms over
    ~100 files on a warm cache, so verdicts are memoized on disk keyed by
    ``(mtime_ns, size)``. A file whose mtime_ns+size match the cached entry is
    trusted without re-reading; any mismatch (or a corrupt/missing cache file)
    falls back to a fresh scan for that file. The cache write is best-effort
    and atomic, so concurrent processes can race harmlessly.
    """
    tools_path = Path(tools_dir) if tools_dir is not None else Path(__file__).resolve().parent

    cache = _load_discovery_cache()
    fresh_cache: Dict[str, list] = {}
    cache_dirty = False

    module_names: List[str] = []
    for path in sorted(tools_path.glob("*.py")):
        if path.name in {"__init__.py", "registry.py", "mcp_tool.py"}:
            continue
        abs_path = str(path.resolve())
        try:
            st = path.stat()
            stat_key = (st.st_mtime_ns, st.st_size)
        except OSError:
            continue
        cached = cache.get(abs_path)
        if (
            isinstance(cached, (list, tuple))
            and len(cached) == 3
            and (cached[0], cached[1]) == stat_key
        ):
            registers = bool(cached[2])
        else:
            registers = _module_registers_tools(path)
            cache_dirty = True
        fresh_cache[abs_path] = [stat_key[0], stat_key[1], registers]
        if registers:
            module_names.append(f"tools.{path.stem}")

    # Drop entries for files that no longer exist; rewrite only when changed.
    if cache_dirty or set(fresh_cache) != set(cache):
        _save_discovery_cache(fresh_cache)

    imported: List[str] = []
    for mod_name in module_names:
        try:
            importlib.import_module(mod_name)
            imported.append(mod_name)
        except Exception as e:
            logger.warning("Could not import tool module %s: %s", mod_name, e)
    return imported


def _discovery_cache_path() -> Optional[Path]:
    """Path of the tool-discovery verdict cache, or None if unresolvable."""
    try:
        # Deferred import keeps tools/registry.py a no-deps leaf at module
        # import time (hermes_constants itself is stdlib-only, so no cycle).
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()) / "cache" / "tool_discovery_cache.json"
    except Exception:
        return None


def _load_discovery_cache() -> Dict[str, list]:
    """Read the discovery cache; any error → empty dict (full scan)."""
    path = _discovery_cache_path()
    if path is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_discovery_cache(cache: Dict[str, list]) -> None:
    """Best-effort atomic write of the discovery cache. Never raises."""
    path = _discovery_cache_path()
    if path is None:
        return
    try:
        from utils import atomic_json_write  # stdlib+yaml only; no cycle

        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, cache, indent=0)
    except Exception as e:
        logger.debug("Could not write tool discovery cache %s: %s", path, e)


class ToolEntry:
    """Metadata for a single registered tool."""

    __slots__ = (
        "name", "toolset", "schema", "handler", "check_fn",
        "requires_env", "is_async", "description", "emoji",
        "max_result_size_chars", "dynamic_schema_overrides",
    )

    def __init__(self, name, toolset, schema, handler, check_fn,
                 requires_env, is_async, description, emoji,
                 max_result_size_chars=None, dynamic_schema_overrides=None):
        self.name = name
        self.toolset = toolset
        self.schema = schema
        self.handler = handler
        self.check_fn = check_fn
        self.requires_env = requires_env
        self.is_async = is_async
        self.description = description
        self.emoji = emoji
        self.max_result_size_chars = max_result_size_chars
        # 可选的无参可调用对象（callable），
        # 返回在调用 get_definitions() 时应用的 Schema 覆盖字典。
        #
        # 适用于依赖运行时配置的字段
        # （例如 delegate_task 的描述必须反映用户当前设置的
        # delegation.max_concurrent_children / max_spawn_depth，
        # 以避免模型获取到错误的限制信息）。
        #
        # 该可调用对象会在每次调用 get_definitions() 时被触发；
        # 其返回的结果会在包装成 {"type": "function", ...} 之前，
        # 浅合并（shallow merge）到基础 Schema 之上。
        self.dynamic_schema_overrides = dynamic_schema_overrides


class _PluginOverridePolicy:
    """Identity-bearing authorization record for one plugin generation."""

    __slots__ = ("allowed",)

    def __init__(self, allowed: bool) -> None:
        self.allowed = bool(allowed)


# ---------------------------------------------------------------------------
# check_fn TTL 缓存
#
# 诸如 tools/terminal_tool.check_terminal_requirements 之类的 check_fn 可调用对象
# 会探测外部状态（例如 Docker 守护进程、Modal SDK 安装情况、playwright 二进制文件
# 可用性）。对于长生命周期的 CLI 或网关进程，在每次调用 get_definitions() 时都进行检查
# 纯属浪费——外部状态通常按人类的时间尺度发生变化。将其结果缓存约 30 秒，这样通过 ``hermes tools``
# 进行的环境变量切换或实时凭据文件更改，就能在一两个轮次内生效，而无需显式使缓存失效。
#
# 瞬时故障抑制（issue #21658 / #5304）：这些探测可能会出现波动。
# 负载下超时的单次 ``subprocess.run([docker, "version"], timeout=5)`` 调用会返回 False，
# 这会默默地从当时正在构建的任何智能体（最明显的是 delegate_task 子智能体，随后它会报告
# “Tool read_file does not exist”）中剥离整个 terminal+file 工具集。为了吸收此类波动，
# 但又不会永久锁定陈旧的“可用”判定，我们会记住每次检查最后一次返回 True 的时间；当新探测
# 在上次成功后的短宽限期内失败时，我们会提供最后一次成功的 True，而不是缓存该失败。
# 持续超出宽限期的一次失败将会被正常采纳，因此真正宕机的后端将停止发布其工具。
# ---------------------------------------------------------------------------

_CHECK_FN_TTL_SECONDS = 30.0
# How long after a successful check a subsequent transient failure is treated
# as a flake (last-good True is served) rather than a real outage. Kept short
# so a genuinely-down backend is reflected within a couple of turns.
_CHECK_FN_FAILURE_GRACE_SECONDS = 60.0
_CHECK_FN_CACHE_MAX = 512
_check_fn_cache: Dict[tuple[Callable, Optional[str]], tuple[float, bool]] = {}
# Monotonic timestamp of the most recent True result per check_fn.
_check_fn_last_good: Dict[tuple[Callable, Optional[str]], float] = {}
_check_fn_cache_lock = threading.Lock()
CHECK_FN_CACHE_BYPASS = ""


def _prune_check_fn_caches(now: float) -> None:
    """Expire stale entries and cap profile-dimensional cache growth.

    Caller must hold ``_check_fn_cache_lock``.
    """
    for key, (timestamp, _) in list(_check_fn_cache.items()):
        if now - timestamp >= _CHECK_FN_TTL_SECONDS:
            _check_fn_cache.pop(key, None)
    for key, timestamp in list(_check_fn_last_good.items()):
        if now - timestamp >= _CHECK_FN_FAILURE_GRACE_SECONDS:
            _check_fn_last_good.pop(key, None)
    while len(_check_fn_cache) >= _CHECK_FN_CACHE_MAX:
        _check_fn_cache.pop(next(iter(_check_fn_cache)))
    while len(_check_fn_last_good) >= _CHECK_FN_CACHE_MAX:
        _check_fn_last_good.pop(next(iter(_check_fn_last_good)))


def check_fn_cache_scope() -> Optional[str]:
    """Return the active profile key when availability is profile-scoped.

    Single-profile processes intentionally keep the historical process-wide
    cache. A multiplex gateway installs a Hermes-home override for every
    profile turn, so the canonical profile key is the stable isolation
    boundary across repeated turns for that profile.
    """
    try:
        from agent.secret_scope import is_multiplex_active

        if not is_multiplex_active():
            return None
        from hermes_constants import get_hermes_home_override

        override = get_hermes_home_override()
        if not override:
            return CHECK_FN_CACHE_BYPASS
        return str(Path(override).expanduser().resolve())
    except Exception:
        # Fail closed: bypass both cache layers rather than aliasing requests
        # whose multiplex profile identity could not be resolved.
        return CHECK_FN_CACHE_BYPASS


def _check_fn_cached(fn: Callable) -> bool:
    """返回 bool(fn())，并在跨多次调用间进行 TTL 缓存。

    发生异常会被当作 False 吞掉。在距离上一次返回 True 的
    ``_CHECK_FN_FAILURE_GRACE_SECONDS`` 秒内发生的短暂 False 或异常会被抑制
    （返回上一次有效的 True，且该失败不会被缓存，因此下一次调用会
    重新探测），以防止不稳定的外部检查（如 Docker 守护进程繁忙、套接字
    竞争、探测超时）在会话中途悄悄移除工具。
    """
    now = time.monotonic()
    scope = check_fn_cache_scope()
    if scope == CHECK_FN_CACHE_BYPASS:
        try:
            return bool(fn())
        except Exception:
            logger.warning(
                "check_fn %s raised while profile cache scope was unresolved; "
                "dependent tools will be unavailable this turn",
                getattr(fn, "__qualname__", fn),
                exc_info=True,
            )
            return False
    cache_key = (fn, scope)
    with _check_fn_cache_lock:
        _prune_check_fn_caches(now)
        cached = _check_fn_cache.get(cache_key)
        if cached is not None:
            ts, value = cached
            if now - ts < _CHECK_FN_TTL_SECONDS:
                return value

    raised = False
    try:
        value = bool(fn())
    except Exception:
        value = False
        raised = True

    with _check_fn_cache_lock:
        _prune_check_fn_caches(now)
        if value:
            _check_fn_last_good[cache_key] = now
            _check_fn_cache[cache_key] = (now, True)
            return True

        last_good = _check_fn_last_good.get(cache_key)
        if last_good is not None and now - last_good < _CHECK_FN_FAILURE_GRACE_SECONDS:
            # Recent success → treat this failure as a flake. Serve last-good
            # True and do NOT cache the failure, so the next call re-probes
            # rather than pinning a stale verdict for the full TTL.
            logger.warning(
                "check_fn %s failed (%s) within %.0fs of last success; "
                "treating as transient and keeping tool(s) available",
                getattr(fn, "__qualname__", fn),
                "raised" if raised else "returned False",
                _CHECK_FN_FAILURE_GRACE_SECONDS,
            )
            return True

        # No recent success (or grace expired) — honor the failure. Log it so
        # silent tool loss in quiet mode (subagents) is diagnosable.
        logger.warning(
            "check_fn %s %s; dependent tools will be unavailable this turn",
            getattr(fn, "__qualname__", fn),
            "raised" if raised else "returned False",
        )
        _check_fn_cache[cache_key] = (now, False)
        return False


def invalidate_check_fn_cache() -> None:
    """Drop all cached ``check_fn`` results. Call after config changes that
    affect tool availability (e.g. ``hermes tools enable``)."""
    with _check_fn_cache_lock:
        _check_fn_cache.clear()
        _check_fn_last_good.clear()


def get_cached_check_fn_result(fn: Callable) -> Optional[bool]:
    """Return the current cached verdict for *fn* if its TTL is still valid.

    Unlike :func:`_check_fn_cached`, this NEVER executes the probe. It is for
    read-only surfaces (e.g. dashboard status panels) that need the last-known
    availability without triggering network / auth / SDK work inside a request
    path. Returns ``None`` when there is no fresh cached verdict.
    """
    now = time.monotonic()
    scope = check_fn_cache_scope()
    if scope == CHECK_FN_CACHE_BYPASS:
        # Unresolved profile identity bypasses the cache entirely; there is no
        # trustworthy cached verdict to report.
        return None
    with _check_fn_cache_lock:
        cached = _check_fn_cache.get((fn, scope))
        if cached is None:
            return None
        ts, value = cached
        if now - ts < _CHECK_FN_TTL_SECONDS:
            return value
        return None


class ToolRegistry:
    """Singleton registry that collects tool schemas + handlers from tool files."""

    def __init__(self):
        # 内置及其他进程全局级的注册表。
        self._tools: Dict[str, ToolEntry] = {}
        # 插件注册表是以解析后的 HERMES_HOME 为键的覆盖层（Overlay）。
        # 某个 Profile 会优先看到属于自己的覆盖层，其次才是全局内置项。
        self._scoped_tools: Dict[str, Dict[str, ToolEntry]] = {}
        # 插件模块命名空间 -> 操作员针对内置覆盖的显式授权（Opt-in）。
        # 授权记录受生命周期管理；
        # 独立的作用域映射表将保持持久化，从而确保延迟回调仍被限制在 Profile 范围内。
        self._plugin_override_policy: Dict[
            tuple[Optional[str], str], _PluginOverridePolicy
        ] = {}
        # 在策略移除后，作用域归属关系仍保持持久，
        # 以便延迟执行的代码依然限定在其模块被加载的 Profile 中。
        self._plugin_module_scopes: Dict[str, Set[Optional[str]]] = {}
        self._toolset_checks: Dict[str, Callable] = {}
        self._toolset_aliases: Dict[str, str] = {}
        # MCP 动态刷新可能会在其他线程读取工具元数据时修改注册表，
        # 因此需要保持修改操作的串行化，并让读取者基于稳定的快照进行访问。
        self._lock = threading.RLock()
        # 单调递增的代际计数器（Generation Counter）。
        # 每次发生修改（注册 / 卸载 / 注册工具集别名 / MCP 刷新）时递增。
        # 外部调用者（如 get_tool_definitions）可以据此进行记忆化缓存：
        # 以该代际值为键的缓存条目，只要代际值未改变就一直有效。
        self._generation: int = 0

    @staticmethod
    def current_scope_key() -> str:
        """Return the active profile's canonical registry scope."""
        return hermes_home_key()

    def _merged_tools(self, scope: Optional[str] = None) -> Dict[str, ToolEntry]:
        """Return global tools overlaid with one profile's plugin tools."""
        active_scope = scope or self.current_scope_key()
        merged = dict(self._tools)
        merged.update(self._scoped_tools.get(active_scope, {}))
        return merged

    def _snapshot_state(
        self,
        scope: Optional[str] = None,
    ) -> tuple[List[ToolEntry], Dict[str, Callable]]:
        """Return a coherent snapshot of registry entries and toolset checks."""
        with self._lock:
            entries = list(self._merged_tools(scope).values())
            checks = dict(self._toolset_checks)
            for entry in entries:
                if entry.check_fn is not None:
                    checks[entry.toolset] = entry.check_fn
            return entries, checks

    def _snapshot_entries(self) -> List[ToolEntry]:
        """Return a stable snapshot of registered tool entries."""
        return self._snapshot_state()[0]

    def _toolset_has_exposable_tools(
        self,
        toolset: str,
        entries: List[ToolEntry],
    ) -> bool:
        """Return True when at least one tool in *toolset* would be exposed.

        Mirrors :meth:`get_tool_definitions` per-tool filtering so doctor,
        banners, and other toolset-level surfaces agree with runtime exposure.
        Mixed toolsets (e.g. ``terminal`` plus desktop-only ``read_terminal``)
        must not be gated solely by the first registered ``check_fn``.
        """
        check_results: Dict[Callable, bool] = {}
        for entry in entries:
            if entry.toolset != toolset:
                continue
            if not entry.check_fn:
                return True
            if entry.check_fn not in check_results:
                check_results[entry.check_fn] = _check_fn_cached(entry.check_fn)
            if check_results[entry.check_fn]:
                return True
        return False

    def get_entry(
        self,
        name: str,
        *,
        scope: Optional[str] = None,
    ) -> Optional[ToolEntry]:
        """Return the active profile's entry by name, falling back to global."""
        with self._lock:
            return self._merged_tools(scope).get(name)

    def snapshot_registration(
        self,
        name: str,
        *,
        scope: Optional[str] = None,
    ) -> Optional[ToolEntry]:
        """Return the local slot state without following global fallback."""
        with self._lock:
            target = self._tools if scope is None else self._scoped_tools.get(scope, {})
            return target.get(name)

    def get_registered_toolset_names(self) -> List[str]:
        """Return sorted unique toolset names present in the registry."""
        return sorted({entry.toolset for entry in self._snapshot_entries()})

    def get_all_entries(self) -> List[ToolEntry]:
        """Return the active profile's merged tool entries."""
        return self._snapshot_entries()

    def get_tool_names_for_toolset(self, toolset: str) -> List[str]:
        """Return sorted tool names registered under a given toolset."""
        return sorted(
            entry.name for entry in self._snapshot_entries()
            if entry.toolset == toolset
        )

    def register_toolset_alias(self, alias: str, toolset: str) -> None:
        """Register an explicit alias for a canonical toolset name."""
        with self._lock:
            existing = self._toolset_aliases.get(alias)
            if existing and existing != toolset:
                logger.warning(
                    "Toolset alias collision: '%s' (%s) overwritten by %s",
                    alias, existing, toolset,
                )
            self._toolset_aliases[alias] = toolset
            self._generation += 1

    def get_registered_toolset_aliases(self) -> Dict[str, str]:
        """Return a snapshot of ``{alias: canonical_toolset}`` mappings."""
        with self._lock:
            return dict(self._toolset_aliases)

    def get_toolset_alias_target(self, alias: str) -> Optional[str]:
        """Return the canonical toolset name for an alias, or None."""
        with self._lock:
            return self._toolset_aliases.get(alias)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_plugin_override_policy(
        self,
        module_namespace: str,
        allowed: bool,
        *,
        scope: Optional[str] = None,
    ) -> _PluginOverridePolicy:
        """将插件模块命名空间绑定到其当前的操作员显式授权（Opt-in）。

        带标识特性的返回结果使得插件在卸载/重新加载时，
        能够撤销过期的授权，
        同时又不会丢失持久的“模块-Profile”归属关系。
        """
        with self._lock:
            policy = _PluginOverridePolicy(allowed)
            self._plugin_override_policy[(scope, module_namespace)] = policy
            self._plugin_module_scopes.setdefault(module_namespace, set()).add(scope)
            return policy

    def snapshot_plugin_override_policy(
        self,
        module_namespace: str,
        *,
        scope: Optional[str] = None,
    ) -> Optional[_PluginOverridePolicy]:
        """Return one local authorization generation without fallback."""
        with self._lock:
            return self._plugin_override_policy.get((scope, module_namespace))

    def restore_plugin_override_policy(
        self,
        module_namespace: str,
        current: _PluginOverridePolicy,
        previous: Optional[_PluginOverridePolicy],
        *,
        scope: Optional[str] = None,
    ) -> bool:
        """CAS-restore policy state while retaining durable scope attribution."""
        with self._lock:
            key = (scope, module_namespace)
            if self._plugin_override_policy.get(key) is not current:
                return False
            if previous is None:
                self._plugin_override_policy.pop(key, None)
            else:
                self._plugin_override_policy[key] = previous
            return True

    def _plugin_override_allowed(
        self,
        scope: Optional[str],
        module_namespace: str,
    ) -> bool:
        policy = self._plugin_override_policy.get((scope, module_namespace))
        if policy is None and scope is not None:
            policy = self._plugin_override_policy.get((None, module_namespace))
        return bool(policy and policy.allowed)

    def _plugin_owner_of(self, handler: Callable) -> Optional[str]:
        """Return the plugin module namespace that defined *handler*, or None
        if it was not defined in a loaded plugin module.

        Authorization is bound to where the handler was DEFINED
        (``handler.__globals__["__name__"]``), which is fixed at definition
        time and cannot drift with the call site, thread, or timing. Lambdas
        and nested functions inherit the defining module's globals, so a
        plugin cannot launder an override through a callback. Built-in/MCP
        handlers live outside the plugin namespace and return None (unchanged
        behavior).
        """
        mod = self._callable_module(handler)
        if not mod:
            return None
        return self._plugin_namespace_of_module(mod)

    @staticmethod
    def _callable_module(handler: Callable) -> str:
        """Resolve defining module through wrappers, partials, and objects."""
        current = handler
        seen: Set[int] = set()
        while id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, functools.partial):
                current = current.func
                continue
            func = getattr(current, "__func__", None)
            if func is not None:
                current = func
                continue
            globals_dict = getattr(current, "__globals__", None)
            if isinstance(globals_dict, dict):
                module_name = globals_dict.get("__name__", "")
                if module_name:
                    return str(module_name)
            wrapped = getattr(current, "__wrapped__", None)
            if wrapped is not None:
                current = wrapped
                continue
            break
        module_name = getattr(current, "__module__", "")
        if module_name:
            return str(module_name)
        return str(getattr(type(current), "__module__", "") or "")

    def _plugin_namespace_of_module(
        self,
        module_namespace: str,
    ) -> Optional[str]:
        """Resolve a module/submodule to its durable plugin namespace."""
        with self._lock:
            matches = [
                namespace
                for namespace in self._plugin_module_scopes
                if module_namespace == namespace
                or module_namespace.startswith(f"{namespace}.")
            ]
            if matches:
                return max(matches, key=len)
        # Also gate plugin modules currently loading but not yet policy-recorded
        # (defensive: a handler defined in the plugin namespace is plugin code).
        if module_namespace.startswith("hermes_plugins."):
            return ".".join(module_namespace.split(".")[:2])
        return None

    def _plugin_scope_of(self, module_namespace: str) -> Optional[str]:
        """Return the profile scope bound to a loaded plugin module."""
        with self._lock:
            scopes = self._plugin_module_scopes.get(module_namespace)
            if not scopes:
                return None
            active_scope = self.current_scope_key()
            if active_scope in scopes:
                return active_scope
            if len(scopes) == 1:
                return next(iter(scopes))
            raise PermissionError(
                f"Plugin module {module_namespace!r} is active in multiple "
                "profiles and cannot register outside one of those scopes."
            )

    def plugin_scope_for_module(self, module_namespace: str) -> Optional[str]:
        """Public host lookup for a loaded plugin module's immutable scope."""
        owner = self._plugin_namespace_of_module(module_namespace)
        return self._plugin_scope_of(owner or module_namespace)

    def plugin_scope_for_callable(self, callback: Callable) -> Optional[str]:
        """Return the durable plugin scope for any supported callable shape."""
        module_name = self._callable_module(callback)
        return self.plugin_scope_for_module(module_name) if module_name else None

    @staticmethod
    def _caller_module() -> str:
        """Best-effort module name of whoever called the registry method that
        invoked this helper (two frames up: this helper, then the registry
        method itself, then the actual caller).

        ``deregister()`` takes only a tool name — unlike ``register()`` it has
        no handler argument to bind authorization to via ``_plugin_owner_of``.
        Frame inspection is the only way to know who is asking.
        """
        try:
            frame = sys._getframe(2)
            return frame.f_globals.get("__name__", "") or ""
        except Exception:
            return ""

    def register(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        check_fn: Callable = None,
        requires_env: list = None,
        is_async: bool = False,
        description: str = "",
        emoji: str = "",
        max_result_size_chars: int | float | None = None,
        dynamic_schema_overrides: Callable = None,
        override: bool = False,
        scope: Optional[str] = None,
    ):
        """注册一个工具。在模块导入时由各个工具文件调用。

        ``override=True`` 是用于插件的显式选入（opt-in）选项，
        旨在替换现有的内置工具实现
        （例如：将默认的浏览器工具替换为带有界面的 Chrome CDP 后端）。
        如果不显式指定该选项，
        凡是会遮蔽来自其他工具集现有工具的注册行为都会被拒绝，
        以防止意外覆盖。
        """
        handler_owner = self._plugin_owner_of(handler)
        caller_owner = self._plugin_namespace_of_module(self._caller_module())
        owner = caller_owner or handler_owner
        if scope is None and owner is not None:
            scope = self._plugin_scope_of(owner)
        with self._lock:
            target = (
                self._tools
                if scope is None
                else self._scoped_tools.setdefault(scope, {})
            )
            existing = (
                self._tools.get(name)
                if scope is None
                else self._merged_tools(scope).get(name)
            )
            shadows_global = (
                owner is not None
                and scope is not None
                and name not in target
                and name in self._tools
            )
            if shadows_global:
                if not override:
                    logger.error(
                        "Tool registration REJECTED: plugin %r attempted to "
                        "shadow global tool %r without override=True",
                        owner,
                        name,
                    )
                    return
                if not self._plugin_override_allowed(scope, owner):
                    raise PermissionError(
                        f"Plugin module {owner!r} cannot override built-in "
                        f"tool {name!r} without operator opt-in "
                        f"(allow_tool_override)."
                    )
            if existing and existing.toolset != toolset:
                if override:
                    if owner is not None and not self._plugin_override_allowed(
                        scope, owner
                    ):
                        logger.error(
                            "Tool registration REJECTED: plugin %r attempted to "
                            "override built-in tool %r (existing toolset %r) without "
                            "operator opt-in. Set "
                            "plugins.entries.<plugin_id>.allow_tool_override: true "
                            "in config.yaml to allow it.",
                            owner, name, existing.toolset,
                        )
                        raise PermissionError(
                            f"Plugin module {owner!r} cannot override built-in "
                            f"tool {name!r} without operator opt-in "
                            f"(allow_tool_override)."
                        )
                    # Explicit opt-in (or non-plugin caller): replace the tool.
                    # Logged at INFO so the override is auditable in agent.log.
                    logger.info(
                        "Tool '%s': toolset '%s' overriding existing toolset '%s' "
                        "(override=True opt-in)",
                        name, toolset, existing.toolset,
                    )
                else:
                    # Reject every cross-toolset shadow, including MCP-to-MCP
                    # collisions. Legitimate MCP reconnect/refresh re-registers
                    # within the same canonical toolset and remains allowed.
                    logger.error(
                        "Tool registration REJECTED: '%s' (toolset '%s') would "
                        "shadow existing tool from toolset '%s'. Pass "
                        "override=True to register() if the replacement is "
                        "intentional, or deregister the existing tool first.",
                        name, toolset, existing.toolset,
                    )
                    return
            target[name] = ToolEntry(
                name=name,
                toolset=toolset,
                schema=schema,
                handler=handler,
                check_fn=check_fn,
                requires_env=requires_env or [],
                is_async=is_async,
                description=description or schema.get("description", ""),
                emoji=emoji,
                max_result_size_chars=max_result_size_chars,
                dynamic_schema_overrides=dynamic_schema_overrides,
            )
            # Availability is now derived per-tool (_toolset_has_exposable_tools),
            # so this map no longer gates a toolset. It is still consumed by
            # get_toolset_requirements -> TOOLSET_REQUIREMENTS["check_fn"], which
            # banner.py reads (presence only, never called) to classify an
            # already-unavailable toolset as lazy-init vs disabled. Keep the
            # write path for that classification.
            if scope is None and check_fn and toolset not in self._toolset_checks:
                self._toolset_checks[toolset] = check_fn
            self._generation += 1

    def deregister(self, name: str) -> None:
        """从注册表中移除一个工具。

        如果该工具集中不再包含其他工具，
        同时会清理相关的工具集检查。

        供 MCP 动态工具发现机制在服务器发送 ``notifications/tools/list_changed`` 信号时，
        用于进行全量更新（彻底清空并重新填充）。

        受与 ``register(override=True)`` 相同的运算符显式选择（opt-in）策略约束。
        若无此约束，插件可能会通过注销非自身拥有的工具，
        随后在已清空的槽位上调用普通 ``register()``，
        从而彻底绕过该策略门禁——因为 ``register()`` 仅在存在 ``existing``（现有条目）时才运行覆盖检查，
        若提前移除条目便会完全跳过此检查。

        MCP 工具集（``mcp-*``）在此受豁免：
        动态工具发现机制在每次刷新时，
        理应合法地对其自身工具进行全量重建，
        且不涉及插件覆盖的概念。
        """
        with self._lock:
            caller_mod = self._caller_module()
            caller_owner = self._plugin_namespace_of_module(caller_mod)
            caller_scope = (
                self._plugin_scope_of(caller_owner)
                if caller_owner is not None
                else None
            )
            target = (
                self._scoped_tools.get(caller_scope, {})
                if caller_scope is not None
                else self._tools
            )
            entry = target.get(name)
            if entry is None and caller_scope is not None:
                if name in self._tools:
                    raise PermissionError(
                        f"Scoped plugin module {caller_mod!r} cannot deregister "
                        f"process-global tool {name!r}; register a scoped "
                        "override instead."
                    )
                return
            if entry is None:
                return
            if not entry.toolset.startswith("mcp-"):
                owner = self._plugin_owner_of(entry.handler)
                # Ownership check: bind to the plugin package root
                # (``hermes_plugins.{name}``), not the exact module string.
                # A handler defined in ``hermes_plugins.pkg.handlers`` is
                # still owned by the ``hermes_plugins.pkg`` package — exact
                # string equality would wrongly block root-module cleanup code
                # from removing tools registered by a submodule of the same
                # plugin (egilewski review on #55840).
                same_plugin = bool(owner and caller_owner == owner)
                if (
                    caller_owner is not None
                    and not same_plugin
                    and not self._plugin_override_allowed(
                        caller_scope, caller_owner
                    )
                ):
                    logger.error(
                        "Tool deregistration REJECTED: plugin %r attempted to "
                        "remove tool %r (toolset %r) it does not own, without "
                        "operator opt-in. Set "
                        "plugins.entries.%s.allow_tool_override: true in "
                        "config.yaml to allow it.",
                        caller_mod, name, entry.toolset, caller_mod,
                    )
                    raise PermissionError(
                        f"Plugin module {caller_mod!r} cannot deregister tool "
                        f"{name!r} (toolset {entry.toolset!r}) without operator "
                        f"opt-in (allow_tool_override)."
                    )
            del target[name]
            if caller_scope is not None and not target:
                self._scoped_tools.pop(caller_scope, None)
            # Drop the toolset check and aliases if this was the last tool in
            # that toolset.
            toolset_still_exists = any(
                e.toolset == entry.toolset
                for e in self._merged_tools(caller_scope).values()
            )
            if not toolset_still_exists:
                self._toolset_checks.pop(entry.toolset, None)
                self._toolset_aliases = {
                    alias: target
                    for alias, target in self._toolset_aliases.items()
                    if target != entry.toolset
                }
            self._generation += 1
        logger.debug("Deregistered tool: %s", name)

    def restore_registration(
        self,
        name: str,
        current: ToolEntry,
        previous: Optional[ToolEntry],
        *,
        scope: Optional[str] = None,
    ) -> bool:
        """Restore a host-owned registration if it is still current.

        This is the narrow inverse used by the plugin ownership ledger.  The
        identity check is deliberate: another plugin (or another
        ``PluginManager`` in a multi-profile process) may have registered a
        newer entry under the same name, in which case unloading this entry
        must leave the newer entry untouched.
        """
        with self._lock:
            target = (
                self._tools
                if scope is None
                else self._scoped_tools.setdefault(scope, {})
            )
            if target.get(name) is not current:
                return False

            if previous is None:
                target.pop(name, None)
            else:
                target[name] = previous
            if scope is not None and not target:
                self._scoped_tools.pop(scope, None)

            # Rebuild the affected toolset checks from the surviving entries.
            # A plugin may have replaced an entry in the same toolset, so
            # simply leaving the current check_fn behind would retain stale
            # plugin state after restoration.
            affected_toolsets = {current.toolset}
            if previous is not None:
                affected_toolsets.add(previous.toolset)
            for toolset in affected_toolsets:
                surviving = [
                    entry for entry in self._merged_tools(scope).values()
                    if entry.toolset == toolset
                ]
                check_fn = next(
                    (entry.check_fn for entry in surviving if entry.check_fn),
                    None,
                )
                if scope is None:
                    if check_fn is None:
                        self._toolset_checks.pop(toolset, None)
                    else:
                        self._toolset_checks[toolset] = check_fn
                if not surviving and not any(
                    entry.toolset == toolset
                    for entries in self._scoped_tools.values()
                    for entry in entries.values()
                ):
                    self._toolset_aliases = {
                        alias: target
                        for alias, target in self._toolset_aliases.items()
                        if target != toolset
                    }
            self._generation += 1
        logger.debug("Restored tool registration: %s", name)
        return True

    # ------------------------------------------------------------------
    # Schema retrieval
    # ------------------------------------------------------------------

    def get_definitions(self, tool_names: Set[str], quiet: bool = False) -> List[dict]:
        """根据请求的工具名称返回 OpenAI 格式的工具 schema。

        仅包含 ``check_fn()`` 返回 True（或未设定 check_fn）的
        工具。``check_fn()`` 的结果会通过 :func:`_check_fn_cached`
        缓存约 30 秒，以平摊重复探测的开销（如 check_terminal_
        requirements 探测 modal/docker，浏览器检查探测 playwright
        等）；选择此 TTL（生存时间）是为了让环境变量的变更
        （如 ``hermes tools enable foo``）仍能在近乎实时的情况下生效，
        同时无需在每次调用时都强制进行完整的缓存刷新。
        """
        result = []
        # 基于 30 秒 TTL 之上的单次调用级缓存 —— 用于在单次定义构建流程中
        # 处理对同一个 check_fn 的重复探测，无需再次读取 TTL 时钟。
        check_results: Dict[Callable, bool] = {}
        entries_by_name = {entry.name: entry for entry in self._snapshot_entries()}
        for name in sorted(tool_names):
            entry = entries_by_name.get(name)
            if not entry:
                continue
            if entry.check_fn:
                if entry.check_fn not in check_results:
                    check_results[entry.check_fn] = _check_fn_cached(entry.check_fn)
                if not check_results[entry.check_fn]:
                    if not quiet:
                        logger.debug("Tool %s unavailable (check failed)", name)
                    continue
            # Ensure schema always has a "name" field — use entry.name as fallback
            schema_with_name = {**entry.schema, "name": entry.name}
            # 应用运行时动态重写（例如 delegate_task 的描述
            # 取决于当前的 delegation.max_concurrent_children /
            # max_spawn_depth）。调用方（model_tools.get_tool_definitions）
            # 已经根据 config.yaml 的 mtime + size 对其备忘（memo）进行了键控，
            # 因此 config 中 delegation.* 的更改会自动使缓存失效。
            if entry.dynamic_schema_overrides is not None:
                try:
                    overrides = entry.dynamic_schema_overrides()
                    if isinstance(overrides, dict):
                        schema_with_name.update(overrides)
                except Exception as exc:
                    logger.warning(
                        "dynamic_schema_overrides for tool %s raised %s; "
                        "using static schema",
                        name, exc,
                    )
            result.append({"type": "function", "function": schema_with_name})
        return result

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_handler_result(name: str, result):
        """强制约束 Agent 工具流水线所支持的结果格式。

        常规的工具返回结果应为字符串。
        唯一的结构化例外，是供 Agent 执行器（executor）调用的多模态包（multimodal envelope）。

        将其他所有返回值都作为字符串类型的错误返回，
        可以防止日志记录、钩子函数（hooks）、预算管理以及持久化模块
        接收到无法安全进行切片或大小计算的值。
        """
        if isinstance(result, str):
            return _bound_json_error_result(result)
        if (
            isinstance(result, dict)
            and result.get("_multimodal") is True
            and isinstance(result.get("content"), list)
        ):
            return result

        result_type = type(result).__name__
        logger.error(
            "Tool %s handler returned unsupported result type: %s",
            name,
            result_type,
        )
        return tool_error(
            f"Tool handler returned unsupported result type: {result_type}",
            error_type="tool_result_contract",
            tool=name,
            result_type=result_type,
        )

    def dispatch(
        self,
        name: str,
        args: dict,
        *,
        scope: Optional[str] = None,
        **kwargs,
    ) -> str | dict:
        """Execute a tool handler by name.

        * 异步处理程序（Async handlers）会自动通过 ``_run_async()`` 进行桥接适配。
        * 处理程序的返回结果在离开注册表前，
          会被规范化（normalize）为字符串或受支持的多模态包（multimodal envelope）。
        * 所有捕获到的异常都会以 ``{"error": "..."}`` 的统一格式返回，
          以保证错误输出格式的一致性。
        """
        entry = self.get_entry(name, scope=scope)
        if not entry:
            return tool_error(f"Unknown tool: {name}")
        try:
            if entry.is_async:
                from model_tools import _run_async
                result = _run_async(entry.handler(args, **kwargs))
            else:
                result = entry.handler(args, **kwargs)
            return self._normalize_handler_result(name, result)
        except Exception as e:
            # exc_info already renders the exception, so keep the message copy bounded.
            logger.exception(
                "Tool %s dispatch error: %s", name, _bound_error_text(str(e))
            )
            # Route through the sanitizer so framing tokens / CDATA / fences
            # in exception strings don't reach the model as structural noise.
            # See model_tools._sanitize_tool_error for rationale.
            raw = f"Tool execution failed: {type(e).__name__}: {e}"
            try:
                from model_tools import _sanitize_tool_error
                sanitized = _sanitize_tool_error(raw)
            except Exception:
                sanitized = raw  # defensive: never let the sanitizer block error propagation
            return tool_error(sanitized)

    # ------------------------------------------------------------------
    # Query helpers  (replace redundant dicts in model_tools.py)
    # ------------------------------------------------------------------

    def get_max_result_size(self, name: str, default: int | float | None = None) -> int | float:
        """Return per-tool max result size, or *default* (or global default)."""
        entry = self.get_entry(name)
        if entry and entry.max_result_size_chars is not None:
            return entry.max_result_size_chars
        if default is not None:
            return default
        from tools.budget_config import DEFAULT_RESULT_SIZE_CHARS
        return DEFAULT_RESULT_SIZE_CHARS

    def get_all_tool_names(self) -> List[str]:
        """Return sorted list of all registered tool names."""
        return sorted(entry.name for entry in self._snapshot_entries())

    def get_schema(self, name: str) -> Optional[dict]:
        """Return a tool's raw schema dict, bypassing check_fn filtering.

        Useful for token estimation and introspection where availability
        doesn't matter — only the schema content does.
        """
        entry = self.get_entry(name)
        return entry.schema if entry else None

    def get_toolset_for_tool(self, name: str) -> Optional[str]:
        """Return the toolset a tool belongs to, or None."""
        entry = self.get_entry(name)
        return entry.toolset if entry else None

    def get_emoji(self, name: str, default: str = "⚡") -> str:
        """Return the emoji for a tool, or *default* if unset."""
        entry = self.get_entry(name)
        return (entry.emoji if entry and entry.emoji else default)

    def get_tool_to_toolset_map(self) -> Dict[str, str]:
        """Return ``{tool_name: toolset_name}`` for every registered tool."""
        return {entry.name: entry.toolset for entry in self._snapshot_entries()}

    def is_toolset_available(self, toolset: str) -> bool:
        """Check if a toolset has at least one exposable tool.

        Returns False (rather than crashing) when a per-tool check raises
        an unexpected exception (e.g. network error, missing import, bad config).
        """
        entries, _ = self._snapshot_state()
        return self._toolset_has_exposable_tools(toolset, entries)

    def check_toolset_requirements(self) -> Dict[str, bool]:
        """Return ``{toolset: available_bool}`` for every toolset."""
        entries, _ = self._snapshot_state()
        toolsets = sorted({entry.toolset for entry in entries})
        return {
            toolset: self._toolset_has_exposable_tools(toolset, entries)
            for toolset in toolsets
        }

    def get_available_toolsets(self) -> Dict[str, dict]:
        """Return toolset metadata for UI display."""
        toolsets: Dict[str, dict] = {}
        entries, _ = self._snapshot_state()
        for entry in entries:
            ts = entry.toolset
            if ts not in toolsets:
                toolsets[ts] = {
                    "available": self._toolset_has_exposable_tools(ts, entries),
                    "tools": [],
                    "description": "",
                    "requirements": [],
                }
            toolsets[ts]["tools"].append(entry.name)
            if entry.requires_env:
                for env in entry.requires_env:
                    if env not in toolsets[ts]["requirements"]:
                        toolsets[ts]["requirements"].append(env)
        return toolsets

    def get_toolset_requirements(self) -> Dict[str, dict]:
        """Build a TOOLSET_REQUIREMENTS-compatible dict for backward compat."""
        result: Dict[str, dict] = {}
        entries, toolset_checks = self._snapshot_state()
        for entry in entries:
            ts = entry.toolset
            if ts not in result:
                result[ts] = {
                    "name": ts,
                    "env_vars": [],
                    "check_fn": toolset_checks.get(ts),
                    "setup_url": None,
                    "tools": [],
                }
            if entry.name not in result[ts]["tools"]:
                result[ts]["tools"].append(entry.name)
            for env in entry.requires_env:
                if env not in result[ts]["env_vars"]:
                    result[ts]["env_vars"].append(env)
        return result

    def check_tool_availability(self, quiet: bool = False):
        """Return (available_toolsets, unavailable_info) like the old function."""
        available = []
        unavailable = []
        entries, _ = self._snapshot_state()
        for ts in sorted({entry.toolset for entry in entries}):
            ts_entries = [entry for entry in entries if entry.toolset == ts]
            if self._toolset_has_exposable_tools(ts, entries):
                available.append(ts)
            else:
                unavailable.append({
                    "name": ts,
                    "env_vars": ts_entries[0].requires_env if ts_entries else [],
                    "tools": [entry.name for entry in ts_entries],
                })
        return available, unavailable


# Module-level singleton
registry = ToolRegistry()


# ---------------------------------------------------------------------------
# 工具响应序列化辅助函数
# ---------------------------------------------------------------------------
# 每个工具处理程序（handler）都必须返回一个 JSON 字符串。
# 这些辅助函数消除了在各个工具文件中
# 出现数百次的模板化代码 ``json.dumps({"error": msg}, ensure_ascii=False)``。
#
# 使用方法：
#   from tools.registry import registry, tool_error, tool_result
#
#   return tool_error("something went wrong")
#   return tool_error("not found", code=404)
#   return tool_result(success=True, data=payload)
#   return tool_result(items)            # 直接传递字典对象

def tool_error(message, **extra) -> str:
    """Return a JSON error string for tool handlers.

    >>> tool_error("file not found")
    '{"error": "file not found"}'
    >>> tool_error("bad input", success=False)
    '{"error": "bad input", "success": false}'
    """
    # Bound the context-bound copy so a raw exception can't bloat history across retries.
    result = {"error": _bound_error_text(str(message))}
    if extra:
        result.update(extra)
    return json.dumps(result, ensure_ascii=False)


def tool_result(data=None, **kwargs) -> str:
    """Return a JSON result string for tool handlers.

    Accepts a dict positional arg *or* keyword arguments (not both):

    >>> tool_result(success=True, count=42)
    '{"success": true, "count": 42}'
    >>> tool_result({"key": "value"})
    '{"key": "value"}'
    """
    if data is not None:
        return json.dumps(data, ensure_ascii=False)
    return json.dumps(kwargs, ensure_ascii=False)
