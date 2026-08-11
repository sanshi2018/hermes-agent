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
import importlib
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


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
    """
    try:
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
    except (OSError, SyntaxError):
        return False

    return any(_is_registry_register_call(stmt) for stmt in tree.body)


def discover_builtin_tools(tools_dir: Optional[Path] = None) -> List[str]:
    """Import built-in self-registering tool modules and return their module names."""
    tools_path = Path(tools_dir) if tools_dir is not None else Path(__file__).resolve().parent
    module_names = [
        f"tools.{path.stem}"
        for path in sorted(tools_path.glob("*.py"))
        if path.name not in {"__init__.py", "registry.py", "mcp_tool.py"}
        and _module_registers_tools(path)
    ]

    imported: List[str] = []
    for mod_name in module_names:
        try:
            importlib.import_module(mod_name)
            imported.append(mod_name)
        except Exception as e:
            logger.warning("Could not import tool module %s: %s", mod_name, e)
    return imported


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
_check_fn_cache: Dict[Callable, tuple[float, bool]] = {}
# Monotonic timestamp of the most recent True result per check_fn.
_check_fn_last_good: Dict[Callable, float] = {}
_check_fn_cache_lock = threading.Lock()


def _check_fn_cached(fn: Callable) -> bool:
    """返回 bool(fn())，并在跨多次调用间进行 TTL 缓存。

    发生异常会被当作 False 吞掉。在距离上一次返回 True 的
    ``_CHECK_FN_FAILURE_GRACE_SECONDS`` 秒内发生的短暂 False 或异常会被抑制
    （返回上一次有效的 True，且该失败不会被缓存，因此下一次调用会
    重新探测），以防止不稳定的外部检查（如 Docker 守护进程繁忙、套接字
    竞争、探测超时）在会话中途悄悄移除工具。
    """
    now = time.monotonic()
    with _check_fn_cache_lock:
        cached = _check_fn_cache.get(fn)
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
        if value:
            _check_fn_last_good[fn] = now
            _check_fn_cache[fn] = (now, True)
            return True

        last_good = _check_fn_last_good.get(fn)
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
        _check_fn_cache[fn] = (now, False)
        return False


def invalidate_check_fn_cache() -> None:
    """Drop all cached ``check_fn`` results. Call after config changes that
    affect tool availability (e.g. ``hermes tools enable``)."""
    with _check_fn_cache_lock:
        _check_fn_cache.clear()
        _check_fn_last_good.clear()


class ToolRegistry:
    """Singleton registry that collects tool schemas + handlers from tool files."""

    def __init__(self):
        self._tools: Dict[str, ToolEntry] = {}
        # 持久化映射表：插件模块命名空间（handler.__globals__["__name__"]）
        # -> 用于覆盖内置功能的运算符显式选择（opt-in）策略。
        #
        # 该映射表在插件加载时填充且永不清空，
        # 因此插件的覆盖授权会严格绑定到定义处理程序（handler）的代码上，
        # 无论 register() 何时被调用
        # （无论是在加载期间同步调用，还是在之后通过延迟/线程回调调用）。
        self._plugin_override_policy: Dict[str, bool] = {}
        self._toolset_checks: Dict[str, Callable] = {}
        self._toolset_aliases: Dict[str, str] = {}
        # MCP 的动态刷新机制可能会在其他线程读取工具元数据时对注册表进行修改，
        # 因此需要保持修改操作的序列化执行，
        # 并确保读取者始终基于稳定的快照进行访问。
        self._lock = threading.RLock()
        # 单调递增的代际计数器（generation counter）。
        # 每当发生变更操作（如注册、注销、注册工具集别名、MCP 刷新）时该值都会自增。
        #
        # 外部调用方（例如 get_tool_definitions）可以基于它进行缓存记忆：
        # 只要该代际计数没有改变，以该代际值为键（key）的缓存条目就一直有效。
        self._generation: int = 0

    def _snapshot_state(self) -> tuple[List[ToolEntry], Dict[str, Callable]]:
        """Return a coherent snapshot of registry entries and toolset checks."""
        with self._lock:
            return list(self._tools.values()), dict(self._toolset_checks)

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

    def get_entry(self, name: str) -> Optional[ToolEntry]:
        """Return a registered tool entry by name, or None."""
        with self._lock:
            return self._tools.get(name)

    def get_registered_toolset_names(self) -> List[str]:
        """Return sorted unique toolset names present in the registry."""
        return sorted({entry.toolset for entry in self._snapshot_entries()})

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

    def register_plugin_override_policy(self, module_namespace: str, allowed: bool) -> None:
        """Bind a plugin module namespace to its operator opt-in for built-in
        override. Called once per plugin at load time. Durable: never cleared,
        so later (even threaded/delayed) register() calls from that module are
        still gated by the same policy.
        """
        with self._lock:
            self._plugin_override_policy[module_namespace] = bool(allowed)

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
        try:
            mod = handler.__globals__.get("__name__", "")  # type: ignore[attr-defined]
        except AttributeError:
            return None
        if mod in self._plugin_override_policy:
            return mod
        # Also gate plugin modules currently loading but not yet policy-recorded
        # (defensive: a handler defined in the plugin namespace is plugin code).
        if isinstance(mod, str) and mod.startswith("hermes_plugins."):
            return mod
        return None

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
    ):
        """注册一个工具。在模块导入时由各个工具文件调用。

        ``override=True`` 是用于插件的显式选入（opt-in）选项，
        旨在替换现有的内置工具实现
        （例如：将默认的浏览器工具替换为带有界面的 Chrome CDP 后端）。
        如果不显式指定该选项，
        凡是会遮蔽来自其他工具集现有工具的注册行为都会被拒绝，
        以防止意外覆盖。
        """
        with self._lock:
            existing = self._tools.get(name)
            if existing and existing.toolset != toolset:
                # 允许 MCP 之间的相互覆盖（合法场景：服务器刷新，
                # 或两个 MCP 服务器拥有重名的工具）。
                both_mcp = (
                    existing.toolset.startswith("mcp-")
                    and toolset.startswith("mcp-")
                )
                if both_mcp:
                    logger.debug(
                        "Tool '%s': MCP toolset '%s' overwriting MCP toolset '%s'",
                        name, toolset, existing.toolset,
                    )
                elif override:
                    _owner = self._plugin_owner_of(handler)
                    if _owner is not None and not self._plugin_override_policy.get(_owner, False):
                        logger.error(
                            "Tool registration REJECTED: plugin %r attempted to "
                            "override built-in tool %r (existing toolset %r) without "
                            "operator opt-in. Set "
                            "plugins.entries.<plugin_id>.allow_tool_override: true "
                            "in config.yaml to allow it.",
                            _owner, name, existing.toolset,
                        )
                        raise PermissionError(
                            f"Plugin module {_owner!r} cannot override built-in "
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
                    # Reject shadowing — prevent plugins/MCP from overwriting
                    # built-in tools or vice versa.
                    logger.error(
                        "Tool registration REJECTED: '%s' (toolset '%s') would "
                        "shadow existing tool from toolset '%s'. Pass "
                        "override=True to register() if the replacement is "
                        "intentional, or deregister the existing tool first.",
                        name, toolset, existing.toolset,
                    )
                    return
            self._tools[name] = ToolEntry(
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
            if check_fn and toolset not in self._toolset_checks:
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
            entry = self._tools.get(name)
            if entry is None:
                return
            if not entry.toolset.startswith("mcp-"):
                caller_mod = self._caller_module()
                owner = self._plugin_owner_of(entry.handler)
                # 所有权检查：绑定到插件包根目录（``hermes_plugins.{name}``），
                # 而不是精确的模块字符串。
                #
                # 定义在 ``hermes_plugins.pkg.handlers`` 中的处理程序（handler），
                # 其所有权依然属于 ``hermes_plugins.pkg`` 包——
                # 如果使用精确的字符串相等校验，会导致根模块的清理代码
                # 错误地无法移除由同一插件的子模块所注册的工具
                # （参照 #55840 中 egilewski 的 Code Review 意见）。
                caller_root = ".".join(caller_mod.split(".")[:2])
                owner_root = ".".join(owner.split(".")[:2]) if owner else ""
                same_plugin = bool(owner and caller_root == owner_root)
                if (
                    caller_mod.startswith("hermes_plugins.")
                    and not same_plugin
                    and not self._plugin_override_policy.get(caller_root, False)
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
            del self._tools[name]
            # 如果这是该工具集中的最后一个工具，
            # 则移除对应工具集的校验及别名。
            toolset_still_exists = any(
                e.toolset == entry.toolset for e in self._tools.values()
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
            return result
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
        return json.dumps({
            "error": f"Tool handler returned unsupported result type: {result_type}",
            "error_type": "tool_result_contract",
            "tool": name,
            "result_type": result_type,
        }, ensure_ascii=False)

    def dispatch(self, name: str, args: dict, **kwargs) -> str | dict:
        """按名称执行指定的工具处理程序（handler）。

        * 异步处理程序（Async handlers）会自动通过 ``_run_async()`` 进行桥接适配。
        * 处理程序的返回结果在离开注册表前，
          会被规范化（normalize）为字符串或受支持的多模态包（multimodal envelope）。
        * 所有捕获到的异常都会以 ``{"error": "..."}`` 的统一格式返回，
          以保证错误输出格式的一致性。
        """
        entry = self.get_entry(name)
        if not entry:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            if entry.is_async:
                from model_tools import _run_async
                result = _run_async(entry.handler(args, **kwargs))
            else:
                result = entry.handler(args, **kwargs)
            return self._normalize_handler_result(name, result)
        except Exception as e:
            logger.exception("Tool %s dispatch error: %s", name, e)
            # Route through the sanitizer so framing tokens / CDATA / fences
            # in exception strings don't reach the model as structural noise.
            # See model_tools._sanitize_tool_error for rationale.
            raw = f"Tool execution failed: {type(e).__name__}: {e}"
            try:
                from model_tools import _sanitize_tool_error
                sanitized = _sanitize_tool_error(raw)
            except Exception:
                sanitized = raw  # defensive: never let the sanitizer block error propagation
            return json.dumps({"error": sanitized})

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
    result = {"error": str(message)}
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
