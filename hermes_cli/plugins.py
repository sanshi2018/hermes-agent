"""
Hermes 插件系统
====================

探索、加载并管理来自以下四个来源的插件：

1. **内置插件**   – ``<repo>/plugins/<name>/``（随 hermes-agent 一起发布；
   跳过 ``memory/`` 和 ``context_engine/`` 子目录 — 它们拥有自己的探索路径）
2. **用户插件**   – ``~/.hermes/plugins/<name>/``
3. **项目插件**   – ``./.hermes/plugins/<name>/``（通过设置
   ``HERMES_ENABLE_PROJECT_PLUGINS`` 手动开启）
4. **Pip 插件**    – 暴露了 ``hermes_agent.plugins``
   入口点组（entry-point group）的 Python 包。

发生名称冲突时，后列出的来源会覆盖前面的来源，
因此与内置插件同名的用户插件或项目插件将会替换原有的内置插件。

每个目录形式的插件都必须包含一个 ``plugin.yaml`` 清单文件，
**以及**一个带有 ``register(ctx)`` 函数的 ``__init__.py`` 文件。

生命周期 Hook
---------------
插件可以针对 ``VALID_HOOKS`` 中的任意 Hook 注册回调函数。
Agent 核心会在适当的节点调用 ``invoke_hook(name, **kwargs)``。

工具注册
-----------------
``PluginContext.register_tool()`` 会委托给 ``tools.registry.register()`` 处理，
从而使插件定义的工具能够与内置工具并列展示和使用。
"""
from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import inspect
import logging
import os
import sys
import threading
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from hermes_constants import get_hermes_home
from utils import env_var_enabled, fast_safe_load
from hermes_cli.config import cfg_get
from hermes_cli.middleware import OBSERVER_SCHEMA_VERSION, VALID_MIDDLEWARE


def get_bundled_plugins_dir() -> Path:
    """定位内置的 ``plugins/`` 目录。

    优先遵循 ``HERMES_BUNDLED_PLUGINS`` 环境变量
    （由 Nix 封装器 / 打包安装程序设置），
    以便首先调取只读存储路径。
    若未设置，则回退到开发期间使用的代码库内路径。
    """
    env_override = os.getenv("HERMES_BUNDLED_PLUGINS")
    if env_override:
        return Path(env_override)
    return Path(__file__).resolve().parent.parent / "plugins"

try:
    import yaml
except ImportError:  # pragma: no cover – yaml is optional at import time
    yaml = None  # type: ignore[assignment]


class PluginToolOverrideError(PermissionError):
    """Raised when a plugin attempts to override a built-in tool without
    operator opt-in via ``plugins.entries.<plugin_id>.allow_tool_override``.
    """


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plugin developer debug logging
# ---------------------------------------------------------------------------
#
# Set ``HERMES_PLUGINS_DEBUG=1`` to surface verbose plugin-discovery logs to
# stderr in addition to ~/.hermes/logs/agent.log. Aimed at plugin authors
# trying to figure out why their plugin isn't showing up: which directories
# were scanned, which manifests parsed, which plugins were skipped (and why),
# what each ``register(ctx)`` call registered, and full tracebacks on load
# failure.
#
# The env var is read once at import time; tests that need to flip it
# mid-process can call ``_install_plugin_debug_handler(force=True)``.

_PLUGINS_DEBUG = os.getenv("HERMES_PLUGINS_DEBUG", "").strip().lower() in {
    "1", "true", "yes", "on",
}
_DEBUG_HANDLER_INSTALLED = False


def _install_plugin_debug_handler(force: bool = False) -> None:
    """When HERMES_PLUGINS_DEBUG is on, tee plugin logs to stderr at DEBUG.

    Idempotent: only attaches the handler once per process unless ``force``
    is passed. Does not touch the root logger or other Hermes loggers.
    """
    global _DEBUG_HANDLER_INSTALLED, _PLUGINS_DEBUG
    if force:
        _PLUGINS_DEBUG = os.getenv("HERMES_PLUGINS_DEBUG", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
    if not _PLUGINS_DEBUG or _DEBUG_HANDLER_INSTALLED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("[plugins] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # Don't double-emit through the root logger when the central logging
    # config also writes to stderr. agent.log still captures everything.
    logger.propagate = True
    _DEBUG_HANDLER_INSTALLED = True
    logger.debug(
        "HERMES_PLUGINS_DEBUG=1 — verbose plugin discovery logging enabled"
    )


_install_plugin_debug_handler()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_HOOKS: Set[str] = {
    "pre_tool_call",
    "post_tool_call",
    "transform_terminal_output",
    "transform_tool_result",
    # 在将 LLM 输出返回给用户之前对其进行转换。
    # 插件可以返回一个字符串来替换响应文本，或者返回 None/空值以保持内容不变。
    # 以第一个非 None 的字符串结果为准。常用于词汇转换或人设风格塑造。
    "transform_llm_output",
    "pre_llm_call",
    "post_llm_call",
    # 验证循环门控（Verification-loop gate）。
    # 当 Agent 编辑了代码并准备进行验证/结束时（在 verify-on-stop 守卫之后），每轮会触发一次。
    # 回调可以通过返回以下内容，让 Agent 继续运行（例如运行一项检查、延迟验证或清理 diff），而不是直接停止：
    #   {"action": "continue", "message": "<后续指令>"}
    # 同时也兼容 Claude-Code Stop 的格式：{"decision": "block", "reason": "..."}（阻止停止 == 继续运行）。
    # 返回其他任何内容都会允许本轮正常结束。
    # Hermes 内置的引导逻辑基于循证验证停止的提醒机制；
    # 本 Hook 则用于用户/插件的策略管控，且受限于 agent.max_verify_nudges 的次数上限。
    "pre_verify",
    "pre_api_request",
    "post_api_request",
    "api_request_error",
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "subagent_start",
    "subagent_stop",
    # 网关预分发 Hook。
    # 在内部事件守卫之后、但在身份验证/配对以及 Agent 分发之前，
    # 对每个接收到的 MessageEvent 触发一次。
    # 插件可以返回一个字典来影响执行流程：
    #   {"action": "skip",    "reason": "..."}  -> 丢弃消息（不作回复）
    #   {"action": "rewrite", "text": "..."}    -> 替换 event.text，然后继续
    #   {"action": "allow"}  /  None             -> 正常分发
    # 关键字参数：event: MessageEvent, gateway: GatewayRunner, session_store。
    "pre_gateway_dispatch",
    # 审批生命周期 Hook。
    # 当危险命令需要审批决策时，由 tools/approval.py 触发 ——
    # 覆盖 CLI 交互式提示、网关/ACP 审批以及智能模式（smart-mode）辅助 LLM 决策场景。
    # 仅作为观察者：返回值会被忽略。
    # 插件无法通过这些 Hook 否决或提前响应审批（若想在工具进入审批前阻止它，请使用 pre_tool_call）。
    #
    # pre_approval_request 的关键字参数：
    #   command: str, description: str, pattern_key: str, pattern_keys: list[str],
    #   session_key: str, surface: "cli" | "gateway" | "smart"
    # post_approval_response 的关键字参数：与上述相同，并附加：
    #   choice: "once" | "session" | "always" | "deny" | "timeout"
    #           | "smart_approve" | "smart_deny"
    #   decided_by: "aux_llm"  --仅在 surface="smart" 时提供
    "pre_approval_request",
    "post_approval_response",
    # 看板任务生命周期 Hook。
    # 当任务发生状态转变、且变更已提交至看板数据库后，由 hermes_cli.kanban_db 触发
    # （因此该 Hook 看到的始终是持久化状态，且慢速插件绝不会占用 SQLite 写锁）。
    # 仅作为观察者：返回值会被忽略。
    #
    # 每个 Hook 在【哪个进程】中触发非常关键，
    # 因为看板 Worker 是作为独立的 `hermes -p <profile> chat -q` 子进程运行的：
    #   - kanban_task_claimed   -> 在 分发器进程（网关嵌入式分发器或 `hermes kanban dispatch`）中触发，
    #                              紧接在 Worker 子进程派生之前。
    #   - kanban_task_completed -> 在 WORKER 进程中触发，当其调用 kanban_complete
    #                              （或通过 CLI/手动完成）时。
    #   - kanban_task_blocked   -> 在 WORKER 进程（Worker 发起的阻塞）
    #                              或驱动该阻塞的对应进程中触发。
    # 如果插件需要集中观察每一次状态转变，应当在分发器中挂载 Hook；
    # 如果需要在会话内获取每个任务的具体上下文，则应当在 Worker 中挂载 Hook。
    #
    # 通用关键字参数：task_id: str, board: str | None, assignee: str | None,
    #   run_id: int | None, profile_name: str。
    # kanban_task_completed 附加参数：summary: str | None。
    # kanban_task_blocked 附加参数：  reason: str | None。
    "kanban_task_claimed",
    "kanban_task_completed",
    "kanban_task_blocked",
}

ENTRY_POINTS_GROUP = "hermes_agent.plugins"

_NS_PARENT = "hermes_plugins"


def _env_enabled(name: str) -> bool:
    """Return True when an env var is set to a truthy opt-in value."""
    return env_var_enabled(name)


def _get_disabled_plugins() -> set:
    """Read the disabled plugins list from config.yaml.

    Kept for backward compat and explicit deny-list semantics. A plugin
    name in this set will never load, even if it appears in
    ``plugins.enabled``.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        disabled = cfg_get(config, "plugins", "disabled", default=[])
        return set(disabled) if isinstance(disabled, list) else set()
    except Exception:
        return set()


def _get_enabled_plugins() -> Optional[set]:
    """Read the enabled-plugins allow-list from config.yaml.

    Plugins are opt-in by default — only plugins whose name appears in
    this set are loaded. Returns:

    * ``None`` — the key is missing or malformed. Callers should treat
      this as "nothing enabled yet" (the opt-in default); the first
      ``migrate_config`` run populates the key with a grandfathered set
      of currently-installed user plugins so existing setups don't
      break on upgrade.
    * ``set()`` — an empty list was explicitly set; nothing loads.
    * ``set(...)`` — the concrete allow-list.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        plugins_cfg = config.get("plugins")
        if not isinstance(plugins_cfg, dict):
            return None
        if "enabled" not in plugins_cfg:
            return None
        enabled = plugins_cfg.get("enabled")
        if not isinstance(enabled, list):
            return None
        return set(enabled)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

_VALID_PLUGIN_KINDS: Set[str] = {"standalone", "backend", "exclusive", "platform", "model-provider"}


@dataclass
class PluginManifest:
    """Parsed representation of a plugin.yaml manifest."""

    name: str
    version: str = ""
    description: str = ""
    author: str = ""
    requires_env: List[Union[str, Dict[str, Any]]] = field(default_factory=list)
    provides_tools: List[str] = field(default_factory=list)
    provides_hooks: List[str] = field(default_factory=list)
    source: str = ""        # "user", "project", or "entrypoint"
    path: Optional[str] = None
    # 插件类型 — 语义请参阅 plugins.py 模块文档字符串。
    # ``standalone``（默认）：拥有自身的 hook/tool；通过 ``plugins.enabled`` 手动开启。
    # ``backend``：现有核心工具的可插拔后端（例如 image_gen）。
    #              内置（随源码发布）的后端会自动加载；
    #              用户安装的后端仍受 ``plugins.enabled`` 管控。
    # ``exclusive``：同一时刻仅能有一个活动提供方的分类（例如 memory）。
    #               通过 ``<category>.provider`` 配置项进行选择；
    #               该分类自身的探索系统负责具体加载，通用扫描程序会跳过这些插件。
    # ``platform``：网关消息平台适配器（例如 IRC）。
    #              内置的平台插件会自动加载，以便所有随软件发布的平台均可开箱即用；
    #              位于 ~/.hermes/plugins/ 中用户安装的平台插件（非信任代码）
    #              仍受 ``plugins.enabled`` 管控。
    kind: str = "standalone"
    # 注册表键（Key）— 基于路径生成，用于 ``plugins.enabled``/``disabled``
    # 的查找以及 ``hermes plugins list`` 命令。
    # 对于位于 ``plugins/disk-cleanup/`` 的扁平插件，键为 ``disk-cleanup``；
    # 对于位于 ``plugins/image_gen/openai/`` 的嵌套分类插件，
    # 键为 ``image_gen/openai``。当该值为空时，回退使用 ``name``。
    key: str = ""


@dataclass
class LoadedPlugin:
    """Runtime state for a single loaded plugin."""

    manifest: PluginManifest
    module: Optional[types.ModuleType] = None
    tools_registered: List[str] = field(default_factory=list)
    hooks_registered: List[str] = field(default_factory=list)
    middleware_registered: List[str] = field(default_factory=list)
    commands_registered: List[str] = field(default_factory=list)
    enabled: bool = False
    error: Optional[str] = None
    # True for a bundled platform plugin recorded as a deferred (not-yet-
    # imported) loader. The module loads on first real use via the
    # platform_registry; see PluginManager._register_deferred_platform.
    deferred: bool = False


# ---------------------------------------------------------------------------
# PluginContext  – handed to each plugin's ``register()`` function
# ---------------------------------------------------------------------------

class PluginContext:
    """Facade given to plugins so they can register tools and hooks."""

    def __init__(self, manifest: PluginManifest, manager: "PluginManager"):
        self.manifest = manifest
        self._manager = manager
        # Lazy-built host-owned LLM facade — see ctx.llm property below.
        self._llm: Any = None

    # -- host-owned LLM access ----------------------------------------------

    @property
    def llm(self) -> Any:
        """返回插件的 :class:`agent.plugin_llm.PluginLlm` 门面（facade）。

        允许受信任的插件针对用户当前活动的模型和身份验证，
        运行由宿主拥有的聊天或结构化补全，而无需自带提供商密钥。
        覆盖功能（模型、agent id、auth 配置文件）默认采用故障关闭（fail-closed）策略，
        并通过 ``plugins.entries.<plugin_id>.llm.*`` 配置键进行管控。

        有关完整的接口表面，请参阅 :mod:`agent.plugin_llm`。"""
        if self._llm is None:
            from agent.plugin_llm import PluginLlm
            plugin_id = self.manifest.key or self.manifest.name
            self._llm = PluginLlm(plugin_id=plugin_id)
        return self._llm

    # -- profile awareness --------------------------------------------------

    @property
    def profile_name(self) -> str:
        """返回当前活动的 Hermes 配置文件名称（例如 ``"default"``）。

        通过 :func:`hermes_cli.profiles.get_active_profile_name` 从 ``HERMES_HOME`` 推导得出，
        因此它在所有执行上下文（交互式 CLI、网关以及看板派生的 Worker 会话）中均可正常工作，
        且不依赖于 ``_cli_ref``（在交互式 CLI 运行之外，该值均为 ``None``）。

        针对默认配置文件返回 ``"default"``；
        当运行在 ``~/.hermes/profiles/<name>`` 下时返回配置文件 ID；
        当 ``HERMES_HOME`` 指向无法识别的位置时，则返回 ``"custom"``。
        """
        try:
            from hermes_cli.profiles import get_active_profile_name
            return get_active_profile_name()
        except Exception:
            return "default"

    # -- tool registration --------------------------------------------------

    def register_tool(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        check_fn: Callable | None = None,
        requires_env: list | None = None,
        is_async: bool = False,
        description: str = "",
        emoji: str = "",
        override: bool = False,
    ) -> None:
        """在全局注册表中注册一个工具，**同时**将其标记为由插件提供。

        传入 ``override=True`` 可以替换同名的现存内置工具
        （例如：将默认的 ``browser_navigate`` 替换为自定义的基于 CDP 的实现）。
        如果不传该参数，尝试注册已被其他工具集占用的名称将被拒绝。

        针对内置工具使用 ``override=True`` 需要操作员在 config.yaml 中
        通过 ``plugins.entries.<plugin_id>.allow_tool_override: true`` 进行显式开启
        —— 这与用于 ``ctx.llm`` 提供商/模型覆盖的信任门控模式（trust gate pattern）保持一致（#23194）。
        如果没有该门控，任何已启用的插件都可以悄悄替换像 ``shell_exec`` 或 ``write_file``
        这样的特权内置工具，并窃取模型通过它们调用的所有内容。
        """
        if override and not self._tool_override_allowed(name):
            plugin_id = self.manifest.key or self.manifest.name
            raise PluginToolOverrideError(
                f"Plugin {self.manifest.name!r} cannot override built-in tool "
                f"{name!r}. Set "
                f"plugins.entries.{plugin_id}.allow_tool_override: true "
                f"in config.yaml to allow this plugin to replace built-in tools."
            )

        from tools.registry import registry

        registry.register(
            name=name,
            toolset=toolset,
            schema=schema,
            handler=handler,
            check_fn=check_fn,
            requires_env=requires_env,
            is_async=is_async,
            description=description,
            emoji=emoji,
            override=override,
        )
        self._manager._plugin_tool_names.add(name)
        logger.debug(
            "Plugin %s registered tool: %s%s",
            self.manifest.name, name, " (override)" if override else "",
        )

    # -- override trust gate ------------------------------------------------

    def _tool_override_allowed(self, tool_name: str) -> bool:
        """如果该插件被配置为允许覆盖内置工具，则返回 True。

        内置插件（随 Hermes 核心一同发布）默认受信任 —
        其中的覆盖行为属于维护者的有意选择，
        而非试图进行权限提升的第三方插件。
        对于其他所有来源的插件，均需要在 config.yaml 中
        针对 ``plugins.entries.<plugin_id>`` 配置
        ``allow_tool_override: true``。
        """
        source = getattr(self.manifest, "source", "") or ""
        if source == "bundled":
            return True
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
        except Exception:
            # If we can't load config, fail closed — better to break the
            # override than silently grant it.
            return False
        plugin_id = self.manifest.key or self.manifest.name
        entries = (cfg.get("plugins") or {}).get("entries") or {}
        entry = entries.get(plugin_id) or {}
        return bool(entry.get("allow_tool_override", False))

    # -- message injection --------------------------------------------------

    def inject_message(self, content: str, role: str = "user") -> bool:
        """向当前活动会话中注入一条消息。

        如果 Agent 处于空闲状态（正在等待用户输入），这将开启一个新的轮次（turn）。
        如果 Agent 正在运行中，这会打断当前进程并注入该消息。

        这使得插件（例如远程控制查看器、消息桥接器）
        能够从外部源向会话中发送消息。

        如果消息成功排队，则返回 True。
        """
        cli = self._manager._cli_ref
        if cli is None:
            logger.warning("inject_message: no CLI reference (not available in gateway mode)")
            return False

        msg = content if role == "user" else f"[{role}] {content}"

        if getattr(cli, "_agent_running", False):
            # Agent is mid-turn — interrupt with the message
            cli._interrupt_queue.put(msg)
        else:
            # Agent is idle — queue as next input
            cli._pending_input.put(msg)
        return True

    # -- CLI command registration --------------------------------------------

    def register_cli_command(
        self,
        name: str,
        help: str,
        setup_fn: Callable,
        handler_fn: Callable | None = None,
        description: str = "",
    ) -> None:
        """
        注册一个 CLI 子命令（例如 ``hermes honcho ...``）。

        *setup_fn* 接收一个 argparse 的子解析器（subparser），
        并应当在其中添加对应的参数或下级子解析器。

        如果提供了 *handler_fn*，
        它将被设置为默认的分发函数（通过 ``set_defaults(func=...)``）。
        """
        self._manager._cli_commands[name] = {
            "name": name,
            "help": help,
            "description": description,
            "setup_fn": setup_fn,
            "handler_fn": handler_fn,
            "plugin": self.manifest.name,
        }
        logger.debug("Plugin %s registered CLI command: %s", self.manifest.name, name)

    # -- slash command registration -------------------------------------------

    def register_command(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        args_hint: str = "",
    ) -> None:
        """注册一个可在 CLI 和网关会话中使用的斜杠命令（例如 ``/lcm``）。

        处理函数的签名应为 ``fn(raw_args: str) -> str | None``。
        它也可以是一个异步可调用对象 —— 网关分发机制同时支持这两种形式。

        与 ``register_cli_command()``（用于创建 ``hermes <subcommand>`` 终端命令）不同，
        本方法注册的是会话内的斜杠命令，供用户在对话过程中调用。

        ``args_hint`` 是一个可选的简短字符串（例如 ``"<file>"`` 或 ``"dias:7 formato:json"``），
        网关适配器可通过它向用户展示带有参数输入框的命令
        —— 例如 Discord 原生的斜杠命令选择器。
        未设置 ``args_hint`` 的插件命令在 Discord 中将注册为无参命令，
        但以自由文本形式调用时仍可接收后续追加的文本。

        若命令名称与内置命令冲突，该注册将被拒绝并发出警告。
        """
        clean = name.lower().strip().lstrip("/").replace(" ", "-")
        if not clean:
            logger.warning(
                "Plugin '%s' tried to register a command with an empty name.",
                self.manifest.name,
            )
            return

        # Reject if it conflicts with a built-in command
        try:
            from hermes_cli.commands import resolve_command
            if resolve_command(clean) is not None:
                logger.warning(
                    "Plugin '%s' tried to register command '/%s' which conflicts "
                    "with a built-in command. Skipping.",
                    self.manifest.name, clean,
                )
                return
        except Exception:
            pass  # If commands module isn't available, skip the check

        self._manager._plugin_commands[clean] = {
            "handler": handler,
            "description": description or "Plugin command",
            "plugin": self.manifest.name,
            "args_hint": (args_hint or "").strip(),
        }
        logger.debug("Plugin %s registered command: /%s", self.manifest.name, clean)

    # -- tool dispatch -------------------------------------------------------

    def dispatch_tool(self, tool_name: str, args: dict, **kwargs) -> str:
        """
        在父 Agent 上下文中，通过注册表分发工具调用。

        这是面向插件斜杠命令（slash commands）的公开接口，用于需要调用诸如
        ``delegate_task`` 等工具的场景，无需直接触及框架内部机制。
        父 Agent（如果可用）会被自动解析——插件永远无需直接访问 Agent。

        参数：
            tool_name: 工具在注册表中的名称（例如 ``"delegate_task"``）。
            args: 工具参数字典（与模型传递的格式相同）。
            **kwargs: 转发给注册表分发函数的额外关键字参数。

        返回：
            来自工具处理函数的 JSON 字符串（格式与模型工具调用相同）。
        """
        from tools.registry import registry

        # 当父 Agent 上下文可用时进行关联（CLI 模式）。
        # 在 Gateway 模式下，_cli_ref 为 None —— 工具会优雅降级
        # （工作区提示退回使用 TERMINAL_CWD，且不显示加载动画）。
        if "parent_agent" not in kwargs:
            cli = self._manager._cli_ref
            agent = getattr(cli, "agent", None) if cli else None
            if agent is not None:
                kwargs["parent_agent"] = agent

        return registry.dispatch(tool_name, args, **kwargs)

    # -- context engine registration -----------------------------------------

    def register_context_engine(self, engine) -> None:
        """注册一个上下文引擎以替换内置的 ContextCompressor。

        仅允许注册一个上下文引擎插件。
        若有第二个插件尝试注册，将被拒绝并收到警告。

        该引擎必须是 ``agent.context_engine.ContextEngine`` 的实例。
        """
        if self._manager._context_engine is not None:
            logger.warning(
                "Plugin '%s' tried to register a context engine, but one is "
                "already registered. Only one context engine plugin is allowed.",
                self.manifest.name,
            )
            return
        # Defer the import to avoid circular deps at module level
        from agent.context_engine import ContextEngine
        if not isinstance(engine, ContextEngine):
            logger.warning(
                "Plugin '%s' tried to register a context engine that does not "
                "inherit from ContextEngine. Ignoring.",
                self.manifest.name,
            )
            return
        self._manager._context_engine = engine
        logger.info(
            "Plugin '%s' registered context engine: %s",
            self.manifest.name, engine.name,
        )

    # -- image gen provider registration ------------------------------------

    def register_image_gen_provider(self, provider) -> None:
        """注册一个图像生成后端。

        ``provider`` 必须是 :class:`agent.image_gen_provider.ImageGenProvider` 的实例。
        在路由 ``image_generate`` 工具调用时，
        ``config.yaml`` 中的 ``image_gen.provider`` 会与 ``provider.name`` 属性进行匹配。
        """
        from agent.image_gen_provider import ImageGenProvider
        from agent.image_gen_registry import register_provider

        if not isinstance(provider, ImageGenProvider):
            logger.warning(
                "Plugin '%s' tried to register an image_gen provider that does "
                "not inherit from ImageGenProvider. Ignoring.",
                self.manifest.name,
            )
            return
        register_provider(provider)
        logger.info(
            "Plugin '%s' registered image_gen provider: %s",
            self.manifest.name, provider.name,
        )

    # -- dashboard auth provider registration --------------------------------

    def register_dashboard_auth_provider(self, provider) -> None:
        """注册仪表盘身份验证提供者。

        ``provider`` 必须是
        :class:`hermes_cli.dashboard_auth.DashboardAuthProvider` 的实例。
        用于仪表盘的 OAuth 身份验证网关，当仪表盘绑定到
        非回环主机且未指定 ``--insecure`` 参数时会被激活。

        行为异常的提供者（类型错误或名称重复）会被记录为 WARNING 级别的日志，
        并被静默忽略 —— 绝不会抛出异常 —— 从而确保损坏的插件
        不会导致宿主机崩溃。该规范与
        ``register_image_gen_provider`` 保持一致。
        """
        from hermes_cli.dashboard_auth import (
            DashboardAuthProvider, register_provider,
        )

        if not isinstance(provider, DashboardAuthProvider):
            logger.warning(
                "Plugin '%s' tried to register a dashboard-auth provider "
                "that does not inherit from DashboardAuthProvider. Ignoring.",
                self.manifest.name,
            )
            return
        try:
            register_provider(provider)
        except (TypeError, ValueError) as e:
            logger.warning(
                "Plugin '%s' failed to register dashboard-auth provider "
                "%r: %s",
                self.manifest.name, getattr(provider, "name", "?"), e,
            )
            return
        logger.info(
            "Plugin '%s' registered dashboard-auth provider: %s (%s)",
            self.manifest.name, provider.name, provider.display_name,
        )

    # -- video gen provider registration -------------------------------------

    def register_video_gen_provider(self, provider) -> None:
        """注册一个视频生成后端。

        ``provider`` 必须是
        :class:`agent.video_gen_provider.VideoGenProvider` 的实例。
        在路由 ``video_generate`` 工具调用时，
        ``config.yaml`` 中的 ``video_gen.provider`` 会与
        ``provider.name`` 属性进行匹配。
        """
        from agent.video_gen_provider import VideoGenProvider
        from agent.video_gen_registry import register_provider as _register_video_provider

        if not isinstance(provider, VideoGenProvider):
            logger.warning(
                "Plugin '%s' tried to register a video_gen provider that does "
                "not inherit from VideoGenProvider. Ignoring.",
                self.manifest.name,
            )
            return
        _register_video_provider(provider)
        logger.info(
            "Plugin '%s' registered video_gen provider: %s",
            self.manifest.name, provider.name,
        )

    # -- web search/extract provider registration ----------------------------

    def register_web_search_provider(self, provider) -> None:
        """注册一个 Web 搜索/提取后端。

        `provider` 必须是
        :class:`agent.web_search_provider.WebSearchProvider` 的实例。

        当路由 `web_search` / `web_extract` 工具调用时，
        `config.yaml` 中的 `web.search_backend` /
        `web.extract_backend` / `web.backend` 配置项
        将与 `provider.name` 属性进行匹配。
        """
        from agent.web_search_provider import WebSearchProvider
        from agent.web_search_registry import register_provider as _register_web_provider

        if not isinstance(provider, WebSearchProvider):
            logger.warning(
                "Plugin '%s' tried to register a web provider that does "
                "not inherit from WebSearchProvider. Ignoring.",
                self.manifest.name,
            )
            return
        _register_web_provider(provider)
        logger.info(
            "Plugin '%s' registered web provider: %s",
            self.manifest.name, provider.name,
        )

    # -- browser provider registration ---------------------------------------

    def register_browser_provider(self, provider) -> None:
        """注册一个云浏览器后端。

        ``provider`` 必须是
        :class:`agent.browser_provider.BrowserProvider` 的实例。
        在路由云端模式的 ``browser_*`` 工具调用时，
        ``config.yaml`` 中的 ``browser.cloud_provider`` 会与
        ``provider.name`` 属性进行匹配。

        与 :meth:`register_web_search_provider` 完全对称 —— 拥有相同的
        注册形式、门控机制和日志记录。浏览器子系统的
        分发器（:func:`tools.browser_tool._get_cloud_provider`）
        会查询由这些调用构建起来的注册表。
        """
        from agent.browser_provider import BrowserProvider
        from agent.browser_registry import register_provider as _register_browser_provider

        if not isinstance(provider, BrowserProvider):
            logger.warning(
                "Plugin '%s' tried to register a browser provider that does "
                "not inherit from BrowserProvider. Ignoring.",
                self.manifest.name,
            )
            return
        _register_browser_provider(provider)
        logger.info(
            "Plugin '%s' registered browser provider: %s",
            self.manifest.name, provider.name,
        )

    # -- secret source registration -------------------------------------------

    def register_secret_source(self, source) -> None:
        """注册一个外部密钥管理器后端。

        ``source`` 必须是
        :class:`agent.secret_sources.base.SecretSource` 的实例。
        当 ``secrets.<source.name>`` 配置项启用时，
        已注册的源将在启动期间的 ``load_hermes_dotenv()`` 流程中运行 ——
        即在加载 ``~/.hermes/.env`` 之后、Hermes 读取凭据之前。
        编排器（``agent.secret_sources.registry.apply_all``）负责掌控
        加载顺序、映射模式与批量模式的优先级、冲突警告以及来源追溯；
        密钥源本身仅负责提取数据。

        关于时序的说明：插件发现机制在启动流程中的发生节点
        晚于首次调用 ``load_hermes_dotenv()``，
        因此，发现该插件的当前进程在初始加载环境变量时，
        并不会查询由插件注册的密钥源。
        但此后派生的每个 Hermes 进程（网关子进程、Cron 会愿、
        Subagent），以及在执行 ``reset_secret_source_cache()`` 重新拉取后，
        **都会**对其进行查询。
        因此，插件密钥源最适合用于为运行中的集群提供凭据；
        而内置密钥源则用于覆盖首个进程的引导启动（Bootstrap）。

        契约要求（若不满足将被拒绝并发出警告）：
        继承自 ``SecretSource``；``api_version`` 需匹配 ``SECRET_SOURCE_API_VERSION``；
        拥有小写的唯一 ``name``；``shape`` 为 ``"mapped"`` 或 ``"bulk"``；
        拥有唯一的 ``scheme``（若设置）；
        以及实现一个绝不抛出异常、也绝不进行交互提示的 ``fetch()`` 方法。
        完整的契约要求请参见基类模块的文档字符串。
        """
        from agent.secret_sources.base import SecretSource
        from agent.secret_sources.registry import register_source

        if not isinstance(source, SecretSource):
            logger.warning(
                "Plugin '%s' tried to register a secret source that does "
                "not inherit from SecretSource. Ignoring.",
                self.manifest.name,
            )
            return
        if register_source(source):
            logger.info(
                "Plugin '%s' registered secret source: %s",
                self.manifest.name, source.name,
            )

    # -- TTS provider registration -------------------------------------------

    def register_tts_provider(self, provider) -> None:
        """注册一个文本转语音（TTS）后端。

        ``provider`` 必须是
        :class:`agent.tts_provider.TTSProvider` 的实例。
        在路由 ``text_to_speech`` 工具调用时，
        ``config.yaml`` 中的 ``tts.provider`` 会与
        ``provider.name`` 属性进行匹配 —— **但仅在满足以下条件时生效**：

        1. ``provider.name`` **不是** 内置的 TTS 提供者名称
           （如 ``edge``, ``openai``, ``elevenlabs`` 等）。
           内置项始终优先 —— 注册表会拒绝覆盖同名内置项并发出警告。
        2. 配置中 **不存在** 同名的 ``tts.providers.<name>: type: command`` 条目。
           发生名称冲突时，命令行提供者（Command-provider，PR #17843）会优先于插件注册，
           因为配置文件的作用域比插件安装更为局域化。

        本机制与命令行提供者注册表共存，而非替换后者 ——
        完整的设计原理请参见 Issue #30398。
        """
        from agent.tts_provider import TTSProvider
        from agent.tts_registry import register_provider as _register_tts_provider

        if not isinstance(provider, TTSProvider):
            logger.warning(
                "Plugin '%s' tried to register a TTS provider that does "
                "not inherit from TTSProvider. Ignoring.",
                self.manifest.name,
            )
            return
        _register_tts_provider(provider)
        logger.info(
            "Plugin '%s' registered TTS provider: %s",
            self.manifest.name, provider.name,
        )

    # -- transcription (STT) provider registration ---------------------------

    def register_transcription_provider(self, provider) -> None:
        """注册一个语音转文本（STT）后端。

        ``provider`` 必须是
        :class:`agent.transcription_provider.TranscriptionProvider` 的实例。
        在路由 :func:`tools.transcription_tools.transcribe_audio` 调用时，
        ``config.yaml`` 中的 ``stt.provider`` 会与 ``provider.name`` 属性进行匹配 ——
        **但仅在满足以下条件时生效**：

        1. ``provider.name`` **不是** 内置的 STT 提供者名称
           （如 ``local``, ``local_command``, ``groq``, ``openai``,
           ``mistral``, ``xai``）。内置项始终优先 —— 注册表会拒绝
           覆盖同名内置项并发出警告。
        2. 配置中 **不存在** 同名的 ``stt.providers.<name>: type: command`` 条目。
           发生名称冲突时，命令行提供者会优先于插件注册，
           因为配置文件的作用域比插件安装更为局域化 ——
           该优先级规则与 TTS 保持一致。

        本机制与内置分发器以及 STT 命令行提供者注册表共存，
        而非替换它们。6 个内置的 STT 后端继续保留它们在
        ``tools/transcription_tools.py`` 中的原生实现；
        本 Hook 专用于 *新型* Python 引擎（如 OpenRouter,
        SenseAudio, Gemini-STT, 自定义专有后端）。
        """
        from agent.transcription_provider import TranscriptionProvider
        from agent.transcription_registry import register_provider as _register_stt_provider

        if not isinstance(provider, TranscriptionProvider):
            logger.warning(
                "Plugin '%s' tried to register a transcription provider that "
                "does not inherit from TranscriptionProvider. Ignoring.",
                self.manifest.name,
            )
            return
        _register_stt_provider(provider)
        logger.info(
            "Plugin '%s' registered transcription provider: %s",
            self.manifest.name, provider.name,
        )

    # -- platform adapter registration ---------------------------------------

    def register_platform(
        self,
        name: str,
        label: str,
        adapter_factory: Callable,
        check_fn: Callable,
        validate_config: Callable | None = None,
        required_env: list | None = None,
        install_hint: str = "",
        **entry_kwargs: Any,
    ) -> None:
        """注册一个网关平台适配器。

        适配器工厂（adapter_factory）接收一个 ``PlatformConfig`` 参数，
        并返回一个 ``BasePlatformAdapter`` 子类的实例。网关在实例化前
        会先调用 ``check_fn()`` 以验证相关的依赖条件。

        额外的关键字参数将被转发给 ``PlatformEntry``
        （例如 ``setup_fn``、``emoji``、``allowed_users_env``、``platform_hint`` 等）。
        若传入未知参数键，数据类构造函数将抛出 TypeError 异常。

        示例：:

            ctx.register_platform(
                name="irc",
                label="IRC",
                adapter_factory=lambda cfg: IRCAdapter(cfg),
                check_fn=lambda: True,
                emoji="💬",
                setup_fn=irc_interactive_setup,
            )
        """
        from gateway.platform_registry import platform_registry, PlatformEntry

        entry_kwargs.setdefault("plugin_name", self.manifest.name)
        entry = PlatformEntry(
            name=name,
            label=label,
            adapter_factory=adapter_factory,
            check_fn=check_fn,
            validate_config=validate_config,
            required_env=required_env or [],
            install_hint=install_hint,
            source="plugin",
            **entry_kwargs,
        )
        platform_registry.register(entry)
        self._manager._plugin_platform_names.add(name)
        logger.debug(
            "Plugin %s registered platform: %s",
            self.manifest.name,
            name,
        )

    # -- slack action handler registration ----------------------------------

    def register_slack_action_handler(
        self,
        action_id: Any,
        callback: Callable,
    ) -> None:
        """从插件中注册一个 Slack Block Kit 操作处理器。

        Hermes 的 Slack 适配器会在连接时将已注册的处理器绑定到其
        ``slack_bolt.AsyncApp`` 中。当用户点击按钮
        （或与其他 Block Kit 操作元素交互）且其 ``action_id`` 匹配时，
        该回调函数将被调用。

        回调函数签名遵循 slack_bolt 的约定规范：:

            async def handler(ack, body, action) -> None:
                await ack()  # 必须在 3 秒内执行
                ...

        参数：
            action_id: ``slack_bolt.App.action()`` 所接受的任意形式 ——
                字符串字面量的 ``action_id``、用于匹配多个 ID 的已编译 ``re.Pattern``，
                或是约束条件字典（例如 ``{"action_id": "...", "block_id": "..."}``）。
            callback: 接收 ``(ack, body, action)`` 参数的异步可调用对象。

        抛出异常：
            ValueError: 当 ``callback`` 不是可调用对象，
                或 ``action_id`` 为空/None 时抛出。

        示例：:

            async def _on_approve(ack, body, action):
                await ack()
                # 根据 action["value"] 执行某些工作流操作

            ctx.register_slack_action_handler("inbox_sweep_approve", _on_approve)
        """
        if not callable(callback):
            raise ValueError(
                f"Plugin '{self.manifest.name}' tried to register a Slack "
                f"action handler with a non-callable callback."
            )
        if action_id is None or (isinstance(action_id, str) and not action_id.strip()):
            raise ValueError(
                f"Plugin '{self.manifest.name}' tried to register a Slack "
                f"action handler with an empty action_id."
            )
        self._manager._slack_action_handlers.append(
            (action_id, callback, self.manifest.name)
        )
        logger.debug(
            "Plugin %s registered Slack action handler: %s",
            self.manifest.name,
            action_id,
        )

    # -- hook registration --------------------------------------------------

    # -- auxiliary task registration ---------------------------------------

    def register_auxiliary_task(
        self,
        key: str,
        *,
        display_name: str,
        description: str,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        """注册一个由插件定义的辅助 LLM 任务。

        辅助任务是基于 LLM 的侧边任务（视觉分析、网页提取、
        压缩、智能审批等），其调用路由均经过 ``auxiliary_client.py`` 处理。
        每个任务拥有独立的 ``auxiliary.<key>`` 配置块，
        用户可以在其中固定与主 Chat 模型无关的 Provider 和模型。

        插件利用此功能来声明自身的辅助任务，而无需修改核心文件。
        注册成功后，该任务会：

          - 显示在 ``hermes model → Configure auxiliary models`` 的选择器中
          - 在网关启动时，其 Provider/Model/Base_URL/API_Key 会从 config.yaml
            桥接映射至 ``AUXILIARY_<KEY_UPPER>_*`` 环境变量中
          - 将默认路由字段（provider="auto", model="" 等）合并入已加载的配置中，
            使得 ``cfg.get("auxiliary", {}).get(key)`` 可以正常读取

        参数：
            key: 稳定的任务标识键（snake_case 命名）。用于配置项 ``auxiliary.<key>``
                以及环境变量 ``AUXILIARY_<KEY_UPPER>_*``。不得与内置任务标识键相冲突
                （如 vision, compression, web_extract, approval,
                mcp, title_generation, skills_hub, curator）。
            display_name: 显示在选择器中的易读名称。
            description: 显示在名称旁边的简短单行说明。
            defaults: 包含默认路由字段的可选字典。可识别的键包括：
                ``provider``（默认为 "auto"）、``model``（默认为 ""）、
                ``base_url``（默认为 ""）、``api_key``（默认为 ""）、
                ``timeout``（默认为 60）、``extra_body``（默认为 {}），
                以及任何特定于任务的额外参数（例如 ``download_timeout``）。
                未知键将被原样保留 —— 插件自行负责其任务的 Schema。

        抛出异常：
            ValueError: 当 *key* 为空、包含非法字符，
                或覆盖了内置的辅助任务标识键时抛出。

        示例：
            ctx.register_auxiliary_task(
                key="memory_retain_filter",
                display_name="Memory retain filter",
                description="hindsight pre-retain dedup/extract",
                defaults={"provider": "auto", "timeout": 30},
            )
        """
        # 校验 key 的格式
        if not key or not isinstance(key, str):
            raise ValueError(
                f"Plugin '{self.manifest.name}' tried to register auxiliary task "
                f"with invalid key {key!r}"
            )
        if not all(c.isalnum() or c == "_" for c in key):
            raise ValueError(
                f"Plugin '{self.manifest.name}' auxiliary task key {key!r} "
                f"must contain only alphanumeric characters and underscores"
            )

        # Lazy import to avoid circular: hermes_cli.main imports plugins indirectly
        from hermes_cli.main import _AUX_TASKS as _BUILTIN_AUX_TASKS

        builtin_keys = {k for k, _name, _desc in _BUILTIN_AUX_TASKS}
        if key in builtin_keys:
            raise ValueError(
                f"Plugin '{self.manifest.name}' cannot register auxiliary task "
                f"{key!r} — that key is reserved for a built-in task. "
                f"Pick a plugin-namespaced key (e.g. '{self.manifest.name}_{key}')."
            )

        # Reject duplicate registrations across plugins
        existing = self._manager._aux_tasks.get(key)
        if existing is not None and existing.get("plugin") != self.manifest.name:
            raise ValueError(
                f"Plugin '{self.manifest.name}' cannot register auxiliary task "
                f"{key!r} — already registered by plugin "
                f"'{existing.get('plugin')}'"
            )

        # Normalize defaults — plugin owns the schema, but we ensure routing
        # fields exist with sensible types so consumers don't crash.
        merged_defaults: Dict[str, Any] = {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 60,
            "extra_body": {},
        }
        if defaults:
            for k, v in defaults.items():
                merged_defaults[k] = v

        self._manager._aux_tasks[key] = {
            "key": key,
            "display_name": display_name,
            "description": description,
            "defaults": merged_defaults,
            "plugin": self.manifest.name,
        }
        logger.debug(
            "Plugin %s registered auxiliary task: %s (%s)",
            self.manifest.name,
            key,
            display_name,
        )

    # -- redaction pattern registration --------------------------------------

    def register_redaction_patterns(self, patterns) -> int:
        """向脱敏引擎中追加注册密钥 Token 的正则表达式。

        接收的模式会加入到 :mod:`agent.redact` 中的供应商前缀交替匹配中，
        并在内置模式适用的所有地方（日志、终端输出、传输错误、对话记录）被掩码遮蔽，
        且保持相同的首尾掩码规则以及在 ``file_read`` 内容上的不可复用标记。
        在过去，每个新的供应商 Token 格式都需要提交核心 PR 来追加到
        ``_PREFIX_PATTERNS`` 中；而提供者插件应当自行管理其格式。

        该注册表是**仅允许追加**的：插件只能扩展需要被掩码的内容，
        而无法移除或削弱内置的模式，因此插件只会产生过度脱敏，
        而绝不会导致数据泄露。运维人员的全局 Opt-out 配置
        （``security.redact_secrets: false``）对插件模式的效力与内置模式完全一致。

        每个模式都必须能够编译为正则表达式，且至少以 2 个字面字符开头
        （例如 ``r"nvapi-[A-Za-z0-9_-]{20,}"``）。
        非法的条目会被警告并跳过 —— 绝不抛出异常。

        返回被接收的模式数量。
        """
        from agent.redact import register_redaction_patterns as _register

        try:
            count = _register(
                patterns, source=f"plugin:{self.manifest.name}",
            )
        except Exception as exc:
            logger.warning(
                "Plugin '%s' redaction pattern registration failed: %s",
                self.manifest.name, exc,
            )
            return 0
        logger.debug(
            "Plugin %s registered %d redaction pattern(s)",
            self.manifest.name, count,
        )
        return count

    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """Register a lifecycle hook callback.

        Unknown hook names produce a warning but are still stored so
        forward-compatible plugins don't break.
        """
        if hook_name not in VALID_HOOKS:
            logger.warning(
                "Plugin '%s' registered unknown hook '%s' "
                "(valid: %s)",
                self.manifest.name,
                hook_name,
                ", ".join(sorted(VALID_HOOKS)),
            )
        self._manager._hooks.setdefault(hook_name, []).append(callback)
        logger.debug("Plugin %s registered hook: %s", self.manifest.name, hook_name)

    # -- middleware registration -------------------------------------------

    def register_middleware(self, kind: str, callback: Callable) -> None:
        """注册一个用于改变行为的中间件回调函数。

        中间件与观察者 Hook（Observer hooks）是相互独立的：
        请求中间件（Request middleware）可以重写生效的 Payload，
        而执行中间件（Execution middleware）则可以包装真实的回调函数。
        为了保持向前兼容性，未知的类型会被保存，
        但系统会发出警告，以便插件开发者能够及时发现拼写错误。
        """
        if kind not in VALID_MIDDLEWARE:
            logger.warning(
                "Plugin '%s' registered unknown middleware '%s' "
                "(valid: %s)",
                self.manifest.name,
                kind,
                ", ".join(sorted(VALID_MIDDLEWARE)),
            )
        self._manager._middleware.setdefault(kind, []).append(callback)
        logger.debug("Plugin %s registered middleware: %s", self.manifest.name, kind)

    # -- skill registration -------------------------------------------------

    def register_skill(
        self,
        name: str,
        path: Path,
        description: str = "",
    ) -> None:
        """注册一个由该插件提供的只读技能（Skill）。

        该技能可以通过 ``skill_view()`` 以 ``'<plugin_name>:<name>'`` 的形式进行解析。
        它**不会**进入平铺的 ``~/.hermes/skills/`` 目录树，
        也**不会**被列入系统提示词（System Prompt）的 ``<available_skills>`` 索引中 ——
        插件技能仅支持通过显式指定进行选择性加载（Opt-in）。

        抛出异常：
            ValueError: 当 *name* 包含 ``':'`` 或非法字符时抛出。
            FileNotFoundError: 当 *path* 路径不存在时抛出。
        """
        from agent.skill_utils import _NAMESPACE_RE

        if ":" in name:
            raise ValueError(
                f"Skill name '{name}' must not contain ':' "
                f"(the namespace is derived from the plugin name "
                f"'{self.manifest.name}' automatically)."
            )
        if not name or not _NAMESPACE_RE.match(name):
            raise ValueError(
                f"Invalid skill name '{name}'. Must match [a-zA-Z0-9_-]+."
            )
        if not path.exists():
            raise FileNotFoundError(f"SKILL.md not found at {path}")

        qualified = f"{self.manifest.name}:{name}"
        self._manager._plugin_skills[qualified] = {
            "path": path,
            "plugin": self.manifest.name,
            "bare_name": name,
            "description": description,
        }
        logger.debug(
            "Plugin %s registered skill: %s",
            self.manifest.name, qualified,
        )


# ---------------------------------------------------------------------------
# PluginManager
# ---------------------------------------------------------------------------

class PluginManager:
    """Central manager that discovers, loads, and invokes plugins."""

    def __init__(self) -> None:
        self._plugins: Dict[str, LoadedPlugin] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._middleware: Dict[str, List[Callable]] = {}
        self._plugin_tool_names: Set[str] = set()
        self._plugin_platform_names: Set[str] = set()
        self._cli_commands: Dict[str, dict] = {}
        self._context_engine = None  # Set by a plugin via register_context_engine()
        self._plugin_commands: Dict[str, dict] = {}  # Slash commands registered by plugins
        self._discovered: bool = False
        self._cli_ref = None  # Set by CLI after plugin discovery
        # Plugin skill registry: qualified name → metadata dict.
        self._plugin_skills: Dict[str, Dict[str, Any]] = {}
        # Plugin-registered auxiliary tasks: key → {key, display_name,
        # description, defaults, plugin}. See PluginContext.register_auxiliary_task.
        self._aux_tasks: Dict[str, Dict[str, Any]] = {}
        # Slack Block Kit action handlers registered by plugins. Each entry
        # is (matcher, callback, plugin_name); the Slack adapter wires them
        # into its slack_bolt App at connect() time. ``matcher`` is whatever
        # ``app.action()`` accepts (a literal action_id string, a compiled
        # ``re.Pattern``, or a constraint dict); ``callback`` is an async
        # function with the slack_bolt signature ``(ack, body, action)``.
        self._slack_action_handlers: List[tuple] = []

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def discover_and_load(self, force: bool = False) -> None:
        """扫描所有插件源并加载找到的每个插件。

        当 ``force`` 为 true 时，会先清除缓存的探测状态，
        以便在无需完全重启 Agent 的情况下，
        让配置变更或新添加的内置后端在长生命周期的会话中生效。
        """
        if self._discovered and not force:
            return
        if env_var_enabled("HERMES_SAFE_MODE"):
            logger.info("HERMES_SAFE_MODE=1 — plugin discovery skipped")
            self._discovered = True
            return
        if force:
            self._plugins.clear()
            self._hooks.clear()
            self._middleware.clear()
            self._plugin_tool_names.clear()
            self._plugin_platform_names.clear()
            self._cli_commands.clear()
            self._plugin_commands.clear()
            self._plugin_skills.clear()
            self._aux_tasks.clear()
            self._slack_action_handlers.clear()
            self._context_engine = None
        # 预先设置该标志作为重入保护（插件的 register()
        # 可能会间接再次触发探索过程），
        # 但如果清理（sweep）引发异常，则重置该标志，
        # 从而避免将失败的扫描错误地缓存为“以空注册表完成探索”——
        # 调用方会捕获并消化该异常，
        # 否则将会被永久困在上方的提前返回逻辑中
        # （即“未配置 Web 提供商”这类失败情况）。
        self._discovered = True
        try:
            self._discover_and_load_inner()
        except BaseException:
            self._discovered = False
            raise

    def _discover_and_load_inner(self) -> None:
        """The actual discovery sweep — see :meth:`discover_and_load`."""
        manifests: List[PluginManifest] = []

        # 1. 内置插件（<repo>/plugins/<name>/）
        #
        # 随代码库发布的插件与 hermes_cli/ 平级放置。支持以下两种布局形式
        # （详情参阅 ``_scan_directory``）：
        #
        #   - 扁平结构（flat）：``plugins/disk-cleanup/plugin.yaml``（独立插件）
        #   - 分类结构（category）：``plugins/image_gen/openai/plugin.yaml``（后端驱动）
        #
        # 在顶层目录下，``memory/``、``context_engine/`` 以及 ``model-providers/``
        # 会被跳过 — 它们拥有各自独立的探索机制
        # （plugins/memory/__init__.py, providers/__init__.py）。
        # ``platforms/`` 则是存放平台适配器的分类目录
        # （需在其下一层级进行扫描）。
        repo_plugins = get_bundled_plugins_dir()
        logger.debug("Scanning bundled plugins: %s", repo_plugins)
        bundled = self._scan_directory(
            repo_plugins,
            source="bundled",
            skip_names={"memory", "context_engine", "platforms", "model-providers"},
        )
        logger.debug("  bundled (top-level): %d manifest(s)", len(bundled))
        manifests.extend(bundled)
        bundled_platforms = self._scan_directory(
            repo_plugins / "platforms", source="bundled"
        )
        logger.debug("  bundled/platforms: %d manifest(s)", len(bundled_platforms))
        manifests.extend(bundled_platforms)

        # 2. User plugins (~/.hermes/plugins/)
        user_dir = get_hermes_home() / "plugins"
        logger.debug("Scanning user plugins: %s", user_dir)
        user_manifests = self._scan_directory(user_dir, source="user")
        logger.debug("  user: %d manifest(s)", len(user_manifests))
        manifests.extend(user_manifests)

        # 3. Project plugins (./.hermes/plugins/)
        if _env_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
            project_dir = Path.cwd() / ".hermes" / "plugins"
            logger.debug("Scanning project plugins: %s", project_dir)
            project_manifests = self._scan_directory(project_dir, source="project")
            logger.debug("  project: %d manifest(s)", len(project_manifests))
            manifests.extend(project_manifests)
        else:
            logger.debug(
                "Project plugins disabled (set HERMES_ENABLE_PROJECT_PLUGINS=1 to enable)"
            )

        # 4. Pip / entry-point plugins
        ep_manifests = self._scan_entry_points()
        logger.debug("  entrypoints: %d manifest(s)", len(ep_manifests))
        manifests.extend(ep_manifests)

        # 加载各个清单文件（跳过用户禁用的插件）。
        # 当键（Key）发生冲突时，后列出的来源会覆盖前面的来源 —
        # 用户插件优先于内置插件，项目插件优先于用户插件。
        # 此处进行去重，以便我们只加载最终胜出的插件。
        # 键是由路径生成（例如 ``image_gen/openai``、``disk-cleanup``），
        # 因此即便两个清单文件中的 ``name: openai`` 相同，
        # ``tts/openai`` 与 ``image_gen/openai`` 也不会发生冲突。
        disabled = _get_disabled_plugins()
        enabled = _get_enabled_plugins()  # None = opt-in default (nothing enabled)
        winners: Dict[str, PluginManifest] = {}
        for manifest in manifests:
            winners[manifest.key or manifest.name] = manifest
        for manifest in winners.values():
            lookup_key = manifest.key or manifest.name

            # Explicit disable always wins (matches on key or on legacy
            # bare name for back-compat with existing user configs).
            if lookup_key in disabled or manifest.name in disabled:
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = "disabled via config"
                self._plugins[lookup_key] = loaded
                logger.debug("Skipping disabled plugin '%s'", lookup_key)
                continue

            # Exclusive plugins (memory providers) have their own
            # discovery/activation path. The general loader records the
            # manifest for introspection but does not load the module.
            if manifest.kind == "exclusive":
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = (
                    "exclusive plugin — activate via <category>.provider config"
                )
                self._plugins[lookup_key] = loaded
                logger.debug(
                    "Skipping '%s' (exclusive, handled by category discovery)",
                    lookup_key,
                )
                continue

            # 模型提供方（Model provider）插件由 providers/__init__.py 负责加载
            # （其自身会在首次调用 get_provider_profile() 时触发延迟探索）。
            # 此处我们仅记录清单信息以供自省，但不直接导入（import）该模块 —
            # 二次导入会创建两个 ProviderProfile 实例，
            # 从而破坏内置插件与用户插件之间“最后写入者胜出”的覆盖机制。
            if manifest.kind == "model-provider":
                loaded = LoadedPlugin(manifest=manifest, enabled=True)
                self._plugins[lookup_key] = loaded
                logger.debug(
                    "Skipping '%s' (model-provider, handled by providers/ discovery)",
                    lookup_key,
                )
                continue

            # 内置后端会自动加载 — 它们随 hermes 一起发布且必须开箱即用。
            # 它们之间的选择（例如由哪个 image_gen 后端来处理调用）
            # 由 ``<category>.provider`` 配置项控制，
            # 并由工具包装器（tool wrapper）强制执行。
            if manifest.source == "bundled" and manifest.kind == "backend":
                self._load_plugin(manifest)
                continue

            # 内置的平台插件（网关适配器：telegram、discord、
            # 飞书、teams 等）采用延迟（LAZY）方式进行注册。
            # 它们的模块会在模块层级导入开销巨大且平台特定的 SDK
            # （如 lark_oapi、microsoft_teams、discord.py、slack_bolt 等），
            # 因此如果同步预加载全部约 20 个插件，会导致每次执行 `hermes` 命令时
            # 都增加几秒钟的开销 — 甚至包括完全不涉及网关平台的普通 `hermes chat` 命令。
            # 作为替代方案，我们在 platform_registry 中注册一个以平台名称为键（Key）的
            # 轻量级延迟加载器；只有当网关、定时任务（cron）、初始化设置（setup）
            # 或消息发送（send_message）路径确实请求该平台时，才会去真实导入对应的模块。
            # Hermes 随附的所有平台依然保持开箱即用 — 只是变成了首次使用时才加载。
            if manifest.source == "bundled" and manifest.kind == "platform":
                self._register_deferred_platform(manifest)
                continue

            # 其他所有内容（独立插件、用户安装的后端、
            # 入口点插件）均需通过 plugins.enabled 手动开启。
            # 同时兼容由路径生成的键（key）以及传统的纯名称，
            # 从而确保现有配置能够继续正常工作。
            is_enabled = (
                enabled is not None
                and (lookup_key in enabled or manifest.name in enabled)
            )
            if not is_enabled:
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = (
                    "not enabled in config (run `hermes plugins enable {}` to activate)"
                    .format(lookup_key)
                )
                self._plugins[lookup_key] = loaded
                logger.debug(
                    "Skipping '%s' (not in plugins.enabled)", lookup_key
                )
                continue
            self._load_plugin(manifest)

        if manifests:
            logger.info(
                "Plugin discovery complete: %d found, %d enabled",
                len(self._plugins),
                sum(1 for p in self._plugins.values() if p.enabled),
            )

    # -----------------------------------------------------------------------
    # Directory scanning
    # -----------------------------------------------------------------------

    def _scan_directory(
        self,
        path: Path,
        source: str,
        skip_names: Optional[Set[str]] = None,
    ) -> List[PluginManifest]:
        """从 *path* 的子目录中读取 ``plugin.yaml`` 清单文件。

        支持以下两种布局，且两者可混合使用：

        * **扁平结构（Flat）** — ``<root>/<plugin-name>/plugin.yaml``。
          键（Key）为 ``<plugin-name>``（例如 ``disk-cleanup``）。
        * **分类结构（Category）** — ``<root>/<category>/<plugin-name>/plugin.yaml``，
          其中 ``<category>`` 目录本身不包含 ``plugin.yaml``。
          键（Key）为 ``<category>/<plugin-name>``（例如 ``image_gen/openai``）。
          目录层级深度上限为两级。

        *skip_names* 是一个可选的忽略名称白名单，用于跳过顶层的指定目录
        （保留该参数是为了向下兼容；由于分类结构现已成为一级支持，
        目前的调用位置已不再传递此参数）。
        """
        return self._scan_directory_level(
            path, source, skip_names=skip_names, prefix="", depth=0
        )

    def _scan_directory_level(
        self,
        path: Path,
        source: str,
        *,
        skip_names: Optional[Set[str]],
        prefix: str,
        depth: int,
    ) -> List[PluginManifest]:
        """ :meth:`_scan_directory` 的递归实现。

        ``prefix`` 是已累积的分类路径（根目录下为 ""，
        下钻一层后为 "image_gen"）。``depth`` 为递归深度；
        我们将其上限设为 2，因此 ``<root>/a/b/c/`` 会被忽略。
        """
        manifests: List[PluginManifest] = []
        if not path.is_dir():
            return manifests

        for child in sorted(path.iterdir()):
            if not child.is_dir():
                continue
            if depth == 0 and skip_names and child.name in skip_names:
                continue
            manifest_file = child / "plugin.yaml"
            if not manifest_file.exists():
                manifest_file = child / "plugin.yml"

            if manifest_file.exists():
                manifest = self._parse_manifest(
                    manifest_file, child, source, prefix
                )
                if manifest is not None:
                    manifests.append(manifest)
                continue

            # No manifest at this level. If we're still within the depth
            # cap, treat this directory as a category namespace and recurse
            # one level in looking for children with manifests.
            if depth >= 1:
                logger.debug("Skipping %s (no plugin.yaml, depth cap reached)", child)
                continue

            sub_prefix = f"{prefix}/{child.name}" if prefix else child.name
            manifests.extend(
                self._scan_directory_level(
                    child,
                    source,
                    skip_names=None,
                    prefix=sub_prefix,
                    depth=depth + 1,
                )
            )

        return manifests

    def _parse_manifest(
        self,
        manifest_file: Path,
        plugin_dir: Path,
        source: str,
        prefix: str,
    ) -> Optional[PluginManifest]:
        """Parse a single ``plugin.yaml`` into a :class:`PluginManifest`.

        Returns ``None`` on parse failure (logs a warning).
        """
        try:
            if yaml is None:
                logger.warning("PyYAML not installed – cannot load %s", manifest_file)
                return None
            data = fast_safe_load(manifest_file.read_text(encoding="utf-8")) or {}

            name = data.get("name", plugin_dir.name)
            key = f"{prefix}/{plugin_dir.name}" if prefix else name

            raw_kind = data.get("kind", "standalone")
            if not isinstance(raw_kind, str):
                raw_kind = "standalone"
            kind = raw_kind.strip().lower()
            if kind not in _VALID_PLUGIN_KINDS:
                logger.warning(
                    "Plugin %s: unknown kind '%s' (valid: %s); treating as 'standalone'",
                    key, raw_kind, ", ".join(sorted(_VALID_PLUGIN_KINDS)),
                )
                kind = "standalone"

            # Auto-coerce user-installed memory providers to kind="exclusive"
            # so they're routed to plugins/memory discovery instead of being
            # loaded by the general PluginManager (which has no
            # register_memory_provider on PluginContext). Mirrors the
            # heuristic in plugins/memory/__init__.py:_is_memory_provider_dir.
            # Bundled memory providers are already skipped via skip_names.
            if kind == "standalone" and "kind" not in data:
                init_file = plugin_dir / "__init__.py"
                if init_file.exists():
                    try:
                        source_text = init_file.read_text(errors="replace")[:8192]
                        if (
                            "register_memory_provider" in source_text
                            or "MemoryProvider" in source_text
                        ):
                            kind = "exclusive"
                            logger.debug(
                                "Plugin %s: detected memory provider, "
                                "treating as kind='exclusive'",
                                key,
                            )
                        elif (
                            "register_provider" in source_text
                            and "ProviderProfile" in source_text
                        ):
                            # Model provider plugin (calls register_provider()
                            # from ``providers`` with a ProviderProfile). Route
                            # to providers/__init__.py discovery.
                            kind = "model-provider"
                            logger.debug(
                                "Plugin %s: detected model provider, "
                                "treating as kind='model-provider'",
                                key,
                            )
                    except Exception:
                        pass

            logger.debug(
                "Parsed manifest: key=%s name=%s kind=%s source=%s path=%s",
                key, name, kind, source, plugin_dir,
            )
            return PluginManifest(
                name=name,
                version=str(data.get("version", "")),
                description=data.get("description", ""),
                author=data.get("author", ""),
                requires_env=data.get("requires_env", []),
                provides_tools=data.get("provides_tools", []),
                provides_hooks=data.get("provides_hooks", []),
                source=source,
                path=str(plugin_dir),
                kind=kind,
                key=key,
            )
        except Exception as exc:
            logger.warning(
                "Failed to parse %s: %s", manifest_file, exc, exc_info=_PLUGINS_DEBUG,
            )
            return None

    # -----------------------------------------------------------------------
    # Entry-point scanning
    # -----------------------------------------------------------------------

    def _scan_entry_points(self) -> List[PluginManifest]:
        """Check ``importlib.metadata`` for pip-installed plugins."""
        manifests: List[PluginManifest] = []
        try:
            eps = importlib.metadata.entry_points()
            # Python 3.12+ returns a SelectableGroups; earlier returns dict
            if hasattr(eps, "select"):
                group_eps = eps.select(group=ENTRY_POINTS_GROUP)
            elif isinstance(eps, dict):
                group_eps = eps.get(ENTRY_POINTS_GROUP, [])
            else:
                group_eps = [ep for ep in eps if ep.group == ENTRY_POINTS_GROUP]

            for ep in group_eps:
                manifest = PluginManifest(
                    name=ep.name,
                    source="entrypoint",
                    path=ep.value,
                    key=ep.name,
                )
                manifests.append(manifest)
        except Exception as exc:
            logger.debug("Entry-point scan failed: %s", exc)

        return manifests

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------

    def _platform_name_from_manifest(self, manifest: PluginManifest) -> str:
        """Derive the gateway platform name (e.g. ``feishu``) for a platform plugin.

        The platform name registered via ``register_platform(name=...)`` lives
        inside the adapter module (which we are explicitly trying NOT to import
        early). It is not carried in ``plugin.yaml``. Across every bundled
        platform plugin the manifest name is ``<platform>-platform`` and the
        plugin directory basename is ``<platform>``, so we derive the name
        without importing: strip a trailing ``-platform`` from the manifest
        name, falling back to the directory basename. This is also a sensible
        convention for third-party platform plugins.
        """
        name = manifest.name or ""
        if name.endswith("-platform"):
            return name[: -len("-platform")]
        if manifest.path:
            return Path(manifest.path).name
        return name

    def _register_deferred_platform(self, manifest: PluginManifest) -> None:
        """Register a lazy loader for a bundled platform plugin.

        The platform adapter module is imported only when the gateway / cron /
        setup / send_message path first asks the ``platform_registry`` for this
        platform. Until then we record a lightweight ``LoadedPlugin`` so
        ``hermes plugins list`` still shows the platform as available, and we
        hand the registry a loader that runs the normal eager-load path.
        """
        lookup_key = manifest.key or manifest.name
        platform_name = self._platform_name_from_manifest(manifest)

        # Record an enabled placeholder for introspection (`hermes plugins
        # list`). The real module load swaps in a fully-populated LoadedPlugin
        # (tools/hooks/commands attribution) when the loader fires.
        loaded = LoadedPlugin(manifest=manifest, enabled=True)
        loaded.deferred = True
        self._plugins[lookup_key] = loaded

        def _loader(_manifest: PluginManifest = manifest) -> None:
            self._load_plugin(_manifest)

        try:
            from gateway.platform_registry import platform_registry

            platform_registry.register_deferred(platform_name, _loader)
            logger.debug(
                "Registered deferred platform loader: %s (plugin=%s)",
                platform_name,
                lookup_key,
            )
        except Exception:
            # If the registry import fails for any reason, fall back to eager
            # loading so the platform is never silently lost.
            logger.debug(
                "Deferred platform registration failed for '%s'; eager-loading",
                lookup_key,
                exc_info=True,
            )
            self._load_plugin(manifest)

    def _load_plugin(self, manifest: PluginManifest) -> None:
        """Import a plugin module and call its ``register(ctx)`` function."""
        loaded = LoadedPlugin(manifest=manifest)
        logger.debug(
            "Loading plugin '%s' (source=%s, kind=%s, path=%s)",
            manifest.key or manifest.name, manifest.source, manifest.kind, manifest.path,
        )

        from tools.registry import registry as _registry
        _plugin_id = manifest.key or manifest.name
        _slug = _plugin_id.replace("/", "__").replace("-", "_")
        _registry.register_plugin_override_policy(
            f"{_NS_PARENT}.{_slug}",
            PluginContext(manifest, self)._tool_override_allowed(""),
        )
        try:
            if manifest.source in {"user", "project", "bundled"}:
                module = self._load_directory_module(manifest)
            else:
                module = self._load_entrypoint_module(manifest)

            loaded.module = module

            # Call register()
            register_fn = getattr(module, "register", None)
            if register_fn is None:
                loaded.error = "no register() function"
                logger.warning("Plugin '%s' has no register() function", manifest.name)
            else:
                ctx = PluginContext(manifest, self)
                # 在执行 register() 之前快照注册表的状态，
                # 从而使各个注册表的归属统计仅包含“本插件”实际新增的内容。
                # 此前的实现方式是将名称与所有已加载的插件进行对比差异，
                # 这会导致对注册了已被早期插件使用过的 Hook / 中间件 / 工具名称的插件归属错误：
                # 重名的部分仅被归入第一个插件，
                # 从而导致后续插件在 `hermes plugins list` 中少报了注册项。
                _tools_before = set(self._plugin_tool_names)
                _hook_counts_before = {
                    h: len(cbs) for h, cbs in self._hooks.items()
                }
                _mw_counts_before = {
                    kind: len(cbs) for kind, cbs in self._middleware.items()
                }
                # Key met 例如调用disk-cleanup#def register()方法
                register_fn(ctx)
                loaded.tools_registered = [
                    t for t in self._plugin_tool_names
                    if t not in _tools_before
                ]
                loaded.hooks_registered = [
                    h
                    for h, cbs in self._hooks.items()
                    if len(cbs) > _hook_counts_before.get(h, 0)
                ]
                loaded.middleware_registered = [
                    kind
                    for kind, cbs in self._middleware.items()
                    if len(cbs) > _mw_counts_before.get(kind, 0)
                ]
                loaded.commands_registered = [
                    c for c in self._plugin_commands
                    if self._plugin_commands[c].get("plugin") == manifest.name
                ]
                loaded.enabled = True
                logger.debug(
                    "  registered: %d tool(s), %d hook(s), %d middleware, %d slash command(s), %d CLI command(s)",
                    len(loaded.tools_registered),
                    len(loaded.hooks_registered),
                    len(loaded.middleware_registered),
                    len(loaded.commands_registered),
                    sum(
                        1 for c in self._cli_commands
                        if self._cli_commands[c].get("plugin") == manifest.name
                    ),
                )

        except Exception as exc:
            loaded.error = str(exc)
            logger.warning(
                "Failed to load plugin '%s': %s",
                manifest.name, exc, exc_info=_PLUGINS_DEBUG,
            )
        self._plugins[manifest.key or manifest.name] = loaded

    def _load_directory_module(self, manifest: PluginManifest) -> types.ModuleType:
        """将基于目录的插件导入为 ``hermes_plugins.<slug>``。

        模块 slug 是基于 ``manifest.key`` 生成的，
        因此带有分类命名空间的插件（例如 ``image_gen/openai``）
        会导入为 ``hermes_plugins.image_gen__openai``，
        而不会与未来可能出现的 ``tts/openai`` 发生冲突。
        """
        plugin_dir = Path(manifest.path)  # type: ignore[arg-type]
        init_file = plugin_dir / "__init__.py"
        if not init_file.exists():
            raise FileNotFoundError(f"No __init__.py in {plugin_dir}")

        # Ensure the namespace parent package exists
        if _NS_PARENT not in sys.modules:
            ns_pkg = types.ModuleType(_NS_PARENT)
            ns_pkg.__path__ = []  # type: ignore[attr-defined]
            ns_pkg.__package__ = _NS_PARENT
            sys.modules[_NS_PARENT] = ns_pkg

        key = manifest.key or manifest.name
        slug = key.replace("/", "__").replace("-", "_")
        module_name = f"{_NS_PARENT}.{slug}"
        spec = importlib.util.spec_from_file_location(
            module_name,
            init_file,
            submodule_search_locations=[str(plugin_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {init_file}")

        module = importlib.util.module_from_spec(spec)
        module.__package__ = module_name
        module.__path__ = [str(plugin_dir)]  # type: ignore[attr-defined]
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _load_entrypoint_module(self, manifest: PluginManifest) -> types.ModuleType:
        """Load a pip-installed plugin via its entry-point reference."""
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            group_eps = eps.select(group=ENTRY_POINTS_GROUP)
        elif isinstance(eps, dict):
            group_eps = eps.get(ENTRY_POINTS_GROUP, [])
        else:
            group_eps = [ep for ep in eps if ep.group == ENTRY_POINTS_GROUP]

        for ep in group_eps:
            if ep.name == manifest.name:
                return ep.load()

        raise ImportError(
            f"Entry point '{manifest.name}' not found in group '{ENTRY_POINTS_GROUP}'"
        )

    # -----------------------------------------------------------------------
    # Hook invocation
    # -----------------------------------------------------------------------

    def invoke_hook(self, hook_name: str, **kwargs: Any) -> List[Any]:
        """调用针对 *hook_name* 注册的所有回调函数。

        每个回调都包裹在各自的 try/except 块中，因此行为异常的
        插件不会破坏核心智能体循环（core agent loop）。

        返回由回调函数返回的非 ``None`` 值所组成的列表。

        对于 ``pre_llm_call``，回调可以返回一个字典，用于描述
        要注入到当前轮次用户消息中的上下文：

            {"context": "被召回的文本..."}
            "被召回的文本..."          # 纯字符串，等效效果

        上下文总是被注入到用户消息中，绝不会注入到系统提示词（system prompt）中。
        这样可以保留提示词缓存前缀（prompt cache prefix）—— 系统提示词在
        不同轮次之间保持完全相同，从而可以复用已缓存的 Token。所有注入的
        上下文都是瞬态的（ephemeral）—— 绝不会持久化到会话数据库中。
        """
        kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
        callbacks = self._hooks.get(hook_name, [])
        results: List[Any] = []
        for cb in callbacks:
            try:
                ret = cb(**kwargs)
                if ret is not None:
                    results.append(ret)
            except Exception as exc:
                logger.warning(
                    "Hook '%s' callback %s raised: %s",
                    hook_name,
                    getattr(cb, "__name__", repr(cb)),
                    exc,
                )
        return results

    def has_hook(self, hook_name: str) -> bool:
        """Return True when at least one callback is registered for a hook."""
        return bool(self._hooks.get(hook_name))

    def has_middleware(self, kind: str) -> bool:
        """Return True when at least one callback is registered for middleware."""
        return bool(self._middleware.get(kind))

    def invoke_middleware(self, kind: str, **kwargs: Any) -> List[Any]:
        """调用针对 *kind* 注册的中间件回调。

        每个回调都是相互隔离的，因此单个插件不会破坏基础运行时
        路径。想要改变行为的中间件必须返回调用方特定契约（contract）
        所说明的数据结构。
        """
        callbacks = self._middleware.get(kind, [])
        results: List[Any] = []
        for cb in callbacks:
            try:
                ret = cb(**kwargs)
                if ret is not None:
                    results.append(ret)
            except Exception as exc:
                logger.warning(
                    "Middleware '%s' callback %s raised: %s",
                    kind,
                    getattr(cb, "__name__", repr(cb)),
                    exc,
                )
        return results

    # -----------------------------------------------------------------------
    # Slack action handler accessor
    # -----------------------------------------------------------------------

    def get_slack_action_handlers(self) -> List[tuple]:
        """Return the list of plugin-registered Slack action handlers.

        Each entry is a ``(action_id, callback, plugin_name)`` tuple.
        Consumed by the Slack adapter at connect time to wire callbacks
        into its ``slack_bolt.AsyncApp``.

        Plugins register handlers via
        :meth:`PluginContext.register_slack_action_handler`.
        """
        return list(self._slack_action_handlers)

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    def list_plugins(self) -> List[Dict[str, Any]]:
        """Return a list of info dicts for all discovered plugins."""
        result: List[Dict[str, Any]] = []
        for key, loaded in sorted(self._plugins.items()):
            result.append(
                {
                    "name": loaded.manifest.name,
                    "key": loaded.manifest.key or loaded.manifest.name,
                    "kind": loaded.manifest.kind,
                    "version": loaded.manifest.version,
                    "description": loaded.manifest.description,
                    "source": loaded.manifest.source,
                    "enabled": loaded.enabled,
                    "tools": len(loaded.tools_registered),
                    "hooks": len(loaded.hooks_registered),
                    "middleware": len(loaded.middleware_registered),
                    "commands": len(loaded.commands_registered),
                    "error": loaded.error,
                }
            )
        return result

    # -----------------------------------------------------------------------
    # Plugin skill lookups
    # -----------------------------------------------------------------------

    def find_plugin_skill(self, qualified_name: str) -> Optional[Path]:
        """Return the ``Path`` to a plugin skill's SKILL.md, or ``None``."""
        entry = self._plugin_skills.get(qualified_name)
        return entry["path"] if entry else None

    def list_plugin_skills(self, plugin_name: str) -> List[str]:
        """Return sorted bare names of all skills registered by *plugin_name*."""
        prefix = f"{plugin_name}:"
        return sorted(
            e["bare_name"]
            for qn, e in self._plugin_skills.items()
            if qn.startswith(prefix)
        )

    def remove_plugin_skill(self, qualified_name: str) -> None:
        """Remove a stale registry entry (silently ignores missing keys)."""
        self._plugin_skills.pop(qualified_name, None)


# ---------------------------------------------------------------------------
# Module-level singleton & convenience functions
# ---------------------------------------------------------------------------

_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Return (and lazily create) the global PluginManager singleton."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def discover_plugins(force: bool = False) -> None:
    """Discover and load all plugins.

    Default behavior is idempotent. Pass ``force=True`` to rescan plugin
    manifests and reload state in the current process.
    """
    get_plugin_manager().discover_and_load(force=force)


def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Invoke a lifecycle hook on all loaded plugins.

    Returns a list of non-``None`` return values from plugin callbacks.
    """
    return get_plugin_manager().invoke_hook(hook_name, **kwargs)


def invoke_middleware(kind: str, **kwargs: Any) -> List[Any]:
    """Invoke registered middleware callbacks.

    Returns a list of non-``None`` return values from middleware callbacks.
    """
    return get_plugin_manager().invoke_middleware(kind, **kwargs)


def has_middleware(kind: str) -> bool:
    """Return True when middleware callbacks are registered for ``kind``."""
    manager = get_plugin_manager()
    method = getattr(manager, "has_middleware", None)
    if callable(method):
        return bool(method(kind))
    return bool(getattr(manager, "_middleware", {}).get(kind))


def has_hook(hook_name: str) -> bool:
    """Return True when a hook has registered callbacks."""
    return get_plugin_manager().has_hook(hook_name)


_thread_tool_whitelist = threading.local()


@dataclass(frozen=True)
class _PreToolCallDirective:
    action: Optional[str] = None
    message: Optional[str] = None
    rule_key: Optional[str] = None


def set_thread_tool_whitelist(
    allowed: Optional[Set[str]],
    deny_msg_fmt: str = "Tool '{tool_name}' denied: not in this thread's tool whitelist",
) -> None:
    _thread_tool_whitelist.allowed = allowed
    _thread_tool_whitelist.fmt = deny_msg_fmt


def clear_thread_tool_whitelist() -> None:
    _thread_tool_whitelist.allowed = None


def _get_pre_tool_call_directive_details(
    tool_name: str,
    args: Optional[Dict[str, Any]],
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> _PreToolCallDirective:
    """Check ``pre_tool_call`` hooks for a blocking or approval directive.

    Plugins that need to enforce policy (rate limiting, security
    restrictions, approval workflows) can return one of::

        {"action": "block",   "message": "Reason the tool was blocked"}
        {"action": "approve", "message": "Why this needs human confirmation"}
        {"action": "approve", "message": "...", "rule_key": "write_file:ssh"}

    from their ``pre_tool_call`` callback.

    - ``block`` vetoes the tool call outright (the message becomes the tool
      result the model sees).
    - ``approve`` ESCALATES to the existing human-approval gate
      (``prompt_dangerous_approval`` on CLI, the approval callback on the
      gateway) — the same mechanism Tier-2 dangerous shell patterns use.
      This lets a plugin require a human ``[o]nce/[s]ession/[a]lways/[d]eny``
      decision on ANY tool, not just terminal command strings. The caller is
      responsible for invoking the gate (see
      :func:`tools.approval.request_tool_approval`).
    - ``rule_key`` is optional and only honored for ``approve`` directives. It
      lets plugins choose the allowlist grain for `[a]lways` approvals.

    The first valid directive wins. Invalid or irrelevant hook return values
    are silently ignored so existing observer-only hooks are unaffected.
    """
    allowed = getattr(_thread_tool_whitelist, "allowed", None)
    if allowed is not None and tool_name not in allowed:
        fmt = getattr(_thread_tool_whitelist, "fmt", "Tool '{tool_name}' denied")
        return _PreToolCallDirective(
            action="block",
            message=fmt.format(tool_name=tool_name),
        )

    hook_results = invoke_hook(
        "pre_tool_call",
        tool_name=tool_name,
        args=args if isinstance(args, dict) else {},
        task_id=task_id,
        session_id=session_id,
        tool_call_id=tool_call_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
        middleware_trace=list(middleware_trace or []),
    )

    for result in hook_results:
        if not isinstance(result, dict):
            continue
        action = result.get("action")
        if action not in ("block", "approve"):
            continue
        message = result.get("message")
        message = message if isinstance(message, str) and message else None
        # A block directive requires a message (it becomes the tool result);
        # an approve directive can carry an optional reason.
        if action == "block" and not message:
            continue
        rule_key = result.get("rule_key") if action == "approve" else None
        rule_key = rule_key.strip() if isinstance(rule_key, str) else None
        if not rule_key:
            rule_key = None
        return _PreToolCallDirective(action=action, message=message, rule_key=rule_key)

    return _PreToolCallDirective()


def get_pre_tool_call_directive(
    tool_name: str,
    args: Optional[Dict[str, Any]],
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Check ``pre_tool_call`` hooks for a blocking or approval directive.

    Backward-compatible public helper: returns ``(directive, message)`` where
    ``directive`` is ``"block"``, ``"approve"``, or ``None``. Internal callers
    that need approve-specific metadata use
    :func:`_get_pre_tool_call_directive_details`.
    """
    details = _get_pre_tool_call_directive_details(
        tool_name, args, task_id=task_id, session_id=session_id,
        tool_call_id=tool_call_id, turn_id=turn_id,
        api_request_id=api_request_id, middleware_trace=middleware_trace,
    )
    return (details.action, details.message)


def get_pre_tool_call_block_message(
    tool_name: str,
    args: Optional[Dict[str, Any]],
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Back-compat shim: return only a ``block`` message (or ``None``).

    Deprecated in favor of :func:`get_pre_tool_call_directive`, which also
    surfaces the ``approve`` escalation directive. Kept so any external caller
    importing the old name keeps working; ``approve`` directives are invisible
    to this shim (it only reports blocks).
    """
    directive, message = get_pre_tool_call_directive(
        tool_name, args, task_id=task_id, session_id=session_id,
        tool_call_id=tool_call_id, turn_id=turn_id,
        api_request_id=api_request_id, middleware_trace=middleware_trace,
    )
    return message if directive == "block" else None


def resolve_pre_tool_block(
    tool_name: str,
    args: Optional[Dict[str, Any]],
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Resolve the pre_tool_call directive to a final block message (or None).

    Single entry point for every tool-dispatch site: fetches the plugin
    directive and, for an ``approve`` escalation, invokes the human-approval
    gate (:func:`tools.approval.request_tool_approval`). Returns the message
    the tool result should carry when the call is blocked, or ``None`` when
    the call may proceed.

    Centralizing this keeps the security-critical fail-closed logic in ONE
    place instead of copy-pasted across the concurrent/sequential/helper
    dispatch paths: an ``approve`` directive whose gate errors, denies, or
    times out is fail-closed to a block; ``block`` blocks with its message;
    anything else proceeds.
    """
    details = _get_pre_tool_call_directive_details(
        tool_name, args, task_id=task_id, session_id=session_id,
        tool_call_id=tool_call_id, turn_id=turn_id,
        api_request_id=api_request_id, middleware_trace=middleware_trace,
    )
    if details.action == "block":
        return details.message
    if details.action == "approve":
        try:
            from tools.approval import request_tool_approval
            result = request_tool_approval(
                tool_name,
                details.message or "",
                rule_key=details.rule_key or tool_name,
            )
        except Exception:
            # Fail-closed: if the gate itself errors, block rather than
            # silently execute an action a plugin flagged for approval.
            return f"BLOCKED: plugin approval gate failed for {tool_name}"
        if not result.get("approved"):
            return str(
                result.get("message")
                or f"BLOCKED: plugin approval required for {tool_name}"
            )
    return None


def get_pre_verify_continue_message(
    *,
    session_id: str = "",
    platform: str = "",
    model: str = "",
    coding: bool = False,
    attempt: int = 0,
    final_response: str = "",
    changed_paths: Optional[List[str]] = None,
) -> Optional[str]:
    """Check user ``pre_verify`` hooks for a directive to keep the agent going.

    Fired once per turn when the agent edited code and is about to verify/finish.
    A hook keeps the turn going (run a check, defer it, tidy the diff) by
    returning::

        {"action": "continue", "message": "<follow-up for the model>"}

    The Claude-Code Stop shape ``{"decision": "block", "reason": "..."}`` (block
    the stop == keep going) is accepted too. The first directive carrying a
    non-empty message wins; any other return lets the turn finish. Mirrors
    :func:`get_pre_tool_call_block_message` — the call site stays a one-liner.

    ``coding`` / ``attempt`` let a hook scope itself (``if not coding`` …) and
    self-throttle (``if attempt`` …), the same way a ``pre_tool_call`` hook
    scopes on ``tool_name``.
    """
    hook_results = invoke_hook(
        "pre_verify",
        session_id=session_id,
        platform=platform,
        model=model,
        coding=coding,
        attempt=attempt,
        final_response=final_response,
        changed_paths=list(changed_paths or []),
    )

    for result in hook_results:
        if not isinstance(result, dict):
            continue
        action = str(result.get("action") or result.get("decision") or "").strip().lower()
        if action not in ("continue", "block"):
            continue
        message = result.get("message") or result.get("reason")
        if isinstance(message, str) and message.strip():
            return message.strip()

    return None


def _ensure_plugins_discovered(force: bool = False) -> PluginManager:
    """Return the global manager after ensuring plugin discovery has run.

    Pass ``force=True`` to rescan in the current process.
    """
    manager = get_plugin_manager()
    manager.discover_and_load(force=force)
    return manager


def get_plugin_context_engine():
    """Return the plugin-registered context engine, or None."""
    return _ensure_plugins_discovered()._context_engine


def get_plugin_command_handler(name: str) -> Optional[Callable]:
    """Return the handler for a plugin-registered slash command, or ``None``."""
    entry = _ensure_plugins_discovered()._plugin_commands.get(name)
    return entry["handler"] if entry else None


_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS = 30.0


def resolve_plugin_command_result(result: Any) -> Any:
    """Resolve a plugin command return value, awaiting async handlers when needed.

    Sync CLI/TUI dispatch sites call plugin handlers from plain functions.
    If a handler is async, await it directly when no loop is running; if
    we're already inside an active loop, run it in a helper thread with its
    own loop so the caller still gets a concrete result synchronously. The
    threaded path is bounded by a 30s timeout so a hung async handler cannot
    wedge the terminal indefinitely.
    """
    if not inspect.isawaitable(result):
        return result

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(result)

    outcome: Dict[str, Any] = {}
    failure: Dict[str, BaseException] = {}
    done = threading.Event()

    def _runner() -> None:
        try:
            outcome["value"] = asyncio.run(result)
        except BaseException as exc:  # pragma: no cover - re-raised below
            failure["exc"] = exc
        finally:
            done.set()

    thread = threading.Thread(
        target=_runner,
        name="hermes-plugin-command-await",
        daemon=True,
    )
    thread.start()
    if not done.wait(timeout=_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS):
        raise TimeoutError(
            "Plugin command async handler did not complete within "
            f"{_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS:.0f}s"
        )
    if "exc" in failure:
        raise failure["exc"]
    return outcome.get("value")


def get_plugin_commands() -> Dict[str, dict]:
    """Return the full plugin commands dict (name → {handler, description, plugin}).

    Triggers idempotent plugin discovery so callers can use plugin commands
    before any explicit discover_plugins() call.
    """
    return _ensure_plugins_discovered()._plugin_commands


def get_plugin_auxiliary_tasks() -> List[Dict[str, Any]]:
    """将所有由插件注册的辅助任务以稳定排序的列表形式返回。

    每个列表项都是来自
    :meth:`PluginContext.register_auxiliary_task` 的注册字典：
    ``{key, display_name, description, defaults, plugin}``。

    该方法会触发幂等的插件发现机制，
    因此调用方可以在进行任何显式 ``discover_plugins()`` 调用之前读取注册表。
    结果按 ``key`` 进行排序，以确保选择器和测试中的确定性顺序。
    """
    manager = _ensure_plugins_discovered()
    return [manager._aux_tasks[k] for k in sorted(manager._aux_tasks)]


def get_plugin_toolsets() -> List[tuple]:
    """Return plugin toolsets as ``(key, label, description)`` tuples.

    Used by the ``hermes tools`` TUI so plugin-provided toolsets appear
    alongside the built-in ones and can be toggled on/off per platform.
    """
    manager = get_plugin_manager()
    if not manager._plugin_tool_names:
        return []

    try:
        from tools.registry import registry
    except Exception:
        return []

    # Group plugin tool names by their toolset
    toolset_tools: Dict[str, List[str]] = {}
    toolset_plugin: Dict[str, LoadedPlugin] = {}
    for tool_name in manager._plugin_tool_names:
        entry = registry.get_entry(tool_name)
        if not entry:
            continue
        ts = entry.toolset
        toolset_tools.setdefault(ts, []).append(entry.name)

    # Map toolsets back to the plugin that registered them
    for _name, loaded in manager._plugins.items():
        for tool_name in loaded.tools_registered:
            entry = registry.get_entry(tool_name)
            if entry and entry.toolset in toolset_tools:
                toolset_plugin.setdefault(entry.toolset, loaded)

    result = []
    for ts_key in sorted(toolset_tools):
        plugin = toolset_plugin.get(ts_key)
        label = f"🔌 {ts_key.replace('_', ' ').title()}"
        if plugin and plugin.manifest.description:
            desc = plugin.manifest.description
        else:
            desc = ", ".join(sorted(toolset_tools[ts_key]))
        result.append((ts_key, label, desc))

    return result
