"""Environment variable passthrough registry.

Skills that declare ``required_environment_variables`` in their frontmatter
need those vars available in sandboxed execution environments (execute_code,
terminal).  By default both sandboxes strip secrets from the child process
environment for security.  This module provides a session-scoped allowlist
so skill-declared vars (and user-configured overrides) pass through.

Two sources feed the allowlist:

1. **Skill declarations** — when a skill is loaded via ``skill_view``, its
   ``required_environment_variables`` are registered here automatically.
2. **User config** — ``terminal.env_passthrough`` in config.yaml lets users
   explicitly allowlist vars for non-skill use cases.

Both ``code_execution_tool.py`` and ``tools/environments/local.py`` consult
:func:`is_env_passthrough` before stripping a variable.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Iterable
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)

# Session-scoped set of env var names that should pass through to sandboxes.
# Backed by ContextVar to prevent cross-session data bleed in the gateway pipeline.
_allowed_env_vars_var: ContextVar[set[str]] = ContextVar("_allowed_env_vars")


def _get_allowed() -> set[str]:
    """Get or create the allowed env vars set for the current context/session."""
    try:
        return _allowed_env_vars_var.get()
    except LookupError:
        val: set[str] = set()
        _allowed_env_vars_var.set(val)
        return val


# Cache for the config-based allowlist (loaded once per process).
_config_passthrough: frozenset[str] | None = None


def _is_hermes_provider_credential(name: str) -> bool:
    """如果依据 ``_HERMES_PROVIDER_ENV_BLOCKLIST`` 判定，
    ``name`` 属于 Hermes 管理的提供商凭据（API 密钥、Token 或类似凭据），
    则返回 True。

    Skill 声明的 ``required_environment_variables`` Frontmatter
    绝不能覆盖此列表 —— 这正是 GHSA-rhgp-j443-p4rf 中的绕过漏洞：
    恶意 Skill 将 ``ANTHROPIC_TOKEN`` / ``OPENAI_API_KEY`` 注册为透传变量，
    从而在 ``execute_code`` 子进程中接收到了该凭据，
    破坏了沙箱的凭据清理（scrubbing）保证。

    非 Hermes 的 API 密钥（如 TENOR_API_KEY、NOTION_TOKEN 等）
    并不在黑名单中，依然可以合法注册 ——
    封装第三方 API 的 Skill 仍可正常工作。

    故障收紧（Fail closed）：如果无法导入权威黑名单
    （如部分安装、导入期错误等），
    我们将该名称视为受保护的提供商凭据并拒绝透传，
    而不是故障放开（fall open）导致 Skill 将 Hermes 凭据隧穿透传至 execute_code 子进程。
    """
    try:
        from tools.environments.local import (
            _HERMES_PROVIDER_ENV_BLOCKLIST,
            _is_hermes_internal_secret,
        )
    except Exception as e:
        logger.warning(
            "env passthrough: provider credential blocklist import failed; "
            "failing closed and refusing passthrough registration for %r: %s",
            name,
            e,
        )
        return True
    # 动态生成的 Hermes 内部密钥
    # （辅助侧 LLM 凭据 AUXILIARY_*_API_KEY / _BASE_URL，
    # 以及网关中继认证密钥 GATEWAY_RELAY_*）
    # 均属于静态黑名单无法穷举列出的提供商凭据 ——
    # 它们是在网关启动时根据每个任务/中继动态注入的。
    # Skill 绝不能将它们注册为透传变量，
    # 并将其隧穿透传至 execute_code / terminal 子进程中。
    if _is_hermes_internal_secret(name):
        return True
    return name in _HERMES_PROVIDER_ENV_BLOCKLIST


def register_env_passthrough(var_names: Iterable[str]) -> None:
    """将环境变量名称注册为沙箱环境中允许使用的变量。

    通常在 Skill 声明 ``required_environment_variables`` 时被调用。

    对于属于 Hermes 管理的提供商凭据的变量
    （来自 ``_HERMES_PROVIDER_ENV_BLOCKLIST``），
    此处会予以拒绝，以维护 GHSA-rhgp-j443-p4rf 规范下
    ``execute_code`` 沙箱对凭据清理（credential-scrubbing）的安全性保证。
    如果某个 Skill 需要与 Hermes 管理的提供商通信，
    应当通过 Agent 主进程的工具（如 web_search、web_extract 等）来进行，
    这样凭据就可以安全地保留在主进程中。dd

    非 Hermes 的第三方 API 密钥（如 TENOR_API_KEY、NOTION_TOKEN 等）
    可以正常通过 —— 它们从来不在沙箱的清理列表中。
    """
    for name in var_names:
        name = name.strip()
        if not name:
            continue
        if _is_hermes_provider_credential(name):
            logger.warning(
                "env passthrough: refusing to register Hermes provider "
                "credential %r (blocked by _HERMES_PROVIDER_ENV_BLOCKLIST). "
                "Skills must not override the execute_code sandbox's "
                "credential scrubbing; see GHSA-rhgp-j443-p4rf.",
                name,
            )
            continue
        _get_allowed().add(name)
        logger.debug("env passthrough: registered %s", name)


def _load_config_passthrough() -> frozenset[str]:
    """Load ``tools.env_passthrough`` from config.yaml (cached)."""
    global _config_passthrough
    if _config_passthrough is not None:
        return _config_passthrough

    result: set[str] = set()
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        passthrough = cfg_get(cfg, "terminal", "env_passthrough")
        if isinstance(passthrough, list):
            for item in passthrough:
                if not isinstance(item, str) or not item.strip():
                    continue
                name = item.strip()
                # Mirror the skill-path filter in register_env_passthrough:
                # Hermes-managed provider credentials must not be passed
                # through to execute_code / terminal children, regardless of
                # whether the request came from a skill or from config.yaml.
                # See GHSA-rhgp-j443-p4rf.
                if _is_hermes_provider_credential(name):
                    logger.warning(
                        "env passthrough: refusing to register Hermes "
                        "provider credential %r from config.yaml (blocked "
                        "by _HERMES_PROVIDER_ENV_BLOCKLIST). Operator "
                        "configuration must not override the execute_code "
                        "sandbox's credential scrubbing; see "
                        "GHSA-rhgp-j443-p4rf.",
                        name,
                    )
                    continue
                result.add(name)
    except Exception as e:
        logger.debug("Could not read tools.env_passthrough from config: %s", e)

    _config_passthrough = frozenset(result)
    return _config_passthrough


def is_env_passthrough(var_name: str) -> bool:
    """Check whether *var_name* is allowed to pass through to sandboxes.

    Returns ``True`` if the variable was registered by a skill or listed in
    the user's ``tools.env_passthrough`` config.
    """
    if var_name in _get_allowed():
        return True
    return var_name in _load_config_passthrough()


def get_all_passthrough() -> frozenset[str]:
    """Return the union of skill-registered and config-based passthrough vars."""
    return frozenset(_get_allowed()) | _load_config_passthrough()


def clear_env_passthrough() -> None:
    """Reset the skill-scoped allowlist (e.g. on session reset)."""
    _get_allowed().clear()


