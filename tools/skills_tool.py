#!/usr/bin/env python3
"""
Skill 工具模块

本模块用于提供列出和查看 Skill 文档的工具。
Skill 通常以目录的形式进行组织，其中包含一个 SKILL.md 文件（主指令文档），
以及可选的支撑文件，如参考文档、模板和示例等。

灵感来源于 Anthropic 的 Claude Skills 系统，采用了渐进式披露（Progressive Disclosure）架构：
- 元数据（名称 ≤ 64 字符，描述 ≤ 1024 字符） - 展示在 skills_list 中
- 完整指令内容 - 需要时通过 skill_view 进行加载
- 关联文件（参考文档、模板等） - 按需加载

目录结构：
    skills/
    ├── my-skill/
    │   ├── SKILL.md           # 主指令文档（必需）
    │   ├── references/        # 支撑文档
    │   │   ├── api.md
    │   │   └── examples.md
    │   ├── templates/         # 输出模板
    │   │   └── template.md
    │   └── assets/            # 补充文件（agentskills.io 标准）
    └── category/              # 用于分类整理的类别文件夹
        └── another-skill/
            └── SKILL.md

SKILL.md 格式（YAML Frontmatter，兼容 agentskills.io）：
    ---
    name: skill-name              # 必需，最大 64 字符
    description: 简短描述          # 必需，最大 1024 字符
    version: 1.0.0                # 可选
    license: MIT                  # 可选 (agentskills.io)
    platforms: [macos]            # 可选 — 限制特定操作系统平台
                                  #   有效值：macos, linux, windows
                                  #   省略则默认在所有平台上加载
    prerequisites:                # 可选 — 旧版运行时依赖要求
      env_vars: [API_KEY]         #   旧版环境变量名在加载时会被
                                  #   归一化（Normalize）转换为 required_environment_variables。
      commands: [curl, jq]        #   命令检查仍仅作为建议性检查。
    compatibility: 需要 X         # 可选 (agentskills.io)
    metadata:                     # 可选，任意键值对 (agentskills.io)
      hermes:
        tags: [fine-tuning, llm]
        related_skills: [peft, lora]
    ---

    # Skill 标题

    此处为完整的指令和内容...

可用工具：
- skills_list：列出包含元数据的 Skill 列表（渐进式披露第 1 层）
- skill_view：加载完整 Skill 内容（渐进式披露第 2-3 层）

用法示例：
    from tools.skills_tool import skills_list, skill_view, check_skills_requirements

    # 列出所有 Skill（仅返回元数据 - 节省 Token）
    result = skills_list()

    # 查看某个 Skill 的主内容（加载完整指令）
    content = skill_view("axolotl")

    # 查看某个 Skill 内部的参考文件（加载关联文件）
    content = skill_view("axolotl", "references/dataset-formats.md")
"""

import json
import logging
import time

from hermes_constants import get_hermes_home, display_hermes_home
import os
import re
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, Any, List, Optional, Set, Tuple

from tools.registry import registry, tool_error
from hermes_cli.config import cfg_get
from utils import env_var_enabled
from agent.skill_utils import (
    EXCLUDED_SKILL_DIRS as _EXCLUDED_SKILL_DIRS,
    is_skill_support_path as _is_skill_support_path,
)

logger = logging.getLogger(__name__)

# Per-session skill discovery cache.  _find_all_skills() re-reads every
# SKILL.md on every call; with hundreds of skills this is wasteful.
# Cache validation (mirrors hermes_cli/profiles.py::_count_skills, d5eee133e):
#   - signature = per-dir max mtime of the dir AND its immediate children
#     (one scandir per dir; catches skill add/remove inside categories,
#     which does NOT bump the root dir's mtime), plus the disabled-set
#     (config-driven — changes with no filesystem mtime bump at all)
#   - a short TTL bounds staleness from in-place SKILL.md edits, which
#     bump only the file's mtime, invisible to any directory signature.
# skip_disabled True/False are cached separately.
_SKILLS_CACHE: dict = {}          # {cache_key: (signature, timestamp, skills_list)}
_SKILLS_CACHE_TTL_SECONDS = 30.0
_SKILLS_CACHE_KEY_DISABLED = "with_disabled"
_SKILLS_CACHE_KEY_FILTERED = "filtered"


def _skills_scan_signature(dirs_to_scan, disabled) -> tuple:
    """Cheap change-signature for the skill scan inputs.

    O(#dirs + #categories) stat calls, not a recursive walk. Includes the
    platform the scan's ``skill_matches_platform`` filter will use (read
    from ``agent.skill_utils``'s ``sys`` so test patches of that module
    are honored) — the scan result is platform-dependent.
    """
    from agent import skill_utils as _skill_utils

    platform = getattr(getattr(_skill_utils, "sys", None), "platform", "")
    sig = []
    for d in dirs_to_scan:
        try:
            m = d.stat().st_mtime
        except OSError:
            continue
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            em = entry.stat(follow_symlinks=False).st_mtime
                            if em > m:
                                m = em
                    except OSError:
                        continue
        except OSError:
            pass
        sig.append((str(d), m))
    return (tuple(sig), frozenset(disabled), platform)


# All skills live in ~/.hermes/skills/ (seeded from bundled skills/ on install).
# This is the single source of truth -- agent edits, hub installs, and bundled
# skills all coexist here without polluting the git repo.
HERMES_HOME = get_hermes_home()
SKILLS_DIR = HERMES_HOME / "skills"
_SKILLS_DIR_AT_IMPORT = SKILLS_DIR


def _skills_dir() -> Path:
    """返回在调用时活跃配置（Active Profile）下的 Skill 目录。

    一些长久运行的运行时（Runtimes）会在活跃配置设置 HERMES_HOME 之前导入此模块。
    此处保留旧版 SKILLS_DIR 模块属性以兼容测试和外部补丁（External Patchers）；
    但当其未被打补丁时，每次调用均会从当前活跃的配置作用域下的 HERMES_HOME 中动态解析。
    """
    configured = Path(SKILLS_DIR)
    if configured != _SKILLS_DIR_AT_IMPORT:
        return configured
    return get_hermes_home() / "skills"


# Anthropic-recommended limits for progressive disclosure efficiency
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024

# Platform identifiers for the 'platforms' frontmatter field.
# Maps user-friendly names to sys.platform prefixes.
_PLATFORM_MAP = {
    "macos": "darwin",
    "linux": "linux",
    "windows": "win32",
}
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REMOTE_ENV_BACKENDS = frozenset(
    {"docker", "singularity", "modal", "ssh", "daytona"}
)
_secret_capture_callback = None


def _skill_lookup_path_error(name: str) -> Optional[str]:
    """如果本地 Skill 查询名称 *name* 能够越界跳出搜索根目录，则返回错误。

    Skill 的名称 ``name`` 会与每个受信任的搜索目录进行拼接，
    以构建磁盘上的查找路径，因此它必须保持为相对路径且不得包含 ``..`` 片段 ——
    否则 ``name="../outside"`` 或绝对路径可能会选中 Skill 目录之外的 Skill
    （并读取其中的文件）。此处的校验与后续通过 ``tools.path_security``
    对 ``file_path`` 进行的校验相匹配。同时，我们还会拒绝 Windows 驱动器路径
    （例如 ``C:\\skills``），否则其包含的 ``:`` 会被误读为插件命名空间的分隔符。
    """
    from tools.path_security import has_traversal_component

    if not isinstance(name, str):
        return "Skill name must be a string."
    candidate = name.strip()
    if (
        PurePosixPath(candidate).is_absolute()
        or PureWindowsPath(candidate).is_absolute()
        or PureWindowsPath(candidate).drive
    ):
        return "Skill name must be a relative path within the skills directory."
    if has_traversal_component(candidate):
        return "Skill name cannot contain '..' path traversal components."
    return None


def load_env() -> Dict[str, str]:
    """Load profile-scoped environment variables from HERMES_HOME/.env."""
    env_path = get_hermes_home() / ".env"
    env_vars: Dict[str, str] = {}
    if not env_path.exists():
        return env_vars

    with env_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                if line.startswith("export "):
                    line = line[7:]
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip().strip("\"'")
    return env_vars


class SkillReadinessStatus(str, Enum):
    AVAILABLE = "available"
    SETUP_NEEDED = "setup_needed"
    UNSUPPORTED = "unsupported"


# Prompt injection detection — shared by local-skill and plugin-skill paths.
_INJECTION_PATTERNS: list = [
    "ignore previous instructions",
    "ignore all previous",
    "you are now",
    "disregard your",
    "forget your instructions",
    "new instructions:",
    "system prompt:",
    "<system>",
    "]]>",
]


def set_secret_capture_callback(callback) -> None:
    global _secret_capture_callback
    _secret_capture_callback = callback


def skill_matches_platform(frontmatter: Dict[str, Any]) -> bool:
    """检查 Skill 是否与当前操作系统平台兼容。

    委派给 ``agent.skill_utils.skill_matches_platform`` 处理 ——
    此处保留作为公共接口的重新导出（re-export），
    以便现有的调用方无需更改。
    """
    from agent.skill_utils import skill_matches_platform as _impl
    return _impl(frontmatter)


def skill_matches_environment(frontmatter: Dict[str, Any]) -> bool:
    """检查 Skill 是否与当前的运行时环境相关。

    委派给 ``agent.skill_utils.skill_matches_environment`` 处理 ——
    此处保留作为公共接口的重新导出（re-export），
    以便现有的调用方无需更改。
    这属于主动推荐时的相关性门禁（如 kanban/docker/s6），
    而**非**强兼容性限制；
    显式加载 Skill 可以绕过此限制。
    """
    from agent.skill_utils import skill_matches_environment as _impl
    return _impl(frontmatter)


def _normalize_prerequisite_values(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(item) for item in value if str(item).strip()]


def _collect_prerequisite_values(
    frontmatter: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    prereqs = frontmatter.get("prerequisites")
    if not prereqs or not isinstance(prereqs, dict):
        return [], []
    return (
        _normalize_prerequisite_values(prereqs.get("env_vars")),
        _normalize_prerequisite_values(prereqs.get("commands")),
    )


def _normalize_setup_metadata(frontmatter: Dict[str, Any]) -> Dict[str, Any]:
    setup = frontmatter.get("setup")
    if not isinstance(setup, dict):
        return {"help": None, "collect_secrets": []}

    help_text = setup.get("help")
    normalized_help = (
        str(help_text).strip()
        if isinstance(help_text, str) and help_text.strip()
        else None
    )

    collect_secrets_raw = setup.get("collect_secrets")
    if isinstance(collect_secrets_raw, dict):
        collect_secrets_raw = [collect_secrets_raw]
    if not isinstance(collect_secrets_raw, list):
        collect_secrets_raw = []

    collect_secrets: List[Dict[str, Any]] = []
    for item in collect_secrets_raw:
        if not isinstance(item, dict):
            continue

        env_var = str(item.get("env_var") or "").strip()
        if not env_var:
            continue

        prompt = str(item.get("prompt") or f"Enter value for {env_var}").strip()
        provider_url = str(item.get("provider_url") or item.get("url") or "").strip()

        entry: Dict[str, Any] = {
            "env_var": env_var,
            "prompt": prompt,
            "secret": bool(item.get("secret", True)),
        }
        if provider_url:
            entry["provider_url"] = provider_url
        collect_secrets.append(entry)

    return {
        "help": normalized_help,
        "collect_secrets": collect_secrets,
    }


def _get_required_environment_variables(
    frontmatter: Dict[str, Any],
    legacy_env_vars: List[str] | None = None,
) -> List[Dict[str, Any]]:
    setup = _normalize_setup_metadata(frontmatter)
    required_raw = frontmatter.get("required_environment_variables")
    if isinstance(required_raw, dict):
        required_raw = [required_raw]
    if not isinstance(required_raw, list):
        required_raw = []

    required: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _append_required(entry: Dict[str, Any]) -> None:
        env_name = str(entry.get("name") or entry.get("env_var") or "").strip()
        if not env_name or env_name in seen:
            return
        if not _ENV_VAR_NAME_RE.match(env_name):
            return

        normalized: Dict[str, Any] = {
            "name": env_name,
            "prompt": str(entry.get("prompt") or f"Enter value for {env_name}").strip(),
        }

        help_text = (
            entry.get("help")
            or entry.get("provider_url")
            or entry.get("url")
            or setup.get("help")
        )
        if isinstance(help_text, str) and help_text.strip():
            normalized["help"] = help_text.strip()

        required_for = entry.get("required_for")
        if isinstance(required_for, str) and required_for.strip():
            normalized["required_for"] = required_for.strip()

        if entry.get("optional"):
            normalized["optional"] = True

        seen.add(env_name)
        required.append(normalized)

    for item in required_raw:
        if isinstance(item, str):
            _append_required({"name": item})
            continue
        if isinstance(item, dict):
            _append_required(item)

    for item in setup["collect_secrets"]:
        _append_required(
            {
                "name": item.get("env_var"),
                "prompt": item.get("prompt"),
                "help": item.get("provider_url") or setup.get("help"),
            }
        )

    if legacy_env_vars is None:
        legacy_env_vars, _ = _collect_prerequisite_values(frontmatter)
    for env_var in legacy_env_vars:
        _append_required({"name": env_var})

    return required


def _capture_required_environment_variables(
    skill_name: str,
    missing_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not missing_entries:
        return {
            "missing_names": [],
            "setup_skipped": False,
            "gateway_setup_hint": None,
        }

    missing_names = [entry["name"] for entry in missing_entries]
    # 大多数网关界面（如消息平台）无法直接弹窗请求输入密钥，
    # 因此它们会直接回退触发“不支持”的提示。
    # 而交互式网关界面——例如桌面应用或终端用户界面（TUI）——
    # 会设置 HERMES_INTERACTIVE 标志，
    # 并注册一个捕获密钥的回调函数，
    # 该回调会路由至安全的 secret.request 覆盖层（overlay），
    # 从而跳过回退逻辑，真正向用户发起输入请求。
    # （HERMES_INTERACTIVE 与 tools/approval.py 中使用的标志相同，
    # 用于区分交互式界面与消息接收界面。）
    if _is_gateway_surface() and not env_var_enabled("HERMES_INTERACTIVE"):
        return {
            "missing_names": missing_names,
            "setup_skipped": False,
            "gateway_setup_hint": _gateway_setup_hint(),
        }

    if _secret_capture_callback is None:
        return {
            "missing_names": missing_names,
            "setup_skipped": False,
            "gateway_setup_hint": None,
        }

    setup_skipped = False
    remaining_names: List[str] = []

    for entry in missing_entries:
        metadata = {"skill_name": skill_name}
        if entry.get("help"):
            metadata["help"] = entry["help"]
        if entry.get("required_for"):
            metadata["required_for"] = entry["required_for"]

        try:
            callback_result = _secret_capture_callback(
                entry["name"],
                entry["prompt"],
                metadata,
            )
        except Exception:
            logger.warning(
                f"Secret capture callback failed for {entry['name']}", exc_info=True
            )
            callback_result = {
                "success": False,
                "stored_as": entry["name"],
                "validated": False,
                "skipped": True,
            }

        success = isinstance(callback_result, dict) and bool(
            callback_result.get("success")
        )
        skipped = isinstance(callback_result, dict) and bool(
            callback_result.get("skipped")
        )
        if success and not skipped:
            continue

        setup_skipped = True
        remaining_names.append(entry["name"])

    return {
        "missing_names": remaining_names,
        "setup_skipped": setup_skipped,
        "gateway_setup_hint": None,
    }


def _is_gateway_surface() -> bool:
    if env_var_enabled("HERMES_GATEWAY_SESSION"):
        return True
    from gateway.session_context import get_session_env
    return bool(get_session_env("HERMES_SESSION_PLATFORM"))


def _get_terminal_backend_name() -> str:
    return str(os.getenv("TERMINAL_ENV", "local")).strip().lower() or "local"


def _is_env_var_persisted(
    var_name: str, env_snapshot: Dict[str, str] | None = None
) -> bool:
    if env_snapshot is None:
        env_snapshot = load_env()
    if var_name in env_snapshot:
        return bool(env_snapshot.get(var_name))
    return bool(os.getenv(var_name))


def _remaining_required_environment_names(
    required_env_vars: List[Dict[str, Any]],
    capture_result: Dict[str, Any],
    *,
    env_snapshot: Dict[str, str] | None = None,
) -> List[str]:
    missing_names = set(capture_result["missing_names"])

    if env_snapshot is None:
        env_snapshot = load_env()
    remaining = []
    for entry in required_env_vars:
        name = entry["name"]
        if entry.get("optional"):
            continue
        if name in missing_names or not _is_env_var_persisted(name, env_snapshot):
            remaining.append(name)
    return remaining


def _gateway_setup_hint() -> str:
    try:
        from gateway.platforms.base import GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE

        return GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE
    except Exception:
        return f"Secure secret entry is not available. Load this skill in the local CLI to be prompted, or add the key to {display_hermes_home()}/.env manually."


def _build_setup_note(
    readiness_status: SkillReadinessStatus,
    missing: List[str],
    setup_help: str | None = None,
) -> str | None:
    if readiness_status == SkillReadinessStatus.SETUP_NEEDED:
        missing_str = ", ".join(missing) if missing else "required prerequisites"
        note = f"Setup needed before using this skill: missing {missing_str}."
        if setup_help:
            return f"{note} {setup_help}"
        return note
    return None


def check_skills_requirements() -> bool:
    """Skills are always available -- the directory is created on first use if needed."""
    return True


def _parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """从 Markdown 内容中解析 YAML Frontmatter（前置元数据）。

    委派给 ``agent.skill_utils.parse_frontmatter`` 处理 ——
    此处保留作为公共接口的重新导出（re-export），
    以便现有的调用方无需更改。
    """
    from agent.skill_utils import parse_frontmatter
    return parse_frontmatter(content)


def _get_category_from_path(skill_path: Path) -> Optional[str]:
    """
    根据目录结构从 Skill 路径中提取分类（category）。

    对于形如 ~/.hermes/skills/mlops/axolotl/SKILL.md 的路径 -> 提取出 "mlops"
    该方法同样适用于通过 skills.external_dirs 配置的外部 Skill 目录。
    """
    # 优先尝试当前活跃配置文件的 Skill 目录（适配测试中的 Monkeypatch 机制），
    # 若未匹配，则回退并尝试配置文件中的外部目录。
    dirs_to_check = [_skills_dir()]
    try:
        from agent.skill_utils import get_external_skills_dirs
        dirs_to_check.extend(get_external_skills_dirs())
    except Exception:
        pass
    for skills_dir in dirs_to_check:
        try:
            rel_path = skill_path.relative_to(skills_dir)
            parts = rel_path.parts
            if len(parts) >= 3:
                return parts[0]
        except ValueError:
            continue
    return None


def _parse_tags(tags_value) -> List[str]:
    """
    Parse tags from frontmatter value.

    Handles:
    - Already-parsed list (from yaml.safe_load): [tag1, tag2]
    - String with brackets: "[tag1, tag2]"
    - Comma-separated string: "tag1, tag2"

    Args:
        tags_value: Raw tags value — may be a list or string

    Returns:
        List of tag strings
    """
    if not tags_value:
        return []

    # yaml.safe_load already returns a list for [tag1, tag2]
    if isinstance(tags_value, list):
        return [str(t).strip() for t in tags_value if t]

    # String fallback — handle bracket-wrapped or comma-separated
    tags_value = str(tags_value).strip()
    if tags_value.startswith("[") and tags_value.endswith("]"):
        tags_value = tags_value[1:-1]

    return [t.strip().strip("\"'") for t in tags_value.split(",") if t.strip()]



def _get_disabled_skill_names() -> Set[str]:
    """Load disabled skill names from config.

    Delegates to ``agent.skill_utils.get_disabled_skill_names`` — kept here
    as a public re-export so existing callers don't need updating.
    """
    from agent.skill_utils import get_disabled_skill_names
    return get_disabled_skill_names()


def _get_session_platform() -> str:
    """Resolve the current platform from gateway session context.

    Mirrors the platform-resolution logic in
    ``agent.skill_utils.get_disabled_skill_names`` so that
    ``_is_skill_disabled`` respects ``HERMES_SESSION_PLATFORM``.
    """
    try:
        from gateway.session_context import get_session_env
        return get_session_env("HERMES_SESSION_PLATFORM") or ""
    except Exception:
        return ""


def _is_skill_disabled(name: str, platform: str = None) -> bool:
    """Check if a skill is disabled in config.

    Resolves the active platform from (in order of precedence):
    1. Explicit ``platform`` argument
    2. ``HERMES_PLATFORM`` environment variable
    3. ``HERMES_SESSION_PLATFORM`` from gateway session context
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        skills_cfg = config.get("skills", {})
        resolved_platform = platform or os.getenv("HERMES_PLATFORM") or _get_session_platform()
        global_disabled = skills_cfg.get("disabled", [])
        if resolved_platform:
            platform_disabled = cfg_get(skills_cfg, "platform_disabled", resolved_platform)
            if platform_disabled is not None:
                # A globally-disabled skill stays disabled on every platform;
                # the platform list adds to it rather than replacing it. Keep
                # in sync with agent.skill_utils.get_disabled_skill_names.
                return name in platform_disabled or name in global_disabled
        return name in global_disabled
    except Exception:
        return False


def _find_all_skills(*, skip_disabled: bool = False) -> List[Dict[str, Any]]:
    """递归查找 ~/.hermes/skills/ 及外部目录中的所有 Skill。

    参数：
        skip_disabled: 若为 True，则忽略禁用状态返回所有 Skill
            （供 ``hermes skills`` 配置界面使用）。
            默认为 False，会自动过滤掉已禁用的 Skill。

    返回：
        Skill 元数据字典的列表（包含 name、description、category）。

    结果按会话进行缓存；当扫描特征发生变化
    （如目录/分类的 mtime 修改时间改变，或禁用集合变更）时，
    缓存会自动失效，且缓存包含较短的 TTL 存活时间，
    以限制因原地修改 SKILL.md 所带来的数据滞后问题。
    """
    from agent.skill_utils import get_external_skills_dirs, iter_skill_index_files

    cache_key = _SKILLS_CACHE_KEY_DISABLED if skip_disabled else _SKILLS_CACHE_KEY_FILTERED

    # 一次性加载已禁用 Skill 集合（而非按每个 Skill 逐个加载）。
    # 这是缓存特征（cache signature）的一部分：
    # 禁用某项 Skill 属于配置变更，不会触发文件系统的修改时间（mtime）变动。
    disabled = set() if skip_disabled else _get_disabled_skill_names()

    # 收集要扫描的目录 —— 解析逻辑与下方的扫描循环保持一致
    # （_skills_dir() 会解析当前活跃配置文件的 HERMES_HOME；
    # 模块级别的 SKILLS_DIR 在长生命周期的运行时中可能会陈旧过期）。
    dirs_to_scan: list = []
    active_skills_dir = _skills_dir()
    if active_skills_dir.exists():
        dirs_to_scan.append(active_skills_dir)
    dirs_to_scan.extend(get_external_skills_dirs())

    signature = _skills_scan_signature(dirs_to_scan, disabled)
    now = time.monotonic()

    cached = _SKILLS_CACHE.get(cache_key)
    if (
        cached is not None
        and cached[0] == signature
        and (now - cached[1]) < _SKILLS_CACHE_TTL_SECONDS
    ):
        # Per-call shallow copies: callers mutate the returned dicts
        # (e.g. web_server annotates s["enabled"]/s["usage"]) — handing
        # out the cached objects would poison the cache for everyone else.
        return [dict(s) for s in cached[2]]

    skills = []
    seen_names: set = set()

    # 先扫描本地目录，再扫描外部目录（本地优先）——
    # 上文生成特征签名时，已将 dirs_to_scan 解析完毕。
    for scan_dir in dirs_to_scan:
        for skill_md in iter_skill_index_files(scan_dir, "SKILL.md"):
            if any(part in _EXCLUDED_SKILL_DIRS for part in skill_md.parts):
                continue

            skill_dir = skill_md.parent

            try:
                content = skill_md.read_text(encoding="utf-8")[:4000]
                frontmatter, body = _parse_frontmatter(content)

                if not skill_matches_platform(frontmatter):
                    continue

                if not skill_matches_environment(frontmatter):
                    continue

                name = frontmatter.get("name", skill_dir.name)[:MAX_NAME_LENGTH]
                if name in seen_names:
                    continue
                if name in disabled:
                    continue

                description = frontmatter.get("description", "")
                if not description:
                    for line in body.strip().split("\n"):
                        line = line.strip()
                        if line and not line.startswith("#"):
                            description = line
                            break

                if len(description) > MAX_DESCRIPTION_LENGTH:
                    description = description[:MAX_DESCRIPTION_LENGTH - 3] + "..."

                category = _get_category_from_path(skill_md)

                seen_names.add(name)
                skills.append({
                    "name": name,
                    "description": description,
                    "category": category,
                })

            except (UnicodeDecodeError, PermissionError) as e:
                logger.debug("Failed to read skill file %s: %s", skill_md, e)
                continue
            except Exception as e:
                logger.debug(
                    "Skipping skill at %s: failed to parse: %s", skill_md, e, exc_info=True
                )
                continue

    # 使用在扫描前计算出的特征签名作为键存入缓存
    # （如果在扫描过程中发生并发写入，则特征签名会被改变，
    # 从而确保下次调用会重新扫描，而不是在 TTL 期间提供被损坏/不完整的结果）。
    # 遵循与缓存命中路径相同的浅拷贝约定 —— 调用方可以对其进行修改。
    _SKILLS_CACHE[cache_key] = (signature, now, skills)
    return [dict(s) for s in skills]


def _sort_skills(skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep every skill listing path ordered the same way."""
    return sorted(skills, key=lambda s: (s.get("category") or "", s["name"]))


def skills_list(category: str = None, task_id: str = None) -> str:
    """列出所有可用的 Skill（渐进式披露第 1 级 —— 最小化元数据）。

    仅返回名称与描述以最小化 Token 消耗。
    如需加载完整内容、标签、关联文件等信息，请使用 skill_view()。

    参数：
        category: 可选的分类过滤器（例如 "mlops"）
        task_id: 可选的任务标识符，用于探测当前激活的后端

    返回：
        包含 Skill 最小化信息的 JSON 字符串：name、description、category
    """
    try:
        active_skills_dir = _skills_dir()
        if not active_skills_dir.exists():
            active_skills_dir.mkdir(parents=True, exist_ok=True)
            return json.dumps(
                {
                    "success": True,
                    "skills": [],
                    "categories": [],
                    "message": f"No skills found. Skills directory created at {display_hermes_home()}/skills/",
                },
                ensure_ascii=False,
            )

        # Find all skills
        all_skills = _find_all_skills()

        if not all_skills:
            return json.dumps(
                {
                    "success": True,
                    "skills": [],
                    "categories": [],
                    "message": "No skills found in skills/ directory.",
                },
                ensure_ascii=False,
            )

        # Filter by category if specified
        if category:
            all_skills = [s for s in all_skills if s.get("category") == category]

        # Sort by category then name
        all_skills = _sort_skills(all_skills)

        # Extract unique categories
        categories = sorted(
            {s.get("category") for s in all_skills if s.get("category")}
        )

        return json.dumps(
            {
                "success": True,
                "skills": all_skills,
                "categories": categories,
                "count": len(all_skills),
                "hint": "Use skill_view(name) to see full content, tags, and linked files",
            },
            ensure_ascii=False,
        )

    except Exception as e:
        return tool_error(str(e), success=False)


# ── Plugin skill serving ──────────────────────────────────────────────────


def _serve_plugin_skill(
    skill_md: Path,
    namespace: str,
    bare: str,
    *,
    preprocess: bool = True,
    session_id: str | None = None,
) -> str:
    """Read a plugin-provided skill, apply guards, return JSON."""
    from hermes_cli.plugins import _get_disabled_plugins, get_plugin_manager

    if namespace in _get_disabled_plugins():
        return json.dumps(
            {
                "success": False,
                "error": (
                    f"Plugin '{namespace}' is disabled. "
                    f"Re-enable with: hermes plugins enable {namespace}"
                ),
            },
            ensure_ascii=False,
        )

    try:
        content = skill_md.read_text(encoding="utf-8")
    except Exception as e:
        return json.dumps(
            {"success": False, "error": f"Failed to read skill '{namespace}:{bare}': {e}"},
            ensure_ascii=False,
        )

    parsed_frontmatter: Dict[str, Any] = {}
    try:
        parsed_frontmatter, _ = _parse_frontmatter(content)
    except Exception:
        pass

    if not skill_matches_platform(parsed_frontmatter):
        return json.dumps(
            {
                "success": False,
                "error": f"Skill '{namespace}:{bare}' is not supported on this platform.",
                "readiness_status": SkillReadinessStatus.UNSUPPORTED.value,
            },
            ensure_ascii=False,
        )

    # Injection scan — log but still serve (matches local-skill behaviour)
    if any(p in content.lower() for p in _INJECTION_PATTERNS):
        logger.warning(
            "Plugin skill '%s:%s' contains patterns that may indicate prompt injection",
            namespace, bare,
        )

    description = str(parsed_frontmatter.get("description", ""))
    if len(description) > MAX_DESCRIPTION_LENGTH:
        description = description[: MAX_DESCRIPTION_LENGTH - 3] + "..."

    # Bundle context banner — tells the agent about sibling skills
    try:
        siblings = [
            s for s in get_plugin_manager().list_plugin_skills(namespace)
            if s != bare
        ]
        if siblings:
            sib_list = ", ".join(siblings)
            banner = (
                f"[Bundle context: This skill is part of the '{namespace}' plugin.\n"
                f"Sibling skills: {sib_list}.\n"
                f"Use qualified form to invoke siblings (e.g. {namespace}:{siblings[0]}).]\n\n"
            )
        else:
            banner = f"[Bundle context: This skill is part of the '{namespace}' plugin.]\n\n"
    except Exception:
        banner = ""

    rendered_content = content
    if preprocess:
        try:
            from agent.skill_preprocessing import preprocess_skill_content

            rendered_content = preprocess_skill_content(
                content,
                skill_md.parent,
                session_id=session_id,
            )
        except Exception:
            logger.debug(
                "Could not preprocess plugin skill %s:%s", namespace, bare, exc_info=True
            )

    return json.dumps(
        {
            "success": True,
            "name": f"{namespace}:{bare}",
            "content": f"{banner}{rendered_content}" if banner else rendered_content,
            "description": description,
            "linked_files": None,
            "readiness_status": SkillReadinessStatus.AVAILABLE.value,
        },
        ensure_ascii=False,
    )


def skill_view(
    name: str,
    file_path: str = None,
    task_id: str = None,
    preprocess: bool = True,
) -> str:
    """查看某个 Skill 的内容或该 Skill 目录下的指定文件。

    参数：
        name: Skill 的名称或路径（例如 "axolotl" 或 "03-fine-tuning/axolotl"）。
            形如 "plugin:skill" 的限定名称，将解析为由插件提供的 Skill。
        file_path: Skill 内部指定文件的可选路径（例如 "references/api.md"）。
        task_id: 用于探测当前活跃后端的可选任务标识符。
        preprocess: 将已配置的 SKILL.md 模板与内联 Shell 渲染应用至主 Skill 内容。
            内部斜杠命令/预加载调用方会禁用此功能，
            因为它们会自行渲染 Skill 消息。

    返回：
        包含 Skill 内容或错误信息的 JSON 字符串。
    """
    try:
        # 在进行带 ':' 的限定名称（qualified-name）分发之前执行校验，
        # 以防 Windows 驱动器路径（例如 C:\skills\foo）被误解为插件命名空间（plugin namespace），
        # 并确保路径遍历或绝对名称永远不会传给下方用于构建 direct_path 的搜索目录拼接逻辑。
        lookup_error = _skill_lookup_path_error(name)
        if lookup_error:
            return json.dumps(
                {
                    "success": False,
                    "error": lookup_error,
                    "hint": "Use a skill name or relative path within the skills directory.",
                },
                ensure_ascii=False,
            )

        local_category_name: str | None = None
        # ── 限定名称分发（插件 Skill）──────────────────
        # 包含 ':' 的名称将被路由到插件 Skill 注册表。
        # 纯名称（不含 ':'）则下沉转入下方的平铺树扫描流程。
        if ":" in name:
            from agent.skill_utils import is_valid_namespace, parse_qualified_name
            from hermes_cli.plugins import discover_plugins, get_plugin_manager

            namespace, bare = parse_qualified_name(name)
            if not is_valid_namespace(namespace):
                return json.dumps(
                    {
                        "success": False,
                        "error": (
                            f"Invalid namespace '{namespace}' in '{name}'. "
                            f"Namespaces must match [a-zA-Z0-9_-]+."
                        ),
                    },
                    ensure_ascii=False,
                )

            discover_plugins()  # idempotent
            pm = get_plugin_manager()
            plugin_skill_md = pm.find_plugin_skill(name)

            if plugin_skill_md is not None:
                if not plugin_skill_md.exists():
                    # Stale registry entry — file deleted out of band
                    pm.remove_plugin_skill(name)
                    return json.dumps(
                        {
                            "success": False,
                            "error": (
                                f"Skill '{name}' file no longer exists at "
                                f"{plugin_skill_md}. The registry entry has "
                                f"been cleaned up — try again after the "
                                f"plugin is reloaded."
                            ),
                        },
                        ensure_ascii=False,
                    )
                return _serve_plugin_skill(
                    plugin_skill_md,
                    namespace,
                    bare,
                    preprocess=preprocess,
                    session_id=task_id,
                )

            # Plugin exists but this specific skill is missing?
            available = pm.list_plugin_skills(namespace)
            if available:
                return json.dumps(
                    {
                        "success": False,
                        "error": f"Skill '{bare}' not found in plugin '{namespace}'.",
                        "available_skills": [f"{namespace}:{s}" for s in available],
                        "hint": f"The '{namespace}' plugin provides {len(available)} skill(s).",
                    },
                    ensure_ascii=False,
                )
            # Plugin itself not found — fall through to flat-tree scan.
            # Categorized local skills also use `category:skill` in config and
            # gateway prompts, so preserve that form and translate it to the
            # on-disk `category/skill` path during the local scan below.
            if bare:
                local_category_name = f"{namespace}/{bare}"

        from agent.skill_utils import get_external_skills_dirs

        # 分类后的回退形式（命名空间/纯名称）也会拼接至各个
        # 搜索目录；请重新校验该路径，因为 `bare`（纯名称）未经过命名空间检查。
        if local_category_name:
            lookup_error = _skill_lookup_path_error(local_category_name)
            if lookup_error:
                return json.dumps(
                    {
                        "success": False,
                        "error": lookup_error,
                        "hint": "Use a skill name or relative path within the skills directory.",
                    },
                    ensure_ascii=False,
                )

        # Build list of all skill directories to search
        all_dirs = []
        active_skills_dir = _skills_dir()
        if active_skills_dir.exists():
            all_dirs.append(active_skills_dir)
        all_dirs.extend(get_external_skills_dirs())

        if not all_dirs:
            return json.dumps(
                {
                    "success": False,
                    "error": "Skills directory does not exist yet. It will be created on first install.",
                },
                ensure_ascii=False,
            )

        skill_dir = None
        skill_md = None

        # 碰撞检测：跨越所有目录，利用每一种查找策略
        # （直接路径、按父目录名递归、旧版平铺 <name>.md）收集所有的候选项。
        # 如果匹配到了多个候选，则拒绝处理并告知调用方 ——
        # 同名的外部 Skill 静默遮蔽（Shadowing）本地 Skill 属于一种极其真实的 Bug 类型
        # （例如 `/skills` 显示的是这一个，但 Agent 实际加载的却是另一个），
        # 因此我们选择明确报错暴露问题，而非盲目进行猜测。
        from agent.skill_utils import iter_skill_index_files

        candidates: List[Tuple[Optional[Path], Path]] = []  # (skill_dir, skill_md)
        seen_md: set = set()

        def _record(sd: Optional[Path], smd: Path) -> None:
            try:
                key = smd.resolve()
            except Exception:
                key = smd
            if key in seen_md:
                return
            seen_md.add(key)
            candidates.append((sd, smd))

        for search_dir in all_dirs:
            # 策略 1：直接路径
            # （例如："mlops/axolotl"，
            # 或者是位于目录顶层单独的 "axolotl"）。
            direct_path = search_dir / name
            if (
                not _is_skill_support_path(direct_path)
                and direct_path.is_dir()
                and (direct_path / "SKILL.md").exists()
            ):
                _record(direct_path, direct_path / "SKILL.md")
            elif direct_path.with_suffix(".md").exists() and not _is_skill_support_path(
                direct_path.with_suffix(".md")
            ):
                _record(None, direct_path.with_suffix(".md"))

            # 策略 1b：插件命名空间降级回退的分类形式
            # （例如：对于未注册任何插件的名称 "myplugin:explore"，
            # 也会尝试寻找磁盘路径 "myplugin/explore"）。
            if local_category_name:
                categorized_path = search_dir / local_category_name
                if (
                    not _is_skill_support_path(categorized_path)
                    and categorized_path.is_dir()
                    and (categorized_path / "SKILL.md").exists()
                ):
                    _record(categorized_path, categorized_path / "SKILL.md")
                elif categorized_path.with_suffix(
                    ".md"
                ).exists() and not _is_skill_support_path(
                    categorized_path.with_suffix(".md")
                ):
                    _record(None, categorized_path.with_suffix(".md"))

            # 策略 2：按目录名递归搜索
            # （捕获通过简短名称调用的嵌套 Skill，
            # 例如 "foundations/runtime/explore-codebase"），
            # 另外再加上 Frontmatter 中的 `name:` 查找。
            # 由于 `skills_list()` 会暴露 Frontmatter 中定义的名称，
            # 因此即使磁盘上的目录使用的是较短的分类名称或别名，
            # `skill_view(name)` 也必须能够接受该名称。
            for found_skill_md in iter_skill_index_files(search_dir, "SKILL.md"):
                if found_skill_md.parent.name == name:
                    _record(found_skill_md.parent, found_skill_md)
                    continue
                try:
                    fm_content = found_skill_md.read_text(encoding="utf-8")
                    fm, _ = _parse_frontmatter(fm_content)
                except Exception:
                    fm = {}
                if fm.get("name") == name:
                    _record(found_skill_md.parent, found_skill_md)

            # 策略 3：位于该目录下任意位置的旧版扁平 <name>.md 文件。
            # 排除 Skill 的支持文档：
            # references/templates/assets/scripts 目录下的文件
            # 是通过 skill_view(skill, file_path=...) 加载的，
            # 绝不能遮蔽（shadow）或碰撞（collide）共享相同基准名称（basename）的真实 Skill。
            for found_md in search_dir.rglob(f"{name}.md"):
                if found_md.name != "SKILL.md" and not _is_skill_support_path(
                    found_md
                ):
                    _record(None, found_md)

        if len(candidates) > 1:
            paths = [str(smd) for _, smd in candidates]
            logging.getLogger(__name__).warning(
                "Skill name collision for '%s': %d candidates — %s",
                name, len(candidates), "; ".join(paths),
            )
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        f"Ambiguous skill name '{name}': {len(candidates)} skills "
                        "match across your local skills dir and external_dirs. "
                        "Refusing to guess — load one explicitly by its categorized path."
                    ),
                    "matches": paths,
                    "hint": (
                        "Pass the full relative path instead of the bare name "
                        "(e.g., 'category/skill-name'), or rename one of the "
                        "colliding skills so each name is unique."
                    ),
                },
                ensure_ascii=False,
            )

        if candidates:
            skill_dir, skill_md = candidates[0]

        if not skill_md or not skill_md.exists():
            available = [s["name"] for s in _sort_skills(_find_all_skills())[:20]]
            return json.dumps(
                {
                    "success": False,
                    "error": f"Skill '{name}' not found.",
                    "available_skills": available,
                    "hint": "Use skills_list to see all available skills",
                },
                ensure_ascii=False,
            )

        # Read the file once — reused for platform check and main content below
        try:
            content = skill_md.read_text(encoding="utf-8")
        except Exception as e:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Failed to read skill '{name}': {e}",
                },
                ensure_ascii=False,
            )

        # 安全性提醒：如果 Skill 从受信任目录之外加载，则发出警告
        # （本地 skills 目录与配置的 external_dirs 均属于受信任目录）
        _outside_skills_dir = True
        _trusted_dirs = [active_skills_dir.resolve()]
        try:
            _trusted_dirs.extend(d.resolve() for d in all_dirs[1:])
        except Exception:
            pass
        for _td in _trusted_dirs:
            try:
                skill_md.resolve().relative_to(_td)
                _outside_skills_dir = False
                break
            except ValueError:
                continue

        # 安全性检测：检测常见的提示词注入（prompt injection）模式
        # （模式列表在模块层级定义为 _INJECTION_PATTERNS）
        _content_lower = content.lower()
        _injection_detected = any(p in _content_lower for p in _INJECTION_PATTERNS)

        if _outside_skills_dir or _injection_detected:
            _warnings = []
            if _outside_skills_dir:
                _warnings.append(f"skill file is outside the trusted skills directory (~/.hermes/skills/): {skill_md}")
            if _injection_detected:
                _warnings.append("skill content contains patterns that may indicate prompt injection")
            logging.getLogger(__name__).warning("Skill security warning for '%s': %s", name, "; ".join(_warnings))

        parsed_frontmatter: Dict[str, Any] = {}
        try:
            parsed_frontmatter, _ = _parse_frontmatter(content)
        except Exception:
            parsed_frontmatter = {}

        if not skill_matches_platform(parsed_frontmatter):
            return json.dumps(
                {
                    "success": False,
                    "error": f"Skill '{name}' is not supported on this platform.",
                    "readiness_status": SkillReadinessStatus.UNSUPPORTED.value,
                },
                ensure_ascii=False,
            )

        # Check if the skill is disabled by the user
        resolved_name = parsed_frontmatter.get("name", skill_md.parent.name)
        if _is_skill_disabled(resolved_name):
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        f"Skill '{resolved_name}' is disabled. "
                        "Enable it with `hermes skills` or inspect the files directly on disk."
                    ),
                },
                ensure_ascii=False,
            )

        # 如果指定请求了具体的文件路径，则改为读取该路径的文件
        if file_path and skill_dir:
            from tools.path_security import validate_within_dir, has_traversal_component

            # 安全性防护：防止路径穿越攻击（path traversal attacks）
            if has_traversal_component(file_path):
                return json.dumps(
                    {
                        "success": False,
                        "error": "Path traversal ('..') is not allowed.",
                        "hint": "Use a relative path within the skill directory",
                    },
                    ensure_ascii=False,
                )

            target_file = skill_dir / file_path

            # Security: Verify resolved path is still within skill directory
            traversal_error = validate_within_dir(target_file, skill_dir)
            if traversal_error:
                return json.dumps(
                    {
                        "success": False,
                        "error": traversal_error,
                        "hint": "Use a relative path within the skill directory",
                    },
                    ensure_ascii=False,
                )
            if not target_file.exists():
                # List available files in the skill directory, organized by type
                available_files = {
                    "references": [],
                    "templates": [],
                    "assets": [],
                    "scripts": [],
                    "other": [],
                }

                # Scan for all readable files
                for f in skill_dir.rglob("*"):
                    if f.is_file() and f.name != "SKILL.md":
                        rel = str(f.relative_to(skill_dir))
                        if rel.startswith("references/"):
                            available_files["references"].append(rel)
                        elif rel.startswith("templates/"):
                            available_files["templates"].append(rel)
                        elif rel.startswith("assets/"):
                            available_files["assets"].append(rel)
                        elif rel.startswith("scripts/"):
                            available_files["scripts"].append(rel)
                        elif f.suffix in {
                            ".md",
                            ".py",
                            ".yaml",
                            ".yml",
                            ".json",
                            ".tex",
                            ".sh",
                        }:
                            available_files["other"].append(rel)

                # Remove empty categories
                available_files = {k: v for k, v in available_files.items() if v}

                return json.dumps(
                    {
                        "success": False,
                        "error": f"File '{file_path}' not found in skill '{name}'.",
                        "available_files": available_files,
                        "hint": "Use one of the available file paths listed above",
                    },
                    ensure_ascii=False,
                )

            # Read the file content
            try:
                content = target_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Binary file - return info about it instead
                return json.dumps(
                    {
                        "success": True,
                        "name": name,
                        "file": file_path,
                        "content": f"[Binary file: {target_file.name}, size: {target_file.stat().st_size} bytes]",
                        "is_binary": True,
                    },
                    ensure_ascii=False,
                )

            try:
                from tools.skill_manager_tool import mark_background_review_skill_read

                mark_background_review_skill_read(target_file)
            except Exception:
                logger.debug(
                    "Could not record background-review skill read for %s",
                    target_file,
                    exc_info=True,
                )

            return json.dumps(
                {
                    "success": True,
                    "name": name,
                    "file": file_path,
                    "content": content,
                    "file_type": target_file.suffix,
                },
                ensure_ascii=False,
            )

        # Reuse the parse from the platform check above
        frontmatter = parsed_frontmatter

        # Get reference, template, asset, and script files if this is a directory-based skill
        reference_files = []
        template_files = []
        asset_files = []
        script_files = []

        if skill_dir:
            references_dir = skill_dir / "references"
            if references_dir.exists():
                reference_files = [
                    str(f.relative_to(skill_dir)) for f in references_dir.glob("*.md")
                ]

            templates_dir = skill_dir / "templates"
            if templates_dir.exists():
                for ext in [
                    "*.md",
                    "*.py",
                    "*.yaml",
                    "*.yml",
                    "*.json",
                    "*.tex",
                    "*.sh",
                ]:
                    template_files.extend(
                        [
                            str(f.relative_to(skill_dir))
                            for f in templates_dir.rglob(ext)
                        ]
                    )

            # assets/ — agentskills.io standard directory for supplementary files
            assets_dir = skill_dir / "assets"
            if assets_dir.exists():
                for f in assets_dir.rglob("*"):
                    if f.is_file():
                        asset_files.append(str(f.relative_to(skill_dir)))

            scripts_dir = skill_dir / "scripts"
            if scripts_dir.exists():
                for ext in ["*.py", "*.sh", "*.bash", "*.js", "*.ts", "*.rb"]:
                    script_files.extend(
                        [str(f.relative_to(skill_dir)) for f in scripts_dir.glob(ext)]
                    )

        # 读取 tags/related_skills 并保持向下兼容（backward compat）：
        # 优先检查 metadata.hermes.*（agentskills.io 规范），
        # 如果不存在则降级回退（fall back）到顶层字段
        hermes_meta = {}
        metadata = frontmatter.get("metadata")
        if isinstance(metadata, dict):
            hermes_meta = metadata.get("hermes", {}) or {}

        tags = _parse_tags(hermes_meta.get("tags") or frontmatter.get("tags", ""))
        related_skills = _parse_tags(
            hermes_meta.get("related_skills") or frontmatter.get("related_skills", "")
        )

        # Build linked files structure for clear discovery
        linked_files = {}
        if reference_files:
            linked_files["references"] = reference_files
        if template_files:
            linked_files["templates"] = template_files
        if asset_files:
            linked_files["assets"] = asset_files
        if script_files:
            linked_files["scripts"] = script_files

        try:
            rel_path = str(skill_md.relative_to(active_skills_dir))
        except ValueError:
            # External skill — use path relative to the skill's own parent dir
            rel_path = str(skill_md.relative_to(skill_md.parent.parent)) if skill_md.parent.parent else skill_md.name
        skill_name = frontmatter.get(
            "name", skill_md.stem if not skill_dir else skill_dir.name
        )
        legacy_env_vars, _ = _collect_prerequisite_values(frontmatter)
        required_env_vars = _get_required_environment_variables(
            frontmatter, legacy_env_vars
        )
        backend = _get_terminal_backend_name()
        env_snapshot = load_env()
        missing_required_env_vars = [
            e
            for e in required_env_vars
            if not e.get("optional")
            and not _is_env_var_persisted(e["name"], env_snapshot)
        ]
        capture_result = _capture_required_environment_variables(
            skill_name,
            missing_required_env_vars,
        )
        if missing_required_env_vars:
            env_snapshot = load_env()
        remaining_missing_required_envs = _remaining_required_environment_names(
            required_env_vars,
            capture_result,
            env_snapshot=env_snapshot,
        )
        setup_needed = bool(remaining_missing_required_envs)

        # 注册可用的 Skill 环境变量，
        # 以便将它们传递到沙箱化执行环境中（如 execute_code、terminal）。
        # 只有实际已设置的变量会被注册 ——
        # 未设置的变量将被报告为 setup_needed。
        available_env_names = [
            e["name"]
            for e in required_env_vars
            if e["name"] not in remaining_missing_required_envs
        ]
        if available_env_names:
            try:
                from tools.env_passthrough import register_env_passthrough

                register_env_passthrough(available_env_names)
            except Exception:
                logger.debug(
                    "Could not register env passthrough for skill %s",
                    skill_name,
                    exc_info=True,
                )

        # 注册凭据文件，以便挂载到远程沙箱
        # （如 Modal、Docker）中。
        # 主机上已存在的文件会被注册；
        # 缺失的文件则会被添加到 setup_needed 状态指标中。
        required_cred_files_raw = frontmatter.get("required_credential_files", [])
        if not isinstance(required_cred_files_raw, list):
            required_cred_files_raw = []
        missing_cred_files: list = []
        if required_cred_files_raw:
            try:
                from tools.credential_files import register_credential_files

                missing_cred_files = register_credential_files(required_cred_files_raw)
                if missing_cred_files:
                    setup_needed = True
            except Exception:
                logger.debug(
                    "Could not register credential files for skill %s",
                    skill_name,
                    exc_info=True,
                )

        rendered_content = content
        if preprocess:
            try:
                from agent.skill_preprocessing import preprocess_skill_content

                rendered_content = preprocess_skill_content(
                    content,
                    skill_dir,
                    session_id=task_id,
                )
            except Exception:
                logger.debug(
                    "Could not preprocess skill content for %s", skill_name, exc_info=True
                )

        result = {
            "success": True,
            "name": skill_name,
            "description": frontmatter.get("description", ""),
            "tags": tags,
            "related_skills": related_skills,
            "content": rendered_content,
            "path": rel_path,
            "skill_dir": str(skill_dir) if skill_dir else None,
            "linked_files": linked_files if linked_files else None,
            "usage_hint": "To view linked files, call skill_view(name, file_path) where file_path is e.g. 'references/api.md' or 'assets/config.yaml'"
            if linked_files
            else None,
            "required_environment_variables": required_env_vars,
            "required_commands": [],
            "missing_required_environment_variables": remaining_missing_required_envs,
            "missing_credential_files": missing_cred_files,
            "missing_required_commands": [],
            "setup_needed": setup_needed,
            "setup_skipped": capture_result["setup_skipped"],
            "readiness_status": SkillReadinessStatus.SETUP_NEEDED.value
            if setup_needed
            else SkillReadinessStatus.AVAILABLE.value,
        }

        setup_help = next((e["help"] for e in required_env_vars if e.get("help")), None)
        if setup_help:
            result["setup_help"] = setup_help

        if capture_result["gateway_setup_hint"]:
            result["gateway_setup_hint"] = capture_result["gateway_setup_hint"]

        try:
            from tools.skill_manager_tool import mark_background_review_skill_read

            mark_background_review_skill_read(skill_md)
        except Exception:
            logger.debug(
                "Could not record background-review skill read for %s",
                skill_md,
                exc_info=True,
            )

        if setup_needed:
            missing_items = [
                f"env ${env_name}" for env_name in remaining_missing_required_envs
            ] + [
                f"file {path}" for path in missing_cred_files
            ]
            setup_note = _build_setup_note(
                SkillReadinessStatus.SETUP_NEEDED,
                missing_items,
                setup_help,
            )
            if backend in _REMOTE_ENV_BACKENDS and setup_note:
                setup_note = f"{setup_note} {backend.upper()}-backed skills need these requirements available inside the remote environment as well."
            if setup_note:
                result["setup_note"] = setup_note

        # Surface agentskills.io optional fields when present
        if frontmatter.get("compatibility"):
            result["compatibility"] = frontmatter["compatibility"]
        if isinstance(metadata, dict):
            result["metadata"] = metadata

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return tool_error(str(e), success=False)




if __name__ == "__main__":
    """Test the skills tool"""
    print("🎯 Skills Tool Test")
    print("=" * 60)

    # Test listing skills
    print("\n📋 Listing all skills:")
    result = json.loads(skills_list())
    if result["success"]:
        print(
            f"Found {result['count']} skills in {len(result.get('categories', []))} categories"
        )
        print(f"Categories: {result.get('categories', [])}")
        print("\nFirst 10 skills:")
        for skill in result["skills"][:10]:
            cat = f"[{skill['category']}] " if skill.get("category") else ""
            print(f"  • {cat}{skill['name']}: {skill['description'][:60]}...")
    else:
        print(f"Error: {result['error']}")

    # Test viewing a skill
    print("\n📖 Viewing skill 'axolotl':")
    result = json.loads(skill_view("codex"))
    if result["success"]:
        print(f"Name: {result['name']}")
        print(f"Description: {result.get('description', 'N/A')[:100]}...")
        print(f"Content length: {len(result['content'])} chars")
        if result.get("linked_files"):
            print(f"Linked files: {result['linked_files']}")
    else:
        print(f"Error: {result['error']}")

    # Test viewing a reference file
    print("\n📄 Viewing reference file 'axolotl/references/dataset-formats.md':")
    result = json.loads(skill_view("axolotl", "references/dataset-formats.md"))
    if result["success"]:
        print(f"File: {result['file']}")
        print(f"Content length: {len(result['content'])} chars")
        print(f"Preview: {result['content'][:150]}...")
    else:
        print(f"Error: {result['error']}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# SKILLS_LIST_SCHEMA = {
#     "name": "skills_list",
#     "description": "列出所有可用的 Skill（包含名称和描述）。若要加载完整内容，请使用 skill_view(name)。",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "category": {
#                 "type": "string",
#                 "description": "可选的分类过滤器，用于缩小搜索结果范围",
#             }
#         },
#         "required": [],
#     },
# }
#
# SKILL_VIEW_SCHEMA = {
#     "name": "skill_view",
#     "description": "Skill 用于加载特定任务和工作流的相关信息，以及脚本和模板。该工具可加载 Skill 的完整内容，或访问其关联的文件（参考文档、模板、脚本）。首次调用将返回 SKILL.md 的内容，以及展示可用的参考文档/模板/脚本的 'linked_files' 字典。若要访问这些关联文件，请携带 file_path 参数再次调用。",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "name": {
#                 "type": "string",
#                 "description": "Skill 的名称（可使用 skills_list 查看所有可用 Skill）。对于插件提供的 Skill，请使用限定名称格式 'plugin:skill'（例如 'superpowers:writing-plans'）。",
#             },
#             "file_path": {
#                 "type": "string",
#                 "description": "可选参数：Skill 内部关联文件的相对路径（例如 'references/api.md'、'templates/config.yaml'、'scripts/validate.py'）。省略此参数则获取主要的 SKILL.md 内容。",
#             },
#         },
#         "required": ["name"],
#     },
# }
SKILLS_LIST_SCHEMA = {
    "name": "skills_list",
    "description": "List available skills (name + description). Use skill_view(name) to load full content.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Optional category filter to narrow results",
            }
        },
        "required": [],
    },
}

SKILL_VIEW_SCHEMA = {
    "name": "skill_view",
    "description": "Skills allow for loading information about specific tasks and workflows, as well as scripts and templates. Load a skill's full content or access its linked files (references, templates, scripts). First call returns SKILL.md content plus a 'linked_files' dict showing available references/templates/scripts. To access those, call again with file_path parameter.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The skill name (use skills_list to see available skills). For plugin-provided skills, use the qualified form 'plugin:skill' (e.g. 'superpowers:writing-plans').",
            },
            "file_path": {
                "type": "string",
                "description": "OPTIONAL: Path to a linked file within the skill (e.g., 'references/api.md', 'templates/config.yaml', 'scripts/validate.py'). Omit to get the main SKILL.md content.",
            },
        },
        "required": ["name"],
    },
}

registry.register(
    name="skills_list",
    toolset="skills",
    schema=SKILLS_LIST_SCHEMA,
    handler=lambda args, **kw: skills_list(
        category=args.get("category"), task_id=kw.get("task_id")
    ),
    check_fn=check_skills_requirements,
    emoji="📚",
)
def _skill_view_with_bump(args, **kw):
    """Invoke skill_view, then bump view_count on success. Best-effort: a
    telemetry failure never breaks the tool call."""
    name = args.get("name", "")
    result = skill_view(
        name, file_path=args.get("file_path"), task_id=kw.get("task_id")
    )
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict) and parsed.get("success"):
            # Use the resolved skill name from the payload when present —
            # qualified forms ("plugin:skill") return with the canonical name.
            resolved = parsed.get("name") or name
            if resolved:
                from tools.skill_usage import bump_use, bump_view
                bump_view(str(resolved))
                # A skill_view tool call is the agent actively loading the skill
                # to act on it — that counts as use, not just a browse/view.
                # Curator's stale timer keys off last_used_at (see agent/curator.py).
                bump_use(str(resolved))
    except Exception:
        pass
    return result


registry.register(
    name="skill_view",
    toolset="skills",
    schema=SKILL_VIEW_SCHEMA,
    handler=_skill_view_with_bump,
    check_fn=check_skills_requirements,
    emoji="📚",
)
