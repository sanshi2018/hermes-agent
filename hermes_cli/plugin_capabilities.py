# 插件能力声明 + 授权同意状态（#64228）。
#
# 将分散的、按插件划分的信任门控（`plugins.entries.<id>.allow_*`）
# 统一为一个声明式的、可进行差异对比的**能力模型**，
# 并在安装或更新时要求用户授权同意。
#
# **这不是一个沙箱。**
# 进程内的 Python 插件仍然是受信任的代码——
# 恶意插件可以导入任何内容、对核心代码进行猴子补丁（monkey-patch），
# 并无视上述所有限制。
# 能力（Capabilities）管控的是 Hermes 提供给插件的 *宿主 API 接口面*
# （即决定哪些注册会成功，哪些 `ctx` 方法处于激活状态），
# 从而为用户提供真实的授权同意流程与审计跟踪记录。
# 实际的代码隔离机制属于另一个独立的研究方向。
#
# ## 规范注册表
#
# 每个能力 ID 都与执行面上**已存在**的信任门控进行 1:1 映射。
# 我们刻意避免创建没有对应执行门控的能力 ID：
#
# | 能力 ID (Capability id) | 旧版配置门控 (`plugins.entries.<id>.…`) |
# | :--- | :--- |
# | `tools.override` | `allow_tool_override` |
# | `llm.provider_override` | `llm.allow_provider_override` |
# | `llm.model_override` | `llm.allow_model_override` |
# | `llm.agent_id_override` | `llm.allow_agent_id_override` |
# | `llm.profile_override` | `llm.allow_profile_override` |
# | `llm.task_override` | `llm.allow_task_override` |
# | `gateway.platform_actions` | `allow_platform_actions` |
#
# 旧版的 `allow_*` 键会按原样继续生效（已被废弃但仍受支持）：
# 当旧版键为 true **或者** 相应能力被授予时，门控即为开启状态。
#
# ## 授权同意状态
#
# 存储在插件的配置项下：
#
#     plugins:
#       entries:
#         <plugin_id>:
#           granted_capabilities: [tools.override]
#           capabilities_consent:
#             hash: "<用户授权同意时声明的能力集合的 sha256 哈希值>"
#             granted_at: "2026-08-12T00:00:00+00:00"
#
# 该哈希值记录了用户在授权同意时*所看到的内容*。
# 当插件更新后，其声明的能力集合的哈希值若发生变化，
# 新增的能力将保持未授权状态，
# 直到用户重新授权同意（`hermes plugins update` 命令会展示这些差异）。
#
# 基本原则：所有功能默认**关闭 (OFF)**。
# 任何读取授权同意状态的失败（如配置缺失、YAML 损坏、类型错误），
# 均被视为**未授权 (not granted)**。

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CapabilitySpec:
    """One declarable capability and the legacy gate it maps to."""

    id: str
    # Path of the deprecated boolean under ``plugins.entries.<plugin_id>``,
    # e.g. ("allow_tool_override",) or ("llm", "allow_model_override").
    legacy_path: Tuple[str, ...]
    # One-line risk description shown on the consent screen.
    description: str


# Canonical registry — ONLY capabilities with an existing enforcing surface.
CAPABILITY_REGISTRY: Dict[str, CapabilitySpec] = {
    spec.id: spec
    for spec in (
        CapabilitySpec(
            id="tools.override",
            legacy_path=("allow_tool_override",),
            description=(
                "Replace built-in tools (e.g. shell_exec, write_file) — an "
                "override can intercept everything routed through that tool"
            ),
        ),
        CapabilitySpec(
            id="llm.provider_override",
            legacy_path=("llm", "allow_provider_override"),
            description=(
                "Run host-owned LLM calls against a provider other than your "
                "active one (uses your credentials)"
            ),
        ),
        CapabilitySpec(
            id="llm.model_override",
            legacy_path=("llm", "allow_model_override"),
            description=(
                "Choose which model host-owned LLM calls use (spend follows "
                "the chosen model)"
            ),
        ),
        CapabilitySpec(
            id="llm.agent_id_override",
            legacy_path=("llm", "allow_agent_id_override"),
            description="Attribute its LLM calls to a different agent id",
        ),
        CapabilitySpec(
            id="llm.profile_override",
            legacy_path=("llm", "allow_profile_override"),
            description="Run LLM calls under a different auth profile",
        ),
        CapabilitySpec(
            id="llm.task_override",
            legacy_path=("llm", "allow_task_override"),
            description=(
                "Route its LLM calls through the host's built-in auxiliary "
                "task lanes"
            ),
        ),
        CapabilitySpec(
            id="gateway.platform_actions",
            legacy_path=("allow_platform_actions",),
            description=(
                "Act on connected chat platforms as the gateway bot "
                "(add reactions, rename threads) via ctx.platform_actions"
            ),
        ),
    )
}

VALID_CAPABILITY_IDS = frozenset(CAPABILITY_REGISTRY)

# Config keys under ``plugins.entries.<plugin_id>``.
GRANTED_KEY = "granted_capabilities"
CONSENT_KEY = "capabilities_consent"


# ---------------------------------------------------------------------------
# Declaration parsing
# ---------------------------------------------------------------------------

def parse_declared_capabilities(raw: Any, plugin_name: str = "?") -> List[str]:
    """Normalize a manifest ``capabilities:`` value into known capability ids.

    Unknown ids are dropped with a warning (forward compat: a plugin built
    for a newer Hermes may declare ids this build doesn't know; they can
    never be granted here, so hiding them from the consent screen is the
    fail-closed choice — the plugin must degrade gracefully).
    """
    if not raw:
        return []
    if not isinstance(raw, (list, tuple)):
        logger.warning(
            "Plugin %s: manifest 'capabilities' must be a list, got %s — ignoring",
            plugin_name, type(raw).__name__,
        )
        return []
    out: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            logger.warning(
                "Plugin %s: ignoring non-string capability entry %r",
                plugin_name, item,
            )
            continue
        cap = item.strip()
        if cap in VALID_CAPABILITY_IDS:
            if cap not in out:
                out.append(cap)
        else:
            logger.warning(
                "Plugin %s: unknown capability %r (known: %s) — ignoring",
                plugin_name, cap, ", ".join(sorted(VALID_CAPABILITY_IDS)),
            )
    return out


def capability_set_hash(capabilities: Iterable[str]) -> str:
    """Deterministic sha256 over a capability set (order-insensitive)."""
    canon = "\n".join(sorted(set(capabilities)))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Consent state (read side — fail closed on ANY error)
# ---------------------------------------------------------------------------

def _plugin_entry(plugin_id: str, config: Optional[Mapping[str, Any]] = None) -> dict:
    """Return ``plugins.entries.<plugin_id>`` or ``{}`` — never raises."""
    try:
        cfg: Any = config
        if cfg is None:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
        entries = (cfg.get("plugins") or {}).get("entries") or {}
        entry = entries.get(plugin_id) or {}
        return entry if isinstance(entry, dict) else {}
    except Exception:
        # Ground rule: failure to read consent state = not granted.
        return {}


def granted_capabilities(
    plugin_id: str, config: Optional[Mapping[str, Any]] = None
) -> frozenset:
    """Return the set of capabilities the user has granted this plugin.

    Fail-closed: missing/corrupt state yields the empty set.
    """
    entry = _plugin_entry(plugin_id, config)
    raw = entry.get(GRANTED_KEY)
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(
        c.strip() for c in raw
        if isinstance(c, str) and c.strip() in VALID_CAPABILITY_IDS
    )


def _legacy_gate_set(entry: Mapping[str, Any], spec: CapabilitySpec) -> bool:
    """True when the deprecated ``allow_*`` key for *spec* is truthy."""
    node: Any = entry
    for part in spec.legacy_path:
        if not isinstance(node, Mapping):
            return False
        node = node.get(part)
    return bool(node) and node is not None


def plugin_capability_granted(
    plugin_id: str,
    capability: str,
    config: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Canonical check: is *capability* live for *plugin_id*?

    True when EITHER:

    * the capability appears in ``granted_capabilities`` (consent flow), OR
    * the legacy ``allow_*`` config key is set (deprecated, still honored so
      existing configs keep working).

    Unknown capability ids and any failure to read state return ``False``
    (ground rule 4: fail closed).
    """
    spec = CAPABILITY_REGISTRY.get(capability)
    if spec is None:
        logger.debug(
            "capability check for unknown id %r (plugin %s) — denied",
            capability, plugin_id,
        )
        return False
    entry = _plugin_entry(plugin_id, config)
    if capability in granted_capabilities(plugin_id, config={"plugins": {"entries": {plugin_id: entry}}}):
        _log_capability_decision(plugin_id, capability, True, "granted_capabilities")
        return True
    if _legacy_gate_set(entry, spec):
        _log_capability_decision(
            plugin_id, capability, True,
            f"legacy key plugins.entries.{plugin_id}.{'.'.join(spec.legacy_path)} (deprecated)",
        )
        return True
    _log_capability_decision(plugin_id, capability, False, "not granted")
    return False


def _log_capability_decision(
    plugin_id: str, capability: str, allowed: bool, evidence: str
) -> None:
    """Audit line for capability gate decisions (the ``checked_by`` trail)."""
    logger.info(
        "capability_check plugin=%s capability=%s decision=%s checked_by=plugin_capability_granted evidence=%s",
        plugin_id, capability, "allow" if allowed else "deny", evidence,
    )


# ---------------------------------------------------------------------------
# Consent state (write side)
# ---------------------------------------------------------------------------

def record_consent(
    plugin_id: str,
    granted: Iterable[str],
    declared: Iterable[str],
) -> None:
    """Persist a consent decision for *plugin_id*.

    Writes ``granted_capabilities`` (union with any previously granted set),
    the consent record (hash of the *declared* set the user saw + UTC
    timestamp), and — so every existing enforcement site keeps working
    without changes — the corresponding legacy ``allow_*`` keys for each
    newly granted capability.
    """
    from hermes_cli.config import load_config, save_config

    granted_list = [c for c in dict.fromkeys(granted) if c in VALID_CAPABILITY_IDS]
    declared_list = [c for c in dict.fromkeys(declared) if c in VALID_CAPABILITY_IDS]

    config = load_config()
    plugins_cfg = config.setdefault("plugins", {})
    if not isinstance(plugins_cfg, dict):
        plugins_cfg = {}
        config["plugins"] = plugins_cfg
    entries = plugins_cfg.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        plugins_cfg["entries"] = entries
    entry = entries.setdefault(plugin_id, {})
    if not isinstance(entry, dict):
        entry = {}
        entries[plugin_id] = entry

    previous = entry.get(GRANTED_KEY)
    merged = list(previous) if isinstance(previous, list) else []
    for cap in granted_list:
        if cap not in merged:
            merged.append(cap)
    entry[GRANTED_KEY] = sorted(
        c for c in dict.fromkeys(merged)
        if isinstance(c, str) and c in VALID_CAPABILITY_IDS
    )
    entry[CONSENT_KEY] = {
        "hash": capability_set_hash(declared_list),
        "granted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # Bridge: mirror each granted capability into its legacy gate so the
    # existing enforcement sites (which still read allow_*) honor the grant.
    for cap in entry[GRANTED_KEY]:
        spec = CAPABILITY_REGISTRY[cap]
        node = entry
        for part in spec.legacy_path[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[spec.legacy_path[-1]] = True

    save_config(config)
    logger.info(
        "capability_consent plugin=%s granted=%s declared_hash=%s",
        plugin_id, ",".join(entry[GRANTED_KEY]) or "(none)",
        entry[CONSENT_KEY]["hash"][:12],
    )


def consent_hash(plugin_id: str, config: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """Return the stored consent hash, or None when absent/corrupt."""
    entry = _plugin_entry(plugin_id, config)
    consent = entry.get(CONSENT_KEY)
    if not isinstance(consent, dict):
        return None
    h = consent.get("hash")
    return h if isinstance(h, str) and h else None


def pending_capabilities(
    plugin_id: str,
    declared: Iterable[str],
    config: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Capabilities declared by the plugin but not yet granted.

    Used both at first consent (everything is pending) and on update
    re-consent: when a new version declares capabilities the granted set
    lacks, those additions are returned and must be re-consented before
    they go live. The stored consent hash tells whether the *declared* set
    changed since the user last saw it.
    """
    declared_list = [c for c in dict.fromkeys(declared) if c in VALID_CAPABILITY_IDS]
    granted = granted_capabilities(plugin_id, config)
    return [c for c in declared_list if c not in granted]


def declared_set_changed(
    plugin_id: str,
    declared: Iterable[str],
    config: Optional[Mapping[str, Any]] = None,
) -> bool:
    """True when the declared set differs from what the user consented to.

    No stored consent at all counts as changed (never consented).
    """
    stored = consent_hash(plugin_id, config)
    if stored is None:
        return True
    return stored != capability_set_hash(
        c for c in declared if c in VALID_CAPABILITY_IDS
    )
