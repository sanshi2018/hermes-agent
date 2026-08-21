"""security-guidance 插件 — 针对文件写入操作提供基于模式匹配的高速安全警告。

绑定并实现如下单一行为：

* ``transform_tool_result`` 钩子函数 — 扫描由 ``write_file`` / ``patch`` / ``skill_manage``
  （write/patch模式）*正在写入的内容*，检查是否存在已知的危险代码模式
  （例如：eval(、pickle.load、yaml.load、os.system、subprocess(shell=True)、
  dangerouslySetInnerHTML、verify=False、ECB 模式、易受 XXE 攻击的 XML 解析器、
  GitHub Actions ``${{ github.event.* }}`` 注入、未设置 ``weights_only=True`` 的 torch.load 等）。
  当匹配到任何已知模式时，插件会在 JSON 格式的工具返回结果字符串末尾追加一个
  ``⚠️ Security warning``（安全警告）区块。文件依然会被正常写入；
  模型将在下一轮对话的工具消息中查看到该警告，并能够进行自我纠正。

为什么不直接拦截（阻止写入）？
模式匹配具有不容忽视的误报率（例如：分词器中的 ``eval(``、已被包装在 ``yaml.SafeLoader``
中的 ``yaml.load``，或者测试夹具内部使用的 ECB 模式）。
如果直接拦截，每次误报都会强制触发确认提示或打断工作流。
对于第 1 层防护机制来说，“警告”是恰当的严重程度 ——
Agent 在读取警告后，既可以修复代码，也可以简要说明为什么该构造在此处是安全的。

如需启用拦截模式（完全拒绝写入），请设置环境变量 ``SECURITY_GUIDANCE_BLOCK=1``。
这是牺牲便利性来换取严格性，适用于将“默认不安全模式”视为违反安全策略的共享开发环境。

模式数据存储在 ``patterns.py`` 中，基于 Apache-2.0 协议逐字 Fork 自 Anthropic 的
``claude-plugins-official`` 项目。详见本目录下的 ``LICENSE`` 和 ``NOTICE`` 文件。
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from . import patterns as _patterns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Tool names whose args carry "code being written to disk" we want to scan.
# Maps tool name -> (path_arg_name, content_arg_names).  For tools with multiple
# possible content fields (patch's old/new_string vs raw patch text), we scan
# every populated string field.
_TARGET_TOOLS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "write_file": ("path", ("content",)),
    "patch": ("path", ("new_string", "patch")),
    # skill_manage write_file / patch sub-actions land here. file_path holds
    # the relative path inside the skill dir; we scan it the same way.
    "skill_manage": ("file_path", ("file_content", "new_string")),
}

# Cap on how much content we scan. Above this we skip — pattern matching a
# 10 MB blob has poor signal-to-noise and would slow down the agent loop.
_MAX_SCAN_BYTES = 256 * 1024


def _block_mode_enabled() -> bool:
    return os.environ.get("SECURITY_GUIDANCE_BLOCK", "").lower() in {"1", "true", "yes", "on"}


def _plugin_disabled() -> bool:
    return os.environ.get("SECURITY_GUIDANCE_DISABLE", "").lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


# Pre-compile the regex patterns once.  Substring patterns stay as plain
# strings — ``str.__contains__`` is faster than a regex of literal chars.
_COMPILED: List[Dict[str, Any]] = []
for _rule in _patterns.SECURITY_PATTERNS:
    _entry: Dict[str, Any] = {
        "ruleName": _rule["ruleName"],
        "reminder": _rule["reminder"],
        "path_filter": _rule.get("path_filter"),
        "path_check": _rule.get("path_check"),
        "substrings": tuple(_rule.get("substrings", ())),
        "regex": None,
    }
    _re_src = _rule.get("regex")
    if _re_src:
        try:
            _entry["regex"] = re.compile(_re_src)
        except re.error as _err:
            logger.warning(
                "security-guidance: skipping rule %s — invalid regex %r: %s",
                _rule["ruleName"], _re_src, _err,
            )
            continue
    _COMPILED.append(_entry)


def _scan_content(path: str, content: str) -> List[Tuple[str, str]]:
    """Return [(ruleName, reminder), ...] for every pattern that matches.

    ``path`` is used by per-rule path filters (path_filter / path_check).
    Each rule fires at most once per call — multiple matches of the same
    rule collapse into a single warning entry.
    """
    if not content or len(content.encode("utf-8", errors="ignore")) > _MAX_SCAN_BYTES:
        return []
    hits: List[Tuple[str, str]] = []
    for entry in _COMPILED:
        # path_check: rule fires PURELY on path match (no content regex). Used
        # for blanket "you're editing a sensitive file, here are reminders"
        # warnings — github_actions_workflow is the canonical example.
        path_check = entry.get("path_check")
        if path_check is not None:
            try:
                if path_check(path or ""):
                    hits.append((entry["ruleName"], entry["reminder"]))
            except Exception:
                pass
            # Path-check rules don't also pattern-match content; move on.
            continue
        # path_filter: rule is skipped when the path filter returns False
        # (e.g. Python-only rules skip .js files; eval_injection skips .md)
        path_filter = entry.get("path_filter")
        if path_filter is not None:
            try:
                if not path_filter(path or ""):
                    continue
            except Exception:
                continue
        matched = False
        for sub in entry["substrings"]:
            if sub in content:
                matched = True
                break
        if not matched and entry["regex"] is not None:
            if entry["regex"].search(content):
                matched = True
        if matched:
            hits.append((entry["ruleName"], entry["reminder"]))
    return hits


def _extract_path_and_content(tool_name: str, args: Any) -> List[Tuple[str, str]]:
    """Return [(path, content), ...] for a tool call.  Empty if nothing to scan."""
    spec = _TARGET_TOOLS.get(tool_name)
    if spec is None or not isinstance(args, dict):
        return []
    path_key, content_keys = spec
    path = args.get(path_key) or ""
    if not isinstance(path, str):
        path = ""
    out: List[Tuple[str, str]] = []
    for ck in content_keys:
        val = args.get(ck)
        if isinstance(val, str) and val:
            out.append((path, val))
    return out


def _format_warning_block(findings: List[Tuple[str, str]]) -> str:
    """Render findings into a Markdown block appended to the tool result."""
    names = ", ".join(name for name, _ in findings)
    lines = [
        "",
        "---",
        f"⚠️ Security guidance — {len(findings)} pattern{'s' if len(findings) != 1 else ''} matched ({names})",
        "",
    ]
    for _, reminder in findings:
        lines.append(reminder)
        lines.append("")
    lines.append(
        "Pattern matches can be false positives. If the construct is safe in this "
        "context, briefly document why in a code comment and continue. Otherwise, "
        "fix the code before moving on."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def _scan_args(tool_name: str, args: Any) -> List[Tuple[str, str]]:
    """Common scan path used by both pre_tool_call (block mode) and
    transform_tool_result (warn mode)."""
    if _plugin_disabled():
        return []
    findings: List[Tuple[str, str]] = []
    for path, content in _extract_path_and_content(tool_name, args):
        findings.extend(_scan_content(path, content))
    return findings


def _on_pre_tool_call(
    tool_name: str = "",
    args: Any = None,
    **_: Any,
) -> Optional[Dict[str, str]]:
    """In block mode, refuse the write if any pattern matches.

    Default mode is non-blocking — we return None here and let
    ``transform_tool_result`` append a warning to the result instead.
    """
    if not _block_mode_enabled():
        return None
    findings = _scan_args(tool_name, args)
    if not findings:
        return None
    return {
        "action": "block",
        "message": (
            "security-guidance refused this write: "
            + _format_warning_block(findings)
            + "\n\nTo override, unset SECURITY_GUIDANCE_BLOCK and retry."
        ),
    }


def _on_transform_tool_result(
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    **_: Any,
) -> Optional[str]:
    """警告模式钩子：向工具执行结果追加安全警告区块。

    返回字符串将替换模型在下一轮对话中看到的结果。
    返回 None 则保持结果不变。
    """
    # 拦截模式会通过 pre_tool_call 处理发现的问题；
    # 在那种情况下，该钩子无需执行任何操作（工具尚未运行，因此没有可供包装的结果）。
    if _block_mode_enabled():
        return None
    findings = _scan_args(tool_name, args)
    if not findings:
        return None
    if not isinstance(result, str):
        return None
    # Don't decorate error results — the model already has bigger problems.
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict) and "error" in parsed and len(parsed) <= 2:
            return None
    except (ValueError, TypeError):
        pass
    return result + "\n\n" + _format_warning_block(findings)


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
