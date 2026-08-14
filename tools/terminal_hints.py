"""terminal（终端）工具的输出模式失败提示。

当命令以非零退出码结束时，原始的 stderr（标准错误）常会让模型产生混淆，
从而浪费额外的轮次去排查诊断（例如：在系统中仅存在 `python3` 时盲目重试 `python`；
或重新发送当前已安装的 gh 不支持的 gh 字段列表）。

本模块扩展了 ``terminal_tool`` 中的退出码语义表，
新增了*输出模式*（output-pattern）匹配层：
通过对命令输出进行有限范围的扫描，
将常见的失败模式映射为一条简短、可立即采取行动的恢复提示。

设计规则（添加新模式时请遵循以下原则）：

* 仅在非零退出码时触发 —— 绝不对成功执行的命令进行标注。
* 每个工具结果最多生成一条提示，优先匹配成功者生效；
  模式按生产轨迹中的出现频率（根据 2026 年 8 月 state.db 挖掘数据）进行排序。
* 仅扫描输出的前 ``_SCAN_CHARS`` 个字符 —— 提示必须基于错误的头部特征，
  而非深层的上下文细节。
* 提示应当直接指出*下一步操作*，而非撰写长篇大论的诊断分析。控制在 1-2 句话内。
* 纯函数设计，无 I/O，无配置读取 —— 极易进行单元测试。

以下引用的出现频率来源于生产环境会话数据库中 25 万次终端结果的采样窗口（2026 年 8 月）：
这些类型共同覆盖了约 1.4 万次失败的工具调用，
这些失败调用的重试链平均额外消耗了 1.4 个工具轮次。
"""

from __future__ import annotations

import re
from typing import Callable, Optional

# Bounded scan window: error headers appear early; deep output is noise.
_SCAN_CHARS = 4000


def _hint_gh_unknown_json_field(command: str, output: str) -> Optional[str]:
    # ~9,175x: gh CLI version drift — model asks for fields the installed
    # gh doesn't know. gh already prints the valid field list.
    m = re.search(r'Unknown JSON field: "?(\w+)', output)
    if not m:
        return None
    return (
        f"The installed gh does not support the JSON field '{m.group(1)}'. "
        "The valid field list is printed in the output above — retry using "
        "only fields from that list."
    )


def _hint_command_not_found(command: str, output: str) -> Optional[str]:
    # ~1,010x generic; 837x of them are bare `python` on python3-only distros.
    m = re.search(r"(?:bash: line \d+: |bash: |sh: \d*:? ?)?([\w.+-]+): command not found", output)
    if not m:
        return None
    missing = m.group(1)
    if missing == "python":
        return (
            "This system has no bare `python` — use `python3`, or the "
            "project venv's interpreter (e.g. .venv/bin/python)."
        )
    if missing == "pip":
        return (
            "This system has no bare `pip` — use `pip3`, `python3 -m pip`, "
            "or the project venv's pip (e.g. .venv/bin/pip)."
        )
    return (
        f"`{missing}` is not installed or not on PATH. Verify with "
        f"`which {missing}`; install it or use an absolute path instead of "
        "retrying the same command."
    )


def _hint_module_not_found(command: str, output: str) -> Optional[str]:
    # ~739x: almost always a venv-activation slip, not a missing dependency.
    m = re.search(r"(?:ModuleNotFoundError|ImportError): No module named '?([\w.]+)", output)
    if not m:
        return None
    return (
        f"Python cannot import '{m.group(1)}'. Most often the wrong "
        "interpreter is running: activate the project venv (e.g. `source "
        ".venv/bin/activate`) or invoke its python directly. Only pip "
        "install if the package is genuinely absent from that venv."
    )


def _hint_merge_conflict(command: str, output: str) -> Optional[str]:
    # ~1,172x: models sometimes re-run the failing merge/rebase verbatim.
    if not re.search(r"^CONFLICT |Automatic merge failed|needs merge", output, re.M):
        return None
    return (
        "Git merge conflict. Do not retry this command. Resolve the "
        "conflicted files listed above (edit, then `git add`), then continue "
        "(`git rebase --continue` / commit the merge) — or abort with "
        "`--abort`."
    )


def _hint_already_exists(command: str, output: str) -> Optional[str]:
    # ~633x: branch/dir/file already exists → retrying unchanged always fails.
    m = re.search(r"(?:fatal|error):.*?'([^']+)' already exists", output)
    if not m:
        return None
    return (
        f"'{m.group(1)}' already exists — retrying unchanged will keep "
        "failing. Reuse it, choose another name, or delete it first if it is "
        "genuinely stale."
    )


def _hint_gh_rate_limit(command: str, output: str) -> Optional[str]:
    # ~133x: immediate retries burn turns; the limit is time-based.
    if "API rate limit" not in output and "was submitted too quickly" not in output:
        return None
    return (
        "GitHub API rate limit hit — immediate retries will keep failing. "
        "Continue with other work and retry this operation later."
    )


def _hint_permission_denied(command: str, output: str) -> Optional[str]:
    if "Permission denied" not in output and "EACCES" not in output:
        return None
    return (
        "Permission denied. Check ownership/mode of the target path "
        "(`ls -la`); prefer a user-writable location. Only escalate to sudo "
        "if the task genuinely requires it."
    )


# Ordered by production frequency — first match wins.
_OUTPUT_HINTS: list[Callable[[str, str], Optional[str]]] = [
    _hint_gh_unknown_json_field,
    _hint_merge_conflict,
    _hint_command_not_found,
    _hint_module_not_found,
    _hint_already_exists,
    _hint_gh_rate_limit,
    _hint_permission_denied,
]

# Exit-code-only hints for codes the semantics table in terminal_tool does
# not cover per-command. Checked after output patterns.
_EXIT_CODE_HINTS: dict[int, str] = {
    126: "Exit 126: the file was found but is not executable — `chmod +x` it or invoke it via its interpreter (e.g. `bash script.sh`).",
    137: "Exit 137: the process was SIGKILLed — usually out-of-memory or an external kill. Reduce memory use or check `dmesg | tail` before retrying.",
    124: "Exit 124: the command hit its timeout. Raise timeout= (foreground max 600s) or run it with background=true and notify_on_complete=true.",
}


def annotate_failure(command: str, exit_code: int, output: str) -> Optional[str]:
    """当命令执行失败时，返回一条简短的恢复/修复提示；若无匹配提示则返回 None。

    参数：
        command: 已运行的命令字符串。
        exit_code: 该命令的退出码（非零值表示失败）。
        output: 返回给模型的标准输出与标准错误（stdout/stderr）合并后的内容。

    仅对 output 的前 ``_SCAN_CHARS`` 个字符进行检查，
    且最多返回一条提示。
    当 exit_code == 0 时返回 None。
    """
    if exit_code == 0:
        return None
    window = (output or "")[:_SCAN_CHARS]
    if window:
        for fn in _OUTPUT_HINTS:
            try:
                hint = fn(command or "", window)
            except Exception:
                continue
            if hint:
                return hint
    return _EXIT_CODE_HINTS.get(exit_code)
