#!/usr/bin/env python3
"""
终端工具模块

一种支持在本地、Docker、Modal、SSH、
Singularity 和 Daytona 环境中执行命令的终端工具。
支持本地执行、容器化后端以及云沙箱（包括托管 Modal 模式）。

支持的环境：
- "local": 直接在宿主机上执行（默认，速度最快）
- "docker": 在 Docker 容器中执行（隔离环境，需要安装 Docker）
- "modal": 在 Modal 云沙箱中执行（直接 Modal 或托管网关）

特性功能：
- 支持多种执行后端（local、docker、modal）
- 支持后台任务
- 虚拟机/容器生命周期管理
- 空闲时自动清理

云沙箱说明：
- 持久化文件系统会在沙箱重新创建时保留工作状态
- 持久化文件系统**不能**保证同一个活动沙箱或长运行进程能够在清理、空闲回收或 Hermes 退出后继续存活

用法示例：
    from terminal_tool import terminal_tool

    # 执行简单命令
    result = terminal_tool("ls -la")

    # 在后台执行
    result = terminal_tool("python server.py", background=True)
"""

import importlib.util
import json
import logging
import os
import platform
import re
import time
import threading
import atexit
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

from utils import env_var_enabled

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 全局中断事件：当收到用户中断指令时由 agent 设置。
# 终端工具会在命令执行期间对该事件进行轮询，
# 以便能够立即终止长运行的子进程，
# 而不必一直阻塞等待直至超时。
# ---------------------------------------------------------------------------
from tools.interrupt import is_interrupted, _interrupt_event  # noqa: F401 — re-exported
# display_hermes_home imported lazily at call site (stale-module safety during hermes update)




# =============================================================================
# Custom Singularity Environment with more space
# =============================================================================

# Singularity helpers (scratch dir, SIF cache) now live in tools/environments/singularity.py
from tools.environments.singularity import _get_scratch_dir
from tools.tool_backend_helpers import (
    coerce_modal_mode,
    has_direct_modal_credentials,
    managed_nous_tools_enabled,
    nous_tool_gateway_unavailable_message,
    resolve_modal_backend_state,
)


def _safe_parse_import_env(
    name: str,
    default: Any,
    converter,
    type_label: str,
):
    """Parse module-level numeric env vars without breaking import.

    Terminal tool is imported by CLI, ACP, tests, and tool discovery. A single
    malformed env var must not make the whole module unloadable at import time.
    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return converter(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value for %s: %r (expected %s). Falling back to %r.",
            name,
            raw,
            type_label,
            default,
        )
        return default


# Hard cap on foreground timeout; override via TERMINAL_MAX_FOREGROUND_TIMEOUT env var.
FOREGROUND_MAX_TIMEOUT = _safe_parse_import_env(
    "TERMINAL_MAX_FOREGROUND_TIMEOUT",
    600,
    int,
    "integer",
)

# Disk usage warning threshold (in GB)
DISK_USAGE_WARNING_THRESHOLD_GB = _safe_parse_import_env(
    "TERMINAL_DISK_WARNING_GB",
    500.0,
    float,
    "number",
)


def _check_disk_usage_warning():
    """Check if total disk usage exceeds warning threshold."""
    try:
        scratch_dir = _get_scratch_dir()

        # Get total size of hermes directories
        total_bytes = 0
        import glob
        for path in glob.glob(str(scratch_dir / "hermes-*")):
            for f in Path(path).rglob('*'):
                if f.is_file():
                    try:
                        total_bytes += f.stat().st_size
                    except OSError as e:
                        logger.debug("Could not stat file %s: %s", f, e)
        
        total_gb = total_bytes / (1024 ** 3)
        
        if total_gb > DISK_USAGE_WARNING_THRESHOLD_GB:
            logger.warning("Disk usage (%.1fGB) exceeds threshold (%.0fGB). Consider running cleanup_all_environments().",
                           total_gb, DISK_USAGE_WARNING_THRESHOLD_GB)
            return True
        
        return False
    except Exception as e:
        logger.debug("Disk usage warning check failed: %s", e, exc_info=True)
        return False


# Interactive sudo password cache.
#
# Scope the cache to the active session when a session key is available, then
# fall back to callback identity (ACP / CLI interactive callbacks), then the
# current thread. This prevents one interactive session from reusing another
# session's cached sudo password inside the same long-lived process.
_sudo_password_cache: dict[str, str] = {}
_sudo_password_cache_lock = threading.Lock()

# Optional UI callbacks for interactive prompts. When set, these are called
# instead of the default /dev/tty or input() readers. The CLI registers these
# so prompts route through prompt_toolkit's event loop.
# Callback slots used by the approval prompt and sudo password prompt
# routines. Stored in thread-local state so overlapping ACP sessions —
# each running in its own ThreadPoolExecutor thread — don't stomp on
# each other's callbacks. See GHSA-qg5c-hvr5-hjgr.
#
# CLI mode is single-threaded, so each thread (the only one) holds its
# own callback exactly like before. Gateway mode resolves approvals via
# the per-session queue in tools.approval, not through these callbacks,
# so it's unaffected.
_callback_tls = threading.local()


def _get_sudo_password_callback():
    return getattr(_callback_tls, "sudo_password", None)


def _get_approval_callback():
    return getattr(_callback_tls, "approval", None)


def set_sudo_password_callback(cb):
    """Register a callback for sudo password prompts (used by CLI).

    Per-thread scope — ACP sessions that run concurrently in a
    ThreadPoolExecutor each have their own callback slot.
    """
    _callback_tls.sudo_password = cb


def set_approval_callback(cb):
    """注册一个针对危险命令审批提示的回调函数。

    线程级作用域 —— 在 ThreadPoolExecutor 中并发运行的
    ACP 会话各自拥有独立的回调槽位。参见
    GHSA-qg5c-hvr5-hjgr。
    """
    _callback_tls.approval = cb


def _get_sudo_password_cache_scope() -> str:
    """Return the cache scope for interactive sudo passwords."""
    try:
        from gateway.session_context import get_session_env

        session_key = get_session_env("HERMES_SESSION_KEY", "")
    except Exception:
        session_key = os.getenv("HERMES_SESSION_KEY", "")
    if session_key:
        return f"session:{session_key}"

    callback = _get_sudo_password_callback()
    if callback is not None:
        owner = getattr(callback, "__self__", None)
        func = getattr(callback, "__func__", None)
        if owner is not None and func is not None:
            return f"callback-owner:{id(owner)}:{id(func)}"
        return f"callback:{id(callback)}"

    return f"thread:{threading.get_ident()}"


def _get_cached_sudo_password() -> str:
    """Return the cached sudo password for the current scope."""
    scope = _get_sudo_password_cache_scope()
    with _sudo_password_cache_lock:
        return _sudo_password_cache.get(scope, "")


def _set_cached_sudo_password(password: str) -> None:
    """Persist a sudo password for the current scope."""
    scope = _get_sudo_password_cache_scope()
    with _sudo_password_cache_lock:
        if password:
            _sudo_password_cache[scope] = password
        else:
            _sudo_password_cache.pop(scope, None)


def _reset_cached_sudo_passwords() -> None:
    """Clear all cached sudo passwords.

    Internal helper for tests and process teardown paths.
    """
    with _sudo_password_cache_lock:
        _sudo_password_cache.clear()

# =============================================================================
# Dangerous Command Approval System
# =============================================================================

# Dangerous command detection + approval now consolidated in tools/approval.py
from tools.approval import (
    check_all_command_guards as _check_all_guards_impl,
)


def _docker_volume_uses_host_path(volume_spec: str) -> bool:
    """Return True when a docker volume spec bind-mounts a host path."""
    if not isinstance(volume_spec, str):
        return False

    vol = volume_spec.strip()
    return bool(vol) and (
        vol.startswith(("/", "~", "./", "../")) or
        (len(vol) >= 3 and vol[1] == ":" and vol[2] in ("/", "\\"))
    )


def _docker_has_host_access(config: Dict[str, Any]) -> bool:
    """Return True when a Docker sandbox exposes host paths through bind mounts."""
    if config.get("env_type") != "docker":
        return False
    if config.get("host_cwd") and config.get("docker_mount_cwd_to_workspace"):
        return True
    return any(_docker_volume_uses_host_path(vol) for vol in config.get("docker_volumes", []))


def _check_all_guards(command: str, env_type: str,
                      has_host_access: bool = False) -> dict:
    """Delegate to consolidated guard (tirith + dangerous cmd) with CLI callback."""
    return _check_all_guards_impl(command, env_type,
                                  approval_callback=_get_approval_callback(),
                                  has_host_access=has_host_access)


# Allowlist: characters that can legitimately appear in directory paths.
# Covers alphanumeric, path separators, Windows drive/UNC separators, tilde,
# dot, hyphen, underscore, space, plus, at, equals, and comma.  Everything
# else is rejected.
_WORKDIR_SAFE_RE = re.compile(r'^[A-Za-z0-9/\\:_\-.~ +@=,]+$')


def _validate_workdir(workdir: str) -> str | None:
    """拒绝看起来不符合文件系统路径规范的 workdir 值。

    使用安全字符白名单而非黑名单机制，
    从而防止新型 Shell 元字符（metacharacters）绕过检查。

    如果路径安全则返回 None，
    如果存在风险则返回错误信息字符串。
    """
    if not workdir:
        return None
    if not _WORKDIR_SAFE_RE.match(workdir):
        # Find the first offending character for a helpful message.
        for ch in workdir:
            if not _WORKDIR_SAFE_RE.match(ch):
                return (
                    f"Blocked: workdir contains disallowed character {repr(ch)}. "
                    "Use a simple filesystem path without shell metacharacters."
                )
        return "Blocked: workdir contains disallowed characters."
    return None


def _handle_sudo_failure(output: str, env_type: str) -> str:
    """
    Check for sudo failure and add helpful message for messaging contexts.
    
    Returns enhanced output if sudo failed in messaging context, else original.
    """
    is_gateway = env_var_enabled("HERMES_GATEWAY_SESSION")
    
    if not is_gateway:
        return output
    
    # Check for sudo failure indicators
    sudo_failures = [
        "sudo: a password is required",
        "sudo: no tty present",
        "sudo: a terminal is required",
    ]
    
    for failure in sudo_failures:
        if failure in output:
            from hermes_constants import display_hermes_home as _dhh
            return output + f"\n\n💡 Tip: To enable sudo over messaging, add SUDO_PASSWORD to {_dhh()}/.env on the agent machine."
    
    return output


# sudo -S rejects a bad cached/interactive password with these messages.
_SUDO_WRONG_PASSWORD_MARKERS = (
    "sudo: authentication failed",
    "sudo: incorrect password attempt",
    "sudo: maximum 3 incorrect authentication attempts",
    "sudo: 3 incorrect password attempts",
)


def _sudo_wrong_password_failure(output: str) -> bool:
    """Return True when sudo rejected a piped password."""
    if not output:
        return False
    lowered = output.lower()
    return any(marker in lowered for marker in _SUDO_WRONG_PASSWORD_MARKERS)


def _invalidate_cached_sudo_on_auth_failure(
    command: str | None, output: str
) -> bool:
    """当 sudo 拒绝密码后，丢弃会话中缓存的 sudo 密码。

    通过环境变量配置的 ``SUDO_PASSWORD`` 将保持不变 ——
    那是操作人员的显式选择，
    而非交互式的缓存条目。
    """
    if "SUDO_PASSWORD" in os.environ:
        return False
    if not _sudo_wrong_password_failure(output):
        return False
    if _count_real_sudo_invocations(command or "") == 0:
        return False
    if not _get_cached_sudo_password():
        return False
    _set_cached_sudo_password("")
    return True


def _prompt_for_sudo_password(timeout_seconds: int = 45) -> str:
    """
    Prompt user for sudo password with timeout.
    
    Returns the password if entered, or empty string if:
    - User presses Enter without input (skip)
    - Timeout expires (45s default)
    - Any error occurs
    
    Only works in interactive mode (HERMES_INTERACTIVE=1).
    If a _sudo_password_callback is registered (by the CLI), delegates to it
    so the prompt integrates with prompt_toolkit's UI.  Otherwise reads
    directly from /dev/tty with echo disabled.
    """
    import sys
    
    # Use the registered callback when available (prompt_toolkit-compatible)
    _sudo_cb = _get_sudo_password_callback()
    if _sudo_cb is not None:
        try:
            return _sudo_cb() or ""
        except Exception:
            return ""

    result = {"password": None, "done": False}
    
    def read_password_thread():
        """Read password with echo disabled. Uses msvcrt on Windows, /dev/tty on Unix."""
        tty_fd = None
        old_attrs = None
        try:
            if platform.system() == "Windows":
                import msvcrt
                chars = []
                while True:
                    c = msvcrt.getwch()
                    if c in {"\r", "\n"}:
                        break
                    if c == "\x03":
                        raise KeyboardInterrupt
                    chars.append(c)
                result["password"] = "".join(chars)
            else:
                import termios
                tty_fd = os.open("/dev/tty", os.O_RDONLY)
                old_attrs = termios.tcgetattr(tty_fd)
                new_attrs = termios.tcgetattr(tty_fd)
                new_attrs[3] = new_attrs[3] & ~termios.ECHO
                termios.tcsetattr(tty_fd, termios.TCSAFLUSH, new_attrs)
                chars = []
                while True:
                    b = os.read(tty_fd, 1)
                    if not b or b in {b"\n", b"\r"}:
                        break
                    chars.append(b)
                result["password"] = b"".join(chars).decode("utf-8", errors="replace")
        except (EOFError, KeyboardInterrupt, OSError):
            result["password"] = ""
        except Exception:
            result["password"] = ""
        finally:
            if tty_fd is not None and old_attrs is not None:
                try:
                    import termios as _termios
                    _termios.tcsetattr(tty_fd, _termios.TCSAFLUSH, old_attrs)
                except Exception as e:
                    logger.debug("Failed to restore terminal attributes: %s", e)
            if tty_fd is not None:
                try:
                    os.close(tty_fd)
                except Exception as e:
                    logger.debug("Failed to close tty fd: %s", e)
            result["done"] = True
    
    try:
        os.environ["HERMES_SPINNER_PAUSE"] = "1"
        time.sleep(0.2)
        
        print()
        print("┌" + "─" * 58 + "┐")
        print("│  🔐 SUDO PASSWORD REQUIRED" + " " * 30 + "│")
        print("├" + "─" * 58 + "┤")
        print("│  Enter password below (input is hidden), or:            │")
        print("│    • Press Enter to skip (command fails gracefully)     │")
        print(f"│    • Wait {timeout_seconds}s to auto-skip" + " " * 27 + "│")
        print("└" + "─" * 58 + "┘")
        print()
        print("  Password (hidden): ", end="", flush=True)
        
        password_thread = threading.Thread(target=read_password_thread, daemon=True)
        password_thread.start()
        password_thread.join(timeout=timeout_seconds)
        
        if result["done"]:
            password = result["password"] or ""
            print()  # newline after hidden input
            if password:
                print("  ✓ Password received (cached for this session)")
            else:
                print("  ⏭ Skipped - continuing without sudo")
            print()
            sys.stdout.flush()
            return password
        else:
            print("\n  ⏱ Timeout - continuing without sudo")
            print("    (Press Enter to dismiss)")
            print()
            sys.stdout.flush()
            return ""
            
    except (EOFError, KeyboardInterrupt):
        print()
        print("  ⏭ Cancelled - continuing without sudo")
        print()
        sys.stdout.flush()
        return ""
    except Exception as e:
        print(f"\n  [sudo prompt error: {e}] - continuing without sudo\n")
        sys.stdout.flush()
        return ""
    finally:
        if "HERMES_SPINNER_PAUSE" in os.environ:
            del os.environ["HERMES_SPINNER_PAUSE"]

def _safe_command_preview(command: Any, limit: int = 200) -> str:
    """Return a log-safe preview for possibly-invalid command values."""
    if command is None:
        return "<None>"
    if isinstance(command, str):
        return command[:limit]
    try:
        return repr(command)[:limit]
    except Exception:
        return f"<{type(command).__name__}>"

def _looks_like_env_assignment(token: str) -> bool:
    """Return True when *token* is a leading shell environment assignment."""
    if "=" not in token or token.startswith("="):
        return False
    name, _value = token.split("=", 1)
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name))


def _read_shell_token(command: str, start: int) -> tuple[str, int]:
    """Read one shell token, preserving quotes/escapes, starting at *start*."""
    i = start
    n = len(command)

    while i < n:
        ch = command[i]
        if ch.isspace() or ch in ";|&()":
            break
        if ch == "'":
            i += 1
            while i < n and command[i] != "'":
                i += 1
            if i < n:
                i += 1
            continue
        if ch == '"':
            i += 1
            while i < n:
                inner = command[i]
                if inner == "\\" and i + 1 < n:
                    i += 2
                    continue
                if inner == '"':
                    i += 1
                    break
                i += 1
            continue
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        i += 1

    return command[start:i], i


def _rewrite_real_sudo_invocations(command: str) -> tuple[str, int]:
    """Rewrite only real unquoted sudo command words, not plain text mentions.

    Returns the rewritten command and the number of sudo invocations rewritten.
    """
    out: list[str] = []
    i = 0
    n = len(command)
    command_start = True
    sudo_count = 0

    while i < n:
        ch = command[i]

        if ch.isspace():
            out.append(ch)
            if ch == "\n":
                command_start = True
            i += 1
            continue

        if ch == "#" and command_start:
            comment_end = command.find("\n", i)
            if comment_end == -1:
                out.append(command[i:])
                break
            out.append(command[i:comment_end])
            i = comment_end
            continue

        if command.startswith("&&", i) or command.startswith("||", i) or command.startswith(";;", i):
            out.append(command[i:i + 2])
            i += 2
            command_start = True
            continue

        if ch in ";|&(":
            out.append(ch)
            i += 1
            command_start = True
            continue

        if ch == ")":
            out.append(ch)
            i += 1
            command_start = False
            continue

        token, next_i = _read_shell_token(command, i)
        if command_start and token == "sudo":
            out.append("sudo -S -p ''")
            sudo_count += 1
        else:
            out.append(token)

        if command_start and _looks_like_env_assignment(token):
            command_start = True
        else:
            command_start = False
        i = next_i

    return "".join(out), sudo_count


def _count_real_sudo_invocations(command: str) -> int:
    """返回 *command* 中实际出现的 sudo 命令单词数量。

    这是一个轻量级扫描，
    它重用了与 ``_rewrite_real_sudo_invocations`` 相同的分词器（tokeniser），
    但跳过了字符串构建过程，
    因此在结果处理路径中进行调用是非常轻量且高效的。
    """
    count = 0
    i = 0
    n = len(command)
    command_start = True

    while i < n:
        ch = command[i]

        if ch.isspace():
            if ch == "\n":
                command_start = True
            i += 1
            continue

        if ch == "#" and command_start:
            comment_end = command.find("\n", i)
            if comment_end == -1:
                break
            i = comment_end
            continue

        if command.startswith("&&", i) or command.startswith("||", i) or command.startswith(";;", i):
            i += 2
            command_start = True
            continue

        if ch in ";|&(":
            i += 1
            command_start = True
            continue

        if ch == ")":
            i += 1
            command_start = False
            continue

        token, next_i = _read_shell_token(command, i)
        if command_start and token == "sudo":
            count += 1

        if command_start and _looks_like_env_assignment(token):
            command_start = True
        else:
            command_start = False
        i = next_i

    return count


def _sudo_nopasswd_works() -> bool:
    """Return True when local sudo currently works without prompting.

    Only probes for the `local` terminal backend; Docker/SSH/Modal/etc. must
    not inherit the host's sudo state. Re-probes every call (no process-level
    cache) so an expired sudo timestamp cannot make a later command silently
    block waiting for a password.
    """
    terminal_env = os.getenv("TERMINAL_ENV", "local").strip().lower() or "local"
    if terminal_env != "local":
        return False

    try:
        probe = subprocess.run(
            ["sudo", "-n", "true"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
        return probe.returncode == 0
    except Exception:
        return False


def _rewrite_compound_background(command: str) -> str:
    """
    在深度为 0 时，将 `A && B &`（或 `A || B &`）包裹改写为 `A && { B & }`。

    Bash 在解析 `A && B &` 时，`&&` 的优先级高于 `&`，
    因此它会为整个 `A && B` 复合命令 fork 一个子 Shell 并将其放入后台运行。
    在子 Shell 内部，`B` 是在前台运行的，所以子 Shell 会等待 `B` 执行结束。
    当 `B` 是一个长期运行的进程时（例如 `python3 -m http.server`、
    `yes > /dev/null` 或任何不会自动退出的命令），该子 Shell 就永远不会退出。
    它会作为一个永久卡在 `wait4` 状态的进程发生泄漏 ——
    在此过程中，它打开的 stdout 管道还会导致终端工具无法及时返回。

    将尾部重写为 `A && { B & }` 既保留了 `&&` 的错误处理语义（若 A 失败则跳过 B），
    同时又用花括号组合（brace group）替代了子 Shell。
    花括号组合会在当前 Shell 中运行（不发起 fork），
    将 B 作为简单命令放入后台（Bash 在非交互模式下不会等待它），
    并立即退出。
    B 则作为普通的后台子进程运行，在父 Shell 退出时被孤儿化。

    此逻辑支持处理重定向（`&>`、`2>&1`），
    并会跳过引号字符串以及带括号的子 Shell 内部的内容。
    对于简单的 `cmd &` 结构则保持原样 —— 因为该结构不存在子 Shell 等待的 bug。
    """
    n = len(command)
    i = 0
    paren_depth = 0
    brace_depth = 0
    # Position in *command* just after the most recent `&&` / `||` at depth 0
    # in the current statement; -1 when no chain operator is active.
    last_chain_op_end = -1
    rewrites: list[tuple[int, int]] = []  # (chain_op_end, amp_pos)

    while i < n:
        ch = command[i]

        # Newline terminates a statement at depth 0 — reset chain state.
        # Checked before the whitespace skip so we don't miss it.
        if ch == "\n" and paren_depth == 0 and brace_depth == 0:
            last_chain_op_end = -1
            i += 1
            continue

        if ch.isspace():
            i += 1
            continue

        # Comments (only at statement start — conservative: any `#` not inside
        # a token ends the line). `_read_shell_token` handles quoted strings
        # below so `#` inside quotes is safe.
        if ch == "#":
            nl = command.find("\n", i)
            if nl == -1:
                break
            i = nl
            continue

        if ch == "\\" and i + 1 < n:
            i += 2
            continue

        # Quoted tokens — consume whole string via the shared tokenizer.
        if ch in {"'", '"'}:
            _, next_i = _read_shell_token(command, i)
            i = max(next_i, i + 1)
            continue

        if ch == "(":
            paren_depth += 1
            i += 1
            continue

        if ch == ")":
            paren_depth = max(0, paren_depth - 1)
            i += 1
            continue

        # Brace groups: `{ ... }` is a group (no subshell fork), and bash
        # requires whitespace after `{`. We track depth so already-rewritten
        # output (`A && { B & }`) is idempotent — the inner `&` is part of
        # the group, not a new compound to rewrite. Also skip content inside
        # the group since `A && B &` there is separately well-formed.
        if ch == "{" and i + 1 < n and (command[i + 1].isspace() or command[i + 1] == "\n"):
            brace_depth += 1
            i += 1
            continue
        if ch == "}" and brace_depth > 0:
            brace_depth -= 1
            # Closing a group completes a compound statement; reset chain.
            last_chain_op_end = -1
            i += 1
            continue

        # Inside parens or brace groups, skip operators — they parse in their
        # own scope. `(...)` subshells have the same bug class but are not the
        # common agent pattern; leave for a follow-up.
        if paren_depth > 0 or brace_depth > 0:
            i += 1
            continue

        # Chain operators at depth 0
        if command.startswith("&&", i) or command.startswith("||", i):
            last_chain_op_end = i + 2
            i += 2
            continue

        # Statement terminators reset the chain state
        if ch == ";":
            last_chain_op_end = -1
            i += 1
            continue

        # Single `|` (pipe) starts a new pipeline stage; don't rewrite
        # across it. `||` handled above.
        if ch == "|":
            last_chain_op_end = -1
            i += 1
            continue

        # `&` handling: distinguish `&&`, `&>`, fd redirect (`>&`, `<&`),
        # and a true backgrounding `&`.
        if ch == "&":
            # `&&` handled above; won't reach here
            if i + 1 < n and command[i + 1] == ">":
                # `&>` redirect — consume
                i += 2
                continue
            # `>&` / `<&` fd target — look back past whitespace
            j = i - 1
            while j >= 0 and command[j].isspace():
                j -= 1
            if j >= 0 and command[j] in "<>":
                i += 1
                continue
            # Real background operator
            if last_chain_op_end >= 0:
                rewrites.append((last_chain_op_end, i))
            last_chain_op_end = -1
            i += 1
            continue

        # Regular unquoted token — advance past it via the shared tokenizer
        _, next_i = _read_shell_token(command, i)
        i = max(next_i, i + 1)

    if not rewrites:
        return command

    # Apply rewrites back-to-front so earlier indices remain valid.
    result = command
    for chain_end, amp_pos in reversed(rewrites):
        # Skip whitespace right after the `&&`/`||` so the brace group
        # opens flush against the inner command.
        insert_pos = chain_end
        while insert_pos < amp_pos and result[insert_pos].isspace():
            insert_pos += 1
        prefix = result[:insert_pos]
        middle = result[insert_pos:amp_pos]  # inner command + trailing space
        suffix = result[amp_pos + 1 :]
        # `{` needs a trailing space in bash; the closing `}` needs to be
        # preceded by `;` or `&` — we're providing `&` from the backgrounding.
        result = prefix + "{ " + middle + "& }" + suffix

    return result


def _transform_sudo_command(command: str | None) -> tuple[str | None, str | None]:
    """如果 SUDO_PASSWORD 可用，将 sudo 命令转换为使用 -S 标志的形式。

    这是一个由所有执行环境共用的辅助函数，
    旨在跨本地、SSH 和容器环境提供一致的 sudo 处理逻辑。

    返回值：
        (transformed_command, sudo_stdin) 其中：
        - transformed_command 将每个单独的 ``sudo`` 替换为
          ``sudo -S -p ''``，以便 sudo 从 stdin 读取密码。
        - sudo_stdin 是末尾带有换行符的密码字符串，
          调用方必须将其追加到进程 stdin 流的最前面。
          sudo -S 会精确读取一行（即密码），
          并将剩余的 stdin 传递给子命令，
          因此即使调用方自身也有需要传输的 stdin_data，提前追加密码也是安全的。
        - 如果没有可用密码，则 sudo_stdin 为 None，
          且命令将原样返回，从而优雅地失败并提示
          "sudo: a password is required"。

    直接驱动子进程的调用方（local, ssh, docker, singularity）
    应当将 sudo_stdin 追加到其 stdin_data 的最前面，
    并将合并后的字节流传递给 Popen 的 stdin 管道。

    无法通过管道传输子进程 stdin 的调用方（modal, daytona）
    必须自行将密码嵌入到命令字符串中；
    有关它们如何处理非 None 的 sudo_stdin 情况，请参阅其对应的 execute() 方法。

    如果未设置 SUDO_PASSWORD 且交互式 UI 可用
    （HERMES_INTERACTIVE=1 或已注册 sudo 密码回调）：
      提示用户输入密码（超时时间 45 秒），并针对当前会话进行缓存。

    如果未设置 SUDO_PASSWORD 且不可交互：
      命令将原样运行（优雅地失败并提示 "sudo: a password is required"）。
    """
    if command is None:
        return None, None
    transformed, sudo_count = _rewrite_real_sudo_invocations(command)
    if sudo_count == 0:
        return command, None

    has_configured_password = "SUDO_PASSWORD" in os.environ
    sudo_password = (
        os.environ.get("SUDO_PASSWORD", "")
        if has_configured_password
        else _get_cached_sudo_password()
    )

    # Local hosts with sudoers NOPASSWD should not be forced through the
    # interactive Hermes password prompt or the sudo -S password-pipe path.
    # Scoped to the local terminal backend so Docker/SSH/Modal/etc. can't
    # inherit host sudo state. Re-probes every call (no process-lifetime
    # cache) so an expired sudo timestamp doesn't make a later command block
    # silently without Hermes prompting.
    if not has_configured_password and not sudo_password and _sudo_nopasswd_works():
        return command, None

    has_sudo_prompt_callback = _get_sudo_password_callback() is not None
    should_prompt_for_sudo = (
        env_var_enabled("HERMES_INTERACTIVE") or has_sudo_prompt_callback
    )
    if not has_configured_password and not sudo_password and should_prompt_for_sudo:
        sudo_password = _prompt_for_sudo_password(timeout_seconds=45)
        if sudo_password:
            _set_cached_sudo_password(sudo_password)

    if has_configured_password or sudo_password:
        # Trailing newline is required: sudo -S reads one line per invocation.
        # Compound commands (`sudo a && sudo b`) need one password line each.
        password_line = sudo_password + "\n"
        return transformed, password_line * sudo_count

    return command, None


# Environment classes now live in tools/environments/
from tools.environments.local import LocalEnvironment as _LocalEnvironment
from tools.environments.singularity import SingularityEnvironment as _SingularityEnvironment
from tools.environments.ssh import SSHEnvironment as _SSHEnvironment
from tools.environments.docker import DockerEnvironment as _DockerEnvironment
from tools.environments.modal import ModalEnvironment as _ModalEnvironment
from tools.environments.managed_modal import ManagedModalEnvironment as _ManagedModalEnvironment
from tools.managed_tool_gateway import is_managed_tool_gateway_ready
import sys


# Tool description for LLM
TERMINAL_TOOL_DESCRIPTION = """Execute shell commands on a Linux environment. Filesystem, current working directory, and exported environment variables persist between calls.

Do NOT use cat/head/tail to read files — use read_file instead.
Do NOT use grep/rg/find to search — use search_files instead.
Do NOT use ls to list directories — use search_files(target='files') instead.
Do NOT use sed/awk to edit files — use patch instead.
Do NOT use echo/cat heredoc to create files — use write_file instead.
Reserve terminal for: builds, installs, git, processes, scripts, network, package managers, and anything that needs a shell.
Because exported environment state persists, activate a virtualenv or export setup variables once per session; do not re-source the same environment before every command unless a command proves the shell state was reset.

Foreground (default): Commands return INSTANTLY when done, even if the timeout is high. Set timeout=300 for long builds/scripts — you'll still get the result in seconds if it's fast. Prefer foreground for short commands.
Background: Set background=true to get a session_id. Almost always pair with notify_on_complete=true — bg without notify runs SILENTLY and you have no way to learn it finished short of calling process(action='poll') yourself. Two legitimate uses:
  (1) Long-lived processes that never exit (servers, watchers, daemons) — silent is correct, there's no exit to notify on.
  (2) Long-running bounded tasks (tests, builds, deploys, CI pollers, batch jobs) — MUST set notify_on_complete=true. Without it you'll either forget to poll or sit blocked waiting for the user to surface the result.
For servers/watchers, do NOT use shell-level background wrappers (nohup/disown/setsid/trailing '&') in foreground mode. Use background=true so Hermes can track lifecycle and output.
After starting a server, verify readiness with a health check or log signal, then run tests in a separate terminal() call. Avoid blind sleep loops.
Use process(action="poll") for progress checks, process(action="wait") to block until done.
Working directory: Use 'workdir' for per-command cwd.
PTY mode: Set pty=true for interactive CLI tools (Codex, Claude Code, Python REPL).

Do NOT use vim/nano/interactive tools without pty=true — they hang without a pseudo-terminal. Pipe git output to cat if it might page.
"""

# Global state for environment lifecycle management
_active_environments: Dict[str, Any] = {}
_last_activity: Dict[str, float] = {}
_env_lock = threading.Lock()
_creation_locks: Dict[str, threading.Lock] = {}  # Per-task locks for sandbox creation
_creation_locks_lock = threading.Lock()  # Protects _creation_locks dict itself
_cleanup_thread = None
_cleanup_running = False

# Once-per-process guard for the docker orphan reaper (issue #20561).
# Set when _maybe_reap_docker_orphans first runs; concurrent _create_environment
# calls for parallel subagents won't re-trigger the sweep.
_docker_orphan_reaper_ran = False
_docker_orphan_reaper_lock = threading.Lock()


def _maybe_reap_docker_orphans(container_config: Dict[str, Any]) -> None:
    """如果已启用，则在每个进程中运行一次 Docker 孤儿回收器。

    扫描标记有当前配置文件 ``hermes-agent=1`` 的长期处于已退出（Exited）状态的容器，
    这些容器符合 issue #20561 泄漏类的特征 — 即 Hermes 进程在未触发 ``atexit``
    的情况下退出（如遭遇 SIGKILL、OOM、终端窗口关闭）所留下的容器。
    默认情况下，回收策略较为保守：
    仅针对停用时间超过 ``2 × lifetime_seconds`` 且作用域属于当前配置文件的已退出容器。

    门控条件：

    * ``terminal.docker_orphan_reaper: false`` 会将其彻底禁用（运维人员选择了退出 —
      通常是因为他们要在同一个配置文件下运行多个 Hermes 进程，
      且不信任保守的默认设定）。
    * ``_docker_orphan_reaper_ran`` 标识 — 扫描会在每个 Python 解释器中仅运行一次，
      而不是在每次子 agent 调用 / RL-rollout / 并发调用 ``terminal()`` 时都重复运行。
    """
    global _docker_orphan_reaper_ran
    if not container_config.get("docker_orphan_reaper", True):
        return
    # 廉价的双重检查锁定（Double-Checked Locking）：
    # 无锁状态下先读取一次；仅在首次运行时获取锁，并在锁内部再次检查。
    if _docker_orphan_reaper_ran:
        return
    with _docker_orphan_reaper_lock:
        if _docker_orphan_reaper_ran:
            return
        _docker_orphan_reaper_ran = True

    # 2 × lifetime_seconds 为兄弟 Hermes 进程提供了充裕的宽限期。
    # 下限设置为 60 秒，以防运维人员将 TERMINAL_LIFETIME_SECONDS 设为 0 时，
    # 触发立即回收，从而与其自身的初始化设置产生竞态条件。
    # ``container_config`` 仅包含 container_* 类型的配置键，
    # 因此请从该模块其他部分所使用的环境变量中读取 lifetime_seconds。
    try:
        lifetime = int(os.getenv("TERMINAL_LIFETIME_SECONDS", "300"))
    except (TypeError, ValueError):
        lifetime = 300
    lifetime = max(60, lifetime)
    max_age = lifetime * 2

    try:
        from tools.environments.docker import (
            reap_orphan_containers, _get_active_profile_name,
        )
    except ImportError:
        return
    try:
        profile = _get_active_profile_name()
        removed = reap_orphan_containers(
            max_age_seconds=max_age, profile_filter=profile,
        )
        if removed:
            logger.info(
                "Docker orphan reaper removed %d stale container(s) for profile %s",
                removed, profile,
            )
    except Exception as e:
        # Never fail the env-creation path because of a janitor problem.
        logger.debug("Docker orphan reaper raised: %s", e)


# 按任务划分的环境覆盖注册表。
# 允许特定环境（例如 TerminalBench2Env）在 agent 循环开始之前，
# 为特定的 task_id 指定自定义的 Docker/Modal 镜像。
# 当终端或文件工具为该 task_id 创建新沙箱时，
# 会首先检查此注册表；如果未设置任何覆盖项，
# 则退回到 TERMINAL_MODAL_IMAGE（等）环境变量。
#
# 此注册表绝不对模型暴露 — 仅基础架构代码会调用它。
# 线程安全，因为每个 rollout 中的 task_id 都是唯一的。
_task_env_overrides: Dict[str, Dict[str, Any]] = {}


def register_task_env_overrides(task_id: str, overrides: Dict[str, Any]):
    """
    Register environment overrides for a specific task/rollout.

    Called by Atropos environments before the agent loop to configure
    per-task sandbox settings (e.g., a custom Dockerfile for the Modal image).

    Supported override keys:
        - modal_image: str -- Path to Dockerfile or Docker Hub image name
        - docker_image: str -- Docker image name
        - cwd: str -- Working directory inside the sandbox

    Args:
        task_id: The rollout's unique task identifier
        overrides: Dict of config keys to override
    """
    _task_env_overrides[task_id] = overrides

    # If a live environment already exists for this task, a freshly registered
    # ``cwd`` override (e.g. the ACP client switching the editor's project root
    # mid-session via ``session/load`` / ``session/resume``) must take effect on
    # the cached env too. ``terminal_tool`` resolves the per-command cwd as
    # ``workdir > env.cwd > config/override cwd`` so that ordinary in-session
    # ``cd`` state is preserved; without syncing here the override would sit
    # below the (already-set) ``env.cwd`` and be silently ignored once any
    # command has run. Pushing it onto the live env keeps ``cd`` tracking intact
    # while letting an explicit ACP cwd change win, as the client expects.
    new_cwd = overrides.get("cwd")
    if isinstance(new_cwd, str) and new_cwd.strip():
        # The live env is cached under the raw task_id for per-session surfaces
        # (ACP/gateway/dashboard) and under the collapsed container id for
        # isolation-keyed rollouts. Try the raw id first, then the container id,
        # so a CWD-only override (which collapses to "default") still finds and
        # updates the originating session's env.
        container_id = _resolve_container_task_id(task_id)
        with _env_lock:
            env = _active_environments.get(task_id) or _active_environments.get(container_id)
        if env is not None and getattr(env, "cwd", None) is not None:
            env.cwd = new_cwd


def clear_task_env_overrides(task_id: str):
    """
    Clear environment overrides for a task after rollout completes.

    Called during cleanup to avoid stale entries accumulating.
    """
    _task_env_overrides.pop(task_id, None)


def _resolve_container_task_id(task_id: Optional[str]) -> str:
    """
    将工具调用的 ``task_id`` 映射为
    ``_active_environments`` 所使用的容器/沙箱键。

    顶级 agent 传入 ``task_id=None`` 并映射至 ``"default"``。
    ``delegate_task`` 子进程传入它们自己的子 agent ID，以便
    文件状态跟踪、活跃子 agent 注册表和 TUI 事件对于每个子进程
    保持独立 — 但我们在此时有意将该 ID 折叠回
    ``"default"``，从而让子 agent 共享父进程的常驻容器
    （同一个 bash、同一个 /workspace 以及同一套已安装的软件包）。

    特例：RL / 基准测试环境（TerminalBench2, HermesSweEnv, ...）
    会调用 ``register_task_env_overrides(task_id, {...})`` 来申请
    按任务隔离的 Docker/Modal 镜像。当某个 task_id 注册了覆盖配置时，
    我们会保持 task_id 原样返回以尊重该配置 — 这些 Rollout 任务
    需要属于它们自己的独立沙箱，这正是覆盖配置的初衷所在。

    仅针对工作目录（CWD）的覆盖（由 ACP 适配器注册用于工作区跟踪）
    *不属于* 隔离信号 — 它们不应该导致每个会话都去
    启动各自独立的容器。只有包含特定后端镜像键或
    ``env_type`` 的覆盖配置才会触发隔离。
    """
    _ISOLATION_KEYS = frozenset({
        "docker_image", "modal_image", "singularity_image",
        "daytona_image", "env_type",
    })
    if task_id and task_id in _task_env_overrides:
        overrides = _task_env_overrides[task_id]
        if set(overrides.keys()) & _ISOLATION_KEYS:
            return task_id
    return "default"


def resolve_task_overrides(task_id: Optional[str]) -> Dict[str, Any]:
    """返回 *task_id* 的环境覆盖配置，优先读取原始键，再读取折叠后的键。

    ``register_task_env_overrides`` 在*原始*任务/会话 ID 下写入配置，
    但仅针对工作目录（CWD）的覆盖配置会折叠（:func:`_resolve_container_task_id`）
    为共享的 ``"default"`` 容器，从而避免按会话划分的界面
    （ACP/网关/仪表盘）各自启动独立的沙箱。
    因此，需要该覆盖配置的调用方（终端命令设置、文件工具工作目录解析）
    必须**优先**读取原始 ID，并在其不存在时才退回到折叠后的容器 ID，
    否则原始会话的覆盖配置会被静默丢弃。
    这是该查找逻辑的唯一数据源，可确保终端层和文件层不会产生不一致。
    """
    raw = task_id or "default"
    return (
        _task_env_overrides.get(raw)
        or _task_env_overrides.get(_resolve_container_task_id(raw))
        or {}
    )


# Configuration from environment variables

def _parse_env_var(name: str, default: str, converter: Any = int, type_label: str = "integer"):
    """Parse an environment variable with *converter*, raising a clear error on bad values.

    Without this wrapper, a single malformed env var (e.g. TERMINAL_TIMEOUT=5m)
    causes an unhandled ValueError that kills every terminal command.
    """
    raw = os.getenv(name, default)
    try:
        return converter(raw)
    except (ValueError, json.JSONDecodeError):
        raise ValueError(
            f"Invalid value for {name}: {raw!r} (expected {type_label}). "
            f"Check ~/.hermes/.env or environment variables."
        )


def _safe_getcwd() -> str:
    """Return the current working directory, tolerating a deleted CWD.

    ``os.getcwd()`` raises FileNotFoundError when the process's working
    directory has been removed out from under it (e.g. a scratch workspace
    that was cleaned up mid-session). Fall back to TERMINAL_CWD, then the
    user's home directory, so terminal setup never crashes on a stale CWD.
    """
    try:
        return os.getcwd()
    except FileNotFoundError:
        return os.getenv("TERMINAL_CWD") or os.path.expanduser("~")


# Path prefixes that identify a *host* working directory which cannot exist
# inside a container sandbox. Covers POSIX user dirs and Windows drive paths
# (``C:\Users\...`` / ``C:/Users/...``) — the latter is how a Windows host's
# cwd looks when it leaks toward a Linux container's ``-w`` flag.
_HOST_CWD_PREFIXES = ("/Users/", "/home/", "C:\\", "C:/")

_CONTAINER_BACKENDS = frozenset({"docker", "singularity", "modal", "daytona"})


def _is_ssh_remote_tilde_cwd(backend: str, cwd: str) -> bool:
    """Return True when *cwd* is a tilde path that the remote SSH shell must
    expand itself, so the Hermes host/container must NOT ``expanduser`` it.

    SSH ``cwd`` is interpreted by the *remote* shell (``cd ~`` / ``cd ~/x``
    over ``ssh ... bash -c``). Expanding ``~`` locally would rewrite it to the
    Hermes host HOME (often ``/opt/data`` under Docker) and inject a
    nonexistent path into the remote session. Only ``~`` / ``~/...`` on the
    ``ssh`` backend qualify; absolute remote paths still pass through
    unchanged, and every other backend keeps expanding locally.
    """
    if (backend or "").strip().lower() != "ssh":
        return False
    return cwd == "~" or cwd.startswith("~/")


def _is_unusable_container_cwd(cwd: str) -> bool:
    """如果 *cwd* 是宿主机路径或相对路径，且无法在容器沙箱内
    用作工作目录，则返回 True。

    容器的 cwd 必须是*存在于沙箱内部*的绝对路径
    （例如 ``/workspace`` 或 ``/root``）。
    宿主机路径（如 ``/home/user``、``C:\\Users\\me``）
    或相对路径（如 ``.``、``src/``）对于 ``docker run -w`` 而言没有任何意义，
    会导致容器无法启动（退出码 125）。
    """
    if not cwd:
        return False
    if any(cwd.startswith(p) for p in _HOST_CWD_PREFIXES):
        return True
    # 相对路径（例如 "."、"src/"）同样不能用作容器的工作目录。
    # Windows 驱动器路径在 Windows 系统上是绝对路径，
    # 但在 POSIX 宿主机上 os.path.isabs() 会返回 False，
    # 因此它们已经会被上方的前缀检查捕获。
    if not os.path.isabs(cwd):
        return True
    return False


def _get_env_config() -> Dict[str, Any]:
    """Get terminal environment configuration from environment variables."""
    # Default image with Python and Node.js for maximum compatibility
    default_image = "nikolaik/python-nodejs:python3.11-nodejs20"
    env_type = os.getenv("TERMINAL_ENV", "local")
    
    mount_docker_cwd = os.getenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "false").lower() in {"true", "1", "yes"}
    container_backend = env_type in {"docker", "singularity", "modal", "daytona"}
    docker_backend = env_type == "docker"

    # Docker/container-only env vars may be bridged from config.yaml even when
    # the active backend is local/ssh.  Do not parse their JSON/numeric payloads
    # until a backend that can consume them is selected; a stale or invalid
    # Docker value should not make local terminal/execute_code unusable.
    if container_backend:
        container_cpu = _parse_env_var("TERMINAL_CONTAINER_CPU", "1", float, "number")
        container_memory = _parse_env_var("TERMINAL_CONTAINER_MEMORY", "5120")
        container_disk = _parse_env_var("TERMINAL_CONTAINER_DISK", "51200")
    else:
        container_cpu = 1.0
        container_memory = 5120
        container_disk = 51200

    if docker_backend:
        docker_forward_env = _parse_env_var("TERMINAL_DOCKER_FORWARD_ENV", "[]", json.loads, "valid JSON")
        docker_volumes = _parse_env_var("TERMINAL_DOCKER_VOLUMES", "[]", json.loads, "valid JSON")
        docker_env = _parse_env_var("TERMINAL_DOCKER_ENV", "{}", json.loads, "valid JSON")
        docker_extra_args = _parse_env_var("TERMINAL_DOCKER_EXTRA_ARGS", "[]", json.loads, "valid JSON")
    else:
        docker_forward_env = []
        docker_volumes = []
        docker_env = {}
        docker_extra_args = []

    # Default cwd: local uses the host's current directory, ssh uses the
    # remote home, and everything else starts in the backend's default
    # root-like cwd.
    if env_type == "local":
        default_cwd = _safe_getcwd()
    elif env_type == "ssh":
        default_cwd = "~"
    else:
        default_cwd = "/root"

    # Read TERMINAL_CWD but sanity-check it for container backends.
    # If Docker cwd passthrough is explicitly enabled, remap the host path to
    # /workspace and track the original host path separately. Otherwise keep the
    # normal sandbox behavior and discard host paths.
    cwd = os.getenv("TERMINAL_CWD", default_cwd)
    if cwd and not _is_ssh_remote_tilde_cwd(env_type, cwd):
        cwd = os.path.expanduser(cwd)
    host_cwd = None
    if env_type == "docker" and mount_docker_cwd:
        docker_cwd_source = os.getenv("TERMINAL_CWD") or _safe_getcwd()
        candidate = os.path.abspath(os.path.expanduser(docker_cwd_source))
        if (
            any(candidate.startswith(p) for p in _HOST_CWD_PREFIXES)
            or (os.path.isabs(candidate) and os.path.isdir(candidate) and not candidate.startswith(("/workspace", "/root")))
        ):
            host_cwd = candidate
            cwd = "/workspace"
    elif env_type in _CONTAINER_BACKENDS and cwd:
        # Host paths and relative paths that won't work inside containers
        if _is_unusable_container_cwd(cwd) and cwd != default_cwd:
            logger.info("Ignoring TERMINAL_CWD=%r for %s backend "
                        "(host/relative path won't work in sandbox). Using %r instead.",
                        cwd, env_type, default_cwd)
            cwd = default_cwd

    return {
        "env_type": env_type,
        "modal_mode": coerce_modal_mode(os.getenv("TERMINAL_MODAL_MODE", "auto")),
        "docker_image": os.getenv("TERMINAL_DOCKER_IMAGE", default_image),
        "docker_forward_env": docker_forward_env,
        "singularity_image": os.getenv("TERMINAL_SINGULARITY_IMAGE", f"docker://{default_image}"),
        "modal_image": os.getenv("TERMINAL_MODAL_IMAGE", default_image),
        "daytona_image": os.getenv("TERMINAL_DAYTONA_IMAGE", default_image),
        "cwd": cwd,
        "host_cwd": host_cwd,
        "docker_mount_cwd_to_workspace": mount_docker_cwd,
        "timeout": _parse_env_var("TERMINAL_TIMEOUT", "180"),
        "lifetime_seconds": _parse_env_var("TERMINAL_LIFETIME_SECONDS", "300"),
        # SSH-specific config
        "ssh_host": os.getenv("TERMINAL_SSH_HOST", ""),
        "ssh_user": os.getenv("TERMINAL_SSH_USER", ""),
        "ssh_port": _parse_env_var("TERMINAL_SSH_PORT", "22"),
        "ssh_key": os.getenv("TERMINAL_SSH_KEY", ""),
        # Persistent shell: SSH defaults to the config-level persistent_shell
        # setting (true by default for non-local backends); local is always opt-in.
        # Per-backend env vars override if explicitly set.
        "ssh_persistent": os.getenv(
            "TERMINAL_SSH_PERSISTENT",
            os.getenv("TERMINAL_PERSISTENT_SHELL", "true"),
        ).lower() in {"true", "1", "yes"},
        "local_persistent": os.getenv("TERMINAL_LOCAL_PERSISTENT", "false").lower() in {"true", "1", "yes"},
        # Container resource config (applies to docker, singularity, modal,
        # daytona -- ignored for local/ssh)
        "container_cpu": container_cpu,
        "container_memory": container_memory,     # MB (default 5GB)
        "container_disk": container_disk,        # MB (default 50GB)
        "container_persistent": os.getenv("TERMINAL_CONTAINER_PERSISTENT", "true").lower() in {"true", "1", "yes"},
        "docker_volumes": docker_volumes,
        "docker_env": docker_env,
        "docker_run_as_host_user": os.getenv("TERMINAL_DOCKER_RUN_AS_HOST_USER", "false").lower() in {"true", "1", "yes"},
        "docker_network": os.getenv("TERMINAL_DOCKER_NETWORK", "true").lower() in {"true", "1", "yes"},
        "docker_extra_args": docker_extra_args,
        # Cross-process container reuse (issue #20561).  The docs claim
        # "ONE long-lived container shared across sessions" — this toggle
        # makes that real by probing for a labeled container at startup and
        # attaching to it instead of always starting a fresh one.  Set to
        # ``false`` for hard per-process isolation (no reuse, container is
        # removed on exit).
        "docker_persist_across_processes": os.getenv(
            "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES", "true"
        ).lower() in {"true", "1", "yes"},
        # Startup orphan reaper for hermes-tagged containers left behind by
        # crashed / SIGKILL'd previous processes that bypassed atexit.
        # Conservative: only sweeps Exited containers older than 2× the
        # idle-reap window AND scoped to the current profile. Issue #20561.
        "docker_orphan_reaper": os.getenv(
            "TERMINAL_DOCKER_ORPHAN_REAPER", "true"
        ).lower() in {"true", "1", "yes"},
    }


def _get_modal_backend_state(modal_mode: object | None) -> Dict[str, Any]:
    """Resolve direct vs managed Modal backend selection."""
    return resolve_modal_backend_state(
        modal_mode,
        has_direct=has_direct_modal_credentials(),
        managed_ready=is_managed_tool_gateway_ready("modal"),
    )


def _create_environment(env_type: str, image: str, cwd: str, timeout: int,
                        ssh_config: dict = None, container_config: dict = None,
                        local_config: dict = None,
                        task_id: str = "default",
                        host_cwd: str = None):
    """
    创建一个用于沙箱化命令执行的执行环境。

    参数:
        env_type: "local"、"docker"、"singularity"、"modal"、
            "daytona"、"ssh" 中的一种
        image: Docker/Singularity/Modal 镜像名称（对于 local/ssh 会被忽略）
        cwd: 工作目录
        timeout: 默认命令超时时间
        ssh_config: SSH 连接配置（用于 env_type="ssh"）
        container_config: 容器后端的资源配置（包含 cpu、memory、disk、persistent）
        task_id: 用于环境复用和快照键定的任务标识符
        host_cwd: 显式启用时，要挂载/绑定到 Docker 中的可选宿主机工作目录

    返回:
        带有 execute() 方法的环境实例
    """
    cc = container_config or {}
    cpu = cc.get("container_cpu", 1)
    memory = cc.get("container_memory", 5120)
    disk = cc.get("container_disk", 51200)
    persistent = cc.get("container_persistent", True)
    volumes = cc.get("docker_volumes", [])
    docker_forward_env = cc.get("docker_forward_env", [])
    docker_env = cc.get("docker_env", {})
    docker_extra_args = cc.get("docker_extra_args", [])
    docker_network = cc.get("docker_network", True)

    if env_type == "local":
        return _LocalEnvironment(cwd=cwd, timeout=timeout)
    
    elif env_type == "docker":
        # 一次性孤儿回收器：清理先前 Hermes 进程留下的带标签容器，
        # 这些进程在 atexit 清理钩子运行前就遭遇了 SIGKILL / OOM / 终端关闭。
        # 每个进程受门控保护仅运行一次，
        # 因此并发的 _create_environment 调用（并行子 agent、RL 基准测试）
        # 不会重复运行回收器 N 次。
        # 可通过设置 ``terminal.docker_orphan_reaper: false`` 来禁用（issue #20561）。
        _maybe_reap_docker_orphans(cc)
        return _DockerEnvironment(
            image=image, cwd=cwd, timeout=timeout,
            cpu=cpu, memory=memory, disk=disk,
            persistent_filesystem=persistent, task_id=task_id,
            volumes=volumes,
            host_cwd=host_cwd,
            auto_mount_cwd=cc.get("docker_mount_cwd_to_workspace", False),
            forward_env=docker_forward_env,
            env=docker_env,
            run_as_host_user=cc.get("docker_run_as_host_user", False),
            network=docker_network,
            extra_args=docker_extra_args,
            persist_across_processes=cc.get("docker_persist_across_processes", True),
        )
    
    elif env_type == "singularity":
        return _SingularityEnvironment(
            image=image, cwd=cwd, timeout=timeout,
            cpu=cpu, memory=memory, disk=disk,
            persistent_filesystem=persistent, task_id=task_id,
        )
    
    elif env_type == "modal":
        sandbox_kwargs = {}
        if cpu > 0:
            sandbox_kwargs["cpu"] = cpu
        if memory > 0:
            sandbox_kwargs["memory"] = memory
        if disk > 0:
            try:
                import inspect, modal
                if "ephemeral_disk" in inspect.signature(modal.Sandbox.create).parameters:
                    sandbox_kwargs["ephemeral_disk"] = disk
            except Exception:
                pass

        modal_state = _get_modal_backend_state(cc.get("modal_mode"))

        if modal_state["selected_backend"] == "managed":
            return _ManagedModalEnvironment(
                image=image, cwd=cwd, timeout=timeout,
                modal_sandbox_kwargs=sandbox_kwargs,
                persistent_filesystem=persistent, task_id=task_id,
            )

        if modal_state["selected_backend"] != "direct":
            if modal_state["managed_mode_blocked"]:
                raise ValueError(
                    "Modal backend is configured for managed mode, but "
                    "Nous Tool Gateway access is not currently available and no direct "
                    "Modal credentials/config were found. "
                    + nous_tool_gateway_unavailable_message(
                        "managed Modal execution",
                    )
                    + " Choose TERMINAL_MODAL_MODE=direct/auto to use direct Modal credentials."
                )
            if modal_state["mode"] == "managed":
                raise ValueError(
                    "Modal backend is configured for managed mode, but the managed tool gateway is unavailable. "
                    + nous_tool_gateway_unavailable_message(
                        "managed Modal execution",
                    )
                )
            if modal_state["mode"] == "direct":
                raise ValueError(
                    "Modal backend is configured for direct mode, but no direct Modal credentials/config were found."
                )
            message = "Modal backend selected but no direct Modal credentials/config was found."
            if managed_nous_tools_enabled():
                message = (
                    "Modal backend selected but no direct Modal credentials/config or managed tool gateway was found."
                )
            raise ValueError(message)

        return _ModalEnvironment(
            image=image, cwd=cwd, timeout=timeout,
            modal_sandbox_kwargs=sandbox_kwargs,
            persistent_filesystem=persistent, task_id=task_id,
        )
    
    elif env_type == "daytona":
        # Lazy import so daytona SDK is only required when backend is selected.
        from tools.environments.daytona import DaytonaEnvironment as _DaytonaEnvironment
        return _DaytonaEnvironment(
            image=image, cwd=cwd, timeout=timeout,
            cpu=int(cpu), memory=memory, disk=disk,
            persistent_filesystem=persistent, task_id=task_id,
        )

    elif env_type == "ssh":
        if not ssh_config or not ssh_config.get("host") or not ssh_config.get("user"):
            raise ValueError("SSH environment requires ssh_host and ssh_user to be configured")
        return _SSHEnvironment(
            host=ssh_config["host"],
            user=ssh_config["user"],
            port=ssh_config.get("port", 22),
            key_path=ssh_config.get("key", ""),
            cwd=cwd,
            timeout=timeout,
        )

    else:
        raise ValueError(
            f"Unknown environment type: {env_type}. Use 'local', 'docker', "
            f"'singularity', 'modal', 'daytona', or 'ssh'"
        )


def _cleanup_inactive_envs(lifetime_seconds: int = 300):
    """清理停用时间超过 lifetime_seconds 的环境。"""
    current_time = time.time()

    # 检查进程注册表 — 跳过对包含活跃后台进程的沙箱的清理
    # （这些沙箱的 _last_activity 会被刷新以维持其存活状态）。
    try:
        from tools.process_registry import process_registry
        for task_id in list(_last_activity.keys()):
            if process_registry.has_active_processes(task_id):
                _last_activity[task_id] = current_time  # Keep sandbox alive
    except ImportError:
        pass

    # 阶段 1：在持有锁的同时，收集过期的条目并从跟踪字典中将其移除。
    # 切勿在锁内部调用 env.cleanup() —— Modal 和 Docker 的销毁过程
    # 可能会阻塞 10 到 15 秒，这会拖慢所有正在等待 _env_lock 的
    # 并发终端/文件工具调用。
    envs_to_stop = []  # list of (task_id, env) pairs

    with _env_lock:
        for task_id, last_time in list(_last_activity.items()):
            if current_time - last_time > lifetime_seconds:
                env = _active_environments.pop(task_id, None)
                _last_activity.pop(task_id, None)
                if env is not None:
                    envs_to_stop.append((task_id, env))

        # Also purge per-task creation locks for cleaned-up tasks
        with _creation_locks_lock:
            for task_id, _ in envs_to_stop:
                _creation_locks.pop(task_id, None)

    # 阶段 2：在锁外部停止实际的沙箱，
    # 从而在 Modal/Docker 沙箱关闭时
    # 不会阻塞其他的工具调用。
    for task_id, env in envs_to_stop:
        # Invalidate stale file_ops cache entry (Bug fix: prevents
        # ShellFileOperations from referencing a dead sandbox)
        try:
            from tools.file_tools import clear_file_ops_cache
            clear_file_ops_cache(task_id)
        except ImportError:
            pass

        try:
            if hasattr(env, 'cleanup'):
                env.cleanup()
            elif hasattr(env, 'stop'):
                env.stop()
            elif hasattr(env, 'terminate'):
                env.terminate()

            logger.info("Cleaned up inactive environment for task: %s", task_id)

        except Exception as e:
            error_str = str(e)
            if "404" in error_str or "not found" in error_str.lower():
                logger.info("Environment for task %s already cleaned up", task_id)
            else:
                logger.warning("Error cleaning up environment for task %s: %s", task_id, e)


def _cleanup_thread_worker():
    """Background thread worker that periodically cleans up inactive environments."""
    while _cleanup_running:
        try:
            config = _get_env_config()
            _cleanup_inactive_envs(config["lifetime_seconds"])
        except Exception as e:
            logger.warning("Error in cleanup thread: %s", e, exc_info=True)

        for _ in range(60):
            if not _cleanup_running:
                break
            time.sleep(1)


def _start_cleanup_thread():
    """Start the background cleanup thread if not already running."""
    global _cleanup_thread, _cleanup_running

    with _env_lock:
        if _cleanup_thread is None or not _cleanup_thread.is_alive():
            _cleanup_running = True
            _cleanup_thread = threading.Thread(target=_cleanup_thread_worker, daemon=True)
            _cleanup_thread.start()


def _stop_cleanup_thread():
    """Stop the background cleanup thread."""
    global _cleanup_running
    _cleanup_running = False
    if _cleanup_thread is not None:
        try:
            _cleanup_thread.join(timeout=5)
        except (SystemExit, KeyboardInterrupt):
            pass


def get_active_env(task_id: str):
    """Return the active BaseEnvironment for *task_id*, or None."""
    lookup = _resolve_container_task_id(task_id)
    with _env_lock:
        return _active_environments.get(lookup) or _active_environments.get(task_id)


def is_persistent_env(task_id: str) -> bool:
    """Return True if the active environment for task_id is configured for
    cross-turn persistence (``persistent_filesystem=True``).

    Used by the agent loop to skip per-turn teardown for backends whose whole
    point is to survive between turns (docker with ``container_persistent``,
    daytona, modal, etc.). Non-persistent backends (e.g. Morph) still get torn
    down at end-of-turn to prevent leakage. The idle reaper
    (``_cleanup_inactive_envs``) handles persistent envs once they exceed
    ``terminal.lifetime_seconds``.
    """
    env = get_active_env(task_id)
    if env is None:
        return False
    return bool(getattr(env, "_persistent", False))




def cleanup_all_environments():
    """Clean up ALL active environments. Use with caution."""
    task_ids = list(_active_environments.keys())
    cleaned = 0
    
    for task_id in task_ids:
        try:
            cleanup_vm(task_id)
            cleaned += 1
        except Exception as e:
            logger.error("Error cleaning %s: %s", task_id, e, exc_info=True)
    
    # Also clean any orphaned directories
    scratch_dir = _get_scratch_dir()
    import glob
    for path in glob.glob(str(scratch_dir / "hermes-*")):
        try:
            shutil.rmtree(path, ignore_errors=True)
            logger.info("Removed orphaned: %s", path)
        except OSError as e:
            logger.debug("Failed to remove orphaned path %s: %s", path, e)
    
    if cleaned > 0:
        logger.info("Cleaned %d environments", cleaned)
    return cleaned


def cleanup_vm(task_id: str, *, force_remove: bool = False):
    """根据 task_id 手动清理指定的环境。

    *force_remove*（默认为 False）会转发给支持该参数的后端
    —— 目前仅有 ``DockerEnvironment``。
    False 的默认值符合会话生命周期的语义：
    此函数会在 ``AIAgent.close()``（TUI 会话关闭、网关会话销毁）
    以及非持久化环境的单轮清理分支中被调用，
    这两者都应当尊重用户的持久化模式偏好。
    若在此处停止容器，将破坏“跨会话共享单个长生命周期容器”的约定
    —— 这正是 Ben 汇报过的 Bug（当时每次 TUI 会话关闭都会杀死容器）。

    对于真正由用户发起的销毁操作
    （例如尚未接入的 ``/reset`` 风格流程，或未来的“销毁我的沙盒”命令），
    请传递 ``force_remove=True``。

    空闲回收程序会直接通过 ``env.cleanup()`` 处理环境（而不通过此函数），
    因此处于持久化模式的空闲环境同样会执行空操作（no-op）
    —— 只有下次启动时的孤立资源回收程序（orphan reaper）才会回收它们。
    """
    # 在持有锁的同时从追踪字典中移除，
    # 但将实际（可能较慢）的 env.cleanup() 调用延迟到锁释放后执行，
    # 以免阻塞其他工具调用。
    env = None
    with _env_lock:
        env = _active_environments.pop(task_id, None)
        _last_activity.pop(task_id, None)

    # Clean up per-task creation lock
    with _creation_locks_lock:
        _creation_locks.pop(task_id, None)

    # Invalidate stale file_ops cache entry
    try:
        from tools.file_tools import clear_file_ops_cache
        clear_file_ops_cache(task_id)
    except ImportError:
        pass

    if env is None:
        return

    try:
        if hasattr(env, 'cleanup'):
            # Pass force_remove only if the env's cleanup() accepts it
            # (DockerEnvironment after issue #20561; other backends don't).
            import inspect
            sig = inspect.signature(env.cleanup)
            if "force_remove" in sig.parameters:
                env.cleanup(force_remove=force_remove)
            else:
                env.cleanup()
        elif hasattr(env, 'stop'):
            env.stop()
        elif hasattr(env, 'terminate'):
            env.terminate()

        logger.info("Manually cleaned up environment for task: %s", task_id)

    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "not found" in error_str.lower():
            logger.info("Environment for task %s already cleaned up", task_id)
        else:
            logger.warning("Error cleaning up environment for task %s: %s", task_id, e)


def _atexit_cleanup():
    """Stop cleanup thread and shut down all remaining sandboxes on exit."""
    _stop_cleanup_thread()
    if _active_environments:
        count = len(_active_environments)
        logger.info("Shutting down %d remaining sandbox(es)...", count)
        # Snapshot the env objects BEFORE cleanup_all_environments empties
        # the dict; we need them to wait on docker cleanup threads after the
        # registry has been cleared.
        envs_to_wait = list(_active_environments.values())
        cleanup_all_environments()
        # Block briefly so docker stop/rm actually completes before the
        # interpreter exits. Issue #20561 — without this join, the daemon
        # cleanup threads were getting torn down mid-`docker stop`, leaving
        # Exited containers piled up on the host.
        for env in envs_to_wait:
            wait_fn = getattr(env, "wait_for_cleanup", None)
            if wait_fn is None:
                continue
            try:
                wait_fn(timeout=15.0)
            except Exception as e:  # never block shutdown on a bad backend
                logger.debug("wait_for_cleanup raised on exit: %s", e)

atexit.register(_atexit_cleanup)


# =============================================================================
# 常用 CLI 工具的退出码上下文
# =============================================================================
# 许多 Unix 命令会使用非零退出码来表示提示性信息，而非指示命令执行失败。
# 模型在看到来自 `grep` 的原始 exit_code=1 时，
# 会白白浪费轮次去排查一个仅仅代表“未匹配到结果”的现象。
# 此查找表添加了人类可读的备注说明，
# 以便 Agent 能够直接继续下一步操作。

def _interpret_exit_code(command: str, exit_code: int) -> str | None:
    """当非零退出码并不代表错误时，返回一条人类可读的备注说明。

    当退出码为 0 或确实指示错误时返回 None。
    该备注说明会被附加到工具执行结果中，
    以避免模型浪费轮次去排查预期内的退出码。
    """
    if exit_code == 0:
        return None

    # 提取管道或命令链中的最后一个命令 —— 它决定了最终的退出码。
    # 支持 `cmd1 && cmd2`、`cmd1 | cmd2` 以及 `cmd1; cmd2`。
    # 逻辑故意保持简洁：直接按 Shell 运算符分割，并获取最后一个片段。
    segments = re.split(r'\s*(?:\|\||&&|[|;])\s*', command)
    last_segment = (segments[-1] if segments else command).strip()

    # Get base command name (first word), stripping env var assignments
    # like  VAR=val cmd ...
    words = last_segment.split()
    base_cmd = ""
    for w in words:
        if "=" in w and not w.startswith("-"):
            continue  # skip VAR=val
        base_cmd = w.split("/")[-1]  # handle /usr/bin/grep -> grep
        break

    if not base_cmd:
        return None

    # Command-specific semantics
    semantics: dict[str, dict[int, str]] = {
        # grep/rg/ag/ack: 1=no matches found (normal), 2+=real error
        "grep":  {1: "No matches found (not an error)"},
        "egrep": {1: "No matches found (not an error)"},
        "fgrep": {1: "No matches found (not an error)"},
        "rg":    {1: "No matches found (not an error)"},
        "ag":    {1: "No matches found (not an error)"},
        "ack":   {1: "No matches found (not an error)"},
        # diff: 1=files differ (expected), 2+=real error
        "diff":  {1: "Files differ (expected, not an error)"},
        "colordiff": {1: "Files differ (expected, not an error)"},
        # find: 1=some dirs inaccessible but results may still be valid
        "find":  {1: "Some directories were inaccessible (partial results may still be valid)"},
        # test/[: 1=condition is false (expected)
        "test":  {1: "Condition evaluated to false (expected, not an error)"},
        "[":     {1: "Condition evaluated to false (expected, not an error)"},
        # curl: common non-error codes
        "curl":  {
            6: "Could not resolve host",
            7: "Failed to connect to host",
            22: "HTTP response code indicated error (e.g. 404, 500)",
            28: "Operation timed out",
        },
        # git: 1 is context-dependent but often normal (e.g. git diff with changes)
        "git":   {1: "Non-zero exit (often normal — e.g. 'git diff' returns 1 when files differ)"},
    }

    cmd_semantics = semantics.get(base_cmd)
    if cmd_semantics and exit_code in cmd_semantics:
        return cmd_semantics[exit_code]

    return None


def _command_requires_pipe_stdin(command: str) -> bool:
    """当 PTY 模式会导致由 stdin 驱动的命令运行异常时，返回 True。

    某些 CLI 工具在 stdin 为 TTY 时会改变行为。
    特别是在使用 `gh auth login --with-token` 时，
    它期望通过管道传输的 stdin 接收令牌并等待 EOF 结束符；
    而当我们在 PTY 模式下启动该命令时，`process.submit()` 仅会发送一个换行符，
    导致该命令看起来在没有任何可见进度的状态下永久挂起。
    """
    normalized = " ".join(command.lower().split())
    return (
        normalized.startswith("gh auth login")
        and "--with-token" in normalized
    )


_SHELL_LEVEL_BACKGROUND_RE = re.compile(
    r"(?:^|[;&|]\s*|&&\s*|\|\|\s*|\$\(\s*)(?:nohup|disown|setsid)\b", re.IGNORECASE | re.MULTILINE
)
_INLINE_BACKGROUND_AMP_RE = re.compile(r"\s&\s")
_TRAILING_BACKGROUND_AMP_RE = re.compile(r"\s&\s*(?:#.*)?$")


def _strip_quotes(command: str) -> str:
    """移除单引号和双引号内的内容，防止正则表达式在字符串内部进行匹配。

    这可以避免当 'nohup' 或 'setsid' 等关键字出现在
    提交信息、Python -c 代码、echo 参数或 PR 正文文本中时触发误报。
    同时也会剥离反引号包裹的内容以及 heredoc 风格的内联文本。
    """
    # Remove single-quoted strings (no escaping inside single quotes in shell)
    result = re.sub(r"'[^']*'", "''", command)
    # Remove double-quoted strings (handle escaped quotes)
    result = re.sub(r'"(?:[^"\\]|\\.)*"', '""', result)
    # Remove backtick-quoted strings
    result = re.sub(r"`[^`]*`", "``", result)
    return result


_LONG_LIVED_FOREGROUND_PATTERNS = (
    re.compile(r"\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:dev|start|serve|watch)\b", re.IGNORECASE),
    re.compile(r"\bdocker\s+compose\s+up\b", re.IGNORECASE),
    re.compile(r"\bnext\s+dev\b", re.IGNORECASE),
    re.compile(r"\bvite(?:\s|$)", re.IGNORECASE),
    re.compile(r"\bnodemon\b", re.IGNORECASE),
    re.compile(r"\buvicorn\b", re.IGNORECASE),
    re.compile(r"\bgunicorn\b", re.IGNORECASE),
    re.compile(r"\bpython(?:3)?\s+-m\s+http\.server\b", re.IGNORECASE),
)


def _looks_like_help_or_version_command(command: str) -> bool:
    """Return True for informational invocations that should never be blocked."""
    normalized = " ".join(command.lower().split())
    return (
        " --help" in normalized
        or normalized.endswith(" -h")
        or " --version" in normalized
        or normalized.endswith(" -v")
    )


def _foreground_background_guidance(command: str) -> str | None:
    """当前台命令看起来属于长生命周期任务时，建议使用后台模式。

    防止因启动服务器/监听进程而导致工作流停滞，
    进而无法执行后续的检查或测试命令。
    """
    if _looks_like_help_or_version_command(command):
        return None

    # 剥离引用内容（字符串），避免字符串/参数内部的关键字触发
    # 误报（例如：git commit -m "... setsid ...", python3 -c "os.setsid"）。
    unquoted = _strip_quotes(command)

    if _SHELL_LEVEL_BACKGROUND_RE.search(unquoted):
        # return (
        #     "前台命令使用了 Shell 层面的后台包装器（nohup/disown/setsid）。"
        #     "请改用 terminal(background=true)，以便 Hermes 能够跟踪该进程，"
        #     "然后再通过独立的命令执行就绪性检查与测试。"
        # )
        return (
            "Foreground command uses shell-level background wrappers (nohup/disown/setsid). "
            "Use terminal(background=true) so Hermes can track the process, then run "
            "readiness checks and tests in separate commands."
        )

    if _INLINE_BACKGROUND_AMP_RE.search(unquoted) or _TRAILING_BACKGROUND_AMP_RE.search(unquoted):
        return (
            "Foreground command uses '&' backgrounding. Use terminal(background=true) for long-lived "
            "processes, then run health checks and tests in follow-up terminal calls."
        )

    for pattern in _LONG_LIVED_FOREGROUND_PATTERNS:
        if pattern.search(unquoted):
            return (
                "This foreground command appears to start a long-lived server/watch process. "
                "Run it with background=true, verify readiness (health endpoint/log signal), "
                "then execute tests in a separate command."
            )

    return None


def _resolve_notification_flag_conflict(
    *,
    notify_on_complete: bool,
    watch_patterns,
    background: bool,
) -> tuple:
    """决定当同时设置了 notify_on_complete 和 watch_patterns 时的处理策略。

    当这两者组合使用时，会产生重复且延迟的通知 ——
    既包含每一次监听模式匹配的通知，也包含进程退出时的通知，
    并且由于异步交付机制，可能会在进程结束很久之后依然向用户发送垃圾消息骚扰。
    当两者同时被设置时，我们将丢弃 watch_patterns，
    优先保留 notify_on_complete（即更有用的“当它完成时通知我”这一信号），
    并返回一条易于阅读的说明（human-readable note）。

    返回值：
        (watch_patterns_to_use, conflict_note)。
        当不存在冲突时，conflict_note 为 ""。
    """
    if background and notify_on_complete and watch_patterns:
        note = (
            "watch_patterns ignored because notify_on_complete=True; "
            "these two flags produce duplicate notifications when combined"
        )
        return None, note
    return watch_patterns, ""


def _resolve_command_cwd(
    *,
    workdir: Optional[str],
    env: Any,
    default_cwd: str,
) -> str:
    """返回命令的工作目录（cwd），优先使用当前会话的实时工作目录。

    在过去，``terminal_tool`` 每次调用时
    都会重新发送初始化阶段或配置中所设定的 cwd。
    这破坏了会话内部的 ``cd`` 状态：
    环境已经在 ``env.cwd`` 中追踪到了新的目录，
    但前台/后台的调用却持续通过 ``env.execute(..., cwd=...)``
    强制切回旧的 cwd。
    当然，显式传入的 ``workdir=`` 参数仍必须拥有最高优先级，覆盖所有其他设置。
    """
    if workdir:
        return workdir

    live_cwd = getattr(env, "cwd", None)
    if isinstance(live_cwd, str) and live_cwd.strip():
        return live_cwd

    return default_cwd


def terminal_tool(
    command: str,
    background: bool = False,
    timeout: Optional[int] = None,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    force: bool = False,
    workdir: Optional[str] = None,
    pty: bool = False,
    notify_on_complete: bool = False,
    watch_patterns: Optional[List[str]] = None,
) -> str:
    """
    在配置好的终端环境中执行命令。

    参数:
        command: 要执行的命令
        background: 是否在后台运行（默认：False）
        timeout: 命令超时时间，单位为秒（默认：来自配置）
        task_id: 用于环境隔离的唯一标识符（可选）
        session_id: 用于持久化可观测性的会话/对话标识符
        force: 如果为 True，则跳过危险命令检查（在用户确认后使用）
        workdir: 此命令的工作目录（可选，未设置时使用会话的当前工作目录）
        pty: 如果为 True，则为交互式 CLI 工具使用伪终端（仅限本地后端）
        notify_on_complete: 如果为 True 且 background=True，将在进程退出时仅接收一次通知。
        几乎适用于所有长时任务的明智选择。与 watch_patterns 互斥。
        watch_patterns: 在后台输出中要监听的字符串列表。
            存在严格的频率限制：每个进程每 15 秒最多发送 1 次通知。
            在连续触发 3 个惩罚窗口后，watch_patterns 将被禁用，且会话会自动升级为 notify_on_complete。
            仅可用于长生存期进程中罕见且一次性的进程中途信号（如服务器就绪信号、迁移完成标识）。
            切勿在循环或批量作业中使用 — 否则其中的错误匹配模式会触发惩罚限制并被禁用。
            与 notify_on_complete 互斥 — 只能设置二者之一。

    返回:
        str: 包含 output（输出）、exit_code（退出码）和 error（错误）字段的 JSON 字符串

    示例:
        # 执行简单命令
        >>> result = terminal_tool(command="ls -la /tmp")

        # 运行后台任务
        >>> result = terminal_tool(command="python server.py", background=True)

        # 使用自定义超时时间
        >>> result = terminal_tool(command="long_task.sh", timeout=300)

        # 用户确认后强制运行
        # 注意：force 参数仅限内部使用，不对模型 API 暴露
    """
    try:
        if not isinstance(command, str):
            logger.warning(
                "Rejected invalid terminal command value: %s",
                type(command).__name__,
            )
            return json.dumps({
                "output": "",
                "exit_code": -1,
                "error": f"Invalid command: expected string, got {type(command).__name__}",
                "status": "error",
            }, ensure_ascii=False)

        # Get configuration
        config = _get_env_config()
        env_type = config["env_type"]

        # 使用 task_id 进行环境隔离。在默认情况下，所有子 agent
        # 的 task_id 都会折叠回 "default"，因此顶级 agent 和
        # 每个 delegate_task 子进程都会共享同一个容器；只有配置了
        # 显式环境覆盖（如 RL 基准测试）的 task_id 才会获得独立的隔离沙箱。
        effective_task_id = _resolve_container_task_id(task_id)

        # 在退回到全局环境变量配置之前，先检查按任务设置的覆盖项
        # （由 TerminalBench2Env 等环境进行设置）。
        # ``resolve_task_overrides`` 会先读取原始任务 ID，然后再读取
        # 折叠后的容器 ID；因此，仅针对工作目录（CWD）的覆盖项
        # （它会将 ``effective_task_id`` 折叠为 ``"default"``）
        # 仍然可以在其原始会话 ID 下被找到，而带有隔离键的
        # RL/基准测试覆盖项则继续像以前一样进行解析。
        overrides = resolve_task_overrides(task_id)
        
        # Select image based on env type, with per-task override support
        if env_type == "docker":
            image = overrides.get("docker_image") or config["docker_image"]
        elif env_type == "singularity":
            image = overrides.get("singularity_image") or config["singularity_image"]
        elif env_type == "modal":
            image = overrides.get("modal_image") or config["modal_image"]
        elif env_type == "daytona":
            image = overrides.get("daytona_image") or config["daytona_image"]
        else:
            image = ""

        cwd = overrides.get("cwd") or config["cwd"]
        # 按任务设置的工作目录（cwd）覆盖项（由网关/TUI 注册用于工作区跟踪，
        # 或由 RL/基准测试环境注册）优先级高于 config["cwd"] — 但 config["cwd"]
        # 已经在 _get_env_config() 中针对容器后端进行了规范化处理，而该覆盖项则是原始路径。
        # 在容器后端上，原始宿主机路径（例如 Windows 桌面会话的 C:\Users\<user>，
        # 或 POSIX 系统上的 /home/<user>）会直接传递给 `docker run -w <host-path>`，
        # 从而导致容器启动失败（退出码 125）。因此需要对*解析后*的 cwd 重新应用
        # 相同的宿主机/相对路径防护机制，确保覆盖项无法绕过该约束。
        # 容器内部的有效覆盖路径（如将 cwd 设置为 /workspace、/root 等的 RL/基准测试沙箱）
        # 属于绝对非宿主机路径，会不受影响地直接通过。
        if env_type in _CONTAINER_BACKENDS and _is_unusable_container_cwd(cwd):
            if cwd != config["cwd"]:
                logger.info(
                    "Ignoring host/relative cwd override %r for %s backend "
                    "(won't exist in sandbox). Using %r instead.",
                    cwd, env_type, config["cwd"],
                )
            cwd = config["cwd"]
        default_timeout = config["timeout"]
        effective_timeout = timeout or default_timeout

        # 拒绝模型显式请求超时时间超过 FOREGROUND_MAX_TIMEOUT 的前台命令
        # — 引导其改为使用后台方式运行。
        if not background and timeout and timeout > FOREGROUND_MAX_TIMEOUT:
            return json.dumps({
                "error": (
                    f"Foreground timeout {timeout}s exceeds the maximum of "
                    f"{FOREGROUND_MAX_TIMEOUT}s. Use background=true with "
                    f"notify_on_complete=true for long-running commands."
                ),
            }, ensure_ascii=False)

        # 防护机制：长寿命服务器/监控命令应当作为受管后台会话运行，
        # 而不应使用前台 Shell 技巧。
        if not background:
            guidance = _foreground_background_guidance(command)
            if guidance:
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": guidance,
                    "status": "error",
                }, ensure_ascii=False)

        # Start cleanup thread
        _start_cleanup_thread()

        # 获取或创建环境。
        # 使用按任务划分的创建锁，以便针对同一个 task_id 的并发工具调用
        # 能够等待第一个调用完成沙箱创建，
        # 而不是各自创建一个沙箱（避免浪费 Modal 资源）。
        with _env_lock:
            # 优先使用折叠后的容器 ID，但在找不到时退回到
            # 在原始 task_id 下缓存的环境。
            # 带有仅针对 CWD 覆盖项的按会话界面（ACP/网关/仪表盘）
            # 会折叠为 "default" 以实现容器共享，
            # 但环境可能已经缓存于原始 task_id 下；
            # 此时应直接使用该环境，避免重复生成。
            _existing_key = (
                effective_task_id if effective_task_id in _active_environments
                else (task_id if task_id and task_id in _active_environments else None)
            )
            if _existing_key is not None:
                _last_activity[_existing_key] = time.time()
                env = _active_environments[_existing_key]
                needs_creation = False
            else:
                needs_creation = True

        if needs_creation:
            # Per-task lock: only one thread creates the sandbox, others wait
            with _creation_locks_lock:
                if effective_task_id not in _creation_locks:
                    _creation_locks[effective_task_id] = threading.Lock()
                task_lock = _creation_locks[effective_task_id]

            with task_lock:
                # Double-check after acquiring the per-task lock
                with _env_lock:
                    _existing_key = (
                        effective_task_id if effective_task_id in _active_environments
                        else (task_id if task_id and task_id in _active_environments else None)
                    )
                    if _existing_key is not None:
                        _last_activity[_existing_key] = time.time()
                        env = _active_environments[_existing_key]
                        needs_creation = False

                if needs_creation:
                    if env_type == "singularity":
                        _check_disk_usage_warning()
                    logger.info("Creating new %s environment for task %s...", env_type, effective_task_id[:8])
                    try:
                        ssh_config = None
                        if env_type == "ssh":
                            ssh_config = {
                                "host": config.get("ssh_host", ""),
                                "user": config.get("ssh_user", ""),
                                "port": config.get("ssh_port", 22),
                                "key": config.get("ssh_key", ""),
                                "persistent": config.get("ssh_persistent", False),
                            }

                        container_config = None
                        if env_type in {"docker", "singularity", "modal", "daytona"}:
                            container_config = {
                                "container_cpu": config.get("container_cpu", 1),
                                "container_memory": config.get("container_memory", 5120),
                                "container_disk": config.get("container_disk", 51200),
                                "container_persistent": config.get("container_persistent", True),
                                "modal_mode": config.get("modal_mode", "auto"),
                                "docker_volumes": config.get("docker_volumes", []),
                                "docker_mount_cwd_to_workspace": config.get("docker_mount_cwd_to_workspace", False),
                                "docker_forward_env": config.get("docker_forward_env", []),
                                "docker_env": config.get("docker_env", {}),
                                "docker_run_as_host_user": config.get("docker_run_as_host_user", False),
                                "docker_extra_args": config.get("docker_extra_args", []),
                                "docker_network": config.get("docker_network", True),
                                "docker_persist_across_processes": config.get("docker_persist_across_processes", True),
                                "docker_orphan_reaper": config.get("docker_orphan_reaper", True),
                            }

                        local_config = None
                        if env_type == "local":
                            local_config = {
                                "persistent": config.get("local_persistent", False),
                            }

                        new_env = _create_environment(
                            env_type=env_type,
                            image=image,
                            cwd=cwd,
                            timeout=effective_timeout,
                            ssh_config=ssh_config,
                            container_config=container_config,
                            local_config=local_config,
                            task_id=effective_task_id,
                            host_cwd=config.get("host_cwd"),
                        )
                    except ImportError as e:
                        return json.dumps({
                            "output": "",
                            "exit_code": -1,
                            "error": f"Terminal tool disabled: environment creation failed ({e})",
                            "status": "disabled"
                        }, ensure_ascii=False)

                    with _env_lock:
                        _active_environments[effective_task_id] = new_env
                        _last_activity[effective_task_id] = time.time()
                        env = new_env
                    logger.info("%s environment ready for task %s", env_type, effective_task_id[:8])

        # 硬性阻断：网关生命周期命令
        # （针对 hermes-gateway 的 systemctl/launchctl/hermes restart|stop）
        # 绝不能在网关进程内部本身运行。
        # 重启操作会向网关发送 SIGTERM 信号，
        # 这会在该子进程完成之前将其杀死 ——
        # 导致该服务可能永远无法重新启动。
        # 这与 hermes_cli/gateway.py 中的 `hermes gateway restart` 防护
        # 以及 hermes_cli/cron.py 中的 cron 路径防护保持一致，
        # 但此处是无条件生效的（即使设置 force=True 也无济于事）。
        if os.environ.get("_HERMES_GATEWAY") == "1":
            from hermes_cli.cron import _contains_gateway_lifecycle_command
            if _contains_gateway_lifecycle_command(command):
                return json.dumps({
                    "output": "",
                    "exit_code": 1,
                    "error": (
                        "Blocked: cannot restart or stop the gateway from inside the "
                        "gateway process. The gateway would kill this command before "
                        "it could complete (SIGTERM propagates to child processes). "
                        "Run `hermes gateway restart` from a separate shell outside "
                        "the running gateway."
                    ),
                    "status": "error",
                }, ensure_ascii=False)

        # 执行前的安全检查（tirith 检查 + 危险命令检测）
        # 如果 force=True 则跳过检查（用户已确认他们希望运行该命令）
        approval_note = None
        # 当用户显式批准此次运行（或已通过 force 预先确认）时为 True。
        # 用于在 env.execute 执行前清空中断状态，
        # 从而确保已批准的命令不会被审批等待期间
        # 传入的 SIGINT 信号所杀死（详见 clear_current_thread_interrupt）。
        _approved_run = bool(force)
        if not force:
            approval = _check_all_guards(
                command, env_type,
                has_host_access=_docker_has_host_access(config),
            )
            if not approval["approved"]:
                # Check if this is an approval_required (gateway ask mode)
                if approval.get("status") == "pending_approval":
                    return json.dumps({
                        "output": "",
                        "exit_code": -1,
                        "error": "",
                        "status": "pending_approval",
                        "approval_pending": True,
                        "command": approval.get("command", command),
                        "description": approval.get("description", "command flagged"),
                        "pattern_key": approval.get("pattern_key", ""),
                        "smart_denied": approval.get("smart_denied", False),
                        "allow_permanent": approval.get("allow_permanent", True),
                    }, ensure_ascii=False)
                # Command was blocked
                desc = approval.get("description", "command flagged")
                fallback_msg = (
                    f"Command denied: {desc}. "
                    "Use the approval prompt to allow it, or rephrase the command."
                )
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": approval.get("message", fallback_msg),
                    "status": "blocked"
                }, ensure_ascii=False)
            # Track whether approval was explicitly granted by the user
            if approval.get("user_approved"):
                desc = approval.get("description", "flagged as dangerous")
                approval_note = f"Command required approval ({desc}) and was approved by the user."
                _approved_run = True
            elif approval.get("smart_approved"):
                desc = approval.get("description", "flagged as dangerous")
                approval_note = f"Command was flagged ({desc}) and auto-approved by smart approval."

        # Validate workdir against shell injection
        if workdir:
            workdir_error = _validate_workdir(workdir)
            if workdir_error:
                logger.warning("Blocked dangerous workdir: %s (command: %s)",
                               workdir[:200], _safe_command_preview(command))
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": workdir_error,
                    "status": "blocked"
                }, ensure_ascii=False)

        # Prepare command for execution
        pty_disabled_reason = None
        effective_pty = pty
        if pty and _command_requires_pipe_stdin(command):
            effective_pty = False
            pty_disabled_reason = (
                "PTY disabled for this command because it expects piped stdin/EOF "
                "(for example gh auth login --with-token). For local background "
                "processes, call process(action='close') after writing so it receives "
                "EOF."
            )

        # 为触发此命令的会话声明（共享的“默认”）终端环境（env）。
        # 文件工具通过读取 env.cwd_owner
        # 来判断该环境的实时工作目录（live cwd）
        # 究竟属于“当前”会话的 `cd`，还是属于另一个工作树（worktree）会话 ——
        # 如果没有该标识，共享同一环境的两个打开的工作树会话
        # 就会将彼此的编辑操作路由到错误的检出（checkout）目录。
        # get_current_session_key() 的 contextvar 无法跨越工具工作线程（tool-worker threads），
        # 因此会降级退回到使用原始的 task_id
        # （对于顶层智能体而言，该 ID 即为 session_key）——
        # 这是一个稳定且线程安全锚点。
        from tools.approval import get_current_session_key

        session_key = get_current_session_key(default="") or (task_id or "")
        try:
            env.cwd_owner = session_key
        except Exception:
            pass

        if background:
            # 通过进程注册表（process registry）生成一个被追踪的后台进程。
            # 对于本地后端：使用带有输出缓冲区的 subprocess.Popen。
            # 对于非本地后端：通过 env.execute() 在沙盒内部运行。
            from tools.process_registry import process_registry

            effective_cwd = _resolve_command_cwd(
                workdir=workdir,
                env=env,
                default_cwd=cwd,
            )
            try:
                if env_type == "local":
                    proc_session = process_registry.spawn_local(
                        command=command,
                        cwd=effective_cwd,
                        task_id=effective_task_id,
                        session_key=session_key,
                        env_vars=env.env if hasattr(env, 'env') else None,
                        use_pty=effective_pty,
                    )
                else:
                    proc_session = process_registry.spawn_via_env(
                        env=env,
                        command=command,
                        cwd=effective_cwd,
                        task_id=effective_task_id,
                        session_key=session_key,
                    )

                result_data = {
                    "output": "Background process started",
                    "session_id": proc_session.id,
                    "pid": proc_session.pid,
                    "exit_code": 0,
                    "error": None,
                }
                # 后台进程在脱离状态下生成并立即返回退出码 0；
                # 它绝不会在内联中轮询 is_interrupted()，
                # 因此此处不会发生陈旧比特位（stale-bit）引发的杀进程行为，
                # 且此标记也绝不会与 rc=130 同时出现。
                if approval_note:
                    result_data["approval"] = approval_note
                if pty_disabled_reason:
                    result_data["pty_note"] = pty_disabled_reason

                # 提示（Nudge）：设置了 background=True
                # 但未设置 notify_on_complete=True 或 watch_patterns
                # 会导致进程变成一个“静默进程”。
                # 除非显式调用 process(action="poll"/"wait")，
                # 否则智能体（agent）将“没有任何途径”得知该进程已经执行完毕。
                # 这种机制仅适用于那些永不退出的真正长周期运行进程（如服务器、监听器等）。
                # 而对于每一个有界任务（如测试、构建、CI 轮询器、部署、批处理任务），
                # 智能体几乎毫无疑问是希望能收到通知的，只是忘记了设置该标志。
                # 2026年5月的 PR #31231 事件即是教训：后台 CI 轮询器运行正常并顺利通过退出，
                # 但智能体从未注意到，最终只能由用户手动提示结果。
                # 此处加入轻量级的提示，在服务器场景下仅需付出大约一次读取的成本（误报），
                # 却能在有界任务场景下防止因静默无感知而导致的“失明”（漏报）。
                if background and not notify_on_complete and not watch_patterns:
                    # result_data["hint"] = (
                    #     "设置 background=true 但未设置 notify_on_complete=true "
                    #     "意味着该进程将以【静默模式】运行 —— "
                    #     "当它退出时系统将不会主动告知你。"
                    #     "如果这是一个有界任务（测试套件、构建、"
                    #     "CI 轮询器、部署或任何有明确结束时间的任务），"
                    #     "你几乎肯定需要设置 notify_on_complete=true，"
                    #     "以便系统在进程退出时给你发送 Ping 提醒。"
                    #     "请使用 notify_on_complete=true 重新启动，"
                    #     "或者自行调用 process(action='poll') "
                    #     "/ process(action='wait') 来获取运行结果。"
                    #     "只有对于真正永不退出的长周期进程"
                    #     "（如服务器、监听器、守护进程），才可以忽略此提示。"
                    # )
                    result_data["hint"] = (
                        "background=true without notify_on_complete=true means "
                        "this process runs SILENTLY — you will not be told when "
                        "it exits. If this is a bounded task (test suite, build, "
                        "CI poller, deploy, anything with a defined end), you "
                        "almost certainly wanted notify_on_complete=true so the "
                        "system pings you on exit. Re-launch with "
                        "notify_on_complete=true, or call process(action='poll') "
                        "/ process(action='wait') yourself to learn the outcome. "
                        "Only ignore this hint for genuine long-lived processes "
                        "that never exit (servers, watchers, daemons)."
                    )

                # 提示（Nudge）：通过 `gh pr view` 的 `--json statusCheckRollup`
                # 或将 `gh pr checks` 管道传输给 `jq`
                # 来自制的 CI 监听器（CI watcher），
                # 是 hermes-agent 开发工作中导致 CI 监听器静默失败的“罪魁祸首”（#1 原因）。
                # 2026年5月暴露这一相同失败模式的 PR 包含：
                # #31329、#31448、#31695、#31709、#31745、#32264、#33131。
                # 已发现的失败模式包括：
                #   * `gh pr view --json statusCheckRollup --jq ...` 配合 `from_entries`
                #     在遇到 null 值的 `conclusion` 键时卡死，
                #     循环静默退出且状态为空，永远不会终止。
                #   * `for i in $(seq 1 60); do ... 2>&1` 块缓冲的 stdout
                #     从未刷新并捕获到后台进程中；
                #     SIGTERM 在刷新前切断了缓冲区；
                #     导致 `process(action='log')` 永久返回 total_lines=0。
                #   * 混淆了 conclusion 与 status 字段：
                #     对 `.conclusion` 中的 `PENDING` 进行过滤，
                #     而进行中的检查其 conclusion 字段为空
                #     → 轮询器在 23 个检查中有 18 个仍在 IN_PROGRESS 时，就宣布全部通过（all-green）。
                #   * grep 仅在 TTY 环境下才显示的标语（如 "All checks were successful"），
                #     而当 stdout 被管道传输时，该标语绝不会出现。
                # green-ci-policy skill 中的规范模式可以避开上述每一个坑 ——
                # 依靠退出码驱动循环，
                # 或者基于以制表符分隔的 `awk -F"\t" "$2==\"pending\""`（第 2 列）驱动循环。
                # 此处的检测器被有意设计得较为严格：
                # 它只标记 statusCheckRollup JSON-API 路径以及 `gh pr checks` + jq 的组合，
                # 但“不会”标记规范的第 2 列 awk 轮询器
                # （后者针对制表符使用 awk，而非将其作为通用的 stdout 解析器）。
                # 当我们检测到自制轮询器的特征时，
                # 直接将智能体指向规范的代码片段，
                # 而不是任由其再次交付一个损坏的轮询器。
                # https://gemini.google.com/app/6d9db637d14f3782 说人话版
                if background and command:
                    _gh = ("gh pr view" in command or "gh pr checks" in command)
                    _has_jq = (
                        " jq " in command or "| jq" in command or "$(jq" in command
                    )
                    _bad_shape = (
                        # JSON-API 模式反例。即便不使用 jq，
                        # 通过 `--json statusCheckRollup` 加解析的方式，
                        # 也会让你陷入 conclusion 与 status 字段混淆的困境。
                            "statusCheckRollup" in command
                            # 将 gh pr checks 通过管道传给 jq 也是错误的 ——
                            # `gh pr checks` 并不会输出 JSON 数据，
                            # 因此在此处使用 `| jq` 属于意图混淆。
                            # 规范的第 2 列轮询器使用的是基于制表符的 awk，而非 jq。
                            or (_gh and _has_jq)
                    )
                    if _bad_shape:
                        existing = result_data.get("hint", "")
                        # canonical_hint = (
                        #     "这看起来像是一个通过 `gh pr view --json statusCheckRollup` "
                        #     "和/或 `gh pr checks | jq` 自制的 CI 轮询器。"
                        #     "这种形式在 hermes-agent 的开发工作中屡次引发问题 "
                        #     "（PRs #31329, #31448, #31695, #31709, #31745, #32264, #33131）—— "
                        #     "stdout 缓冲区会导致输出捕获失效，"
                        #     "jq 对 null 键的边缘情况处理会静默退出循环，"
                        #     "混淆 conclusion 与 status 字段会导致带着虚假的全绿（all-green）结论提前退出，"
                        #     "而仅在 TTY 下显示的汇总标语在管道传输时则永远不会出现。"
                        #     "请改用 green-ci-policy skill 中的规范代码片段："
                        #     "对于“遇首错即退出”的行为，使用由退出码驱动的 `gh pr checks $PR >/dev/null` "
                        #     "（rc 0 = 通过，8 = 进行中，其他 = 失败）；"
                        #     "对于分片矩阵（sharded matrices），使用基于制表符的第 2 列 awk 轮询器 "
                        #     "（`awk -F\"\\t\" \"$2==\\\"pending\\\"\"`）。"
                        #     "可以加载 skill_view("
                        #     "name='github/hermes-agent-dev', "
                        #     "file_path='references/green-ci-policy.md') "
                        #     "来获取逐字对应的代码片段。"
                        #     "如果你必须编写包含丰富结构化输出的自定义循环，"
                        #     "请将每次 tick 的结果写入已知文件（`tee -a /tmp/ci.log`），"
                        #     "并依靠 `process(action='log')` 去读取该文件 —— "
                        #     "对于行缓冲的 Shell 循环，切勿依赖后台进程的 stdout 捕获功能。"
                        # )
                        canonical_hint = (
                            "This looks like a homebrewed CI poller built from "
                            "`gh pr view --json statusCheckRollup` and/or "
                            "`gh pr checks | jq`. That shape has burned us "
                            "repeatedly in hermes-agent dev work (PRs #31329, "
                            "#31448, #31695, #31709, #31745, #32264, #33131) — "
                            "stdout buffering kills output capture, jq null-key "
                            "edge cases silently exit the loop, conclusion-vs-"
                            "status field confusion exits early with bogus "
                            "all-green verdicts, TTY-only summary banners "
                            "never appear when piped. Use the canonical "
                            "snippets in the green-ci-policy skill instead: "
                            "the exit-code-driven `gh pr checks $PR >/dev/null` "
                            "(rc 0 = green, 8 = pending, else fail) for "
                            "exit-on-first-fail behavior, or the column-2 "
                            "awk-on-tabs poller "
                            "(`awk -F\"\\t\" \"$2==\\\"pending\\\"\"`) for "
                            "sharded matrices. Load skill_view("
                            "name='github/hermes-agent-dev', "
                            "file_path='references/green-ci-policy.md') for "
                            "the verbatim snippets. If you must roll a custom "
                            "loop with rich structured output, write each tick "
                            "to a known file (`tee -a /tmp/ci.log`) and rely "
                            "on `process(action='log')` to read THAT file — "
                            "do not rely on background-process stdout capture "
                            "for line-buffered shell loops."
                        )
                        result_data["hint"] = (
                            existing + "\n\n" + canonical_hint if existing
                            else canonical_hint
                        )

                # 在会话（session）上填充路由元数据（routing metadata），
                # 以便将监听模式（watch-pattern）和完成通知
                # 正确路由回对应的聊天/线程（chat/thread）中。
                if background and (notify_on_complete or watch_patterns):
                    from gateway.session_context import (
                        async_delivery_supported as _async_ok,
                        get_session_env as _gse,
                    )

                    # 无状态的请求/响应会话（API 服务器 / WebUI 路径）
                    # 无法在轮次（turn）结束后将完成通知路由回智能体（agent）——
                    # 因为不存在持久化通道，且 send() 会是一个空操作（no-op）。
                    # 在此类会话中注册监听程序（watcher）只会静默地失效（issue #10760）。
                    # 因此此处直接拒绝该承诺（promise）：
                    # 丢弃这些标志，并告知智能体去主动轮询（poll）。
                    if not _async_ok():
                        notify_on_complete = False
                        watch_patterns = None
                        result_data["notify_on_complete"] = False
                        result_data["notify_unsupported"] = (
                            "notify_on_complete / watch_patterns are not available on "
                            "this endpoint (stateless HTTP API — no channel to deliver "
                            "an async completion after the turn ends). The process is "
                            "running in the background; retrieve its result with "
                            "process(action='poll') or process(action='wait')."
                        )
                        logger.info(
                            "background proc %s: async delivery unsupported on this "
                            "session; notify_on_complete/watch_patterns disabled",
                            proc_session.id,
                        )
                    else:
                        _gw_platform = _gse("HERMES_SESSION_PLATFORM", "")
                        if _gw_platform:
                            _gw_chat_id = _gse("HERMES_SESSION_CHAT_ID", "")
                            _gw_thread_id = _gse("HERMES_SESSION_THREAD_ID", "")
                            _gw_user_id = _gse("HERMES_SESSION_USER_ID", "")
                            _gw_user_name = _gse("HERMES_SESSION_USER_NAME", "")
                            _gw_message_id = _gse("HERMES_SESSION_MESSAGE_ID", "")
                            proc_session.watcher_platform = _gw_platform
                            proc_session.watcher_chat_id = _gw_chat_id
                            proc_session.watcher_user_id = _gw_user_id
                            proc_session.watcher_user_name = _gw_user_name
                            proc_session.watcher_thread_id = _gw_thread_id
                            proc_session.watcher_message_id = _gw_message_id

                # 互斥机制：如果同时设置了 notify_on_complete和 watch_patterns，则丢弃 watch_patterns。
                # 这两者的组合会产生重复的通知（每次匹配发送一次 + 进程退出时发送一次），
                # 这些通知异步交付，可能在进程结束很久之后依然不断打扰用户。
                # 对于“任务完成时通知我”这一需求，
                # notify_on_complete 是更有用的信号； 而 watch_patterns 应当留给长周期运行进程中
                # 独立的进程中途信号。
                watch_patterns, conflict_note = _resolve_notification_flag_conflict(
                    notify_on_complete=bool(notify_on_complete),
                    watch_patterns=watch_patterns,
                    background=bool(background),
                )
                if conflict_note:
                    logger.warning("background proc %s: %s", proc_session.id, conflict_note)
                    result_data["watch_patterns_ignored"] = conflict_note

                # Mark for agent notification on completion
                if notify_on_complete and background:
                    proc_session.notify_on_complete = True
                    result_data["notify_on_complete"] = True

                    # 在网关（gateway）模式下，自动注册一个快速监听程序（fast watcher），
                    # 以便网关能够检测到完成状态并触发新的智能体轮次（agent turn）。
                    # CLI 模式则直接使用 completion_queue。
                    if proc_session.watcher_platform:
                        proc_session.watcher_interval = 5
                        process_registry.pending_watchers.append({
                            "session_id": proc_session.id,
                            "check_interval": 5,
                            "session_key": session_key,
                            "platform": proc_session.watcher_platform,
                            "chat_id": proc_session.watcher_chat_id,
                            "user_id": proc_session.watcher_user_id,
                            "user_name": proc_session.watcher_user_name,
                            "thread_id": proc_session.watcher_thread_id,
                            "message_id": proc_session.watcher_message_id,
                            "notify_on_complete": True,
                        })

                # Set watch patterns for output monitoring
                if watch_patterns and background:
                    proc_session.watch_patterns = list(watch_patterns)
                    result_data["watch_patterns"] = proc_session.watch_patterns

                return json.dumps(result_data, ensure_ascii=False)
            except Exception as e:
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": f"Failed to start background process: {str(e)}"
                }, ensure_ascii=False)
        else:
            # Run foreground command with retry logic
            max_retries = 3
            retry_count = 0
            result = None
            command_cwd = None

            # 在重试循环开始“前”，针对已批准的命令清除中断标志（仅执行一次）：
            # 清除在等待批准期间落在该线程上的陈旧比特位，
            # 以防其向刚获批运行的命令发送 SIGINT 信号。
            # 切勿在循环“内部”重复清除 ——
            # 在重试间隙的退避休眠（backoff sleep）期间收到的真正中断，
            # 必须保留下来以终止该命令
            # （会被下一次尝试中的 _wait_for_process 轮询循环捕获并返回 130）。
            if _approved_run:
                from tools.interrupt import clear_current_thread_interrupt
                clear_current_thread_interrupt()

            while retry_count <= max_retries:
                try:
                    command_cwd = _resolve_command_cwd(
                        workdir=workdir,
                        env=env,
                        default_cwd=cwd,
                    )
                    execute_kwargs = {
                        "timeout": effective_timeout,
                        "cwd": command_cwd,
                        # Foreground model-facing output: cap retention while
                        # streaming (head/tail window) so a verbose command
                        # can't OOM the gateway before truncation (#64435).
                        # Internal env.execute() consumers (file ops cat
                        # reads, RPC reads) intentionally stay unbounded.
                        "bounded_capture": True,
                    }
                    result = env.execute(command, **execute_kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    if "timeout" in error_str:
                        return json.dumps({
                            "output": "",
                            "exit_code": 124,
                            "error": f"Command timed out after {effective_timeout} seconds"
                        }, ensure_ascii=False)
                    
                    # Retry on transient errors
                    if retry_count < max_retries:
                        retry_count += 1
                        wait_time = 2 ** retry_count
                        logger.warning("Execution error, retrying in %ds (attempt %d/%d) - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                                       wait_time, retry_count, max_retries, _safe_command_preview(command), type(e).__name__, e, effective_task_id, env_type)
                        time.sleep(wait_time)
                        continue
                    
                    logger.error("Execution failed after %d retries - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                                 max_retries, _safe_command_preview(command), type(e).__name__, e, effective_task_id, env_type)
                    return json.dumps({
                        "output": "",
                        "exit_code": -1,
                        "error": f"Command execution failed: {type(e).__name__}: {str(e)}"
                    }, ensure_ascii=False)
                
                # Got a result
                break
            
            # Extract output
            output = result.get("output", "")
            returncode = result.get("returncode", 0)

            # Add helpful message for sudo failures in messaging context
            output = _handle_sudo_failure(output, env_type)

            sudo_auth_failed = _sudo_wrong_password_failure(output)
            sudo_cache_cleared = _invalidate_cached_sudo_on_auth_failure(
                command, output
            )
            if sudo_cache_cleared:
                has_sudo_prompt_callback = _get_sudo_password_callback() is not None
                if has_sudo_prompt_callback or env_var_enabled("HERMES_INTERACTIVE"):
                    output += (
                        "\n\n⚠️ Sudo authentication failed — cached password "
                        "cleared. You will be prompted again on the next sudo "
                        "command."
                    )

            # 前台终端输出规范化切入点（seam）：插件会在默认截断之前
            # 接收到完整的输出字符串，并且只能通过从 transform_terminal_output
            # 返回一个字符串来对其进行替换。
            # 该钩子采用故障开放（fail-open）机制，首个返回的有效字符串生效。
            try:
                from hermes_cli.plugins import invoke_hook
                hook_results = invoke_hook(
                    "transform_terminal_output",
                    command=command,
                    output=output,
                    returncode=returncode,
                    task_id=effective_task_id or "",
                    env_type=env_type,
                )
                for hook_result in hook_results:
                    if isinstance(hook_result, str):
                        output = hook_result
                        break
            except Exception:
                pass
            
            # Truncate output if too long, keeping both head and tail
            from tools.tool_output_limits import get_max_bytes
            MAX_OUTPUT_CHARS = get_max_bytes()
            if len(output) > MAX_OUTPUT_CHARS:
                head_chars = int(MAX_OUTPUT_CHARS * 0.4)  # 40% head (error messages often appear early)
                tail_chars = MAX_OUTPUT_CHARS - head_chars  # 60% tail (most recent/relevant output)
                omitted = len(output) - head_chars - tail_chars
                truncated_notice = (
                    f"\n\n... [OUTPUT TRUNCATED - {omitted} chars omitted "
                    f"out of {len(output)} total] ...\n\n"
                )
                output = output[:head_chars] + truncated_notice + output[-tail_chars:]

            # 剥离 ANSI 转义序列，以便模型永远不会看到终端
            # 格式化内容——防止它将转义字符复制到文件写入中。
            from tools.ansi_strip import strip_ansi
            output = strip_ansi(output)

            # 从命令输出中隐删敏感信息（如密钥等）。对于源码或配置转储
            # （例如 MAX_TOKENS=100、"apiKey": "x" 等测试用例，以及 postgresql:// 格式的 f-string 模板），
            # 系统会跳过环境变量、JSON 及模板匹配阶段，以避免误报（即设置 code_file=True）。
            # 但对于环境变量导出命令（如 env/printenv/set/export/declare），
            # 其输出本身就是 KEY=value 形式的凭据转储，
            # 因此 redact_terminal_output 会执行环境变量匹配阶段（即设置 code_file=False），
            # 从而掩码那些没有供应商前缀的不透明 Token。
            # 至于真实的前缀、身份验证头（Auth Headers）、JWT 以及私钥，
            # 在这两种模式下均会被掩码。详情参见 Issue #43025。
            from agent.redact import redact_terminal_output
            output = redact_terminal_output(output.strip(), command) if output else ""

            # 解释非零退出码（这些退出码并不代表真正的错误，
            # 例如 grep=1 表示“未匹配到内容”，diff=1 表示“文件存在差异”）
            exit_note = _interpret_exit_code(command, returncode)

            # 输出模式失败提示：
            # 将常见的错误类型（如命令未找到、ModuleNotFoundError、gh 字段偏差、合并冲突等）映射为一条简短的修复提示，
            # 从而让模型在下一次调用时就能修复根因，而不必浪费轮次去重新诊断。
            # 详情请参见 tools/terminal_hints.py。
            failure_hint = None
            if returncode != 0 and not exit_note:
                try:
                    from tools.terminal_hints import annotate_failure
                    failure_hint = annotate_failure(command, returncode, output)
                except Exception:
                    failure_hint = None

            result_dict = {
                "output": output,
                "exit_code": returncode,
                "error": None,
            }
            try:
                from agent.verification_evidence import record_terminal_result

                evidence = record_terminal_result(
                    command=command,
                    cwd=command_cwd,
                    session_id=session_id or task_id or effective_task_id or "default",
                    exit_code=returncode,
                    output=output,
                )
                if evidence:
                    result_dict["verification_evidence"] = {
                        "status": evidence.get("status"),
                        "kind": evidence.get("kind"),
                        "scope": evidence.get("scope"),
                        "canonical_command": evidence.get("canonical_command"),
                    }
            except Exception:
                logger.debug("verification evidence recording failed", exc_info=True)
            if approval_note:
                # 仅当存在执行器的标记（marker）时，才将 rc=130 视为中断信号。
                #
                # 命令自身也完全可能合法地主动返回退出码 130
                # （例如执行 `bash -c 'exit 130'`）；
                # 在这种情况下，_wait_for_process 会返回该子进程原生的 returncode，
                # 且不带任何标记，此时绝对不能在审计备注（audit note）中
                # 将其重新归类/标记为用户主动中断。
                if returncode == 130 and "[Command interrupted]" in output:
                    # 经过批准的命令在运行过程中被真实的停止（Stop）操作中断。
                    #
                    # 保留审计追踪记录，但绝不传达“成功”的语义：
                    # 纯粹的 “...approved by the user.” 备注说明
                    # 绝不能与中断退出码同时出现
                    # （满足包含三部分签名的 DONE 条件）。
                    result_dict["approval"] = approval_note.rstrip(".") + ", then interrupted."
                else:
                    result_dict["approval"] = approval_note
            if exit_note:
                result_dict["exit_code_meaning"] = exit_note
            if failure_hint:
                result_dict["hint"] = failure_hint
            if sudo_auth_failed:
                result_dict["sudo_auth_failed"] = True
            if sudo_cache_cleared:
                result_dict["sudo_cache_cleared"] = True

            return json.dumps(result_dict, ensure_ascii=False)

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        logger.error("terminal_tool exception:\n%s", tb_str)
        return json.dumps({
            "output": "",
            "exit_code": -1,
            "error": f"Failed to execute command: {str(e)}",
            "traceback": tb_str,
            "status": "error"
        }, ensure_ascii=False)


def check_terminal_requirements() -> bool:
    """Check if all requirements for the terminal tool are met."""
    try:
        config = _get_env_config()
        env_type = config["env_type"]

        if env_type == "local":
            return True

        elif env_type == "docker":
            from tools.environments.docker import find_docker
            docker = find_docker()
            if not docker:
                logger.error("Docker executable not found in PATH or common install locations")
                return False
            result = subprocess.run([docker, "version"], capture_output=True, timeout=5, stdin=subprocess.DEVNULL)
            return result.returncode == 0

        elif env_type == "singularity":
            executable = shutil.which("apptainer") or shutil.which("singularity")
            if executable:
                result = subprocess.run([executable, "--version"], capture_output=True, timeout=5, stdin=subprocess.DEVNULL)
                return result.returncode == 0
            return False

        elif env_type == "ssh":
            if not config.get("ssh_host") or not config.get("ssh_user"):
                logger.error(
                    "SSH backend selected but TERMINAL_SSH_HOST and TERMINAL_SSH_USER "
                    "are not both set. Configure both or switch TERMINAL_ENV to 'local'."
                )
                return False
            return True

        elif env_type == "modal":
            modal_state = _get_modal_backend_state(config.get("modal_mode"))
            if modal_state["selected_backend"] == "managed":
                return True

            if modal_state["selected_backend"] != "direct":
                if modal_state["managed_mode_blocked"]:
                    logger.error(
                        "Modal backend selected with TERMINAL_MODAL_MODE=managed, but "
                        "Nous Tool Gateway access is not currently available and no direct "
                        "Modal credentials/config were found. %s Choose "
                        "TERMINAL_MODAL_MODE=direct/auto to use direct Modal credentials.",
                        nous_tool_gateway_unavailable_message(
                            "managed Modal execution",
                        ),
                    )
                    return False
                if modal_state["mode"] == "managed":
                    logger.error(
                        "Modal backend selected with TERMINAL_MODAL_MODE=managed, but the managed "
                        "tool gateway is unavailable. %s",
                        nous_tool_gateway_unavailable_message(
                            "managed Modal execution",
                        ),
                    )
                    return False
                elif modal_state["mode"] == "direct":
                    if managed_nous_tools_enabled():
                        logger.error(
                            "Modal backend selected with TERMINAL_MODAL_MODE=direct, but no direct "
                            "Modal credentials/config were found. Configure Modal or choose "
                            "TERMINAL_MODAL_MODE=managed/auto."
                        )
                    else:
                        logger.error(
                            "Modal backend selected with TERMINAL_MODAL_MODE=direct, but no direct "
                            "Modal credentials/config were found. Configure Modal or choose "
                            "TERMINAL_MODAL_MODE=auto."
                        )
                    return False
                else:
                    if managed_nous_tools_enabled():
                        logger.error(
                            "Modal backend selected but no direct Modal credentials/config or managed "
                            "tool gateway was found. Configure Modal, set up the managed gateway, "
                            "or choose a different TERMINAL_ENV."
                        )
                    else:
                        logger.error(
                            "Modal backend selected but no direct Modal credentials/config was found. "
                            "Configure Modal or choose a different TERMINAL_ENV."
                        )
                    return False

            if importlib.util.find_spec("modal") is None:
                logger.error("modal is required for direct modal terminal backend: pip install modal")
                return False

            return True

        elif env_type == "daytona":
            from daytona import Daytona  # noqa: F401 — SDK presence check
            return os.getenv("DAYTONA_API_KEY") is not None

        else:
            logger.error(
                "Unknown TERMINAL_ENV '%s'. Use one of: local, docker, singularity, "
                "modal, daytona, ssh.",
                env_type,
            )
            return False
    except Exception as e:
        logger.error("Terminal requirements check failed: %s", e, exc_info=True)
        return False


if __name__ == "__main__":
    # Simple test when run directly
    print("Terminal Tool Module")
    print("=" * 50)
    
    config = _get_env_config()
    print("\nCurrent Configuration:")
    print(f"  Environment type: {config['env_type']}")
    print(f"  Docker image: {config['docker_image']}")
    print(f"  Modal image: {config['modal_image']}")
    print(f"  Working directory: {config['cwd']}")
    print(f"  Default timeout: {config['timeout']}s")
    print(f"  Lifetime: {config['lifetime_seconds']}s")

    if not check_terminal_requirements():
        print("\n❌ Requirements not met. Please check the messages above.")
        sys.exit(1)

    print("\n✅ All requirements met!")
    print("\nAvailable Tool:")
    print("  - terminal_tool: Execute commands in sandboxed environments")

    print("\nUsage Examples:")
    print("  # Execute a command")
    print("  result = terminal_tool(command='ls -la')")
    print("  ")
    print("  # Run a background task")
    print("  result = terminal_tool(command='python server.py', background=True)")

    print("\nEnvironment Variables:")
    default_img = "nikolaik/python-nodejs:python3.11-nodejs20"
    print(
        "  TERMINAL_ENV: "
        f"{os.getenv('TERMINAL_ENV', 'local')} "
        "(local/docker/singularity/modal/daytona/ssh)"
    )
    print(f"  TERMINAL_DOCKER_IMAGE: {os.getenv('TERMINAL_DOCKER_IMAGE', default_img)}")
    print(f"  TERMINAL_SINGULARITY_IMAGE: {os.getenv('TERMINAL_SINGULARITY_IMAGE', f'docker://{default_img}')}")
    print(f"  TERMINAL_MODAL_IMAGE: {os.getenv('TERMINAL_MODAL_IMAGE', default_img)}")
    print(f"  TERMINAL_DAYTONA_IMAGE: {os.getenv('TERMINAL_DAYTONA_IMAGE', default_img)}")
    print(f"  TERMINAL_CWD: {os.getenv('TERMINAL_CWD', _safe_getcwd())}")
    from hermes_constants import display_hermes_home as _dhh
    print(f"  TERMINAL_SANDBOX_DIR: {os.getenv('TERMINAL_SANDBOX_DIR', f'{_dhh()}/sandboxes')}")
    print(f"  TERMINAL_TIMEOUT: {os.getenv('TERMINAL_TIMEOUT', '60')}")
    print(f"  TERMINAL_LIFETIME_SECONDS: {os.getenv('TERMINAL_LIFETIME_SECONDS', '300')}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

TERMINAL_SCHEMA = {
    "name": "terminal",
    "description": TERMINAL_TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command to execute on the VM"
            },
            "background": {
                "type": "boolean",
                "description": "Run the command in the background. Almost always pair with notify_on_complete=true — without it, the process runs silently and you'll have no way to learn it finished short of calling process(action='poll') yourself (easy to forget, leading to silent blindness on long jobs). Two legitimate patterns: (1) Long-lived processes that never exit (servers, watchers, daemons) — these stay silent because there's no exit to notify on. (2) Long-running bounded tasks (tests, builds, deploys, CI pollers, batch jobs) — these MUST set notify_on_complete=true. For short commands, prefer foreground with a generous timeout instead.",
                "default": False
            },
            "timeout": {
                "type": "integer",
                "description": f"Max seconds to wait (default: 180, foreground max: {FOREGROUND_MAX_TIMEOUT}). Returns INSTANTLY when command finishes — set high for long tasks, you won't wait unnecessarily. Foreground timeout above {FOREGROUND_MAX_TIMEOUT}s is rejected; use background=true for longer commands.",
                "minimum": 1
            },
            "workdir": {
                "type": "string",
                "description": "Working directory for this command (absolute path). Defaults to the session working directory."
            },
            "pty": {
                "type": "boolean",
                "description": "Run in pseudo-terminal (PTY) mode for interactive CLI tools like Codex, Claude Code, or Python REPL. Only works with local and SSH backends. Default: false.",
                "default": False
            },
            "notify_on_complete": {
                "type": "boolean",
                "description": "When true (and background=true), you'll be automatically notified exactly once when the process finishes. **This is the right choice for almost every long-running task** — tests, builds, deployments, multi-item batch jobs, anything that takes over a minute and has a defined end. Use this and keep working on other things; the system notifies you on exit. MUTUALLY EXCLUSIVE with watch_patterns — when both are set, watch_patterns is dropped.",
                "default": False
            },
            "watch_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Strings to watch for in background process output. HARD RATE LIMIT: at most 1 notification per 15 seconds per process — matches arriving inside the cooldown are dropped. After 3 consecutive 15-second windows with dropped matches, watch_patterns is automatically disabled for that process and promoted to notify_on_complete behavior (one notification on exit, no more mid-process spam). USE ONLY for truly rare, one-shot mid-process signals on LONG-LIVED processes that will never exit on their own — e.g. ['Application startup complete'] on a server so you know when to hit its endpoint, or ['migration done'] on a daemon. DO NOT use for: (1) end-of-run markers like 'DONE'/'PASS' — use notify_on_complete instead; (2) error patterns like 'ERROR'/'Traceback' in loops or multi-item batch jobs — they fire on every iteration and you'll hit the strike limit fast; (3) anything you'd ever combine with notify_on_complete. When in doubt, choose notify_on_complete. MUTUALLY EXCLUSIVE with notify_on_complete — set one, not both."
            }
        },
        "required": ["command"]
    }
}


def _handle_terminal(args, **kw):
    return terminal_tool(
        command=args.get("command"),
        background=args.get("background", False),
        timeout=args.get("timeout"),
        task_id=kw.get("task_id"),
        session_id=kw.get("session_id"),
        workdir=args.get("workdir"),
        pty=args.get("pty", False),
        notify_on_complete=args.get("notify_on_complete", False),
        watch_patterns=args.get("watch_patterns"),
    )


registry.register(
    name="terminal",
    toolset="terminal",
    schema=TERMINAL_SCHEMA,
    handler=_handle_terminal,
    check_fn=check_terminal_requirements,
    emoji="💻",
    max_result_size_chars=100_000,
)
