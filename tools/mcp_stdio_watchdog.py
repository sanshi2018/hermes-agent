#!/usr/bin/env python3
"""
用于 stdio MCP 子进程的父进程死亡（parent-death）watchdog 监控程序。

解决的问题（#待定）：stdio MCP 服务器（例如 ``npx -y mcp-remote <url>``）
是作为 Hermes 进程的直接子进程被派生的。Hermes 自身的清理路径
（在最终退出时执行的 ``MCPServerTask.shutdown()`` / ``_kill_orphaned_mcp_children``）
能在*优雅*退出时干净地回收它。但如果派生它的 Hermes 进程发生了硬死亡 ——
例如 ``kill -9``、操作系统级崩溃、或强制退出 TUI/桌面应用 ——
该清理代码就永远不会运行，导致子进程（及其自身的任何子代进程，
例如 mcp-remote 派生的 ``node`` 进程）变成孤儿进程。
macOS 没有直接等效于 Linux ``prctl(PR_SET_PDEATHSIG)`` 的机制
来让内核在父进程死亡时自动杀掉子进程，因此除非在下一次 Hermes 启动时
显式调用并扫描 ``_kill_orphaned_mcp_children()``（这也只有在有代码调用它时才会运行），
否则没有任何机制回收它们。多次非优雅的会话重启可能会堆积 N 个孤儿进程，
它们会竞争抢占同一个上游 SSE 会话，从而在*合法*的新连接上引发类似于
“Invalid request parameters”（无效的请求参数）或
“Received request before initialization was complete”（在初始化完成前收到了请求）等错误。

修复方案：不要直接派生 MCP 服务器命令。改为派生此监控程序，它会：
  1. 将真正的命令作为其子进程执行（通过 ``start_new_session`` 建立独立的进程组，
     这样它就不会异常地继承监控程序的控制终端，同时我们也能干净地对其执行 killpg）；
  2. 透明地透传 stdin/stdout/stderr —— MCP stdio 协议直接通过这些管道通信，
     因此监控程序必须是一个无操作的转发器（no-op relay），而不是中间字节代理（bytes-in-the-middle proxy）；
  3. 运行一个后台线程，使用已经在 ``tui_gateway/slash_worker.py``（``_is_orphaned``）中
     被验证过的孤儿检测算法来轮询原始父进程 PID：对比当前的 ``getppid()`` 与记录的原始 PID，
     并通过 ``psutil`` 进程创建时间来防止 PID 复用问题；
  4. 一旦原始父进程消失，立即终止真实子进程所在的进程组
     （发送 SIGTERM，等待宽限期，若未退出则发送 SIGKILL）并退出。

这是一个有意设计的轻量级、低依赖脚本（仅依赖 ``psutil``，
且该库已通过 ``tui_gateway/slash_worker.py`` 成为硬性依赖），
因此它启动迅速，且本身不会成为资源泄漏点。

用法（参见 ``tools/mcp_tool.py::_run_stdio``）::

    python3 -m tools.mcp_stdio_watchdog \
        --ppid <original_parent_pid> --create-time <original_parent_create_time> \
        -- <real_command> <arg1> <arg2> ...
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a hard dependency elsewhere
    psutil = None

_POLL_INTERVAL_S = 2.0
_TERM_GRACE_S = 3.0


def _is_orphaned(original_ppid: int, parent_create_time: float, getppid=os.getppid) -> bool:
    """Mirrors ``tui_gateway.slash_worker._is_orphaned`` exactly.

    True once the process that spawned us is gone. Never trusts a bare
    ``getppid() == 1`` check (Linux reparents orphans to a subreaper, not
    always PID 1), and guards against PID reuse via the recorded creation
    time of the original parent.
    """
    if getppid() != original_ppid:
        return True
    if psutil is None:
        # No reliable staleness check available; fall back to the ppid
        # comparison alone (still catches the common case).
        return False
    try:
        if not psutil.pid_exists(original_ppid):
            return True
        return psutil.Process(original_ppid).create_time() != parent_create_time
    except psutil.Error:
        return True


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Best-effort SIGTERM-then-SIGKILL of the child's process group.

    This module only ever runs on POSIX (the wrap site in tools/mcp_tool.py
    gates on ``os.name == "posix"``), but guard the POSIX-only primitives
    anyway so an accidental Windows import/execute degrades to a plain
    child kill instead of AttributeError.
    """
    killpg = getattr(os, "killpg", None)
    if killpg is None:  # windows-footgun: ok — non-POSIX fallback
        try:
            proc.terminate()
            proc.wait(timeout=_TERM_GRACE_S)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return
    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
    for sig in (signal.SIGTERM, sigkill):
        try:
            killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            return
        try:
            proc.wait(timeout=_TERM_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            continue


def _watchdog_loop(proc: subprocess.Popen, original_ppid: int, parent_create_time: float) -> None:
    while proc.poll() is None:
        if _is_orphaned(original_ppid, parent_create_time):
            _terminate_process_group(proc)
            return
        time.sleep(_POLL_INTERVAL_S)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parent-death watchdog for a stdio MCP subprocess.",
    )
    parser.add_argument("--ppid", type=int, required=True)
    parser.add_argument("--create-time", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    real_argv = list(args.command)
    if real_argv and real_argv[0] == "--":
        real_argv = real_argv[1:]
    if not real_argv:
        print("mcp_stdio_watchdog: no command given after '--'", file=sys.stderr)
        return 2

    # New process group so we can killpg() the whole tree the real command
    # may spawn (e.g. mcp-remote's own child `node` process), without
    # touching our own group or the (already-gone) original parent's.
    proc = subprocess.Popen(
        real_argv,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,
    )

    # Because the real server lives in its OWN process group (above), the
    # parent's graceful-shutdown killpg of *our* group no longer reaches it.
    # Forward SIGTERM/SIGINT to the child's group so graceful teardown
    # (`_kill_orphaned_mcp_children`, shutdown sweeps) still kills a wedged
    # server that ignores stdin EOF — otherwise the watchdog wrap would
    # invert the bug it fixes.
    def _forward_shutdown(signum, frame):  # noqa: ARG001
        _terminate_process_group(proc)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _forward_shutdown)
    signal.signal(signal.SIGINT, _forward_shutdown)

    watchdog = threading.Thread(
        target=_watchdog_loop,
        args=(proc, args.ppid, args.create_time),
        daemon=True,
    )
    watchdog.start()

    try:
        return proc.wait()
    except KeyboardInterrupt:
        _terminate_process_group(proc)
        return 130


if __name__ == "__main__":
    sys.exit(main())
