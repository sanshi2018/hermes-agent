"""Base class for all Hermes execution environment backends.

Unified spawn-per-call model: every command spawns a fresh ``bash -c`` process.
A session snapshot (env vars, functions, aliases) is captured once at init and
re-sourced before each command. CWD persists via in-band stdout markers (remote)
or a temp file (local).
"""

import codecs
import json
import logging
import os
import select
import shlex
import subprocess
import threading
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Callable, Protocol

from hermes_constants import get_hermes_home
from hermes_cli._subprocess_compat import windows_hide_flags
from tools.interrupt import is_interrupted

logger = logging.getLogger(__name__)

# Opt-in debug tracing for the interrupt/activity/poll machinery.  Set
# HERMES_DEBUG_INTERRUPT=1 to log loop entry/exit, periodic heartbeats, and
# every is_interrupted() state change from _wait_for_process.  Off by default
# to avoid flooding production gateway logs.
_DEBUG_INTERRUPT = bool(os.getenv("HERMES_DEBUG_INTERRUPT"))

if _DEBUG_INTERRUPT:
    # AIAgent's quiet_mode path (run_agent.py) forces the `tools` logger to
    # ERROR on CLI startup, which would silently swallow every trace we emit.
    # Force this module's own logger back to INFO so the trace is visible in
    # agent.log regardless of quiet-mode.  Scoped to the opt-in case only.
    logger.setLevel(logging.INFO)

# Thread-local activity callback.  The agent sets this before a tool call so
# long-running _wait_for_process loops can report liveness to the gateway.
_activity_callback_local = threading.local()


def set_activity_callback(cb: Callable[[str], None] | None) -> None:
    """Register a callback that _wait_for_process fires periodically."""
    _activity_callback_local.callback = cb


def _get_activity_callback() -> Callable[[str], None] | None:
    return getattr(_activity_callback_local, "callback", None)


def touch_activity_if_due(
    state: dict,
    label: str,
) -> None:
    """Fire the activity callback at most once every ``state['interval']`` seconds.

    *state* must contain ``last_touch`` (monotonic timestamp) and ``start``
    (monotonic timestamp of the operation start).  An optional ``interval``
    key overrides the default 10 s cadence.

    Swallows all exceptions so callers don't need their own try/except.
    """
    now = time.monotonic()
    interval = state.get("interval", 10.0)
    if now - state["last_touch"] < interval:
        return
    state["last_touch"] = now
    try:
        cb = _get_activity_callback()
        if cb:
            elapsed = int(now - state["start"])
            cb(f"{label} ({elapsed}s elapsed)")
    except Exception:
        pass


def get_sandbox_dir() -> Path:
    """Return the host-side root for all sandbox storage (Docker workspaces,
    Singularity overlays/SIF cache, etc.).

    Configurable via TERMINAL_SANDBOX_DIR. Defaults to {HERMES_HOME}/sandboxes/.
    """
    custom = os.getenv("TERMINAL_SANDBOX_DIR")
    if custom:
        p = Path(custom)
    else:
        p = get_hermes_home() / "sandboxes"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Shared constants and utilities
# ---------------------------------------------------------------------------


def _pipe_stdin(proc: subprocess.Popen, data: str) -> None:
    """在守护线程中将 *data* 写入 proc.stdin，以避免管道缓冲区死锁。

    在 Windows 系统上，文本模式下的 stdin（``text=True`` / ``encoding="utf-8"``）
    在数据流经管道时会将 ``\\n`` 转换为 ``\\r\\n``
    —— 这会损坏每一次 write_file 或 patch 调用，
    因为写入磁盘的字节中包含了被额外注入的回车符（carriage return）。
    文件虽然会被正常创建，
    但随后与调用方仅含 ``\\n`` 的字符串进行字节数/内容对比时均会宣告失败。

    变通解决办法（Workaround）：
    直接通过底层字节缓冲区 ``proc.stdin.buffer`` 进行写入，
    并由我们自行编码为 UTF-8。
    这样可以在所有平台上完全绕过 Python 的换行符转换机制。
    在 POSIX 系统上不会有行为变化
    —— 写入的字节序列与文本模式在此处产生的结果完全一致。
    """

    def _write():
        try:
            # 当 Popen 设置了 text=True 时，proc.stdin 是一个 TextIOWrapper 对象。
            # 它的 ``.buffer`` 属性为原始的 BufferedWriter，
            # 能够绕过换行符转换机制。
            #
            # 当 Popen 是以字节模式创建时，
            # proc.stdin 本身就已经是一个没有 ``.buffer`` 属性的 BufferedWriter
            # —— 此时直接回退并使用 .write() 方法即可。
            raw = data.encode("utf-8") if isinstance(data, str) else data
            target = getattr(proc.stdin, "buffer", proc.stdin)
            target.write(raw)
            target.close()
        except (BrokenPipeError, OSError):
            pass

    threading.Thread(target=_write, daemon=True).start()


def _popen_bash(
    cmd: list[str], stdin_data: str | None = None, **kwargs
) -> subprocess.Popen:
    """生成一个具有标准 stdout/stderr/stdin
    设置的子进程。

    如果提供了 *stdin_data*，
    则通过 :func:`_pipe_stdin` 异步写入该数据。
    具有特殊 Popen 需求的后端
    （例如本地的 ``preexec_fn``）可以绕过此过程，
    直接调用 :func:`_pipe_stdin`。
    """
    kwargs.setdefault("creationflags", windows_hide_flags())
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
        text=True,
        **kwargs,
    )
    if stdin_data is not None:
        _pipe_stdin(proc, stdin_data)
    return proc


def _load_json_store(path: Path) -> dict:
    """Load a JSON file as a dict, returning ``{}`` on any error."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_json_store(path: Path, data: dict) -> None:
    """Write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _file_mtime_key(host_path: str) -> tuple[float, int] | None:
    """Return ``(mtime, size)`` for cache comparison, or ``None`` if unreadable."""
    try:
        st = Path(host_path).stat()
        return (st.st_mtime, st.st_size)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# ProcessHandle protocol
# ---------------------------------------------------------------------------


class ProcessHandle(Protocol):
    """Duck type that every backend's _run_bash() must return.

    subprocess.Popen satisfies this natively.  SDK backends (Modal, Daytona)
    return _ThreadedProcessHandle which adapts their blocking calls.
    """

    def poll(self) -> int | None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...

    @property
    def stdout(self) -> IO[str] | None: ...

    @property
    def returncode(self) -> int | None: ...


class _ThreadedProcessHandle:
    """Adapter for SDK backends (Modal, Daytona) that have no real subprocess.

    Wraps a blocking ``exec_fn() -> (output_str, exit_code)`` in a background
    thread and exposes a ProcessHandle-compatible interface.  An optional
    ``cancel_fn`` is invoked on ``kill()`` for backend-specific cancellation
    (e.g. Modal sandbox.terminate, Daytona sandbox.stop).
    """

    def __init__(
        self,
        exec_fn: Callable[[], tuple[str, int]],
        cancel_fn: Callable[[], None] | None = None,
    ):
        self._cancel_fn = cancel_fn
        self._done = threading.Event()
        self._returncode: int | None = None
        self._error: Exception | None = None

        # Pipe for stdout — drain thread in _wait_for_process reads the read end.
        read_fd, write_fd = os.pipe()
        self._stdout = os.fdopen(read_fd, "r", encoding="utf-8", errors="replace")
        self._write_fd = write_fd

        def _worker():
            try:
                output, exit_code = exec_fn()
                self._returncode = exit_code
                # Write output into the pipe so drain thread picks it up.
                try:
                    os.write(self._write_fd, output.encode("utf-8", errors="replace"))
                except OSError:
                    pass
            except Exception as exc:
                self._error = exc
                self._returncode = 1
            finally:
                try:
                    os.close(self._write_fd)
                except OSError:
                    pass
                self._done.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    @property
    def stdout(self):
        return self._stdout

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def poll(self) -> int | None:
        return self._returncode if self._done.is_set() else None

    def kill(self):
        if self._cancel_fn:
            try:
                self._cancel_fn()
            except Exception:
                pass

    def wait(self, timeout: float | None = None) -> int:
        self._done.wait(timeout=timeout)
        return self._returncode


# ---------------------------------------------------------------------------
# CWD marker for remote backends
# ---------------------------------------------------------------------------


def _cwd_marker(session_id: str) -> str:
    return f"__HERMES_CWD_{session_id}__"


# ---------------------------------------------------------------------------
# BaseEnvironment
# ---------------------------------------------------------------------------


class BaseEnvironment(ABC):
    """Common interface and unified execution flow for all Hermes backends.

    Subclasses implement ``_run_bash()`` and ``cleanup()``.  The base class
    provides ``execute()`` with session snapshot sourcing, CWD tracking,
    interrupt handling, and timeout enforcement.
    """

    # Subclasses that embed stdin as a heredoc (Modal, Daytona) set this.
    _stdin_mode: str = "pipe"  # "pipe" or "heredoc"

    # Snapshot creation timeout (override for slow cold-starts).
    _snapshot_timeout: int = 30

    def get_temp_dir(self) -> str:
        """Return the backend temp directory used for session artifacts.

        Most sandboxed backends use ``/tmp`` inside the target environment.
        LocalEnvironment overrides this on platforms like Termux where ``/tmp``
        may be missing and ``TMPDIR`` is the portable writable location.
        """
        return "/tmp"

    def __init__(self, cwd: str, timeout: int, env: dict = None):
        self.cwd = cwd
        self.timeout = timeout
        self.env = env or {}

        self._session_id = uuid.uuid4().hex[:12]
        temp_dir = self.get_temp_dir().rstrip("/") or "/"
        self._snapshot_path = f"{temp_dir}/hermes-snap-{self._session_id}.sh"
        self._cwd_file = f"{temp_dir}/hermes-cwd-{self._session_id}.txt"
        self._cwd_marker = _cwd_marker(self._session_id)
        self._snapshot_ready = False

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ) -> ProcessHandle:
        """Spawn a bash process to run *cmd_string*.

        Returns a ProcessHandle (subprocess.Popen or _ThreadedProcessHandle).
        Must be overridden by every backend.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _run_bash()")

    @abstractmethod
    def cleanup(self):
        """Release backend resources (container, instance, connection)."""
        ...

    # ------------------------------------------------------------------
    # Session snapshot (init_session)
    # ------------------------------------------------------------------

    def init_session(self):
        """将登录 Shell 的环境变量捕获并保存至快照文件中。

        在后端构建完成后调用一次。
        成功后，会将 ``_snapshot_ready`` 设置为 ``True``，
        以便后续命令直接加载（source）该快照，
        而非通过 ``bash -l`` 运行。
        """
        # 完整捕获：环境变量、函数、别名以及 Shell 选项。
        # 在登录 Shell 的配置文件脚本执行完毕后，恢复所配置的工作目录；
        # 因为这些脚本可能会改变工作目录（例如 bashrc 中的 `cd ~`）。
        # 若不进行此恢复，
        # `pwd -P` 捕获的将是配置文件所设定的目录，而非 terminal.cwd。
        # 路由通过 ``_quote_cwd_for_cd``（而非单纯的 ``shlex.quote``）处理，
        # 从而使 Windows 子类的覆盖方法能够将原生的 ``C:\Users\x`` 工作目录
        # 转换为 Git-Bash 的 ``/c/Users/x`` 形式，以便引导程序中的 ``cd`` 可以解析。
        # 如果没有此步骤，在 Windows 上下方快照引导中的 ``cd`` 就会失败，
        # 且 ``pwd -P`` 捕获到的将是登录 Shell 的目录，而非 ``terminal.cwd``。
        _quoted_cwd = self._quote_cwd_for_cd(self.cwd)
        # 通过 ``_quote_shell_path`` 对快照 / 工作目录文件的路径进行转义，
        # 从而使 LocalEnvironment 覆盖方法可以在转义之前，
        # 将 ``C:/...``（以及混合形式如 ``/c/Users\\...``）重写为 ``/c/...`` ——
        # 在引导脚本中使用未处理的盘符路径会导致 MSYS 触发
        # ``Directory \\drivers\\etc does not exist`` 类型的错误。
        # 在 POSIX 系统上，这等同于普通的 ``shlex.quote``。
        _quoted_snap = self._quote_shell_path(self._snapshot_path)
        _quoted_cwd_file = self._quote_shell_path(self._cwd_file)
        # 使用原子文件替换：先在临时文件中组装快照，
        # 然后使用 mv 命令将其覆盖到最终路径上。
        # 这可以防止在另一个终端命令完成并重写环境变量时，
        # 并发的 source() 调用读取到写入了一半的快照（issue #38249）。
        # 当源文件和目标文件位于同一文件系统时，`mv` 在 POSIX 上是原子性的，
        # 因此 source() 要么看到旧的完整快照，要么看到新的完整快照 ——
        # 绝不会读取到部分写入或截断的文件。
        #
        # 对于每个并发写入者，临时文件名必须保持唯一。
        # ``$$`` 表示 bash 的 PID，但在通过 ``&`` 启动的子 Shell 中
        # （即并发终端调用的运行方式），``$$`` 仍保持为*父* Shell 的 PID ——
        # 这会导致两个并发写入者选择相同的临时文件名，
        # 在写入过程中相互覆盖，进而导致 mv 发布一个损坏的文件（竞争问题只是减少，并未消除）。
        # ``$BASHPID`` 则是实际子 Shell 的 PID，对于每个写入者来说是真正唯一的，
        # 从而彻底解决了该竞争条件。
        # 静态路径进行了 Shell 转义（用于处理 Windows/Git-Bash 盘符及空格），
        # 并将 ``$BASHPID`` 留于引号之外，以便其依然能够被正常展开。
        _snap_tmp = self._quote_shell_path(self._snapshot_path + ".tmp.") + "$BASHPID"
        bootstrap = (
            f"umask 077\n"
            f"export -p > {_snap_tmp}\n"
            # 导出函数定义，并通过“名称”而非“行”
            # 来过滤掉私有（前缀为 ``_``）的辅助函数
            # —— 主要为 bash-completion 的内部函数（如 ``_git``、``_make``…）。
            # 简单的 ``declare -f | grep -vE '^_[^_]'`` 是基于行处理的：
            # 它会剔除函数的*头部*行，但却留下了孤立的 ``{ … }`` 函数体，
            # 这会破坏快照，并导致后续每个 source 的命令都失败（例如返回退出码 127）。
            # 先使用 ``declare -F`` 筛选出需要的函数名称，
            # 然后仅导出这些完整的函数定义，
            # 这样可以在不破坏任何函数体的前提下，完美实现过滤目的。
            # 此处非空检查（non-empty guard）非常重要：
            # 不带名称参数的纯 ``declare -f`` 会导出“所有”函数，
            # 因此如果函数名称列表为空（即仅存在私有函数），
            # 反而会导致我们本打算丢弃的那些函数泄漏出来。
            f"__hermes_fns=$(declare -F | awk '{{print $3}}' | grep -vE '^_[^_]') || true\n"
            f"[ -n \"$__hermes_fns\" ] && declare -f $__hermes_fns "
            f">> {_snap_tmp} 2>/dev/null || true\n"
            f"alias -p >> {_snap_tmp}\n"
            f"echo 'shopt -s expand_aliases' >> {_snap_tmp}\n"
            f"echo 'set +e' >> {_snap_tmp}\n"
            f"echo 'set +u' >> {_snap_tmp}\n"
            # Publish atomically only if assembly succeeded; otherwise drop the
            # partial temp rather than leave it to be sourced or orphaned.
            f"mv -f {_snap_tmp} {_quoted_snap} || rm -f {_snap_tmp}\n"
            f"builtin cd -- {_quoted_cwd} 2>/dev/null || true\n"
            f"pwd -P > {_quoted_cwd_file} 2>/dev/null || true\n"
            f"printf '\\n{self._cwd_marker}%s{self._cwd_marker}\\n' \"$(pwd -P)\"\n"
        )
        try:
            proc = self._run_bash(bootstrap, login=True, timeout=self._snapshot_timeout)
            result = self._wait_for_process(proc, timeout=self._snapshot_timeout)
            if int(result.get("returncode") or 0) != 0:
                raise RuntimeError(
                    f"snapshot bootstrap failed with exit code {result.get('returncode')}"
                )
            self._snapshot_ready = True
            self._update_cwd(result)
            logger.info(
                "Session snapshot created (session=%s, cwd=%s)",
                self._session_id,
                self.cwd,
            )
        except Exception as exc:
            logger.warning(
                "init_session failed (session=%s): %s — "
                "falling back to bash -l per command",
                self._session_id,
                exc,
            )
            self._snapshot_ready = False

    # ------------------------------------------------------------------
    # Command wrapping
    # ------------------------------------------------------------------

    @staticmethod
    def _quote_cwd_for_cd(cwd: str) -> str:
        """Quote a ``cd`` target while preserving ``~`` expansion."""
        if cwd == "~":
            return cwd
        if cwd == "~/":
            return "$HOME"
        if cwd.startswith("~/"):
            return f"$HOME/{shlex.quote(cwd[2:])}"
        return shlex.quote(cwd)

    def _quote_shell_path(self, path: str) -> str:
        """对 *path* 进行加引号/转义处理，以便插值嵌入到 Bash 脚本中。

        LocalEnvironment 重写了此方法，
        在加引号之前会将原生或混合格式的 Windows 路径转换为 ``/c/...`` 格式。
        远程后端则保持路径不变（因为它们本身就是 POSIX 格式）。
        """
        return shlex.quote(path)

    def _wrap_command(self, command: str, cwd: str) -> str:
        """构建完整的 Bash 脚本：

        该脚本用于加载 snapshot 环境变量、切换工作目录（cd）、执行命令、重新导出环境变量，并输出当前工作目录（CWD）标记。
        https://gemini.google.com/app/4e5f9fa5708cc2be
        """
        escaped = command.replace("'", "'\\''")

        # 对快照/当前工作目录（cwd）的文件路径进行加引号处理
        # （参见 init_session —— LocalEnvironment 会将 ``C:/...`` 重写为 ``/c/...``，
        # 以避免 MSYS 对路径造成损坏）。
        _quoted_snap = self._quote_shell_path(self._snapshot_path)
        _quoted_cwd_file = self._quote_shell_path(self._cwd_file)
        # 对于环境变量快照（env snapshot）的更新，使用原子文件替换机制（参见 issue #38249）。
        #
        # 先将内容组装到一个针对各写入者（writer）唯一的临时文件中，
        # 随后通过 ``mv`` 命令进行原子替换，
        # 从而确保并发的 ``source()`` 调用绝不会读取到截断或仅写入了一半的文件。
        #
        # 使用 ``$BASHPID``（而非 ``$$``）来获取真实的子 Shell PID
        # —— 该 PID 对于每个通过 ``&`` 后台并发启动的写入者而言都是唯一的
        # —— 这样两个写入者就绝不会共享同一个临时文件名，
        # 也不会在执行 ``mv`` 操作前互相覆盖损坏。
        #
        # 静态路径进行了 Shell 转义加引号（适应 Windows/含空格路径）；
        # ``$BASHPID`` 则保留用于变量展开。
        _snap_tmp = self._quote_shell_path(self._snapshot_path + ".tmp.") + "$BASHPID"

        parts = []

        # 加载（source）环境变量快照（来自先前命令的环境变量）。
        #
        # 将标准输出（stdout）重定向至 /dev/null：
        # 在 macOS（bash 3.2 及部分 Homebrew 构建的 bash）上，
        # 加载包含 ``declare -x`` 的文件可能会将变量声明输出到标准输出，
        # 导致约 60 行环境变量泄露至每个工具的响应结果中（参见 issue #15459）。
        # Linux 上的 bash 在此处是静默无输出的，但添加重定向并无害处。
        if self._snapshot_ready:
            parts.append(
                f"source {_quoted_snap} >/dev/null 2>&1 || true"
            )

        # 保留单独 ``~`` 的展开机制，
        # 但将 ``~/...`` 重写为 ``$HOME`` 路径形式，
        # 从而确保带有空格的后缀部分依然保持为单一的 Shell 词词元（word）。
        quoted_cwd = self._quote_cwd_for_cd(cwd)
        # ``--`` 可防止以连字符（-）开头的文件/目录名被误解析为选项参数。
        parts.append(f"builtin cd -- {quoted_cwd} || exit 126")

        # Run the actual command
        parts.append(f"eval '{escaped}'")
        parts.append("__hermes_ec=$?")
        # 在不改变用户命令 umask 的前提下，
        # 限制 Hermes 元数据文件的访问权限。
        # 快照文件（Snapshot files）中可能包含由环境变量携带来的敏感密钥。
        parts.append("umask 077")

        # 重新将环境变量导出（dump）至快照中（采用原子替换以避免竞争条件）。
        #
        # 仅在导出成功时才链式触发 ``mv`` 替换，
        # 确保失败或部分完成的导出绝不会覆盖良好的快照文件；
        # 并在失败时清除临时文件以防其遗留残留
        # （在 LocalEnvironment.cleanup 中也会进行统一的批量清理）。
        if self._snapshot_ready:
            parts.append(
                f"{{ export -p > {_snap_tmp} && mv -f {_snap_tmp} {_quoted_snap}; }} "
                f"2>/dev/null || rm -f {_snap_tmp} 2>/dev/null || true"
            )

        # Write CWD to file (local reads this) and stdout marker (remote parses this)
        parts.append(f"pwd -P > {_quoted_cwd_file} 2>/dev/null || true")
        # 标记（marker）需独占一行。
        # 前置的 \n 可确保即使命令末尾没有换行符（例如执行 printf 'exact'），
        # 标记依然能从新的一行开始。
        # 我们会在 _extract_cwd_from_output 中将这个注入的换行符剥离去除。
        parts.append(
            f"printf '\\n{self._cwd_marker}%s{self._cwd_marker}\\n' \"$(pwd -P)\""
        )
        parts.append("exit $__hermes_ec")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Stdin heredoc embedding (for SDK backends)
    # ------------------------------------------------------------------

    @staticmethod
    def _embed_stdin_heredoc(command: str, stdin_data: str) -> str:
        """Append stdin_data as a shell heredoc to the command string."""
        delimiter = f"HERMES_STDIN_{uuid.uuid4().hex[:12]}"
        return f"{command} << '{delimiter}'\n{stdin_data}\n{delimiter}"

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------

    def _wait_for_process(self, proc: ProcessHandle, timeout: int = 120) -> dict:
        """
        基于轮询的等待机制，集成了中断检查与标准输出（stdout）的清空。

        该方法在所有后端共用，不可重写。

        在进程运行期间，每隔 10 秒触发一次 ``activity_callback``（若当前实例已设置），
        防止网关的非活动超时机制误杀长时间运行的命令。

        此外，该方法将轮询循环包裹在 ``try/finally`` 块中，
        以确保在因 ``KeyboardInterrupt`` 或 ``SystemExit`` 退出时，
        一定会调用 ``self._kill_process(proc)``。
        如果没有这一机制，本地后端（通过 ``os.setsid`` 将子进程生成到其独立进程组中）
        在 Python 中途关机时会留下一个 ``PPID=1`` 的孤儿进程——
        也就是 Physikal 和我共同踩过的“在 30 分钟后 ``sleep 300`` 依然残留”的 Bug。
        """
        output_chunks: list[str] = []

        # 通过 select() 实现非阻塞的数据清空。
        #
        # 旧有的处理模式——``for line in proc.stdout``——会在 Pipe 未到达 EOF 时
        # 一直阻塞在 ``readline()`` 上。当用户的命令将进程放入后台
        # （例如 ``cmd &``、``setsid cmd & disown`` 等）时，
        # 该后台运行的孙进程会通过 ``fork()`` 继承我们 stdout Pipe 的写入端。
        # 此时即使 ``bash`` 本身已经退出，Pipe 依然保持打开状态，
        # 因为孙进程仍持有该写入端——这会导致数据清空线程永远无法返回，
        # 从而使工具在孙进程的整个生命周期内一直挂起
        # （Issue #8340：用户反馈在使用 ``setsid ... & disown`` 重启 uvicorn 时出现无限挂起）。
        #
        # 修复方案：使用带有短轮询间隔的 select()，
        # 并在 ``bash`` 退出后不久即停止清空，即使 Pipe 尚未到达 EOF。
        # 孙进程在此之后写入的任何输出都会进入一个孤立的 Pipe 中
        # （无害——当我们这一端关闭时，内核会自动回收它）。
        #
        # 解码逻辑：我们通过 ``os.read()`` 以固定大小的块（4096 字节）读取原始字节，
        # 因此单个多字节 UTF-8 字符可能会被拆分到不同的读取块中。
        # 增量解码器（Incremental Decoder）可以在跨块读取时缓存未完成的字节序列；
        # 同时 ``errors="replace"`` 保持了与底层 ``TextIOWrapper``
        # （在 ``Popen`` 上构建时使用了 ``encoding="utf-8", errors="replace"``）一致的行为，
        # 从而对二进制或编码错误的输出使用 U+FFFD 进行替换，而不是损坏整个缓冲区。
        # https://gemini.google.com/app/42952ff33b0cedd9
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        def _drain_iterable(stream):
            # 降级备用路径：``stream`` 并非由真实的操作系统文件描述符所支持
            # （缺少可用的 ``fileno()``）。
            # 该逻辑适用于内存中的 ProcessHandle 适配器，
            # 它们将 stdout 暴露为已收集输出的普通迭代器
            # （即旧有的 ``for line in proc.stdout`` 约定时），
            # 而非实时连接的 Pipe。
            # 对其进行迭代直至到达 EOF。
            # 如果缺少这一处理，清空线程将会抛出未捕获的异常并默默终止，
            # 从而丢失该进程的所有输出数据。
            try:
                for piece in stream:
                    if piece is None:
                        continue
                    if isinstance(piece, bytes):
                        output_chunks.append(decoder.decode(piece))
                    else:
                        output_chunks.append(str(piece))
            except Exception:
                pass
            finally:
                try:
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        output_chunks.append(tail)
                except Exception:
                    pass

        def _drain():
            # 预先解析出真实的操作系统文件描述符（file descriptor）。
            # 真实的子进程与 SDK 中的 ``_ThreadedProcessHandle``（由 os.pipe 支持）
            # 此处都会返回一个整型文件描述符（fd）。
            # 至于 Mock 对象或迭代器类型的 stdout 流，
            # 要么完全缺少 ``fileno()`` 方法，要么会返回非整数值——
            # 在这种情况下，将退回到将该流作为可迭代对象进行清空，
            # 而不是直接导致线程崩溃
            # （引发的 Issue：'list_iterator' 对象没有 'fileno' 属性）。
            stream = proc.stdout
            if stream is None:
                return
            fileno = getattr(stream, "fileno", None)
            try:
                fd = fileno() if callable(fileno) else None
            except Exception:
                fd = None
            if not isinstance(fd, int) or fd < 0:
                _drain_iterable(stream)
                return
            # select.select 在 Windows 系统上无法应用于 Pipe 文件描述符（仅支持 Socket）。
            # 取而代之的是在守护线程（Daemon Thread）中使用阻塞式的 os.read——
            # 这一方案是安全的，因为当 bash 退出时，EOF 会被迅速触发。
            if os.name == "nt":
                try:
                    while True:
                        chunk = os.read(fd, 4096)
                        if not chunk:
                            break
                        output_chunks.append(decoder.decode(chunk))
                except (ValueError, OSError):
                    pass
                finally:
                    try:
                        tail = decoder.decode(b"", final=True)
                        if tail:
                            output_chunks.append(tail)
                    except Exception:
                        pass
                return
            idle_after_exit = 0
            try:
                while True:
                    try:
                        ready, _, _ = select.select([fd], [], [], 0.1)
                    except (ValueError, OSError):
                        break  # fd already closed
                    if ready:
                        try:
                            chunk = os.read(fd, 4096)
                        except (ValueError, OSError):
                            break
                        if not chunk:
                            break  # true EOF — all writers closed
                        output_chunks.append(decoder.decode(chunk))
                        idle_after_exit = 0
                    elif proc.poll() is not None:
                        # bash is gone and the pipe was idle for ~100ms.  Give
                        # it two more cycles to catch any buffered tail, then
                        # stop — otherwise we wait forever on a grandchild pipe.
                        idle_after_exit += 1
                        if idle_after_exit >= 3:
                            break
            finally:
                # Flush any bytes buffered mid-sequence.  With ``errors="replace"``
                # this emits U+FFFD for any final incomplete sequence rather than
                # raising.
                try:
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        output_chunks.append(tail)
                except Exception:
                    pass

        drain_thread = threading.Thread(target=_drain, daemon=True)
        drain_thread.start()
        deadline = time.monotonic() + timeout
        _now = time.monotonic()
        _activity_state = {
            "last_touch": _now,
            "start": _now,
        }

        # --- 调试追踪（通过设置 HERMES_DEBUG_INTERRUPT=1 显式开启）-------------
        # 捕获循环的进入/退出、中断状态变更以及定期心跳信息，
        # 以便我们在无需本地复现的情况下，诊断“Agent 始终未收到中断信号”的相关问题。
        _tid = threading.current_thread().ident
        _pid = getattr(proc, "pid", None)
        _iter_count = 0
        _last_heartbeat = _now
        _last_interrupt_state = False
        _cb_was_none = _get_activity_callback() is None
        if _DEBUG_INTERRUPT:
            logger.info(
                "[interrupt-debug] _wait_for_process ENTER tid=%s pid=%s "
                "timeout=%ss activity_cb=%s initial_interrupt=%s",
                _tid, _pid, timeout,
                "set" if not _cb_was_none else "MISSING",
                is_interrupted(),
            )

        try:
            _poll_sleep = 0.005
            while proc.poll() is None:
                _iter_count += 1
                if is_interrupted():
                    if _DEBUG_INTERRUPT:
                        logger.info(
                            "[interrupt-debug] _wait_for_process INTERRUPT DETECTED "
                            "tid=%s pid=%s iter=%d elapsed=%.1fs — killing process group",
                            _tid, _pid, _iter_count, time.monotonic() - _activity_state["start"],
                        )
                    self._kill_process(proc)
                    drain_thread.join(timeout=2)
                    return {
                        "output": "".join(output_chunks) + "\n[Command interrupted]",
                        "returncode": 130,
                    }
                if time.monotonic() > deadline:
                    if _DEBUG_INTERRUPT:
                        logger.info(
                            "[interrupt-debug] _wait_for_process TIMEOUT "
                            "tid=%s pid=%s iter=%d timeout=%ss",
                            _tid, _pid, _iter_count, timeout,
                        )
                    self._kill_process(proc)
                    drain_thread.join(timeout=2)
                    partial = "".join(output_chunks)
                    timeout_msg = f"\n[Command timed out after {timeout}s]"
                    return {
                        "output": partial + timeout_msg
                        if partial
                        else timeout_msg.lstrip(),
                        "returncode": 124,
                    }
                # Periodic activity touch so the gateway knows we're alive
                touch_activity_if_due(_activity_state, "terminal command running")

                # 每隔约 30 秒发送一次心跳：
                # 证明循环处于活跃状态，
                # 并汇报 activity-callback 的状态
                # （该状态为线程局部变量，可能会被嵌套的工具调用或执行器线程的复用所损坏）。
                if _DEBUG_INTERRUPT and time.monotonic() - _last_heartbeat >= 30.0:
                    _cb_now_none = _get_activity_callback() is None
                    logger.info(
                        "[interrupt-debug] _wait_for_process HEARTBEAT "
                        "tid=%s pid=%s iter=%d elapsed=%.0fs "
                        "interrupt=%s activity_cb=%s%s",
                        _tid, _pid, _iter_count,
                        time.monotonic() - _activity_state["start"],
                        is_interrupted(),
                        "set" if not _cb_now_none else "MISSING",
                        " (LOST during run)" if _cb_now_none and not _cb_was_none else "",
                    )
                    _last_heartbeat = time.monotonic()
                    _cb_was_none = _cb_now_none

                # 自适应轮询：初始间隔设为 5ms，以便快速执行的命令
                # （如 echo、pwd、date、查看短文件内容等）可以在约 6ms 内返回，
                # 而不必被卡住等待下一个 200ms 的轮询 Tick。
                # 随后指数级退避（Back off）至 200ms，
                # 确保长时间运行的命令（构建、测试、sleep 等）
                # 不会在轮询循环中消耗可察觉的 CPU 资源。
                # 对于 `echo` 命令，每次工具调用可节省约 195ms 的时间；
                # 对于耗时 10 秒的构建任务，其稳态下的轮询频率则与旧有行为完全一致。
                time.sleep(_poll_sleep)
                if _poll_sleep < 0.2:
                    _poll_sleep = min(_poll_sleep * 1.5, 0.2)
        except (KeyboardInterrupt, SystemExit):
            # Signal arrived (SIGTERM/SIGHUP/SIGINT) or sys.exit() was called
            # while we were polling.  The local backend spawns subprocesses
            # with os.setsid, which puts them in their own process group — so
            # if we let the interrupt propagate without killing the child,
            # python exits and the child is reparented to init (PPID=1) and
            # keeps running as an orphan.  Killing the process group here
            # guarantees the tool's side effects stop when the agent stops.
            if _DEBUG_INTERRUPT:
                logger.info(
                    "[interrupt-debug] _wait_for_process EXCEPTION_EXIT "
                    "tid=%s pid=%s iter=%d elapsed=%.1fs — killing subprocess group before re-raise",
                    _tid, _pid, _iter_count,
                    time.monotonic() - _activity_state["start"],
                )
            try:
                self._kill_process(proc)
                drain_thread.join(timeout=2)
            except Exception:
                pass  # cleanup is best-effort
            raise

        # 数据清空线程现在会在 bash 退出后迅速退出
        # （经过约 300ms 的空闲检查）。
        # 此处使用短时间的 join 就足够了；
        # 如果耗时很长则意味着存在 Bug，
        # 说明非阻塞循环本身已经停止协作。
        drain_thread.join(timeout=2)

        try:
            proc.stdout.close()
        except Exception:
            pass

        if _DEBUG_INTERRUPT:
            logger.info(
                "[interrupt-debug] _wait_for_process EXIT (natural) "
                "tid=%s pid=%s iter=%d elapsed=%.1fs returncode=%s",
                _tid, _pid, _iter_count,
                time.monotonic() - _activity_state["start"],
                proc.returncode,
            )

        return {"output": "".join(output_chunks), "returncode": proc.returncode}

    def _kill_process(self, proc: ProcessHandle):
        """Terminate a process. Subclasses may override for process-group kill."""
        try:
            proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass

    # ------------------------------------------------------------------
    # CWD extraction
    # ------------------------------------------------------------------

    def _update_cwd(self, result: dict):
        """Extract CWD from command output. Override for local file-based read."""
        self._extract_cwd_from_output(result)

    def _extract_cwd_from_output(self, result: dict):
        """Parse the __HERMES_CWD_{session}__ marker from stdout output.

        Updates self.cwd and strips the marker from result["output"].
        Used by remote backends (Docker, SSH, Modal, Daytona, Singularity).
        """
        output = result.get("output", "")
        marker = self._cwd_marker
        last = output.rfind(marker)
        if last == -1:
            return

        # Find the opening marker before this closing one
        search_start = max(0, last - 4096)  # CWD path won't be >4KB
        first = output.rfind(marker, search_start, last)
        if first == -1 or first == last:
            return

        cwd_path = output[first + len(marker) : last].strip()
        if cwd_path:
            self.cwd = cwd_path

        # Strip the marker line AND the \n we injected before it.
        # The wrapper emits: printf '\n__MARKER__%s__MARKER__\n'
        # So the output looks like: <cmd output>\n__MARKER__path__MARKER__\n
        # We want to remove everything from the injected \n onwards.
        line_start = output.rfind("\n", 0, first)
        if line_start == -1:
            line_start = first
        line_end = output.find("\n", last + len(marker))
        line_end = line_end + 1 if line_end != -1 else len(output)

        result["output"] = output[:line_start] + output[line_end:]

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _before_execute(self) -> None:
        """Hook called before each command execution.

        Remote backends (SSH, Modal, Daytona) override this to trigger
        their FileSyncManager.  Bind-mount backends (Docker, Singularity)
        and Local don't need file sync — the host filesystem is directly
        visible inside the container/process.
        """
        pass

    # ------------------------------------------------------------------
    # Unified execute()
    # ------------------------------------------------------------------

    def execute(
        self,
        command: str,
        cwd: str = "",
        *,
        timeout: int | None = None,
        stdin_data: str | None = None,
        rewrite_compound_background: bool = True,
    ) -> dict:
        """Execute a command, return {"output": str, "returncode": int}."""
        self._before_execute()

        exec_command, sudo_stdin = self._prepare_command(command)
        # 默认防范 `A && B &` 的子 Shell 等待陷阱（subshell-wait trap）。
        # 某些调用方（如 spawn_via_env）已经生成了 Shell 安全的包装程序，
        # 并会传入 rewrite_compound_background=False。
        if rewrite_compound_background:
            from tools.terminal_tool import _rewrite_compound_background
            exec_command = _rewrite_compound_background(exec_command)
        effective_timeout = timeout or self.timeout
        effective_cwd = cwd or self.cwd

        # Merge sudo stdin with caller stdin
        if sudo_stdin is not None and stdin_data is not None:
            effective_stdin = sudo_stdin + stdin_data
        elif sudo_stdin is not None:
            effective_stdin = sudo_stdin
        else:
            effective_stdin = stdin_data

        # Embed stdin as heredoc for backends that need it
        if effective_stdin and self._stdin_mode == "heredoc":
            exec_command = self._embed_stdin_heredoc(exec_command, effective_stdin)
            effective_stdin = None

        wrapped = self._wrap_command(exec_command, effective_cwd)

        # Use login shell if snapshot failed (so user's profile still loads)
        login = not self._snapshot_ready

        proc = self._run_bash(
            wrapped, login=login, timeout=effective_timeout, stdin_data=effective_stdin
        )
        result = self._wait_for_process(proc, timeout=effective_timeout)
        self._update_cwd(result)

        return result

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def stop(self):
        """Alias for cleanup (compat with older callers)."""
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def _prepare_command(self, command: str) -> tuple[str, str | None]:
        """Transform sudo commands if SUDO_PASSWORD is available."""
        from tools.terminal_tool import _transform_sudo_command

        return _transform_sudo_command(command)
