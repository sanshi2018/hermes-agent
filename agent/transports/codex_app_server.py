"""Codex 应用服务器（app-server）JSON-RPC 客户端。

采用 codex-rs/app-server/README.md（codex 0.125+）中记录的协议进行通信。
传输层使用基于 stdio 的换行符分隔的 JSON-RPC 2.0 协议：启动 `codex app-server` 子进程，
进行 `initialize` 握手，然后驱动 `thread/start` + `turn/start`，
并持续消耗流式的 `item/*` 通知，直到收到 `turn/completed` 为止。

本模块仅负责传输线缆级（wire-level）的协议通信。更高层级的业务逻辑（如将事件
投射到 Hermes 的显示界面、审批对接、将对话记录投射到 AIAgent.messages 中、
插件迁移等）均位于同级的兄弟模块中。

状态：可选的自主加入型（opt-in）运行时，受控制于 `model.openai_runtime ==
"codex_app_server"` 门槛。当未选择此运行时，Hermes 默认的工具分发逻辑保持不变。
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from tools.environments.local import hermes_subprocess_env

# Default minimum codex version we test against. The PR sets this from the
# `codex --version` parsed at install time; bumping is a one-line change here.
MIN_CODEX_VERSION = (0, 125, 0)


@dataclass
class CodexAppServerError(RuntimeError):
    """Raised on JSON-RPC errors from the app-server."""

    code: int
    message: str
    data: Optional[Any] = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"codex app-server error {self.code}: {self.message}"


@dataclass
class _Pending:
    queue: queue.Queue
    method: str
    sent_at: float = field(default_factory=time.time)


class CodexAppServerClient:
    """基于 stdio 的 `codex app-server` 极简 JSON-RPC 2.0 客户端。

    线程模型：
      - 启动线程（调用者）以同步方式驱动请求/响应对（request/response pairs）。
      - 一个读取器线程（reader thread）解析标准输出（stdout），将回复分发给正确的
        挂起期货（pending future），并将通知（notifications）和服务器发起的请求路由到
        有界队列中，由调用者按自己的节奏进行消耗。
      - 另一个读取器线程捕获标准错误（stderr）用于诊断；Codex 会在其中以
        RUST_LOG 控制的级别输出追踪日志（tracing logs）。

    故意【不】采用异步编程（async）。AIAgent.run_conversation() 是同步的且运行在
    主线程上；仅为了驱动一个 stdio 子进程而分层引入 asyncio 会产生令人意外的
    中断语义。我们使用带有超时机制的阻塞队列，并依赖 `turn/interrupt` 来实现取消操作。
    """

    def __init__(
        self,
        codex_bin: str = "codex",
        codex_home: Optional[str] = None,
        extra_args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        self._codex_bin = codex_bin
        # codex app-server 是一个由模型驱动的 CLI 执行器：它运行一个由模型选择的智能体循环（agentic loop）
        # 来执行 shell 命令，因此它理所当然需要 LLM 服务商的凭证（inherit_credentials=True）
        # 来对模型端点进行身份验证。但是，之前使用的 `os.environ.copy()` 会将每一个 Tier-1 Hermes 密钥
        # —— 包括网关机器人 Token、GitHub 认证信息、Modal/Daytona 基础设施 Token、仪表盘
        # 会话 Token、AUXILIARY_* 辅助 LLM 密钥、GATEWAY_RELAY_* 认证信息 —— 全部同步塞给它，
        # 而这些对于一个代码编译/执行子进程来说完全毫无用处。现改为通过集中式辅助函数进行路由，
        # 以便在确保服务商凭证依然能够传递的同时，始终剥离掉 Tier-1 及动态内部密钥，
        # 从而与 copilot_acp_client 相匹配（修复了 #29157 中同级的子进程启动点漏洞）。
        spawn_env = hermes_subprocess_env(inherit_credentials=True)
        if env:
            spawn_env.update(env)
        if codex_home:
            spawn_env["CODEX_HOME"] = codex_home

        app_server_args = list(extra_args or [])
        # 看板作业人员（Kanban workers）必须能够将他们的交接/状态信息写回到
        # 看板数据库（board DB）中，该数据库位于每项任务的工作空间之外。保持
        # Codex 沙箱处于开启状态，但将看板根目录（Kanban root）添加为唯一的额外可写根目录。
        # 如果不进行此配置，Codex 运行时作业人员即使完成了其实际工作，
        # 也会在 kanban_complete/kanban_block 写入 SQLite 时发生崩溃或受阻。
        if spawn_env.get("HERMES_KANBAN_TASK"):
            kanban_db = spawn_env.get("HERMES_KANBAN_DB")
            kanban_root = (
                os.path.dirname(kanban_db)
                if kanban_db
                else spawn_env.get(
                    "HERMES_KANBAN_ROOT",
                    os.path.join(
                        spawn_env.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
                        "kanban",
                    ),
                )
            )
            app_server_args.extend(
                [
                    "-c",
                    'sandbox_mode="workspace-write"',
                    "-c",
                    f'sandbox_workspace_write.writable_roots=["{kanban_root}"]',
                    "-c",
                    "sandbox_workspace_write.network_access=false",
                ]
            )

        cmd = [codex_bin, "app-server"] + app_server_args
        # Codex emits tracing to stderr; default WARN keeps it quiet for users.
        spawn_env.setdefault("RUST_LOG", "warn")

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=spawn_env,
        )
        self._next_id = 1
        self._pending: dict[int, _Pending] = {}
        self._pending_lock = threading.Lock()
        self._notifications: queue.Queue = queue.Queue()
        self._server_requests: queue.Queue = queue.Queue()
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        self._closed = False
        self._initialized = False

        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()

    # ---------- lifecycle ----------

    def initialize(
        self,
        client_name: str = "hermes",
        client_title: str = "Hermes Agent",
        client_version: str = "0.1",
        capabilities: Optional[dict] = None,
        timeout: float = 10.0,
    ) -> dict:
        """Send `initialize` + `initialized` handshake. Returns the server's
        InitializeResponse (userAgent, codexHome, platformFamily, platformOs)."""
        if self._initialized:
            raise RuntimeError("already initialized")
        params = {
            "clientInfo": {
                "name": client_name,
                "title": client_title,
                "version": client_version,
            },
            "capabilities": capabilities or {},
        }
        result = self.request("initialize", params, timeout=timeout)
        self.notify("initialized")
        self._initialized = True
        return result

    def close(self, timeout: float = 3.0) -> None:
        """Close stdin and wait for the subprocess to exit, escalating to kill."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                self._proc.kill()
                self._proc.wait(timeout=1.0)
            except Exception:
                pass

    def __enter__(self) -> "CodexAppServerClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- send/receive ----------

    def request(
        self,
        method: str,
        params: Optional[dict] = None,
        timeout: float = 30.0,
    ) -> dict:
        """Send a JSON-RPC request and block on the response. Returns `result`,
        raises CodexAppServerError on `error`."""
        rid = self._take_id()
        q: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[rid] = _Pending(queue=q, method=method)
        self._send({"id": rid, "method": method, "params": params or {}})
        try:
            msg = q.get(timeout=timeout)
        except queue.Empty:
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise TimeoutError(
                f"codex app-server method {method!r} timed out after {timeout}s"
            )
        if "error" in msg:
            err = msg["error"]
            raise CodexAppServerError(
                code=err.get("code", -1),
                message=err.get("message", ""),
                data=err.get("data"),
            )
        return msg.get("result", {})

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        self._send({"method": method, "params": params or {}})

    def respond(self, request_id: Any, result: dict) -> None:
        """Reply to a server-initiated request (e.g. approval prompts)."""
        self._send({"id": request_id, "result": result})

    def respond_error(
        self, request_id: Any, code: int, message: str, data: Optional[Any] = None
    ) -> None:
        """Reply to a server-initiated request with an error."""
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        self._send({"id": request_id, "error": err})

    def take_notification(self, timeout: float = 0.0) -> Optional[dict]:
        """Pop the next streaming notification, or return None on timeout.

        timeout=0.0 means non-blocking. Use small positive timeouts inside the
        AIAgent turn loop to interleave reads with interrupt checks."""
        try:
            if timeout <= 0:
                return self._notifications.get_nowait()
            return self._notifications.get(timeout=timeout)
        except queue.Empty:
            return None

    def take_server_request(self, timeout: float = 0.0) -> Optional[dict]:
        """Pop the next server-initiated request (e.g. exec/applyPatch approval)."""
        try:
            if timeout <= 0:
                return self._server_requests.get_nowait()
            return self._server_requests.get(timeout=timeout)
        except queue.Empty:
            return None

    # ---------- diagnostics ----------

    def stderr_tail(self, n: int = 20) -> list[str]:
        """Return last n lines of codex's stderr (for error reports)."""
        with self._stderr_lock:
            return list(self._stderr_lines[-n:])

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    # ---------- internals ----------

    def _take_id(self) -> int:
        # JSON-RPC ids only need to be unique per-connection. A simple
        # monotonically increasing int is the common choice and matches what
        # codex's own clients use.
        rid = self._next_id
        self._next_id += 1
        return rid

    def _send(self, obj: dict) -> None:
        if self._closed:
            raise RuntimeError("codex app-server client is closed")
        if self._proc.stdin is None:
            raise RuntimeError("codex app-server stdin not available")
        try:
            self._proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise RuntimeError(
                f"codex app-server stdin closed unexpectedly: {exc}"
            ) from exc

    def _read_stdout(self) -> None:
        if self._proc.stdout is None:
            return
        try:
            for line in iter(self._proc.stdout.readline, b""):
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # Non-JSON output is unexpected on stdout; tracing belongs
                    # on stderr. Surface it via stderr buffer for diagnostics.
                    with self._stderr_lock:
                        self._stderr_lines.append(
                            f"<non-json on stdout> {line[:200]!r}"
                        )
                    continue
                self._dispatch(msg)
        except Exception as exc:
            with self._stderr_lock:
                self._stderr_lines.append(f"<stdout reader error> {exc}")

    def _dispatch(self, msg: dict) -> None:
        # Reply (has id + result/error, no method)
        if "id" in msg and ("result" in msg or "error" in msg):
            with self._pending_lock:
                pending = self._pending.pop(msg["id"], None)
            if pending is not None:
                try:
                    pending.queue.put_nowait(msg)
                except queue.Full:  # pragma: no cover - defensive
                    pass
            return
        # Server-initiated request (has id + method)
        if "id" in msg and "method" in msg:
            self._server_requests.put(msg)
            return
        # Notification (no id)
        if "method" in msg:
            self._notifications.put(msg)

    def _read_stderr(self) -> None:
        if self._proc.stderr is None:
            return
        try:
            for line in iter(self._proc.stderr.readline, b""):
                if not line:
                    break
                with self._stderr_lock:
                    self._stderr_lines.append(
                        line.decode("utf-8", "replace").rstrip()
                    )
                    # Bound memory: keep last 500 lines.
                    if len(self._stderr_lines) > 500:
                        self._stderr_lines = self._stderr_lines[-500:]
        except Exception:  # pragma: no cover
            pass


def parse_codex_version(output: str) -> Optional[tuple[int, int, int]]:
    """Parse `codex --version` output. Returns (major, minor, patch) or None."""
    # Output format: "codex-cli 0.130.0" possibly followed by metadata.
    import re

    match = re.search(r"(\d+)\.(\d+)\.(\d+)", output or "")
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def check_codex_binary(
    codex_bin: str = "codex", min_version: tuple[int, int, int] = MIN_CODEX_VERSION
) -> tuple[bool, str]:
    """Verify codex CLI is installed and meets minimum version.

    Returns (ok, message). Used by setup wizard and runtime startup."""
    try:
        proc = subprocess.run(
            [codex_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False, (
            f"codex CLI not found at {codex_bin!r}. Install with: "
            f"npm i -g @openai/codex"
        )
    except subprocess.TimeoutExpired:
        return False, "codex --version timed out"
    if proc.returncode != 0:
        return False, f"codex --version exited {proc.returncode}: {proc.stderr.strip()}"
    version = parse_codex_version(proc.stdout)
    if version is None:
        return False, f"could not parse codex version from: {proc.stdout!r}"
    if version < min_version:
        return False, (
            f"codex {'.'.join(map(str, version))} is older than required "
            f"{'.'.join(map(str, min_version))}. Run: npm i -g @openai/codex"
        )
    return True, ".".join(map(str, version))
