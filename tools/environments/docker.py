"""Docker execution environment for sandboxed command execution.

Security hardened (cap-drop ALL, no-new-privileges, PID limits),
configurable resource limits (CPU, memory, disk), and optional filesystem
persistence via bind mounts.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

from tools.environments.base import BaseEnvironment, _popen_bash
from tools.environments.local import (
    _HERMES_PROVIDER_ENV_BLOCKLIST,
    _is_hermes_internal_secret,
)

logger = logging.getLogger(__name__)


# Common Docker Desktop install paths checked when 'docker' is not in PATH.
# macOS Intel: /usr/local/bin, macOS Apple Silicon (Homebrew): /opt/homebrew/bin,
# Docker Desktop app bundle: /Applications/Docker.app/Contents/Resources/bin
_DOCKER_SEARCH_PATHS = [
    "/usr/local/bin/docker",
    "/opt/homebrew/bin/docker",
    "/Applications/Docker.app/Contents/Resources/bin/docker",
]

_docker_executable: Optional[str] = None  # resolved once, cached
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _normalize_forward_env_names(forward_env: list[str] | None) -> list[str]:
    """Return a deduplicated list of valid environment variable names."""
    normalized: list[str] = []
    seen: set[str] = set()

    for item in forward_env or []:
        if not isinstance(item, str):
            logger.warning("Ignoring non-string docker_forward_env entry: %r", item)
            continue

        key = item.strip()
        if not key:
            continue
        if not _ENV_VAR_NAME_RE.match(key):
            logger.warning("Ignoring invalid docker_forward_env entry: %r", item)
            continue
        if key in seen:
            continue

        seen.add(key)
        normalized.append(key)

    return normalized


def _normalize_env_dict(env: dict | None) -> dict[str, str]:
    """Validate and normalize a docker_env dict to {str: str}.

    Filters out entries with invalid variable names or non-string values.
    """
    if not env:
        return {}
    if not isinstance(env, dict):
        logger.warning("docker_env is not a dict: %r", env)
        return {}

    normalized: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not _ENV_VAR_NAME_RE.match(key.strip()):
            logger.warning("Ignoring invalid docker_env key: %r", key)
            continue
        key = key.strip()
        if not isinstance(value, str):
            # Coerce simple scalar types (int, bool, float) to string;
            # reject complex types.
            if isinstance(value, (int, float, bool)):
                value = str(value)
            else:
                logger.warning("Ignoring non-string docker_env value for %r: %r", key, value)
                continue
        normalized[key] = value

    return normalized


def _load_hermes_env_vars() -> dict[str, str]:
    """Load ~/.hermes/.env values without failing Docker command execution."""
    try:
        from hermes_cli.config import load_env

        return load_env() or {}
    except Exception:
        return {}


# Docker label values must match [a-zA-Z0-9_.-] and stay ≤63 chars to round-trip
# safely through `docker ps --filter label=key=value`. Profile and task names
# can technically contain other characters; sanitize defensively.
_LABEL_VALUE_OK_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _sanitize_label_value(value: str) -> str:
    """Coerce *value* into a Docker label-safe form (alnum + ``_.-``, ≤63 chars).

    Empty or all-invalid inputs collapse to ``"unknown"`` so the resulting
    label is always queryable. Used at container-create time; never round-trip
    a sanitized value back into application logic.
    """
    if not isinstance(value, str) or not value:
        return "unknown"
    cleaned = _LABEL_VALUE_OK_RE.sub("_", value)
    cleaned = cleaned[:63] or "unknown"
    return cleaned


def _get_active_profile_name() -> str:
    """Return the active Hermes profile name, or ``"default"`` on any error.

    Resolved at container-create time so a single container is permanently
    tagged with the profile that created it. Profile switches inside the
    same process don't retroactively relabel running containers.
    """
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def reap_orphan_containers(
    *,
    max_age_seconds: int = 600,
    profile_filter: str | None = None,
    docker_exe: str | None = None,
) -> int:
    """移除由先前进程遗留的且带有 hermes 标签的过期容器。

    清理对象为同时满足以下所有条件的容器：

    * ``label=hermes-agent=1``（由此代码库创建）
    * ``status=exited``（绝不会清理正在运行的容器 — 它们可能属于
      某个兄弟 Hermes 进程，且后续复用逻辑会将其拾取；
      强行杀死它们会导致兄弟进程在执行命令途中崩溃）
    *（可选）``label=hermes-profile=<profile_filter>``（默认仅扫描
      调用方的配置文件；配置文件 A 中的 hermes 进程不得
      拆除配置文件 B 的容器）
    * ``State.FinishedAt`` 早于 *max_age_seconds* 之前（从而确保一个刚刚退出、
      即将被替换的兄弟进程的容器不会在其脚下被抽离）

    返回已移除的容器数量。尽力而为原则：任何失败
    （Docker 守护进程无法连接、inspect 缓慢、解析错误）都会在
    调试日志级别记录，且该函数会返回在失败前成功完成的清理数量。
    可重复安全调用；具有幂等性。

    Issue #20561 — 这是针对绕过 ``atexit`` 清理钩子的
    SIGKILL / OOM / 终端崩溃等退出的安全防护网。如果没有它，
    即使在前一个提交中修补了清理机制，一个被强行杀死的 Hermes
    进程仍会永久遗留其容器，因为后续将没有被调度的 Hermes
    进程去复用那个完全相同的 (task, profile) 配对。
    """
    docker = docker_exe or find_docker() or "docker"
    filters = ["--filter", "label=hermes-agent=1", "--filter", "status=exited"]
    if profile_filter:
        filters.extend(["--filter", f"label=hermes-profile={_sanitize_label_value(profile_filter)}"])

    try:
        listing = subprocess.run(
            [docker, "ps", "-a", *filters, "--format", "{{.ID}}"],
            capture_output=True, text=True, timeout=15, check=False,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("orphan reaper docker ps failed: %s", e)
        return 0
    if listing.returncode != 0:
        logger.debug(
            "orphan reaper docker ps returned %d: %s",
            listing.returncode, listing.stderr.strip(),
        )
        return 0

    candidate_ids = [ln.strip() for ln in listing.stdout.splitlines() if ln.strip()]
    if not candidate_ids:
        return 0

    # 检查每个候选容器以获取 FinishedAt 字段；仅回收那些已退出
    # 足够长时间的容器。按容器逐个进行处理（而非批量 inspect）
    # 可以将失败的影响范围限制在单次一个容器之内。
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    removed = 0
    for cid in candidate_ids:
        finished_at = _container_finished_at(docker, cid)
        if finished_at is None:
            # Couldn't determine age — be conservative and leave it alone.
            continue
        age = (now - finished_at).total_seconds()
        if age < max_age_seconds:
            continue
        try:
            result = subprocess.run(
                [docker, "rm", "-f", cid],
                capture_output=True, text=True, timeout=30,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                removed += 1
                logger.info(
                    "Reaped orphan container %s (exited %d seconds ago)",
                    cid[:12], int(age),
                )
            else:
                logger.debug(
                    "docker rm -f %s failed: %s",
                    cid[:12], result.stderr.strip(),
                )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("orphan reaper docker rm %s failed: %s", cid[:12], e)
    return removed


def _container_finished_at(docker_exe: str, container_id: str):
    """Parse ``docker inspect`` FinishedAt for *container_id*.

    Returns a timezone-aware datetime, or ``None`` if the field is missing,
    unparseable, or the zero-value ``0001-01-01T00:00:00Z`` Docker emits
    for never-finished containers. ``None`` means "don't reap" — the caller
    leaves the container alone.
    """
    try:
        result = subprocess.run(
            [docker_exe, "inspect", "--format", "{{.State.FinishedAt}}", container_id],
            capture_output=True, text=True, timeout=10, check=False,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("orphan reaper docker inspect %s failed: %s", container_id[:12], e)
        return None
    if result.returncode != 0:
        return None
    raw = result.stdout.strip()
    if not raw or raw.startswith("0001-01-01"):
        return None
    # Docker emits RFC3339 with nanoseconds (e.g. "2026-05-28T13:45:00.123456789Z").
    # Python's fromisoformat handles microseconds but not nanoseconds; trim.
    import re as _re
    raw = _re.sub(r"(\.\d{6})\d+", r"\1", raw)
    raw = raw.replace("Z", "+00:00")
    try:
        import datetime
        return datetime.datetime.fromisoformat(raw)
    except ValueError as e:
        logger.debug("could not parse FinishedAt %r for %s: %s", raw, container_id[:12], e)
        return None


def find_docker() -> Optional[str]:
    """查找 docker（或 podman）CLI 的二进制文件。

    解析顺序：
    1. ``HERMES_DOCKER_BINARY`` 环境变量 —— 显式覆盖（例如 ``/usr/bin/podman``）
    2. 通过 ``shutil.which`` 在 PATH 环境变量中查找 ``docker``
    3. 通过 ``shutil.which`` 在 PATH 环境变量中查找 ``podman``
    4. macOS 上 Docker Desktop 的已知安装路径

    返回绝对路径，
    如果无法找到任何一个运行时（runtime）环境，
    则返回 ``None``。
    """
    global _docker_executable
    if _docker_executable is not None:
        return _docker_executable

    # 1. Explicit override via env var (e.g. for Podman on immutable distros)
    override = os.getenv("HERMES_DOCKER_BINARY")
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        _docker_executable = override
        logger.info("Using HERMES_DOCKER_BINARY override: %s", override)
        return override

    # 2. docker on PATH
    found = shutil.which("docker")
    if found:
        _docker_executable = found
        return found

    # 3. podman on PATH (drop-in compatible for our use case)
    found = shutil.which("podman")
    if found:
        _docker_executable = found
        logger.info("Using podman as container runtime: %s", found)
        return found

    # 4. Well-known macOS Docker Desktop locations
    for path in _DOCKER_SEARCH_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            _docker_executable = path
            logger.info("Found docker at non-PATH location: %s", path)
            return path

    return None


# 应用于每个容器的安全标志。
# 容器本身就是安全边界（与主机隔离）。
# 我们会丢弃所有的系统权能（capabilities），
# 然后仅重新添加最低限度的必需权能：
#   DAC_OVERRIDE - root 用户可以写入由主机用户所有的、绑定挂载（bind-mounted）的目录
#   CHOWN/FOWNER - 包管理器（如 pip、npm、apt）需要设置文件所有权
#   SETUID/SETGID - 镜像的初始化进程（init）会从 root 降权至 'hermes'
#       用户（通过打包镜像中的 `s6-setuidgid`，
#       或者用户镜像所使用的任何降权辅助工具），
#       这一操作需要这些权能。
#       结合 `no-new-privileges`，
#       降权后的进程依然无法重新提权回 root，
#       从而保持了系统的安全态势。
#       当容器通过 --user 参数以非 root 用户身份启动时，
#       由于在该模式下不需要进行降权，
#       因此会完全省略这些设置。
# 阻止权限提升。
# /tmp 目录受到大小限制，并禁用了 suid（nosuid），
# 但允许执行操作（exec，这是 pip/npm 构建所必需的）。
#
# 注意：``--pids-limit`` *并不仅* 存在于此列表中 ——
# 它位于 ``resource_args`` 里，
# 并且受 ``_cgroup_limits_available(image)`` 的控制，
# 因为它要求委派 ``pids`` cgroup 控制器，
# 而在诸如无特权的 LXC 等主机环境下通常无法满足此条件。
# 基于相同的原因，
# ``--cpus`` 和 ``--memory`` 也同样受到该控制。
_BASE_SECURITY_ARGS = [
    "--cap-drop", "ALL",
    "--cap-add", "DAC_OVERRIDE",
    "--cap-add", "CHOWN",
    "--cap-add", "FOWNER",
    "--security-opt", "no-new-privileges",
    "--tmpfs", "/tmp:rw,nosuid,size=512m",
    "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=256m",
]

# Default per-container PID limit. Applied as ``--pids-limit`` only when the
# cgroup ``pids`` controller is available (see ``_cgroup_limits_available``).
_DEFAULT_PIDS_LIMIT = "256"

# /run is split out from _BASE_SECURITY_ARGS because s6-overlay images need it
# mounted ``exec``: s6 stage0 later runs ``exec /run/s6/basedir/bin/init``, which
# fails with "Permission denied" (exit 126) on a ``noexec`` mount. For all other
# images we keep the hardened ``noexec`` default.
_RUN_TMPFS_NOEXEC = "--tmpfs", "/run:rw,noexec,nosuid,size=64m"
_RUN_TMPFS_EXEC = "--tmpfs", "/run:rw,exec,nosuid,size=64m"

# Extra caps needed when the container starts as root and an init/entrypoint
# must drop privileges (via `s6-setuidgid`, `gosu`, `su`, or similar).
# Skipped when --user is passed because the container already starts
# unprivileged and never needs to switch.
_PRIVDROP_CAP_ARGS = [
    "--cap-add", "SETUID",
    "--cap-add", "SETGID",
]


def _build_security_args(run_as_host_user: bool, run_exec: bool = False) -> list[str]:
    """返回针对特定特权模式定制的
    安全/权能（cap）/tmpfs 参数。

    ``run_exec`` 会使用 ``exec`` 选项挂载 ``/run``，
    而非使用经过加固的默认选项 ``noexec``。
    对于其 ``/init`` 入口点会在启动期间
    执行 ``/run/s6/basedir/bin/init`` 的 s6-overlay 镜像而言，
    这一设置是必需的；
    详见 ``_image_uses_init_entrypoint``。
    """
    run_tmpfs = list(_RUN_TMPFS_EXEC if run_exec else _RUN_TMPFS_NOEXEC)
    args = list(_BASE_SECURITY_ARGS) + run_tmpfs
    if run_as_host_user:
        return args
    return args + list(_PRIVDROP_CAP_ARGS)


def _image_uses_init_entrypoint(docker_exe: str, image: str) -> bool:
    """如果 ``image`` 的入口点是 s6-overlay 的 ``/init``，则返回 True。

    这类镜像
    （例如任何基于 ``s6-overlay`` 构建的镜像，
    包括 ``hermes-agent:latest``）
    已经提供了其自身的 PID-1 初始化进程，
    并在 stage0 启动阶段
    执行 ``/run/s6/basedir/bin/init``。
    它们与 Docker 的 ``--init`` 参数不兼容
    （会导致两个 PID-1 初始化进程相互冲突），
    同时也与带有 ``noexec`` 选项的 ``/run`` 挂载不兼容。
    这种检测是尽力而为的：
    如果检查过程中出现任何失败，
    我们将返回 False，
    并保留经过安全加固的默认设置。
    """
    try:
        result = subprocess.run(
            [docker_exe, "image", "inspect", image,
             "--format", "{{json .Config.Entrypoint}}"],
            capture_output=True,
            text=True,
            timeout=15,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("Docker: could not inspect entrypoint for %s: %s", image, e)
        return False
    if result.returncode != 0:
        # Image may not be pulled yet; the run will pull it. Defaults are safe
        # for non-s6 images, so don't block on this.
        logger.debug(
            "Docker: image inspect for %s returned %d (stderr=%s)",
            image, result.returncode, result.stderr.strip(),
        )
        return False
    raw = (result.stdout or "").strip()
    if not raw or raw == "null":
        return False
    try:
        entrypoint = json.loads(raw)
    except (ValueError, TypeError):
        return False
    if isinstance(entrypoint, str):
        entrypoint = [entrypoint]
    if not isinstance(entrypoint, list) or not entrypoint:
        return False
    first = str(entrypoint[0]).strip()
    return first in ("/init", "/package/admin/s6-overlay/command/init")


def _resolve_host_user_spec() -> Optional[str]:
    """Return ``<uid>:<gid>`` for the current host user, or ``None`` on platforms
    where this is not meaningful (e.g. Windows without posix ids).

    We intentionally read ``os.getuid()``/``os.getgid()`` directly rather than
    going through ``getpass``/``pwd`` so this stays cheap and never raises on
    nameless UIDs (nss lookups can fail inside sandboxed launchers).
    """
    get_uid = getattr(os, "getuid", None)
    get_gid = getattr(os, "getgid", None)
    if get_uid is None or get_gid is None:
        return None
    try:
        return f"{get_uid()}:{get_gid()}"
    except Exception:  # pragma: no cover - defensive
        return None


_storage_opt_ok: Optional[bool] = None  # cached result across instances
_cgroup_limits_ok: Optional[bool] = None  # cached result across instances


def _cgroup_limits_available(image: str) -> bool:
    """探测当前环境中 cgroup 资源限制是否有效。

    通过从 *image* 启动一个一次性容器来集中测试
    ``--cpus``、``--memory`` 和 ``--pids-limit`` 参数
    （这是我们即将投入实际使用的同一个沙盒镜像，
    因此无需额外拉取，也不依赖于公共镜像仓库）。
    该容器会运行 ``sleep 0`` 命令 ——
    由于沙盒本身使用 ``sleep 2h`` 作为其常驻入口点，
    因此可以确保 sleep 命令一定存在。

    在未将相应的 cgroup 控制器委派给当前进程的主机上
    （常见于无特权的 LXC 环境以及部分 rootless 设置中），
    这些参数标志会导致每次容器启动都失败，
    并抛出 ``OCI runtime error`` / 退出码 126。
    该探测在每个进程中仅运行一次，
    并且探测结果会被缓存下来
    （该结果作用于主机全局范围，而非针对特定镜像）。
    """
    global _cgroup_limits_ok
    if _cgroup_limits_ok is not None:
        return _cgroup_limits_ok

    docker_exe = find_docker()
    if not docker_exe or not image:
        _cgroup_limits_ok = False
        return False

    try:
        result = subprocess.run(
            [docker_exe, "run", "--rm",
             "--cpus", "0.5", "--memory", "64m", "--pids-limit", "32",
             image, "sleep", "0"],
            capture_output=True,
            text=True,
            timeout=60,
            stdin=subprocess.DEVNULL,
        )
        _cgroup_limits_ok = result.returncode == 0
        if not _cgroup_limits_ok:
            logger.warning(
                "Cgroup resource limits (--cpus/--memory/--pids-limit) not "
                "available in this environment. Containers will run without "
                "CPU, memory or PID limits. To enable, delegate the cpu, "
                "memory and pids cgroup controllers to this container. "
                "Probe stderr: %s",
                (result.stderr or "").strip()[:500],
            )
    except Exception as e:
        _cgroup_limits_ok = False
        logger.warning("Cgroup limit probe failed; disabling resource limits: %s", e)

    return _cgroup_limits_ok


def _ensure_docker_available() -> None:
    """在使用前，尽力检查 Docker CLI 是否可用。

    复用 ``find_docker()``，
    以便此预先检查能够与 Docker 后端的其余部分保持一致，
    包括已知的、未在 PATH 环境变量中的 Docker Desktop 路径。
    """
    docker_exe = find_docker()
    if not docker_exe:
        logger.error(
            "Docker backend selected but no docker executable was found in PATH "
            "or known install locations. Install Docker Desktop and ensure the "
            "CLI is available."
        )
        raise RuntimeError(
            "Docker executable not found in PATH or known install locations. "
            "Install Docker and ensure the 'docker' command is available."
        )

    try:
        result = subprocess.run(
            [docker_exe, "version"],
            capture_output=True,
            text=True,
            timeout=5,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        logger.error(
            "Docker backend selected but the resolved docker executable '%s' could "
            "not be executed.",
            docker_exe,
            exc_info=True,
        )
        raise RuntimeError(
            "Docker executable could not be executed. Check your Docker installation."
        )
    except subprocess.TimeoutExpired:
        logger.error(
            "Docker backend selected but '%s version' timed out. "
            "The Docker daemon may not be running.",
            docker_exe,
            exc_info=True,
        )
        raise RuntimeError(
            "Docker daemon is not responding. Ensure Docker is running and try again."
        )
    except Exception:
        logger.error(
            "Unexpected error while checking Docker availability.",
            exc_info=True,
        )
        raise
    else:
        if result.returncode != 0:
            logger.error(
                "Docker backend selected but '%s version' failed "
                "(exit code %d, stderr=%s)",
                docker_exe,
                result.returncode,
                result.stderr.strip(),
            )
            raise RuntimeError(
                "Docker command is available but 'docker version' failed. "
                "Check your Docker installation."
            )


class DockerEnvironment(BaseEnvironment):
    """具备资源限制与持久化特性的加固型 Docker 容器执行环境。

    安全性：剥离所有 Capability 特权，禁止特权提升，限制 PID 数量，
    为临时目录配置限制容量的 tmpfs。容器本身即为安全边界 —
    其内部文件系统可写，以便 Agent 能够根据需要安装软件包
    （如 pip、npm、apt）。通过 tmpfs 或绑定挂载提供可写的工作空间。

    持久化：启用时，绑定挂载将在容器重启间
    保留 /workspace 与 /root 目录的内容。
    """

    def __init__(
        self,
        image: str,
        cwd: str = "/root",
        timeout: int = 60,
        cpu: float = 0,
        memory: int = 0,
        disk: int = 0,
        persistent_filesystem: bool = False,
        task_id: str = "default",
        volumes: list = None,
        forward_env: list[str] | None = None,
        env: dict | None = None,
        network: bool = True,
        host_cwd: str = None,
        auto_mount_cwd: bool = False,
        run_as_host_user: bool = False,
        extra_args: list = None,
        persist_across_processes: bool = True,
    ):
        if cwd == "~":
            cwd = "/root"
        super().__init__(cwd=cwd, timeout=timeout)
        self._persistent = persistent_filesystem
        self._persist_across_processes = persist_across_processes
        self._task_id = task_id
        self._forward_env = _normalize_forward_env_names(forward_env)
        self._env = _normalize_env_dict(env)
        self._container_id: Optional[str] = None
        self._labels: dict[str, str] = {}
        self._image: str = ""
        self._container_name: str = ""
        self._image_uses_s6_init: bool = False
        self._all_run_args: list[str] = []
        logger.info(f"DockerEnvironment volumes: {volumes}")
        # Ensure volumes is a list (config.yaml could be malformed)
        if volumes is not None and not isinstance(volumes, list):
            logger.warning(f"docker_volumes config is not a list: {volumes!r}")
            volumes = []

        # Fail fast if Docker is not available.
        _ensure_docker_available()

        # 构建资源限制参数
        # （受 cgroup 可用性探测的控制，
        # 因此它们在没有控制器委派的主机上能够优雅地降级，
        # 例如无特权的 LXC 环境）。
        # 该探测每个进程仅运行一次，
        # 并在主机全局范围内进行缓存。
        resource_args = []
        if cpu > 0 and _cgroup_limits_available(image):
            resource_args.extend(["--cpus", str(cpu)])
        if memory > 0 and _cgroup_limits_available(image):
            resource_args.extend(["--memory", f"{memory}m"])
        if _cgroup_limits_available(image):
            resource_args.extend(["--pids-limit", _DEFAULT_PIDS_LIMIT])
        if disk > 0 and sys.platform != "darwin":
            if self._storage_opt_supported():
                resource_args.extend(["--storage-opt", f"size={disk}m"])
            else:
                logger.warning(
                    "Docker storage driver does not support per-container disk limits "
                    "(requires overlay2 on XFS with pquota). Container will run without disk quota."
                )
        if not network:
            resource_args.append("--network=none")

        # 通过绑定挂载（bind mounts）一个可配置的主机目录
        # 来实现持久化工作区
        # （由 TERMINAL_SANDBOX_DIR 指定，默认为 ~/.hermes/sandboxes/）。
        # 非持久化模式则使用 tmpfs
        # （临时且快速，在清理后即消失）。
        from tools.environments.base import get_sandbox_dir

        # User-configured volume mounts (from config.yaml docker_volumes)
        volume_args = []
        workspace_explicitly_mounted = False
        for vol in (volumes or []):
            if not isinstance(vol, str):
                logger.warning(f"Docker volume entry is not a string: {vol!r}")
                continue
            vol = vol.strip()
            if not vol:
                continue
            if ":" in vol:
                volume_args.extend(["-v", vol])
                if ":/workspace" in vol:
                    workspace_explicitly_mounted = True
            else:
                logger.warning(f"Docker volume '{vol}' missing colon, skipping")

        host_cwd_abs = os.path.abspath(os.path.expanduser(host_cwd)) if host_cwd else ""
        bind_host_cwd = (
            auto_mount_cwd
            and bool(host_cwd_abs)
            and os.path.isdir(host_cwd_abs)
            and not workspace_explicitly_mounted
        )
        if auto_mount_cwd and host_cwd and not os.path.isdir(host_cwd_abs):
            logger.debug(f"Skipping docker cwd mount: host_cwd is not a valid directory: {host_cwd}")

        self._workspace_dir: Optional[str] = None
        self._home_dir: Optional[str] = None
        writable_args = []
        if self._persistent:
            sandbox = get_sandbox_dir() / "docker" / task_id
            self._home_dir = str(sandbox / "home")
            os.makedirs(self._home_dir, exist_ok=True)
            writable_args.extend([
                "-v", f"{self._home_dir}:/root",
            ])
            if not bind_host_cwd and not workspace_explicitly_mounted:
                self._workspace_dir = str(sandbox / "workspace")
                os.makedirs(self._workspace_dir, exist_ok=True)
                writable_args.extend([
                    "-v", f"{self._workspace_dir}:/workspace",
                ])
        else:
            if not bind_host_cwd and not workspace_explicitly_mounted:
                writable_args.extend([
                    "--tmpfs", "/workspace:rw,exec,size=10g",
                ])
            writable_args.extend([
                "--tmpfs", "/home:rw,exec,size=1g",
                "--tmpfs", "/root:rw,exec,size=1g",
            ])

        if bind_host_cwd:
            logger.info(f"Mounting configured host cwd to /workspace: {host_cwd_abs}")
            volume_args = ["-v", f"{host_cwd_abs}:/workspace", *volume_args]
        elif workspace_explicitly_mounted:
            logger.debug("Skipping docker cwd mount: /workspace already mounted by user config")

        # 挂载由技能声明的凭据文件
        # （例如 OAuth 令牌等）。
        # 设置为只读模式，
        # 以便容器能够进行身份验证，
        # 但无法修改主机的凭据。
        try:
            from tools.credential_files import (
                get_credential_file_mounts,
                get_skills_directory_mount,
                get_cache_directory_mounts,
            )

            for mount_entry in get_credential_file_mounts():
                src = Path(mount_entry["host_path"])
                if src.is_dir():
                    # Docker-in-Docker: Docker auto-created the source path as
                    # a directory when it didn't exist on the host.  Mounting a
                    # directory over a file destination causes exit 125.
                    logger.warning(
                        "Docker: skipping credential mount — source is a directory "
                        "(likely Docker-in-Docker auto-creation): %s",
                        src,
                    )
                    continue
                if not src.is_file():
                    logger.warning(
                        "Docker: skipping credential mount — source not found: %s", src,
                    )
                    continue
                volume_args.extend([
                    "-v",
                    f"{mount_entry['host_path']}:{mount_entry['container_path']}:ro",
                ])
                logger.info(
                    "Docker: mounting credential %s -> %s",
                    mount_entry["host_path"],
                    mount_entry["container_path"],
                )

            # Mount skill directories (local + external) so skill
            # scripts/templates are available inside the container.
            for skills_mount in get_skills_directory_mount():
                src = Path(skills_mount["host_path"])
                if not src.is_dir():
                    logger.warning(
                        "Docker: skipping skills mount — source is not a directory: %s",
                        src,
                    )
                    continue
                volume_args.extend([
                    "-v",
                    f"{skills_mount['host_path']}:{skills_mount['container_path']}:ro",
                ])
                logger.info(
                    "Docker: mounting skills dir %s -> %s",
                    skills_mount["host_path"],
                    skills_mount["container_path"],
                )

            # 挂载主机端的缓存目录
            # （文档、图像、音频、屏幕截图），
            # 以便智能体能够从容器内部
            # 访问已上传的文件以及其他缓存的媒体资源。
            # 设置为只读模式 ——
            # 容器仅能读取这些内容，
            # 而主机网关负责管理写入操作。
            for cache_mount in get_cache_directory_mounts():
                src = Path(cache_mount["host_path"])
                if not src.is_dir():
                    logger.warning(
                        "Docker: skipping cache mount — source is not a directory: %s",
                        src,
                    )
                    continue
                volume_args.extend([
                    "-v",
                    f"{cache_mount['host_path']}:{cache_mount['container_path']}:ro",
                ])
                logger.info(
                    "Docker: mounting cache dir %s -> %s",
                    cache_mount["host_path"],
                    cache_mount["container_path"],
                )
        except Exception as e:
            logger.debug("Docker: could not load credential file mounts: %s", e)

        # 显式环境变量（docker_env 配置）——
        # 在容器创建时进行设置，
        # 从而使它们能够被所有进程
        # （包括入口点 entrypoint）所使用。
        env_args = []
        for key in sorted(self._env):
            env_args.extend(["-e", f"{key}={self._env[key]}"])

        # 可选操作：以主机用户身份运行容器，
        # 以便写入到绑定挂载目录
        # （如 /workspace、/root 以及 docker_volumes 项）中的文件
        # 在主机上能够归属该用户所有，而非 root 用户。
        # 在不具备 POSIX uid/gid 的平台上
        # （例如原生的 Windows Docker 环境），则平滑地跳过此操作。
        user_args: list[str] = []
        if run_as_host_user:
            user_spec = _resolve_host_user_spec()
            if user_spec is not None:
                user_args = ["--user", user_spec]
                logger.info("Docker: running container as host user %s", user_spec)
            else:
                logger.warning(
                    "docker_run_as_host_user is enabled but this platform does "
                    "not expose POSIX uid/gid; container will start as its "
                    "image default user."
                )
                # Fall back to the full cap set — without --user, an image's
                # init may still need s6-setuidgid/gosu/su to drop privileges.

        # Resolve the docker executable once so it works even when
        # /usr/local/bin is not in PATH (common on macOS gateway/service).
        self._docker_exe = find_docker() or "docker"

        # s6-overlay 镜像（例如 hermes-agent:latest）
        # 已经使用 /init 作为 PID 1，
        # 并在启动期间执行 /run/s6/basedir/bin/init。
        # 对于这些镜像，我们必须
        # (a) 跳过 Docker 的 --init 参数（以避免两个 PID-1 初始化进程相互冲突），
        # 并且 (b) 使用 exec（而非 noexec）选项来挂载 /run 目录，
        # 否则 s6 的 stage0 将会异常终止，
        # 并返回退出码 126 "Permission denied"（权限被拒绝）。
        # 此处仅检测一次；
        # 若检查失败，则保留默认设置。
        # 详见 issue #34628。
        image_uses_s6_init = _image_uses_init_entrypoint(self._docker_exe, image)
        if image_uses_s6_init:
            logger.info(
                "Docker: image %s uses /init (s6-overlay) as entrypoint — "
                "skipping --init and mounting /run with exec.",
                image,
            )
        security_args = _build_security_args(
            run_as_host_user and bool(user_args),
            run_exec=image_uses_s6_init,
        )

        logger.info(f"Docker volume_args: {volume_args}")
        # User-supplied extra docker run flags (docker_extra_args in config.yaml).
        # Appended last so they can override defaults if needed.
        validated_extra = []
        for arg in (extra_args or []):
            if not isinstance(arg, str):
                logger.warning("Ignoring non-string docker_extra_args entry: %r", arg)
                continue
            validated_extra.append(arg)

        all_run_args = (
            security_args
            + user_args
            + writable_args
            + resource_args
            + volume_args
            + env_args
            + validated_extra
        )
        logger.info(f"Docker run_args: {all_run_args}")

        # Start the container directly via `docker run -d`.
        container_name = f"hermes-{uuid.uuid4().hex[:8]}"
        # 标签（Labels）使 Hermes 创建的容器能够被以下对象识别：
        #   * 孤儿回收程序（使用 `hermes-agent=1` 作为全局扫描过滤器）
        #   * 未来的跨进程复用（`hermes-task-id`, `hermes-profile`）
        #   * 运行 `docker ps --filter label=hermes-agent=1` 的操作人员
        # 标签值仅限于由 _sanitize_label_value()
        # 所定义的安全字符集；
        # 当前生效的 Hermes profile 会在容器启动时被捕获，
        # 且在容器的整个生命周期内保持不变。
        profile_name = _sanitize_label_value(_get_active_profile_name())
        task_label = _sanitize_label_value(task_id)
        label_args = [
            "--label", "hermes-agent=1",
            "--label", f"hermes-task-id={task_label}",
            "--label", f"hermes-profile={profile_name}",
        ]
        # Save args for container recreation on "No such container" recovery.
        self._image = image
        self._container_name = container_name
        self._image_uses_s6_init = image_uses_s6_init
        self._all_run_args = all_run_args

        self._labels = {
            "hermes-agent": "1",
            "hermes-task-id": task_label,
            "hermes-profile": profile_name,
        }

        # 跨进程容器复用（issue #20561 —— 文档声明“多个会话共享一个长久运行的容器”）。
        # 如果先前的 Hermes 进程已经为当前 (task_id, profile)
        # 启动了一个容器，并且该容器仍然存在，
        # 则直接附着（attach）到该容器，而不是启动一个全新的容器。
        # 这恢复了文档中所约定的行为逻辑；
        # 可通过设置 ``terminal.docker_persist_across_processes: false`` 来选择退出此机制。
        #
        # 复用仅根据标签（labels）进行匹配 ——
        # 我们特意不对镜像、挂载项以及资源配置进行比较。
        # 操作人员如果在修改这些设置后需要一个全新的容器，
        # 应当设置 ``docker_persist_across_processes: false``
        # （或者针对带有对应标签的容器运行 ``docker rm -f``），
        # 以强制进行干净的重新启动。
        reused = False
        if persist_across_processes:
            existing = self._find_reusable_container(task_label, profile_name)
            if existing is not None:
                container_id, state = existing
                # 网络模式卫士（ guard ）：复用绝不能在无声无息中破坏出站锁定（ egress lockdown ）。
                # 在操作人员设置 ``docker_network: false`` 之前创建的容器会保持其原有的网桥 NetworkMode，
                # 因此仅基于标签的复用会导致：
                # 尽管配置已更改，智能体（ agent ）依然会拿到一个带网络权限的容器。
                # 当网络模式不匹配时，我们会移除过期（ stale ）的容器并重新启动 ——
                # 如果保留该容器，会导致下一次基于标签的复用再次选中它。
                # 我们仅对锁定操作（即从允许联网变为禁止联网）进行保护：
                # 在默认网络配置下，处于 ``none`` 模式的容器将被原封不动地保留，
                # 这样使用 ``docker_extra_args: ["--network=none"]`` 的操作人员
                # 就不会在每次启动时都触发容器的频繁重建与替换。
                # ---
                # https://gemini.google.com/app/2fe24ca020da437f
                # “为了防止旧容器带着‘能联网’的旧权限偷偷绕过现在的‘断网封锁’指令，
                # 只要发现旧容器的网络权限比现在的要求大，就必须当场干掉它重做。”
                mode_mismatch = False
                actual_mode = None
                if not network:
                    actual_mode = self._container_network_mode(container_id)
                    mode_mismatch = actual_mode != "none"
                if mode_mismatch:
                    logger.warning(
                        "Existing container %s has NetworkMode=%s but "
                        "docker_network=false requests an air-gapped "
                        "container — removing it and starting fresh "
                        "(task=%s, profile=%s).",
                        container_id[:12], actual_mode or "unknown",
                        task_label, profile_name,
                    )
                    try:
                        subprocess.run(
                            [self._docker_exe, "rm", "-f", container_id],
                            capture_output=True,
                            text=True,
                            timeout=30,
                            check=False,
                            stdin=subprocess.DEVNULL,
                        )
                    except (subprocess.TimeoutExpired, OSError) as e:
                        logger.warning("Failed to remove mismatched container %s: %s", container_id[:12], e)
                    existing = None


            if existing is not None:
                container_id, state = existing
                self._container_id = container_id
                if state != "running":
                    try:
                        subprocess.run(
                            [self._docker_exe, "start", container_id],
                            capture_output=True,
                            text=True,
                            timeout=30,
                            check=True,
                            stdin=subprocess.DEVNULL,
                        )
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                        logger.warning(
                            "Failed to start existing container %s (state=%s): "
                            "%s — falling back to a fresh container.",
                            container_id[:12], state, e,
                        )
                        self._container_id = None
                if self._container_id:
                    logger.info(
                        "Reusing container %s (task=%s, profile=%s, prior state=%s)",
                        container_id[:12], task_label, profile_name, state,
                    )
                    reused = True

        if not reused:
            # tini/catatonit 作为 PID 1 进程可以回收僵尸子进程 ——
            # 但 s6-overlay 镜像已经提供了其自身的 /init PID 1，
            # 因此在该环境下添加 --init 会产生两个相互冲突的初始化进程，
            # 并导致启动失败（#34628）。
            init_args = [] if image_uses_s6_init else ["--init"]
            run_cmd = [
                self._docker_exe, "run", "-d",
                *init_args,
                "--name", container_name,
                *label_args,
                "-w", cwd,
                *all_run_args,
                image,
                "sleep", "infinity",  # no fixed lifetime — idle reaper handles cleanup
            ]
            logger.debug(f"Starting container: {' '.join(run_cmd)}")
            try:
                result = subprocess.run(
                    run_cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,  # image pull may take a while
                    check=True,
                    stdin=subprocess.DEVNULL,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                # Docker may create the container object before `docker run`
                # fails to start it (e.g. exit code 125 when the daemon isn't
                # ready, or a timeout mid-pull). That orphan is left in
                # "Created" state — which the exited-only orphan reaper
                # (reap_orphan_containers, status=exited) never catches, so it
                # leaks permanently. Remove it by its known name before
                # re-raising. See #7439.
                logger.warning(
                    "docker run failed for %s, cleaning up orphaned container: %s",
                    container_name, e,
                )
                subprocess.run(
                    [self._docker_exe, "rm", "-f", container_name],
                    capture_output=True, timeout=10,
                    stdin=subprocess.DEVNULL,
                )
                raise
            self._container_id = result.stdout.strip()
            logger.info(f"Started container {container_name} ({self._container_id[:12]})")

        # 构建初始化阶段的环境变量转发参数
        # （仅在 init_session 中使用，
        # 用于将主机环境变量注入快照；
        # 后续命令将直接从快照文件中获取这些变量）。
        self._init_env_args = self._build_init_env_args()

        # 在容器内部初始化会话快照
        self.init_session()

    def _build_init_env_args(self) -> list[str]:
        """Build -e KEY=VALUE args for injecting host env vars into init_session.

        These are used once during init_session() so that export -p captures
        them into the snapshot.  Subsequent execute() calls don't need -e flags.
        """
        exec_env: dict[str, str] = dict(self._env)

        explicit_forward_keys = set(self._forward_env)
        passthrough_keys: set[str] = set()
        try:
            from tools.env_passthrough import get_all_passthrough
            passthrough_keys = set(get_all_passthrough())
        except Exception:
            pass
        # Explicit docker_forward_env entries are an intentional opt-in and must
        # win over the generic Hermes secret blocklist. Only implicit passthrough
        # keys are filtered. Also strip Hermes-internal dynamic secrets
        # (AUXILIARY_*_API_KEY / _BASE_URL, GATEWAY_RELAY_* auth) that the
        # name-based blocklist doesn't cover — see _is_hermes_internal_secret.
        _implicit_forward = {
            k for k in passthrough_keys if not _is_hermes_internal_secret(k)
        }
        forward_keys = explicit_forward_keys | (_implicit_forward - _HERMES_PROVIDER_ENV_BLOCKLIST)
        hermes_env = _load_hermes_env_vars() if forward_keys else {}
        for key in sorted(forward_keys):
            value = os.getenv(key)
            if not value:
                value = hermes_env.get(key)
            if value:
                exec_env[key] = value

        args = []
        for key in sorted(exec_env):
            args.extend(["-e", f"{key}={exec_env[key]}"])
        return args

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        """Spawn a bash process inside the Docker container."""
        assert self._container_id, "Container not started"
        cmd = [self._docker_exe, "exec"]
        if stdin_data is not None:
            cmd.append("-i")

        # Only inject -e env args during init_session (login=True).
        # Subsequent commands get env vars from the snapshot.
        if login:
            cmd.extend(self._init_env_args)

        cmd.extend([self._container_id])

        if login:
            cmd.extend(["bash", "-l", "-c", cmd_string])
        else:
            cmd.extend(["bash", "-c", cmd_string])

        return _popen_bash(cmd, stdin_data)

    # ------------------------------------------------------------------
    # "No such container" recovery (issue #36266)
    # ------------------------------------------------------------------

    _NO_CONTAINER_PATTERNS = (
        "No such container",
        "is not running",
        "no such container",
    )

    def _is_container_gone(self, output: str) -> bool:
        """Return True if the output indicates the container no longer exists."""
        return any(p in output for p in self._NO_CONTAINER_PATTERNS)

    def _recreate_container(self) -> bool:
        """在容器被外部流程移除后，重新创建该容器。

        优先尝试基于标签（label-based）的复用机制；
        如果未找到现有容器，
        则使用相同的镜像和运行参数启动一个全新的容器。
        成功时返回 True，
        若重新创建失败则返回 False（调用方应向上抛出原始错误）。
        """
        old_id = (self._container_id or "")[:12]
        logger.warning(
            "Container %s appears to be gone — attempting recovery", old_id,
        )
        self._container_id = None

        # 1. Try label-based reuse (another process may have recreated it).
        task_label = self._labels.get("hermes-task-id", "")
        profile_label = self._labels.get("hermes-profile", "")
        existing = self._find_reusable_container(task_label, profile_label)
        if existing is not None:
            cid, state = existing
            if state == "running":
                self._container_id = cid
                logger.info("Recovery: reusing running container %s", cid[:12])
            else:
                try:
                    subprocess.run(
                        [self._docker_exe, "start", cid],
                        capture_output=True, text=True, timeout=30, check=True,
                        stdin=subprocess.DEVNULL,
                    )
                    self._container_id = cid
                    logger.info("Recovery: restarted container %s", cid[:12])
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    logger.warning("Recovery: failed to start container %s: %s", cid[:12], e)

        # 2. No reusable container — create a fresh one.
        if not self._container_id:
            if not self._image:
                logger.error("Recovery: no saved image name, cannot recreate container")
                return False
            try:
                import uuid as _uuid
                new_name = f"hermes-{_uuid.uuid4().hex[:8]}"
                init_args = [] if self._image_uses_s6_init else ["--init"]
                label_args = []
                for k, v in self._labels.items():
                    label_args.extend(["--label", f"{k}={v}"])
                run_cmd = [
                    self._docker_exe, "run", "-d",
                    *init_args,
                    "--name", new_name,
                    *label_args,
                    "-w", self.cwd,
                    *self._all_run_args,
                    self._image,
                    "sleep", "infinity",
                ]
                result = subprocess.run(
                    run_cmd, capture_output=True, text=True, timeout=120, check=True,
                    stdin=subprocess.DEVNULL,
                )
                self._container_id = result.stdout.strip()
                self._container_name = new_name
                logger.info(
                    "Recovery: created fresh container %s (%s)",
                    new_name, self._container_id[:12],
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
                logger.error("Recovery: failed to create new container: %s", e)
                return False

        # 3. Re-initialize session snapshot in the (re)created container.
        try:
            self._snapshot_ready = False
            self.init_session()
        except Exception as e:
            logger.error("Recovery: init_session failed in new container: %s", e)
            return False

        logger.info("Recovery successful — new container %s", (self._container_id or "")[:12])
        return True

    def execute(self, command: str, cwd: str = "", **kwargs) -> dict:
        """执行命令，并自动从已失效的容器中恢复。

        如果容器被外部流程移除（例如空闲回收清理器、docker prune、OOM 杀进程、守护进程重启等），
        检测该错误并在透明地重新创建容器后，
        进行一次重试。
        """
        result = super().execute(command, cwd, **kwargs)
        if (
            result.get("returncode", 0) != 0
            and self._is_container_gone(result.get("output", ""))
            and self._persist_across_processes
        ):
            if self._recreate_container():
                result = super().execute(command, cwd, **kwargs)
        return result

    @staticmethod
    def _storage_opt_supported() -> bool:
        """Check if Docker's storage driver supports --storage-opt size=.
        
        Only overlay2 on XFS with pquota supports per-container disk quotas.
        Ubuntu (and most distros) default to ext4, where this flag errors out.
        """
        global _storage_opt_ok
        if _storage_opt_ok is not None:
            return _storage_opt_ok
        try:
            docker = find_docker() or "docker"
            result = subprocess.run(
                [docker, "info", "--format", "{{.Driver}}"],
                capture_output=True, text=True, timeout=10,
                stdin=subprocess.DEVNULL,
            )
            driver = result.stdout.strip().lower()
            if driver != "overlay2":
                _storage_opt_ok = False
                return False
            # overlay2 only supports storage-opt on XFS with pquota.
            # Probe by attempting a dry-ish run — the fastest reliable check.
            probe = subprocess.run(
                [docker, "create", "--storage-opt", "size=1m", "hello-world"],
                capture_output=True, text=True, timeout=15,
                stdin=subprocess.DEVNULL,
            )
            if probe.returncode == 0:
                # Clean up the created container
                container_id = probe.stdout.strip()
                if container_id:
                    subprocess.run([docker, "rm", container_id],
                                   capture_output=True, timeout=5,
                                   stdin=subprocess.DEVNULL)
                _storage_opt_ok = True
            else:
                _storage_opt_ok = False
        except Exception:
            _storage_opt_ok = False
        logger.debug("Docker --storage-opt support: %s", _storage_opt_ok)
        return _storage_opt_ok

    def _container_network_mode(self, container_id: str) -> Optional[str]:
        """返回容器的 ``HostConfig.NetworkMode``
        （例如 ``bridge``、``none``、``host``），
        或者在检查（inspection）失败时返回 ``None``。

        供复用路径使用，
        以确保已持久化容器的网络模式
        仍然与操作人员的 ``docker_network`` 设置相匹配；
        当请求进行网络锁定（lockdown）时，
        调用方会将 ``None``（未知状态）视为不匹配，
        因此检查失败时会按安全闭合（fail closed）而非开放的方式处理。
        """
        try:
            result = subprocess.run(
                [
                    self._docker_exe, "inspect",
                    "--format", "{{.HostConfig.NetworkMode}}",
                    container_id,
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("docker inspect NetworkMode failed: %s", e)
            return None
        if result.returncode != 0:
            logger.debug(
                "docker inspect NetworkMode returned %d: %s",
                result.returncode, result.stderr.strip(),
            )
            return None
        mode = result.stdout.strip()
        return mode or None

    def _find_reusable_container(self, task_label: str, profile_label: str) -> Optional[tuple[str, str]]:
        """查找带有为此 (task, profile) 标记的现有容器。

        命中时返回 ``(container_id, state)``，
        未命中或发生任何失败（包括 ``docker ps`` 本身失败）时返回 ``None``。
        状态（state）是 Docker 通过 ``{{.State}}`` 报告的值之一 ——
        例如 ``running``、``exited``、``created``、``paused``、``restarting``、``dead``。
        调用方自行决定该状态在复用前是否需要执行 ``docker start``。

        仅限于此类创建的、存放在 Docker 中的标签集合；
        绝不会匹配那些恰好命名为 ``hermes-*``
        但由其他工具启动的容器。
        """
        try:
            result = subprocess.run(
                [
                    self._docker_exe, "ps", "-a",
                    "--filter", "label=hermes-agent=1",
                    "--filter", f"label=hermes-task-id={task_label}",
                    "--filter", f"label=hermes-profile={profile_label}",
                    "--format", "{{.ID}}\t{{.State}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("docker ps probe failed: %s — will start a fresh container", e)
            return None
        if result.returncode != 0:
            logger.debug(
                "docker ps probe returned %d: %s — will start a fresh container",
                result.returncode, result.stderr.strip(),
            )
            return None
        lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
        if not lines:
            return None
        # Multiple matches are unusual (one (task, profile) should produce one
        # container) but can happen if a previous Hermes process crashed
        # mid-cleanup. Prefer a running one if present; otherwise pick the
        # first listed. Stale duplicates get reaped by the orphan-reaper in a
        # follow-up commit; we don't try to be heroic about them here.
        running = None
        first = None
        for ln in lines:
            parts = ln.split("\t", 1)
            if len(parts) != 2:
                continue
            cid, state = parts[0], parts[1].lower()
            if first is None:
                first = (cid, state)
            if state == "running" and running is None:
                running = (cid, state)
        return running or first

    def cleanup(self, *, force_remove: bool = False):
        """Tear down the container according to persist mode and *force_remove*.

        Persist-mode (``persist_across_processes=True``, the default) leaves the
        container **running** untouched. The docs promise "ONE long-lived
        container shared across sessions" and stopping it on every Hermes exit
        breaks that promise:

        * Background processes inside the container (``npm run dev``, watchers,
          long-running pytest) get killed every time the user runs ``/quit``.
        * Every reuse requires ``docker start`` + waiting for the container to
          come back up, adding 1–2s to the first tool call of the new session.
        * The user-visible difference between "ONE long-lived container" and
          "a new container that happens to share state" is exactly this:
          processes survive in the former, die in the latter.

        Resource reclamation for the persist-mode case lives in the
        ``reap_orphan_containers()`` path (see issue #20561 commit 3): if no
        Hermes process touches a labeled container for ``2 × lifetime_seconds``
        it gets ``docker rm -f``'d at the next Hermes startup. That covers the
        SIGKILL / OOM / abandoned-laptop cases without us needing to stop the
        container on every graceful exit.

        Opt-out mode (``persist_across_processes=False``) still does
        ``docker stop`` + ``docker rm -f`` on every cleanup, matching the
        pre-PR behavior for users who explicitly want per-process isolation.

        ``force_remove=True`` overrides persist mode and always tears the
        container down (``docker stop`` + ``docker rm -f``). This is the
        explicit-teardown path for ``/reset``, ``cleanup_vm(task_id)``-driven
        resets, or any caller that wants a guaranteed fresh container on next
        ``DockerEnvironment(task_id=...)``. No current caller passes
        ``force_remove=True``; the parameter is here so the explicit-teardown
        semantics can be wired up later without changing this method's
        signature.

        Cleanup runs on a daemon thread with bounded ``subprocess.run`` calls
        (not the racy ``Popen(... &)`` pattern from before PR #33645). The
        atexit hook in ``tools/terminal_tool.py`` waits up to 15s for the
        thread to finish before the interpreter exits, so ``docker stop`` /
        ``docker rm`` actually completes when we do trigger it.
        """
        container_id = self._container_id
        if not container_id:
            # Still drop the bind-mount dirs if any were allocated and we're
            # NOT in persist mode (persist mode preserves them).
            if not self._persistent:
                for d in (self._workspace_dir, self._home_dir):
                    if d:
                        shutil.rmtree(d, ignore_errors=True)
            return

        # Decide what to actually do. Three cases:
        #
        #   force_remove=True             → stop + rm (explicit teardown)
        #   persist_across_processes=True → no-op (leave container running)
        #   persist_across_processes=False → stop + rm (per-process isolation)
        #
        # The persist-mode no-op is the issue-#20561 contract: the container
        # outlives Hermes processes, processes inside it stay alive, and
        # reuse on next startup is instant.
        if force_remove:
            should_stop = True
            should_remove = True
        elif self._persist_across_processes:
            # No-op for the container. Drop the in-process handle so a fresh
            # __init__ will re-probe via labels (and find the running
            # container) instead of trying to reuse a stale Python reference.
            self._container_id = None
            return
        else:
            should_stop = True
            should_remove = True

        # Capture state needed by the worker before we null out the attrs —
        # the worker thread can outlive ``self``.
        docker_exe = self._docker_exe
        log_id = container_id[:12]

        def _do_cleanup() -> None:
            if should_stop:
                try:
                    subprocess.run(
                        [docker_exe, "stop", "-t", "10", container_id],
                        capture_output=True, timeout=30,
                        stdin=subprocess.DEVNULL,
                    )
                except (subprocess.TimeoutExpired, OSError) as e:
                    logger.warning("docker stop %s timed out / failed: %s", log_id, e)
            if should_remove:
                try:
                    subprocess.run(
                        [docker_exe, "rm", "-f", container_id],
                        capture_output=True, timeout=30,
                        stdin=subprocess.DEVNULL,
                    )
                except (subprocess.TimeoutExpired, OSError) as e:
                    logger.warning("docker rm -f %s failed: %s", log_id, e)

        # Daemon thread: doesn't block interpreter exit (atexit returns
        # promptly), but unlike the old ``Popen(... &)`` shell trick the
        # Python-level join semantics let the thread actually run to
        # completion if the interpreter is still alive. atexit registers
        # ``_atexit_cleanup`` in terminal_tool.py which waits up to ~60s for
        # outstanding cleanups, so most exits complete the work cleanly.
        import threading
        t = threading.Thread(target=_do_cleanup, daemon=True, name=f"hermes-cleanup-{log_id}")
        t.start()
        self._cleanup_thread = t
        self._container_id = None

        # Bind-mount dir teardown only runs when we actually removed the
        # container (the dirs are the container's filesystem state; keeping
        # them around with no container would orphan the data on disk).
        if should_remove and not self._persistent:
            for d in (self._workspace_dir, self._home_dir):
                if d:
                    shutil.rmtree(d, ignore_errors=True)

    def wait_for_cleanup(self, timeout: float = 30.0) -> bool:
        """Block up to *timeout* seconds for the cleanup worker thread.

        Returns ``True`` if the thread finished (or no thread was started),
        ``False`` on timeout. The atexit hook in terminal_tool.py calls this
        on every active environment so docker stop/rm actually completes
        before the Python process exits — without this, ``hermes /quit``
        races the interpreter shutdown and leaves stopped containers behind.
        """
        thread = getattr(self, "_cleanup_thread", None)
        if thread is None or not thread.is_alive():
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()
