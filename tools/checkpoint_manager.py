"""
检查点管理器（Checkpoint Manager）— 通过单个共享的
影子 Git 仓库实现的透明文件系统快照。

在执行可修改文件的操作（``write_file``、``patch``、带有破坏性标志的 ``terminal`` 命令）前，
自动创建工作目录的快照，每个对话轮次触发一次。
提供恢复（rollback）至任意历史检查点的功能。

这**不是**一个工具 — LLM 永远看不到它。
它是由 ``checkpoints`` 配置项或 ``--checkpoints`` CLI 标志控制的透明基础设施。

存储布局（单个共享存储，Git 对象在跨项目间自动去重）
-----------------------------------------------------------------------------

    ~/.hermes/checkpoints/
        store/                          — 单个类 bare Git 仓库
            HEAD, config, objects/      — 标准 Git 内部文件（共享）
            refs/hermes/<hash16>        — 每个项目的分支 Head
            indexes/<hash16>            — 每个项目的 Git 索引
            projects/<hash16>.json      — {workdir, created_at, last_touch}
            info/exclude                — 默认排除规则（共享）
        .last_prune                     — 自动清理的等幂性标记
        legacy-<timestamp>/             — 归档的 v2 版本前按项目划分的影子仓库
                                          （首次初始化时自动迁移）

为什么使用单个存储？
-------------------

v2 版本之前的设计是为每个工作目录保留一个完整的影子仓库。
每个仓库都在自己的 ``objects/`` 树下重新存储项目的大部分文件，
即使是同一个项目的不同工作树（worktree）之间也完全无法共享。
若单个用户拥有同一个仓库的十几个工作树，
每份都会消耗约 40 MB（总计约 500 MB），反复存储相同的 Blob 对象。
改用单个共享存储后，Git 基于内容寻址（content-addressable）的对象数据库
能够跨项目、跨对话轮次进行数据去重，因此新增一个工作树的成本几乎为零。

影子存储通过结合使用 ``GIT_DIR`` + ``GIT_WORK_TREE`` + ``GIT_INDEX_FILE``，
确保不会有任何 Git 状态泄漏到用户的项目目录中。

自动维护
--------

影子状态会随着时间不断累积。``prune_checkpoints`` 会删除那些
对应的工作目录已不存在（孤立/orphan）或最后触碰时间已超过 ``retention_days``（过期/stale）的引用（refs），
随后运行 ``git gc --prune=now`` 来回收对象存储空间。
如果总存储空间超出限制，容量上限清理逻辑（size-cap pass）还会逐个项目丢弃最老旧的检查点，
直到存储库总大小降至 ``max_total_size_mb`` 以下。
"""

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from hermes_constants import get_hermes_home
from hermes_cli._subprocess_compat import windows_hide_flags
from typing import Dict, List, Optional, Set, Tuple

from utils import env_int

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_BASE = get_hermes_home() / "checkpoints"

# Single shared store directory under CHECKPOINT_BASE.
_STORE_DIRNAME = "store"
_REFS_PREFIX = "refs/hermes"
_INDEXES_DIRNAME = "indexes"
_PROJECTS_DIRNAME = "projects"
_LEGACY_PREFIX = "legacy-"

DEFAULT_EXCLUDES = [
    # Dependency / build output
    "node_modules/",
    "dist/",
    "build/",
    "target/",
    "out/",
    ".next/",
    ".nuxt/",
    # Caches
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".cache/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "coverage/",
    ".coverage",
    # Virtualenvs
    ".venv/",
    "venv/",
    "env/",
    # VCS
    ".git/",
    ".hg/",
    ".svn/",
    # Worktrees (Hermes convention — don't recursively snapshot siblings)
    ".worktrees/",
    # Native / compiled binaries
    "*.so",
    "*.dylib",
    "*.dll",
    "*.o",
    "*.a",
    "*.jar",
    "*.class",
    "*.exe",
    "*.obj",
    # Media / large binaries
    "*.mp4",
    "*.mov",
    "*.mkv",
    "*.webm",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.7z",
    "*.rar",
    "*.iso",
    # Secrets
    ".env",
    ".env.*",
    ".env.local",
    ".env.*.local",
    # OS junk
    ".DS_Store",
    "Thumbs.db",
    # Logs
    "*.log",
]

# Git subprocess timeout (seconds).
_GIT_TIMEOUT: int = max(10, min(60, env_int("HERMES_CHECKPOINT_TIMEOUT", 30)))

# Max files to snapshot — skip huge directories to avoid slowdowns.
_MAX_FILES = 50_000

# Valid git commit hash pattern: 4–40 hex chars (short or full SHA-1/SHA-256).
_COMMIT_HASH_RE = re.compile(r'^[0-9a-fA-F]{4,64}$')


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _validate_commit_hash(commit_hash: str) -> Optional[str]:
    """Validate a commit hash to prevent git argument injection.

    Returns an error string if invalid, None if valid.
    Values starting with '-' would be interpreted as git flags
    (e.g., '--patch', '-p') instead of revision specifiers.
    """
    if not commit_hash or not commit_hash.strip():
        return "Empty commit hash"
    if commit_hash.startswith("-"):
        return f"Invalid commit hash (must not start with '-'): {commit_hash!r}"
    if not _COMMIT_HASH_RE.match(commit_hash):
        return f"Invalid commit hash (expected 4-64 hex characters): {commit_hash!r}"
    return None


def _validate_file_path(file_path: str, working_dir: str) -> Optional[str]:
    """Validate a file path to prevent path traversal outside the working directory.

    Returns an error string if invalid, None if valid.
    """
    if not file_path or not file_path.strip():
        return "Empty file path"
    if os.path.isabs(file_path):
        return f"File path must be relative, got absolute path: {file_path!r}"
    abs_workdir = _normalize_path(working_dir)
    resolved = (abs_workdir / file_path).resolve()
    try:
        resolved.relative_to(abs_workdir)
    except ValueError:
        return f"File path escapes the working directory via traversal: {file_path!r}"
    return None


# ---------------------------------------------------------------------------
# Path / hash helpers
# ---------------------------------------------------------------------------

def _normalize_path(path_value: str) -> Path:
    """Return a canonical absolute path for checkpoint operations."""
    return Path(path_value).expanduser().resolve()


def _project_hash(working_dir: str) -> str:
    """Deterministic per-project hash: sha256(abs_path)[:16]."""
    abs_path = str(_normalize_path(working_dir))
    return hashlib.sha256(abs_path.encode()).hexdigest()[:16]


def _store_path(base: Optional[Path] = None) -> Path:
    """Return the single shared shadow store path."""
    return (base or CHECKPOINT_BASE) / _STORE_DIRNAME


def _shadow_repo_path(working_dir: str) -> Path:  # pragma: no cover — kept for BC
    """Return the shared store path.

    Retained for backward-compatibility with callers / tests that imported
    this helper.  Under v2 the shadow git storage is shared across all
    projects — per-project isolation lives in refs and indexes, not in
    separate repo directories.
    """
    return _store_path()


def _index_path(store: Path, dir_hash: str) -> Path:
    return store / _INDEXES_DIRNAME / dir_hash


def _ref_name(dir_hash: str) -> str:
    return f"{_REFS_PREFIX}/{dir_hash}"


def _project_meta_path(store: Path, dir_hash: str) -> Path:
    return store / _PROJECTS_DIRNAME / f"{dir_hash}.json"


# ---------------------------------------------------------------------------
# Git env
# ---------------------------------------------------------------------------

def _git_env(
    store: Path,
    working_dir: str,
    index_file: Optional[Path] = None,
) -> dict:
    """Build env dict that redirects git to the shared store.

    The shared store is internal Hermes infrastructure — it must NOT inherit
    the user's global or system git config.  User-level settings like
    ``commit.gpgsign = true``, signing hooks, or credential helpers would
    either break background snapshots or, worse, spawn interactive prompts
    (pinentry GUI windows) mid-session every time a file is written.

    Isolation strategy:
    * ``GIT_CONFIG_GLOBAL=<os.devnull>`` — ignore ``~/.gitconfig`` (git 2.32+).
    * ``GIT_CONFIG_SYSTEM=<os.devnull>`` — ignore ``/etc/gitconfig`` (git 2.32+).
    * ``GIT_CONFIG_NOSYSTEM=1`` — legacy belt-and-suspenders for older git.

    ``index_file``, if given, forces git to use a per-project index under
    ``store/indexes/<hash>`` so projects don't race on a shared index.
    """
    normalized_working_dir = _normalize_path(working_dir)
    env = os.environ.copy()
    env["GIT_DIR"] = str(store)
    env["GIT_WORK_TREE"] = str(normalized_working_dir)
    env.pop("GIT_NAMESPACE", None)
    env.pop("GIT_ALTERNATE_OBJECT_DIRECTORIES", None)
    if index_file is not None:
        env["GIT_INDEX_FILE"] = str(index_file)
    else:
        env.pop("GIT_INDEX_FILE", None)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


def _repair_bare_repo_dirs(store: Path) -> None:
    """Recreate refs/ and branches/ dirs that ``git gc`` may have removed.

    ``git gc --prune=now`` on a bare repo with only packed refs can remove
    the empty ``refs/heads/`` directory.  Git 2.34+ requires ``refs/`` (and
    some versions require ``branches/``) to exist even when all refs are
    packed in ``packed-refs``.  Without them, ``git add -A`` returns
    ``fatal: not a git repository`` and all checkpoint operations fail
    silently.
    """
    for subdir in ("refs/heads", "branches"):
        path = store / subdir
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.debug("Repaired missing %s in checkpoint store", subdir)
            except OSError as exc:
                logger.warning(
                    "Cannot create %s in checkpoint store: %s", subdir, exc,
                )


def _run_git(
    args: List[str],
    store: Path,
    working_dir: str,
    timeout: int = _GIT_TIMEOUT,
    allowed_returncodes: Optional[Set[int]] = None,
    index_file: Optional[Path] = None,
) -> Tuple[bool, str, str]:
    """Run a git command against the shared store.  Returns (ok, stdout, stderr).

    ``allowed_returncodes`` suppresses error logging for known/expected non-zero
    exits while preserving the normal ``ok = (returncode == 0)`` contract.
    Example: ``git diff --cached --quiet`` returns 1 when changes exist.
    """
    normalized_working_dir = _normalize_path(working_dir)
    if not normalized_working_dir.exists():
        msg = f"working directory not found: {normalized_working_dir}"
        logger.error("Git command skipped: %s (%s)", " ".join(["git"] + list(args)), msg)
        return False, "", msg
    if not normalized_working_dir.is_dir():
        msg = f"working directory is not a directory: {normalized_working_dir}"
        logger.error("Git command skipped: %s (%s)", " ".join(["git"] + list(args)), msg)
        return False, "", msg

    env = _git_env(store, str(normalized_working_dir), index_file=index_file)
    cmd = ["git"] + list(args)
    allowed_returncodes = allowed_returncodes or set()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(normalized_working_dir),
            stdin=subprocess.DEVNULL,
            # Checkpoints fire several bare git calls per turn from the
            # console-less desktop/gateway backend; suppress the per-call
            # conhost flash on Windows (no-op on POSIX).
            creationflags=windows_hide_flags(),
        )
        ok = result.returncode == 0
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if not ok and result.returncode not in allowed_returncodes:
            logger.error(
                "Git command failed: %s (rc=%d) stderr=%s",
                " ".join(cmd), result.returncode, stderr,
            )
        return ok, stdout, stderr
    except subprocess.TimeoutExpired:
        msg = f"git timed out after {timeout}s: {' '.join(cmd)}"
        logger.error(msg, exc_info=True)
        return False, "", msg
    except FileNotFoundError as exc:
        missing_target = getattr(exc, "filename", None)
        if missing_target == "git":
            logger.error("Git executable not found: %s", " ".join(cmd), exc_info=True)
            return False, "", "git not found"
        msg = f"working directory not found: {normalized_working_dir}"
        logger.error("Git command failed before execution: %s (%s)", " ".join(cmd), msg, exc_info=True)
        return False, "", msg
    except Exception as exc:
        logger.error("Unexpected git error running %s: %s", " ".join(cmd), exc, exc_info=True)
        return False, "", str(exc)


# ---------------------------------------------------------------------------
# Store initialisation + legacy migration
# ---------------------------------------------------------------------------

def _migrate_legacy_store(base: Path) -> Optional[Path]:
    """Move pre-v2 per-project shadow repos into a ``legacy-<ts>/`` dir.

    The pre-v2 layout had one shadow git repo per working directory directly
    under ``CHECKPOINT_BASE``.  The v2 layout wants a single ``store/`` dir.
    Rather than delete the old data (users might want to recover), rename
    everything except our own v2 entries into ``legacy-<timestamp>/``.  The
    legacy dir is subject to the same retention sweep and can be manually
    cleared with ``hermes checkpoints clear-legacy``.

    Returns the legacy-archive path, or None if nothing to migrate.
    """
    if not base.exists():
        return None
    store = _store_path(base)
    legacy_root: Optional[Path] = None
    # Reserved top-level entries managed by v2.
    reserved = {_STORE_DIRNAME, _PRUNE_MARKER_NAME}
    for child in list(base.iterdir()):
        name = child.name
        if name in reserved or name.startswith(_LEGACY_PREFIX):
            continue
        # Candidate: pre-v2 shadow repo (has HEAD) OR stray dir.  Either way
        # we archive it so v2 starts clean.
        if legacy_root is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            legacy_root = base / f"{_LEGACY_PREFIX}{stamp}"
            try:
                legacy_root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("Could not create legacy archive dir: %s", exc)
                return None
        dest = legacy_root / name
        try:
            shutil.move(str(child), str(dest))
        except OSError as exc:
            logger.warning("Could not archive legacy checkpoint %s: %s", child, exc)
    # If the store still hasn't been created, create it here.
    _ = store
    if legacy_root is not None:
        logger.info(
            "Migrated pre-v2 checkpoint repos to %s. "
            "Clear with `hermes checkpoints clear-legacy` when safe.",
            legacy_root,
        )
    return legacy_root


def _init_store(store: Path, working_dir: str) -> Optional[str]:
    """Initialise the shared shadow store if needed.  Returns error or None.

    Also performs one-time migration of pre-v2 per-directory shadow repos
    into ``legacy-<timestamp>/``.
    """
    base = store.parent
    # One-time legacy migration before we create the store.
    if not store.exists():
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return f"Could not create checkpoint base: {exc}"
        # Only migrate if the base dir has pre-existing content that isn't
        # our own v2 layout.
        _migrate_legacy_store(base)

    if (store / "HEAD").exists():
        return None

    store.mkdir(parents=True, exist_ok=True)
    (store / _INDEXES_DIRNAME).mkdir(exist_ok=True)
    (store / _PROJECTS_DIRNAME).mkdir(exist_ok=True)

    # ``git init --bare`` rejects GIT_WORK_TREE, so we can't use _run_git
    # here (which always sets GIT_DIR + GIT_WORK_TREE).  Use a raw
    # subprocess with just the config-isolation env vars.
    init_env = os.environ.copy()
    init_env["GIT_CONFIG_GLOBAL"] = os.devnull
    init_env["GIT_CONFIG_SYSTEM"] = os.devnull
    init_env["GIT_CONFIG_NOSYSTEM"] = "1"
    # Drop any inherited GIT_* that would interfere.
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_NAMESPACE",
              "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        init_env.pop(k, None)
    try:
        result = subprocess.run(
            ["git", "init", "--bare", str(store)],
            capture_output=True, text=True,
            env=init_env, timeout=_GIT_TIMEOUT,
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
        if result.returncode != 0:
            return f"Shadow store init failed: {result.stderr.strip()}"
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return f"Shadow store init failed: {exc}"

    # Per-store config (isolated by env vars above, but belt-and-suspenders).
    # Use the base dir as the working_dir for config commands — it always
    # exists since we just created the store inside it.
    cfg_wd = str(base)
    _run_git(["config", "user.email", "hermes@local"], store, cfg_wd)
    _run_git(["config", "user.name", "Hermes Checkpoint"], store, cfg_wd)
    _run_git(["config", "commit.gpgsign", "false"], store, cfg_wd)
    _run_git(["config", "tag.gpgSign", "false"], store, cfg_wd)
    _run_git(["config", "gc.auto", "0"], store, cfg_wd)

    info_dir = store / "info"
    info_dir.mkdir(exist_ok=True)
    (info_dir / "exclude").write_text(
        "\n".join(DEFAULT_EXCLUDES) + "\n", encoding="utf-8"
    )

    logger.debug("Initialised checkpoint store at %s", store)
    return None


def _register_project(store: Path, working_dir: str) -> None:
    """Create or update ``projects/<hash>.json`` with workdir + timestamps."""
    dir_hash = _project_hash(working_dir)
    meta_path = _project_meta_path(store, dir_hash)
    now = time.time()
    meta: Dict = {"workdir": str(_normalize_path(working_dir)),
                  "created_at": now, "last_touch": now}
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                meta["created_at"] = existing.get("created_at", now)
        except (OSError, ValueError):
            pass
    try:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not write project metadata %s: %s", meta_path, exc)


def _touch_project(store: Path, working_dir: str) -> None:
    """Update last_touch for a project, preserving created_at."""
    dir_hash = _project_hash(working_dir)
    meta_path = _project_meta_path(store, dir_hash)
    if not meta_path.exists():
        _register_project(store, working_dir)
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    meta["workdir"] = str(_normalize_path(working_dir))
    meta["last_touch"] = time.time()
    meta.setdefault("created_at", meta["last_touch"])
    try:
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not update project metadata %s: %s", meta_path, exc)


def _list_projects(store: Path) -> List[Dict]:
    """Return all registered projects under the store."""
    projects_dir = store / _PROJECTS_DIRNAME
    if not projects_dir.exists():
        return []
    out: List[Dict] = []
    for meta_path in projects_dir.glob("*.json"):
        dir_hash = meta_path.stem
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        meta["_hash"] = dir_hash
        out.append(meta)
    return out


def _dir_file_count(path: str) -> int:
    """Quick file count estimate (stops early if over _MAX_FILES)."""
    count = 0
    try:
        for _ in Path(path).rglob("*"):
            count += 1
            if count > _MAX_FILES:
                return count
    except (PermissionError, OSError):
        pass
    return count


def _dir_size_bytes(path: Path) -> int:
    """Best-effort recursive size in bytes.  Returns 0 on error."""
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


# Backwards-compatibility shim — some tests import ``_init_shadow_repo`` and
# look for ``HEAD``/``info/exclude``/``HERMES_WORKDIR``.  In v2 we also write
# those markers, but inside the shared store + under ``projects/<hash>.json``.
# The shim initialises the store and registers the project so the old
# surface keeps roughly the same shape.
def _init_shadow_repo(shadow_repo: Path, working_dir: str) -> Optional[str]:
    """Backwards-compatible initialiser.

    In v1 ``shadow_repo`` was a per-project dir; in v2 it's the shared
    ``store/`` path (or a test path that we respect).  We initialise the
    store at ``shadow_repo``, create per-project markers, and return None
    on success.
    """
    err = _init_store(shadow_repo, working_dir)
    if err:
        return err
    _register_project(shadow_repo, working_dir)
    # Compat marker for tests that look at HERMES_WORKDIR
    # (write in addition to the JSON metadata).
    try:
        (shadow_repo / "HERMES_WORKDIR").write_text(
            str(_normalize_path(working_dir)) + "\n", encoding="utf-8"
        )
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """管理自动文件系统检查点。

    设计由 AIAgent 独占管理。请在每个对话轮次开始时
    调用 ``new_turn()``，并在执行任何会修改文件的工具调用前
    调用 ``ensure_checkpoint(dir, reason)``。
    管理器会自动去重，确保每个轮次内每个目录最多只创建一个快照。

    参数
    ----------
    enabled : bool
        主开关（来自配置或 CLI 标志）。
    max_snapshots : int
        每个目录最多保留的检查点数量。
    max_total_size_mb : int
        存储库总容量的硬性上限。提交后若存储库大小超出此限制，
        将按项目删除最老旧的检查点。
    max_file_size_mb : int
        跳过将任何大于此尺寸的单个文件添加到检查点中。
        （通过 ``.gitignore`` 排除规则以及暂存后的文件大小检查来实现。）
    """

    def __init__(
        self,
        enabled: bool = False,
        max_snapshots: int = 20,
        max_total_size_mb: int = 500,
        max_file_size_mb: int = 10,
    ):
        self.enabled = enabled
        self.max_snapshots = max(1, int(max_snapshots))
        self.max_total_size_mb = max(0, int(max_total_size_mb))
        self.max_file_size_mb = max(0, int(max_file_size_mb))
        self._checkpointed_dirs: Set[str] = set()
        self._git_available: Optional[bool] = None  # lazy probe

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------

    def new_turn(self) -> None:
        """Reset per-turn dedup.  Call at the start of each agent iteration."""
        self._checkpointed_dirs.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_checkpoint(self, working_dir: str, reason: str = "auto") -> bool:
        """Take a checkpoint if enabled and not already done this turn.

        Returns True if a checkpoint was taken, False otherwise.
        Never raises — all errors are silently logged.
        """
        if not self.enabled:
            return False

        if self._git_available is None:
            self._git_available = shutil.which("git") is not None
            if not self._git_available:
                logger.debug("Checkpoints disabled: git not found")
        if not self._git_available:
            return False

        abs_dir = str(_normalize_path(working_dir))

        # Skip root, home, and other overly broad directories
        if abs_dir in {"/", str(Path.home())}:
            logger.debug("Checkpoint skipped: directory too broad (%s)", abs_dir)
            return False

        if abs_dir in self._checkpointed_dirs:
            return False

        self._checkpointed_dirs.add(abs_dir)

        try:
            return self._take(abs_dir, reason)
        except Exception as e:
            logger.debug("Checkpoint failed (non-fatal): %s", e)
            return False

    def list_checkpoints(self, working_dir: str) -> List[Dict]:
        """List available checkpoints for a directory (most recent first)."""
        abs_dir = str(_normalize_path(working_dir))
        store = _store_path(CHECKPOINT_BASE)

        if not (store / "HEAD").exists():
            return []

        ref = _ref_name(_project_hash(abs_dir))
        ok, stdout, _ = _run_git(
            ["log", ref, "--format=%H|%h|%aI|%s", "-n", str(self.max_snapshots)],
            store, abs_dir,
            allowed_returncodes={128, 129},
        )

        if not ok or not stdout:
            return []

        results: List[Dict] = []
        for line in stdout.splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                entry = {
                    "hash": parts[0],
                    "short_hash": parts[1],
                    "timestamp": parts[2],
                    "reason": parts[3],
                    "files_changed": 0,
                    "insertions": 0,
                    "deletions": 0,
                }
                stat_ok, stat_out, _ = _run_git(
                    ["diff", "--shortstat", f"{parts[0]}~1", parts[0]],
                    store, abs_dir,
                    allowed_returncodes={128, 129},
                )
                if stat_ok and stat_out:
                    self._parse_shortstat(stat_out, entry)
                results.append(entry)
        return results

    @staticmethod
    def _parse_shortstat(stat_line: str, entry: Dict) -> None:
        """Parse git --shortstat output into entry dict."""
        m = re.search(r'(\d+) file', stat_line)
        if m:
            entry["files_changed"] = int(m.group(1))
        m = re.search(r'(\d+) insertion', stat_line)
        if m:
            entry["insertions"] = int(m.group(1))
        m = re.search(r'(\d+) deletion', stat_line)
        if m:
            entry["deletions"] = int(m.group(1))

    def diff(self, working_dir: str, commit_hash: str) -> Dict:
        """Show diff between a checkpoint and the current working tree."""
        hash_err = _validate_commit_hash(commit_hash)
        if hash_err:
            return {"success": False, "error": hash_err}

        abs_dir = str(_normalize_path(working_dir))
        store = _store_path(CHECKPOINT_BASE)

        if not (store / "HEAD").exists():
            return {"success": False, "error": "No checkpoints exist for this directory"}

        ok, _, err = _run_git(
            ["cat-file", "-t", commit_hash], store, abs_dir,
        )
        if not ok:
            return {"success": False, "error": f"Checkpoint '{commit_hash}' not found"}

        dir_hash = _project_hash(abs_dir)
        index_file = _index_path(store, dir_hash)

        # Stage current state into the per-project index to compare.
        _run_git(["add", "-A"], store, abs_dir,
                 timeout=_GIT_TIMEOUT * 2, index_file=index_file)

        ok_stat, stat_out, _ = _run_git(
            ["diff", "--stat", commit_hash, "--cached"],
            store, abs_dir, index_file=index_file,
        )
        ok_diff, diff_out, _ = _run_git(
            ["diff", commit_hash, "--cached", "--no-color"],
            store, abs_dir, index_file=index_file,
        )

        # Reset staged tree back to the project's last checkpoint so the
        # index doesn't drift out of sync with the ref.
        ref = _ref_name(dir_hash)
        _run_git(["read-tree", ref], store, abs_dir,
                 index_file=index_file,
                 allowed_returncodes={128})

        if not ok_stat and not ok_diff:
            return {"success": False, "error": "Could not generate diff"}

        return {
            "success": True,
            "stat": stat_out if ok_stat else "",
            "diff": diff_out if ok_diff else "",
        }

    def restore(self, working_dir: str, commit_hash: str, file_path: str = None) -> Dict:
        """Restore files to a checkpoint state."""
        hash_err = _validate_commit_hash(commit_hash)
        if hash_err:
            return {"success": False, "error": hash_err}

        abs_dir = str(_normalize_path(working_dir))

        if file_path:
            path_err = _validate_file_path(file_path, abs_dir)
            if path_err:
                return {"success": False, "error": path_err}

        store = _store_path(CHECKPOINT_BASE)

        if not (store / "HEAD").exists():
            return {"success": False, "error": "No checkpoints exist for this directory"}

        ok, _, err = _run_git(
            ["cat-file", "-t", commit_hash], store, abs_dir,
        )
        if not ok:
            return {"success": False, "error": f"Checkpoint '{commit_hash}' not found",
                    "debug": err or None}

        # Take a pre-rollback snapshot so you can undo the undo.
        self._take(abs_dir, f"pre-rollback snapshot (restoring to {commit_hash[:8]})")

        dir_hash = _project_hash(abs_dir)
        index_file = _index_path(store, dir_hash)

        restore_target = file_path if file_path else "."
        ok, stdout, err = _run_git(
            ["checkout", commit_hash, "--", restore_target],
            store, abs_dir, timeout=_GIT_TIMEOUT * 2,
            index_file=index_file,
        )

        if not ok:
            return {"success": False, "error": f"Restore failed: {err}",
                    "debug": err or None}

        ok2, reason_out, _ = _run_git(
            ["log", "--format=%s", "-1", commit_hash], store, abs_dir,
        )
        reason = reason_out if ok2 else "unknown"

        result = {
            "success": True,
            "restored_to": commit_hash[:8],
            "reason": reason,
            "directory": abs_dir,
        }
        if file_path:
            result["file"] = file_path
        return result

    def get_working_dir_for_path(self, file_path: str) -> str:
        """Resolve a file path to its working directory for checkpointing."""
        path = _normalize_path(file_path)
        if path.is_dir():
            candidate = path
        else:
            candidate = path.parent

        markers = {".git", "pyproject.toml", "package.json", "Cargo.toml",
                    "go.mod", "Makefile", "pom.xml", ".hg", "Gemfile"}
        check = candidate
        while check != check.parent:
            if any((check / m).exists() for m in markers):
                return str(check)
            check = check.parent

        return str(candidate)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _take(self, working_dir: str, reason: str) -> bool:
        """Take a snapshot.  Returns True on success."""
        store = _store_path(CHECKPOINT_BASE)

        err = _init_store(store, working_dir)
        if err:
            logger.debug("Checkpoint store init failed: %s", err)
            return False

        _touch_project(store, working_dir)

        # Quick size guard — don't try to snapshot enormous directories
        if _dir_file_count(working_dir) > _MAX_FILES:
            logger.debug("Checkpoint skipped: >%d files in %s", _MAX_FILES, working_dir)
            return False

        dir_hash = _project_hash(working_dir)
        index_file = _index_path(store, dir_hash)
        ref = _ref_name(dir_hash)

        # Seed the per-project index from the last checkpoint, if any, so the
        # diff/commit machinery sees only changes since then.  On first call,
        # clear the index so ``git add -A`` produces a clean tree.
        if index_file.exists():
            # Reset index to current ref tip to avoid accumulating stale paths.
            ok_ref, ref_commit, _ = _run_git(
                ["rev-parse", "--verify", ref + "^{commit}"],
                store, working_dir,
                allowed_returncodes={128},
            )
            if ok_ref and ref_commit:
                _run_git(
                    ["read-tree", ref_commit],
                    store, working_dir,
                    index_file=index_file,
                    allowed_returncodes={128},
                )
            else:
                try:
                    index_file.unlink()
                except OSError:
                    pass
        else:
            # First snapshot for this project.
            index_file.parent.mkdir(parents=True, exist_ok=True)

        # Stage with per-project index.  Include a per-stage file-size filter
        # via ``core.bigFileThreshold`` is not what we want — instead, we
        # rely on the exclude file for broad patterns and post-stage prune
        # any path whose size exceeds max_file_size_mb.
        ok, _, err = _run_git(
            ["add", "-A"], store, working_dir,
            timeout=_GIT_TIMEOUT * 2, index_file=index_file,
        )
        if not ok:
            logger.debug("Checkpoint git-add failed: %s", err)
            return False

        if self.max_file_size_mb > 0:
            self._drop_oversize_from_index(store, working_dir, index_file)

        # Compare against the current ref tip (not HEAD — HEAD points to a
        # branch that doesn't exist on a bare store, so ``diff --cached``
        # against HEAD would always show "new file" for every staged path).
        ok_ref, ref_commit, _ = _run_git(
            ["rev-parse", "--verify", ref + "^{commit}"],
            store, working_dir,
            allowed_returncodes={128},
        )
        has_ref = ok_ref and bool(ref_commit)

        if has_ref:
            ok_diff, _, _ = _run_git(
                ["diff-index", "--cached", "--quiet", ref_commit],
                store, working_dir,
                allowed_returncodes={1},
                index_file=index_file,
            )
            if ok_diff:
                logger.debug("Checkpoint skipped: no changes in %s", working_dir)
                return False
        else:
            # No ref yet — skip only if the index is empty.
            ok_ls, ls_out, _ = _run_git(
                ["ls-files", "--cached"],
                store, working_dir,
                index_file=index_file,
            )
            if ok_ls and not ls_out.strip():
                logger.debug("Checkpoint skipped: empty tree in %s", working_dir)
                return False

        # Write tree from per-project index.
        ok_tree, tree_sha, err = _run_git(
            ["write-tree"], store, working_dir,
            index_file=index_file,
        )
        if not ok_tree or not tree_sha:
            logger.debug("Checkpoint write-tree failed: %s", err)
            return False

        # Build commit (parent = current ref tip, if any).
        commit_args = ["commit-tree", tree_sha, "-m", reason, "--no-gpg-sign"]
        if has_ref:
            commit_args = ["commit-tree", tree_sha, "-p", ref_commit, "-m", reason, "--no-gpg-sign"]
        ok_commit, new_sha, err = _run_git(
            commit_args, store, working_dir,
            index_file=index_file,
        )
        if not ok_commit or not new_sha:
            logger.debug("Checkpoint commit-tree failed: %s", err)
            return False

        # Update the per-project ref.
        update_args = ["update-ref", ref, new_sha]
        if has_ref:
            update_args = ["update-ref", ref, new_sha, ref_commit]
        ok_update, _, err = _run_git(
            update_args, store, working_dir,
        )
        if not ok_update:
            logger.debug("Checkpoint update-ref failed: %s", err)
            return False

        logger.debug("Checkpoint taken in %s: %s (%s)", working_dir, reason, new_sha[:8])

        # Real pruning — drop old commits beyond max_snapshots.
        self._prune(store, working_dir, ref)

        # Enforce global size cap.
        self._enforce_size_cap(store)

        return True

    def _drop_oversize_from_index(
        self, store: Path, working_dir: str, index_file: Path,
    ) -> None:
        """Remove any staged file larger than ``max_file_size_mb`` from the index.

        Lets the agent keep snapshotting source code while refusing to
        swallow generated assets (datasets, model weights, logs, videos).
        """
        cap = self.max_file_size_mb * 1024 * 1024
        if cap <= 0:
            return
        ok, stdout, _ = _run_git(
            ["ls-files", "--cached", "-z"],
            store, working_dir, index_file=index_file,
        )
        if not ok or not stdout:
            return
        # ls-files -z output is NUL-separated. _run_git strips trailing
        # whitespace but that leaves NULs alone; rebuild list.
        paths = [p for p in stdout.split("\x00") if p]
        abs_workdir = _normalize_path(working_dir)
        oversize: List[str] = []
        for rel in paths:
            try:
                size = (abs_workdir / rel).stat().st_size
            except OSError:
                continue
            if size > cap:
                oversize.append(rel)
        if not oversize:
            return
        logger.debug(
            "Checkpoint: dropping %d oversize file(s) (>%d MB) from index",
            len(oversize), self.max_file_size_mb,
        )
        # Use --pathspec-from-file for safety with many paths.
        # Chunk into manageable batches.
        BATCH = 200
        for i in range(0, len(oversize), BATCH):
            chunk = oversize[i:i + BATCH]
            _run_git(
                ["rm", "--cached", "--quiet", "--"] + chunk,
                store, working_dir, index_file=index_file,
                allowed_returncodes={128},
            )

    def _prune(self, store: Path, working_dir: str, ref: str) -> None:
        """Keep only the last ``max_snapshots`` commits on the per-project ref.

        v1's ``_prune`` was documented as a no-op (``git``'s pack mechanism
        was supposed to handle it, but only the log view was limited — loose
        objects accumulated forever).  v2 actually rewrites the ref to drop
        commits older than ``max_snapshots`` and then runs ``git gc`` on the
        store so unreachable objects are reclaimed.
        """
        ok, stdout, _ = _run_git(
            ["rev-list", "--count", ref], store, working_dir,
            allowed_returncodes={128},
        )
        if not ok:
            return
        try:
            count = int(stdout)
        except ValueError:
            return
        if count <= self.max_snapshots:
            return

        # Collect commits oldest → newest, take last N.
        ok_list, list_out, _ = _run_git(
            ["rev-list", "--reverse", ref], store, working_dir,
        )
        if not ok_list or not list_out:
            return
        commits = list_out.splitlines()
        keep = commits[-self.max_snapshots:]

        # Rebuild a linear chain off keep[0]'s tree.
        new_parent: Optional[str] = None
        for sha in keep:
            ok_tree, tree_sha, _ = _run_git(
                ["rev-parse", f"{sha}^{{tree}}"], store, working_dir,
            )
            if not ok_tree or not tree_sha:
                return
            ok_msg, msg, _ = _run_git(
                ["log", "--format=%s", "-1", sha], store, working_dir,
            )
            commit_msg = msg if ok_msg and msg else "checkpoint"
            args = ["commit-tree", tree_sha, "-m", commit_msg, "--no-gpg-sign"]
            if new_parent is not None:
                args = ["commit-tree", tree_sha, "-p", new_parent,
                        "-m", commit_msg, "--no-gpg-sign"]
            ok_commit, new_sha, _ = _run_git(args, store, working_dir)
            if not ok_commit or not new_sha:
                return
            new_parent = new_sha

        if new_parent is None:
            return
        _run_git(["update-ref", ref, new_parent], store, working_dir)

        # Reclaim objects from the dropped commits.
        _run_git(
            ["reflog", "expire", "--expire=now", "--all"],
            store, working_dir,
        )
        _run_git(
            ["gc", "--prune=now", "--quiet"],
            store, working_dir, timeout=_GIT_TIMEOUT * 3,
        )
        _repair_bare_repo_dirs(store)

    def _enforce_size_cap(self, store: Path) -> None:
        """If total store size exceeds ``max_total_size_mb``, drop oldest
        checkpoints across ALL projects until under the cap.
        """
        if self.max_total_size_mb <= 0:
            return
        cap_bytes = self.max_total_size_mb * 1024 * 1024
        size = _dir_size_bytes(store)
        if size <= cap_bytes:
            return
        logger.info(
            "Checkpoint store exceeded %d MB (actual %d MB) — pruning oldest",
            self.max_total_size_mb, size // (1024 * 1024),
        )

        # Collect (commit_time, ref, sha) across all per-project refs.
        ok, stdout, _ = _run_git(
            ["for-each-ref", "--format=%(refname)", _REFS_PREFIX],
            store, str(store.parent),
            allowed_returncodes={128},
        )
        if not ok or not stdout:
            return
        refs = [r for r in stdout.splitlines() if r.strip()]

        any_dropped = False
        # Round-robin-drop oldest commit per ref until under cap.
        for _ in range(20):  # hard upper bound to avoid pathological loops
            size = _dir_size_bytes(store)
            if size <= cap_bytes:
                break
            for ref in refs:
                ok_count, count_out, _ = _run_git(
                    ["rev-list", "--count", ref], store, str(store.parent),
                    allowed_returncodes={128},
                )
                try:
                    count = int(count_out) if ok_count else 0
                except ValueError:
                    count = 0
                if count <= 1:
                    continue  # keep at least one snapshot per project
                ok_list, list_out, _ = _run_git(
                    ["rev-list", "--reverse", ref], store, str(store.parent),
                )
                if not ok_list or not list_out:
                    continue
                commits = list_out.splitlines()
                keep = commits[1:]  # drop oldest
                new_parent: Optional[str] = None
                fail = False
                for sha in keep:
                    ok_tree, tree_sha, _ = _run_git(
                        ["rev-parse", f"{sha}^{{tree}}"], store, str(store.parent),
                    )
                    if not ok_tree or not tree_sha:
                        fail = True
                        break
                    ok_msg, msg, _ = _run_git(
                        ["log", "--format=%s", "-1", sha], store, str(store.parent),
                    )
                    commit_msg = msg if ok_msg and msg else "checkpoint"
                    args = ["commit-tree", tree_sha, "-m", commit_msg, "--no-gpg-sign"]
                    if new_parent is not None:
                        args = ["commit-tree", tree_sha, "-p", new_parent,
                                "-m", commit_msg, "--no-gpg-sign"]
                    ok_commit, new_sha, _ = _run_git(args, store, str(store.parent))
                    if not ok_commit or not new_sha:
                        fail = True
                        break
                    new_parent = new_sha
                if fail or new_parent is None:
                    continue
                _run_git(["update-ref", ref, new_parent], store, str(store.parent))
                any_dropped = True
            if not any_dropped:
                break

        _run_git(
            ["reflog", "expire", "--expire=now", "--all"],
            store, str(store.parent),
        )
        _run_git(
            ["gc", "--prune=now", "--quiet"],
            store, str(store.parent), timeout=_GIT_TIMEOUT * 3,
        )
        _repair_bare_repo_dirs(store)


def format_checkpoint_list(checkpoints: List[Dict], directory: str) -> str:
    """Format checkpoint list for display to user."""
    if not checkpoints:
        return f"No checkpoints found for {directory}"

    lines = [f"📸 Checkpoints for {directory}:\n"]
    for i, cp in enumerate(checkpoints, 1):
        ts = cp["timestamp"]
        if "T" in ts:
            ts = ts.split("T")[1].split("+")[0].split("-")[0][:5]
            date = cp["timestamp"].split("T")[0]
            ts = f"{date} {ts}"

        files = cp.get("files_changed", 0)
        ins = cp.get("insertions", 0)
        dele = cp.get("deletions", 0)
        if files:
            stat = f"  ({files} file{'s' if files != 1 else ''}, +{ins}/-{dele})"
        else:
            stat = ""

        lines.append(f"  {i}. {cp['short_hash']}  {ts}  {cp['reason']}{stat}")

    lines.append("\n  /rollback <N>             restore to checkpoint N")
    lines.append("  /rollback diff <N>        preview changes since checkpoint N")
    lines.append("  /rollback <N> <file>      restore a single file from checkpoint N")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-maintenance
# ---------------------------------------------------------------------------
#
# v2 rewrite.  The sweep now operates on per-project refs inside the shared
# store rather than per-project shadow repos.  Legacy-archive dirs
# (``legacy-<ts>/``) are swept with the same retention policy.

_PRUNE_MARKER_NAME = ".last_prune"


def _delete_ref(store: Path, ref: str) -> bool:
    """Delete a ref from the store.  Returns True on success."""
    ok, _, _ = _run_git(
        ["update-ref", "-d", ref], store, str(store.parent),
        allowed_returncodes={128},
    )
    return ok


def prune_checkpoints(
    retention_days: int = 7,
    delete_orphans: bool = True,
    checkpoint_base: Optional[Path] = None,
    max_total_size_mb: int = 0,
) -> Dict[str, int]:
    """Delete stale/orphan checkpoints and reclaim store space.

    A project entry is deleted when either:

    * ``delete_orphans=True`` and its ``workdir`` no longer exists on disk
      (the original project was deleted / moved); OR
    * its ``last_touch`` is older than ``retention_days`` days.

    Additionally, if ``max_total_size_mb > 0`` and the store exceeds that
    after orphan/stale pruning, the oldest commit per remaining project is
    dropped until the store is under the cap.

    Legacy-archive dirs (``legacy-*``) older than ``retention_days`` are
    also deleted.

    Returns a dict with counts ``{"scanned", "deleted_orphan",
    "deleted_stale", "errors", "bytes_freed"}``.

    Never raises — maintenance must never block interactive startup.
    """
    base = checkpoint_base or CHECKPOINT_BASE
    result = {
        "scanned": 0,
        "deleted_orphan": 0,
        "deleted_stale": 0,
        "errors": 0,
        "bytes_freed": 0,
    }
    if not base.exists():
        return result

    size_before = _dir_size_bytes(base)

    # --- Legacy pre-v2 per-project shadow repos (kept directly under base) ---
    # Pre-v2 layout: ``base/<hash>/HEAD`` etc.  We treat these exactly as the
    # v1 pruner did so behaviour is unchanged for anyone still on that layout
    # or sitting on a mid-migration system.
    cutoff = 0.0
    if retention_days > 0:
        cutoff = time.time() - retention_days * 86400

    for child in base.iterdir():
        if not child.is_dir():
            continue
        if child.name == _STORE_DIRNAME:
            continue
        if child.name.startswith(_LEGACY_PREFIX):
            # Legacy archive: prune by dir mtime using same retention rule.
            if retention_days <= 0:
                continue
            try:
                m = child.stat().st_mtime
            except OSError:
                continue
            if m >= cutoff:
                continue
            try:
                size = _dir_size_bytes(child)
                shutil.rmtree(child)
                result["bytes_freed"] += size
                result["deleted_stale"] += 1
            except OSError as exc:
                result["errors"] += 1
                logger.warning("Failed to delete legacy archive %s: %s", child, exc)
            continue
        # Only count as a pre-v2 shadow repo if it has a HEAD.
        if not (child / "HEAD").exists():
            continue
        result["scanned"] += 1
        reason: Optional[str] = None
        if delete_orphans:
            workdir: Optional[str] = None
            wd_marker = child / "HERMES_WORKDIR"
            if wd_marker.exists():
                try:
                    workdir = wd_marker.read_text(encoding="utf-8").strip()
                except (OSError, UnicodeDecodeError):
                    workdir = None
            if workdir is None or not Path(workdir).exists():
                reason = "orphan"
        if reason is None and retention_days > 0:
            newest = 0.0
            try:
                for p in child.rglob("*"):
                    try:
                        mt = p.stat().st_mtime
                        newest = max(newest, mt)
                    except OSError:
                        continue
            except OSError:
                pass
            if newest > 0 and newest < cutoff:
                reason = "stale"
        if reason is None:
            continue
        try:
            size = _dir_size_bytes(child)
            shutil.rmtree(child)
            result["bytes_freed"] += size
            if reason == "orphan":
                result["deleted_orphan"] += 1
            else:
                result["deleted_stale"] += 1
        except OSError as exc:
            result["errors"] += 1
            logger.warning("Failed to prune checkpoint repo %s: %s", child.name, exc)

    # --- v2 shared store: per-project ref pruning via metadata ---
    store = _store_path(base)
    if (store / "HEAD").exists():
        for meta in _list_projects(store):
            dir_hash = meta.get("_hash") or ""
            workdir = meta.get("workdir") or ""
            if not dir_hash:
                continue
            result["scanned"] += 1
            reason = None
            if delete_orphans and (not workdir or not Path(workdir).exists()):
                reason = "orphan"
            elif retention_days > 0:
                last_touch = float(meta.get("last_touch", 0) or 0)
                if last_touch > 0 and last_touch < cutoff:
                    reason = "stale"
            if reason is None:
                continue
            ref = _ref_name(dir_hash)
            _delete_ref(store, ref)
            # Drop per-project index and metadata.
            try:
                idx = _index_path(store, dir_hash)
                if idx.exists():
                    idx.unlink()
            except OSError:
                pass
            try:
                mp = _project_meta_path(store, dir_hash)
                if mp.exists():
                    mp.unlink()
            except OSError:
                pass
            if reason == "orphan":
                result["deleted_orphan"] += 1
            else:
                result["deleted_stale"] += 1

        # GC the store to reclaim unreachable objects from dropped refs.
        _run_git(
            ["reflog", "expire", "--expire=now", "--all"],
            store, str(base),
        )
        _run_git(
            ["gc", "--prune=now", "--quiet"],
            store, str(base), timeout=_GIT_TIMEOUT * 3,
        )
        _repair_bare_repo_dirs(store)

        # Size-cap pass across remaining projects.
        if max_total_size_mb > 0:
            cap_bytes = max_total_size_mb * 1024 * 1024
            for _i in range(20):
                size = _dir_size_bytes(store)
                if size <= cap_bytes:
                    break
                ok, stdout, _ = _run_git(
                    ["for-each-ref", "--format=%(refname)", _REFS_PREFIX],
                    store, str(base),
                    allowed_returncodes={128},
                )
                refs = [r for r in stdout.splitlines() if r.strip()] if ok else []
                if not refs:
                    break
                any_drop = False
                for ref in refs:
                    ok_c, count_out, _ = _run_git(
                        ["rev-list", "--count", ref], store, str(base),
                        allowed_returncodes={128},
                    )
                    try:
                        count = int(count_out) if ok_c else 0
                    except ValueError:
                        count = 0
                    if count <= 1:
                        continue
                    ok_l, lo, _ = _run_git(
                        ["rev-list", "--reverse", ref], store, str(base),
                    )
                    if not ok_l or not lo:
                        continue
                    commits = lo.splitlines()
                    keep = commits[1:]
                    new_parent: Optional[str] = None
                    fail = False
                    for sha in keep:
                        ok_t, tsha, _ = _run_git(
                            ["rev-parse", f"{sha}^{{tree}}"], store, str(base),
                        )
                        if not ok_t or not tsha:
                            fail = True
                            break
                        ok_m, m, _ = _run_git(
                            ["log", "--format=%s", "-1", sha], store, str(base),
                        )
                        msg = m if ok_m and m else "checkpoint"
                        args = ["commit-tree", tsha, "-m", msg, "--no-gpg-sign"]
                        if new_parent is not None:
                            args = ["commit-tree", tsha, "-p", new_parent,
                                    "-m", msg, "--no-gpg-sign"]
                        ok_cm, new_sha, _ = _run_git(args, store, str(base))
                        if not ok_cm or not new_sha:
                            fail = True
                            break
                        new_parent = new_sha
                    if fail or new_parent is None:
                        continue
                    _run_git(["update-ref", ref, new_parent], store, str(base))
                    any_drop = True
                if not any_drop:
                    break
            _run_git(
                ["reflog", "expire", "--expire=now", "--all"],
                store, str(base),
            )
            _run_git(
                ["gc", "--prune=now", "--quiet"],
                store, str(base), timeout=_GIT_TIMEOUT * 3,
            )
            _repair_bare_repo_dirs(store)

    size_after = _dir_size_bytes(base)
    delta = size_before - size_after
    result["bytes_freed"] = max(result["bytes_freed"], delta)

    return result


def maybe_auto_prune_checkpoints(
    retention_days: int = 7,
    min_interval_hours: int = 24,
    delete_orphans: bool = True,
    checkpoint_base: Optional[Path] = None,
    max_total_size_mb: int = 0,
) -> Dict[str, object]:
    """Idempotent wrapper around ``prune_checkpoints`` for startup hooks.

    Writes ``CHECKPOINT_BASE/.last_prune`` on completion so subsequent
    calls within ``min_interval_hours`` short-circuit.

    Returns ``{"skipped": bool, "result": prune_checkpoints-dict,
    "error": optional str}``.
    """
    base = checkpoint_base or CHECKPOINT_BASE
    out: Dict[str, object] = {"skipped": False}

    try:
        if not base.exists():
            out["result"] = {
                "scanned": 0, "deleted_orphan": 0, "deleted_stale": 0,
                "errors": 0, "bytes_freed": 0,
            }
            return out

        marker = base / _PRUNE_MARKER_NAME
        now = time.time()
        if marker.exists():
            try:
                last_ts = float(marker.read_text(encoding="utf-8").strip())
                if now - last_ts < min_interval_hours * 3600:
                    out["skipped"] = True
                    return out
            except (OSError, ValueError):
                pass  # corrupt marker — treat as no prior run

        result = prune_checkpoints(
            retention_days=retention_days,
            delete_orphans=delete_orphans,
            checkpoint_base=base,
            max_total_size_mb=max_total_size_mb,
        )
        out["result"] = result

        try:
            marker.write_text(str(now), encoding="utf-8")
        except OSError as exc:
            logger.debug("Could not write checkpoint prune marker: %s", exc)

        total = result["deleted_orphan"] + result["deleted_stale"]
        if total > 0:
            logger.info(
                "checkpoint auto-maintenance: pruned %d entry(ies) "
                "(%d orphan, %d stale), reclaimed %.1f MB",
                total,
                result["deleted_orphan"],
                result["deleted_stale"],
                result["bytes_freed"] / (1024 * 1024),
            )
    except Exception as exc:
        logger.warning("checkpoint auto-maintenance failed: %s", exc)
        out["error"] = str(exc)

    return out


# ---------------------------------------------------------------------------
# Public helpers for `hermes checkpoints` CLI
# ---------------------------------------------------------------------------

def store_status(checkpoint_base: Optional[Path] = None) -> Dict:
    """Return a summary of the shadow store.

    ``{"base": path, "store_size_bytes": N, "legacy_size_bytes": N,
       "total_size_bytes": N, "project_count": N, "projects": [...],
       "legacy_archives": [...]}``
    """
    base = checkpoint_base or CHECKPOINT_BASE
    out: Dict = {
        "base": str(base),
        "store_size_bytes": 0,
        "legacy_size_bytes": 0,
        "total_size_bytes": 0,
        "project_count": 0,
        "projects": [],
        "legacy_archives": [],
    }
    if not base.exists():
        return out

    store = _store_path(base)
    if store.exists():
        out["store_size_bytes"] = _dir_size_bytes(store)
        if (store / "HEAD").exists():
            for meta in _list_projects(store):
                dir_hash = meta.get("_hash") or ""
                workdir = meta.get("workdir") or ""
                ref = _ref_name(dir_hash)
                ok, count_out, _ = _run_git(
                    ["rev-list", "--count", ref], store, str(base),
                    allowed_returncodes={128},
                )
                try:
                    commits = int(count_out) if ok else 0
                except ValueError:
                    commits = 0
                out["projects"].append({
                    "hash": dir_hash,
                    "workdir": workdir,
                    "exists": bool(workdir) and Path(workdir).exists(),
                    "created_at": meta.get("created_at"),
                    "last_touch": meta.get("last_touch"),
                    "commits": commits,
                })
    out["project_count"] = len(out["projects"])

    for child in base.iterdir():
        if child.is_dir() and child.name.startswith(_LEGACY_PREFIX):
            try:
                size = _dir_size_bytes(child)
            except OSError:
                size = 0
            out["legacy_size_bytes"] += size
            try:
                mt = child.stat().st_mtime
            except OSError:
                mt = 0
            out["legacy_archives"].append({
                "name": child.name,
                "size_bytes": size,
                "mtime": mt,
            })

    out["total_size_bytes"] = _dir_size_bytes(base)
    return out


def clear_all(checkpoint_base: Optional[Path] = None) -> Dict[str, int]:
    """Nuke the entire checkpoint base (store + legacy).  Irreversible.

    Returns ``{"bytes_freed": N, "deleted": bool}``.
    """
    base = checkpoint_base or CHECKPOINT_BASE
    out = {"bytes_freed": 0, "deleted": False}
    if not base.exists():
        return out
    size = _dir_size_bytes(base)
    try:
        shutil.rmtree(base)
        out["bytes_freed"] = size
        out["deleted"] = True
    except OSError as exc:
        logger.warning("Could not clear checkpoint base %s: %s", base, exc)
    return out


def clear_legacy(checkpoint_base: Optional[Path] = None) -> Dict[str, int]:
    """Delete all ``legacy-*`` archive directories.

    Returns ``{"bytes_freed": N, "deleted": count}``.
    """
    base = checkpoint_base or CHECKPOINT_BASE
    out = {"bytes_freed": 0, "deleted": 0}
    if not base.exists():
        return out
    for child in list(base.iterdir()):
        if not child.is_dir() or not child.name.startswith(_LEGACY_PREFIX):
            continue
        try:
            size = _dir_size_bytes(child)
            shutil.rmtree(child)
            out["bytes_freed"] += size
            out["deleted"] += 1
        except OSError as exc:
            logger.warning("Could not delete legacy archive %s: %s", child, exc)
    return out
