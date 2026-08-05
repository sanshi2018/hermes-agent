"""Curator — 后台 Skill 维护编排器。

Curator 是一个使用辅助模型（Auxiliary Model）的后台任务，
负责定期审查由 Agent 创建的 Skill 并对其集合进行维护。
它通过“空闲触发”机制运行（无需 Cron 守护进程）：
当 Agent 处于空闲状态，且距离上一次 Curator 运行的时间
超过了 ``interval_hours`` 时，``maybe_run_curator()`` 会
派生（Fork）一个 AIAgent 来执行审查。

主要职责：
  - 根据派生的 Skill 活动时间戳，自动过渡生命周期状态
  - 衍生后台审查 Agent，该 Agent 可通过 skill_manage 工具
    对 Agent 创建的 Skill 执行固定（Pin）、归档（Archive）、合并（Consolidate）或补丁（Patch）操作
  - 在 .curator_state 中持久化保存 Curator 状态（如 last_run_at、paused 等）

严格的不变性约束：
  - 仅处理由 Agent 创建的 Skill（参见 tools/skill_usage.is_agent_created）
  - 绝不自动删除 — 仅进行归档。归档是可恢复的
  - 被固定的 Skill 会跳过所有自动状态过渡
  - 使用辅助客户端；绝不触碰主会话（Main Session）的 Prompt 缓存
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set

from hermes_constants import get_hermes_home
from tools import skill_usage
from utils import atomic_json_write

logger = logging.getLogger(__name__)


def _strip_aux_credential(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class _ReviewRuntimeBinding(NamedTuple):
    """Provider/model for the curator review fork plus per-slot overrides."""

    provider: str
    model: str
    explicit_api_key: Optional[str]
    explicit_base_url: Optional[str]
    request_overrides: Dict[str, Any]


def _merge_request_overrides(
    runtime_overrides: Any,
    slot_extra_body: Any,
) -> Dict[str, Any]:
    """Merge resolver metadata with task-local request body fields."""
    merged = dict(runtime_overrides or {})
    if isinstance(slot_extra_body, dict) and slot_extra_body:
        extra_body = dict(merged.get("extra_body") or {})
        extra_body.update(slot_extra_body)
        merged["extra_body"] = extra_body
    return merged


DEFAULT_INTERVAL_HOURS = 24 * 7  # 7 days
DEFAULT_MIN_IDLE_HOURS = 2
DEFAULT_STALE_AFTER_DAYS = 30
DEFAULT_ARCHIVE_AFTER_DAYS = 90
# Consolidation (the LLM umbrella-building fork) is OFF by default. The
# deterministic inactivity prune (apply_automatic_transitions) still runs
# whenever the curator is enabled; only the opinionated, aux-model-cost
# consolidation pass is opt-in.
DEFAULT_CONSOLIDATE = False


# ---------------------------------------------------------------------------
# .curator_state — persistent scheduler + status
# ---------------------------------------------------------------------------

def _state_file() -> Path:
    return get_hermes_home() / "skills" / ".curator_state"


def _default_state() -> Dict[str, Any]:
    return {
        "last_run_at": None,
        "last_run_duration_seconds": None,
        "last_run_summary": None,
        "last_run_summary_shown_at": None,
        "last_report_path": None,
        "paused": False,
        "run_count": 0,
    }


def load_state() -> Dict[str, Any]:
    path = _state_file()
    if not path.exists():
        return _default_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            base = _default_state()
            base.update({k: v for k, v in data.items() if k in base or k.startswith("_")})
            return base
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to read curator state: %s", e)
    return _default_state()


def save_state(data: Dict[str, Any]) -> None:
    path = _state_file()
    try:
        atomic_json_write(path, data, indent=2, sort_keys=True)
    except Exception as e:
        logger.debug("Failed to save curator state: %s", e, exc_info=True)


def set_paused(paused: bool) -> None:
    state = load_state()
    state["paused"] = bool(paused)
    save_state(state)


def is_paused() -> bool:
    return bool(load_state().get("paused"))


# ---------------------------------------------------------------------------
# Config access
# ---------------------------------------------------------------------------

def _load_config() -> Dict[str, Any]:
    """Read curator.* config from ~/.hermes/config.yaml. Tolerates missing file."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception as e:
        logger.debug("Failed to load config for curator: %s", e)
        return {}
    if not isinstance(cfg, dict):
        return {}
    cur = cfg.get("curator") or {}
    if not isinstance(cur, dict):
        return {}
    return cur


def is_enabled() -> bool:
    """Default ON when no config says otherwise."""
    cfg = _load_config()
    return bool(cfg.get("enabled", True))


def get_interval_hours() -> int:
    cfg = _load_config()
    try:
        return int(cfg.get("interval_hours", DEFAULT_INTERVAL_HOURS))
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL_HOURS


def get_min_idle_hours() -> float:
    cfg = _load_config()
    try:
        return float(cfg.get("min_idle_hours", DEFAULT_MIN_IDLE_HOURS))
    except (TypeError, ValueError):
        return DEFAULT_MIN_IDLE_HOURS


def get_stale_after_days() -> int:
    cfg = _load_config()
    try:
        return int(cfg.get("stale_after_days", DEFAULT_STALE_AFTER_DAYS))
    except (TypeError, ValueError):
        return DEFAULT_STALE_AFTER_DAYS


def get_archive_after_days() -> int:
    cfg = _load_config()
    try:
        return int(cfg.get("archive_after_days", DEFAULT_ARCHIVE_AFTER_DAYS))
    except (TypeError, ValueError):
        return DEFAULT_ARCHIVE_AFTER_DAYS


def get_prune_builtins() -> bool:
    """Curator 是否也可以清理（归档）打包自带的内置 Skill。

    默认开启（ON）。开启时，内置 Skill 将成为维护候选对象，
    并在经历与 Agent 创建的 Skill 相同的空闲期后被归档，
    同时通过抑制列表（Suppression List）确保它们在执行 `hermes update` 重新填充时
    依然保持归档状态。
    无论此标志如何设置，通过 Hub 安装的 Skill 均绝不会被清理。
    """
    cfg = _load_config()
    return bool(cfg.get("prune_builtins", True))


def get_consolidate() -> bool:
    """是否让 Curator 运行其 LLM 合并（框架构建）流程。

    默认关闭（OFF）。关闭时，Curator 仅执行确定性的空闲清理
    （标记过期 / 归档长期未使用的 Skill），
    并完全跳过派生的辅助模型审查 ——
    不进行合并，不构建框架，亦无辅助模型成本。
    将 ``curator.consolidate: true`` 设置为 true，
    可重新启用将重叠 Skill 合并为类级别框架的 LLM 流程。

    无论配置值如何，显式使用 ``hermes curator run --consolidate`` 标志
    均可在单次调用中覆盖此选项。
    """
    cfg = _load_config()
    return bool(cfg.get("consolidate", DEFAULT_CONSOLIDATE))


# ---------------------------------------------------------------------------
# Idle / interval check
# ---------------------------------------------------------------------------

def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def should_run_now(now: Optional[datetime] = None) -> bool:
    """如果 Curator 应当立即运行，则返回 True。

    关卡门控条件：
      - curator.enabled == True
      - 未暂停（not paused）
      - 存在 last_run_at，且其时间早于指定的 interval_hours

    首次运行行为：当不存在 ``last_run_at`` 时
    （例如全新安装，或先于 Curator 版本安装的情况），
    我们**不会**立即运行。
    Curator 的设计初衷是在 Skill 产生至少 ``interval_hours``
    （默认为 7 天）的活动之后才运行，
    而不是在执行 ``hermes update`` 后的首次后台 Tick 时立即运行。
    在首次检测时，我们会将 ``last_run_at`` 初始化为“当前时间”，
    并将第一次真正的审查推迟整整一个周期。
    如果用户希望尽早运行，可以随时显式调用
    ``hermes curator run``（无论是否带 ``--dry-run`` 参数）——
    该路径会绕过这里的门控限制。

    空闲检测（min_idle_hours）会在能够获取 Agent
    是否正在活跃运行的调用方（Call Site）处执行 ——
    在此处，我们仅强制校验静态门控条件。
    """
    if not is_enabled():
        return False
    if is_paused():
        return False

    state = load_state()
    last = _parse_iso(state.get("last_run_at"))
    if last is None:
        # Never run before. Seed state so we wait a full interval before the
        # first real pass. Report-only; do not auto-mutate the library the
        # very first time a gateway ticks after an update.
        if now is None:
            now = datetime.now(timezone.utc)
        try:
            state["last_run_at"] = now.isoformat()
            state["last_run_summary"] = (
                "deferred first run — curator seeded, will run after one "
                "interval; use `hermes curator run --dry-run` to preview now"
            )
            save_state(state)
        except Exception as e:  # pragma: no cover — best-effort persistence
            logger.debug("Failed to seed curator last_run_at: %s", e)
        return False

    if now is None:
        now = datetime.now(timezone.utc)
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    interval = timedelta(hours=get_interval_hours())
    return (now - last) >= interval


# ---------------------------------------------------------------------------
# Automatic state transitions (pure function, no LLM)
# ---------------------------------------------------------------------------

def _cron_referenced_skills() -> Set[str]:
    """任何 Cron 任务（包括已暂停/已禁用的任务）所引用的 Skill 名称列表。

    尽力而为（Best-effort）：Cron 模块导入错误或损坏的 Jobs 存储库
    绝不能导致 Curator 崩溃，
    因此任何异常都会返回一个空集合
    （即不进行特殊保护，但能保证系统不崩溃）。
    """
    try:
        from cron.jobs import referenced_skill_names as _refs
        return _refs()
    except Exception as e:
        logger.debug("Curator could not read cron skill references: %s", e, exc_info=True)
        return set()


def apply_automatic_transitions(now: Optional[datetime] = None) -> Dict[str, int]:
    """遍历每个由 Curator 管理的 Skill，并根据其最新的实际活动时间戳
    在 active（活跃）、stale（陈旧）和 archived（归档）状态之间进行切换过渡。
    被固定（Pinned）的 Skill 绝不会被触碰变动。

    内置 Skill（仅在启用 ``curator.prune_builtins`` 时才符合处理条件）
    会在首次发现时初始化并写入一条基线记录，
    以便其空闲计时器从“此刻”开始计算，而不是从 Unix 纪元（Epoch）开始 ——
    因此，长期未使用的内置 Skill 只有在经历了完整的 ``archive_after_days``
    未使用的天数后才会被归档，而不会在配置开关刚开启后的首次审查流程中就被立即归档。

    返回一个描述变更情况的 Counter 字典。
    """
    from tools import skill_usage as _u

    if now is None:
        now = datetime.now(timezone.utc)
    stale_cutoff = now - timedelta(days=get_stale_after_days())
    archive_cutoff = now - timedelta(days=get_archive_after_days())

    cron_referenced = _cron_referenced_skills()

    counts = {"marked_stale": 0, "archived": 0, "reactivated": 0, "checked": 0, "seeded": 0}

    for row in _u.agent_created_report():
        counts["checked"] += 1
        name = row["name"]
        if row.get("pinned"):
            continue

        # 只要被任意 Cron 任务（包括已暂停/已禁用的任务）引用的 Skill，
        # 按定义均视为“在使用中” —— 因为恢复运行或下一次触发时必须确保其存在。
        # 调度器仅在任务实际触发时才会更新（Bump）使用率，
        # 因此对于触发频率低于 archive_after_days 的任务、已暂停的任务，
        # 以及执行时间久远的单次（One-shot）任务，若不加以保护，
        # 其依赖的 Skill 就会在后台自动因过期而被移除。
        # 因此，请将受引用的 Skill 视为与“固定（Pinned）”相同：绝不进行自动状态过渡。
        if name in cron_referenced:
            continue

        # 首次发现符合 Curator 管理条件但无持久化记录的 Skill
        # （例如新纳入管理范围的内置 Skill）：
        # 将其计时器锚定为当前时间，并推迟后续处理。
        if not row.get("_persisted", True):
            _u.seed_record_if_missing(name)
            counts["seeded"] += 1
            continue

        last_activity = _parse_iso(row.get("last_activity_at"))
        # 若从未活跃过，则将 created_at 作为时间锚点，
        # 以防止新创建的 Skill 被立即自动归档。
        anchor = last_activity or _parse_iso(row.get("created_at")) or now
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)

        current = row.get("state", _u.STATE_ACTIVE)

        # 从未使用过的 Skill（use_count == 0）将享有宽限期：
        # 在其年龄至少达到 stale_after_days 之前，不会对其进行归档。
        # 一个 use=0 的 Skill 仅代表“缺乏使用证据”，并不等同于“已陈旧” ——
        # 新创建的 Skill 可能只是尚未遇到触发它的场景而已。
        never_used = int(row.get("use_count", 0) or 0) == 0
        if never_used and anchor > stale_cutoff:
            # Younger than the stale window — leave it alone entirely.
            if current == _u.STATE_STALE:
                _u.set_state(name, _u.STATE_ACTIVE)
                counts["reactivated"] += 1
            continue

        if anchor <= archive_cutoff and current != _u.STATE_ARCHIVED:
            ok, _msg = _u.archive_skill(name)
            if ok:
                counts["archived"] += 1
        elif anchor <= stale_cutoff and current == _u.STATE_ACTIVE:
            _u.set_state(name, _u.STATE_STALE)
            counts["marked_stale"] += 1
        elif anchor > stale_cutoff and current == _u.STATE_STALE:
            # Skill got used again after being marked stale — reactivate.
            _u.set_state(name, _u.STATE_ACTIVE)
            counts["reactivated"] += 1

    return counts


# ---------------------------------------------------------------------------
# Review prompt for the forked agent
# ---------------------------------------------------------------------------
# CURATOR_DRY_RUN_BANNER = (
#     "═══════════════════════════════════════════════════════════════\n"
#     "预演模式（DRY-RUN）— 仅生成报告。请勿修改技能库（SKILL LIBRARY）。\n"
#     "═══════════════════════════════════════════════════════════════\n"
#     "\n"
#     "这是一个【预览】环节。请遵循以下所有指令，但【排除】以下操作：\n"
#     "\n"
#     "  • 请勿调用 skill_manage 并传入 action=patch, create, delete, "
#     "write_file 或 remove_file。\n"
#     "  • 请勿调用 terminal 将技能目录移动（mv）至 .archive/ 中。\n"
#     "  • 请勿调用 terminal 去移动（mv）、复制（cp）、删除（rm）或重写 "
#     "~/.hermes/skills/ 下的任何文件。\n"
#     "  • skills_list 与 skill_view 可以正常使用 — 尽情读取你需要的内容。\n"
#     "\n"
#     "你的输出【就是】最终交付物。请生成与实际运行（live run）完全一致的"
#     "可读摘要和结构化 YAML 块 — 但请描述你【打算】采取的操作，"
#     "而非【已执行】的操作。后续的审核人员将阅读此报告，"
#     "并决定是否批准使用 `hermes curator run`（不带标记）进行实际运行。\n"
#     "\n"
#     "如果你意外执行了任何修改类操作，请在摘要中明确说明，"
#     "以便审核人员能够进行还原。\n"
#     "═══════════════════════════════════════════════════════════════"
# )
CURATOR_DRY_RUN_BANNER = (
    "═══════════════════════════════════════════════════════════════\n"
    "DRY-RUN — REPORT ONLY. DO NOT MUTATE THE SKILL LIBRARY.\n"
    "═══════════════════════════════════════════════════════════════\n"
    "\n"
    "This is a PREVIEW pass. Follow every instruction below EXCEPT:\n"
    "\n"
    "  • DO NOT call skill_manage with action=patch, create, delete, "
    "write_file, or remove_file.\n"
    "  • DO NOT call terminal to mv skill directories into .archive/.\n"
    "  • DO NOT call terminal to mv, cp, rm, or rewrite any file under "
    "~/.hermes/skills/.\n"
    "  • skills_list and skill_view are FINE — read as much as you need.\n"
    "\n"
    "Your output IS the deliverable. Produce the exact same "
    "human-readable summary and structured YAML block you would "
    "produce on a live run — but describe the actions you WOULD take, "
    "not actions you took. A downstream reviewer will read the report "
    "and decide whether to approve a live run with "
    "`hermes curator run` (no flag).\n"
    "\n"
    "If you accidentally take a mutating action, say so explicitly in "
    "the summary so the reviewer can revert it.\n"
    "═══════════════════════════════════════════════════════════════"
)
#
# CURATOR_REVIEW_PROMPT = (
#     "你正在作为 Hermes 的后台 Skill 维护者（CURATOR）运行。"
#     "这是一次**框架构建（UMBRELLA-BUILDING）**合并审查，而不是被动审计，也不是重复项查找器。\n\n"
#     "Skill 集合的目标是建立一个**类级别（CLASS-LEVEL）指令和经验知识的库**。"
#     "包含几百个狭隘 Skill（其中每个只捕获单次会话的特定 Bug）的集合是该库的**失败**，而不是特性。"
#     "Agent 搜索 Skill 时是通过描述进行匹配，而不是通过精确名称匹配；"
#     "一个带有清晰标注小节的广义框架（Umbrella）Skill，在可发现性上远胜于五个狭隘的同类 Skill，而不是相反。\n\n"
#     "正确的目标形态是带有丰富 SKILL.md 主体，
#     以及用于存放会话特定细节的 `references/`、`templates/`
#     和 `scripts/` 子文件的**类级别 Skill** —— 而不是“一次会话一个 Skill”的微型条目。\n\n"
#     "硬性规则 —— 切勿违反：\n"
#     "1. 切勿触碰打包自带（bundled）、Hub 安装或外部目录（`skills.external_dirs`）中的 Skill。
#     下方的候选列表已经过滤为仅包含本地 Curator 管理的 Skill；
#     外部 Skill 归外部所有，对本后台 Curator 而言是只读的。\n"
#     "2. 切勿删除任何 Skill。归档（将 Skill 目录移动到 ~/.hermes/skills/.archive/）
#     是最大的破坏性操作。归档是可恢复的；删除不可恢复。\n"
#     "3. 切勿触碰显示为 pinned=yes 的 Skill。完全跳过它们。\n"
#     "3b. 切勿对受保护的内置列表（目前为：plan）中命名的任何 Skill 进行归档、删除、合并、移动或以其他方式修改。
#     这些支撑着关键的 UX（文档和提示中引用的斜杠命令入口点），
#     并且已从下方的候选列表中过滤掉 —— 切勿将其作为归档或吸收目标重新复活。\n"
#     "3c. 切勿归档或清理候选列表中标记为 `cron=yes` 的任何 Skill。Cron 任务依赖于它，下次运行时将无法加载它。
#     你**仍可以**将其合并到框架 Skill 中 ——
#     但这仅是因为 Curator 会重写 Cron 任务的 Skill 引用以跟进合并；绝不能直接清理它。\n"
#     "4. 切勿将使用计数器作为跳过合并的理由。
#     计数器是新功能，通常大多为零。请根据**内容**判断重叠度，而不是根据 use_count。
#     “use=0”不是 Skill 毫无价值的证据；无论如何它只是缺乏证据。
#     推论：“use=0”也**不是**清理（PRUNE）Skill 的理由。
#     切勿归档从未使用的 Skill（use=0），除非它至少有 30 天的历史（检查 last_activity / created 日期）
#     **并且**其内容确实已过时或完全被其他地方吸收 —— 新创建的 Skill 可能只是尚未遇到触发它的场景而已。\n"
#     "5. 切勿以“每个 Skill 都有不同的触发条件”为由拒绝合并。
#     成对的差异性是错误的标准。正确的标准是：“人类维护者会将其写成 N 个独立的 Skill，
#     还是写成带有 N 个标注小节的一个 Skill？”当答案是后者时，进行合并。\n\n"
#     "工作方式 —— 强制执行：\n"
#     "1. 扫描完整的候选列表。
#     识别**前缀聚类（PREFIX CLUSTERS）**（共享首个单词或领域关键字的 Skill）。
#     你可能找到的示例包括：hermes-config-*、hermes-dashboard-*、
#     gateway-*、codex-*、ollama-*、anthropic-*、
#     gemini-*、mcp-*、salvage-*、pr-*、competitor-*、
#     python-*、security-* 等。预计会有 10-25 个聚类。\n"
#     "2. 对于每个包含 2 个以上成员的聚类，**不要**问“这些成对的 Skill 是否重叠？”
#     —— 要问“这些 Skill 共同服务的**框架类（UMBRELLA CLASS）**是什么？
#     维护者是否会命名该类并为其编写一个 Skill？”
#     如果是，选择（或创建）框架 Skill，并将同类 Skill 吸收进去。\n"
#     "3. 合并的三种方式 —— 每个聚类使用合适的方式：\n"
#     "   a. **合并到现有的框架 Skill 中** —— 聚类中的一个 Skill 已经足够宽泛，
#     可以作为框架 Skill（例如 PR 审查聚类中的 `pr-triage-salvage`）。
#     对其进行打补丁（Patch），为每个同类 Skill 的独到见解添加一个带标注的章节，然后归档这些同类 Skill。\n"

#     "   b. **创建一个新的框架 SKILL.md** ——
#     现有的成员都不够宽泛。使用 skill_manage action=create 编写一个新的类级别 Skill，
#     其 SKILL.md 涵盖共享的工作流并包含简短的带标注小节。归档现已被吸收的狭隘同类 Skill。\n"

#     "   c. **降级为 REFERENCES/TEMPLATES/SCRIPTS** —— 同类 Skill 拥有狭隘但有价值的会话特定内容。
#     将其移动到框架 Skill 对应的支持目录中：\n"
#     "      • `references/<topic>.md`：用于会话特定的细节或精简的知识库（引用的研究、API 文档摘要、领域笔记、提供者特性、复现配方）\n"
#     "      • `templates/<name>.<ext>`：用于旨在复制和修改的初始模板文件\n"
#     "      • `scripts/<name>.<ext>`：用于可静态重复运行的操作（验证脚本、Fixture 生成器、探测器）\n"
#     "      然后归档旧的同类 Skill。使用 `terminal` 命令，
#     如 `mkdir -p ~/.hermes/skills/<umbrella>/references/ && mv ... <umbrella>/references/<topic>.md`（或 templates/ / scripts/）。\n\n"
#     "包完整性 —— 强制执行：\n"
#     "在降级或归档 Skill 之前，请将其作为一个**完整的目录包**进行检查，而不仅仅是 SKILL.md。
#     Skill 根目录可能包含 `references/`、`templates/`、`scripts/` 和 `assets/`；
#     `skill_view` 会相对于 Skill 根目录发现这些文件。
#     另一个 Skill 内部的参考 Markdown 文件**并不是**新的 Skill 根目录，不会获得其自身的关联文件发现。\n"
#     "如果源 Skill 包含支持文件，或 SKILL.md 包含相对链接（
#     例如 `references/...`、`templates/...`、`scripts/...` 或 `assets/...`），
#     **切勿**仅将 SKILL.md 扁平化合并到 `<umbrella>/references/<old>.md`。请选择以下一种安全路径：\n"
#     "   • 将其保留为独立 Skill，或者\n"
#     "   • 通过将每个需要的支持文件重新归位到框架 Skill 的规范 `references/`、
#     `templates/`、`scripts/` 或 `assets/` 目录中来进行完全合并，**并且**将目标指令重写为新路径，或者\n"
#     "   • 保持整个原始 Skill 包不变并直接归档。\n"
#     "切勿留下指向旧 Skill 目录下遗留文件的已归档/已降级指令。\n"
#     "4. 标记那些**名称**过于狭隘的 Skill（包含 PR 编号、功能代号、特定的错误字符串、
#     “audit”/“diagnosis”/“salvage” 会话产物）。
#     这些几乎总是属于类级别框架下的子章节或支持文件。\n"
#     "5. 迭代处理。在一个合并轮次之后，扫描剩余的集合，寻找**下一个**框架构建机会。
#     不要在 3 次合并后就停止。\n\n"
#     "你的工具集：\n"
#     "  - skills_list, skill_view        — 读取当前的全局状况\n"
#     "  - skill_manage action=patch      — 为框架 Skill 添加章节\n"
#     "  - skill_manage action=create     — 创建一个新的框架 SKILL.md\n"
#     "  - skill_manage action=write_file — 在现有的 Skill 下添加 references/、templates/ 或 scripts/ 文件（该 Skill 必须已经存在）\n"
#     "  - skill_manage action=delete     — 归档一个 Skill。当你将其内容合并到另一个 Skill 时，
#     **必须**传递 `absorbed_into=<umbrella>`；当你在没有任何转发目标的情况下进行纯粹清理时，
#     传递 `absorbed_into=\"\"`。这将驱动 Cron 任务的 Skill 引用迁移 ——
#     事后从你的 YAML 总结中猜测是很脆弱的。\n"
#     "  - terminal                       — 当包完整性需要时，将本地候选内容移动到支持子文件中；
#     切勿对打包自带、Hub 安装或外部目录中的 Skill 执行 mv、cp、rm、patch 或重写操作\n\n"
#     "只有当 Skill 已经是一个类级别框架，且提议的合并都无法改善可发现性时，'keep'（保留）才是一个合理的决策。
#     “这个 Skill 虽然狭隘但与其同类 Skill 不同”**不是**保留的理由 —— 这是将其作为子章节或支持文件移动到框架 Skill 下的理由。\n\n"
#     "期望的输出：真正的框架化（umbrella-ification）。处理每一个明显的聚类。如果你结束本次审查时归档数量少于 10 个，说明你停止得太早了 —— 请重新查看你未触碰的聚类。\n\n"
#     "完成后，撰写一份人类可读的总结**以及**一个结构化的机器可读文本块，以便下游工具能够区分“合并”与“清理”。格式**严格如下**：\n\n"
#     "## Structured summary (required)\n"
#     "```yaml\n"
#     "consolidations:\n"
#     "  - from: <old-skill-name>\n"
#     "    into: <umbrella-skill-name>\n"
#     "    reason: <简短的一句话 —— 解释为何合并，而不仅仅是'相似'>\n"
#     "prunings:\n"
#     "  - name: <skill-name>\n"
#     "    reason: <简短的一句话 —— 解释为何在无合并目标的情况下归档>\n"
#     "```\n\n"
#     "你移动到 .archive/ 的每个 Skill **必须**恰好出现在上述两个列表之一中。如果你将 X 合并到了框架 Y 中（打补丁到 Y、向 Y 写入参考文件，或创建了吸收 X 内容的 Y），X 放在 `consolidations` 下并带上 `into: Y`。如果你归档了 X 且没有任何吸收 —— 属于真正陈旧、不相关或过时 —— X 放在 `prunings` 下。如果没有相应项目，请保留空列表（`consolidations: []`）。切勿省略该文本块。该文本块应当放在你对已处理聚类、所做补丁和未触碰决策的人类可读总结**之后**。"
# )
CURATOR_REVIEW_PROMPT = (
    "You are running as Hermes' background skill CURATOR. This is an "
    "UMBRELLA-BUILDING consolidation pass, not a passive audit and not a "
    "duplicate-finder.\n\n"
    "The goal of the skill collection is a LIBRARY OF CLASS-LEVEL "
    "INSTRUCTIONS AND EXPERIENTIAL KNOWLEDGE. A collection of hundreds of "
    "narrow skills where each one captures one session's specific bug is "
    "a FAILURE of the library — not a feature. An agent searching skills "
    "matches on descriptions, not on exact names; one broad umbrella "
    "skill with labeled subsections beats five narrow siblings for "
    "discoverability, not the other way around.\n\n"
    "The right target shape is CLASS-LEVEL skills with rich SKILL.md "
    "bodies + `references/`, `templates/`, and `scripts/` subfiles for "
    "session-specific detail — not one-session-one-skill micro-entries.\n\n"
    "Hard rules — do not violate:\n"
    "1. DO NOT touch bundled, hub-installed, or external-dir skills "
    "(`skills.external_dirs`). The candidate list below is already filtered "
    "to local curator-managed skills only; external skills are externally "
    "owned and read-only to this background curator.\n"
    "2. DO NOT delete any skill. Archiving (moving the skill's directory "
    "into ~/.hermes/skills/.archive/) is the maximum destructive action. "
    "Archives are recoverable; deletion is not.\n"
    "3. DO NOT touch skills shown as pinned=yes. Skip them entirely.\n"
    "3b. DO NOT archive, delete, consolidate, move, or otherwise modify any "
    "skill named in the protected built-ins list (currently: plan). These "
    "back load-bearing UX (slash-command entry points referenced in docs and "
    "tips) and are filtered out of the candidate list below — never resurrect "
    "one as an archive or absorb target.\n"
    "3c. DO NOT archive or prune any skill marked `cron=yes` in the candidate "
    "list. A cron job depends on it and will fail to load it on its next "
    "run. You MAY still consolidate it into an umbrella — but only because "
    "the curator rewrites cron job skill references to follow consolidations; "
    "never simply prune it.\n"
    "4. DO NOT use usage counters as a reason to skip consolidation. The "
    "counters are new and often mostly zero. Judge overlap on CONTENT, "
    "not on use_count. 'use=0' is not evidence a skill is valuable; it's "
    "absence of evidence either way. Corollary: 'use=0' is ALSO not a "
    "reason to PRUNE a skill. Never archive a never-used skill (use=0) "
    "unless it is at least 30 days old (check last_activity / created date) "
    "AND its content is genuinely obsolete or fully absorbed elsewhere — a "
    "recently-created skill simply may not have had its trigger come up yet.\n"
    "5. DO NOT reject consolidation on the grounds that 'each skill has "
    "a distinct trigger'. Pairwise distinctness is the wrong bar. The "
    "right bar is: 'would a human maintainer write this as N separate "
    "skills, or as one skill with N labeled subsections?' When the "
    "answer is the latter, merge.\n\n"
    "How to work — not optional:\n"
    "1. Scan the full candidate list. Identify PREFIX CLUSTERS (skills "
    "sharing a first word or domain keyword). Examples you are likely "
    "to find: hermes-config-*, hermes-dashboard-*, gateway-*, codex-*, "
    "ollama-*, anthropic-*, gemini-*, mcp-*, salvage-*, pr-*, "
    "competitor-*, python-*, security-*, etc. Expect 10-25 clusters.\n"
    "2. For each cluster with 2+ members, do NOT ask 'are these pairs "
    "overlapping?' — ask 'what is the UMBRELLA CLASS these skills all "
    "serve? Would a maintainer name that class and write one skill for "
    "it?' If yes, pick (or create) the umbrella and absorb the siblings "
    "into it.\n"
    "3. Three ways to consolidate — use the right one per cluster:\n"
    "   a. MERGE INTO EXISTING UMBRELLA — one skill in the cluster is "
    "already broad enough to be the umbrella (example: `pr-triage-"
    "salvage` for the PR review cluster). Patch it to add a labeled "
    "section for each sibling's unique insight, then archive the "
    "siblings.\n"
    "   b. CREATE A NEW UMBRELLA SKILL.md — no existing member is broad "
    "enough. Use skill_manage action=create to write a new class-level "
    "skill whose SKILL.md covers the shared workflow and has short "
    "labeled subsections. Archive the now-absorbed narrow siblings.\n"
    "   c. DEMOTE TO REFERENCES/TEMPLATES/SCRIPTS — a sibling has "
    "narrow-but-valuable session-specific content. Move it into the "
    "umbrella's appropriate support directory:\n"
    "      • `references/<topic>.md` for session-specific detail OR "
    "condensed knowledge banks (quoted research, API docs excerpts, "
    "domain notes, provider quirks, reproduction recipes)\n"
    "      • `templates/<name>.<ext>` for starter files meant to be "
    "copied and modified\n"
    "      • `scripts/<name>.<ext>` for statically re-runnable actions "
    "(verification scripts, fixture generators, probes)\n"
    "      Then archive the old sibling. Use `terminal` with `mkdir -p "
    "~/.hermes/skills/<umbrella>/references/ && mv ... <umbrella>/"
    "references/<topic>.md` (or templates/ / scripts/).\n\n"
    "Package integrity — not optional:\n"
    "Before demoting or archiving a skill, inspect it as a COMPLETE "
    "directory package, not just SKILL.md. A skill root may include "
    "`references/`, `templates/`, `scripts/`, and `assets/`; `skill_view` "
    "discovers those relative to the skill root. A reference markdown file "
    "inside another skill is NOT a new skill root and does not get its own "
    "linked-file discovery.\n"
    "If the source skill has support files OR SKILL.md contains relative "
    "links such as `references/...`, `templates/...`, `scripts/...`, or "
    "`assets/...`, DO NOT flatten only SKILL.md into "
    "`<umbrella>/references/<old>.md`. Choose one safe path instead:\n"
    "   • keep it as a standalone skill, OR\n"
    "   • fully merge it by re-homing every needed support file into the "
    "umbrella's canonical `references/`, `templates/`, `scripts/`, or "
    "`assets/` directories AND rewrite the destination instructions to "
    "the new paths, OR\n"
    "   • archive the entire original skill package unchanged.\n"
    "Never leave archived/demoted instructions pointing at files that were "
    "left behind under the old skill directory.\n"
    "4. Also flag skills whose NAME is too narrow (contains a PR number, "
    "a feature codename, a specific error string, an 'audit' / "
    "'diagnosis' / 'salvage' session artifact). These almost always "
    "belong as a subsection or support file under a class-level umbrella.\n"
    "5. Iterate. After one consolidation round, scan the remaining set "
    "and look for the NEXT umbrella opportunity. Don't stop after 3 "
    "merges.\n\n"
    "Your toolset:\n"
    "  - skills_list, skill_view        — read the current landscape\n"
    "  - skill_manage action=patch      — add sections to the umbrella\n"
    "  - skill_manage action=create     — create a new umbrella SKILL.md\n"
    "  - skill_manage action=write_file — add a references/, templates/, "
    "or scripts/ file under an existing skill (the skill must already "
    "exist)\n"
    "  - skill_manage action=delete     — archive a skill. MUST pass "
    "`absorbed_into=<umbrella>` when you've merged its content into another "
    "skill, or `absorbed_into=\"\"` when you're truly pruning with no "
    "forwarding target. This drives cron-job skill-reference migration — "
    "guessing from your YAML summary after the fact is fragile.\n"
    "  - terminal                       — move LOCAL candidate content into "
    "a support subfile when package integrity requires it; never mv, cp, rm, "
    "patch, or rewrite bundled, hub-installed, or external-dir skills\n\n"
    "'keep' is a legitimate decision ONLY when the skill is already a "
    "class-level umbrella and none of the proposed merges would improve "
    "discoverability. 'This is narrow but distinct from its siblings' "
    "is NOT a reason to keep — it's a reason to move it under an "
    "umbrella as a subsection or support file.\n\n"
    "Expected output: real umbrella-ification. Process every obvious "
    "cluster. If you end the pass with fewer than 10 archives, you "
    "stopped too early — go back and look at the clusters you left "
    "alone.\n\n"
    "When done, write a human summary AND a structured machine-readable "
    "block so downstream tooling can distinguish consolidation from "
    "pruning. Format EXACTLY:\n\n"
    "## Structured summary (required)\n"
    "```yaml\n"
    "consolidations:\n"
    "  - from: <old-skill-name>\n"
    "    into: <umbrella-skill-name>\n"
    "    reason: <one short sentence — why merged, not just 'similar'>\n"
    "prunings:\n"
    "  - name: <skill-name>\n"
    "    reason: <one short sentence — why archived with no merge target>\n"
    "```\n\n"
    "Every skill you moved to .archive/ MUST appear in exactly one of the "
    "two lists. If you consolidated X into umbrella Y (patched Y, wrote "
    "a references file to Y, or created Y with X's content absorbed), X "
    "goes under `consolidations` with `into: Y`. If you archived X with "
    "no absorption — truly stale, irrelevant, or obsolete — X goes under "
    "`prunings`. Leave a list empty (`consolidations: []`) if none. Do "
    "not omit the block. The block comes AFTER your human-readable "
    "summary of clusters processed, patches made, and decisions left alone."
)


# ---------------------------------------------------------------------------
# Per-run reports — {YYYYMMDD-HHMMSS}/run.json + REPORT.md under logs/curator/
# ---------------------------------------------------------------------------

def _reports_root() -> Path:
    """Directory where curator run reports are written.

    Lives under the profile-aware logs dir (``~/.hermes/logs/curator/``)
    alongside ``agent.log`` and ``gateway.log`` so it's found by anyone
    looking for operational telemetry, not mixed in with the user's
    authored skill data in ``~/.hermes/skills/``.

    ``ensure_hermes_home()`` pre-creates this dir on every CLI launch and
    the v22→v23 migration backfills it for existing profiles, but we
    still mkdir here as a belt-and-suspenders so the curator works even
    from an odd entry path (e.g. gateway-only install, bare library use)
    that bypasses both.
    """
    root = get_hermes_home() / "logs" / "curator"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.debug("Curator reports dir create failed: %s", e)
    return root


def _needle_in_path_component(needle: str, path: str) -> bool:
    """Check if *needle* is a complete filename stem or directory name in *path*.

    Unlike simple substring matching, this avoids false positives where short
    skill names are embedded in longer filenames (e.g. "api" matching
    "references/api-design.md").  Hyphens and underscores are normalised so
    "open-webui-setup" matches "open_webui_setup.md".
    """
    norm_needle = needle.replace("-", "_")
    for part in path.replace("\\", "/").split("/"):
        if not part:
            continue
        stem = part.rsplit(".", 1)[0] if "." in part else part
        if stem.replace("-", "_") == norm_needle:
            return True
    return False


def _classify_removed_skills(
    removed: List[str],
    added: List[str],
    after_names: Set[str],
    tool_calls: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Split ``removed`` into consolidated vs pruned.

    A removed skill is "consolidated" when the curator absorbed its content
    into another skill (an umbrella) during this run — the content still
    lives, just under a different name. A removed skill is "pruned" when the
    curator archived it for staleness/irrelevance without preserving its
    content elsewhere.

    Heuristic: scan this run's ``skill_manage`` tool calls and look for
    ``write_file``/``patch``/``create``/``edit`` actions whose target skill
    (the ``name`` argument) is NOT the removed skill and whose
    ``file_path`` / ``file_content`` / ``content`` arguments reference the
    removed skill's name. That's the textbook "absorbed into umbrella"
    signal. Ties are broken by first-match (earliest tool call wins).

    Returns ``{"consolidated": [{"name", "into", "evidence"}, ...],
               "pruned":       [{"name"}, ...]}``.
    """
    consolidated: List[Dict[str, Any]] = []
    pruned: List[Dict[str, Any]] = []

    # Pre-parse tool calls: we only care about skill_manage.
    parsed_calls: List[Dict[str, Any]] = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        if tc.get("name") != "skill_manage":
            continue
        raw = tc.get("arguments") or ""
        # Arguments can be a JSON string (standard) or a dict (defensive).
        args: Dict[str, Any] = {}
        if isinstance(raw, dict):
            args = raw
        elif isinstance(raw, str):
            try:
                args = json.loads(raw)
            except Exception:
                # Truncated or malformed — fall back to substring match on
                # the raw string so we still catch the common case.
                args = {"_raw": raw}
        if not isinstance(args, dict):
            continue
        parsed_calls.append(args)

    # Build a set of "destination" skill names: anything still present after
    # the run plus anything newly added this run. A removed skill being
    # referenced from one of these is the consolidation signal.
    destinations = set(after_names) | set(added or [])

    for name in removed:
        if not name:
            continue
        into: Optional[str] = None
        evidence: Optional[str] = None

        # Normalise name variants we'll search for in path/content strings.
        needles = {name, name.replace("-", "_"), name.replace("_", "-")}

        for args in parsed_calls:
            target = args.get("name")
            if not isinstance(target, str) or not target:
                continue
            # A call that operates on the removed skill itself isn't
            # consolidation evidence.
            if target == name:
                continue
            # The target must be a surviving or newly-created skill —
            # otherwise we're pointing to a skill that doesn't exist.
            if target not in destinations:
                continue

            # Look for the removed skill's name in file_path / content / raw.
            # Matching strategy differs by field type:
            #   file_path — needle must be a complete path component
            #     (filename stem or directory name), so "api" does NOT
            #     falsely match "references/api-design.md".
            #   content fields — word-boundary regex so "test" does NOT
            #     falsely match "latest" or "testing".
            haystacks: List[tuple[str, str]] = []
            for key in ("file_path", "file_content", "content", "new_string", "_raw"):
                v = args.get(key)
                if isinstance(v, str):
                    haystacks.append((key, v))
            hit = False
            for key, hay in haystacks:
                for needle in needles:
                    if not needle:
                        continue
                    if key == "file_path":
                        matched = _needle_in_path_component(needle, hay)
                    else:
                        matched = bool(
                            re.search(rf'\b{re.escape(needle)}\b', hay)
                        )
                    if matched:
                        hit = True
                        evidence = (
                            f"skill_manage action={args.get('action', '?')} "
                            f"on '{target}' referenced '{name}' "
                            f"in {hay[:80]}"
                        )
                        break
                if hit:
                    break
            if hit:
                into = target
                break

        if into:
            consolidated.append({"name": name, "into": into, "evidence": evidence})
        else:
            pruned.append({"name": name})

    return {"consolidated": consolidated, "pruned": pruned}


def _parse_structured_summary(
    llm_final: str,
) -> Dict[str, List[Dict[str, str]]]:
    """Extract the structured YAML block from the curator's final response.

    The curator prompt requires a fenced ```yaml block under
    ``## Structured summary (required)`` with ``consolidations:`` and
    ``prunings:`` lists. This parses it tolerantly:

    - Missing block → returns empty lists (we'll fall back to heuristic).
    - Malformed YAML → returns empty lists and we rely on heuristic.
    - Partial block (e.g. only consolidations) → returns what we could parse.

    Returns ``{"consolidations": [{"from", "into", "reason"}, ...],
               "prunings":       [{"name", "reason"}, ...]}``.
    """
    empty = {"consolidations": [], "prunings": []}
    if not llm_final or not isinstance(llm_final, str):
        return empty

    # Find the YAML fenced block. We look for ```yaml ... ``` specifically
    # rather than any fenced block so we don't accidentally pick up a code
    # sample the model quoted elsewhere.
    import re
    match = re.search(
        r"```ya?ml\s*\n(.*?)\n```",
        llm_final,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return empty

    body = match.group(1)

    # Prefer PyYAML when available — every hermes install already has it
    # (config.yaml loader). Fall back to a hand parser for paranoia.
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(body)
    except Exception:
        return empty

    if not isinstance(data, dict):
        return empty

    out: Dict[str, List[Dict[str, str]]] = {"consolidations": [], "prunings": []}
    cons_raw = data.get("consolidations") or []
    prun_raw = data.get("prunings") or []

    if isinstance(cons_raw, list):
        for entry in cons_raw:
            if not isinstance(entry, dict):
                continue
            frm = entry.get("from")
            into = entry.get("into")
            if not (isinstance(frm, str) and frm.strip()
                    and isinstance(into, str) and into.strip()):
                continue
            reason = entry.get("reason")
            out["consolidations"].append({
                "from": frm.strip(),
                "into": into.strip(),
                "reason": (reason or "").strip() if isinstance(reason, str) else "",
            })

    if isinstance(prun_raw, list):
        for entry in prun_raw:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not (isinstance(name, str) and name.strip()):
                continue
            reason = entry.get("reason")
            out["prunings"].append({
                "name": name.strip(),
                "reason": (reason or "").strip() if isinstance(reason, str) else "",
            })

    return out


def _extract_absorbed_into_declarations(
    tool_calls: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Walk this run's tool calls and extract model-declared absorption targets.

    The curator prompt requires every ``skill_manage(action='delete')`` call
    to pass ``absorbed_into=<umbrella>`` when consolidating, or
    ``absorbed_into=""`` when truly pruning. This is the single authoritative
    signal for classification — the model's own declaration at the moment of
    deletion, which beats both post-hoc YAML summary parsing and substring
    heuristics on other tool calls.

    Returns ``{skill_name: {"into": "<umbrella>" | "", "declared": True}}``.
    Entries with ``into == ""`` are explicit prunings.
    Skills without a ``skill_manage(delete)`` call, or with one that omitted
    ``absorbed_into``, are not in the returned dict — caller falls back to
    the existing heuristic/YAML logic for those (backward compat with older
    curator runs and any callers that don't populate the arg).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        if tc.get("name") != "skill_manage":
            continue
        raw = tc.get("arguments") or ""
        args: Dict[str, Any] = {}
        if isinstance(raw, dict):
            args = raw
        elif isinstance(raw, str):
            try:
                args = json.loads(raw)
            except Exception:
                continue
        if not isinstance(args, dict):
            continue
        if args.get("action") != "delete":
            continue
        name = args.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        # absorbed_into must be present (even empty string is meaningful);
        # missing key means the model didn't declare intent.
        if "absorbed_into" not in args:
            continue
        target = args.get("absorbed_into")
        if target is None:
            continue
        if not isinstance(target, str):
            continue
        out[name.strip()] = {"into": target.strip(), "declared": True}
    return out


def _reconcile_classification(
    removed: List[str],
    heuristic: Dict[str, List[Dict[str, Any]]],
    model_block: Dict[str, List[Dict[str, str]]],
    destinations: Set[str],
    absorbed_declarations: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Merge heuristic (tool-call evidence) with the model's structured block.

    Rules (evaluated in order; first match wins):
    - **Model-declared `absorbed_into` at delete time is authoritative.** Any
      entry in ``absorbed_declarations`` beats every other signal. This is
      the model telling us directly, at the moment of deletion, what it did.
      ``into != ""`` and target exists → consolidated. ``into == ""`` →
      pruned. ``into != ""`` but target doesn't exist → hallucination; fall
      through to the usual signals.
    - Model-declared consolidation wins when its ``into`` target exists
      in ``destinations`` (survived or newly-created). This gives the
      model authority over intent + rationale.
    - Model-declared consolidation whose ``into`` target does NOT exist is
      downgraded: the model hallucinated an umbrella. We prefer the
      heuristic's finding for that skill, or fall back to pruned.
    - Heuristic-only finding (model didn't mention it, tool calls confirm)
      is preserved as a consolidation, marked ``source="tool-call audit"``.
    - Model-declared pruning is accepted unless the heuristic has
      tool-call evidence that contradicts it (rare — the heuristic would
      have flagged consolidation). In that case we log both.

    Every removed skill is placed in exactly one bucket.
    """
    heur_cons = {e["name"]: e for e in heuristic.get("consolidated", [])}
    heur_pruned = {e["name"] for e in heuristic.get("pruned", [])}

    model_cons = {e["from"]: e for e in model_block.get("consolidations", [])}
    model_pruned = {e["name"]: e for e in model_block.get("prunings", [])}

    declared = absorbed_declarations or {}

    consolidated: List[Dict[str, Any]] = []
    pruned: List[Dict[str, Any]] = []

    for name in removed:
        mc = model_cons.get(name)
        mp = model_pruned.get(name)
        hc = heur_cons.get(name)
        dec = declared.get(name)

        # Authoritative: model declared `absorbed_into` at the delete call.
        if dec is not None:
            into_claim = dec.get("into", "")
            if into_claim and into_claim in destinations:
                entry: Dict[str, Any] = {
                    "name": name,
                    "into": into_claim,
                    "source": "absorbed_into (model-declared at delete)",
                    "reason": (mc.get("reason") or "") if mc else "",
                }
                if hc and hc.get("evidence"):
                    entry["evidence"] = hc["evidence"]
                consolidated.append(entry)
                continue
            if into_claim == "":
                # Explicit prune declaration
                pruned.append({
                    "name": name,
                    "source": "absorbed_into=\"\" (model-declared prune)",
                    "reason": (mp.get("reason") or "") if mp else "",
                })
                continue
            # into_claim is non-empty but target doesn't exist: the model
            # named a nonexistent umbrella at delete time. The tool already
            # rejects this at the skill_manage layer, so we shouldn't see it
            # in practice — but if it slips through (e.g. the umbrella was
            # deleted LATER in the same run), fall through to the usual
            # signals rather than trusting a broken reference.

        # Model says consolidated — trust it if the destination is real.
        if mc and mc.get("into") in destinations:
            entry: Dict[str, Any] = {
                "name": name,
                "into": mc["into"],
                "source": "model" + ("+audit" if hc else ""),
                "reason": mc.get("reason") or "",
            }
            if hc and hc.get("evidence"):
                entry["evidence"] = hc["evidence"]
            consolidated.append(entry)
            continue

        # Model says consolidated but the umbrella doesn't exist —
        # hallucination. Fall back to heuristic or prune.
        if mc and mc.get("into") not in destinations:
            if hc:
                consolidated.append({
                    "name": name,
                    "into": hc["into"],
                    "source": "tool-call audit (model named missing umbrella)",
                    "reason": "",
                    "evidence": hc.get("evidence", ""),
                    "model_claimed_into": mc["into"],
                })
            else:
                pruned.append({
                    "name": name,
                    "source": "fallback (model named missing umbrella, no tool-call evidence)",
                    "reason": "",
                })
            continue

        # Heuristic found consolidation the model didn't mention.
        if hc:
            consolidated.append({
                "name": name,
                "into": hc["into"],
                "source": "tool-call audit (model omitted from structured block)",
                "reason": "",
                "evidence": hc.get("evidence", ""),
            })
            continue

        # Model says pruned (or no mention + no heuristic evidence).
        reason = mp.get("reason", "") if mp else ""
        pruned.append({
            "name": name,
            "source": "model" if mp else "no-evidence fallback",
            "reason": reason,
        })

    return {"consolidated": consolidated, "pruned": pruned}


def _build_rename_summary(
    *,
    before_names: Set[str],
    after_report: List[Dict[str, Any]],
    tool_calls: List[Dict[str, Any]],
    model_final: str,
) -> str:
    """为策展人（curator）运行格式化用户可见的重命名映射。

    渲染“我的技能去哪儿了？”相关行，这些行会被追加到
    传递给网关/CLI接收器的 `final_summary` 字符串中。
    如果本次运行没有归档任何内容，则为空字符串 ——
    大多数心跳检测（ticks）都是无操作的，不应增加额外的日志噪声。

    格式如下：::

        archived 4 skill(s):
          • pdf-extraction → document-tools
          • docx-extraction → document-tools
          • flaky-thing — pruned (stale)
          • old-utility → spreadsheet-ops
        full report: hermes curator status
        keep an umbrella stable: hermes curator pin document-tools

    上限为 10 个条目，以避免 50 个技能的整合撑爆
    agent.log；完整列表始终保存在 REPORT.md 中。固定提示仅在
    至少产生了一个值得固定的伞形技能的整合时才会出现
    （仅修剪的运行会跳过此提示）。
    """
    after_by_name = {r.get("name"): r for r in after_report if isinstance(r, dict)}
    after_names = set(after_by_name.keys())
    removed = sorted(before_names - after_names)
    added = sorted(after_names - before_names)
    if not removed:
        return ""

    heuristic = _classify_removed_skills(
        removed=removed,
        added=added,
        after_names=after_names,
        tool_calls=tool_calls,
    )
    model_block = _parse_structured_summary(model_final)
    destinations = set(after_names) | set(added)
    absorbed_declarations = _extract_absorbed_into_declarations(tool_calls)
    classification = _reconcile_classification(
        removed=removed,
        heuristic=heuristic,
        model_block=model_block,
        destinations=destinations,
        absorbed_declarations=absorbed_declarations,
    )
    consolidated = classification["consolidated"]
    pruned = classification["pruned"]

    SHOW = 10
    lines: List[str] = []
    total = len(consolidated) + len(pruned)
    lines.append(f"archived {total} skill(s):")
    shown = 0
    for entry in consolidated:
        if shown >= SHOW:
            break
        name = entry.get("name", "?")
        into = entry.get("into", "?")
        lines.append(f"  • {name} → {into}")
        shown += 1
    for entry in pruned:
        if shown >= SHOW:
            break
        name = entry.get("name", "?") if isinstance(entry, dict) else str(entry)
        lines.append(f"  • {name} — pruned (stale)")
        shown += 1
    if total > SHOW:
        lines.append(f"  … and {total - SHOW} more")
    lines.append("full report: hermes curator status")
    # Pin hint — only surface it when there's actually a destination skill
    # worth pinning. The umbrella skills that absorbed content are the natural
    # candidates: pinning one tells future curator runs to leave it alone.
    # Pruned-only runs don't get this hint (nothing surviving to pin).
    if consolidated:
        umbrellas = sorted({e.get("into") for e in consolidated if e.get("into")})
        if umbrellas:
            example = umbrellas[0]
            lines.append(
                f"keep an umbrella stable: hermes curator pin {example}"
            )
    return "\n".join(lines)


def _write_run_report(
    *,
    started_at: datetime,
    elapsed_seconds: float,
    auto_counts: Dict[str, int],
    auto_summary: str,
    before_report: List[Dict[str, Any]],
    before_names: Set[str],
    after_report: List[Dict[str, Any]],
    llm_meta: Dict[str, Any],
) -> Optional[Path]:
    """在 logs/curator/{YYYYMMDD-HHMMSS}/ 目录下写入 run.json 和 REPORT.md 文件。

    成功时返回报告目录路径，如果写入未能完成则返回 None
    （调用方会记录日志并继续 —— 报告属于尽力而为的操作）。
    """
    root = _reports_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.debug("Curator report dir create failed: %s", e)
        return None

    stamp = started_at.strftime("%Y%m%d-%H%M%S")
    run_dir = root / stamp
    # If we crash-reran within the same second, append a disambiguator
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"{stamp}-{suffix}"
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except Exception as e:
        logger.debug("Curator run dir create failed: %s", e)
        return None

    # Diff before/after
    after_by_name = {r.get("name"): r for r in after_report if isinstance(r, dict)}
    after_names = set(after_by_name.keys())
    removed = sorted(before_names - after_names)   # archived during this run
    added = sorted(after_names - before_names)     # new skills this run
    before_by_name = {r.get("name"): r for r in before_report if isinstance(r, dict)}

    # State transitions between the two snapshots (e.g. active -> stale)
    transitions: List[Dict[str, str]] = []
    for name in sorted(after_names & before_names):
        s_before = (before_by_name.get(name) or {}).get("state")
        s_after = (after_by_name.get(name) or {}).get("state")
        if s_before and s_after and s_before != s_after:
            transitions.append({"name": name, "from": s_before, "to": s_after})

    # Classify LLM tool calls
    tc_counts: Dict[str, int] = {}
    for tc in llm_meta.get("tool_calls", []) or []:
        name = tc.get("name", "unknown")
        tc_counts[name] = tc_counts.get(name, 0) + 1

    # Split "removed" into consolidated (absorbed into umbrella) vs pruned
    # (archived for staleness, content not preserved elsewhere). The old
    # "Skills archived" section lumped both together, which misled users
    # into thinking consolidated skills had been pruned.
    #
    # Classification strategy:
    # 1. Parse the curator's structured YAML block from its final response.
    #    The curator is now prompted to emit consolidations/prunings lists
    #    with short rationale. The model has intent visibility the tool
    #    calls don't.
    # 2. Run the tool-call heuristic as a ground-truth audit.
    # 3. Reconcile: model gets authority over intent + rationale, heuristic
    #    catches hallucination (umbrella doesn't exist) and omission
    #    (model forgot to list an actual consolidation).
    heuristic = _classify_removed_skills(
        removed=removed,
        added=added,
        after_names=after_names,
        tool_calls=llm_meta.get("tool_calls", []) or [],
    )
    model_block = _parse_structured_summary(llm_meta.get("final", "") or "")
    destinations = set(after_names) | set(added or [])
    # Authoritative signal: extract per-delete `absorbed_into` declarations
    # from this run's tool calls. These beat both the YAML summary block and
    # the substring heuristic — the model is telling us directly, at the
    # moment of deletion, whether each archived skill was consolidated
    # (into=<umbrella>) or pruned (into="").
    absorbed_declarations = _extract_absorbed_into_declarations(
        llm_meta.get("tool_calls", []) or []
    )
    classification = _reconcile_classification(
        removed=removed,
        heuristic=heuristic,
        model_block=model_block,
        destinations=destinations,
        absorbed_declarations=absorbed_declarations,
    )
    consolidated = classification["consolidated"]
    pruned = classification["pruned"]

    # Rewrite cron job skill references. When the curator consolidates
    # skill X into umbrella Y, any cron job that lists X fails to load
    # it at run time — the scheduler skips it and the job runs without
    # the instructions it was scheduled to follow. Rewriting the
    # references in-place keeps scheduled jobs working across
    # consolidation passes. Best-effort: never let a cron-module issue
    # break the curator.
    cron_rewrites: Dict[str, Any] = {"rewrites": [], "jobs_updated": 0, "jobs_scanned": 0}
    try:
        consolidated_map = {
            e["name"]: e["into"]
            for e in consolidated
            if isinstance(e, dict) and e.get("name") and e.get("into")
        }
        pruned_names = [
            e["name"] for e in pruned
            if isinstance(e, dict) and e.get("name")
        ]
        if consolidated_map or pruned_names:
            from cron.jobs import rewrite_skill_refs as _rewrite_cron_refs
            cron_rewrites = _rewrite_cron_refs(
                consolidated=consolidated_map,
                pruned=pruned_names,
            )
    except Exception as e:
        logger.debug("Curator cron skill rewrite failed: %s", e, exc_info=True)
        cron_rewrites = {
            "rewrites": [],
            "jobs_updated": 0,
            "jobs_scanned": 0,
            "error": str(e),
        }

    payload = {
        "started_at": started_at.isoformat(),
        "duration_seconds": round(elapsed_seconds, 2),
        "model": llm_meta.get("model", ""),
        "provider": llm_meta.get("provider", ""),
        "auto_transitions": auto_counts,
        "counts": {
            "before": len(before_names),
            "after": len(after_names),
            "delta": len(after_names) - len(before_names),
            "archived_this_run": len(removed),
            "added_this_run": len(added),
            "consolidated_this_run": len(consolidated),
            "pruned_this_run": len(pruned),
            "state_transitions": len(transitions),
            "cron_jobs_rewritten": int(cron_rewrites.get("jobs_updated", 0)),
            "tool_calls_total": sum(tc_counts.values()),
        },
        "tool_call_counts": tc_counts,
        "archived": removed,
        "consolidated": consolidated,
        "pruned": pruned,
        "pruned_names": [p["name"] for p in pruned],
        "added": added,
        "state_transitions": transitions,
        "cron_rewrites": cron_rewrites,
        "llm_final": llm_meta.get("final", ""),
        "llm_summary": llm_meta.get("summary", ""),
        "llm_error": llm_meta.get("error"),
        "tool_calls": llm_meta.get("tool_calls", []),
    }

    # run.json — machine-readable, full fidelity
    try:
        (run_dir / "run.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception as e:
        logger.debug("Curator run.json write failed: %s", e)

    # REPORT.md — human-readable
    try:
        md = _render_report_markdown(payload)
        (run_dir / "REPORT.md").write_text(md, encoding="utf-8")
    except Exception as e:
        logger.debug("Curator REPORT.md write failed: %s", e)

    # cron_rewrites.json — only when at least one job was touched, to
    # keep run dirs uncluttered for the common no-op case.
    try:
        if int(cron_rewrites.get("jobs_updated", 0)) > 0:
            (run_dir / "cron_rewrites.json").write_text(
                json.dumps(cron_rewrites, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    except Exception as e:
        logger.debug("Curator cron_rewrites.json write failed: %s", e)

    return run_dir


def _render_report_markdown(p: Dict[str, Any]) -> str:
    """Render the human-readable report."""
    lines: List[str] = []
    started = p.get("started_at", "")
    duration = p.get("duration_seconds", 0) or 0
    mins, secs = divmod(int(duration), 60)
    dur_label = f"{mins}m {secs}s" if mins else f"{secs}s"

    lines.append(f"# Curator run — {started}\n")
    model = p.get("model") or "(not resolved)"
    prov = p.get("provider") or "(not resolved)"
    counts = p.get("counts") or {}
    lines.append(
        f"Model: `{model}` via `{prov}`  ·  Duration: {dur_label}  ·  "
        f"Agent-created skills: {counts.get('before', 0)} → {counts.get('after', 0)} "
        f"({counts.get('delta', 0):+d})\n"
    )

    error = p.get("llm_error")
    if error:
        lines.append(f"> ⚠ LLM pass error: `{error}`\n")

    # Auto-transitions (pure, no LLM)
    auto = p.get("auto_transitions") or {}
    lines.append("## Auto-transitions (pure, no LLM)\n")
    lines.append(f"- checked: {auto.get('checked', 0)}")
    lines.append(f"- marked stale: {auto.get('marked_stale', 0)}")
    lines.append(f"- archived (no LLM, pure time-based staleness): {auto.get('archived', 0)}")
    lines.append(f"- reactivated: {auto.get('reactivated', 0)}")
    lines.append("")

    # LLM pass numbers
    tc_counts = p.get("tool_call_counts") or {}
    lines.append("## LLM consolidation pass\n")
    lines.append(f"- tool calls: **{counts.get('tool_calls_total', 0)}** "
                 f"(by name: {', '.join(f'{k}={v}' for k, v in sorted(tc_counts.items())) or 'none'})")
    lines.append(f"- consolidated into umbrellas: **{counts.get('consolidated_this_run', 0)}**")
    lines.append(f"- pruned (archived for staleness): **{counts.get('pruned_this_run', 0)}**")
    lines.append(f"- new skills this run: **{counts.get('added_this_run', 0)}**")
    lines.append(f"- state transitions (active ↔ stale ↔ archived): "
                 f"**{counts.get('state_transitions', 0)}**")
    lines.append("")

    # Consolidated list — content absorbed into an umbrella. The directory
    # on disk still lives under ~/.hermes/skills/.archive/ (every removal is
    # recoverable by design), but the "live" content for these skills
    # continues to exist inside the destination umbrella.
    consolidated = p.get("consolidated") or []
    if consolidated:
        lines.append(f"### Consolidated into umbrella skills ({len(consolidated)})\n")
        lines.append(
            "_These skills were **absorbed into another skill** during this run — "
            "their content still lives, just under a different name. "
            "The original directory was moved to `~/.hermes/skills/.archive/` for "
            "safety and can be restored via `hermes curator restore <name>` if the "
            "consolidation was wrong._\n"
        )
        SHOW = 50
        for entry in consolidated[:SHOW]:
            name = entry.get("name", "?")
            into = entry.get("into", "?")
            reason = (entry.get("reason") or "").strip()
            source = entry.get("source", "")
            line = f"- `{name}` → merged into `{into}`"
            if reason:
                line += f" — {reason}"
            if source and source.startswith("tool-call audit"):
                # The model didn't enumerate this one — surface that to the
                # user so they know why the row has no rationale.
                line += f"  _(detected via {source})_"
            lines.append(line)
            if entry.get("model_claimed_into"):
                lines.append(
                    f"  ⚠ The curator's summary named `{entry['model_claimed_into']}` "
                    "as the umbrella but that skill doesn't exist post-run; "
                    "showing the tool-call audit's finding instead."
                )
        if len(consolidated) > SHOW:
            lines.append(f"- … and {len(consolidated) - SHOW} more (see `run.json`)")
        lines.append("")

    # Pruned list — archived without consolidation. These are the
    # "stale skill pruned" cases the UI should mark clearly.
    pruned = p.get("pruned") or []
    if pruned:
        lines.append(f"### Pruned — archived for staleness ({len(pruned)})\n")
        lines.append(
            "_These skills were archived without being merged into an umbrella "
            "(e.g. stale, unused, or judged irrelevant). "
            "Directories live under `~/.hermes/skills/.archive/`. "
            "Restore any via `hermes curator restore <name>`._\n"
        )
        SHOW = 50
        for entry in pruned[:SHOW]:
            # Entries are dicts with {name, source, reason} when written via
            # the reconciler, or bare strings when an older format slipped
            # through. Handle both.
            if isinstance(entry, dict):
                name = entry.get("name", "?")
                reason = (entry.get("reason") or "").strip()
                line = f"- `{name}`"
                if reason:
                    line += f" — {reason}"
                lines.append(line)
            else:
                lines.append(f"- `{entry}`")
        if len(pruned) > SHOW:
            lines.append(f"- … and {len(pruned) - SHOW} more (see `run.json`)")
        lines.append("")

    # Added list
    added = p.get("added") or []
    if added:
        lines.append(f"### New skills this run ({len(added)})\n")
        lines.append("_Usually these are new class-level umbrellas created via `skill_manage action=create`._\n")
        for n in added:
            lines.append(f"- `{n}`")
        lines.append("")

    # State transitions
    trans = p.get("state_transitions") or []
    if trans:
        lines.append(f"### State transitions ({len(trans)})\n")
        for t in trans:
            lines.append(f"- `{t.get('name')}`: {t.get('from')} → {t.get('to')}")
        lines.append("")

    # Cron job rewrites — show which scheduled jobs had their skill
    # references updated so users can audit that the auto-rewrite did
    # the right thing. Only present when at least one job changed.
    cron_rw = p.get("cron_rewrites") or {}
    cron_rewrites_list = cron_rw.get("rewrites") or []
    if cron_rewrites_list:
        lines.append(f"### Cron job skill references rewritten ({len(cron_rewrites_list)})\n")
        lines.append(
            "_Cron jobs that referenced a consolidated or pruned skill were "
            "updated in-place so they keep loading the right instructions "
            "on their next run. See `cron_rewrites.json` for the full record._\n"
        )
        SHOW = 25
        for entry in cron_rewrites_list[:SHOW]:
            job_name = entry.get("job_name") or entry.get("job_id") or "?"
            before = entry.get("before") or []
            after = entry.get("after") or []
            mapped = entry.get("mapped") or {}
            dropped = entry.get("dropped") or []
            lines.append(
                f"- `{job_name}`: `{', '.join(before)}` → `{', '.join(after) or '(none)'}`"
            )
            for old, new in mapped.items():
                lines.append(f"    - `{old}` → `{new}` (consolidated)")
            for name in dropped:
                lines.append(f"    - `{name}` dropped (pruned)")
        if len(cron_rewrites_list) > SHOW:
            lines.append(
                f"- … and {len(cron_rewrites_list) - SHOW} more "
                "(see `cron_rewrites.json`)"
            )
        lines.append("")

    # Full LLM final response
    final = (p.get("llm_final") or "").strip()
    if final:
        lines.append("## LLM final summary\n")
        lines.append(final)
        lines.append("")
    elif not error:
        llm_sum = p.get("llm_summary") or ""
        if llm_sum:
            lines.append("## LLM summary\n")
            lines.append(llm_sum)
            lines.append("")

    # Recovery footer
    lines.append("## Recovery\n")
    lines.append("- Restore an archived skill: `hermes curator restore <name>`")
    lines.append("- All archives live under `~/.hermes/skills/.archive/` and are recoverable by `mv`")
    lines.append("- See `run.json` in this directory for the full machine-readable record.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator — spawn a forked AIAgent for the LLM review pass
# ---------------------------------------------------------------------------

def _render_candidate_list() -> str:
    """Human/agent-readable list of agent-created skills with usage stats."""
    rows = skill_usage.agent_created_report()
    if not rows:
        return "No agent-created skills to review."
    cron_referenced = _cron_referenced_skills()
    lines = [f"Agent-created skills ({len(rows)}):\n"]
    for r in rows:
        lines.append(
            f"- {r['name']}  "
            f"state={r['state']}  "
            f"pinned={'yes' if r.get('pinned') else 'no'}  "
            f"cron={'yes' if r['name'] in cron_referenced else 'no'}  "
            f"activity={r.get('activity_count', 0)}  "
            f"use={r.get('use_count', 0)}  "
            f"view={r.get('view_count', 0)}  "
            f"patches={r.get('patch_count', 0)}  "
            f"last_activity={r.get('last_activity_at') or 'never'}"
        )
    return "\n".join(lines)


def run_curator_review(
    on_summary: Optional[Callable[[str], None]] = None,
    synchronous: bool = False,
    dry_run: bool = False,
    consolidate: Optional[bool] = None,
) -> Dict[str, Any]:
    """执行单次 Curator 审查流程。

    步骤：
      1. 应用自动状态过渡（纯逻辑控制，无 LLM 参与）。
      2. 如果启用了合并功能（Consolidation）且存在由 Agent 创建的 Skill，
         则派生（Fork）一个 AIAgent，针对当前的候选列表运行 LLM 审查 Prompt。
      3. 使用 last_run_at 以及单行总结更新 .curator_state。
      4. 传入用户可见的描述内容并调用 *on_summary*。

    如果 *synchronous* 为 True，LLM 审查将在调用线程中同步运行；
    默认行为是派生一个守护线程（Daemon Thread），以便调用方能够立即返回。

    *consolidate* 参数用于控制 LLM 级的整合/归类 pass 流程。
    默认值 ``None`` 会从配置中读取 ``curator.consolidate``（默认关闭 OFF）。
    显式传入 ``True``/``False`` 则会覆盖本次调用的默认配置 ——
    这一特性常用于 ``hermes curator run --consolidate`` 命令行标志。
    当合并功能关闭时，仅运行确定性的空闲清理逻辑，
    完全跳过派生的辅助模型审查（不会产生辅助模型的消耗成本）。

    如果 *dry_run* 为 True，系统将跳过自动过期/归档的状态过渡，
    同时指示 LLM 审查流程仅生成报告 ——
    不调用 skill_manage 进行变更，也不执行最终的归档移动。
    系统仍会写入 REPORT.md 并将其记录在 ``state.last_report_path`` 中，
    以便用户查看 Curator 本“计划”执行的操作。
    试运行（Dry-run）同样遵循 *consolidate* 的设定：
    当合并功能关闭时，预览结果仅包含确定性清理的候选项目。
    """
    if consolidate is None:
        consolidate = get_consolidate()
    start = datetime.now(timezone.utc)
    if dry_run:
        # Count candidates without mutating state.
        try:
            report = skill_usage.agent_created_report()
            counts = {
                "checked": len(report),
                "marked_stale": 0,
                "archived": 0,
                "reactivated": 0,
            }
        except Exception:
            counts = {"checked": 0, "marked_stale": 0, "archived": 0, "reactivated": 0}
    else:
        # 变动前快照 — 尽力而为，绝不阻塞流程运行。
        # 快照失败仅输出 Debug 日志并继续执行
        # （若非如此，一次短暂的磁盘问题就会静默禁用 Curator 且无法恢复，后果更为严重）。
        # 若用户要求必须生成快照，可在修复磁盘空间前彻底禁用 Curator。
        try:
            from agent import curator_backup
            snap = curator_backup.snapshot_skills(reason="pre-curator-run")
            if snap is not None and on_summary:
                try:
                    on_summary(f"curator: snapshot created ({snap.name})")
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Curator pre-run snapshot failed: %s", e, exc_info=True)
        counts = apply_automatic_transitions(now=start)

    auto_summary_parts = []
    if counts["marked_stale"]:
        auto_summary_parts.append(f"{counts['marked_stale']} marked stale")
    if counts["archived"]:
        auto_summary_parts.append(f"{counts['archived']} archived")
    if counts["reactivated"]:
        auto_summary_parts.append(f"{counts['reactivated']} reactivated")
    auto_summary = ", ".join(auto_summary_parts) if auto_summary_parts else "no changes"

    # 在进行 LLM 审查前先持久化保存状态，
    # 这样即使在审查过程中发生崩溃，仍会记录本次运行，
    # 从而避免立即重新触发。
    # 在试运行（Dry-run）模式下，我们**不会**更新 last_run_at 或 run_count ——
    # 预览操作不应当将下一次计划的正式审查推迟。
    # 但我们仍会记录一份总结，
    # 以便通过 `hermes curator status` 可以查看到曾执行过预览。
    state = load_state()
    if not dry_run:
        state["last_run_at"] = start.isoformat()
        state["run_count"] = int(state.get("run_count", 0)) + 1
    prefix = "dry-run auto: " if dry_run else "auto: "
    state["last_run_summary"] = f"{prefix}{auto_summary}"
    save_state(state)

    def _llm_pass():
        nonlocal auto_summary
        # Snapshot skill state BEFORE the LLM pass so the report can diff.
        try:
            before_report = skill_usage.agent_created_report()
        except Exception:
            before_report = []
        before_names = {r.get("name") for r in before_report if isinstance(r, dict)}

        # 合并流程门控（Consolidation gate）。
        # 当该功能关闭时（默认关闭），Curator 仅会执行上述确定性的空闲清理 ——
        # 不会派生辅助模型进行审查，不会构建框架，亦无辅助模型产生的成本。
        # 记录本次运行，写入仅反映清理结果的报告，
        # 并直接返回，不再派生（Fork）新进程。
        if not consolidate:
            final_summary = (
                f"{prefix}{auto_summary}; llm: skipped (consolidation off)"
            )
            llm_meta = {
                "final": "",
                "summary": "skipped (consolidation off)",
                "model": "",
                "provider": "",
                "tool_calls": [],
                "error": None,
            }
            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            state2 = load_state()
            state2["last_run_duration_seconds"] = elapsed
            state2["last_run_summary"] = final_summary
            try:
                after_report = skill_usage.agent_created_report()
            except Exception:
                after_report = []
            try:
                report_path = _write_run_report(
                    started_at=start,
                    elapsed_seconds=elapsed,
                    auto_counts=counts,
                    auto_summary=auto_summary,
                    before_report=before_report,
                    before_names=before_names,
                    after_report=after_report,
                    llm_meta=llm_meta,
                )
                if report_path is not None:
                    state2["last_report_path"] = str(report_path)
            except Exception as e:
                logger.debug("Curator report write failed: %s", e, exc_info=True)
            save_state(state2)
            if on_summary:
                try:
                    on_summary(f"curator: {final_summary}")
                except Exception:
                    pass
            return

        llm_meta: Dict[str, Any] = {}
        try:
            candidate_list = _render_candidate_list()
            if "No agent-created skills" in candidate_list:
                final_summary = f"{prefix}{auto_summary}; llm: skipped (no candidates)"
                llm_meta = {
                    "final": "",
                    "summary": "skipped (no candidates)",
                    "model": "",
                    "provider": "",
                    "tool_calls": [],
                    "error": None,
                }
            else:
                # 当启用清理内置 Skill 功能时，候选列表将包含打包自带的 Skill。
                # 针对这些 Skill，覆盖默认的“请勿触碰打包内置内容”规则 ——
                # 但仅允许进行归档操作，
                # 且通过 Hub 安装的 Skill 仍严格禁止触碰。
                builtins_note = ""
                if get_prune_builtins():
                    # builtins_note = (
                    #     "\n\n清理内置 Skill 模式已开启：打包自带的内置 Skill "
                    #     "已包含在下方的候选列表中，"
                    #     "且可能会因陈旧/不相关而被归档，"
                    #     "仅针对打包内置 Skill 覆盖硬性规则 #1。"
                    #     "通过 Hub 安装的 Skill 仍严格禁止触碰。"
                    #     "对待陈旧内置 Skill 的方式与对待 Agent 创建的陈旧 Skill 相同："
                    #     "对其进行归档（绝不删除）。"
                    #     "只有当用户显式恢复时，它才会在执行 `hermes update` 时被恢复。"
                    # )
                    builtins_note = (
                        "\n\nPRUNE-BUILTINS MODE IS ON: bundled built-in skills "
                        "ARE included in the candidate list below and MAY be "
                        "archived for staleness/irrelevance, overriding hard "
                        "rule #1 for bundled skills ONLY. Hub-installed skills "
                        "remain strictly off-limits. Treat a stale built-in the "
                        "same as a stale agent-created skill: archive it (never "
                        "delete). It will be restored on `hermes update` only if "
                        "the user explicitly restores it."
                    )
                if dry_run:
                    prompt = (
                        f"{CURATOR_DRY_RUN_BANNER}\n\n"
                        f"{CURATOR_REVIEW_PROMPT}{builtins_note}\n\n"
                        f"{candidate_list}"
                    )
                else:
                    prompt = f"{CURATOR_REVIEW_PROMPT}{builtins_note}\n\n{candidate_list}"
                llm_meta = _run_llm_review(prompt)
                final_summary = (
                    f"{prefix}{auto_summary}; llm: {llm_meta.get('summary', 'no change')}"
                )
        except Exception as e:
            logger.debug("Curator LLM pass failed: %s", e, exc_info=True)
            final_summary = f"{prefix}{auto_summary}; llm: error ({e})"
            llm_meta = {
                "final": "",
                "summary": f"error ({e})",
                "model": "",
                "provider": "",
                "tool_calls": [],
                "error": str(e),
            }

        # 将重命名映射（`旧名称 → 伞形名称`）追加到用户可见的
        # 摘要中，这样大家就不需要深入查阅 REPORT.md
        # 才能知道自己的技能去哪儿了。尽力而为：
        # 分类是纯粹的，但绝不要因为格式问题而阻塞运行。
        try:
            rename_lines = _build_rename_summary(
                before_names=before_names,
                after_report=skill_usage.agent_created_report(),
                tool_calls=llm_meta.get("tool_calls", []) or [],
                model_final=llm_meta.get("final", "") or "",
            )
            if rename_lines:
                final_summary = f"{final_summary}\n{rename_lines}"
        except Exception as e:
            logger.debug("Curator rename summary build failed: %s", e, exc_info=True)

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        state2 = load_state()
        state2["last_run_duration_seconds"] = elapsed
        state2["last_run_summary"] = final_summary

        # Write the per-run report. Runs in a best-effort try so a
        # reporting bug never breaks the curator itself. Report path is
        # recorded in state so `hermes curator status` can point at it.
        try:
            after_report = skill_usage.agent_created_report()
        except Exception:
            after_report = []
        try:
            report_path = _write_run_report(
                started_at=start,
                elapsed_seconds=elapsed,
                auto_counts=counts,
                auto_summary=auto_summary,
                before_report=before_report,
                before_names=before_names,
                after_report=after_report,
                llm_meta=llm_meta,
            )
            if report_path is not None:
                state2["last_report_path"] = str(report_path)
        except Exception as e:
            logger.debug("Curator report write failed: %s", e, exc_info=True)

        save_state(state2)

        if on_summary:
            try:
                on_summary(f"curator: {final_summary}")
            except Exception:
                pass

    if synchronous:
        _llm_pass()
    else:
        t = threading.Thread(target=_llm_pass, daemon=True, name="curator-review")
        t.start()

    return {
        "started_at": start.isoformat(),
        "auto_transitions": counts,
        "summary_so_far": auto_summary,
    }


def _resolve_review_runtime(cfg: Dict[str, Any]) -> _ReviewRuntimeBinding:
    """Resolve provider/model and per-slot credentials for the curator review fork.

    Same precedence as `_resolve_review_model()`. Non-empty ``api_key`` /
    ``base_url`` from the active slot are returned as explicit overrides so
    ``resolve_runtime_provider`` does not silently reuse the main chat
    credential chain for a routed auxiliary model.
    """
    _main = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    _main_provider = _main.get("provider") or "auto"
    _main_model = _main.get("default") or _main.get("model") or ""

    # 1. Canonical aux task slot
    _aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}
    _cur_task = _aux.get("curator", {}) if isinstance(_aux.get("curator"), dict) else {}
    _task_provider = (_cur_task.get("provider") or "").strip() or None
    _task_model = (_cur_task.get("model") or "").strip() or None
    if _task_provider and _task_provider != "auto" and _task_model:
        return _ReviewRuntimeBinding(
            _task_provider,
            _task_model,
            _strip_aux_credential(_cur_task.get("api_key")),
            _strip_aux_credential(_cur_task.get("base_url")),
            _merge_request_overrides({}, _cur_task.get("extra_body")),
        )

    # 2. Legacy curator.auxiliary.{provider,model} (deprecated, pre-unification)
    _cur = cfg.get("curator", {}) if isinstance(cfg.get("curator"), dict) else {}
    _legacy = _cur.get("auxiliary", {}) if isinstance(_cur.get("auxiliary"), dict) else {}
    _legacy_provider = _legacy.get("provider") or None
    _legacy_model = _legacy.get("model") or None
    if _legacy_provider and _legacy_model:
        logger.info(
            "curator: using deprecated curator.auxiliary.{provider,model} "
            "config — please migrate to auxiliary.curator.{provider,model}"
        )
        return _ReviewRuntimeBinding(
            str(_legacy_provider),
            str(_legacy_model),
            _strip_aux_credential(_legacy.get("api_key")),
            _strip_aux_credential(_legacy.get("base_url")),
            _merge_request_overrides({}, _legacy.get("extra_body")),
        )

    # 3. Fall through to the main chat model
    return _ReviewRuntimeBinding(_main_provider, _main_model, None, None, {})


def _resolve_review_model(cfg: Dict[str, Any]) -> tuple[str, str]:
    """Pick (provider, model) for the curator review fork.

    Curator is a regular auxiliary task slot — ``auxiliary.curator.{provider,model}``
    — so it participates in the canonical aux-model plumbing (``hermes model`` →
    auxiliary picker, the dashboard Models tab, ``auxiliary.curator.{timeout,
    base_url,api_key,extra_body}``). ``provider: "auto"`` with an empty model
    means "use the main chat model" — same default as every other aux task.

    Legacy fallback: users who configured ``curator.auxiliary.{provider,model}``
    under the previous one-off schema still work. Precedence:
      1. ``auxiliary.curator.{provider,model}`` when both are set non-auto
      2. Legacy ``curator.auxiliary.{provider,model}`` when both are set
      3. Main ``model.{provider,default/model}`` pair
    """
    b = _resolve_review_runtime(cfg)
    return b.provider, b.model


def _run_llm_review(prompt: str) -> Dict[str, Any]:
    """派生一个 AIAgent 子进程来运行策展人（curator）审核提示词。

    返回一个包含以下内容的字典：
      - final：来自审核员的完整（未截断）最终响应
      - summary：适用于状态文件的简短摘要（上限 240 个字符）
      - model, provider：子进程实际运行所使用的模型和提供商
      - tool_calls：在运行期间发生的所有工具调用的 {name, arguments} 列表
        （为了可读性，参数可能会被截断）
      - error：如果运行中途失败则会设置此项；final/summary 可能会为空

    绝不抛出异常；调用方会获取一个结构化的失败结果。
    """
    import contextlib
    result_meta: Dict[str, Any] = {
        "final": "",
        "summary": "",
        "model": "",
        "provider": "",
        "tool_calls": [],
        "error": None,
    }
    try:
        from run_agent import AIAgent
    except Exception as e:
        result_meta["error"] = f"AIAgent import failed: {e}"
        result_meta["summary"] = result_meta["error"]
        return result_meta

    # 以与 CLI 相同的方式解析提供商和模型，
    # 这样 curator 分支就能继承用户的活动主配置，
    # 而不会回退到空的提供商/模型对
    # （这会导致发送 HTTP 400 错误“未提供模型”）。
    # 如果 AIAgent() 没有显式提供商/模型参数，
    # 就会触发自动解析路径，该路径对于仅支持 OAuth 的提供商
    # 以及由凭据池支持的提供商会失败。
    #
    # `_resolve_review_runtime()` 优先使用 `auxiliary.curator.{provider,model,...}`
    # （规范的辅助任务槽位，通过 `hermes model` → 辅助选择器
    # 以及仪表板的“模型”选项卡连接），
    # 同时对旧版配置 `curator.auxiliary.{provider,model,...}` 提供向下兼容的回退。
    # 详情请参见 docs/user-guide/features/curator.md。
    _api_key = None
    _base_url = None
    _api_mode = None
    _resolved_provider = None
    _credential_pool = None
    _request_overrides: Dict[str, Any] = {}
    _max_tokens = None
    _acp_command = None
    _acp_args = None
    _model_name = ""
    try:
        from hermes_cli.config import load_config
        from hermes_cli.runtime_provider import resolve_runtime_provider
        _cfg = load_config()
        _binding = _resolve_review_runtime(_cfg)
        _provider, _model_name = _binding.provider, _binding.model
        _rp = resolve_runtime_provider(
            requested=_provider,
            target_model=_model_name,
            explicit_api_key=_binding.explicit_api_key,
            explicit_base_url=_binding.explicit_base_url,
        )
        _api_key = _rp.get("api_key")
        _base_url = _rp.get("base_url")
        _api_mode = _rp.get("api_mode")
        _resolved_provider = _rp.get("provider") or _provider
        _credential_pool = _rp.get("credential_pool")
        _request_overrides = _merge_request_overrides(
            _rp.get("request_overrides"),
            _binding.request_overrides.get("extra_body"),
        )
        _max_tokens = _rp.get("max_output_tokens")
        _acp_command = _rp.get("command")
        _acp_args = list(_rp.get("args") or [])
        if isinstance(_rp.get("model"), str) and _rp["model"].strip():
            _model_name = _rp["model"].strip()
    except Exception as e:
        logger.debug("Curator provider resolution failed: %s", e, exc_info=True)

    result_meta["model"] = _model_name
    result_meta["provider"] = _resolved_provider or ""

    review_agent = None
    try:
        _agent_kwargs: Dict[str, Any] = {}
        if isinstance(_max_tokens, int):
            _agent_kwargs["max_tokens"] = _max_tokens
        if isinstance(_acp_command, str) and _acp_command:
            _agent_kwargs["acp_command"] = _acp_command
            _agent_kwargs["acp_args"] = _acp_args or []
        review_agent = AIAgent(
            model=_model_name,
            provider=_resolved_provider,
            api_key=_api_key,
            base_url=_base_url,
            api_mode=_api_mode,
            credential_pool=_credential_pool,
            request_overrides=_request_overrides,
            **_agent_kwargs,
            # 针对庞大的技能集合构建伞形技能值得设定较高的
            # 迭代上限 —— 针对数百个候选技能，这一过程通常需要进行
            # 50 到 100 次 API 调用。而单次会话的审核路径
            # 将其上限限制在小得多的数量，
            # 因为它并没有进行策展扫描。
            max_iterations=9999,
            quiet_mode=True,
            platform="curator",
            skip_context_files=True,
            skip_memory=True,
        )
        # 禁用递归提示 — curator 绝不能生成它自己的审核。
        review_agent._memory_nudge_interval = 0
        review_agent._skill_nudge_interval = 0
        # 将此分支标记为自主后台策展（autonomous background curation），
        # 从而触发 skill_manage 的后台审核写入保护机制。
        # 若无此标记，该分支将继承默认的 "assistant_tool" 来源，
        # 导致 is_background_review() 返回 False，
        # 从而使外部/内置/从 hub 安装的 skill_manage 保护机制
        # 在其原本用于防范的策展期间永不触发。
        # turn_context.py 会在回合开始时将此标记绑定到
        # 写入来源的 ContextVar 上（参见 agent/turn_context.py）。
        review_agent._memory_write_origin = "background_review"

        # 将派生代理的 stdout/stderr 重定向到 /dev/null，
        # 以免其工具调用啰嗦信息污染前台终端。
        # 后台线程运行器也会将其隐藏；当调用方
        # 从 CLI 调用 run_curator_review(synchronous=True) 时，
        # 这种双重保险（belt-and-suspenders）路径就显得尤为重要。
        with open(os.devnull, "w", encoding="utf-8") as _devnull, \
             contextlib.redirect_stdout(_devnull), \
             contextlib.redirect_stderr(_devnull):
            conv_result = review_agent.run_conversation(user_message=prompt)

        final = ""
        if isinstance(conv_result, dict):
            final = str(conv_result.get("final_response") or "").strip()
        result_meta["final"] = final
        result_meta["summary"] = (final[:240] + "…") if len(final) > 240 else (final or "no change")

        # 收集用于报告的工具调用。遍历分叉代理的
        # 会话消息，并提取在运行期间发出的所有 tool_call。
        # 截断参数有效载荷，以防庞大的 skill_manage 创建
        # 撑爆报告。
        _calls: List[Dict[str, Any]] = []
        for msg in getattr(review_agent, "_session_messages", []) or []:
            if not isinstance(msg, dict):
                continue
            tcs = msg.get("tool_calls") or []
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                args_raw = fn.get("arguments") or ""
                if isinstance(args_raw, str) and len(args_raw) > 400:
                    args_raw = args_raw[:400] + "…"
                _calls.append({"name": name, "arguments": args_raw})
        result_meta["tool_calls"] = _calls
    except Exception as e:
        result_meta["error"] = f"error: {e}"
        result_meta["summary"] = result_meta["error"]
    finally:
        if review_agent is not None:
            try:
                review_agent.close()
            except Exception:
                pass
    return result_meta


# ---------------------------------------------------------------------------
# Public entrypoint for the session-start hook
# ---------------------------------------------------------------------------

def maybe_run_curator(
    *,
    idle_for_seconds: Optional[float] = None,
    on_summary: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, Any]]:
    """Best-effort: run a curator pass if all gates pass. Returns the result
    dict if a pass was started, else None. Never raises."""
    try:
        if not should_run_now():
            return None
        # Idle gating: only enforce when the caller provided a measurement.
        if idle_for_seconds is not None:
            min_idle_s = get_min_idle_hours() * 3600.0
            if idle_for_seconds < min_idle_s:
                return None
        return run_curator_review(on_summary=on_summary)
    except Exception as e:
        logger.debug("maybe_run_curator failed: %s", e, exc_info=True)
        return None
