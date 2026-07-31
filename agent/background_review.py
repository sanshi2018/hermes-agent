"""Background memory/skill review — fork the agent to evaluate the turn.

After every turn, ``AIAgent.run_conversation`` may call
:func:`spawn_background_review` to fire off a daemon thread that replays
the conversation snapshot in a forked :class:`AIAgent` and asks itself
"should any skill/memory be saved or updated?".  Writes go straight to
the memory + skill stores.  Main conversation and prompt cache are never
touched.

The fork inherits the parent's live runtime (provider, model, base_url,
credentials, cached system prompt) so it hits the same prefix cache and
uses the same auth.  It runs with a tool whitelist limited to memory and
skill management tools; everything else is denied at runtime.

See the ``hermes-agent-dev`` skill (``references/self-improvement-loop.md``)
for invariants and PR review criteria.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from agent.thread_scoped_output import thread_scoped_silence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background-review aux-model selector + routed digest.
#
# The review fork runs on the MAIN model by default ("auto"), replaying the
# full conversation — already warm in the prompt cache, so cheap cache reads.
# Optimal and unchanged. A user can route the review to a different, cheaper
# model via auxiliary.background_review.{provider,model}. A different model
# cannot reuse the parent's cache (different key), so the fork is cold
# regardless — replaying the full transcript would just cold-write it. So when
# (and only when) routed to a different model, we replay a compact DIGEST to
# minimise cold-written tokens. Same model -> full replay; different model ->
# digest. That's the whole policy.
# ---------------------------------------------------------------------------


def _resolve_review_runtime(agent: Any) -> Dict[str, Any]:
    """Resolve provider/model/credentials for the review fork.

    Default (auto / unset / same as parent): inherit the parent's live runtime
    (with codex_app_server -> codex_responses downgrade). ``routed`` is False —
    the fork uses the main model and the warm cache, exactly as before. When
    ``auxiliary.background_review.{provider,model}`` names a concrete model
    different from the parent's, resolve that runtime and set ``routed=True``.
    """
    parent_runtime = agent._current_main_runtime()
    parent_api_mode = parent_runtime.get("api_mode") or None
    if parent_api_mode == "codex_app_server":
        parent_api_mode = "codex_responses"
    parent = {
        "provider": agent.provider,
        "model": agent.model,
        "api_key": parent_runtime.get("api_key") or None,
        "base_url": parent_runtime.get("base_url") or None,
        "api_mode": parent_api_mode,
        "credential_pool": getattr(agent, "_credential_pool", None),
        "request_overrides": dict(getattr(agent, "request_overrides", {}) or {}),
        "max_tokens": getattr(agent, "max_tokens", None),
        "command": getattr(agent, "acp_command", None),
        "args": list(getattr(agent, "acp_args", []) or []),
        "routed": False,
    }
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        return parent
    aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}
    task = aux.get("background_review", {}) if isinstance(aux.get("background_review"), dict) else {}
    task_provider = (str(task.get("provider", "")).strip() or None)
    task_model = (str(task.get("model", "")).strip() or None)
    task_base_url = (str(task.get("base_url", "")).strip() or None)
    task_api_key = (str(task.get("api_key", "")).strip() or None)
    if not (task_provider and task_provider != "auto" and task_model):
        return parent
    if task_provider == (agent.provider or "") and task_model == (agent.model or ""):
        return parent  # same model/provider as parent -> not routed
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        rp = resolve_runtime_provider(
            requested=task_provider,
            target_model=task_model,
            explicit_api_key=task_api_key,
            explicit_base_url=task_base_url,
        )
        return {
            "provider": rp.get("provider") or task_provider,
            "model": rp.get("model") or task_model,
            "api_key": rp.get("api_key"),
            "base_url": rp.get("base_url"),
            "api_mode": rp.get("api_mode"),
            "credential_pool": rp.get("credential_pool"),
            "request_overrides": dict(rp.get("request_overrides") or {}),
            "max_tokens": rp.get("max_output_tokens"),
            "command": rp.get("command"),
            "args": list(rp.get("args") or []),
            "routed": True,
        }
    except Exception as e:
        logger.debug("background-review aux routing failed (%s); using main model", e)
        return parent


def _msg_text(m: Dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        return " ".join(b.get("text", "") for b in c if isinstance(b, dict)).strip()
    return ""


def _digest_history(messages_snapshot: List[Dict], tail: int = 24) -> List[Dict]:
    """仅用于路由到不同模型（routed/different-model）路径的简短重放（replay）。

    原样保留最近的 ``tail`` 消息，
    将较早的对话轮次折叠为一条合成的用户角色摘要（digest），
    并维持角色的交替顺序。
    仅在路由到不同模型时使用（因为该模型的缓存无论如何都是冷的，
    所以减少冷写入的 Token 数量是纯粹的性能提升）。
    绝不用于主模型路径（主模型的完整重发能保持热缓存状态）。
    """
    msgs = list(messages_snapshot or [])
    if len(msgs) <= tail:
        return msgs
    keep = msgs[-tail:]
    while keep and isinstance(keep[0], dict) and keep[0].get("role") == "tool":
        tail += 1
        if len(msgs) <= tail:
            return msgs
        keep = msgs[-tail:]
    old = msgs[:-len(keep)]
    lines: List[str] = []
    for m in old:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        text = _msg_text(m).replace("\n", " ")
        if role == "user" and text:
            lines.append(f"USER: {text[:300]}")
        elif role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                names = [(tc.get("function") or {}).get("name", "?") for tc in tcs if isinstance(tc, dict)]
                lines.append(f"ASSISTANT[tools: {', '.join(names)}]")
            if text:
                lines.append(f"ASSISTANT: {text[:200]}")
    digest = {
        "role": "user",
        "content": (
            "[Earlier conversation digest — older turns summarised to bound the "
            "review's cold-write cost on the routed aux model. Recent turns "
            "follow verbatim below.]\n" + "\n".join(lines)
        ),
    }
    return [digest] + keep

# # 审查提示词字符串 —— 由 ``spawn_background_review_thread`` 使用，
# # 用于构建分叉（forked）出来的审查智能体所接收的用户消息。
# # AIAgent 将它们作为类属性（如 ``_MEMORY_REVIEW_PROMPT`` 等）暴露以兼容旧版本；
# # 实际文本存于此处，以便未来的修改能在单点集中完成。
# _MEMORY_REVIEW_PROMPT = (
#     "审查上述对话，并在适当情况下考虑保存到记忆中。\n\n"
#     "重点关注：\n"
#     "1. 用户是否透露了关于他们自己的信息 —— 例如个人画像、愿望、"
#     "偏好或值得记住的个人细节？\n"
#     "2. 用户是否对你应当如何表现、他们的工作风格，"
#     "或者他们希望你运作的方式表达过期望？\n\n"
#     "如果有值得注意的内容，请使用 memory 工具进行保存。"
#     "如果没有什么值得保存的，只需说 'Nothing to save.' 并停止。"
# )
# Review-prompt strings — used by ``spawn_background_review_thread`` to build
# the user-message that the forked review agent receives.  AIAgent exposes
# them as class attributes (``_MEMORY_REVIEW_PROMPT`` etc.) for back-compat;
# the actual text lives here so future edits are one-place.
_MEMORY_REVIEW_PROMPT = (
    "Review the conversation above and consider saving to memory if appropriate.\n\n"
    "Focus on:\n"
    "1. Has the user revealed things about themselves — their persona, desires, "
    "preferences, or personal details worth remembering?\n"
    "2. Has the user expressed expectations about how you should behave, their work "
    "style, or ways they want you to operate?\n\n"
    "If something stands out, save it using the memory tool. "
    "If nothing is worth saving, just say 'Nothing to save.' and stop."
)
# _SKILL_REVIEW_PROMPT = (
#     "审查上述对话并更新技能库。要保持**主动** —— "
#     "绝大多数会话都会产出至少一个技能更新（即使很微小）。"
#     "毫无作为的审查是一次错失的学习机会，而非中立的结果。\n\n"
#     "技能库的目标形态：**类级别（CLASS-LEVEL）**的技能，"
#     "每个技能包含丰富内容的 SKILL.md 以及用于存放特定会话细节的 `references/` 目录。"
#     "绝不要写成一串由狭隘的“单会话-单技能”构成的冗长平铺列表。"
#     "这决定了你**如何**更新，而不是**是否**更新。\n\n"
#     "需要寻找的信号（满足以下任意一条即值得采取行动）：\n"
#     "  • 用户纠正了你的风格、语气、格式、易读性或冗长程度。"
#     "诸如“停止做 X”、“太冗长了”、“不要这样排版”、"
#     "“为什么还在解释”、“直接给我答案”、“你总是做 Y 我很讨厌”、"
#     "或明确的“记住这个”等表达负面情绪的信号，都是**一等（FIRST-CLASS）**技能信号，"
#     "而不仅仅是记忆信号。请更新相关技能以嵌入这些偏好，"
#     "以便在下一次会话启动时就已经知晓。\n"
#     "  • 用户纠正了你的工作流、处理方法或步骤顺序。"
#     "将此纠正编码为管辖该类任务技能中的陷阱或显式步骤。\n"
#     "  • 涌现出了未来的会话能够从中受益的非同寻常的技术、修复方案、"
#     "变通办法（workaround）、调试路径或工具使用模式。捕获它们。\n"
#     "  • 本会话中被加载或查阅的某个技能经证实是错误的、缺少步骤的或过时的。"
#     "**立即**为其打补丁。\n\n"
#     "优先级顺序 —— 当触发上述信号时，优先选择最靠前的适用操作，但务必选择一项：\n"
#     "  1. **更新当前已加载的技能**。回顾对话中用户通过 /skill-name "
#     "或你通过 skill_view 读取的技能。如果其中有任何技能涵盖了新学习到的领域，"
#     "优先对该技能打**补丁**。因为它是正在起作用的技能，也是扩展内容的最佳位置。\n"
#     "  2. **更新现有的主技能（UMBRELLA）**（通过 skills_list + skill_view）。"
#     "如果没有已加载的技能适用，但现有的某个类级别技能适用，请对它打补丁。"
#     "添加一个子章节、一个陷阱说明，或扩大其触发条件。\n"
#     "  3. 在现有的主技能下**添加支持文件**。技能可以与三种支持文件打捆封装 —— "
#     "请根据类型使用对应的目录：\n"
#     "     • `references/<topic>.md` —— 特定会话的细节（错误文本转录、"
#     "重现方法、提供者特性）**以及**提炼后的知识库：引用的研究、API 文档、"
#     "外部权威摘录，或在解决问题时发现的领域笔记。"
#     "编写时要精简且聚焦于任务价值，而不是完整镜像上游文档。\n"
#     "     • `templates/<name>.<ext>` —— 旨在被复制和修改的起始模板文件"
#     "（样板配置、脚手架、智能体可以“带修改重现”的已验证示例）。\n"
#     "     • `scripts/<name>.<ext>` —— 技能可以直接调用的可静态重复运行的操作"
#     "（验证脚本、测试数据生成器、确定性探针，以及任何智能体应当直接运行而不是每次手动输入的脚本）。\n"
#     "     通过 skill_manage action=write_file 添加支持文件，"
#     "文件路径需以 'references/'、'templates/' 或 'scripts/' 开头。"
#     "主技能的 SKILL.md 中应添加一行指向新支持文件的指针，以便未来的智能体知道其存在。\n"
#     "  4. 在无任何现有技能涵盖该类别时，**创建新的类级别（CLASS-LEVEL）主技能**。"
#     "命名**必须**处于“类”的层面。名称**绝不能**是特定的 PR 编号、错误字符串、"
#     "功能代号、单独的库名，或是类似 'fix-X / debug-Y / audit-Z-today' "
#     "这种特定会话的产物。如果拟定的名称仅对今天的任务有意义，那它就是错误的 —— "
#     "请退回到方案 (1)、(2) 或 (3)。\n\n"
#     "用户偏好嵌入（重要）：当用户表达了风格/格式/工作流偏好时，"
#     "该更新属于 SKILL.md 正文，而不仅仅存在于记忆中。"
#     "memory捕获的是“用户是谁，以及当前的操作状况与状态如何”；"
#     "skill捕获的是“如何为该用户处理此类任务”。"
#     "当他们抱怨你处理任务的方式时，管辖该任务的技能需要吸收这一教训。\n\n"
#     "如果你注意到现有的两个技能存在重叠，请在回复中予以说明 —— "
#     "后台整理器（curator）会在大规模范围内处理合并事宜。\n\n"
#     "受保护的技能（**切勿编辑这些**）：\n"
#     "  • 内置技能（随 Hermes 一同发布的技能，例如 'hermes-agent'）。\n"
#     "  • Hub 安装的技能（通过 'hermes skills install' 安装）。\n"
#     "固定技能（通过 'hermes curator pin' 标记）**可以**被改进 —— "
#     "固定（pin）仅阻断整理器的删除/归档/合并操作，而不阻断内容更新。"
#     "当发现陷阱或缺失步骤时，请像对待其他智能体创建的技能一样为其打补丁。\n"
#     "如果唯一需要更新的技能是受保护的，请输入 'Nothing to save.' 并停止。\n\n"
#     "**切勿捕获**（这些会变成持续存在的自我限制，日后环境变化时会反噬自己）：\n"
#     "  • 依赖于环境的失败：缺少二进制文件、全新的安装错误、"
#     "迁移后的路径不匹配、'command not found'、未配置的凭据、未安装的包。"
#     "用户可以修复这些问题 —— 它们不是持久规则。\n"
#     "  • 关于工具或功能的否定陈述（如“浏览器工具无法工作”、"
#     "“X 工具损坏了”、“无法在 execute_code 中使用 Y”）。"
#     "在实际问题解决后的几个月里，这些陈述会固化为智能体用来自我引用的拒绝理由。\n"
#     "  • 会话特有的、在对话结束前已解决的临时错误。"
#     "如果重试成功了，教训在于“重试模式”，而不是最初的失败。\n"
#     "  • 一次性的任务叙述。用户要求“总结今天的市场”或“分析这个 PR”"
#     "并不属于需要建立技能的工作类别。\n\n"
#     "如果某个工具因配置状态而失败，请将**修复方法**（安装命令、配置步骤、"
#     "需要设置的环境变量）捕获在现有的安装或排错技能下 —— "
#     "绝不要将“该工具无法工作”作为独立的约束条款。\n\n"
#     "'Nothing to save.' 是一个真实的选项，但**不应**作为默认选择。"
#     "如果会话运行顺利，没有任何纠正，也没有产出新技术，只需说 'Nothing to save.' 并停止。"
#     "否则，请采取行动。"
# )
_SKILL_REVIEW_PROMPT = (
    "Review the conversation above and update the skill library. Be "
    "ACTIVE — most sessions produce at least one skill update, even if "
    "small. A pass that does nothing is a missed learning opportunity, "
    "not a neutral outcome.\n\n"
    "Target shape of the library: CLASS-LEVEL skills, each with a rich "
    "SKILL.md and a `references/` directory for session-specific detail. "
    "Not a long flat list of narrow one-session-one-skill entries. This "
    "shapes HOW you update, not WHETHER you update.\n\n"
    "Signals to look for (any one of these warrants action):\n"
    "  • User corrected your style, tone, format, legibility, or "
    "verbosity. Frustration signals like 'stop doing X', 'this is too "
    "verbose', 'don't format like this', 'why are you explaining', "
    "'just give me the answer', 'you always do Y and I hate it', or an "
    "explicit 'remember this' are FIRST-CLASS skill signals, not just "
    "memory signals. Update the relevant skill(s) to embed the "
    "preference so the next session starts already knowing.\n"
    "  • User corrected your workflow, approach, or sequence of steps. "
    "Encode the correction as a pitfall or explicit step in the skill "
    "that governs that class of task.\n"
    "  • Non-trivial technique, fix, workaround, debugging path, or "
    "tool-usage pattern emerged that a future session would benefit "
    "from. Capture it.\n"
    "  • A skill that got loaded or consulted this session turned out "
    "to be wrong, missing a step, or outdated. Patch it NOW.\n\n"
    "Preference order — prefer the earliest action that fits, but do "
    "pick one when a signal above fired:\n"
    "  1. UPDATE A CURRENTLY-LOADED SKILL. Look back through the "
    "conversation for skills the user loaded via /skill-name or you "
    "read via skill_view. If any of them covers the territory of the "
    "new learning, PATCH that one first. It is the skill that was in "
    "play, so it's the right one to extend.\n"
    "  2. UPDATE AN EXISTING UMBRELLA (via skills_list + skill_view). "
    "If no loaded skill fits but an existing class-level skill does, "
    "patch it. Add a subsection, a pitfall, or broaden a trigger.\n"
    "  3. ADD A SUPPORT FILE under an existing umbrella. Skills can be "
    "packaged with three kinds of support files — use the right "
    "directory per kind:\n"
    "     • `references/<topic>.md` — session-specific detail (error "
    "transcripts, reproduction recipes, provider quirks) AND "
    "condensed knowledge banks: quoted research, API docs, external "
    "authoritative excerpts, or domain notes you found while working "
    "on the problem. Write it concise and for the value of the task, "
    "not as a full mirror of upstream docs.\n"
    "     • `templates/<name>.<ext>` — starter files meant to be "
    "copied and modified (boilerplate configs, scaffolding, a "
    "known-good example the agent can `reproduce with modifications`).\n"
    "     • `scripts/<name>.<ext>` — statically re-runnable actions "
    "the skill can invoke directly (verification scripts, fixture "
    "generators, deterministic probes, anything the agent should run "
    "rather than hand-type each time).\n"
    "     Add support files via skill_manage action=write_file with "
    "file_path starting 'references/', 'templates/', or 'scripts/'. "
    "The umbrella's SKILL.md should gain a one-line pointer to any "
    "new support file so future agents know it exists.\n"
    "  4. CREATE A NEW CLASS-LEVEL UMBRELLA SKILL when no existing "
    "skill covers the class. The name MUST be at the class level. "
    "The name MUST NOT be a specific PR number, error string, feature "
    "codename, library-alone name, or 'fix-X / debug-Y / audit-Z-today' "
    "session artifact. If the proposed name only makes sense for "
    "today's task, it's wrong — fall back to (1), (2), or (3).\n\n"
    "User-preference embedding (important): when the user expressed a "
    "style/format/workflow preference, the update belongs in the "
    "SKILL.md body, not just in memory. Memory captures 'who the user "
    "is and what the current situation and state of your operations "
    "are'; skills capture 'how to do this class of task for this "
    "user'. When they complain about how you handled a task, the "
    "skill that governs that task needs to carry the lesson.\n\n"
    "If you notice two existing skills that overlap, note it in your "
    "reply — the background curator handles consolidation at scale.\n\n"
    "Protected skills (DO NOT edit these):\n"
    "  • Bundled skills (shipped with Hermes, e.g. 'hermes-agent').\n"
    "  • Hub-installed skills (installed via 'hermes skills install').\n"
    "Pinned skills (marked via 'hermes curator pin') CAN be improved — "
    "pin only blocks deletion/archive/consolidation by the curator, not "
    "content updates. Patch them when a pitfall or missing step turns up, "
    "same as any other agent-created skill.\n"
    "If the only skills that need updating are protected, say\n"
    "'Nothing to save.' and stop.\n\n"
    "Do NOT capture (these become persistent self-imposed constraints "
    "that bite you later when the environment changes):\n"
    "  • Environment-dependent failures: missing binaries, fresh-install "
    "errors, post-migration path mismatches, 'command not found', "
    "unconfigured credentials, uninstalled packages. The user can fix "
    "these — they are not durable rules.\n"
    "  • Negative claims about tools or features ('browser tools do not "
    "work', 'X tool is broken', 'cannot use Y from execute_code'). These "
    "harden into refusals the agent cites against itself for months "
    "after the actual problem was fixed.\n"
    "  • Session-specific transient errors that resolved before the "
    "conversation ended. If retrying worked, the lesson is the retry "
    "pattern, not the original failure.\n"
    "  • One-off task narratives. A user asking 'summarize today's "
    "market' or 'analyze this PR' is not a class of work that warrants "
    "a skill.\n\n"
    "If a tool failed because of setup state, capture the FIX (install "
    "command, config step, env var to set) under an existing setup or "
    "troubleshooting skill — never 'this tool does not work' as a "
    "standalone constraint.\n\n"
    "'Nothing to save.' is a real option but should NOT be the "
    "default. If the session ran smoothly with no corrections and "
    "produced no new technique, just say 'Nothing to save.' and stop. "
    "Otherwise, act."
)
#
# _COMBINED_REVIEW_PROMPT = (
#     "审查上述对话并更新以下两项内容：\n\n"
#     "**记忆（Memory）**：用户是谁。用户是否透露了个人画像、"
#     "愿望、偏好、个人细节，或者对你应当如何表现的期望？"
#     "请使用 memory 工具保存关于用户的相关事实和持久偏好。\n\n"
#     "**技能（Skills）**：如何处理此类任务。要保持**主动** —— "
#     "绝大多数会话都应至少产出一个技能更新。毫无作为的审查是一次错失的"
#     "学习机会，而非中立的结果。\n\n"
#     "技能库的目标形态：**类级别（CLASS-LEVEL）**的技能，包含丰富内容的 "
#     "SKILL.md 以及用于存放特定会话细节的 `references/` 目录。"
#     "绝不要写成一串由狭隘的“单会话-单技能”构成的冗长平铺列表。\n\n"
#     "触发技能更新的信号（满足任意一条即可）：\n"
#     "  • 用户纠正了你的风格、语气、格式、易读性、"
#     "冗长程度或处理方法。负面情绪是**一等（FIRST-CLASS）**技能信号，"
#     "而不仅仅是记忆信号。“停止做 X”、“不要这样排版”、"
#     "“我讨厌你做出 Y” —— 请将教训嵌入到管辖该任务的技能中，"
#     "以便在下一次会话启动时问题已被修复。\n"
#     "  • 涌现出了非同寻常的技术、修复方案、变通办法（workaround）或调试路径。\n"
#     "  • 某个被加载或查阅的技能经证实是错误的、缺失的或过时的 —— 立即为其打补丁。\n\n"
#     "技能的优先级顺序 —— 选择最靠前且匹配的一项：\n"
#     "  1. **更新当前已加载的技能**。检查对话中通过 /skill-name 或 skill_view "
#     "加载了哪些技能。如果其中某个技能涵盖了本次学习到的内容，优先对其打**补丁**。"
#     "因为它本就在发挥作用，这里就是最合适的位置。\n"
#     "  2. **更新现有的主技能（UMBRELLA）**（通过 skills_list + skill_view "
#     "找到合适的那一个）。对其打补丁。\n"
#     "  3. 通过 skill_manage action=write_file 在现有的主技能下**添加支持文件**。"
#     "包含三种类型：`references/<topic>.md` 用于存放特定会话的细节，"
#     "或编写精简且以任务为焦点的提炼知识库（引用的研究、API 文档摘要、领域笔记）；"
#     "`templates/<name>.<ext>` 用于存放旨在被复制和修改的起始模板文件；"
#     "`scripts/<name>.<ext>` 用于存放可静态重复运行的操作"
#     "（验证脚本、测试数据生成器、探针）。在 SKILL.md 中添加一行单行指针，"
#     "以便未来的智能体能够找到它们。\n"
#     "  4. 在无任何现有技能匹配时，**创建新的类级别（CLASS-LEVEL）主技能**。"
#     "从“类”的层面命名 —— **绝不要**使用 PR 编号、错误字符串、"
#     "代号、单独的库名，或是类似“fix-X / debug-Y”这种特定会话的产物。"
#     "如果命名仅适用于今天的任务，请退回到方案 (1)、(2) 或 (3)。\n\n"
#     "用户偏好嵌入：当用户抱怨你处理任务的方式时，更新管辖该任务的技能 —— "
#     "仅靠记忆是不够的。记忆回答的是“用户是谁，以及当前的操作状况与状态如何”；"
#     "技能回答的是“如何为该用户处理此类任务”。两者在相关时都应当承载用户偏好教训。\n\n"
#     "如果你注意到现有的技能存在重叠，请提出来 —— 后台整理器（curator）"
#     "会负责进行合并。\n\n"
#     "受保护的技能（**切勿编辑这些**）：\n"
#     "  • 内置技能（随 Hermes 一同发布的技能，例如 'hermes-agent'）。\n"
#     "  • Hub 安装的技能（通过 'hermes skills install' 安装）。\n"
#     "固定技能（通过 'hermes curator pin' 标记）**可以**被改进 —— "
#     "固定（pin）仅阻断整理器的删除/归档/合并操作，而不阻断内容更新。"
#     "当发现陷阱或缺失步骤时，请像对待其他智能体创建的技能一样为其打补丁。\n"
#     "如果唯一需要更新的技能是受保护的，请输入 'Nothing to save.' 并停止。\n\n"
#     "**切勿捕获**为技能（这些会变成持续存在的自我限制，"
#     "日后环境变化时会反噬自己）：\n"
#     "  • 依赖于环境的失败：缺少二进制文件、全新的安装错误、"
#     "迁移后的路径不匹配、'command not found'、未配置的凭据、未安装的包。"
#     "用户可以修复这些问题 —— 它们不是持久规则。\n"
#     "  • 关于工具或功能的否定陈述（如“浏览器工具无法工作”、"
#     "“X 工具损坏了”、“无法在 execute_code 中使用 Y”）。"
#     "在实际问题解决后的几个月里，这些陈述会固化为智能体用来自我引用的拒绝理由。\n"
#     "  • 会话特有的、在对话结束前已解决的临时错误。"
#     "如果重试成功了，教训在于“重试模式”，而不是最初的失败。\n"
#     "  • 一次性的任务叙述。用户要求“总结今天的市场”或“分析这个 PR”"
#     "并不属于需要建立技能的工作类别。\n\n"
#     "如果某个工具因配置状态而失败，请将**修复方法**（安装命令、配置步骤、"
#     "需要设置的环境变量）捕获在现有的安装或排错技能下 —— "
#     "绝不要将“该工具无法工作”作为独立的约束条款。\n\n"
#     "对包含真实有效信号的维度采取行动。如果两个维度上确实都无突出之处，"
#     "请输入 'Nothing to save.' 并停止 —— 但绝不要默认走向这一结论。"
# )
_COMBINED_REVIEW_PROMPT = (
    "Review the conversation above and update two things:\n\n"
    "**Memory**: who the user is. Did the user reveal persona, "
    "desires, preferences, personal details, or expectations about "
    "how you should behave? Save facts about the user and durable "
    "preferences with the memory tool.\n\n"
    "**Skills**: how to do this class of task. Be ACTIVE — most "
    "sessions produce at least one skill update. A pass that does "
    "nothing is a missed learning opportunity, not a neutral outcome.\n\n"
    "Target shape of the skill library: CLASS-LEVEL skills with a rich "
    "SKILL.md and a `references/` directory for session-specific detail. "
    "Not a long flat list of narrow one-session-one-skill entries.\n\n"
    "Signals that warrant a skill update (any one is enough):\n"
    "  • User corrected your style, tone, format, legibility, "
    "verbosity, or approach. Frustration is a FIRST-CLASS skill "
    "signal, not just a memory signal. 'stop doing X', 'don't format "
    "like this', 'I hate when you Y' — embed the lesson in the skill "
    "that governs that task so the next session starts fixed.\n"
    "  • Non-trivial technique, fix, workaround, or debugging path "
    "emerged.\n"
    "  • A skill that was loaded or consulted turned out wrong, "
    "missing, or outdated — patch it now.\n\n"
    "Preference order for skills — pick the earliest that fits:\n"
    "  1. UPDATE A CURRENTLY-LOADED SKILL. Check what skills were "
    "loaded via /skill-name or skill_view in the conversation. If one "
    "of them covers the learning, PATCH it first. It was in play; "
    "it's the right place.\n"
    "  2. UPDATE AN EXISTING UMBRELLA (skills_list + skill_view to "
    "find the right one). Patch it.\n"
    "  3. ADD A SUPPORT FILE under an existing umbrella via "
    "skill_manage action=write_file. Three kinds: "
    "`references/<topic>.md` for session-specific detail OR condensed "
    "knowledge banks (quoted research, API docs excerpts, domain "
    "notes) written concise and task-focused; `templates/<name>.<ext>` "
    "for starter files meant to be copied and modified; "
    "`scripts/<name>.<ext>` for statically re-runnable actions "
    "(verification, fixture generators, probes). Add a one-line "
    "pointer in SKILL.md so future agents find them.\n"
    "  4. CREATE A NEW CLASS-LEVEL UMBRELLA when nothing exists. "
    "Name at the class level — NOT a PR number, error string, "
    "codename, library-alone name, or 'fix-X / debug-Y' session "
    "artifact. If the name only fits today's task, fall back to (1), "
    "(2), or (3).\n\n"
    "User-preference embedding: when the user complains about how "
    "you handled a task, update the skill that governs that task — "
    "memory alone isn't enough. Memory says 'who the user is and "
    "what the current situation and state of your operations are'; "
    "skills say 'how to do this class of task for this user'. Both "
    "should carry user-preference lessons when relevant.\n\n"
    "If you notice overlapping existing skills, mention it — the "
    "background curator handles consolidation.\n\n"
    "Protected skills (DO NOT edit these):\n"
    "  • Bundled skills (shipped with Hermes, e.g. 'hermes-agent').\n"
    "  • Hub-installed skills (installed via 'hermes skills install').\n"
    "Pinned skills (marked via 'hermes curator pin') CAN be improved — "
    "pin only blocks deletion/archive/consolidation by the curator, not "
    "content updates. Patch them when a pitfall or missing step turns up, "
    "same as any other agent-created skill.\n"
    "If the only skills that need updating are protected, say\n"
    "'Nothing to save.' and stop.\n\n"
    "Do NOT capture as skills (these become persistent self-imposed "
    "constraints that bite you later when the environment changes):\n"
    "  • Environment-dependent failures: missing binaries, fresh-install "
    "errors, post-migration path mismatches, 'command not found', "
    "unconfigured credentials, uninstalled packages. The user can fix "
    "these — they are not durable rules.\n"
    "  • Negative claims about tools or features ('browser tools do not "
    "work', 'X tool is broken', 'cannot use Y from execute_code'). These "
    "harden into refusals the agent cites against itself for months "
    "after the actual problem was fixed.\n"
    "  • Session-specific transient errors that resolved before the "
    "conversation ended. If retrying worked, the lesson is the retry "
    "pattern, not the original failure.\n"
    "  • One-off task narratives. A user asking 'summarize today's "
    "market' or 'analyze this PR' is not a class of work that warrants "
    "a skill.\n\n"
    "If a tool failed because of setup state, capture the FIX (install "
    "command, config step, env var to set) under an existing setup or "
    "troubleshooting skill — never 'this tool does not work' as a "
    "standalone constraint.\n\n"
    "Act on whichever of the two dimensions has real signal. If "
    "genuinely nothing stands out on either, say 'Nothing to save.' "
    "and stop — but don't reach for that conclusion as a default."
)



def summarize_background_review_actions(
    review_messages: List[Dict],
    prior_snapshot: List[Dict],
    notification_mode: str = "on",
) -> List[str]:
    """构建针对后台审查阶段面向用户的操作摘要。

    遍历审查 Agent 的会话消息，收集成功的记忆
    与技能管理操作，以便展示给用户。
    已存在于 ``prior_snapshot`` 中的工具消息会被跳过，
    从而避免继承自过去的旧结果被重新当作新的后台任务展示（issue #14944）。

    ``notification_mode`` 用于控制显示的详细程度：
    - ``off``：不返回任何操作。
    - ``on``：通用的“Memory updated”（记忆已更新）/ 工具消息。
    - ``verbose``：包含来自工具调用参数的简短内容预览。
    """
    mode = str(notification_mode or "on").lower()
    if mode == "off":
        return []
    verbose = mode == "verbose"

    existing_tool_call_ids = set()
    existing_tool_contents = set()
    for prior in prior_snapshot or []:
        if not isinstance(prior, dict) or prior.get("role") != "tool":
            continue
        tcid = prior.get("tool_call_id")
        if tcid:
            existing_tool_call_ids.add(tcid)
        else:
            content = prior.get("content")
            if isinstance(content, str):
                existing_tool_contents.add(content)

    # 将审查 Agent 的工具执行结果映射回生成它们的调用（calls）。
    # 返回的 JSON 结果仅显示“Entry added”（条目已添加）；
    # 而调用参数中才包含具体的操作、目标以及内容预览。
    # 将范围限定在 notify_tools 内，还能防止辅助工具仅因执行成功
    # 就被误当作记忆更新操作展示出来。
    notify_tools = {"memory", "skill_manage"}
    all_tool_call_ids: set = set()
    call_details: dict = {}
    for msg in review_messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {}) or {}
            fn_name = fn.get("name", "")
            tcid = tc.get("id")
            if tcid:
                all_tool_call_ids.add(tcid)
            if fn_name not in notify_tools:
                continue
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            if tcid:
                call_details[tcid] = {
                    "tool": fn_name,
                    "action": args.get("action", "?"),
                    "target": args.get("target", "memory"),
                    "content": args.get("content", ""),
                    "old_text": args.get("old_text", ""),
                    "operations": args.get("operations") or [],
                    "name": args.get("name", ""),
                    "old_string": args.get("old_string", ""),
                    "new_string": args.get("new_string", ""),
                }

    actions: List[str] = []
    for msg in review_messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        tcid = msg.get("tool_call_id")
        if tcid and tcid in existing_tool_call_ids:
            continue
        if not tcid:
            content_str = msg.get("content")
            if isinstance(content_str, str) and content_str in existing_tool_contents:
                continue
        if tcid and all_tool_call_ids and tcid not in call_details:
            continue
        try:
            data = json.loads(msg.get("content", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
        # ``data`` 可能不是一个字典（dict）——在较旧的代码路径
        # 或包装器 MCP 服务器中，某些记忆/技能工具的响应会返回一个顶层的 JSON
        # 列表（例如 ``[{"success": true, ...}]``）或者标量。
        # 下方原始的 isinstance 检查会静默跳过非字典类型的载荷，
        # 这本身是正确的；但后文的 ``data.get("_change")`` 仍可能
        # 返回一个列表，从而导致 ``change.get("description", "")`` 报错崩溃。
        # 此处防御性地将所有内容归一化为一个字典类型的别名，
        # 从而让函数的后续逻辑保持简洁，无需在每次调用时都加上 ``isinstance`` 防护（#59437）。
        if not isinstance(data, dict) or not data.get("success"):
            continue
        message = data.get("message", "")
        detail = call_details.get(tcid) or {}
        if not isinstance(detail, dict):
            detail = {}
        target = data.get("target", "") or detail.get("target", "")
        is_skill = detail.get("tool") == "skill_manage"

        message_lower = message.lower()
        if not verbose:
            if "created" in message_lower:
                actions.append(message)
                continue
            if "updated" in message_lower:
                actions.append(message)
                continue
            if is_skill and "patched" in message_lower:
                actions.append(message)
                continue

        if is_skill:
            label = "Skill"
        elif target:
            label = "Memory" if target == "memory" else "User profile" if target == "user" else target
        else:
            continue

        if verbose:
            action = detail.get("action", "")
            content = detail.get("content", "")
            old_text = detail.get("old_text", "")
            skill_name = detail.get("name", "")
            # ``operations`` may be anything callable put into the JSON
            # arguments.  Anything non-iterable that isn't a list[str]
            # of dicts becomes unusable here, so coerce defensively.
            ops_raw = detail.get("operations")
            operations: list = (
                ops_raw if isinstance(ops_raw, list) else []
            )
            max_preview = 120
            if is_skill:
                # ``operations`` 可以是 JSON 参数中的任何可调用对象（callable）。
                # 任何非可迭代对象或不是由字典组成的 ``list[str]``
                # 在此处都会变得无法使用，因此要进行防御性类型强转。
                change_raw = data.get("_change")
                change: dict = (
                    change_raw if isinstance(change_raw, dict) else {}
                )
                old_string = (
                    change.get("old", "") or detail.get("old_string", "")
                )
                new_string = (
                    change.get("new", "") or detail.get("new_string", "")
                )
                description = change.get("description", "")
                if action == "patch" and (old_string or new_string):
                    old_preview = old_string[:80].replace("\n", " ") + (
                        "…" if len(old_string) > 80 else ""
                    )
                    new_preview = new_string[:80].replace("\n", " ") + (
                        "…" if len(new_string) > 80 else ""
                    )
                    actions.append(
                        f"📝 Skill '{skill_name}' patched: "
                        f"\"{old_preview}\" → \"{new_preview}\""
                    )
                elif action == "create" and description:
                    actions.append(f"📝 Skill '{skill_name}' created: {description}")
                elif action == "edit" and description:
                    actions.append(f"📝 Skill '{skill_name}' rewritten: {description}")
                else:
                    actions.append(f"📝 {message}" if message else f"Skill {action}")
            elif operations:
                for op in operations:
                    # ``_change`` 是技能工具（skill tool）在响应中留下的自由格式字典（dict）。
                    # 较旧的或包装器 MCP 后端会将其作为列表、整数或 JSON 标量返回 —
                    # 此处将其归一化为字典，以确保后续的 .get() 调用不会引发 AttributeError (#59437)。
                    if not isinstance(op, dict):
                        continue
                    op_act = op.get("action", "")
                    op_content = (op.get("content") or "")
                    op_old = (op.get("old_text") or "")
                    if op_act == "add" and op_content:
                        preview = op_content[:max_preview] + ("…" if len(op_content) > max_preview else "")
                        actions.append(f"{label} ➕ {preview}")
                    elif op_act == "replace" and op_content:
                        preview = op_content[:max_preview] + ("…" if len(op_content) > max_preview else "")
                        actions.append(f"{label} ✏️ {preview}")
                    elif op_act == "remove" and op_old:
                        preview = op_old[:60] + ("…" if len(op_old) > 60 else "")
                        actions.append(f"{label} ➖ {preview}")
            elif action == "add" and content:
                preview = content[:max_preview] + ("…" if len(content) > max_preview else "")
                actions.append(f"{label} ➕ {preview}")
            elif action == "replace" and content:
                preview = content[:max_preview] + ("…" if len(content) > max_preview else "")
                actions.append(f"{label} ✏️ {preview}")
            elif action == "remove" and old_text:
                preview = old_text[:60] + ("…" if len(old_text) > 60 else "")
                actions.append(f"{label} ➖ {preview}")
            else:
                actions.append(f"{label} updated")
        elif (
            "added" in message_lower
            or "replaced" in message_lower
            or "removed" in message_lower
            or "applied" in message_lower
            or (target and "add" in message.lower())
            or "Entry added" in message
        ):
            actions.append(f"{label} updated")
    return actions


def build_memory_write_metadata(
    agent: Any,
    *,
    write_origin: Optional[str] = None,
    execution_context: Optional[str] = None,
    task_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build provenance metadata for external memory-provider mirrors."""
    metadata: Dict[str, Any] = {
        "write_origin": write_origin or getattr(agent, "_memory_write_origin", "assistant_tool"),
        "execution_context": (
            execution_context
            or getattr(agent, "_memory_write_context", "foreground")
        ),
        "session_id": agent.session_id or "",
        "parent_session_id": agent._parent_session_id or "",
        "platform": agent.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
        "tool_name": "memory",
    }
    if task_id:
        metadata["task_id"] = task_id
    if tool_call_id:
        metadata["tool_call_id"] = tool_call_id
    return {k: v for k, v in metadata.items() if v not in {None, ""}}


def _run_review_in_thread(
    agent: Any,
    messages_snapshot: List[Dict],
    prompt: str,
) -> None:
    """在后台审查守护线程（daemon thread）中执行的工作函数（Worker function）。

    生成一个继承父级运行时（runtime）的派生（forked）``AIAgent``，
    运行审查提示词（prompt），并借助 ``agent._safe_print`` 和
    ``agent.background_review_callback`` 向用户输出简短的操作摘要。
    """
    # 局部导入（Local import），以避免在模块加载时发生硬循环依赖。
    from run_agent import AIAgent
    from tools.terminal_tool import set_approval_callback as _set_approval_callback

    # 在该工作线程上安装非交互式审批回调函数，
    # 使得审查 Agent 触发的任何危险命令防护机制
    # 都会直接判定为“拒绝”（deny），而不是回退到 input() ——
    # 后者会与父线程的 prompt_toolkit TUI 产生死锁（#15216）。
    # 此模式与 tools/delegate_tool.py 中的 _subagent_auto_deny 一致。
    def _bg_review_auto_deny(command, description, **kwargs):
        logger.warning(
            "Background review auto-denied dangerous command: %s (%s)",
            command, description,
        )
        return "deny"
    try:
        _set_approval_callback(_bg_review_auto_deny)
    except Exception:
        pass

    review_agent = None
    review_messages: List[Dict] = []
    try:
        # 仅对“当前”工作线程静默 stdout/stderr。
        # 如果在此处使用进程全局的 ``contextlib.redirect_stdout(devnull)``，
        # 会在整个审查期间（持续数十秒）同时清空所有其他线程的 ``sys.stdout``/``sys.stderr``
        # —— 包括驱动 Telegram 长轮询的网关事件循环线程 ——
        # 从而吞掉它们的控制台输出（#55769 / #55925）。
        # ``thread_scoped_silence`` 仅将当前线程的写入重定向到 devnull，
        # 并让所有其他线程继续保留在真实流上。
        with thread_scoped_silence():
            # 继承父 Agent 的实时运行时参数（provider、model、
            # base_url、api_key、api_mode），以确保 Fork 出来的子 Agent
            # 使用与主轮次完全相同的凭据。如果不这样做，
            # AIAgent.__init__ 会重新从环境变量中进行自动解析，
            # 这会导致仅支持 OAuth 的提供商、会话作用域凭据、
            # 或凭据池配置解析失败（因为解析器无法从头重建身份验证），
            # 从而在轮次结束时引发伪报错：“No LLM provider configured”。
            # _resolve_review_runtime() 默认返回父 Agent 的实时运行时参数
            # （routed=False；主模型，热缓存）；
            # 当用户将 auxiliary.background_review.{provider,model} 设置为其他模型时，
            # 则返回该模型的运行时参数（routed=True）。
            # codex_app_server -> codex_responses 的降级策略已在解析器内部处理完成。
            _rt = _resolve_review_runtime(agent)
            # 总结Agent是否需要路由新模型
            _routed = bool(_rt.get("routed"))
            # skip_memory=True 可以防止审查 Fork 出来的子 Agent
            # 触碰外部记忆插件（如 honcho、mem0、
            # supermemory 等）。如果没有设置此项，Fork 的
            # __init__ 会根据配置重新构建自己的 _memory_manager，
            # 并作用于父 Agent 的 session_id，随后
            # run_conversation() 会通过三个写入点将 Harness 提示词
            # 泄露到用户的真实记忆命名空间中：
            # on_turn_start（节奏 + 轮次消息）、prefetch_all（召回查询）、
            # 以及 sync_all（Harness 提示词 + 审查输出被记录为
            # (user, assistant) 轮次对）。
            # 内置的 MEMORY.md / USER.md 状态会在下方从
            # 父 Agent 重新绑定，因此来自审查的 memory(action="add") 写入
            # 仍会落盘；审查过程仅对外部提供商做到零副作用。
            # 匹配父 Agent 的工具集配置，使得请求体中的 ``tools[]``
            # 在字节层面上完全一致 —— Anthropic 的缓存键包含了该项。
            # （下方的运行时白名单仍会限制工具的分发调度。）
            _fork_kwargs: Dict[str, Any] = {}
            if isinstance(_rt.get("max_tokens"), int):
                _fork_kwargs["max_tokens"] = _rt["max_tokens"]
            if isinstance(_rt.get("command"), str) and _rt["command"]:
                _fork_kwargs["acp_command"] = _rt["command"]
                _fork_kwargs["acp_args"] = _rt.get("args") or []
            review_agent = AIAgent(
                model=_rt.get("model") or agent.model,
                max_iterations=16,
                quiet_mode=True,
                platform=agent.platform,
                provider=_rt.get("provider") or agent.provider,
                api_mode=_rt.get("api_mode"),
                base_url=_rt.get("base_url") or None,
                api_key=_rt.get("api_key") or None,
                credential_pool=_rt.get("credential_pool"),
                request_overrides=_rt.get("request_overrides") or {},
                parent_session_id=agent.session_id,
                enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                disabled_toolsets=getattr(agent, "disabled_toolsets", None),
                skip_memory=True,
                **_fork_kwargs,
            )
            review_agent._memory_write_origin = "background_review"
            review_agent._memory_write_context = "background_review"
            # 审查 Fork 出来的子 Agent 锁定了父级缓存的系统提示词（system prompt），
            # 并且保持 ``tools[]`` 在字节层面上与父级完全一致，
            # 从而使其发送的请求能够命中同一个提供商的缓存前缀（参见上方的工具集对齐说明）。
            # build_turn_context 中轮次间的 MCP 刷新机制
            # 可能会将后连入的 MCP 工具添加到此 Fork 中，进而破坏这种一致性，
            # 因此这里让审查 Fork 选择退出（opt out）该刷新机制。
            review_agent._skip_mcp_refresh = True
            review_agent._memory_store = agent._memory_store
            review_agent._memory_enabled = agent._memory_enabled
            review_agent._user_profile_enabled = agent._user_profile_enabled
            review_agent._memory_nudge_interval = 0
            review_agent._skill_nudge_interval = 0
            # 持久化隔离（curator接管问题 / curator-takeover 的根本原因）：
            # Fork 出来的子 Agent 共享了父 Agent 的 session_id（在下方设置，用于保持提示词缓存的热度），
            # 因此如果不做隔离，它会将自身的 Harness 轮次（“审查上述对话并更新技能库……”）
            # 以及其自身的响应直接写入 state.db 中用户的“真实”会话中。
            # 在用户的下一个实时轮次中，Agent 会将这条注入的用户消息重新读取为常驻指令（standing instruction），
            # 从而“变成”策展人（curator），并拒绝执行用户的实际任务。
            # _persist_disabled 会硬性阻止所有的数据库写入/延迟打开路径
            # （_flush_messages_to_session_db、_ensure_db_session、_get_session_db_for_recall）；
            # 审查过程仅通过其工具写入技能库和记忆存储，这已经完全满足其需求。
            review_agent._persist_disabled = True
            review_agent._session_db = None
            review_agent._session_json_enabled = False
            # 抑制 Fork 出来的子 Agent 发出的所有状态/警告输出，
            # 从而使用户仅能看到最终成功的操作摘要。
            # 如果不进行抑制，审查过程中的“达到迭代次数上限”（Iteration budget exhausted）、
            # 速率限制重试、压缩警告以及其他生命周期消息
            # 都会通过 _emit_status -> _vprint 向上冒泡，
            # 进而绕过 stdout 重定向发生泄露
            # （因为它们是通过 _print_fn/status_callback 输出的，这会绕过 sys.stdout）。
            review_agent.suppress_status_output = True
            # 逐字逐句继承父级缓存的系统提示词（system prompt），
            # 使得审查 Fork 发出的 HTTP 请求能够命中
            # 父级已经预热的同一 Anthropic/OpenRouter 前缀缓存（prefix cache）。
            # 如果不这样做，Fork 会从头重新构建系统提示词
            # （全新的 _hermes_now() 时间戳、全新的 session_id、
            # 较窄的工具集 → 不同的 skills_prompt），
            # 从而导致字节级完全匹配的前缀缓存键未命中。
            # 完整分析及实测影响（在 Sonnet 4.5 上实现了约 26% 的端到端成本降低）
            # 请参见 issue #25322 和 PR #17276。
            # 仅当审查运行在“同一模型”（未发生路由）上时，
            # 才共享父级预热缓存的系统提示词。
            # 当被路由至不同模型时，父级缓存的提示词对应的是错误的模型/缓存键，
            # 本就会导致缓存未命中，因此让路由后的 Fork 自行构建即可。
            if not _routed:
                review_agent._cached_system_prompt = agent._cached_system_prompt
                # 防御性措施：将 session_start + session_id 锁定为
                # 父级的对应值，以确保任何重新渲染系统提示词（system prompt）
                # 局部内容的代码路径（如压缩、插件钩子/plugin hooks）
                # 仍能生成在字节层面上完全一致的输出。
                # 上方的缓存提示词赋值过程虽然已经对正常的重新构建路径进行了短路处理（short-circuits），
                # 但这些锁定操作可以确保即使未来的代码路径绕过了缓存，
                # 依然能够保持输出的一致性。
                review_agent.session_start = agent.session_start
            review_agent.session_id = agent.session_id
            # 该 Fork 继承并共享了父级的实时 session_id（已在上方锁定，用于保持前缀缓存的一致性）。
            # 该 Fork 属于单次生命周期，并在执行完此 run_conversation() 后便会立即调用 close()；
            # 如果不选择退出（opting out），close() 将会在对话进行的中途
            # 终止/完结（finalize）父级依然处于活跃状态的会话行（因为后台审查大约每 10 个轮次就会触发一次）。
            # 请将会话完结的处理交由真正的拥有者（CLI 关闭 / 网关重置 / 定时任务 cron）。
            review_agent._end_session_on_close = False
            # 绝不要允许审查 Fork 进行压缩（compress）。
            # 它共享了父级的 session_id，因此如果它在压缩竞争中胜出，
            # 它会将父级轮转（rotate）为一个网关（gateway）永远不会接管的“新”子节点
            # （因为该 Fork 属于单次生命周期，在此次 run_conversation 结束后就会销毁）。
            # 前台轮次随后会从陈旧的父级重新开始并再次对其进行压缩，
            # 从而导致同一个父级下留有两个兄弟子节点（issue #38727）。
            # 此外，审查也需要完整的上下文来生成高质量的记忆/技能摘要 ——
            # 进行压缩会剥离这些细节。
            # conversation_loop.py 中的两个压缩触发点都受控于 agent.compression_enabled，
            # 因此将此项禁用可同时短路（short-circuit）这两条路径。
            review_agent.compression_enabled = False

            from model_tools import get_tool_definitions
            from hermes_cli.plugins import (
                set_thread_tool_whitelist,
                clear_thread_tool_whitelist,
            )

            # 根据 Profile（配置）中的 memory_enabled 标志来控制（Gate）内置 memory 工具的开启。
            # 硬编码 ["memory", "skills"] 会使得审查 LLM 即使在 Profile 设置了
            # memory_enabled: false 时，依然拥有 MEMORY.md 的读写工具，
            # 从而污染禁用了记忆功能的 Profile（#54937 第二层）。
            review_toolsets = ["skills"]
            if review_agent._memory_enabled or review_agent._user_profile_enabled:
                review_toolsets.insert(0, "memory")
            review_whitelist = {
                t["function"]["name"]
                for t in get_tool_definitions(
                    enabled_toolsets=review_toolsets,
                    quiet_mode=True,
                )
            }
            set_thread_tool_whitelist(
                review_whitelist,
                deny_msg_fmt=(
                    "Background review denied non-whitelisted tool: "
                    "{tool_name}. Only memory/skill tools are allowed."
                ),
            )
            try:
                from tools.skill_manager_tool import _reset_background_review_read_marks

                _reset_background_review_read_marks()
            except Exception:
                pass

            try:
                # 用户配置了总结模型，不使用主agent的内容
                # 路由至不同模型 -> 重放摘要（digest）
                # （反正该模型的缓存是冷的，因此要尽量减少冷写入的 Token 数）。
                # 路由至同一模型 -> 重放完整的快照（snapshot）
                # （属于热缓存读取）。
                _review_history = (
                    _digest_history(messages_snapshot) if _routed
                    else messages_snapshot
                )
                review_agent.run_conversation(
                    user_message=(
                        prompt
                        + "\n\nYou can only call memory and skill "
                        "management tools. Other tools will be denied "
                        "at runtime — do not attempt them."
                    ),
                    conversation_history=_review_history,
                )
            finally:
                clear_thread_tool_whitelist()

            # 在拆卸/清理（teardown）之前对审查操作进行快照记录。
            # close() 允许清理每个会话的状态（per-session state），
            # 但用于向用户展示的自我改进摘要
            # 依然需要已完成审查的 Agent 所返回的工具结果。
            review_messages = list(getattr(review_agent, "_session_messages", []))

            # 在 stdout 仍处于重定向状态时拆卸/清理记忆提供商（memory providers），
            # 以便后台线程的清理工作（Honcho 刷新、Hindsight 同步等）
            # 保持静默。
            # 下方的 finally 块是针对异常路径的安全防护网。
            try:
                review_agent.shutdown_memory_provider()
            except Exception:
                pass
            try:
                review_agent.close()
            except Exception:
                pass
            review_agent = None

        # 扫描审查 Agent 的消息以获取成功的工具操作，
        # 并向用户展示一份简短的摘要。
        # 必须跳过已经存在于 messages_snapshot 中的工具消息，
        # 因为审查 Agent 继承了这部分历史记录；
        # 否则它会重新展示先前对话中陈旧的“created”（已创建）/“updated”（已更新）消息，
        # 仿佛它们刚刚发生一样（issue #14944）。
        #
        # 包装在 try/except 中：针对异常/旧版的工具响应格式
        # （例如 ``_change`` 作为列表返回而非字典，#59437），
        # 决不能引发 AttributeError 并使整个审查过程崩溃，
        # 因为调用方的外层 except 只会记录“Background memory/skill review failed”（后台记忆/技能审查失败），
        # 并丢弃该 Fork 在崩溃前“确实”完成的所有成功操作。
        # 将捕获的异常强制转换为空的操作列表，
        # 从而返回消息中更早前完成的部分有效操作。
        try:
            actions = summarize_background_review_actions(
                review_messages,
                messages_snapshot,
                notification_mode=getattr(agent, "memory_notifications", "on"),
            )
        except Exception as e:
            # logger.warning(
            #     "summarize_background_review_actions 在发生异常后返回了部分结果"
            #     "（已按空结果处理）；正抑制此前曾导致"
            #     "整个审查中断的 AttributeError (#59437)：%s",
            #     e,
            # )
            logger.warning(
                "summarize_background_review_actions returned partial results "
                "after exception (treating as empty); suppressing AttributeError "
                "that previously aborted the entire review (#59437): %s",
                e,
            )
            actions = []

        if actions:
            summary = " · ".join(dict.fromkeys(actions))
            agent._safe_print(
                f"  💾 Self-improvement review: {summary}"
            )
            _bg_cb = agent.background_review_callback
            if _bg_cb:
                try:
                    _bg_cb(
                        f"💾 Self-improvement review: {summary}"
                    )
                except Exception:
                    pass

    except Exception as e:
        logger.warning("Background memory/skill review failed: %s", e)
        agent._emit_auxiliary_failure("background review", e)
    finally:
        # 异常路径的安全性清理（Safety-net cleanup）。
        # 正常完成时，已在上方线程作用域的静默机制（thread-scoped silence）中关闭。
        # 此处重新进入线程作用域静默，
        # 以确保拆卸/清理输出（Honcho 刷新、Hindsight 同步、后台线程 join）
        # 即使在发生异常的路径上也能保持静默，
        # 且不会清空其他线程的流。
        if review_agent is not None:
            try:
                with thread_scoped_silence():
                    try:
                        review_agent.shutdown_memory_provider()
                    except Exception:
                        pass
                    try:
                        review_agent.close()
                    except Exception:
                        pass
            except Exception:
                pass
        # Clear the approval callback on this bg-review thread so a
        # recycled thread-id doesn't inherit a stale reference.
        try:
            _set_approval_callback(None)
        except Exception:
            pass


def spawn_background_review_thread(
    agent: Any,
    messages_snapshot: List[Dict],
    review_memory: bool = False,
    review_skills: bool = False,
):
    """构建后台审查所需的线程目标函数（target）与提示词（prompt）。

    返回一个 ``(target, prompt)`` 元组。
    调用方（``AIAgent._spawn_background_review``）负责实际构建
    ``threading.Thread``，以便测试层面对 ``run_agent.threading.Thread``
    进行的补丁（patch）能够继续有效。
    """
    # 根据触发的条件选择合适的提示词。
    # 允许按 agent 进行覆盖（提示词已重构为模块级常量，
    # 但直接设置 agent._MEMORY_REVIEW_PROMPT 等属性的旧代码路径仍可正常工作）。
    if review_memory and review_skills:
        prompt = getattr(agent, "_COMBINED_REVIEW_PROMPT", _COMBINED_REVIEW_PROMPT)
    elif review_memory:
        # 针对USER.md , MEMORY.md 的修改提示词
        prompt = getattr(agent, "_MEMORY_REVIEW_PROMPT", _MEMORY_REVIEW_PROMPT)
    else:
        prompt = getattr(agent, "_SKILL_REVIEW_PROMPT", _SKILL_REVIEW_PROMPT)

    def _target() -> None:
        _run_review_in_thread(agent, messages_snapshot, prompt)

    return _target, prompt


__all__ = [
    "_MEMORY_REVIEW_PROMPT",
    "_SKILL_REVIEW_PROMPT",
    "_COMBINED_REVIEW_PROMPT",
    "spawn_background_review_thread",
    "summarize_background_review_actions",
    "build_memory_write_metadata",
]
