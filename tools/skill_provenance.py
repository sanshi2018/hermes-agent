"""技能写入来源溯源 — 用于区分智能体沉淀的技能写入与前台用户直接发起的技能写入的 ContextVar。

管理员（curator）仅对通过后台自我改进审查分支自主创建的技能进行整合/修剪。
用户要求前台智能体编写的技能属于用户，绝不能被自动管理。

本模块公开了一个 ContextVar，由 run_agent.py 在每个工具循环之前设置，
以便工具处理程序（例如 skill_manage create）可以检查它们是否在后台审查分支内部执行。

该信号依托于 AIAgent._memory_write_origin，
对于审查分支实例，该属性已被设置为 "background_review"
（参见 run_agent.py 中的 _spawn_background_review），
而对于普通（前台）智能体，则默认值为 "assistant_tool"。

用法示例：
    from tools.skill_provenance import (
        set_current_write_origin,
        reset_current_write_origin,
        get_current_write_origin,
    )

    token = set_current_write_origin("background_review")
    try:
        ...  # 工具在此处运行
    finally:
        reset_current_write_origin(token)

    # 在工具内部：
    if get_current_write_origin() == "background_review":
        mark_agent_created(skill_name)
"""

import contextvars


_write_origin: contextvars.ContextVar[str] = contextvars.ContextVar(
    "skill_write_origin",
    default="foreground",
)

# The sentinel value the background review fork uses; mirrors
# run_agent.py's AIAgent._memory_write_origin override in
# _spawn_background_review().
BACKGROUND_REVIEW = "background_review"


def set_current_write_origin(origin: str) -> contextvars.Token[str]:
    """Bind the active write origin to the current context.

    Returns a Token the caller must pass to reset_current_write_origin
    in a finally block.
    """
    return _write_origin.set(origin or "foreground")


def reset_current_write_origin(token: contextvars.Token[str]) -> None:
    """Restore the prior write origin context."""
    _write_origin.reset(token)


def get_current_write_origin() -> str:
    """返回当前处于激活状态的写入来源（write origin）。

    默认值："foreground" —— 任何由常规（非审查）Agent、
    命令行界面（CLI）、网关（gateway）、定时任务（cron）或子 Agent 调用的工具。

    "background_review" —— 自我改进的审查分支（review fork）；
    仅在此来源下创建的 Skill 才会被标记为 Agent 创建，
    以供 Curator 进行管理。
    """
    return _write_origin.get()


def is_background_review() -> bool:
    """Convenience: True iff the current write origin is the background
    review fork."""
    return get_current_write_origin() == BACKGROUND_REVIEW
