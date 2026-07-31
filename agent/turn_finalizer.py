"""Post-loop turn finalization for ``run_conversation``.

Extracted from ``agent/conversation_loop.py`` as part of the god-file
decomposition campaign (``~/.hermes/plans/god-file-decomposition.md``, Phase 1
step 4 — the post-loop ``TurnFinalizer`` seam). ``run_conversation``'s tail
(everything after the main tool-calling ``while`` loop) is lifted here verbatim:
budget-exhaustion summary, trajectory save, session persist, turn diagnostics,
response transforms, result-dict assembly, steer drain, and the memory/skill
review trigger.

Behavior-neutral: the body is moved unchanged. All ``agent.*`` side effects fire
exactly as before; only the post-loop *locals* are passed in as keyword args, and
the assembled ``result`` dict is returned to ``run_conversation`` which returns it
to the caller. The function is synchronous with a single return — mirroring the
region it replaces (no awaits, no early returns).

Module ``logger`` is imported lazily inside the body (``from
agent.conversation_loop import logger``) so this module never imports
``agent.conversation_loop`` at import time -> no import cycle, and the log records
keep the exact logger name (``"agent.conversation_loop"``).
"""

from __future__ import annotations

import os

from agent.codex_responses_adapter import _summarize_user_message_for_log


def finalize_turn(
    agent,
    *,
    final_response,
    api_call_count,
    interrupted,
    failed,
    messages,
    conversation_history,
    effective_task_id,
    turn_id,
    user_message,
    original_user_message,
    _should_review_memory,
    _turn_exit_reason,
    _pending_verification_response=None,
):
    """执行循环结束后的清理收尾工作，并返回轮次 (turn) 的 ``result`` 字典。

    此部分直接逐字提取自 ``run_conversation``（主 Agent 循环之后的区域）。
    具体请参阅模块文档字符串 (module docstring)。
    """
    from agent.conversation_loop import logger

    budget_exhausted = (
        api_call_count >= agent.max_iterations
        or agent.iteration_budget.remaining <= 0
    )
    budget_fallback_eligible = (
        budget_exhausted
        and not interrupted
        and not failed
        and str(_turn_exit_reason) in {"unknown", "budget_exhausted"}
    )
    continuation_budget_exhausted = (
        final_response is None
        and bool(_pending_verification_response)
        and budget_fallback_eligible
    )

    iteration_limit_fallback = False
    preserved_verification_fallback = False
    if continuation_budget_exhausted:
        # 验证/延续门控机制此前刻意保留了一个已生成的回答，
        # 随后的消耗用尽了剩余预算，未能生成更新的回答。
        # 此处直接保留该回答，而不是使用另一个可能出错的模型调用来替换它。
        # 明确的挂起值（pending value）起到了出处保护作用：
        # 无关的错误或恢复退出流程绝不可能进入此分支。
        final_response = _pending_verification_response
        _turn_exit_reason = f"max_iterations_reached({api_call_count}/{agent.max_iterations})"
        iteration_limit_fallback = True
        preserved_verification_fallback = True
    elif final_response is None and budget_fallback_eligible:
        # 预算已耗尽 —— 在剥离工具（tools）的前提下，
        # 通过额外发起一次 API 调用来让模型生成总结。
        # _handle_max_iterations 会注入一条用户消息，
        # 并执行单次不携带工具的请求。
        _turn_exit_reason = f"max_iterations_reached({api_call_count}/{agent.max_iterations})"
        agent._emit_status(
            f"⚠️ Iteration budget exhausted ({api_call_count}/{agent.max_iterations}) "
            "— asking model to summarise"
        )
        if not agent.quiet_mode:
            agent._safe_print(
                f"\n⚠️  Iteration budget exhausted ({api_call_count}/{agent.max_iterations}) "
                "— requesting summary..."
            )
        final_response = agent._handle_max_iterations(messages, api_call_count)
        iteration_limit_fallback = True

    if iteration_limit_fallback:
        # 如果作为看板工作节点（kanban worker）运行，
        # 则向调度器（dispatcher）发送该 worker 无法完成的信号
        # （而不是将其视为协议违规）。
        # 无论面向用户的降级处理是来自总结调用，
        # 还是来自明确挂起的延续逻辑，该规则均适用；
        # 这两种情况都耗尽了任务预算，因此必须触发失败电路演进。
        #
        # 我们通过 ``_record_task_failure(outcome="timed_out")`` 进行路由，
        # 而不是使用 ``kanban_block``，
        # 这样才能将其计入调度器的连续失败熔断机制（参见 #29747 gap 2）。
        _kanban_task = os.environ.get("HERMES_KANBAN_TASK")
        if _kanban_task:
            try:
                from hermes_cli import kanban_db as _kb
                _conn = _kb.connect()
                try:
                    _kb._record_task_failure(
                        _conn,
                        _kanban_task,
                        error=(
                            f"Iteration budget exhausted "
                            f"({api_call_count}/{agent.max_iterations}) — "
                            "task could not complete within the allowed "
                            "iterations"
                        ),
                        outcome="timed_out",
                        release_claim=True,
                        end_run=True,
                        event_payload_extra={
                            "budget_used": api_call_count,
                            "budget_max": agent.max_iterations,
                        },
                    )
                    logger.info(
                        "recorded budget-exhausted failure for task %s (%d/%d)",
                        _kanban_task, api_call_count, agent.max_iterations,
                    )
                finally:
                    try:
                        _conn.close()
                    except Exception:
                        pass
            except Exception:
                logger.warning(
                    "Failed to record budget-exhausted failure for task %s",
                    _kanban_task,
                    exc_info=True,
                )

    # Determine if conversation completed successfully
    normal_text_response = str(_turn_exit_reason).startswith("text_response(")
    completed = (
        final_response is not None
        and not failed
        and (
            api_call_count < agent.max_iterations
            or normal_text_response
        )
    )

    # 循环后的清理逻辑绝不能丢失响应。
    # 轨迹保存、资源销毁以及会话持久化等环节，均涉及易出错的操作 ——
    # 包括文件 I/O 与 JSON 序列化（_save_trajectory）、
    # 跨网络的远程虚拟机/浏览器销毁（_cleanup_task_resources），
    # 以及 SQLite 写入（_persist_session）。
    #
    # 此前，其中任何一步引发异常都会直接抛出 run_conversation 之外，
    # 从而丢弃调用方正在等待的局部 final_response
    # （导致子进程封装层只能捕获到没有任何 Traceback 的空标准输出 —— #8049）。
    #
    # 现在，每个步骤都已被独立防护：
    # 某一步的失败不会跳过后续步骤，
    # 且任何错误都会通过 ``cleanup_errors`` 呈现在结果字典中，
    # 而不会直接终止当前轮次（turn）。
    _cleanup_errors = []

    # 若已启用，则保存运行轨迹。
    # ``user_message`` 可能是一个包含多模态组件的列表；
    # 而轨迹格式要求提供纯字符串。
    try:
        agent._save_trajectory(messages, _summarize_user_message_for_log(user_message), completed)
    except Exception as _save_err:
        _cleanup_errors.append(f"save_trajectory: {_save_err}")
        logger.error("finalize_turn: _save_trajectory failed: %s", _save_err, exc_info=True)

    # Clean up VM and browser for this task after conversation completes
    try:
        # TODO KEY 关闭vm sandBox browser下一轮的关键
        agent._cleanup_task_resources(effective_task_id)
    except Exception as _cleanup_err:
        _cleanup_errors.append(f"cleanup_task_resources: {_cleanup_err}")
        logger.error("finalize_turn: _cleanup_task_resources failed: %s", _cleanup_err, exc_info=True)

    # 只有在移除私有重试（private retry）骨架逻辑之后，
    # 才将会话持久化写入 JSON 日志和 SQLite。
    # 否则，后续用户的“继续（continue）”轮次
    # 可能会重放 assistant("(empty)") 或恢复提示语，
    # 从而再次陷入相同的空响应循环中。
    try:
        agent._drop_trailing_empty_response_scaffolding(messages)

        # 当当前轮次（turn）被中断且最后一条消息是工具执行结果（tool result）时，
        # 追加一条合成的 assistant 消息来结束该工具调用序列。
        # 如果不进行此补全，会话持久化后将留存 ``tool → user`` 的交替关系，
        # 严格遵循协议的提供商（如 Gemini、Claude）会拒绝这种格式，
        # 从而导致它们在下一轮对话中对用户消息的续写产生幻觉（#48879）。
        #
        # ``_drop_trailing_empty_response_scaffolding`` 仅在存在空响应骨架标记时，
        # 才会倒回并清理工具尾部；
        # 而成功执行工具后的正常 ``/stop`` 中断不会设置此类标记，
        # 因此工具结果会保留在尾部，我们需要在此处对其进行闭合。
        # 在发生中断时，``final_response`` 通常为空，
        # 因此会降级使用明确的占位符，而不是持久化一个内容为空的 assistant 轮次。
        if interrupted:
            from agent.message_sanitization import close_interrupted_tool_sequence
            close_interrupted_tool_sequence(messages, final_response)

        # 某些恢复/降级路径会返回一个真实的 final_response，
        # 但不会向对话记录（transcript）中添加闭合的 assistant 消息
        # （例如 ``conversation_loop`` 中部分流（partial-stream）
        # 以及前一轮内容（prior-turn-content）恢复时的 ``break`` 位置）。
        # 如果原样进行持久化，持久化会话（durable session）可能会停留在 tool/user 消息处，
        # 哪怕调用方 —— 以及网关平台 —— 已经看到了完成的 assistant 响应。
        # 随后在下一轮对话中，系统会重放仅包含 user 的积压消息，
        # 导致模型对每一条“未解答的”消息重新进行回答。
        # 因此，需要在源头（即所有恢复 ``break`` 流程都会经过的单一关口）闭合该持久化轮次，
        # 从而确保无论是由哪条路径生成的响应，
        # 都能满足“已交付 final_response ⇒ 对话记录中存在对应的 assistant 行”这一不变性（invariant）。
        # （#43849 / #44100）
        if final_response and not interrupted:
            try:
                _tail_role = messages[-1].get("role") if messages else None
            except Exception:
                _tail_role = None
            if _tail_role != "assistant":
                messages.append({"role": "assistant", "content": final_response})
        # TODO KEY 存储message到db
        agent._persist_session(messages, conversation_history)
    except Exception as _persist_err:
        _cleanup_errors.append(f"persist_session: {_persist_err}")
        logger.error("finalize_turn: _persist_session failed: %s", _persist_err, exc_info=True)

    # ── 回合退出诊断日志 ─────────────────────────────────────
    # 始终以 INFO 级别记录，
    # 以便 agent.log 能够捕获每个回合结束的原因。
    #
    # 当最后一条消息是工具结果时（表明智能体正处于工作的中途），
    # 则以 WARNING 级别进行记录 ——
    # 这正是用户所报告的“突然停止”的场景。
    _last_msg_role = messages[-1].get("role") if messages else None
    _last_tool_name = None
    if _last_msg_role == "tool":
        # Walk back to find the assistant message with the tool call
        for _m in reversed(messages):
            if _m.get("role") == "assistant" and _m.get("tool_calls"):
                _tcs = _m["tool_calls"]
                if _tcs and isinstance(_tcs[0], dict):
                    _last_tool_name = _tcs[-1].get("function", {}).get("name")
                break

    _turn_tool_count = sum(
        1 for m in messages
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
    )
    _resp_len = len(final_response) if final_response else 0
    _budget_used = agent.iteration_budget.used if agent.iteration_budget else 0
    _budget_max = agent.iteration_budget.max_total if agent.iteration_budget else 0

    _diag_msg = (
        "Turn ended: reason=%s model=%s api_calls=%d/%d budget=%d/%d "
        "tool_turns=%d last_msg_role=%s response_len=%d session=%s"
    )
    _diag_args = (
        _turn_exit_reason, agent.model, api_call_count, agent.max_iterations,
        _budget_used, _budget_max,
        _turn_tool_count, _last_msg_role, _resp_len,
        agent.session_id or "none",
    )

    if _last_msg_role == "tool" and not interrupted:
        # Agent was mid-work — this is the "just stops" case.
        logger.warning(
            "Turn ended with pending tool result (agent may appear stuck). "
            + _diag_msg + " last_tool=%s",
            *_diag_args, _last_tool_name,
        )
    else:
        logger.info(_diag_msg, *_diag_args)

    # 文件修改验证器页脚。
    #
    # 如果在本回合中，一个或多个 ``write_file`` / ``patch`` 调用失败，
    # 并且随后没有对同一路径成功执行写入操作来覆盖它们，
    # 则在助手的回复末尾追加一个提示性页脚。
    #
    # 此举旨在捕获一种特定情况（由 Ben Eng 报告，与 #15524 相关）——
    # 即模型发出了一批并行的'/patch'操作，
    # 其中一半因“找不到旧字符串 (Could not find old_string)”而失败，
    # 但模型在总结本回合时，却声称每个文件都已编辑成功。
    # 随后用户不得不手动运行 ``git status`` 来戳穿这个谎言。
    #
    # 有了这个页脚，每个回合都会直接呈现真实情况，
    # 从而在结构上杜绝了模型过度夸大事实的可能性。
    #
    # 限制条件：仅当本回合存在实质性的文本回复，
    # 且用户未进行打断时才会应用。
    # 空回合或被打断的回合已经包含了其他的表层文本，
    # 不应再对其进行内容追加。
    if final_response and not interrupted:
        try:
            _failed = getattr(agent, "_turn_failed_file_mutations", None) or {}
            if _failed and agent._file_mutation_verifier_enabled():
                footer = agent._format_file_mutation_failure_footer(_failed)
                if footer:
                    final_response = final_response.rstrip() + "\n\n" + footer
        except Exception as _ver_err:
            logger.debug("file-mutation verifier footer failed: %s", _ver_err)

    # 对“轮次完成（Turn-completion）”原因的说明。
    # 当一个轮次在执行了实质性工作后异常结束 —— 例如重试后内容仍为空、
    # 响应流截断/部分缺失、工具结果仍在等待中、或者超出了迭代/预算限制 ——
    # 如果不加以处理，用户只会看到一个空白或碎片化的响应框，
    # 无法了解智能体停止工作的综合原因（参见 #34452）。
    # 此处借鉴上方文件变更验证器页脚的模式，向用户展示由 ``_turn_exit_reason``
    # 推导出的单一可视化说明。
    #
    # 严格控制触发条件，确保正常轮次保持简洁静默：
    #   - 经由 ``text_response(...)`` 退出时绝不生成说明
    #     （在格式化器内部处理），因此简短的“Done.”是无感的。
    #   - 我们仅在本轮没有真正可用回复时才采取动作：
    #     响应为空、出现终止标记 "(empty)"、或者仅有极短且缺少末尾标点
    #     的碎片片段（例如 "The"）。对于正常的简短回答，则保留其原有文本。
    if not interrupted:
        try:
            if agent._turn_completion_explainer_enabled():
                _stripped = (final_response or "").strip()
                _is_empty_terminal = _stripped == "" or _stripped == "(empty)"
                # A short fragment that is not a normal text_response exit
                # and lacks sentence-ending punctuation is treated as a
                # truncated partial (the "The" case from #34452).
                _is_partial_fragment = (
                    not _is_empty_terminal
                    and not preserved_verification_fallback
                    and not str(_turn_exit_reason).startswith("text_response")
                    and len(_stripped) <= 24
                    and _stripped[-1:] not in {".", "!", "?", "。", "！", "？", "`", ")"}
                )
                _is_partial_stream_recovery = (
                    str(_turn_exit_reason) == "partial_stream_recovery"
                )
                if (
                    _is_empty_terminal
                    or _is_partial_fragment
                    or _is_partial_stream_recovery
                ):
                    _explanation = agent._format_turn_completion_explanation(
                        _turn_exit_reason
                    )
                    if _explanation:
                        if _is_empty_terminal:
                            # Replace the bare "(empty)"/blank sentinel with
                            # the actionable explanation.
                            final_response = _explanation
                        else:
                            # Keep the partial fragment, append the reason so
                            # the user sees both what arrived and why it
                            # stopped.
                            final_response = (
                                _stripped + "\n\n" + _explanation
                            )
        except Exception as _exp_err:
            logger.debug("turn-completion explainer failed: %s", _exp_err)

    _response_transformed = False

    # 插件钩子：transform_llm_output
    # 在工具调用循环完成后，每轮触发一次。
    # 插件可以在 LLM 的输出文本被返回之前对其进行转换。
    # 第一个返回字符串的钩子将胜出；返回 None 或空值则保持文本不变。
    if final_response and not interrupted:
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _transform_results = _invoke_hook(
                "transform_llm_output",
                response_text=final_response,
                session_id=agent.session_id or "",
                model=agent.model,
                platform=getattr(agent, "platform", None) or "",
            )
            for _hook_result in _transform_results:
                if isinstance(_hook_result, str) and _hook_result:
                    final_response = _hook_result
                    _response_transformed = True
                    break  # First non-empty string wins
        except Exception as exc:
            logger.warning("transform_llm_output hook failed: %s", exc)

    # 插件钩子：post_llm_call
    # 在工具调用循环完成后，每轮触发一次。
    # 插件可以使用此钩子来持久化对话数据
    # （例如同步到外部记忆系统中）。
    if final_response and not interrupted:
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _invoke_hook(
                "post_llm_call",
                session_id=agent.session_id,
                task_id=effective_task_id,
                turn_id=turn_id,
                user_message=original_user_message,
                assistant_response=final_response,
                conversation_history=list(messages),
                model=agent.model,
                platform=getattr(agent, "platform", None) or "",
            )
        except Exception as exc:
            logger.warning("post_llm_call hook failed: %s", exc)

    # 仅提取来自“当前”轮次的推理内容（reasoning）。
    # 向后遍历，但止于触发本轮次的用户消息 ——
    # 任何早于该消息的内容均来自先前的轮次，切勿泄漏至推理框中
    # （以免造成陈旧信息的展示混淆；参见 #17055）。
    # 在当前轮次内，我们仍希望获取“最新”的非空推理内容：
    # 许多提供者（如 Claude thinking、DeepSeek v4、Codex Responses）
    # 会在工具调用步骤中输出推理，而在最终回答步骤中留下 reasoning=None；
    # 因此，若仅选择最后一个助手消息，将会无意间丢弃同轮次中合规的推理内容。
    last_reasoning = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            break  # turn boundary — don't cross into prior turns
        if msg.get("role") == "assistant" and msg.get("reasoning"):
            last_reasoning = msg["reasoning"]
            break

    # Build result with interrupt info if applicable
    result = {
        "final_response": final_response,
        "last_reasoning": last_reasoning,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": completed,
        "turn_exit_reason": _turn_exit_reason,
        "failed": failed,
        "partial": False,  # True only when stopped due to invalid tool calls
        "interrupted": interrupted,
        "response_transformed": _response_transformed,
        "response_previewed": getattr(agent, "_response_was_previewed", False),
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        "input_tokens": agent.session_input_tokens,
        "output_tokens": agent.session_output_tokens,
        "cache_read_tokens": agent.session_cache_read_tokens,
        "cache_write_tokens": agent.session_cache_write_tokens,
        "reasoning_tokens": agent.session_reasoning_tokens,
        "prompt_tokens": agent.session_prompt_tokens,
        "completion_tokens": agent.session_completion_tokens,
        "total_tokens": agent.session_total_tokens,
        "last_prompt_tokens": getattr(agent.context_compressor, "last_prompt_tokens", 0) or 0,
        "estimated_cost_usd": agent.session_estimated_cost_usd,
        "cost_status": agent.session_cost_status,
        "cost_source": agent.session_cost_source,
        # Requested service tier (from request_overrides.extra_body), for
        # billing audits by callers like `hermes -z --usage-file`.
        "service_tier": (
            (getattr(agent, "request_overrides", {}) or {}).get("extra_body") or {}
        ).get("service_tier"),
        "session_id": agent.session_id,
    }
    if agent._tool_guardrail_halt_decision is not None:
        result["guardrail"] = agent._tool_guardrail_halt_decision.to_metadata()
    # 暴露循环结束后的所有清理失败信息，以便调用方区分
    # 正常结束的轮次与那些在轨迹/会话/资源销毁时引发异常的轮次
    # （无论属于哪种情况，响应都会照常返回 —— 参见 #8049）。
    if _cleanup_errors:
        result["cleanup_errors"] = _cleanup_errors
    # 如果在最后一个助手轮次之后收到了 /steer 指令
    # （即已经没有可以继续处理的工具批次），
    # 则将其交还给调用方，以便作为下一个用户轮次发送，
    # 而不是静默丢弃。
    _leftover_steer = agent._drain_pending_steer()
    if _leftover_steer:
        result["pending_steer"] = _leftover_steer
    agent._response_was_previewed = False

    # Include interrupt message if one triggered the interrupt
    if interrupted and agent._interrupt_message:
        result["interrupt_message"] = agent._interrupt_message

    # Clear interrupt state after handling
    agent.clear_interrupt()

    # Clear stream callback so it doesn't leak into future calls
    agent._stream_callback = None

    # 立即检查技能触发条件 ——
    # 基于本轮次（THIS turn）已使用的工具迭代次数。
    _should_review_skills = False
    if (agent._skill_nudge_interval > 0
            and agent._iters_since_skill >= agent._skill_nudge_interval
            and "skill_manage" in agent.valid_tool_names):
        _should_review_skills = True
        agent._iters_since_skill = 0

    # TODO KEY 外部memory provider
    # 外部记忆提供者：同步已完成的轮次，并对下一次预取进行排队。
    agent._sync_external_memory_for_turn(
        original_user_message=original_user_message,
        final_response=final_response,
        interrupted=interrupted,
        messages=messages,
    )

    # 后台记忆/技能审查 —— 在响应交付后运行，
    # 因此绝不会与用户的任务竞争模型的注意力。
    if final_response and not interrupted and (_should_review_memory or _should_review_skills):
        try:
            # TODO KEY 后台审查线程
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=_should_review_memory,
                review_skills=_should_review_skills,
            )
        except Exception:
            pass  # Background review is best-effort

    # 注意：此处切勿调用记忆提供者（Memory provider）的 on_session_end()
    # 与 shutdown_all() —— 在多轮对话会话中，每收到一条用户消息，
    # run_conversation() 都会被调用一次。如果在每个轮次结束后就进行关闭，
    # 会导致提供者在第二条消息到达之前就被销毁。
    # 实际的会话结束清理工作是由 CLI（atexit / /reset）
    # 以及网关（会话过期 / _reset_session）来负责处理的。

    # 插件钩子：on_session_end
    # 在每次 run_conversation 调用的最末端触发。
    # 插件可以使用此钩子来进行清理工作、刷新缓冲区等操作。
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        _invoke_hook(
            "on_session_end",
            session_id=agent.session_id,
            task_id=effective_task_id,
            turn_id=turn_id,
            completed=completed,
            interrupted=interrupted,
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
        )
    except Exception as exc:
        logger.warning("on_session_end hook failed: %s", exc)

    return result
