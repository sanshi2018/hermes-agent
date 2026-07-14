# Pre-API 上下文压缩逻辑深度解析

> 分析对象：`agent/conversation_loop.py` 第 985 行起的 **Pre-API pressure check（API 调用前压力检查）** 代码块（约 968–1043 行）。
> 本文梳理其完整调用链路，总结设计目的、解决的问题与解决方案。

---

## 1. 这段代码在做什么（一句话）

在**每一次真正发起模型 API 调用之前**，用"消息 + 系统提示词 + 工具 schema"的粗略 token 估算值重新检查一次上下文压力；若已逼近上下文/输出上限，则先对会话历史做一次有损摘要压缩（compaction），再退还本次迭代计数并 `continue` 重试，从而避免请求被服务商以"上下文超限"拒绝。

```python
# conversation_loop.py:985 起（核心骨架）
_compressor = agent.context_compressor
if (
    agent.compression_enabled                       # 压缩功能开启
    and len(messages) > 1                           # 有可压缩的历史
    and compression_attempts < 3                    # 每回合硬性上限（防死循环）
    and not _defer_preflight(request_pressure_tokens)  # 粗估已知偏高时让位给真实用量
    and not _compression_cooldown                   # 摘要失败冷却期内不重试
    and _compressor.should_compress(request_pressure_tokens)  # 权威阈值判断
):
    compression_attempts += 1
    messages, active_system_prompt = agent._compress_context(...)
    conversation_history = conversation_history_after_compression(agent, messages)
    api_call_count -= 1                             # 本次不算 API 调用
    agent.iteration_budget.refund()                 # 退还迭代预算
    continue                                        # 用压缩后的历史重新进入循环
```

---

## 2. 为什么需要这个检查点 —— 三道压缩闸门的分工

Hermes 的自动压缩体系一共有**三个检查点**，本段代码是第二个，专门补另外两个的盲区：

| 检查点 | 位置 | 依据的 token 信号 | 盲区 |
|---|---|---|---|
| ① 回合开场预检（turn-prologue preflight） | `turn_context.py:350-462` | `estimate_request_tokens_rough`（粗估） | 只在**用户消息进来时**跑一次；看不到本回合后续工具循环中新增的大量工具结果 |
| ② **Pre-API 压力检查（本段代码）** | `conversation_loop.py:968-1043` | `estimate_request_tokens_rough`（粗估，含工具 schema） | —— 就是为补①③的盲区而生 |
| ③ 工具循环尾部 post-response 闸门 | `conversation_loop.py:4732-4763` | API 返回的真实 `prompt_tokens` | 该值**滞后一拍**：它反映的是上一次请求的大小，不包含"刚刚追加的巨大工具结果" |

代码注释里点名的真实事故（"the live 271k/272k Codex failure"）正是这个盲区的具体化：

> 单个回合内，模型连续调用工具，一次性追加了多个巨大的工具结果。回合开场预检早已通过（那时只有用户消息），而检查点③用的是 API 上报的 `last_prompt_tokens`——它是**上一次**请求的用量，尚未计入刚追加的工具结果。于是下一次 API 调用以 ~271K tokens 撞上 272K 的上下文墙，请求直接被拒。

因此在"即将发请求"这个最后时刻，用**当前请求的即时估算值**再查一次，是唯一能兜住该场景的位置。

---

## 3. 完整调用链路

### 3.1 入口前的准备（conversation_loop.py:900-949）

1. **消息规范化**（900-932 行）：对 `api_messages`（发给 API 的副本）做空白裁剪、tool-call 参数 JSON 规范化（紧凑分隔符 + key 排序），保证跨轮次前缀位级一致，最大化 KV 缓存 / 提供商缓存命中。
2. **代理字符净化**（938 行）：`_sanitize_messages_surrogates()`（`message_sanitization.py`）清除孤立 surrogate 字符，防止 OpenAI SDK 内部 `json.dumps()` 崩溃引发重试死循环。
3. **压力估算**（945-949 行）：
   - `estimate_messages_tokens_rough(api_messages)`（`model_metadata.py:2337`）：字符数 ÷ 4 的粗估；base64 图片按每张 ~1500 token 固定计费，避免 1MB 截图被误估为 25 万 token 而触发过早压缩。
   - `estimate_request_tokens_rough(api_messages, tools=agent.tools)`（`model_metadata.py:2413`）：在消息之上**加上工具 schema 的开销**。启用 50+ 工具时 schema 独占 20–30K token，只算消息会留下显著盲区（#14695）。压缩决策用的是这个带工具的值 `request_pressure_tokens`。

### 3.2 守卫链（conversation_loop.py:992-998）—— 与回合开场预检完全镜像

按顺序短路求值，任一不满足即跳过压缩、直接发请求：

**① `compression_attempts < 3` — 每回合硬性退避**
该计数器与下方的上下文溢出错误处理器（400 context-length-exceeded ≈ 3443/3557 行、413 payload-too-large ≈ 3333 行）**共享**，作为整回合的压缩次数总闸。每次成功的正常 API 响应会将其清零（多处 `compression_attempts = 0`）。防止"压缩→仍超限→再压缩"的无限循环。

**② `should_defer_preflight_to_real_usage(rough_tokens)`（`context_compressor.py:1131`）— 粗估噪声抑制（#36718）**

粗估器为了安全**故意高估** schema 重的请求。但如果服务商刚刚用真实 `prompt_tokens` 证明了"上一次请求其实在阈值以下"，就不该因为同样的 schema 高估反复触发压缩。逻辑：

- 压缩刚完成、真实用量还没回来时（`awaiting_real_usage_after_compression=True`，此时 `last_prompt_tokens` 被置为 -1 哨兵值），无条件推迟一回合——否则压缩前的陈旧 `last_real_prompt_tokens`（必然超阈值，压缩就是它触发的）会立刻骗过检查、连续触发第二次压缩；
- 若上次真实 prompt tokens 低于阈值，且当前粗估相对"当时的粗估基线"只增长了不到 `max(4096, 阈值×5%)`，则判定为已知噪声，推迟压缩，把决定权交给下一次 API 响应的真实用量；
- 真实用量由 `update_from_response()`（`context_compressor.py:1117`）在每次 API 成功返回时刷新。

**③ `get_active_compression_failure_cooldown()`（`context_compressor.py:784`）— 会话级失败冷却**
摘要 LLM 失败（429/网络错误等）后会记录一个冷却期，且**持久化到 SQLite 会话行**（`record_compression_failure_cooldown`），跨进程重启依然生效。冷却期内跳过自动压缩；手动 `/compress` 传 `force=True` 可绕过。

**④ `should_compress(request_pressure_tokens)`（`context_compressor.py:1173`）— 权威阈值判断**
- `tokens >= threshold_tokens` 才继续；
- 内置摘要 LLM 冷却检查（#11529：摘要失败后 token 数依然超阈值，若不挡住，每一回合都会重新触发压缩、反复插入失败标记，CLI 表现为"卡死"直到冷却结束）；
- **防抖动（anti-thrash）**：若最近连续 2 次压缩各自节省不足 10%，则拒绝再压，并建议用户 `/new` 或 `/compress <topic>`。

### 3.3 阈值 `threshold_tokens` 的来历（`context_compressor.py:949 _compute_threshold_tokens`）

```
有效输入预算 = context_length - max_tokens(输出预留)      # #43547
基础阈值     = 有效输入预算 × threshold_percent(默认 50%)
下限保护     = max(基础阈值, MINIMUM_CONTEXT_LENGTH)      # 大窗口模型不 50% 就过早压缩
退化保护     = 若下限 ≥ 有效窗口（如 64K 小模型），改为窗口的 85% 触发  # #14690
```

两个关键修正：
- **#43547**：服务商会从同一个窗口里预留 `max_tokens` 作输出空间。自定义 provider 配了 65536 的 `max_tokens` 时，若阈值按整窗计算，会在压缩触发前就吃到 400 错误。所以阈值基于"窗口 − 输出预留"计算——这就是本段代码注释中 "output room already reserved by `_compute_threshold_tokens`" 的含义。
- **#14690**：小窗口模型（≤最低上下文长度）如果直接套 `max(50%, 最低值)`，阈值会等于整个窗口，压缩永远触发不了（服务商在 100% 之前就拒绝了）。改为按 85% 触发。

### 3.4 执行压缩：`agent._compress_context(...)` 的链路

```
conversation_loop.py:1014
  └─ AIAgent._compress_context()                      # run_agent.py:5572（纯转发器）
       └─ compress_context()                          # conversation_compression.py:435
            ├─ codex_app_server 模式 → 转交 codex 线程自身的 compact 机制（#36801）
            ├─ 惰性可行性检查（首次压缩时才探测辅助模型，省 ~400ms 冷启动）
            ├─ 会话级压缩锁（state.db 原子锁）
            │    防止父回合 agent 与后台 review fork 并发压缩同一 session，
            │    各自轮转 session_id 造成孤儿子会话；锁不可用时 fail-open
            │    （宁可冒并发风险也不能陷入"永不压缩"的死循环）
            ├─ memory_manager.on_pre_compress()       # 压缩丢弃上下文前通知外部记忆
            ├─ ContextCompressor.compress()           # context_compressor.py:2707 ★核心算法
            │    Phase 1  剪枝旧工具结果（廉价预处理，无 LLM 调用）：
            │             旧 tool result 替换为一行信息摘要
            │             （如 "[terminal] ran `npm test` -> exit 0, 47 lines"），
            │             相同结果去重、截断超大 tool_call 参数
            │    Phase 2  划定边界：保护头部（system prompt + 首轮交换）
            │             + 按 token 预算保护尾部（最近约 20K tokens）
            │    Phase 3  用结构化 prompt 让摘要 LLM 总结中间轮次；
            │             再次压缩时对上一份摘要做"迭代更新"而非重写
            │    失败路径：认证失败(401/403)/网络瞬断 → 整体 ABORT，
            │             原样返回消息、记录冷却（#29559 #25585：
            │             为一次瞬断丢弃中间上下文是净损失）
            ├─ 压缩后清理孤立的 tool_call/tool_result 配对
            ├─ _ensure_compressed_has_user_turn()     # 严格 chat 模板兜底（#55677）
            └─ 会话持久化：二选一
                 · 传统模式：结束旧 session，轮转出新 session_id 的子会话
                 · in-place 模式（compression.in_place）：同一 session_id 下
                   归档旧行、写入压缩后转录（#38763，消灭 session 轮转 bug 群）
```

`compress_context` 的返回约定：压缩中止时**原样返回输入消息**，调用方通过 `len(returned) == len(input)` 识别 no-op 并停止重试。

### 3.5 压缩成功后的收尾（conversation_loop.py:1020-1043）

1. **重置重试状态**：`_empty_content_retries`、`_thinking_prefill_retries` 等清零——压缩后的请求应获得全新机会，不继承压缩前历史遗留的恢复计数器。
2. **重定 flush 基线**：`conversation_history_after_compression()`（`conversation_compression.py:371`）
   - 传统轮转模式 → 返回 `None`：子会话还没见过压缩后的转录，下次持久化时整体写入；
   - **in-place 模式 → 返回 `list(messages)`（浅拷贝）**：压缩后的行已经以同一 session_id 持久化了。若仍返回 `None`，基于对象身份的 flush 路径会把这些已入库的 dict 当成新消息**再追加一遍，活跃上下文直接翻倍，反过来又触发压缩**——一个精确的自激振荡 bug。浅拷贝恰好把"当前 dict 身份"记为历史，同回合后续 append 的新消息仍会被识别为新。
3. **预算退还**：`api_call_count -= 1` + `iteration_budget.refund()`——压缩这一趟不消耗用户可感知的迭代次数。
4. `continue` 回到循环顶部，用压缩后的消息重新组装请求。

---

## 4. 设计目的、问题与解决方案总结

### 设计目的

在模型请求发出前的最后一刻建立一道**基于即时请求估算**的压缩闸门，与"回合开场预检"（前瞻）和"post-response 真实用量闸门"（滞后）形成三层互补，保证长会话在任何增长模式下都能在撞上下文墙**之前**完成压缩，同时不因粗估噪声或压缩失败而陷入压缩风暴。

### 解决的问题 → 对应方案

| # | 问题 | 方案 | 关联 issue |
|---|---|---|---|
| 1 | 单回合内工具结果暴涨，开场预检看不到、post-response 闸门的 `prompt_tokens` 滞后一拍，下一次调用撞墙（实测 271K/272K Codex 失败） | 在发请求前用 `estimate_request_tokens_rough`（含工具 schema）即时复查，超阈值先压缩再重试 | — |
| 2 | 粗估器对 schema 重请求故意高估；压缩刚结束时旧的真实用量又是陈旧超阈值的，两者都会造成误触发/连续二次压缩 | `should_defer_preflight_to_real_usage`：压缩后等一回合真实用量（-1 哨兵）；粗估相对"已被证明能装下"的基线增长在容忍带内就推迟 | #36718 |
| 3 | 摘要 LLM 失败后 token 依然超阈值，每回合重新触发压缩→插入失败标记→再触发，CLI 看似冻结 | 双保险：`should_compress` 内置冷却检查 + 会话级持久化冷却（`get_active_compression_failure_cooldown`，SQLite 落盘，跨重启有效）；手动 `/compress force=True` 可穿透 | #11529 |
| 4 | 压缩收益递减时（每次只省 1-2 条消息）无限压缩循环 | 防抖动：连续 2 次节省 <10% 即停止自动压缩，建议 `/new`；加上 `compression_attempts < 3` 的每回合硬上限（与 400/413 溢出处理器共享） | #40803 |
| 5 | 阈值按整窗计算，忽略输出预留 `max_tokens`，压缩触发前就吃 400 | 阈值基于"窗口 − 输出预留"的有效输入预算计算 | #43547 |
| 6 | 小窗口模型的最低阈值下限退化为整窗，压缩永不触发 | 下限 ≥ 有效窗口时改为按 85% 触发 | #14690 |
| 7 | in-place 压缩后 flush 基线错误导致已入库消息被二次追加，上下文翻倍并重新触发压缩 | `conversation_history_after_compression`：in-place 返回消息浅拷贝作基线，轮转模式返回 `None` | #38763 |
| 8 | 父 agent 与后台 review fork 并发压缩同一会话，双双轮转 session 产生孤儿子会话 | state.db 原子压缩锁（按旧 session_id 加锁）；锁子系统坏掉时 fail-open | #34351 |
| 9 | 摘要模型认证失败/网络瞬断时销毁中间上下文换一个占位符，得不偿失 | 该类失败一律 ABORT：原样返回、记录冷却、等待重试 | #29559 #25585 |
| 10 | 压缩不应惩罚用户的迭代预算 | `api_call_count -= 1` + `iteration_budget.refund()`，压缩趟次对用户透明 | — |

### 核心设计取舍

- **粗估宁高勿低，但用真实用量纠偏**：`estimate_request_tokens_rough` 故意把工具 schema 全额计入并高估，保证先于服务商拒绝而压缩；再用 `should_defer_preflight_to_real_usage` 以服务商上报的真实 `prompt_tokens` 作为最终仲裁，避免高估造成压缩风暴。
- **三闸门镜像同一条守卫链**：Pre-API 检查刻意与 turn-prologue 预检保持完全一致的守卫顺序（defer → cooldown → should_compress），行为可预测、修一处即三处受益。
- **失败时保守优先**：摘要失败宁可不压（ABORT + 冷却）也不破坏历史；锁失效宁可裸奔（fail-open）也不陷入永不压缩的死循环——两个方向的"宁可"都选了对会话数据损害最小的一侧。

---

## 5. 关键文件索引

| 文件 | 角色 |
|---|---|
| `agent/conversation_loop.py:968-1043` | 本文分析的 Pre-API 压力检查 |
| `agent/conversation_loop.py:4732-4763` | 工具循环尾部 post-response 压缩闸门（真实用量） |
| `agent/conversation_loop.py:~3333/~3443/~3557` | 413 / context-length-exceeded 溢出错误处理器（共享 `compression_attempts`） |
| `agent/turn_context.py:350-462` | 回合开场预检压缩（最多 3 轮渐进压缩） |
| `agent/context_compressor.py:699` | `ContextCompressor` — 默认上下文引擎（剪枝→保护头尾→摘要中段） |
| `agent/context_compressor.py:949` | `_compute_threshold_tokens` — 阈值计算（输出预留 + 小窗口退化保护） |
| `agent/context_compressor.py:1131` | `should_defer_preflight_to_real_usage` — 粗估噪声抑制 |
| `agent/context_compressor.py:1173` | `should_compress` — 阈值 + 冷却 + 防抖动 |
| `agent/context_compressor.py:2707` | `compress()` — 压缩核心算法 |
| `agent/conversation_compression.py:435` | `compress_context()` — 压缩编排（锁、会话轮转/in-place、失败处理） |
| `agent/conversation_compression.py:371` | `conversation_history_after_compression` — 压缩后 flush 基线 |
| `agent/model_metadata.py:2337/2413` | 粗略 token 估算器（消息级 / 请求级含工具 schema） |
| `run_agent.py:5572` | `AIAgent._compress_context` — 转发器 |
