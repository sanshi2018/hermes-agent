# ContextCompressor：新一轮压缩中"已总结过的消息"是如何处理的

> 源码：`agent/context_compressor.py`（入口锚点：`compress()` 中第 2934 行的
> `_find_latest_context_summary(...)` 调用）
> 结论先行：**旧总结不会被当成普通消息再总结一遍，也不会被原样保留两份。**
> 压缩器把旧总结"摘出来"作为迭代基线（`_previous_summary`），只把旧总结
> **之后**的新增轮次送给总结器做**增量更新**，最后用一条合并了新旧内容的
> 新总结消息替换掉整个压缩窗口（旧总结消息随窗口一起被丢弃）。

---

## 1. 问题背景

自动压缩会周期性触发。第二次及以后触发时，消息列表里已经存在上一轮
压缩产生的"交接总结"（handoff summary）消息。如果不做特殊处理，会出现
两类经典问题：

1. **总结套总结（summary-of-summary）**：旧总结被当作普通文本再次喂给
   总结器，指令前缀、结构化标题被反复嵌套，信息逐轮劣化、token 逐轮膨胀。
2. **重复/化石化**：旧总结消息和新总结消息同时留在上下文里，或早期被
   保护的消息在每轮压缩中被反复复制，永远无法被总结掉（#11996）。

## 2. 整体调用链路

```
compress(messages, ...)                                    # 主入口
 ├─ Phase 1: _prune_old_tool_results()                     # 2896  廉价裁剪旧工具结果，无 LLM 调用
 ├─ Phase 2: 确定压缩窗口边界
 │   ├─ compress_start = _protect_head_size()              # 2904  受保护头部（system + 衰减后的 protect_first_n）
 │   │    └─ _effective_protect_first_n()                  # 2417  压缩过一次后衰减为 0（防化石化 #11996）
 │   ├─ _align_boundary_forward()                          # 2905  避免窗口起点切在 tool 结果中间
 │   └─ compress_end = _find_tail_cut_by_tokens()          # 2908  按 token 预算保护尾部
 │
 ├─ ★ 旧总结定位与迭代状态恢复（本文核心，2927-2948）
 │   ├─ _find_latest_context_summary(messages,
 │   │       summary_search_start, compress_end)           # 2934  从窗口尾部向前扫描，找最新的旧总结
 │   │    ├─ _is_context_summary_content()                 # 2246  按 SUMMARY_PREFIX / 遗留 / 历史前缀识别
 │   │    └─ _strip_summary_prefix()                       # 2214  剥掉前缀、合并分隔符、结束标记，取纯正文
 │   ├─ 找到 → 恢复 _previous_summary（若为空）             # 2940-2941
 │   │         并截断 turns_to_summarize 到旧总结之后       # 2942
 │   └─ 没找到但 _previous_summary 非空 → 判定跨会话残留，
 │         直接丢弃 _previous_summary（#38788 防泄漏）      # 2943-2948
 │
 ├─ Phase 3: _generate_summary(turns_to_summarize, ...)    # 2974
 │   ├─ _previous_summary 存在 → 迭代更新 prompt            # 1946-1960
 │   │    （PREVIOUS SUMMARY + NEW TURNS → 更新后的完整总结）
 │   ├─ _previous_summary 为空 → 从零总结 prompt            # 1961-1972
 │   ├─ 成功后：_previous_summary = 新总结正文               # 2055  （存无前缀正文，供下一轮迭代）
 │   └─ 返回 _with_summary_prefix(summary)                 # 2061  （加当前版本 SUMMARY_PREFIX）
 │
 └─ Phase 4: 组装压缩后消息列表（3036-3178）
     ├─ 复制受保护头部（system 上追加压缩提示注记）          # 3038-3048
     ├─ 插入一条新总结消息（角色按交替规则选定，
     │    带 _compressed_summary 元数据 + END MARKER）      # 3128-3136
     │    （角色冲突无解时合并进首条尾部消息 3140-3165）
     ├─ 复制受保护尾部                                      # 3138-3166
     ├─ _sanitize_tool_pairs() 修复孤儿 tool_call/result    # 3170
     └─ 旧总结消息因位于 [compress_start, compress_end)
         窗口内而被自然丢弃 —— 由新总结取代
```

## 3. 对"已总结消息"的具体处理机制

### 3.1 定位：`_find_latest_context_summary`（2305-2316，调用点 2934）

- 搜索范围是 `[summary_search_start, compress_end)`，其中
  `summary_search_start` 为第一条非 system 消息（2933 行）。
  **故意从受保护头部就开始搜**，而不是从 `compress_start` 开始：会话恢复
  （resume）后，持久化的交接总结可能紧跟在系统提示词之后、落在受保护
  头部内（2928-2932 行注释），也必须能被找到以恢复迭代状态。
- 从后往前扫，**取最新的一条**总结（一个 lineage 里可能残留多条历史总结）。
- 识别依据是内容前缀，而非元数据：`_is_context_summary_content`（2246）
  同时匹配当前 `SUMMARY_PREFIX`（44 行）、遗留 `[CONTEXT SUMMARY]:`
  （71 行）以及冻结的历史前缀表 `_HISTORICAL_SUMMARY_PREFIXES`
  （153 行）——保证旧版本产生的总结在新版本下依然被认出。对
  "合并进尾部"式总结，还会先跳过 `_MERGED_SUMMARY_DELIMITER` 之前的
  包裹内容再匹配（2252-2253）。

### 3.2 提取正文：`_strip_summary_prefix`（2214-2237）

找到旧总结后不会原样使用，而是清洗出**纯正文**：

- 剥掉当前/遗留/所有历史版本的指令前缀——否则旧版前缀携带的陈旧指令
  （如早期"resume exactly from Active Task"式措辞）会内嵌在正文里，
  持续劫持后续回复（#35344）；
- 若是合并式总结，丢弃 `_MERGED_SUMMARY_DELIMITER` 之前的旧尾部包裹
  内容，防止 `[PRIOR CONTEXT]` 头和陈旧尾部泄漏进下一轮总结器 prompt；
- 剥掉尾部 `_SUMMARY_END_MARKER`（插入时会重新追加）。

### 3.3 恢复迭代状态 + 截断总结窗口（2939-2948）

```python
if summary_idx is not None:
    if summary_body and not self._previous_summary:
        self._previous_summary = summary_body          # 会话恢复：从消息里回填迭代基线
    turns_to_summarize = messages[max(compress_start, summary_idx + 1):compress_end]
elif self._previous_summary:
    self._previous_summary = None                      # 跨会话残留，丢弃（#38788）
```

三个关键点：

1. **回填而不覆盖**：只有当内存中 `_previous_summary` 为空（典型场景：
   进程重启后 resume 会话）才从消息中回填。同一会话内连续压缩时，
   内存里的 `_previous_summary` 已是最新（2055 行每次成功都会更新），
   不会被消息中较旧的副本覆盖。
2. **窗口截断是防"总结套总结"的核心**：
   `max(compress_start, summary_idx + 1)` 保证送入总结器的
   `turns_to_summarize` **不包含旧总结消息本身，也不包含它之前的任何
   消息**——那些内容已经浓缩在 `_previous_summary` 里了，重复喂入只会
   造成嵌套与膨胀。真正需要总结的只有旧总结之后新产生的轮次。
3. **跨会话防泄漏（elif 分支）**：消息里找不到总结、但内存里
   `_previous_summary` 却有值，说明它来自另一个已结束的会话（cron 任务、
   上一次 `/new` 等）。此时必须丢弃，否则迭代更新路径会把别的会话内容
   注入总结器 prompt。这是点位防御；`on_session_reset()`（726 行）和
   `on_session_end()`（763 行）在会话边界还会整体清空，构成双保险。

### 3.4 增量总结：`_generate_summary` 的迭代更新路径（1946-1960）

`_previous_summary` 存在时，prompt 不是"从头总结"，而是"更新总结"：

```
PREVIOUS SUMMARY:
{旧总结正文}

NEW TURNS TO INCORPORATE:
{仅新增轮次的序列化内容}

Update the summary using this exact structure.
PRESERVE all existing information that is still relevant.
ADD new completed actions to the numbered list (continue numbering).
Move items from "In Progress" to "Completed Actions" when done.
Move answered questions to "Resolved Questions".
Update "Active State" to reflect current state.
Remove information only if it is clearly obsolete.
CRITICAL: Update "## Active Task" to the user's most recent unfulfilled input...
```

即：旧信息**保留并演进**（编号续增、状态迁移、仅删明确过时项），
新信息**增量并入**，输出仍是同一套结构化模板（Active Task / Goal /
Completed Actions / Active State / In Progress / Blocked / Key Decisions /
Resolved Questions / Pending Asks / Relevant Files / Remaining Work /
Critical Context）。这样总结的规模由模板 + token 预算约束，而不随
压缩轮数无限增长。

成功后（2048-2061）：剥离思考块 → 脱敏 → **无前缀正文存入
`_previous_summary`**（下一轮迭代的基线）→ 返回
`_with_summary_prefix(summary)`（带当前版本指令前缀的完整消息文本）。

### 3.5 旧总结消息的最终去向（Phase 4，3036-3166）

组装阶段只保留三段：受保护头部 + **一条新总结消息** + 受保护尾部。
旧总结消息通常位于 `[compress_start, compress_end)` 窗口内，因此
**随窗口整体被丢弃**——它的信息已通过迭代更新融入新总结，不会重复出现。

新总结消息的形态：

- 内容 = `SUMMARY_PREFIX` + 正文 + `_SUMMARY_END_MARKER`（3129 行；
  前缀告诉模型这是历史参考不是活跃指令，结束标记防止弱模型把
  "## Active Task" 里逐字引用的旧请求当成新输入，#11475/#14521/#33256）；
- 角色按"避免相邻同角色"规则在 user/assistant 间选择（3064-3119），
  且在 Anthropic 首消息必须为 user（#52160）、或压缩后无任何 user 轮
  （#58753）时强制 user；两头都冲突时降级为**合并进首条尾部消息**
  （3140-3165，带 `_MERGED_SUMMARY_DELIMITER` 包裹——这正是 3.1/3.2 中
  需要"看穿"合并格式的原因）；
- 打上 `_compressed_summary` 元数据（86 行，下划线开头以便 wire 层
  剥除），供前端识别，但**跨进程识别仍靠内容前缀**（元数据不持久化）。

边缘情况：会话 resume 后首轮压缩时，若旧总结落在受保护头部
（`summary_idx < compress_start`），它本轮会原样保留在头部；但由于
`_effective_protect_first_n()`（2417-2432）在 `compression_count >= 1`
**或 `_previous_summary` 非空**时衰减为 0，下一轮压缩窗口就会从
system 之后开始，这条旧总结随即落入窗口被正常替换——不会永久残留。

### 3.6 总结失败时的处理

- LLM 总结失败且属于鉴权/网络终结性故障，或配置了
  `abort_on_summary_failure`：**整体中止**，消息原样返回（2998-3034），
  旧总结与 `_previous_summary` 均不动，下次重试仍走迭代路径。
- 其余失败：走确定性静态兜底 `_build_static_fallback_summary`
  （3053-3062），其中若 `_previous_summary` 存在，会显式注记
  "上一轮总结仍作为背景连续性上下文有效，但本次 LLM 更新失败"
  （1701-1706）。

## 4. 设计目的（Why）

| 目的 | 对应机制 |
|---|---|
| 防止"总结套总结"导致的信息劣化与 token 膨胀 | 窗口截断到 `summary_idx + 1`；`_previous_summary` 走专用迭代 prompt 而非当普通文本重总结 |
| 长会话中总结信息**连续演进**而非每轮丢失重来 | 迭代更新 prompt：PRESERVE + ADD + 状态迁移，结构化模板固定总结规模 |
| 进程重启 / 会话 resume 后迭代能力不中断 | 从消息中按前缀识别旧总结并回填 `_previous_summary`，搜索范围覆盖受保护头部 |
| 跨版本兼容：旧版前缀的总结仍能被识别与清洗 | `_HISTORICAL_SUMMARY_PREFIXES` 冻结表 + `_strip_summary_prefix` 归一化（#35344） |
| 防跨会话内容泄漏进总结器 prompt | "消息中无总结但内存有基线 → 丢弃"守卫（2943-2948）+ 会话边界整体清空（#38788） |
| 防早期消息在多轮压缩中化石化、头部无限增长 | `protect_first_n` 压缩一次后衰减为 0（#11996） |
| 防模型把旧总结当活跃指令重放 | `SUMMARY_PREFIX` 强指令 + `_SUMMARY_END_MARKER` 边界 + Active Task 标注为 STALE |

## 5. 解决方案一句话概括

**"摘出旧总结做基线，只喂新增轮次，LLM 做结构化增量合并，一条新总结
整体替换旧窗口（含旧总结）"**——由此，无论压缩多少轮，上下文中始终
只有一条最新的、持续演进的交接总结，且其体量被模板与预算约束在常数级。
