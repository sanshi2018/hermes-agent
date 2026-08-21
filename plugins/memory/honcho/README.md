# Honcho 记忆提供程序（Honcho Memory Provider）

具备多轮辩证推理、会话摘要、双向 Peer 工具以及持久化结论的 AI 原生跨会话用户建模方案。

> **Honcho 文档：** https://docs.honcho.dev/v3/guides/integrations/hermes

## 环境要求

- `pip install honcho-ai`
- Honcho Cloud 账号 — 通过 OAuth 登录或从 [app.honcho.dev](https://app.honcho.dev/) 获取 API 密钥进行连接 — 或自托管实例

## 配置设置

Bash

```
hermes memory setup honcho   # 直接配置 Honcho（适用于全新安装）
hermes memory setup          # 通用选择器，从列表中选择 Honcho
```

对于云端版本，配置向导会询问 **OAuth、设备代码（device code）或 API 密钥**。OAuth 会打开浏览器登录并自动存储授权 — 无需手动复制；Token 会自动刷新。在 SSH 或无头（headless）机器上，请选择 **device**：CLI 会输出一段简短的代码和链接，你可以在其他任何机器的浏览器中打开该链接；在浏览器中批准后设置即告完成。桌面端应用会在记忆提供程序下拉菜单旁提供一个 **Connect** 链接用于打开浏览器授权流程。

或手动配置：

Bash

```
hermes config set memory.provider honcho
echo "HONCHO_API_KEY=***" >> ~/.hermes/.env
```

> `hermes honcho setup` 同样可用，但前提必须是 Honcho **已被激活**为当前记忆提供程序 — `honcho` 子命令仅针对当前处于激活状态的提供程序注册。对于全新安装，请使用 `hermes memory setup honcho`。

## 架构概览

### 双层上下文注入

上下文会在 API 调用时注入到 **用户消息（user message）** 中（而非系统提示词中），以保留 Prompt 缓存。系统提示词中仅包含静态模式标头。注入的文本块包裹在 `<memory-context>` 标签中，并附带系统备注，明确指出这是背景数据而非新的用户输入。

分为两个独立层，每层拥有各自的执行节奏（cadence）：

**第 1 层 — 基础上下文（Base context）**（每 `contextCadence` 轮刷新一次）：

1. **会话摘要（SESSION SUMMARY）** — 来自 `session.context(summary=True)`，置于首位
2. **用户画像（User Representation）** — Honcho 持续演进的用户模型
3. **用户 Peer 卡片（User Peer Card）** — 关键事实快照
4. **AI 自我画像（AI Self-Representation）** — Honcho 构建的 AI Peer 模型
5. **AI 身份卡片（AI Identity Card）** — AI Peer 的事实数据

**第 2 层 — 辩证补充（Dialectic supplement）**（每 `dialecticCadence` 轮触发一次）：

针对用户展开的多轮 `.chat()` 推理，附加在基础上下文之后。

两层合并后，通过 `_truncate_to_budget` 按 `contextTokens` 预算进行截断（Token 数 × 4 个字符，且保证词边界安全）。

### 最新消息查询重写（可选项）

当设置 `queryRewrite: true` 时，辩证第 0 轮（pass 0）首先会使用共享的 `memory_query_rewrite` 辅助任务，将最新消息重写为一个简短的记忆检索问题。重写后的问题将用于辩证请求；而基础上下文检索仍使用原始消息作为搜索查询。如果重写超时或返回无效结果，插件将回退到下文所述的冷/热提示词。开启该标志后，会跳过通用的辩证预热，从而不会覆盖第一条用户消息。

**默认关闭** — 该重写会在每个辩证周期（并非每轮 pass）增加一次辅助模型调用。请在 `hermes model` -> auxiliary models -> **Memory query rewrite** 下选择一个快速且低成本的模型；其请求超时时间在 config.yaml 中的 `auxiliary.memory_query_rewrite.timeout` 进行配置（默认 8 秒）。该任务和模块（`plugins/memory/query_rewrite.py`）与具体提供程序无关 — 任何记忆提供程序均可复用它们。`dialecticCadence` 依然控制周期运行的频率。

### 冷启动 vs 热会话提示词

当最新消息重写不可用时，辩证第 0 轮会根据会话状态自动选择回退提示词：

- **冷启动（Cold）**（未缓存基础上下文）：“这个人是谁？他们的偏好、目标和工作风格是什么？重点关注能够帮助 AI 助手立即发挥作用的事实。”
- **热会话（Warm）**（已存在基础上下文）：“根据本会话目前讨论的内容，与当前对话最相关的用户上下文是什么？优先考虑活跃的上下文，而非背景履历等事实。”

不可配置 — 自动判定。

### 辩证深度（多轮推理）

`dialecticDepth`（1–3，超限将被限制）控制每个辩证周期触发的 `.chat()` 调用次数：

| **深度** | **轮数（Passes）** | **行为**                                                     |
| -------- | ------------------ | ------------------------------------------------------------ |
| 1        | 单次 `.chat()`     | 仅基础查询（冷启动或热会话提示词）                           |
| 2        | 审查 + 综合        | 对 Pass 0 结果进行自我审查；Pass 1 进行针对性综合。如果 Pass 0 返回强信号（>300 字符，或包含列表/分段的结构化内容 >100 字符），则条件性提前退出 |
| 3        | 审查 + 综合 + 调和 | Pass 2 调和先前各轮之间的矛盾，输出最终的综合结果            |

### 等比例推理级别

当未设置 `dialecticDepthLevels` 时，每一轮会使用相对于 `dialecticReasoningLevel`（“基准级别”）的等比例级别：

| **深度** | **各轮推理级别**     |
| -------- | -------------------- |
| 1        | [base]               |
| 2        | [minimal, base]      |
| 3        | [minimal, base, low] |

可使用 `dialecticDepthLevels` 进行重写：显式指定每一轮推理级别字符串数组。

### 查询自适应推理级别

自动注入的辩证逻辑会根据查询长度动态调整 `dialecticReasoningLevel`：当 ≥120 字符时提升 1 级，≥400 字符时提升 2 级，最高不超过 `reasoningLevelCap`（默认为 `"high"`）。设置 `reasoningHeuristic: false` 可禁用此特性，将每次自动调用固定为 `dialecticReasoningLevel`。

### 三个独立的辩证控制旋钮

| **控制项**                | **控制内容**                              | **类型** |
| ------------------------- | ----------------------------------------- | -------- |
| `dialecticCadence`        | 触发频率 — 两次辩证触发之间的最小对话轮数 | int      |
| `dialecticDepth`          | 触发轮数 — 单次触发的轮数（1–3）          | int      |
| `dialecticReasoningLevel` | 推理强度 — 每次 `.chat()` 调用的推理上限  | string   |

### 输入清理（Sanitization）

`run_conversation` 会在处理前从用户输入中剥离泄露的 `<memory-context>` 块。当 `saveMessages` 持久化包含注入上下文的对话轮次时，该块可能会通过消息历史记录重新出现在后续轮次中。清理器会移除 `<memory-context>` 块及关联的系统备注。

## 工具

提供 5 个双向工具。所有工具均接受可选的 `peer` 参数（`"user"` 或 `"ai"`，默认值为 `"user"`）。

| **工具**           | **是否调用 LLM？** | **描述**                                                     |
| ------------------ | ------------------ | ------------------------------------------------------------ |
| `honcho_profile`   | 否                 | Peer 卡片 — 关键事实快照                                     |
| `honcho_search`    | 否                 | 跨会话消息搜索（混合语义 + 关键词，按相关度排序的摘录；默认 800 tokens，上限 2000） |
| `honcho_context`   | 否                 | 完整会话上下文：摘要、画像、卡片、消息                       |
| `honcho_reasoning` | 是                 | 通过辩证 `.chat()` 综合生成的 LLM 回答                       |
| `honcho_conclude`  | 否                 | 写入、列表/搜索或删除持久化结论（列表会展示删除所需的 ID）   |

工具可见性取决于 `recallMode`：在 `context` 模式下隐藏，在 `tools` 和 `hybrid` 模式下始终存在。

## 配置解析逻辑

配置将从第一个存在的文件中读取：

| **优先级** | **路径**                   | **作用域**                         |
| ---------- | -------------------------- | ---------------------------------- |
| 1          | `$HERMES_HOME/honcho.json` | 配置文件级别（隔离的 Hermes 实例） |
| 2          | `~/.hermes/honcho.json`    | 默认配置文件（共享的主机块）       |
| 3          | `~/.honcho/config.json`    | 全局（跨应用互操作）               |

主机键（Host key）衍生自处于激活状态的 Hermes Profile：`hermes`（默认）或 `hermes_<profile>`。

对于每个配置键，解析顺序依次为：**主机块（host block） > 根配置（root） > 环境变量（env var） > 默认值（default）**。

## 完整配置项参考

### 身份与连接

| **键名**      | **类型** | **默认值**     | **描述**                                                     |
| ------------- | -------- | -------------- | ------------------------------------------------------------ |
| `apiKey`      | string   | —              | API 密钥。回退使用 `HONCHO_API_KEY` 环境变量。通过 OAuth 连接时，该键将存储自动刷新的访问令牌 |
| `oauth`       | object   | —              | OAuth 授权信息（刷新令牌、过期时间、客户端、令牌端点）。由连接/登录流程自动写入并自动轮换 — 请勿手动编辑。可选：单独配置 API 密钥即可生效，无需 OAuth |
| `baseUrl`     | string   | —              | 自托管 Honcho 的基础 URL。本地 URL 会自动跳过 API 密钥身份验证 |
| `environment` | string   | `"production"` | SDK 环境映射                                                 |
| `enabled`     | bool     | auto           | 主开关。当存在 `apiKey` 或 `baseUrl` 时自动启用              |
| `workspace`   | string   | host key       | Honcho 工作区 ID。共享环境 — 同一工作区中的所有 Profile 都可以查看相同的用户身份和相关记忆 |
| `peerName`    | string   | —              | 用户 Peer 身份                                               |
| `aiPeer`      | string   | host key       | AI Peer 身份                                                 |

### 身份映射（网关多用户）

在网关部署（Telegram、Discord、Slack 等）中，每个用户都会携带平台原生的运行时 ID（Telegram UID、Discord snowflake、Slack 用户名等）。以下三个配置项控制这些运行时 ID 如何映射到 Honcho Peer。解析器完全由配置驱动且具有确定性 — 不会进行自动合并或运行时推断。

| **键名**            | **类型** | **默认值** | **描述**                                                     |
| ------------------- | -------- | ---------- | ------------------------------------------------------------ |
| `pinUserPeer`       | bool     | `false`    | 设置为 `true` 时，每个网关运行时用户都会折叠为 `peerName`。适用于单操作员部署，以便所有平台（及其他用户）共享同一个 Peer |
| `userPeerAliases`   | object   | `{}`       | 运行时 ID 到 Peer ID 的映射（如 `{"7654321": "alice"}`）。多对一是预期用法 — 将所有运行时 ID 别名映射到一个 Peer 名称。不支持一对多；一个运行时 ID 仅精准解析为一个 Peer |
| `runtimePeerPrefix` | string   | `""`       | 添加在未知运行时 ID 前面的前缀，用于划分命名空间（例如 `"telegram_"` → `telegram_7654321`）。仅在没有匹配的别名时使用。防止运行时 ID 形状相同的平台之间发生冲突 |

> **已废弃：** `pinPeerName` 是 `pinUserPeer` 的旧别名，目前仍可读取以保持向下兼容（两者同时设置时 `pinUserPeer` 优先）。`hermes honcho setup` 在触发时会将其迁移至 `pinUserPeer`，且不再写入该旧字段。

**解析阶梯**（首次匹配即止）：

```
1. pinUserPeer / pinPeerName=true → 返回 peerName（忽略运行时 ID）
2. userPeerAliases[runtime_id]   → 返回别名对应的 peer
3. userPeerAliases[runtime_id_alt] → 检查备用 ID（Telegram UID + 用户名等）
4. runtimePeerPrefix + runtime_id → 划分命名空间的 peer，带 sha256 冲突升级处理
5. 原始清理后的 runtime_id      → 回退 peer
6. peerName                      → 无运行时 ID（CLI/TUI）
7. session-key 回退              → 无相关配置
```

**为什么没有 `pinAiPeer`？** AI Peer 在结构设计上已经固定 — `aiPeer` 是 AI 侧唯一的身份设置，解析器绝不会覆盖它。只有用户侧 Peer 存在需要由 `pinUserPeer` 解决的“运行时 vs 配置”冲突。

**主机块 vs 根语义。** 这三个配置键在根级别和 `hosts.<host>` 级别均被接受。主机级别优先。对于映射和前缀，主机级别会**整体替换**根级别的值（而非合并），因此主机可以刻意拥有独立的身份体系，或通过 `userPeerAliases: {}` / `runtimePeerPrefix: ""` 将其清空。

**设置 — 网关身份树。** `hermes honcho setup` 仅在检测到已连接网关平台时才会询问身份映射（它会检查网关配置；在无网关环境下将跳过此步骤，因为没有运行时用户 ID 时这些键不会起作用）。运行时，它会询问*谁在与此网关通信？*并导出配置键：

- **只有我（just me）** → `pinUserPeer: true`。所有非 Agent 网关用户都会归并为 `peerName`；固定设置会覆盖所有别名，因此仅在不需要为用户侧身份分配独立 Peer 时选择此项。适用于将 Hermes 连接到个人 Telegram/Discord 等的个人使用场景。如果有多个独立的 Agent 接入网关且各自需要不同的 Peer，**请勿**锁定 — 保持 `pinUserPeer: false` 并通过 `userPeerAliases`（`[e]` 编辑器）进行映射。
- **我 + 其他人，池化共享（me + other people, pooled）** → `pinUserPeer: false` + `userPeerAliases`（将你的运行时 ID 映射到 `peerName`）。你可以继续使用共享历史记录；其他所有人拥有各自独立的 Peer。
- **我 + 其他人 / 仅其他人（me + other people / only other people）** → `pinUserPeer: false`，可选设置 `runtimePeerPrefix`。每个运行时用户 → 拥有各自的 Peer。适用于服务众多用户的 Bot。

在提示符处选择 **[e]** 可直接设置这三个键，而无需通过向导树。

**解绑固定（单用户 → 每用户）。** 将 `pinUserPeer` 从 `true` 切换为 `false` **不会**迁移已有数据。固定期间在 `peerName` 下积累的记忆将保留在该处；运行时用户现在将解析为全新的空白 Peer。为了保持你个人记忆的连续性，请选择 **池化（pooled）** 路径 — 将你的运行时 ID 别名映射回 `peerName`，这样你的对话将继续存入池化历史记录中，而其他用户则获得各自的 Peer。当向导检测到你正在取消固定先前固定的配置文件时，会自动提供此引导。

### 记忆与召回

| **键名**          | **类型** | **默认值**      | **描述**                                                     |
| ----------------- | -------- | --------------- | ------------------------------------------------------------ |
| `recallMode`      | string   | `"hybrid"`      | `"hybrid"`（自动注入 + 工具），`"context"`（仅自动注入，隐藏工具），`"tools"`（仅工具，不注入）。旧版 `"auto"` → `"hybrid"` |
| `observationMode` | string   | `"directional"` | 预设：`"directional"`（全部开启）或 `"unified"`（用户观察自我，AI 观察他人）。若要进行精细控制，请使用 `observation` 对象 |
| `observation`     | object   | —               | 单 Peer 观察配置（见 Observation 章节）                      |

### 写入行为

| **键名**         | **类型**   | **默认值** | **描述**                                                     |
| ---------------- | ---------- | ---------- | ------------------------------------------------------------ |
| `writeFrequency` | string/int | `"async"`  | `"async"`（后台异步），`"turn"`（每轮同步），`"session"`（结束时批量写入），或整数 N（每 N 轮写入一次） |
| `saveMessages`   | bool       | `true`     | 将消息持久化保存至 Honcho API。当设置为 `false` 时，所有自动写入都将被跳过 — 包含原始对话轮次（`sync_turn`）、结论镜像（`on_memory_write`）以及会话结束/关闭时的刷新 — 而读取和工具路径仍保持完备的功能。 |

### 会话解析

| **键名**            | **类型** | **默认值**        | **描述**                                                     |
| ------------------- | -------- | ----------------- | ------------------------------------------------------------ |
| `sessionStrategy`   | string   | `"per-directory"` | `"per-directory"`（按目录）、`"per-session"`（按会话）、`"per-repo"`（按 Git 仓库根目录）、`"global"`（全局） |
| `sessionPeerPrefix` | bool     | `false`           | 在会话键前加上 Peer 名称                                     |
| `sessions`          | object   | `{}`              | 手动的“目录到会话名称”映射                                   |

#### 会话名称解析

Honcho 会话名称决定了记忆落入哪个对话桶中。解析遵循优先级链 — 首次匹配即止：

| **优先级** | **来源**                           | **示例会话名称**                      |
| ---------- | ---------------------------------- | ------------------------------------- |
| 1          | 手动映射（`sessions` 配置）        | `"myproject-main"`                    |
| 2          | `/title` 命令（会话中重命名）      | `"refactor-auth"`                     |
| 3          | 网关会话键（Telegram、Discord 等） | `"agent-main-telegram-dm-8439114563"` |
| 4          | `per-session` 策略                 | Hermes 会话 ID (`20260415_a3f2b1`)    |
| 5          | `per-repo` 策略                    | Git 根目录名称 (`hermes-agent`)       |
| 6          | `per-directory` 策略               | 当前目录基准名 (`src`)                |
| 7          | `global` 策略                      | 工作区名称 (`hermes`)                 |

无论 `sessionStrategy` 如何设置，网关平台始终通过优先级 3（按聊天隔离）进行解析。该策略设置仅影响 CLI 会话。

如果 `sessionPeerPrefix` 设置为 `true`，则会附加 Peer 名称前缀：`alice-hermes-agent`。

#### 各策略的产出效果

- **`per-directory`** — `$PWD` 的基准目录名。在 `~/code/myapp` 和 `~/code/other` 中打开 hermes 会产生两个独立的会话。相同目录 = 跨运行共享同一会话。
- **`per-repo`** — Git 根目录名称。仓库中的所有子目录共享同一个会话。如果不在 Git 仓库内，则回退至 `per-directory`。
- **`per-session`** — Hermes 会话 ID（时间戳 + 十六进制）。每次调用 `hermes` 都会启动一个新的 Honcho 会话。如果没有可用会话 ID，则回退至 `per-directory`。
- **`global`** — 工作区名称。所有内容使用同一个会话。记忆会在所有目录和运行中累积。

### 多 Profile 模式

多个 Hermes Profile 可以共享一个工作区，同时保持独立的 AI 身份。配置解析逻辑为 **主机块 > 根 > 环境变量 > 默认值** — 主机块继承自根，因此共享设置只需声明一次：

JSON

```
{
  "apiKey": "***",
  "workspace": "hermes",
  "peerName": "yourname",
  "hosts": {
    "hermes": {
      "aiPeer": "hermes",
      "recallMode": "hybrid",
      "sessionStrategy": "per-directory"
    },
    "hermes_coder": {
      "aiPeer": "coder",
      "recallMode": "tools",
      "sessionStrategy": "per-repo"
    }
  }
}
```

两个 Profile 在同一个共享环境（`hermes`）中看到相同的用户（`yourname`），但每个 AI Peer 都会构建各自的观察结果、结论和行为模式。Coder 的记忆保持以代码为导向；主 Agent 的记忆则保持通用广泛。

主机键继承自激活的 Hermes Profile：`hermes`（默认）或 `hermes_<profile>`（例如 `hermes -p coder` -> 主机键为 `hermes_coder`）。为了兼容性，旧版的 `hermes.<profile>` 主机块仍可读取，并在 CLI 写入 Profile 作用域的 Honcho 配置时进行迁移。

### 辩证与推理

| **键名**                  | **类型** | **默认值** | **描述**                                                     |
| ------------------------- | -------- | ---------- | ------------------------------------------------------------ |
| `dialecticDepth`          | int      | `1`        | 每个辩证周期发生的轮数（1–3，超限将被限制）。1=单次查询，2=审查+综合，3=审查+综合+调和 |
| `dialecticDepthLevels`    | array    | —          | 每一轮可配置的推理级别字符串数组（可选）。重写等比例默认值。示例：`["minimal", "low", "medium"]` |
| `dialecticReasoningLevel` | string   | `"low"`    | `.chat()` 的基础推理级别：`"minimal"`、`"low"`、`"medium"`、`"high"`、`"max"` |
| `dialecticDynamic`        | bool     | `true`     | 设置为 `true` 时，模型可以通过 `honcho_reasoning` 工具按需覆盖每次调用的推理级别。设置为 `false` 时，始终使用 `dialecticReasoningLevel` |
| `dialecticMaxChars`       | int      | `600`      | 自动注入辩证补充的最大字符数。仅适用于自动注入 — 显式调用的 `honcho_reasoning` 工具结果将完整返回 |
| `dialecticMaxInputChars`  | int      | `10000`    | 传给 `.chat()` 的辩证查询输入最大字符数。Honcho 云端限制：10k |
| `reasoningHeuristic`      | bool     | `true`     | 查询自适应：根据查询长度自动上调自动注入辩证的级别（≥120 字符 +1，≥400 字符 +2），上限为 `reasoningLevelCap`。设置为 `false` 将每次自动调用固定为 `dialecticReasoningLevel` |
| `reasoningLevelCap`       | string   | `"high"`   | `reasoningHeuristic` 动态调整的上限：`"minimal"`、`"low"`、`"medium"`、`"high"`、`"max"` |

### Token 预算

| **键名**          | **类型** | **默认值** | **描述**                                                     |
| ----------------- | -------- | ---------- | ------------------------------------------------------------ |
| `contextTokens`   | int      | SDK 默认值 | `context()` API 调用的 Token 预算。同时也限制预取截断的上限（tokens × 4 字符） |
| `messageMaxChars` | int      | `25000`    | 通过 `add_messages()` 发送的单条消息最大字符数。超出此限制将触发带有 `[continued]` 标记的分块。Honcho 云端限制：25k |

### 节奏控制（成本控制）

| **键名**                 | **类型** | **默认值**     | **描述**                                                     |
| ------------------------ | -------- | -------------- | ------------------------------------------------------------ |
| `contextCadence`         | int      | `1`            | 基础上下文刷新的最小间隔轮数（会话摘要 + 画像 + 卡片）       |
| `dialecticCadence`       | int      | `1`            | 辩证 `.chat()` 触发的最小间隔轮数                            |
| `injectionFrequency`     | string   | `"every-turn"` | `"every-turn"`（每轮）或 `"first-turn"`（仅在首条用户消息注入基础上下文；辩证补充保持自身的节奏） |
| `queryRewrite`           | bool     | `false`        | 在辩证前将最新消息重写为检索查询（每个周期增加一次辅助 LLM 调用） |
| `firstTurnBaseWait`      | float    | `3.0`          | 第 1 轮等待基础上下文/会话初始化的最大秒数。`0` 表示禁用等待（完全异步；上下文将在后续轮次展现）。第 2 轮及以后绝不等待卡住的初始化 |
| `firstTurnDialecticWait` | float    | `2.0`          | 第 1 轮等待辩证结果的最大秒数。`0` 表示禁用                  |

### 观察模式（精细化）

1:1 映射到 Honcho 的单 Peer 配置 `SessionPeerConfig`。存在时将覆盖 `observationMode` 预设。

JSON

```
"observation": {
  "user": { "observeMe": true, "observeOthers": true },
  "ai":   { "observeMe": true, "observeOthers": true }
}
```

| **字段**             | **默认值** | **描述**                                  |
| -------------------- | ---------- | ----------------------------------------- |
| `user.observeMe`     | `true`     | 用户 Peer 自我观察（Honcho 构建用户画像） |
| `user.observeOthers` | `true`     | 用户 Peer 观察 AI 消息                    |
| `ai.observeMe`       | `true`     | AI Peer 自我观察（Honcho 构建 AI 画像）   |
| `ai.observeOthers`   | `true`     | AI Peer 观察用户消息（启用跨 Peer 辩证）  |

预设值：

- `"directional"`（默认）：上述 4 项均为 `true`
- `"unified"`：用户 `observeMe=true`，AI `observeOthers=true`，其余为 `false`

### 硬编码限制

| **限制项**           | **数值**                      |
| -------------------- | ----------------------------- |
| 搜索工具最大 tokens  | 2000（硬上限），800（默认值） |
| Peer 卡片获取 tokens | 200                           |

## 环境变量

| **环境变量**                   | **回退/对应的配置项**                                        |
| ------------------------------ | ------------------------------------------------------------ |
| `HONCHO_API_KEY`               | `apiKey`                                                     |
| `HONCHO_BASE_URL`              | `baseUrl`                                                    |
| `HONCHO_ENVIRONMENT`           | `environment`                                                |
| `HERMES_HONCHO_HOST`           | 主机键（Host key）重写                                       |
| `HONCHO_OAUTH_DASHBOARD`       | OAuth 授权 Source 域（默认：云端 Dashboard；本地开发为 `localhost:3000`） |
| `HONCHO_OAUTH_AUTHORIZE_URL`   | 完整授权 URL（覆盖 Dashboard 域名）                          |
| `HONCHO_OAUTH_TOKEN_URL`       | Token 端点（默认：云端 API；本地开发为 `localhost:8000`）    |
| `HONCHO_OAUTH_DEVICE_AUTH_URL` | 设备授权端点（默认：由 Token URL 派生）                      |
| `HONCHO_OAUTH_CLIENT_ID`       | OAuth 客户端 ID（默认为 `hermes-agent`）                     |
| `HONCHO_OAUTH_SCOPE`           | 请求的作用域 Scope（默认为 `write`）                         |

## CLI 命令

| **命令**                               | **描述**                                                     |
| -------------------------------------- | ------------------------------------------------------------ |
| `hermes memory setup honcho`           | 直接配置 Honcho — 适用于全新安装                             |
| `hermes honcho setup`                  | 交互式设置向导（仅在 Honcho 作为激活提供程序时注册；重定向至 `hermes memory setup`） |
| `hermes honcho status`                 | 显示当前激活 Profile 的已解析配置                            |
| `hermes honcho enable` / `disable`     | 切换激活 Profile 的 Honcho 开关状态                          |
| `hermes honcho mode <mode>`            | 更改召回（recall）或观察（observation）模式                  |
| `hermes honcho peer --user <name>`     | 更新用户 Peer 名称                                           |
| `hermes honcho peer --ai <name>`       | 更新 AI Peer 名称                                            |
| `hermes honcho tokens --context <N>`   | 设置上下文 Token 预算                                        |
| `hermes honcho tokens --dialectic <N>` | 设置辩证最大字符数                                           |
| `hermes honcho map <name>`             | 将当前目录映射至某个会话名称                                 |
| `hermes honcho sync`                   | 为所有 Hermes Profile 创建主机块                             |

## 配置示例

JSON

```
{
  "apiKey": "***",
  "workspace": "hermes",
  "peerName": "username",
  "contextCadence": 2,
  "dialecticCadence": 3,
  "dialecticDepth": 2,
  "hosts": {
    "hermes": {
      "enabled": true,
      "aiPeer": "hermes",
      "recallMode": "hybrid",
      "observation": {
        "user": { "observeMe": true, "observeOthers": true },
        "ai": { "observeMe": true, "observeOthers": true }
      },
      "writeFrequency": "async",
      "sessionStrategy": "per-directory",
      "dialecticReasoningLevel": "low",
      "dialecticDepth": 2,
      "dialecticMaxChars": 600,
      "saveMessages": true
    },
    "hermes_coder": {
      "enabled": true,
      "aiPeer": "coder",
      "sessionStrategy": "per-repo",
      "dialecticDepth": 1,
      "dialecticDepthLevels": ["low"],
      "observation": {
        "user": { "observeMe": true, "observeOthers": false },
        "ai": { "observeMe": true, "observeOthers": true }
      }
    }
  },
  "sessions": {
    "/home/user/myproject": "myproject-main"
  }
}
```