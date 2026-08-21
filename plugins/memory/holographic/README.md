```markdown
# 全息记忆提供程序（Holographic Memory Provider）

基于本地 SQLite 的事实存储，具备 FTS5 全文搜索、信任度评分、实体解析以及基于 HRR（全息简化表示）的组合检索功能。

## 环境要求

无 —— 使用 SQLite（始终可用）。NumPy 为可选依赖，用于 HRR 代数运算。

## 安装与设置

```bash
hermes memory setup    # 选择 "holographic"
或手动配置：

Bash

```
hermes config set memory.provider holographic
```

## 配置说明

在 `config.yaml` 中的 `plugins.hermes-memory-store` 项下进行配置：

| **配置项**      | **默认值**                     | **描述**                 |
| --------------- | ------------------------------ | ------------------------ |
| `db_path`       | `$HERMES_HOME/memory_store.db` | SQLite 数据库路径        |
| `auto_extract`  | `false`                        | 在会话结束时自动提取事实 |
| `default_trust` | `0.5`                          | 新事实的默认信任度评分   |
| `hrr_dim`       | `1024`                         | HRR 向量维度             |

## 工具列表

| **工具**        | **描述**                                                     |
| --------------- | ------------------------------------------------------------ |
| `fact_store`    | 包含 9 种操作：add（添加）、search（搜索）、probe（探测）、related（相关）、reason（推理）、contradict（矛盾检查）、update（更新）、remove（删除）、list（列表） |
| `fact_feedback` | 将事实评价为“有帮助/无帮助”（用于训练和更新信任度评分）      |