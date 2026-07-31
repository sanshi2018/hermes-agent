"""可插拔内存提供程序（Memory Provider）的抽象基类。

内存提供程序为 Agent 提供跨会话的持久化记忆能力。
MemoryManager 强制限制最多只能启用一个外部提供程序，
以防止工具 Schema 膨胀以及内存后端冲突。

外部提供程序（如 Honcho、Hindsight、Mem0 等）通过
MemoryManager 进行注册和管理。同一时间仅允许运行一个外部提供程序。

注册机制：
  插件存放在 plugins/memory/<name>/ 目录中，
  并通过配置项 memory.provider 进行激活。

生命周期（由 MemoryManager 调用，并挂载于 run_agent.py 中）：
  initialize()          — 连接服务、创建资源、系统预热
  system_prompt_block()  — 用于系统提示词（System Prompt）的静态文本
  prefetch(query)        — 在每轮对话开始前进行后台记忆检索
  sync_turn(user, asst)  — 在每轮对话结束后进行异步写入
  get_tool_schemas()     — 暴露给模型的工具 Schema 列表
  handle_tool_call()     — 分发并处理工具调用
  shutdown()             — 安全退出 / 优雅关闭

可选钩子函数（重写以启用相应功能）：
  on_turn_start(turn, message, **kwargs) — 带有运行时上下文的单轮对话起始钩子
  on_session_end(messages)               — 会话结束时的记忆提取
  on_session_switch(new_session_id, **kwargs) — 运行过程中的 session_id 切换/轮换
  on_pre_compress(messages) -> str       — 在上下文压缩前提取信息
  on_memory_write(action, target, content, metadata=None) — 镜像/同步内置内存的写入操作
  on_delegation(task, result, **kwargs)  — 父 Agent 侧对子 Agent 工作的观察/监听
  backup_paths() -> list[str]            — 包含在 `hermes backup` 备份中的额外磁盘路径
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryProvider(ABC):
    """Abstract base class for memory providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """该提供程序的简短标识符（例如：'builtin'、'honcho'、'hindsight'）。"""

        # -- 核心生命周期（需实现以下方法）------------------------------------

    @abstractmethod
    def is_available(self) -> bool:
        """如果该提供程序已正确配置、具备凭据且准备就绪，则返回 True。

        在 Agent 初始化期间被调用，用于决定是否激活该提供程序。
        不应发起网络请求 —— 仅检查配置项和已安装的依赖项。
        """

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None:
        """为会话进行初始化。

        在 Agent 启动时调用一次。可用于创建资源（如数据库/表）、
        建立网络连接、启动后台线程等。

        kwargs 始终包含：
          - hermes_home (str): 当前启用的 HERMES_HOME 目录路径。请使用此路径
            进行配置文件（Profile）作用域的存储，而不是硬编码为 ``~/.hermes``。
          - platform (str): 运行平台，例如 "cli"、"telegram"、"discord"、"cron" 等。

        kwargs 还可能包含：
          - agent_context (str): 上下文类型，例如 "primary"、"subagent"、"cron" 或 "flush"。
            对于非主（non-primary）上下文，提供程序应跳过写入操作
            （因为定时任务的系统提示词可能会损坏用户的记忆表示）。
          - agent_identity (str): 配置文件（Profile）名称（例如 "coder"）。用于
            按 Profile 对提供程序身份进行作用域划分。
          - agent_workspace (str): 共享工作区名称（例如 "hermes"）。
          - parent_session_id (str): 针对子 Agent，为其父级的 session_id。
          - user_id (str): 平台用户标识符（适用于网关会话）。
          - user_id_alt (str): 可选的备用稳定平台用户标识符。
        """

    def system_prompt_block(self) -> str:
        """返回要包含在系统提示词中的文本。

        在组装系统提示词期间被调用。返回空字符串以跳过。
        这适用于静态的提供方信息（指令、状态）。预取
        的召回上下文是通过 prefetch() 单独注入的。
        """
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """召回即将开始的对话轮次所需的相关上下文。

                在每次 API 调用前执行。返回格式化后的文本，
                以便作为上下文注入；若没有相关内容，则返回空字符串。

                实现应保持高效——实际的召回工作应放在后台线程中执行，
                此处只返回缓存结果。

                ``session_id`` 用于支持并发会话的提供方，
                例如网关群聊或缓存代理。
                不需要按会话隔离的提供方可以忽略该参数。
                """
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """为【下一轮对话】排队预加载后台记忆检索。

        在每轮对话完成后被调用。检索结果将在
        下一轮对话中的 prefetch() 被消费使用。默认不执行任何操作（no-op）——
        需要进行后台预检索的提供程序应当重写此方法。
        """
    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """将已完成的轮次持久化存储到后端。

        在每个轮次结束后被调用。应当是非阻塞的 —— 如果后端存在延迟，
        请将其放入队列以进行后台处理。

        ``messages`` 是截至该已完成轮次为止、符合 OpenAI 格式的对话消息列表，
        包含任何助手的工具调用以及工具执行结果。
        不需要原始轮次上下文的提供者（providers）可以忽略此参数。
        """

    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """返回该提供程序暴露的工具 Schema 列表。

        每个 Schema 均遵循 OpenAI 的函数调用（Function Calling）格式：
        {"name": "...", "description": "...", "parameters": {...}}

        如果该提供程序不包含任何工具（仅提供上下文），则返回空列表。
        """

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """处理该提供程序旗下某个工具的调用。

        必须返回一个 JSON 字符串（即工具的执行结果）。
        仅针对由 get_tool_schemas() 所返回的工具名称进行调用。
        """
        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

    def shutdown(self) -> None:
        """安全退出 / 优雅关闭 —— 刷新缓冲区队列，关闭网络连接。"""

    # -- 可选钩子函数（重写以启用相应功能）---------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """在每轮对话开始时，使用用户消息调用。

                可用于轮次计数、作用域管理和定期维护。

                ``kwargs`` 可能包含：``remaining_tokens``、``model``、
                ``platform``、``tool_count``。

                各提供方可按需使用其中的参数；
                未使用的额外参数将被忽略。
                """

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """当会话结束时调用（显式退出或超时）。

        用于会话结束时的真实信息提取、摘要生成等。
        messages 是完整的对话历史记录。

        并非在每轮对话后都调用 — 仅在真正的会话边界处调用
        （如 CLI 退出、/reset、网关会话过期）。
        """

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        """当 Agent 在进程运行期间切换 session_id 时被调用。

        在执行 ``/resume``、``/branch``、``/reset``、``/new``（CLI 命令）、
        网关端等效操作以及上下文压缩时触发 —— 即任何在不销毁提供程序的前提下
        重新分配 ``AIAgent.session_id`` 的路径。

        如果在 ``initialize()`` 中缓存了特定于会话的状态
        （如 ``_session_id``、``_document_id``、累积的对话轮次缓冲区、计数器等），
        提供程序应当在此处更新或重置这些状态，
        以确保后续的写入操作能落入正确会话的记录中。

        参数
        ----------
        new_session_id:
            Agent 刚刚切换到的目标 session_id。
        parent_session_id:
            先前的 session_id（如果该信息有意义）—— 适用于 ``/branch``（分叉血统/谱系关系）、
            上下文压缩（延续血统/谱系关系）以及 ``/resume``（正在离开的会话）。
            在无血统/谱系关联时为空字符串。
        reset:
            当这是一个全新的对话而非对现有对话的恢复时，该值为 ``True``。
            由 ``/reset`` 或 ``/new`` 触发。
            在此参数设为 ``True`` 时，提供程序应当刷新并清空已累积的单会话缓冲区
            （如 ``_session_turns``、``_turn_counter`` 等）。
            对于 ``/resume``、``/branch`` 或上下文压缩等逻辑对话在新的 ID 下继续进行的情况，
            该值为 ``False``。
        rewound:
            如果 session_id 未发生改变但对话记录（transcript）被截断，该值为 ``True``；
            缓存了单轮对话文档状态的提供程序应当将此类缓存标记为失效。

        为保持向下兼容性，默认不执行任何操作（no-op）。
        """

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """在上下文压缩丢弃旧消息之前被调用。

        可用于从即将被压缩的消息中提取关键信息/洞察。
        messages 参数即是将要被总结或丢弃的消息列表。

        返回希望包含在压缩总结提示词（Prompt）中的文本，
        以便压缩器能够保留由提供程序提取出的关键洞察。
        若无需补充任何内容，则返回空字符串（保持向下兼容的默认行为）。
        """
        return ""

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """当子代理（subagent）完成任务时，在父代理（PARENT agent）上调用。

        父代理的内存提供者（memory provider）会将“任务+结果”对
        作为“被委托事项及返回结果”的观察记录（observation）进行获取。
        子代理本身没有提供者会话（skip_memory=True）。

        task: 委托提示词（delegation prompt）
        result: 子代理的最终响应
        child_session_id: 子代理的 session_id
        """

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """返回该提供程序进行安装/设置时所需的配置字段。

        供 'hermes memory setup' 命令使用，引导用户完成配置过程。
        每个字段都是一个字典（dict），结构如下：
          key:         配置键名（例如 'api_key', 'mode'）
          description: 易于理解的文字描述
          secret:      若该配置应保存至 .env 文件中则为 True（默认：False）
          required:    若该项为必填项则为 True（默认：False）
          default:     默认值（可选）
          choices:     有效可选值列表（可选）
          url:         用户获取该凭据的 URL 网址（可选）
          env_var:     针对敏感信息（secrets）显式指定的环境变量名（默认：自动生成）

        若无需任何配置（例如仅本地运行的提供程序），则返回空列表。
        """
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """将非敏感配置写入提供程序的原生位置。

        由 'hermes memory setup' 命令在收集完用户输入后调用。
        ``values`` 仅包含非敏感字段（敏感信息保存至 .env 文件中）。
        ``hermes_home`` 为当前启用的 HERMES_HOME 目录路径。

        拥有原生配置文件（如 JSON、YAML）的提供程序应当重写此方法，
        以将配置写入其预期的存储位置。仅使用环境变量的提供程序
        保持默认实现即可（不执行任何操作）。

        所有新增的内存提供程序插件必须实现以下方案之一：
        - 实现 save_config() 以支持原生配置文件格式；或
        - 仅使用环境变量（此时 get_config_schema() 中的所有字段
          都应设置 ``env_var`` 属性，且本方法保持默认的 no-op 状态）。
        """
    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """当内置内存工具写入条目时被调用。

        action: 操作类型，为 'add'（添加）、'replace'（替换）或 'remove'（移除）
        target: 写入目标，为 'memory'（内存）或 'user'（用户）
        content: 写入的条目内容
        metadata: 写入操作的结构化出处/溯源信息（若可用）。常见键名包括：
          ``write_origin``、``execution_context``、``session_id``、
          ``parent_session_id``、``platform`` 以及 ``tool_name``。

        用于将内置内存的写入操作同步/镜像至你的后端。
        """

    def backup_paths(self) -> List[str]:
        """返回该提供程序存储在 HERMES_HOME 【之外】的额外磁盘路径。

        ``hermes backup`` 仅会遍历 HERMES_HOME 目录，因此如果不在此处声明，
        任何保存在 ``~/.honcho``、``~/.hindsight``、``~/.openviking`` 等路径下的
        提供程序状态，都将在“备份/导入”循环中丢失。

        返回绝对路径字符串（文件或目录）组成的列表。
        备份命令会解析每个路径，并将存在且位于用户主目录（home directory）下的路径
        提取并存入归档文件中保留的 ``_external/`` 子树中；
        随后的 ``hermes import`` 则会将它们还原至原始位置。
        为了安全起见，主目录之外的路径将被忽略。

        本方法【必须】可以在未调用 ``initialize()`` 且无网络连接的情况下被调用 ——
        仅从配置或环境变量中解析。默认返回空列表（无外部路径）。
        """
        return []
