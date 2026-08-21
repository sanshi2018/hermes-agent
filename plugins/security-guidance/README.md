# security-guidance（安全指导）

针对 Agent 编写的代码，提供基于模式匹配（Pattern-matched）的安全警告。

当 Agent 调用 `write_file`、`patch` 或 `skill_manage` 工具，
且写入的内容包含已知危险代码模式（如：`eval`、`pickle.load`、`yaml.load`、`os.system`、
设置了 `shell=True` 的 `subprocess`、`dangerouslySetInnerHTML`、`verify=False`、
ECB 模式、GitHub Actions 的 `${{ github.event.* }}` 注入、未设置 `weights_only=True` 的 `torch.load` 等）时，
插件会在工具的执行结果末尾追加一条警告。

文件依然会被正常写入；模型会在下一轮对话中看到该警告，
从而修复代码或简要说明为什么该写法在此处是安全的。

这是 Anthropic 的 `security-guidance` 插件设计中的第 1 层 —— 
一种快速的本地首轮检查，不消耗任何 LLM Token。
第 2 层和第 3 层（对话结束时的 LLM Diff 审查、Agent 级提交审查）未被移植；
Agent 本身已支持通过 `delegate_task` 按需运行此类审查。

## 规则覆盖范围（25 条规则）

模式规则集基于 Apache-2.0 协议完整 Fork 自 Anthropic 的 `claude-plugins-official`。
分类如下：

| 类别                   | 规则                                                         |
| ---------------------- | ------------------------------------------------------------ |
| 不安全的反序列化       | `pickle.load`, `cPickle/cloudpickle/dill.load`, `marshal.loads`, `shelve.open`, `yaml.load`, `yaml.unsafe_load`, 未设置 `weights_only=True` 的 `torch.load`, `joblib.load`, `pandas.read_pickle`, `numpy.load(allow_pickle=True)` |
| 命令注入               | `os.system`, `subprocess(..., shell=True)`, JS 的 `child_process.exec`, Go 的 `exec.Command("sh"...)` |
| 代码注入               | `eval(`, JS 的 `new Function(...)`                           |
| XSS 漏洞接收端 (Sinks) | `.innerHTML =`, `.outerHTML =`, `.insertAdjacentHTML(`, `document.write`, React `dangerouslySetInnerHTML` |
| 密码学陷阱             | AES ECB 模式, Node 的 `crypto.createCipher`（未指定 IV）, 禁用 TLS 证书校验（`verify=False`, `rejectUnauthorized: false`, `InsecureSkipVerify: true` 等） |
| XXE (XML 外部实体注入) | 未使用 `defusedxml` 的 `xml.etree`、`minidom`、`xml.sax`     |
| 供应链安全             | 未设置 `integrity=` SRI 哈希的 `<script src="https://..."`   |
| CI/CD 注入             | GitHub Actions 工作流文件中在 `run:` 中使用 `${{ github.event.* }}` |

模式匹配采用了 Python 正则表达式 + 字面子字符串匹配的方式。
每条规则都包含一个针对文件扩展名的 `path_filter` lambda 表达式 —— 
仅适用于 Python 的规则会跳过 `.js`，仅适用于 JS 的规则会跳过 `.py`，
所有规则均跳过 `.md/.txt/.rst/.json/.yaml`。

使用了后向断言（Lookbehind assertions）来排除方法调用
（因此 `model.eval()` 和 `redis.eval()` 不会触发 `eval(` 规则）。
误报率虽然一般但尚可接受；正因如此，该插件默认仅采取“警告”模式。

## 启用方式

插件为按需启用（Opt-in）。将其添加到你的允许列表中：

```bash
hermes plugins enable security-guidance
# 或手动编辑 ~/.hermes/config.yaml：
plugins:
  enabled:
    - security-guidance
```

## 运行模式

| **环境变量**                  | **默认值** | **作用 / 效果**                                              |
| ----------------------------- | ---------- | ------------------------------------------------------------ |
| (无)                          | warn       | 在工具返回结果中追加一个 `⚠️ Security guidance` 警告块。文件依然正常写入。 |
| `SECURITY_GUIDANCE_BLOCK=1`   | 未设置     | 完全拒绝写入操作，并将该警告作为拦截原因。适用于要求更严格的环境。 |
| `SECURITY_GUIDANCE_DISABLE=1` | 未设置     | 紧急开关 —— 加载插件但不执行任何操作。                       |

## 目前**尚未**支持的功能

- **暂无 LLM Diff 审查：** Anthropic 的第 2 层会在每次修改了文件的对话轮次结束时，

  发起一次辅助 LLM 调用。在 hermes 中，默认情况下这会路由到主模型

  （`auxiliary_client._resolve_auto()` 优先选择主模型），

  对于推理模型来说会产生实际费用。后续可以通过单独的 PR，

  将第 2 层连接到低成本的辅助模型上，并提供显式启用选项。

- **暂无 Agent 级提交审查：** Anthropic 的第 3 层会启动一个带有 `Read`/`Grep`/`Glob` 权限的 SDK 子 Agent，

  以在执行 `git commit` 时追踪数据流。这是基于 `delegate_task` 构建的后续功能。

- **暂无项目本地规则文件：** Anthropic 的 `.claude/claude-security-guidance.md`

  是给其第 2/3 层的 LLM Prompt 读取的，而不是给模式扫描器读取的。

  一旦第 2 层落地，我们可以添加类似的 `.hermes/security-guidance.md`。

## 局限性

这是一个尽力而为的辅助工具。

模式匹配可能会漏报漏洞，也可能会产生误报。

请将警告视为建议，它不能替代代码审查、SAST（静态代码安全测试）、依赖项扫描或渗透测试。

## 归属与许可协议

- `patterns.py` 完整 Fork 自

  [`anthropics/claude-plugins-official`](https://github.com/anthropics/claude-plugins-official/tree/main/plugins/security-guidance/hooks)

  （提交版本 `0bde168`，2026-05-26），遵循 [Apache License 2.0](https://www.google.com/search?q=./LICENSE) 协议。

  完整归属信息请参见 [NOTICE](https://www.google.com/search?q=./NOTICE)。

- `__init__.py`、`plugin.yaml`、`README.md` 以及测试代码均为 NousResearch 的原创工作，

  与 hermes-agent 的其余部分一样遵循 MIT 协议。