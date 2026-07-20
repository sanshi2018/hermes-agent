# 在 PyCharm 中以终端方式运行 Hermes Agent（CLI / TUI）

## 结论先行

项目自带的虚拟环境 `.venv`（Python 3.11.15）已经以可编辑模式安装了 `hermes-agent`
包（`hermes_agent.egg-info` 存在，`.venv/Scripts/hermes.exe` 已生成），所以不需要重新
安装依赖，直接配置 PyCharm 的 Run Configuration 指向这个解释器即可。

我已经在 `.idea/runConfigurations/` 下生成了两个共享的 Run Configuration 文件（PyCharm
项目文件，纳入版本控制后其他人打开工程也能直接用）：

- `Hermes_CLI.xml` → 名称 **Hermes (CLI)**，等价于在终端执行 `hermes --cli`
- `Hermes_TUI.xml` → 名称 **Hermes (TUI)**，等价于在终端执行 `hermes --tui`

两者都：
- 使用项目解释器 `.venv/Scripts/python.exe`（SDK 名称 `uv (hermes-agent)`，与
  `.idea/hermes-agent.iml` 中登记的一致）
- 以 `RUN_TOOL` 方式直接调用已安装的控制台脚本 `hermes`（即 `.venv/Scripts/hermes.exe`
  对应的入口 `hermes_cli.main:main`），而不是用某个 `.py` 文件路径
- 勾选了 **Emulate terminal in output console**（`EMULATE_TERMINAL=true`）——这一项是
  关键：Hermes 的 TUI 基于 curses / 全屏终端渲染、CLI 也依赖多行编辑和 ANSI
  控制序列，不开这个选项在 PyCharm 的普通 Run 面板里会显示乱码或无法交互。

## 如何在 PyCharm 里看到并运行

因为我只能操作文件系统、无法直接点击 PyCharm 的图形界面，请按下面步骤在 IDE 里确认：

1. 用 PyCharm 打开（或切回）`hermes-agent` 工程。
2. 如果 PyCharm 已经在开着这个工程，配置文件是在外部写入的，PyCharm 一般会自动感知
   `.idea/runConfigurations/` 的变化；如果右上角运行配置下拉框里没有出现，执行一次
   **File → Reload All from Disk**（或者关闭工程重新打开）即可刷新。
3. 右上角运行配置下拉框中应能看到 **Hermes (CLI)** 和 **Hermes (TUI)** 两个选项。
4. 选中后点绿色三角形运行（或 Shift+F10 / Ctrl+Shift+F10）。
   - 运行面板会以「模拟终端」的方式启动，可以直接输入文字、使用方向键、Ctrl+C 中断，
     和在系统终端里跑 `hermes` / `hermes --tui` 效果一致。
5. 首次运行如果尚未做过 `hermes setup`（配置模型 Provider / API Key 等），CLI 会引导
   走配置向导；这是正常现象，与 PyCharm 配置无关。

## 如果下拉框里没有出现这两个配置（兜底手动步骤）

在极少数版本/缓存问题下 PyCharm 不会自动加载新增的共享配置文件，此时手动新建一份，
参数完全对照上面即可：

1. `Run → Edit Configurations… → +（左上角加号） → Python`
2. **Name**：`Hermes (CLI)`（或 `Hermes (TUI)`）
3. 把 **Script path** 切换成 **Module name**，不要用脚本路径；这里不用 Module，
   而是要选 **Run 是「工具」模式**——即在最新版 PyCharm 里，把「Script path」下拉切换
   为 **"Custom" / 直接填工具名**：
   - 若界面上是 "Script path" 单选，改成勾选旁边的可执行工具选项，填 `hermes`
   - 如果找不到该切换项，最简单等价方式：**Script path** 选中项目根目录下的
     `hermes` 文件（无扩展名的 Python 脚本，`#!/usr/bin/env python3` 开头，内容是
     `from hermes_cli.main import main; main()`），一样能跑。
4. **Parameters**：CLI 填 `--cli`；TUI 填 `--tui`
5. **Python interpreter**：选择项目自带的 `.venv`（`uv (hermes-agent)`）
6. **Working directory**：项目根目录 `C:\Users\Administrator\PycharmProjects\example\hermes-agent`
7. 展开 **Modify options**，勾选 **Emulate terminal in output console**（必须勾选，
   否则 TUI/多行编辑无法正常显示）
8. Apply → OK，即可在下拉框里看到并运行。

## 命令行等价对照

| PyCharm 配置 | 等价的终端命令 |
| --- | --- |
| Hermes (CLI) | `hermes --cli`（或直接 `hermes`，默认走经典 REPL） |
| Hermes (TUI) | `hermes --tui` |

其他常用命令（也可以照同样方式复制一份 Run Configuration，把 Parameters 换掉）：

```
hermes setup      # 交互式设置向导（选择模型 Provider / API Key）
hermes gateway    # 前台运行消息网关（Telegram/Discord 等）
hermes doctor     # 诊断配置与依赖问题
hermes model      # 切换模型
```

## 背景信息（供排查问题参考）

- 入口定义：`pyproject.toml` 的 `[project.scripts]` 中
  `hermes = "hermes_cli.main:main"`。
- `hermes` 是否走 CLI 还是 TUI 的优先级（见 `hermes_cli/main.py` 顶部注释）：
  显式 `--cli` 最高优先 → 显式 `--tui` / 环境变量 `HERMES_TUI=1` → 真实 TTY 检测 →
  `~/.hermes/config.yaml` 里的 `display.interface` 配置项兜底。
- 已验证 `.venv/Scripts/python.exe -c "import hermes_cli.main"` 可以无报错导入，
  说明依赖环境是完整可用的，PyCharm 里同一个解释器跑起来不会因为缺包而失败。
