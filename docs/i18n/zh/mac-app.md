# Basalt Mac 版

> Languages: [English](../../mac-app.md) · [Русский](../ru/mac-app.md) · **中文** · [Español](../es/mac-app.md) · [Français](../fr/mac-app.md) · [日本語](../ja/mac-app.md) · [Deutsch](../de/mac-app.md)

围绕 basalt 构建的原生应用：包含一个聊天窗口和一个 Monitor 窗口，基于由应用自行启动并监管的本地引擎运行。设计方案与阶段规划见 #434。本文档介绍第 1 阶段。

## 构建与运行

```bash
./scripts/bundle_app.swift            # release build, Metal kernels, then .build/Basalt.app, signed ad hoc
open .build/Basalt.app
```

- `--identity "Developer ID Application: …"` 使用强化运行时（hardened runtime）进行签名，用于分发。
- `--bin PATH` 直接基于已构建的二进制文件进行组装。
- `--dry-run` 打印打包计划。

若从构建目录直接运行而不打包，可通过环境变量将应用指向引擎：`BASALT_HELPER=.build/release/Basalt .build/release/BasaltMac`。

## 工作原理

- **引擎是一个独立进程：**即随应用打包在 `Contents/Helpers` 中的 `basalt serve`。
  - 应用会在空闲的回环端口上启动该引擎（`--port 0 --announce-ready`），并通过 `BASALT_API_KEY` 向其传入每次启动生成的令牌。
  - 应用通过与编辑器和脚本相同的 REST API 与引擎通信。
- **崩溃恢复：**引擎崩溃不会导致应用崩溃。监管器（supervisor）会在 1 s、2 s……最长至 30 s 后自动重启引擎；若一分钟内连续崩溃三次则放弃重启，并显示引擎的最后输出。
- **退出：**退出应用会优雅地停止引擎：先发送 SIGINT，10 s 后再发送 SIGKILL。
- **重新启动：**应用会根据记录文件（`~/Library/Application Support/Basalt/engine.json`，权限模式 0600）接管仍在运行的引擎，从而避免重复加载模型。
- **对话记录**通过 SwiftData 存储在 `~/Library/Application Support/Basalt/Conversations.store` 中。因崩溃而中断的回答会被标记为已停止。
- **模型**在设置（Settings）中配置（默认为 `mlx-community/gemma-4-12B-it-qat-4bit`），将在引擎下次启动时生效。

## 菜单栏与快速提问

- **引擎随应用一同启动**，而非随窗口打开才启动。即使关闭所有窗口，菜单栏图标（一个立方体）仍会保留。它会显示引擎的状态和当前模型，并提供以下操作：
  - 新建对话与快速提问；
  - 最近五个对话；
  - 启动、重启和停止引擎；
  - 打开 Monitor、设置（Settings）以及退出（Quit）。
- **快速提问（Quick ask）：**在任何应用中（包括全屏应用）按下 ⌥Space 即可呼出浮动面板，且不会抢占当前应用的焦点。
  - 输入问题并按 ↩，回答会在面板中流式输出。
  - 按 ⌘↩ 可将当前问答作为普通对话在主窗口中打开。
  - 按 Esc 关闭面板并丢弃本次问答。在按下 ⌘↩ 之前，快速提问的内容不会被保存。
  - 该快捷键通过 `RegisterEventHotKey` 注册，无需辅助功能（Accessibility）权限。可在设置中关闭，更改将在下次启动时生效。
- **设置（Settings）：**
  - 通用（General）：模型选择、快速提问以及登录时打开（`SMAppService`；需应用已签名并位于“应用程序”文件夹中才会生效）。
  - 云端密钥（Cloud Keys）：例如 `OPENROUTER_API_KEY` 等环境变量，其值保存在钥匙串中（服务名 `app.basalt.mac`）。引擎在下次启动时接收这些密钥以支持 compose 云端模型。应用绝不会将其写入磁盘。
- **状态恢复：**重新启动后会恢复上次选中的对话。每个对话未发送的草稿也会保存在 user defaults 中并在重启后恢复。

## 模型管理

- **模型窗口**（⌥⌘L，或通过菜单栏中的 Models 打开）会列出 Hugging Face 缓存（`~/.cache/huggingface/hub`）中的模型，并附带大小以及来自 `GET /v1/models/preview` 的当前 Mac 适配性评估结果。
- **加载（Load）**会将引擎切换至该模型（`POST /v1/models/load`），并将其设为引擎启动时的默认模型。评估结果为“does not fit（不适配）”的模型无法从应用中加载；窗口会说明具体原因。
- **下载（Download）**接受仓库 ID，例如 `mlx-community/gemma-4-12B-it-qat-4bit`。应用会在自身进程中执行下载、显示进度并记住未完成的下载任务。下次启动时，应用会从部分下载的文件处断点续传。下载失败的任务会等待点击重试（Retry）。
- **删除（Delete）**会先弹出确认提示，然后从缓存中移除该模型的文件夹。当前已加载的模型无法被删除。

## 附件

- 通过回形针图标**添加附件**，或直接将文件拖放至对话中：支持 PDF、图片（PNG、JPEG、HEIC、TIFF、GIF、WebP）以及音频（WAV、MP3、M4A、AAC、FLAC、CAF），单个文件上限为 32 MB。
- **带有附件的消息会经由 `/v1/pipeline` 处理。**引擎会使用其 OCR 模型（若无则使用 Apple Vision）读取文档和图片，并使用其语音模型（或聊天模型）转录音频。随后由聊天模型回答该消息（若消息文本为空，则默认使用 "Summarize the attached files."）。
- **识别出的文本会保留在对话中。**每个附件都以标签（chip）形式显示，展开标签即可查看引擎读取到的文本。后续消息发送时会自动将该文本附在问题之前，因此多轮追问可以直接引用文件内容。
- 回答下方的**回执（receipt）**会列出流水线的每个阶段及其使用的模型和耗时（毫秒）。

## 检查器（Inspector）

- 按 **⌥⌘I** 或点击某条回答，即可在对话旁打开检查器。
- **对话（Conversation）：**显示当前模型以及该对话的系统提示词（system prompt）。系统提示词随对话一并保存，并在每条消息前发送。清空提示词即可清除该设置。
- **回答（Answer）：**显示所选回答的回执，包括模型、提示词与补全 token 数、首字延迟（TTFT）、生成速度，以及流水线各阶段的模型和耗时（毫秒）。
- **附件（Attachments）：**逐页显示引擎从该问题的每个文件中读取到的文本。

## 快捷指令、服务与程序坞

- **快捷指令（Shortcuts）与聚焦搜索（Spotlight）：**“Ask Basalt”“Summarize File”“Transcribe Audio”和“Switch Model”均已注册为 App Intents。快捷指令可在应用处于后台时运行它们。每项指令都会等待引擎就绪（最长等待三分钟），且 Switch Model 会拒绝不适配当前设备的模型。由于 SwiftPM 不会处理意图元数据，`scripts/bundle_app.swift` 会使用 `appintentsmetadataprocessor` 将其提取至 `Contents/Resources/Metadata.appintents`。
- **系统服务（Services）：**在任意应用中选中文本并选择“服务 → Ask Basalt”，即可打开一个新对话并以该文本作为提问内容。
- **从 Hugging Face 一键通过 `basalt://` 启动：**`Basalt.app` 注册了 `basalt` URL scheme（`CFBundleURLTypes`）。打开 `basalt://run?model=<owner/repo>[&role=chat|mtp|moe-stream|ocr|asr|decide|embed|rerank]`——或在任意 [`goodmai`](https://huggingface.co/goodmai) 模型卡片上点击 **Run with Basalt**——会校验 Hugging Face 仓库 ID、更新已保存的模型选择、将 Basalt 置前并把受监管的引擎切换至该模型或流水线角色。
- **程序坞（Dock）：**将 PDF、图片或音频文件拖放到程序坞图标上，即可新建一个已附加该文件的对话。
- **通知：**当 Basalt 处于后台且回答生成完成时，通知会显示回答的前几行内容。应用仅请求一次通知权限。

## Compose 系统与云端模型

- **设置 → 引擎（Settings → Engine）：**既可运行单个模型，也可从配置文件运行 compose 系统（`basalt serve --compose <file>`）。点击“Restart the Engine with This”会停止当前引擎并使用新参数启动。该选择也会在下次启动时沿用。
- **按参数接管：**应用仅在遗留运行中引擎的启动参数完全一致时才会接管它。更改模型或 compose 文件后会启动新的引擎进程。退出应用时也会一并停止已接管的引擎。
- **Monitor：**针对 compose 系统，展示其名称、流水线及今日云端消费，以及每个模型的状态（`ready`、`loading`、`no-key`、`over-budget` 等）、保留策略、消费金额与预算上限以及任何拒绝记录。云端模型会带有云朵图标标识。
- **云端标记：**使用了云端模型的回答会在回执行和检查器中显示 "☁︎ cloud" 及其费用。所有未带该标记的回答均完全在本台 Mac 上本地运行。

## 辅助功能、多语言与更新

- **旁白（VoiceOver）：**
  - 每个纯图标控件都配有朗读名称：菜单栏图标、发送、停止、附件、删除以及检查器按钮。
  - 每条消息作为一个整体元素朗读，格式为 "You: …" 或 "Basalt: …"。
  - 进度指示器会说明当前正在等待的操作。
  - 适配性徽章会朗读为 "fits on this Mac" 或 "does not fit"。
  - 待发送附件会朗读为 "Remove <file>"。
- **多语言：**应用跟随系统语言，目前已随附英语和俄语翻译。翻译文件位于 `Localization/<language>.lproj/Localizable.strings`，由 `scripts/bundle_app.swift` 复制进应用包并列入 `CFBundleLocalizations`。当视图新增了俄语翻译表中缺失的字符串时，对应测试将会失败。
- **更新检查：**应用每天向 GitHub API 查询一次最新的 basalt 发布版本。若存在更新的非草稿、非预发布版本，菜单栏中会显示 "Update available"，并附带前往发布页面的链接。该请求仅向 GitHub 发送应用的 IP 地址和 User-Agent。可在“设置 → 通用（Settings → General）”中关闭此功能。

## 发布打包：签名、公证与 Homebrew Cask

- **本地构建：**运行 `./scripts/bundle_app.swift --identity "Developer ID Application: Name (TEAMID)" --notarize basalt-notary --dmg` 将使用强化运行时进行签名并生成 `Basalt-<version>.zip`。随后脚本通过 `notarytool` 提交压缩包进行公证、将公证票据（ticket）装订（staple）到应用中并重新打包，最终生成 `.dmg` 镜像。
  - 仅指定 `--zip` 时只打包而不进行公证。
  - `--notarize` 会拒绝临时（ad hoc）签名或未签名的构建。
  - 可通过以下命令一次性创建钥匙串配置文件：`xcrun notarytool store-credentials basalt-notary --apple-id <id> --team-id <TEAMID> --password <app-specific password>`。
- **版本发布流程：**当仓库中配置了 `MAC_SIGNING_IDENTITY`（runner 钥匙串中的证书名称）和 `MAC_NOTARY_PROFILE`（对应的钥匙串配置文件）两个机密变量时，推送 `v*` 标签会自动构建、签名并公证该应用。
  - 构建产物 `Basalt-<version>.zip` 会上传附加到对应的 GitHub Release。
  - 流程会在更新 formula 的同一 PR 中基于 `Casks/basalt-app.rb.template` 生成 `Casks/basalt-app.rb`。
  - 若缺少上述机密变量，发布流程将仅更新 formula 并给出提示。
- **安装：**当某个 Release 包含应用包后，可运行 `brew install --cask goodmai/basalt/basalt-app` 进行安装。

## Compose 视频工作室、通用转换器与实时 Hub（`v0.12`）

- **Compose 视频工作室（Compose Video Studio）：**支持在 24 GB UMA 上分阶段生成视频并内联播放（`LTX-2.5 AV-DiT`、`Kandinsky 6.0`、`LingBot-Video`、`Wan 2.2`），支持导出 `.mp4`（`#611`）。
- **通用转换器（`ConvertModelSheet`）：**提供完整的任意格式互转（any-to-any）量化界面（`MACaQ` 稠密/MoE、`FlatQuant`、`LIRQuant`、`SINQ`、`QTIP`、`E8P`、`PV-Tuning`、`QAT STE` + `QuantizationAuditCore`），支持英语和俄语（`#546`、`#599`、`#623`）。
- **实时 Hub 与 10 Hz SSE 遥测：**基于 `swift-huggingface` 驱动，支持 Xet 下载、Hub 实时搜索/发布以及 Monitor 中的 10 Hz 流式遥测（`#587`）。

## 性能预算

以下数据于 530aa28 提交在配备 24 GB 内存的 M4 Pro 上测得，引擎中加载了 Gemma 4 12B（#434）：

| 预算项 | 实测值 | 上限 | 测量方式 |
|---|---|---|---|
| 热启动时窗口出现耗时 | 5 次中位数 257 ms（冷启动：1.6 s） | 400 ms | `swift scripts/app_budgets.swift` |
| 引擎就绪后的空闲唤醒次数 | 10 s 内为 0 | 0 | 同上 |
| 应用进程内存占用 | 40 MB，打开包含 2,000 条消息的对话记录时额外增加 33.5 MB | 150 MB | 同上，以及 `AppBudgetTests` |
| 向 2,000 条消息的对话中流式写入单个 token | p95 为 7.4 ms，中位数为 6.0 ms | 16.7 ms (60 fps) | `BASALT_APP_BUDGETS=1 swift test --filter AppBudgetTests` |
| 快速提问：面板与首 token 超出引擎自身耗时的开销 | 热启动 40 ms（面板 15 ms）；进程内首个 SwiftUI 视图 175 ms | 150 ms | 同上 |

由于引擎需要加载模型，`app_budgets.swift` 会获取 GPU 锁。请在空闲的 Mac 上运行上述测试，并确保此时没有 CI 任务正在执行。以下三项仍需手动验证，尚未自动化：全屏应用上方呼出浮动面板（该面板为带有 `.fullScreenAuxiliary` 属性的非激活面板）、在应用关闭时从“快捷指令”触发 App Intents，以及在干净的 Mac 上通过 cask 安装（需要签名密钥）。
