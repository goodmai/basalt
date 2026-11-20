# 架构

> Languages: [English](../../architecture.md) · [Русский](../ru/architecture.md) · **中文** · [Español](../es/architecture.md) · [Français](../fr/architecture.md) · [日本語](../ja/architecture.md) · [Deutsch](../de/architecture.md)

```
┌──────────────────────────────────────────────────────┐
│                       Basalt                           │
│                                                      │
│   MCP stdio ──┐                                      │
│               ├──► ModelOrchestratorActor ──► MLX   │
│   REST :8080 ─┘        (actor, FIFO)      Metal GPU  │
└──────────────────────────────────────────────────────┘
```

两种传输协议共享同一个 actor 实例——用于 IDE 集成（Claude Desktop、Cursor）的 **MCP stdio**，以及用于智能体间工作流和 Claude Code 后端的 **REST HTTP**。

## 项目结构

```
Sources/                    — one SwiftPM target per directory; dependencies point down only
  BasaltCore/               — DTOs, errors, config, logger, profilers, calculators, HF cache, orchestrator
    Chat/                   — ModelProfile: everything that differs between model families
    Config/                 — ServerConfig value type
  BasaltInference/          — MLX inference engine, ContextDegradationProfiler, RepetitionGuard  (→ Core)
  BasaltCloud/              — OpenRouter / cloud model fallback (CostTracker, ModelRouter)    (→ Core)
  BasaltServer/             — REST (Hummingbird 2.x, Host/Origin/Content-Type guard, API key and rate-limit middleware) and MCP JSON-RPC stdio server (→ Core)
  BasaltUI/                 — Terminal UI (Markdown, spinner, progress bar, diff, table)       (→ Core)
  BasaltApp/                — SwiftUI REST chat client, not linked into the CLI (launch: #55)  (→ Core)
  BasaltCLI/                — Subcommands and the CLI root command (App/)                     (→ all of the above)
  BasaltBin/                — Thin executable wrapper (product `Basalt`, installed as `basalt`)
  PerformanceBenchmark/     — Standalone benchmark CLI target

Tests/
  BasaltTests/              — Unit + integration tests
    CLI/                    — Command parsing, Cloud, Fit, PromptContextBuilder
    Cloud/                  — CostTracker, ModelRouter, OpenRouterClient
    UI/                     — Clipboard, Diff, Markdown, ProgressBar, Spinner, Table, TerminalUI
    MockInferenceEngine.swift — Controllable mock for orchestrator tests

basalt-claude                       — Executable Swift launcher (build + serve + claude)
scripts/                    — Build and maintenance Swift scripts
docs/                       — Extended documentation
```

只有 `BasaltApp` 可以导入 SwiftUI/AppKit；CLI 所链接的所有 target 都保持无界面（headless）（由 `ModuleBoundaryTests` 强制检查）。
跨 target 新共享的符号使用 `package` 访问级别（既有公共 API 保持 `public`）。

---

## 架构说明

**动态量化、MACaQ 与 MoE** —— `MLXInferenceEngine` 会解析 `config.json`（包括 `macaq_quantization`、`flatquant_quantization`、`lirquant_quantization`），将稠密（Dense）和稀疏 MoE 检查点分派到经过优化的 Metal 内核上执行（`SwitchGLU`、单遍 `MACaQSwitchLinear` WHT + `MLX.gatherQuantizedMM` `#628`、`E8P` 2-bit 格点量化、`QTIP`、`STA` 3D 视频注意力 `#609`）。

**SDK、治理与分层内存（`v0.12`）** —— `BasaltFoundationModels` 提供 Swift SDK 桥接层（`#606`）；`VideoRunnerCore` 面向 24 GB 统一内存实现分阶段视频生成（`#611`）；`TieredKVHierarchyCore` 管理 L0/L1/L2 统一内存 KV 层级（`#627`）；`SemanticCacheActor`、`HITLApprovalActor` 和 `SymbolicKnowledgeGraph` 负责企业级路由与人工介入（Human-in-the-Loop）审批（`#625`、`#626`）。

**Actor 隔离** —— `ModelOrchestratorActor` 是一个 Swift 6 actor。所有推理调用无需显式加锁即可按 FIFO 顺序串行执行。MCP 与 REST 共享同一个 actor 实例。

**模型热切换** —— `switchModel(to:)` 会从本地缓存解析 HuggingFace 仓库 ID，卸载当前模型（`container = nil` + `MLX.GPU.clearCache()`），然后加载新权重。actor 会确保所有正在执行的请求完成后再开始切换。

**超时保护** —— 每次 `generate` 和 `generateStream` 调用都包裹在 5 分钟超时保护中，并支持协作式 Task 取消，防止模型卡死无限期阻塞服务端。

**流式输出** —— 使用 Hummingbird 2 的 `AsyncStream<ByteBuffer>` 响应体同时支持两种 SSE 格式：OpenAI（`data: {...}`）和 Anthropic（`event: content_block_delta`）。

**思考块剥离** —— `generateStream` 运行一个状态机，在将 token 转发给客户端前滤除 `<think>...</think>` 推理思考块。

**ID 转换** —— `ModelsController` 会在 HuggingFace 仓库 ID（`mlx-community/Qwen3.5-4B-4bit`）与 Claude Code 模型发现过滤器所要求的 `claude-local/mlx-community--Qwen3.5-4B-4bit` 格式之间进行相互转换。API 请求中同时接受这两种格式。

---

## 维护脚本

| 脚本 | 用途 |
|---|---|
| `scripts/build_metal.swift` | 编译 Metal GPU 内核（若已是最新则跳过） |
| `scripts/cleanup.swift` | `--mode=daily` 轮转日志/图像/报告，`--mode=release` 在发布前清理构建和运行时产物，`--mode=all` 同时执行两者；未加 `--apply` 时默认空跑（dry run） |
| `scripts/archive.sh` | 归档日志/报告/截图 |

## 依赖项

整个项目为纯 SwiftPM 包；运行时不依赖 CocoaPods、Node 或 Python。`Package.resolved` 锁定了精确的版本修订号。

| 包 | 用途 |
|---|---|
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | 基于 Metal 的张量数组与神经网络层 |
| [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | 模型架构与加载 —— `MLXLLM`、`MLXVLM`、`MLXLMCommon` |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | 分词器（tokenizer）与对话模板（chat template） |
| [hummingbird](https://github.com/hummingbird-project/hummingbird) | HTTP 服务器 |
| [swift-argument-parser](https://github.com/apple/swift-argument-parser) | 命令行解析 |
| [Rainbow](https://github.com/onevcat/Rainbow) | 终端着色与布局 |
| [swift-markdown](https://github.com/apple/swift-markdown), [Splash](https://github.com/JohnSundell/Splash) | 终端内的 Markdown 渲染与语法高亮 |

`MLXVLM` 是必需依赖而非可选项：Gemma 4 和 Ornith 检查点声明了 `*ForConditionalGeneration`，这些架构注册在 `MLXVLM` 中而非 `MLXLLM` 中。

依赖更新通过 Dependabot PR 推送——SwiftPM 依赖每周一次，GitHub Actions 每月一次。mlx-swift 迭代很快，新的模型架构往往只差一个版本发布，这也是每周更新一次的原因。

---

## 参与贡献

欢迎提交 Issue 和 Pull Request。提交 PR 前请先确认以下检查通过：

```bash
swift build          # must compile
swift test           # must stay green
```

切勿提交凭据、令牌或 `*.sqlite3` 状态文件——`.gitignore` 已屏蔽常见敏感文件，且发布工作流会对整个代码树执行密钥扫描。
