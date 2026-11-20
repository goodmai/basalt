# アーキテクチャ

> Languages: [English](../../architecture.md) · [Русский](../ru/architecture.md) · [中文](../zh/architecture.md) · [Español](../es/architecture.md) · [Français](../fr/architecture.md) · **日本語** · [Deutsch](../de/architecture.md)

```
┌──────────────────────────────────────────────────────┐
│                       Basalt                           │
│                                                      │
│   MCP stdio ──┐                                      │
│               ├──► ModelOrchestratorActor ──► MLX   │
│   REST :8080 ─┘        (actor, FIFO)      Metal GPU  │
└──────────────────────────────────────────────────────┘
```

2 つのトランスポートが単一のアクターインスタンスを共有します — IDE 統合（Claude Desktop、Cursor）向けの **MCP stdio** と、エージェント間ワークフローおよび Claude Code バックエンド向けの **REST HTTP** です。

## プロジェクト構成

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

SwiftUI/AppKit をインポートできるのは `BasaltApp` のみです。CLI がリンクするすべてのターゲットはヘッドレスのまま維持されます（`ModuleBoundaryTests` によって強制）。
ターゲット間で新たに共有されるシンボルには `package` アクセスレベルを使用します（既存の公開 API は `public` を維持）。

---

## アーキテクチャノート

**動的量子化、MACaQ、MoE** — `MLXInferenceEngine` は `config.json`（`macaq_quantization`、`flatquant_quantization`、`lirquant_quantization` を含む）を解析し、Dense および Sparse MoE チェックポイントを最適化された Metal カーネル（`SwitchGLU`、シングルパス `MACaQSwitchLinear` WHT + `MLX.gatherQuantizedMM` `#628`、`E8P` 2-bit ラティス、`QTIP`、`STA` 3D 動画アテンション `#609`）にディスパッチします。

**SDK、ガバナンス、階層型メモリ（`v0.12`）** — `BasaltFoundationModels` は Swift SDK ブリッジを提供し（`#606`）、`VideoRunnerCore` は 24 GB UMA での動画生成をフェーズ分割します（`#611`）。`TieredKVHierarchyCore` は L0/L1/L2 のユニファイドメモリ KV 階層を管理し（`#627`）、`SemanticCacheActor`、`HITLApprovalActor`、`SymbolicKnowledgeGraph` はエンタープライズ向けルーティングと Human-in-the-Loop 承認を適用します（`#625`、`#626`）。

**アクター分離（Actor isolation）** — `ModelOrchestratorActor` は Swift 6 の actor です。すべての推論呼び出しは明示的なロックなしで（FIFO 順に）直列化されます。MCP と REST の両方が 1 つのアクターインスタンスを共有します。

**モデルのホットスワップ** — `switchModel(to:)` はローカルキャッシュから HuggingFace リポジトリ ID を解決し、現在のモデルをアンロード（`container = nil` + `MLX.GPU.clearCache()`）してから新しい重みをロードします。アクターにより、処理中のリクエストが完了してからスワップが開始されることが保証されます。

**タイムアウト保護** — すべての `generate` および `generateStream` 呼び出しは協調的タスクキャンセル（cooperative Task cancellation）付きの 5 分間タイムアウトでラップされており、モデルのハングによってサーバーが永久にブロックされるのを防ぎます。

**ストリーミング** — Hummingbird 2 の `AsyncStream<ByteBuffer>` レスポンスボディを、OpenAI（`data: {...}`）と Anthropic（`event: content_block_delta`）の両方の SSE フォーマットに使用しています。

**think ブロックの除去** — `generateStream` はステートマシンを実行し、クライアントへトークンを転送する前に `<think>...</think>` 推論ブロックを抑制します。

**ID 変換** — `ModelsController` は HuggingFace リポジトリ ID（`mlx-community/Qwen3.5-4B-4bit`）と、Claude Code のモデル検出フィルターが要求する `claude-local/mlx-community--Qwen3.5-4B-4bit` 形式との間で相互変換を行います。API リクエストではどちらの形式も受け付けます。

---

## メンテナンススクリプト

| スクリプト | 用途 |
|---|---|
| `scripts/build_metal.swift` | Metal GPU カーネルをコンパイル（最新の場合はスキップ） |
| `scripts/cleanup.swift` | `--mode=daily` はログ/画像/レポートをローテーション、`--mode=release` は公開前にビルド/ランタイム成果物を削除、`--mode=all` は両方を実行。`--apply` がない場合はドライラン |
| `scripts/archive.sh` | ログ/レポート/スクリーンショットをアーカイブ |

## 依存関係

すべてが SwiftPM パッケージであり、CocoaPods、Node、実行時の Python は一切不要です。`Package.resolved` が正確なリビジョンを固定しています。

| パッケージ | 用途 |
|---|---|
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | Metal 上の配列 / NN レイヤー |
| [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | モデルアーキテクチャとローディング — `MLXLLM`、`MLXVLM`、`MLXLMCommon` |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | トークナイザーとチャットテンプレート |
| [hummingbird](https://github.com/hummingbird-project/hummingbird) | HTTP サーバー |
| [swift-argument-parser](https://github.com/apple/swift-argument-parser) | CLI |
| [Rainbow](https://github.com/onevcat/Rainbow) | ターミナルの配色とレイアウト |
| [swift-markdown](https://github.com/apple/swift-markdown), [Splash](https://github.com/JohnSundell/Splash) | ターミナルでの Markdown レンダリングとシンタックスハイライト |

`MLXVLM` は必須です。Gemma 4 と Ornith のチェックポイントは `*ForConditionalGeneration` を宣言しており、それらのアーキテクチャは `MLXLLM` ではなく `MLXVLM` 側に登録されています。

アップデートは Dependabot PR として届きます（SwiftPM は毎週、GitHub Actions は毎月）。mlx-swift は開発速度が速く、新しいモデルアーキテクチャが 1 リリース先で対応されることも多いため、毎週の更新頻度としています。

---

## コントリビュート

Issue やプルリクエストを歓迎します。PR を作成する前に以下を確認してください：

```bash
swift build          # must compile
swift test           # must stay green
```

認証情報、トークン、`*.sqlite3` の状態ファイルを決してコミットしないでください。主要なものは `.gitignore` でブロックされており、リリースワークフローでもツリー全体に対してシークレットスキャンが実行されます。
