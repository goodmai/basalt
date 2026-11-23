# Architektur

> Languages: [English](../../architecture.md) · [Русский](../ru/architecture.md) · [中文](../zh/architecture.md) · [Español](../es/architecture.md) · [Français](../fr/architecture.md) · [日本語](../ja/architecture.md) · **Deutsch** · [한국어](../ko/architecture.md) · [Bahasa Indonesia](../id/architecture.md)

```
┌──────────────────────────────────────────────────────┐
│                       Basalt                           │
│                                                      │
│   MCP stdio ──┐                                      │
│               ├──► ModelOrchestratorActor ──► MLX   │
│   REST :8080 ─┘        (actor, FIFO)      Metal GPU  │
└──────────────────────────────────────────────────────┘
```

Zwei Transporte teilen sich eine einzige Actor-Instanz — **MCP stdio** für die IDE-Integration (Claude Desktop, Cursor) und **REST HTTP** für Agent-to-Agent-Workflows sowie Claude-Code-Backends.

## Projektstruktur

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

Nur `BasaltApp` darf SwiftUI/AppKit importieren; jedes Target, das die CLI linkt, bleibt headless (durch `ModuleBoundaryTests` erzwungen).
Symbole, die neu über Targets hinweg geteilt werden, nutzen die Zugriffsstufe `package` (die bestehende öffentliche API bleibt `public`).

---

## Architekturhinweise

**Dynamische Quantisierung, MACaQ & MoE** — `MLXInferenceEngine` parst `config.json` (einschließlich `macaq_quantization`, `flatquant_quantization`, `lirquant_quantization`), um Dense- und Sparse-MoE-Checkpoints auf optimierte Metal-Kernel zu verteilen (`SwitchGLU`, Single-Pass-`MACaQSwitchLinear` WHT + `MLX.gatherQuantizedMM` `#628`, `E8P` 2-bit Lattice, `QTIP`, `STA`-3D-Video-Attention `#609`).

**SDK, Governance & Tiered Memory (`v0.12`)** — `BasaltFoundationModels` stellt eine Swift-SDK-Brücke bereit (`#606`); `VideoRunnerCore` steuert die Phasen der 24-GB-UMA-Videogenerierung (`#611`); `TieredKVHierarchyCore` verwaltet L0/L1/L2-Unified-Memory-KV-Tiers (`#627`); `SemanticCacheActor`, `HITLApprovalActor` und `SymbolicKnowledgeGraph` setzen Enterprise-Routing und Human-in-the-Loop-Freigaben durch (`#625`, `#626`).

**Actor-Isolation** — `ModelOrchestratorActor` ist ein Swift-6-Actor. Alle Inferenzaufrufe werden ohne explizites Locking (FIFO) serialisiert. Sowohl MCP als auch REST teilen sich eine einzige Actor-Instanz.

**Modell-Hot-Swap** — `switchModel(to:)` löst die HuggingFace-Repo-ID aus dem lokalen Cache auf, entlädt das aktuelle Modell (`container = nil` + `MLX.GPU.clearCache()`) und lädt dann die neuen Gewichte. Der Actor stellt sicher, dass laufende Anfragen abgeschlossen werden, bevor der Wechsel beginnt.

**Timeout-Schutz** — jeder Aufruf von `generate` und `generateStream` ist in ein 5-Minuten-Timeout mit kooperativer Task-Cancellation eingebettet, sodass hängende Modelle den Server nicht dauerhaft blockieren können.

**Streaming** — der `AsyncStream<ByteBuffer>`-Response-Body von Hummingbird 2 wird für beide SSE-Formate verwendet: OpenAI (`data: {...}`) und Anthropic (`event: content_block_delta`).

**Entfernen von Think-Blöcken** — `generateStream` betreibt eine Zustandsmaschine, um `<think>...</think>`-Reasoning-Blöcke zu unterdrücken, bevor Tokens an Clients weitergeleitet werden.

**ID-Übersetzung** — `ModelsController` übersetzt zwischen HuggingFace-Repo-IDs (`mlx-community/Qwen3.5-4B-4bit`) und dem Format `claude-local/mlx-community--Qwen3.5-4B-4bit`, das der Modell-Erkennungsfilter von Claude Code erfordert. In API-Anfragen werden beide Formate akzeptiert.

---

## Wartungsskripte

| Skript | Zweck |
|---|---|
| `scripts/build_metal.swift` | Metal-GPU-Kernel kompilieren (wird übersprungen, falls aktuell) |
| `scripts/cleanup.swift` | `--mode=daily` rotiert Logs/Bilder/Berichte, `--mode=release` entfernt Build-/Laufzeitartefakte vor dem Veröffentlichen, `--mode=all` beides; Dry-Run, sofern nicht `--apply` angegeben ist |
| `scripts/archive.sh` | Logs/Berichte/Screenshots archivieren |

## Abhängigkeiten

Alles ist ein SwiftPM-Paket; zur Laufzeit gibt es weder CocoaPods noch Node oder Python.
`Package.resolved` fixiert die exakten Revisionen.

| Paket | Verwendet für |
|---|---|
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | die Array-/NN-Schicht auf Metal |
| [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | Modellarchitekturen und Laden — `MLXLLM`, `MLXVLM`, `MLXLMCommon` |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | Tokenizer und Chat-Templates |
| [hummingbird](https://github.com/hummingbird-project/hummingbird) | der HTTP-Server |
| [swift-argument-parser](https://github.com/apple/swift-argument-parser) | die CLI |
| [Rainbow](https://github.com/onevcat/Rainbow) | Farben und Layout im Terminal |
| [swift-markdown](https://github.com/apple/swift-markdown), [Splash](https://github.com/JohnSundell/Splash) | Markdown-Rendering und Syntaxhervorhebung im Terminal |

`MLXVLM` ist nicht optional: Gemma 4 und die Ornith-Checkpoints deklarieren
`*ForConditionalGeneration`, und diese Architekturen sind dort statt in `MLXLLM`
registriert.

Updates treffen als Dependabot-PRs ein — wöchentlich für SwiftPM, monatlich für die GitHub
Actions. mlx-swift entwickelt sich schnell und eine Modellarchitektur ist oft nur ein Release
entfernt, was der Grund für den wöchentlichen Rhythmus ist.

---

## Mitwirken

Issues und Pull Requests sind willkommen. Vor dem Erstellen eines PR:

```bash
swift build          # must compile
swift test           # must stay green
```

Committen Sie niemals Zugangsdaten, Tokens oder `*.sqlite3`-Statusdateien — `.gitignore` blockiert die
üblichen Verdächtigen, und der Release-Workflow führt einen Secret-Scan über den gesamten Dateibaum aus.
