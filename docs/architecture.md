# Architecture

```
┌──────────────────────────────────────────────────────┐
│                       Basalt                           │
│                                                      │
│   MCP stdio ──┐                                      │
│               ├──► ModelOrchestratorActor ──► MLX   │
│   REST :8080 ─┘        (actor, FIFO)      Metal GPU  │
└──────────────────────────────────────────────────────┘
```

Two transports share a single actor instance — **MCP stdio** for IDE integration (Claude Desktop, Cursor) and **REST HTTP** for agent-to-agent workflows and Claude Code backends.

## Project Structure

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

Only `BasaltApp` may import SwiftUI/AppKit; every target the CLI links stays headless (enforced by `ModuleBoundaryTests`).
Symbols newly shared across targets use the `package` access level (existing public API stays `public`).

---

## Architecture Notes

**Dynamic Quantization & MoE** — `MLXInferenceEngine` automatically parses `config.json` to detect 2/3/4/8-bit quantization and dynamically switches between Dense and Sparse MoE (Mixture of Experts) architectures (e.g. using highly optimized `SwitchGLU` Metal kernels).

**Actor isolation** — `ModelOrchestratorActor` is a Swift 6 actor. All inference calls are serialised (FIFO) without explicit locking. Both MCP and REST share one actor instance.

**Model hot-swap** — `switchModel(to:)` resolves the HuggingFace repo ID from the local cache, unloads the current model (`container = nil` + `MLX.GPU.clearCache()`), then loads the new weights. The actor ensures in-flight requests finish before the swap begins.

**Timeout protection** — every `generate` and `generateStream` call is wrapped in a 5-minute timeout with cooperative Task cancellation, preventing model hangs from blocking the server indefinitely.

**Streaming** — Hummingbird 2's `AsyncStream<ByteBuffer>` response body is used for both SSE formats: OpenAI (`data: {...}`) and Anthropic (`event: content_block_delta`).

**Think-block stripping** — `generateStream` runs a state machine to suppress `<think>...</think>` reasoning blocks before forwarding tokens to clients.

**ID translation** — `ModelsController` translates between HuggingFace repo IDs (`mlx-community/Qwen3.5-4B-4bit`) and the `claude-local/mlx-community--Qwen3.5-4B-4bit` form required by Claude Code's model discovery filter. Both forms are accepted in API requests.

---

## Maintenance Scripts

| Script | Purpose |
|---|---|
| `scripts/build_metal.swift` | Compile Metal GPU kernels (skips if up to date) |
| `scripts/cleanup.swift` | `--mode=daily` rotates logs/images/reports, `--mode=release` removes build/runtime artifacts before publishing, `--mode=all` both; dry run unless `--apply` |
| `scripts/archive.sh` | Archive logs/reports/screenshots |

## Dependencies

Everything is a SwiftPM package; there is no CocoaPods, no Node, no Python at
runtime. `Package.resolved` pins the exact revisions.

| Package | Used for |
|---|---|
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | the array/NN layer on Metal |
| [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | model architectures and loading — `MLXLLM`, `MLXVLM`, `MLXLMCommon` |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | tokenizers and chat templates |
| [hummingbird](https://github.com/hummingbird-project/hummingbird) | the HTTP server |
| [swift-argument-parser](https://github.com/apple/swift-argument-parser) | the CLI |
| [Rainbow](https://github.com/onevcat/Rainbow) | terminal colour and layout |
| [swift-markdown](https://github.com/apple/swift-markdown), [Splash](https://github.com/JohnSundell/Splash) | Markdown rendering and syntax highlighting in the terminal |

`MLXVLM` is not optional: Gemma 4 and the Ornith checkpoints declare
`*ForConditionalGeneration`, and those architectures are registered there rather
than in `MLXLLM`.

Updates arrive as Dependabot PRs — weekly for SwiftPM, monthly for the GitHub
Actions. mlx-swift moves fast and a model architecture is often one release
away, which is the reason for the weekly cadence.

---

## Contributing

Issues and pull requests are welcome. Before opening a PR:

```bash
swift build          # must compile
swift test           # must stay green
```

Never commit credentials, tokens, or `*.sqlite3` state — `.gitignore` blocks the
usual suspects, and the release workflow runs a secret scan over the tree.
