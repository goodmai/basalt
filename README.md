# Gemm

[English](README.md) · [Русский](README.ru.md)

[![Release](https://github.com/goodmai/basalt/actions/workflows/release.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/release.yml)
[![Security Audit](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

[![Platform](https://img.shields.io/badge/platform-macOS%2015%2B%20·%20Apple%20Silicon-black.svg)](#requirements)
[![Swift 6](https://img.shields.io/badge/Swift-6-black.svg)](Package.swift)
[![Homebrew](https://img.shields.io/badge/brew-goodmai%2Fbasalt%2Fgemm-black.svg)](#install-with-homebrew)
[![Ladder](https://img.shields.io/badge/ladder-6%20tasks-black.svg)](#benchmarks)
[![ARC-AGI](https://img.shields.io/badge/ARC--AGI-pass%402-black.svg)](#benchmarks)

Local LLM inference server for Apple Silicon. Runs Gemma 4, Qwen 3, and other MLX-compatible models entirely on-device (Metal GPU). No authentication, no cloud calls — designed for local development and agentic workflows.

```
┌──────────────────────────────────────────────────────┐
│                       Gemm                           │
│                                                      │
│   MCP stdio ──┐                                      │
│               ├──► ModelOrchestratorActor ──► MLX   │
│   REST :8080 ─┘        (actor, FIFO)      Metal GPU  │
│   WebSocket ──┘                                      │
└──────────────────────────────────────────────────────┘
```

Two transports share a single actor instance — **MCP stdio** for IDE integration (Claude Desktop, Cursor) and **REST HTTP** for agent-to-agent workflows and Claude Code backends.

---

## Requirements

| | |
|---|---|
| **macOS** | 15+ (Sequoia) |
| **Xcode / Swift** | 16+ / Swift 6 |
| **Metal toolchain** | `xcodebuild -downloadComponent MetalToolchain` — Xcode 26 ships the Metal compiler separately, and MLX's kernels cannot be built without it |
| **Hardware** | Apple Silicon M1–M4, Unified Memory |
| **Disk** | 2–30 GB depending on model |

---

## Quick Start

### Install with Homebrew

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew install goodmai/basalt/gemm
```

The tap needs the explicit URL because the repository is not named
`homebrew-basalt`. There is no bottle: the formula builds from source, so Xcode
16+ must be installed, along with the Metal toolchain — on Xcode 26 that is a
separate download:

```bash
xcodebuild -downloadComponent MetalToolchain
```

The formula checks for it before building and stops with that command if it is
missing, rather than burying you in per-kernel compiler errors.

The build does two things, and both matter:

1. `swift build -c release` — the server itself.
2. `scripts/build_metal.swift` — compiles MLX's Metal kernels and puts
   `mlx.metallib` next to the binary in `libexec`. A SwiftPM build of mlx-swift
   ships no metallib, and MLX's fallback lookup is relative to the working
   directory, so without this step `gemm` only runs from the directory holding
   the library. Expect this step to dominate the install time.

```bash
brew install --HEAD goodmai/basalt/gemm   # build main instead of the last tag
brew upgrade goodmai/basalt/gemm
brew uninstall gemm && brew untap goodmai/basalt
```

### Build from source

```bash
git clone https://github.com/goodmai/basalt
cd basalt

# Build
swift build -c release

# Compile the MLX Metal kernels. Required once per checkout: a SwiftPM build of
# mlx-swift ships no metallib, and without it the first inference call fails
# with "Failed to load the default metallib". The script places it next to the
# built binary, so the binary works from any directory.
./scripts/build_metal.swift

# Interactive chat
.build/release/gemm chat --model mlx-community/Qwen3.5-4B-4bit

# REST server on :8080 (OpenAI + Anthropic compatible)
.build/release/gemm serve --model mlx-community/Qwen3.5-4B-4bit --rest

# MCP stdio server (for Claude Desktop / Cursor)
.build/release/gemm serve --model mlx-community/gemma-4-e4b-it-4bit --mcp
```

### One-command launcher: `./Gemma`

The repo includes a self-contained Swift launcher that builds the server (if needed), waits for readiness, and opens Claude Code — all with local env vars scoped to that session:

```bash
chmod +x ./Gemma

./Gemma                                             # Qwen 4B, port 8080
./Gemma --model mlx-community/gemma-4-31b-it-4bit  # Gemma 4 31B
./Gemma --port 8081                                 # custom port
./Gemma -- --model haiku                            # pass --model haiku to claude
```

`Gemma` pins `ANTHROPIC_API_KEY=local` rather than inheriting it, so a real key is never forwarded to the local server or written into the `--settings` payload. Other terminals are untouched.

---

## Commands

`gemm` with no subcommand starts `chat`.

| Command | What it does |
|---|---|
| `gemm onboard` | first-run wizard: picks a model for your machine and downloads it |
| `gemm fit` | reads the hardware and ranks the model catalogue against it |
| `gemm chat --model <id>` | interactive terminal chat |
| `gemm serve --model <id> --rest` | REST server on :8080 (OpenAI + Anthropic compatible) |
| `gemm serve --model <id> --mcp` | MCP stdio server for Claude Desktop / Cursor |
| `gemm models list --author <org>` | browse a HuggingFace author's models |
| `gemm models download <repo-id>` | download into the shared HF cache |
| `gemm models info <repo-id>` | size, quantization, context window |
| `gemm models cache` | what is on disk, and how much of it |
| `gemm models check` | verify a cached model is complete and loadable |
| `gemm cloud configure` | OpenRouter fallback for what does not fit locally |
| `gemm cloud cost` | spend so far on the cloud fallback |

Inside `gemm chat`:

| Input | Effect |
|---|---|
| `/clear` | wipe the conversation and the screen |
| `/color`, `/theme` | cycle the terminal colour theme |
| `exit`, `quit` | leave (no slash) |

The flags worth knowing on `serve`:

| Flag | Why you would reach for it |
|---|---|
| `--reasoning-effort none` | stop a reasoning model from spending the whole budget inside `<think>` |
| `--reasoning-effort xhigh\|medium\|low` | Qwen-family thinking budget |
| `--quant 4bit` | pick a quantization subfolder in repos that ship several |
| `--kv-bits 4\|8` | quantize the KV cache — buys context on a memory-tight machine |
| `--min-p`, `--top-k`, `--seed` | sampling; `--seed` makes a non-greedy run reproducible |
| `--max-tokens` | default generation ceiling (2048–128000) |
| `--dry-run` | memory feasibility check, then exit without loading |
| `--port`, `--host` | default 127.0.0.1:8080 |

---

## Verified Models

Models are downloaded from HuggingFace and cached in `~/.cache/huggingface/hub/`.

```bash
huggingface-cli download mlx-community/Qwen3.5-4B-4bit
huggingface-cli download mlx-community/gemma-4-e4b-it-4bit
```

| Model | Params | RAM | Status on 24GB Mac |
|---|---|---|---|
| `mlx-community/gemma-4-e2b-it-4bit` | 2B | 2.7 GB | ✅ ~110 TPS |
| `mlx-community/gemma-4-e4b-it-4bit` | 4B | 4.3 GB | ✅ ~85 TPS |
| `mlx-community/Qwen3.5-4B-4bit` | 4B | 2.3 GB | ✅ ~92 TPS |
| `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 7B | 4.1 GB | ✅ ~60 TPS |
| `mlx-community/Qwen3.5-9B-OptiQ-4bit` | 9B | 5.8 GB | ✅ ~37 TPS |
| `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit` | 27B | 15.2 GB | ✅ ~12 TPS (Abliterated) |
| `Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX` | 26B MoE | 14.5 GB | ✅ ~14 TPS (Dynamic Quant) |
| `Ex0bit/Qwen3.6-35B-A3B-PRISM-MLX-NVFP4` | 35B MoE | 20.5 GB | ✅ ~8 TPS (NVFP4) |
| `Ex0bit/Elbaz-Olmo-3-7B-Instruct-abliterated` | 7B | 4.5 GB | ✅ ~55 TPS |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` | 9B | 5.8 GB | ✅ ~45 TPS (reasoning — see note) |
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | 35B MoE | 21 GB | ✅ untested above 24 GB |
| `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (Base BF16) | 27B | ~54 GB | ❌ Needs 64GB+ (Use 4-bit MLX version on 24GB) |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | 26B MoE | 14.5 GB | ❌ Gibberish output |
| `mlx-community/Qwen3.6-27B-4bit` | 27B | 14.5 GB | ❌ Gibberish output |
| `mlx-community/gemma-4-31b-it-4bit` | 31B | 17 GB | ❌ OOM (Needs 32GB+) |

> **Note on 24GB RAM Macs:** 
> - Unquantized 27B/35B models (~54 GB weights) exceed physical memory. Use MLX 4-bit / Dynamic Quant (DQ) quantized weights (e.g. `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit`, `Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX`) which comfortably fit in ~15 GB RAM.
> - Models > 10GB constrain KV-cache size under heavy multi-turn contexts. Gemm's `TokenBudgetCalculator` automatically calculates and caps context budget based on free system RAM.

---

## REST API

Base URL: `http://127.0.0.1:8080` — **no authentication required**.

### Model management

```bash
# List all locally cached models (OpenAI-compatible format)
curl http://localhost:8080/v1/models

# Currently loaded model + readiness status
curl http://localhost:8080/v1/models/current

# Hot-swap to a different model at runtime (blocks until loaded)
curl -s http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/gemma-4-31b-it-4bit"}'
```

`GET /v1/models` returns IDs prefixed with `claude-local/` so Claude Code's automatic model picker adds them. The `display_name` field carries the original HuggingFace repo ID.

### Raw generation

```bash
curl -s http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum entanglement.", "maxTokens": 256}'
```

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | required | Input text |
| `maxTokens` | int | 8192 | Max tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0–2) |
| `topP` | float | 0.9 | Nucleus sampling |

### OpenAI-compatible

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemm",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

Supports streaming SSE, system prompts, and multi-turn conversation. Sending a HuggingFace model ID in the `"model"` field triggers a hot-swap automatically.

### Anthropic-compatible

```bash
curl -s http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemm",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Implements the full Anthropic SSE event sequence (`message_start`, `content_block_start`, `content_block_delta`, `message_stop`). Accepts plain-string and block-array content formats.

### WebSocket streaming

```js
const ws = new WebSocket("ws://localhost:8080/ws/generate");
ws.send(JSON.stringify({ prompt: "Hello", maxTokens: 512 }));
ws.onmessage = e => console.log(JSON.parse(e.data));
```

### Swagger UI

Visit `http://localhost:8080/swagger` for interactive API docs.

---

## Claude Code Integration

### Option A — environment variables (per terminal session)

```bash
# Start Gemm
.build/release/gemm serve --model mlx-community/Qwen3.5-4B-4bit --rest

# In another terminal — scope vars only to this claude process
ANTHROPIC_BASE_URL=http://localhost:8080 \
ANTHROPIC_AUTH_TOKEN=local \
claude
```

`ANTHROPIC_AUTH_TOKEN` sends as `Authorization: Bearer local` (not `x-api-key`), so your real `ANTHROPIC_API_KEY` is never touched.

### Option B — shell function (add to `~/.zshrc`)

```bash
function gemm-claude() {
  ANTHROPIC_BASE_URL=http://localhost:8080          \
  ANTHROPIC_AUTH_TOKEN=local                        \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=mlx-community/gemma-4-e4b-it-4bit    \
  ANTHROPIC_DEFAULT_SONNET_MODEL=mlx-community/Qwen3.5-4B-4bit       \
  ANTHROPIC_DEFAULT_OPUS_MODEL=mlx-community/gemma-4-31b-it-4bit     \
  claude "$@"
}

gemm-claude                   # sonnet alias → Qwen 4B
gemm-claude --model haiku     # haiku alias → Gemma 4B (fastest)
gemm-claude --model opus      # opus alias → Gemma 31B (strongest)
```

### Option C — `./Gemma` launcher

Starts the server and Claude Code in one command (see Quick Start above).

### Model discovery

Claude Code (v2.1.126+) queries `GET /v1/models` at startup and adds returned models to the `/model` picker — but only if the ID starts with `claude` or `anthropic`. Gemm returns IDs in `claude-local/<hf-id>` form so discovery works automatically.

---

## OpenCode Integration

[OpenCode](https://github.com/opencode-ai/opencode) is a terminal-native AI coding agent. Gemm provides a native OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`) and Anthropic-compatible API that connects directly to OpenCode without cloud dependencies or API keys.

### 1. Start Gemm Server

Start Gemm with your desired local model (e.g. Huihui Qwen3.8 Abliterated, Ex0bit MYTHOS 26B, or Qwen3.5 4B):

```bash
# High capability (Abliterated / MoE)
.build/release/gemm serve --model AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit --rest

# Or fast coding model
.build/release/gemm serve --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --rest
```

### 2. Connect OpenCode

#### Option A — Terminal Environment Variables

```bash
# In your project folder:
OPENAI_BASE_URL=http://localhost:8080/v1 \
OPENAI_API_KEY=local \
OPENAI_MODEL=gemm \
opencode
```

#### Option B — OpenCode Configuration File (`~/.config/opencode/config.json` or `opencode.json`)

```json
{
  "provider": "openai",
  "base_url": "http://localhost:8080/v1",
  "api_key": "local",
  "model": "gemm",
  "temperature": 0.7,
  "max_tokens": 16384
}
```

#### Option C — Shell Alias (add to `~/.zshrc`)

```bash
function gemm-opencode() {
  OPENAI_BASE_URL=http://localhost:8080/v1 \
  OPENAI_API_KEY=local \
  OPENAI_MODEL=gemm \
  opencode "$@"
}
```

---

## Downloading Models (mlx-community, huihui-ai, Ex0bit)

Gemm can browse, inspect, and download models from any creator or organization on Hugging Face:

```bash
# List models by author
gemm models list --author Ex0bit
gemm models list --author huihui-ai
gemm models list --author mlx-community

# Search with filters
gemm models list --author Ex0bit --search PRISM
gemm models list --search Huihui-Qwen3.8

# Download specific models
gemm models download AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit
gemm models download Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX
gemm models download Ex0bit/Qwen3.6-35B-A3B-PRISM-MLX-NVFP4

# Interactive picker with custom author
gemm models download --author Ex0bit
```

---

## MCP Integration (Claude Desktop / Cursor)

Add to your MCP config (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "gemm": {
      "command": "/path/to/.build/release/gemm",
      "args": ["serve", "--model", "mlx-community/Qwen3.5-4B-4bit", "--mcp"]
    }
  }
}
```

Available MCP tools:

| Tool | Description |
|---|---|
| `gemma_generate` | Generate text — `prompt`, `maxTokens`, `temperature`, `topP` |
| `gemma_status` | Returns server readiness, version, and current model |
| `playwright_screenshot` | Capture a webpage screenshot via Playwright |
| `gemma_add_knowledge` | Inject custom context into the session |

---

## Project Structure

```
Sources/
  Gem/                      — GemCore library (all logic, importable by tests)
    App/                    — Entry point: CLI root command + routing
    CLI/                    — Subcommands: chat, serve, models, fit, cloud, onboard
    Cloud/                  — OpenRouter / cloud model fallback (CostTracker, ModelRouter)
    Config/                 — ServerConfig value type
    Core/                   — Inference engine, orchestrator, DTOs, errors, utilities
    MCP/                    — MCP JSON-RPC 2.0 stdio server
    REST/                   — Hummingbird 2.x HTTP server
      Middleware/            — (stubs — auth/JWT removed)
    UI/                     — Terminal UI: Markdown, spinner, progress bar, diff, table
  GemBin/                   — Thin executable wrapper: calls GemCore.GemCLI.main()
  PerformanceBenchmark/     — Standalone benchmark CLI target

Tests/
  GemTests/                 — Unit + integration tests
    CLITests/               — FitCommand, PromptContextBuilder
    CloudTests/             — CostTracker, ModelRouter, OpenRouterClient
    UI/                     — DiffRenderer
    UITests/                — Clipboard, Markdown, ProgressBar, Spinner, Table, TerminalUI
    MockInferenceEngine.swift — Controllable mock for orchestrator tests

Gemma                       — Executable Swift launcher (build + serve + claude)
scripts/                    — Build and maintenance Swift scripts
docs/                       — Extended documentation
```

### Source file count (by module)

| Module | Files | Role |
|---|---|---|
| `App/` | 1 | CLI entry point |
| `CLI/` | 10 | Subcommands |
| `Cloud/` | 4 | Cloud routing + cost tracking |
| `Config/` | 1 | Server configuration |
| `Core/` | 13 | Inference engine, orchestrator, DTOs, errors |
| `MCP/` | 1 | MCP stdio server |
| `REST/` | 7 | HTTP controllers + WebSocket |
| `UI/` | 9 | Terminal rendering |
| `GemBin/` | 1 | Executable wrapper |
| `PerformanceBenchmark/` | 1 | Benchmark CLI |
| **Total** | **48** | |

*(4 stubs with `// removed` content remain in `REST/Middleware/` and `UI/SwiftUI/` — they are empty placeholders kept for git history continuity.)*

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
| `scripts/cleanup.swift` | Clean build artifacts and temp files |
| `scripts/cleanup_daily.swift` | Rotate logs and old benchmarks |
| `scripts/archive.sh` | Archive logs/reports/screenshots |
| `scripts/clean_for_github.swift` | Remove secrets and heavy binaries before push |

---

## Benchmarks

Two harnesses, both pointed at a running server, both reporting numbers you can
reproduce on your own machine.

**The ladder to the moon** — `benchmarks/ladder/run.py`. Five rungs, each
strictly harder than the one below it, ending in a full Earth-to-Moon mission.
Nothing is graded on how the answer reads: the model's own code is executed
against inputs the prompt never showed it, so a plausible-looking wrong answer
still fails.

| Rung | Task | What it actually tests |
|---|---|---|
| 1 | Biquadratic, real roots | closed form plus the edge cases: none, repeated, zero |
| 2 | Biquadratic, complex roots | branch handling — all four roots, with multiplicity |
| 3 | Fourier coefficients | numeric integration, checked on sawtooth, square wave, cosine |
| 4 | Pump head | units (mm vs m, L/s vs m³/s) + Colebrook, which has no closed form |
| 5 | **Earth → Moon mission** | exhaust velocity from `sqrt(2ηQ)`, Tsiolkovsky per stage, four-body propagation, lunar braking, soft landing — every constant is in the prompt, nothing recallable from a textbook |

The top rung is scored on six invariants rather than one answer, so a model that
gets the energetics right but never propagates the trajectory still shows a
number. Its prompt lives in `benchmarks/ladder/lunar_prompt.md`; the reference
values in `lunar_task.md` are never shown to the model.

`--warmup` prepends three cheap smoke tests (algebra, Fibonacci, translation)
that any instruct model should clear. They are not part of the ladder score.

**ARC-AGI** — `benchmarks/arc-agi/arc_agi_benchmark.py`, official pass@2 rules.
The corpus is not vendored; fetch it into `benchmarks/arc-agi/dataset/` first
(see `benchmarks/arc-agi/ARC_AGI.md`).

### Results

Measured on an M-series Mac with 24 GB unified memory. `thinking` is the chat
template's `<think>` block: on by default for reasoning models, disabled with
`--reasoning-effort none`.

| Model | Warm-up | Ladder | Top rung | ARC-AGI mini | tok/s |
|---|---|---|---|---|---|
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` (thinking off) | 3/3 | 1/4 ² | not run | 0/3 ¹ | ~45 |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` (thinking on) | 2/3 | 0/4 ² | not run | not run | ~43 |

Ornith 9B rung by rung, thinking off:

| Rung | Result | Detail |
|---|---|---|
| 1 · biquadratic real | ✅ | roots correct, including the repeated and empty cases |
| 2 · biquadratic complex | ❌ | `x⁴ + 1` returns 2 roots instead of 4 |
| 3 · Fourier | ❌ | `a0 = 10.86` for `f(x) = x`, want 0 — the integration is wrong, not the formula |
| 4 · pump head | ❌ | `v = 0.0298` m/s, want 1.79 — never converted mm to m |
| 5 · lunar mission | — | not run |

Warm-up, thinking off: algebra ✅ (163 tok, 4 s), Fibonacci ✅ (757 tok, 16 s),
translation ✅ (303 tok, 7 s — German idiomatic, French paraphrased rather than
translated).

² Rungs 1 and 2 were measured before the ladder was split into separate
prompts: one prompt asked for both functions, the real half was correct and the
complex half was not. Rung 5 has not been run against this model yet.

¹ The ARC-AGI column is a partial run: three training tasks (`007bbfb7`,
`00d62c1b`, `017c7c7b`), none solved at pass@2, 330 s total. It was stopped
before the remaining two, so treat it as a smoke test rather than a score.

The validators are themselves checked, without a model or a GPU:

```bash
python3 benchmarks/ladder/selfcheck.py    # 9 assertions, both directions
```

With thinking **on**, all three hard tasks and the translation ended in
`finish_reason: length` — the budget went into the `<think>` block and no answer
was ever emitted. That is a budget failure, not a capability ceiling, which is
why the table above reports the two modes separately.

### Reproduce

```bash
gemm serve --model ornith-ai/Ornith-1.5-9B-MLX-4bit --rest --reasoning-effort none &

python3 benchmarks/ladder/run.py --port 8080 --warmup --json ladder.json
python3 benchmarks/arc-agi/arc_agi_benchmark.py --engine mlx --split training --limit 5
```

Adding a model to the table is one run of each — PRs with new rows are welcome.

---

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
| [Rainbow](https://github.com/onevcat/Rainbow), [console-kit](https://github.com/vapor/console-kit) | terminal colour and layout |
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

---

## License

[MIT](LICENSE) © 2026 goodmai

Model weights are **not** covered by this license — each model on HuggingFace
carries its own terms.
