# Gemm

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
| **Hardware** | Apple Silicon M1–M4, Unified Memory |
| **Disk** | 2–30 GB depending on model |

---

## Quick Start

```bash
git clone https://github.com/your-org/Gemm
cd Gemm

# Build
swift build -c release

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

`Gemma` sets `ANTHROPIC_AUTH_TOKEN=local` (not `ANTHROPIC_API_KEY`), so your real Anthropic credentials in other terminals are untouched.

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
