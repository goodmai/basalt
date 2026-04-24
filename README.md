# GemmaServer

Local Gemma 4 inference server for Apple Silicon.  
Dual interface: **MCP** (stdio) for IDE integration + **REST/A2A** (HTTP) for agent-to-agent calls.

```
┌─────────────────────────────────────────────────────┐
│                   GemmaServer                       │
│                                                     │
│   MCP stdio ──┐                                     │
│               ├──► ModelOrchestratorActor ──► MLX  │
│   REST :8080 ─┘         (actor, FIFO)     Metal GPU │
└─────────────────────────────────────────────────────┘
```

---

## Requirements

| | |
|---|---|
| **macOS** | 15+ (Sequoia) |
| **Xcode** | 16+ / Swift 6 |
| **Hardware** | Apple Silicon (M1–M4), Unified Memory |
| **Disk** | 2–20 GB depending on model |

---

## Install

```bash
git clone https://github.com/your-org/GemmaServer
cd GemmaServer
swift build -c release
```

Optional — install to PATH:
```bash
cp .build/release/GemmaServer /usr/local/bin/
```

---

## Quick Start

### 1. Browse available models

```bash
swift run GemmaServer models list
```

```
Fetching mlx-community/gemma-4* from Hugging Face…

  MODEL                                           QUANT   PARAMS   ↓
  ────────────────────────────────────────────────────────────────────
  mlx-community/gemma-4-e2b-it-4bit               4bit    2B       228k
  mlx-community/gemma-4-e4b-it-4bit               4bit    4B        46k
  mlx-community/gemma-4-26b-a4b-it-4bit           4bit    26B      101k
  mlx-community/gemma-4-31b-it-4bit               4bit    31B       64k
```

Filter by quantization:
```bash
swift run GemmaServer models list --quant 8bit
swift run GemmaServer models list --search gemma-3
```

### 2. Download a model

**Interactive picker** (recommended for first use):
```bash
swift run GemmaServer models download
```

**Direct download:**
```bash
swift run GemmaServer models download mlx-community/gemma-4-e2b-it-4bit
```

Models go to `~/.cache/huggingface/hub/` — same cache as Python `transformers`.

Check what's already downloaded:
```bash
swift run GemmaServer models cache
```

### 3. Start the server

```bash
# Both interfaces (MCP + REST) — default
swift run GemmaServer serve --model mlx-community/gemma-4-e2b-it-4bit

# REST only (for curl / agent-to-agent)
swift run GemmaServer serve --model mlx-community/gemma-4-e2b-it-4bit --rest

# MCP only (for Cursor / Claude Desktop)
swift run GemmaServer serve --model mlx-community/gemma-4-e2b-it-4bit --mcp

> **Note:** The `--mcp` flag enables the Model Context Protocol via standard input/output using JSON-RPC 2.0. **You cannot chat with this interface by typing plain text (e.g., "hello").** If you type raw text, you will get a `"Parse error"`. This interface is meant to be consumed programmatically by IDEs like Cursor or Claude Desktop. To interact with the model manually, use the REST API (`curl`).
```

---

## REST API

Base URL: `http://127.0.0.1:8080`

### POST /api/v1/generate

```bash
curl -s http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum entanglement in one paragraph.",
    "maxTokens": 256,
    "temperature": 0.7,
    "topP": 0.9
  }' | python3 -m json.tool
```

**Request:**

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | required | Input text |
| `maxTokens` | int | 1024 | Max tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0–2) |
| `topP` | float | 0.9 | Nucleus sampling p |

**Response:**

```json
{
  "generatedText": "Quantum entanglement is a phenomenon...",
  "promptTokens": 12,
  "completionTokens": 87,
  "tokensPerSecond": 24.5,
  "finishReason": "stop"
}
```

### GET /api/v1/health

```bash
curl -s http://localhost:8080/api/v1/health | python3 -m json.tool
```

```json
{
  "status": "ok",
  "modelId": "gemma-4-e2b-it-4bit",
  "isReady": true,
  "version": "0.1.0"
}
```

---

## MCP Integration

Add to `~/.cursor/mcp.json` (Cursor) or `claude_desktop_config.json` (Claude Desktop):

```json
{
  "mcpServers": {
    "gemma-local": {
      "command": "swift",
      "args": [
        "run", "--package-path", "/path/to/GemmaServer",
        "GemmaServer", "serve",
        "--model", "mlx-community/gemma-4-e2b-it-4bit",
        "--mcp"
      ]
    }
  }
}
```

**Available MCP tools:**

| Tool | Description |
|---|---|
| `gemma_generate` | Generate text from a prompt |
| `gemma_status` | Check model readiness and version |

---

## Models

Recommended models by RAM:

| RAM | Model | Command |
|---|---|---|
| 8 GB | `gemma-4-e2b-it-4bit` | `--model mlx-community/gemma-4-e2b-it-4bit` |
| 16 GB | `gemma-4-e4b-it-4bit` | `--model mlx-community/gemma-4-e4b-it-4bit` |
| 32 GB | `gemma-4-26b-a4b-it-4bit` | `--model mlx-community/gemma-4-26b-a4b-it-4bit` |
| 64 GB+ | `gemma-4-31b-it-4bit` | `--model mlx-community/gemma-4-31b-it-4bit` |

---

## Architecture

```
Sources/
├── GemmaServer/          # Library (GemmaServerCore)
│   ├── App/
│   │   └── EntryPoint.swift        # Root AsyncParsableCommand
│   ├── Config/
│   │   └── ServerConfig.swift      # CLI options
│   ├── Core/
│   │   ├── GemmaServerError.swift  # Typed throws — E = E_io ∪ E_memory ∪ E_inference
│   │   ├── DTOs.swift              # GenerationRequest / Response (Sendable, Codable)
│   │   ├── ModelOrchestratorActor.swift  # Single queue for both transports
│   │   └── MLXInferenceEngine.swift      # Real MLXLLM inference (actor-isolated)
│   ├── MCP/
│   │   └── MCPServer.swift         # JSON-RPC 2.0 over newline-delimited stdio
│   ├── REST/
│   │   ├── RESTServer.swift        # Hummingbird 2.x HTTP server
│   │   └── GenerateController.swift
│   └── CLI/
│       ├── HuggingFaceHub.swift    # HF Hub API + downloader (URLSession)
│       ├── ModelsCommand.swift     # list / download / info / cache
│       └── ServeCommand.swift      # serve subcommand
└── GemmaServerBin/       # Executable (thin wrapper)
    └── main.swift                  # GemmaServerCLI.main()

Tests/
└── GemmaServerTests/     # Swift Testing — 34 tests
    ├── MockInferenceEngine.swift
    ├── DTOTests.swift
    ├── ErrorTests.swift
    ├── ModelCacheTests.swift
    └── OrchestratorTests.swift
```

**Stability invariant:** λ_mcp + λ_rest < μ_inference  
Swift `actor` guarantees FIFO serialization without explicit locks.

---

## Development

```bash
# Run tests
swift test

# Build release binary
swift build -c release

# Watch logs (MCP and REST log to stderr)
swift run GemmaServer serve --model mlx-community/gemma-4-e2b-it-4bit 2>&1 | tee server.log
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `ml-explore/mlx-swift` | 0.31.3 | MLX core + Metal ops |
| `ml-explore/mlx-swift-lm` | 3.31.3 | MLXLLM — Gemma 4 model loading + inference |
| `huggingface/swift-transformers` | 1.3.0 | Tokenizers for model loading |
| `hummingbird-project/hummingbird` | 2.22.0 | HTTP server (REST/A2A) |
| `apple/swift-argument-parser` | 1.7.1 | CLI subcommands |
| `stephencelis/SQLite.swift` | 0.15.3 | Local SQLite database for auth |
| `vapor/jwt-kit` | 4.13.0 | JWT generation and verification |
s |
| `ml-explore/mlx-swift-lm` | 3.31.3 | MLXLLM — Gemma 4 model loading + inference |
| `huggingface/swift-transformers` | 1.3.0 | Tokenizers for model loading |
| `hummingbird-project/hummingbird` | 2.22.0 | HTTP server (REST/A2A) |
| `apple/swift-argument-parser` | 1.7.1 | CLI subcommands |
