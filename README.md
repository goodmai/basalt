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
| **Disk** | 2–30 GB depending on model |

---

## Install

**Option 1: Homebrew (Recommended)**
```bash
brew tap your-org/gemma
brew install gemma
```

**Option 2: Install Script**
```bash
curl -fsSL https://raw.githubusercontent.com/your-org/GemmaServer/main/install.sh | bash
```

**Option 3: Build from source**
```bash
git clone https://github.com/your-org/GemmaServer
cd GemmaServer
swift build -c release
sudo cp .build/release/GemmaServer /usr/local/bin/gemma
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
  mlx-community/gemma-4-31b-it-4bit               4bit    31B       64k
```

### 2. Download a model

**Interactive picker** (features fast multi-threaded download):
```bash
swift run GemmaServer models download
```

**Direct download:**
```bash
swift run GemmaServer models download mlx-community/gemma-4-e4b-it-4bit
```

### 3. Start the server

```bash
# Start with the balanced 4B model (REST + MCP)
swift run GemmaServer serve --model mlx-community/gemma-4-e4b-it-4bit --rest
```

---

## Interactive Chat

You can chat with the model directly in your terminal:

```bash
swift run GemmaServer chat --model mlx-community/gemma-4-e4b-it-4bit
```

---

## REST API

Base URL: `http://127.0.0.1:8080`

### POST /api/v1/auth/login

```bash
curl -s http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'
```

**Response:**
```json
{ "token": "eyJ0eXAiOiJKV1QiLCJhbGci..." }
```

### POST /api/v1/generate

```bash
curl -s http://localhost:8080/api/v1/generate \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum entanglement in one paragraph.",
    "maxTokens": 256
  }' | python3 -m json.tool
```

**Request Fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | required | Input text |
| `maxTokens` | int | 65536 | Max tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0–2) |
| `topP` | float | 0.9 | Nucleus sampling p |

**Response:**

```json
{
  "generatedText": "Quantum entanglement is a phenomenon...",
  "promptTokens": 12,
  "completionTokens": 87,
  "tokensPerSecond": 24.5,
  "generationTime": 3.55,
  "timeToFirstToken": 0.15,
  "memory": {
    "peakBytes": 2708605545,
    "activeBytes": 2639005122,
    "cacheBytes": 142525990
  },
  "finishReason": "stop"
}
```

---

## Models & Performance

**Tested on Apple Silicon M-series (24 GB Unified Memory)**

All models below are verified working with GemmaServer. Performance metrics: TPS (tokens/sec), TTFT (time to first token), RAM (active memory during inference).

### Recommended Models (by use case)

| Use Case | Model | Size | RAM | TPS | TTFT | Model ID |
|---|---|---|---|---|---|---|
| **Fastest** | Qwen3.5 4B | 4B | 2.3 GB | 92.0 | 0.053s | `mlx-community/Qwen3.5-4B-4bit` |
| **Balanced** | Qwen3.5 9B OptiQ | 9B | 5.8 GB | 36.9 | 0.212s | `mlx-community/Qwen3.5-9B-OptiQ-4bit` |
| **Code** | Qwen2.5-Coder 7B | 7B | 4.1 GB | 59.8 | 0.094s | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` |
| **Flagship** | Qwen3.6 27B | 27B | 14.5 GB | 10.8 | 1.959s | `mlx-community/Qwen3.6-27B-4bit` |

### All Verified Models

#### Gemma 4 (Google)
- `mlx-community/gemma-4-e2b-it-4bit` — 2B, ~2.7 GB RAM
- `mlx-community/gemma-4-e4b-it-4bit` — 4B, ~4.3 GB RAM  
- `mlx-community/gemma-4-31b-it-4bit` — 31B, ~17 GB RAM

#### Qwen3.5 / Qwen3.6 (Alibaba, April 2026)
- `mlx-community/Qwen3.5-4B-4bit` — 4B, 2.3 GB, **92 TPS** ⚡
- `mlx-community/Qwen3.5-9B-OptiQ-4bit` — 9B, 5.8 GB, 37 TPS (best quantization)
- `mlx-community/Qwen3.6-27B-4bit` — 27B, 14.5 GB, 11 TPS (newest flagship)

#### Qwen2.5-Coder (code generation)
- `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` — 7B, 4.1 GB, **60 TPS** 💻

---

## MCP Integration (Cursor / Claude)

Add to your MCP config:

```json
{
  "mcpServers": {
    "gemma": {
      "command": "swift",
      "args": [
        "run", "--package-path", "/path/to/GemmaServer",
        "GemmaServer", "serve", "--model", "mlx-community/gemma-4-e4b-it-4bit", "--mcp"
      ]
    }
  }
}
```

---

## Development

```bash
# Run tests (51 tests)
swift test

# Performance Benchmark CLI
swift run PerformanceBenchmark --model mlx-community/gemma-4-e4b-it-4bit
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `ml-explore/mlx-swift` | 0.31.3 | MLX core + Metal ops |
| `ml-explore/mlx-swift-lm` | 3.31.3 | MLXLLM inference engine |
| `hummingbird-project/hummingbird` | 2.6.0 | HTTP server (REST/A2A) |
| `hummingbird-project/hummingbird-auth` | 2.0.0 | Authentication middleware |
| `stephencelis/SQLite.swift` | 0.16.0 | Local SQLite database for auth |
| `vapor/jwt-kit` | 4.13.0 | JWT generation and verification |
| `apple/swift-crypto` | 3.0.0 | Password hashing |
| `apple/swift-argument-parser` | 1.5.0 | CLI subcommands |
