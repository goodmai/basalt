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

Benchmarked on Apple Silicon (Unified Memory):

| Model | Size | RAM Usage | TTFT | Avg TPS | Command |
|---|---|---|---|---|---|
| **Gemma 4 2B** | 2B | ~2.7 GB | 0.16s | 60.4 | `--model mlx-community/gemma-4-e2b-it-4bit` |
| **Gemma 4 4B** | 4B | ~4.3 GB | 0.09s | 51.8 | `--model mlx-community/gemma-4-e4b-it-4bit` |
| **Gemma 4 31B** | 31B | **~17.3 GB** | **10.2s** | 13.5 | `--model mlx-community/gemma-4-31b-it-4bit` |

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
