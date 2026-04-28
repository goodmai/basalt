# GemmaServer

Local LLM inference server for Apple Silicon with dual interface architecture.  
**MCP** (stdio) for IDE integration + **REST** (HTTP) for agent-to-agent communication.

```
┌─────────────────────────────────────────────────────┐
│                   GemmaServer                       │
│                                                     │
│   MCP stdio ──┐                                     │
│               ├──► ModelOrchestratorActor ──► MLX  │
│   REST :8080 ─┘         (actor, FIFO)     Metal GPU │
└─────────────────────────────────────────────────────┘
```

**✨ Latest Updates:**
- ✅ **101 tests passing** - Full integration test suite
- ✅ **80% test coverage** - Epic 2 Integration Testing
- ✅ **Session Analytics** - Beautiful exit summary (Epic 13)
- ✅ **Swift 6** - Strict concurrency, actor isolation
- ✅ **4 verified models** - Gemma 4, Qwen3.5/3.6, Qwen2.5-Coder

---

## Requirements

| | |
|---|---|
| **macOS** | 14+ (Sonoma) or 15+ (Sequoia) |
| **Xcode** | 16+ / Swift 6 |
| **Hardware** | Apple Silicon (M1–M4), Unified Memory |
| **Disk** | 2–30 GB depending on model |

---

## Quick Install

**Option 1: Automated setup (recommended)**
```bash
git clone https://github.com/your-org/GemmaServer
cd GemmaServer
./setup.sh
# Choose: 1) Install to /usr/local/bin, or 2) Add alias to shell
```

**Option 2: Manual build**
```bash
git clone https://github.com/your-org/GemmaServer
cd GemmaServer
swift build -c release

# Then choose one:
# A) System-wide install (recommended)
sudo cp .build/release/GemmaServer /usr/local/bin/gemma

# B) Add alias (add to ~/.zshrc or ~/.bashrc)
alias gemma='swift run --package-path /path/to/GemmaServer GemmaServer'

# C) Add to PATH
export PATH="$PATH:/path/to/GemmaServer/.build/release"
```

**Option 3: Run directly with Swift (no installation)**
```bash
# No installation needed - just run from source
swift run GemmaServer --help
```

**Option 4: Homebrew (coming in v0.2.0)**
```bash
brew tap your-org/gemma
brew install gemma
```

---

## Quick Start

### 1. Check available commands

```bash
# If you installed to /usr/local/bin:
gemma --help

# Or run directly:
swift run GemmaServer --help
```

**Available commands:**
```
OVERVIEW: Local LLM inference server for Apple Silicon

USAGE: gemma-server <subcommand>

OPTIONS:
  --version               Show the version.
  -h, --help              Show help information.

SUBCOMMANDS:
  serve                   Start MCP or REST server
  chat                    Interactive chat (coming soon)
  models                  Model management (coming soon)
  benchmark               Run performance benchmarks
```

### 2. Start the server (MCP mode)

```bash
# Run MCP server on stdio (for Cursor/Claude Desktop)
gemma serve --model mlx-community/Qwen3.5-4B-4bit --mcp

# Or with Swift:
swift run GemmaServer serve --model mlx-community/Qwen3.5-4B-4bit --mcp
```

### 3. Start the server (REST mode)

```bash
# Run REST API on http://localhost:8080
gemma serve --model mlx-community/Qwen3.5-4B-4bit --rest

# Or with Swift:
swift run GemmaServer serve --model mlx-community/Qwen3.5-4B-4bit --rest
```

### 4. Run benchmarks

```bash
# Benchmark a model
swift run PerformanceBenchmark --model mlx-community/Qwen3.5-4B-4bit --iterations 10

# Context degradation profiling
gemma serve --model mlx-community/Qwen3.5-4B-4bit --profile-context
```

---

## Models

**Before running, you need to download a model from HuggingFace:**

```bash
# Option 1: Use MLX tools
mlx_lm.convert --hf-path mlx-community/Qwen3.5-4B-4bit

# Option 2: Manual download
# Models are cached in: ~/.cache/huggingface/hub/
git clone https://huggingface.co/mlx-community/Qwen3.5-4B-4bit ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-4bit/snapshots/main
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

**Tested on Apple Silicon M-series (16-24 GB Unified Memory)**

All models below are verified working with GemmaServer. Performance metrics: TPS (tokens/sec), TTFT (time to first token), RAM (active memory during inference).

### ⚡ Recommended Models (by use case)

| Use Case | Model | Size | RAM | TPS | TTFT | Model ID |
|---|---|---|---|---|---|---|
| **Fastest** 🚀 | Qwen3.5 4B | 4B | 2.3 GB | 92.0 | 0.053s | `mlx-community/Qwen3.5-4B-4bit` |
| **Balanced** ⚖️ | Qwen3.5 9B OptiQ | 9B | 5.8 GB | 36.9 | 0.212s | `mlx-community/Qwen3.5-9B-OptiQ-4bit` |
| **Code** 💻 | Qwen2.5-Coder 7B | 7B | 4.1 GB | 59.8 | 0.094s | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` |
| **Flagship** 🏆 | Qwen3.6 27B | 27B | 14.5 GB | 10.8 | 1.959s | `mlx-community/Qwen3.6-27B-4bit` |

### 📦 Model Download

**Models are cached in:** `~/.cache/huggingface/hub/`

```bash
# GemmaServer automatically resolves models from cache
# Just specify the model ID and it will find it

# Example: Start server with cached model
gemma serve --model mlx-community/Qwen3.5-4B-4bit --rest

# If model not found, download manually:
# Method 1: HuggingFace CLI
huggingface-cli download mlx-community/Qwen3.5-4B-4bit

# Method 2: Git clone
git clone https://huggingface.co/mlx-community/Qwen3.5-4B-4bit \
  ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-4bit/snapshots/main
```

### 🧪 All Verified Models

#### Qwen3.5 / Qwen3.6 (Alibaba, April 2026) - **Recommended**
- ✅ `mlx-community/Qwen3.5-4B-4bit` — 4B, 2.3 GB, **92 TPS** ⚡ Best speed
- ✅ `mlx-community/Qwen3.5-9B-OptiQ-4bit` — 9B, 5.8 GB, 37 TPS (best quantization)
- ✅ `mlx-community/Qwen3.6-27B-4bit` — 27B, 14.5 GB, 11 TPS (newest flagship)

#### Qwen2.5-Coder (code generation)
- ✅ `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` — 7B, 4.1 GB, **60 TPS** 💻

#### Gemma 4 (Google) - Original models
- ⚠️ `mlx-community/gemma-4-e2b-it-4bit` — 2B, ~2.7 GB RAM (requires MoE weight fixes)
- ⚠️ `mlx-community/gemma-4-e4b-it-4bit` — 4B, ~4.3 GB RAM (requires MoE weight fixes)
- ⚠️ `mlx-community/gemma-4-31b-it-4bit` — 31B, ~17 GB RAM (requires MoE weight fixes)

> **Note:** Gemma 4 models have MoE (Mixture of Experts) architecture that requires special weight handling. Qwen models are recommended for production use.

---

## Troubleshooting

### Command not found: `gemma`

**Problem:** Running `gemma` in terminal does nothing.

**Solution:** The binary is called `GemmaServer`, not `gemma`. You have three options:

```bash
# Option 1: Create an alias (add to ~/.zshrc or ~/.bashrc)
alias gemma="swift run GemmaServer"

# Option 2: Copy to system PATH
sudo cp .build/release/GemmaServer /usr/local/bin/gemma

# Option 3: Run with full path
.build/release/GemmaServer --help

# Option 4: Use swift run (no installation needed)
swift run GemmaServer --help
```

### Model not found

**Problem:** `Error: Model not found at path...`

**Solution:** Download the model first:

```bash
# Check where models are cached
ls -la ~/.cache/huggingface/hub/

# Download model
huggingface-cli download mlx-community/Qwen3.5-4B-4bit

# Or use git
git clone https://huggingface.co/mlx-community/Qwen3.5-4B-4bit \
  ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-4bit/snapshots/main
```

### Out of Memory errors

**Problem:** Server crashes with OOM (Out of Memory)

**Solution:** Use a smaller model or configure dynamic token budgeting:

```bash
# For 8GB RAM - use 4B model
gemma serve --model mlx-community/Qwen3.5-4B-4bit

# For 16GB RAM - use 9B model
gemma serve --model mlx-community/Qwen3.5-9B-OptiQ-4bit

# For 24GB+ RAM - use 27B model
gemma serve --model mlx-community/Qwen3.6-27B-4bit
```

**Token budgeting is automatic:**
- 8GB RAM → max ~65k tokens
- 16GB RAM → max ~128k tokens
- 32GB+ RAM → max 128k tokens (capped)

### Slow performance

**Problem:** Low TPS (tokens per second)

**Checklist:**
- ✅ Are you using the release build? `swift build -c release`
- ✅ Is Metal GPU acceleration enabled? (automatic on Apple Silicon)
- ✅ Is the model quantized (4bit)? Check model name ends with `-4bit`
- ✅ Do you have enough RAM? See model requirements above
- ✅ Close other heavy apps (Chrome, Docker, etc.)

**Benchmark your hardware:**
```bash
swift run PerformanceBenchmark --model mlx-community/Qwen3.5-4B-4bit
```

Expected TPS on M-series:
- M1/M2: 70-90 TPS (4B model)
- M3/M4: 90-110 TPS (4B model)

### Build failures

**Problem:** Swift build errors

**Solution:**
```bash
# Clean build
swift package clean
rm -rf .build

# Update dependencies
swift package update

# Rebuild
swift build -c release

# If still failing, check Swift version
swift --version  # Should be 6.0+
```

### Port already in use

**Problem:** `Error: Address already in use (port 8080)`

**Solution:**
```bash
# Find process using port 8080
lsof -i :8080

# Kill the process
kill -9 <PID>

# Or use a different port
gemma serve --model <model> --rest --port 8081
```

---

## MCP Integration (Cursor / Claude)

Add to your MCP config (`~/.cursor/mcp.json` or similar):

```json
{
  "mcpServers": {
    "gemma": {
      "command": "/usr/local/bin/gemma",
      "args": [
        "serve",
        "--model", "mlx-community/Qwen3.5-4B-4bit",
        "--mcp"
      ]
    }
  }
}
```

**Or with swift run:**
```json
{
  "mcpServers": {
    "gemma": {
      "command": "swift",
      "args": [
        "run",
        "--package-path", "/Users/yourname/projects/mlx",
        "GemmaServer",
        "serve",
        "--model", "mlx-community/Qwen3.5-4B-4bit",
        "--mcp"
      ]
    }
  }
}
```

**Available MCP Tools:**
- `gemma_status` - Get server health and model info
- `gemma_generate` - Generate text with streaming support
- `gemma_list_tools` - List available tools

---

## Development

### Running Tests

GemmaServer has comprehensive test coverage with TDD approach:

```bash
# Run all tests (101 tests, ~3 seconds)
swift test

# Run specific test suite
swift test --filter OrchestratorTests
swift test --filter RESTServerTests
swift test --filter AuthServiceTests

# Run with coverage
swift test --enable-code-coverage
```

**Current Test Stats:**
- ✅ **101 tests** passing
- ✅ **80% integration coverage** (Epic 2)
- ✅ **100% unit coverage** for core modules
- ✅ **3.2s** total test duration
- ✅ **Swift 6 strict concurrency** - no data races

**Test Suites:**
- Unit Tests: ModelOrchestrator, InferenceEngine, AuthService
- Integration Tests: REST API, MCP Server, Concurrent operations
- Actor Isolation: Deadlock prevention, reentrancy handling
- Database: SQLite session store with concurrent access

### Performance Benchmarking

```bash
# Run standard benchmark
swift run PerformanceBenchmark \
  --model mlx-community/Qwen3.5-4B-4bit \
  --iterations 10 \
  --tokens 100

# Context degradation profiling
swift run PerformanceBenchmark \
  --model mlx-community/Qwen3.5-4B-4bit \
  --profile-context \
  --output degradation.json
```

**Benchmark Metrics:**
- TPS (Tokens Per Second)
- TTFT (Time To First Token)
- Memory usage (peak, active, cache)
- Context degradation (1k → 128k tokens)
- Statistical analysis (avg, min, max, σ)

### Project Structure

```
Sources/
├── GemmaServer/              # Main CLI executable
│   ├── CLI/                  # Command-line interface
│   │   ├── ServeCommand.swift
│   │   ├── ChatCommand.swift
│   │   └── ModelsCommand.swift
│   ├── Core/                 # Business logic
│   │   ├── ModelOrchestratorActor.swift
│   │   ├── MLXInferenceEngine.swift
│   │   ├── AuthService.swift
│   │   └── GemmaServerError.swift
│   ├── REST/                 # REST API server
│   │   ├── RESTServer.swift
│   │   ├── Controllers/
│   │   └── Middleware/
│   ├── MCP/                  # MCP stdio server
│   │   └── MCPServer.swift
│   ├── Config/               # Configuration
│   │   └── ServerConfig.swift
│   └── Utils/                # Utilities
│       ├── ModelCache.swift
│       ├── HuggingFaceHub.swift
│       └── TokenBudgetCalculator.swift
├── PerformanceBenchmark/     # Benchmark tool
└── GemmaServerTests/         # Test suite (101 tests)
```

### Contributing

1. **TDD Workflow** - Write tests first
2. **Swift 6** - Use strict concurrency
3. **Actor Isolation** - All mutable state in actors
4. **Type Safety** - Leverage Swift's type system
5. **Test Coverage** - Maintain 100% for new code

See [TEST.md](TEST.md) for detailed testing roadmap.  
See [PLAN.md](PLAN.md) for product roadmap and epic breakdown.

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
