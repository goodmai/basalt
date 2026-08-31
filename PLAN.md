# Gemm — Development Plan

**Product:** Local LLM inference server for Apple Silicon, designed as a drop-in backend for agentic workflows (Claude Code, MCP, REST).  
**Current version:** v0.7.0  
**Status:** Active development

---

## Completed

- [x] MLX inference engine with Gemma 4 and Qwen 3.x support (Apple Silicon Metal GPU)
- [x] Swift 6 strict concurrency — `ModelOrchestratorActor` with FIFO actor serialisation
- [x] 5-minute generation timeout with cooperative Task cancellation
- [x] MCP stdio server (JSON-RPC 2.0) — `gemma_generate`, `gemma_status`, `playwright_screenshot`, `gemma_add_knowledge`
- [x] REST server (Hummingbird 2.x) — all endpoints public, no auth required
- [x] OpenAI-compatible `/v1/chat/completions` with SSE streaming
- [x] Anthropic-compatible `/v1/messages` with full SSE event sequence
- [x] WebSocket `/ws/generate` streaming
- [x] Think-block stripping (`<think>...</think>`) in stream output
- [x] Smart Metal shader build script — skips if `default.metallib` is newer than sources
- [x] Dynamic token budget calculation based on available RAM
- [x] Rainbow terminal UI (Markdown, spinner, progress bar, diff renderer, table)
- [x] Benchmark suite (`PerformanceBenchmark` target) with SQLite result store
- [x] **Model hot-swap** — `POST /v1/models/load` switches model at runtime; actor serialises with in-flight requests
- [x] **Proper model unload** — `MLX.GPU.clearCache()` called before loading new model; prevents dual-model peak RAM spike
- [x] **Model listing** — `GET /v1/models` scans HF cache, returns size + load status; `GET /v1/models/current`
- [x] **Claude Code model discovery** — `/v1/models` returns `claude-local/<id>` prefix so models appear in `/model` picker
- [x] **ID translation** — `ModelsController.hfRepoId(from:)` / `claudeLocalId(from:)` handle both ID forms end-to-end
- [x] **`claude-local/` prefix stripping in controllers** — `AnthropicController` and `OpenAIController` strip prefix before passing to orchestrator
- [x] **`./Gemma` launcher script** — builds server, waits for readiness, launches Claude Code with local env vars
- [x] Auth/JWT removed — server is fully public (local-only by design)
- [x] Dead test files stubbed out (`AuthServiceTests`, `RESTServerTests`, `MetalE2ETests`)

---

## Near-term (v0.7.1)

- [ ] **Context window management** — auto-truncate conversation history when approaching context limit to prevent silent truncation errors
- [ ] **Tool use / function calling** — pass `tools` array through in Anthropic and OpenAI formats; return `tool_use` content blocks
- [ ] **`gemm models download`** — download a model from HuggingFace directly from the CLI with progress bar
- [ ] **Delete `// removed` stubs** — `AuthService.swift`, `AuthController.swift`, `JWTAuthenticator.swift`, `RateLimitMiddleware.swift`, `UI/SwiftUI/*.swift` after git history is cleaned; remove empty `Middleware/` directory
- [ ] **`UICommand` cleanup** — either implement the web UI command or remove it from the CLI entirely

---

## Medium-term (v0.8.0)

- [ ] **Structured output** — JSON schema enforcement via constrained decoding
- [ ] **Embeddings endpoint** — `POST /v1/embeddings` for RAG use cases
- [ ] **Session logging** — optional request/response logging with configurable rotation
- [ ] **Multi-model routing** — keep multiple models warm and route by request header or URL prefix
- [ ] **Homebrew tap** — `brew install gemm`

---

## Architecture Decisions

**No authentication** — local server for personal development. Auth adds attack surface with no benefit on loopback.

**Single actor, two transports** — `ModelOrchestratorActor` is shared between MCP and REST. Swift actor guarantees FIFO without explicit locking.

**Model unload before load** — when hot-swapping, `container = nil` + `MLX.GPU.clearCache()` frees the old model before the new one allocates GPU memory. Without this, both models exist simultaneously (peak = A + B RAM) which causes OOM on constrained hardware.

**Hummingbird 2.x not Vapor** — lighter, Swift 6 native, better `AsyncStream` support.

**MLX not llama.cpp** — Apple Silicon Metal GPU path is significantly faster than CPU paths on macOS.

**Prompt templating** — a flat `[System]/[User]/[Assistant]` string is sent to `UserInput(chat:)`, letting the MLX chat template apply model-specific formatting (Gemma instruction tuning, Qwen ChatML, etc.). Stop tokens are model-defined by the template. Investigate passing structured message arrays directly to `MLXLLM` to avoid double-wrapping.

**ID strategy** — HuggingFace repo IDs (`org/name`) are the canonical form inside the server. The `claude-local/org--name` form exists only at the `/v1/models` API boundary for Claude Code compatibility. All controllers call `ModelsController.hfRepoId(from:)` to normalise before hitting the orchestrator.

---

## Known Issues

- Gemma 4 26B MoE (`gemma-4-26b-a4b-it-4bit`) loads in dense-fallback mode — MoE expert routing weights are currently bypassed for compatibility with the MLX-LM loader
- Think-block stripping uses a rolling buffer which can split tags across chunk boundaries; a proper state machine with unbounded accumulation would be more robust
- `UICommand` is a no-op stub left from an earlier SwiftUI phase; will be removed or repurposed

---

## Infrastructure

| Directory | Purpose |
|---|---|
| `logs/` | Runtime logs — gitignored, rotated by `cleanup_daily.swift` |
| `reports/` | Benchmark JSON reports — gitignored |
| `screenshots/` | Test captures — gitignored |
| `scripts/` | Build and maintenance Swift scripts |
| `docs/` | Extended API and architecture documentation |
