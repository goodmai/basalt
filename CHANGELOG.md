# Changelog

## Unreleased

### Sampling
- Per-request `min_p` (`minP` on `/api/v1/generate` and MCP) overrides `--min-p` and the model profile default; `0` turns Min-P off for that request. Validated to 0–1 (#40).
- `dynamic_temperature` (`dynamicTemperature`): entropy-based dynamic temperature (EDT, arXiv:2403.14541), `T = T0·N^(θ/H) + γ·R` with N = 0.8, θ = 1.5, γ = 0.5, clamped to 0.05–2 (a request colder than 0.05 keeps its own temperature as the floor). Confident tokens sample near-greedy; a forming repetition loop raises the temperature. With EDT on, `top_p`/`min_p` still cut the original distribution, not the heated one. Off by default; no effect at temperature 0 (#40).
- `RepetitionGuard` exposes a continuous repetition score in [0, 1] alongside its stop decision (#40).

### Upgrading
- Anthropic `/v1/messages` usage: `input_tokens` is now only the uncached remainder, as in Anthropic's API; the reused part is in `cache_read_input_tokens`. A client that took `input_tokens` as the whole prompt size must add the two (#56).

### Prompt cache
- Prefix KV-cache reuse across conversations: a radix tree keeps the KV cache of recent prompts, one node per shared token run. A request takes the longest stored prefix and prefills only the rest, so the next turn of a conversation, and a second conversation with the same system prompt and tools (a Claude Code subagent, another Codex thread), reuse what is already computed instead of prefilling it again; concurrent requests on one prefix both reuse it. All clients share one tree; `prompt_cache_key`, `metadata.user_id` and `x-session-id` are not needed and are ignored; service requests are never cached. Sliding-window layers such as Gemma 3/4 and `--kv-bits` prefill in full (#56, #32).
- Prefix reuse for hybrid attention/SSM models (Qwen3.5/3.6, Ornith-1.5): the prompt is prefilled in two phases and the recurrent state is snapshotted at the end of the request text, before the chat template's closing tokens; the next turn that extends the text restores the snapshot and prefills only the tail, with `cached_tokens` reporting the restored length. Snapshots count against `--prompt-cache-mb`. Ornith-1.5-9B, 12.6k-token agent prompt: second turn ≈1 s instead of ≈36 s. A conversation sharing only the system prompt with another still prefills in full (#79).
- `--prompt-cache-mb` (default: half of what Metal's recommended GPU working set leaves after the weights; `0` disables) caps the memory of the tree, counted on the stored prompt tokens only, each shared prefix once. Over budget the least recently used branch is cut from its end, so an idle conversation keeps its head (system prompt and tools) longest; a branch in use by a running request is never evicted; a prompt larger than the budget keeps the longest head that fits (#56, #32).
- The tree also holds at most 4096 branches (leaves) in all; beyond that the least recently used free branch is dropped, a branch in use never (#32).
- Usage reports the tokens actually reused: `prompt_tokens_details.cached_tokens`, `input_tokens_details.cached_tokens`, `cache_read_input_tokens` (see Upgrading for Anthropic `input_tokens`). `cached_tokens` reveals how much of a prompt matched an earlier prompt from any client; see the README before sharing a server (#56, #32).
- Removed the unused string-based `MultiSlotConversationCache`, `PromptContextBuilder.extractToolPrefix` / `compactWithToolPrefixProtection` and the orchestrator's working-session tracker; the engine's token-level prefix tree replaces them (#56, #32).

### Fixed
- Non-streaming `/v1/chat/completions` and `/v1/responses` answered 500 to an invalid request (e.g. `temperature` or `min_p` out of range); they now return the error's own status, 400 for a bad field.

## 0.3.0 — 2026-10-19

### Upgrading from 0.2.0
- `basalt serve --host` off loopback now refuses to start without a key: set `BASALT_API_KEY`, pass `--api-key`, or `--auto-generate-key`. Clients send it as `Authorization: Bearer` or `x-api-key`.
- Off loopback the server answers 429 beyond 120 requests per minute; raise it with `--rate-limit N`, turn it off with `--rate-limit 0`.
- Building from source (Homebrew included) needs Xcode 26.4+ (Swift 6.3): mlx-swift now declares swift-tools-version 6.3.
- The config stays in `~/.gem/`. The command is still `basalt`.
- Earlier versions left the last request body in `/tmp/basalt_anthropic_request.json` and `/tmp/basalt_openai_request.json`: delete them.

### Security
- An API key is mandatory when serving off loopback (`BASALT_API_KEY`, `--api-key` or `--auto-generate-key`); the key check runs on every route except health and docs. `~/.gem/config.json` is written owner-only and merged, never overwritten (#28, #48).
- Token-bucket rate limit, 429 with `Retry-After`: on by default off loopback (120 req/min), `--rate-limit N` or `rate_limit` in the config, `0` disables (#19, #57).
- `/v1/messages` and `/v1/chat/completions` no longer write every request body (prompts, code, anything pasted) to world-readable files in `/tmp` (`basalt_anthropic_request.json`, `basalt_openai_request.json`). Delete those files if they exist.

### Coding agents
- OpenAI Responses API (`/v1/responses`) with the event sequence Codex expects, `reasoning.effort`, and tool calling end to end: tool definitions reach the model, tool calls stream back to the client (#27, #52).
- Malformed tool-call JSON from the model is repaired against the declared tools; a call that cannot be repaired is shown, not dropped (#26 part, #53).
- Fixed: with tools in the request, the answer was replaced by its first `{…}` fragment, which mangled every Claude Code reply containing code (#53).
- Usage no longer reports `cached_tokens` / `cache_read_input_tokens`: they were an estimate for a cache that was never used (#53).
- `/v1/messages` accepts `thinking` / `reasoning_effort` (#52).

### Models and hardware
- `models inspect` verifies safetensors shards and measures quantization bits from the weights; `models check` exits non-zero when a model does not fit (#29, #49).
- `fit` scoring: one decode-speed model with per-chip bandwidth, MoE accounting, the best quantization per model, and benchmark profiles that survive a restart (#31, #51).
- `--stats` (`--stats-interval`) and `GET /api/v1/stats`: tok/s, TTFT, RAM and GPU memory, streaming requests included; MCP `status` reports the same snapshot (#30, #50).
- New: `models check --system [--json]`, `PerformanceBenchmark --share`; MCP tools `status`, `check_hardware`, `inspect_model`, `switch_model` (#49, #50).

### Architecture
- The single `Gem` target is split into `BasaltCore`, `BasaltInference`, `BasaltCloud`, `BasaltServer`, `BasaltUI`, `BasaltApp` and `BasaltCLI`; SwiftUI is no longer linked into the `basalt` binary. The command name is unchanged (#20, #60).
- Tests, scripts and assets consolidated: one cleanup script (`scripts/cleanup.swift --mode=…`, dry run unless `--apply`), one audit script with the OSV scan built in, REST controllers covered on a mock engine (#21, #61).

### CI and release
- Each PR is tested once, incrementally, on the self-hosted runner; release checks the tag against the source version and opens the Homebrew formula bump as a PR; the formula is audited with `brew audit --strict` (#54, #58).
- Dependencies: hummingbird 2.27 (capped below 2.28), swift-transformers 1.3.5, Rainbow 4.2.2, swift-markdown 0.9.0.

### Removed
- Files that only held a `// removed` comment (#57).
- The WebSocket section of the README: the route was already gone before 0.2.0.

### Changed
- The SwiftUI client's login form now takes the API key (sent as Bearer) instead of calling a `/auth/login` route the server never had (#57).

### Known limits
- Prefix KV-cache reuse across turns is not implemented yet (#56); the GUI client has no launch command (#55).
- Some names still say Gem: the MCP `serverInfo.name`, the logger label, the `gem_` prefix of generated API keys, `~/.gem/` (#64).
