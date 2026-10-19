# Changelog

## 0.3.0 — 2026-10-19

### Security
- An API key is mandatory when serving off loopback (`BASALT_API_KEY`, `--api-key` or `--auto-generate-key`); the key check runs on every route except health and docs. `~/.gem/config.json` is written owner-only and merged, never overwritten (#28, #48).
- Token-bucket rate limit, 429 with `Retry-After`: on by default off loopback (120 req/min), `--rate-limit N` or `rate_limit` in the config, `0` disables (#19, #57).

### Coding agents
- OpenAI Responses API (`/v1/responses`) with the event sequence Codex expects, `reasoning.effort`, and tool calling end to end: tool definitions reach the model, tool calls stream back to the client (#27, #52).
- Malformed tool-call JSON from the model is repaired against the declared tools; a call that cannot be repaired is shown, not dropped (#26 part, #53).

### Models and hardware
- `models inspect` verifies safetensors shards and measures quantization bits from the weights; `models check` exits non-zero when a model does not fit (#29, #49).
- `fit` scoring: one decode-speed model with per-chip bandwidth, MoE accounting, the best quantization per model, and benchmark profiles that survive a restart (#31, #51).
- `--stats` and `GET /api/v1/stats`: tok/s, TTFT, RAM and GPU memory, streaming requests included; MCP `status` reports the same snapshot (#30, #50).

### Architecture
- The single `Gem` target is split into `BasaltCore`, `BasaltInference`, `BasaltCloud`, `BasaltServer`, `BasaltUI`, `BasaltApp` and `BasaltCLI`; SwiftUI is no longer linked into the `basalt` binary. The command name is unchanged (#20, #60).
- Tests, scripts and assets consolidated: one cleanup script (`scripts/cleanup.swift --mode=…`, dry run unless `--apply`), one audit script with the OSV scan built in, REST controllers covered on a mock engine (#21, #61).

### CI and release
- Each PR is tested once, incrementally, on the self-hosted runner; release checks the tag against the source version and opens the Homebrew formula bump as a PR; the formula is audited with `brew audit --strict` (#54, #58).

### Removed
- Files that only held a `// removed` comment, the login screen of the SwiftUI client (the server has no login route), the WebSocket section of the README (no such route).

### Known limits
- Prefix KV-cache reuse across turns is not implemented yet (#56); the GUI client has no launch command (#55).
