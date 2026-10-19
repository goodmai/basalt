# Changelog

## Unreleased

### Security
- REST on loopback no longer trusts browsers (#72). DNS rebinding (an attacker's domain re-resolved to 127.0.0.1, the class of Ollama CVE-2024-28224) is stopped by a `Host` check: on loopback only `localhost`, `127.0.0.1`, `[::1]` (any port), else 403; a specific non-loopback bind also accepts its own address, a `0.0.0.0`/`::` bind accepts any `Host` because the API key is mandatory there. A request carrying an `Origin` other than the server's own loopback origin (`localhost`, `127.0.0.1`, `[::1]` on the bound port; no port means 80/443) gets 403 unless allow-listed, so a page on another local port (a dev server, Jupyter) cannot drive the API either; allow-list it with `--allow-origin` (repeatable) or `allowed_origins` in `~/.gem/config.json`; requests without `Origin` (curl, SDKs, Claude Code, Codex) are unaffected. Every `POST` needs `Content-Type: application/json`, else 415, which removes the no-preflight CORS "simple request". These checks run before the API key and the rate limit.
- `POST /v1/models/load` accepts only Hugging Face repo ids (`org/name`); local paths, `..` and `claude-local/..--x` are 400. In the `model` field of `/v1/messages`, `/v1/chat/completions` or `/v1/responses` a path form (starting with `/`, `.` or `~`, or containing `..`, after trimming whitespace) is 400 too; any other value that is not a repo id (`basalt`, `meta-llama/llama-3.1-8b-instruct:free`, `org/name@rev`) is an alias served by the loaded model, with no switch. Local paths stay available to `basalt serve --model-path` and MCP `switch_model` (#72).
- A model directory without a readable `config.json` is refused before the serving model is unloaded, so a bad load no longer leaves the server without a model (#72).
- REST error bodies no longer contain filesystem paths or echo the requested model id of a failed load (the endpoint was a file-existence oracle); the full error is in the server log (#72).

- ARC-AGI benchmark `--mode hybrid` no longer runs model-written Python with `exec()` inside the benchmark process (#78). Each attempt's code runs in a child `python3 -I -S` (grid in on stdin, result out as JSON on stdout) in an empty temp directory with an empty environment, under a `sandbox-exec` profile on macOS: deny by default, exec only of the Python binary, reads only of the Python install, `numpy` and its libraries, `/usr/lib`, `/System` and the two benchmark module files (their directory cannot be listed), writes only to the temp directory, `sysctl` reads only of the five names `os.uname()` needs, no network. `_ctypes` and the `_test*` extension modules cannot be loaded: Seatbelt does not mediate `sysctl` `KERN_PROCARGS2`, which returns the argv and environment of any same-user process (the benchmark parent included), so the child must not reach raw syscalls; a CPython or `numpy` memory-corruption exploit remains a residual path. A wall-clock timeout (`--exec-timeout`, 10 s), `RLIMIT_CPU`, `RLIMIT_NPROC=0`, `RLIMIT_NOFILE`, `RLIMIT_FSIZE` and a 1 GiB memory watchdog stop runaway code (the parent samples the child's `ri_phys_footprint` via `proc_pid_rusage` every 10 ms; a soft cap: a 4 GiB `memset` on a fully loaded machine peaked at 1.08-1.23 GiB); a 64 MiB cap on the temp directory (every entry counts as at least 4 KiB, so empty files count too) stops code that writes many files; the attempt is recorded as `exec_timeout` / `exec_memory_limit` / `exec_disk_limit` and the run continues. macOS refuses `RLIMIT_AS`/`RLIMIT_DATA`, hence the watchdog. Without `sandbox-exec` (Linux) the default mode is now `direct`, and an explicit `--mode hybrid` warns that the code can still read files and reach the network. The DSL classes moved to `benchmarks/arc-agi/arc_helpers.py`; `--engine-url` points the benchmark at a server on any port. The default results file name now ends with the mode (`arc_benchmark_results_<engine>_<split>_<mode>.json`), so a hybrid and a direct run no longer overwrite or `--resume` each other.

### Coding agents
- Tool calling on `/v1/messages` (Claude Code): `tools` (`input_schema`) reach the chat template; `tool_use` / `tool_result` history blocks (string or block `content`, `is_error`) are rendered into the prompt like the Responses API's `function_call` / `function_call_output`; calls come back as `tool_use` blocks with `toolu_…` ids and `stop_reason: "tool_use"`, streamed as `content_block_start {tool_use}`, `input_json_delta`, `content_block_stop` after any thinking or text block (#73).
- Tool calling on `/v1/chat/completions`: `tools` in, assistant `tool_calls` and role `tool` messages rendered into the prompt, `message.tool_calls` / streamed `delta.tool_calls` with `finish_reason: "tool_calls"`. The stream's last chunk now reports `length` when generation hit the cap instead of always `stop` (#73).
- `tool_choice` on all three endpoints: `none` keeps the tools in the prompt (same prefix as the next turn) and rejects every call with `[tool call to X rejected: tool_choice is none]`; a named tool keeps every tool in the prompt and only that one is let through (a name not in `tools` is 400); `any` / `required` are not enforced and behave as `auto` (#42); `parallel_tool_calls: false` / `disable_parallel_tool_use: true` end the generation after the first call (#73).
- `/v1/messages`: tool arguments that are not a JSON object go out as `input: {}` (and `partial_json: "{}"`) with the original text in a visible note (#73).
- A tool call the engine parses for a tool that was not offered (or not the forced one) is no longer passed to the client; it becomes the visible `[tool call to X rejected: …]` note, like an unrepairable call (#73, #53).
- Tool results in the history render as `[Tool result for <name>]`, matched by `tool_use_id` / `tool_call_id` / `call_id`, so parallel results are not attributed by order; images in tool results render as `[image omitted]`. Tool schemas are serialized with sorted keys for a stable prompt prefix (#73).

### Sampling
- Per-request `min_p` (`minP` on `/api/v1/generate` and MCP) overrides `--min-p` and the model profile default; `0` turns Min-P off for that request. Validated to 0–1 (#40).
- `dynamic_temperature` (`dynamicTemperature`): entropy-based dynamic temperature (EDT, arXiv:2403.14541), `T = T0·N^(θ/H) + γ·R` with N = 0.8, θ = 1.5, γ = 0.5, clamped to 0.05–2 (a request colder than 0.05 keeps its own temperature as the floor). Confident tokens sample near-greedy; a forming repetition loop raises the temperature. With EDT on, `top_p`/`min_p` still cut the original distribution, not the heated one. Off by default; no effect at temperature 0 (#40).
- `RepetitionGuard` exposes a continuous repetition score in [0, 1] alongside its stop decision (#40).

### Upgrading
- `curl -d '…'` without `-H "Content-Type: application/json"` now gets 415; a browser app on any other origin, a UI on another local port included, needs `--allow-origin` (#72).
- `POST /v1/models/load` accepts only a Hugging Face repo id `org/name` (or its `claude-local/org--name` form), characters `A-Z a-z 0-9 - _ .`; anything else, local paths included, is 400 (#72).
- Anthropic `/v1/messages` usage: `input_tokens` is now only the uncached remainder, as in Anthropic's API; the reused part is in `cache_read_input_tokens`. A client that took `input_tokens` as the whole prompt size must add the two (#56).

### Prompt cache
- Prefix KV-cache reuse across conversations: a radix tree keeps the KV cache of recent prompts, one node per shared token run. A request takes the longest stored prefix and prefills only the rest, so the next turn of a conversation, and a second conversation with the same system prompt and tools (a Claude Code subagent, another Codex thread), reuse what is already computed instead of prefilling it again; concurrent requests on one prefix both reuse it. All clients share one tree; `prompt_cache_key`, `metadata.user_id` and `x-session-id` are not needed and are ignored; service requests are never cached. Sliding-window layers such as Gemma 3/4 and `--kv-bits` prefill in full (#56, #32).
- Prefix reuse for hybrid attention/SSM models (Qwen3.5/3.6, Ornith-1.5): the prompt is prefilled in two phases and the recurrent state is snapshotted at the end of the request text, before the chat template's closing tokens; the next turn that extends the text restores the snapshot and prefills only the tail, with `cached_tokens` reporting the restored length. Snapshots count against `--prompt-cache-mb`. Ornith-1.5-9B, 12.6k-token agent prompt: second turn ≈1 s instead of ≈36 s. A reused turn can differ from an uncached one after its first tokens (bf16 drift from a different prefill chunking: restored state ≡ two-phase prefill from scratch bitwise, vs one-phase max |Δlogit| 1.56 with the same top 5). Under the byte budget a snapshot branch is evicted whole, and a growing conversation keeps only its two newest snapshots (the newest to continue from, its parent so an edited or regenerated last message still restores); older ones on a single-child chain are dropped and the chain is merged. A conversation sharing only the system prompt with another still prefills in full (#79).
- `--prompt-cache-mb` (default: half of what Metal's recommended GPU working set leaves after the weights; `0` disables) caps the memory of the tree, counted on the stored prompt tokens only, each shared prefix once. Over budget the least recently used branch is cut from its end, so an idle conversation keeps its head (system prompt and tools) longest; a branch in use by a running request is never evicted; a prompt larger than the budget keeps the longest head that fits (#56, #32).
- The tree also holds at most 4096 branches (leaves) in all; beyond that the least recently used free branch is dropped, a branch in use never (#32).
- Usage reports the tokens actually reused: `prompt_tokens_details.cached_tokens`, `input_tokens_details.cached_tokens`, `cache_read_input_tokens` (see Upgrading for Anthropic `input_tokens`). `cached_tokens` reveals how much of a prompt matched an earlier prompt from any client; see the README before sharing a server (#56, #32).
- `--cache-blend` (opt-in, approximate): CacheBlend non-prefix KV reuse (arXiv:2405.16444). Prompt chunks (split after blank lines) keep their pre-RoPE KV and are reused at any position; the first layer is recomputed for all tokens and only the `--cache-blend-ratio` (default 0.5) most deviating reused tokens on later layers. Six reordered files of 550–900 tokens on Qwen3-4B-4bit, teacher-forced top-1 agreement with a full prefill on 200 positions / time to first token: full prefill 0.995 / 10.9 s, ratio 0.5 0.950 / 6.7 s, ratio 0.15 0.930 / 3.7 s, ratio 0 0.925 / 1.1 s — the paper's 0.15 barely beats no recomputation on this sample, hence 0.5. Chunks end at a blank line in one token or across two newline tokens (SentencePiece); chunks under 16 tokens are recomputed; a prompt is blended only when (1 − ratio) × reused share ≥ 0.2 (the blended forward costs ~1.2× native per recomputed token) and the cache stays within 16 384 tokens, otherwise it takes the native chunked prefill and stores its chunks from that cache; ratio 1 is always native. Llama and Qwen3 dense models only; the prompt-cache budget is split in half with the tree; `cached_tokens` counts only exact prefix-tree tokens, blended ones are logged (#36).
- Removed the unused string-based `MultiSlotConversationCache`, `PromptContextBuilder.extractToolPrefix` / `compactWithToolPrefixProtection` and the orchestrator's working-session tracker; the engine's token-level prefix tree replaces them (#56, #32).

### Fixed
- KV-cache memory is estimated from the model's geometry instead of 2 bytes per token (token budget) and 8 bytes per token (memory check): `2 · layers · kv_heads · head_dim · bytes` per token from `config.json` (`text_config` for VLM wrappers), only full-attention layers growing for hybrid models (Ornith-1.5/Qwen3.5: 8 of 32) plus their fixed SSM state, sliding-window layers capped at the window, Gemma 4 full-attention layers with `global_head_dim` (and `num_global_key_value_heads` when `attention_k_eq_v`) and without the `num_kv_shared_layers` that hold no cache, MLA (DeepSeek V2/V3) as the per-head K and V that MLX caches, Jamba and NemotronH attention layers from `attn_layer_period` / `hybrid_override_pattern`, `--kv-bits` as `bits + 0.5` bits per element (`bits + 1` when `head_dim` is not a multiple of 64 and MLX quantizes in groups of 32); `--kv-bits` does not apply to sliding-window layers. The memory-derived token budget is now computed correctly and subtracts the prompt (Qwen3-4B on 16 GB free: 77.5k instead of 128k); cap = max(--max-tokens, memory budget); the budget can only raise it above --max-tokens (see #85); `serve --dry-run` (which takes `--kv-bits`), `models check` and `fit` (its quantization grid and decode-speed estimate; the RAM column and fit level still count weights only) include the KV cache for every model: Qwen3-4B at 32k tokens needs 4.5 GiB of KV, Ornith-1.5-9B 1.05 GiB. Without a cached config the parameter-count estimate of `fit` is the fallback (#75).
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
