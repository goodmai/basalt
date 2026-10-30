<p align="center"><img src="images/logo.svg" alt="basalt logo" width="128"></p>

# basalt — turn your Mac into a local LLM server. Point Claude Code at it.

> Gemma 4, Qwen 3 and any MLX model on your Apple Silicon GPU, behind the **OpenAI API, the Anthropic API and MCP**. One Swift binary. No Python at runtime. No cloud. No API bill.

[English](README.md) · [Русский](README.ru.md)

[![Release](https://github.com/goodmai/basalt/actions/workflows/release.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/release.yml)
[![Security Audit](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

[![Platform](https://img.shields.io/badge/platform-macOS%2015%2B%20·%20Apple%20Silicon-black.svg)](#requirements)
[![Swift 6](https://img.shields.io/badge/Swift-6-black.svg)](Package.swift)
[![Homebrew](https://img.shields.io/badge/brew-goodmai%2Fbasalt%2Fbasalt-black.svg)](#install-with-homebrew)
[![Ladder](https://img.shields.io/badge/ladder-6%20tasks-black.svg)](#benchmarks)
[![ARC-AGI](https://img.shields.io/badge/ARC--AGI-pass%402-black.svg)](#benchmarks)

```
┌──────────────────────────────────────────────────────┐
│                       Basalt                           │
│                                                      │
│   MCP stdio ──┐                                      │
│               ├──► ModelOrchestratorActor ──► MLX   │
│   REST :8080 ─┘        (actor, FIFO)      Metal GPU  │
└──────────────────────────────────────────────────────┘
```

Two transports share a single actor instance — **MCP stdio** for IDE integration (Claude Desktop, Cursor) and **REST HTTP** for agent-to-agent workflows and Claude Code backends.


<p align="center"><img src="images/models-hive.svg" alt="Open models basalt runs: Qwen, Gemma, Ornith, gpt-oss, DeepSeek, Llama" width="240"></p>
<p align="center"><sub>Runs open-weight models on Apple Silicon. Verified here today: Gemma, Qwen, Ornith (see <a href="#verified-models">the table</a>); gpt-oss, DeepSeek and Llama are MLX-compatible but not benchmarked here yet. Logos are trademarks of their owners, shown only to indicate compatibility. Anthropic's models are closed-weight, so they are not in the hive; the Anthropic <i>API</i> is supported so that Claude Code can talk to a local model.</sub></p>

## Why basalt

- **One server, two transports.** REST speaks OpenAI *and* Anthropic (`/v1/messages`), MCP stdio plugs into Claude Desktop and Cursor, and every generation route can stream over SSE. They share a single model actor, so nothing loads twice.
- **Claude Code on a local model in two lines.** Set `ANTHROPIC_BASE_URL` and run `claude` — your real API key is never touched. See [Claude Code Integration](#claude-code-integration).
- **Fast on a laptop.** On a 24 GB Mac: ~110 tok/s on Gemma 4 E2B, ~92 tok/s on Qwen3.5-4B, ~37 tok/s on a 9B, ~12 tok/s on a 27B ([full table](#verified-models)).
- **Pure Swift.** SwiftPM only — no Node, no Python, no Docker at runtime. Metal kernels compiled once, `mlx.metallib` shipped next to the binary.
- **Honest numbers.** The model table lists what *broke* too (gibberish output, OOM). The benchmark ladder executes the model's own code against inputs the prompt never showed, so a plausible-looking wrong answer still fails ([Benchmarks](#benchmarks)).

## 60-second demo

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew install goodmai/basalt/basalt          # builds from source, needs Xcode 26.4+

basalt onboard                              # picks a model that fits your Mac and downloads it
basalt serve --model mlx-community/Qwen3.5-4B-4bit --rest

# in another terminal: Claude Code, backed by your own GPU
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude
```

Local 4B–9B models are not a drop-in replacement for a frontier model — they are for private, offline, zero-cost agentic work and for tinkering. `basalt fit` tells you what your machine can actually run.

**Star it** if you want more MLX-native tooling in Swift, and open an issue with your Mac and your tok/s — [PRs adding a row to the benchmark table](#benchmarks) are welcome.

---

## Requirements

| | |
|---|---|
| **macOS** | 15+ (Sequoia) |
| **Xcode / Swift** | 26.4+ / Swift 6.3 |
| **Metal toolchain** | `xcodebuild -downloadComponent MetalToolchain` — Xcode 26 ships the Metal compiler separately, and MLX's kernels cannot be built without it |
| **Hardware** | Apple Silicon M1–M4, Unified Memory |
| **Disk** | 2–30 GB depending on model |

---

## Quick Start

### Install with Homebrew

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew install goodmai/basalt/basalt
```

The tap needs the explicit URL because the repository is not named
`homebrew-basalt`. There is no bottle: the formula builds from source, so Xcode
26.4+ must be installed, along with the Metal toolchain — on Xcode 26 that is a
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
   directory, so without this step `basalt` only runs from the directory holding
   the library. Expect this step to dominate the install time.

```bash
brew install --HEAD goodmai/basalt/basalt   # build main instead of the last tag
brew upgrade goodmai/basalt/basalt
brew uninstall basalt && brew untap goodmai/basalt
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
.build/release/basalt chat --model mlx-community/Qwen3.5-4B-4bit

# REST server on :8080 (OpenAI + Anthropic compatible)
.build/release/basalt serve --model mlx-community/Qwen3.5-4B-4bit --rest

# MCP stdio server (for Claude Desktop / Cursor)
.build/release/basalt serve --model mlx-community/gemma-4-e4b-it-4bit --mcp
```

### One-command launcher: `./basalt-claude`

The repo includes a self-contained Swift launcher that builds the server (if needed), waits for readiness, and opens Claude Code — all with local env vars scoped to that session:

```bash
chmod +x ./basalt-claude

./basalt-claude                                             # Qwen 4B, port 8080
./basalt-claude --model mlx-community/gemma-4-31b-it-4bit  # Gemma 4 31B
./basalt-claude --port 8081                                 # custom port
./basalt-claude -- --model haiku                            # pass --model haiku to claude
```

`basalt-claude` pins `ANTHROPIC_API_KEY=local` rather than inheriting it, so a real key is never forwarded to the local server or written into the `--settings` payload. Other terminals are untouched.

---

## Commands

`basalt` with no subcommand starts `chat`.

| Command | What it does |
|---|---|
| `basalt onboard` | first-run wizard: picks a model for your machine and downloads it |
| `basalt fit` | reads the hardware and ranks the model catalogue against it |
| `basalt chat --model <id>` | interactive terminal chat |
| `basalt serve --model <id> --rest` | REST server on :8080 (OpenAI + Anthropic compatible) |
| `basalt serve --model <id> --mcp` | MCP stdio server for Claude Desktop / Cursor |
| `basalt serve --model <id> --stats` | print telemetry to stderr every 5 s (`--stats-interval`); the same numbers at `GET /api/v1/stats` and in MCP `status` |
| `basalt models list --author <org>` | browse a HuggingFace author's models |
| `basalt models download <repo-id>` | download into the shared HF cache |
| `basalt models info <repo-id>` | size, quantization, context window |
| `basalt models cache` | what is on disk, and how much of it |
| `basalt models check <repo-id>` | will it fit in memory (non-zero exit when it does not) |
| `basalt models inspect <repo-id>` | verify a cached model is complete and measure its real bit width |
| `basalt cloud configure` | OpenRouter fallback for what does not fit locally |
| `basalt cloud cost` | spend so far on the cloud fallback |

Inside `basalt chat`:

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
| `--kv-bits 4\|8\|auto` | quantize the KV cache — buys generation room on a memory-tight machine; turns off prompt-cache reuse. MLX prefills the prompt in fp16 and quantizes it afterwards, so the prompt itself is no longer than in fp16 (shorter: the conversion briefly holds both copies). `auto` stays fp16, with prompt-cache reuse, until a request's prompt plus `max_tokens` does not fit, then runs that request at 8, else 4 bits |
| `--kv-bits k8v4` | with `--kv-zones`: the quantized middle stores keys at 8 bits and values at 4 (13 bits per K+V pair with group scales, 0.41× fp16, between 8-bit's 0.53× and 4-bit's 0.28×). `auto` with `--kv-zones` tries it between 8 and 4 bits. On a model the zones do not apply to (Gemma 4) it runs as uniform 8-bit. **Experimental:** `k8v4h` also applies a Walsh–Hadamard transform to the values before 4-bit quantization (and its inverse on read), to spread outlier channels; it is opt-in only, `auto` never picks it. Measured with 8k tokens scored after an 8k context, perplexity against `k8v4`: Qwen3-4B 6.568 vs 6.562, Ornith-1.5-9B 4.937 vs 4.920, LFM2.5-8B 8.723 vs 8.750 (one run each, within noise), and peak memory after a 16k prompt rose by about 85 MB on Ornith and 30–150 MB on Qwen3-4B (the fp32 copy of the values it rotates). It rotates head dimensions 2^k and 12·2^k, where MLX's transform is its own inverse; other head dimensions stay unrotated, and `serve` logs that at load. It may matter on longer contexts (32k+) or on models with outlier channels in V; that is untested |
| `--kv-zones` | with `--kv-bits`: keep the first 64 and the last 1024 tokens of each full-attention layer in fp16 and quantize only the tokens between them, as the prompt is prefilled. No fp16 prefill to convert, so the prompt may use the whole quantized room and peak memory drops; on Qwen3-4B, 8k tokens scored after an 8k context, 4-bit perplexity is 6.58 against fp16 6.56 and uniform 4-bit 8.66 |
| `--kv-evict F` | **Experimental**, with `--kv-zones`: drop the fraction `F` (0–0.9) of the quantized middle of each full-attention layer as tokens leave the recent window, per KV head the ones whose keys have the largest L2 norm (arXiv:2406.11430). Needs no attention weights, so attention stays on the fused kernels; sinks and the recent window are never dropped, and the room `serve` and `models check` report counts only the kept tokens. After a 16k prompt with k8v4, KV on Qwen3-4B drops from 1.11 GB to 0.89 GB at 0.25 and 0.66 GB at 0.5. It costs quality: perplexity (8k scored after 8k) 6.71 / 7.04 against 6.56 on Qwen3-4B, 9.06 / 9.52 against 8.75 on LFM2.5-8B, unchanged on Ornith-1.5-9B; a 16k passkey was lost at one of five depths on Qwen3-4B and Ornith at 0.25, and greedy output diverges from fp16 within the first tokens. Gemma 4, and a model that prefills more than the recent window in one pass (checked at load), keep every token and are sized that way |
| `--prompt-cache-mb N` | memory kept for the prompt-cache prefix tree (default: half of what Metal's recommended GPU working set leaves after the weights, logged at load; `0` disables), see below |
| `--prompt-cache-disk-mb N` | persist the prompt cache to `~/.basalt/prompt-cache/` (`--prompt-cache-dir`) on graceful shutdown, before a model switch and when `basalt chat` exits, so a resumed session or restarted server restores its prefix from disk instead of prefilling it (default `0`, off). The hottest branches are written first, up to `N` MB and only those whose write fits the shutdown deadline at an assumed 1 GB/s; a longer path of the same conversation replaces the shorter files it covers, the least recently used files are deleted to make room, files unused for `--prompt-cache-disk-days` (7) go at start, and `basalt sessions prune` does the same offline. The directory must be new, empty or an earlier prompt-cache directory (one holding other files is refused). A file is used only by the same model (weight blobs, or sampled weight content outside the HF cache, config, tokenizer and chat template), basalt version and cache format; anything else, a truncated file or a different layer layout falls back to a normal prefill. Needs `--session-log full` (the files hold the conversation as KV tensors) and the in-memory prompt cache; `--kv-bits`, `--kv-zones` and multimodal prompts are never persisted |
| `--cache-blend` | reuse cached prompt chunks in any order, approximately (CacheBlend); `--cache-blend-ratio R` sets the share recomputed (default 0.3), see below |
| `--min-p`, `--top-k`, `--seed` | sampling; `--seed` makes a non-greedy run reproducible; a request's `min_p` overrides `--min-p` |
| `--max-tokens` | default generation ceiling (2048–128000) |
| `--dry-run` | memory feasibility check, then exit without loading |
| `--port`, `--host` | default 127.0.0.1:8080; any other host needs an API key (`BASALT_API_KEY` or `--auto-generate-key`) |
| `virtual_keys` (config only) | extra API keys with their own cloud budget: `"virtual_keys": [{"key": "vk-…", "id": "team-a", "max_budget": 2.0}]` in `~/.gem/config.json`; a key is at least 16 characters, ids and keys are unique; spend per key survives restarts |
| `--max-concurrent N`, `--max-queue N` | generations decoding at once (default 1; each holds its own KV cache) and requests waiting for a slot (default 8), by priority — `basalt chat` first, then service requests (titles, summaries) and harness probes, then API traffic, then benchmarks — and in arrival order within a level; a higher-priority request finding the queue full turns away the lowest-priority waiter; beyond the queue the server answers 503 with `Retry-After` (a stream that already sent its headers gets an error event instead). A model switch waits for running generations and new requests queue behind it; two loads of the same model share one, a load of another model meanwhile is 503. On SIGTERM/SIGINT generations get 5 s to finish, then they are cancelled so the process exits |
| `--session-log full\|meta\|off` | local JSONL session log under `~/.basalt/sessions/<date>/<session>.jsonl` (`0700`/`0600`): one `request` line before the engine call and one `response` line after it (status, tokens, timing, memory, finish reason) for every endpoint, plus `error` lines for rejected requests (size only, never the body or headers) and `model_switch`. `full` (default) also stores the conversation and the reply, `meta` only sizes and SHA-256 hashes; `--session-log-dir`, `--session-log-retention-days` (14), `--session-log-max-total-mb` (1024); the session is named by `X-Basalt-Session` (or `x-session-id`); without one, a request that extends an earlier conversation joins its session and anything else starts a new one. A known id appends to its existing file after a server restart (`session_resume`), and a session idle for `--session-idle-minutes` (30) is suspended. `basalt sessions list | show <id\|latest> | stats <id\|latest>` reads the files without loading a model; `basalt chat` keeps a multi-turn history in its own session and `basalt chat --resume <id\|latest>` continues it (not for `meta` sessions). In `full` mode private-key blocks and API tokens (`sk-`, `ghp_`/`github_pat_`, `AKIA`, `xox*-`, `Bearer`) are replaced by `[REDACTED:*]` (best effort, not a guarantee; `--session-log-redact off`), and a request that extends an earlier one of the session stores only its new messages plus `parent` and `prefix_count` |
| `--rate-limit N` | server-wide requests per minute, 429 beyond it — default off on loopback, 120 otherwise; `0` disables; also `rate_limit` in `~/.gem/config.json` |
| `--allow-origin URL` | a browser origin (`https://app.example:3000`, or a local UI such as `http://localhost:3000`) allowed to call the API besides the server's own; repeatable; also `allowed_origins` in `~/.gem/config.json` |
| `--cloud-fallback MODEL` | answer from a model on an OpenAI-compatible server when the local model cannot: the request's KV does not fit in the generation memory budget (the same KV room that caps `max_tokens`), the prompt is longer than `--local-context-limit N`, or the cost router rates the task beyond the local model and the gain beats the price. `--cloud-base-url URL` picks the server (default OpenRouter; any `/v1` endpoint such as vLLM, Ollama, LM Studio or another Mac, no key needed there); `--no-cloud-fallback` turns it off. Also `cloud_fallback` in `~/.gem/config.json`: `{"enabled": true, "base_url": "http://192.168.1.20:8000/v1", "model": "qwen3-235b", "api_key_env": "LAN_KEY", "local_context_limit": 32768}`; `enabled: false` keeps it off. The OpenRouter key is sent only to openrouter.ai. Spend is capped by the `basalt cloud configure` daily/monthly budgets, by `cloud_fallback.session_budget` (USD per `x-session-id`) and by each virtual key's `max_budget`; over budget the cloud is not called. `--router-params FILE` (trained Bradley–Terry parameters: `strongEmbedding`, `weakEmbedding`, `weights`, `projection`) turns on the capability check, tuned by `--cost-threshold θ` (default 0.5) and `--cache-stickiness λ` (default 0.1; a warm `x-session-id` keeps its side until the margin beats λ times the switch cost); also `router_params`, `cost_threshold`, `cache_stickiness` in `cloud_fallback`. Requests with tools or `response_format` and service requests stay local; a failed cloud call is answered locally |

**Prompt-cache reuse.** The server keeps the model's KV cache for recent prompts in a prefix tree: prompts that start with the same tokens share one stored copy of that start, and a request prefills only what follows the longest stored prefix. So the next turn of a conversation costs only its new tokens (a tool result appended to a long agent transcript, not the whole history again), and a second conversation that opens with the same system prompt and tools — a Claude Code subagent, another Codex thread — starts from the shared part instead of prefilling it again. Matching is by tokens, so no conversation id is needed (`prompt_cache_key`, `metadata.user_id` and `x-session-id` are ignored); title and other service requests never touch the cache. Under the `--prompt-cache-mb` budget the least recently used branch is cut from its end (a hybrid model's snapshot branch, see below, is removed whole, since its recurrent state is valid only at its end), so an idle conversation keeps its system prompt and tools longest, and the node a running request reuses up to is never evicted (for a hybrid model that is its snapshot node; the request has its own copy of the data, so evicting the rest of its branch cannot affect it); the tree also keeps at most 4096 branches, dropping the least recently used free one beyond that. Usage reports what was actually reused: `prompt_tokens_details.cached_tokens` (Chat Completions), `input_tokens_details.cached_tokens` (Responses), `cache_read_input_tokens` (Anthropic, where `input_tokens` counts only the prefilled rest). Hybrid attention/SSM models such as Qwen3.5/3.6 and Ornith-1.5 reuse a conversation's previous turn too: their recurrent state cannot be rewound, so while prefilling a prompt the server keeps a snapshot of it at the end of the request text, before the chat template's closing tokens, and the next turn that extends that text restores the snapshot and prefills only what follows (Ornith-1.5-9B, 12.6k-token agent prompt: the second turn takes about 1 s instead of about 36 s). A reused turn can differ from an uncached run after its first tokens: the prefill is chunked differently, which shifts bf16 results slightly (measured on Ornith: the restored state is bitwise equal to a from-scratch prefill cut at the same points, and differs from a one-phase prefill by at most 1.56 in the last-position logits with the same top 5). They reuse only from such a snapshot, so a second conversation with the same system prompt prefills in full. Sliding-window layers such as Gemma 3/4 and `--kv-bits` always prefill the full prompt.

**CacheBlend (opt-in, approximate).** The prefix tree only helps when a prompt starts the same way; an agent that pastes the same files in another order prefills them all again. `--cache-blend` also keeps the KV of each prompt chunk — a chunk ends after every blank line, whether the tokenizer writes it as one token or as two newline tokens, so files and messages separated by blank lines keep their own chunks; chunks under 16 tokens are always recomputed — and reuses a chunk wherever it lands in a later prompt (keys are stored before RoPE and rotated to the new position). The cached chunks were computed next to other text, so their KV is stale: following CacheBlend (arXiv:2405.16444), the first layer is recomputed for every token, and only `--cache-blend-ratio` of the reused tokens (default 0.3), plus every token not in the store, are recomputed on the later layers. The budget goes first to the first 32 tokens of each reused chunk, whose context changed the most when the chunk moved (as in EPIC, arXiv:2410.15332), ranked by how far their KV deviates on the second layer, and only then to the most deviating tokens of the rest. The blended forward costs about 1.2× the model's own prefill per recomputed token, so a prompt is blended only when (1 − ratio) × the share of reused tokens is at least 0.2, and only while the whole cache stays within 16 384 tokens; otherwise — a fresh prompt, little reuse, `--cache-blend-ratio 1`, a long context — it takes the model's own chunked prefill and its chunks are stored from that cache (keys are un-rotated from post-RoPE, exact up to the cache's dtype rounding, ~2⁻⁸ relative in bf16). The result is approximate. Measured on Qwen3-4B-4bit (M4 Pro) with six frozen files of 550–900 tokens (`benchmarks/cacheblend`) in four new orders, each with a new question (4 854 of ~5 066 tokens from chunks), scoring the full-prefill greedy answer (50 tokens) under teacher forcing — mean log-prob of the reference tokens and top-1 agreement per position, on this sample of 200 positions (0.01 is two positions):

| cache | log-prob | top-1 | perplexity |
|---|---|---|---|
| full prefill | −0.094 | 0.995 | 1.099 |
| ratio 0 | −0.180 | 0.925 | 1.198 |
| ratio 0.15 (the paper's value) | −0.166 | 0.945 | 1.181 |
| **ratio 0.3 (default)** | −0.123 | 0.960 | 1.131 |
| ratio 0.5 | −0.125 | 0.970 | 1.134 |
| ratio 1 (native prefill) | −0.094 | 0.995 | 1.099 |

Recomputing chunk heads first is what makes 0.3 the default: with layer-1 deviation alone, ratio 0.3 left a perplexity gap of 0.085 and even 0.5 left 0.038; with chunk heads first, 0.3 leaves 0.032 (0.50: 0.035). On four other orders and questions of the same files the gap is larger (0.096 at 0.3, against 0.169 without chunk heads and 0.120 for the old default 0.5), so the 0.05 bound holds on the benchmark sample, not on every prompt. `--cache-blend-ratio 0` trades quality for the shortest prefill. On Meta-Llama-3.1-8B-Instruct-4bit (same files, 4 816 of ~5 045 tokens from chunks) reordering hurts much less: the gap is 0.031 even at ratio 0 and 0.030 at 0.3, where the blended prefill is 2.78× faster than the native one (release build, quiet host). Llama- and Qwen3-family dense models only; other models keep the exact prefix path. With the flag on, the prompt-cache budget is split in half between the tree and the chunk store. `cached_tokens` still counts only the exact prefix-tree tokens; blended tokens are counted in `prompt_tokens` and logged (`[CacheBlend] Blended N prompt tokens: R from cached chunks, X of them recomputed`), and the tree only stores the exact head of a blended prompt. A blended pass briefly holds the assembled cache and the chunks' un-rotated keys and values at once (about twice the prompt's KV, e.g. ~5 GB for 16k tokens of Qwen3-4B), which the memory estimates do not include. At small ratios the fixed cost of layers 0–1 and the deviation pass dominates, so 0.15 and 0.3 are slower than the linear cost model suggests. Off by default because it changes outputs; `--prompt-cache-mb 0` turns it off as well.

Privacy: all clients share one prefix tree, and `cached_tokens` reveals how many leading tokens a prompt shares with earlier prompts, so a caller can probe what another client sent. On a server shared by people who must not see each other's prompts, run with `--prompt-cache-mb 0`; with `--cache-blend` the time to first token also shows which chunks another client sent.

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
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` | 9B | 4.7 GB | ✅ ~48 TPS (reasoning — see note) |
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | 35B MoE | 21 GB | ✅ untested above 24 GB |
| `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` | 8B MoE | 4.5 GB | ✅ ~76 TPS (always reasons) |
| `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` | 24B MoE | 12.6 GB | ✅ ~40 TPS (always reasons) |
| `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (Base BF16) | 27B | ~54 GB | ❌ Needs 64GB+ (Use 4-bit MLX version on 24GB) |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | 26B MoE | 14.5 GB | ❌ Gibberish output |
| `mlx-community/Qwen3.6-27B-4bit` | 27B | 14.5 GB | ❌ Gibberish output |
| `mlx-community/gemma-4-31b-it-4bit` | 31B | 17 GB | ❌ OOM (Needs 32GB+) |

> **Note on 24GB RAM Macs:** 
> - Unquantized 27B/35B models (~54 GB weights) exceed physical memory. Use MLX 4-bit / Dynamic Quant (DQ) quantized weights (e.g. `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit`, `Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX`) which comfortably fit in ~15 GB RAM.
> - Models > 10GB constrain KV-cache size under heavy multi-turn contexts. The KV cache takes `2 · layers · kv_heads · head_dim · 2 B` per token (fp16), read from the model's `config.json`; hybrid models count only their full-attention layers plus a fixed recurrent state, sliding-window layers stop at the window, `--kv-bits 4|8` stores `bits + 0.5` bits per element (`--kv-bits` does not apply to sliding-window layers). Qwen3-4B: 144 KiB per token, 4.5 GiB at 32k tokens; Ornith-1.5-9B (8 of 32 layers full attention): 32 KiB per token, 1.05 GiB at 32k. `serve --dry-run`, `models check`, `fit` (its quantization grid and decode-speed estimate — the RAM column and fit level still count weights only) and the memory-derived token budget use this figure (cap = min(requested max_tokens, --max-tokens, KV room − prompt tokens), where the KV room is Metal's recommended working set minus the loaded weights and the prompt-cache budget, and the prompt is counted with the model's tokenizer; a prompt that does not fit at all is refused with 413 before generation); without a downloaded config they fall back to a parameter-count estimate that is close for dense 7B–32B models but several times off for small, hybrid and MoE ones. Estimates, not measurements: the recurrent state of Mamba layers (Jamba, NemotronH) is not counted, only their attention layers; MLA models (DeepSeek V2/V3) are counted as the decompressed per-head K and V that MLX caches, not the compressed latent.

---

## REST API

Base URL: `http://127.0.0.1:8080` — **no authentication required**.

### Security

On loopback any local process may call the API without a key, but a web page in your browser may not:

- **Host.** Bound to loopback, the server answers only requests whose `Host` is `localhost`, `127.0.0.1` or `[::1]` (any port), anything else gets 403; this stops DNS rebinding, where an attacker's domain re-resolves to 127.0.0.1 and its page reads your responses. Bound to a specific address, that address is accepted as well. Bound to `0.0.0.0`/`::`, any `Host` is accepted: the API key is mandatory there.
- **Origin.** A request with an `Origin` header is refused (403) unless it is the server's own loopback origin (`http(s)://localhost|127.0.0.1|[::1]` on the port the server listens on; no port means 80 for http, 443 for https) or listed with `--allow-origin` / `allowed_origins` (`scheme://host[:port]`, no path, no trailing slash, no `*`; default ports match, so `https://app.example:443` equals `https://app.example`). Other local ports are not trusted: a page on a dev server (`localhost:3000`) or Jupyter could otherwise load models and run generation, so such a UI needs `--allow-origin http://localhost:3000`. Clients that send no `Origin` (curl, SDKs, Claude Code, Codex) are unaffected.
- **Content-Type.** Every `POST` must send `Content-Type: application/json` (a `charset` parameter is fine), otherwise 415. `curl -d` without `-H "Content-Type: application/json"` sends a form type and is refused.
- **Model ids.** REST `model` values are Hugging Face repo ids: `/v1/models/load` takes only `org/name`; in a generation request's `model` a path form (starting with `/`, `.` or `~`, or containing `..`, also hidden in `claude-local/..--x`; surrounding whitespace is ignored) is a 400, while any other value that is not a repo id (`basalt`, `meta-llama/llama-3.1-8b-instruct:free`) is an alias: the loaded model answers and nothing is switched. Local paths stay available to `basalt serve --model-path` and MCP. A failed load leaves the current model loaded, and error bodies never contain filesystem paths (they are in the server log).

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
| `temperature` | float | model | Sampling temperature (0–2); unset takes the model's `generation_config.json`, else 0.7 |
| `topP` | float | model | Nucleus sampling (0–1); unset takes `generation_config.json`, else 0.9 |
| `topK` | int | model | Keep the k most likely tokens; unset takes `--top-k`, then `generation_config.json`, else off |
| `presencePenalty` | float | off | −2…2, subtracted from every token among the last 1024 generated tokens (the prompt is not penalised) |
| `repetitionPenalty` | float | `--repetition-penalty` | (0, 2], multiplicative penalty on recent tokens |
| `thinkingBudget` | int | off | Reasoning tokens before the server forces the end of thinking (Qwen: "Considering the limited time by the user…" + `</think>`) so the answer fits in `maxTokens`; lowered to fit `maxTokens` minus the transition and an answer reserve of min(512, maxTokens/4); a `maxTokens` leaving fewer than 64 tokens after the reserve is 400; ignored when thinking is off, the model has no reasoning protocol, or it is a non-Qwen model whose protocol declares no transition |
| `minP` | float | server | Min-P 0–1: drop tokens below `minP × max p`. Overrides `--min-p` and the model profile; `0` turns it off |
| `dynamicTemperature` | bool | false | Entropy-based dynamic temperature (EDT): per token `T = T0·0.8^(1.5/H) + 0.5·R`, clamped to 0.05–2 (a colder request keeps its own temperature as the floor; `top_p`/`min_p` cut the original distribution), where `H` is the entropy of the next-token distribution and `R ∈ [0,1]` a repetition index. Near-greedy on confident tokens, warmer when a loop starts. No effect at `temperature: 0` |
| `speculativeLookup` | bool | `--speculative-lookup` | Prompt-lookup speculative decoding for this request: up to 16 tokens copied from an n-gram match in the prompt or the output are verified in one forward pass; after a rejected draft it pauses drafting for 1, 2, 4… up to 64 tokens. With `serve --speculative-recycling`, when no n-gram matches it drafts up to 6 tokens from the model's own argmax successors recorded twice in a row (Token Recycling, #38), with a pause of its own. Attention-only models and dense recurrent hybrids (Qwen3.5/Ornith, Nemotron-H: a rejected draft restores the recurrent state and replays the accepted tokens), without `--kv-bits` and text-only input; hybrids with expert routing (LFM2.5 MoE) and sliding-window models are ignored |

`/v1/chat/completions`, `/v1/responses` and `/v1/messages` take the same two fields as `min_p` and `dynamic_temperature`; MCP `gemma_generate` as `minP` and `dynamicTemperature`. The OpenAI endpoints take `top_k`, `presence_penalty`, `repetition_penalty` and `thinking_budget`; `/v1/messages` takes `top_k` and `thinking: {"type": "enabled", "budget_tokens": N}` as the thinking budget (1 ≤ N < `max_tokens`, else 400). All three take `speculative_lookup` (a non-boolean is 400); MCP `gemma_generate` takes `speculativeLookup` (a non-boolean is -32602). Usage reports `completion_tokens_details.reasoning_tokens` (chat; streamed with `stream_options.include_usage`) and `output_tokens_details.reasoning_tokens` (Responses): `0` when the model did not reason, and the forced transition counts as reasoning.

### OpenAI-compatible

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "basalt",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

Supports streaming SSE, system prompts, and multi-turn conversation. Sending a HuggingFace model ID in the `"model"` field triggers a hot-swap automatically.

A reasoning model's thinking comes back as `message.reasoning` (stream: `delta.reasoning`), the OpenRouter / Ollama name, and the same text again as `reasoning_content`, the name vLLM, DeepSeek, llama.cpp and LM Studio clients read. Assistant turns sent back with either field are accepted; the reasoning itself is not put back into the prompt.

### Anthropic-compatible

```bash
curl -s http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "basalt",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Implements the full Anthropic SSE event sequence (`message_start`, `content_block_start`, `content_block_delta`, `message_stop`). Accepts plain-string and block-array content formats.

`thinking` blocks carry an empty `signature`: basalt has no key to sign them with. They round-trip through basalt, but the real Anthropic API rejects a conversation that replays them, so drop them before sending the same history there.

**Prefill.** A conversation that ends with an assistant message continues that message instead of answering it in a new turn, on `/v1/messages`, `/v1/chat/completions` and `/v1/responses`, as on the Anthropic API, vLLM's `continue_final_message` and llama.cpp. The earlier turns are rendered up to the generation prompt with thinking off, the assistant text follows it unclosed, and the response holds only what the model writes after it: prefill `{"name":` and the reply is ` "Whiskers", "age": 3}`. With `response_format` or a forced tool the prefill is the start of the JSON document (a prefill the schema cannot start with is 400), and a forced tool's arguments include it. A prefill that opens a thinking block (`<think>…`) keeps the requested thinking and continues as reasoning. Several trailing assistant messages are joined as one prefill; an empty one is dropped; an assistant message with tool calls is still a finished turn. A prompt with images cannot be continued (400). Control tokens in the prefill (`<|im_end|>`, `<|im_start|>`, …) are neutralized like the rest of the conversation, so a prefill cannot forge a turn; only the thinking markers that open and close its reasoning stay real, and on a template that already opens `<think>` itself (Ornith, Qwen3.5+) the prefill's own leading `<think>` is dropped instead of doubled. With `--cloud-fallback` a prefilled request always stays local, since not every provider continues a trailing assistant message. On Nemotron Nano v2, which turns thinking off with a `/no_think` line in the system turn, a prefilled request renders a different system turn than a thinking-on request on the same history, so the two share only the prompt cache before it.

### SSE streaming (native API)

```bash
curl -N http://localhost:8080/api/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "maxTokens": 512}'
```

Each event is a `data: {...}` line; the stream ends with `data: [DONE]`. A failure mid-stream arrives as `event: error` with `data: {"error": ...}`.

### Swagger UI

Visit `http://localhost:8080/swagger` for interactive API docs.

---

## Claude Code Integration

### Option A — environment variables (per terminal session)

```bash
# Start Basalt
.build/release/basalt serve --model mlx-community/Qwen3.5-4B-4bit --rest

# In another terminal — scope vars only to this claude process
ANTHROPIC_BASE_URL=http://localhost:8080 \
ANTHROPIC_AUTH_TOKEN=local \
claude
```

`ANTHROPIC_AUTH_TOKEN` sends as `Authorization: Bearer local` (not `x-api-key`), so your real `ANTHROPIC_API_KEY` is never touched.

### Option B — shell function (add to `~/.zshrc`)

```bash
function claude-local() {
  ANTHROPIC_BASE_URL=http://localhost:8080          \
  ANTHROPIC_AUTH_TOKEN=local                        \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=mlx-community/gemma-4-e4b-it-4bit    \
  ANTHROPIC_DEFAULT_SONNET_MODEL=mlx-community/Qwen3.5-4B-4bit       \
  ANTHROPIC_DEFAULT_OPUS_MODEL=mlx-community/gemma-4-31b-it-4bit     \
  claude "$@"
}

claude-local                   # sonnet alias → Qwen 4B
claude-local --model haiku     # haiku alias → Gemma 4B (fastest)
claude-local --model opus      # opus alias → Gemma 31B (strongest)
```

### Option C — `./basalt-claude` launcher

Starts the server and Claude Code in one command (see Quick Start above).

### Tool calling

Claude Code's tools work on `/v1/messages`: the `tools` it sends reach the model's chat template, the model's calls come back as `tool_use` blocks (streamed as `content_block_start` / `input_json_delta` / `content_block_stop`, `stop_reason: "tool_use"`), and the `tool_use` / `tool_result` blocks of the history are rendered into the prompt, so the model sees what its tools returned. `/v1/chat/completions` does the same in the OpenAI form (`tools`, `message.tool_calls` / `delta.tool_calls`, `finish_reason: "tool_calls"`, role `tool` messages), as `/v1/responses` already did. Tool results in the history are matched to their calls by `tool_use_id` / `tool_call_id`, so parallel calls keep their names; images in a tool result show as `[image omitted]`. `tool_choice: none` keeps the tools in the prompt and lets no call through; a named tool keeps every tool in the prompt and lets only that one through (naming a tool that is not in `tools` is a 400); `any` / `required` are not enforced and work as `auto` (see #42); `parallel_tool_calls: false` / `disable_parallel_tool_use: true` end the generation after the first call. Tool input that is not a JSON object goes out as `{}` with the original arguments in a visible note. A call the model botches beyond repair, or a call to a tool that was not offered, arrives as a visible `[tool call … rejected: …]` note in the text. The legacy Chat Completions `functions` / `function_call` fields are ignored. Checked with Claude Code 2.1 on `mlx-community/Qwen3-4B-4bit`: list a directory with `Bash`, and read a file then edit it with `Read` / `Edit`.

### Model discovery

Claude Code (v2.1.126+) queries `GET /v1/models` at startup and adds returned models to the `/model` picker — but only if the ID starts with `claude` or `anthropic`. Basalt returns IDs in `claude-local/<hf-id>` form so discovery works automatically.

---

## OpenCode Integration

[OpenCode](https://github.com/opencode-ai/opencode) is a terminal-native AI coding agent. Basalt provides a native OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`) and Anthropic-compatible API that connects directly to OpenCode without cloud dependencies or API keys.

### 1. Start Basalt Server

Start Basalt with your desired local model (e.g. Huihui Qwen3.8 Abliterated, Ex0bit MYTHOS 26B, or Qwen3.5 4B):

```bash
# High capability (Abliterated / MoE)
.build/release/basalt serve --model AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit --rest

# Or fast coding model
.build/release/basalt serve --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --rest
```

### 2. Connect OpenCode

#### Option A — Terminal Environment Variables

```bash
# In your project folder:
OPENAI_BASE_URL=http://localhost:8080/v1 \
OPENAI_API_KEY=local \
OPENAI_MODEL=basalt \
opencode
```

#### Option B — OpenCode Configuration File (`~/.config/opencode/config.json` or `opencode.json`)

```json
{
  "provider": "openai",
  "base_url": "http://localhost:8080/v1",
  "api_key": "local",
  "model": "basalt",
  "temperature": 0.7,
  "max_tokens": 16384
}
```

#### Option C — Shell Alias (add to `~/.zshrc`)

```bash
function basalt-opencode() {
  OPENAI_BASE_URL=http://localhost:8080/v1 \
  OPENAI_API_KEY=local \
  OPENAI_MODEL=basalt \
  opencode "$@"
}
```

---

## Downloading Models (mlx-community, huihui-ai, Ex0bit)

Basalt can browse, inspect, and download models from any creator or organization on Hugging Face:

```bash
# List models by author
basalt models list --author Ex0bit
basalt models list --author huihui-ai
basalt models list --author mlx-community

# Search with filters
basalt models list --author Ex0bit --search PRISM
basalt models list --search Huihui-Qwen3.8

# Download specific models
basalt models download AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit
basalt models download Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX
basalt models download Ex0bit/Qwen3.6-35B-A3B-PRISM-MLX-NVFP4

# Interactive picker with custom author
basalt models download --author Ex0bit
```

---

## MCP Integration (Claude Desktop / Cursor)

Add to your MCP config (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "basalt": {
      "command": "/path/to/.build/release/basalt",
      "args": ["serve", "--model", "mlx-community/Qwen3.5-4B-4bit", "--mcp"]
    }
  }
}
```

Available MCP tools:

| Tool | Description |
|---|---|
| `gemma_generate` | Generate text — `prompt`, `maxTokens`, `temperature`, `topP`, `minP`, `dynamicTemperature` |
| `gemma_status` | Returns server readiness, version, and current model |
| `status` | Engine health, loaded model, memory snapshot and telemetry — optional `modelId` to check |
| `check_hardware` | Apple Silicon resources: chip, unified memory, GPU, disk |
| `inspect_model` | `modelId` (repo ID or local path) — verify weight shards and detect the real bit width of MLX/Safetensors weights |
| `switch_model` | Switch to another model from the local cache — `modelId` |
| `gemma_add_knowledge` | Store a lesson in the Harness Experience Library (`~/.basalt/harness-library`) |
| `harness_validate` | Validate a Claude Code / opencode / AGENTS.md harness under the working directory or `BASALT_HARNESS_ROOTS` |
| `harness_optimize` | One Harness² propose–probe–compose pass on the loaded model (also `basalt harness optimize`) |
| `harness_library_query` | Search the Experience Library |

---

## Project Structure

```
Sources/                    — one SwiftPM target per directory; dependencies point down only
  BasaltCore/               — DTOs, errors, config, logger, profilers, calculators, HF cache, orchestrator
    Chat/                   — ModelProfile: everything that differs between model families
    Config/                 — ServerConfig value type
  BasaltInference/          — MLX inference engine, ContextDegradationProfiler, RepetitionGuard  (→ Core)
  BasaltCloud/              — OpenRouter / cloud model fallback (CostTracker, ModelRouter)    (→ Core)
  BasaltServer/             — REST (Hummingbird 2.x, Host/Origin/Content-Type guard, API key and rate-limit middleware) and MCP JSON-RPC stdio server (→ Core)
  BasaltUI/                 — Terminal UI (Markdown, spinner, progress bar, diff, table)       (→ Core)
  BasaltApp/                — SwiftUI REST chat client, not linked into the CLI (launch: #55)  (→ Core)
  BasaltCLI/                — Subcommands and the CLI root command (App/)                     (→ all of the above)
  BasaltBin/                — Thin executable wrapper (product `Basalt`, installed as `basalt`)
  PerformanceBenchmark/     — Standalone benchmark CLI target

Tests/
  BasaltTests/              — Unit + integration tests
    CLI/                    — Command parsing, Cloud, Fit, PromptContextBuilder
    Cloud/                  — CostTracker, ModelRouter, OpenRouterClient
    UI/                     — Clipboard, Diff, Markdown, ProgressBar, Spinner, Table, TerminalUI
    MockInferenceEngine.swift — Controllable mock for orchestrator tests

basalt-claude                       — Executable Swift launcher (build + serve + claude)
scripts/                    — Build and maintenance Swift scripts
docs/                       — Extended documentation
```

Only `BasaltApp` may import SwiftUI/AppKit; every target the CLI links stays headless (enforced by `ModuleBoundaryTests`).
Symbols newly shared across targets use the `package` access level (existing public API stays `public`).

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
| `scripts/cleanup.swift` | `--mode=daily` rotates logs/images/reports, `--mode=release` removes build/runtime artifacts before publishing, `--mode=all` both; dry run unless `--apply` |
| `scripts/archive.sh` | Archive logs/reports/screenshots |

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
basalt serve --model ornith-ai/Ornith-1.5-9B-MLX-4bit --rest --reasoning-effort none &

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
| [Rainbow](https://github.com/onevcat/Rainbow) | terminal colour and layout |
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
