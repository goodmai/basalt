# REST API

Base URL: `http://127.0.0.1:8080` — **no authentication required**.

## Security

On loopback any local process may call the API without a key, but a web page in your browser may not:

- **Host.** Bound to loopback, the server answers only requests whose `Host` is `localhost`, `127.0.0.1` or `[::1]` (any port), anything else gets 403; this stops DNS rebinding, where an attacker's domain re-resolves to 127.0.0.1 and its page reads your responses. Bound to a specific address, that address is accepted as well. Bound to `0.0.0.0`/`::`, any `Host` is accepted: the API key is mandatory there.
- **Origin.** A request with an `Origin` header is refused (403) unless it is the server's own loopback origin (`http(s)://localhost|127.0.0.1|[::1]` on the port the server listens on; no port means 80 for http, 443 for https) or listed with `--allow-origin` / `allowed_origins` (`scheme://host[:port]`, no path, no trailing slash, no `*`; default ports match, so `https://app.example:443` equals `https://app.example`). Other local ports are not trusted: a page on a dev server (`localhost:3000`) or Jupyter could otherwise load models and run generation, so such a UI needs `--allow-origin http://localhost:3000`. Clients that send no `Origin` (curl, SDKs, Claude Code, Codex) are unaffected.
- **Content-Type.** Every `POST` must send `Content-Type: application/json` (a `charset` parameter is fine), otherwise 415. `curl -d` without `-H "Content-Type: application/json"` sends a form type and is refused.
- **Model ids.** REST `model` values are Hugging Face repo ids: `/v1/models/load` takes only `org/name`; in a generation request's `model` a path form (starting with `/`, `.` or `~`, or containing `..`, also hidden in `claude-local/..--x`; surrounding whitespace is ignored) is a 400, while any other value that is not a repo id (`basalt`, `qwen/qwen3-8b-instruct:free`) is an alias: the loaded model answers and nothing is switched. Local paths stay available to `basalt serve --model-path` and MCP. A failed load leaves the current model loaded, and error bodies never contain filesystem paths (they are in the server log).

## Model management

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

## Raw generation

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
| `speculativeLookup` | bool | `--speculative-lookup` | Prompt-lookup speculative decoding for this request: up to 16 tokens copied from an n-gram match in the prompt or the output are verified in one forward pass; after a rejected draft it pauses drafting for 1, 2, 4… up to 64 tokens. With `serve --speculative-recycling`, when no n-gram matches it drafts up to 6 tokens from the model's own argmax successors recorded twice in a row (Token Recycling, #38), with a pause of its own. Attention-only models and dense recurrent hybrids (Qwen3.5/Ornith, Nemotron-H: a rejected draft restores the recurrent state and replays the accepted tokens), sliding-window models (Gemma 4: each verify round is staged, so a rejected draft never overwrites the window's ring), without `--kv-bits` and text-only input; recurrent hybrids with expert routing (LFM2.5 MoE) are ignored. On Gemma 4 26B-A4B (MoE) lookup gains 1.97× on code edits and costs 3–6% on prose and reasoning |

`/v1/chat/completions`, `/v1/responses` and `/v1/messages` take the same two fields as `min_p` and `dynamic_temperature`; MCP `gemma_generate` as `minP` and `dynamicTemperature`. The OpenAI endpoints take `top_k`, `presence_penalty`, `repetition_penalty` and `thinking_budget`; `/v1/messages` takes `top_k` and `thinking: {"type": "enabled", "budget_tokens": N}` as the thinking budget (1 ≤ N < `max_tokens`, else 400). All three take `speculative_lookup` (a non-boolean is 400); MCP `gemma_generate` takes `speculativeLookup` (a non-boolean is -32602). Usage reports `completion_tokens_details.reasoning_tokens` (chat; streamed with `stream_options.include_usage`) and `output_tokens_details.reasoning_tokens` (Responses): `0` when the model did not reason, and the forced transition counts as reasoning.

## OpenAI-compatible

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

## Anthropic-compatible

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

**Constrained output.** `response_format` constrains the answer, after any reasoning block, with XGrammar:
- `{"type": "json_object"}` or `{"type": "json_schema", "json_schema": {"schema": …}}`;
- `{"type": "regex", "regex": "[A-Z]{3}-\\d{4}"}`;
- `{"type": "grammar", "grammar": "root ::= …"}`, XGrammar's EBNF/GBNF dialect with root rule `root`;
- `{"type": "structural_tag", "format": …}`, XGrammar's structural-tag JSON (`sequence`, `const_string`, `regex`, `json_schema`, `tag`, …).

vLLM's `guided_regex` and `guided_grammar` request fields work too, one of them at a time. The MCP `generate` tool takes `regex` or `grammar`. A source over 64 KB, or one XGrammar does not compile, is a 400 that names the format.

**Prefill.** A conversation that ends with an assistant message continues that message instead of answering it in a new turn, on `/v1/messages`, `/v1/chat/completions` and `/v1/responses`, as on the Anthropic API, vLLM's `continue_final_message` and llama.cpp. The earlier turns are rendered up to the generation prompt with thinking off, the assistant text follows it unclosed, and the response holds only what the model writes after it: prefill `{"name":` and the reply is ` "Whiskers", "age": 3}`. With `response_format` or a forced tool the prefill is the start of the JSON document (a prefill the schema cannot start with is 400), and a forced tool's arguments include it. A prefill that opens a thinking block (`<think>…`) keeps the requested thinking and continues as reasoning. Several trailing assistant messages are joined as one prefill; an empty one is dropped; an assistant message with tool calls is still a finished turn. A prompt with images cannot be continued (400). Control tokens in the prefill (`<|im_end|>`, `<|im_start|>`, …) are neutralized like the rest of the conversation, so a prefill cannot forge a turn; only the thinking markers that open and close its reasoning stay real, and on a template that already opens `<think>` itself (Ornith, Qwen3.5+) the prefill's own leading `<think>` is dropped instead of doubled. With `--cloud-fallback` a prefilled request always stays local, since not every provider continues a trailing assistant message. On Nemotron Nano v2, which turns thinking off with a `/no_think` line in the system turn, a prefilled request renders a different system turn than a thinking-on request on the same history, so the two share only the prompt cache before it.

## Image input

A vision model (Gemma 4, Qwen3.5-VL, … — any model MLXVLM loads) takes images on `/v1/chat/completions`, `/v1/responses`, `/v1/messages` and the MCP `generate` tool:

- **OpenAI:** a content part `{"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}` (`/v1/responses`: `{"type": "input_image", "image_url": "data:…"}`).
- **Anthropic:** a content block `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "…"}}`.
- **MCP:** `images: [{"data": "<base64>", "mime": "image/png"}]`, or `[{"path": "/abs/file.png"}]` for a file inside a directory given with `serve --image-dir <dir>` (repeatable). The path is checked after resolving symlinks; with no `--image-dir` every path is refused.

PNG, JPEG, WebP and HEIC, at most 8 images per request, 20 MB of encoded bytes per image and 32 MB for all images of a request; each overflow is a 400 that names the limit. Remote URLs (`http(s)://` in `image_url`, `source.type: "url"`) are refused with 400 — basalt does not fetch them; send the image as a `data:` URI. An image sent to a model that is not a vision model is a 400 (`<model> does not take images`), never a silent drop, and a request with images never goes to `--cloud-fallback`. The prompt cache is keyed by the SHA-256 of each image: the same images in the same order reuse the cached prefix, different images reuse nothing. Session logs record digests and byte counts, never image bytes. Images inside tool results are still shown to the model as `[image omitted]`.

## SSE streaming (native API)

```bash
curl -N http://localhost:8080/api/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "maxTokens": 512}'
```

Each event is a `data: {...}` line; the stream ends with `data: [DONE]`. A failure mid-stream arrives as `event: error` with `data: {"error": ...}`.

## Telemetry and Prometheus

- **`GET /api/v1/stats`** returns the telemetry snapshot as JSON.
  - It also carries `windows`: the trailing 1 minute, 5 minutes and 1 hour with requests, tokens, p50/p95 time to first token and tok/s, the prompt-cache hit ratio and the draft acceptance ratio.
  - It also carries `decisions`: per decision model, requests, answers, abstentions, escalations, cloud cost, latency and answers per confidence decile.
- **`GET /api/v1/metrics`** returns the same data as OpenMetrics text for Prometheus or Grafana.
  - Metric names start with `basalt_`. Window and quantile are labels, as is the decision model.
  - It goes through the server's API key like every other endpoint: give the scraper the bearer token.

```yaml
scrape_configs:
  - job_name: basalt
    metrics_path: /api/v1/metrics
    authorization: { credentials_file: /etc/prometheus/basalt.token }
    static_configs: [{ targets: ["127.0.0.1:8080"] }]
```

Telemetry holds counts, timings and histograms only: no prompt or answer text (#450).

## Swagger UI

Visit `http://localhost:8080/swagger` for interactive API docs.
