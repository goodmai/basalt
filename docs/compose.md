# basalt compose

`basalt compose` serves a model system described in one YAML file. The file lists local and cloud models and how each is kept in memory. It also describes the pipelines that route a request between them. Each pipeline runs perceive (OCR, speech), then decide (System One), then answer (a chat model). This is a mixture of models, in the style of a docker-compose file for models.

```bash
basalt compose plan -f basalt.yaml        # validate and show the compute plan on this Mac
basalt compose up -f basalt.yaml          # serve it; run it again with a changed file to apply the diff
basalt compose ps                         # what is loaded, when on-demand models unload, cloud spend
basalt compose down <name>                # stop it
basalt compose up document-qa             # bundled files: document-qa, voice-agent
```

`compose up` starts `basalt serve` with the flags the file expands to; it prints them at start. It serves the same REST and MCP surfaces. Flags after `--` are passed to serve, for example `basalt compose up -f basalt.yaml -- --port 8090`.

## The file

```yaml
version: 1
name: support-desk            # the system's name, used by ps, down and up-as-diff
memory: auto                  # resident | sequential | auto
budget: { usd_per_day: 2 }    # optional cap on all cloud spend

models:
  chat:     { source: mlx-community/gemma-4-12B-it-qat-4bit }
  big:      { source: lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly, keep: on-demand, idle_unload: 10m }
  ocr:      { source: mlx-community/PaddleOCR-VL-1.5-8bit, keep: on-demand, idle_unload: 2m }
  asr:      { source: divergentlabs/Nemotron-3.5-Transcribe-30B-A4B-MLX, keep: sequential }
  jev:      { source: "cloud:openrouter/~typesafe/jev-latest", api_key_env: OPENROUTER_API_KEY }
  frontier: { source: "cloud:openrouter/anthropic/claude-haiku-5.5", api_key_env: OPENROUTER_API_KEY, budget: { usd_per_day: 1 } }

pipeline:                     # the default pipeline: /v1/pipeline and MCP basalt_pipeline
  perceive:
    documents: [ocr, apple-vision]     # the first available reads pdf and image attachments
    audio: [asr, speech-analyzer]
  decide:
    cascade: [jev, chat]               # up to three cloud models, then one local model
    escalate_below: 0.6
    abstain_below: 0.75                # questions below this confidence go to the answer model
  answer:
    routes:                            # the first matching rule wins
      - when: { decide: { route: code }, cost_left: true }
        model: frontier
        fallback: chat
      - when: { prompt_tokens: { over: 8000 } }
        model: big
      - model: chat                    # the last rule has no when and names a local model

pipelines:                    # more pipelines, served at /v1/pipelines/<name>
  triage:
    decide: { cascade: [chat] }
    answer: chat
```

### Models
- **`source`:** a Hugging Face repo, a local path (`./`, `/` or `~`), or `cloud:openrouter/<model>`.
- **`keep`** applies to local models:
  - `resident` is the default and stays loaded;
  - `on-demand` loads on the first request that needs the model and unloads after `idle_unload`. Durations look like `30s`, `5m` or `1h`, and the default is 120 s;
  - `sequential` loads a perceiver for its stage only. In the default pipeline it can also unload the chat model around the perceiver, if the two don't fit together.
- **Cloud models:**
  - Name the environment variable that holds the key with `api_key_env`. A key written in the file is refused, and so is `${…}` interpolation.
  - `budget: { usd_per_day }` caps one model's spend.
  - `data: no-perceived` is the only data rule, and it is the default: text read from documents or audio never goes to a cloud model.

### Memory and the plan
`compose plan` adds up the resident bytes of every local model. It counts an on-demand model as resident, because it can be loaded at any time.

- `memory: resident` needs the sum to fit the working set.
- `memory: sequential` needs only the largest stage plus 2 GB.
- `memory: auto` picks resident when it fits.

A system that does not fit is refused with the numbers before anything loads. A model that is not downloaded is reported as size unknown.

### Decide
- **Order:** `cascade` lists the decision models in order. Each stage answers the questions it is confident about (at least `escalate_below`) and passes the rest on.
- **The local stage:** the last stage is local. It is either the chat model (read out as `self`) or a decision model such as a Laya Core AI bundle. Every pipeline shares this one local decision model.
- **What the cloud gets:** requests with documents or audio go only to the local stage.

### Answer routes
- **Matching:** a rule's `when` matches when every condition holds. Conditions:
  - `modality`: documents, audio or text;
  - `decide`: question ID to label, from System One;
  - `prompt_tokens: { over: N }`: estimated at 4 bytes per token;
  - `escalated`: the decide stage escalated, or a question went to the answer model;
  - `cost_left`: the rule's model still has budget.
- **Order of tries:** the rule's `model` first, then its `fallback`, then the pipeline's default local model.
- **Skips:** a cloud model is skipped for perceived text, without budget, after a refused key, or while rate limited. Each skip is named in the receipt.

### Cloud failures and spend
- A refused key (HTTP 401/403) disables that cloud model until the server restarts, with one log line saying so.
- A rate limit (429) pauses it for `Retry-After`, or 60 s if the header is missing.
- While a cloud model is disabled or paused, the next stage or the fallback answers.
- The day's spend per model and in total is kept in `~/.basalt/cloud-spend.json`, so a restart does not reset a cap. Days are UTC.

## Running systems
- **`compose up` on a running system:** if a system with the same `name` runs, `compose up` sends it the file (`POST /v1/compose/apply`) and applies the difference.
  - It loads added models, reloads changed ones and unloads removed ones.
  - It swaps the pipelines between requests.
  - It prints `load`, `reload`, `unload` and `keep` per model.
  - A cloud model's budget or key variable changes without a reload.
- **Changes that need a restart:** a different chat model, memory mode or local decision model, or a sequential perceiver. Run `compose down` and then `compose up`.
- **`/v1/systemone`** keeps the cascade it started with.
- **`compose ps`:** lists each running system (`GET /v1/compose`) with its pipelines and, per model:
  - whether it is loaded;
  - for an on-demand model, its phase, when it was last used and when it unloads;
  - for a cloud model, its spend today against its cap, and a refusal if one applies.
- **`compose down`:** sends the server SIGINT. It finishes in-flight requests and unloads.
- **Where state lives:** the running system's pid and port are in `~/.basalt/compose/<name>.json`.
- **API key:** when the server has an API key, set `BASALT_API_KEY` for `ps` and up-as-diff.

## Receipts
Each `/v1/pipeline` response, and each session log record, carries a receipt. Per stage it gives the model, milliseconds and tokens. Decide stages also carry their cost and any cloud stages that were skipped. Answer stages carry the rule that fired (`route`), the cost and the skipped models. `swap_ms` is the time spent loading on-demand and sequential models.

## Examples
- `docs/examples/compose/mac-16gb.yaml`: Ornith 9B local, OCR on demand, long prompts to Claude Haiku within $0.50 a day.
- `docs/examples/compose/mac-24gb.yaml`: Gemma 4 12B, OCR on demand, code to Haiku, a second `support` pipeline, $1 a day in total.
- `docs/examples/compose/mac-64gb.yaml`: Gemma 4 26B-A4B with Qwen3.8-27B and Nemotron Transcribe on demand, no cloud answers.
- `docs/examples/compose/document-qa-cloud.yaml`, `mom.yaml` and `too-big.yaml`: the files behind the README tables.

The measurements are in the README section "basalt compose" and in #412.
