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

The server binds its port first and loads the models after. While they load, `compose ps` shows them as `loading`, `GET /v1/health/ready` answers 503, and the run record already exists, so a second `compose up` finds the system instead of starting another. Once the models are loaded, each gets a startup probe:
- the chat model and each local answer model generate one token;
- each local decision model answers one question;
- an on-demand model's files are checked without loading it.

A model that fails its probe shows as `failed`, with the reason, in `compose ps`. `/v1/health/ready` stays 503 until a `compose up` replaces the model. A model that `compose up` loads or reloads gets the same probe. With `--mcp` the models load before the server starts, as before.

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
    cascade: [jev, chat]               # up to four stages, cloud and local in any order, the last one local
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
- **`source`:** a Hugging Face repo, a local path (`./`, `/` or `~`), `cloud:openrouter/<model>`, or `cloud:gemini/<model>` for Google's decision models such as `gemini-decision-1` (decide cascades only, key usually in `GEMINI_API_KEY`).
- **`keep`** applies to local models:
  - `resident` is the default and stays loaded;
  - `on-demand` loads on the first request that needs the model and unloads after `idle_unload`. Durations look like `30s`, `5m` or `1h`, and the default is 120 s;
  - `sequential` loads a perceiver for its stage only. In the default pipeline it can also unload the chat model around the perceiver, if the two don't fit together.
- **Cloud models:**
  - Name the environment variable that holds the key with `api_key_env`. A key written in the file is refused, and so is `${…}` interpolation.
  - `budget: { usd_per_day }` caps one model's spend.
  - `data: no-perceived` is the default: text read from documents or audio never goes to a cloud model.
  - `data: media`, only for `cloud:gemini` decision models, sends a pipeline's attached files themselves (PDFs, images, audio as WAV) and their perceived text to that model for the decision. Use it only for documents that may leave this Mac.

### Memory and the plan
`compose plan` adds up the resident bytes of every local model. It counts an on-demand model as resident, because it can be loaded at any time.

- `memory: resident` needs the sum to fit the working set.
- `memory: sequential` needs only the largest stage plus 2 GB.
- `memory: auto` picks resident when it fits.

A system that does not fit is refused with the numbers before anything loads. A model that is not downloaded is reported as size unknown.

### Decide
- **Order:** `cascade` lists the decision models in order. Each stage answers the questions it is confident about (at least `escalate_below`) and passes the rest on.
- **Stages in any order:** up to four stages, cloud and local in any order. An example is a small local judge first, a cloud model next and a larger local model last: `[laya, jev, chat]`.
  - The last stage is local.
  - Every local stage is held as a decision engine: the chat model, read out as `self`, or a decision model such as a Laya Core AI bundle or an OpenJev bundle.
  - Each pipeline names its own stages. A decision model that several pipelines name is loaded once.
  - A decision model only decides; only the chat model both answers and decides.
- **Calibration:** a readout decision model takes its temperatures from a `basalt_readout.json` in its folder (`{"temperature": …, "noul_temperature": …, "noul_bias": …}`), if there is one. Otherwise it uses the known values for its name, and otherwise none. #448 plans to fit these files against Gemini Decision 1.
- **Sizes:** `compose plan` sizes an OpenJev head bundle as its base model plus its adapter, because the LoRA is applied unmerged on the base as it is stored, 4-bit included. serve's own plan does the same.
  - Measured on OpenJev-0.6B: 100 labelled states, 300 questions, M4 Pro, commit 066da0a. The reference is the merged float32 path, which matches PyTorch within 1e-4 (#406).

    | Base, LoRA unmerged | Accuracy | Same top answer as float32 | Active memory | Median per state |
    |---|---|---|---|---|
    | merged float32 (reference) | 176/300 | — | 2,273 MB | 185 ms |
    | bf16, as the bundle names it (the default) | 175/300 | 288/300 | 1,141 MB | 181 ms |
    | 8-bit | 176/300 | 279/300 | 608 MB | 184 ms |
    | 4-bit | 149/300 | 215/300 | 324 MB | 185 ms |

  - A 4-bit base costs this 0.6B model 9 points, while 8-bit keeps its accuracy at a quarter of the memory. serve and compose load the base the bundle names, so a quantized base is only reached through `JevEvaluator` directly.
  - Rerun it with `BASALT_E2E=1 BASALT_E2E_JEV_MODEL=<bundle> BASALT_E2E_JEV_BASE=<bf16 base> BASALT_E2E_JEV_BASE_8BIT=… BASALT_E2E_JEV_BASE_4BIT=… swift test --filter JevRealModelTests/unmergedArms`.
- **What the cloud gets:** requests with documents or audio skip the cloud stages and go only to the local ones, in their order.
- **Thresholds:** questions whose confidence is below `abstain_below`, or whose top two options are closer than `min_margin`, go to the answer model.

### Answer routes
- **Matching:** a rule's `when` matches when every condition holds. Conditions:
  - `modality`: documents, audio or text;
  - `decide`: question ID to label, from System One;
  - `prompt_tokens: { over: N }`: estimated at 4 bytes per token;
  - `escalated`: the decide stage escalated, or a question went to the answer model;
  - `cost_left`: the rule's model still has budget.
- **Order of tries:** the rule's `model` first, then its `fallback`, then the pipeline's default local model.
- **Skips:** a cloud model is skipped for perceived text, without budget, after a refused key, or while rate limited. Each skip is named in the receipt.

### Version 2: variables and profiles
`version: 2` adds two blocks, so that one file serves several Macs. A version 1 file keeps working; using either block in it is refused with "needs version: 2".

```yaml
version: 2
name: desk
variables:
  chat_repo: mlx-community/gemma-4-12B-it-qat-4bit
profiles:
  small:
    when: { memory_gb: { under: 20 } }
    variables: { chat_repo: mlx-community/gemma-4-E4B-it-4bit }
  big:
    when: { memory_gb: { over: 48 }, chip: Max }
models:
  chat:  { source: "${chat_repo}" }
  large: { source: lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly, keep: on-demand, profiles: [big] }
```

- **`variables`:** lowercase names with one-line plain values. `${name}` is replaced anywhere in the file. An undeclared `${…}` is refused, so environment variables still never reach the file; secrets come only by `api_key_env`.
- **`profiles`:** a profile is on when its `when` holds on this Mac: `memory_gb` with `under` and `over` (unified memory in GB), and `chip` (a case-insensitive part of the chip name, such as `M4 Pro` or `Max`). A profile without `when` is on only when named with `--profile`.
  - An on profile's `variables` override the file's.
  - A model or a named pipeline with `profiles: [...]` exists only while one of them is on.
- **Overrides:** `--set name=value` overrides a declared variable and `--profile name` turns on exactly the named profiles, for `config`, `plan` and `up`. `compose up` passes them on to the server, and up-as-diff sends them with the file.
- **Paths:** a `./` or `../` model path is relative to the compose file's folder.
- **`basalt compose config -f <file>`** prints the file as it resolves on this Mac, with the profiles that are on. **`basalt compose schema`** prints the JSON Schema of the format for editors.

### Cloud failures and spend
- A refused key (HTTP 401/403) disables that cloud model until the server restarts, with one log line saying so.
- A rate limit (429) pauses it for `Retry-After`, or 60 s if the header is missing.
- While a cloud model is disabled or paused, the next stage or the fallback answers.
- The day's spend per model and in total is kept in `~/.basalt/cloud-spend.json`, so a restart does not reset a cap. Days are UTC.

## Running systems
- **`compose pull`:** downloads every local model of the file into the Hugging Face cache and verifies each file, printing progress every tenth. After a pull, `compose plan` has exact sizes and `compose up` loads without waiting for downloads.
- **Sizes in `compose plan`:** a model in the cache is measured on disk. A model that is not downloaded is sized from the Hugging Face API's file list. If neither knows the size, `plan` refuses and names the model: run `compose pull`, or pass `--assume-size <GB>` to count it at that size. `plan` no longer counts an unknown model as 0 bytes.
- **`compose diff`:** shows what `compose up` would change on the running system with the same name, without changing it (`POST /v1/compose/diff`). It lists loads, unloads, keys and budgets, a needed restart, or a refusal: a missing key, or a system that does not fit. It exits non-zero on a restart or a refusal.
- **Events:** `GET /v1/compose/events` is a server-sent event stream of what the system does. Event kinds are `loading`, `ready`, `registered` (on demand), `failed` (a startup probe), `applied`, `refused` and `up`. A new client first gets the last 100 events. `compose up --wait` prints them while it waits.
- **`compose up` on a running system:** if a system with the same `name` runs, `compose up` sends it the file (`POST /v1/compose/apply`) and applies the difference.
  - While the system is still loading, or applying another change, it answers 503. `compose up` waits and sends the file again, up to `--wait-timeout` seconds (default 1800).
  - With `--wait`, `compose up` then waits until `GET /v1/health/ready` answers 200, prints the models it is waiting for, and exits non-zero if the system is not ready within the timeout.
  - It loads added models, reloads changed ones and unloads removed ones.
  - It swaps the pipelines between requests.
  - It checks the new system against this Mac's memory first, as `compose plan` does, with the models of every pipeline counted. A system that does not fit is refused, and nothing changes.
  - When the new system fits but old and new models do not fit side by side, it unloads the models it replaces before loading their successors. A request that needs one of them in that window waits for the load or fails. If a load then fails, it loads the old models back.
  - It prints `load`, `reload`, `unload` and `keep` per model, `budget` for each changed cloud budget (`total` for the file's) and `key` for each cloud model whose `api_key_env` changed.
  - A cloud model's budget or key variable changes without a reload. The new limits apply at once, and a model disabled by a refused key is enabled again when its key variable changes.
  - A different chat model is loaded into the serving model's place and printed as `chat`. Requests wait while it loads. If a later step of the apply fails, the old chat model is loaded back.
  - A pipeline's new local decision model is loaded like any added model. The default pipeline's is also printed as `decide`. The pipelines and `/v1/systemone` use it from the next request. A model that starts or stops deciding is reloaded in its new role.
- **Changes that need a restart:** a different memory mode or a sequential perceiver. Run `compose down` and then `compose up`.
- **`compose ps`:** lists each running system (`GET /v1/compose`) with its pipelines and, per model:
  - its state, one of `loading`, `ready`, `unloading`, `unloaded`, `idle-unloaded` (on demand, not loaded now), `decision` (held by the decision engine), `cloud`, `no-key`, `over-budget`, `rate-limited`, `refused` and `failed`;
  - for an on-demand model, when it was last used and when it unloads;
  - for a cloud model, its spend today against its cap, and a refusal if one applies.
- **Health:** `GET /v1/health/live` answers 200 while the server answers HTTP. `GET /v1/health/ready` answers 200 once every model can serve. Until then it answers 503 and lists the models still waiting, such as `chat (loading)`. An idle on-demand model, an over-budget cloud model and a rate-limited cloud model count as able to serve; the cascade or the router goes past them.
- **Cloud keys:** `compose up` refuses to start when a cloud model's `api_key_env` is not set, and names the variable. Up-as-diff checks the running server's environment instead.
- **`compose down`:** sends the server SIGINT. It finishes in-flight requests and unloads. If the process is still running after 30 s, `down` says so and exits non-zero.
- **Where state lives:** the running system's pid, port and process start time are in `~/.basalt/compose/<name>.json`. A record whose pid now belongs to another process, because the pid was reused, counts as not running.
- **API key:** when the server has an API key, set `BASALT_API_KEY` for `ps` and up-as-diff.

## Receipts
Each `/v1/pipeline` response, and each session log record, carries a receipt. Per stage it gives the model, milliseconds and tokens. Decide stages also carry their cost and any cloud stages that were skipped. Answer stages carry the rule that fired (`route`), the cost and the skipped models. `swap_ms` is the time spent loading on-demand and sequential models.

## Examples
- `docs/examples/compose/any-mac.yaml`: one version 2 file for 16, 24 and 64 GB Macs. The chat model comes from a profile chosen by memory, and the 64 GB profile adds a 27B model on demand and a `long` pipeline.
- `docs/examples/compose/mac-16gb.yaml`: Ornith 9B local, OCR on demand, long prompts to Claude Haiku within $0.50 a day.
- `docs/examples/compose/mac-24gb.yaml`: Gemma 4 12B, OCR on demand, code to Haiku, a second `support` pipeline, $1 a day in total.
- `docs/examples/compose/mac-64gb.yaml`: Gemma 4 26B-A4B with Qwen3.8-27B and Nemotron Transcribe on demand, no cloud answers.
- `docs/examples/compose/document-qa-cloud.yaml`, `mom.yaml` and `too-big.yaml`: the files behind the measurements below.

More measurements are in #412.

## Measured

```yaml
version: 1
name: document-qa-cloud
models:
  chat:  { source: mlx-community/gemma-4-12B-it-qat-4bit }
  ocr:   { source: mlx-community/PaddleOCR-VL-1.5-8bit, keep: on-demand, idle_unload: 30s }
  jev:   { source: "cloud:openrouter/~typesafe/jev-latest", api_key_env: OPENROUTER_API_KEY }
pipeline:
  perceive: { documents: [ocr, apple-vision] }
  decide:   { cascade: [jev, chat], escalate_below: 0.6 }
  answer: chat
```

```bash
basalt compose plan -f docs/examples/compose/document-qa-cloud.yaml
basalt compose up -f docs/examples/compose/document-qa-cloud.yaml -- --port 8080
basalt compose up voice-agent            # bundled: document-qa, voice-agent
```

```yaml
pipeline:
  decide: { cascade: [jev, chat], escalate_below: 0.6 }
  answer:
    routes:
      - when: { decide: { route: code }, cost_left: true }
        model: haiku          # cloud:openrouter/anthropic/claude-haiku-5.5, budget: { usd_per_day: 0.5 }
        fallback: chat
      - model: chat
```

Mixed workload of 55 requests through `/v1/pipeline` (30 decisions, 10 code, 10 chat, 5 documents), M4 Pro, release build (`scripts/compose_mom_bench.swift`, `docs/examples/compose/mom.yaml`):

| Setup | Labels right | Cloud per 1000 requests | p50 | p95 | Local stage time |
|---|---|---|---|---|---|
| local only (Gemma 4 12B decides and answers) | 55/55 | $0 | 0.92 s | 10.0 s | 176 s |
| cloud first (Jev decides, Haiku answers all text) | 55/55 | $0.046 | 1.25 s | 4.8 s | 24 s |
| MoM (Jev decides, code to Haiku while budget lasts, the rest Gemma) | 55/55 | $0.035 | 1.21 s | 4.8 s | 53 s |

Measured with `compose up` on an M4 Pro with 24 GB, release build:

| File | Request | Receipt |
|---|---|---|
| document-qa-cloud | document (first) | OCR loaded on demand (489 ms), decided locally, answered: 6.1 s |
| | document (again) | OCR already loaded (0 ms): 5.0 s |
| | after 40 s idle | OCR unloaded, footprint 12 → 11 GB |
| | typed question | decided by Jev in the cloud: 0.6 s |
| | document | OCR reloaded (267 ms): 5.3 s |
| voice-agent (bundled) | audio | chat unloaded around Nemotron Transcribe (swap 6.9 s): 17.3 s |
| too-big (example) | `compose plan` | refused: one stage needs 20.7 GB plus 2 GB against a 19.1 GB working set |
