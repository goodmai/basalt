# Model X-ray and research tools (#328)

Basalt's MCP server can show what the loaded model is doing and run small, reversible experiments on it. The read-only X-ray tools are always listed. The experiment tools are listed and accepted only when the server starts with `basalt serve --mcp --research-tools`.

Every call runs through the generation gate at background priority, so user requests go first. Responses are bounded: statistics and small slices, never a whole tensor or a full attention matrix.

## Tools

The examples are the `arguments` object of an MCP `tools/call` request.

### Read-only X-ray

| Tool | What it returns | Example arguments |
|---|---|---|
| `model_describe` | Model type, architecture, per-layer kind (full attention, sliding window, linear attention, mamba, conv), heads, KV heads, head size, window, experts, KV-cache class, quantization, parameter count and bytes | `{}` |
| `model_tensors` | Parameter names with shape, dtype and whether they are quantized; `filter` is a substring of the name | `{"filter": "layers.3.mlp"}` |
| `model_tensor` | Mean, std, min, max, L2 and the p1/p5/p50/p95/p99 of \|x\| for one tensor (dequantized), plus up to `maxValues` values of a slice | `{"name": "model.layers.3.input_layernorm.weight", "slice": [[0, 64]], "maxValues": 64}` |
| `kv_inspect` | The prompt-cache tree per namespace: nodes, leaves, tokens, bytes, recurrent snapshots, pins, last use; budget and evictions | `{"namespace": "claude-code"}` |
| `trace_run` | One capture run of a prompt: per-layer hidden-state norm, logit lens top-k, per-head top attended positions and entropy at one position, MoE router choices | `{"prompt": "The capital of France is", "layers": [0, 12, 23], "topK": 5}` |

### Experiments (`--research-tools`)

| Tool | What it does | Example arguments |
|---|---|---|
| `intervene` | Adds a steering vector to the residual stream after a layer, or zeroes one attention head or one expert. Applies to research calls (`experiment_compare`, `trace_run`) only. | `{"kind": "ablate_head", "layer": 10, "head": 3}` |
| `weights_delta` | Adds `values × scale` to a slice of one float tensor in memory (one value is broadcast). Changes the live model, so user requests see it too (see below). | `{"tensor": "model.layers.10.post_attention_layernorm.weight", "slice": [[0, 128]], "values": [0.05], "scale": 1}` |
| `overlays_list` | Active overlays: id, description, creation and expiry time | `{}` |
| `overlay_undo` | Undoes one overlay by id; a weight delta is restored bit for bit | `{"id": 4}` |
| `experiment_compare` | Greedy before/after on a prompt set with the given overlays: both texts, KL divergence of the logits, both perplexities | `{"prompts": ["2 + 2 =", "Name a prime above 100."], "overlays": [3, 4], "maxTokens": 32}` |
| `train_experts` | Self-training (#332 mode 3, long-running): collects answers to checkable tasks, keeps only the verified ones, trains a LoRA, saves it, reports loss and held-out accuracy before/after, then applies it through the quality gate. See [Self-training](#self-training-train_experts) | `{"count": 200, "seed": 1, "layers": [20, 21, 22, 23], "iterations": 200, "heldOut": ["Summarise the plot of Hamlet in two sentences."]}` |

A slice is one half-open `[start, end)` pair per leading dimension; dimensions left out are taken whole.

## Safety model

- **Local only.** The tools are MCP tools served over stdio to the client that launched the server; they have no REST route.
- **Off by default.** Without `--research-tools` the experiment tools are neither listed nor accepted.
- **Overlays, TTL, undo, journal.** Every experiment is an overlay with an id, a description and a 30-minute TTL. `overlay_undo` removes it early; on expiry it is undone automatically. Adds, undos and expiries are appended to `~/.basalt/experiments/<model fingerprint>.jsonl` (directory 0700, file 0600). Loading another model drops every overlay.
- **Files on disk are never written.** A delta exists only in memory; restarting the server or reloading the model discards it.
- **Interventions stay in research calls.** Steering and ablations apply to `experiment_compare` and `trace_run`, never to user requests.
- **Weight deltas are visible and isolated from the caches.** A delta changes the weights every request uses, so while one is active:
  - every response carries `modelModified: true` (also a response whose generation overlapped an apply or undo);
  - the telemetry snapshot (MCP `status`, `GET /api/v1/stats`) lists the active deltas under `activeDeltas`;
  - prompt-cache reuse is off, including restores from the disk cache, because the cached KV was computed with the old weights;
  - the in-memory prompt caches are dropped on apply and on undo, and a request that was running across either change does not store its cache.
- **Validated at the boundary.** The tensor must exist, the slice must be in range and non-empty (at most 16M elements), there must be one value per element or one to broadcast, every value must be finite and `|scale| ≤ 10`. One active delta per tensor.
- **Float tensors only.** Deltas write norms, biases and unquantized embeddings or projections (float16, bfloat16, float32). A tensor of a quantized layer (its packed weight and its scales and biases alike) is refused: a quantized write path is out of scope.

## How this leads to learning loops (#328 layer 3)

These tools are the measuring instrument and the sandbox for a later layer: `distill.collect` gathers prompts and reference answers, `adapter.train` / `apply` / `evaluate` train and try a small adapter, and `weights.propose` takes an externally proposed correction. Each candidate first runs as an overlay or a delta here, is compared with `experiment_compare`, and is accepted only if the load-time probe, the reference logits, the parity runner and the feature matrix still pass. Nothing in this layer persists a change; the learning loop decides what, if anything, becomes permanent.

## Self-training (`train_experts`)

One call runs the whole loop at background priority through the generation gate, holding the model for the training itself:

1. **Tasks.** `count` + `seed` generate integer `+`, `-`, `*` problems with known answers (the same seed gives the same problems); `file` reads JSONL lines `{"prompt": "...", "answer": "..."}` instead. A fifth of the tasks (at least one) is held out and never trained on.
2. **Collect.** The model answers every task greedily through the chat template (thinking off, 64 tokens). With `teacher` (base URL of an OpenAI-compatible server; `teacherModel` names its model, `BASALT_TEACHER_API_KEY` is sent as a bearer token if set) the teacher answers the training tasks too.
3. **Verify.** An answer is kept only if its final integer (commas ignored) equals a numeric known answer, or its last line, case and surrounding punctuation ignored, equals a text answer (or ends in `: <answer>`). Wrong, empty and malformed answers are dropped. **Nothing unverified is ever learned:** if no answer verifies, the report has `verified: 0`, no adapter is trained, nothing is written and the model is untouched.
4. **Train.** A LoRA (`rank`, Adam at `learningRate`, `iterations`) is trained with the fork's `LoRATrain` on the verified prompt + answer pairs rendered as the model saw them, on the attention and dense or shared-expert projections of `layers` (default all). Per-expert LoRA is not available — the fork's `SwitchLinear` expert layers take no adapter — so `experts` is refused with an error. A model that already carries an adapter is refused too. Cancelling the call stops training at the next iteration and leaves the model unchanged.
5. **Save.** `adapter_config.json` + `adapters.safetensors` (the mlx-lm adapter format) go to `output` or `~/.basalt/adapters/<model fingerprint>/<timestamp>` (0700). An `output` inside the model directory is refused: nothing is ever written next to the source weights.
6. **Measure and apply.** Held-out accuracy is measured with the adapter in place, the training layers are removed again, and the adapter is applied through `applyAdapter` with the quality gate built from `heldOut`, `maxKL` and `maxPerplexityRise`: it becomes a marked, undoable overlay with the usual TTL, or is refused by the gate.

The report lists `collected` (answers gathered for training), `verified`, `losses` (mean training loss per reporting step, at most 20), `accuracyBefore` / `accuracyAfter` on the held-out tasks, `adapterDirectory` and the gate's `verdict`.
