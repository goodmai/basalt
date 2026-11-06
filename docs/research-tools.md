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
| `model_tensor` | Mean, std, min, max, L2 and the p1/p5/p50/p95/p99 of \|x\| for one tensor (dequantized), plus up to `maxValues` values of a slice. With `"spectrum": true` also the spectrum of the matrix, or of one expert of a 3-D stack picked by the slice: Marchenko–Pastur noise variance, bulk edges and eigenvalue mass above the bulk, Gavish–Donoho rank, power-law tail exponent `alpha` and the seconds it took (Gram WᵀW of the smaller side on the GPU, `eigvalsh` on the CPU; smaller side ≤ 8192; only with `serve --research-tools`, since it holds the model for seconds) | `{"name": "model.layers.3.input_layernorm.weight", "slice": [[0, 64]], "maxValues": 64}`, `{"name": "model.layers.3.mlp.down_proj.weight", "maxValues": 0, "spectrum": true}` |
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
| `adapter_apply` | Applies a LoRA adapter to the live model: an mlx-lm adapter directory (`adapter_config.json` + `adapters.safetensors`), or a rank-1 update `scale × (x·input) output` of one linear layer, quantized layers included. Marks the model modified like a weight delta; `overlay_undo` restores the layers bit for bit. With `gate`, the gate runs after applying: on a pass the adapter stays without expiry, on a failure it is undone and the report comes back with `overlay: null`. Without `gate` it expires after 30 minutes. | `{"module": "model.layers.10.mlp.down_proj", "input": [0.01, …], "output": [0.02, …], "scale": 1, "gate": {"heldOut": ["The capital of France is"], "maxKL": 0.5}}` |
| `gate_check` | The quality gate on the model as it is now against the pristine model (every live adapter and weight delta lifted for the baseline): each check with its value, limit and verdict, and the held-out comparison rows | `{"heldOut": ["The capital of France is", "2 + 2 ="], "maxKL": 0.5, "maxPerplexityRise": 0.1, "target": [{"prompt": "17 + 25 =", "expected": "42"}]}` |
| `tensor_pack` | Re-packs the weight of one linear layer in memory: `affine` (`bits` 2, 3, 4, 5, 6 or 8, `groupSize` 32, 64 or 128), `mxfp4`, or `ternary` (absmean scale·{-1, 0, 1} per group of 64). A quantized layer is dequantized first. The layer is swapped live, like a weight delta (see below) | `{"tensor": "model.layers.10.mlp.down_proj.weight", "format": "affine", "bits": 2, "groupSize": 64}` |
| `tensor_unpack` | Replaces one quantized linear layer by a dense `float16` or `bfloat16` layer (default `bfloat16`) holding its dequantized weight, live | `{"tensor": "model.layers.10.mlp.down_proj.weight", "format": "bfloat16"}` |
| `train_experts` | Self-training (#332 mode 3, long-running): collects answers to checkable tasks, keeps only the verified ones, trains a LoRA, saves it, reports loss and held-out accuracy before/after, then applies it through the quality gate. See [Self-training](#self-training-train_experts) | `{"count": 200, "seed": 1, "layers": [20, 21, 22, 23], "iterations": 200, "heldOut": ["Summarise the plot of Hamlet in two sentences."]}` |
| `steer_search` | Contrastive steering search: per layer the direction mean(positive) − mean(negative) of the last-position residual, its norm and class separation; the best layer's unit direction; the heads whose ablation shrinks the gap most; KL and projection shift when the held-out prompts are steered at each scale. Leaves no overlay behind. | `{"positive": ["I love this", "What a great day"], "negative": ["I hate this", "What an awful day"], "heldOut": ["The movie was"], "topHeads": 4, "scales": [2, 4, 8]}` |
| `steer_commit` | Turns a direction into a rank-1 adapter on the layer's output projection and applies it through the quality gate on `heldOut`. Changes the live weights like `weights_delta` (undoable, TTL unless accepted, never on disk). | `{"layer": 12, "direction": [0.01, -0.02], "scale": 4, "positive": ["I love this"], "heldOut": ["The movie was"]}` |

A slice is one half-open `[start, end)` pair per leading dimension; dimensions left out are taken whole.

## Safety model

- **Local only.** The tools are MCP tools served over stdio to the client that launched the server; they have no REST route.
- **Off by default.** Without `--research-tools` the experiment tools are neither listed nor accepted.
- **Overlays, TTL, undo, journal.** Every experiment is an overlay with an id, a description and a 30-minute TTL, except an adapter that passed its quality gate (`adapter_apply` with `gate`, `steer_commit`, `train_experts`), which stays until `overlay_undo`. `overlay_undo` removes any overlay early; on expiry it is undone automatically. Adds, undos and expiries are appended to `~/.basalt/experiments/<model fingerprint>.jsonl` (directory 0700, file 0600). Loading another model drops every overlay.
- **Model files on disk are never written.** Deltas, packs and adapters exist only in memory; restarting the server or reloading the model discards them. The only files written are the journal and the adapter `train_experts` saves to a new directory.
- **Interventions stay in research calls.** Steering and ablations apply to `experiment_compare` and `trace_run`, never to user requests.
- **Weight deltas are visible and isolated from the caches.** A delta changes the weights every request uses, so while one is active:
  - every response carries `modelModified: true` (also a response whose generation overlapped an apply or undo);
  - the telemetry snapshot (MCP `status`, `GET /api/v1/stats`) lists the active deltas under `activeDeltas`;
  - prompt-cache reuse is off, including restores from the disk cache, because the cached KV was computed with the old weights;
  - the in-memory prompt caches are dropped on apply and on undo, and a request that was running across either change does not store its cache.
- **Validated at the boundary.** The tensor must exist, the slice must be in range and non-empty (at most 16M elements), there must be one value per element or one to broadcast, every value must be finite and `|scale| ≤ 10`. One active delta per tensor.
- **Float tensors only.** Deltas write norms, biases and unquantized embeddings or projections (float16, bfloat16, float32). A tensor of a quantized layer (its packed weight and its scales and biases alike) is refused: a quantized write path is out of scope.

## Quality gate and adapters (#332)

`gate_check` and the `gate` of `adapter_apply` compare the current model with the pristine one, inside one exclusive model call that lifts every live adapter and weight delta for the baseline and puts them back. The checks:

- **`probe perplexity`**: the load probe's perplexity on its fixed text must stay under the family ceiling, or, when the pristine model is already above it, rise no more than `maxPerplexityRise` over the pristine value.
- **`probe known answer …`**: each load-probe known answer the pristine model gets right must still be right (value 1 or 0, limit the pristine result).
- **`held-out mean KL`** and **`held-out perplexity rise`**: on each `heldOut` prompt, `experiment_compare`'s machinery runs with the pristine model as the baseline; the mean KL divergence of the next-token distributions along the baseline text must be at most `maxKL` (default 0.5) and the mean relative perplexity rise of that text at most `maxPerplexityRise` (default 0.1).
- **`target …`**: each target prompt's greedy answer (chat template first, then the raw prompt, 16 tokens) must contain `expected`, case-insensitively.

Non-finite values are reported as the largest finite float and fail. Adapters:

- A directory adapter is loaded with mlx-swift-lm's LoRA support; one directory adapter at a time, and its tensors must match the layers' shapes and the configured rank.
- A rank-1 adapter needs one `input` value per input feature and one `output` value per output feature of the layer; the layer must be a plain `Linear` or `QuantizedLinear` without an adapter.
- Adapters wrap only plain `Linear` and `QuantizedLinear` layers, so undo puts back the same weights and the logits match the pristine model bit for bit. Nothing is written to disk.

## How this leads to learning loops (#328 layer 3)

These tools are the measuring instrument and the sandbox for a later layer: `distill.collect` gathers prompts and reference answers, `adapter.train` / `apply` / `evaluate` train and try a small adapter, and `weights.propose` takes an externally proposed correction. Each candidate first runs as an overlay or a delta here, is compared with `experiment_compare`, and is accepted only if the load-time probe, the reference logits, the parity runner and the feature matrix still pass. Nothing in this layer persists a change; the learning loop decides what, if anything, becomes permanent.

## Tensor pack and unpack (#332)

`tensor_pack` and `tensor_unpack` both change one linear layer in place: they read its weight (dequantizing a quantized layer), convert it, and swap the whole module (`Linear` ↔ `QuantizedLinear`), so the bytes really change. The answer reports what the conversion cost (illustrative values):

```json
{"overlay": {"id": 7, "description": "tensor_pack model.layers.10.mlp.down_proj.weight: 4-bit affine g64 → 2-bit affine g64", "createdAt": "…", "expiresAt": "…"},
 "bytesBefore": 9437184, "bytesAfter": 5505024, "rmse": 0.0041, "maxAbsError": 0.031, "relativeError": 0.29}
```

`rmse`, `maxAbsError` and `relativeError` (‖W − Ŵ‖ / ‖W‖) compare the new layer's dequantized weight with the values the old layer held. Unpacking to the scales' own float type is exact (all three are 0).

The pack is an overlay like a weight delta: the model is marked modified, prompt-cache reuse is off, `overlay_undo` (or the 30-minute TTL) swaps the original layer back bit for bit, and nothing is written to disk. Only plain `Linear` and `QuantizedLinear` layers are packed; MoE expert stacks, embeddings, LoRA-wrapped layers and a layer that carries a delta or another pack are refused.

`experiment_compare` runs on the live weights, so the effect of a pack is the difference between a run before and a run after it:

1. `experiment_compare` `{"prompts": ["2 + 2 =", "Name a prime above 100."], "maxTokens": 32}`: note `baselineText` and `baselinePerplexity`.
2. `tensor_pack` `{"tensor": "model.layers.10.mlp.down_proj.weight", "format": "ternary"}`: note the overlay id and the error.
3. `experiment_compare` with the same arguments: the change of `baselineText` and `baselinePerplexity` from step 1 is the effect of the pack on the logits.
4. `overlay_undo` `{"id": 7}`, then step 1 again gives step 1's answer bit for bit.
## Introspection (`introspect`, #332 mode 1)

With `--research-tools`, `introspect` lets the served model study itself. Given a goal, the server runs the model through the normal generation path with the X-ray tools offered as tool definitions, executes each call it makes through the same gated research path as the MCP tools, feeds the result back and asks again.

| Argument | Meaning |
|---|---|
| `goal` (required) | What to find out, e.g. `"find which heads attend to the last tool call"` |
| `maxSteps` | Model turns before the loop stops, 1–32 (default 8) |
| `allowInterventions` | Also offer `intervene`, `experiment_compare` and `overlay_undo` (default false; the read-only tools are always offered) |
| `resultBudget` | Characters of each tool result fed back to the model, 200–20000 (default 2000); longer results are cut and marked |

- **One call per turn.** The loop stops at an answer without a tool call, after `maxSteps`, or when the model repeats a call it already made (same tool and arguments, key order ignored); `stop` in the report says which.
- **Hypotheses.** The model is asked to write each hypothesis as `Hypothesis:` / `Test:` / `Outcome:` lines. They are parsed leniently (markdown, numbering, `Result:` for `Outcome:`), and a restated hypothesis replaces the earlier one, keeping its `Test:` or `Outcome:` when the restatement leaves that line out.
- **Errors go back to the model.** A malformed or failing tool call becomes an `error: …` result the model sees; a failed generation or a cancellation ends the loop.
- **Interventions never outlive the loop.** Every overlay created during the loop is undone when it ends, also on an error or a cancellation; `undone` lists them. If the server refuses the cleanup (busy), `cleanupFailure` says so and those overlays expire at their TTL. `overlay_undo` refuses overlays that existed before the loop.
- **A failed turn keeps the report.** If a generation fails mid-loop, the steps so far come back with `stop: "failed"` and the error in `failure`.
- **Journal.** Each step appends one `introspect` line (model, step, tool, arguments and the start of the result) to `~/.basalt/experiments/introspect.jsonl`.

The report holds `goal`, `steps` (the model's text, and for a call its `tool`, `arguments` and truncated `result`), `hypotheses` (`statement`, `test`, `outcome`), `answer`, `stop`, `failure`, `undone` and `cleanupFailure`.

```json
{"goal": "find which heads attend to the last tool call", "maxSteps": 6, "allowInterventions": true}
```
## Self-training (`train_experts`)

One call runs the whole loop at background priority through the generation gate, holding the model for the training itself:

1. **Tasks.** `count` + `seed` generate integer `+`, `-`, `*` problems with known answers (the same seed gives the same problems); `file` reads JSONL lines `{"prompt": "...", "answer": "..."}` instead. A fifth of the tasks (at least one) is held out and never trained on.
2. **Collect.** The model answers every task greedily through the chat template (thinking off, 64 tokens). With `teacher` (base URL of an OpenAI-compatible server; `teacherModel` names its model, `BASALT_TEACHER_API_KEY` is sent as a bearer token only when the teacher's scheme, host and port match `BASALT_TEACHER_URL`, so a client cannot send the key elsewhere) the teacher answers the training tasks too.
3. **Verify.** An answer is kept only if its final integer (commas ignored) equals a numeric known answer, or its last line, case and surrounding punctuation ignored, equals a text answer (or ends in `: <answer>`). Wrong, empty and malformed answers are dropped. **Nothing unverified is ever learned:** if no answer verifies, the report has `verified: 0`, no adapter is trained, nothing is written and the model is untouched.
4. **Train.** A LoRA (`rank`, Adam at `learningRate`, `iterations`) is trained with the fork's `LoRATrain` on the verified prompt + answer pairs rendered as the model saw them, on the attention and dense or shared-expert projections of `layers` (default all). Per-expert LoRA is not available — the fork's `SwitchLinear` expert layers take no adapter — so `experts` is refused with an error. A model that already carries an adapter is refused too. Cancelling the call stops training at the next iteration and leaves the model unchanged.
5. **Save.** `adapter_config.json` + `adapters.safetensors` (the mlx-lm adapter format) go to `output` or `~/.basalt/adapters/<model fingerprint>/<timestamp>` (0700). `output` must not exist yet, and one inside the model directory is refused: nothing is overwritten and nothing is written next to the source weights.
6. **Measure and apply.** Held-out accuracy is measured with the adapter in place, the training layers are removed again, and the adapter is applied through `applyAdapter` with the quality gate built from `heldOut`, `maxKL` and `maxPerplexityRise`: it becomes a marked overlay without expiry that `overlay_undo` removes, or is refused by the gate.

The report lists `collected` (answers gathered for training), `verified`, `losses` (mean training loss per reporting step, at most 20), `accuracyBefore` / `accuracyAfter` on the held-out tasks, `adapterDirectory` and the gate's `verdict`.

## Contrastive steering ("meditation", #332 mode 2)

This is directed search for activation directions, not self-awareness. Nothing here means the model reflects on itself: basalt runs two sets of prompts through the model, subtracts the averages of the hidden states, and measures what the difference does.

`steer_search` runs every positive and negative prompt through one probed forward and keeps the residual after each layer at the last position.

- **Direction per layer:** mean(positive) − mean(negative). `norm` is its length; `separation` is the gap between the class means of the projections onto the unit direction (equal to `norm`) divided by the pooled standard deviation of those projections. The best layer has the highest separation, and `direction` is its unit vector (hidden size).
- **Heads:** each attention head of the best layer and of the layer before it is ablated in turn (the `ablate_head` intervention), and `effect` is how much the positive-minus-negative projection gap at the best layer shrinks. Heads of later layers cannot change that residual and are not tried. At most `topHeads` are reported, largest effect first. This is one forward per head and prompt, so keep the prompt sets small on large models, or pass `topHeads: 0`.
- **Trials:** for each scale the held-out prompts are steered by `direction × scale` after the best layer (the `steer` intervention, research-only). `klDivergence` is the mean KL of the next-token distribution against the unsteered run; `projectionShift` is the mean change of the projection onto `direction` read after the last layer. At the steered layer that shift is the scale by construction, so the last layer shows how much of the push survives.
- No overlay is registered: the ablations and steering exist only inside the call.

`steer_commit` makes a direction that helped permanent for the session, through the gate. It picks the layer's dense projection that writes into the residual (`mlp.down_proj`, else `feed_forward.w2`, else `self_attn.o_proj`; a MoE layer without a dense one is refused), measures that module's mean input `m` at the last position over the `positive` prompts, and calls the adapter path with `.rankOne(module: <full module path>, input: m / (m·m), output: direction, scale: scale)`, so the update is ΔW = scale · direction ⊗ m / (m·m).

The approximation, stated plainly:

- The update adds `scale · (x·m / m·m) · direction` to the projection's output for every token. That equals the steering vector exactly only where the module input `x` has the same projection on `m` as the positive mean; other tokens get more, less or the opposite. `steer` adds the same vector to every position.
- It is exact in the residual only when the projection's output is added straight to the residual stream (pre-norm blocks such as Qwen and LFM). Where a norm follows the projection (Gemma's post-attention and post-feedforward norms) the vector is rescaled and only its direction is roughly kept. When the fallback `self_attn.o_proj` is used, the vector also passes through the layer's MLP before it reaches the residual after the layer.
- The adapter is accepted only if the gate passes (`maxKL`, `maxPerplexityRise` on `heldOut`); run `steer_search` trials first and commit a scale whose KL was acceptable.

