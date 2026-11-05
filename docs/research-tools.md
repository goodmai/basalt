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
| `adapter_apply` | Applies a LoRA adapter to the live model: an mlx-lm adapter directory (`adapter_config.json` + `adapters.safetensors`), or a rank-1 update `scale × (x·input) output` of one linear layer, quantized layers included. Marks the model modified like a weight delta; `overlay_undo` restores the layers bit for bit. With `gate`, the gate runs after applying: on a pass the adapter stays without expiry, on a failure it is undone and the report comes back with `overlay: null`. Without `gate` it expires after 30 minutes. | `{"module": "model.layers.10.mlp.down_proj", "input": [0.01, …], "output": [0.02, …], "scale": 1, "gate": {"heldOut": ["The capital of France is"], "maxKL": 0.5}}` |
| `gate_check` | The quality gate on the model as it is now against the pristine model (every live adapter and weight delta lifted for the baseline): each check with its value, limit and verdict, and the held-out comparison rows | `{"heldOut": ["The capital of France is", "2 + 2 ="], "maxKL": 0.5, "maxPerplexityRise": 0.1, "target": [{"prompt": "17 + 25 =", "expected": "42"}]}` |
| `tensor_pack` | Re-packs the weight of one linear layer in memory: `affine` (`bits` 2, 3, 4, 5, 6 or 8, `groupSize` 32, 64 or 128), `mxfp4`, or `ternary` (absmean scale·{-1, 0, 1} per group of 64). A quantized layer is dequantized first. The layer is swapped live, like a weight delta (see below) | `{"tensor": "model.layers.10.mlp.down_proj.weight", "format": "affine", "bits": 2, "groupSize": 64}` |
| `tensor_unpack` | Replaces one quantized linear layer by a dense `float16` or `bfloat16` layer (default `bfloat16`) holding its dequantized weight, live | `{"tensor": "model.layers.10.mlp.down_proj.weight", "format": "bfloat16"}` |

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
