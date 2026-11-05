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
| `steer_search` | Contrastive steering search: per layer the direction mean(positive) − mean(negative) of the last-position residual, its norm and class separation; the best layer's unit direction; the heads whose ablation shrinks the gap most; KL and projection shift when the held-out prompts are steered at each scale. Leaves no overlay behind. | `{"positive": ["I love this", "What a great day"], "negative": ["I hate this", "What an awful day"], "heldOut": ["The movie was"], "topHeads": 4, "scales": [2, 4, 8]}` |
| `steer_commit` | Turns a direction into a rank-1 adapter on the layer's output projection and applies it through the quality gate on `heldOut`. Changes the live weights like `weights_delta` (undoable, TTL unless accepted, never on disk). | `{"layer": 12, "direction": [0.01, -0.02], "scale": 4, "positive": ["I love this"], "heldOut": ["The movie was"]}` |

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

