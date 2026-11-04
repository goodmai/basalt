---
name: model-xray
description: Use basalt's Model X-ray MCP tools (model_describe, model_tensors, model_tensor, kv_inspect, trace_run) and, with `serve --research-tools`, the experiment tools (intervene, weights_delta, overlays_list, overlay_undo, experiment_compare) to look inside the running model instead of guessing. Use when a model answers wrongly or degenerately and you want to know where (which layer, head, expert, norm), when a prompt-cache or KV question needs the actual tree, when you need a layer's anatomy or a tensor's statistics for a design or bug report, or when you want to test a hypothesis about a head, expert or norm with a reversible experiment.
---

# Model X-ray

Reference for every tool, argument and the safety model: `docs/research-tools.md`.

## When to reach for it

- **Before guessing about the architecture.** `model_describe` gives layer kinds, heads, KV heads, windows, experts and the KV-cache class per layer. Use it instead of reading config.json by hand when a cache, quantization or routing decision depends on the layout.
- **A wrong or degenerate answer.** `trace_run` on the failing prompt: the logit lens shows the layer where the right token falls out of the top-k, attention shows what the position reads, router choices show which experts ran. Name the layer in the issue.
- **A suspicious tensor.** `model_tensors` with a filter, then `model_tensor` for statistics. An outlier max, a zero std or a NaN-free but huge L2 points at a bad conversion or quantization.
- **A prompt-cache question** (no reuse, memory growth, evictions): `kv_inspect` shows namespaces, tokens, bytes, snapshots and pins as the engine holds them.
- **Testing a hypothesis** (needs `--research-tools`): ablate a head or an expert, or steer a layer, with `intervene`, then `experiment_compare` on a few prompts against the baseline. Interventions affect research calls only.

## Weight deltas: use sparingly

`weights_delta` changes the live weights, so user requests on the same server see it. Only use it on a server you started yourself for the experiment.

- Float tensors only (norms, biases, unquantized embeddings); a quantized layer is refused.
- While a delta is active, responses carry `modelModified: true`, prompt-cache reuse is off and telemetry lists it. Undo it with `overlay_undo` as soon as the comparison is done; otherwise it expires after 30 minutes.
- Undo restores the tensor bit for bit. Nothing is written to disk.

## Rules

- Prefer the read-only tools. Experiment only when a read-only result leaves two explanations open.
- Ask for small slices and a small `topK`; responses are bounded anyway, and large ones waste context.
- Quote tool results (layer, head, values) in the issue or PR, not impressions. A change justified by X-ray still needs the usual parity run (`basalt-model-runs`).
- This layer persists nothing. Turning a finding into a permanent change goes through the learning-loop layer (#328 layer 3) and its quality gates.
