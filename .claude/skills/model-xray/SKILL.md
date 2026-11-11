---
name: model-xray
description: Use basalt's Model X-ray MCP tools (model_describe, model_tensors, model_tensor, kv_inspect, trace_run) and, with `serve --research-tools`, the experiment tools (intervene, weights_delta, overlays_list, overlay_undo, experiment_compare, trace_tokens, activation_contrast) to look inside the running model instead of guessing; also how to reach basalt's own MCP server from a coding session. Use when a model answers wrongly or degenerately and you want to know where (which layer, head, expert, norm), when a prompt-cache or KV question needs the actual tree, when you need a layer's anatomy or a tensor's statistics for a design or bug report, or when you want to test a hypothesis about a head, expert or norm with a reversible experiment.
---

# Model X-ray

Reference for every tool, argument and the safety model: `docs/research-tools.md`.

## Reaching the tools

The tools belong to basalt's own MCP server (`basalt serve --mcp`, JSON-RPC 2.0 on stdin and stdout, logs on stderr). It is not registered in `.mcp.json`, so a coding session does not see them as `mcp__…` tools. Two ways in:

- **One-shot from the shell** (checked 2026-11-11 on gemma-4-12B). The server answers each line and exits when stdin closes:

  ```bash
  printf '%s\n' \
    '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
    '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"model_describe","arguments":{}}}' \
    | .build/release/Basalt serve --mcp --model mlx-community/gemma-4-12B-it-qat-4bit 2>mcp.err
  ```

  Each answer is one JSON line; the tool's payload is a JSON string in `result.content[0].text`. Put every call for one question in the same pipe: the model loads once.
- **A client that keeps the server open** (Claude Desktop, an agent): configure it as in `docs/AI_SETUP.md`.

Rules for both:
- The server loads weights, so it is a real-model run: take the GPU lock and use a release build (`basalt-model-runs`).
- Add `--research-tools` only for the experiment tools. `tools/list` shows what the running server accepts.
- Read stderr when a call returns nothing: load errors go there, not to stdout.

Other tools on the same server, for when the question is about serving and not about the model's insides:

| Tool | Use |
|---|---|
| `status`, `basalt_status`, `check_hardware` | readiness, telemetry, what this Mac can hold |
| `inspect_model`, `switch_model` | a model's files and fit; swap the served model |
| `basalt_generate`, `basalt_pipeline`, `basalt_ocr`, `basalt_transcribe`, `basalt_systemone_decide` | the serving paths themselves, same code as REST |
| `harness_validate`, `harness_optimize`, `harness_library_query` | agent harness files |

## When to reach for it

- **Before guessing about the architecture.** `model_describe` gives layer kinds, heads, KV heads, windows, experts and the KV-cache class per layer. Use it instead of reading config.json by hand when a cache, quantization or routing decision depends on the layout.
- **A wrong or degenerate answer.** `trace_run` on the failing prompt: the logit lens shows the layer where the right token falls out of the top-k, attention shows what the position reads, router choices show which experts ran. Name the layer in the issue.
- **A suspicious tensor.** `model_tensors` with a filter, then `model_tensor` for statistics. An outlier max, a zero std or a NaN-free but huge L2 points at a bad conversion or quantization.
- **A prompt-cache question** (no reuse, memory growth, evictions): `kv_inspect` shows namespaces, tokens, bytes, snapshots and pins as the engine holds them.
- **Testing a hypothesis** (needs `--research-tools`): ablate a head or an expert, or steer a layer, with `intervene`, then `experiment_compare` on a few prompts against the baseline. Interventions affect research calls only.
- **A behaviour over a whole answer** (refusal, a style switch; needs `--research-tools`): `trace_tokens` to see which heads fire and which experts route on every prompt and generated token, `activation_contrast` on a set that shows it against one that does not, then `intervene` with `when` (a projection or expert trigger) to ablate the top components only on the tokens where the behaviour starts, and `experiment_compare` against the static ablation. Procedure: `docs/research-tools.md`, "Per-token trace and conditional ablation".

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

## Related skills

All skills live in `.claude/skills/<name>/SKILL.md` (mirrored at `.agents/skills/`). Start from `swiftcoder` when unsure.

| Skill | Use it for |
|---|---|
| `swiftcoder` | the role and entry point: which skill to load, in what order, and the gates |
| `ponytail` | scope and the smallest complete change |
| `ponytail-review` | review of a change for correctness, safety, load, tests and need |
| `swift-64-architecture` | targets, boundaries, isolation domains, value core and actor shell |
| `swift-64-optimization-coding` | writing Swift 6.4 code: ownership, typed throws, size limits, Metal kernels |
| `swift-64-pr-review` | the 5-gate review of a Swift diff |
| `swift-concurrency-pro` | async/await, actors, tasks, streams and cancellation in depth |
| `swift-race-review` | races and deadlocks the compiler cannot see; MLX and Metal threading |
| `swift-testing-pro` | Swift Testing in depth: structure, async tests, new features |
| `xcode-mcp` | checking an API against Apple's documentation; Xcode workspace tools |
| `basalt-model-runs` | runs that load model weights: benchmarks, ppl, memory, parity, GPU lock |
