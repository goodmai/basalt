---
name: model-surgery
description: Perform surgical, targeted weight edits on MLX models in basalt — live MCP overlays (weights_delta, rank-1 adapter_apply, steer_search + steer_commit, tensor_pack/tensor_unpack, gate_check, overlay_undo) and Swift-level MLXNN.Module weight/code mutations (abliteration, Procrustes alignment, PV-Tuning, mixed-bit repacking). Use whenever you modify specific layers, heads, projections, norms or quantized codes instead of full fine-tuning, or when turning an X-ray localization into a gated weight change.
---

# Model Surgery

Model surgery changes the smallest set of parameters that fixes or steers a localized behaviour while keeping the rest of the model bit-identical or within a measured KL and perplexity gate. Never edit weights before localizing the target (`model-xray`), and never keep a surgical edit that fails its quality gate.

Reference for the live MCP tools and safety invariants: `docs/research-tools.md` and `Sources/BasaltInference/MLXInferenceEngine+{XRayDeltas,LabGate,LabPack,LabSteer}.swift`.

## 1. Pick the smallest scalpel

| Target parameter | Live MCP tool (`serve --mcp --research-tools`) | Swift engine / offline path | Scope & constraints |
|---|---|---|---|
| Activation hypothesis (before touching weights) | `intervene` (`ablate_head`, `ablate_expert`, `steer`, optional `when`) + `experiment_compare` | `ConditionalEdit` / `ForwardProbe` | Research calls only; zero effect on live user traffic. Always test here first. |
| Unquantized float slice (norms, biases, dense embeddings) | `weights_delta` (`tensor`, `slice`, `values`, `scale`) | `MLXInferenceEngine.applyDelta` | Float tensors only (`float16`, `bfloat16`, `float32`); `\|scale\| <= 10`, `<= 16M` elements; one active delta per tensor. |
| Single linear projection (`down_proj`, `w2`, `o_proj`, `q/k/v`), dense or quantized | `adapter_apply` (`module`, `input`, `output`, `scale`, `gate`) | `MLXInferenceEngine.applyAdapter(.rankOne(...))` | Rank-1 update `ΔW = scale · output ⊗ input`; wraps `Linear` and `QuantizedLinear` without unpacking; undo restores bit for bit. |
| Contrastive residual direction (`mean(pos) - mean(neg)`) | `steer_search` → `steer_commit` (`layer`, `direction`, `scale`, `positive`, `heldOut`) | `ContrastSearch` + `commitSteering` | Measures module mean input `m` on `positive`, commits rank-1 `input = m / (m·m)` on `mlp.down_proj` / `feed_forward.w2` / `self_attn.o_proj` through `gate_check`. |
| Single-layer precision or codebook swap | `tensor_pack` / `tensor_unpack` | `TensorPacker.pack` / `WeightConverter` | Swaps one `Linear` ↔ `QuantizedLinear` (`affine` 2/3/4/5/6/8-bit, `nf4:32\|64\|128`, `mxfp4`, `ternary`, `bfloat16`, `float16`); returns `rmse`, `maxAbsError`, `relativeError`. |
| Discrete quantized codes / group scales / orthogonal subspace | Offline runner or lab pass | `Module.update` + `MLXInferenceEngine.settle` | Used for abliteration, orthogonal Procrustes alignment, mixed-bit allocation and PV-Tuning; requires real-model parity and KL/PPL gates (`basalt-model-runs`). |

## 2. Surgical workflow (Diagnose → Test → Cut → Gate → Undo or Persist)

1. **Localize (`model-xray`).** Run `model_describe` for layer kinds (pre-norm Qwen/LFM vs post-norm Gemma, dense vs MoE, linear vs full attention), then `trace_run`, `trace_tokens` + `activation_contrast`, or `steer_search` to name the exact layer, head, expert or residual direction. Inspect suspect weights with `model_tensor` (`"spectrum": true` when checking Gavish–Donoho rank or tail anomalies).
2. **Verify causally in activations first.** Apply `intervene` (static or conditional `when`) and measure with `experiment_compare` on positive, negative and held-out prompts. If the activation edit does not move the target behaviour cleanly, a weight edit at that site will not either.
3. **Apply the surgical edit through the gate.**
   - **Pre-norm vs post-norm geometry:** On pre-norm blocks (Qwen, LFM), `ΔW` on `mlp.down_proj` writes straight into the residual stream. On post-norm blocks (Gemma 4), `post_feedforward_layernorm` rescales the output, preserving direction only approximately — verify post-layer shift with `steer_search` trials before committing.
   - **Gate every edit:** Pass `gate` (`heldOut`, `maxKL`, `maxPerplexityRise`, `target`) to `adapter_apply` / `steer_commit`, or run `gate_check` immediately after `weights_delta` / `tensor_pack`. `gate_check` lifts all live deltas and adapters inside one exclusive model lock to compare against the pristine baseline (`probe perplexity`, `probe known answers`, `held-out mean KL`, `held-out perplexity rise`, `target` exact match).
4. **Clean up or promote.**
   - Failed gates undo automatically (`overlay: null`). For manual deltas or packs, call `overlay_undo` as soon as the comparison finishes (otherwise they expire after 30 minutes).
   - While any weight edit is live, `modelModified` is `true`, prompt-cache reuse (RAM and disk) is fenced and disabled, and `~/.basalt/experiments/<fingerprint>.jsonl` logs the lifecycle.
   - Never mutate source snapshots in the Hugging Face cache; write exported adapters or converted checkpoints to `~/.basalt/adapters/` (`0700`) or an explicit `--out` directory.

## 3. Five `mlx-swift` weight-surgery traps (mandatory when writing Swift code)

1. **`MLXArray` in-place aliasing under `Module.update`.** In `mlx-swift`, `MLXArray` is a reference class and `Module.update(parameters:)` calls `p._updateInternal(newArray)`, mutating existing `MLXArray` handles **in place**. Saving `let saved = ql.weight` aliases the live parameter: the first candidate `root.update(...)` overwrites `saved` too, making rollback a silent no-op. Clone saved tensors explicitly (`materialize(w)` or `let copy = w + MLXArray(0, dtype: w.dtype); eval(copy)`).
2. **`QuantizedMatmul` raw scale/bias pointer dtypes.** MLX Metal `QuantizedMatmul::eval_gpu` binds `scales` and `biases` directly as raw `bfloat16_t*` / `half*` pointers matching `x.dtype` without runtime casting. Optimizers such as `MLXOptimizers.Adam` promote `.bfloat16` parameters to `.float32`, corrupting quantized matmuls. Keep `.float32` master copies out-of-tree and cast updates back to `origDType` before `Module.update`.
3. **Compiled layer traces and fused projections.** Hybrid models (`Qwen35`) compile decoder layers when `training == false` and fuse input projections (`in_proj_qkv/z/b/a` → `fusedInputProjection.fused`) inside `model.prepare()`. After mutating weights or swapping modules:
   - call `model.invalidateCompiledTraces()` before evaluating candidates;
   - call `MLXInferenceEngine.settle(model)` (`invalidateCompiledTraces()` + `prepare()`) once when surgery finishes so fused projections rebuild from the edited weights;
   - when reading gradients through `Qwen35`, touch `in_proj_qkv.weight` via `update` once at entry so `fusedInputProjection` invalidates and individual projections stay on the autograd tape.
4. **Held-out KL gating on discrete code flips.** On 26B–35B models (`d >= 14336`), flipping discrete quantized codes (`uint32`-packed 2/3/4-bit affine or `nf4`) from gradients on a 128–256 token batch overfits that subspace and spikes orthogonal quantization noise. Average gradients over `>= 512–1024` tokens, sweep micro-fractions (`0.05%` down to `0.002%`), and accept a chunk only when **both** training and held-out validation KL strictly improve.
5. **Streaming in-place swaps and high-expert MoE VJP on 24 GB unified memory.** Never accumulate replacement `Module`s for all layers in an array before swapping (that holds `2×` the model in RAM and triggers macOS swap), and never call `MLXInferenceEngine.replace` inside a 300-layer loop (it runs `settle`/`prepare` every iteration). Snapshot `let entries = model.namedModules()`, swap each layer in place (`parent.update(modules:)`), call `Memory.clearCache()` every ~8 swaps, and call `MLXInferenceEngine.settle(model)` once at the end. In `mlx-swift`, `GatherQMM::vjp` on `QuantizedSwitchLinear` runs fast (`gather_qmm` on routed experts only) when expert parameters stay frozen (`arg == 0`), whereas unfreezing `scales`/`biases` (`arg == 2/3`) triggers `gather_mm_grad` + full `dequantize` across all experts (~1 GB per 128-expert projection and Metal watchdog timeouts) — keep `QuantizedSwitchLinear` frozen above ~16 experts (`PVTuner`).

## Related skills

All skills live in `.claude/skills/<name>/SKILL.md` (mirrored at `.agents/skills/`). Start from `swiftcoder` when unsure.

| Skill | Use it for |
|---|---|
| `swiftcoder` | the role and entry point: which skill to load, in what order, and the gates |
| `model-xray` | read-only anatomy, tensor stats, per-token traces and activation interventions before surgery |
| `basalt-model-runs` | runs that load model weights: benchmarks, ppl, memory, parity, GPU lock |
| `ponytail` | scope and the smallest complete change |
| `ponytail-review` | review of a change for correctness, safety, load, tests and need |
| `swift-64-architecture` | targets, boundaries, isolation domains, value core and actor shell |
| `swift-64-optimization-coding` | writing Swift 6.4 code: ownership, typed throws, size limits, Metal kernels |
| `swift-64-pr-review` | the 5-gate review of a Swift diff |
| `swift-concurrency-pro` | async/await, actors, tasks, streams and cancellation in depth |
| `swift-race-review` | races and deadlocks the compiler cannot see; MLX and Metal threading |
| `swift-testing-pro` | Swift Testing in depth: structure, async tests, new features |
| `xcode-mcp` | checking an API against Apple's documentation; Xcode workspace tools |
