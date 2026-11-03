# MLX and Metal hazards

Each section names the shared state, the contexts that race on it, how it showed up in basalt, and the fix that held.

## 1. A compiled trace swaps weights while it traces

**State:** the module weights a `CompiledTrace` declares as compile inputs.

`MLX.compile(inputs:)` traces once per new input shape. While it traces, it replaces the arrays it was given as inputs, here the module's weights, with tracer arrays, then restores them.

**Contexts:** any two forwards of the same model, for example:
- request A's first decode step at a new batch size, tracing;
- request B's prefill, which reads the same projection weights through the uncompiled path.

`CompiledTrace` in mlx-swift-lm takes its lock only to install the trace, and calls the compiled function outside it.

**Symptom:** `Fatal error: [eval] Attempting to eval an array without a primitive`.
- It appears on a Qwen3.5 hybrid (Ornith, Qwen3.8, Bonsai 2), never on Llama, Nemotron-H or LFM2, which have no compiled traces.
- It appears under `--max-concurrent 4` even with batching off.
- In the batch decoder it surfaced on the second step, as soon as a new row prefilled while the batch thread decoded.

**Not sufficient:**
- A lock inside `CompiledTrace`: an uncompiled reader of the same weights still races with the trace.
- `Stream.defaultStream.synchronize()`: it orders GPU work, not tracing.

**Fix:** serialize all forwards of the model at one shared entry point: `ModelEvaluationLock.shared` around each generation step, around `launch` (every prefill path), and around each batch step.

## 2. Arrays produced on one thread, evaluated on another

**State:** MLX's per-thread default stream and Metal command encoders.

**Contexts:** prefill in `container.perform`, then decode on the generation worker. Work still encoded on the first thread's encoder can be committed from two threads at once.

**Symptom:** crashes in `~CommandEncoder()` in crash reports. #214 found the pattern but couldn't reproduce the crash.

**Fix:** `Stream.defaultStream.synchronize()` at the hand-off (`MLXInferenceEngine.handOff`), so no generation's work sits in two thread-local encoders. This is the right tool for hand-offs. It is the wrong tool for item 1.

## 3. A cache object shared between a request, a snapshot and a batch

**State:** `KVCacheSimple` / `MambaCache` / `ArraysCache` instances.

**Contexts and pitfalls:**
- **Joining a batch.** `HybridPrefixSnapshot.snapshot(at:)` copies only snapshot-only (recurrent) layers and shares attention layers by reference. The batch takes the first row's recurrent cache object itself and `extend`s the others into it.
- **`ArraysCache.copy()` returns a base `ArraysCache`.** Model code casts `cache as? MambaCache`, so a copy breaks the model silently. Keep the original object.
- **Speculative checkpoints.** They hold references to cache slots. A replay restores them, so a checkpoint must not outlive the round.

**What to check:**
- Who else holds a reference after a cache joins a batch or a snapshot?
- Does anyone write into it later?

## 4. Two threads, one model, and a synchronous wait

**Contexts:**
- The generation worker calls `BatchedTokenIterator.next()`, which blocks until the batch thread delivers a token.
- The batch thread needs the model lock to step.

**Deadlock:** if the generation loop holds the model lock around `next()`, the batch thread can never take it.

**Fix:** iterators that only wait conform to `SelfSerializingTokenIterator`; the loop calls them without the lock, and the batch thread takes it per step.

## 5. Numerics that look like a race

A batched `[B,1]` forward or a multi-token verify forward uses different quantized-matmul kernels and reduction orders than `[1,1]`. On real 4-bit models, greedy text can therefore differ at a near-tie with no race involved: prompt lookup, MTP, batching. Tiny float models usually stay bitwise equal.

Before calling a divergence a race:
- check the logits at the first differing token;
- check that the behaviour is deterministic run to run, so it is not timing-dependent;
- compare accuracy on `scripts/task_pool_eval.swift` between the two modes.

Two quick discriminators:
- **Same prompt in every row.** Put one prompt in all B rows of one batch. Every row runs the same kernels, so the rows must be bitwise equal to each other. If they differ, something per row is wrong: padding, offsets, a cache shared between rows, or a race.
- **Run the same pass twice.** It must reproduce itself.

Rule out confounds before comparing modes:
- **Prompt-cache hits.** A repeated prompt reuses the cache, and that changes numerics: chunking drift. Compare first runs only, or turn the cache off.
- **Seeds and sampling.** Different seeds or different sampling settings between the two passes.
- **Inputs.** Different prompts or token limits between the two passes.

A race shows up as a crash, a run-to-run difference, or an accuracy drop. Numerics show up as a stable, reproducible difference with unchanged accuracy.
