---
name: swift-race-review
description: Hunt data races, deadlocks and cross-thread hazards in Swift code that the Swift 6 compiler cannot see — `@unchecked Sendable` types, `NSLock`/`NSCondition`/`Mutex`, raw `Thread`s, `Task.detached`, synchronous waits, global or C-library state, and MLX/Metal specifics (compiled traces that swap weights while tracing, thread-local streams and command encoders, two threads running one model). Use whenever you review, write or debug code in basalt that touches the model from more than one thread or task (batching, `--max-concurrent`, speculative iterators, prefill/decode hand-off, generation workers), adds a lock, condition or line to `scripts/concurrency-hatches.allow`, or when a test or server crashes with "eval an array without a primitive", a `~CommandEncoder` assertion, a hang, or output that differs only under load.
---

# Swift race review

The Swift 6 checker proves isolation only for what it can see. Everything below `@unchecked Sendable`, a lock, a C library, a GPU stream or a `Thread {}` is invisible to it, and that is where basalt's real concurrency bugs live: they compile cleanly, pass tiny-model tests most of the time, and crash or corrupt output on a real model under load.

basalt's `Sources/` has no `@unchecked Sendable`, `nonisolated(unsafe)`, `NSLock`, `NSCondition` or `DispatchQueue`. The few hatches left are listed, with the reason and what serializes each, in `scripts/concurrency-hatches.allow`; `scripts/check_concurrency_hatches.swift` fails CI on any hatch not listed there and on a line that no longer matches, so the list only shrinks. The unchecked bridge itself (`ModelEvaluationLock`, `FairRecursiveLock`, the `SendableBox` behind `performSending`) lives in the mlx-swift-lm fork.

Use this skill for that blind spot. For ordinary actor, `Task` and async/await correctness, `swift-concurrency-pro` and gate 1 of `swift-64-pr-review` already apply.

## How to review

1. **List the execution contexts.** For the code under review, name every thread or task that runs it: the caller's task, `container.perform` / `performSending`, the mlx-swift-lm generation worker that calls `iterator.next()` (a batched row steps the whole batch from its own worker), the `@concurrent` stream tasks (engine token stream, orchestrator relay), the allowlisted raw `Thread`s, timers, `deinit`. A race needs two of them. If you cannot say which two, you have not found a race yet.
2. **List the shared state they reach**, including state that isn't a Swift property:
   - the model's weights and module tree, read by every forward;
   - `CompiledTrace` / `MLX.compile` caches, which swap weights for tracers while they trace a new input shape;
   - MLX's default stream and thread-local command encoders;
   - KV / `MambaCache` objects shared between a request, a prompt-cache snapshot and a batch;
   - the `ModelContext`'s `batch` (`BatchCore`) and `mtpDrafter`, shared by every request on the model;
   - globals and `static var`s;
   - anything behind `@unchecked Sendable` or `nonisolated(unsafe)`, here or in the fork.
3. **For each pair (context A, context B, shared state S), ask what serializes them.** Name the lock, actor, queue or happens-before edge (e.g. `Stream.defaultStream.synchronize()` before handing arrays to another thread). "The GPU runs one command buffer at a time" serializes GPU execution. It does not serialize graph construction, tracing, or CPU reads of a module's weights.
4. **For every lock, check it can't deadlock.** Look for code that holds lock L and then waits (condition wait, semaphore, `take()`, synchronous `next()` that blocks on another thread) for a thread that needs L. Recursive acquisition of a non-recursive `NSLock` also hangs, for example `launch` holding the lock and calling a helper that takes it again.
5. **Check the tests can see it.**
   - A race test on a tiny model often passes without the fix, because the window is microseconds wide.
   - A test that finishes every prefill before the first decode step never overlaps them at all.
   - Say so, and ask for the real-model reproduction: the server under `--max-concurrent 4`, or staggered arrivals into the batch on Ornith or Qwen3.5. A green unit test is not enough.
   - Check that every new path is exercised on every model family it enables, not only on dense Qwen3. For a batch decoder that means join, retire and filter.

6. **Check the allowlist.** Run `swift scripts/check_concurrency_hatches.swift`. A new line in `scripts/concurrency-hatches.allow` needs a reason that names what serializes the site, and that reason must be true in the code; prefer the structural fix (a value, an actor, `sending`, `Mutex`) and no new line. A PR that removes a hatch removes its line.

Load `references/mlx-hazards.md` when the code calls a model, touches caches or MLX arrays across threads, or uses `CompiledTrace`. Load `references/lock-patterns.md` when it adds or changes a lock, condition or an allowlisted hatch. Run `scripts/race_hotspots.sh <base-ref>` first to list the risky lines in the diff, then read each hit in context: it finds candidates, it does not judge them.

## What to report

Check what an API guarantees (`Mutex`, `Atomic`, `withTaskCancellationShield`, `Task` cancellation inheritance, `MTLCommandBuffer` completion handlers) with Xcode's `DocumentationSearch`; see the `xcode-mcp` skill (`swift .claude/skills/xcode-mcp/scripts/xcode_mcp.swift DocumentationSearch '{"query":"Mutex","frameworks":["Synchronization"]}'`). Don't rely on memory. For a crash, use CLI `lldb` (`bt all`).

Read `scripts/concurrency-hatches.allow` for today's escape hatches and #300 for the target design.
- Data races stay inside one bridge.
- Actor state changes are synchronous reducers, with epoch checks after each `await`.
- Shared state is released by its owner (`consuming`), never in a `deinit`.
- One owner serializes MLX.

A finding that breaks one of these gets the structural fix, not an extra lock.

Report only hazards you can trace to two named contexts and one piece of shared state, or to a lock cycle you can draw. For each finding:

- **Contexts:** which two threads or tasks, with the code location each runs from.
- **Shared state and missing edge:** what they both touch, and why nothing orders them.
- **Failure:** what goes wrong and when. Use the concrete symptom when you know it: crash text, hang, a row that diverges, a corrupted cache.
- **Smallest fix:** usually one lock taken at the right level, which must be the level every context passes through, or moving the work onto the context that already owns the state. Don't propose a lock per call site when one at the shared entry point covers all of them.
- **Evidence it is fixed:** the test or run that would fail without the fix, and whether a tiny model can show it at all.

If you find nothing, say which contexts and shared state you checked. "No races" without that list is not a review.

## Basalt specifics worth remembering

- **One model, many threads.**
  - Generation runs `next()` on an mlx-swift-lm worker per request.
  - Prefill runs inside `container.performSending(nonSendable:)` in `launch`; the leased `PromptCache` rides in as the non-`Sendable` value and the `GenerationLaunch` comes back as a `sending` result, parked on the engine actor by lease id.
  - Batched decode has no thread of its own. It is flat-combined: a row whose consumer needs a token takes the model lock and, if the token is still missing, runs the next batch step for every row. Tokens and seats go through `BatchedDecoder`'s `Mutex<Mailbox>`; the MLX batch state is a `BatchCore` in `ModelContext.batch`.
  - The MTP drafter is `ModelContext.mtpDrafter`, loaded in `container.update`; the engine keeps only `hasMTPDrafter`.
  - All of them call the same model.
  - `ModelEvaluationLock.lock(for: model)` (a fair recursive lock per model, in the mlx-swift-lm fork) is the serialization point. The generation loops take it around each step; basalt takes it around `launch`, each batched prefill and each batch step.
  - `BatchedTokenIterator` conforms to `SelfSerializingTokenIterator`, so the loop calls its `next()` without the lock and the row takes it itself only when it steps the batch.
- **`--max-concurrent` > 1 is concurrency even with batching off.** A model with compiled traces (Qwen3.5 hybrids) crashed there before the shared lock existed.
- **Bitwise equality has limits.** Tiny-model tests can demand bitwise equality. On real 4-bit models a batched or multi-token forward reduces in a different order. There, an output difference is not by itself a race: compare accuracy on a task pool (`scripts/task_pool_eval.swift`) and look for a near-tie at the divergence point before calling it a bug.
