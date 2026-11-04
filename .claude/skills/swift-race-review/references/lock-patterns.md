# Locks, conditions and `@unchecked Sendable`

## `@unchecked Sendable` and the allowlist
basalt's `Sources/` has none. It is a promise the compiler stops checking, so a new one needs a line in `scripts/concurrency-hatches.allow` (`path:symbol: why it is needed and what serializes it`), which CI enforces and which only shrinks. Prefer a value type, an actor, `sending` (`performSending`) or a `Mutex` around value state, as `BatchedDecoder`'s mailbox does. Accept a hatch only when the type says how it keeps the promise:
- every mutable stored property is behind one named lock (or `Mutex`), or is only touched from one named thread;
- no method returns a reference into the protected state (an array of class instances, a closure capturing `self`) that a caller can then mutate unlocked;
- `deinit` does not run code that needs the lock while another thread may still hold a reference.

Ask the author to name the lock or thread. If they cannot, it is a race waiting for load.

## Lock level
A lock protects the state, not the call site. Two call sites guarding the same state with two different locks is no protection. One lock at the entry point every context passes through beats N locks around N callers.

## Deadlock checklist
- Holding lock L, does the code wait for anything (condition, semaphore, continuation, a synchronous `next()`, `DispatchQueue.sync`, `Task` result through a semaphore) whose producer needs L?
- Is L taken again on the same thread through a helper? `Mutex` is not recursive. Re-entry is by design only for the model lock (`ModelEvaluationLock`, a recursive lock in the fork: `launch` holds it while a batched row's prefill takes it again).
- Two locks: is the order the same everywhere?
- A condition wait (in the fork's `FairRecursiveLock`, or anywhere a hatch allows one) loops on its predicate and is signalled after the state changes, under the lock.

## Locks across `await`
A lock held across a suspension point can be released on a different thread, or never if the task is cancelled; `NSLock` across `await` is banned in this repo. Synchronous critical sections inside async code are fine.

## Raw `Thread` and `Task.detached`
A `Thread {}` has no task-local values, no cancellation, and its own MLX default stream. It is justified only for a call that must block a thread and must not block a cooperative-pool thread: today the stdin reader in `MCPServer` and the pipe writer in `ClipboardManager`, both allowlisted. Batched decode no longer has a thread; rows step the batch themselves (`mlx-hazards.md` §4). Check: who stops it, what happens to waiters when it exits, and that it never touches actor state directly.

`Task.detached` drops actor isolation, priority and task-locals. The two allowlisted stream tasks (engine and orchestrator) need that because they outlive the call that made them; they carry only `Sendable` values and hand state back through `await` on the actor. Anything else uses `Task {}` or structured concurrency.

## Reading Thread Sanitizer on MLX
`swift test --sanitize=thread` on anything that evaluates MLX arrays prints dozens of reports from MLX's Metal backend: command-buffer completion handlers, `scheduler::notify_task_completion`, `MetalAllocator`, fences. Metal orders these in ways TSan cannot see, and they appear even in single-threaded test setup.

What to do with them:
- Group the reports by the top frame.
- Set aside the ones whose stacks are MLX/Metal internals plus a basalt frame that merely called `eval`.
- What remains, reports where both sides touch basalt state (a decoder's rows, a cache array, an engine property), are the real findings.
- Say how many you triaged and how.

## Tests for races
- Prefer a deterministic interleaving (a condition or continuation the test controls) over sleeps.
- A tiny-model test that passes without the fix proves nothing about the fix; say what reproduces it on a real model (server under `--max-concurrent`, batch on a compiled model) and run that.
