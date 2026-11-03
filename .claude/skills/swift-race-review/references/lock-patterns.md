# Locks, conditions and `@unchecked Sendable`

## `@unchecked Sendable`
It is a promise the compiler stops checking. Accept it only when the type says how it keeps the promise:
- every mutable stored property is behind one named lock (or `Mutex`), or is only touched from one named thread;
- no method returns a reference into the protected state (an array of class instances, a closure capturing `self`) that a caller can then mutate unlocked;
- `deinit` does not run code that needs the lock while another thread may still hold a reference.

Ask the author to name the lock or thread. If they cannot, it is a race waiting for load.

## Lock level
A lock protects the state, not the call site. Two call sites guarding the same state with two different locks is no protection. One lock at the entry point every context passes through beats N locks around N callers.

## Deadlock checklist
- Holding lock L, does the code wait for anything (condition, semaphore, continuation, a synchronous `next()`, `DispatchQueue.sync`, `Task` result through a semaphore) whose producer needs L?
- Is L taken again on the same thread through a helper? Use `NSRecursiveLock` only when re-entry is by design, and say so.
- Two locks: is the order the same everywhere?
- `NSCondition.wait()` without a loop on the predicate misses spurious wake-ups; `broadcast` after state changes, under the lock.

## Locks across `await`
A lock held across a suspension point can be released on a different thread, or never if the task is cancelled; `NSLock` across `await` is banned in this repo. Synchronous critical sections inside async code are fine.

## Raw `Thread`
A `Thread {}` has no task-local values, no cancellation, and its own MLX default stream. It is justified for a loop that must block (the batch decoder waits on a condition between steps). Check: who stops it, what happens to waiters when it exits, and that it never touches actor state directly.

## Tests for races
- Prefer a deterministic interleaving (a condition or continuation the test controls) over sleeps.
- A tiny-model test that passes without the fix proves nothing about the fix; say what reproduces it on a real model (server under `--max-concurrent`, batch on a compiled model) and run that.
