---
name: swift-64-optimization-coding
description: Use when writing high-performance Swift 6.4 code, optimizing memory allocations, eliminating ARC traffic, fixing strict concurrency issues, or implementing low-level systems and async workflows.
---

# Swift 6.4 High-Performance Coding & Zero-Cost Optimization Skill

You are an Expert Swift 6.4 Systems & Performance Engineer. Write deterministic, allocation-conscious, data-race-free Swift 6.4 code that maximizes stack allocation, static dispatch, and structured concurrency.

**Check APIs before relying on memory**: before using a Swift 6.2–6.4 API from this skill (`InlineArray`, `Span`/`RawSpan`, `UniqueBox`, `Mutex`, `withTaskCancellationShield`, ownership modifiers), look it up with Xcode's `DocumentationSearch`. Use `mcp__xcode__DocumentationSearch`, or `swift .claude/skills/xcode-mcp/scripts/xcode_mcp.swift DocumentationSearch '{"query":"RawSpan unsafeLoad"}'`. See the `xcode-mcp` skill.

## 1. Zero-Copy Memory & Ownership Idioms (Swift 6.2 - 6.4)
- **Contiguous Stack Buffers**: Use `InlineArray<N, Element>` for fixed-size buffers instead of heap-allocated `Array<Element>`.
- **Safe Non-Owning Views**: Use `Span<Element>` and `RawSpan` instead of `UnsafeBufferPointer` for bounds-checked, lifetime-tracked contiguous memory access.
- **Zero-Refcount Heap Ownership (SE-0517 / SE-0527)**:
  - Use `UniqueBox<T>` for single-owner heap allocation without ARC atomic reference-counting overhead.
  - Use `UniqueArray<Element>` for dynamically growing buffers of `~Copyable` elements or hot-path buffers where Copy-on-Write (CoW) checks must be eliminated.
- **First-Class Borrowing References (SE-0507 / SE-0519 / SE-0532)**:
  - Implement `borrow` and `mutate` accessors on properties exposing `Span`, `InlineArray`, or `~Copyable` state to prevent implicit copies.
  - Use `Ref<T>` and `MutableRef<T>` when storing temporary borrowed references without consuming or copying the underlying value.
  - Inspect `Optional<T>` where `T: ~Copyable` via borrowing pattern matching without consuming the optional payload.
- **Non-Copying Iteration (SE-0516)**: Conform performance-critical containers to `Iterable` and iterate via borrowing loops rather than `Sequence` when elements are large value types or `~Copyable`.
- **Parameter Ownership**: Annotate large parameters explicitly with `borrowing` (read-only), `inout` (in-place mutation), or `consuming` (ownership transfer / final move).

## 2. Strict Concurrency & Async Lifecycle (Swift 6.4)
- **Async Cleanup in `defer` (SE-0493)**: Use `defer { await resource.close() }` directly inside async functions to guarantee deterministic asynchronous teardown across all exit paths.
- **Cancellation Shielding (SE-0504)**: Wrap critical teardown, transaction rollback, or state-flush logic inside `withTaskCancellationShield { ... }` so cleanup completes even if the enclosing `Task` is cancelled.
- **Region-Based Isolation & `sending` (SE-0414 / SE-0430)**: Use `sending` parameters and return values to transfer non-`Sendable` mutable graphs safely across actor boundaries without defensive copying or `@unchecked Sendable`.
- **Actor Contention Reduction**:
  - Mark pure helper methods inside actors as `nonisolated`.
  - Use `nonisolated(nonsending)` and `@concurrent` accurately to avoid unnecessary executor hops.
  - Never hold resources across an `await` suspension point without re-validating actor invariants afterward (actor reentrancy guard).
- **Ban Legacy Primitives**: Never generate `DispatchQueue`, `DispatchSemaphore`, `NSLock` across async contexts, or `Task.detached` (unless an explicit unstructured root lifetime is documented).

## 3. Dispatch, Generics & Compiler Optimization Controls
- Eliminate existential boxing (`any Protocol` witness table + 3-word existential buffer) in hot loops; specialize via `some Protocol` or generic constraints.
- Apply `@inlinable`, `@inline(always)`, `@specialize`, and `@export(implementation)` judiciously on cross-module hot-path APIs.
- Use `throws(SpecificError)` in performance-critical or embedded paths to avoid existential `any Error` heap boxing.
- For C/C++ interoperability (Swift 6.3/6.4), use `@c` and `@implementation` instead of manual C bridging wrappers.

## 4. Size, Complexity & Leaving Code Better (basalt)
- **Limits (CONTRIBUTING "Size and complexity")**:
  - a function stays at CCN ≤ 15 (aim ≤ 10), ≤ 60 lines of code and ≤ 6 parameters;
  - a file stays at ≤ 600 lines (aim ≤ 400);
  - a new type body stays at ≤ 200 lines of code, and an extension at ≤ 150;
  - a type older than 2026-11-04 aims for ≤ 450, and any type over 200 must not grow;
  - code goes in a value core (structs, enums, pure functions) with a thin actor or protocol shell for effects (CONTRIBUTING "Value core, actor shell");
  - check with `uvx lizard==1.24.0 -l swift -C 15 -L 60 -a 6 <files>` before committing.
- **Ratchet**: a function or file already over a limit must not grow. Before adding a branch to one, split it along an existing seam: an engine mode, a `switch` case, a stage of a long function.
- **Apply the toolset in the code you touch**, not only in new code:
  - `Mutex`/actor over `@unchecked Sendable` and locks;
  - `~Copyable` leases over flags such as `isHeld`;
  - cancellation shields on teardown;
  - typed throws;
  - `RawSpan` over unsafe byte pointers;
  - `InlineArray` on per-token scratch, only with a micro-benchmark that shows the gain.

  Tracking issue: #284.

## 5. Traps the compiler reports late (seen in basalt diffs)
- **Typed throws stop at some closures.** `map`, `mapValues` and `Optional.map` carry `throws(E)`. `compactMap`, sequence `flatMap`, `reduce` and `sorted(by:)` are `rethrows` and widen to `any Error`. Inside a `throws(BasaltError)` function, write the loop.
- **No `await` in an autoclosure.** `try await a() ?? b()` fails with "'async' call in an autoclosure that does not support concurrency". Use `if let` in a small function.
- **A `static let` of a non-`Sendable` type is rejected** (`MLXArray`, a kernel input table). Make it a computed `static var`, or build it where it is used. Do not reach for `nonisolated(unsafe)`.
- **`swift run -c release Basalt`**, not `basalt`: the product name is capitalized. `models convert` takes `--out`.

## 6. Custom Metal kernels (`MLXFast.metalKernel`)
- **A kernel that does not compile is a `fatalError` at its first call**, once per template instantiation. A parity test must call every combination the model will use (each type, each row count path), or the crash lands on a user's first token.
- **MSL has its own reserved words.** A parameter named `thread` fails with "parameter may not be qualified with an address space". Name helpers' parameters plainly (`first`, `stride`).
- **Admission is #228:** parity against a CPU reference on generated blocks for every type, and exact greedy tokens on a `TinyModel` fixture. A second kernel is not a reference.
- **Find the floor before tuning.** Replace the decode with a trivial read of the same bytes and time it. That number is the memory floor; the rest is arithmetic. Then remove one cost at a time (lookups, sign application, conversion) to see which one pays.
- **Benchmark shape:** release build, GPU lock, many calls evaluated in one `eval` (a single `eval` per call measures launch overhead), the stock kernel timed in the same process as the candidate.
- **Experiment switches** go in as template integers read from the environment, so one build sweeps every variant with `--skip-build`. They never ship: remove them before the PR.
- **What did not matter on M4 Pro** (#408): where a 512-entry table lives (constant, threadgroup or device buffer), rows per simdgroup from 2 to 8, simdgroups per threadgroup from 2 to 32, `half` arithmetic. A pre-converted `float4` table in threadgroup memory was three times slower.

## Related skills

All skills live in `.claude/skills/<name>/SKILL.md` (mirrored at `.agents/skills/`). Start from `swiftcoder` when unsure.

| Skill | Use it for |
|---|---|
| `swiftcoder` | the role and entry point: which skill to load, in what order, and the gates |
| `ponytail` | scope and the smallest complete change |
| `ponytail-review` | review of a change for correctness, safety, load, tests and need |
| `swift-64-architecture` | targets, boundaries, isolation domains, value core and actor shell |
| `swift-64-pr-review` | the 5-gate review of a Swift diff |
| `swift-concurrency-pro` | async/await, actors, tasks, streams and cancellation in depth |
| `swift-race-review` | races and deadlocks the compiler cannot see; MLX and Metal threading |
| `swift-testing-pro` | Swift Testing in depth: structure, async tests, new features |
| `xcode-mcp` | checking an API against Apple's documentation; Xcode workspace tools |
| `basalt-model-runs` | runs that load model weights: benchmarks, ppl, memory, parity, GPU lock |
| `model-xray` | looking inside a running model and calling basalt's own MCP server |
