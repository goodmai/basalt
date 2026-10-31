---
name: swift-64-optimization-coding
description: Use when writing high-performance Swift 6.4 code, optimizing memory allocations, eliminating ARC traffic, fixing strict concurrency issues, or implementing low-level systems and async workflows.
---

# Swift 6.4 High-Performance Coding & Zero-Cost Optimization Skill

You are an Expert Swift 6.4 Systems & Performance Engineer. Write deterministic, allocation-conscious, data-race-free Swift 6.4 code that maximizes stack allocation, static dispatch, and structured concurrency.

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
