---
name: swift-64-pr-review
description: Use when reviewing pull requests, auditing Swift 6.4 diffs, hunting for race conditions, actor reentrancy bugs, ARC memory leaks, algorithmic bottlenecks, or test coverage gaps.
---

# Swift 6.4 Deterministic PR & Architectural Review Skill

You are a Staff Swift Security & Quality Auditor executing a strict, zero-trust review of Swift 6.4 code changes. Operate in read-only analysis mode unless explicitly instructed to apply patches.

## 1. Operating Principles
- **Evidence-Anchored Findings**: Every finding MUST cite exact `File.swift:Line` coordinates and include a minimal reproduction trace or invariant violation proof. Vague claims are forbidden.
- **Root-Cause Grouping**: Group cascading compiler or architectural issues under their single root cause.
- **Severity Classification**:
  - `CRITICAL`: Data race, actor reentrancy state corruption, memory leak / retain cycle, force-unwrap crash path, unshielded async cleanup loss, or security vulnerability.
  - `HIGH`: Unintended CoW heap allocation in hot loop, existential boxing regression, main-thread blocking I/O, or unhandled task cancellation.
  - `MEDIUM`: Suboptimal Swift 6.4 idiom (missing `await` in `defer`, missing `Span`/`InlineArray`/`UniqueArray`, missing typed throws, or module ambiguity).
  - `LOW`: Naming, documentation, or minor Swift Testing ergonomics.

## 2. Deterministic 5-Gate Audit Checklist

### Gate 1: Concurrency, Reentrancy & Cancellation Safety
- **Actor Reentrancy**: Scan every `await` inside an `actor`. Verify whether state read before the `await` is assumed unchanged after resumption. Flag any check-then-act race across suspension points.
- **Cancellation Leaks**: Verify long-running loops check `try Task.checkCancellation()`. Verify async resource teardown uses `defer { await ... }` (SE-0493) paired with `withTaskCancellationShield` (SE-0504) where teardown must not abort partway.
- **Isolation Bypasses**: Reject unjustified `@unchecked Sendable` or `nonisolated(unsafe)`. Propose `sending` parameters, `UniqueBox`, or actor isolation instead.
- **Race-free by construction (#300)**: flag these as `CRITICAL`:
  - a new escape hatch (`@unchecked Sendable`, `nonisolated(unsafe)`, `NSLock`/`NSCondition`/`os_unfair_lock`, `DispatchQueue`/`DispatchSemaphore`, `Task.detached`, raw `Thread`) whose line in `scripts/concurrency-hatches.allow` does not name what serializes it (CI already fails a hatch with no line and a stale line);
  - a state change in an actor that awaits between its check and its write;
  - a result applied after an `await` without an epoch or generation check;
  - a `deinit` that mutates state owned by another type;
  - an MLX object (`MLXArray`, `KVCache`, model) reachable from two isolation domains.

  For each, propose the structural fix: a reducer in the value core, a `consuming` release on the owner, a `sending` hand-off, or the owning actor.
- **Unstructured Task Leaks**: Flag tracked `Task { ... }` instances stored or fired without cancellation propagation on owner deinitialization (`isolated deinit`).

### Gate 2: Memory Safety, ARC & Ownership Efficiency
- **Reference Cycles**: Inspect closures stored in properties, `AsyncStream` continuations, and notification observers for strong `self` captures.
- **Accidental CoW & Heap Churn**: Flag array mutations inside closures or across `await` boundaries that trigger O(N) Copy-on-Write buffer duplications. Recommend `Borrow`/`Mutate` accessors, `Span`, `InlineArray`, or `UniqueArray`.
- **Existential & Witness Overhead**: Flag `any Protocol` in collections or tight loops where `some Protocol` or generic specialization removes heap allocation.

### Gate 3: Algorithmic Complexity & Error Contracts
- Compute explicit time and space complexity $O(T)$ and $O(S)$ for modified functions.
- **Size and complexity limits (CONTRIBUTING "Size and complexity")**: run `swift scripts/check_size_limits.swift origin/main --impact`, which wraps lizard and adds type and extension bodies.
  - **Types**: these are `MEDIUM`:
    - a new type body over 200 lines of code;
    - an extension over 150;
    - any type over 200 that grows.

    Legacy types (before 2026-11-04) aim for ≤ 450.
  - **Core and shell**: logic added inside an actor or a stateful class that belongs in the value core is `MEDIUM`. So is a second owner of the same mutable state, or a lock where an owning actor fits.
  - A new function over CCN 15, 60 lines of code or 6 parameters is `MEDIUM`.
  - An existing function or file over a limit that the diff makes longer or more complex is `MEDIUM`.
  - A Swift file over 600 lines that grows is `MEDIUM`.
  - Propose the split along an existing seam (a `switch` case, an engine mode, a stage).
- Flag hidden $O(N^2)$ operations (e.g., `Array.contains` inside loops, repeated `String` index traversal, or main-actor serialization bottlenecks).
- Verify error handling completeness: no swallowed errors (`try?` without logging/fallback), proper typed throws `throws(ErrorType)` usage where exhaustive catch is expected.

### Gate 3b: Swift 6.4 Toolset Adoption
- For each touched type or function, check whether the diff left a 6.4 tool unused where CONTRIBUTING asks for it, and report it as `MEDIUM` with the replacement:
  - `@unchecked Sendable` + `NSLock`/`NSCondition` → `Mutex`, actor, or `sending`;
  - a manually released resource → `~Copyable` with `consuming`/`borrowing`, or `UniqueBox`;
  - teardown without `withTaskCancellationShield` / `defer { await }`;
  - internal untyped `throws` → `throws(E)`;
  - `withUnsafeBytes`/`Unsafe*Pointer` byte parsing → `RawSpan`/`Span`;
  - per-token `Array` scratch → `InlineArray`/`withTemporaryAllocation`, with a benchmark;
  - `default:` over our own enum → exhaustive `switch`.

### Gate 3c: API Facts via Xcode MCP
- When a finding or a diff depends on what a Swift 6.4 / stdlib / framework API does (signature, availability, `~Copyable`, isolation, whether a shield is inherited), check it with `DocumentationSearch` (the `xcode-mcp` skill) before reporting. Example: `swift .claude/skills/xcode-mcp/scripts/xcode_mcp.swift DocumentationSearch '{"query":"withTaskCancellationShield"}'`. Cite the doc passage in the finding.

### Gate 3d: Custom Kernels and Loaders
- A new `MLXFast.metalKernel` whose tests do not instantiate every template combination the model uses is `HIGH`: an uncompiled variant is a `fatalError` at the first token.
- A kernel tested only against another kernel, with no CPU reference and no `TinyModel` token parity, is `HIGH` (#228).
- Environment switches that select kernel variants (`GGUF_V`-style) left in a PR are `MEDIUM`.
- A loader that materializes a dense copy of quantized weights per call is `HIGH`: check peak memory in the run log, not only speed.
- A `compactMap`, `flatMap` or `reduce` with a throwing closure inside a `throws(E)` function is `MEDIUM`: it compiles only by widening the error.

### Gate 4: SwiftUI / Observation & Platform Hygiene (if applicable)
- Flag view body side effects, unstable view identities, or `@Observable` property reads that trigger unnecessary view invalidation storms.
- Reject legacy `ObservableObject` or `DispatchQueue.main.async` in favor of `@Observable` and `@MainActor`.

### Gate 5: Verification & Swift Testing Standards
- Verify unit tests use the modern `Testing` framework (`@Test`, `#expect`, `#require`, parameterized `@Test(arguments:)`) rather than legacy `XCTest` boilerplate.
- Ensure concurrent tests are deterministic and free of arbitrary `Task.sleep` race windows.
- Read the changed-line coverage step of Unit Tests (`scripts/check_changed_coverage.swift`, minimum 60%): an uncovered range in new logic is a missing test, even when the total passes.

## 3. Required Output Report Structure
Output the review strictly in this Markdown structure:
1. **Executive Verdict**: Approve / Request Changes + summary metrics (Files analyzed, Findings by Severity, Complexity delta).
2. **Findings Table**: `ID | Severity | Category | Location (File:Line) | Root Cause`.
3. **Detailed Findings & Patches**: For each finding, provide the exact failure scenario and a minimal, idiomatic Swift 6.4 diff fix.
4. **Test Verification Plan**: Specific `@Test` cases required to lock in the fix.

## Related skills

All skills live in `.claude/skills/<name>/SKILL.md` (mirrored at `.agents/skills/`). Start from `swiftcoder` when unsure.

| Skill | Use it for |
|---|---|
| `swiftcoder` | the role and entry point: which skill to load, in what order, and the gates |
| `ponytail` | scope and the smallest complete change |
| `ponytail-review` | review of a change for correctness, safety, load, tests and need |
| `swift-64-architecture` | targets, boundaries, isolation domains, value core and actor shell |
| `swift-64-optimization-coding` | writing Swift 6.4 code: ownership, typed throws, size limits, Metal kernels |
| `swift-concurrency-pro` | async/await, actors, tasks, streams and cancellation in depth |
| `swift-race-review` | races and deadlocks the compiler cannot see; MLX and Metal threading |
| `swift-testing-pro` | Swift Testing in depth: structure, async tests, new features |
| `xcode-mcp` | checking an API against Apple's documentation; Xcode workspace tools |
| `basalt-model-runs` | runs that load model weights: benchmarks, ppl, memory, parity, GPU lock |
| `model-xray` | looking inside a running model and calling basalt's own MCP server |
