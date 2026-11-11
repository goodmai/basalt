---
name: swift-concurrency-pro
description: Reviews Swift code for concurrency correctness, modern API usage, and common async/await pitfalls, with a basalt section that overrides the generic advice where the repo bans a primitive. Use when reading, writing, or reviewing Swift concurrency code (async/await, actors, tasks, AsyncStream, continuations, cancellation).
license: MIT
metadata:
  author: Paul Hudson
  version: "1.0"
  modified: "basalt section and related-skills table added 2026-11-11; the references are upstream text"
---

Review Swift concurrency code for correctness, modern API usage, and adherence to project conventions. Report only genuine problems - do not nitpick or invent issues.

## In basalt

This skill is general Swift concurrency guidance. Where it and the repo disagree, the repo wins (`CLAUDE.md` section 4, CONTRIBUTING "Value core, actor shell", #300):

- **Target Swift 6.4**, strict concurrency. "Swift 6.2" in the references is a floor, not the target. Check newer APIs (`withTaskCancellationShield`, `defer { await }`) with `xcode-mcp`.
- **GCD and locks are not acceptable in new code**, whatever the references say about low-level use: `DispatchQueue`, `DispatchSemaphore`, `NSLock`, `NSCondition`, `os_unfair_lock`.
- **`@unchecked Sendable`, `nonisolated(unsafe)` and `Task.detached` are banned**, including the "type with internal locking" and "last resort" cases in `references/bridging.md`, `diagnostics.md` and `bug-patterns.md`. An existing site needs a line in `scripts/concurrency-hatches.allow`, and that list only shrinks. Redesign with an owning actor, a `sending` hand-off or a value type.
- **State changes are synchronous reducers** in the value core; a result applied after an `await` checks an epoch. Use `references/actors.md` to find reentrancy, and fix it this way.
- **MLX objects never leave their owner.** For two tasks touching one model, compiled traces or Metal encoders, load `swift-race-review`: those hazards are not in the references here.
- **Tests:** no `Task.sleep`, including `.zero` yields used as synchronization (`references/testing.md` shows one). Use continuations, `confirmation` or an injected clock.
- **Output:** inside a PR review, report findings in the `swift-64-pr-review` format with severities; the per-file format below is for a standalone concurrency read.

Review process:

1. Scan for known-dangerous patterns using `references/hotspots.md` to prioritize what to inspect.
1. Check for recent Swift 6.2 concurrency behavior using `references/new-features.md`.
1. Validate actor usage for reentrancy and isolation correctness using `references/actors.md`.
1. Ensure structured concurrency is preferred over unstructured where appropriate using `references/structured.md`.
1. Check unstructured task usage for correctness using `references/unstructured.md`.
1. Verify cancellation is handled correctly using `references/cancellation.md`.
1. Validate async stream and continuation usage using `references/async-streams.md`.
1. Check bridging code between sync and async worlds using `references/bridging.md`.
1. Review any legacy concurrency migrations using `references/interop.md`.
1. Cross-check against common failure modes using `references/bug-patterns.md`.
1. If the project has strict-concurrency errors, map diagnostics to fixes using `references/diagnostics.md`.
1. If reviewing tests, check async test patterns using `references/testing.md`.

If doing a partial review, load only the relevant reference files.


## Core Instructions

- Target Swift 6.2 or later with strict concurrency checking.
- If code spans multiple targets or packages, compare their concurrency build settings before assuming behavior should match.
- Prefer structured concurrency (task groups) over unstructured (`Task {}`).
- Prefer Swift concurrency over Grand Central Dispatch for new code. GCD is still acceptable in low-level code, framework interop, or performance-critical synchronous work where queues and locks are the right tool – don't flag these as errors.
- If an API offers both `async`/`await` and closure-based variants, always prefer `async`/`await`.
- Do not introduce third-party concurrency frameworks without asking first.
- Do not suggest `@unchecked Sendable` to fix compiler errors. It silences the diagnostic without fixing the underlying race. Prefer actors, value types, or `sending` parameters instead. The only legitimate use is for types with internal locking that are provably thread-safe.


## Output Format

Organize findings by file. For each issue:

1. State the file and relevant line(s).
2. Name the rule being violated.
3. Show a brief before/after code fix.

Skip files with no issues. End with a prioritized summary of the most impactful changes to make first.

Example output:

### DataLoader.swift

**Line 18: Actor reentrancy – state may have changed across the `await`.**

```swift
// Before
actor Cache {
    var items: [String: Data] = [:]

    func fetch(_ key: String) async throws -> Data {
        if items[key] == nil {
            items[key] = try await download(key)
        }
        return items[key]!
    }
}

// After
actor Cache {
    var items: [String: Data] = [:]

    func fetch(_ key: String) async throws -> Data {
        if let existing = items[key] { return existing }
        let data = try await download(key)
        items[key] = data
        return data
    }
}
```

**Line 34: Use `withTaskGroup` instead of creating tasks in a loop.**

```swift
// Before
for url in urls {
    Task { try await fetch(url) }
}

// After
try await withThrowingTaskGroup(of: Data.self) { group in
    for url in urls {
        group.addTask { try await fetch(url) }
    }

    for try await result in group {
        process(result)
    }
}
```

### Summary

1. **Correctness (high):** Actor reentrancy bug on line 18 may cause duplicate downloads and a force-unwrap crash.
2. **Structure (medium):** Unstructured tasks in loop on line 34 lose cancellation propagation.

End of example.


## References

- `references/hotspots.md` - Grep targets for code review: known-dangerous patterns and what to check for each.
- `references/new-features.md` - Swift 6.2 changes that alter review advice: default actor isolation, isolated conformances, caller-actor async behavior, `@concurrent`, `Task.immediate`, task naming, and priority escalation.
- `references/actors.md` - Actor reentrancy, shared-state annotations, global actor inference, and isolation patterns.
- `references/structured.md` - Task groups over loops, discarding task groups, concurrency limits.
- `references/unstructured.md` - Task vs Task.detached, when Task {} is a code smell.
- `references/cancellation.md` - Cancellation propagation, cooperative checking, broken cancellation patterns.
- `references/async-streams.md` - AsyncStream factory, continuation lifecycle, back-pressure.
- `references/bridging.md` - Checked continuations, wrapping legacy APIs, `@unchecked Sendable`.
- `references/interop.md` - Migrating from GCD, `Mutex`/locks, completion handlers, delegates, and Combine.
- `references/bug-patterns.md` - Common concurrency failure modes and their fixes.
- `references/diagnostics.md` - Strict-concurrency compiler errors, protocol conformance fixes, and likely remedies.
- `references/testing.md` - Async test strategy with Swift Testing, race detection, avoiding timing-based tests.

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
| `swift-race-review` | races and deadlocks the compiler cannot see; MLX and Metal threading |
| `swift-testing-pro` | Swift Testing in depth: structure, async tests, new features |
| `xcode-mcp` | checking an API against Apple's documentation; Xcode workspace tools |
| `basalt-model-runs` | runs that load model weights: benchmarks, ppl, memory, parity, GPU lock |
| `model-xray` | looking inside a running model and calling basalt's own MCP server |
