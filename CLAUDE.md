# Swift 6.4 Engineering & Agent Orchestration Contract

## 1. Mandatory Skill Routing (Progressive Disclosure)
Before proposing designs, writing code, running real-model benchmarks, or reviewing changes, you MUST load and follow the corresponding skill from `.claude/skills/` (mirrored at `.agents/skills/`):

| Task Intent | Trigger Condition | Required Skill Path |
|---|---|---|
| **Architecture & Modularity** | Designing targets, `Package.swift` topology, DI boundaries, actor isolation domains, or choosing patterns | `.claude/skills/swift-64-architecture/SKILL.md` |
| **Coding & Optimization** | Writing features, optimizing memory/ARC, using `Span`/`UniqueBox`/`Ref`, or implementing async workflows | `.claude/skills/swift-64-optimization-coding/SKILL.md` |
| **PR Review & Security Audit** | Reviewing diffs, auditing concurrency/reentrancy, checking CoW allocations, or verifying complexity | `.claude/skills/swift-64-pr-review/SKILL.md` |
| **Lean Diff & YAGNI Discipline** | Choosing the smallest complete solution (`YAGNI -> reuse -> stdlib -> native`) and reviewing lean diffs | `.claude/skills/ponytail/SKILL.md`, `.claude/skills/ponytail-review/SKILL.md` |
| **Real-Model Benchmarks & Parity** | Running MLX weights on Metal (`PerformanceBenchmark`, PPL/passkey/memory, GPU lock, PR tables) | `.claude/skills/basalt-model-runs/SKILL.md` |

*Rule:* Never implement architectural changes or performance-critical paths from base memory alone—always read the matching `SKILL.md` and `CONTRIBUTING.md` first. External MCP/CLI runtime operations live in `docs/AGENT_OPERATIONS.md`.

---

## 2. Toolchain & Strict Language Baseline
- **Toolchain**: Swift 6.4 (`swift-tools-version: 6.4`, Xcode 27 / Swift Build engine, `.macOS(.v27)`).
- **Language Mode**: Strict Swift 6 concurrency mode (`swiftLanguageModes: [.v6]`).
- **Module Disambiguation (SE-0491)**: Resolve symbol collisions explicitly via module selectors (`ModuleName::SymbolName`).
- **Opaque Optional Types (SE-0521)**: Use `some Protocol?` instead of existential `any Protocol?` unless runtime heterogeneity is required.

---

## 3. Build & Test Execution Policy (`swift build` / `swift test`)

All code modifications must pass these deterministic CLI verification gates before task completion:

### Build Gates
1. **Debug Strict Gate (Zero Warnings)**:
   ```bash
   swift build -Xswiftc -warnings-as-errors
   ```
2. **Release Optimization & Specialization Check** (required when modifying generics, `@inlinable`, `~Copyable`, or hot paths):
   ```bash
   swift build -c release -Xswiftc -warnings-as-errors
   ```

### Test Gates (`Swift Testing` Framework Only)
1. **Standard Parallel Execution**:
   ```bash
   swift test --parallel
   ```
2. **Targeted Suite Execution** (use during iterative TDD before full run):
   ```bash
   swift test --parallel --filter <TargetTests.SuiteName>
   ```
3. **Thread Sanitizer (TSan) / Repetition Gate** (mandatory for any change touching `actor`, `AsyncSequence`, `Task`, `sending`, or shared state):
   ```bash
   swift test --parallel --sanitize=thread
   ```

---

## 4. Non-Negotiable Code & Test Constraints
- **Zero Code Comments (`CONTRIBUTING.md`)**: Do not add `//`, `///`, or `/* */` comments in Swift code; remove surrounding comments when modifying code. Verify with `python3 scripts/check_no_new_comments.py origin/main`.
- **Testing Standard**: Use the modern `Testing` module (`import Testing`, `@Suite`, `@Test`, `#expect`, `#require`). Never create new `XCTestCase` classes. No arbitrary `Task.sleep` in unit tests; use deterministic continuations or clock injection.
- **Memory & Ownership (Swift 6.4)**:
  - Prefer `InlineArray` and `Span` / `RawSpan` over heap arrays and unsafe pointers.
  - Prefer `UniqueBox` and `UniqueArray` (SE-0517/0527) for single-owner heap buffers to eliminate ARC and Copy-on-Write (CoW) overhead.
  - Use `borrow` / `mutate` accessors and `Ref` / `MutableRef` (SE-0519) for zero-copy views into noncopyable (`~Copyable`) or large value types.
  - Use `Iterable` (SE-0516) for borrowing iteration over noncopyable or large element sequences.
- **Concurrency & Lifecycle (Swift 6.4)**:
  - Use `defer { await cleanup() }` (SE-0493) for deterministic async resource teardown.
  - Wrap critical rollback/flush logic in `withTaskCancellationShield` (SE-0504).
  - Re-validate actor state after every `await` suspension point (actor reentrancy guard).
  - **Banned Primitives**: `DispatchQueue`, `DispatchSemaphore`, `NSLock` across async boundaries, `ObservableObject` / `@Published`, and unjustified `@unchecked Sendable` or `nonisolated(unsafe)`.

---

## 5. Pre-Completion Definition of Done (DoD)
1. Target architecture and module boundaries comply with `swift-64-architecture` and `ponytail`.
2. Zero warnings on `swift build -Xswiftc -warnings-as-errors` and zero new comments (`check_no_new_comments.py`).
3. All unit and parameterized `@Test` suites pass under `swift test --parallel`.
4. Self-review completed against `swift-64-pr-review` (5 Gates) and `ponytail-review`.
