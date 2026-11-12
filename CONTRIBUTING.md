# Contributing to Basalt

## Toolchain

- Swift 6.4 (Xcode 27) and `swift-tools-version: 6.4`. Every target builds in the Swift 6 language mode, so data-race checking is complete; a target does not opt out.
- Zero warnings in our own targets. A warning is fixed, not silenced; `@diagnose` (SE-0522) lowers one only for code we do not control, and the commit message says why.
- **No new concurrency escape hatch:**
  - `@unchecked Sendable`, `nonisolated(unsafe)`;
  - `NSLock`, `NSCondition`, `NSRecursiveLock`, `os_unfair_lock`;
  - `DispatchQueue`, `DispatchSemaphore`;
  - `Task.detached`.

  Prefer a value type, an actor, `sending`, `Mutex` or an atomic. `Sources/` has no `@unchecked Sendable` and no `nonisolated(unsafe)`. A task that must outlive its caller without the actor's isolation is `Task { @concurrent in … }`, not `Task.detached`. The remaining sites (two raw `Thread`s around blocking reads and writes) are listed in `scripts/concurrency-hatches.allow`, one line per site: `path:symbol: why it is needed and what serializes it`.
- **The allowlist only shrinks.** `scripts/check_concurrency_hatches.swift` (`swift scripts/check_concurrency_hatches.swift`, CI: `.github/workflows/concurrency-hatches.yml`) lexes every file in `Sources/` like the comment checker, so comments and strings don't count, and fails on a hatch without an allowlist line (raw `Thread` included) and on an allowlist line that no longer matches. Removing a hatch removes its line in the same PR; a new line needs a reason the reviewer accepts.

## Types first

The type says what the value can be, so the compiler rejects what the code must never do.

- **Model state as an enum with associated values**, not as flags plus optionals. A request that is `queued`, `prefilling(tokens:)`, `decoding(cache:)` or `finished(reason:)` is one enum; four optionals that "can't all be set" are a bug the compiler cannot see.
- **Exhaustive `switch` over our own enums, without `default`.** A new case must fail to compile wherever it is not handled. `default` is for types we do not own; a non-frozen enum from another module gets `@unknown default`.
- **A distinct type for each kind of number or id that must not mix.** Token counts, byte counts, positions and ids of different things are separate structs (or `RawRepresentable` wrappers), not bare `Int` or `String`. A function that takes `TokenCount` cannot be handed bytes.
- **Typed throws** (`throws(BasaltError)`, `throws(Malformed)`): every thrown error is a case the caller can switch over. `any Error` only at the boundary to code that throws untyped.
- **Parse at the boundary, trust inside.** DTOs decode and validate once in the controller; the engine receives a type that can only hold a valid request, not a dictionary it re-checks.
- **Resources with one owner are `~Copyable`.** A lock, a cache lease, a GPU lock, a file handle is a non-copyable value whose `deinit` releases it, so it cannot be released twice or leaked (`consuming` hands it on, `borrowing` lends it). Use `UniqueBox` (SE-0517) when it must live on the heap and optional non-copyable values (SE-0532) instead of a separate `isHeld` flag.
- **Cleanup that must finish runs in a cancellation shield.** Persisting state on shutdown, flushing a log or releasing a lease goes in `withTaskCancellationShield` (SE-0504); async cleanup can `await` inside `defer` (SE-0493).
- **`some` over `any`.** Generic and opaque types are specialised and inlined; an existential is a box and a dynamic call. `any` is for collections of different concrete types and for API boundaries, not for hot paths.

## Performance

Speed is measured, not assumed. A change that claims to be faster names the benchmark (a micro-benchmark in an env-gated Swift test or `PerformanceBenchmark`) and the ratio before/after on this machine class; a change to a hot path that claims to be neutral says it was measured.

On the per-token path (sampling, logit processors, speculative drafting, constraint masks) and in byte parsing:

- **No allocation per token.** Reuse buffers; prefer `InlineArray` for small fixed sizes, `withTemporaryAllocation` (SE-0524) for scratch space, `UniqueArray` (SE-0527) for growable non-copyable storage.
- **No copies of large values.** Pass with `borrowing`/`consuming`; iterate by borrowing (`Iterable`, SE-0516); expose views as `Span`/`MutableSpan`, using `borrow`/`mutate` accessors (SE-0507).
- **No `unsafe` pointer code for bytes.** Parse files and headers through `RawSpan` loads and stores (SE-0525) and `Span`; an `Unsafe*Pointer` is only acceptable where a C API demands it, wrapped once.
- **Value types and `final` classes**; watch for copy-on-write on arrays that are shared and then mutated.
- **Keep MLX lazy.** Build the graph, evaluate once per step (`asyncEval` to overlap CPU and GPU), never call `.item()` or `asArray` inside a loop over tokens, and keep work off the GPU's critical path rather than blocking on it.
- **No actor hop per token.** Actors guard state between requests, not inside the decode loop.

## Smallest change that works

- **Reuse before building.** Check the standard library, the pinned dependencies (`.build/checkouts`: mlx-swift, mlx-swift-lm, swift-transformers, swift-collections, …) and upstream before writing a subsystem; the issue records the build-or-reuse decision.
- **No speculative abstraction.** No protocol with one conformer, no factory for one product, no configuration for a value that never changes, no scaffolding for later. Add the second implementation when it exists.
- **Delete over add.** The shortest correct diff wins; dead code, unused parameters and options nobody sets are removed.
- **Fix the cause where all callers meet**, not the symptom on the path the bug report names.
- **Opt-in until measured.** A behaviour that is not proven at least as fast and as exact as the current one ships off by default behind a flag, with its numbers in the PR.

## Size and complexity

Measured with lizard, which reports per function the cyclomatic complexity (CCN), the lines of code without blanks and comments, and the parameter count: `uvx lizard==1.24.0 -l swift -C 15 -L 60 -a 6 <files>`.

| Unit | Limit |
|---|---|
| Function | CCN ≤ 15 (aim for ≤ 10), ≤ 60 lines of code, ≤ 6 parameters |
| Type body (`struct`, `enum`, `class`, `actor`, `protocol`) | New type: ≤ 200 lines of code (a core type or a state type). A type that existed on 2026-11-04 aims for ≤ 450. Any type over 200 must not grow. |
| Extension body | ≤ 150 lines of code |
| File | ≤ 600 lines (aim for ≤ 400) |

- **CI enforces the ratchet**: `scripts/check_size_limits.swift` runs lizard on the base and head versions of every changed Swift file (`.github/workflows/size-limits.yml`). Run it locally with `swift scripts/check_size_limits.swift origin/main`; `--impact` adds the share of functions the change touched and the CCN sum and peak before and after, for the PR description.
- **Type and extension bodies are counted** in lines of code (comments, strings and blank lines don't count) by the same script. A type older than 2026-11-04 that is over 200 is legacy. It may stay up to 450 and only has to stop growing. Refactoring targets the legacy types over 450, which `swift scripts/check_size_limits.swift --inventory` lists. A type that needs to shrink splits into a value core plus one extension per concern, each extension within 150. It does not split into a second type that shares the same state.
- **New code stays within the limits.** Tests follow the function limits; a test file may be longer than 600 lines.
- **Code already over a limit does not get worse.** On 2026-11-04 that was 6 files over 600 lines, 20 functions over CCN 15 and 20 over 60 lines (#284). A change that touches one of them must not make it longer or more complex. A change that adds a branch to one first splits it.
- **Split along the seams the code already has.** A `switch` case or an engine mode becomes its own function or type, and a stage of a long function becomes a method named after what it does. Cutting a function into `part1`/`part2` to get under the number does not count.
- **Use the Swift 6.4 tools when touching code.**
  - A lock or `@unchecked Sendable` becomes `Mutex` or an actor.
  - A resource that is released by hand becomes a `~Copyable` value.
  - Teardown gets `withTaskCancellationShield` and `defer { await … }`.
  - An internal `throws` becomes typed.
  - Byte parsing moves to `RawSpan`.

  The rules are in "Types first" and "Performance" above. Leave the code you touch closer to them.

## Value core, actor shell

Thread safety comes from where state lives, not from locks.

- **Core: values and functions.** Decisions, parsing, scheduling policy, cache geometry, routing scores, request validation and sampling math are `struct`s and `enum`s with pure functions, or `mutating` methods on a value the caller owns. A core type is `Sendable` because its stored properties are values, so it needs no `@unchecked`, no lock and no actor. A test calls it directly with no engine, server or clock.
- **Shell: actors and protocols.** Anything with identity or effects is an `actor`, or a type that one actor owns: a loaded model, a prompt cache, a session log, a socket, a file, the GPU lock, a cloud client. The shell takes a request, asks the core what to do, does it, and stores the new value. Dependencies cross the boundary as protocols with one narrow job (`InferenceEngine`, a clock, a store), so tests swap the effect, not the logic.
- **One owner per mutable state.** A piece of mutable state belongs to exactly one actor or one value. Everyone else gets a copy, a `sending` transfer or a message. A `final class` with mutable fields is acceptable only inside one actor's isolation, as with MLX caches and model weights, and never escapes it.
- **The shell stays thin.** An actor method reads its state, calls the core and writes the result. It does not hold logic of its own. Logic that grows inside an actor moves into the core value, and that is also how a type over 200 lines splits.
- **State changes are synchronous reducers.** Every state change is a synchronous `state = Core.reduce(state, event)`; effects run before or after it, and a result that crosses an `await` carries the epoch it was started under. A result that comes back after an `await` is applied only if the epoch or generation token it carries still matches, as the engine does with `modelEpoch`. Check-then-act across an `await` is a race even inside an actor.
- **Shared state is released by its owner, not by a `deinit`.** A lease, pin or slot that changes another type's state is given back with a `consuming` method called on the owning actor. A `deinit` runs on whichever thread drops the value last, so it must not touch shared state unless the compiler keeps the value on its owner. `PromptCacheLease` still unpins in `deinit`, but neither it nor `PromptCacheNode` is `Sendable`, so the lease can only be dropped on the engine actor (#300).
- **MLX objects are shell state.** `MLXArray`, `KVCache`, `PromptCache` and `LanguageModel` are not `Sendable`. Model-wide MLX state lives in the mlx-swift-lm `ModelContext` next to the model: the MTP drafter (`mtpDrafter`) and the batch state (`batch`, a `BatchCore`) are created in `container.update` and read only inside `container.perform`. A per-request value crosses into the model through `container.performSending(nonSendable:)` and comes back as a `sending` result (`GenerationLaunch`); the engine actor keeps only `Sendable` handles, such as `BatchedDecoder`'s `Mutex`-guarded mailbox. Two forwards of one model never run at once: today `ModelEvaluationLock` enforces this, and every model call takes it (batched rows take it and step the whole batch themselves; there is no decode thread). #300 replaces it with one global model actor, so the type system enforces it.

## No comments in code

Code carries no comments: no `//`, `///`, `/* */`, no `// MARK:`, no commented-out code. Names, types and small functions say what the code does; tests say what it must keep doing.

The *why* lives where history keeps it, next to the change it explains:

- **Commit message** — the reason for the change, the alternative that was rejected, the incident or measurement behind a constant. `git blame` leads every line to it.
- **Pull request description** — design, trade-offs, what was verified and how, what was not.
- **Issue** — the problem, the plan, deferred work.
- **Tests** — the test name states the behaviour; a regression test names the bug it pins.

When a line seems to need a comment, change the code instead: extract a function whose name is the comment, name the constant, introduce a type, or write the test.

Only directives the toolchain reads are allowed: `// swift-tools-version:` on the first line of `Package.swift` and the `#!` line of a script.

Existing comments are removed when the code around them changes; a change does not add new ones — and a line that is moved or re-indented counts as added, so its old comment goes with the move. CI enforces this for Swift on every pull request (other languages follow the same rule by review): `scripts/check_no_new_comments.swift` (`swift scripts/check_no_new_comments.swift origin/main`) lexes each changed Swift file (strings, multi-line and raw strings, interpolation, nested block comments, `/regex/` and `#/regex/#` literals) and fails when a comment lands on an added line. A renamed file counts as added in full.

## Tiny models and kernel admission

Unit tests run real model code with random weights, not mocks. `TinyModel.make(_:)` (`Tests/BasaltTests/Fixtures/TinyModels.swift`) builds every fixture the same way: a shrunk real config, a fixed seed, `prepare()` and `eval` as the loader does. A test that needs another architecture adds a `TinyModel.Family` case there, with a greedy and a prompt-cache test in `TinyModelFixtureTests`, rather than a builder of its own.

A custom kernel or a new decode path lands only with:

- a parity test against the stock MLX graph on a tiny fixture: exact for token ids and expert indices, within a stated float tolerance for logits and weights;
- no mocked engine below `BasaltServer`. `MockInferenceEngine` is for REST and controller tests.

## Real-model runs

Anything that loads a real model — a benchmark, a quality or memory measurement, a passkey check, a parity run — runs from Swift: `PerformanceBenchmark`, a Swift test gated on `BASALT_E2E*` environment variables, or a `scripts/*.swift` runner. A shell script may only launch it; it does not loop over models, parse output or decide pass/fail.

The runner:

- checks the model snapshot is complete before loading it (`basalt models inspect <model>`), and fails instead of measuring a half-downloaded model;
- writes one timestamped line per step — model, mode, parameters, result — to a log file under `logs/` and to stdout;
- logs every failure (a thrown error, a failed precondition, running out of memory, a missing snapshot) with the step it happened in, and exits non-zero when any step failed; a run that skips a step says so in the log;
- records the commit, the model repo and revision, and the machine (chip, memory) next to the results;
- holds the GPU for one model at a time, so parallel runs on one Mac don't skew each other's numbers.

Shell pipelines lose exit statuses and stderr, `tail` hides a crash in the middle of a log, and a killed wrapper leaves an orphaned process holding the GPU. A Swift runner keeps the error next to the result it invalidates.

Measure more than one model family. A claim about speed or quality names the models it was measured on; a feature that disables itself on some architectures (hybrid SSM, sliding-window, multimodal) is also run on one of those to show the fallback gives the same output.

## Coverage

- **Changed lines are tested.** The Unit Tests job builds and runs the tests with `--enable-code-coverage`, then `scripts/check_changed_coverage.swift` compares the `llvm-cov export` JSON with the diff against the PR base. It lists, per changed file in `Sources/`, the added executable lines no test ran, and fails when fewer than 60% of them ran (#316). 60% leaves room for lines a unit test cannot reach (CLI and UI glue, Metal-only and real-model-only branches); it rises as PRs show the usual numbers. Locally:

  ```bash
  swift test --enable-code-coverage
  swift scripts/check_changed_coverage.swift "$(swift test --show-codecov-path)" origin/main
  ```
- **Mutation testing is on demand, not in CI.** Each mutant is a rebuild and a test run, so run it on the files a change is about, when a suite's strength is in question:

  ```bash
  brew install muter-mutation-testing/formulae/muter
  muter init
  muter --files-to-mutate Sources/BasaltCore/ToolCallGate.swift
  ```

  `muter init` writes `muter.conf.yml` (keep it local; check that it runs `swift test`). A surviving mutant is a missing assertion: name it in the PR or add the test.

## Pull requests

- One issue per PR; link it with `Closes #N`.
- No function or file over the limits in "Size and complexity", and none already over them grown.
- `swift build --build-tests` and `swift test` pass locally with no new warnings (run `./scripts/build_metal.swift --skip-unsupported` once if MLX reports a missing metallib). Run the full suite with `SWT_EXPERIMENTAL_MAXIMUM_PARALLELIZATION_WIDTH` set to half the cores, as CI does: MLX evaluates under one mutex, and more parallel tests only park cooperative threads on it until live-server tests hit their time limits (#427).
- A test that guards against nondeterminism (sampling parity, speculative exactness, concurrency) runs with repetitions (ST-0024: `swift test --maximum-repetitions 10 --repeat-until fail --filter <SuiteName>`) before it is trusted.
- Behaviour changes come with a test that fails without the change; the changed-line coverage step (see Coverage) checks the tests run it.
- No new concurrency hatch outside `scripts/concurrency-hatches.allow`, and no stale line left in it.
- Claims that need a real model (speed, output quality) are marked as unverified in the PR unless they were measured; measured ones name the models and link the runner's log (see Real-model runs).
