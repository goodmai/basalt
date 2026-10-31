# Contributing to Basalt

## Toolchain

- Swift 6.4 (Xcode 27) and `swift-tools-version: 6.4`. Every target builds in the Swift 6 language mode, so data-race checking is complete; a target does not opt out.
- Zero warnings in our own targets. A warning is fixed, not silenced; `@diagnose` (SE-0522) lowers one only for code we do not control, and the commit message says why.
- No `@unchecked Sendable` and no `nonisolated(unsafe)` in new code. Prefer a value type, an actor, `Mutex` or an atomic; an existing escape hatch is removed when the code around it changes.

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

## No comments in code

Code carries no comments: no `//`, `///`, `/* */`, no `// MARK:`, no commented-out code. Names, types and small functions say what the code does; tests say what it must keep doing.

The *why* lives where history keeps it, next to the change it explains:

- **Commit message** — the reason for the change, the alternative that was rejected, the incident or measurement behind a constant. `git blame` leads every line to it.
- **Pull request description** — design, trade-offs, what was verified and how, what was not.
- **Issue** — the problem, the plan, deferred work.
- **Tests** — the test name states the behaviour; a regression test names the bug it pins.

When a line seems to need a comment, change the code instead: extract a function whose name is the comment, name the constant, introduce a type, or write the test.

Only directives the toolchain reads are allowed: `// swift-tools-version:` on the first line of `Package.swift` and the `#!` line of a script.

Existing comments are removed when the code around them changes; a change does not add new ones — and a line that is moved or re-indented counts as added, so its old comment goes with the move. CI enforces this for Swift on every pull request (other languages follow the same rule by review): `scripts/check_no_new_comments.py` lexes each changed Swift file (strings, multi-line and raw strings, interpolation, nested block comments, `/regex/` and `#/regex/#` literals) and fails when a comment lands on an added line. A renamed file counts as added in full.

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

## Pull requests

- One issue per PR; link it with `Closes #N`.
- `swift build --build-tests` and `swift test` pass locally with no new warnings (run `./scripts/build_metal.swift --skip-unsupported` once if MLX reports a missing metallib).
- A test that guards against nondeterminism (sampling parity, speculative exactness, concurrency) runs with repetitions (ST-0024: `swift test --maximum-repetitions 10 --repeat-until fail --filter <SuiteName>`) before it is trusted.
- Behaviour changes come with a test that fails without the change.
- Claims that need a real model (speed, output quality) are marked as unverified in the PR unless they were measured; measured ones name the models and link the runner's log (see Real-model runs).
