# Contributing to Basalt

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
- `swift build --build-tests` and `swift test` pass locally (run `./scripts/build_metal.swift --skip-unsupported` once if MLX reports a missing metallib).
- Behaviour changes come with a test that fails without the change.
- Claims that need a real model (speed, output quality) are marked as unverified in the PR unless they were measured; measured ones name the models and link the runner's log (see Real-model runs).
