# Contributing to Basalt

## No comments in code

Code carries no comments: no `//`, `///`, `/* */`, no `// MARK:`, no commented-out code. Names, types and small functions say what the code does; tests say what it must keep doing.

The *why* lives where history keeps it, next to the change it explains:

- **Commit message** — the reason for the change, the alternative that was rejected, the incident or measurement behind a constant. `git blame` leads every line to it.
- **Pull request description** — design, trade-offs, what was verified and how, what was not.
- **Issue** — the problem, the plan, deferred work.
- **Tests** — the test name states the behaviour; a regression test names the bug it pins.

When a line seems to need a comment, change the code instead: extract a function whose name is the comment, name the constant, introduce a type, or write the test.

Only directives the toolchain reads are allowed: `// swift-tools-version:` in `Package.swift` and the `#!` line of a script.

Existing comments are removed when the code around them changes; a change does not add new ones — and a line that is moved or re-indented counts as added, so its old comment goes with the move. CI enforces this on every pull request: `scripts/check_no_new_comments.py` lexes each changed Swift file (strings, multi-line and raw strings, interpolation, nested block comments, regex literals) and fails when a comment lands on an added line.

## Pull requests

- One issue per PR; link it with `Closes #N`.
- `swift build --build-tests` and `swift test` pass locally (run `./scripts/build_metal.swift --skip-unsupported` once if MLX reports a missing metallib).
- Behaviour changes come with a test that fails without the change.
- Claims that need a real model (speed, output quality) are marked as unverified in the PR unless they were measured.
