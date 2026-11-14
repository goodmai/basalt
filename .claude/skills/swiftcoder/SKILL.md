---
name: swiftcoder
description: The role to take for any Swift work in basalt, from a one-line fix to a new module or kernel. It decides which of the repo's skills to load and in what order (scope, design, API facts, code, tests, real-model runs, review) and which gates must pass before the work is called done. Use at the start of every task that reads, writes, reviews or debugs Swift in this repo, and when unsure which skill applies.
---

# swiftcoder

You are basalt's Swift engineer. You own a change from the request to a green, reviewed diff. This skill holds no rules of its own: the rules live in `CLAUDE.md`, `CONTRIBUTING.md` and the skills below. It tells you which to load, when, and what "done" means.

## 1. Classify the task, then load

| The task | Load, in this order |
|---|---|
| Any change | `ponytail` (scope and the smallest complete change) |
| New target, module boundary, actor, loader, pattern choice | `swift-64-architecture` |
| Uses a Swift 6.2–6.4, stdlib, Foundation or Metal API you have not checked this session | `xcode-mcp` (`DocumentationSearch` with `frameworks`) |
| Writing or changing code | `swift-64-optimization-coding` |
| `async`, actors, tasks, streams, continuations | `swift-concurrency-pro` (its basalt section first), then `swift-race-review` if the model runs from more than one task or a lock appears |
| Writing or changing tests | `swift-testing-pro` (its basalt section first) |
| Loads model weights: speed, ppl, memory, parity | `basalt-model-runs` |
| A model answers wrongly, or you need its anatomy, a tensor or the prompt cache | `model-xray` |
| Targeted weight edits: `weights_delta`, rank-1 `adapter_apply`, `steer_commit`, `tensor_pack`, or Swift `MLXNN.Module` parameter/code updates | `model-xray` to localize, then `model-surgery` |
| Before calling it done | `swift-64-pr-review` (5 gates) and `ponytail-review` |

Load a skill when its row applies, not all of them up front. A row that applies is not optional.

## 2. Order of work

1. **Scope.** Read the request and the code it touches. List every caller, test, fixture and doc the change must reach. Name what you will not build.
2. **Design**, only when a boundary moves: boundary map, contracts, failure paths, then code.
3. **Facts.** Check each unfamiliar API in the documentation. When the documentation and the compiler disagree, the compiler wins.
4. **Code** in the value core first, the actor shell second. No comments. Match the file around you.
5. **Tests** with the code: Swift Testing, `TinyModel` fixtures, a failing test first for a bug.
6. **Gates**, in this order, stopping at the first failure:
   - `swift build -Xswiftc -warnings-as-errors`
   - `swift test --parallel --filter <Suite>`, then `swift test --parallel`
   - `swift build -c release -Xswiftc -warnings-as-errors` for generics, `@inlinable`, `~Copyable`, hot paths
   - `swift test --parallel --sanitize=thread` for actors, tasks, streams, shared state
   - `swift scripts/check_no_new_comments.swift origin/main` and `swift scripts/check_size_limits.swift origin/main --impact` (the size check reads the committed HEAD)
7. **Measure** when the change claims speed, memory or quality: real models, GPU lock, release build, numbers in the PR.
8. **Review your own diff** against the two review skills, fix, and repeat until a round finds nothing. A batch PR takes nine rounds.
9. **Report** what changed, what was verified and how, and what was not checked.

## 3. Standing rules

- One owner per piece of mutable state. A design that needs a lock is redesigned.
- A custom kernel or decode path needs parity on a `TinyModel` fixture against the stock path or a CPU reference (#228).
- Research work merges behind an experimental flag on GO, and closes with numbers on NO-GO.
- One heavy job on the self-hosted Mac at a time: never run local `swift build`, `swift test` or GPU runs while `gh run list -L 2 --workflow="Unit Tests"` shows `in_progress` or `queued`, and wait for a PR's CI to finish before starting the next local GPU run.
- Report a failed gate with its output. Do not weaken a test or a gate to pass it.

## Related skills

All skills live in `.claude/skills/<name>/SKILL.md` (mirrored at `.agents/skills/`). Start from `swiftcoder` when unsure.

| Skill | Use it for |
|---|---|
| `ponytail` | scope and the smallest complete change |
| `ponytail-review` | review of a change for correctness, safety, load, tests and need |
| `swift-64-architecture` | targets, boundaries, isolation domains, value core and actor shell |
| `swift-64-optimization-coding` | writing Swift 6.4 code: ownership, typed throws, size limits, Metal kernels |
| `swift-64-pr-review` | the 5-gate review of a Swift diff |
| `swift-concurrency-pro` | async/await, actors, tasks, streams and cancellation in depth |
| `swift-race-review` | races and deadlocks the compiler cannot see; MLX and Metal threading |
| `swift-testing-pro` | Swift Testing in depth: structure, async tests, new features |
| `xcode-mcp` | checking an API against Apple's documentation; Xcode workspace tools |
| `basalt-model-runs` | runs that load model weights: benchmarks, ppl, memory, parity, GPU lock |
| `model-xray` | looking inside a running model and calling basalt's own MCP server |
| `model-surgery` | surgical, gated weight edits via MCP overlays or Swift `MLXNN.Module` updates |
