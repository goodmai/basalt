---
name: xcode-mcp
description: Use Xcode 27's built-in MCP tools from basalt work — above all DocumentationSearch, to check a Swift 6.4 / stdlib / Foundation / Metal API against Apple's documentation before writing or reviewing code that uses it (InlineArray, Span/RawSpan, UniqueBox, Mutex, withTaskCancellationShield, typed throws, ~Copyable, Synchronization, Observation). Use whenever you are about to rely on memory for a Swift 6.2–6.4 API, an SE-proposal spelling, availability, or a framework signature; when a reviewer asks "does this API exist / what is its signature"; and when the swift-64-* skills tell you to apply a 6.4 tool. Also covers setup, agent approval and which Xcode MCP tools do not fit an SPM package.
---

# Xcode MCP tools in basalt

Xcode 27 ships an MCP server (`xcrun mcpbridge`, server name `xcode-tools`) that exposes Xcode's own tools to agents. On this machine it is enabled in headless mode and the repo folder is permitted (`xcrun mcp-server status`).

## Calling it

- **In a session that loaded `.mcp.json`**, the tools appear as `mcp__xcode__<Tool>`, for example `mcp__xcode__DocumentationSearch`.
- **In any other session, or from a script**, use the helper, which handles the handshake and agent approval:

  ```bash
  swift .claude/skills/xcode-mcp/scripts/xcode_mcp.swift list
  swift .claude/skills/xcode-mcp/scripts/xcode_mcp.swift <Tool> '<json arguments>' [timeout s]
  ```

Run it from the repo root. Xcode approves each launching process (zsh, `swift-frontend` for `swift script.swift`, the Claude Code process) for 24 hours. When the first call answers "This agent isn't approved", the helper calls `XcodeOpenWorkspace` on the current folder, which approves the agent, and retries. When it answers "Xcode is waiting for the user to approve this request", approve the pending agent from the Xcode MCP menu bar icon or with `sudo xcrun mcp-server approve <id>` (the id is in `xcrun mcp-server status`), then rerun.

## What to use it for

### 1. DocumentationSearch: check an API before relying on it
It searches Apple's documentation semantically, Swift stdlib included. Arguments: `query`, plus optional `frameworks` (an array). It needs no open workspace.

Use it before writing or approving code that uses a Swift 6.2–6.4 API from memory: spelling, generic parameters, availability, ownership (`borrowing`/`consuming`), whether something is `~Copyable`. The CONTRIBUTING toolset list is the usual trigger.

```bash
swift .claude/skills/xcode-mcp/scripts/xcode_mcp.swift DocumentationSearch '{"query":"InlineArray"}'
swift .claude/skills/xcode-mcp/scripts/xcode_mcp.swift DocumentationSearch '{"query":"withTaskCancellationShield"}'
swift .claude/skills/xcode-mcp/scripts/xcode_mcp.swift DocumentationSearch '{"query":"Mutex withLock","frameworks":["Synchronization"]}'
swift .claude/skills/xcode-mcp/scripts/xcode_mcp.swift DocumentationSearch '{"query":"RawSpan unsafeLoad"}'
swift .claude/skills/xcode-mcp/scripts/xcode_mcp.swift DocumentationSearch '{"query":"MTLResidencySet"}'
```

**Pass `frameworks` for a symbol.** A bare symbol query searches every framework and returns about 20 long documents. `{"query":"UniqueBox"}` put the stdlib type eleventh, behind AudioToolbox and AppKit boxes; `{"query":"UniqueBox","frameworks":["Swift"]}` returned one document. Use `["Swift"]` for the stdlib, `["Synchronization"]`, `["Foundation"]`, `["Metal"]` and so on. Ask for one symbol per call.

**What it does not hold.** The Metal Shading Language specification is a PDF that is not indexed: the search answers on `MTL*` APIs and the compute articles (threadgroups, SIMD groups, `[[thread_index_in_simdgroup]]`), not on MSL keywords, address-space rules or `simdgroup_matrix`. For shader text the Metal compiler decides: instantiate the kernel in a test.

Results already checked:
- `InlineArray`: the type can be spelled `[3 of Int]`, and every element must be initialized.
- `withTaskCancellationShield(operation:)`: child tasks do not inherit the shield, and `UnsafeCurrentTask.hasActiveCancellationShield` is for debugging only.
- Typed throws through closures: `Sequence.map`, `Dictionary.mapValues`, `Optional.map` and `Optional.flatMap` are `throws(E)`. `compactMap`, `flatMap` on sequences, `reduce` and `sorted(by:)` are still `rethrows`, so a `throws(BasaltError)` function that calls them with a throwing closure needs a loop.
- `UniqueBox` and `UniqueArray` are in the stdlib under "Manual Memory Management: Heap Storage".

When the documentation and the code disagree, the compiler decides: build with `swift build`. Quote the doc passage in the PR when a review turns on it.

### 2. Workspace tools
`XcodeOpenWorkspace` with the repo folder works (checked 2026-11-11): it returns a `workspaceIdentifier`, scheme `Basalt-Package`, destination "My Mac". Pass that identifier, or the repo's absolute path, as `workspaceIdentifier` to the other tools. `XcodeListWorkspaces` shows what is open.

| Tool | State on basalt |
|---|---|
| `XcodeRefreshCodeIssuesInFile` | works. `filePath` starts with the root folder's name: `mlx/Sources/BasaltCore/ComposeDiff.swift`. Without the prefix it fails with `SourceEditorCallableDiagnosticError error 5` |
| `XcodeGrep`, `XcodeGlob`, `XcodeLS`, `XcodeRead` | not needed; the normal file tools are faster |
| `BuildProject`, `GetBuildLog`, `GetTestList`, `RunSomeTests`, `RunAllTests`, `RunCodeSnippet`, `GetTargetBuildSettings` | not tried on basalt. Try one before relying on it and record the result here |

- Use `XcodeRefreshCodeIssuesInFile` for a quick diagnostic on one file while editing. It is not a gate.
- The gates stay on the command line, because CI runs them there: `swift build -Xswiftc -warnings-as-errors`, `swift test --parallel --filter …`.
- A worktree is a different folder: open it by its own path, and prefix file paths with its folder name.
- If `XcodeOpenWorkspace` times out ("hasResolvedInitialRunDestination … after 10.0s", seen before 2026-11), call it again; the `LSP` tool (sourcekit-lsp) covers definitions, references and diagnostics meanwhile.

### 3. Debugger
`InvokeDebuggerCommand` drives Xcode's active debug session. basalt runs from the command line, so there is usually none. For a crash in `swift test` or `basalt serve`, run `lldb` from the CLI (`lldb .build/debug/basalt -- serve …`, then `bt all`). `xcrun lldb-mcp` exists in Xcode 27 but has not been tried here.

## Do not use
- **Project-editing tools:** `XcodeWrite`, `XcodeUpdate`, `XcodeMV`, `XcodeRM`, `XcodeMakeDir`, `XcodeNewTarget`, `UpdateTargetBuildSetting`, `UpdateFileCompilerFlags`, `AddEntitlement`, `AddInfoPlist`. basalt is an SPM package whose layout lives in `Package.swift`, so edit files with the normal tools.
- **App, device and App Store tools:** the `DeviceInteraction*`, `RunProject`, `RenderPreview`, `GetTopCrashIssues` and `StringCatalog*` tools. basalt has no app target in Xcode.

## Setup (already done on this Mac)
```bash
sudo xcrun mcp-server enable
sudo xcrun mcp-server allow-folder --always /Users/alexeiboklag/projects/mlx
xcrun mcp-server status
```
`.mcp.json` at the repo root registers `xcode` → `xcrun mcpbridge` for Claude Code sessions. To reset: `sudo xcrun mcp-server clear-permissions`.

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
| `swift-concurrency-pro` | async/await, actors, tasks, streams and cancellation in depth |
| `swift-race-review` | races and deadlocks the compiler cannot see; MLX and Metal threading |
| `swift-testing-pro` | Swift Testing in depth: structure, async tests, new features |
| `basalt-model-runs` | runs that load model weights: benchmarks, ppl, memory, parity, GPU lock |
| `model-xray` | looking inside a running model and calling basalt's own MCP server |
