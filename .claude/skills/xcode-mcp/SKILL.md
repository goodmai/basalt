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

Two results already checked:
- `InlineArray`: the type can be spelled `[3 of Int]`, and every element must be initialized.
- `withTaskCancellationShield(operation:)`: child tasks do not inherit the shield, and `UnsafeCurrentTask.hasActiveCancellationShield` is for debugging only.

When the documentation and the code disagree, the compiler decides: build with `swift build`. Quote the doc passage in the PR when a review turns on it.

### 2. Workspace tools (not usable on basalt yet)
These tools need an open workspace:
- `XcodeRefreshCodeIssuesInFile`, compiler diagnostics for one file;
- `RunCodeSnippet`, runs a snippet in the context of a file;
- `GetTestList` / `RunSomeTests` / `RunAllTests`;
- `BuildProject` / `GetBuildLog`;
- `GetTargetBuildSettings`.

On basalt, `XcodeOpenWorkspace` with the repo folder currently fails with "timed out waiting for hasResolvedInitialRunDestination … after 10.0s": headless Xcode does not settle a run destination for the SPM package in time.

Until that works:
- build with `swift build -Xswiftc -warnings-as-errors`;
- test with `swift test --parallel --filter …`;
- use the `LSP` tool (sourcekit-lsp) for definitions, references and diagnostics.

To retry, call `XcodeOpenWorkspace '{"path":"<repo>"}'`, then `XcodeListWorkspaces` to get the `workspaceIdentifier` the other tools take. If it succeeds, update this section.

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
