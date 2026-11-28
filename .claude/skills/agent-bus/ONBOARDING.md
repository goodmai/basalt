# Agent Bus Onboarding (`claude`, `agy`, `glm`, `qwen`)

How each agent harness connects to the macOS Darwin-notification event bus (`ai.basalt.agents.bus`), stays awake with sub-second latency, and follows the `#892` multi-agent pact.

## 1. Agent Roster & Workspace Isolation

| Agent Name (`AGENT_NAME`) | Model / Harness | Dedicated Workspace |
|---|---|---|
| `claude` | Claude Code | `~/projects/mlx-claude`, `~/projects/mlx-claude-b` |
| `agy` | Antigravity CLI (Gemini) | `~/projects/mlx-agy` |
| `glm` | GLM 5.4 (3rd agent slot) | `~/projects/mlx-glm` (dedicated `jj` workspace) |
| `qwen` | Local Qwen3.8 served by basalt on this Mac (4th agent slot) | `~/projects/mlx-qwen` (dedicated `jj` workspace) |

Any additional agent may join using a lowercase identifier matching `^[a-z][a-z0-9-]*$`.

## 2. First-Time Join Checklist (for `glm` or any new agent)

1. **Set identity**: prefix every bus command with `AGENT_NAME=glm` (or export `AGENT_NAME=glm` in your shell).
2. **Verify bus health**:
   ```bash
   BUS=.claude/skills/agent-bus/bus.swift
   swift "$BUS" status
   swift "$BUS" agents
   ```
3. **Announce presence (first `heartbeat`)**:
   ```bash
   AGENT_NAME=glm swift "$BUS" send all heartbeat "online; workspace=mlx-glm; ready for assignment"
   ```
4. **Read unread context**:
   ```bash
   AGENT_NAME=glm swift "$BUS" peek glm
   ```
5. **Arm your background listener** (Section 3 below) *before* starting work so you react to `gpu.release`, `review.request`, or owner redirects in <1 second.

## 3. Harness Wakeup Recipes

### A. Antigravity CLI (`agy`) — Reactive Wakeup via Background `run_command`

Antigravity does **not** need polling loops or 5-minute `schedule` timers. Its `run_command` tool automatically wakes the agent with a high-priority `<SYSTEM_MESSAGE>` the moment a background command exits.

1. **Arm the waiter** with a short sync wait (`WaitMsBeforeAsync: 500`) so it backgrounds immediately:
   ```bash
   AGENT_NAME=agy swift .claude/skills/agent-bus/bus.swift wait agy --timeout 600
   ```
   - Call `run_command` with `WaitMsBeforeAsync: 500`, `IsDaemon: false`, `NotificationTimeoutSeconds: 660`.
   - Continue working on non-conflicting tasks (editing code, reading files) or end your turn to sleep at zero token cost.
2. **Instant wakeup**:
   - The moment `claude` or `glm` posts an event to `agy` or `all`, `notify_post("ai.basalt.agents.bus")` wakes `bus.swift wait agy`.
   - `bus.swift` prints the new event(s), one line each, advances `agy`'s cursor, and exits with code **`0`**.
   - Antigravity immediately wakes `agy` with the command output in a `<SYSTEM_MESSAGE>`.
3. **Handle & re-arm**:
   - **Exit `0`**: Process the event(s) (e.g., respond to `review.request`, claim GPU after `gpu.release`), then **immediately launch a fresh background `bus.swift wait agy --timeout 600`** so you are never deaf.
   - **Exit `3`** (timeout, no new events): Simply re-arm `bus.swift wait agy --timeout 600` (optional `heartbeat` if working on a long task).
   - **Rule**: Keep exactly **one** active `bus.swift wait agy` background task running at all times during a multi-agent session.

### B. Claude Code (`claude`) — Streaming `watch` or Background `wait`

- Prefer `Monitor` / background streaming:
  ```bash
  AGENT_NAME=claude swift .claude/skills/agent-bus/bus.swift watch claude
  ```
- Or single-shot `wait claude --timeout 600` re-armed on exit `0` / `3`.

### C. GLM 5.4 (`glm`) — Universal Background `wait` Loop

- If the GLM 5.4 harness supports async/background command callbacks (like Antigravity), launch `AGENT_NAME=glm swift "$BUS" wait glm --timeout 600` as a background task and re-arm on exit `0` or `3`.
- If the harness only supports synchronous shell calls, run `AGENT_NAME=glm swift "$BUS" wait glm --timeout 30` between work chunks, or `AGENT_NAME=glm swift "$BUS" peek glm` before every GPU/push step.

### D. Local Qwen3.8 (`qwen`) — a model on the shared GPU

- `qwen` is served by basalt on this Mac, so it occupies the GPU for as long as it is loaded. Its harness runs `gpu claim "qwen agent loaded" --minutes <N>` before loading the model and `gpu release` after unloading it.
- When another agent asks for the GPU with a `msg`, `qwen` finishes its current step, unloads, releases, and waits for the next `gpu.release` before loading again.
- Its harness wakes it the same way as recipe C: a background `wait qwen --timeout 600`, re-armed on exit `0` or `3`.

## 4. Operating Rules (`#892` Pact)

1. **Before any GPU / heavy build / benchmark**:
   ```bash
   AGENT_NAME=<me> swift "$BUS" gpu claim "<task description>" --minutes <N> --ref "#<issue>"
   ```
   - If exit code is **`2`**, another agent holds the GPU. Do **not** build, test, or run MLX workloads; wait for `gpu.release` via `wait <me>`.
2. **Immediately after GPU work finishes (even on failure)**:
   ```bash
   AGENT_NAME=<me> swift "$BUS" gpu release
   ```
3. **Before `jj git push` / `git push`**:
   - Run `swift "$BUS" status` to confirm no other agent holds `gpu.claim` and CI is not mid-benchmark.
   - After pushing, emit `ci.busy`; after merge (and cancelling duplicate push-main CI), emit `pr.merged` and `ci.idle`.
4. **Cross-review discipline**:
   - Post `review.request` with branch/PR ref and key invariants.
   - Reviewer posts `review.verdict` with **all blockers in Round 1** (`LGTM` or `CHANGES_REQUESTED`).

## 5. Strictly Forbidden (DO NOTs)

- **DO NOT** run `swift build`, `swift test`, `PerformanceBenchmark`, or model inference while another agent holds an active `gpu.claim`.
- **DO NOT** push to GitHub while another agent holds `gpu.claim` (self-hosted `Runner.Worker` shares this Mac's CPU/Metal).
- **DO NOT** edit another agent's workspace directory or touch `Package.swift` / `Package.resolved` without explicit owner/peer agreement.
- **DO NOT** merge before CI is green and the reviewer's `review.verdict` is `LGTM`. A harness that cannot run the merge command asks another agent to merge with a `msg`, and checks that it happened.
- **DO NOT** leave a dead `gpu.claim` hanging or ignore `review.request` for >5 minutes—always keep `bus.swift wait <me>` armed.
- **DO NOT** use `sleep` loops to poll `events.jsonl`—always block on Darwin notifications via `bus.swift wait` or `watch`.
