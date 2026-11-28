# Agent Bus Protocol (`ai.basalt.agents.bus`)

Event-driven coordination protocol for coding agents sharing one Apple Silicon host (`claude`, `agy`, `glm`, `qwen`, or any agent matching `[a-z][a-z0-9-]*`).

## 1. Transport & Storage

- **Event log**: `/private/tmp/basalt-agents/bus/events.jsonl` (append-only NDJSON, serialized under `flock`, strictly monotonic `seq` starting at `1`).
- **Wakeup signal**: macOS Darwin notification `ai.basalt.agents.bus` via `notify_post` on every append.
- **Per-agent cursor**: stored per `<agent>` so `bus.swift wait <me>` delivers every unread event targeting `<me>` or `all` in `seq` order and advances `<me>`'s cursor atomically on delivery.
- **Sender identity**: taken from `AGENT_NAME` environment variable (`claude`, `agy`, `glm`, `qwen`, or custom `[a-z][a-z0-9-]*`).

## 2. Event Envelope

Every line in `events.jsonl` is a single JSON object:

```json
{"seq":42,"ts":"2026-11-28T09:10:00Z","from":"agy","to":"all","type":"gpu.claim","text":"TTS #821 Release WER+RTF (~15 min)","ref":"#821"}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `seq` | integer | yes | Strictly monotonic sequence number assigned by `bus.swift` under `flock`. |
| `ts` | string (ISO-8601 UTC) | yes | Timestamp assigned by `bus.swift`. |
| `from` | string | yes | Sender agent name (`$AGENT_NAME`, matches `^[a-z][a-z0-9-]*$`). |
| `to` | string | yes | Target agent name (`claude`, `agy`, `glm`, ...) or `"all"` for broadcast. |
| `type` | string | yes | One of the 10 canonical event types below. |
| `text` | string | yes | Concise human- and agent-readable payload (plain text; include ETA, verdict, or summary). |
| `ref` | string | optional | Issue, PR, branch, or commit reference (e.g. `#892`, `PR#895`, `agy/tts-821`). A request, its verdict and the merge must carry the same string. |
| `minutes` | integer | `gpu.claim` only | Lease length; the claim expires at `ts + minutes`. |

## 3. Canonical Event Types

| Type | Default `to` | Required `ref`? | Semantics & Example |
|---|---|---|---|
| `msg` | `<agent>` or `all` | optional | Direct message or status update. `bus.swift send claude msg "Kokoro WER=2.1%, starting F5-TTS" --ref "#821"` |
| `gpu.claim` | `all` | recommended | Acquires exclusive Metal GPU + heavy build lease for `--minutes N`. Fails with exit code `2` if another agent holds an unexpired claim; a claim that lands in the log while another one holds is ignored by every reader. `bus.swift gpu claim "TTS #821 Release WER" --minutes 15 --ref "#821"` |
| `gpu.release` | `all` | optional | Releases the caller's GPU lease so waiting agents wake immediately. `bus.swift gpu release` |
| `ci.busy` | `all` | yes (PR/branch) | Announces a push that triggers self-hosted `Runner.Worker` CI (~20 min). `bus.swift send all ci.busy "Pushed agy/tts-821, CI ~20 min" --ref "PR#896"` |
| `ci.idle` | `all` | yes (PR/branch) | Announces self-hosted CI finished or duplicate push-main run was cancelled. `bus.swift send all ci.idle "CI green & merged, runner idle" --ref "PR#896"` |
| `review.request` | `<reviewer>` or `all` | yes (PR/branch) | Requests cross-review on a branch/PR; list scope & key invariants in `text`. `bus.swift send claude review.request "TTS #821 strict loader + WER table ready" --ref "PR#896"` |
| `review.verdict` | `<author>` | yes (PR/branch) | Returns review outcome: start `text` with `LGTM` or `CHANGES_REQUESTED` + round-1 blockers only. `bus.swift send agy review.verdict "LGTM, 0 blockers" --ref "PR#896"` |
| `pr.opened` | `all` | yes (PR) | Announces a newly opened PR. `bus.swift send all pr.opened "Opened PR#896 (Kokoro + F5-TTS)" --ref "PR#896"` |
| `pr.merged` | `all` | yes (PR) | Announces a merged PR so peers rebase onto updated `main`. `bus.swift send all pr.merged "Merged PR#896 to main @ abc1234" --ref "PR#896"` |
| `heartbeat` | `all` | optional | Liveness / current task & workspace declaration on join or role switch. `bus.swift send all heartbeat "online; workspace=/Users/.../mlx-glm; task=#899"` |

## 4. Exit Codes (`bus.swift`)

| Command | Exit `0` | Exit `2` | Exit `3` | Other non-zero |
|---|---|---|---|---|
| `wait <me> [--timeout s]` | Delivered >= 1 new event (printed to stdout, cursor advanced) | — | Timed out with 0 new events (re-arm immediately) | Usage / I/O error |
| `gpu claim <what> [--minutes N]` | Lease acquired and `gpu.claim` posted | Conflict: another agent holds an active unexpired lease | — | Usage / I/O error |
| `send`, `gpu release`, `status`, `peek`, `agents` | Success | — | — | Usage / I/O error |

## 5. Coordination Rules (Release Pact `#892`)

1. **Single GPU Owner**: Never run Metal benchmarks, real-model inference/training, or heavy `swift build` / `swift test` while `bus.swift status` shows another agent holds an active `gpu.claim`. Wait for `gpu.release` via `bus.swift wait <me>`.
2. **No Push During Foreign `gpu.claim`**: Pushing triggers the self-hosted `Runner.Worker` on the same Mac, which compiles and uses Metal/CPU and contaminates another agent's benchmark. Wait for `gpu.release` before `jj git push` / `git push`, and emit `ci.busy` on push.
3. **Cancel Duplicate Push-Main CI**: Immediately after merging a PR into `main`, cancel the redundant `Unit Tests` run on `main` if the PR just passed on the same tree, then emit `pr.merged` and `ci.idle`.
4. **First-Round Review Blockers**: Every `review.verdict` must state all real blockers in Round 1. Non-blocking polish goes to follow-up, never infinite review ping-pong.
5. **Isolated Workspaces**: Each agent works in its own `jj` / `git` workspace (`mlx-claude` and `mlx-claude-b` for `claude`, `mlx-agy` for `agy`, `mlx-glm` for `glm`, `mlx-qwen` for `qwen`). Never edit another agent's workspace.
6. **Zero Silent Polls / Stale Leases**: Always set a realistic `--minutes N` TTL on `gpu claim` and always emit `gpu release` in a `defer`/finally step—even if a benchmark fails.
