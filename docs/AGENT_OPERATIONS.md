# AGENTS.md

Autonomous Agent Operations & Lifecycle Guide for `basalt`

This document defines the deterministic protocol for external AI agents (Claude Code, Cursor, Codex, OpenCode, Antigravity) interacting with the local MLX inference engine and dual-transport server.

---

## 1. Fast-Path Protocol for Agents

Agents MUST execute operations in the following deterministic sequence:

```mermaid
flowchart TD
    A0["Phase 1: Hardware Diagnosis<br/>basalt fit --check"] --> A["Fit check<br/>basalt models check &lt;modelId&gt; --json"]
    A --> B{"Feasible?<br/>(exit status 0)"}
    B -- Yes --> C["Phase 2: Weight Inspection<br/>basalt models inspect &lt;modelId&gt;"]
    B -- No --> X["Select smaller model / Warn user"]
    C --> D{"Intact &amp; Verified?"}
    D -- Yes --> E["Phase 3: Launch MCP Server<br/>basalt serve --mcp"]
    D -- No --> F["Download / re-fetch shards<br/>basalt models download &lt;modelId&gt;"]
    F --> C
    E --> G["Phase 4: Agent Control via MCP<br/>status | inspect_model | check_hardware | switch_model"]
    G --> H["Phase 5: Golden Vector Verification<br/>temperature = 0 greedy bit-exact parity"]
```

---

## 2. CLI Diagnostics & Inspection

### 2.1 Hardware Diagnosis (`--check`)
Without downloading or loading any model weights, query the host system's hardware capability:

```bash
# Output machine-readable JSON via SystemProfiler
basalt fit --check

# Alternative equivalent diagnostic
basalt models check --system --json
```

**JSON Output Schema:**
```json
{
  "availableRAMBytes": 25769803776,
  "availableRAMGB": 24.0,
  "chip": "M4",
  "compatible": true,
  "cpuCores": 10,
  "diskSpaceBytes": 104857600000,
  "diskSpaceGB": 97,
  "gpu": "Apple M4",
  "gpuMemoryGB": 24,
  "osVersion": "Version 15.0 (Build 24A335)",
  "totalRAMBytes": 34359738368,
  "totalRAMGB": 32
}
```

### 2.2 Model Weight Shard & Bit Width Inspection (`models inspect`)
Verify shard completeness, Safetensors headers, and actual quantization bit width:

```bash
# Human-readable summary
basalt models inspect ornith-ai/Ornith-1.5-9B-MLX-4bit

# Machine-readable JSON output
basalt models inspect ornith-ai/Ornith-1.5-9B-MLX-4bit --json
```

**Validation Checks Performed:**
1. The directory `serve` would load (HF cache `refs/main` snapshot, quant subfolders) or a local path.
2. Complete shard presence against `model.safetensors.index.json`; dangling HF blob links.
3. Safetensors header integrity and completeness: each shard must end exactly at its largest tensor offset, so an interrupted download is caught even with an intact header.
4. Actual bit width measured from the packed weights per layer; mixed widths (DQ) reported as `mixed a/b-bit`; a `config.json` claim that disagrees is an error.

Exit status is non-zero (and the MCP `inspect_model` result carries `isError: true`) whenever `intact` is false.

---

## 3. MCP Lifecycle Interface

`basalt` provides an MCP (Model Context Protocol) stdio server over JSON-RPC 2.0.

### Starting and stopping the MCP Server
```bash
basalt serve --mcp --model mlx-community/Qwen3.5-4B-4bit
```
The server speaks JSON-RPC on stdin/stdout for as long as stdin is open. To stop it, close its stdin (or send SIGTERM); there is no stop tool.

### Standard MCP Lifecycle Tools

| Tool Name | Arguments | Description |
|-----------|-----------|-------------|
| `status` | `{"modelId": "..."}` *(optional)* | Returns engine readiness, loaded model ID and the serving telemetry (`telemetry`: tok/s, TTFT, RAM, MLX GPU memory, active/total requests — the same snapshot as `serve --stats` and `GET /api/v1/stats`; `diagnostics` repeats the request/token totals from it). With `modelId`, `requestedModelLoaded` says whether that model is the one actually serving. |
| `check_hardware` | `{}` | Detects Apple Silicon chip, available Unified Memory, GPU and disk space as structured JSON. |
| `inspect_model` | `{"modelId": "..."}` | Validates weight shard integrity and detects actual bit width for the target model. |
| `switch_model` | `{"modelId": "..."}` | Hot-swaps the loaded model to a different model present in the local cache. |
| `gemma_generate` | `{"prompt": "...", "temperature": 0.0}` | Generates text using the loaded model via MLX. |
| `gemma_status` | `{}` | Legacy text status check for backward compatibility. |

---

## 4. Real-Model Regression Gate (#291)

`GoldenLogitsRealModelTests` in `Tests/BasaltTests/GoldenLogitsTests.swift` runs each model through `MLXInferenceEngine` at `temperature = 0` on eight fixed prompts (three short chats, code, an ~8k-token agent trace, a JSON tool call, a non-English chat, a hybrid-anchor resume over one `cacheKey`). A `LogitProbe` on the request records the top-64 log-probs of every greedy step, 32 steps per prompt. The gate fails when:

- the greedy tokens differ from the fixture, or
- at any step, KL(P_ref ‖ P_new) over the reference top-64, both renormalized over the reference ids, exceeds the fixture's `tolerance`. A reference id missing from the new top-64 gets the new smallest log-prob minus ln 1000.

Fixtures live in `Tests/BasaltTests/Fixtures/golden/<org>--<name>.json` with the snapshot revision and the tolerance. The same suite asserts the memory invariant: an ~8k-token prompt with 256 generated tokens causes 0 new swap-outs (`host_statistics64`) and keeps the peak `phys_footprint` within the `basalt models check` fit estimate.

Models: `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`, `LiquidAI/LFM2.5-24B-A2B-MLX-4bit`, `mlx-community/gemma-4-26B-A4B-it-qat-4bit`. An entry is a snapshot path or a repo id in the HF cache. The suite takes the GPU lock per model and logs to `logs/golden-logits-<date>.log` and `logs/golden-swap-<date>.log`.

```bash
M=lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly,LiquidAI/LFM2.5-24B-A2B-MLX-4bit,mlx-community/gemma-4-26B-A4B-it-qat-4bit
BASALT_E2E=1 BASALT_E2E_GOLDEN_MODELS=$M swift test --filter GoldenLogitsRealModelTests                          # compare
BASALT_E2E=1 BASALT_E2E_GOLDEN_MODELS=$M BASALT_GOLDEN_NOISE=1 swift test --filter 'GoldenLogitsRealModelTests/goldenLogits'   # run-to-run max KL
BASALT_E2E=1 BASALT_E2E_GOLDEN_MODELS=$M BASALT_GOLDEN_UPDATE=1 BASALT_GOLDEN_TOLERANCE=<from noise> swift test --filter 'GoldenLogitsRealModelTests/goldenLogits'  # record
```

An update keeps the fixture's tolerance unless `BASALT_GOLDEN_TOLERANCE` is set. A PR that rewrites a fixture says why. Without `BASALT_E2E=1` the suite is reported as skipped, and a green run without it proves nothing. CI runs it through `.github/workflows/model-tests.yml` (manual dispatch, or a PR labelled `model-gate` that touches `Sources/BasaltInference/**` or `Package.resolved`).
