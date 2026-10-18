# AI Agent Setup & Onboarding Guide

Deterministic Setup Protocol for AI Agents (Claude Code, Cursor, Codex, OpenCode, Antigravity)

---

## Overview

This guide walks an external AI agent through diagnosing the host Mac hardware, inspecting downloaded weights, launching the MCP inference server, and verifying bit-exact reproducibility.

---

## Step 1: Hardware Diagnosis (`--check`)

Before requesting or downloading model weights, query hardware resources without initiating any downloads:

```bash
gemm fit --check
```

Or via the models subcommand:

```bash
gemm models check --system --json
```

### Expected JSON Output
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

### Verification Rules:
- If `compatible == false`: Notify user that Apple Silicon / Metal GPU is required for optimal performance.
- Compute maximum model RAM budget: `availableRAMGB * 0.85`.

---

## Step 2: Model Memory Feasibility Check

Before downloading, check whether a specific model fits in RAM:

```bash
gemm models check mlx-community/Qwen3.5-4B-4bit --json
```

Or for larger reasoning models:

```bash
gemm models check ornith-ai/Ornith-1.5-9B-MLX-4bit --json
```

If feasible, proceed to download:

```bash
gemm models download ornith-ai/Ornith-1.5-9B-MLX-4bit
```

---

## Step 3: Weight Shard Integrity & Bit Width Inspection (`--inspect`)

After downloading, verify tensor shard integrity and the actual quantization bit width:

```bash
gemm models inspect ornith-ai/Ornith-1.5-9B-MLX-4bit --json
```

### Inspection Checks Performed:
1. **Directory Structure**: Validates snapshot path in HuggingFace cache or local filesystem.
2. **Shard Completeness**: If `model.safetensors.index.json` is present, ensures every shard referenced in `weight_map` exists and is non-empty.
3. **Safetensors Header**: Reads the 8-byte little-endian header length and verifies JSON tensor descriptors.
4. **Actual Bit Width**: Reads `quantization.bits`, `group_size`, and mode from `config.json` and tensor dtypes.

### Sample JSON Output
```json
{
  "bitWidthDescription": "4-bit quantized (optiq)",
  "detectedBitWidth": 4,
  "errors": [],
  "groupSize": 64,
  "intact": true,
  "modelId": "ornith-ai/Ornith-1.5-9B-MLX-4bit",
  "path": "/Users/.../.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-9B-MLX-4bit/snapshots/main",
  "quantizationMode": "optiq",
  "shardFiles": [
    {
      "name": "model-00001-of-00003.safetensors",
      "sizeBytes": 2048000000,
      "sizeFormatted": "1.91 GB",
      "valid": true
    }
  ],
  "tensorDtype": "uint32, BF16",
  "totalSizeBytes": 5800000000,
  "totalSizeFormatted": "5.40 GB",
  "totalShards": 3
}
```

---

## Step 4: MCP Lifecycle Management

Agents can launch the dual-transport server in MCP stdio mode and control it via JSON-RPC 2.0.

### Starting the Server
```bash
gemm serve --mcp --model ornith-ai/Ornith-1.5-9B-MLX-4bit
```

### 1. Initialize
Send:
```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
```
Server responds with capabilities and version info.

### 2. Query System Status (`status`)
Send:
```json
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"status","arguments":{}}}
```
Returns:
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"diagnostics\":{\"requestCount\":0,\"totalTokensGenerated\":0},\"loadedModel\":\"ornith-ai/Ornith-1.5-9B-MLX-4bit\",\"ready\":true,\"status\":\"ok\",\"version\":\"1.0.0\"}"
    }
  ]
}
```

### 3. Check Hardware via MCP (`check_hardware`)
Send:
```json
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"check_hardware","arguments":{}}}
```

### 4. Inspect Model via MCP (`inspect_model`)
Send:
```json
{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"inspect_model","arguments":{"modelId":"ornith-ai/Ornith-1.5-9B-MLX-4bit"}}}
```

### 5. Switch Model via MCP (`switch_model`)
Send:
```json
{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"switch_model","arguments":{"modelId":"mlx-community/Qwen3.5-4B-4bit"}}}
```

---

## Step 5: Reproducibility & Golden Test Vector

The project enforces strict reproducibility standards verified by test suites.

### Golden Test Vector
- **Model**: `ornith-ai/Ornith-1.5-9B-MLX-4bit`
- **Benchmark**: ARC-AGI Task 1 (`benchmarks/arc-agi/compare-tasks/task_1.json`)
- **Reasoning Flag**: `--reasoning-effort none` (prevents infinite reasoning loops).

### Parity Invariants:
1. **Greedy Parity (`temperature = 0`)**:
   Consecutive runs with `temperature = 0.0` must produce the exact same token sequence and generated text.
2. **Reference Softmax Parity**:
   Non-greedy sampling at temperature $T$ must empirically match reference softmax probabilities $P(i) \propto \exp(z_i / T)$.
3. **Seeded Determinism**:
   When `--seed <val>` is passed, non-greedy generation is strictly repeatable.

Run reproducibility test suite:
```bash
swift test --filter SamplerAndGreedyParityTests
```
