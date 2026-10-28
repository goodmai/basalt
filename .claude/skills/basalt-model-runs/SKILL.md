---
name: basalt-model-runs
description: Run a real MLX model for basalt — benchmarks, PPL/passkey/memory measurements, speculative-decoding or KV-cache checks, parity runs. Use before launching any run that loads model weights, and when writing the measured-numbers section of a PR.
---

# Real-model runs in basalt

The rule is in CONTRIBUTING.md, "Real-model runs". This is how to follow it.

## Pick the runner

1. `PerformanceBenchmark` covers decode/prefill speed: `swift run -c release PerformanceBenchmark -m <repo> --kv-bits 4|8 --speculative-lookup --temperature <t> --seed <n> --print-output`.
2. Quality, memory or behaviour on a real model is an env-gated Swift test: `@Test(.enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_<X>_MODEL_PATH"))`, following `CacheBlendQualityTests` and `HybridPrefixDriftTests`. Run it with `swift test --filter <Suite>`.
3. Anything that fits neither goes in `scripts/<name>.swift` (see `scripts/build_metal.swift`).

Bash only starts one of these. Don't write `for model in …; do …; done | tail`, and don't decide pass/fail with grep.

## What the runner does

- Calls `basalt models inspect <repo> --json` first and fails when `intact` is false.
- Appends one line per step to `logs/<runner>-<date>.log` and prints it: timestamp, model, mode, parameters, result.
- Catches every error per step, logs `FAIL <step> <error>`, carries on to the next step only when the steps are independent, and exits non-zero if anything failed.
- Writes a header line with `git rev-parse HEAD`, the model repo and snapshot revision, and the chip and memory.
- Takes the GPU lock before loading weights and releases it in `defer`: create the directory `/private/tmp/basalt-gpu.lock` (failing while it exists means another run holds the GPU, so wait and retry) and remove it on exit, including on error.

## Which models

Cover at least two families for any speed or quality claim, plus one model the feature must refuse:

| Class | Cached examples | Use for |
|---|---|---|
| Dense, all `KVCacheSimple` | `mlx-community/Qwen3-4B-4bit`, `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | CacheBlend, prompt lookup, KV quantization |
| Sliding window + full attention, attention calls `cache.update` | `mlx-community/gemma-4-12B-it-qat-4bit` | KV cache layouts, fallback paths |
| Hybrid linear/full attention (qwen3_5) | `ornith-ai/Ornith-1.5-9B-MLX-4bit`, `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` | hybrid snapshots, fallback paths |
| Hybrid conv/SSM + attention | `LiquidAI/LFM2.5-8B-A1B-MLX-4bit`, `mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit` | hybrid snapshots, fallback paths |

Gemma 3 and Ministral 3 are out of scope. One model at a time on a 24 GB Mac.

## In the PR

A table per model with the numbers, the commit they were measured on, and the path or attachment of the runner's log. A model that wasn't run is listed as not measured, not left out.
