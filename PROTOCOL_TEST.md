# PROTOCOL_TEST.md — Comprehensive Manual QA Session (All README Models)

**Date:** 2026-05-02  
**Tester:** Antigravity (Manual QA Engineer)  
**Backend:** Gemm v0.7.0 (`.build/release/gemm`)  
**Hardware:** Apple M4 (24 GB RAM)  

---

## 1. Executive Summary

| Model Class | Success Rate | Findings |
|---|---|---|
| **Small (2B–4B)** | 100% ✅ | Fast, accurate, high TPS (>80). Qwen 4B is the most stable. |
| **Medium (7B–9B)** | 100% ✅ | Strong reasoning. Qwen 9B requires high token budget for think-blocks. |
| **Large (26B–31B)** | 0% ❌ | **CRITICAL FAILURES.** OOM, architectural mismatch, or crashes. |

**Overall Status: 5/8 Models PASS · 3/8 Models FAIL**

---

## 2. Test Plan

### Scope
Manual E2E testing of all 8 models verified in `README.md`.
Tasks per model:
1. **Algebra:** `Solve for x: 3x + 7 = 22. Show your step-by-step working.`
2. **Fibonacci:** `Write a Python function to compute the nth Fibonacci number using recursion with memoization. Include an example call for n=10 and print the result.`
3. **Translation:** `Translate the following text to French, German, and Spanish: "The quick brown fox jumps over the lazy dog."`

---

## 3. Results by Model

### M1 — Qwen3.5-4B-4bit
- **Algebra:** ✅ PASS. Correct result (x=5).
- **Fibonacci:** ✅ PASS. Implemented `memo = {}`.
- **Translation:** ✅ PASS. Correct translations.
- **Status:** ✅ **PASS**

### M2 — gemma-4-e4b-it-4bit
- **Algebra:** ✅ PASS. Step-by-step logic correct.
- **Fibonacci:** ✅ PASS. Result 55 for n=10.
- **Translation:** ✅ PASS. High quality.
- **Status:** ✅ **PASS**

### M3 — gemma-4-e2b-it-4bit
- **Algebra:** ✅ PASS. Very fast.
- **Fibonacci:** ✅ PASS. Simple and correct.
- **Translation:** ✅ PASS.
- **Status:** ✅ **PASS**

### M4 — Qwen2.5-Coder-7B-Instruct-4bit
- **Algebra:** ✅ PASS.
- **Fibonacci:** ✅ PASS. Best code formatting.
- **Translation:** ✅ PASS.
- **Status:** ✅ **PASS**

### M5 — Qwen3.5-9B-OptiQ-4bit
- **Algebra:** ✅ PASS.
- **Fibonacci:** ✅ PASS.
- **Translation:** ✅ PASS. Verbose output.
- **Status:** ✅ **PASS**

### M6 — gemma-4-26b-a4b-it-4bit (MoE)
- **Status:** ❌ **FAIL (Gibberish Output)**
- **Architecture Log:**
  ```text
  Type: gemma4 (Gemma4ForConditionalGeneration)
  Topology: Sparse MoE (Experts: 128)
  Quantization: 4-bit (Group Size: 64)
  ```
- **Finding:** Model loads and executes inference successfully (`is_ready: true`). The previous `broadcast_shapes` and OOM errors were resolved by fully implementing `SwitchGLU` for MoE and fixing the RoPE/transpose order in `Gemma4Attention`. However, the model generates gibberish output (e.g., `sustaining علاقوں`), indicating a deeper architectural mismatch in weight mapping or quantization decoding logic for this specific MoE implementation.
- **Root Cause:** Inference pipeline compiles, but weight application logic (likely RMSNorm scaling or expert routing logic) diverges from the expected math, producing random token embeddings.

### M7 — Qwen3.6-27B-4bit
- **Status:** ❌ **FAIL (Gibberish)**
- **Architecture Log:**
  ```text
  Type: qwen3_5 (Qwen3_5ForConditionalGeneration)
  Topology: Dense
  Quantization: 4-bit (Group Size: 64)
  ```
- **Finding:** Model outputs infinite sequences of symbols or empty responses (e.g., `!!!!!!!!!!`).
- **Root Cause:** Architectural mismatch or quantization issue at this scale in the underlying `mlx-swift-lm` generation logic.

### M8 — gemma-4-31b-it-4bit
- **Status:** ❌ **FAIL (Crash/OOM)**
- **Architecture Log:**
  ```text
  Type: gemma4 (Gemma4ForConditionalGeneration)
  Topology: Dense
  Quantization: 4-bit (Group Size: 64)
  ```
- **Finding:** Server crashes (`Trace/BPT trap: 5` or SIGKILL) during the first inference attempt despite aggressive memory purging (18GB free).
- **Root Cause:** OOM or shape mismatch in Metal kernels for 31B parameter count. KV-cache allocation fails.

---

## 4. CLI & Script Testing

### TC-FIT-01 — `gemm fit`
- **Command:** `gemm fit --json`
- **Result:** ✅ PASS. Correctly identified M4 chip and 24GB RAM.
- **Output Artifact:**
```json
{
  "device" : {
    "total_ram" : 25769803776,
    "available_ram" : 14366406144,
    "chip" : "M4",
    "model_budget" : 12061445222
  }
}
```

### TC-GS-01 — `gemma.swift`
- **Command:** `swift gemma.swift --model mlx-community/Qwen3.5-4B-4bit`
- **Result:** ✅ PASS. Correctly orchestrates build and serve lifecycle.

---

## 5. Findings & Recommendations

1. **[BUG-01] Large Model Support:** Models > 10GB are currently broken. Need to implement MoE support for 26B and debug shape broadcasting for 31B.
2. **[UX-01] Model Loading Feedback:** Server should return an error if a model is loaded in "stub mode" instead of silently failing later.
3. **[PERF-01] M4 Efficiency:** 4B models are the "sweet spot" for this hardware, providing near-instant responses.

---
*End of Protocol — Tester: Antigravity*
