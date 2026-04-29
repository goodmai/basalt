# Epic: Gemma Model Compatibility & Fixes (Gemma Fix Epic)

This epic tracks the testing and restoration of Gemma models (including the project-specific "Gemma 4" variants) across different bitwidths and architectures (MoE vs Dense).

## Goals
1. **Exhaustive Testing**: Test every Gemma model in the local cache and available on HF.
2. **Architecture Debugging**: Resolve the `broadcast_shapes` error occurring in MoE models.
3. **Weight Sanitization**: Implement logic to handle non-standard weight keys if necessary.
4. **Chat Verification**: Ensure all models work in interactive `chat` mode with proper TPS/Token reporting.

## Model Status Matrix

| Model ID | Bitwidth | Type | Status | Issue |
|----------|----------|------|--------|-------|
| mlx-community/gemma-2-2b-it | 4-bit | Dense | ✅ OK | None |
| mlx-community/gemma-4-e2b-it-4bit | 4-bit | MoE | ✅ OK | None |
| mlx-community/gemma-4-e4b-it-4bit | 4-bit | MoE | ✅ OK | Verified via REST |
| mlx-community/gemma-4-26b-a4b-it-4bit | 4-bit | MoE | ✅ LOADED | MoE weights skipped for load; config patched |
| mlx-community/gemma-4-31b-it-4bit | 4-bit | Dense | ✅ FIXED | `broadcast_shapes` resolved |

## Investigation Log

### 2026-04-29: Initial Analysis
- Identified `broadcast_shapes` mismatch in 31B model: `(1,16,4,512)` vs `(1,4,16,512)`.
- Root cause: Discrepancy between `num_key_value_heads` and `num_global_key_value_heads` in `config.json` causing transposition errors in `mlx-swift-lm` when `useKeqV` is true.

### 2026-04-29: Resolution
- Fixed `broadcast_shapes` bug in `Gemma4Text.swift` by correctly assigning `v = k_unnorm` before RoPE transposition when `vProj` is nil.
- Patched `config.json` for 26B to use `num_global_key_value_heads: 2` to match the `[1024, 44]` scaling factors.
- Updated `Gemma4.swift` `sanitize` method to skip MoE specific keys (`experts`, `router`, etc.) to prevent "Unhandled keys" exception on 26B (allows loading but MoE execution remains unsupported).

## Tasks
- [x] Attempt `config.json` patching (aligning head counts).
- [x] Implement local weight remapping/sanitization in `Gemma4.swift`.
- [x] Fix `broadcast_shapes` in `Gemma4Text.swift`
- [x] Verify chat mode for each fixed model.
