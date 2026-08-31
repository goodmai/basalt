# Epic 22: Dynamic Quantization and MoE Support

## Objective
Support all MLX quantizations (2-bit, 3-bit, 4-bit, 8-bit) dynamically and automatically identify MoE (Mixture of Experts) models at runtime. Enhance logging to provide deep technical visibility into the loaded model's architecture, quantization configuration, and KV-cache parameters.

## Scope
1. **Dynamic Configuration Parsing:** Read `config.json` directly or through `MLX-Swift-LM` to extract `quantization` metadata (bits, group size) and MoE metadata (`num_experts`, `top_k_experts`, `enable_moe_block`).
2. **Technical Logging:** Print detailed architectural specs when a model loads (e.g., "Quantization: 4-bit, Group Size: 64", "Architecture: MoE, Experts: 128, Top-K: 8").
3. **MoE Adaptability:** Ensure the `Gemma4` (and potentially other models) implementation dynamically switches between dense and sparse (MoE) execution paths based on the parsed configuration, without hardcoded fallbacks.
4. **Verification:** Validate that the server can load differently quantized models and dense/MoE variants, recording the output in the test protocol.

## Implementation Steps
- [ ] Inject configuration parsing logic in `MLXInferenceEngine` or `ModelOrchestratorActor` before/after `loadModelContainer`.
- [ ] Extract and log:
  - Total parameters (estimated).
  - Quantization bits and group size.
  - Architecture type (Dense vs MoE).
  - Expert counts (if MoE).
- [ ] Refine `Gemma4Text.swift` (local patch) to fully rely on the dynamic config flags for MoE instantiation.
- [ ] Run test suite (`run_tests.sh`) to verify successful loading and logging.
