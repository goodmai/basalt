import Testing
@testable import BasaltInference

@Suite("The architecture lines logged at load come from config.json alone")
struct ArchitectureLinesTests {
    @Test("A quantized MoE names its experts, bits, group size and mode")
    func quantizedMoE() {
        let lines = architectureLines(config: [
            "architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4",
            "text_config": ["num_experts": 128], "quantization_config": ["bits": 4, "group_size": 64, "mode": "affine"],
        ])
        #expect(lines == [
            "--- Model Architecture ---", "Type: gemma4 (Gemma4ForConditionalGeneration)", "Topology: Sparse MoE (Experts: 128)",
            "Quantization: 4-bit (Group Size: 64) [affine]", "--------------------------",
        ])
    }

    @Test("A dense model with per-layer quantization and no bits reports dynamic quantization; no quantization reports the base precision")
    func denseVariants() {
        let dynamic = architectureLines(config: ["model_type": "qwen3_5", "num_experts": 0, "quantization": ["layers.0": ["bits": 3], "layers.1": ["bits": 6]]])
        #expect(dynamic[1] == "Type: qwen3_5 (Unknown)")
        #expect(dynamic[2] == "Topology: Dense")
        #expect(dynamic[3] == "Quantization: Dynamic Quantization (DQ - per-layer, 2 layers)")
        #expect(architectureLines(config: [:])[3] == "Quantization: None (FP16/BF16 Base)")
    }
}
