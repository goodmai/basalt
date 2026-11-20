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

    @Test("Module-manifest quantization reports MACaQ/FlatQuant/LIRQuant details alongside any base quantization (#629)")
    func moduleManifestQuantization() {
        let hybrid = architectureLines(config: [
            "model_type": "qwen3_5",
            "quantization": ["bits": 8, "group_size": 64, "mode": "affine"],
            "macaq_quantization": [
                "version": 1,
                "modules": [
                    "layers.0.mlp.gate_proj": ["bits": 4, "group_size": 64, "mode": "affine"],
                    "layers.0.mlp.up_proj": ["bits": 4, "group_size": 64, "mode": "affine"],
                ],
            ],
        ])
        #expect(hybrid[3] == "Quantization: MACaQ 4-bit (Group Size: 64) [affine] (2 modules) + 8-bit (Group Size: 64) [affine]")

        let mixed = architectureLines(config: [
            "model_type": "gemma4",
            "macaq_quantization": [
                "version": 1,
                "modules": [
                    "layers.0.mlp.gate_proj": ["bits": 3, "group_size": 64, "mode": "affine"],
                    "layers.0.mlp.down_proj": ["bits": 4, "group_size": 32, "mode": "mxfp4"],
                ],
            ],
        ])
        #expect(mixed[3] == "Quantization: MACaQ 3/4-bit (Group Size: 32/64) [affine/mxfp4] (2 modules)")

        let flat = architectureLines(config: [
            "flatquant_quantization": [
                "modules": ["layers.0.mlp.up_proj": ["bits": 4, "group_size": 64, "mode": "affine"]],
            ],
        ])
        #expect(flat[3] == "Quantization: FlatQuant 4-bit (Group Size: 64) [affine] (1 module)")

        let emptyManifest = architectureLines(config: [
            "quantization": ["bits": 4, "group_size": 64, "mode": "affine"],
            "macaq_quantization": ["version": 1, "modules": [String: Any]()],
        ])
        #expect(emptyManifest[3] == "Quantization: 4-bit (Group Size: 64) [affine]")
    }
}
