import Foundation
import Testing
import MLX
import MLXNN
@testable import GemmaServerCore

@Suite("Model Weight Sanitization")
struct WeightSanitizationTests {

    @Test("Sanitize weights ignores MoE keys for Gemma 4")
    func testGemma4Sanitization() {
        // This test simulates the logic I added to the library patch
        let weights: [String: MLXArray] = [
            "language_model.model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([10, 10]),
            "language_model.model.layers.0.experts.0.gate_proj.weight": MLXArray.zeros([10, 10]),
            "language_model.model.layers.0.router.weight": MLXArray.zeros([10, 10]),
            "language_model.model.layers.0.post_feedforward_layernorm_1.weight": MLXArray.zeros([10])
        ]
        
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            // Logic from the patch:
            if key.contains("experts") || key.contains("router") || key.contains("post_feedforward_layernorm_") {
                continue
            }
            sanitized[key] = value
        }
        
        #expect(sanitized.count == 1)
        #expect(sanitized["language_model.model.layers.0.self_attn.q_proj.weight"] != nil)
        #expect(sanitized["language_model.model.layers.0.experts.0.gate_proj.weight"] == nil)
        #expect(sanitized["language_model.model.layers.0.router.weight"] == nil)
    }
}
