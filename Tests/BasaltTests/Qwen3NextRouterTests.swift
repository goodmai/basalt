import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXLLM
@testable import MLXLMCommon
@testable import BasaltInference

@Suite("Qwen3-Next P1: fused 512-expert router against the stock chain (#642, #228)", .serialized)
struct Qwen3NextRouterTests {
    static func chain(_ block: Qwen3NextSparseMoeBlock, _ x: MLXArray) -> (MLXArray, MLXArray) {
        let gates = MLX.softmax(block.gate(x), axis: -1, precise: true)
        let indices = MLX.argSort(-gates, axis: -1)[.ellipsis, ..<block.topK]
        let picked = MLX.takeAlong(gates, indices, axis: -1)
        let scores = picked / picked.sum(axis: -1, keepDims: true)
        let routed = (block.switchMLP(x, indices) * scores[.ellipsis, .newAxis]).sum(axis: -2)
        return (indices.asType(.int32), routed + sigmoid(block.sharedExpertGate(x)) * block.sharedExpert(x))
    }

    static func wideBlock(experts: Int, topK: Int) throws -> Qwen3NextSparseMoeBlock {
        let json = """
            {"model_type": "qwen3_next", "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "partial_rotary_factor": 0.25,
             "linear_num_value_heads": 4, "linear_num_key_heads": 2, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
             "linear_conv_kernel_dim": 4, "full_attention_interval": 4, "num_experts": \(experts), "num_experts_per_tok": \(topK),
             "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32, "decoder_sparse_step": 1, "mlp_only_layers": [],
             "norm_topk_prob": true, "rms_norm_eps": 1e-6, "vocab_size": 256, "rope_theta": 10000.0, "tie_word_embeddings": false}
            """
        return Qwen3NextSparseMoeBlock(try JSONDecoder().decode(Qwen3NextConfiguration.self, from: Data(json.utf8)))
    }

    static func blocks() throws -> [Qwen3NextSparseMoeBlock] {
        let fixture = try TinyModel.make(.qwen3Next).namedModules().compactMap { $0.1 as? Qwen3NextSparseMoeBlock }
        let wide = try Self.wideBlock(experts: 512, topK: 10)
        let quantized = try Self.wideBlock(experts: 512, topK: 10)
        quantize(model: quantized, groupSize: 32, bits: 4) { path, _ in path.hasPrefix("switch_mlp") }
        return fixture + [wide, quantized]
    }

    @Test("Every MoE block of the fixture, a 512-expert top-10 block and its 4-bit copy pick exactly the stock chain's experts and match its output, at 1 and 5 tokens")
    func fusedRouterMatchesChain() throws {
        let blocks = try Self.blocks()
        #expect(blocks.count == 6)
        for (index, block) in blocks.enumerated() {
            for tokens in [1, 5] {
                let x = MLXRandom.normal([1, tokens, 64], key: MLXRandom.key(UInt64(6420 + 10 * index + tokens)))
                let gates = MLX.softmax(block.gate(x), axis: -1, precise: true)
                let fused = moeRouterTopK(gates, k: block.topK, normalize: true).indices.asType(.int32)
                let (indices, expected) = Self.chain(block, x)
                let actual = block(x)
                #expect(sorted(fused, axis: -1).asArray(Int32.self) == sorted(indices, axis: -1).asArray(Int32.self), "block \(index) tokens \(tokens)")
                let gap = abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
                #expect(gap <= 1e-5, "block \(index) tokens \(tokens): \(gap)")
            }
        }
    }
}
