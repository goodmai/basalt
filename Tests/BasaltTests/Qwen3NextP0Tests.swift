import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Qwen3-Next P0: tiny fixture, decode parity and template-driven profile (#642)", .serialized)
struct Qwen3NextP0Tests {
    static let tokens: [Int32] = [5, 19, 42, 7, 88, 13, 64, 21, 9, 31, 200, 3, 3, 150, 77, 12, 44, 101, 6, 250]

    static let coderTemplate = "{%- for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{%- endfor %}"
        + "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}"

    static let thinkingTemplate = Self.coderTemplate.replacingOccurrences(of: "assistant\n", with: "assistant\n<think>\n")

    static func model() throws -> Qwen3NextModel {
        try #require(try TinyModel.make(.qwen3Next) as? Qwen3NextModel)
    }

    @Test("The fixture loads as the fork's Qwen3NextModel: three Gated DeltaNet layers then one attention layer, 16 routed experts per MoE layer")
    func fixtureLoads() throws {
        let model = try Self.model()
        let caches = try model.newCache(parameters: nil)
        #expect(caches.map { $0 is MambaCache } == [true, true, true, false])
        #expect(caches[3] is KVCacheSimple)
        let experts = try #require(model.parameters().flattened().first { $0.0 == "model.layers.0.mlp.switch_mlp.gate_proj.weight" })
        #expect(experts.1.shape == [16, 32, 64])
    }

    @Test("Decode through the caches in chunks gives the full-sequence logits within 1e-3 and the same greedy token at every position", arguments: [1, 3, 7])
    func decodeMatchesFullForward(chunk: Int) throws {
        let model = try Self.model()
        let full = model(MLXArray(Self.tokens)[.newAxis], cache: nil).asType(.float32)
        let caches = try model.newCache(parameters: nil)
        var steps: [MLXArray] = []
        var start = 0
        while start < Self.tokens.count {
            let end = min(Self.tokens.count, start + chunk)
            steps.append(model(MLXArray(Array(Self.tokens[start ..< end]))[.newAxis], cache: caches).asType(.float32))
            start = end
        }
        let stepped = concatenated(steps, axis: 1)
        let gap = abs(stepped - full).max().item(Float.self)
        #expect(stepped.shape == [1, Self.tokens.count, 256])
        #expect(gap <= 1e-3, "max logit gap \(gap)")
        #expect(argMax(stepped, axis: -1).asArray(Int32.self) == argMax(full, axis: -1).asArray(Int32.self))
    }

    @Test("A qwen3_next template without <think> gets the non-thinking Qwen profile: no markers, reasoning_effort ignored, no thinking budget")
    func coderTemplateIsNonThinking() {
        let profile = ModelProfileRegistry.profile(forModelType: "qwen3_next", chatTemplate: Self.coderTemplate)
        #expect(!(profile is QwenProfile))
        #expect(profile.thinkingMarkers.open.isEmpty && profile.thinkingMarkers.close.isEmpty)
        #expect(profile.templateContext(for: .effort("high")).isEmpty)
        #expect(profile.templateContext(for: .disabled).isEmpty)
        #expect(MLXInferenceEngine.reasoningProtocol(for: profile) == nil)
    }

    @Test("Qwen3-Next-Thinking shares qwen3_next, and its <think> template keeps the thinking Qwen profile; other Qwen types are unchanged")
    func thinkingTemplateKeepsQwenProfile() {
        #expect(ModelProfileRegistry.profile(forModelType: "qwen3_next", chatTemplate: Self.thinkingTemplate) is QwenProfile)
        #expect(ModelProfileRegistry.profile(forModelType: "qwen3_5_moe", chatTemplate: Self.coderTemplate) is QwenProfile)
        #expect(ModelProfileRegistry.profile(forModelType: "qwen3_next", chatTemplate: Self.coderTemplate).name == "qwen3-instruct")
    }
}
