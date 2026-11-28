import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("DeepSeek-V3 (deepseek_v3, the GigaChat3 architecture) on the tiny fixture: MLA attention with a plain q_proj, sigmoid group-limited routing, one shared expert, YaRN RoPE; cached decode, prefix reuse on the engine and streamed experts (#893)", .serialized)
struct DeepseekV3Tests {
    static let tokens: [Int32] = [5, 19, 42, 7, 88, 13, 64, 21, 9, 31, 200, 3, 3, 150, 77, 12, 44, 101, 6, 250]
    static let tolerance: Float = 1e-3

    static func model() throws -> DeepseekV3Model {
        try #require(try TinyModel.make(.deepseekV3) as? DeepseekV3Model)
    }

    static func shapes(_ model: DeepseekV3Model) -> [String: [Int]] {
        Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
    }

    static func stepped(_ model: DeepseekV3Model, cuts: [Int]) throws -> MLXArray {
        let cache = try model.newCache(parameters: nil)
        let bounds = [0] + cuts + [tokens.count]
        let chunks = zip(bounds, bounds.dropFirst()).map { start, end in
            model(MLXArray(Array(tokens[start ..< end]))[.newAxis], cache: cache).asType(.float32)
        }
        return concatenated(chunks, axis: 1)
    }

    static func expectFullForward(_ model: DeepseekV3Model, cuts: [Int]) throws {
        let full = model(MLXArray(tokens)[.newAxis], cache: nil).asType(.float32)
        let cached = try stepped(model, cuts: cuts)
        let gap = abs(cached - full).max().item(Float.self)
        #expect(cached.shape == [1, tokens.count, 256])
        #expect(gap <= tolerance, "max logit gap \(gap) with cuts \(cuts)")
        #expect(argMax(cached, axis: -1).asArray(Int32.self) == argMax(full, axis: -1).asArray(Int32.self), "cuts \(cuts)")
    }

    static func engine(promptCacheBytes: Int) async throws -> MLXInferenceEngine {
        let capabilities = try #require(ModelCapabilities.of(modelType: "deepseek_v3"))
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: promptCacheBytes))
        await engine.record(.loadStarted(path: "tiny-deepseekV3"))
        await engine.attach(try TinyModel.container(.deepseekV3), facts: ModelFacts(capabilities: capabilities), epoch: engine.modelEpoch)
        return engine
    }

    @Test("With q_lora_rank null the fixture builds as the fork's DeepseekV3Model: a plain q_proj and no low-rank query pair, a 32-rank KV latent beside 8 RoPE dims, one dense layer then two MoE layers of 8 routed experts and one shared expert, a random router, and a stock KV cache per layer")
    func fixtureLoads() throws {
        let model = try Self.model()
        let shapes = Self.shapes(model)
        let caches = try model.newCache(parameters: nil)
        #expect(caches.count == 3 && caches.allSatisfy { type(of: $0) == KVCacheSimple.self })
        #expect(shapes["model.layers.0.self_attn.q_proj.weight"] == [128, 64])
        #expect(shapes["model.layers.0.self_attn.q_a_proj.weight"] == nil)
        #expect(shapes["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"] == [40, 64])
        #expect(shapes["model.layers.0.self_attn.kv_b_proj.weight"] == [160, 32])
        #expect(shapes["model.layers.0.self_attn.o_proj.weight"] == [64, 64])
        #expect(shapes["model.layers.0.mlp.gate_proj.weight"] == [128, 64])
        #expect(shapes["model.layers.0.mlp.switch_mlp.gate_proj.weight"] == nil)
        for layer in [1, 2] {
            #expect(shapes["model.layers.\(layer).mlp.switch_mlp.gate_proj.weight"] == [8, 32, 64])
            #expect(shapes["model.layers.\(layer).mlp.shared_experts.gate_proj.weight"] == [32, 64])
            #expect(shapes["model.layers.\(layer).mlp.gate.weight"] == [8, 64])
        }
        let router = model.parameters().flattened().filter { $0.0.hasSuffix(".gate.weight") || $0.0.hasSuffix(".e_score_correction_bias") }
        #expect(router.count == 4)
        #expect(router.allSatisfy { abs($0.1).max().item(Float.self) > 0 })
    }

    @Test("With an integer q_lora_rank the same config builds the low-rank query path, q_a_proj then q_b_proj and no q_proj, and gives finite logits")
    func lowRankQueriesBuild() throws {
        let config = TinyModel.Family.deepseekV3.config.replacingOccurrences(of: "\"q_lora_rank\": null", with: "\"q_lora_rank\": 16")
        try #require(config != TinyModel.Family.deepseekV3.config)
        let model = try #require(try TinyModel.Family.deepseekV3.model(Data(config.utf8)) as? DeepseekV3Model)
        TinyModel.seedSpecialParameters(model, family: .deepseekV3)
        let shapes = Self.shapes(model)
        #expect(shapes["model.layers.0.self_attn.q_proj.weight"] == nil)
        #expect(shapes["model.layers.0.self_attn.q_a_proj.weight"] == [16, 64])
        #expect(shapes["model.layers.0.self_attn.q_b_proj.weight"] == [128, 16])
        let logits = model(MLXArray(Self.tokens)[.newAxis], cache: nil).asType(.float32)
        #expect(logits.shape == [1, Self.tokens.count, 256])
        #expect(abs(logits).max().item(Float.self).isFinite)
    }

    @Test("A prefill of n − k tokens followed by k single-token steps through the caches gives the logits of one full forward over n tokens within 1e-3, the tolerance of the Qwen3-Next and PLT decode tests, and the same greedy token at every position",
          arguments: [1, 4, 12])
    func cachedDecodeMatchesFullForward(steps: Int) throws {
        let first = Self.tokens.count - steps
        try Self.expectFullForward(try Self.model(), cuts: Array(first ..< Self.tokens.count))
    }

    @Test("A prefill cut into chunks at one, two or four points gives the logits of one pass within 1e-3 and the same greedy tokens",
          arguments: [[10], [3, 11], [1, 7, 8, 19]])
    func chunkedPrefillMatchesOnePass(cuts: [Int]) throws {
        try Self.expectFullForward(try Self.model(), cuts: cuts)
    }

    @Test("On the engine, with the capabilities the table gives model_type deepseek_v3, a second turn that extends the first prefills only the appended tokens and generates what a fresh engine without a cache generates")
    func engineReusesThePrefix() async throws {
        let engine = try await Self.engine(promptCacheBytes: 1 << 30)
        let fresh = try await Self.engine(promptCacheBytes: 0)
        let first = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        let second = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "ns"))
        let cold = try await fresh.generate(request: greedy(Conversation.afterToolCall, key: nil))
        #expect(first.cachedPromptTokens == 0)
        #expect(second.cachedPromptTokens == TinyModel.tokens(Conversation.firstTurn).count - 1)
        #expect(second.promptTokens == cold.promptTokens)
        #expect(cold.cachedPromptTokens == 0)
        #expect(!second.generatedText.isEmpty)
        #expect(second.generatedText == cold.generatedText)
        #expect(await engine.checkInvariants(idle: true) == [])
    }

    @Test("With the routed experts of both MoE layers streamed from the fixture's safetensors, logits are bitwise equal to the resident model and 16 greedy tokens identical with a cache that holds every expert, half of them, or none beyond the current call",
          arguments: [false, true])
    func streamedExpertsMatchResident(quantized: Bool) async throws {
        let source = try ExpertStreamingParityTests.source("deepseek_v3")
        let directory = quantized ? try await ExpertStreamingParityTests.quantized(source) : source
        defer { ExpertStreamingParityTests.removeGenerated(source, directory) }
        let resident = try await ExpertStreamingParityTests.outputs(try await loadModelContainer(from: directory, using: ByteTokenizerLoader()))
        let full = try await ExpertStreamingParityTests.streamed(directory, cache: ByteCount(1 << 30))
        let total = try #require(await full.perform { ExpertStreamingLoader.store(in: $0.model)?.expertBytes })
        let quantizedLayouts = await full.perform { ExpertStreamingLoader.store(in: $0.model)?.layouts.map { $0.quantization != nil } }
        #expect(quantizedLayouts == Array(repeating: quantized, count: 6))
        for cache in [ByteCount(1 << 30), total / 2, ByteCount(1)] {
            let container = cache == ByteCount(1 << 30) ? full : try await ExpertStreamingParityTests.streamed(directory, cache: cache)
            let streamed = try await ExpertStreamingParityTests.outputs(container)
            #expect(streamed.tokens == resident.tokens, "quantized \(quantized) cache \(cache)")
            #expect(streamed.logits == resident.logits, "quantized \(quantized) cache \(cache)")
            let stats = try #require(await container.perform { ExpertStreamingLoader.store(in: $0.model)?.stats })
            #expect(stats.misses > 0)
            #expect(cache != ByteCount(1) || stats.hits == 0)
        }
    }

    @Test("A chat template whose macro takes a parameter named call (GigaChat 3.1) is rewritten so the Jinja parser accepts it, and names inside string literals and other macros stay as they are")
    func macroParameterNamedCallIsRenamed() throws {
        let template = """
            {%- macro render_function_call(call) -%}
            {%- if call.function -%}{%- set call = call.function -%}{%- endif -%}
            {{- '{"name": "' ~ call['name'] ~ '", "call": 1}' -}}
            {%- endmacro -%}
            {%- macro other(x, call_count=0) -%}{{ x }}{%- endmacro -%}
            {{- render_function_call(messages[0]) -}}
            """
        let compatible = ChatTemplateCompatibility.renamingReservedLoopVariables(template)
        #expect(compatible.contains("macro render_function_call(call_item)"))
        #expect(compatible.contains("call_item['name']") && compatible.contains("set call_item = call_item.function"))
        #expect(compatible.contains(#""call": 1"#))
        #expect(compatible.contains("macro other(x, call_count=0)"))
        #expect(!compatible.contains("render_function_call_item"))
    }
}
