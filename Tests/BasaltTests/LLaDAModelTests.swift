import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("LLaDA masked-diffusion LM behind --experimental-dllm on the llada TinyModel (#819)", .serialized)
struct LLaDAModelTests {
    static let tokens: [Int32] = [5, 19, 42, 7, 88, 13, 64, 21]

    static func model() throws -> LLaDAModel {
        try #require(try TinyModel.make(.llada) as? LLaDAModel)
    }

    static func reference(_ model: LLaDAModel, _ tokens: [Int32]) -> MLXArray {
        let w = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.asType(.float32)) })
        let c = model.config
        let (t, half) = (tokens.count, c.headDim / 2)
        func rms(_ x: MLXArray, _ key: String) -> MLXArray { x * rsqrt(mean(x * x, axis: -1, keepDims: true) + c.rmsNormEps) * w[key]! }
        func linear(_ x: MLXArray, _ key: String) -> MLXArray { matmul(x, w[key + ".weight"]!.T) }
        let inverse = pow(MLXArray(c.ropeTheta), -MLXArray(stride(from: 0, to: c.headDim, by: 2).map(Float.init)) / Float(c.headDim))
        let angles = MLXArray((0 ..< t).map(Float.init))[0..., .newAxis] * inverse[.newAxis]
        func rope(_ x: MLXArray) -> MLXArray {
            let (x1, x2) = (x[.ellipsis, ..<half], x[.ellipsis, half...])
            return concatenated([x1 * cos(angles) - x2 * sin(angles), x2 * cos(angles) + x1 * sin(angles)], axis: -1)
        }
        let group = c.heads / c.kvHeads
        var h = w["model.embed_tokens.weight"]![MLXArray(tokens)]
        for layer in 0 ..< c.layers {
            let p = "model.layers.\(layer)."
            let x = rms(h, p + "attn_norm.weight")
            let q = rope(linear(x, p + "q_proj").reshaped(t, c.heads, c.headDim).transposed(1, 0, 2))
            let k = repeated(rope(linear(x, p + "k_proj").reshaped(t, c.kvHeads, c.headDim).transposed(1, 0, 2)), count: group, axis: 0)
            let v = repeated(linear(x, p + "v_proj").reshaped(t, c.kvHeads, c.headDim).transposed(1, 0, 2), count: group, axis: 0)
            let scores = softmax(matmul(q, k.transposed(0, 2, 1)) / Float(c.headDim).squareRoot(), axis: -1)
            h = h + linear(matmul(scores, v).transposed(1, 0, 2).reshaped(t, -1), p + "attn_out")
            let f = rms(h, p + "ff_norm.weight")
            h = h + linear(silu(linear(f, p + "ff_proj")) * linear(f, p + "up_proj"), p + "ff_out")
        }
        return linear(rms(h, "model.norm.weight"), "model.lm_head")[.newAxis]
    }

    @Test("The full forward equals a plain-math bidirectional reference: logits within 1e-4 and identical argmax ids (#228)")
    func matchesReference() throws {
        let model = try Self.model()
        let logits = model(MLXArray(Self.tokens)[.newAxis], cache: nil).asType(.float32)
        let reference = Self.reference(model, Self.tokens)
        #expect(abs(logits - reference).max().item(Float.self) < 1e-4)
        #expect(argMax(logits, axis: -1).asArray(Int32.self) == argMax(reference, axis: -1).asArray(Int32.self))
    }

    @Test("Attention is bidirectional: changing the last token changes the logits at the first position")
    func bidirectional() throws {
        let model = try Self.model()
        var other = Self.tokens
        other[other.count - 1] = 99
        let a = model(MLXArray(Self.tokens)[.newAxis], cache: nil)[0, 0]
        let b = model(MLXArray(other)[.newAxis], cache: nil)[0, 0]
        #expect(abs(a - b).max().item(Float.self) > 0)
    }

    @Test("Prompt prefill through the dual block cache is bidirectional for LLaDA: its logits equal the uncached full forward, which a causal prefill would not")
    func promptPrefillIsBidirectional() throws {
        let model = try Self.model()
        let full = model(MLXArray(Self.tokens)[.newAxis], cache: nil).asType(.float32)
        let dual = DualBlockKVCache(layerCount: model.config.layers, bidirectionalPrompt: model.bidirectionalPrompt)
        dual.preparePromptPrefill()
        let prefilled = model(MLXArray(Self.tokens)[.newAxis], cache: dual.kvCaches).asType(.float32)
        #expect(abs(prefilled - full).max().item(Float.self) < 1e-4)
        let causal = DualBlockKVCache(layerCount: model.config.layers)
        causal.preparePromptPrefill()
        #expect(abs(model(MLXArray(Self.tokens)[.newAxis], cache: causal.kvCaches).asType(.float32) - full).max().item(Float.self) > 1e-3)
    }

    @Test("A config with zero heads or zero key-value heads is refused instead of trapping on a division by zero", arguments: ["\"n_heads\": 0", "\"n_kv_heads\": 0"])
    func zeroHeadsRefused(override: String) throws {
        var config = TinyModel.Family.llada.config
        config = config.replacingOccurrences(of: override.hasPrefix("\"n_heads") ? "\"n_heads\": 4" : "\"n_kv_heads\": 2", with: override)
        #expect(throws: (any Error).self) { _ = try LLaDAModel(configurationData: Data(config.utf8)) }
    }

    @Test("Fast-dLLM over the model unmasks every requested position, never emits the mask token, and repeats itself exactly")
    func fastDLLMGenerates() throws {
        let model = try Self.model()
        let adapter = try LanguageModelBlockAdapter(model)
        let request = FastDLLMRequest(promptTokens: Self.tokens.map(Int.init), maxNewTokens: 16)
        let config = BlockDiffusionConfig(blockSize: 8, maskTokenId: model.maskTokenId)
        let first = FastDLLMRunner.generate(model: adapter, request: request, config: config)
        #expect(first.generatedTokens.count == 16)
        #expect(!first.generatedTokens.contains(model.maskTokenId))
        #expect(FastDLLMRunner.generate(model: adapter, request: request, config: config).generatedTokens == first.generatedTokens)
    }

    @Test("With --experimental-dllm the engine generates through the diffusion iterator; a llada checkpoint is refused at load without it")
    func engineRoutesAndRefuses() async throws {
        var configuration = EngineConfiguration(promptCacheBytes: nil)
        configuration.dllm = true
        let engine = MLXInferenceEngine(configuration)
        await engine.attach(try TinyModel.container(.llada))
        let reply = try await engine.generate(request: greedy(Conversation.firstTurn, key: nil, maxTokens: 16))
        #expect(reply.completionTokens > 0)
        #expect(throws: BasaltError.self) { try ModelCapabilities.refuseDiffusionWithoutFlag(dllm: false, modelType: "llada", path: "tiny") }
        #expect(throws: Never.self) { try ModelCapabilities.refuseDiffusionWithoutFlag(dllm: true, modelType: "llada", path: "tiny") }
        #expect(throws: Never.self) { try ModelCapabilities.refuseDiffusionWithoutFlag(dllm: false, modelType: "qwen3", path: "tiny") }
    }
}
