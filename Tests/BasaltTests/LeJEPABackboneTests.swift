import Foundation
import Testing
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import MLXEmbedders
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCLI

@Suite("LeJEPA backbone: bidirectional mask, float32 mean pooling, and headless LoRA (#384)", .serialized)
struct LeJEPABackboneTests {
    static func tinyQwen3Encoder() throws -> MLXEmbedders::Qwen3Model {
        try withRandomState(MLXRandom.RandomState(seed: 36)) {
            let model = try LeJEPABackbone.makeQwen3Encoder(configData: Data(TinyModel.Family.qwen3Small.config.utf8))
            model.update(parameters: model.parameters().mapValues { $0 * Float(4.0) })
            eval(model)
            return model
        }
    }

    @Test("With isCausal = false, mutating token at L - 1 changes hidden state at 0; mutating padded positions yields bit-identical pooled embeddings")
    func bidirectionalMaskAndPaddingInvariance() throws {
        let model = try Self.tinyQwen3Encoder()
        let seqA = MLXArray([Int32(10), 20, 30, 40, 50, 60]).reshaped(1, 6)
        let seqB = MLXArray([Int32(10), 20, 30, 40, 50, 99]).reshaped(1, 6)

        let biA = try #require(LeJEPABackbone.encode(model: model, inputIds: seqA, isCausal: false).hiddenStates)
        let biB = try #require(LeJEPABackbone.encode(model: model, inputIds: seqB, isCausal: false).hiddenStates)
        let causalA = try #require(LeJEPABackbone.encode(model: model, inputIds: seqA, isCausal: true).hiddenStates)
        let causalB = try #require(LeJEPABackbone.encode(model: model, inputIds: seqB, isCausal: true).hiddenStates)
        eval(biA, biB, causalA, causalB)

        let biDiffAtZero = MLX.max(MLX.abs(biA[0, 0, 0...] - biB[0, 0, 0...])).item(Float.self)
        let causalDiffAtZero = MLX.max(MLX.abs(causalA[0, 0, 0...] - causalB[0, 0, 0...])).item(Float.self)
        #expect(biDiffAtZero > 1e-4, "bidirectional attention at index 0 must see index L - 1")
        #expect(causalDiffAtZero == 0, "causal attention at index 0 must not see index L - 1")

        let mask = MLXArray([Int32(1), 1, 1, 1, 0, 0]).reshaped(1, 6)
        let padA = MLXArray([Int32(10), 20, 30, 40, 5, 6]).reshaped(1, 6)
        let padB = MLXArray([Int32(10), 20, 30, 40, 123, 211]).reshaped(1, 6)
        let outPadA = LeJEPABackbone.encode(model: model, inputIds: padA, attentionMask: mask, isCausal: false)
        let outPadB = LeJEPABackbone.encode(model: model, inputIds: padB, attentionMask: mask, isCausal: false)
        let pooledA = LeJEPABackbone.pool(outPadA, mask: mask, strategy: .mean)
        let pooledB = LeJEPABackbone.pool(outPadB, mask: mask, strategy: .mean)
        eval(pooledA, pooledB)
        #expect(pooledA.dtype == .float32)
        #expect(pooledA.asArray(Float.self) == pooledB.asArray(Float.self))
    }

    @Test("Float32 mean pooling guards all-zero masks against division by zero and supports CLS fallback")
    func safeMeanAndCLSPooling() {
        let hidden = MLXArray([Float(2.0), 4.0, 6.0, 8.0, 10.0, 12.0]).reshaped(1, 2, 3).asType(.bfloat16)
        let zeroMask = MLXArray([Int32(0), 0]).reshaped(1, 2)
        let output = LeJEPABackbone.Output(hiddenStates: hidden, pooledOutput: nil)
        let zeroPooled = LeJEPABackbone.pool(output, mask: zeroMask, strategy: .mean)
        let clsFallback = LeJEPABackbone.pool(output, mask: nil, strategy: .cls)
        let explicitCLS = LeJEPABackbone.pool(
            LeJEPABackbone.Output(hiddenStates: hidden, pooledOutput: MLXArray([Float(9.0), 8.0, 7.0]).reshaped(1, 3)),
            mask: nil, strategy: .cls)
        eval(zeroPooled, clsFallback, explicitCLS)
        #expect(zeroPooled.dtype == .float32)
        #expect(zeroPooled.asArray(Float.self) == [0, 0, 0])
        #expect(clsFallback.asArray(Float.self) == [2, 4, 6])
        #expect(explicitCLS.asArray(Float.self) == [9, 8, 7])
    }

    @Test("lm_head weights are stripped when loading a decoder checkpoint in encoder mode")
    func lmHeadNotResidentInMemory() throws {
        let llm = try TinyModel.make(.qwen3Small)
        let rawWeights = Dictionary(uniqueKeysWithValues: llm.parameters().flattened())
        #expect(rawWeights.keys.contains(where: { $0.contains("lm_head") }))

        let encoder = try LeJEPABackbone.makeQwen3Encoder(
            configData: Data(TinyModel.Family.qwen3Small.config.utf8), weights: rawWeights)
        let encoderKeys = Set(encoder.parameters().flattened().map(\.0))
        #expect(!encoderKeys.isEmpty)
        #expect(!encoderKeys.contains(where: { $0.contains("lm_head") }))

        let headless = LeJEPABackbone.headlessBackbone(of: llm)
        let headlessKeys = Set(headless.parameters().flattened().map(\.0))
        #expect(!headlessKeys.isEmpty)
        #expect(!headlessKeys.contains(where: { $0.contains("lm_head") }))
    }

    @Test("attachLoRA on a headless encoder freezes all base weights, leaves only lora_a and lora_b in trainableParameters(), and round-trips through LoRAContainer")
    func headlessLoRAInjectionAndRoundTrip() throws {
        let model = try Self.tinyQwen3Encoder()
        let baseKeys = Set(model.parameters().flattened().map(\.0))
        let config = try MLXInferenceEngine.attachLoRA(to: model, layers: [0, 1], rank: 4)
        let trainable = model.trainableParameters().flattened()
        let trainableKeys = Set(trainable.map(\.0))
        #expect(!trainableKeys.isEmpty)
        #expect(trainableKeys.allSatisfy { $0.hasSuffix(".lora_a") || $0.hasSuffix(".lora_b") })
        #expect(trainableKeys.isDisjoint(with: baseKeys))

        let directory = FileManager.default.temporaryDirectory.appending(path: "lejepa-lora-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONEncoder().encode(config).write(to: directory.appending(component: "adapter_config.json"))
        try LoRATrain.saveLoRAWeights(model: model, url: directory.appending(component: "adapters.safetensors"))

        MLXInferenceEngine.detachLoRA(from: model)
        #expect(Set(model.parameters().flattened().map(\.0)) == baseKeys)

        let fresh = try Self.tinyQwen3Encoder()
        let loaded = try LoRAContainer.from(directory: directory)
        try LeJEPABackbone.applyLoRA(loaded, to: fresh)
        #expect(fresh.namedModules().contains(where: { $0.1 is LoRALayer }))
    }

    @Test("EmbeddingService relaxes Qwen3 chat refusal when an embedding adapter or pooling config is present, and ServeConfigFile maps model.embedding_adapter")
    func adapterRefusalAndConfigMapping() throws {
        let qwenChatConfig: [String: Any] = ["model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]]
        #expect(EmbeddingService.embedderRefusal(
            config: qwenChatConfig, sentenceTransformersLayout: false, name: "Qwen/Qwen3-4B", hasAdapter: false) != nil)
        #expect(EmbeddingService.embedderRefusal(
            config: qwenChatConfig, sentenceTransformersLayout: false, name: "Qwen/Qwen3-4B", hasAdapter: true) == nil)
        #expect(EmbeddingService.embedderRefusal(
            config: ["model_type": "qwen3", "pooling": "mean"], sentenceTransformersLayout: false, name: "Qwen/Qwen3-4B",
            hasAdapter: false) == nil)

        let yaml = "version: 1\ndefaults:\n  model:\n    embedding: Qwen/Qwen3-4B\n    embedding_adapter: /tmp/jepa-lora\n"
        let tokens = try ServeConfigFile.tokens(yaml: yaml, file: "basalt.yaml", cli: [])
        #expect(tokens.contains("--embedding-model=Qwen/Qwen3-4B"))
        #expect(tokens.contains("--embedding-adapter=/tmp/jepa-lora"))
    }
}
