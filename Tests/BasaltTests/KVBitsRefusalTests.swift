import Testing
import Foundation
import MLXLMCommon
import MLXLLM
@testable import BasaltCore
@testable import BasaltInference

@Suite("--kv-bits is refused at load for a model whose attention would trap in QuantizedKVCache.update (#165)", .serialized)
struct KVBitsRefusalTests {
    static let trappingTypes = ["gemma2", "apertus", "bitnet", "glm4_moe_lite", "gemma3n", "baichuan_m1"]

    static func gemma2() throws -> Gemma2Model {
        let config = """
            {"model_type": "gemma2", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-6,
             "vocab_size": 256, "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0,
             "query_pre_attn_scalar": 16.0}
            """
        return Gemma2Model(try JSONDecoder().decode(Gemma2Configuration.self, from: Data(config.utf8)))
    }

    @Test("serve --kv-bits 4 refuses the load, before any weight is read, with a typed error naming the model_type", arguments: trappingTypes)
    func fixedBitsRefusedAtLoad(modelType: String) async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("kvbits-\(modelType)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(#"{"model_type": "\#(modelType)"}"#.utf8).write(to: dir.appendingPathComponent("config.json"))
        let engine = MLXInferenceEngine(EngineConfiguration(kvPrecision: .q4))
        let error = await #expect(throws: BasaltError.self) { try await engine.load(modelPath: dir.path) }
        guard case .modelConfigurationInvalid(let path, let reason)? = error else {
            Issue.record("expected modelConfigurationInvalid, got \(String(describing: error))")
            return
        }
        #expect(path == dir.path)
        #expect(reason.contains("'\(modelType)'"))
        #expect(await engine.isLoaded == false)
    }

    @Test("A verified family under --kv-bits, and any family without it, loads as before")
    func supportedFamiliesUnchanged() {
        for modelType in ["qwen3_5", "lfm2_moe", "gemma4", "qwen3"] {
            #expect(throws: Never.self) { try ModelCapabilities.refuseUnverifiedKVBits(.q4, modelType: modelType, path: "") }
        }
        for modelType in Self.trappingTypes {
            #expect(throws: Never.self) { try ModelCapabilities.refuseUnverifiedKVBits(nil, modelType: modelType, path: "") }
            #expect(ModelCapabilities.of(modelType: modelType) == nil)
        }
    }

    @Test("Under auto a Gemma 2 decodes on an fp16 cache even when handed a 4-bit rung, instead of trapping in QuantizedKVCache.update")
    func autoKeepsFP16OnTrappingModel() async throws {
        let engine = MLXInferenceEngine(EngineConfiguration(quantizedKVStart: 0))
        let container = ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-gemma2"), model: try Self.gemma2(),
            processor: ByteInputProcessor(), tokenizer: ByteTokenizer()))
        await engine.record(.loadStarted(path: "tiny-gemma2"))
        await engine.attach(container, facts: ModelFacts(capabilities: .conservative), epoch: engine.modelEpoch)
        #expect(await engine.supportsKVQuantization == false)
        let request = GenerationRequest(prompt: "hello there", maxTokens: 4, temperature: 0).withKVPrecision(.q4)
        let response = try await engine.generate(request: request)
        #expect(response.completionTokens > 0)
    }

    @Test("A client cannot set the KV precision of one request: the field is not part of the request body")
    func requestBodyCarriesNoKVPrecision() throws {
        let body = #"{"prompt": "x", "kvPrecision": "q4", "kv_bits": 4}"#
        #expect(try JSONDecoder().decode(GenerationRequest.self, from: Data(body.utf8)).kvPrecision == nil)
    }
}
