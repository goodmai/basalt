import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("TurboQuant KV behind --kv-bits (#471): the turbo3, turbo4 and turbo8v4 presets reach GenerateParameters as a scheme and the fork's cache runs them")
struct KVTurboTests {
    @Test("--kv-bits takes turbo3, turbo4 and turbo8v4, names them back, and lists them among the accepted values", arguments: ["turbo3", "turbo4", "turbo8v4"])
    func parses(argument: String) throws {
        let precision = try #require(KVPrecision(argument: argument))
        #expect(precision.turbo && precision.description == argument && precision.scheme == argument && precision.uniform == precision)
        #expect(KVPrecision.accepted.contains(argument))
    }

    @Test("A turbo preset is refused with --kv-zones, with the zoned alternative named; alone it needs no zones")
    func zonesConflict() throws {
        let conflict = try #require(KVPrecision.flagConflict(kvBitsSet: true, fixed: .turbo8v4, zones: true))
        #expect(conflict.contains("--kv-zones") && conflict.contains("k8v4"))
        #expect(KVPrecision.flagConflict(kvBitsSet: true, fixed: .turbo8v4, zones: false) == nil)
        #expect(KVPrecision.flagConflict(kvBitsSet: true, fixed: .k8v4, zones: false) != nil)
    }

    @Test("The engine puts a turbo preset in GenerateParameters as kvScheme with no affine kvBits; a model without verified KV quantization gets neither")
    func parameters() async throws {
        var configuration = EngineConfiguration(promptCacheBytes: 0)
        configuration.kvPrecision = .turbo3
        let engine = MLXInferenceEngine(configuration)
        await engine.attach(try TinyModel.container(.qwen3))
        let params = await engine.sampling(for: GenerationRequest(prompt: "hi", maxTokens: 2, temperature: 0)).0
        #expect(params.kvScheme == "turbo3" && params.kvBits == nil)
        #expect(ModelCapabilities.conservative.kvScheme(.turbo3) == nil && ModelCapabilities.unrestricted.kvBits(.turbo4) == nil)
        #expect(ModelCapabilities.unrestricted.kvBits(.q4) == 4 && ModelCapabilities.unrestricted.kvScheme(.q4) == nil)
    }

    @Test("A tiny model generates through the fork's TurboQuant cache from the first token", arguments: [KVPrecision.turbo3, .turbo4, .turbo8v4])
    func generates(precision: KVPrecision) async throws {
        var configuration = EngineConfiguration(promptCacheBytes: 0)
        configuration.kvPrecision = precision
        configuration.quantizedKVStart = 0
        let engine = MLXInferenceEngine(configuration)
        await engine.attach(try TinyModel.container(.qwen3))
        let reply = try await engine.generate(request: GenerationRequest(prompt: String(repeating: "basalt keeps keys small. ", count: 8), maxTokens: 12, temperature: 0))
        #expect(reply.completionTokens > 0, "\(precision): \(reply.generatedText)")
    }
}
