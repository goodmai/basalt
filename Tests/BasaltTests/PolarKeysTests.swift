import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Polar keys in the zoned middle (#351, experimental): one byte per RoPE pair, the rest affine, served through ZonedKVCache")
struct PolarKeysTests {
    @Test("A polar round trip keeps every pair's phase within π/32 and the unrotated dims bit-exact with affine")
    func roundTrip() throws {
        let keys = MLXRandom.normal([1, 2, 300, 128], key: MLXRandom.key(1))
        var rows = PolarKeyRows(rotated: 32, restBits: 4, groupSize: 32, headDim: 128)
        rows.append(keys[.ellipsis, ..<200, 0...])
        rows.append(keys[.ellipsis, 200..., 0...])
        let decoded = try #require(rows.dequantizedRows(dtype: .float32))
        let (re, im) = (keys[.ellipsis, ..<16], keys[.ellipsis, 16 ..< 32])
        let (reHat, imHat) = (decoded[.ellipsis, ..<16], decoded[.ellipsis, 16 ..< 32])
        let difference = atan2(imHat, reHat) - atan2(im, re)
        #expect(abs(atan2(sin(difference), cos(difference))).max().item(Float.self) <= Float.pi / 32 + 1e-4)
        var affine = ZonedKVCache.QuantizedRows(bits: 4, groupSize: 32)
        affine.append(keys[.ellipsis, ..<200, 32...])
        affine.append(keys[.ellipsis, 200..., 32...])
        #expect(arrayEqual(decoded[.ellipsis, 32...], try #require(affine.dequantizedRows())).item(Bool.self))
        #expect(rows.count == 300 && rows.trim(100) == 100 && rows.count == 200)
    }

    @Test("Polar decode scores equal a matmul against the dequantized keys")
    func scores() throws {
        var keys = MiddleKeys(precision: KVPrecision.polar4.resolvingPolar(rotated: 32), groupSize: 32, headDim: 64)
        keys.append(MLXRandom.normal([1, 2, 96, 64], key: MLXRandom.key(2)))
        let grouped = MLXRandom.normal([1, 2, 3, 1, 64], key: MLXRandom.key(3))
        let expected = matmul(grouped, expandedDimensions(try #require(keys.dequantizedRows(dtype: .float32)), axis: 2).swappedAxes(-1, -2))
        #expect((abs(try #require(keys.scores(grouped)) - expected).max()).item(Float.self) < 1e-5)
    }

    static func decode(_ precision: KVPrecision, keys: MLXArray, query: MLXArray) -> MLXArray {
        let cache = ZonedKVCache(zones: KVZones(sinkTokens: 4, windowTokens: 32, migrationStep: 64), precision: precision)
        for start in stride(from: 0, to: keys.dim(2), by: 32) {
            let chunk = keys[.ellipsis, start ..< min(start + 32, keys.dim(2)), 0...]
            _ = cache.updateAndAttend(queries: chunk, keys: chunk, values: chunk, scale: 0.125, mask: .causal)
        }
        let one = keys[.ellipsis, (keys.dim(2) - 1)..., 0...]
        return cache.updateAndAttend(queries: query, keys: one, values: one, scale: 0.125, mask: .none)
    }

    @Test("In ZonedKVCache, unresolved polar falls back to affine 4-bit exactly; resolved polar stays within 2.5x the affine 4-bit error against fp16 (it is not better; #351)")
    func zoned() {
        let keys = MLXRandom.normal([1, 2, 320, 64], key: MLXRandom.key(4))
        let query = MLXRandom.normal([1, 2, 1, 64], key: MLXRandom.key(5))
        let affine = Self.decode(.q4, keys: keys, query: query)
        #expect(arrayEqual(Self.decode(.polar4, keys: keys, query: query), affine).item(Bool.self))
        let exact = MLXFast.scaledDotProductAttention(queries: query, keys: concatenated([keys, keys[.ellipsis, (keys.dim(2) - 1)..., 0...]], axis: 2),
                                                      values: concatenated([keys, keys[.ellipsis, (keys.dim(2) - 1)..., 0...]], axis: 2), scale: 0.125, mask: .none)
        let polar = Self.decode(KVPrecision.polar4.resolvingPolar(rotated: 64), keys: keys, query: query)
        let error = { (output: MLXArray) in (sqrt((output - exact).square().sum()) / sqrt(exact.square().sum())).item(Float.self) }
        print("POLAR zoned decode error: polar \(error(polar)), affine q4 \(error(affine))")
        #expect(error(polar) <= 2.5 * error(affine), "polar \(error(polar)) against affine \(error(affine))")
    }

    @Test("--kv-bits polar4 parses, names itself, needs --kv-zones, and is resolved per model from config.json")
    func flagAndGeometry() throws {
        let polar = try #require(KVPrecision(argument: "polar4"))
        #expect(polar.isPolar && polar.description == "polar4" && KVPrecision.accepted.contains("polar4"))
        #expect(KVPrecision.flagConflict(kvBitsSet: true, fixed: polar, zones: false) != nil && KVPrecision.flagConflict(kvBitsSet: true, fixed: polar, zones: true) == nil)
        let qwen = ["model_type": "qwen3_5", "text_config": ["model_type": "qwen3_5_text", "head_dim": 256, "num_attention_heads": 24, "partial_rotary_factor": 0.25]] as [String: Any]
        #expect(PolarGeometry.rotatedDims(config: qwen) == 64)
        #expect(PolarGeometry.rotatedDims(config: ["model_type": "lfm2_moe", "hidden_size": 2048, "num_attention_heads": 32]) == 64)
        #expect(PolarGeometry.rotatedDims(config: ["model_type": "gemma4_text", "head_dim": 256, "num_attention_heads": 8]) == nil)
    }

    @Test("Through the engine: polar4 resolves the rotated dims from the model config, an unknown family falls back to affine keys, and a tiny model decodes through polar zones")
    func engine() async throws {
        var configuration = EngineConfiguration(promptCacheBytes: 0)
        (configuration.kvPrecision, configuration.kvZones) = (.polar4, KVZones(sinkTokens: 4, windowTokens: 32, migrationStep: 64))
        let engine = MLXInferenceEngine(configuration)
        await engine.attach(try TinyModel.container(.qwen3HeadDim32))
        await engine.learnPolarGeometry(["model_type": "gemma4_text", "head_dim": 32, "num_attention_heads": 4], epoch: await engine.modelEpoch)
        #expect(await engine.effectiveKVPrecision?.polarRotated == 0)
        await engine.learnPolarGeometry(["model_type": "qwen3", "head_dim": 32, "num_attention_heads": 4], epoch: await engine.modelEpoch)
        #expect(await engine.effectiveKVPrecision?.polarRotated == 32)
        await engine.probeKVZones()
        let reply = try await engine.generate(request: GenerationRequest(prompt: String(repeating: "polar keys turn with position. ", count: 12), maxTokens: 8, temperature: 0))
        #expect(reply.completionTokens > 0)
    }
}
