import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("GaugeRoPE KV cache: latent codes in place of full-attention K/V, parity with the stock cache at full rank (#647, #228)", .serialized)
struct GaugeRoPEKVCacheTests {
    static let tokens = (0 ..< 96).map { ($0 * 37 + 11) % 200 }
    static let rotaryDims = 4
    static let jointDims = 2 * (16 + 16)

    static func layout(carrierRank: Int, codeRank: Int?, calibrationTokens: Int = 64) -> GaugeRoPELayout {
        GaugeRoPELayout(
            rotaryDims: rotaryDims, ropeBase: 100_000, calibrationTokens: calibrationTokens, carrierRank: carrierRank, codeRank: codeRank)
    }

    static func caches(_ model: any LanguageModel, layout: GaugeRoPELayout?) throws -> [any KVCache] {
        let stock = try model.newCache(parameters: nil)
        return layout.map { GaugeRoPEKVCache.replacingFullAttention(stock, layout: $0) } ?? stock
    }

    static func chunkedLogits(_ model: any LanguageModel, caches: [any KVCache]) -> MLXArray {
        let parts = stride(from: 0, to: tokens.count, by: 32).map { start in
            model(MLXArray(Array(tokens[start ..< (start + 32)]))[.newAxis], cache: caches).asType(.float32)
        }
        return concatenated(parts, axis: 1)
    }

    static func greedy(_ model: any LanguageModel, caches: [any KVCache]) throws -> [Int] {
        let input = LMInput(tokens: MLXArray(tokens.map(Int32.init)))
        return Array(try TokenIterator(input: input, model: model, cache: caches, parameters: GenerateParameters(maxTokens: 8, temperature: 0)))
    }

    static func relativeError(_ approx: (MLXArray, MLXArray), _ exact: (MLXArray, MLXArray)) -> Double {
        let num = MLX.sum(MLX.square(approx.0 - exact.0)) + MLX.sum(MLX.square(approx.1 - exact.1))
        let den = MLX.sum(MLX.square(exact.0)) + MLX.sum(MLX.square(exact.1))
        return Double((num / den).item(Float.self)).squareRoot()
    }

    @Test("Full carrier and code rank: chunked logits within 1e-4 of the stock cache and identical greedy tokens, with every full-attention layer encoded")
    func fullRankMatchesStock() throws {
        let model = try TinyModel.make(.qwen35)
        let full = Self.layout(carrierRank: 2, codeRank: Self.jointDims)
        let gaugeCaches = try Self.caches(model, layout: full)
        let stock = Self.chunkedLogits(model, caches: try Self.caches(model, layout: nil))
        let gauge = Self.chunkedLogits(model, caches: gaugeCaches)
        #expect(abs(gauge - stock).max().item(Float.self) < 1e-4)
        let encoded = gaugeCaches.compactMap { $0 as? GaugeRoPEKVCache }
        #expect(!encoded.isEmpty)
        #expect(encoded.allSatisfy { $0.codec != nil && $0.encodedTokens == Self.tokens.count && $0.offset == Self.tokens.count })
        #expect(try Self.greedy(model, caches: try Self.caches(model, layout: full)) == Self.greedy(model, caches: try Self.caches(model, layout: nil)))
    }

    @Test("Reduced rank: K/V reconstruction error of captured full-attention K/V is below 1 and does not grow as the code rank rises to exact")
    func reducedRankErrorIsBounded() throws {
        let model = try TinyModel.make(.qwen35)
        let stock = try Self.caches(model, layout: nil)
        eval(model(MLXArray(Self.tokens)[.newAxis], cache: stock))
        let captured = try #require(stock.first { type(of: $0) == KVCacheSimple.self })
        let (keys, values) = (captured.state[0], captured.state[1])
        let errors = [8, 24, 48, Self.jointDims].map { rank -> Double in
            let cache = GaugeRoPEKVCache(layout: Self.layout(carrierRank: 1, codeRank: rank, calibrationTokens: Self.tokens.count))
            let error = Self.relativeError(cache.update(keys: keys, values: values), (keys, values))
            #expect(cache.codec?.elementsPerToken == Self.rotaryDims + rank)
            return error
        }
        #expect(errors[0] < 1)
        #expect(zip(errors, errors.dropFirst()).allSatisfy { $1 <= $0 + 1e-5 }, "\(errors)")
        #expect(errors[errors.count - 1] < 1e-4, "\(errors)")
    }

    @Test("Reduced rank decodes: finite chunked logits and the requested number of greedy tokens")
    func reducedRankDecodeIsFinite() throws {
        let model = try TinyModel.make(.qwen35)
        let reduced = Self.layout(carrierRank: 1, codeRank: 16, calibrationTokens: 32)
        let logits = Self.chunkedLogits(model, caches: try Self.caches(model, layout: reduced))
        #expect(all(isFinite(logits)).item(Bool.self))
        #expect(try Self.greedy(model, caches: try Self.caches(model, layout: reduced)).count == 8)
    }

    @Test("Before the calibration window fills the cache holds exact K/V and returns them unchanged")
    func exactUntilCalibrated() throws {
        let keys = MLXRandom.normal([1, 2, 10, 16], key: MLXRandom.key(6476))
        let values = MLXRandom.normal([1, 2, 10, 16], key: MLXRandom.key(6477))
        let cache = GaugeRoPEKVCache(layout: Self.layout(carrierRank: 1, codeRank: 8))
        let (k, v) = cache.update(keys: keys, values: values)
        #expect(cache.codec == nil && cache.offset == 10 && cache.encodedTokens == 0)
        #expect(abs(k - keys).max().item(Float.self) == 0 && abs(v - values).max().item(Float.self) == 0)
    }

    @Test("The per-frequency carrier basis is the same whether calibrated on pre-RoPE or post-RoPE keys")
    func carrierIgnoresRoPE() {
        let keys = MLXRandom.normal([128, 4, 16], key: MLXRandom.key(6478)) * MLXArray([3.0, 1.5, 0.5, 0.1] as [Float]).reshaped([1, 4, 1])
        let projector = { (carrier: GaugeRoPECarrier) in matmul(carrier.basis, carrier.basis.transposed(0, 2, 1)) }
        let pre = GaugeRoPECarrier.calibrate(keys: keys, rotaryDims: 16, carrierHeads: 2)
        let post = GaugeRoPECarrier.calibrate(keys: GaugeRoPERotary.apply(keys, rotaryDims: 16, base: 10_000, offset: 300), rotaryDims: 16, carrierHeads: 2)
        #expect(abs(projector(pre) - projector(post)).max().item(Float.self) < 1e-3)
    }

    @Test("Default code rank fills 0.6x the k8v4 bytes of Qwen3.8 at 2 bytes per code, and the geometry sizes gaugerope at that budget")
    func byteBudget() throws {
        let rank = GaugeRoPECodec.defaultCodeRank(heads: 4, keyDim: 256, valueDim: 256, carrierElements: 2 * 64)
        #expect(rank == 371)
        let k8v4PerLayer = 4 * (256 * KVCacheGeometry.bytesPerElement(kvBits: 8) + 256 * KVCacheGeometry.bytesPerElement(kvBits: 4))
        #expect(Double(2 * (128 + rank)) <= 0.6 * k8v4PerLayer)
        let config: [String: Any] = [
            "num_hidden_layers": 4, "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
            "full_attention_interval": 4,
        ]
        let geometry = try #require(KVCacheGeometry(config: config))
        #expect(abs(geometry.bytesPerToken(.gaugeRoPE) - 0.6 * geometry.bytesPerToken(.k8v4)) < 1e-9)
        #expect(geometry.bytesPerToken(.gaugeRoPE) < geometry.bytesPerToken(nil) / 4)
    }
}

@Suite("GaugeRoPE routing and refusals: uncached launch plan, qwen3_5 allowlist, Gemma 4 and YaRN refused (#647)")
struct GaugeRoPEWiringTests {
    static var qwen38: [String: Any] { [
        "model_type": "qwen3_5",
        "text_config": [
            "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
            "rope_parameters": ["rope_type": "default", "rope_theta": 10_000_000, "partial_rotary_factor": 0.25] as [String: Any],
        ] as [String: Any],
    ] }

    static func refusal(_ config: [String: Any], ropeYarn: Double? = nil) -> String? {
        do {
            _ = try ModelCapabilities.gaugeRoPELayout(.gaugeRoPE, config: config, ropeYarn: ropeYarn, path: "m")
            return nil
        } catch {
            guard case .modelConfigurationInvalid(_, let reason) = error else { return "\(error)" }
            return reason
        }
    }

    @Test("A Qwen3.8 config gives the rotary layout; non-GaugeRoPE precisions need no layout")
    func qwenLayout() throws {
        let layout = try #require(try ModelCapabilities.gaugeRoPELayout(.gaugeRoPE, config: Self.qwen38, ropeYarn: nil, path: "m"))
        #expect(layout.rotaryDims == 64 && layout.ropeBase == 10_000_000)
        #expect(layout.calibrationTokens == 512 && layout.carrierRank == 2 && layout.codeRank == nil)
        #expect(try ModelCapabilities.gaugeRoPELayout(.q4, config: ["model_type": "gemma4"], ropeYarn: nil, path: "m") == nil)
        #expect(try ModelCapabilities.gaugeRoPELayout(nil, config: nil, ropeYarn: nil, path: "m") == nil)
    }

    @Test("Gemma 4 is refused naming p-RoPE, other families as unverified, YaRN and non-default RoPE as not plain")
    func refusals() {
        let gemma = Self.refusal(["model_type": "gemma4"])
        #expect(gemma?.contains("'gemma4'") == true && gemma?.contains("p-RoPE") == true)
        #expect(Self.refusal(["model_type": "gemma4_text"])?.contains("p-RoPE") == true)
        #expect(Self.refusal(["model_type": "llama"])?.contains("'llama'") == true)
        #expect(Self.refusal(Self.qwen38, ropeYarn: 2)?.contains("--rope-yarn") == true)
        var yarn = Self.qwen38
        let yarnRope: [String: Any] = ["rope_type": "yarn", "rope_theta": 1e7]
        yarn["text_config"] = ["num_attention_heads": 24, "head_dim": 256, "rope_parameters": yarnRope] as [String: Any]
        #expect(Self.refusal(yarn)?.contains("not plain") == true)
    }

    @Test("serve --kv-bits gaugerope refuses a Gemma 4 load before any weight is read")
    func gemmaLoadRefused() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("gaugerope-gemma4-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(#"{"model_type": "gemma4"}"#.utf8).write(to: dir.appendingPathComponent("config.json"))
        let engine = MLXInferenceEngine(EngineConfiguration(kvPrecision: .gaugeRoPE))
        let error = await #expect(throws: BasaltError.self) { try await engine.load(modelPath: dir.path) }
        guard case .modelConfigurationInvalid(_, let reason)? = error else {
            Issue.record("expected modelConfigurationInvalid, got \(String(describing: error))")
            return
        }
        #expect(reason.contains("p-RoPE"))
        #expect(await engine.isLoaded == false)
    }

    @Test("GaugeRoPE launches uncached on its own cache plan, never as quantized KV bits or a prompt-cache namespace")
    func routing() {
        let layout = GaugeRoPELayout(rotaryDims: 64, ropeBase: 10_000_000)
        let options = LaunchOptions(params: GenerateParameters(), kvPrecision: .gaugeRoPE, gaugeRoPE: layout)
        #expect(FreshCache.plan(options: options) == .gaugeRoPE(layout))
        #expect(FreshCache.plan(options: LaunchOptions(params: GenerateParameters(), kvPrecision: .gaugeRoPE)) == .stock)
        #expect(ModelCapabilities.unrestricted.kvBits(.gaugeRoPE) == nil)
        let settings = GenerationPlan.Settings(kvPrecision: .gaugeRoPE)
        #expect(GenerationPlan.cache(precision: .gaugeRoPE, settings).kvPrecision == .gaugeRoPE)
        let plan = GenerationPlan.make(GenerationRequest(prompt: "hi"), settings, GenerationPlan.Facts())
        #expect(!plan.batchable && !plan.drafts)
        #expect(GenerationPlan.cacheNamespace(GenerationRequest(prompt: "hi"), settings, cacheEnabled: true, media: .text, sameEpoch: true) == nil)
    }
}
