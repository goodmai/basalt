import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXLLM
import MLXNN
@testable import BasaltCore
@testable import BasaltInference

@Suite("Zoned KV cache")
struct ZonedKVCacheTests {
    static let zones = KVZones(sinkTokens: 4, windowTokens: 8, migrationStep: 4)
    static let (kvHeads, heads, dim) = (2, 4, 64)

    static func tokens(_ count: Int, seed: UInt64) -> MLXArray {
        MLXRandom.normal([1, kvHeads, count, dim], key: MLXRandom.key(seed)).asType(.float16)
    }

    static func queries(_ count: Int, seed: UInt64) -> MLXArray {
        MLXRandom.normal([1, heads, count, dim], key: MLXRandom.key(seed)).asType(.float16)
    }

    static let precisions = [KVPrecision.q8, .q4, .k8v4, .k8v4h]

    static func roundTripped(_ x: MLXArray, bits: Int) -> MLXArray {
        let q = quantized(x, groupSize: 64, bits: bits)
        return dequantized(q.wq, scales: q.scales, biases: q.biases, groupSize: 64, bits: bits)
    }

    static func rotated(_ x: MLXArray) -> MLXArray {
        hadamardTransform(x.asType(.float32)).asType(x.dtype)
    }

    static func roundTrippedValues(_ x: MLXArray, _ precision: KVPrecision) -> MLXArray {
        precision.rotatesValues
            ? rotated(roundTripped(rotated(x), bits: precision.valueBits))
            : roundTripped(x, bits: precision.valueBits)
    }

    static func middleBytes(_ cache: ZonedKVCache) -> Int {
        cache.state.dropFirst(2).reduce(0) { $0 + $1.nbytes }
    }

    static func counts(_ cache: ZonedKVCache) -> [Int] {
        [cache.sinkCount, cache.quantizedCount, cache.windowCount]
    }

    static func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    @Test("The first sink tokens and the last window tokens stay exact; the tokens between them are quantized, moved one step at a time")
    func zonesFillInOrder() {
        let cache = ZonedKVCache(zones: Self.zones, precision: .q4)
        _ = cache.update(keys: Self.tokens(3, seed: 1), values: Self.tokens(3, seed: 2))
        #expect(Self.counts(cache) == [3, 0, 0] as [Int])
        _ = cache.update(keys: Self.tokens(12, seed: 3), values: Self.tokens(12, seed: 4))
        #expect(Self.counts(cache) == [4, 0, 11] as [Int])
        _ = cache.update(keys: Self.tokens(1, seed: 5), values: Self.tokens(1, seed: 6))
        #expect(Self.counts(cache) == [4, 4, 8] as [Int])
        _ = cache.update(keys: Self.tokens(30, seed: 7), values: Self.tokens(30, seed: 8))
        #expect(Self.counts(cache) == [4, 34, 8] as [Int])
        #expect(cache.offset == 46)
        #expect(cache.windowCount <= Self.zones.maxFullPrecisionTokens - Self.zones.sinkTokens)
    }

    @Test("update returns every token in position order: the sink and window bit-exact, the middle as its quantize round trip")
    func updateReturnsPositionOrder() {
        let cache = ZonedKVCache(zones: Self.zones, precision: .q8)
        let keys = Self.tokens(20, seed: 11)
        let values = Self.tokens(20, seed: 12)
        let (k, v) = cache.update(keys: keys, values: values)
        #expect(k.shape == keys.shape && v.shape == values.shape)
        #expect(Self.maxDifference(k[.ellipsis, ..<4, 0...], keys[.ellipsis, ..<4, 0...]) == 0)
        #expect(Self.maxDifference(k[.ellipsis, 12..., 0...], keys[.ellipsis, 12..., 0...]) == 0)
        let middle = Self.roundTripped(keys[.ellipsis, 4..<12, 0...], bits: 8)
        #expect(Self.maxDifference(k[.ellipsis, 4..<12, 0...], middle) == 0)
        #expect(Self.maxDifference(v[.ellipsis, 4..<12, 0...], values[.ellipsis, 4..<12, 0...]) > 0)
    }

    @Test("update returns the middle keys at the key bits and the middle values at the value bits, k8v4h values rotated, quantized and rotated back",
          arguments: precisions)
    func middleRoundTripsKeysAndValuesAtTheirOwnBits(precision: KVPrecision) {
        let cache = ZonedKVCache(zones: Self.zones, precision: precision)
        let (keys, values) = (Self.tokens(20, seed: 13), Self.tokens(20, seed: 14))
        let (k, v) = cache.update(keys: keys, values: values)
        let middle = (keys[.ellipsis, 4..<12, 0...], values[.ellipsis, 4..<12, 0...])
        #expect(Self.maxDifference(k[.ellipsis, 4..<12, 0...], Self.roundTripped(middle.0, bits: precision.keyBits)) == 0)
        #expect(Self.maxDifference(v[.ellipsis, 4..<12, 0...], Self.roundTrippedValues(middle.1, precision)) == 0)
        #expect(cache.rotatesValues == precision.rotatesValues)
    }

    @Test("Decode attention over the split zones equals fused attention over the same dequantized keys and values, for every precision",
          arguments: precisions)
    func decodeMatchesFusedReference(precision: KVPrecision) {
        let cache = ZonedKVCache(zones: Self.zones, precision: precision)
        _ = cache.update(keys: Self.tokens(37, seed: 21), values: Self.tokens(37, seed: 22))
        let (newKey, newValue) = (Self.tokens(1, seed: 23), Self.tokens(1, seed: 24))
        let reference = cache.copy()
        let (allKeys, allValues) = reference.update(keys: newKey, values: newValue)
        let q = Self.queries(1, seed: 25)
        let expected = MLXFast.scaledDotProductAttention(queries: q, keys: allKeys, values: allValues, scale: 0.125, mask: .none)
        let got = cache.updateAndAttend(queries: q, keys: newKey, values: newValue, scale: 0.125, mask: .none)
        #expect(cache.quantizedCount > 0)
        #expect(got.shape == expected.shape)
        #expect(Self.maxDifference(got, expected) < 2e-3)
    }

    @Test("Without sinks, the window is the only exact zone and the oldest tokens are the quantized ones")
    func noSinks() {
        let cache = ZonedKVCache(zones: KVZones(sinkTokens: 0, windowTokens: 8, migrationStep: 4), precision: .q8)
        let keys = Self.tokens(20, seed: 91)
        let (k, _) = cache.update(keys: keys, values: keys)
        #expect(Self.counts(cache) == [0, 12, 8] as [Int])
        #expect(Self.maxDifference(k[.ellipsis, 12..., 0...], keys[.ellipsis, 12..., 0...]) == 0)
        #expect(Self.maxDifference(k[.ellipsis, ..<12, 0...], Self.roundTripped(keys[.ellipsis, ..<12, 0...], bits: 8)) == 0)
        let q = Self.queries(1, seed: 92)
        let (newKey, reference) = (Self.tokens(1, seed: 93), cache.copy())
        let (allKeys, allValues) = reference.update(keys: newKey, values: newKey)
        let expected = MLXFast.scaledDotProductAttention(queries: q, keys: allKeys, values: allValues, scale: 0.125, mask: .none)
        let got = cache.updateAndAttend(queries: q, keys: newKey, values: newKey, scale: 0.125, mask: .none)
        #expect(Self.maxDifference(got, expected) < 2e-3)
    }

    @Test("Prefill attention with a causal mask equals fused attention over the dequantized keys", arguments: precisions)
    func prefillMatchesFusedReference(precision: KVPrecision) {
        let cache = ZonedKVCache(zones: Self.zones, precision: precision)
        _ = cache.update(keys: Self.tokens(10, seed: 31), values: Self.tokens(10, seed: 32))
        let (keys, values, q) = (Self.tokens(9, seed: 33), Self.tokens(9, seed: 34), Self.queries(9, seed: 35))
        let reference = cache.copy()
        let (allKeys, allValues) = reference.update(keys: keys, values: values)
        let expected = MLXFast.scaledDotProductAttention(queries: q, keys: allKeys, values: allValues, scale: 0.125, mask: .causal)
        let got = cache.updateAndAttend(queries: q, keys: keys, values: values, scale: 0.125, mask: .causal)
        #expect(Self.maxDifference(got, expected) == 0)
    }

    @Test("Before anything is quantized, attention is exactly fp16 attention")
    func shortContextIsExact() {
        let cache = ZonedKVCache(zones: Self.zones, precision: .q4)
        let (keys, values) = (Self.tokens(10, seed: 41), Self.tokens(10, seed: 42))
        _ = cache.update(keys: keys[.ellipsis, ..<9, 0...], values: values[.ellipsis, ..<9, 0...])
        let q = Self.queries(1, seed: 43)
        let got = cache.updateAndAttend(queries: q, keys: keys[.ellipsis, 9..., 0...], values: values[.ellipsis, 9..., 0...], scale: 0.125, mask: .none)
        let expected = MLXFast.scaledDotProductAttention(queries: q, keys: keys, values: values, scale: 0.125, mask: .none)
        #expect(cache.quantizedCount == 0)
        #expect(Self.maxDifference(got, expected) == 0)
    }

    @Test("trim removes the newest tokens across the window, middle and sink, and appending after it refills from the trimmed end",
          arguments: [KVPrecision.q4, .k8v4])
    func trimCrossesZones(precision: KVPrecision) {
        let cache = ZonedKVCache(zones: Self.zones, precision: precision)
        _ = cache.update(keys: Self.tokens(30, seed: 51), values: Self.tokens(30, seed: 52))
        #expect(Self.counts(cache) == [4, 18, 8] as [Int])
        #expect(cache.trim(10) == 10)
        #expect(Self.counts(cache) == [4, 16, 0] as [Int])
        #expect(cache.trim(18) == 18)
        #expect(Self.counts(cache) == [2, 0, 0] as [Int])
        let (k, _) = cache.update(keys: Self.tokens(3, seed: 53), values: Self.tokens(3, seed: 54))
        #expect(Self.counts(cache) == [4, 0, 1] as [Int])
        #expect(k.dim(2) == 5)
        #expect(cache.trim(99) == 5)
        #expect(cache.offset == 0)
    }

    @Test("A head dimension no quantization group divides stays in fp16 instead of crashing in QuantizedKVCache")
    func unquantizableHeadsStayExact() {
        let cache = ZonedKVCache(zones: Self.zones, precision: .q4)
        let keys = MLXRandom.normal([1, 2, 40, 16], key: MLXRandom.key(81)).asType(.float16)
        let (k, _) = cache.update(keys: keys, values: keys)
        #expect(Self.counts(cache) == [4, 0, 36] as [Int])
        #expect(Self.maxDifference(k, keys) == 0)
    }

    @Test("Head dimension 96 quantizes in groups of 32 past the window, and attention still matches the dequantized reference",
          arguments: [96, 160], [KVPrecision.q4, .k8v4, .k8v4h])
    func groupThirtyTwoHeads(dim: Int, precision: KVPrecision) {
        let cache = ZonedKVCache(zones: Self.zones, precision: precision)
        let keys = MLXRandom.normal([1, 2, 30, dim], key: MLXRandom.key(101)).asType(.float16)
        _ = cache.update(keys: keys, values: keys)
        #expect(Self.counts(cache) == [4, 18, 8] as [Int])
        let q = MLXRandom.normal([1, 4, 1, dim], key: MLXRandom.key(102)).asType(.float16)
        let newKey = MLXRandom.normal([1, 2, 1, dim], key: MLXRandom.key(103)).asType(.float16)
        let (allKeys, allValues) = cache.copy().update(keys: newKey, values: newKey)
        let expected = MLXFast.scaledDotProductAttention(queries: q, keys: allKeys, values: allValues, scale: 0.125, mask: .none)
        let got = cache.updateAndAttend(queries: q, keys: newKey, values: newKey, scale: 0.125, mask: .none)
        #expect(Self.maxDifference(got, expected) < 2e-3)
    }

    @Test("k8v4h's 4-bit middle reconstructs values with four x20 outlier channels closer than plain k8v4")
    func rotationLowersValueErrorOnOutlierChannels() {
        let outliers = MLXArray((0..<Self.dim).map { [3, 17, 40, 61].contains($0) ? Float(20) : 1 })
        let values = (MLXRandom.normal([1, Self.kvHeads, 40, Self.dim], key: MLXRandom.key(71)) * outliers).asType(.float16)
        let keys = Self.tokens(40, seed: 72)
        let errors = [KVPrecision.k8v4, .k8v4h].map { precision in
            let (_, v) = ZonedKVCache(zones: Self.zones, precision: precision).update(keys: keys, values: values)
            let (got, want) = (v[.ellipsis, 4..<32, 0...].asType(.float32), values[.ellipsis, 4..<32, 0...].asType(.float32))
            return (sqrt(((got - want) * (got - want)).sum()) / sqrt((want * want).sum())).item(Float.self)
        }
        #expect(errors[1] < errors[0], "rotated \(errors[1]) vs plain \(errors[0])")
    }

    @Test("k8v4h rotates head dimensions 2^k and 12·2^k, where the Hadamard transform is its own inverse, and not 20·2^k or 28·2^k, where it is not",
          arguments: [64, 96, 128, 160, 192, 256, 320, 448])
    func rotationIsItsOwnInverse(dim: Int) {
        let x = MLXRandom.normal([2, 3, dim], key: MLXRandom.key(131))
        let involution = Self.maxDifference(hadamardTransform(hadamardTransform(x)), x) < 1e-4
        #expect(ZonedKVCache.rotationApplies(toHeadDim: dim) == involution)
        #expect(involution == ![160, 320, 448].contains(dim))
    }

    @Test("A head dimension the Hadamard transform cannot rotate keeps k8v4h values unrotated, bit-exact with k8v4")
    func unrotatableHeadsStoreValuesPlain() {
        let keys = MLXRandom.normal([1, 2, 30, 160], key: MLXRandom.key(111)).asType(.float16)
        let values = MLXRandom.normal([1, 2, 30, 160], key: MLXRandom.key(112)).asType(.float16)
        let rotated = ZonedKVCache(zones: Self.zones, precision: .k8v4h)
        let plain = ZonedKVCache(zones: Self.zones, precision: .k8v4)
        let (a, b) = (rotated.update(keys: keys, values: values), plain.update(keys: keys, values: values))
        #expect(!rotated.rotatesValues)
        #expect(rotated.quantizedCount == 18)
        #expect(Self.maxDifference(a.1, b.1) == 0)
    }

    @Test("The k8v4 middle holds 13 bits per K+V element pair: its arrays weigh exactly what the geometry charges per quantized token")
    func middleBytesMatchGeometry() {
        let (heads, dim, count) = (8, 128, 4_096)
        let geometry = KVCacheGeometry(config: [
            "num_hidden_layers": 1, "num_attention_heads": 32, "num_key_value_heads": heads, "head_dim": dim,
        ])!
        for precision in [KVPrecision.q8, .k8v4, .q4] {
            let cache = ZonedKVCache(zones: .standard, precision: precision)
            let tokens = MLXRandom.normal([1, heads, count, dim], key: MLXRandom.key(121)).asType(.float16)
            _ = cache.update(keys: tokens, values: tokens)
            let measured = Double(Self.middleBytes(cache))
            let charged = Double(cache.quantizedCount) * geometry.bytesPerToken(precision)
            #expect(cache.quantizedCount == count - KVZones.standard.sinkTokens - KVZones.standard.windowTokens)
            #expect(abs(measured / charged - 1) < 0.01, "\(precision): \(measured) B held, \(charged) B charged")
        }
    }

    @Test("copy is independent of the original")
    func copyIsIndependent() {
        let cache = ZonedKVCache(zones: Self.zones, precision: .q4)
        _ = cache.update(keys: Self.tokens(20, seed: 61), values: Self.tokens(20, seed: 62))
        let copy = cache.copy()
        _ = cache.update(keys: Self.tokens(5, seed: 63), values: Self.tokens(5, seed: 64))
        #expect(copy.offset == 20)
        #expect(cache.offset == 25)
    }

    @Test("Only full-attention simple caches are replaced; sliding-window, chunked and recurrent caches are kept")
    func replacesOnlyFullAttentionLayers() {
        let layers: [any KVCache] = [KVCacheSimple(), RotatingKVCache(maxSize: 16), ChunkedKVCache(chunkSize: 32), MambaCache()]
        let replaced = ZonedKVCache.replacingFullAttention(layers, zones: Self.zones, precision: .q8)
        #expect(replaced[0] is ZonedKVCache)
        #expect(replaced[1] is RotatingKVCache)
        #expect(replaced[2] is ChunkedKVCache)
        #expect(replaced[3] is MambaCache)
    }
}

@Suite("Zoned KV cache in the engine")
struct ZonedKVCacheEngineTests {
    static let zones = KVZones(sinkTokens: 4, windowTokens: 16, migrationStep: 8)
    static let prompt = (0..<200).map { 1 + ($0 * 97 + 13) % 250 }

    static func container() throws -> ModelContainer {
        let config = """
            {"hidden_size": 128, "num_hidden_layers": 2, "intermediate_size": 256,
             "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-5,
             "vocab_size": 256, "rope_theta": 10000.0, "tie_word_embeddings": false}
            """
        let model = try withRandomState(MLXRandom.RandomState(seed: 7)) {
            LlamaModel(try JSONDecoder().decode(LlamaConfiguration.self, from: Data(config.utf8)))
        }
        model.update(parameters: model.parameters().mapValues { $0 * TinyModel.weightScale })
        eval(model)
        return ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-llama-32"), model: model,
            processor: ByteInputProcessor(), tokenizer: ByteTokenizer()))
    }

    static func headDim80Model() throws -> LlamaModel {
        let config = """
            {"hidden_size": 320, "num_hidden_layers": 1, "intermediate_size": 64,
             "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-5,
             "vocab_size": 256, "rope_theta": 10000.0, "tie_word_embeddings": false}
            """
        return LlamaModel(try JSONDecoder().decode(LlamaConfiguration.self, from: Data(config.utf8)))
    }

    static func text(_ stream: AsyncStream<Generation>) async -> String {
        var text = ""
        for await generation in stream {
            if case .chunk(let chunk) = generation { text += chunk }
        }
        return text
    }

    @Test("The probe accepts a model whose attention goes through attentionWithCacheUpdate and rejects one that reads the cache with update(), as Gemma 4 does")
    func probeTellsHowAttentionReadsTheCache() async throws {
        let llama = try Self.container()
        #expect(try await llama.perform { try ZonedKVCache.modelAttendsThroughCache($0.model) })
        #expect(try !ZonedKVCache.modelAttendsThroughCache(SelfAttendingModel()))
        #expect(try !ZonedKVCache.modelAttendsThroughCache(Self.headDim80Model()))
    }

    @Test("Zones apply only to a quantized request; the zoned request itself runs with kvBits cleared so MLX does not convert the cache again")
    func zonedCacheNeedsBitsAndZones() throws {
        let model = try LlamaModel(JSONDecoder().decode(LlamaConfiguration.self, from: Data(TinyModel.config.utf8)))
        var params = GenerateParameters(maxTokens: 4, temperature: 0)
        #expect(try MLXInferenceEngine.zonedCache(params: params, zones: Self.zones, precision: nil, model: model) == nil)
        params.kvBits = 8
        #expect(try MLXInferenceEngine.zonedCache(params: params, zones: nil, precision: .k8v4, model: model) == nil)
        let zoned = try #require(try MLXInferenceEngine.zonedCache(params: params, zones: Self.zones, precision: .k8v4, model: model))
        #expect(zoned.cache.allSatisfy { ($0 as? ZonedKVCache)?.precision == .k8v4 })
        #expect(zoned.params.kvBits == nil)
        #expect(zoned.params.maxTokens == 4)
    }

    @Test("A quantized launch with zones generates on the zoned cache: the text of a direct zoned run, not the uniform-quantized one")
    func launchGeneratesOnZonedCache() async throws {
        let container = try Self.container()
        var quantized = GenerateParameters(maxTokens: 24, temperature: 0)
        quantized.kvBits = 4
        quantized.quantizedKVStart = 0
        let params = quantized
        let launches = try await container.perform { (context: ModelContext) -> [MLXInferenceEngine.GenerationLaunch] in
            let input = LMInput(tokens: MLXArray(Self.prompt.map(Int32.init)))
            var exact = params
            exact.kvBits = nil
            let reference = try MLXLMCommon.generate(
                input: input,
                cache: ZonedKVCache.replacingFullAttention(try context.model.newCache(parameters: exact), zones: Self.zones, precision: .q4),
                parameters: exact, context: context)
            return try [Self.zones, nil].map { zones in
                try MLXInferenceEngine.launch(
                    input: input, lease: nil, keepsCache: false, boundary: Self.prompt.count - 1, params: params,
                    components: GenerationComponents(), tools: nil, context: context, kvZones: zones,
                    kvPrecision: zones == nil ? nil : .q4)
            } + [MLXInferenceEngine.GenerationLaunch(stream: reference, task: nil, cache: nil, reusedTokens: 0)]
        }
        var texts = [String]()
        for launch in launches { texts.append(await Self.text(launch.stream)) }
        let (zoned, uniform, reference) = (texts[0], texts[1], texts[2])
        #expect(!zoned.isEmpty)
        #expect(zoned == reference)
        #expect(zoned != uniform, "the fixture must tell the two layouts apart")
    }
}

final class SelfAttendingModel: Module, LLMModel {
    var loraLayers: [Module] { [] }

    func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        [KVCacheSimple()]
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokens = MLXArray.zeros([1, 2, inputs.dim(1), 64], dtype: .float16)
        let (keys, _) = cache?.first?.update(keys: tokens, values: tokens) ?? (tokens, tokens)
        return MLXArray.zeros([1, inputs.dim(1), keys.dim(2)])
    }
}
