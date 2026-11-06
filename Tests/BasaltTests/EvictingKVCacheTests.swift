import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXLLM
@testable import BasaltCore
@testable import BasaltInference

@Suite("fp16 evicting KV cache for attention that reads the cache through update()")
struct EvictingKVCacheTests {
    static let zones = KVZones(sinkTokens: 4, windowTokens: 8, migrationStep: 4, evictedMiddleFraction: 0.5)
    static let prompt = (0..<120).map { 1 + ($0 * 97 + 13) % 250 }

    static func counts(_ cache: EvictingKVCache) -> [Int] {
        [cache.sinkCount, cache.middleCount, cache.windowCount]
    }

    @Test("Each update hands back every stored token plus the chunk, then keeps the sinks, the window and the lowest-key-norm half of what left the window, per KV head")
    func keepsSinksWindowAndLowestNormMiddle() {
        let cache = EvictingKVCache(zones: Self.zones)
        let scales: [[Float]] = [
            (0..<20).map { Float(1 + ($0 * 7) % 20) / 4 },
            (0..<20).map { Float(20 - ($0 * 3) % 20) / 4 },
        ]
        let keys = ZonedKVCacheTests.scaledTokens(20, seed: 401, scales: scales)
        let values = ZonedKVCacheTests.tokens(20, seed: 402)
        var handed = [Int]()
        for start in stride(from: 0, to: 20, by: 4) {
            let rows = start..<(start + 4)
            handed.append(cache.update(keys: keys[.ellipsis, rows, 0...], values: values[.ellipsis, rows, 0...]).0.dim(2))
        }
        #expect(handed == [4, 8, 12, 16, 18])
        #expect(Self.counts(cache) == [4, 4, 8] as [Int])
        #expect((cache.evictedCount, cache.storedCount, cache.offset) == (4, 16, 20))
        let stored = cache.state
        for head in 0..<ZonedKVCacheTests.kvHeads {
            let kept = ZonedKVCacheTests.lowestNormRows(keys, head: head, among: 4..<8, keeping: 2)
                + ZonedKVCacheTests.lowestNormRows(keys, head: head, among: 8..<12, keeping: 2)
            let rows = MLXArray(kept.map(Int32.init))
            let headKeys = keys[0..., head..<(head + 1), 0..., 0...].take(rows, axis: 2)
            let headValues = values[0..., head..<(head + 1), 0..., 0...].take(rows, axis: 2)
            #expect(ZonedKVCacheTests.maxDifference(stored[0][0..., head..<(head + 1), 4..<8, 0...], headKeys) == 0)
            #expect(ZonedKVCacheTests.maxDifference(stored[1][0..., head..<(head + 1), 4..<8, 0...], headValues) == 0)
        }
        #expect(ZonedKVCacheTests.maxDifference(stored[0][.ellipsis, ..<4, 0...], keys[.ellipsis, ..<4, 0...]) == 0)
        #expect(ZonedKVCacheTests.maxDifference(stored[0][.ellipsis, 8..., 0...], keys[.ellipsis, 12..., 0...]) == 0)
    }

    @Test("The mask made before an update is exactly as wide as the keys that update hands back, causal over the chunk, for prefill chunks and decode",
          arguments: [[3, 8, 5, 1, 1, 7, 1, 12, 1, 1, 30, 1], [40, 1, 1, 1, 9, 1]])
    func maskMatchesHandedKeys(chunks: [Int]) {
        let cache = EvictingKVCache(zones: Self.zones)
        for (step, n) in chunks.enumerated() {
            let mask = cache.makeMask(n: n, windowSize: nil, returnArray: true)
            let symbolic = cache.makeMask(n: n, windowSize: nil, returnArray: false)
            let tokens = ZonedKVCacheTests.tokens(n, seed: UInt64(500 + step))
            let (keys, _) = cache.update(keys: tokens, values: tokens)
            guard n > 1 else {
                guard case .none = mask else { Issue.record("decode must take no mask, got \(mask)"); continue }
                continue
            }
            guard case .array(let columns) = mask, case .causal = symbolic else {
                Issue.record("a chunk of \(n) must get an array mask on request and the causal mode otherwise")
                continue
            }
            #expect(columns.dim(-1) == keys.dim(2))
            let expected = createCausalMask(n: n, offset: keys.dim(2) - n)
            #expect(all(columns .== expected).item(Bool.self))
        }
        #expect(cache.evictedCount > 0)
        #expect(cache.offset == chunks.reduce(0, +))
    }

    @Test("The evicted share of the middle stays the configured fraction across uneven updates", arguments: [0.25, 0.5, 0.9])
    func evictedShareIsExact(fraction: Double) {
        let zones = Self.zones.evicting(fraction)
        let cache = EvictingKVCache(zones: zones)
        var seen = 0
        for (step, count) in [3, 8, 4, 1, 8, 5, 7, 8, 2, 6, 40, 8, 4].enumerated() {
            let tokens = ZonedKVCacheTests.tokens(count, seed: UInt64(600 + step))
            _ = cache.update(keys: tokens, values: tokens)
            seen += count
            #expect(cache.middleCount == zones.keptMiddleTokens(cache.middleCount + cache.evictedCount))
            #expect(cache.offset == seen)
            #expect(cache.windowCount < zones.windowTokens + zones.migrationStep)
        }
        #expect(cache.evictedCount > 0)
        #expect(!cache.isTrimmable)
    }

    @Test("Without eviction the cache only counts, never rewrites, so it stays trimmable")
    func zeroFractionKeepsEveryToken() {
        let cache = EvictingKVCache(zones: Self.zones.evicting(0))
        let tokens = ZonedKVCacheTests.tokens(30, seed: 701)
        _ = cache.update(keys: tokens, values: tokens)
        #expect((cache.storedCount, cache.evictedCount) == (30, 0))
        #expect(cache.isTrimmable)
        #expect(cache.trim(20) == 20)
        #expect(cache.storedCount == 10 && cache.windowCount + cache.middleCount + cache.sinkCount == 10)
    }

    static func greedy(_ model: any LanguageModel, cache: [any KVCache], maxTokens: Int = 16) throws -> [Int] {
        let input = LMInput(tokens: MLXArray(prompt.map(Int32.init)))
        var iterator = try TokenIterator(input: input, model: model, cache: cache, parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0))
        var tokens = [Int]()
        while let token = iterator.next() { tokens.append(token) }
        return tokens
    }

    @Test("Gemma 4 with nothing evicted decodes token for token as on its stock cache")
    func gemma4WithoutEvictionMatchesStock() throws {
        let model = try TinyModel.make(.gemma4)
        let zones = KVZones(sinkTokens: 4, windowTokens: 16, migrationStep: 8)
        let stock = try Self.greedy(model, cache: try model.newCache(parameters: nil))
        let evicting = try Self.greedy(model, cache: EvictingKVCache.replacingFullAttention(try model.newCache(parameters: nil), zones: zones))
        #expect(stock.count == 16)
        #expect(evicting == stock)
    }

    @Test("Gemma 4 with eviction evicts only on its full-attention layers, keeps its sliding layers stock, and decodes", arguments: [TinyModel.Family.gemma4, .gemma4Moe])
    func gemma4EvictsFullAttentionOnly(family: TinyModel.Family) throws {
        let model = try TinyModel.make(family)
        let zones = KVZones(sinkTokens: 4, windowTokens: 16, migrationStep: 8, evictedMiddleFraction: 0.5)
        let caches = EvictingKVCache.replacingFullAttention(try model.newCache(parameters: nil), zones: zones)
        let evicting = caches.compactMap { $0 as? EvictingKVCache }
        let sliding = try #require(caches.first { !($0 is EvictingKVCache) })
        #expect(!evicting.isEmpty)
        let tokens = try Self.greedy(model, cache: caches)
        #expect(tokens.count == 16)
        #expect(sliding.offset >= Self.prompt.count + tokens.count - 1)
        for cache in evicting {
            #expect(cache.evictedCount > 0)
            #expect(cache.offset == sliding.offset)
            #expect(cache.storedCount < cache.offset)
        }
    }
}

@Suite("fp16 eviction for Gemma 4 in the engine")
struct EvictingKVCacheEngineTests {
    @Test("At load only a model that reads its cache through update() with --kv-evict set evicts in fp16; zones stay off for it and on for Qwen3",
          arguments: [
            (TinyModel.Family.gemma4, KVZones.standard.evicting(0.5), false, true),
            (.gemma4, KVZones.standard, false, false),
            (.qwen3HeadDim32, KVZones.standard.evicting(0.5), true, false),
          ])
    func loadFindsUpdateReaders(family: TinyModel.Family, zones: KVZones, takesZones: Bool, evictsInFP16: Bool) async throws {
        let engine = MLXInferenceEngine(kvPrecision: .q8, kvZones: zones)
        await engine.attach(try TinyModel.container(family))
        await engine.probeKVZones()
        #expect(await engine.supportsKVZones == takesZones)
        #expect(await engine.evictsKVInFullPrecision == evictsInFP16)
        #expect(await engine.planSettings.evictsThroughUpdate == evictsInFP16)
    }

    @Test("A quantized Gemma 4 launch with fp16-middle zones generates on the evicting cache: the text of a direct evicting run")
    func launchGeneratesOnEvictingCache() async throws {
        let container = try TinyModel.container(.gemma4)
        let zones = KVZones(sinkTokens: 4, windowTokens: 16, migrationStep: 8, evictedMiddleFraction: 0.5).keepingMiddleFullPrecision()
        var quantized = GenerateParameters(maxTokens: 24, temperature: 0)
        quantized.kvBits = 8
        let params = quantized
        let prompt = EvictingKVCacheTests.prompt
        let streams = try await container.perform { (context: ModelContext) -> [AsyncStream<Generation>] in
            let input = LMInput(tokens: MLXArray(prompt.map(Int32.init)))
            var exact = params
            exact.kvBits = nil
            let reference = try MLXLMCommon.generate(
                input: input, cache: EvictingKVCache.replacingFullAttention(try context.model.newCache(parameters: exact), zones: zones),
                parameters: exact, context: context)
            let launched = try MLXInferenceEngine.launch(
                input: input, prompt: CachedPrompt(boundary: prompt.count - 1),
                options: LaunchOptions(params: params, kvZones: zones, kvPrecision: .q8), context: context).stream
            return [launched, reference]
        }
        var texts = [String]()
        for stream in streams { texts.append(await ZonedKVCacheEngineTests.text(stream)) }
        #expect(!texts[0].isEmpty)
        #expect(texts[0] == texts[1])
    }
}
