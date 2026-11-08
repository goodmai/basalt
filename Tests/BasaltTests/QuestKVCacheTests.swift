import Testing
import MLX
import MLXLMCommon
@testable import BasaltInference

@Suite("Quest: a decode step attends to sinks, a recent window and the best key blocks between them (#33)")
struct QuestKVCacheTests {
    static let kvHeads = 2, groups = 4, dim = 32
    static let scale = Float(1) / Float(32).squareRoot()

    static func filled(_ cache: QuestKVCache, keys: MLXArray, values: MLXArray, chunk: Int = 256) {
        var start = 0
        while start < keys.dim(2) {
            let end = min(keys.dim(2), start + chunk)
            let queries = normal([1, kvHeads * groups, end - start, dim], UInt64(start))
            eval(cache.updateAndAttend(
                queries: queries, keys: keys[.ellipsis, start ..< end, 0...], values: values[.ellipsis, start ..< end, 0...],
                scale: scale, mask: .causal))
            start = end
        }
    }

    static func normal(_ shape: [Int], _ seed: UInt64) -> MLXArray {
        MLXRandom.normal(shape, key: MLXRandom.key(seed))
    }

    static func dense(_ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray) -> MLXArray {
        MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: scale, mask: .none)
    }

    @Test("While the middle has no more blocks than top-k, a decode step equals dense attention", arguments: [300, 1000])
    func smallMiddleIsDense(tokens: Int) {
        let keys = Self.normal([1, Self.kvHeads, tokens, Self.dim], 1)
        let values = Self.normal([1, Self.kvHeads, tokens, Self.dim], 2)
        let cache = QuestKVCache(topK: 64)
        Self.filled(cache, keys: keys[.ellipsis, ..<(tokens - 1), 0...], values: values[.ellipsis, ..<(tokens - 1), 0...])
        let query = Self.normal([1, Self.kvHeads * Self.groups, 1, Self.dim], 3)
        let output = cache.updateAndAttend(
            queries: query, keys: keys[.ellipsis, (tokens - 1)..., 0...], values: values[.ellipsis, (tokens - 1)..., 0...],
            scale: Self.scale, mask: .none)
        #expect(allClose(output, Self.dense(query, keys, values), atol: 1e-5).item(Bool.self))
        #expect(cache.sparseSteps == 0)
    }

    @Test("On a long context, the block holding the keys the query matches is picked, so the output matches dense attention")
    func needleBlockIsPicked() {
        let tokens = 4096, needle = 1700
        let direction = Self.normal([1, Self.kvHeads, 1, Self.dim], 4)
        let query = (broadcast(direction.expandedDimensions(axis: 2), to: [1, Self.kvHeads, Self.groups, 1, Self.dim])
            + Self.normal([1, Self.kvHeads, Self.groups, 1, Self.dim], 5) * 0.1).reshaped(1, Self.kvHeads * Self.groups, 1, Self.dim)
        let keys = Self.normal([1, Self.kvHeads, tokens, Self.dim], 6) * 0.05
        keys[.ellipsis, needle ..< (needle + 8), 0...] = broadcast(direction * 2, to: [1, Self.kvHeads, 8, Self.dim])
        let values = Self.normal([1, Self.kvHeads, tokens, Self.dim], 7)
        let cache = QuestKVCache(topK: 4)
        Self.filled(cache, keys: keys[.ellipsis, ..<(tokens - 1), 0...], values: values[.ellipsis, ..<(tokens - 1), 0...])
        let output = cache.updateAndAttend(
            queries: query, keys: keys[.ellipsis, (tokens - 1)..., 0...], values: values[.ellipsis, (tokens - 1)..., 0...],
            scale: Self.scale, mask: .none)
        let dense = Self.dense(query, keys, values)
        let farFromNeedle = Self.dense(
            query, concatenated([keys[.ellipsis, ..<needle, 0...], keys[.ellipsis, (needle + 64)..., 0...]], axis: 2),
            concatenated([values[.ellipsis, ..<needle, 0...], values[.ellipsis, (needle + 64)..., 0...]], axis: 2))
        #expect(cache.sparseSteps == 1)
        #expect(allClose(output, dense, atol: 5e-2).item(Bool.self))
        #expect(!allClose(farFromNeedle, dense, atol: 0.3).item(Bool.self))
    }

    @Test("A copy decodes the next step exactly like the cache it was copied from, and the two stay independent")
    func copyIsIndependent() throws {
        let tokens = 2000
        let keys = Self.normal([1, Self.kvHeads, tokens, Self.dim], 12)
        let values = Self.normal([1, Self.kvHeads, tokens, Self.dim], 13)
        let cache = QuestKVCache(topK: 4)
        Self.filled(cache, keys: keys[.ellipsis, ..<(tokens - 1), 0...], values: values[.ellipsis, ..<(tokens - 1), 0...])
        let copied = try #require(cache.copy() as? QuestKVCache)
        let query = Self.normal([1, Self.kvHeads * Self.groups, 1, Self.dim], 14)
        let next = (keys[.ellipsis, (tokens - 1)..., 0...], values[.ellipsis, (tokens - 1)..., 0...])
        let original = cache.updateAndAttend(queries: query, keys: next.0, values: next.1, scale: Self.scale, mask: .none)
        let fromCopy = copied.updateAndAttend(queries: query, keys: next.0, values: next.1, scale: Self.scale, mask: .none)
        #expect(cache.sparseSteps == 1 && copied.sparseSteps == 1)
        #expect(allClose(original, fromCopy, atol: 1e-6).item(Bool.self))
        #expect(cache.offset == tokens && copied.offset == tokens)
        #expect(cache.trim(500) == 500)
        #expect(copied.offset == tokens)
    }

    @Test("A multi-token step stays dense, the block bounds are evaluated with the cache, and trimming drops the bounds of blocks that no longer exist")
    func prefillDenseAndTrim() {
        let tokens = 2000
        let keys = Self.normal([1, Self.kvHeads, tokens, Self.dim], 8)
        let values = Self.normal([1, Self.kvHeads, tokens, Self.dim], 9)
        let cache = QuestKVCache(topK: 2)
        Self.filled(cache, keys: keys[.ellipsis, ..<1900, 0...], values: values[.ellipsis, ..<1900, 0...])
        let queries = Self.normal([1, Self.kvHeads * Self.groups, 100, Self.dim], 10)
        let output = cache.updateAndAttend(
            queries: queries, keys: keys[.ellipsis, 1900..., 0...], values: values[.ellipsis, 1900..., 0...],
            scale: Self.scale, mask: .causal)
        let expected = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: Self.scale,
            mask: .array(createCausalMask(n: 100, offset: 1900)))
        #expect(allClose(output, expected, atol: 1e-4).item(Bool.self))
        #expect(cache.innerState().count == 4)
        #expect(cache.trim(1700) == 1700)
        #expect(cache.offset == 300)
        let query = Self.normal([1, Self.kvHeads * Self.groups, 1, Self.dim], 11)
        let next = (keys[.ellipsis, 300 ..< 301, 0...], values[.ellipsis, 300 ..< 301, 0...])
        let trimmed = cache.updateAndAttend(queries: query, keys: next.0, values: next.1, scale: Self.scale, mask: .none)
        #expect(allClose(trimmed, Self.dense(query, keys[.ellipsis, ..<301, 0...], values[.ellipsis, ..<301, 0...]), atol: 1e-5).item(Bool.self))
    }

    @Test("Quest applies to models that attend through the cache and refuses Gemma 4, whose attention reads the whole cache",
          arguments: [(TinyModel.Family.qwen3, true), (.qwen35, true), (.gemma4, false)])
    func modelSupport(family: TinyModel.Family, applies: Bool) throws {
        #expect(try QuestKVCache.modelAttendsThroughCache(try TinyModel.make(family)) == applies)
    }

    @Test("On a tiny Qwen3, a short prompt decodes exactly like the dense cache")
    func tinyModelMatchesDense() throws {
        let model = try TinyModel.make(.qwen3)
        let prompt = MLXArray((0 ..< 40).map { Int32(1 + $0 % 200) }).reshaped(1, -1)
        let dense = try model.newCache(parameters: nil)
        let quest = QuestKVCache.replacingFullAttention(try model.newCache(parameters: nil), topK: 2)
        #expect(quest.contains { $0 is QuestKVCache })
        #expect(allClose(model(prompt, cache: dense)[0, -1], model(prompt, cache: quest)[0, -1], atol: 1e-5).item(Bool.self))
        let next = MLXArray([Int32(5)]).reshaped(1, 1)
        #expect(allClose(model(next, cache: dense)[0, -1], model(next, cache: quest)[0, -1], atol: 1e-4).item(Bool.self))
    }

    @Test("Block scores from the group-summed sign split equal the per-group max(q·upper, q·lower) bound")
    func blockScoresMatchBound() {
        let q = Self.normal([2, Self.kvHeads, Self.groups, Self.dim], 20)
        let a = Self.normal([2, Self.kvHeads, 37, Self.dim], 21), b = Self.normal([2, Self.kvHeads, 37, Self.dim], 22)
        let (upper, lower) = (maximum(a, b), minimum(a, b))
        let wide = q.expandedDimensions(axis: 3)
        let bound = maximum(wide * upper.expandedDimensions(axis: 2), wide * lower.expandedDimensions(axis: 2)).sum(axis: -1).sum(axis: 2)
        #expect(allClose(QuestKVCache.blockScores(q, upper: upper, lower: lower), bound, atol: 1e-4).item(Bool.self))
    }
}

@Suite("Quest kernels: block selection and sparse attention match the gathered MLX graph (#280)")
struct QuestKernelParityTests {
    static func normal(_ shape: [Int], _ seed: UInt64) -> MLXArray {
        MLXRandom.normal(shape, key: MLXRandom.key(seed))
    }

    static func decoded(_ keys: MLXArray, _ values: MLXArray, query: MLXArray, topK: Int) -> (QuestKVCache, MLXArray) {
        let (cache, tokens) = (QuestKVCache(topK: topK), keys.dim(2))
        _ = cache.update(keys: keys[.ellipsis, ..<(tokens - 1), 0...], values: values[.ellipsis, ..<(tokens - 1), 0...])
        let output = cache.updateAndAttend(
            queries: query, keys: keys[.ellipsis, (tokens - 1)..., 0...], values: values[.ellipsis, (tokens - 1)..., 0...],
            scale: Float(1) / Float(keys.dim(3)).squareRoot(), mask: .none)
        return (cache, output)
    }

    @Test("The selection kernel picks exactly the blocks argPartition picks from the float32 matmul scores",
          arguments: [(32, 4, 16, DType.float32), (128, 4, 16, .float32), (128, 4, 32, .bfloat16), (256, 4, 32, .float32), (64, 8, 1, .float32)])
    func selectionMatchesArgPartition(dim: Int, groups: Int, topK: Int, dtype: DType) {
        let (kvHeads, blocks) = (3, 300)
        let q = Self.normal([2, kvHeads * groups, 1, dim], 30).asType(dtype)
        let (a, b) = (Self.normal([2, kvHeads, blocks, dim], 31), Self.normal([2, kvHeads, blocks, dim], 32))
        let (upper, lower) = (maximum(a, b).asType(dtype), minimum(a, b).asType(dtype))
        let middle = 2 ..< 290
        let reference = QuestKVCache.topBlocks(
            q.asType(.float32), upper: upper.asType(.float32), lower: lower.asType(.float32), middle: middle, topK: topK)
        let kernel = QuestKernel.selectedBlocks(
            q, upper: upper, lower: lower, first: middle.lowerBound, last: middle.upperBound, topK: topK)
        #expect(kernel.shape == reference.shape)
        #expect((sorted(kernel, axis: -1) .== sorted(reference, axis: -1)).all().item(Bool.self))
    }

    @Test("A block whose bound is NaN is always selected, never silently skipped")
    func nonFiniteBoundIsPicked() {
        let q = Self.normal([1, 8, 1, 32], 33)
        let (a, b) = (Self.normal([1, 2, 100, 32], 34), Self.normal([1, 2, 100, 32], 35))
        let (upper, lower) = (maximum(a, b), minimum(a, b))
        upper[0, 1, 40, 3] = MLXArray(Float.nan)
        let chosen = QuestKernel.selectedBlocks(q, upper: upper, lower: lower, first: 2, last: 90, topK: 4)
        #expect((chosen[0, 1] .== Int32(40)).any().item(Bool.self))
    }

    @Test("The sparse attention kernel equals the gathered SDPA over sinks, picked blocks and the window, scored in float32",
          arguments: [(32, 4, 4, DType.float32), (128, 4, 16, .float32), (256, 4, 32, .float32), (128, 4, 16, .bfloat16), (256, 4, 32, .float16)])
    func attentionMatchesGathered(dim: Int, groups: Int, topK: Int, dtype: DType) {
        let (kvHeads, tokens) = (2, 4000)
        let keys = Self.normal([1, kvHeads, tokens, dim], 36).asType(dtype)
        let values = Self.normal([1, kvHeads, tokens, dim], 37).asType(dtype)
        let query = Self.normal([1, kvHeads * groups, 1, dim], 38).asType(dtype)
        let (cache, output) = Self.decoded(keys, values, query: query, topK: topK)
        let (exact, _) = Self.decoded(keys.asType(.float32), values.asType(.float32), query: query.asType(.float32), topK: topK)
        let reference = exact.gatheredAttention(queries: query.asType(.float32), scale: Float(1) / Float(dim).squareRoot())
        #expect(cache.sparseSteps == 1)
        #expect(output.shape == reference.shape && output.dtype == dtype)
        #expect(allClose(output.asType(.float32), reference, atol: dtype == .float32 ? 1e-5 : 5e-3).item(Bool.self))
    }

    @Test("On a tiny Qwen3 with head dim 32, a long prompt decodes the same tokens through the kernels as through the gathered graph")
    func tinyModelKernelMatchesGathered() throws {
        let model = try TinyModel.make(.qwen3HeadDim32)
        let prompt = MLXArray((0 ..< 700).map { Int32(1 + ($0 * 37) % 250) }).reshaped(1, -1)
        let caches = try [true, false].map { QuestKVCache.replacingFullAttention(try model.newCache(parameters: nil), topK: 2, usesKernel: $0) }
        var tokens = caches.map { argMax(model(prompt, cache: $0)[0, -1]).reshaped(1, 1) }
        for _ in 0 ..< 8 {
            let logits = zip(tokens, caches).map { model($0, cache: $1)[0, -1] }
            #expect(allClose(logits[0], logits[1], atol: 1e-4).item(Bool.self))
            tokens = logits.map { argMax($0).reshaped(1, 1) }
            #expect(tokens[0].item(Int32.self) == tokens[1].item(Int32.self))
        }
        #expect(caches.allSatisfy { $0.compactMap { $0 as? QuestKVCache }.allSatisfy { $0.sparseSteps == 8 } })
    }
}
