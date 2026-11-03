import Testing
import MLX
import MLXLMCommon
@testable import BasaltInference

@Suite("Head-level KV: retrieval heads keep the whole context, streaming heads keep sinks and a recent window (#37)")
struct HeadSplitKVCacheTests {
    static let kvHeads = 4, groups = 2, dim = 16
    static let scale = Float(0.25)

    static func feed(_ cache: HeadSplitKVCache, tokens: Int, chunk: Int, seed: UInt64) -> (keys: MLXArray, values: MLXArray) {
        MLXRandom.seed(seed)
        let keys = MLXRandom.normal([1, kvHeads, tokens, dim])
        let values = MLXRandom.normal([1, kvHeads, tokens, dim])
        var start = 0
        while start < tokens {
            let end = min(tokens, start + chunk)
            let queries = MLXRandom.normal([1, kvHeads * groups, end - start, dim])
            eval(cache.updateAndAttend(
                queries: queries, keys: keys[.ellipsis, start ..< end, 0...], values: values[.ellipsis, start ..< end, 0...],
                scale: scale, mask: end - start == 1 ? .none : .causal))
            start = end
        }
        return (keys, values)
    }

    static func reference(queries: MLXArray, keys: MLXArray, values: MLXArray, retrieval: Set<Int>, streamed: Int) -> MLXArray {
        let total = keys.dim(2)
        let recent = streamed - HeadSplitKVCache.sinkTokens
        let kept = streamed >= total
            ? Array(0 ..< total)
            : Array(0 ..< HeadSplitKVCache.sinkTokens) + Array((total - recent) ..< total)
        let heads = (0 ..< kvHeads).map { head -> MLXArray in
            let q = queries[0..., (head * groups) ..< ((head + 1) * groups)]
            let rows = retrieval.contains(head) ? MLXArray(Array(0 ..< Int32(total))) : MLXArray(kept.map(Int32.init))
            let k = keys[0..., head ..< (head + 1)][.ellipsis, rows, 0...]
            let v = values[0..., head ..< (head + 1)][.ellipsis, rows, 0...]
            return MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        }
        return concatenated(heads, axis: 1)
    }

    @Test("Decode attention equals per-head SDPA over each head's visible tokens, in the original head order",
          arguments: [[0, 1, 2, 3], [1, 3], [2], []], [300, 700])
    func decodeMatchesReference(retrieval: [Int], context: Int) {
        let cache = HeadSplitKVCache(retrieval: retrieval)
        let (keys, values) = Self.feed(cache, tokens: context, chunk: 128, seed: 7)
        MLXRandom.seed(11)
        let queries = MLXRandom.normal([1, Self.kvHeads * Self.groups, 1, Self.dim])
        let newKey = MLXRandom.normal([1, Self.kvHeads, 1, Self.dim])
        let newValue = MLXRandom.normal([1, Self.kvHeads, 1, Self.dim])
        let streamed = (retrieval.count < Self.kvHeads ? cache.streamingTokens : context) + 1
        let output = cache.updateAndAttend(queries: queries, keys: newKey, values: newValue, scale: Self.scale, mask: .none)
        let allKeys = concatenated([keys, newKey], axis: 2), allValues = concatenated([values, newValue], axis: 2)
        let expected = Self.reference(queries: queries, keys: allKeys, values: allValues, retrieval: Set(retrieval), streamed: streamed)
        #expect(output.shape == expected.shape)
        #expect(allClose(output, expected, atol: 1e-5).item(Bool.self))
        #expect(cache.offset == context + 1)
        #expect(cache.retrievalTokens == (retrieval.isEmpty ? 0 : context + 1))
        if retrieval.count < Self.kvHeads {
            #expect(cache.streamingTokens <= HeadSplitKVCache.sinkTokens + HeadSplitKVCache.windowTokens + HeadSplitKVCache.trimStep)
        }
    }

    @Test("On a tiny model, a mask with every head retrieving, or a partial mask before any token leaves the window, decodes exactly like the dense cache",
          arguments: [(TinyModel.Family.llama, false), (.llama, true), (.qwen35, false), (.qwen35, true)])
    func matchesDenseModelWhileNothingIsDropped(family: TinyModel.Family, partial: Bool) throws {
        let model = try TinyModel.make(family)
        let prompt = MLXArray((0 ..< 40).map { Int32(1 + $0 % 200) }).reshaped(1, -1)
        let dense = try model.newCache(parameters: nil)
        let fullAttention = dense.filter { type(of: $0) == KVCacheSimple.self }.count
        let heads = Array(0 ..< 8)
        let mask = KVHeadMask(retrieval: Array(repeating: partial ? [1] : heads, count: fullAttention))
        let split = mask.replacingFullAttention(try model.newCache(parameters: nil))
        #expect(split.contains { $0 is HeadSplitKVCache })
        let a = model(prompt, cache: dense)[0, -1]
        let b = model(prompt, cache: split)[0, -1]
        #expect(allClose(a, b, atol: 1e-5).item(Bool.self))
        let next = MLXArray([Int32(5)]).reshaped(1, 1)
        let (x, y) = (model(next, cache: dense)[0, -1], model(next, cache: split)[0, -1])
        #expect(allClose(x, y, atol: 1e-3).item(Bool.self))
        #expect(argMax(x).item(Int.self) == argMax(y).item(Int.self))
        #expect(split.compactMap { $0 as? HeadSplitKVCache }.allSatisfy { !$0.materializedForCaller })
    }

    @Test("Trimming takes tokens off both the retrieval and the streaming part, never the sinks")
    func trimRemovesRecentTokens() {
        let cache = HeadSplitKVCache(retrieval: [0])
        _ = Self.feed(cache, tokens: 200, chunk: 50, seed: 3)
        #expect(cache.trim(8) == 8)
        #expect(cache.offset == 192)
        #expect(cache.retrievalTokens == 192)
        #expect(cache.streamingTokens == 192)
    }

    @Test("Splitting applies only when a few heads carry the retrieval: a hybrid whose every attention head retrieves stays dense")
    func concentrationGuard() {
        #expect(KVHeadMask.isConcentrated([[1.07, 0.28, 0.01, 0.0003], [0.11, 0.02, 0.6, 0.004]]))
        #expect(!KVHeadMask.isConcentrated([[0.41, 0.67, 0.65, 0.65], [0.79, 1.04, 0.94, 1.26]]))
        #expect(!KVHeadMask.isConcentrated([[0, 0], [0, 0]]))
    }

    @Test("The mask keeps the requested share of heads with the highest scores, across layers")
    func classifyKeepsTopHeads() {
        let mask = KVHeadMask.classify(scores: [[0.9, 0.1, 0.5, 0.2], [0.05, 0.8, 0.3, 0.7]], ratio: 0.5)
        #expect(mask.retrieval == [[0, 2], [1, 3]])
        #expect(KVHeadMask.classify(scores: [[0.1, 0.2]], ratio: 0).retrieval == [[]])
    }

    @Test("Head scores cover every full-attention layer and KV head of a model, with the needle's attention mass")
    func classifierScoresEveryHead() throws {
        let scores = try KVHeadClassifier.scores(model: try TinyModel.make(.llama), tokenizer: ByteTokenizer(), length: 256)
        #expect(scores.count == 2)
        #expect(scores.allSatisfy { $0.count == 2 && $0.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 3.0001 } })
    }
}
