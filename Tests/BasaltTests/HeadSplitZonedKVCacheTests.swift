import Testing
import MLX
import MLXLMCommon
import BasaltCore
@testable import BasaltInference

@Suite("Head split with zoned retrieval heads: the heads that keep the whole context store its middle quantized (#294)")
struct HeadSplitZonedKVCacheTests {
    static let kvHeads = 4, groups = 2, dim = 64
    static let scale = HeadSplitKVCacheTests.scale
    static let zones = KVZones(sinkTokens: 16, windowTokens: 64, migrationStep: 32)
    static let zoning = HeadSplitKVCache.Zoning(zones: zones, precision: .k8v4)

    static func queryHeads(_ heads: [Int]) -> MLXArray {
        MLXArray(heads.flatMap { ($0 * groups) ..< (($0 + 1) * groups) }.map(Int32.init))
    }

    static func bytes(_ cache: any KVCache) -> Int { cache.state.reduce(0) { $0 + $1.nbytes } }

    static func normal(_ tokens: Int, heads: Int = kvHeads, _ seed: UInt64) -> MLXArray {
        HeadSplitKVCacheTests.normal([1, heads, tokens, dim], seed)
    }

    @Test("Retrieval heads attend exactly like a zoned cache holding only those heads, streaming heads like the fp16 split, in fewer bytes",
          arguments: [[1, 3], [0, 1, 2, 3], [2]])
    func matchesZonedCacheOfRetrievalHeads(retrieval: [Int]) throws {
        let split = HeadSplitKVCache(retrieval: retrieval, zoning: Self.zoning)
        let plain = HeadSplitKVCache(retrieval: retrieval)
        let reference = ZonedKVCache(zones: Self.zones, precision: .k8v4)
        let (kept, keptQueries) = (MLXArray(retrieval.map(Int32.init)), Self.queryHeads(retrieval))
        let (keys, values) = (Self.normal(601, 21), Self.normal(601, 22))
        var outputs: (split: MLXArray, plain: MLXArray, reference: MLXArray)?
        for (start, end) in [(0, 128), (128, 256), (256, 384), (384, 512), (512, 600), (600, 601)] {
            let queries = Self.normal(end - start, heads: Self.kvHeads * Self.groups, 23 + UInt64(start))
            let (k, v) = (keys[.ellipsis, start ..< end, 0...], values[.ellipsis, start ..< end, 0...])
            let mask: MLXFast.ScaledDotProductAttentionMaskMode = end - start == 1 ? .none : .causal
            outputs = (
                split.updateAndAttend(queries: queries, keys: k, values: v, scale: Self.scale, mask: mask),
                plain.updateAndAttend(queries: queries, keys: k, values: v, scale: Self.scale, mask: mask),
                reference.updateAndAttend(queries: queries[0..., keptQueries], keys: k[0..., kept], values: v[0..., kept], scale: Self.scale, mask: mask))
        }
        let last = try #require(outputs)
        #expect(reference.quantizedCount > 0)
        #expect(allClose(last.split[0..., keptQueries], last.reference, atol: 1e-5).item(Bool.self))
        let streaming = (0 ..< Self.kvHeads).filter { !retrieval.contains($0) }
        if !streaming.isEmpty {
            let streamingQueries = Self.queryHeads(streaming)
            #expect(allClose(last.split[0..., streamingQueries], last.plain[0..., streamingQueries], atol: 1e-5).item(Bool.self))
        }
        #expect(split.offset == 601)
        #expect(split.retrievalTokens == 601)
        #expect(Self.bytes(split) < Self.bytes(plain))
    }

    @Test("A copy decodes like the original, and trimming within the recent window keeps both parts in step")
    func copyAndTrim() throws {
        let split = HeadSplitKVCache(retrieval: [0, 2], zoning: Self.zoning)
        let (keys, values) = (Self.normal(400, 31), Self.normal(400, 32))
        let queries = Self.normal(400, heads: Self.kvHeads * Self.groups, 33)
        eval(split.updateAndAttend(queries: queries, keys: keys, values: values, scale: Self.scale, mask: .causal))
        let copied = try #require(split.copy() as? HeadSplitKVCache)
        let (q, k, v) = (queries[.ellipsis, ..<1, 0...], keys[.ellipsis, ..<1, 0...], values[.ellipsis, ..<1, 0...])
        let a = split.updateAndAttend(queries: q, keys: k, values: v, scale: Self.scale, mask: .none)
        let b = copied.updateAndAttend(queries: q, keys: k, values: v, scale: Self.scale, mask: .none)
        #expect(allClose(a, b).item(Bool.self))
        #expect(split.isTrimmable)
        #expect(split.trim(8) == 8)
        #expect(split.offset == 393)
        #expect(split.retrievalTokens == 393)
        #expect(copied.offset == 401)
    }

    @Test("On a tiny model, a zoned split where every head retrieves decodes exactly like the zoned cache, with a quantized middle")
    func allRetrievalMatchesZonedModel() throws {
        let model = try TinyModel.make(.qwen3HeadDim32)
        let prompt = MLXArray((0 ..< 300).map { Int32(1 + $0 % 200) }).reshaped(1, -1)
        let zoned = ZonedKVCache.replacingFullAttention(try model.newCache(parameters: nil), zones: Self.zones, precision: .k8v4)
        let layers = zoned.filter { $0 is ZonedKVCache }.count
        let split = KVHeadMask(retrieval: Array(repeating: Array(0 ..< 8), count: layers))
            .replacingFullAttention(try model.newCache(parameters: nil), zoning: Self.zoning)
        #expect(split.filter { $0 is HeadSplitKVCache }.count == layers)
        #expect(allClose(model(prompt, cache: zoned)[0, -1], model(prompt, cache: split)[0, -1], atol: 1e-4).item(Bool.self))
        let next = MLXArray([Int32(5)]).reshaped(1, 1)
        let (x, y) = (model(next, cache: zoned)[0, -1], model(next, cache: split)[0, -1])
        #expect(allClose(x, y, atol: 1e-4).item(Bool.self))
        #expect(argMax(x).item(Int.self) == argMax(y).item(Int.self))
        #expect(zoned.compactMap { $0 as? ZonedKVCache }.allSatisfy { $0.quantizedCount > 0 })
        #expect(split.compactMap { $0 as? HeadSplitKVCache }.allSatisfy { !$0.materializedForCaller })
    }
}
