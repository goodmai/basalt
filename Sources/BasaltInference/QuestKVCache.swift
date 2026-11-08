import Foundation
import MLX
import MLXLMCommon

final class QuestKVCache: KVCacheAttentionProtocol {
    static let blockTokens = 64
    static let sinkTokens = 128
    static let windowTokens = 256

    let topK: Int
    private let base = KVCacheSimple()
    private var maxima: MLXArray?
    private var minima: MLXArray?
    private var indexedBlocks = 0
    private(set) var sparseSteps = 0
    private(set) var materializedForCaller = false
    private let sinks = MLXArray(Int32(0) ..< Int32(QuestKVCache.sinkTokens))
    private let windowSteps = MLXArray(Int32(0) ..< Int32(QuestKVCache.windowTokens + QuestKVCache.blockTokens))
    private let blockSteps = MLXArray(Int32(0) ..< Int32(QuestKVCache.blockTokens)).reshaped(1, 1, 1, -1)

    init(topK: Int) {
        self.topK = topK
    }

    static func replacingFullAttention(_ caches: [any KVCache], topK: Int) -> [any KVCache] {
        caches.map { type(of: $0) == KVCacheSimple.self ? QuestKVCache(topK: topK) : $0 }
    }

    static func modelAttendsThroughCache(_ model: any LanguageModel) throws -> Bool {
        let caches = replacingFullAttention(try model.newCache(parameters: nil), topK: 1)
        let quest = caches.compactMap { $0 as? QuestKVCache }
        guard !quest.isEmpty else { return false }
        _ = model(LMInput.Text(tokens: MLXArray([Int32(0)]).reshaped(1, 1)), cache: caches, state: nil)
        return quest.allSatisfy { !$0.materializedForCaller }
    }

    var offset: Int { base.offset }
    var maxSize: Int? { nil }
    var isTrimmable: Bool { true }

    static func blockScores(_ q: MLXArray, upper: MLXArray, lower: MLXArray) -> MLXArray {
        let positive = maximum(q, 0).sum(axis: 2, keepDims: true)
        let negative = minimum(q, 0).sum(axis: 2, keepDims: true)
        return (matmul(positive, upper.swappedAxes(-1, -2)) + matmul(negative, lower.swappedAxes(-1, -2))).squeezed(axis: 2)
    }

    func innerState() -> [MLXArray] { base.innerState() + [maxima, minima].compactMap { $0 } }

    var state: [MLXArray] {
        get { base.state }
        set { preconditionFailure("QuestKVCache is never restored from a serialized prompt cache") }
    }

    var metaState: [String] {
        get { [String(topK)] }
        set { preconditionFailure("QuestKVCache is never restored from a serialized prompt cache") }
    }

    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let (k, v) = base.update(keys: keys, values: values)
        index(k)
        let first = Self.sinkTokens / Self.blockTokens
        let last = (offset - Self.windowTokens) / Self.blockTokens
        guard queries.dim(2) == 1, let maxima, let minima, last - first > topK else {
            return MLXFast.scaledDotProductAttention(queries: queries, keys: k, values: v, scale: scale, mask: mask)
        }
        sparseSteps += 1
        let (batch, kvHeads) = (k.dim(0), k.dim(1))
        let q = queries.reshaped(batch, kvHeads, queries.dim(1) / kvHeads, k.dim(3))
        let score = Self.blockScores(q, upper: maxima[0..., 0..., first ..< last], lower: minima[0..., 0..., first ..< last])
        let chosen = argPartition(-score, kth: topK - 1, axis: -1)[.ellipsis, ..<topK].asType(.int32) + Int32(first)
        let blocks = (chosen.expandedDimensions(axis: -1) * Int32(Self.blockTokens) + blockSteps).reshaped(batch, kvHeads, -1)
        let windowStart = last * Self.blockTokens
        let fixed = concatenated([sinks, windowSteps[..<(offset - windowStart)] + Int32(windowStart)])
        let rows = concatenated([broadcast(fixed.reshaped(1, 1, -1), to: [batch, kvHeads, fixed.dim(0)]), blocks], axis: -1)
            .expandedDimensions(axis: -1)
        let count = rows.dim(2)
        let buffers = base.innerState()
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: takeAlong(buffers[0], broadcast(rows, to: [batch, kvHeads, count, k.dim(3)]), axis: 2),
            values: takeAlong(buffers[1], broadcast(rows, to: [batch, kvHeads, count, v.dim(3)]), axis: 2),
            scale: scale, mask: .none)
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        materializedForCaller = true
        return base.update(keys: keys, values: values)
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        let removed = base.trim(n)
        let kept = min(indexedBlocks, offset / Self.blockTokens)
        if kept < indexedBlocks {
            maxima = kept == 0 ? nil : maxima?[0..., 0..., ..<kept]
            minima = kept == 0 ? nil : minima?[0..., 0..., ..<kept]
            indexedBlocks = kept
        }
        return removed
    }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        base.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    func copy() -> any KVCache {
        let copied = QuestKVCache(topK: topK)
        if !base.state.isEmpty { copied.base.state = base.state }
        copied.maxima = maxima
        copied.minima = minima
        copied.indexedBlocks = indexedBlocks
        copied.sparseSteps = sparseSteps
        copied.materializedForCaller = materializedForCaller
        return copied
    }

    private func index(_ keys: MLXArray) {
        let complete = offset / Self.blockTokens
        guard complete > indexedBlocks else { return }
        let blocks = keys[.ellipsis, (indexedBlocks * Self.blockTokens) ..< (complete * Self.blockTokens), 0...]
            .reshaped(keys.dim(0), keys.dim(1), complete - indexedBlocks, Self.blockTokens, keys.dim(3))
        let upper = blocks.max(axis: 3), lower = blocks.min(axis: 3)
        maxima = maxima.map { concatenated([$0, upper], axis: 2) } ?? upper
        minima = minima.map { concatenated([$0, lower], axis: 2) } ?? lower
        indexedBlocks = complete
    }
}
