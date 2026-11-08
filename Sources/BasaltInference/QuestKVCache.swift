import Foundation
import MLX
import MLXLMCommon

final class QuestKVCache: KVCacheAttentionProtocol {
    static let blockTokens = 64
    static let sinkTokens = 128
    static let windowTokens = 256

    let topK: Int
    let usesKernel: Bool
    private let base = KVCacheSimple()
    private var maxima: MLXArray?
    private var minima: MLXArray?
    private var indexedBlocks = 0
    private(set) var sparseSteps = 0
    private(set) var materializedForCaller = false
    private let sinks = MLXArray(Int32(0) ..< Int32(QuestKVCache.sinkTokens))
    private let windowSteps = MLXArray(Int32(0) ..< Int32(QuestKVCache.windowTokens + QuestKVCache.blockTokens))
    private let blockSteps = MLXArray(Int32(0) ..< Int32(QuestKVCache.blockTokens)).reshaped(1, 1, 1, -1)

    init(topK: Int, usesKernel: Bool = true) {
        self.topK = topK
        self.usesKernel = usesKernel
    }

    static func replacingFullAttention(_ caches: [any KVCache], topK: Int, usesKernel: Bool = true) -> [any KVCache] {
        caches.map { type(of: $0) == KVCacheSimple.self ? QuestKVCache(topK: topK, usesKernel: usesKernel) : $0 }
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
        guard queries.dim(2) == 1, maxima != nil, middle.count > topK else {
            return MLXFast.scaledDotProductAttention(queries: queries, keys: k, values: v, scale: scale, mask: mask)
        }
        sparseSteps += 1
        let buffers = base.innerState()
        guard usesKernel, QuestKernel.fits(queries: queries, keys: buffers[0], values: buffers[1], middleBlocks: middle.count) else {
            return gatheredAttention(queries: queries, scale: scale)
        }
        return kernelAttention(queries: queries, scale: scale)
    }

    func kernelAttention(queries: MLXArray, scale: Float) -> MLXArray {
        guard let maxima, let minima else { preconditionFailure("kernelAttention runs only once blocks are indexed") }
        let buffers = base.innerState()
        let selected = QuestKernel.selectedBlocks(
            queries, upper: maxima, lower: minima, first: middle.lowerBound, last: middle.upperBound, topK: topK)
        return QuestKernel.attention(
            queries: queries, keys: buffers[0], values: buffers[1], selected: selected, offset: offset, scale: scale)
    }

    private var middle: Range<Int> {
        let first = Self.sinkTokens / Self.blockTokens
        return first ..< max(first, (offset - Self.windowTokens) / Self.blockTokens)
    }

    static func topBlocks(_ queries: MLXArray, upper: MLXArray, lower: MLXArray, middle: Range<Int>, topK: Int) -> MLXArray {
        let (batch, kvHeads) = (upper.dim(0), upper.dim(1))
        let q = queries.reshaped(batch, kvHeads, queries.dim(1) / kvHeads, upper.dim(3))
        let score = blockScores(q, upper: upper[0..., 0..., middle], lower: lower[0..., 0..., middle])
        return argPartition(-score, kth: topK - 1, axis: -1)[.ellipsis, ..<topK].asType(.int32) + Int32(middle.lowerBound)
    }

    func gatheredAttention(queries: MLXArray, scale: Float) -> MLXArray {
        guard let maxima, let minima else { preconditionFailure("gatheredAttention runs only once blocks are indexed") }
        let buffers = base.innerState()
        let (batch, kvHeads) = (buffers[0].dim(0), buffers[0].dim(1))
        let chosen = Self.topBlocks(queries, upper: maxima, lower: minima, middle: middle, topK: topK)
        let blocks = (chosen.expandedDimensions(axis: -1) * Int32(Self.blockTokens) + blockSteps).reshaped(batch, kvHeads, -1)
        let windowStart = middle.upperBound * Self.blockTokens
        let fixed = concatenated([sinks, windowSteps[..<(offset - windowStart)] + Int32(windowStart)])
        let rows = concatenated([broadcast(fixed.reshaped(1, 1, -1), to: [batch, kvHeads, fixed.dim(0)]), blocks], axis: -1)
            .expandedDimensions(axis: -1)
        let count = rows.dim(2)
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: takeAlong(buffers[0], broadcast(rows, to: [batch, kvHeads, count, buffers[0].dim(3)]), axis: 2),
            values: takeAlong(buffers[1], broadcast(rows, to: [batch, kvHeads, count, buffers[1].dim(3)]), axis: 2),
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
        let copied = QuestKVCache(topK: topK, usesKernel: usesKernel)
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
