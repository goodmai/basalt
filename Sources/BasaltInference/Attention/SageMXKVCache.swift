import BasaltCore
import Foundation
import MLX
import MLXLMCommon

final class SageMXKVCache: KVCacheAttentionProtocol {
    enum Mode: Equatable, Sendable {
        case sagemx
        case moba(topBlocks: Int)
    }

    static let blockTokens = 64
    static let minTokensForKernel = 64

    let mode: Mode
    private let base = KVCacheSimple()
    private var centroids: MLXArray?
    private var indexedBlocks = 0
    private(set) var kernelSteps = 0
    private(set) var materializedForCaller = false

    init(mode: Mode) {
        self.mode = mode
    }

    static func replacingFullAttention(_ caches: [any KVCache], mode: Mode) -> [any KVCache] {
        caches.map { type(of: $0) == KVCacheSimple.self ? SageMXKVCache(mode: mode) : $0 }
    }

    static func modelAttendsThroughCache(_ model: any LanguageModel) throws -> Bool {
        let caches = replacingFullAttention(try model.newCache(parameters: nil), mode: .sagemx)
        let probes = caches.compactMap { $0 as? SageMXKVCache }
        guard !probes.isEmpty else { return false }
        _ = model(LMInput.Text(tokens: MLXArray([Int32(0)]).reshaped(1, 1)), cache: caches, state: nil)
        return probes.allSatisfy { !$0.materializedForCaller }
    }

    var offset: Int { base.offset }
    var maxSize: Int? { nil }
    var isTrimmable: Bool { true }

    func innerState() -> [MLXArray] { base.innerState() + [centroids].compactMap { $0 } }

    var state: [MLXArray] {
        get { base.state }
        set { preconditionFailure("SageMXKVCache is never restored from a serialized prompt cache") }
    }

    var metaState: [String] {
        get {
            switch mode {
            case .sagemx: ["sagemx"]
            case .moba(let k): ["moba", String(k)]
            }
        }
        set { preconditionFailure("SageMXKVCache is never restored from a serialized prompt cache") }
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
            centroids = kept == 0 ? nil : centroids?[0..., 0..., ..<kept]
            indexedBlocks = kept
        }
        return removed
    }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        base.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    func copy() -> any KVCache {
        let copied = SageMXKVCache(mode: mode)
        if !base.state.isEmpty { copied.base.state = base.state }
        copied.centroids = centroids
        copied.indexedBlocks = indexedBlocks
        copied.kernelSteps = kernelSteps
        copied.materializedForCaller = materializedForCaller
        return copied
    }

    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let (k, v) = base.update(keys: keys, values: values)
        if case .moba = mode { indexCentroids(k) }
        guard Self.supports(queries: queries, keys: k, mask: mask) else {
            return MLXFast.scaledDotProductAttention(queries: queries, keys: k, values: v, scale: scale, mask: mask)
        }
        switch mode {
        case .sagemx:
            guard k.dim(2) >= Self.minTokensForKernel else {
                return MLXFast.scaledDotProductAttention(queries: queries, keys: k, values: v, scale: scale, mask: mask)
            }
            kernelSteps += 1
            let isCausal = Self.isCausal(mask) && queries.dim(2) > 1
            let (kExp, vExp) = Self.expandGQA(k: k, v: v, qHeads: queries.dim(1))
            return SageMXAttentionMetal.attention(
                q: queries, k: kExp, v: vExp,
                options: SageMXAttentionMetal.Options(scale: scale, causal: isCausal))
        case .moba(let topBlocks):
            guard queries.dim(2) == 1, let centroids, indexedBlocks > topBlocks + 2 else {
                return MLXFast.scaledDotProductAttention(queries: queries, keys: k, values: v, scale: scale, mask: mask)
            }
            kernelSteps += 1
            return mobaAttention(queries: queries, keys: k, values: v, centroids: centroids, topBlocks: topBlocks, scale: scale)
        }
    }

    private func mobaAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, centroids: MLXArray, topBlocks: Int, scale: Float
    ) -> MLXArray {
        let qHeads = queries.dim(1)
        let kvHeads = keys.dim(1)
        let group = qHeads / kvHeads
        let lastFull = indexedBlocks - 1
        let middleCount = max(0, lastFull - 1)
        let kSel = min(topBlocks, middleCount)
        let middleCentroids = centroids[0..., 0..., 1 ..< lastFull, 0...]
        let qGrouped = queries.asType(.float32).reshaped(1, kvHeads, group, queries.dim(3))
        let scores = matmul(qGrouped, middleCentroids.swappedAxes(-1, -2)).reshaped(qHeads, 1, middleCount)
        let picked = argPartition(-scores, kth: kSel - 1, axis: -1)[0..., 0..., ..<kSel].asType(.int32) + Int32(1)
        let hasTail = offset > indexedBlocks * Self.blockTokens
        var fixedIds: [Int32] = [0, Int32(lastFull)]
        if hasTail { fixedIds.append(Int32(indexedBlocks)) }
        let fixed = broadcast(MLXArray(fixedIds).reshaped(1, 1, fixedIds.count), to: [qHeads, 1, fixedIds.count])
        let selected = concatenated([fixed, picked], axis: -1)
        let (kExp, vExp) = Self.expandGQA(k: keys, v: values, qHeads: qHeads)
        return BlockSparseRoutingMetal.sparseSelectedAttention(
            q: queries, k: kExp, v: vExp, selectedBlocks: selected, blockSize: Self.blockTokens, scale: scale)
    }

    private func indexCentroids(_ keys: MLXArray) {
        let complete = offset / Self.blockTokens
        guard complete > indexedBlocks else { return }
        let slice = keys[.ellipsis, (indexedBlocks * Self.blockTokens) ..< (complete * Self.blockTokens), 0...]
            .asType(.float32)
            .reshaped(keys.dim(0), keys.dim(1), complete - indexedBlocks, Self.blockTokens, keys.dim(3))
        let meanBlock = mean(slice, axis: 3)
        centroids = centroids.map { concatenated([$0, meanBlock], axis: 2) } ?? meanBlock
        indexedBlocks = complete
    }

    private static func supports(
        queries: MLXArray, keys: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> Bool {
        if case .array = mask { return false }
        let d = queries.dim(-1)
        return queries.dim(0) == 1 && d.isMultiple(of: 32) && d <= SageMXAttentionMetal.maxHeadDim
            && SageMXAttentionMetal.supportedDTypes.contains(queries.dtype)
            && queries.dim(1).isMultiple(of: max(1, keys.dim(1)))
    }

    private static func isCausal(_ mask: MLXFast.ScaledDotProductAttentionMaskMode) -> Bool {
        if case .causal = mask { return true }
        return false
    }

    private static func expandGQA(k: MLXArray, v: MLXArray, qHeads: Int) -> (MLXArray, MLXArray) {
        let kvHeads = k.dim(1)
        guard kvHeads < qHeads else { return (k, v) }
        let group = qHeads / kvHeads
        let shape = [k.dim(0), kvHeads, group, k.dim(2), k.dim(3)]
        let kExp = broadcast(k.expandedDimensions(axis: 2), to: shape).reshaped(k.dim(0), qHeads, k.dim(2), k.dim(3))
        let vExp = broadcast(v.expandedDimensions(axis: 2), to: shape).reshaped(v.dim(0), qHeads, v.dim(2), v.dim(3))
        return (kExp, vExp)
    }
}
