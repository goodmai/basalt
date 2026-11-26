import Foundation
import MLX
import MLXLMCommon

public enum DualBlockCachePhase: Equatable, Sendable {
    case promptPrefill
    case suffixPrefill(offset: Int)
    case activeBlock(offset: Int)
    case blockCommit(offset: Int)
    case uncachedFullSequence
}

public final class DualBlockLayerKVCache: KVCacheAttentionProtocol {
    static let allocationStep = 256

    public private(set) var offset: Int = 0
    private var prefixKeyBuffer: MLXArray?
    private var prefixValueBuffer: MLXArray?
    public private(set) var suffixKeys: MLXArray?
    public private(set) var suffixValues: MLXArray?
    public var phase: DualBlockCachePhase = .promptPrefill
    public var customMask: MLXArray?
    public let bidirectionalPrompt: Bool
    public let kernelMode: DualBlockFlashKernelMode
    package private(set) var kernelInvocations = 0
    package private(set) var fallbackInvocations = 0

    public init(
        bidirectionalPrompt: Bool = false,
        kernelMode: DualBlockFlashKernelMode? = nil
    ) {
        self.bidirectionalPrompt = bidirectionalPrompt
        self.kernelMode = kernelMode ?? DualBlockFlashAttentionKernel.environmentDefaultMode
    }

    public var prefixKeys: MLXArray? {
        guard offset > 0, let buffer = prefixKeyBuffer else { return nil }
        return buffer[.ellipsis, ..<offset, 0...]
    }

    public var prefixValues: MLXArray? {
        guard offset > 0, let buffer = prefixValueBuffer else { return nil }
        return buffer[.ellipsis, ..<offset, 0...]
    }

    public var requiresAttentionThroughCache: Bool { true }
    public var maxSize: Int? { nil }
    public var isTrimmable: Bool { true }
    public var prefixLength: Int { offset }
    public var suffixLength: Int { suffixKeys?.dim(2) ?? 0 }

    public var ropeOffset: RoPEOffset {
        switch phase {
        case .promptPrefill:
            .scalar(prefixLength)
        case .suffixPrefill(let pos), .activeBlock(let pos), .blockCommit(let pos):
            .scalar(pos)
        case .uncachedFullSequence:
            .scalar(0)
        }
    }

    public func innerState() -> [MLXArray] {
        [prefixKeys, prefixValues, suffixKeys, suffixValues].compactMap { $0 }
    }

    public var state: [MLXArray] {
        get {
            guard let preK = prefixKeys, let preV = prefixValues else { return [] }
            return [preK, preV]
        }
        set {
            if newValue.isEmpty {
                prefixKeyBuffer = nil
                prefixValueBuffer = nil
                offset = 0
            } else if newValue.count >= 2 {
                prefixKeyBuffer = newValue[0]
                prefixValueBuffer = newValue[1]
                offset = newValue[0].dim(2)
            }
        }
    }

    public var metaState: [String] {
        get { [String(offset), String(suffixLength)] }
        set { offset = newValue.first.flatMap { Int($0) } ?? 0 }
    }

    public func makeMask(
        n: Int,
        windowSize _: Int?,
        returnArray _: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if let customMask {
            return .array(customMask)
        }
        switch phase {
        case .promptPrefill:
            return bidirectionalPrompt || n <= 1 ? .none : .causal
        case .suffixPrefill, .activeBlock, .blockCommit, .uncachedFullSequence:
            return .none
        }
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        switch phase {
        case .promptPrefill, .blockCommit:
            return appendToPrefix(keys: keys, values: values)
        case .suffixPrefill:
            suffixKeys = keys
            suffixValues = values
            if let preK = prefixKeys, let preV = prefixValues {
                return (concatenated([preK, keys], axis: 2), concatenated([preV, values], axis: 2))
            }
            return (keys, values)
        case .activeBlock:
            return mergeActive(keys: keys, values: values)
        case .uncachedFullSequence:
            return (keys, values)
        }
    }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        guard n > 0, offset > 0 else { return 0 }
        let removed = min(offset, n)
        offset -= removed
        if offset == 0 {
            prefixKeyBuffer = nil
            prefixValueBuffer = nil
        }
        return removed
    }

    public func copy() -> any KVCache {
        let clone = DualBlockLayerKVCache(bidirectionalPrompt: bidirectionalPrompt, kernelMode: kernelMode)
        clone.prefixKeyBuffer = prefixKeys
        clone.prefixValueBuffer = prefixValues
        clone.suffixKeys = suffixKeys
        clone.suffixValues = suffixValues
        clone.phase = phase
        clone.customMask = customMask
        clone.offset = offset
        return clone
    }

    public func clearSuffix() {
        suffixKeys = nil
        suffixValues = nil
    }

    public func advanceSuffix(by count: Int) {
        guard count > 0, let curK = suffixKeys, let curV = suffixValues else { return }
        if count >= curK.dim(2) {
            clearSuffix()
        } else {
            suffixKeys = curK[.ellipsis, count..., 0...]
            suffixValues = curV[.ellipsis, count..., 0...]
        }
    }

    public func reset() {
        prefixKeyBuffer = nil
        prefixValueBuffer = nil
        suffixKeys = nil
        suffixValues = nil
        customMask = nil
        phase = .promptPrefill
        offset = 0
    }
}

extension DualBlockLayerKVCache {
    public func updateAndAttend(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        switch phase {
        case .activeBlock:
            let segments = DualBlockFlashAttentionKernel.Segments(
                prefixKeys: prefixKeys, prefixValues: prefixValues,
                activeKeys: keys, activeValues: values,
                suffixKeys: suffixKeys, suffixValues: suffixValues
            )
            if customMask == nil,
               DualBlockFlashAttentionKernel.isEligible(queries: queries, segments: segments, mask: mask, mode: kernelMode)
            {
                kernelInvocations += 1
                return DualBlockFlashAttentionKernel.attention(queries: queries, segments: segments, scale: scale)
            }
        case .suffixPrefill:
            suffixKeys = keys
            suffixValues = values
            let segments = DualBlockFlashAttentionKernel.Segments(
                prefixKeys: prefixKeys, prefixValues: prefixValues,
                activeKeys: keys, activeValues: values,
                suffixKeys: nil, suffixValues: nil
            )
            if customMask == nil,
               DualBlockFlashAttentionKernel.isEligible(queries: queries, segments: segments, mask: mask, mode: kernelMode)
            {
                kernelInvocations += 1
                return DualBlockFlashAttentionKernel.attention(queries: queries, segments: segments, scale: scale)
            }
            fallbackInvocations += 1
            let mergedK = prefixKeys.map { concatenated([$0, keys], axis: 2) } ?? keys
            let mergedV = prefixValues.map { concatenated([$0, values], axis: 2) } ?? values
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: mergedK, values: mergedV, scale: scale, mask: mask)
        case .promptPrefill, .blockCommit, .uncachedFullSequence:
            break
        }
        fallbackInvocations += 1
        let (allK, allV) = update(keys: keys, values: values)
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: allK, values: allV, scale: scale, mask: mask)
    }

    private func appendToPrefix(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let added = keys.dim(2)
        guard added > 0 else {
            return (prefixKeys ?? keys, prefixValues ?? values)
        }
        let prev = offset
        let next = prev + added
        let step = Self.allocationStep
        if prefixKeyBuffer == nil || next > (prefixKeyBuffer?.dim(2) ?? 0) {
            let (b, kvHeads, kDim, vDim) = (keys.dim(0), keys.dim(1), keys.dim(3), values.dim(3))
            let grow = ((next + step - 1) / step) * step
            let newK = MLXArray.zeros([b, kvHeads, grow, kDim], dtype: keys.dtype)
            let newV = MLXArray.zeros([b, kvHeads, grow, vDim], dtype: values.dtype)
            if prev > 0, let curK = prefixKeyBuffer, let curV = prefixValueBuffer {
                newK[.ellipsis, ..<prev, 0...] = curK[.ellipsis, ..<prev, 0...]
                newV[.ellipsis, ..<prev, 0...] = curV[.ellipsis, ..<prev, 0...]
            }
            prefixKeyBuffer = newK
            prefixValueBuffer = newV
        }
        prefixKeyBuffer?[.ellipsis, prev ..< next, 0...] = keys
        prefixValueBuffer?[.ellipsis, prev ..< next, 0...] = values
        offset = next
        return (prefixKeys ?? keys, prefixValues ?? values)
    }

    private func mergeActive(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let keyParts = [prefixKeys, keys, suffixKeys].compactMap { $0 }
        let valueParts = [prefixValues, values, suffixValues].compactMap { $0 }
        let mergedKeys = keyParts.count == 1 ? keyParts[0] : concatenated(keyParts, axis: 2)
        let mergedValues = valueParts.count == 1 ? valueParts[0] : concatenated(valueParts, axis: 2)
        return (mergedKeys, mergedValues)
    }
}

public struct DualBlockKVCache {
    public let layers: [DualBlockLayerKVCache]

    public init(
        layerCount: Int,
        bidirectionalPrompt: Bool = false,
        kernelMode: DualBlockFlashKernelMode? = nil
    ) {
        let resolvedMode = kernelMode ?? DualBlockFlashAttentionKernel.environmentDefaultMode
        self.layers = (0 ..< max(0, layerCount)).map { _ in
            DualBlockLayerKVCache(bidirectionalPrompt: bidirectionalPrompt, kernelMode: resolvedMode)
        }
    }

    public var kvCaches: [any KVCache] { layers }
    public var prefixLength: Int { layers.first?.prefixLength ?? 0 }
    public var suffixLength: Int { layers.first?.suffixLength ?? 0 }
    public var hasSuffixCache: Bool { suffixLength > 0 }
    package var totalKernelInvocations: Int { layers.reduce(0) { $0 + $1.kernelInvocations } }

    public func preparePromptPrefill() {
        for layer in layers {
            layer.customMask = nil
            layer.phase = .promptPrefill
        }
    }

    public func prepareSuffixPrefill(at offset: Int) {
        for layer in layers {
            layer.customMask = nil
            layer.phase = .suffixPrefill(offset: offset)
        }
    }

    public func prepareActiveBlock(at offset: Int) {
        for layer in layers {
            layer.customMask = nil
            layer.phase = .activeBlock(offset: offset)
        }
    }

    public func prepareBlockCommit(at offset: Int) {
        for layer in layers {
            layer.customMask = nil
            layer.phase = .blockCommit(offset: offset)
        }
    }

    public func clearSuffix() {
        for layer in layers { layer.clearSuffix() }
    }

    public func advanceSuffix(by count: Int) {
        for layer in layers { layer.advanceSuffix(by: count) }
    }

    public func reset() {
        for layer in layers { layer.reset() }
    }

    public static func makeBlockDiffusionMask(
        promptLength: Int,
        finalizedBlockEnds: [Int],
        activeRange: Range<Int>,
        totalLength: Int
    ) -> MLXArray {
        guard totalLength > 0 else { return MLXArray.zeros([0, 0], type: Bool.self) }
        var rowPrefixLimit = [Int32](repeating: 0, count: totalLength)
        let clampedPrompt = min(max(0, promptLength), totalLength)
        for row in 0 ..< clampedPrompt {
            rowPrefixLimit[row] = Int32(row + 1)
        }
        var blockCursor = clampedPrompt
        for end in finalizedBlockEnds {
            let clampedEnd = min(max(blockCursor, end), activeRange.lowerBound)
            for row in blockCursor ..< clampedEnd {
                rowPrefixLimit[row] = Int32(clampedEnd)
            }
            blockCursor = clampedEnd
        }
        let activeLower = min(max(0, activeRange.lowerBound), totalLength)
        let activeUpper = min(max(activeLower, activeRange.upperBound), totalLength)
        for row in activeLower ..< activeUpper {
            rowPrefixLimit[row] = Int32(totalLength)
        }
        for row in activeUpper ..< totalLength {
            rowPrefixLimit[row] = Int32(activeLower)
        }
        let seesSuffix = (0 ..< totalLength).map { $0 >= activeLower }
        let cols = MLXArray(Int32(0) ..< Int32(totalLength))[.newAxis, 0...]
        let limits = MLXArray(rowPrefixLimit)[0..., .newAxis]
        let suffixFlag = MLXArray(seesSuffix)[0..., .newAxis]
        return (cols .< limits) .|| (suffixFlag .&& (cols .>= Int32(activeUpper)))
    }

    public static func uncachedMaskCaches(layerCount: Int, mask: MLXArray) -> [any KVCache] {
        (0 ..< max(0, layerCount)).map { _ in
            let cache = DualBlockLayerKVCache()
            cache.phase = .uncachedFullSequence
            cache.customMask = mask
            return cache
        }
    }
}
