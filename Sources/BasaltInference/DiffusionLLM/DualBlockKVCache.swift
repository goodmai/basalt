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

public final class DualBlockLayerKVCache: KVCache {
    public private(set) var offset: Int = 0
    public private(set) var prefixKeys: MLXArray?
    public private(set) var prefixValues: MLXArray?
    public private(set) var suffixKeys: MLXArray?
    public private(set) var suffixValues: MLXArray?
    public var phase: DualBlockCachePhase = .promptPrefill
    public var customMask: MLXArray?
    public let bidirectionalPrompt: Bool

    public init(bidirectionalPrompt: Bool = false) {
        self.bidirectionalPrompt = bidirectionalPrompt
    }

    public var maxSize: Int? { nil }
    public var isTrimmable: Bool { true }
    public var prefixLength: Int { prefixKeys?.dim(2) ?? 0 }
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
                prefixKeys = nil
                prefixValues = nil
                offset = 0
            } else if newValue.count >= 2 {
                prefixKeys = newValue[0]
                prefixValues = newValue[1]
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
        guard n > 0, let curK = prefixKeys, let curV = prefixValues else { return 0 }
        let current = curK.dim(2)
        let removed = min(current, n)
        let remaining = current - removed
        if remaining == 0 {
            prefixKeys = nil
            prefixValues = nil
        } else {
            prefixKeys = curK[.ellipsis, ..<remaining, 0...]
            prefixValues = curV[.ellipsis, ..<remaining, 0...]
        }
        offset = remaining
        return removed
    }

    public func copy() -> any KVCache {
        let clone = DualBlockLayerKVCache()
        clone.prefixKeys = prefixKeys
        clone.prefixValues = prefixValues
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

    public func reset() {
        prefixKeys = nil
        prefixValues = nil
        suffixKeys = nil
        suffixValues = nil
        customMask = nil
        phase = .promptPrefill
        offset = 0
    }

    private func appendToPrefix(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let nextKeys: MLXArray
        let nextValues: MLXArray
        if let curK = prefixKeys, let curV = prefixValues {
            nextKeys = concatenated([curK, keys], axis: 2)
            nextValues = concatenated([curV, values], axis: 2)
        } else {
            nextKeys = keys
            nextValues = values
        }
        prefixKeys = nextKeys
        prefixValues = nextValues
        offset = nextKeys.dim(2)
        return (nextKeys, nextValues)
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

    public init(layerCount: Int, bidirectionalPrompt: Bool = false) {
        self.layers = (0 ..< max(0, layerCount)).map { _ in DualBlockLayerKVCache(bidirectionalPrompt: bidirectionalPrompt) }
    }

    public var kvCaches: [any KVCache] { layers }
    public var prefixLength: Int { layers.first?.prefixLength ?? 0 }
    public var suffixLength: Int { layers.first?.suffixLength ?? 0 }
    public var hasSuffixCache: Bool { suffixLength > 0 }

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
