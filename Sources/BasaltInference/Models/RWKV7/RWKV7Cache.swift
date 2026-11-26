import Foundation
import MLX
import MLXLMCommon

public final class RWKV7LayerCache: KVCache {
    public var stateMatrix: MLXArray?
    public var prevTimeX: MLXArray?
    public var prevChannelX: MLXArray?
    public private(set) var offset: Int = 0

    public init() {}

    public var maxSize: Int? { nil }
    public var isTrimmable: Bool { false }

    public func isTrimmable(after positions: Int) -> Bool { false }

    public func innerState() -> [MLXArray] {
        [stateMatrix, prevTimeX, prevChannelX].compactMap { $0 }
    }

    public var state: [MLXArray] {
        get { innerState() }
        set {
            switch newValue.count {
            case 0:
                (stateMatrix, prevTimeX, prevChannelX) = (nil, nil, nil)
            case 3:
                (stateMatrix, prevTimeX, prevChannelX) = (newValue[0], newValue[1], newValue[2])
            default:
                stateMatrix = newValue.indices.contains(0) ? newValue[0] : nil
                prevTimeX = newValue.indices.contains(1) ? newValue[1] : nil
                prevChannelX = newValue.indices.contains(2) ? newValue[2] : nil
            }
        }
    }

    public var metaState: [String] {
        get { [String(offset)] }
        set {
            guard let first = newValue.first, let stored = Int(first) else { return }
            offset = max(0, stored)
        }
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }

    @discardableResult
    public func trim(_ n: Int) -> Int { 0 }

    public func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    public func advance(by tokens: Int) {
        offset += max(0, tokens)
    }

    public func reset() {
        stateMatrix = nil
        prevTimeX = nil
        prevChannelX = nil
        offset = 0
    }

    public func copy() -> any KVCache {
        let copied = RWKV7LayerCache()
        copied.stateMatrix = stateMatrix.map { $0[.ellipsis] }
        copied.prevTimeX = prevTimeX.map { $0[.ellipsis] }
        copied.prevChannelX = prevChannelX.map { $0[.ellipsis] }
        copied.offset = offset
        return copied
    }
}
