import Foundation
import MLX
import MLXLMCommon

public enum XLSTMLayerKind: String, Sendable, Equatable {
    case mlstm
    case slstm
}

public final class XLSTMLayerCache: KVCache {
    public let kind: XLSTMLayerKind
    public var cState: MLXArray?
    public var nState: MLXArray?
    public var mState: MLXArray?
    public var hState: MLXArray?
    public private(set) var offset: Int = 0

    public init(kind: XLSTMLayerKind = .mlstm) {
        self.kind = kind
    }

    public var maxSize: Int? { nil }
    public var isTrimmable: Bool { false }

    public func isTrimmable(after positions: Int) -> Bool { false }

    public func innerState() -> [MLXArray] {
        switch kind {
        case .mlstm:
            [cState, nState, mState].compactMap { $0 }
        case .slstm:
            [cState, nState, mState, hState].compactMap { $0 }
        }
    }

    public var state: [MLXArray] {
        get { innerState() }
        set {
            switch newValue.count {
            case 0:
                (cState, nState, mState, hState) = (nil, nil, nil, nil)
            case 3:
                (cState, nState, mState, hState) = (newValue[0], newValue[1], newValue[2], nil)
            case 4:
                (cState, nState, mState, hState) = (newValue[0], newValue[1], newValue[2], kind == .slstm ? newValue[3] : nil)
            default:
                cState = newValue.indices.contains(0) ? newValue[0] : nil
                nState = newValue.indices.contains(1) ? newValue[1] : nil
                mState = newValue.indices.contains(2) ? newValue[2] : nil
                hState = kind == .slstm && newValue.indices.contains(3) ? newValue[3] : nil
            }
        }
    }

    public var metaState: [String] {
        get { [String(offset), kind.rawValue] }
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
        cState = nil
        nState = nil
        mState = nil
        hState = nil
        offset = 0
    }

    public func copy() -> any KVCache {
        let copied = XLSTMLayerCache(kind: kind)
        copied.cState = cState.map(Self.detach)
        copied.nState = nState.map(Self.detach)
        copied.mState = mState.map(Self.detach)
        copied.hState = hState.map(Self.detach)
        copied.offset = offset
        return copied
    }

    private static func detach(_ tensor: MLXArray) -> MLXArray {
        tensor[.ellipsis] + MLXArray(Float(0)).asType(tensor.dtype)
    }
}
