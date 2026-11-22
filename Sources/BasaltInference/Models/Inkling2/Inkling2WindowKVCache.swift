import Foundation
import MLX
import MLXLMCommon

final class Inkling2WindowKVCache: KVCache {
    let window: Int
    private var keys: MLXArray?
    private var values: MLXArray?
    private(set) var offset = 0
    private(set) var firstPosition = 0

    init(window: Int) {
        self.window = window
    }

    var maxSize: Int? { window }
    var isTrimmable: Bool { true }

    func innerState() -> [MLXArray] { [keys, values].compactMap { $0 } }

    var state: [MLXArray] {
        get { [keys, values].compactMap { $0 } }
        set {
            guard newValue.count == 2 else { (keys, values) = (nil, nil); return }
            (keys, values) = (newValue[0], newValue[1])
            firstPosition = offset - newValue[0].dim(2)
        }
    }

    var metaState: [String] {
        get { [String(offset), String(firstPosition)] }
        set {
            guard newValue.count == 2, let stored = Int(newValue[0]), let first = Int(newValue[1]) else { return }
            (offset, firstPosition) = (stored, first)
        }
    }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let length = newKeys.dim(2)
        if let keys, let values {
            self.keys = concatenated([keys, newKeys], axis: 2)
            self.values = concatenated([values, newValues], axis: 2)
        } else {
            (keys, values) = (newKeys, newValues)
        }
        offset += length
        let held = self.keys!.dim(2)
        let needed = window - 1 + length
        let start = max(0, held - needed)
        let visibleKeys = self.keys![0..., 0..., start..., 0...]
        let visibleValues = self.values![0..., 0..., start..., 0...]
        let keyStart = firstPosition + start
        if held > 2 * window + length {
            let keep = max(0, held - (window - 1))
            self.keys = self.keys![0..., 0..., keep..., 0...] + 0
            self.values = self.values![0..., 0..., keep..., 0...] + 0
            firstPosition += keep
        }
        lastKeyStart = keyStart
        return (visibleKeys, visibleValues)
    }

    private(set) var lastKeyStart = 0

    func trim(_ n: Int) -> Int {
        guard let keys, let values else { return 0 }
        let removable = min(n, keys.dim(2))
        self.keys = keys[0..., 0..., ..<(keys.dim(2) - removable), 0...]
        self.values = values[0..., 0..., ..<(values.dim(2) - removable), 0...]
        offset -= removable
        return removable
    }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        n > 1 ? .causal : .none
    }

    func copy() -> any KVCache {
        let copied = Inkling2WindowKVCache(window: window)
        (copied.keys, copied.values, copied.offset, copied.firstPosition) = (keys, values, offset, firstPosition)
        return copied
    }
}
