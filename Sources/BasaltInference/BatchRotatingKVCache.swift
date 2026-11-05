import MLX
import MLXLMCommon

final class BatchRotatingKVCache: KVCache {
    private var keys: MLXArray?
    private var values: MLXArray?
    private(set) var leftPadding: [Int] = []
    private(set) var positions: [Int] = []
    private(set) var idx = 0
    private(set) var wrapped = false
    let window: Int
    let step = 256

    init(window: Int) {
        self.window = window
    }

    static func admits(_ layer: KVCache) -> Bool {
        type(of: layer) == RotatingKVCache.self && layer.metaState.first == "0"
    }

    var offset: Int { positions.max() ?? 0 }
    var maxSize: Int? { window }
    var isTrimmable: Bool { false }

    var ropeOffset: RoPEOffset {
        if let first = positions.first, positions.allSatisfy({ $0 == first }) { return .scalar(first) }
        return .batch(MLXArray(positions.map(Int32.init)))
    }

    private var live: Int { wrapped ? keys?.dim(2) ?? 0 : idx }

    func innerState() -> [MLXArray] { [keys, values].compactMap { $0 } }

    var state: [MLXArray] {
        get { innerState() }
        set { fatalError("BatchRotatingKVCache is built by append(_:), not restored from state") }
    }

    var metaState: [String] {
        get { [] }
        set { fatalError("BatchRotatingKVCache is built by append(_:), not restored from state") }
    }

    func trim(_ n: Int) -> Int { 0 }

    func copy() -> any KVCache { fatalError("BatchRotatingKVCache is owned by one BatchedDecoder and never copied") }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let wraps = wrapped || idx == window
        let padding = wraps ? leftPadding.map { max($0 - 1, 0) } : leftPadding
        guard returnArray || padding.contains(where: { $0 > 0 }) else { return .none }
        let length = wraps ? window : idx + 1
        let oldest = wraps ? ((idx == window ? 0 : idx) + 1) % window : 0
        let age = MLXArray((0 ..< length).map { Int32(($0 - oldest + length) % length) })
        return .array(age .>= MLXArray(padding.map(Int32.init)).reshaped(-1, 1, 1, 1))
    }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        precondition(newKeys.dim(2) == 1, "BatchRotatingKVCache decodes one token per row per step")
        guard var keys, var values else { fatalError("BatchRotatingKVCache updated before any row joined") }
        if !wrapped, idx == keys.dim(2), idx < window {
            let room = min(step, window - idx)
            keys = Self.padded(keys, left: 0, right: room)
            values = Self.padded(values, left: 0, right: room)
        }
        if idx == window {
            idx = 0
            wrapped = true
        }
        if wrapped { leftPadding = leftPadding.map { max($0 - 1, 0) } }
        keys[.ellipsis, idx ..< idx + 1, 0...] = newKeys
        values[.ellipsis, idx ..< idx + 1, 0...] = newValues
        idx += 1
        positions = positions.map { $0 + 1 }
        self.keys = keys
        self.values = values
        guard !wrapped, idx < keys.dim(2) else { return (keys, values) }
        return (keys[.ellipsis, ..<idx, 0...], values[.ellipsis, ..<idx, 0...])
    }

    func append(_ joining: [RotatingKVCache]) {
        var rows: [[MLXArray]] = []
        if let keys, let values { rows.append(Self.temporal([keys, values], idx: idx, wrapped: wrapped)) }
        for row in joining {
            let meta = row.metaState
            let ordered = Self.temporal(row.state, idx: Int(meta[4]) ?? 0, wrapped: meta.count > 6 && meta[6] == "true")
            let kept = min(ordered[0].dim(2), window)
            rows.append(ordered.map { $0[.ellipsis, ($0.dim(2) - kept)..., 0...] })
            positions.append(row.offset)
        }
        let lengths = rows.map { $0[0].dim(2) }
        let width = lengths.max() ?? 0
        if keys != nil { leftPadding = leftPadding.map { $0 + width - lengths[0] } }
        leftPadding += lengths.suffix(joining.count).map { width - $0 }
        keys = concatenated(zip(rows, lengths).map { Self.padded($0[0], left: width - $1, right: 0) }, axis: 0)
        values = concatenated(zip(rows, lengths).map { Self.padded($0[1], left: width - $1, right: 0) }, axis: 0)
        idx = width
        wrapped = false
    }

    func filter(rows kept: [Int]) {
        guard let keys, let values, !kept.isEmpty else {
            self.keys = nil
            self.values = nil
            leftPadding = []
            positions = []
            idx = 0
            wrapped = false
            return
        }
        let index = MLXArray(kept.map(Int32.init))
        leftPadding = kept.map { leftPadding[$0] }
        positions = kept.map { positions[$0] }
        let unused = leftPadding.min() ?? 0
        guard unused > 0 else {
            self.keys = keys[index]
            self.values = values[index]
            return
        }
        let ordered = Self.temporal([keys[index], values[index]], idx: idx, wrapped: wrapped)
        idx = live - unused
        self.keys = ordered[0][.ellipsis, unused..., 0...]
        self.values = ordered[1][.ellipsis, unused..., 0...]
        wrapped = false
        leftPadding = leftPadding.map { $0 - unused }
    }

    private static func temporal(_ arrays: [MLXArray], idx: Int, wrapped: Bool) -> [MLXArray] {
        guard wrapped else { return arrays.map { $0[.ellipsis, ..<idx, 0...] } }
        guard idx < arrays[0].dim(2) else { return arrays }
        return arrays.map { concatenated([$0[.ellipsis, idx..., 0...], $0[.ellipsis, ..<idx, 0...]], axis: 2) }
    }

    private static func padded(_ array: MLXArray, left: Int, right: Int) -> MLXArray {
        guard left > 0 || right > 0 else { return array }
        return MLX.padded(array, widths: [0, 0, IntOrPair((left, right)), 0])
    }
}
