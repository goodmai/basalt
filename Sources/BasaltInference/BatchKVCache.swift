import MLX
import MLXLMCommon

final class BatchKVCache: KVCache {
    private var keys: MLXArray?
    private var values: MLXArray?
    private(set) var leftPadding: [Int] = []
    private(set) var offset = 0
    let step = 256
    var maxSize: Int? { nil }
    var isTrimmable: Bool { false }

    var ropeOffset: RoPEOffset {
        if let first = leftPadding.first, leftPadding.allSatisfy({ $0 == first }) { return .scalar(offset - first) }
        return .batch(MLXArray(leftPadding.map { Int32(offset - $0) }))
    }

    func innerState() -> [MLXArray] { [keys, values].compactMap { $0 } }

    var state: [MLXArray] {
        get { innerState() }
        set { fatalError("BatchKVCache is built by append(_:), not restored from state") }
    }

    var metaState: [String] {
        get { [] }
        set { fatalError("BatchKVCache is built by append(_:), not restored from state") }
    }

    func trim(_ n: Int) -> Int { 0 }

    func copy() -> any KVCache { fatalError("BatchKVCache is owned by one BatchedDecoder and never copied") }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1, !returnArray, windowSize == nil, leftPadding.allSatisfy({ $0 == 0 }) { return .none }
        let causal = createCausalMask(n: n, offset: offset, windowSize: windowSize)
        let columns = MLXArray(Int32(0) ..< Int32(offset + n))
        let padding = MLXArray(leftPadding.map(Int32.init)).reshaped(-1, 1, 1, 1)
        return .array(causal & (columns .>= padding))
    }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let previous = offset
        let added = newKeys.dim(2)
        if previous + added > keys?.dim(2) ?? 0 {
            let capacity = (previous + added + step - 1) / step * step
            keys = Self.widened(keys, used: previous, to: capacity, like: newKeys)
            values = Self.widened(values, used: previous, to: capacity, like: newValues)
        }
        offset += added
        keys?[.ellipsis, previous ..< offset, 0...] = newKeys
        values?[.ellipsis, previous ..< offset, 0...] = newValues
        return (keys![.ellipsis, ..<offset, 0...], values![.ellipsis, ..<offset, 0...])
    }

    func append(_ joining: [KVCacheSimple]) {
        let incoming = joining.map(\.state)
        let width = max(offset, incoming.map { $0[0].dim(2) }.max() ?? 0)
        var keyRows: [MLXArray] = []
        var valueRows: [MLXArray] = []
        if let keys, let values {
            let grow = width - offset
            keyRows.append(Self.padded(keys[.ellipsis, ..<offset, 0...], left: grow))
            valueRows.append(Self.padded(values[.ellipsis, ..<offset, 0...], left: grow))
            leftPadding = leftPadding.map { $0 + grow }
        }
        for row in incoming {
            let pad = width - row[0].dim(2)
            keyRows.append(Self.padded(row[0], left: pad))
            valueRows.append(Self.padded(row[1], left: pad))
            leftPadding.append(pad)
        }
        keys = concatenated(keyRows, axis: 0)
        values = concatenated(valueRows, axis: 0)
        offset = width
    }

    func filter(rows kept: [Int]) {
        guard let keys, let values, !kept.isEmpty else {
            self.keys = nil
            self.values = nil
            leftPadding = []
            offset = 0
            return
        }
        leftPadding = kept.map { leftPadding[$0] }
        let unused = leftPadding.min() ?? 0
        let index = MLXArray(kept.map(Int32.init))
        self.keys = keys[index][.ellipsis, unused ..< offset, 0...]
        self.values = values[index][.ellipsis, unused ..< offset, 0...]
        leftPadding = leftPadding.map { $0 - unused }
        offset -= unused
    }

    private static func padded(_ array: MLXArray, left count: Int) -> MLXArray {
        guard count > 0 else { return array }
        var shape = array.shape
        shape[2] = count
        return concatenated([MLXArray.zeros(shape, dtype: array.dtype), array], axis: 2)
    }

    private static func widened(_ array: MLXArray?, used: Int, to capacity: Int, like template: MLXArray) -> MLXArray {
        var shape = template.shape
        shape[2] = capacity - used
        let room = MLXArray.zeros(shape, dtype: template.dtype)
        guard let array, used > 0 else { return room }
        return concatenated([array[.ellipsis, ..<used, 0...], room], axis: 2)
    }
}
