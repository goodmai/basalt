import BasaltCore
import MLX
import MLXLMCommon

final class BatchQuantizedKVCache: QuantizedKVCacheProtocol {
    typealias QuantizedTuple = (MLXArray, MLXArray, MLXArray?)

    private var keys: QuantizedTuple?
    private var values: QuantizedTuple?
    private(set) var leftPadding: [Int] = []
    private(set) var offset = 0
    private(set) var groupSize: Int
    let bits: Int
    let mode: QuantizationMode
    let step = 256

    init(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
    }

    var maxSize: Int? { nil }
    var isTrimmable: Bool { false }

    var ropeOffset: RoPEOffset {
        if let first = leftPadding.first, leftPadding.allSatisfy({ $0 == first }) { return .scalar(offset - first) }
        return .batch(MLXArray(leftPadding.map { Int32(offset - $0) }))
    }

    func innerState() -> [MLXArray] {
        var arrays: [MLXArray] = []
        if let keys { arrays.append(contentsOf: [keys.0, keys.1, keys.2].compactMap { $0 }) }
        if let values { arrays.append(contentsOf: [values.0, values.1, values.2].compactMap { $0 }) }
        return arrays
    }

    var state: [MLXArray] {
        get { innerState() }
        set { fatalError("BatchQuantizedKVCache is built by append(_:), not restored from state") }
    }

    var metaState: [String] {
        get { [String(step), String(offset), String(groupSize), String(bits)] }
        set { fatalError("BatchQuantizedKVCache is built by append(_:), not restored from state") }
    }

    func trim(_ n: Int) -> Int { 0 }

    func copy() -> any KVCache { fatalError("BatchQuantizedKVCache is owned by one BatchedDecoder and never copied") }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1, !returnArray, windowSize == nil, leftPadding.allSatisfy({ $0 == 0 }) { return .none }
        let causal = createCausalMask(n: n, offset: offset, windowSize: windowSize)
        let columns = MLXArray(Int32(0) ..< Int32(offset + n))
        let padding = MLXArray(leftPadding.map(Int32.init)).reshaped(-1, 1, 1, 1)
        return .array(causal & (columns .>= padding))
    }

    func getQuantizedState() -> (QuantizedTuple, QuantizedTuple)? {
        guard let keys, let values else { return nil }
        return (Self.slice(keys, range: 0 ..< offset), Self.slice(values, range: 0 ..< offset))
    }

    func dequantizedState() -> (MLXArray, MLXArray)? {
        guard let (qk, qv) = getQuantizedState() else { return nil }
        return (
            dequantized(qk.0, scales: qk.1, biases: qk.2, groupSize: groupSize, bits: bits, mode: mode),
            dequantized(qv.0, scales: qv.1, biases: qv.2, groupSize: groupSize, bits: bits, mode: mode)
        )
    }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let (qk, qv) = updateQuantized(keys: newKeys, values: newValues)
        return (
            dequantized(qk.0, scales: qk.1, biases: qk.2, groupSize: groupSize, bits: bits, mode: mode),
            dequantized(qv.0, scales: qv.1, biases: qv.2, groupSize: groupSize, bits: bits, mode: mode)
        )
    }

    func updateQuantized(keys newKeys: MLXArray, values newValues: MLXArray) -> (QuantizedTuple, QuantizedTuple) {
        if keys == nil, values == nil, offset == 0,
           let resolved = KVCacheGeometry.quantizedGroupSize(keyDim: newKeys.dim(3), valueDim: newValues.dim(3)) {
            groupSize = resolved
        }
        let qk = quantized(newKeys, groupSize: groupSize, bits: bits, mode: mode)
        let qv = quantized(newValues, groupSize: groupSize, bits: bits, mode: mode)
        let incomingKeys: QuantizedTuple = (qk.wq, qk.scales, qk.biases)
        let incomingValues: QuantizedTuple = (qv.wq, qv.scales, qv.biases)
        let previous = offset
        let added = newKeys.dim(2)
        if previous + added > (keys?.0.dim(2) ?? 0) {
            let capacity = (previous + added + step - 1) / step * step
            keys = Self.widened(keys, used: previous, to: capacity, like: incomingKeys)
            values = Self.widened(values, used: previous, to: capacity, like: incomingValues)
        }
        offset += added
        guard var curKeys = keys, var curValues = values else {
            preconditionFailure("BatchQuantizedKVCache buffers not initialized")
        }
        Self.assign(&curKeys, range: previous ..< offset, from: incomingKeys)
        Self.assign(&curValues, range: previous ..< offset, from: incomingValues)
        keys = curKeys
        values = curValues
        return (Self.slice(curKeys, range: 0 ..< offset), Self.slice(curValues, range: 0 ..< offset))
    }

    func append(_ joining: [QuantizedKVCache]) {
        append(joining.map { $0 as any KVCache })
    }

    func append(_ joining: [KVCacheSimple]) {
        append(joining.map { $0 as any KVCache })
    }

    func append(_ joining: [any KVCache]) {
        let incoming = joining.compactMap { extractQuantized($0) }
        guard !incoming.isEmpty else { return }
        let width = max(offset, incoming.map { $0.0.0.dim(2) }.max() ?? 0)
        var keyRows: [QuantizedTuple] = []
        var valueRows: [QuantizedTuple] = []
        if let keys, let values {
            let grow = width - offset
            keyRows.append(Self.padded(Self.slice(keys, range: 0 ..< offset), left: grow))
            valueRows.append(Self.padded(Self.slice(values, range: 0 ..< offset), left: grow))
            leftPadding = leftPadding.map { $0 + grow }
        }
        for (qk, qv) in incoming {
            let pad = width - qk.0.dim(2)
            keyRows.append(Self.padded(qk, left: pad))
            valueRows.append(Self.padded(qv, left: pad))
            leftPadding.append(pad)
        }
        keys = Self.concatenatedTuples(keyRows, axis: 0)
        values = Self.concatenatedTuples(valueRows, axis: 0)
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
        self.keys = Self.compacted(keys, batchIndices: index, range: unused ..< offset)
        self.values = Self.compacted(values, batchIndices: index, range: unused ..< offset)
        leftPadding = leftPadding.map { $0 - unused }
        offset -= unused
    }
}

extension BatchQuantizedKVCache {
    private func extractQuantized(_ cache: any KVCache) -> (QuantizedTuple, QuantizedTuple)? {
        if let q = cache as? QuantizedKVCache {
            if keys == nil, offset == 0 { groupSize = q.groupSize }
            return q.getQuantizedState()
        }
        if let simple = cache as? KVCacheSimple {
            let s = simple.state
            guard s.count == 2 else { return nil }
            if keys == nil, offset == 0,
               let resolved = KVCacheGeometry.quantizedGroupSize(keyDim: s[0].dim(3), valueDim: s[1].dim(3)) {
                groupSize = resolved
            }
            let qk = quantized(s[0], groupSize: groupSize, bits: bits, mode: mode)
            let qv = quantized(s[1], groupSize: groupSize, bits: bits, mode: mode)
            return ((qk.wq, qk.scales, qk.biases), (qv.wq, qv.scales, qv.biases))
        }
        return nil
    }

    fileprivate static func slice(_ tuple: QuantizedTuple, range: Range<Int>) -> QuantizedTuple {
        (tuple.0[.ellipsis, range, 0...], tuple.1[.ellipsis, range, 0...], tuple.2?[.ellipsis, range, 0...])
    }

    fileprivate static func assign(_ target: inout QuantizedTuple, range: Range<Int>, from source: QuantizedTuple) {
        target.0[.ellipsis, range, 0...] = source.0
        target.1[.ellipsis, range, 0...] = source.1
        if let srcBias = source.2, let dstBias = target.2 {
            dstBias[.ellipsis, range, 0...] = srcBias
            target.2 = dstBias
        }
    }

    fileprivate static func padded(_ tuple: QuantizedTuple, left count: Int) -> QuantizedTuple {
        guard count > 0 else { return tuple }
        return (
            padArray(tuple.0, left: count),
            padArray(tuple.1, left: count),
            tuple.2.map { padArray($0, left: count) }
        )
    }

    private static func padArray(_ array: MLXArray, left count: Int) -> MLXArray {
        var shape = array.shape
        shape[2] = count
        return concatenated([MLXArray.zeros(shape, dtype: array.dtype), array], axis: 2)
    }

    fileprivate static func widened(
        _ tuple: QuantizedTuple?, used: Int, to capacity: Int, like template: QuantizedTuple
    ) -> QuantizedTuple {
        (
            widenArray(tuple?.0, used: used, to: capacity, like: template.0),
            widenArray(tuple?.1, used: used, to: capacity, like: template.1),
            template.2.map { widenArray(tuple?.2, used: used, to: capacity, like: $0) }
        )
    }

    private static func widenArray(_ array: MLXArray?, used: Int, to capacity: Int, like template: MLXArray) -> MLXArray {
        var shape = template.shape
        shape[2] = capacity - used
        let room = MLXArray.zeros(shape, dtype: template.dtype)
        guard let array, used > 0 else { return room }
        return concatenated([array[.ellipsis, ..<used, 0...], room], axis: 2)
    }

    fileprivate static func concatenatedTuples(_ tuples: [QuantizedTuple], axis: Int) -> QuantizedTuple {
        let biases = tuples.compactMap(\.2)
        return (
            concatenated(tuples.map(\.0), axis: axis),
            concatenated(tuples.map(\.1), axis: axis),
            biases.count == tuples.count ? concatenated(biases, axis: axis) : nil
        )
    }

    fileprivate static func compacted(
        _ tuple: QuantizedTuple, batchIndices: MLXArray, range: Range<Int>
    ) -> QuantizedTuple {
        (
            tuple.0[batchIndices][.ellipsis, range, 0...],
            tuple.1[batchIndices][.ellipsis, range, 0...],
            tuple.2?[batchIndices][.ellipsis, range, 0...]
        )
    }
}
