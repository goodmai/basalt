import MLX
import MLXLMCommon
import BasaltCore

final class ZonedKVCache: KVCacheAttentionProtocol {
    typealias QuantizedTokens = (MLXArray, MLXArray, MLXArray?)

    let zones: KVZones
    private let exact: KVCacheSimple
    private let middle: QuantizedKVCache
    private(set) var materializedForCaller = false

    init(zones: KVZones, bits: Int) {
        self.zones = zones
        exact = KVCacheSimple()
        middle = QuantizedKVCache(groupSize: KVCacheGeometry.defaultQuantizedGroupSize, bits: bits)
    }

    private init(zones: KVZones, exact: KVCacheSimple, middle: QuantizedKVCache) {
        self.zones = zones
        self.exact = exact
        self.middle = middle
    }

    static func modelAttendsThroughCache(_ model: any LanguageModel) throws -> Bool {
        let caches = replacingFullAttention(try model.newCache(parameters: nil), zones: .standard, bits: 8)
        let zoned = caches.compactMap { $0 as? ZonedKVCache }
        guard !zoned.isEmpty else { return false }
        _ = model(LMInput.Text(tokens: MLXArray([Int32(0)]).reshaped(1, 1)), cache: caches, state: nil)
        return zoned.allSatisfy { !$0.materializedForCaller }
    }

    static func replacingFullAttention(_ caches: [any KVCache], zones: KVZones, bits: Int) -> [any KVCache] {
        caches.map { type(of: $0) == KVCacheSimple.self ? ZonedKVCache(zones: zones, bits: bits) : $0 }
    }

    var bits: Int { middle.bits }
    var sinkCount: Int { min(zones.sinkTokens, exact.offset) }
    var quantizedCount: Int { middle.offset }
    var windowCount: Int { exact.offset - sinkCount }
    var offset: Int { exact.offset + middle.offset }
    var maxSize: Int? { nil }
    var isTrimmable: Bool { true }

    func innerState() -> [MLXArray] {
        exact.innerState() + middle.innerState()
    }

    var state: [MLXArray] {
        get { exact.state + middle.state }
        set { preconditionFailure("ZonedKVCache is never restored from a serialized prompt cache") }
    }

    var metaState: [String] {
        get { [String(zones.sinkTokens), String(zones.windowTokens), String(zones.migrationStep)] + middle.metaState }
        set { preconditionFailure("ZonedKVCache is never restored from a serialized prompt cache") }
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        materializedForCaller = true
        append(keys: keys, values: values)
        return materialized()
    }

    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        append(keys: keys, values: values)
        if queries.dim(2) == 1, case .none = mask, let quantized = middle.getQuantizedState() {
            return decodeAttention(queries: queries, quantized: quantized, scale: scale)
        }
        let (allKeys, allValues) = materialized()
        return MLXFast.scaledDotProductAttention(queries: queries, keys: allKeys, values: allValues, scale: scale, mask: mask)
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        var left = max(0, n)
        left -= exact.trim(min(left, windowCount))
        left -= middle.trim(left)
        left -= exact.trim(left)
        return max(0, n) - left
    }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 { return .none }
        if returnArray || windowSize.map({ n > $0 }) == true {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    func copy() -> any KVCache {
        guard let exactCopy = exact.copy() as? KVCacheSimple, let middleCopy = middle.copy() as? QuantizedKVCache else {
            preconditionFailure("KVCacheSimple and QuantizedKVCache copy to their own type")
        }
        return ZonedKVCache(zones: zones, exact: exactCopy, middle: middleCopy)
    }

    private func append(keys: MLXArray, values: MLXArray) {
        _ = exact.update(keys: keys, values: values)
        let overflow = windowCount - zones.windowTokens
        guard overflow >= zones.migrationStep,
              KVCacheGeometry.quantizedGroupSize(keyDim: keys.dim(3), valueDim: values.dim(3)) != nil else { return }
        let (sinks, moved, recent) = (sinkCount, sinkCount..<(sinkCount + overflow), (sinkCount + overflow)...)
        let tokens = exact.state
        _ = middle.updateQuantized(keys: tokens[0][.ellipsis, moved, 0...], values: tokens[1][.ellipsis, moved, 0...])
        exact.state = tokens.map { concatenated([$0[.ellipsis, ..<sinks, 0...], $0[.ellipsis, recent, 0...]], axis: 2) }
        eval(exact.innerState() + middle.innerState())
    }

    private func materialized() -> (MLXArray, MLXArray) {
        let tokens = exact.state
        guard let (k, v) = middle.getQuantizedState() else { return (tokens[0], tokens[1]) }
        let sinks = sinkCount
        let middleTokens = [k, v].map {
            dequantized($0.0, scales: $0.1, biases: $0.2, groupSize: middle.groupSize, bits: middle.bits, mode: middle.mode)
        }
        let ordered = zip(tokens, middleTokens).map { full, middle in
            concatenated([full[.ellipsis, ..<sinks, 0...], middle, full[.ellipsis, sinks..., 0...]], axis: 2)
        }
        return (ordered[0], ordered[1])
    }

    private func decodeAttention(
        queries: MLXArray, quantized: (QuantizedTokens, QuantizedTokens), scale: Float
    ) -> MLXArray {
        let (batch, heads, length, dim) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
        let ((qk, qv), tokens) = (quantized, exact.state.map { expandedDimensions($0, axis: 2) })
        let kvHeads = qk.0.dim(1)
        let grouped = (queries * scale).reshaped(batch, kvHeads, heads / kvHeads, length, dim)
        let perGroup = { (part: QuantizedTokens) in
            (expandedDimensions(part.0, axis: 2), expandedDimensions(part.1, axis: 2), part.2.map { expandedDimensions($0, axis: 2) })
        }
        let (keys, values) = (perGroup(qk), perGroup(qv))
        let quantizedScores = quantizedMM(
            grouped, keys.0, scales: keys.1, biases: keys.2,
            transpose: true, groupSize: middle.groupSize, bits: middle.bits, mode: middle.mode)
        let exactScores = matmul(grouped, tokens[0].swappedAxes(-1, -2))
        let weights = softmax(concatenated([quantizedScores, exactScores], axis: -1), axis: -1, precise: true)
        let split = qk.0.dim(2)
        let output = quantizedMM(
            weights[.ellipsis, ..<split], values.0, scales: values.1, biases: values.2,
            transpose: false, groupSize: middle.groupSize, bits: middle.bits, mode: middle.mode)
            + matmul(weights[.ellipsis, split...], tokens[1])
        return output.reshaped(batch, heads, length, -1)
    }
}
