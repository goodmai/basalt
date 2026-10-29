import MLX
import MLXLMCommon
import BasaltCore

final class ZonedKVCache: KVCacheAttentionProtocol {
    typealias QuantizedTokens = (MLXArray, MLXArray, MLXArray?)

    struct QuantizedRows {
        let bits: Int
        let groupSize: Int
        private(set) var parts: QuantizedTokens?

        init(bits: Int, groupSize: Int) {
            self.bits = bits
            self.groupSize = groupSize
        }

        var count: Int { parts?.0.dim(2) ?? 0 }
        var arrays: [MLXArray] { parts.map { [$0.0, $0.1, $0.2].compactMap { $0 } } ?? [] }

        mutating func append(_ rows: MLXArray) {
            let q = quantized(rows, groupSize: groupSize, bits: bits)
            guard let old = parts else {
                parts = (q.wq, q.scales, q.biases)
                return
            }
            parts = (
                concatenated([old.0, q.wq], axis: 2), concatenated([old.1, q.scales], axis: 2),
                old.2.flatMap { bias in q.biases.map { concatenated([bias, $0], axis: 2) } })
        }

        mutating func trim(_ n: Int) -> Int {
            let removed = min(max(0, n), count)
            guard removed > 0, let old = parts else { return 0 }
            let kept = count - removed
            parts = kept == 0 ? nil : (old.0[.ellipsis, ..<kept, 0...], old.1[.ellipsis, ..<kept, 0...], old.2?[.ellipsis, ..<kept, 0...])
            return removed
        }

        func dequantizedRows() -> MLXArray? {
            parts.map { dequantized($0.0, scales: $0.1, biases: $0.2, groupSize: groupSize, bits: bits) }
        }
    }

    let zones: KVZones
    let precision: KVPrecision
    private let exact: KVCacheSimple
    private var middle: (keys: QuantizedRows, values: QuantizedRows)?
    private(set) var materializedForCaller = false
    private(set) var headsQuantize: Bool?
    private(set) var rotatesValues = false
    private(set) var evictedCount = 0
    private(set) var evictionRefused = false

    init(zones: KVZones, precision: KVPrecision) {
        self.zones = zones
        self.precision = precision
        exact = KVCacheSimple()
    }

    private init(source: ZonedKVCache, exact: KVCacheSimple) {
        zones = source.zones
        precision = source.precision
        self.exact = exact
        middle = source.middle
        materializedForCaller = source.materializedForCaller
        headsQuantize = source.headsQuantize
        rotatesValues = source.rotatesValues
        evictedCount = source.evictedCount
        evictionRefused = source.evictionRefused
    }

    static func rotationApplies(toHeadDim dim: Int) -> Bool {
        [1, 12].contains { factor in
            dim.isMultiple(of: factor) && dim / factor > 0 && (dim / factor) & (dim / factor - 1) == 0
        }
    }

    static func modelAttendsThroughCache(_ model: any LanguageModel) throws -> Bool {
        try probe(model, precision: .q8).attendsThroughCache
    }

    static func probe(_ model: any LanguageModel, precision: KVPrecision) throws -> (attendsThroughCache: Bool, rotatesValues: Bool) {
        let caches = replacingFullAttention(try model.newCache(parameters: nil), zones: .standard, precision: precision)
        let zoned = caches.compactMap { $0 as? ZonedKVCache }
        guard !zoned.isEmpty else { return (false, false) }
        _ = model(LMInput.Text(tokens: MLXArray([Int32(0)]).reshaped(1, 1)), cache: caches, state: nil)
        return (zoned.allSatisfy { !$0.materializedForCaller && $0.headsQuantize == true }, zoned.allSatisfy(\.rotatesValues))
    }

    static func prefillsWithinWindow(
        _ model: any LanguageModel, zones: KVZones, precision: KVPrecision, prefill: PrefillParameters
    ) throws -> Bool {
        let caches = replacingFullAttention(try model.newCache(parameters: nil), zones: zones, precision: precision)
        let zoned = caches.compactMap { $0 as? ZonedKVCache }
        guard !zoned.isEmpty else { return false }
        let prompt = MLXArray((0...(zones.windowTokens + 1)).map { Int32(1 + $0 % 64) })
        if case .tokens(let rest) = try model.prepare(LMInput(tokens: prompt), cache: caches, state: nil, prefill: prefill) {
            _ = model(rest[text: .newAxis], cache: caches, state: nil)
        }
        return zoned.allSatisfy { !$0.evictionRefused }
    }

    static func replacingFullAttention(_ caches: [any KVCache], zones: KVZones, precision: KVPrecision) -> [any KVCache] {
        caches.map { type(of: $0) == KVCacheSimple.self ? ZonedKVCache(zones: zones, precision: precision) : $0 }
    }

    var sinkCount: Int { min(zones.sinkTokens, exact.offset) }
    var quantizedCount: Int { middle?.keys.count ?? 0 }
    var windowCount: Int { exact.offset - sinkCount }
    var storedCount: Int { exact.offset + quantizedCount }
    var offset: Int { storedCount + evictedCount }
    private var middleArrays: [MLXArray] { (middle?.keys.arrays ?? []) + (middle?.values.arrays ?? []) }
    var maxSize: Int? { nil }
    var isTrimmable: Bool { evictedCount == 0 }

    func innerState() -> [MLXArray] {
        exact.innerState() + middleArrays
    }

    var state: [MLXArray] {
        get { exact.state + middleArrays }
        set { preconditionFailure("ZonedKVCache is never restored from a serialized prompt cache") }
    }

    var metaState: [String] {
        get {
            [zones.sinkTokens, zones.windowTokens, zones.migrationStep, precision.keyBits, precision.valueBits].map(String.init)
                + [String(rotatesValues), String(zones.evictedMiddleFraction), String(evictedCount)]
        }
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
        if queries.dim(2) == 1, case .none = mask, let middle {
            return decodeAttention(queries: queries, middle: middle, scale: scale)
        }
        let (allKeys, allValues) = materialized()
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: allKeys, values: allValues, scale: scale, mask: Self.trailingColumns(of: mask, keys: allKeys.dim(2)))
    }

    static func trailingColumns(
        of mask: MLXFast.ScaledDotProductAttentionMaskMode, keys: Int
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard case .array(let columns) = mask, columns.dim(-1) > keys else { return mask }
        return .array(columns[.ellipsis, (columns.dim(-1) - keys)...])
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        var left = max(0, n)
        left -= exact.trim(min(left, windowCount))
        if evictedCount > 0 { return max(0, n) - left }
        if var trimmed = middle {
            let removed = trimmed.keys.trim(left)
            _ = trimmed.values.trim(removed)
            left -= removed
            middle = trimmed.keys.count == 0 ? nil : trimmed
        }
        left -= exact.trim(left)
        return max(0, n) - left
    }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 { return .none }
        if returnArray || windowSize.map({ n > $0 }) == true {
            return .array(createCausalMask(n: n, offset: storedCount, windowSize: windowSize))
        }
        return .causal
    }

    func copy() -> any KVCache {
        guard let exactCopy = exact.copy() as? KVCacheSimple else {
            preconditionFailure("KVCacheSimple copies to its own type")
        }
        return ZonedKVCache(source: self, exact: exactCopy)
    }

    private func append(keys: MLXArray, values: MLXArray) {
        evictionRefused = evictionRefused || keys.dim(2) > zones.windowTokens
        _ = exact.update(keys: keys, values: values)
        let groupSize = KVCacheGeometry.quantizedGroupSize(keyDim: keys.dim(3), valueDim: values.dim(3))
        rotatesValues = precision.rotatesValues && Self.rotationApplies(toHeadDim: values.dim(3))
        headsQuantize = groupSize != nil
        let overflow = windowCount - zones.windowTokens
        guard overflow >= zones.migrationStep, let groupSize else { return }
        let (sinks, moved, recent) = (sinkCount, sinkCount..<(sinkCount + overflow), (sinkCount + overflow)...)
        let tokens = exact.state
        var grown = middle ?? (
            QuantizedRows(bits: precision.keyBits, groupSize: groupSize), QuantizedRows(bits: precision.valueBits, groupSize: groupSize))
        let (keptKeys, keptValues) = evictingHighestKeyNorms(keys: tokens[0][.ellipsis, moved, 0...], values: tokens[1][.ellipsis, moved, 0...])
        if keptKeys.dim(2) > 0 {
            grown.keys.append(keptKeys)
            grown.values.append(rotatedValues(keptValues))
            middle = grown
        }
        exact.state = tokens.map { concatenated([$0[.ellipsis, ..<sinks, 0...], $0[.ellipsis, recent, 0...]], axis: 2) }
        eval(exact.innerState() + middleArrays)
    }

    private func evictingHighestKeyNorms(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let arriving = keys.dim(2)
        let kept = zones.keptMiddleTokens(quantizedCount + evictedCount + arriving) - quantizedCount
        guard !evictionRefused, kept < arriving else { return (keys, values) }
        evictedCount += arriving - kept
        let squaredNorms = keys.asType(.float32).square().sum(axis: -1)
        let rows = expandedDimensions(sorted(argSort(squaredNorms, axis: -1)[.ellipsis, ..<kept], axis: -1), axis: -1)
        return (takeAlong(keys, rows, axis: 2), takeAlong(values, rows, axis: 2))
    }

    private func rotatedValues(_ values: MLXArray) -> MLXArray {
        rotatesValues ? hadamardTransform(values.asType(.float32)).asType(values.dtype) : values
    }

    private func materialized() -> (MLXArray, MLXArray) {
        let tokens = exact.state
        guard let middle, let keys = middle.keys.dequantizedRows(), let values = middle.values.dequantizedRows() else {
            return (tokens[0], tokens[1])
        }
        let sinks = sinkCount
        let middleTokens = [keys, rotatedValues(values)]
        let ordered = zip(tokens, middleTokens).map { full, middle in
            concatenated([full[.ellipsis, ..<sinks, 0...], middle, full[.ellipsis, sinks..., 0...]], axis: 2)
        }
        return (ordered[0], ordered[1])
    }

    private func decodeAttention(
        queries: MLXArray, middle: (keys: QuantizedRows, values: QuantizedRows), scale: Float
    ) -> MLXArray {
        let (batch, heads, length, dim) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
        guard let qk = middle.keys.parts, let qv = middle.values.parts else {
            preconditionFailure("the middle zone holds keys and values for the same tokens")
        }
        let tokens = exact.state.map { expandedDimensions($0, axis: 2) }
        let kvHeads = qk.0.dim(1)
        let grouped = (queries * scale).reshaped(batch, kvHeads, heads / kvHeads, length, dim)
        let perGroup = { (part: QuantizedTokens) in
            (expandedDimensions(part.0, axis: 2), expandedDimensions(part.1, axis: 2), part.2.map { expandedDimensions($0, axis: 2) })
        }
        let (keys, values) = (perGroup(qk), perGroup(qv))
        let quantizedScores = quantizedMM(
            grouped, keys.0, scales: keys.1, biases: keys.2,
            transpose: true, groupSize: middle.keys.groupSize, bits: middle.keys.bits)
        let exactScores = matmul(grouped, tokens[0].swappedAxes(-1, -2))
        let weights = softmax(concatenated([quantizedScores, exactScores], axis: -1), axis: -1, precise: true)
        let split = qk.0.dim(2)
        let middleOutput = quantizedMM(
            weights[.ellipsis, ..<split], values.0, scales: values.1, biases: values.2,
            transpose: false, groupSize: middle.values.groupSize, bits: middle.values.bits)
        let output = rotatedValues(middleOutput) + matmul(weights[.ellipsis, split...], tokens[1])
        return output.reshaped(batch, heads, length, -1)
    }
}
