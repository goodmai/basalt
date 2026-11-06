import MLX
import MLXLMCommon
import BasaltCore

final class EvictingKVCache: KVCache {
    let zones: KVZones
    private let exact: KVCacheSimple
    private(set) var middleCount = 0
    private(set) var evictedCount = 0

    init(zones: KVZones) {
        self.zones = zones
        exact = KVCacheSimple()
    }

    private init(source: EvictingKVCache, exact: KVCacheSimple) {
        zones = source.zones
        self.exact = exact
        middleCount = source.middleCount
        evictedCount = source.evictedCount
    }

    static func replacingFullAttention(_ caches: [any KVCache], zones: KVZones) -> [any KVCache] {
        caches.map { type(of: $0) == KVCacheSimple.self ? EvictingKVCache(zones: zones) : $0 }
    }

    static func keepingLowestKeyNorms(keys: MLXArray, values: MLXArray, count: Int) -> (MLXArray, MLXArray) {
        let squaredNorms = keys.asType(.float32).square().sum(axis: -1)
        let rows = expandedDimensions(sorted(argSort(squaredNorms, axis: -1)[.ellipsis, ..<count], axis: -1), axis: -1)
        return (takeAlong(keys, rows, axis: 2), takeAlong(values, rows, axis: 2))
    }

    var storedCount: Int { exact.offset }
    var sinkCount: Int { min(zones.sinkTokens, storedCount) }
    var windowCount: Int { storedCount - sinkCount - middleCount }
    var offset: Int { storedCount + evictedCount }
    var maxSize: Int? { nil }
    var isTrimmable: Bool { evictedCount == 0 }

    func innerState() -> [MLXArray] {
        exact.innerState()
    }

    var state: [MLXArray] {
        get { exact.state }
        set { preconditionFailure("EvictingKVCache is never restored from a serialized prompt cache") }
    }

    var metaState: [String] {
        get {
            [zones.sinkTokens, zones.windowTokens, zones.migrationStep, middleCount, evictedCount].map(String.init)
                + [String(zones.evictedMiddleFraction)]
        }
        set { preconditionFailure("EvictingKVCache is never restored from a serialized prompt cache") }
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let stored = exact.update(keys: keys, values: values)
        migrateOverflow()
        return stored
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        let removed = exact.trim(min(max(0, n), evictedCount == 0 ? storedCount : windowCount))
        middleCount = min(middleCount, storedCount - sinkCount)
        return removed
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
        return EvictingKVCache(source: self, exact: exactCopy)
    }

    private func migrateOverflow() {
        let overflow = windowCount - zones.windowTokens
        guard overflow >= zones.migrationStep else { return }
        let kept = zones.keptMiddleTokens(middleCount + evictedCount + overflow) - middleCount
        let start = sinkCount + middleCount
        middleCount += kept
        guard kept < overflow else { return }
        evictedCount += overflow - kept
        let tokens = exact.state
        let moved = start..<(start + overflow)
        let (keptKeys, keptValues) = Self.keepingLowestKeyNorms(
            keys: tokens[0][.ellipsis, moved, 0...], values: tokens[1][.ellipsis, moved, 0...], count: kept)
        exact.state = zip(tokens, [keptKeys, keptValues]).map { all, middle in
            concatenated([all[.ellipsis, ..<start, 0...], middle, all[.ellipsis, moved.upperBound..., 0...]], axis: 2)
        }
        eval(exact.innerState())
    }
}
