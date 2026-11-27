import Foundation

public enum ServingComputeRegime: String, Sendable, Codable, Equatable, CaseIterable {
    case memoryBound = "memory_bound"
    case computeBound = "compute_bound"
    case idle = "idle"
}

public struct ServingSLOSnapshot: Sendable, Codable, Equatable {
    public var sampleCount: Int = 0
    public var itlSampleCount: Int = 0
    public var ttftP50Seconds: Double = 0
    public var ttftP95Seconds: Double = 0
    public var ttftP99Seconds: Double = 0
    public var itlP50Seconds: Double = 0
    public var itlP95Seconds: Double = 0
    public var itlP99Seconds: Double = 0
    public var tpotMeanSeconds: Double = 0
    public var prefillTokensPerSecond: Double = 0
    public var decodeTokensPerSecond: Double = 0
    public var queueWaitP50Seconds: Double = 0
    public var queueWaitP95Seconds: Double = 0
    public var achievedMemoryBandwidthGBps: Double = 0
    public var peakMemoryBandwidthGBps: Double = 0
    public var memoryBandwidthUtilization: Double = 0
    public var regime: ServingComputeRegime = .idle

    public init() {}

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sampleCount = try Self.decodeCount(c, .sampleCount)
        itlSampleCount = try Self.decodeCount(c, .itlSampleCount)
        ttftP50Seconds = try Self.decodeMetric(c, .ttftP50Seconds)
        ttftP95Seconds = try Self.decodeMetric(c, .ttftP95Seconds)
        ttftP99Seconds = try Self.decodeMetric(c, .ttftP99Seconds)
        itlP50Seconds = try Self.decodeMetric(c, .itlP50Seconds)
        itlP95Seconds = try Self.decodeMetric(c, .itlP95Seconds)
        itlP99Seconds = try Self.decodeMetric(c, .itlP99Seconds)
        tpotMeanSeconds = try Self.decodeMetric(c, .tpotMeanSeconds)
        prefillTokensPerSecond = try Self.decodeMetric(c, .prefillTokensPerSecond)
        decodeTokensPerSecond = try Self.decodeMetric(c, .decodeTokensPerSecond)
        queueWaitP50Seconds = try Self.decodeMetric(c, .queueWaitP50Seconds)
        queueWaitP95Seconds = try Self.decodeMetric(c, .queueWaitP95Seconds)
        achievedMemoryBandwidthGBps = try Self.decodeMetric(c, .achievedMemoryBandwidthGBps)
        peakMemoryBandwidthGBps = try Self.decodeMetric(c, .peakMemoryBandwidthGBps)
        memoryBandwidthUtilization = try Self.decodeMetric(c, .memoryBandwidthUtilization)
        regime = try c.decodeIfPresent(ServingComputeRegime.self, forKey: .regime) ?? .idle
    }

    private static func decodeCount(_ c: KeyedDecodingContainer<CodingKeys>, _ key: CodingKeys) throws -> Int {
        max(0, try c.decodeIfPresent(Int.self, forKey: key) ?? 0)
    }

    private static func decodeMetric(_ c: KeyedDecodingContainer<CodingKeys>, _ key: CodingKeys) throws -> Double {
        let value = try c.decodeIfPresent(Double.self, forKey: key) ?? 0
        return value.isFinite && value >= 0 ? value : 0
    }

    enum CodingKeys: String, CodingKey {
        case sampleCount = "sample_count"
        case itlSampleCount = "itl_sample_count"
        case ttftP50Seconds = "ttft_p50_seconds"
        case ttftP95Seconds = "ttft_p95_seconds"
        case ttftP99Seconds = "ttft_p99_seconds"
        case itlP50Seconds = "itl_p50_seconds"
        case itlP95Seconds = "itl_p95_seconds"
        case itlP99Seconds = "itl_p99_seconds"
        case tpotMeanSeconds = "tpot_mean_seconds"
        case prefillTokensPerSecond = "prefill_tokens_per_second"
        case decodeTokensPerSecond = "decode_tokens_per_second"
        case queueWaitP50Seconds = "queue_wait_p50_seconds"
        case queueWaitP95Seconds = "queue_wait_p95_seconds"
        case achievedMemoryBandwidthGBps = "achieved_memory_bandwidth_gbps"
        case peakMemoryBandwidthGBps = "peak_memory_bandwidth_gbps"
        case memoryBandwidthUtilization = "memory_bandwidth_utilization"
        case regime
    }
}

public struct ServingSLORecord: Sendable, Equatable {
    public var queueWaitSeconds: Double
    public var ttftSeconds: Double
    public var prefillTokens: Int
    public var prefillSeconds: Double
    public var decodeTokens: Int
    public var decodeSeconds: Double
    public var itlSamplesSeconds: [Double]
    public var activeWeightBytes: UInt64

    public init(
        queueWaitSeconds: Double = 0,
        ttftSeconds: Double = 0,
        prefillTokens: Int = 0,
        prefillSeconds: Double = 0,
        decodeTokens: Int = 0,
        decodeSeconds: Double = 0
    ) {
        self.queueWaitSeconds = queueWaitSeconds
        self.ttftSeconds = ttftSeconds
        self.prefillTokens = prefillTokens
        self.prefillSeconds = prefillSeconds
        self.decodeTokens = decodeTokens
        self.decodeSeconds = decodeSeconds
        self.itlSamplesSeconds = []
        self.activeWeightBytes = 0
    }

    public func withDecodeDetail(
        itlSamplesSeconds: [Double] = [],
        activeWeightBytes: UInt64 = 0
    ) -> ServingSLORecord {
        var copy = self
        copy.itlSamplesSeconds = itlSamplesSeconds
        copy.activeWeightBytes = activeWeightBytes
        return copy
    }
}

public enum AppleSiliconBandwidthTable: Sendable {
    public static let fallbackBandwidthGBps: Double = 100.0
    public static let memoryBoundUtilizationThreshold: Double = 0.50

    public static func peakBandwidthGBps(for chipModel: String, gpuCores: Int? = nil) -> Double {
        let normalized = chipModel.replacingOccurrences(of: "_", with: " ")
        return AppleSiliconChip(normalized)?.memoryBandwidthGBs(gpuCores: gpuCores) ?? fallbackBandwidthGBps
    }

    public static func achievedBandwidthGBps(activeWeightBytes: UInt64, decodeTokensPerSecond: Double) -> Double {
        guard activeWeightBytes > 0, decodeTokensPerSecond.isFinite, decodeTokensPerSecond > 0 else { return 0 }
        let achieved = (Double(activeWeightBytes) * decodeTokensPerSecond) / 1e9
        return achieved.isFinite ? achieved : 0
    }

    public static func utilization(achievedGBps: Double, peakGBps: Double) -> Double {
        guard achievedGBps.isFinite, achievedGBps > 0, peakGBps.isFinite, peakGBps > 0 else { return 0 }
        let ratio = achievedGBps / peakGBps
        return ratio.isFinite ? ratio : 0
    }

    public static func classifyRegime(
        achievedGBps: Double,
        peakGBps: Double,
        prefillTokPerSec: Double,
        decodeTokPerSec: Double
    ) -> ServingComputeRegime {
        let achieved = sanitizePositive(achievedGBps)
        let peak = sanitizePositive(peakGBps)
        let prefill = sanitizePositive(prefillTokPerSec)
        let decode = sanitizePositive(decodeTokPerSec)
        guard max(achieved, max(prefill, decode)) > 0 else { return .idle }
        if peak > 0, achieved > 0 {
            return (achieved / peak) >= memoryBoundUtilizationThreshold ? .memoryBound : .computeBound
        }
        return max(decode, achieved) > 0 ? .memoryBound : .computeBound
    }

    private static func sanitizePositive(_ value: Double) -> Double {
        value.isFinite && value > 0 ? value : 0
    }
}

private struct FixedRingBuffer<Element: Sendable & Equatable>: Sendable, Equatable {
    let capacity: Int
    private var storage: [Element] = []
    private var writeIndex: Int = 0

    init(capacity: Int) {
        self.capacity = max(1, capacity)
    }

    var count: Int { storage.count }
    var isEmpty: Bool { storage.isEmpty }
    var elements: [Element] { storage }

    var orderedElements: [Element] {
        guard storage.count == capacity, writeIndex > 0 else { return storage }
        return Array(storage[writeIndex...]) + Array(storage[..<writeIndex])
    }

    mutating func append(_ element: Element) {
        if storage.count < capacity {
            storage.append(element)
            writeIndex = storage.count % capacity
        } else {
            storage[writeIndex] = element
            writeIndex = (writeIndex + 1) % capacity
        }
    }

    mutating func reset() {
        storage.removeAll(keepingCapacity: false)
        writeIndex = 0
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.capacity == rhs.capacity && lhs.orderedElements == rhs.orderedElements
    }
}

public struct ServingSLOReducer: Sendable, Equatable {
    public static let defaultRequestCapacity = 512
    public static let defaultITLCapacity = 4096

    private struct StoredRequest: Sendable, Equatable {
        let queueWaitSeconds: Double?
        let ttftSeconds: Double?
        let prefillTokens: Int
        let prefillSeconds: Double
        let decodeTokens: Int
        let decodeSeconds: Double
        let activeWeightBytes: UInt64

        init(_ record: ServingSLORecord) {
            queueWaitSeconds = Self.nonNegative(record.queueWaitSeconds)
            ttftSeconds = Self.nonNegative(record.ttftSeconds)
            let pSec = Self.positive(record.prefillSeconds)
            let pTok = max(0, record.prefillTokens)
            prefillTokens = pSec > 0 ? pTok : 0
            prefillSeconds = pTok > 0 ? pSec : 0
            let dSec = Self.positive(record.decodeSeconds)
            let dTok = max(0, record.decodeTokens)
            decodeTokens = dSec > 0 ? dTok : 0
            decodeSeconds = dTok > 0 ? dSec : 0
            activeWeightBytes = record.activeWeightBytes
        }

        private static func nonNegative(_ value: Double) -> Double? {
            value.isFinite && value >= 0 ? value : nil
        }

        private static func positive(_ value: Double) -> Double {
            value.isFinite && value > 0 ? value : 0
        }
    }

    private struct Totals {
        var prefillTokens = 0
        var prefillSeconds = 0.0
        var decodeTokens = 0
        var decodeSeconds = 0.0
        var weightedGB = 0.0
        var weightedSeconds = 0.0
    }

    private var requests: FixedRingBuffer<StoredRequest>
    private var itlSamples: FixedRingBuffer<Double>

    public init(
        requestCapacity: Int = defaultRequestCapacity,
        itlCapacity: Int = defaultITLCapacity
    ) {
        requests = FixedRingBuffer(capacity: requestCapacity)
        itlSamples = FixedRingBuffer(capacity: itlCapacity)
    }

    public var requestCapacity: Int { requests.capacity }
    public var itlCapacity: Int { itlSamples.capacity }
    public var requestCount: Int { requests.count }
    public var itlSampleCount: Int { itlSamples.count }
    public var isEmpty: Bool { requests.isEmpty && itlSamples.isEmpty }

    public mutating func record(_ record: ServingSLORecord) {
        requests.append(StoredRequest(record))
        for sample in record.itlSamplesSeconds where sample.isFinite && sample >= 0 {
            itlSamples.append(sample)
        }
    }

    public mutating func reset() {
        requests.reset()
        itlSamples.reset()
    }

    public func snapshot(peakBandwidthGBps: Double) -> ServingSLOSnapshot {
        var out = ServingSLOSnapshot()
        out.sampleCount = requests.count
        out.itlSampleCount = itlSamples.count
        let peak = peakBandwidthGBps.isFinite && peakBandwidthGBps > 0 ? peakBandwidthGBps : 0
        out.peakMemoryBandwidthGBps = peak
        guard !isEmpty else { return out }
        let totals = accumulateTotals()
        let ttfts = requests.elements.compactMap(\.ttftSeconds).sorted()
        let waits = requests.elements.compactMap(\.queueWaitSeconds).sorted()
        let itls = itlSamples.elements.sorted()
        out.ttftP50Seconds = Self.sortedQuantile(ttfts, quantile: 0.50)
        out.ttftP95Seconds = Self.sortedQuantile(ttfts, quantile: 0.95)
        out.ttftP99Seconds = Self.sortedQuantile(ttfts, quantile: 0.99)
        out.itlP50Seconds = Self.sortedQuantile(itls, quantile: 0.50)
        out.itlP95Seconds = Self.sortedQuantile(itls, quantile: 0.95)
        out.itlP99Seconds = Self.sortedQuantile(itls, quantile: 0.99)
        out.queueWaitP50Seconds = Self.sortedQuantile(waits, quantile: 0.50)
        out.queueWaitP95Seconds = Self.sortedQuantile(waits, quantile: 0.95)
        out.tpotMeanSeconds = Self.safeRate(totals.decodeSeconds, over: Double(totals.decodeTokens))
        out.prefillTokensPerSecond = Self.safeRate(Double(totals.prefillTokens), over: totals.prefillSeconds)
        out.decodeTokensPerSecond = Self.safeRate(Double(totals.decodeTokens), over: totals.decodeSeconds)
        out.achievedMemoryBandwidthGBps = Self.safeRate(totals.weightedGB, over: totals.weightedSeconds)
        out.memoryBandwidthUtilization = AppleSiliconBandwidthTable.utilization(
            achievedGBps: out.achievedMemoryBandwidthGBps, peakGBps: peak)
        out.regime = AppleSiliconBandwidthTable.classifyRegime(
            achievedGBps: out.achievedMemoryBandwidthGBps, peakGBps: peak,
            prefillTokPerSec: out.prefillTokensPerSecond, decodeTokPerSec: out.decodeTokensPerSecond)
        return out
    }

    public static func nearestRankQuantile(_ values: [Double], quantile: Double) -> Double {
        sortedQuantile(values.filter { $0.isFinite && $0 >= 0 }.sorted(), quantile: quantile)
    }

    private func accumulateTotals() -> Totals {
        var totals = Totals()
        for entry in requests.elements {
            totals.prefillTokens += entry.prefillTokens
            totals.prefillSeconds += entry.prefillSeconds
            totals.decodeTokens += entry.decodeTokens
            totals.decodeSeconds += entry.decodeSeconds
            if entry.activeWeightBytes > 0, entry.decodeTokens > 0, entry.decodeSeconds > 0 {
                totals.weightedGB += (Double(entry.activeWeightBytes) / 1e9) * Double(entry.decodeTokens)
                totals.weightedSeconds += entry.decodeSeconds
            }
        }
        return totals
    }

    private static func sortedQuantile(_ sorted: [Double], quantile: Double) -> Double {
        guard !sorted.isEmpty, quantile.isFinite else { return 0 }
        let clamped = min(max(quantile, 0), 1)
        guard clamped > 0 else { return sorted[0] }
        let rank = Int((clamped * Double(sorted.count)).rounded(.up))
        return sorted[min(max(rank - 1, 0), sorted.count - 1)]
    }

    private static func safeRate(_ numerator: Double, over denominator: Double) -> Double {
        guard numerator.isFinite, numerator > 0, denominator.isFinite, denominator > 0 else { return 0 }
        let rate = numerator / denominator
        return rate.isFinite ? rate : 0
    }
}
