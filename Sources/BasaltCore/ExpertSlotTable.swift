import Foundation

public struct ExpertCacheStats: Sendable, Equatable, Codable {
    public var hits = 0
    public var misses = 0
    public var evictions = 0
    public var warmed = 0
    public var prefetched = 0
    public var prefetchHits = 0
    public var bytesRead = ByteCount.zero

    public init() {}

    public var hitRate: Double { hits + misses > 0 ? Double(hits) / Double(hits + misses) : 0 }
    public var prefetchHitRate: Double { prefetched > 0 ? Double(prefetchHits) / Double(prefetched) : 0 }

    public var summary: String {
        let base = String(
            format: "hit rate %.1f%% (%d hits, %d misses, %d evictions, %@ read)",
            hitRate * 100, hits, misses, evictions, formatRAMBytes(bytesRead.rawValue))
        guard warmed > 0 || prefetched > 0 else { return base }
        return String(
            format: "%@, warmed %d, prefetch %d/%d (%.1f%%)",
            base, warmed, prefetchHits, prefetched, prefetchHitRate * 100)
    }
}

public struct ExpertPrefetchGate: Sendable, Equatable {
    public let threshold: Double
    public private(set) var scores: [Double]

    public init(topK: Int, threshold: Double = 0.5) {
        self.threshold = threshold
        let count = max(0, topK)
        let active = max(1, count / 2)
        scores = (0 ..< count).map { $0 < active ? 0.65 : 0.35 }
    }

    public var budget: Int {
        scores.reduce(0) { $0 + ($1 >= threshold ? 1 : 0) }
    }

    public func select(_ predicted: [Int]) -> [Int] {
        var seen: Set<Int> = []
        return zip(predicted, scores).compactMap { expert, score in
            score >= threshold && seen.insert(expert).inserted ? expert : nil
        }
    }

    public mutating func observe(predicted: [Int], actual: Set<Int>) {
        for index in 0 ..< min(scores.count, predicted.count) {
            let hit = actual.contains(predicted[index]) ? 1.0 : 0.0
            scores[index] = scores[index] * 0.8 + hit * 0.2
        }
    }
}

public struct ExpertSlotTable: Sendable, Equatable {
    public struct Placement: Sendable, Equatable {
        public var slots: [Int]
        public var loads: [Load]
        public var misses: Int
        public var evictions: Int
        public var prefetchHits = 0
    }

    public struct Load: Sendable, Equatable {
        public let expert: Int
        public let slot: Int
    }

    public let capacity: Int
    private var slotOf: [Int: Int] = [:]
    private var expertIn: [Int?]
    private var lastUse: [Int]
    private var prefetched: Set<Int> = []
    private var speculative: Set<Int> = []
    private var clock = 0

    public init(slots: Int) {
        capacity = max(0, slots)
        expertIn = Array(repeating: nil, count: capacity)
        lastUse = Array(repeating: -1, count: capacity)
    }

    public func slot(of expert: Int) -> Int? { slotOf[expert] }

    public mutating func place(_ experts: [Int], prefetch: Bool = false, speculative isSpeculative: Bool = false) -> Placement? {
        guard experts.count <= capacity, Set(experts).count == experts.count else { return nil }
        clock += 1
        for expert in experts { if let slot = slotOf[expert] { lastUse[slot] = clock } }
        var placement = Placement(slots: [], loads: [], misses: 0, evictions: 0)
        for expert in experts where slotOf[expert] == nil {
            let slot = lastUse.indices.min { lastUse[$0] < lastUse[$1] }!
            evictIfOccupied(slot: slot, evictions: &placement.evictions)
            (expertIn[slot], slotOf[expert], lastUse[slot]) = (expert, slot, clock)
            placement.loads.append(Load(expert: expert, slot: slot))
            if prefetch { prefetched.insert(expert) }
            if isSpeculative { speculative.insert(expert) }
        }
        placement.slots = experts.map { slotOf[$0]! }
        guard !prefetch else { return placement }
        placement.misses = placement.loads.count
        for expert in experts where prefetched.remove(expert) != nil { placement.misses += 1 }
        for expert in experts where speculative.remove(expert) != nil { placement.prefetchHits += 1 }
        return placement
    }

    public mutating func warm(_ ranked: [Int]) -> Placement {
        var seen: Set<Int> = []
        let target = Array(ranked.filter { seen.insert($0).inserted }.prefix(capacity))
        guard !target.isEmpty else { return Placement(slots: [], loads: [], misses: 0, evictions: 0) }
        let base = clock
        for (rank, expert) in target.enumerated() {
            if let slot = slotOf[expert] { lastUse[slot] = base + target.count - rank }
        }
        var placement = Placement(slots: [], loads: [], misses: 0, evictions: 0)
        for (rank, expert) in target.enumerated() where slotOf[expert] == nil {
            let slot = lastUse.indices.min { lastUse[$0] < lastUse[$1] }!
            evictIfOccupied(slot: slot, evictions: &placement.evictions)
            (expertIn[slot], slotOf[expert], lastUse[slot]) = (expert, slot, base + target.count - rank)
            placement.loads.append(Load(expert: expert, slot: slot))
        }
        clock = base + target.count
        placement.slots = target.map { slotOf[$0]! }
        return placement
    }

    private mutating func evictIfOccupied(slot: Int, evictions: inout Int) {
        guard let old = expertIn[slot] else { return }
        slotOf[old] = nil
        prefetched.remove(old)
        speculative.remove(old)
        evictions += 1
    }
}
