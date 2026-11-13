import Foundation

public struct ExpertCacheStats: Sendable, Equatable, Codable {
    public var hits = 0
    public var misses = 0
    public var evictions = 0
    public var bytesRead = ByteCount.zero

    public init() {}

    public var hitRate: Double { hits + misses > 0 ? Double(hits) / Double(hits + misses) : 0 }

    public var summary: String {
        String(format: "hit rate %.1f%% (%d hits, %d misses, %d evictions, %@ read)",
               hitRate * 100, hits, misses, evictions, formatRAMBytes(bytesRead.rawValue))
    }
}

public struct ExpertSlotTable: Sendable, Equatable {
    public struct Placement: Sendable, Equatable {
        public var slots: [Int]
        public var loads: [Load]
        public var misses: Int
        public var evictions: Int
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
    private var clock = 0

    public init(slots: Int) {
        capacity = max(0, slots)
        expertIn = Array(repeating: nil, count: capacity)
        lastUse = Array(repeating: -1, count: capacity)
    }

    public func slot(of expert: Int) -> Int? { slotOf[expert] }

    public mutating func place(_ experts: [Int], prefetch: Bool = false) -> Placement? {
        guard experts.count <= capacity, Set(experts).count == experts.count else { return nil }
        clock += 1
        for expert in experts { if let slot = slotOf[expert] { lastUse[slot] = clock } }
        var placement = Placement(slots: [], loads: [], misses: 0, evictions: 0)
        for expert in experts where slotOf[expert] == nil {
            let slot = lastUse.indices.min { lastUse[$0] < lastUse[$1] }!
            if let old = expertIn[slot] {
                slotOf[old] = nil
                prefetched.remove(old)
                placement.evictions += 1
            }
            (expertIn[slot], slotOf[expert], lastUse[slot]) = (expert, slot, clock)
            placement.loads.append(Load(expert: expert, slot: slot))
            if prefetch { prefetched.insert(expert) }
        }
        placement.slots = experts.map { slotOf[$0]! }
        guard !prefetch else { return placement }
        placement.misses = placement.loads.count
        for expert in experts where prefetched.remove(expert) != nil { placement.misses += 1 }
        return placement
    }
}
