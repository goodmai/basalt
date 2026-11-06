import Foundation

public struct ExpertSlot: Hashable, Sendable {
    public let module: Int
    public let expert: Int

    public init(module: Int, expert: Int) {
        self.module = module
        self.expert = expert
    }
}

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

public struct ExpertLRU: Sendable {
    public let capacity: ByteCount
    public private(set) var stats = ExpertCacheStats()
    public private(set) var resident = ByteCount.zero
    private var slots: [ExpertSlot: (lastUse: Int, bytes: ByteCount)] = [:]
    private var clock = 0

    public init(capacity: ByteCount) {
        self.capacity = capacity
    }

    public func holds(_ slot: ExpertSlot) -> Bool { slots[slot] != nil }

    public mutating func admit(_ wanted: Set<ExpertSlot>, bytes: ByteCount) -> (missing: [ExpertSlot], evicted: [ExpertSlot]) {
        clock += 1
        let missing = wanted.filter { slots[$0] == nil }
        stats.hits += wanted.count - missing.count
        stats.misses += missing.count
        stats.bytesRead += bytes * Int64(missing.count)
        resident += bytes * Int64(missing.count)
        for slot in wanted { slots[slot] = (clock, slots[slot]?.bytes ?? bytes) }
        let evicted = evictUntilWithinCapacity()
        stats.evictions += evicted.count
        return (Array(missing), evicted)
    }

    private mutating func evictUntilWithinCapacity() -> [ExpertSlot] {
        guard resident > capacity else { return [] }
        var evicted: [ExpertSlot] = []
        let candidates = slots.filter { $0.value.lastUse < clock }.sorted { $0.value.lastUse < $1.value.lastUse }
        for (slot, entry) in candidates where resident > capacity {
            slots[slot] = nil
            resident -= entry.bytes
            evicted.append(slot)
        }
        return evicted
    }
}
