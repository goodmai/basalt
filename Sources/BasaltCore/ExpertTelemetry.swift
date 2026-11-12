import Foundation

public struct ExpertLayerRouting: Codable, Sendable, Equatable {
    public var layer: Int
    public var routed: Int
    public var distinct: Int
    public var top: [ExpertCount]
    public var topShare: Double

    public struct ExpertCount: Codable, Sendable, Equatable {
        public var expert: Int
        public var count: Int
    }

    enum CodingKeys: String, CodingKey {
        case layer, routed, distinct, top
        case topShare = "top_share"
    }
}

public struct ExpertRouting: Sendable, Equatable {
    public private(set) var counts: [Int: [Int: Int]] = [:]

    public init() {}

    public mutating func record(module: Int, _ experts: [Int]) {
        for expert in experts { counts[module, default: [:]][expert, default: 0] += 1 }
    }

    public func layers(top: Int = 8) -> [ExpertLayerRouting] {
        counts.keys.sorted().compactMap { layer in
            guard let experts = counts[layer] else { return nil }
            let routed = experts.values.reduce(0, +)
            let leaders = experts.sorted { ($0.value, -$0.key) > ($1.value, -$1.key) }.prefix(top)
            let share = routed > 0 ? Double(leaders.reduce(0) { $0 + $1.value }) / Double(routed) : 0
            return ExpertLayerRouting(layer: layer, routed: routed, distinct: experts.count, top: leaders.map { .init(expert: $0.key, count: $0.value) }, topShare: share)
        }
    }
}

public struct ExpertTelemetry: Codable, Sendable, Equatable {
    public var cache: ExpertCacheStats
    public var capacityBytes: Int64
    public var layers: [ExpertLayerRouting]

    public init(cache: ExpertCacheStats, capacityBytes: Int64, layers: [ExpertLayerRouting]) {
        (self.cache, self.capacityBytes, self.layers) = (cache, capacityBytes, layers)
    }

    enum CodingKeys: String, CodingKey {
        case cache, layers
        case capacityBytes = "capacity_bytes"
    }
}
