import Foundation

public struct ExpertStreaming: Sendable, Equatable {
    public var cacheBytes: ByteCount?

    public init(cacheBytes: ByteCount? = nil) {
        self.cacheBytes = cacheBytes
    }

    public static func isDisabledByEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        if let raw = environment["BASALT_NO_EXPERT_STREAMING"]?.lowercased(), ["1", "true", "yes"].contains(raw) {
            return true
        }
        if let raw = environment["BASALT_EXPERT_STREAMING"]?.lowercased(), ["0", "false", "no"].contains(raw) {
            return true
        }
        return false
    }

    public func cache(forExperts experts: ByteCount) -> ByteCount {
        cacheBytes ?? ByteCount(Int64(Double(experts.rawValue) * ExpertStreamingFootprint.expertCacheShare))
    }

    public func arenaPlan(workload: ExpertArenaWorkload, host: ExpertArenaHost) -> ExpertArenaPlan {
        let overrideGB = cacheBytes.map { Double($0.rawValue) / Double(ExpertArenaPlanner.gib) }
        return ExpertArenaPlanner.plan(
            ExpertArenaInput(host: host, workload: workload, explicitOverrideGB: overrideGB)
        )
    }

    public func residentBytes(weights: ByteCount, modelId: String) -> ByteCount {
        let total = ModelDefinition.inferTotalParameters(id: modelId, name: "", ramMB: weights.rawValue >> 20)
        let active = ModelDefinition.inferActiveParameters(id: modelId, name: "", total: total)
        let footprint = ExpertStreamingFootprint(weights: weights, totalParameters: total, activeParameters: active, kv: .zero, streaming: self)
        return footprint.map { $0.residentWeights + $0.expertCache } ?? weights
    }
}

public struct ExpertStreamingFootprint: Sendable, Codable, Equatable {
    public static let expertCacheShare = 0.25

    public let residentWeights: ByteCount
    public let expertCache: ByteCount
    public let kv: ByteCount

    public var required: ByteCount { residentWeights + expertCache + kv }

    public init?(
        weights: ByteCount, totalParameters: Double, activeParameters: Double, kv: ByteCount, streaming: ExpertStreaming = ExpertStreaming()
    ) {
        guard activeParameters > 0, activeParameters < totalParameters else { return nil }
        let resident = ByteCount(Int64(Double(weights.rawValue) * activeParameters / totalParameters))
        residentWeights = resident
        expertCache = streaming.cache(forExperts: weights - resident)
        self.kv = kv
    }

    public init?(
        residentWeights: ByteCount, expertBytes: ByteCount, kv: ByteCount, streaming: ExpertStreaming = ExpertStreaming()
    ) {
        guard expertBytes > .zero else { return nil }
        self.residentWeights = max(.zero, residentWeights)
        expertCache = streaming.cache(forExperts: expertBytes)
        self.kv = kv
    }

    public init?(
        residentWeights: ByteCount,
        expertBytes: ByteCount,
        kv: ByteCount,
        host: ExpertArenaHost,
        streaming: ExpertStreaming = ExpertStreaming()
    ) {
        guard expertBytes > .zero else { return nil }
        let resident = max(.zero, residentWeights)
        let workload = ExpertArenaWorkload(
            denseWeightsBytes: resident,
            kvCacheBytes: max(.zero, kv),
            expertBytesTotal: expertBytes
        )
        let plan = streaming.arenaPlan(workload: workload, host: host)
        self.residentWeights = resident
        self.expertCache = plan.arenaBytes
        self.kv = kv
    }

    public var summary: String {
        "resident weights \(formatRAMBytes(residentWeights.rawValue)) + expert cache \(formatRAMBytes(expertCache.rawValue)) + KV \(formatRAMBytes(kv.rawValue)) = \(formatRAMBytes(required.rawValue))"
    }
}
