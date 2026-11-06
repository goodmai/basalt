import Foundation

public struct ExpertStreaming: Sendable, Equatable {
    public var cacheBytes: ByteCount?

    public init(cacheBytes: ByteCount? = nil) {
        self.cacheBytes = cacheBytes
    }

    public func cache(forExperts experts: ByteCount) -> ByteCount {
        cacheBytes ?? ByteCount(Int64(Double(experts.rawValue) * ExpertStreamingFootprint.expertCacheShare))
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

    public var summary: String {
        "resident weights \(formatRAMBytes(residentWeights.rawValue)) + expert cache \(formatRAMBytes(expertCache.rawValue)) + KV \(formatRAMBytes(kv.rawValue)) = \(formatRAMBytes(required.rawValue))"
    }
}
