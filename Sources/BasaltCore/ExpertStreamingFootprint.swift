import Foundation

public struct ExpertStreamingFootprint: Sendable, Codable, Equatable {
    public static let expertCacheShare = 0.25

    public let residentWeights: ByteCount
    public let expertCache: ByteCount
    public let kv: ByteCount

    public var required: ByteCount { residentWeights + expertCache + kv }

    public init?(weights: ByteCount, totalParameters: Double, activeParameters: Double, kv: ByteCount) {
        guard activeParameters > 0, activeParameters < totalParameters else { return nil }
        let resident = ByteCount(Int64(Double(weights.rawValue) * activeParameters / totalParameters))
        residentWeights = resident
        expertCache = ByteCount(Int64(Double((weights - resident).rawValue) * Self.expertCacheShare))
        self.kv = kv
    }

    public var summary: String {
        "resident weights \(formatRAMBytes(residentWeights.rawValue)) + expert cache \(formatRAMBytes(expertCache.rawValue)) + KV \(formatRAMBytes(kv.rawValue)) = \(formatRAMBytes(required.rawValue))"
    }
}
