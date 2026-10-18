import Foundation

// MARK: — Telemetry & Observability (Epic 16 / Issue #30)

/// Real-time telemetry metrics snapshot for CLI (--stats), SwiftUI Monitor tab, and MCP status
public struct ServerTelemetrySnapshot: Codable, Sendable, Equatable {
    public let timestamp: Date
    public let tokensPerSecond: Double
    public let timeToFirstToken: Double
    public let totalRequests: Int
    public let totalTokensGenerated: Int
    public let activeRequests: Int
    public let parkedConversations: Int
    public let kvCacheSizeBytes: Int64
    public let kvCacheSizeFormatted: String
    public let speculativeAcceptanceRate: Double?
    public let totalRAMGB: Int
    public let availableRAMGB: Double
    public let modelRAMMB: Int
    public let chipModel: String
    public let gpuName: String

    public init(
        timestamp: Date = Date(),
        tokensPerSecond: Double = 0.0,
        timeToFirstToken: Double = 0.0,
        totalRequests: Int = 0,
        totalTokensGenerated: Int = 0,
        activeRequests: Int = 0,
        parkedConversations: Int = 0,
        kvCacheSizeBytes: Int64 = 0,
        kvCacheSizeFormatted: String = "0 B",
        speculativeAcceptanceRate: Double? = nil,
        totalRAMGB: Int = 0,
        availableRAMGB: Double = 0.0,
        modelRAMMB: Int = 0,
        chipModel: String = "Apple Silicon",
        gpuName: String = "Metal GPU"
    ) {
        self.timestamp = timestamp
        self.tokensPerSecond = tokensPerSecond
        self.timeToFirstToken = timeToFirstToken
        self.totalRequests = totalRequests
        self.totalTokensGenerated = totalTokensGenerated
        self.activeRequests = activeRequests
        self.parkedConversations = parkedConversations
        self.kvCacheSizeBytes = kvCacheSizeBytes
        self.kvCacheSizeFormatted = kvCacheSizeFormatted
        self.speculativeAcceptanceRate = speculativeAcceptanceRate
        self.totalRAMGB = totalRAMGB
        self.availableRAMGB = availableRAMGB
        self.modelRAMMB = modelRAMMB
        self.chipModel = chipModel
        self.gpuName = gpuName
    }

    public var formattedStatsLine: String {
        let tps = String(format: "%.1f", tokensPerSecond)
        let ttft = String(format: "%.2f", timeToFirstToken)
        let ram = String(format: "%.1f", availableRAMGB)
        let specStr: String
        if let spec = speculativeAcceptanceRate {
            specStr = String(format: " | Speculative: %.1f%%", spec * 100)
        } else {
            specStr = ""
        }
        return "[stats] \(tps) tok/s | TTFT: \(ttft)s | RAM avail: \(ram)/\(totalRAMGB) GB | Model: \(modelRAMMB) MB | KV: \(kvCacheSizeFormatted) | Parked: \(parkedConversations)\(specStr)"
    }
}

/// Actor managing thread-safe telemetry metrics for the inference server
public actor TelemetryTracker {
    public static let shared = TelemetryTracker()

    private var lastTokensPerSecond: Double = 0.0
    private var lastTTFT: Double = 0.0
    private var totalRequests: Int = 0
    private var totalTokens: Int = 0
    private var activeRequests: Int = 0
    private var parkedConversations: Int = 0
    private var kvCacheSizeBytes: Int64 = 0
    private var speculativeAccepted: Int = 0
    private var speculativeDrafted: Int = 0
    private var modelRAMMB: Int = 0

    public init() {}

    public func recordGeneration(tokens: Int, tokensPerSecond: Double, timeToFirstToken: Double) {
        totalRequests += 1
        totalTokens += tokens
        self.lastTokensPerSecond = tokensPerSecond
        self.lastTTFT = timeToFirstToken
    }

    public func setModelMemory(mb: Int) {
        self.modelRAMMB = mb
    }

    public func setActiveRequests(_ count: Int) {
        self.activeRequests = max(0, count)
    }

    public func setParkedConversations(_ count: Int) {
        self.parkedConversations = max(0, count)
    }

    public func setKVCacheSize(bytes: Int64) {
        self.kvCacheSizeBytes = max(0, bytes)
    }

    public func recordSpeculative(drafted: Int, accepted: Int) {
        self.speculativeDrafted += drafted
        self.speculativeAccepted += accepted
    }

    public func snapshot(profiler: SystemProfiler? = nil) async -> ServerTelemetrySnapshot {
        let p = profiler ?? SystemProfiler()
        let res = await p.detectResources()
        let availGB = Double(res.availableRAM) / 1_073_741_824.0

        let acceptanceRate: Double?
        if speculativeDrafted > 0 {
            acceptanceRate = Double(speculativeAccepted) / Double(speculativeDrafted)
        } else {
            acceptanceRate = nil
        }

        return ServerTelemetrySnapshot(
            timestamp: Date(),
            tokensPerSecond: lastTokensPerSecond,
            timeToFirstToken: lastTTFT,
            totalRequests: totalRequests,
            totalTokensGenerated: totalTokens,
            activeRequests: activeRequests,
            parkedConversations: parkedConversations,
            kvCacheSizeBytes: kvCacheSizeBytes,
            kvCacheSizeFormatted: Self.formatBytes(kvCacheSizeBytes),
            speculativeAcceptanceRate: acceptanceRate,
            totalRAMGB: res.totalRAMGB,
            availableRAMGB: (availGB * 10).rounded() / 10,
            modelRAMMB: modelRAMMB,
            chipModel: res.chipModel,
            gpuName: res.gpuName
        )
    }

    public static func formatBytes(_ bytes: Int64) -> String {
        switch bytes {
        case 0..<1024:
            return "\(bytes) B"
        case 0..<1_048_576:
            return String(format: "%.1f KB", Double(bytes) / 1024.0)
        case 0..<1_073_741_824:
            return String(format: "%.1f MB", Double(bytes) / 1_048_576.0)
        default:
            return String(format: "%.2f GB", Double(bytes) / 1_073_741_824.0)
        }
    }
}
