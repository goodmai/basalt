import Foundation

// MARK: — Telemetry & Observability (Epic 16 / Issue #30)

/// Real-time telemetry snapshot shared by `serve --stats`, `GET /api/v1/stats` (Monitor tab)
/// and MCP `status`. Every field has a producer on the serving path; metrics the engine cannot
/// measure yet (KV size, parked conversations, speculative acceptance) are left out rather than
/// reported as zeros.
public struct ServerTelemetrySnapshot: Codable, Sendable, Equatable {
    public let timestamp: Date
    public let tokensPerSecond: Double
    public let timeToFirstToken: Double
    public let totalRequests: Int
    public let totalTokensGenerated: Int
    public let activeRequests: Int
    /// MLX GPU memory after the last generation (weights + KV cache + activations).
    public let gpuActiveBytes: Int64
    public let gpuPeakBytes: Int64
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
        gpuActiveBytes: Int64 = 0,
        gpuPeakBytes: Int64 = 0,
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
        self.gpuActiveBytes = gpuActiveBytes
        self.gpuPeakBytes = gpuPeakBytes
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
        return "[stats] \(tps) tok/s | TTFT: \(ttft)s | RAM avail: \(ram)/\(totalRAMGB) GB | Model: \(modelRAMMB) MB | GPU active: \(formatBytes(gpuActiveBytes)) (peak \(formatBytes(gpuPeakBytes))) | Active: \(activeRequests) | Requests: \(totalRequests)"
    }
}

/// Actor holding the serving process's telemetry counters.
public actor TelemetryTracker {
    public static let shared = TelemetryTracker()

    private var lastTokensPerSecond: Double = 0.0
    private var lastTTFT: Double = 0.0
    private var totalRequests: Int = 0
    private var totalTokens: Int = 0
    private var activeRequests: Int = 0
    private var gpuActiveBytes: Int64 = 0
    private var gpuPeakBytes: Int64 = 0
    private var modelRAMMB: Int = 0
    /// Chip, GPU and total RAM never change for the process: probed once, not on every tick.
    private var hardware: SystemProfiler.SystemResources?

    public init() {}

    public func recordGeneration(_ response: GenerationResponse) {
        totalRequests += 1
        totalTokens += response.completionTokens
        lastTokensPerSecond = response.tokensPerSecond
        lastTTFT = response.timeToFirstToken
        gpuActiveBytes = Int64(response.memory.activeBytes)
        gpuPeakBytes = Int64(response.memory.peakBytes)
    }

    public func setModelMemory(mb: Int) {
        modelRAMMB = mb
    }

    public func requestStarted() { activeRequests += 1 }
    public func requestFinished() { activeRequests = max(0, activeRequests - 1) }

    public func snapshot(profiler: SystemProfiler = SystemProfiler()) async -> ServerTelemetrySnapshot {
        let hw: SystemProfiler.SystemResources
        if let cached = hardware {
            hw = cached
        } else {
            hw = await profiler.detectResources()
            hardware = hw
        }
        let availGB = Double(await profiler.currentAvailableRAM()) / 1_073_741_824.0

        return ServerTelemetrySnapshot(
            timestamp: Date(),
            tokensPerSecond: lastTokensPerSecond,
            timeToFirstToken: lastTTFT,
            totalRequests: totalRequests,
            totalTokensGenerated: totalTokens,
            activeRequests: activeRequests,
            gpuActiveBytes: gpuActiveBytes,
            gpuPeakBytes: gpuPeakBytes,
            totalRAMGB: hw.totalRAMGB,
            availableRAMGB: (availGB * 10).rounded() / 10,
            modelRAMMB: modelRAMMB,
            chipModel: hw.chipModel,
            gpuName: hw.gpuName
        )
    }
}
