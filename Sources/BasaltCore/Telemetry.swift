import Foundation

public struct ServerTelemetrySnapshot: Codable, Sendable, Equatable {
    public let timestamp: Date
    public let tokensPerSecond: Double
    public let timeToFirstToken: Double
    public let totalRequests: Int
    public let totalTokensGenerated: Int
    public let activeRequests: Int
    public let gpuActiveBytes: Int64
    public let gpuPeakBytes: Int64
    public let totalRAMGB: Int
    public let availableRAMGB: Double
    public let modelRAMMB: Int
    public let chipModel: String
    public let gpuName: String
    public var hybridAnchors: Int?
    public var hybridAnchorBytes: Int64?
    public var memory: MemoryTelemetry?
    public var routingBias: [String: [String: Double]]?
    public var draftTokens: Int?
    public var acceptedDraftTokens: Int?
    public var health: HealthTelemetry?
    public var activeDeltas: [OverlayInfo]?

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
            + (memory.map { " | \($0.formattedStats)" } ?? "")
            + (draftTokens.map { " | Drafts: \(acceptedDraftTokens ?? 0) of \($0) accepted" } ?? "")
    }
}

public struct MemoryTelemetry: Codable, Sendable, Equatable {
    public let pressure: MemoryPressure
    public let availableBytes: Int64
    public let step: MemoryDirective
    public let limits: MemoryLimits?
    public let refusedAdmissions: Int

    public init(state: MemoryState, limits: MemoryLimits?, refusedAdmissions: Int) {
        pressure = state.sample.pressure
        availableBytes = state.sample.availableBytes
        step = state.directive
        self.limits = limits
        self.refusedAdmissions = refusedAdmissions
    }

    var formattedStats: String {
        let applied = limits.map {
            ", cache limit \($0.cacheLimitBytes.map { formatBytes(Int64($0)) } ?? "MLX default"), prompt cache \(formatBytes(Int64($0.promptCacheBytes))) of \(formatBytes(Int64($0.promptCacheBudgetBytes))), evicted \($0.promptCacheEvictions)"
        } ?? ""
        return "Memory: \(pressure.rawValue) step \(step.rawValue)\(applied), refused \(refusedAdmissions)"
    }
}

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
    private var draftTokens = 0
    private var acceptedDraftTokens = 0
    private var hardware: SystemProfiler.SystemResources?
    private var model = ""
    private var health = HealthTelemetry()
    private let logger = BasaltLogger(module: "Telemetry")

    public init() {}

    private func recordGeneration(_ response: GenerationResponse) {
        totalRequests += 1
        totalTokens += response.completionTokens
        lastTokensPerSecond = response.tokensPerSecond
        lastTTFT = response.timeToFirstToken
        gpuActiveBytes = Int64(response.memory.activeBytes)
        gpuPeakBytes = Int64(response.memory.peakBytes)
        draftTokens += response.draftTokens ?? 0
        acceptedDraftTokens += response.acceptedDraftTokens ?? 0
        for warning in health.record(response, model: model) { logger.warn(warning) }
    }

    public func setModelMemory(mb: Int) {
        modelRAMMB = mb
    }

    public func setModel(_ name: String) {
        model = name
    }

    public func recordCanary(passed: Bool, at date: Date = Date()) {
        var canary = health.canary ?? CanaryTelemetry()
        canary.record(passed: passed, at: date)
        health.canary = canary
    }

    public func requestStarted() { activeRequests += 1 }

    public func requestFinished(with response: GenerationResponse? = nil) {
        activeRequests = max(0, activeRequests - 1)
        if let response { recordGeneration(response) }
    }

    public func snapshot(profiler: SystemProfiler = SystemProfiler()) async -> ServerTelemetrySnapshot {
        let hw: SystemProfiler.SystemResources
        if let cached = hardware {
            hw = cached
        } else {
            hw = await profiler.detectResources()
            hardware = hw
        }
        let availGB = Double(await profiler.currentAvailableRAM()) / 1_073_741_824.0

        var snapshot = ServerTelemetrySnapshot(
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
        if draftTokens > 0 {
            snapshot.draftTokens = draftTokens
            snapshot.acceptedDraftTokens = acceptedDraftTokens
        }
        snapshot.health = health
        return snapshot
    }
}
