import Foundation

public struct ServerTelemetrySnapshot: Codable, Sendable, Equatable {
    public var timestamp = Date()
    public var tokensPerSecond = 0.0
    public var timeToFirstToken = 0.0
    public var totalRequests = 0
    public var totalTokensGenerated = 0
    public var activeRequests = 0
    public var gpuActiveBytes: Int64 = 0
    public var gpuPeakBytes: Int64 = 0
    public var totalRAMGB = 0
    public var availableRAMGB = 0.0
    public var modelRAMMB = 0
    public var chipModel = "Apple Silicon"
    public var gpuName = "Metal GPU"
    public var gpuArchitecture: String?
    public var gpuCores: Int?
    public var neuralAccelerators: Bool?
    public var hybridAnchors: Int?
    public var hybridAnchorBytes: Int64?
    public var memory: MemoryTelemetry?
    public var thermal: ThermalTelemetry?
    public var routingBias: [String: [String: Double]]?
    public var cascadeDeferrals: Int?
    public var draftTokens: Int?
    public var acceptedDraftTokens: Int?
    public var health: HealthTelemetry?
    public var activeDeltas: [OverlayInfo]?
    public var windows: [TelemetryWindow]?
    public var decisions: [String: DecisionTelemetry]?
    public var experts: ExpertTelemetry?
    public var confidence: ConfidenceTelemetry?
    public var cache: CacheTelemetry?
    public var calibration: DecisionCalibrationTelemetry?
    public var semanticCache: SemanticCacheMetrics?
    public var tieredKV: TieredKVMetrics?
    public var servingSLO: ServingSLOSnapshot?
    public var stateHealth: RecurrentStateHealthSnapshot?
    public var reasoning: ReasoningTelemetrySnapshot?

    public init() {}

    public var formattedStatsLine: String {
        let tps = String(format: "%.1f", tokensPerSecond)
        let ttft = String(format: "%.2f", timeToFirstToken)
        let ram = String(format: "%.1f", availableRAMGB)
        return "[stats] \(tps) tok/s | TTFT: \(ttft)s | RAM avail: \(ram)/\(totalRAMGB) GB | Model: \(modelRAMMB) MB | GPU active: \(formatBytes(gpuActiveBytes)) (peak \(formatBytes(gpuPeakBytes))) | Active: \(activeRequests) | Requests: \(totalRequests)"
            + (memory.map { " | \($0.formattedStats)" } ?? "")
            + (thermal.map { " | \($0.formattedStats)" } ?? "")
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

    private var core = TelemetryCore()
    private var sinks: [any TelemetrySink]
    private let clock: @Sendable () -> Date
    private var hardware: SystemProfiler.SystemResources?
    private let logger = BasaltLogger(module: "Telemetry")

    public init(sinks: [any TelemetrySink] = [], clock: @escaping @Sendable () -> Date = { Date() }) {
        self.sinks = sinks
        self.clock = clock
    }

    public var coreState: TelemetryCore { core }

    public func addSink(_ sink: any TelemetrySink) {
        sinks.append(sink)
    }

    public func installDefaultSinkIfEmpty(_ sink: any TelemetrySink) {
        if sinks.isEmpty { sinks.append(sink) }
    }

    public func sampleConfidence(every requests: Int) {
        core.sampleConfidence(every: requests)
    }

    public func sampled(_ request: GenerationRequest) -> GenerationRequest {
        core.sampled(request)
    }

    public func recordDecision(_ response: SystemOneResponse, latencyMs: Double) {
        core.recordDecision(response, latencyMs: latencyMs)
        for sink in sinks { sink.recordDecision(response, latencyMs: latencyMs) }
    }

    public func setModelMemory(mb: Int) {
        core.modelRAMMB = mb
    }

    public func setModel(_ name: String) {
        core.model = name
    }

    public func recordCanary(passed: Bool, at date: Date = Date()) {
        core.recordCanary(passed: passed, at: date)
    }

    public func requestStarted() {
        let active = core.requestStarted()
        for sink in sinks { sink.recordActiveRequests(active) }
    }

    public func requestFinished(with response: GenerationResponse? = nil, queueWaitSeconds: Double = 0) {
        let warnings = core.requestFinished(with: response, queueWaitSeconds: queueWaitSeconds, at: clock())
        for warning in warnings { logger.warn(warning) }
        let active = core.activeRequests
        let currentModel = core.model
        let currentCache = core.cache
        for sink in sinks {
            sink.recordActiveRequests(active)
            if let response {
                sink.recordGeneration(response, model: currentModel)
                if response.cacheTelemetry != nil { sink.recordCache(currentCache) }
            }
        }
    }

    public func recordRemote(_ response: GenerationResponse) {
        guard response.finishReason != .error else { return }
        core.recordRemote(response, at: clock())
        let currentModel = core.model
        for sink in sinks { sink.recordGeneration(response, model: currentModel) }
    }

    public func recordCache(_ delta: CacheTelemetry) {
        core.recordCache(delta)
        let current = core.cache
        for sink in sinks { sink.recordCache(current) }
    }

    public func recordCalibration(confidence: Double, agreed: Bool) {
        core.recordCalibration(confidence: confidence, agreed: agreed)
        let current = core.calibration
        for sink in sinks { sink.recordCalibration(confidence: confidence, agreed: agreed, telemetry: current) }
    }

    public func recordIntervention(_ event: InterventionTelemetryEvent) {
        core.recordIntervention(event)
        for sink in sinks { sink.recordIntervention(event) }
    }

    public func recordStateHealth(_ health: RecurrentStateHealthSnapshot) {
        core.recordStateHealth(health)
    }

    public func recordReasoning(_ turn: ReasoningTurnRecord) {
        core.recordReasoning(turn)
    }

    public func recordStage(_ span: ComposeStageSpan) {
        for sink in sinks { sink.recordStage(span) }
    }

    public func recordPipeline(
        _ receipt: PipelineReceipt, questions: [String: SystemOneQuestion] = [:], answers: [String: PipelineAnswer] = [:]
    ) {
        for stage in receipt.stages {
            recordStage(ComposeStageSpan(
                stage: stage.stage, model: stage.model, latencyMs: Double(stage.ms), tokens: stage.tokens, escalated: stage.escalated))
        }
        for (id, answer) in answers where answer.answeredBy == .answer {
            guard let question = questions[id], let decision = answer.decision, let confidence = decision.confidence,
                  let predicted = PipelineCore.label(of: decision, question), let verified = answer.label
            else { continue }
            recordCalibration(confidence: confidence, agreed: predicted == verified)
        }
    }

    public func snapshot(profiler: SystemProfiler = SystemProfiler()) async -> ServerTelemetrySnapshot {
        let hw: SystemProfiler.SystemResources
        if let cached = hardware {
            hw = cached
        } else {
            let detected = await profiler.detectResources()
            hardware = hardware ?? detected
            hw = hardware ?? detected
        }
        let availableBytes = await profiler.currentAvailableRAM()
        return core.snapshot(now: clock(), hardware: hw, availableRAMBytes: availableBytes)
    }
}
