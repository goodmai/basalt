import Foundation

public struct CacheTelemetry: Sendable, Codable, Equatable {
    public var hits: Int
    public var misses: Int
    public var evictions: Int
    public var cachedEntries: Int
    public var cachedTokens: Int
    public var cachedBytes: Int64
    public var tiered: TieredKVMetrics?

    public init(
        hits: Int = 0, misses: Int = 0, evictions: Int = 0,
        cachedEntries: Int = 0, cachedTokens: Int = 0, cachedBytes: Int64 = 0
    ) {
        self.hits = max(0, hits)
        self.misses = max(0, misses)
        self.evictions = max(0, evictions)
        self.cachedEntries = max(0, cachedEntries)
        self.cachedTokens = max(0, cachedTokens)
        self.cachedBytes = max(0, cachedBytes)
        self.tiered = nil
    }

    public func withTiered(_ tiered: TieredKVMetrics?) -> CacheTelemetry {
        var copy = self
        copy.tiered = tiered
        return copy
    }

    public var hitRatio: Double? {
        let total = hits + misses
        return total > 0 ? Double(hits) / Double(total) : nil
    }

    public var isEmpty: Bool {
        hits == 0 && misses == 0 && evictions == 0 && cachedEntries == 0 && cachedTokens == 0 && cachedBytes == 0
    }

    public mutating func record(_ delta: CacheTelemetry) {
        hits += max(0, delta.hits)
        misses += max(0, delta.misses)
        evictions += max(0, delta.evictions)
        cachedEntries = max(0, delta.cachedEntries)
        cachedTokens = max(0, delta.cachedTokens)
        cachedBytes = max(0, delta.cachedBytes)
        if let tiered = delta.tiered { self.tiered = tiered }
    }

    enum CodingKeys: String, CodingKey {
        case hits, misses, evictions, tiered
        case cachedEntries = "cached_entries", cachedTokens = "cached_tokens", cachedBytes = "cached_bytes"
    }
}

public struct CalibrationBin: Sendable, Codable, Equatable {
    public var samples: Int
    public var confidenceSum: Double
    public var agreementSum: Int

    public init(samples: Int = 0, confidenceSum: Double = 0, agreementSum: Int = 0) {
        self.samples = max(0, samples)
        self.confidenceSum = confidenceSum
        self.agreementSum = max(0, agreementSum)
    }

    public var meanConfidence: Double { samples > 0 ? confidenceSum / Double(samples) : 0 }
    public var accuracy: Double { samples > 0 ? Double(agreementSum) / Double(samples) : 0 }

    enum CodingKeys: String, CodingKey {
        case samples
        case confidenceSum = "confidence_sum", agreementSum = "agreement_sum"
    }
}

public struct DecisionCalibrationTelemetry: Sendable, Codable, Equatable {
    public static let binCount = 10
    public private(set) var samples = 0
    public private(set) var agreements = 0
    public private(set) var ece = 0.0
    public private(set) var bins = [CalibrationBin](repeating: CalibrationBin(), count: DecisionCalibrationTelemetry.binCount)

    public init() {}

    public var expectedCalibrationError: Double { ece }
    public var agreementRate: Double? { samples > 0 ? Double(agreements) / Double(samples) : nil }

    public mutating func record(confidence: Double, agreed: Bool) {
        guard confidence.isFinite else { return }
        let clamped = min(max(confidence, 0.0), 1.0)
        let index = min(bins.count - 1, max(0, Int(clamped * Double(bins.count))))
        samples += 1
        if agreed { agreements += 1 }
        bins[index].samples += 1
        bins[index].confidenceSum += clamped
        if agreed { bins[index].agreementSum += 1 }
        ece = Self.computeECE(bins: bins, totalSamples: samples)
    }

    static func computeECE(bins: [CalibrationBin], totalSamples: Int) -> Double {
        guard totalSamples > 0 else { return 0 }
        let total = Double(totalSamples)
        return bins.reduce(0.0) { acc, bin in
            guard bin.samples > 0 else { return acc }
            return acc + (Double(bin.samples) / total) * abs(bin.accuracy - bin.meanConfidence)
        }
    }
}

public struct InterventionTelemetryEvent: Sendable, Codable, Equatable {
    public let kind: String
    public let target: String
    public let activeOverlays: Int

    public init(kind: String, target: String = "", activeOverlays: Int = 0) {
        (self.kind, self.target, self.activeOverlays) = (kind, target, max(0, activeOverlays))
    }
}

public struct ComposeStageSpan: Sendable, Codable, Equatable {
    public let stage: PipelineStage
    public let model: String
    public let latencyMs: Double
    public let tokens: Int
    public let escalated: Bool

    public init(stage: PipelineStage, model: String, latencyMs: Double, tokens: Int = 0, escalated: Bool = false) {
        (self.stage, self.model, self.latencyMs, self.tokens, self.escalated) = (stage, model, latencyMs, tokens, escalated)
    }
}

public protocol TelemetrySink: Sendable {
    func recordActiveRequests(_ active: Int)
    func recordGeneration(_ response: GenerationResponse, model: String)
    func recordDecision(_ response: SystemOneResponse, latencyMs: Double)
    func recordCache(_ cache: CacheTelemetry)
    func recordCalibration(confidence: Double, agreed: Bool, telemetry: DecisionCalibrationTelemetry)
    func recordIntervention(_ event: InterventionTelemetryEvent)
    func recordStage(_ span: ComposeStageSpan)
}

extension TelemetrySink {
    public func recordActiveRequests(_ active: Int) {}
    public func recordGeneration(_ response: GenerationResponse, model: String) {}
    public func recordDecision(_ response: SystemOneResponse, latencyMs: Double) {}
    public func recordCache(_ cache: CacheTelemetry) {}
    public func recordCalibration(confidence: Double, agreed: Bool, telemetry: DecisionCalibrationTelemetry) {}
    public func recordIntervention(_ event: InterventionTelemetryEvent) {}
    public func recordStage(_ span: ComposeStageSpan) {}
}

public struct TelemetryCore: Sendable, Equatable {
    public private(set) var lastTokensPerSecond = 0.0
    public private(set) var lastTTFT = 0.0
    public private(set) var totalRequests = 0
    public private(set) var totalTokens = 0
    public private(set) var activeRequests = 0
    public private(set) var gpuActiveBytes: Int64 = 0
    public private(set) var gpuPeakBytes: Int64 = 0
    public var modelRAMMB = 0
    public private(set) var draftTokens = 0
    public private(set) var acceptedDraftTokens = 0
    public var model = ""
    public private(set) var health = HealthTelemetry()
    public private(set) var windows = TelemetryWindows()
    public private(set) var decisions: [String: DecisionTelemetry] = [:]
    public private(set) var confidence = ConfidenceTelemetry()
    public private(set) var cache = CacheTelemetry()
    public private(set) var calibration = DecisionCalibrationTelemetry()
    public private(set) var interventions: [String: Int] = [:]
    public private(set) var servingSLO = ServingSLOReducer()
    public private(set) var stateHealth: RecurrentStateHealthSnapshot?
    public private(set) var reasoning = ThinkingTelemetryReducer()
    private var confidenceEvery = 0
    private var confidenceCounter = 0

    public init() {}

    @discardableResult
    public mutating func requestStarted() -> Int {
        activeRequests += 1
        return activeRequests
    }

    @discardableResult
    public mutating func requestFinished(
        with response: GenerationResponse? = nil, queueWaitSeconds: Double = 0, at date: Date = Date()
    ) -> [String] {
        activeRequests = max(0, activeRequests - 1)
        guard let response else { return [] }
        return recordGeneration(response, queueWaitSeconds: queueWaitSeconds, at: date)
    }

    @discardableResult
    public mutating func recordGeneration(
        _ response: GenerationResponse, queueWaitSeconds: Double = 0, at date: Date = Date()
    ) -> [String] {
        totalRequests += 1
        totalTokens += response.completionTokens
        lastTokensPerSecond = response.tokensPerSecond
        lastTTFT = response.timeToFirstToken
        gpuActiveBytes = Int64(response.memory.activeBytes)
        gpuPeakBytes = Int64(response.memory.peakBytes)
        draftTokens += response.draftTokens ?? 0
        acceptedDraftTokens += response.acceptedDraftTokens ?? 0
        let warnings = health.record(response, model: model)
        var sample = TelemetrySample(
            at: date, ttft: response.timeToFirstToken, tokensPerSecond: response.tokensPerSecond,
            completionTokens: response.completionTokens, promptTokens: response.promptTokens)
        (sample.cachedPromptTokens, sample.draftTokens, sample.acceptedDraftTokens) = (
            response.cachedPromptTokens, response.draftTokens ?? 0, response.acceptedDraftTokens ?? 0)
        windows.record(sample)
        if let measured = response.confidence { confidence.record(measured) }
        if let cacheDelta = response.cacheTelemetry { cache.record(cacheDelta) }
        servingSLO.record(sloRecord(for: response, queueWaitSeconds: queueWaitSeconds))
        recordReasoningIfPresent(from: response)
        return warnings
    }

    public mutating func recordRemote(_ response: GenerationResponse, at date: Date = Date()) {
        guard response.finishReason != .error else { return }
        totalRequests += 1
        totalTokens += response.completionTokens
        windows.record(TelemetrySample(
            at: date, ttft: response.timeToFirstToken, tokensPerSecond: response.tokensPerSecond,
            completionTokens: response.completionTokens, promptTokens: response.promptTokens))
    }

    public mutating func sampleConfidence(every requests: Int) {
        confidenceEvery = max(0, requests)
    }

    public mutating func sampled(_ request: GenerationRequest) -> GenerationRequest {
        guard confidenceEvery > 0, !request.recordConfidence else { return request }
        confidenceCounter += 1
        return confidenceCounter % confidenceEvery == 0 ? request.withConfidenceRecording() : request
    }

    public static let maxDecisionModels = 64

    public mutating func recordDecision(_ response: SystemOneResponse, latencyMs: Double) {
        let bucket = (decisions[response.model] != nil || decisions.count < Self.maxDecisionModels)
            ? response.model
            : "_other"
        decisions[bucket, default: DecisionTelemetry()].record(response, latencyMs: latencyMs)
    }

    public mutating func recordCanary(passed: Bool, at date: Date = Date()) {
        var canary = health.canary ?? CanaryTelemetry()
        canary.record(passed: passed, at: date)
        health.canary = canary
    }

    public mutating func recordCache(_ delta: CacheTelemetry) {
        cache.record(delta)
    }

    public mutating func recordCalibration(confidence value: Double, agreed: Bool) {
        calibration.record(confidence: value, agreed: agreed)
    }

    public mutating func recordIntervention(_ event: InterventionTelemetryEvent) {
        interventions[event.kind, default: 0] += 1
    }

    public func snapshot(
        now: Date = Date(), hardware: SystemProfiler.SystemResources? = nil, availableRAMBytes: Int64 = 0
    ) -> ServerTelemetrySnapshot {
        var snapshot = ServerTelemetrySnapshot()
        snapshot.timestamp = now
        snapshot.tokensPerSecond = lastTokensPerSecond
        snapshot.timeToFirstToken = lastTTFT
        snapshot.totalRequests = totalRequests
        snapshot.totalTokensGenerated = totalTokens
        snapshot.activeRequests = activeRequests
        snapshot.gpuActiveBytes = gpuActiveBytes
        snapshot.gpuPeakBytes = gpuPeakBytes
        if let hardware {
            snapshot.totalRAMGB = hardware.totalRAMGB
            snapshot.chipModel = hardware.chipModel
            snapshot.gpuName = hardware.gpuName
            snapshot.gpuArchitecture = hardware.gpuArchitecture
            snapshot.gpuCores = hardware.gpuCores
            snapshot.neuralAccelerators = hardware.neuralAccelerators
        }
        let availGB = Double(availableRAMBytes) / 1_073_741_824.0
        snapshot.availableRAMGB = (availGB * 10).rounded() / 10
        snapshot.modelRAMMB = modelRAMMB
        if draftTokens > 0 {
            snapshot.draftTokens = draftTokens
            snapshot.acceptedDraftTokens = acceptedDraftTokens
        }
        snapshot.health = health
        snapshot.windows = windows.windows(now: now)
        snapshot.decisions = decisions.isEmpty ? nil : decisions
        snapshot.confidence = confidence.responses > 0 ? confidence : nil
        snapshot.cache = cache.isEmpty ? nil : cache
        snapshot.calibration = calibration.samples > 0 ? calibration : nil
        if !servingSLO.isEmpty {
            let peak = AppleSiliconBandwidthTable.peakBandwidthGBps(
                for: hardware?.chipModel ?? snapshot.chipModel,
                gpuCores: hardware?.gpuCores)
            snapshot.servingSLO = servingSLO.snapshot(peakBandwidthGBps: peak)
        }
        snapshot.stateHealth = stateHealth
        snapshot.reasoning = reasoning.isEmpty ? nil : reasoning.snapshot()
        return snapshot
    }
}

extension TelemetryCore {
    public mutating func recordStateHealth(_ health: RecurrentStateHealthSnapshot) {
        stateHealth = health
    }

    public mutating func recordReasoning(_ turn: ReasoningTurnRecord) {
        reasoning.record(turn)
    }

    mutating func recordReasoningIfPresent(from response: GenerationResponse) {
        if let turn = response.reasoningTurn {
            reasoning.record(turn)
            return
        }
        guard let reasoningTokens = response.reasoningTokens, reasoningTokens > 0 else { return }
        let answerTokens = max(0, response.completionTokens - reasoningTokens)
        let truncated = response.finishReason == .length && answerTokens == 0
        reasoning.record(ReasoningTurnRecord(
            thinkingTokens: reasoningTokens,
            answerTokens: answerTokens,
            hitTokenBudgetBeforeClose: truncated
        ))
    }

    func sloRecord(for response: GenerationResponse, queueWaitSeconds: Double) -> ServingSLORecord {
        let prefillTokens = max(0, response.promptTokens - response.cachedPromptTokens)
        let prefillSeconds = response.prefillSeconds ?? response.timeToFirstToken
        let elapsedDecode = response.generationTime - prefillSeconds
        let decodeSeconds = elapsedDecode > 0
            ? elapsedDecode
            : (response.tokensPerSecond > 0 ? Double(response.completionTokens) / response.tokensPerSecond : 0)
        let activeWeightBytes: UInt64 = modelRAMMB > 0
            ? UInt64(modelRAMMB) * 1_048_576
            : UInt64(max(0, response.memory.activeBytes))
        return ServingSLORecord(
            queueWaitSeconds: queueWaitSeconds,
            ttftSeconds: response.timeToFirstToken,
            prefillTokens: prefillTokens,
            prefillSeconds: prefillSeconds,
            decodeTokens: response.completionTokens,
            decodeSeconds: decodeSeconds
        ).withDecodeDetail(
            itlSamplesSeconds: response.itlSamplesSeconds ?? [],
            activeWeightBytes: activeWeightBytes)
    }
}

extension GenerationResponse {
    package func withCacheTelemetry(_ cache: CacheTelemetry?) -> GenerationResponse {
        var copy = self
        copy.cacheTelemetry = cache
        return copy
    }

    package func withServingTiming(prefillSeconds: Double?, itlSamplesSeconds: [Double]?) -> GenerationResponse {
        var copy = self
        copy.prefillSeconds = prefillSeconds
        copy.itlSamplesSeconds = itlSamplesSeconds
        return copy
    }

    package func withReasoningTurn(_ turn: ReasoningTurnRecord?) -> GenerationResponse {
        var copy = self
        copy.reasoningTurn = turn
        return copy
    }
}
