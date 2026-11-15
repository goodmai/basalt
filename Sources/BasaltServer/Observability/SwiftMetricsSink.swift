import BasaltCore
import CoreMetrics
import Foundation

public struct SwiftMetricsSink: TelemetrySink, Sendable {
    private let factory: any MetricsFactory
    private let requestsTotal: Counter
    private let generatedTokensTotal: Counter
    private let activeRequests: Meter
    private let gpuActiveBytes: Gauge
    private let gpuPeakBytes: Gauge
    private let ttftTimer: CoreMetrics::Timer
    private let generationTimer: CoreMetrics::Timer
    private let tokensPerSecond: Recorder
    private let cacheHits: Gauge
    private let cacheMisses: Gauge
    private let cacheEvictions: Gauge
    private let cacheEntries: Gauge
    private let cacheTokens: Gauge
    private let cacheBytes: Gauge
    private let calibrationSamplesTotal: Counter
    private let calibrationAgreementTotal: Counter
    private let calibrationECE: Gauge

    public init(factory: any MetricsFactory = MetricsSystem.factory) {
        self.factory = factory
        requestsTotal = Counter(label: "basalt_requests_total", factory: factory)
        generatedTokensTotal = Counter(label: "basalt_generated_tokens_total", factory: factory)
        activeRequests = Meter(label: "basalt_active_requests", factory: factory)
        gpuActiveBytes = Gauge(label: "basalt_gpu_active_bytes", factory: factory)
        gpuPeakBytes = Gauge(label: "basalt_gpu_peak_bytes", factory: factory)
        ttftTimer = CoreMetrics::Timer(label: "basalt_ttft", preferredDisplayUnit: .seconds, factory: factory)
        generationTimer = CoreMetrics::Timer(label: "basalt_generation_duration", preferredDisplayUnit: .seconds, factory: factory)
        tokensPerSecond = Recorder(label: "basalt_tokens_per_second", aggregate: true, factory: factory)
        cacheHits = Gauge(label: "basalt_prompt_cache_hits", factory: factory)
        cacheMisses = Gauge(label: "basalt_prompt_cache_misses", factory: factory)
        cacheEvictions = Gauge(label: "basalt_prompt_cache_evictions", factory: factory)
        cacheEntries = Gauge(label: "basalt_prompt_cache_entries", factory: factory)
        cacheTokens = Gauge(label: "basalt_prompt_cache_tokens", factory: factory)
        cacheBytes = Gauge(label: "basalt_prompt_cache_bytes", factory: factory)
        calibrationSamplesTotal = Counter(label: "basalt_calibration_samples_total", factory: factory)
        calibrationAgreementTotal = Counter(label: "basalt_calibration_agreement_total", factory: factory)
        calibrationECE = Gauge(label: "basalt_calibration_ece", factory: factory)
    }

    public func recordActiveRequests(_ active: Int) {
        activeRequests.set(active)
    }

    public func recordGeneration(_ response: GenerationResponse, model: String) {
        requestsTotal.increment()
        if response.completionTokens > 0 {
            generatedTokensTotal.increment(by: response.completionTokens)
        }
        gpuActiveBytes.record(response.memory.activeBytes)
        gpuPeakBytes.record(response.memory.peakBytes)
        if response.timeToFirstToken.isFinite, response.timeToFirstToken >= 0 {
            ttftTimer.recordSeconds(response.timeToFirstToken)
        }
        if response.generationTime.isFinite, response.generationTime >= 0 {
            generationTimer.recordSeconds(response.generationTime)
        }
        if response.tokensPerSecond.isFinite {
            tokensPerSecond.record(response.tokensPerSecond)
        }
        if !model.isEmpty {
            Counter(label: "basalt_model_requests_total", dimensions: [("model", model)], factory: factory).increment()
        }
    }

    public func recordDecision(_ response: SystemOneResponse, latencyMs: Double) {
        let dims = [("model", response.model)]
        Counter(label: "basalt_decision_requests_total", dimensions: dims, factory: factory).increment()
        if response.escalate {
            Counter(label: "basalt_decision_escalations_total", dimensions: dims, factory: factory).increment()
        }
        if latencyMs.isFinite, latencyMs >= 0 {
            CoreMetrics::Timer(label: "basalt_decision_latency", dimensions: dims, preferredDisplayUnit: .milliseconds, factory: factory)
                .recordMilliseconds(latencyMs)
        }
    }

    public func recordCache(_ cache: CacheTelemetry) {
        cacheHits.record(cache.hits)
        cacheMisses.record(cache.misses)
        cacheEvictions.record(cache.evictions)
        cacheEntries.record(cache.cachedEntries)
        cacheTokens.record(cache.cachedTokens)
        cacheBytes.record(cache.cachedBytes)
    }

    public func recordCalibration(confidence: Double, agreed: Bool, telemetry: DecisionCalibrationTelemetry) {
        calibrationSamplesTotal.increment()
        if agreed { calibrationAgreementTotal.increment() }
        if telemetry.ece.isFinite { calibrationECE.record(telemetry.ece) }
    }

    public func recordIntervention(_ event: InterventionTelemetryEvent) {
        Counter(label: "basalt_interventions_total", dimensions: [("kind", event.kind)], factory: factory).increment()
        Gauge(label: "basalt_active_overlays", factory: factory).record(event.activeOverlays)
    }

    public func recordStage(_ span: ComposeStageSpan) {
        let dims = [("stage", span.stage.rawValue), ("model", span.model)]
        Counter(label: "basalt_compose_stage_runs_total", dimensions: dims, factory: factory).increment()
        if span.latencyMs.isFinite, span.latencyMs >= 0 {
            CoreMetrics::Timer(label: "basalt_compose_stage_duration", dimensions: dims, preferredDisplayUnit: .milliseconds, factory: factory)
                .recordMilliseconds(span.latencyMs)
        }
        if span.tokens > 0 {
            Counter(label: "basalt_compose_stage_tokens_total", dimensions: dims, factory: factory).increment(by: span.tokens)
        }
        if span.escalated {
            Counter(label: "basalt_compose_stage_escalations_total", dimensions: dims, factory: factory).increment()
        }
    }
}
