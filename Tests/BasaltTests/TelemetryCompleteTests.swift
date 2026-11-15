import CoreMetrics
import Foundation
import Hummingbird
import HummingbirdTesting
import MetricsTestKit
import MLX
import MLXLMCommon
import NIOCore
import Synchronization
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("X-ray telemetry phase 5: TelemetryCore pure reducer, calibration ECE bins, PromptCacheTree and FusedLogitConfidence (#450)", .serialized)
struct TelemetryCoreAndCalibrationTests {
    static let epoch = Date(timeIntervalSince1970: 1_800_000_000)

    private func sampleResponse(
        tokens: Int = 24, prompt: Int = 16, cached: Int = 8, tps: Double = 45.0, ttft: Double = 0.12,
        cache: CacheTelemetry? = nil
    ) -> GenerationResponse {
        GenerationResponse(
            generatedText: "ok", promptTokens: prompt, cachedPromptTokens: cached, completionTokens: tokens,
            tokensPerSecond: tps, generationTime: 0.6, timeToFirstToken: ttft,
            memory: .init(peakBytes: 2048, activeBytes: 1024, cacheBytes: 512), finishReason: .stop
        ).withCacheTelemetry(cache)
    }

    @Test("TelemetryCore reduces active requests, generations, window eviction, sampling, canary and interventions deterministically")
    func pureCoreDeterminism() {
        var core = TelemetryCore()
        core.model = "qwen3-4b"
        core.modelRAMMB = 2400
        #expect(core.requestStarted() == 1 && core.requestStarted() == 2)
        let firstAt = Self.epoch
        _ = core.requestFinished(
            with: sampleResponse(cache: CacheTelemetry(hits: 0, misses: 1, evictions: 0, cachedEntries: 1, cachedTokens: 16, cachedBytes: 4096)),
            at: firstAt)
        #expect(core.activeRequests == 1 && core.totalRequests == 1 && core.totalTokens == 24)
        _ = core.requestFinished(
            with: sampleResponse(cache: CacheTelemetry(hits: 1, misses: 0, evictions: 2, cachedEntries: 1, cachedTokens: 16, cachedBytes: 4096)),
            at: firstAt.addingTimeInterval(3700))
        #expect(core.activeRequests == 0 && core.totalRequests == 2 && core.windows.samples.count == 1)
        #expect(core.cache == CacheTelemetry(hits: 1, misses: 1, evictions: 2, cachedEntries: 1, cachedTokens: 16, cachedBytes: 4096))
        #expect(core.cache.hitRatio == 0.5)

        core.sampleConfidence(every: 2)
        let r1 = core.sampled(GenerationRequest(prompt: "a"))
        let r2 = core.sampled(GenerationRequest(prompt: "b"))
        #expect(!r1.recordConfidence && r2.recordConfidence)

        core.recordCanary(passed: true, at: firstAt)
        core.recordIntervention(InterventionTelemetryEvent(kind: "weights_delta", target: "layers.0", activeOverlays: 1))
        core.recordIntervention(InterventionTelemetryEvent(kind: "weights_delta", target: "layers.1", activeOverlays: 2))
        #expect(core.interventions["weights_delta"] == 2 && core.health.canary?.passed == 1)

        var clone = TelemetryCore()
        clone.model = "qwen3-4b"
        clone.modelRAMMB = 2400
        _ = clone.requestStarted()
        _ = clone.requestStarted()
        _ = clone.requestFinished(
            with: sampleResponse(cache: CacheTelemetry(hits: 0, misses: 1, evictions: 0, cachedEntries: 1, cachedTokens: 16, cachedBytes: 4096)),
            at: firstAt)
        _ = clone.requestFinished(
            with: sampleResponse(cache: CacheTelemetry(hits: 1, misses: 0, evictions: 2, cachedEntries: 1, cachedTokens: 16, cachedBytes: 4096)),
            at: firstAt.addingTimeInterval(3700))
        clone.sampleConfidence(every: 2)
        _ = clone.sampled(GenerationRequest(prompt: "a"))
        _ = clone.sampled(GenerationRequest(prompt: "b"))
        clone.recordCanary(passed: true, at: firstAt)
        clone.recordIntervention(InterventionTelemetryEvent(kind: "weights_delta", target: "layers.0", activeOverlays: 1))
        clone.recordIntervention(InterventionTelemetryEvent(kind: "weights_delta", target: "layers.1", activeOverlays: 2))
        #expect(core == clone)
    }

    @Test("DecisionCalibrationTelemetry bins confidence into 10 reliability deciles and computes exact Expected Calibration Error")
    func calibrationBinsAndECE() {
        var perfect = DecisionCalibrationTelemetry()
        perfect.record(confidence: 1.0, agreed: true)
        perfect.record(confidence: 0.0, agreed: false)
        #expect(perfect.samples == 2 && perfect.agreements == 1)
        #expect(perfect.bins[9].samples == 1 && perfect.bins[0].samples == 1)
        #expect(abs(perfect.expectedCalibrationError) < 1e-12)

        var skewed = DecisionCalibrationTelemetry()
        skewed.record(confidence: 0.8, agreed: true)
        skewed.record(confidence: 0.8, agreed: false)
        skewed.record(confidence: 0.2, agreed: false)
        skewed.record(confidence: 0.2, agreed: false)
        skewed.record(confidence: .nan, agreed: true)
        #expect(skewed.samples == 4 && skewed.agreements == 1 && skewed.agreementRate == 0.25)
        #expect(skewed.bins[8].samples == 2 && skewed.bins[2].samples == 2)
        let expectedECE = 0.5 * abs(0.5 - 0.8) + 0.5 * abs(0.0 - 0.2)
        #expect(abs(skewed.ece - expectedECE) < 1e-12 && skewed.expectedCalibrationError == skewed.ece)
    }

    @Test("TelemetryTracker clock injection and pipeline teacher verification update windows and calibration")
    func trackerClockAndPipelineCalibration() async {
        let now = Mutex(Self.epoch)
        let tracker = TelemetryTracker(clock: { now.withLock { $0 } })
        await tracker.requestStarted()
        await tracker.requestFinished(with: sampleResponse(tokens: 10, prompt: 20, cached: 10))
        now.withLock { $0 = Self.epoch.addingTimeInterval(120) }
        let snap = await tracker.snapshot()
        let windows = snap.windows ?? []
        #expect(windows.first { $0.seconds == 60 }?.requests == 0)
        #expect(windows.first { $0.seconds == 300 }?.requests == 1)

        var decision = SystemOneAnswer()
        (decision.noul, decision.confidence, decision.abstain) = (0.85, 0.85, true)
        let receipt = PipelineReceipt(
            stages: [
                PipelineStageReceipt(stage: .decide, model: "jev", ms: 12, tokens: 10, confidence: 0.85, escalated: true),
                PipelineStageReceipt(stage: .answer, model: "qwen3", ms: 40, tokens: 4),
            ],
            answeredBy: ["q1": .answer], escalated: true, system2Calls: 1, ms: 52)
        await tracker.recordPipeline(
            receipt,
            questions: ["q1": SystemOneQuestion(type: .noul, instructions: "Is it safe?")],
            answers: ["q1": PipelineAnswer(answeredBy: .answer, label: "true", decision: decision, text: "true")])
        let calibrated = await tracker.snapshot()
        #expect(calibrated.calibration?.samples == 1 && calibrated.calibration?.agreements == 1)
    }

    @Test("PromptCacheTree and MLXInferenceEngine emit hit, miss, eviction and byte telemetry without counting disk persistence")
    func promptCacheTreeAndEngineTelemetry() async throws {
        let fixture = PromptCacheTreeTests()
        var tree = PromptCacheTree(byteBudget: ByteCount(10 * PromptCacheTreeTests.bytesPerToken))
        let tokens = Array(0..<8)
        if tree.lease("ns", for: tokens + [99]) != nil { Issue.record("expected initial miss") }
        tree.checkedStore("ns", tokens: tokens, cache: fixture.cache(tokens))
        let nonCounted = tree.lease("ns", for: tokens + [99], countsAccess: false)
        _ = consume nonCounted
        let counted = tree.lease("ns", for: tokens + [99])
        _ = consume counted
        #expect(tree.hits == 1 && tree.misses == 1)
        tree.checkedRebudget(ByteCount(4 * PromptCacheTreeTests.bytesPerToken))
        let rawDelta = tree.takeTelemetryDelta()
        let delta = try #require(rawDelta)
        #expect(delta.hits == 1 && delta.misses == 1 && delta.evictions > 0)
        #expect(delta.cachedTokens == 4 && delta.cachedBytes == Int64(4 * PromptCacheTreeTests.bytesPerToken))
        let emptyDelta = tree.takeTelemetryDelta()
        #expect(emptyDelta == nil)

        let tracker = TelemetryTracker()
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 1 << 20), telemetry: tracker)
        await engine.attach(try TinyModel.container())
        _ = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns", maxTokens: 4))
        _ = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "ns", maxTokens: 4))
        let cache = try #require(await tracker.snapshot().cache)
        #expect(cache.hits == 1 && cache.misses == 1 && cache.cachedEntries >= 1 && cache.cachedTokens > 0 && cache.cachedBytes > 0)
    }

    @Test("FusedLogitConfidence.summarize matches per-step ConfidenceProbeRecorder across multi-row logits in one pass")
    func fusedSummarizeParity() throws {
        let logits = MLXArray([
            Float(4.0), 1.0, -1.0, 0.0,
            Float(0.5), 0.4, 0.3, 0.2,
            Float(6.0), -2.0, -3.0, -4.0,
        ], [3, 4])
        let probe = ConfidenceProbe()
        let recorder = ConfidenceProbeRecorder(probe: probe)
        for row in 0..<3 { _ = recorder.process(logits: logits[row]) }
        let expected = try #require(probe.summary)
        let actual = try #require(FusedLogitConfidence.summarize(logits))
        #expect(abs(actual.min - expected.min) < 1e-5 && abs(actual.mean - expected.mean) < 1e-5)
        #expect(abs((actual.meanEntropy ?? -1) - (expected.meanEntropy ?? -2)) < 1e-5)
        #expect(abs((actual.minMargin ?? -1) - (expected.minMargin ?? -2)) < 1e-5)
    }
}

@Suite("X-ray telemetry phase 5: SwiftMetricsSink fan-out, OpenMetrics cache/calibration export, /metrics parity and MCP telemetry_query (#450)")
struct TelemetryExportAndSinksTests {
    @Test("TelemetryTracker fans out generations, decisions, cache, calibration, interventions and compose stages to SwiftMetricsSink")
    func swiftMetricsSinkFanOut() async throws {
        let primary = TestMetrics()
        let secondary = TestMetrics()
        let tracker = TelemetryTracker(sinks: [SwiftMetricsSink(factory: primary), SwiftMetricsSink(factory: secondary)])
        await tracker.setModel("qwen3-4b")
        await tracker.requestStarted()
        let response = GenerationResponse(
            generatedText: "hello", promptTokens: 12, cachedPromptTokens: 6, completionTokens: 15,
            tokensPerSecond: 52.5, generationTime: 0.4, timeToFirstToken: 0.15,
            memory: .init(peakBytes: 8192, activeBytes: 4096, cacheBytes: 1024), finishReason: .stop
        ).withCacheTelemetry(CacheTelemetry(hits: 2, misses: 1, evictions: 1, cachedEntries: 2, cachedTokens: 64, cachedBytes: 16_384))
        await tracker.requestFinished(with: response)

        var answer = SystemOneAnswer()
        (answer.confidence, answer.abstain) = (0.4, true)
        let usage = SystemOneUsage(prefixTokens: 8, candidateBranches: 2, prefixEvaluations: 1, branchForwardCalls: 2, suffixTokens: 2)
        await tracker.recordDecision(
            SystemOneResponse(model: "jev-mini", answers: ["q": answer], usage: usage, calibrated: true), latencyMs: 18.0)
        await tracker.recordCalibration(confidence: 0.9, agreed: true)
        await tracker.recordIntervention(InterventionTelemetryEvent(kind: "intervene", target: "layer.0", activeOverlays: 2))
        await tracker.recordStage(ComposeStageSpan(stage: .perceive, model: "glm-ocr", latencyMs: 35.0, tokens: 120, escalated: true))

        for metrics in [primary, secondary] {
            #expect(try metrics.expectCounter("basalt_requests_total").totalValue == 1)
            #expect(try metrics.expectCounter("basalt_generated_tokens_total").totalValue == 15)
            #expect(try metrics.expectCounter("basalt_model_requests_total", [("model", "qwen3-4b")]).totalValue == 1)
            #expect(try metrics.expectMeter("basalt_active_requests").values == [1.0, 0.0])
            #expect(try metrics.expectGauge("basalt_gpu_active_bytes").lastValue == 4096.0)
            #expect(try metrics.expectGauge("basalt_gpu_peak_bytes").lastValue == 8192.0)
            #expect(try metrics.expectRecorder("basalt_tokens_per_second").lastValue == 52.5)
            #expect(try metrics.expectTimer("basalt_ttft").lastValue == 150_000_000)
            #expect(try metrics.expectGauge("basalt_prompt_cache_hits").lastValue == 2.0)
            #expect(try metrics.expectGauge("basalt_prompt_cache_tokens").lastValue == 64.0)
            #expect(try metrics.expectGauge("basalt_prompt_cache_bytes").lastValue == 16_384.0)
            #expect(try metrics.expectCounter("basalt_decision_requests_total", [("model", "jev-mini")]).totalValue == 1)
            #expect(try metrics.expectCounter("basalt_decision_escalations_total", [("model", "jev-mini")]).totalValue == 1)
            #expect(try metrics.expectTimer("basalt_decision_latency", [("model", "jev-mini")]).lastValue == 18_000_000)
            #expect(try metrics.expectCounter("basalt_calibration_samples_total").totalValue == 1)
            #expect(try metrics.expectCounter("basalt_calibration_agreement_total").totalValue == 1)
            #expect(try metrics.expectGauge("basalt_calibration_ece").lastValue != nil)
            #expect(try metrics.expectCounter("basalt_interventions_total", [("kind", "intervene")]).totalValue == 1)
            #expect(try metrics.expectGauge("basalt_active_overlays").lastValue == 2.0)
            let stageDims = [("stage", "perceive"), ("model", "glm-ocr")]
            #expect(try metrics.expectCounter("basalt_compose_stage_runs_total", stageDims).totalValue == 1)
            #expect(try metrics.expectCounter("basalt_compose_stage_tokens_total", stageDims).totalValue == 120)
            #expect(try metrics.expectCounter("basalt_compose_stage_escalations_total", stageDims).totalValue == 1)
        }
    }

    @Test("OpenMetrics exports prompt-cache and decision-calibration metrics, and GET /metrics matches GET /api/v1/metrics including auth")
    func openMetricsAndUnprefixedRouteParity() async throws {
        let tracker = TelemetryTracker()
        await tracker.recordCache(CacheTelemetry(hits: 7, misses: 3, evictions: 2, cachedEntries: 4, cachedTokens: 512, cachedBytes: 65_536))
        await tracker.recordCalibration(confidence: 0.75, agreed: true)
        await tracker.recordCalibration(confidence: 0.75, agreed: false)
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine(), telemetry: tracker)

        let openServer = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "stub"))
        try await Application(router: openServer.makeRouter()).test(.router) { client in
            var unprefixed = ""
            var prefixed = ""
            try await client.execute(uri: "/metrics", method: .get) { res in
                #expect(res.status == .ok && res.headers[.contentType] == TelemetryExport.contentType)
                unprefixed = String(buffer: res.body)
            }
            try await client.execute(uri: "/api/v1/metrics", method: .get) { res in
                #expect(res.status == .ok && res.headers[.contentType] == TelemetryExport.contentType)
                prefixed = String(buffer: res.body)
            }
            #expect(unprefixed == prefixed && unprefixed.hasSuffix("# EOF\n"))
            #expect(unprefixed.contains("basalt_prompt_cache_hits_total 7\n"))
            #expect(unprefixed.contains("basalt_prompt_cache_misses_total 3\n"))
            #expect(unprefixed.contains("basalt_prompt_cache_evictions_total 2\n"))
            #expect(unprefixed.contains("basalt_prompt_cache_entries 4\n"))
            #expect(unprefixed.contains("basalt_prompt_cache_tokens 512\n"))
            #expect(unprefixed.contains("basalt_prompt_cache_bytes 65536\n"))
            #expect(unprefixed.contains("basalt_calibration_samples_total 2\n"))
            #expect(unprefixed.contains("basalt_calibration_agreement_total 1\n"))
            #expect(unprefixed.contains("basalt_calibration_ece 0.25\n"))
        }

        let gatedServer = RESTServer(
            orchestrator: orchestrator, config: try ServerConfig(modelPath: "stub", host: "0.0.0.0", apiKey: "secret-key"))
        try await Application(router: gatedServer.makeRouter()).test(.router) { client in
            for path in ["/metrics", "/api/v1/metrics"] {
                try await client.execute(uri: path, method: .get) { #expect($0.status == .unauthorized) }
                try await client.execute(uri: path, method: .get, headers: [.authorization: "Bearer secret-key"]) {
                    #expect($0.status == .ok)
                }
            }
        }
    }

    @Test("MCP telemetry_query exposes cache and calibration sections by default and when filtered")
    func mcpTelemetryQueryCacheAndCalibration() async throws {
        let tracker = TelemetryTracker()
        await tracker.recordCache(CacheTelemetry(hits: 5, misses: 2, evictions: 1, cachedEntries: 3, cachedTokens: 256, cachedBytes: 32_768))
        await tracker.recordCalibration(confidence: 0.9, agreed: true)
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine(), telemetry: tracker)
        let lines = MCPLines()
        let server = MCPServer(orchestrator: orchestrator, writer: { lines.append($0) })

        await server.dispatch(toolCall(1, "telemetry_query", [:]))
        let full = try lines.toolText(0)
        let cache = try #require(full["cache"] as? [String: Any])
        let calibration = try #require(full["calibration"] as? [String: Any])
        #expect(cache["hits"] as? Int == 5 && cache["misses"] as? Int == 2 && cache["evictions"] as? Int == 1)
        #expect(cache["cached_entries"] as? Int == 3 && cache["cached_tokens"] as? Int == 256 && cache["cached_bytes"] as? Int == 32_768)
        #expect(calibration["samples"] as? Int == 1 && calibration["agreements"] as? Int == 1)
        #expect((calibration["bins"] as? [[String: Any]])?.count == DecisionCalibrationTelemetry.binCount)

        await server.dispatch(toolCall(2, "telemetry_query", ["sections": ["cache", "calibration"]]))
        let filtered = try lines.toolText(1)
        #expect(filtered["cache"] != nil && filtered["calibration"] != nil && filtered["windows"] == nil && filtered["health"] == nil)
    }
}
