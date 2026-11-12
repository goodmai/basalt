import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
@testable import BasaltCore
@testable import BasaltServer

@Suite("X-ray telemetry phase 1: trailing windows with quantiles, prompt-cache and draft ratios, per-model decision aggregates, OpenMetrics at /api/v1/metrics (#450)")
struct TelemetryWindowsTests {
    static let start = Date(timeIntervalSince1970: 1_800_000_000)

    @Test("Windows count what falls inside them, take nearest-rank quantiles, and drop samples older than an hour")
    func windows() {
        var windows = TelemetryWindows()
        for index in 0 ..< 10 {
            var sample = TelemetrySample(at: Self.start.addingTimeInterval(Double(index * 30)), ttft: Double(index + 1) / 10, tokensPerSecond: Double(20 + index), completionTokens: 10, promptTokens: 100)
            (sample.cachedPromptTokens, sample.draftTokens, sample.acceptedDraftTokens) = (index.isMultiple(of: 2) ? 100 : 0, 4, 3)
            windows.record(sample)
        }
        let now = Self.start.addingTimeInterval(270)
        let minute = windows.window(seconds: 60, now: now)
        #expect(minute.requests == 3 && minute.tokens == 30 && minute.ttftP50 == 0.9 && minute.ttftP95 == 1.0)
        let hour = windows.window(seconds: 3600, now: now)
        #expect(hour.requests == 10 && hour.promptCacheHitRatio == 0.5 && hour.draftAcceptance == 0.75 && hour.tokensPerSecondP50 == 25)
        #expect(windows.windows(now: now).map(\.seconds) == [60, 300, 3600])
        windows.record(TelemetrySample(at: Self.start.addingTimeInterval(4000), ttft: 0.1, tokensPerSecond: 30, completionTokens: 1, promptTokens: 0))
        #expect(windows.samples.count == 1 && windows.window(seconds: 60, now: Self.start.addingTimeInterval(4000)).promptCacheHitRatio == nil)
        #expect(TelemetryWindows.percentile([], 0.5) == nil && TelemetryWindows.percentile([.nan, 3], 0.5) == 3)
    }

    @Test("Decision aggregates count answers, abstentions, escalations, cost and confidence deciles")
    func decisions() {
        var aggregate = DecisionTelemetry()
        var sure = SystemOneAnswer()
        sure.confidence = 0.97
        var unsure = SystemOneAnswer()
        (unsure.confidence, unsure.abstain) = (0.31, true)
        var usage = SystemOneUsage(prefixTokens: 0, candidateBranches: 0, prefixEvaluations: 0, branchForwardCalls: 0, suffixTokens: 0)
        usage.cost = 0.0002
        aggregate.record(SystemOneResponse(model: "gemini-decision-1", answers: ["a": sure, "b": unsure], usage: usage, calibrated: true), latencyMs: 210)
        #expect(aggregate.requests == 1 && aggregate.answers == 1 && aggregate.abstains == 1 && aggregate.escalations == 1)
        #expect(aggregate.confidence[9] == 1 && aggregate.confidence[3] == 1 && aggregate.cost == 0.0002 && aggregate.latencyMs == 210)
    }

    @Test("OpenMetrics text has types, help, quantile and model labels with escaping, and ends with EOF")
    func openMetrics() {
        var snapshot = ServerTelemetrySnapshot()
        (snapshot.totalRequests, snapshot.totalTokensGenerated) = (12, 3400)
        snapshot.windows = [TelemetryWindow(seconds: 60, requests: 2, tokens: 40, ttftP50: 0.2, ttftP95: 0.4, tokensPerSecondP50: 30, tokensPerSecondP95: 35, promptCacheHitRatio: 0.5, draftAcceptance: nil)]
        var decision = DecisionTelemetry()
        decision.requests = 3
        snapshot.decisions = ["org/\"odd\"": decision]
        let text = TelemetryExport.openMetrics(snapshot)
        #expect(text.contains("# TYPE basalt_requests counter\n") && text.contains("basalt_requests 12\n"))
        #expect(text.contains("basalt_ttft_seconds{window=\"60s\",quantile=\"0.95\"} 0.4") && text.contains("basalt_prompt_cache_hit_ratio{window=\"60s\"} 0.5"))
        #expect(text.contains("basalt_decision_requests{model=\"org/\\\"odd\\\"\"} 3") && !text.contains("draft_acceptance_ratio{") && text.hasSuffix("# EOF\n"))
    }

    @Test("/api/v1/metrics reports a generation and a decision made through /v1/systemone")
    func endpoint() async throws {
        let engine = MockInferenceEngine()
        let chat = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await chat.loadModel(path: "/models/chat")
        _ = try await chat.generate(request: GenerationRequest(prompt: "Hi", maxTokens: 4))
        let server = RESTServer(orchestrator: chat, config: try ServerConfig(modelPath: "org/mock-model")).withDecisions(StubDecisionEngine())
        try await Application(router: server.makeRouter()).test(.router) { client in
            try await client.execute(uri: "/v1/systemone", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: SystemOneAPITests.body)) { #expect($0.status == .ok) }
            try await client.execute(uri: "/api/v1/metrics", method: .get) { response in
                let text = String(buffer: response.body)
                #expect(response.status == .ok && response.headers[.contentType]?.hasPrefix("application/openmetrics-text") == true)
                #expect(text.contains("basalt_requests 1\n") && text.contains("basalt_window_requests{window=\"60s\"} 1") && text.contains("basalt_decision_requests{model=\"org/stub-jev\"} 1"))
            }
        }
        let snapshot = await chat.telemetrySnapshot()
        #expect(snapshot.windows?.first?.requests == 1 && snapshot.decisions?["org/stub-jev"]?.answers == 2)
    }
}
