import AppKit
import Testing
import Foundation
import SwiftUI
@testable import BasaltCore
@testable import BasaltServer
@testable import BasaltApp
@testable import BasaltAppCore

@Suite("X-ray telemetry phase 4: the telemetry_query MCP tool returns the chosen sections, and the Mac app Monitor shows windows, decisions and confidence (#450)")
struct TelemetryQueryTests {
    @Test("telemetry_query returns totals and only the asked sections, and refuses an unknown section")
    func mcp() async throws {
        let telemetry = TelemetryTracker()
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine(), telemetry: telemetry)
        try await orchestrator.loadModel(path: "/models/chat")
        _ = try await orchestrator.generate(request: GenerationRequest(prompt: "Hi", maxTokens: 4))
        let lines = MCPLines()
        let server = MCPServer(orchestrator: orchestrator, writer: { lines.append($0) })
        await server.dispatch(toolCall(1, "telemetry_query", ["sections": ["windows"]]))
        let result = try lines.toolText(0)
        #expect((result["totals"] as? [String: Any])?["requests"] as? Int == 1)
        #expect((result["windows"] as? [[String: Any]])?.first?["requests"] as? Int == 1 && result["decisions"] == nil && result["memory"] == nil)
        await server.dispatch(toolCall(2, "telemetry_query", ["sections": ["secrets"]]))
        #expect(try lines.json(1)["error"] != nil)
        #expect(MCPToolCatalog.list.tools.contains { $0.name == "telemetry_query" })
        #expect(TelemetryQuery.validated(["windows", "experts"]) == ["windows", "experts"] && TelemetryQuery.validated(["x"]) == nil)
    }

    @MainActor
    @Test("The Monitor lays out recent traffic, decisions and the confidence chart")
    func monitor() {
        var snapshot = ServerTelemetrySnapshot()
        snapshot.windows = [TelemetryWindow(seconds: 60, requests: 3, tokens: 90, ttftP50: 0.21, ttftP95: 0.4, tokensPerSecondP50: 31, tokensPerSecondP95: 36, promptCacheHitRatio: 0.6, draftAcceptance: nil),
                            TelemetryWindow(seconds: 3600, requests: 0, tokens: 0)]
        var decision = DecisionTelemetry()
        (decision.requests, decision.answers, decision.cost) = (4, 7, 0.0003)
        snapshot.decisions = ["gemini-decision-1": decision]
        var confidence = ConfidenceTelemetry()
        confidence.record(ResponseConfidence(min: 0.4, mean: 0.8))
        snapshot.confidence = confidence
        var state = MonitorState()
        (state.snapshot, state.polling) = (snapshot, true)
        let store = Store(state, reduce: MonitorReducer.reduce, handle: { _ in .none })
        let host = NSHostingView(rootView: MonitorView(store: store).frame(width: 520, height: 900))
        host.layoutSubtreeIfNeeded()
        #expect(host.fittingSize.width == 520)
        #expect(WindowsSection.span(60) == "1 min" && WindowsSection.span(3600) == "1 h" && WindowsSection.pair(nil, 1, format: "%.1f") == "–")
        #expect(WindowsSection.pair(0.21, 0.4, format: "%.2f s") == "0.21 s / 0.40 s")
    }
}
