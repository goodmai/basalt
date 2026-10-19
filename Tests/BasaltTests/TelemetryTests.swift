import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltUI
@testable import BasaltCLI

@Suite("Telemetry & Observability Tests")
struct TelemetryTests {

    private func response(tokens: Int, tps: Double, ttft: Double, active: Int, peak: Int) -> GenerationResponse {
        GenerationResponse(
            generatedText: "x", promptTokens: 3, completionTokens: tokens,
            tokensPerSecond: tps, generationTime: 1, timeToFirstToken: ttft,
            memory: .init(peakBytes: peak, activeBytes: active, cacheBytes: 0),
            finishReason: .stop
        )
    }

    @Test("TelemetryTracker records a finished request and its result in one step")
    func testTelemetryTrackerRecording() async {
        let tracker = TelemetryTracker()
        await tracker.requestStarted()
        await tracker.setModelMemory(mb: 5800)
        #expect(await tracker.snapshot().activeRequests == 1)

        await tracker.requestFinished(with: response(tokens: 120, tps: 42.5, ttft: 0.15, active: 6 << 30, peak: 7 << 30))
        let done = await tracker.snapshot()
        #expect(done.activeRequests == 0)
        #expect(done.totalRequests == 1)
        #expect(done.totalTokensGenerated == 120)
        #expect(done.tokensPerSecond == 42.5)
        #expect(done.timeToFirstToken == 0.15)
        #expect(done.gpuActiveBytes == 6 << 30)
        #expect(done.gpuPeakBytes == 7 << 30)
        #expect(done.modelRAMMB == 5800)
        #expect(done.totalRAMGB > 0)

        await tracker.requestFinished()  // never goes negative, records nothing
        let after = await tracker.snapshot()
        #expect(after.activeRequests == 0)
        #expect(after.totalRequests == 1)
    }

    @Test("formattedStatsLine shows only measured metrics")
    func testFormattedStatsLine() {
        let line = ServerTelemetrySnapshot(
            tokensPerSecond: 55.4, timeToFirstToken: 0.12, totalRequests: 9, activeRequests: 2,
            gpuActiveBytes: 1_073_741_824, gpuPeakBytes: 2_147_483_648,
            totalRAMGB: 32, availableRAMGB: 18.5, modelRAMMB: 4096
        ).formattedStatsLine
        #expect(line.hasPrefix("[stats] 55.4 tok/s"))
        #expect(line.contains("TTFT: 0.12s"))
        #expect(line.contains("RAM avail: 18.5/32 GB"))
        #expect(line.contains("GPU active: 1.00 GB (peak 2.00 GB)"))
        #expect(line.contains("Active: 2"))
        #expect(!line.contains("KV"))
        #expect(!line.contains("Parked"))
    }

    @Test("ServerTelemetrySnapshot round-trips through JSON")
    func testSnapshotCodable() throws {
        let original = ServerTelemetrySnapshot(tokensPerSecond: 12.3, activeRequests: 1, gpuActiveBytes: 42)
        let decoded = try JSONDecoder().decode(ServerTelemetrySnapshot.self, from: JSONEncoder().encode(original))
        #expect(decoded == original)
    }

    private func loadedOrchestrator(_ tracker: TelemetryTracker, delay: TimeInterval = 0) async throws -> (ModelOrchestratorActor, MockInferenceEngine) {
        let engine = MockInferenceEngine()
        await engine.setArtificialDelay(delay)
        let orch = ModelOrchestratorActor(engine: engine, telemetry: tracker)
        try await orch.loadModel(path: "test-model-path")
        return (orch, engine)
    }

    @Test("Streaming requests are counted and hold the active slot until the stream ends")
    func testStreamingPathRecordsTelemetry() async throws {
        let tracker = TelemetryTracker()
        // The delay keeps the stream open long enough to observe the slot deterministically.
        let (orch, _) = try await loadedOrchestrator(tracker, delay: 0.2)

        let stream = try await orch.generateStream(request: GenerationRequest(prompt: "hi", maxTokens: 8))
        #expect(await tracker.snapshot().activeRequests == 1)
        for await _ in stream {}
        try? await Task.sleep(for: .milliseconds(20))

        let after = await tracker.snapshot()
        #expect(after.totalRequests == 1)
        #expect(after.activeRequests == 0)

        _ = try await orch.generate(request: GenerationRequest(prompt: "hi", maxTokens: 8))
        let both = await tracker.snapshot()
        #expect(both.totalRequests == 2)
        #expect(both.activeRequests == 0)
    }

    @Test("A failing generation releases its slot and records nothing")
    func testFailedGenerationReleasesSlot() async throws {
        let tracker = TelemetryTracker()
        let (orch, engine) = try await loadedOrchestrator(tracker)
        await engine.setShouldFailOnGenerate(true)

        await #expect(throws: (any Error).self) {
            _ = try await orch.generate(request: GenerationRequest(prompt: "hi", maxTokens: 8))
        }
        await #expect(throws: (any Error).self) {
            _ = try await orch.generateStream(request: GenerationRequest(prompt: "hi", maxTokens: 8))
        }
        let snap = await tracker.snapshot()
        #expect(snap.activeRequests == 0)
        #expect(snap.totalRequests == 0)
    }

    @Test("A client that disconnects mid-stream releases its slot")
    func testDisconnectReleasesSlot() async throws {
        let tracker = TelemetryTracker()
        let (orch, _) = try await loadedOrchestrator(tracker, delay: 5.0)

        /// Polls until `activeRequests` reaches `value` or two seconds pass (no fixed sleeps:
        /// the suite runs in parallel and a fixed wait races the scheduler).
        func waitForActive(_ value: Int) async -> Bool {
            let deadline = ContinuousClock.now + .seconds(2)
            while ContinuousClock.now < deadline {
                if await tracker.snapshot().activeRequests == value { return true }
                try? await Task.sleep(for: .milliseconds(5))
            }
            return false
        }

        // A consumer that stops iterating (e.g. ChatController's task being cancelled)
        // cancels the orchestrator's stream through onTermination.
        let consumer = Task {
            let stream = try await orch.generateStream(request: GenerationRequest(prompt: "hi", maxTokens: 8))
            for await _ in stream {}
        }
        #expect(await waitForActive(1))
        consumer.cancel()
        #expect(await waitForActive(0))
        #expect(await tracker.snapshot().totalRequests == 0)
    }

    @Test("GET /api/v1/stats serves the serving process's snapshot")
    func testStatsEndpoint() async throws {
        let tracker = TelemetryTracker()
        await tracker.requestStarted()
        await tracker.requestFinished(with: response(tokens: 7, tps: 9.5, ttft: 0.2, active: 100, peak: 200))
        let orch = ModelOrchestratorActor(engine: MockInferenceEngine(), telemetry: tracker)
        let server = RESTServer(orchestrator: orch, config: try ServerConfig(modelPath: "stub"))
        try await Application(router: server.makeRouter()).test(.router) { client in
            try await client.execute(uri: "/api/v1/stats", method: .get) { res in
                #expect(res.status == .ok)
                let snap = try JSONDecoder().decode(ServerTelemetrySnapshot.self, from: Data(String(buffer: res.body).utf8))
                #expect(snap.totalRequests == 1)
                #expect(snap.tokensPerSecond == 9.5)
                #expect(snap.gpuPeakBytes == 200)
            }
        }
    }

    @Test("GET /api/v1/stats is behind the API key when one is configured")
    func testStatsEndpointRequiresKey() async throws {
        let orch = ModelOrchestratorActor(engine: MockInferenceEngine(), telemetry: TelemetryTracker())
        let server = RESTServer(orchestrator: orch, config: try ServerConfig(modelPath: "stub", host: "0.0.0.0", apiKey: "k-1"))
        try await Application(router: server.makeRouter()).test(.router) { client in
            try await client.execute(uri: "/api/v1/stats", method: .get) { res in
                #expect(res.status == .unauthorized)
            }
            try await client.execute(uri: "/api/v1/stats", method: .get, headers: [.authorization: "Bearer k-1"]) { res in
                #expect(res.status == .ok)
            }
        }
    }

    @Test("MCP status reports the same numbers as the tracker, in diagnostics and telemetry alike")
    func testMCPStatusMatchesTracker() async throws {
        let tracker = TelemetryTracker()
        let (orch, _) = try await loadedOrchestrator(tracker)
        // One streamed and one plain request: the orchestrator's own counters used to miss the stream.
        let stream = try await orch.generateStream(request: GenerationRequest(prompt: "hi", maxTokens: 8))
        for await _ in stream {}
        try? await Task.sleep(for: .milliseconds(20))
        _ = try await orch.generate(request: GenerationRequest(prompt: "hi", maxTokens: 8))
        let expected = await tracker.snapshot()

        actor OutputCapturer {
            var lines: [String] = []
            func capture(_ line: String) { lines.append(line) }
        }
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: orch) { line in
            Task { await capturer.capture(line) }
        }
        await server.dispatch("""
        {"jsonrpc": "2.0", "id": 77, "method": "tools/call", "params": {"name": "status", "arguments": {}}}
        """)
        try? await Task.sleep(for: .milliseconds(50))

        let output = await capturer.lines
        #expect(output.count == 1)
        let outer = try #require(try JSONSerialization.jsonObject(with: Data(output[0].utf8)) as? [String: Any])
        let content = try #require((outer["result"] as? [String: Any])?["content"] as? [[String: Any]])
        let text = try #require(content.first?["text"] as? String)
        let status = try #require(try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        let telemetry = try #require(status["telemetry"] as? [String: Any])
        let diagnostics = try #require(status["diagnostics"] as? [String: Any])

        #expect(expected.totalRequests == 2)
        #expect(telemetry["totalRequests"] as? Int == expected.totalRequests)
        #expect(telemetry["totalTokensGenerated"] as? Int == expected.totalTokensGenerated)
        #expect(diagnostics["requestCount"] as? Int == expected.totalRequests)
        #expect(diagnostics["totalTokensGenerated"] as? Int == expected.totalTokensGenerated)
    }
}
