import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
@testable import GemCore

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

    @Test("TelemetryTracker records generation events and request lifecycle")
    func testTelemetryTrackerRecording() async {
        let tracker = TelemetryTracker()
        await tracker.requestStarted()
        await tracker.recordGeneration(response(tokens: 120, tps: 42.5, ttft: 0.15, active: 6 << 30, peak: 7 << 30))
        await tracker.setModelMemory(mb: 5800)

        let busy = await tracker.snapshot()
        #expect(busy.activeRequests == 1)
        #expect(busy.totalRequests == 1)
        #expect(busy.totalTokensGenerated == 120)
        #expect(busy.tokensPerSecond == 42.5)
        #expect(busy.timeToFirstToken == 0.15)
        #expect(busy.gpuActiveBytes == 6 << 30)
        #expect(busy.gpuPeakBytes == 7 << 30)
        #expect(busy.modelRAMMB == 5800)
        #expect(busy.totalRAMGB > 0)

        await tracker.requestFinished()
        await tracker.requestFinished()  // never goes negative
        #expect(await tracker.snapshot().activeRequests == 0)
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

    @Test("Streaming requests are counted and release the active slot when the stream ends")
    func testStreamingPathRecordsTelemetry() async throws {
        let tracker = TelemetryTracker()
        let orch = ModelOrchestratorActor(engine: MockInferenceEngine(), telemetry: tracker)
        try await orch.loadModel(path: "test-model-path")

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

    @Test("GET /api/v1/stats serves the serving process's snapshot")
    func testStatsEndpoint() async throws {
        let tracker = TelemetryTracker()
        await tracker.recordGeneration(response(tokens: 7, tps: 9.5, ttft: 0.2, active: 100, peak: 200))
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

    @Test("MCP status tool returns telemetry metrics consistent with tracker")
    func testMCPStatusIncludesTelemetry() async throws {
        let engine = MockInferenceEngine()
        let orch = ModelOrchestratorActor(engine: engine)
        try await orch.loadModel(path: "test-model-path")

        actor OutputCapturer {
            var lines: [String] = []
            func capture(_ line: String) { lines.append(line) }
        }
        let capturer = OutputCapturer()

        let server = MCPServer(orchestrator: orch) { line in
            Task { await capturer.capture(line) }
        }

        let request = """
        {"jsonrpc": "2.0", "id": 77, "method": "tools/call", "params": {"name": "status", "arguments": {}}}
        """

        await server.dispatch(request)
        try? await Task.sleep(for: .milliseconds(50))

        let output = await capturer.lines
        #expect(output.count == 1)
        #expect(output[0].contains("telemetry"))
        #expect(output[0].contains("totalRAMGB"))
        #expect(output[0].contains("tokensPerSecond"))
    }
}
