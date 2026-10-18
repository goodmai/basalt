import Testing
import Foundation
@testable import GemCore

@Suite("Telemetry & Observability Tests")
struct TelemetryTests {

    @Test("TelemetryTracker records generation events and updates metrics")
    func testTelemetryTrackerRecording() async {
        let tracker = TelemetryTracker()
        
        await tracker.recordGeneration(tokens: 120, tokensPerSecond: 42.5, timeToFirstToken: 0.15)
        await tracker.setModelMemory(mb: 5800)
        await tracker.setKVCacheSize(bytes: 268_435_456) // 256 MB
        await tracker.setParkedConversations(3)
        await tracker.setActiveRequests(1)
        await tracker.recordSpeculative(drafted: 100, accepted: 75)

        let snapshot = await tracker.snapshot()
        
        #expect(snapshot.totalRequests == 1)
        #expect(snapshot.totalTokensGenerated == 120)
        #expect(snapshot.tokensPerSecond == 42.5)
        #expect(snapshot.timeToFirstToken == 0.15)
        #expect(snapshot.modelRAMMB == 5800)
        #expect(snapshot.kvCacheSizeBytes == 268_435_456)
        #expect(snapshot.kvCacheSizeFormatted == "256.0 MB")
        #expect(snapshot.parkedConversations == 3)
        #expect(snapshot.activeRequests == 1)
        #expect(snapshot.speculativeAcceptanceRate == 0.75)
    }

    @Test("ServerTelemetrySnapshot formattedStatsLine produces consistent terminal output")
    func testFormattedStatsLine() {
        let snapshot = ServerTelemetrySnapshot(
            timestamp: Date(),
            tokensPerSecond: 38.4,
            timeToFirstToken: 0.18,
            totalRequests: 10,
            totalTokensGenerated: 1500,
            activeRequests: 1,
            parkedConversations: 2,
            kvCacheSizeBytes: 134_217_728,
            kvCacheSizeFormatted: "128.0 MB",
            speculativeAcceptanceRate: 0.80,
            totalRAMGB: 32,
            availableRAMGB: 24.5,
            modelRAMMB: 4000,
            chipModel: "M4",
            gpuName: "Apple M4"
        )

        let line = snapshot.formattedStatsLine
        #expect(line.contains("[stats]"))
        #expect(line.contains("38.4 tok/s"))
        #expect(line.contains("TTFT: 0.18s"))
        #expect(line.contains("RAM avail: 24.5/32 GB"))
        #expect(line.contains("Model: 4000 MB"))
        #expect(line.contains("KV: 128.0 MB"))
        #expect(line.contains("Parked: 2"))
        #expect(line.contains("Speculative: 80.0%"))
    }

    @Test("ServerTelemetrySnapshot can be encoded and decoded to/from JSON")
    func testSnapshotCodable() throws {
        let original = ServerTelemetrySnapshot(
            tokensPerSecond: 55.0,
            timeToFirstToken: 0.12,
            totalRequests: 5,
            totalTokensGenerated: 500,
            activeRequests: 0,
            parkedConversations: 1,
            kvCacheSizeBytes: 67_108_864,
            kvCacheSizeFormatted: "64.0 MB",
            speculativeAcceptanceRate: nil,
            totalRAMGB: 24,
            availableRAMGB: 16.0,
            modelRAMMB: 2700,
            chipModel: "M3",
            gpuName: "Apple M3"
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        let data = try encoder.encode(original)
        let decoded = try JSONDecoder().decode(ServerTelemetrySnapshot.self, from: data)

        #expect(decoded.tokensPerSecond == original.tokensPerSecond)
        #expect(decoded.timeToFirstToken == original.timeToFirstToken)
        #expect(decoded.totalRequests == original.totalRequests)
        #expect(decoded.kvCacheSizeFormatted == original.kvCacheSizeFormatted)
        #expect(decoded.chipModel == original.chipModel)
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
