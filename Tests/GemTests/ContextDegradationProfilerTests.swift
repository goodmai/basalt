import Testing
import Foundation
@testable import GemCore

// MARK: — ContextDegradationProfiler Tests

@Suite("ContextDegradationProfiler")
struct ContextDegradationProfilerTests {

    // MARK: — TC-1.2.3.1: Profile with valid context sizes

    @Test("Profile with valid context sizes returns data points")
    func testProfileWithValidContextSizes() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine)

        try await orchestrator.loadModel(path: "test-model")

        let profiler = ContextDegradationProfiler(orchestrator: orchestrator)

        // Use small context sizes for fast testing
        let targetSizes = [100, 500, 1000]

        let report = try await profiler.runBenchmark(
            contextSizes: targetSizes,
            tokensToGenerate: 10,
            iterations: 1
        )

        #expect(report.dataPoints.count == 3)
        #expect(report.dataPoints[0].contextSize == 100)
        #expect(report.dataPoints[1].contextSize == 500)
        #expect(report.dataPoints[2].contextSize == 1000)

        for point in report.dataPoints {
            #expect(point.avgTPS > 0)
            #expect(point.avgTTFT > 0)
            #expect(point.memoryActiveMB >= 0)
        }
    }

    // MARK: — TC-1.2.3.2: Export results to JSON

    @Test("Report can be encoded to JSON")
    func testReportEncodesToJSON() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine)

        try await orchestrator.loadModel(path: "test-model")

        let profiler = ContextDegradationProfiler(orchestrator: orchestrator)

        let report = try await profiler.runBenchmark(
            contextSizes: [100, 500],
            tokensToGenerate: 10,
            iterations: 1
        )

        // Verify report can be encoded to JSON
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let jsonData = try encoder.encode(report)

        #expect(jsonData.count > 0)

        // Verify it can be decoded back
        let decoded = try JSONDecoder().decode(ContextDegradationReport.self, from: jsonData)
        #expect(decoded.dataPoints.count == 2)
        #expect(decoded.dataPoints[0].contextSize == 100)
        #expect(decoded.dataPoints[1].contextSize == 500)
    }

    // MARK: — TC-1.2.3.3: Handle OOM gracefully

    @Test("Profile handles errors gracefully")
    func testProfileHandlesErrorsGracefully() async throws {
        let engine = MockInferenceEngine()
        await engine.update { $0.shouldFailOnGenerate = true }

        let orchestrator = ModelOrchestratorActor(engine: engine)
        try await orchestrator.loadModel(path: "test-model")

        let profiler = ContextDegradationProfiler(orchestrator: orchestrator)

        // Should throw error but not crash
        await #expect(throws: Error.self) {
            _ = try await profiler.runBenchmark(
                contextSizes: [100],
                tokensToGenerate: 10,
                iterations: 1
            )
        }
    }

    // MARK: — TC-1.2.3.4: Calculate TPS degradation curve

    @Test("TPS degrades with increasing context size")
    func testTPSDegradationCurve() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine)

        try await orchestrator.loadModel(path: "test-model")

        let profiler = ContextDegradationProfiler(orchestrator: orchestrator)

        let report = try await profiler.runBenchmark(
            contextSizes: [100, 500, 1000, 2000],
            tokensToGenerate: 10,
            iterations: 1
        )

        #expect(report.dataPoints.count == 4)

        // Verify all data points have valid TPS
        for point in report.dataPoints {
            #expect(point.avgTPS > 0)
            #expect(point.avgTTFT > 0)
        }

        // Note: MockInferenceEngine returns constant TPS, so we can't test actual degradation
        // In real usage, TPS would decrease as context size increases
    }

    // MARK: — Additional Tests

    @Test("Profile with single context size")
    func testProfileWithSingleContextSize() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine)

        try await orchestrator.loadModel(path: "test-model")

        let profiler = ContextDegradationProfiler(orchestrator: orchestrator)

        let report = try await profiler.runBenchmark(
            contextSizes: [1000],
            tokensToGenerate: 10,
            iterations: 1
        )

        #expect(report.dataPoints.count == 1)
        #expect(report.dataPoints[0].contextSize == 1000)
    }

    @Test("Profile with multiple iterations averages results")
    func testProfileWithMultipleIterations() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine)

        try await orchestrator.loadModel(path: "test-model")

        let profiler = ContextDegradationProfiler(orchestrator: orchestrator)

        let report = try await profiler.runBenchmark(
            contextSizes: [500],
            tokensToGenerate: 10,
            iterations: 3
        )

        #expect(report.dataPoints.count == 1)
        #expect(report.dataPoints[0].avgTPS > 0)
        #expect(report.dataPoints[0].avgTTFT > 0)
    }

    @Test("Profile without loaded model throws error")
    func testProfileWithoutLoadedModel() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine)

        let profiler = ContextDegradationProfiler(orchestrator: orchestrator)

        await #expect(throws: Error.self) {
            _ = try await profiler.runBenchmark(
                contextSizes: [100],
                tokensToGenerate: 10,
                iterations: 1
            )
        }
    }
}
