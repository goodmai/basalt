import XCTest
@testable import GemmaServerCore

final class ContextDegradationProfilerTests: XCTestCase {
    
    func testContextDegradationProfiler() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, maxTokens: 100)
        
        try await orchestrator.loadModel(path: "/mock/path")
        
        let profiler = ContextDegradationProfiler(orchestrator: orchestrator)
        
        // Use a small set of target context sizes for the test
        let targetSizes = [100, 500, 1000]
        
        let report = try await profiler.runBenchmark(contextSizes: targetSizes, tokensToGenerate: 10, iterations: 1)
        
        XCTAssertEqual(report.dataPoints.count, 3)
        XCTAssertEqual(report.dataPoints[0].contextSize, 100)
        XCTAssertEqual(report.dataPoints[1].contextSize, 500)
        XCTAssertEqual(report.dataPoints[2].contextSize, 1000)
        
        for point in report.dataPoints {
            XCTAssertGreaterThan(point.avgTPS, 0)
            XCTAssertGreaterThan(point.avgTTFT, 0)
            // Mock memory is hardcoded to activeBytes = 512
            XCTAssertEqual(point.memoryActiveMB, 512 / 1_048_576)
        }
    }
}
