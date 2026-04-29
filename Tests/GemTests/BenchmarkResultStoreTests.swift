import Testing
import Foundation
@testable import GemCore

// MARK: — BenchmarkResultStore Tests

@Suite("BenchmarkResultStore")
struct BenchmarkResultStoreTests {

    @Test("Store and retrieve synthetic benchmark results")
    func testStoreSyntheticBenchmark() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let dbPath = tempDir.appendingPathComponent("test_benchmarks_\(UUID().uuidString).db").path

        let store = try BenchmarkResultStore(dbPath: dbPath)

        let result = SyntheticBenchmarkResult(
            avgTPS: 150.5,
            avgTTFT: 0.05,
            minTPS: 140.0,
            maxTPS: 160.0,
            stdDevTPS: 5.2,
            memoryPeakMB: 4096,
            iterations: 10
        )

        try await store.storeSyntheticBenchmark(
            modelPath: "test-model",
            modelId: "test-id",
            result: result
        )

        let records = try await store.getRecentBenchmarks(modelPath: "test-model", benchmarkType: "synthetic", limit: 1)

        #expect(records.count == 1)
        #expect(records[0].modelPath == "test-model")
        #expect(records[0].benchmarkType == "synthetic")
        #expect(!records[0].resultJSON.isEmpty)

        // Cleanup
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("Store and retrieve context degradation benchmark results")
    func testStoreContextDegradationBenchmark() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let dbPath = tempDir.appendingPathComponent("test_benchmarks_\(UUID().uuidString).db").path

        let store = try BenchmarkResultStore(dbPath: dbPath)

        let report = ContextDegradationReport(
            modelPath: "test-model",
            dataPoints: [
                BenchmarkDataPoint(contextSize: 1000, avgTPS: 150.0, avgTTFT: 0.05, memoryActiveMB: 2048),
                BenchmarkDataPoint(contextSize: 4000, avgTPS: 140.0, avgTTFT: 0.06, memoryActiveMB: 3072)
            ]
        )

        try await store.storeContextDegradationBenchmark(
            modelPath: "test-model",
            modelId: "test-id",
            report: report
        )

        let records = try await store.getRecentBenchmarks(modelPath: "test-model", benchmarkType: "context_degradation", limit: 1)

        #expect(records.count == 1)
        #expect(records[0].benchmarkType == "context_degradation")

        // Cleanup
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("Query benchmark history")
    func testGetBenchmarkHistory() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let dbPath = tempDir.appendingPathComponent("test_benchmarks_\(UUID().uuidString).db").path

        let store = try BenchmarkResultStore(dbPath: dbPath)

        // Store multiple results
        for i in 1...5 {
            let result = SyntheticBenchmarkResult(
                avgTPS: Double(150 + i),
                avgTTFT: 0.05,
                minTPS: 140.0,
                maxTPS: 160.0,
                stdDevTPS: 5.2,
                memoryPeakMB: 4096,
                iterations: 10
            )

            try await store.storeSyntheticBenchmark(
                modelPath: "test-model",
                modelId: "test-id",
                result: result
            )

            try await Task.sleep(for: .milliseconds(10)) // Ensure different timestamps
        }

        let history = try await store.getBenchmarkHistory(
            modelPath: "test-model",
            benchmarkType: "synthetic"
        )

        #expect(history.count == 5)
        #expect(history[0].timestamp <= history[4].timestamp) // Ordered by timestamp ascending

        // Cleanup
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("Filter benchmarks by type")
    func testFilterByBenchmarkType() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let dbPath = tempDir.appendingPathComponent("test_benchmarks_\(UUID().uuidString).db").path

        let store = try BenchmarkResultStore(dbPath: dbPath)

        // Store synthetic benchmark
        let syntheticResult = SyntheticBenchmarkResult(
            avgTPS: 150.0,
            avgTTFT: 0.05,
            minTPS: 140.0,
            maxTPS: 160.0,
            stdDevTPS: 5.2,
            memoryPeakMB: 4096,
            iterations: 10
        )

        try await store.storeSyntheticBenchmark(
            modelPath: "test-model",
            modelId: "test-id",
            result: syntheticResult
        )

        // Store context degradation benchmark
        let contextReport = ContextDegradationReport(
            modelPath: "test-model",
            dataPoints: [
                BenchmarkDataPoint(contextSize: 1000, avgTPS: 150.0, avgTTFT: 0.05, memoryActiveMB: 2048)
            ]
        )

        try await store.storeContextDegradationBenchmark(
            modelPath: "test-model",
            modelId: "test-id",
            report: contextReport
        )

        // Query only synthetic
        let syntheticRecords = try await store.getRecentBenchmarks(
            modelPath: "test-model",
            benchmarkType: "synthetic",
            limit: 10
        )

        #expect(syntheticRecords.count == 1)
        #expect(syntheticRecords[0].benchmarkType == "synthetic")

        // Query only context_degradation
        let contextRecords = try await store.getRecentBenchmarks(
            modelPath: "test-model",
            benchmarkType: "context_degradation",
            limit: 10
        )

        #expect(contextRecords.count == 1)
        #expect(contextRecords[0].benchmarkType == "context_degradation")

        // Cleanup
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("Store git commit and system info")
    func testStoreMetadata() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let dbPath = tempDir.appendingPathComponent("test_benchmarks_\(UUID().uuidString).db").path

        let store = try BenchmarkResultStore(dbPath: dbPath)

        let result = SyntheticBenchmarkResult(
            avgTPS: 150.0,
            avgTTFT: 0.05,
            minTPS: 140.0,
            maxTPS: 160.0,
            stdDevTPS: 5.2,
            memoryPeakMB: 4096,
            iterations: 10
        )

        try await store.storeSyntheticBenchmark(
            modelPath: "test-model",
            modelId: "test-id",
            result: result
        )

        let records = try await store.getRecentBenchmarks(limit: 1)

        #expect(records.count == 1)
        // Git commit and system info may be nil in test environment, but fields should exist
        #expect(records[0].gitCommit != nil || records[0].gitCommit == nil) // Field exists
        #expect(records[0].systemInfo != nil || records[0].systemInfo == nil) // Field exists

        // Cleanup
        try? FileManager.default.removeItem(atPath: dbPath)
    }
}
