import Testing
import Foundation
@testable import GemCore

@Suite("FitCommand Tests - Epic 16.7 / Issue #31")
struct FitCommandTests {
    
    @Test("FitScore serialization and measured label formatting")
    func testFitScoreMeasuredFormatting() {
        let unmeasured = FitScore(
            modelName: "Qwen 3.5 4B",
            fitLevel: .perfect,
            score: 95.0,
            estimatedRAM: 2300,
            estimatedTPS: 85,
            contextWindow: 32768,
            isMeasured: false,
            measuredTPS: nil,
            calibrationFactor: 1.0,
            selectedQuantization: "4-bit"
        )
        #expect(unmeasured.isMeasured == false)
        #expect(unmeasured.measuredTPS == nil)

        let measured = FitScore(
            modelName: "Qwen 3.5 4B",
            fitLevel: .perfect,
            score: 95.0,
            estimatedRAM: 2300,
            estimatedTPS: 92,
            contextWindow: 32768,
            isMeasured: true,
            measuredTPS: 92.4,
            calibrationFactor: 1.08,
            selectedQuantization: "4-bit"
        )
        #expect(measured.isMeasured == true)
        #expect(measured.measuredTPS == 92.4)
        #expect(measured.calibrationFactor == 1.08)
    }

    @Test("HardwareBenchmarkProfileStore saves and loads measurements")
    func testHardwareBenchmarkProfileStore() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let customPath = tempDir.appendingPathComponent("test_profile.json").path
        let store = HardwareBenchmarkProfileStore(customPath: customPath)

        let measurement = BenchmarkMeasurement(
            modelId: "mlx-community/Qwen3.5-4B-4bit",
            chipModel: "Apple M2 Max",
            measuredTPS: 104.2,
            measuredTTFT: 0.05,
            timestamp: Date(),
            quant: "4bit"
        )

        try await store.record(measurement)
        let loaded = await store.load()

        #expect(loaded.chipModel == "Apple M2 Max")
        #expect(loaded.measurements.count == 1)
        let matched = loaded.measurement(for: "mlx-community/Qwen3.5-4B-4bit")
        #expect(matched != nil)
        #expect(matched?.measuredTPS == 104.2)

        let calib = loaded.calibrationFactor(for: "mlx-community/Qwen3.5-4B-4bit", theoreticalTPS: 100.0)
        #expect(calib > 1.0)
    }
}
