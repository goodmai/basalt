import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltUI
@testable import BasaltCLI

@Suite("FitCommand Tests - Epic 16.7 / Issue #31")
struct FitCommandTests {

    private func score(measured: Bool, optimal: String?) -> FitScore {
        FitScore(
            modelName: "Qwen 3.5 4B", fitLevel: .perfect, score: 95.0, estimatedRAM: 2300,
            estimatedTPS: measured ? 92 : 85, contextWindow: 32768, isMeasured: measured,
            measuredTPS: measured ? 92.4 : nil, calibrationFactor: measured ? 1.08 : 1.0,
            selectedQuantization: "4-bit", optimalQuantization: optimal
        )
    }

    @Test("fit table marks measured TPS and shows the grid's best width")
    func testTableRow() {
        let measured = FitCommand.tableRow(score(measured: true, optimal: "8-bit"))
        #expect(measured[4] == "92 measured ✓")
        #expect(measured[6] == "8-bit")

        let theoretical = FitCommand.tableRow(score(measured: false, optimal: nil))
        #expect(theoretical[4] == "85")
        #expect(theoretical[6] == "—")
        #expect(theoretical[5] == "32k")
    }

    @Test("HardwareBenchmarkProfileStore writes a profile that a later process reads back")
    func testHardwareBenchmarkProfileStore() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let customPath = tempDir.appendingPathComponent("test_profile.json").path

        try await HardwareBenchmarkProfileStore(customPath: customPath).record(BenchmarkMeasurement(
            modelId: "mlx-community/Qwen3.5-4B-4bit", chipModel: "M2 Max",
            measuredTPS: 104.2, measuredTTFT: 0.05, quant: "4bit"
        ))

        // Read from disk, not from the store's cache.
        let loaded = HardwareBenchmarkProfileStore.loadSync(path: customPath)
        #expect(loaded.chipModel == "M2 Max")
        #expect(loaded.measurements.count == 1)
        #expect(loaded.measurement(for: "mlx-community/Qwen3.5-4B-4bit", currentChip: "M2 Max")?.measuredTPS == 104.2)
    }

    @Test("An unreadable profile is not overwritten by the next record")
    func testCorruptProfileIsNotOverwritten() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let url = tempDir.appendingPathComponent("p.json")
        try Data("{\"measurements\": [broken".utf8).write(to: url)

        await #expect(throws: (any Error).self) {
            try await HardwareBenchmarkProfileStore(customPath: url.path).record(BenchmarkMeasurement(
                modelId: "org/m", chipModel: "M4", measuredTPS: 50, measuredTTFT: 0.1))
        }
        #expect(try String(contentsOf: url, encoding: .utf8) == "{\"measurements\": [broken")
    }

    @Test("Basename matching is used only when it is unambiguous")
    func testAmbiguousBasename() {
        let m = { (id: String, tps: Double) in BenchmarkMeasurement(modelId: id, chipModel: "M4", measuredTPS: tps, measuredTTFT: 0.1) }
        let one = HardwareBenchmarkProfile(chipModel: "M4", measurements: ["orgA/X-4bit": m("orgA/X-4bit", 40)])
        #expect(one.measurement(for: "mirror/X-4bit", currentChip: "M4")?.measuredTPS == 40)

        let two = HardwareBenchmarkProfile(chipModel: "M4", measurements: [
            "orgA/X-4bit": m("orgA/X-4bit", 40), "orgB/X-4bit": m("orgB/X-4bit", 90)
        ])
        #expect(two.measurement(for: "orgB/X-4bit", currentChip: "M4")?.measuredTPS == 90, "exact id wins")
        #expect(two.measurement(for: "mirror/X-4bit", currentChip: "M4") == nil, "ambiguous basename")
    }

    @Test("Parameter counts are read from the size token, not from '-4bit'")
    func testParameterInference() {
        let total = { (id: String) in ModelDefinition.inferTotalParameters(id: id, name: "", ramMB: 0) }
        #expect(total("mlx-community/Llama-3.1-8B-Instruct-4bit") == 8)
        #expect(total("Qwen/Qwen3-30B-A3B-4bit") == 30)
        #expect(total("mlx-community/Qwen2.5-0.5B-Instruct-4bit") == 0.5)
        #expect(total("mlx-community/Qwen3.5-4B-4bit") == 4)
        #expect(ModelDefinition.inferActiveParameters(id: "Qwen/Qwen3-30B-A3B-4bit", name: "", total: 30) == 3)
        #expect(ModelDefinition.inferActiveParameters(id: "mlx-community/Llama-3.1-8B-4bit", name: "", total: 8) == 8)
    }
}
