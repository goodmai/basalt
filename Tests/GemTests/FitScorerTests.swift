import Testing
import Foundation
@testable import GemCore

@Suite("FitScorer Tests - Epic 16.7 / Issue #31")
struct FitScorerTests {
    
    private func createMockResources(ramGB: Int, availableGB: Int, chip: String) -> SystemProfiler.SystemResources {
        SystemProfiler.SystemResources(
            totalRAM: Int64(ramGB) * 1_073_741_824,
            availableRAM: Int64(availableGB) * 1_073_741_824,
            cpuCores: 10,
            gpuName: "Apple \(chip)",
            gpuMemory: Int64(ramGB) * 1_073_741_824,
            diskSpace: 500_000_000_000,
            osVersion: "14.0",
            chipModel: chip
        )
    }
    
    @Test("Calculate score for perfect fit on M2 Max")
    func testPerfectFit() async {
        let resources = createMockResources(ramGB: 32, availableGB: 18, chip: "M2 Max")
        let scorer = FitScorer(profile: resources, benchmarkProfile: HardwareBenchmarkProfile())
        let model = ModelDefinition(
            id: "mlx-community/Qwen3.5-4B-4bit",
            name: "Qwen 3.5 4B",
            family: .qwen,
            task: .chat,
            modality: .text,
            ramMB: 2300,
            contextWindow: 32768,
            quantization: "4-bit",
            quality: 0.85,
            totalParameters: 4.0,
            activeParameters: 4.0
        )
        
        let score = await scorer.score(model)
        #expect(score.fitLevel == .perfect)
        #expect(score.score > 80.0)
    }
    
    @Test("Calculate score for marginal fit on low RAM")
    func testMarginalFit() async {
        let resources = createMockResources(ramGB: 8, availableGB: 4, chip: "M1")
        let scorer = FitScorer(profile: resources, benchmarkProfile: HardwareBenchmarkProfile())
        let model = ModelDefinition(
            id: "mlx-community/Qwen3.6-27B",
            name: "Qwen 27B",
            family: .qwen,
            task: .chat,
            modality: .text,
            ramMB: 14500,
            contextWindow: 32768,
            quantization: "4-bit",
            quality: 0.92,
            totalParameters: 27.0,
            activeParameters: 27.0
        )
        
        let score = await scorer.score(model)
        #expect(score.fitLevel == .tooTight)
        #expect(score.score < 50.0)
    }

    @Test("MoE separate accounting: P_total for memory and P_active for decode speed")
    func testMoESeparateAccounting() async {
        let resources = createMockResources(ramGB: 32, availableGB: 20, chip: "M2 Max")
        let scorer = FitScorer(profile: resources, benchmarkProfile: HardwareBenchmarkProfile())

        // MoE model: 26B total parameters, 4B active parameters
        let moeModel = ModelDefinition(
            id: "Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX",
            name: "MYTHOS 26B A4B",
            family: .gemma,
            task: .chat,
            modality: .text,
            ramMB: 14500,
            contextWindow: 262144,
            quantization: "4-bit",
            quality: 0.93,
            totalParameters: 26.0,
            activeParameters: 4.0
        )

        // Dense model: 26B total parameters, 26B active parameters
        let denseModel = ModelDefinition(
            id: "custom/Dense-26B-4bit",
            name: "Dense 26B",
            family: .gemma,
            task: .chat,
            modality: .text,
            ramMB: 14500,
            contextWindow: 262144,
            quantization: "4-bit",
            quality: 0.93,
            totalParameters: 26.0,
            activeParameters: 26.0
        )

        #expect(moeModel.isMoE == true)
        #expect(denseModel.isMoE == false)

        let moeScore = await scorer.score(moeModel)
        let denseScore = await scorer.score(denseModel)

        // Acceptance criterion: MoE model occupies memory by P_total (same estimated RAM as dense)
        #expect(moeScore.estimatedRAM == denseScore.estimatedRAM)
        #expect(moeScore.estimatedRAM == 14500)

        // Acceptance criterion: MoE model is NOT considered slower than by P_active
        // 4B active should decode significantly faster than 26B active (at least 4x higher TPS)
        #expect(moeScore.estimatedTPS > denseScore.estimatedTPS * 3)
        #expect(moeScore.estimatedTPS >= 50)
    }

    @Test("Self-calibration from real hardware benchmark measurements")
    func testHardwareProfileCalibration() async {
        let resources = createMockResources(ramGB: 32, availableGB: 20, chip: "M3 Max")
        
        let benchmarkProfile = HardwareBenchmarkProfile(
            chipModel: "M3 Max",
            measurements: [
                "mlx-community/Qwen3.5-4B-4bit": BenchmarkMeasurement(
                    modelId: "mlx-community/Qwen3.5-4B-4bit",
                    chipModel: "M3 Max",
                    measuredTPS: 115.5,
                    measuredTTFT: 0.08,
                    timestamp: Date()
                )
            ]
        )

        let scorer = FitScorer(profile: resources, benchmarkProfile: benchmarkProfile)
        let model = ModelDefinition(
            id: "mlx-community/Qwen3.5-4B-4bit",
            name: "Qwen 3.5 4B",
            family: .qwen,
            task: .chat,
            modality: .text,
            ramMB: 2300,
            contextWindow: 32768,
            quantization: "4-bit",
            quality: 0.85,
            totalParameters: 4.0,
            activeParameters: 4.0
        )

        let score = await scorer.score(model)
        #expect(score.isMeasured == true)
        #expect(score.measuredTPS == 115.5)
        #expect(score.estimatedTPS == 116)
        #expect(score.calibrationFactor > 0.0)
    }

    @Test("Dynamic quantization selection across 8/6/4/3/2-bit grid")
    func testDynamicQuantizationGrid() {
        let resources = createMockResources(ramGB: 16, availableGB: 8, chip: "M2")
        let model = ModelDefinition(
            id: "mlx-community/Qwen2.5-Coder-7B",
            name: "Qwen2.5-Coder-7B",
            family: .qwen,
            task: .code,
            modality: .text,
            ramMB: 4100,
            contextWindow: 128000,
            quantization: "4-bit",
            quality: 0.89,
            totalParameters: 7.0,
            activeParameters: 7.0
        )

        let result = ModelFitAnalyzer.evaluateQuantizations(model: model, resources: resources, benchmarkProfile: HardwareBenchmarkProfile())
        #expect(result.options.count == 5)
        #expect(result.options.map(\.bits) == [8, 6, 4, 3, 2])

        // The chosen quantization must fit within 85% of available memory (8 GB * 0.85 = 6.8 GB = ~6960 MB)
        let budgetMB = Int(Double(resources.availableRAM) * 0.85 / 1_048_576.0)
        #expect(result.optimal.footprintMB <= budgetMB)
        #expect(result.optimal.fitsBudget == true)
        #expect(result.optimal.score.quality > 0.5)
    }

    @Test("Zero-I/O compile-time ModelDatabase lookup and MoE tagging")
    func testModelDatabaseZeroIO() {
        let found = ModelDatabase.find(id: "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit")
        #expect(found != nil)
        #expect(found?.isMoE == true)
        #expect(found?.totalParameters == 35.0)
        #expect(found?.activeParameters == 3.0)

        let qwenModels = ModelDatabase.models(for: .qwen)
        #expect(!qwenModels.isEmpty)
    }

    @Test("Cross-model suffix collisions are rejected by measurement lookup")
    func testNoCrossModelSuffixCollisions() {
        let profile = HardwareBenchmarkProfile(
            chipModel: "M2 Max",
            measurements: [
                "mlx-community/Qwen3.5-4B-4bit": BenchmarkMeasurement(
                    modelId: "mlx-community/Qwen3.5-4B-4bit",
                    chipModel: "M2 Max",
                    measuredTPS: 100.0,
                    measuredTTFT: 0.1,
                    quant: "4-bit"
                )
            ]
        )

        // Exact repo matches
        #expect(profile.measurement(for: "mlx-community/Qwen3.5-4B-4bit", currentChip: "M2 Max") != nil)
        #expect(profile.measurement(for: "Qwen3.5-4B-4bit", currentChip: "M2 Max") != nil)

        // Suffix matches without slash boundary must NOT match
        #expect(profile.measurement(for: "mlx-community/DeepSeek-4B-4bit", currentChip: "M2 Max") == nil)
        #expect(profile.measurement(for: "4bit", currentChip: "M2 Max") == nil)
        #expect(profile.measurement(for: "B-4bit", currentChip: "M2 Max") == nil)
    }

    @Test("Chip model mismatch rejects calibration from another machine")
    func testChipMismatchRejection() {
        let profile = HardwareBenchmarkProfile(
            chipModel: "Apple M1",
            measurements: [
                "mlx-community/Qwen3.5-4B-4bit": BenchmarkMeasurement(
                    modelId: "mlx-community/Qwen3.5-4B-4bit",
                    chipModel: "Apple M1",
                    measuredTPS: 40.0,
                    measuredTTFT: 0.2,
                    quant: "4-bit"
                )
            ]
        )

        // Same chip family accepts
        #expect(profile.measurement(for: "mlx-community/Qwen3.5-4B-4bit", currentChip: "Apple M1") != nil)

        // Different chip family rejects
        #expect(profile.measurement(for: "mlx-community/Qwen3.5-4B-4bit", currentChip: "Apple M3 Max") == nil)
        #expect(profile.measurement(for: "mlx-community/Qwen3.5-4B-4bit", currentChip: "Apple M2 Pro") == nil)
    }

    @Test("Non-4-bit benchmark measurements rescale according to measurement.quant")
    func testNon4BitRescaling() {
        let resources = createMockResources(ramGB: 64, availableGB: 40, chip: "M2 Max")
        let profile = HardwareBenchmarkProfile(
            chipModel: "M2 Max",
            measurements: [
                "mlx-community/Qwen2.5-Coder-7B": BenchmarkMeasurement(
                    modelId: "mlx-community/Qwen2.5-Coder-7B",
                    chipModel: "M2 Max",
                    measuredTPS: 50.0,
                    measuredTTFT: 0.1,
                    quant: "8-bit" // Measured at 8-bit!
                )
            ]
        )

        let model = ModelDefinition(
            id: "mlx-community/Qwen2.5-Coder-7B",
            name: "Qwen 7B",
            family: .qwen,
            task: .code,
            modality: .text,
            ramMB: 5000,
            contextWindow: 32768,
            quantization: "4-bit",
            quality: 0.90,
            totalParameters: 7.0,
            activeParameters: 7.0
        )

        let result = ModelFitAnalyzer.evaluateQuantizations(
            model: model,
            resources: resources,
            benchmarkProfile: profile
        )

        let opt8 = result.options.first { $0.bits == 8 }!
        let opt4 = result.options.first { $0.bits == 4 }!

        // Since measured was 8-bit at 50 TPS:
        // 8-bit option should be ~50 TPS
        // 4-bit option should be rescaled up to ~100 TPS (8 / 4 * 50 = 100), NOT halved!
        #expect(opt8.estimatedTPS == 50)
        #expect(opt4.estimatedTPS >= 95 && opt4.estimatedTPS <= 105)
    }

    @Test("No-qualifying-option path does not falsely mark optimal with fitsBudget=true")
    func testNoQualifyingOptionFallback() {
        // Only 1GB available RAM for a 70B model with min TPS 100
        let resources = createMockResources(ramGB: 2, availableGB: 1, chip: "M1")
        let hugeModel = ModelDefinition(
            id: "huge/70B",
            name: "Huge 70B",
            family: .llama,
            task: .chat,
            modality: .text,
            ramMB: 40000,
            contextWindow: 4096,
            quantization: "4-bit",
            quality: 0.95,
            totalParameters: 70.0,
            activeParameters: 70.0
        )

        let result = ModelFitAnalyzer.evaluateQuantizations(
            model: hugeModel,
            resources: resources,
            tpsMin: 100.0,
            benchmarkProfile: HardwareBenchmarkProfile()
        )

        // None of the options fit
        #expect(result.options.allSatisfy { !$0.fitsBudget })
        #expect(result.options.allSatisfy { !$0.isOptimal })
        #expect(result.optimal.fitsBudget == false)
    }

    @Test("FitScorer speed calibration rescales when scoring model at different quant from measurement")
    func testFitScorerQuantRescaling() async {
        let resources = createMockResources(ramGB: 32, availableGB: 20, chip: "M2 Max")
        let benchmarkProfile = HardwareBenchmarkProfile(
            chipModel: "M2 Max",
            measurements: [
                "mlx-community/Qwen2.5-Coder-7B": BenchmarkMeasurement(
                    modelId: "mlx-community/Qwen2.5-Coder-7B",
                    chipModel: "M2 Max",
                    measuredTPS: 50.0,
                    measuredTTFT: 0.1,
                    quant: "4-bit" // Measured at 4-bit, 50 TPS
                )
            ]
        )

        let scorer = FitScorer(profile: resources, benchmarkProfile: benchmarkProfile)
        let model8bit = ModelDefinition(
            id: "mlx-community/Qwen2.5-Coder-7B",
            name: "Qwen 7B (8-bit)",
            family: .qwen,
            task: .code,
            modality: .text,
            ramMB: 8000,
            contextWindow: 32768,
            quantization: "8-bit",
            quality: 0.90,
            totalParameters: 7.0,
            activeParameters: 7.0
        )

        let score = await scorer.score(model8bit)
        #expect(score.isMeasured == true)
        // 4-bit 50 TPS rescaled to 8-bit should be 25 TPS, not 50 TPS
        #expect(score.estimatedTPS == 25)
        #expect(score.measuredTPS == 25.0)
    }

    // MARK: — Review round 1 (#51)

    @Test("A recorded profile survives a fresh process: the file is read back with its ISO-8601 dates")
    func testProfileRoundTripsThroughDisk() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let path = dir.appendingPathComponent("hardware_profile.json").path

        try await HardwareBenchmarkProfileStore(customPath: path).record(BenchmarkMeasurement(
            modelId: "org/model-a", chipModel: "M2 Max", measuredTPS: 50, measuredTTFT: 0.1, quant: "4bit"))
        // A second store is a second process: it has no cache and must decode the file.
        try await HardwareBenchmarkProfileStore(customPath: path).record(BenchmarkMeasurement(
            modelId: "org/model-b", chipModel: "M2 Max", measuredTPS: 60, measuredTTFT: 0.1, quant: "4bit"))

        let loaded = HardwareBenchmarkProfileStore.loadSync(path: path)
        #expect(loaded.measurements.count == 2, "the second record must not wipe the first")
        #expect(loaded.measurement(for: "org/model-a", currentChip: "M2 Max")?.measuredTPS == 50)
    }

    @Test("A run that produced (almost) no tokens is not recorded as a measurement")
    func testImplausibleMeasurementRejected() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = HardwareBenchmarkProfileStore(customPath: dir.appendingPathComponent("p.json").path)
        await #expect(throws: HardwareBenchmarkProfileError.self) {
            try await store.record(BenchmarkMeasurement(modelId: "org/m", chipModel: "M4", measuredTPS: 0, measuredTTFT: 0))
        }
        #expect(await store.load().measurements.isEmpty)
    }

    @Test("parseBits reads the explicit width and is not fooled by other digits")
    func testParseBits() {
        #expect(FitScorer.parseBits(from: "4-bit") == 4)
        #expect(FitScorer.parseBits(from: "8bit") == 8)
        #expect(FitScorer.parseBits(from: "1.58-bit") == 1.58)
        #expect(FitScorer.parseBits(from: "4-bit g128") == 4)
        #expect(FitScorer.parseBits(from: "bf16") == 16)
        #expect(FitScorer.parseBits(from: "NVFP4") == 4)
        #expect(FitScorer.parseBits(from: "mxfp8") == 8)
        #expect(FitScorer.parseBits(from: "Dynamic Quant (DQ)") == 4)
    }

    @Test("Bandwidth follows the published per-tier figures")
    func testBandwidthTable() {
        #expect(FitScorer.memoryBandwidthGBs(chip: "M4 Max", gpuName: "Apple M4 Max") == 546)
        #expect(FitScorer.memoryBandwidthGBs(chip: "M4 Pro", gpuName: "Apple M4 Pro") == 273)
        #expect(FitScorer.memoryBandwidthGBs(chip: "M2 Ultra", gpuName: "Apple M2 Ultra") == 800)
        #expect(FitScorer.memoryBandwidthGBs(chip: "M1", gpuName: "Apple M1") == 68)
        #expect(FitScorer.memoryBandwidthGBs(chip: "M3 Max", gpuName: "Apple M3 Max") == 400)
    }

    @Test("A measurement from a different tier of the same generation does not calibrate this one")
    func testTierMismatchRejected() {
        let profile = HardwareBenchmarkProfile(chipModel: "M2", measurements: [
            "org/m": BenchmarkMeasurement(modelId: "org/m", chipModel: "M2", measuredTPS: 30, measuredTTFT: 0.1)
        ])
        #expect(profile.measurement(for: "org/m", currentChip: "M2") != nil)
        #expect(profile.measurement(for: "org/m", currentChip: "M2 Ultra") == nil)
    }

    @Test("fit reports the grid's best quantization, or none when nothing fits")
    func testOptimalQuantizationWired() async {
        let small = ModelDatabase.allModels.first { $0.id == "mlx-community/Qwen3.5-4B-4bit" }!
        let roomy = FitScorer(profile: createMockResources(ramGB: 64, availableGB: 48, chip: "M2 Max"),
                              benchmarkProfile: HardwareBenchmarkProfile())
        #expect(await roomy.score(small).optimalQuantization == "8-bit")

        let huge = ModelDefinition(id: "org/huge-400b", name: "Huge 400B", family: small.family, task: small.task,
                                   modality: small.modality, ramMB: 220_000, contextWindow: 32768,
                                   quantization: "4-bit", quality: 0.95, totalParameters: 400, activeParameters: 400)
        let tight = FitScorer(profile: createMockResources(ramGB: 16, availableGB: 10, chip: "M2"),
                              benchmarkProfile: HardwareBenchmarkProfile())
        #expect(await tight.score(huge).optimalQuantization == nil)
    }

    @Test("fit and the quantization grid agree on the measured speed at the model's own width")
    func testScorerAndGridShareSpeedModel() async {
        let model = ModelDatabase.allModels.first { $0.id == "mlx-community/Qwen3.5-4B-4bit" }!
        let resources = createMockResources(ramGB: 32, availableGB: 24, chip: "M2 Max")
        // No quant recorded: both paths must take it as the model's own 4-bit.
        let profile = HardwareBenchmarkProfile(chipModel: "M2 Max", measurements: [
            model.id: BenchmarkMeasurement(modelId: model.id, chipModel: "M2 Max", measuredTPS: 80, measuredTTFT: 0.1)
        ])
        let fitTPS = await FitScorer(profile: resources, benchmarkProfile: profile).score(model).estimatedTPS
        let grid = ModelFitAnalyzer.evaluateQuantizations(model: model, resources: resources, benchmarkProfile: profile)
        #expect(fitTPS == 80)
        #expect(grid.options.first { $0.bits == 4 }?.estimatedTPS == 80)
        #expect(grid.options.first { $0.bits == 8 }?.estimatedTPS == 40)
    }

    @Test("KV cache estimate is pinned to GQA geometry (~0.25 GB per 4k tokens for a 7B model)")
    func testKVCacheCoefficient() {
        let gb = FitScorer.kvCacheFootprintGB(params: 7, contextTokens: 4096)
        #expect(abs(gb - 0.248) < 0.005)
        #expect(FitScorer.kvCacheMemoryMB(params: 7, contextTokens: 8192) > 500)
    }

    @Test("A profile that failed to decode is not cached and then overwritten")
    func testLoadDoesNotCacheUnreadableProfile() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("p.json")
        try Data("not json".utf8).write(to: url)

        let store = HardwareBenchmarkProfileStore(customPath: url.path)
        #expect(await store.load().measurements.isEmpty)
        await #expect(throws: (any Error).self) {
            try await store.record(BenchmarkMeasurement(modelId: "org/m", chipModel: "M4", measuredTPS: 50, measuredTTFT: 0.1))
        }
        #expect(try String(contentsOf: url, encoding: .utf8) == "not json")
    }
}

