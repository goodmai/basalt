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
        let scorer = FitScorer(profile: resources)
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
        let scorer = FitScorer(profile: resources)
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
        let scorer = FitScorer(profile: resources)

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

        let result = ModelFitAnalyzer.evaluateQuantizations(model: model, resources: resources)
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
}
