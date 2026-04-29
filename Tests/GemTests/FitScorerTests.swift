import Testing
import Foundation
@testable import GemCore

@Suite("FitScorer Tests - Epic 16.7")
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
            quality: 0.85
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
            quality: 0.92
        )
        
        let score = await scorer.score(model)
        #expect(score.fitLevel == .tooTight)
        #expect(score.score < 50.0)
    }
}
