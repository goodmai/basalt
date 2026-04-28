import Testing
import Foundation
@testable import GemmaServerCore

/// Epic 16.7: fit Command - Hardware Profiling & Model Recommendations
/// Following TDD: Write tests FIRST, then implement
@Suite("ModelFitAnalyzer Tests - Epic 16.7")
struct ModelFitAnalyzerTests {
    
    // MARK: - Fit Score Calculation Tests
    
    @Test("Calculate fit score for perfect match")
    func testPerfectFitScore() {
        let resources = createMockResources(ramGB: 32, gpuGB: 64)
        let model = createMockModel(name: "Qwen3.5-4B", ramMB: 2300, recommendedRAM: 8)
        
        let score = ModelFitAnalyzer.calculateFitScore(
            model: model,
            resources: resources
        )
        
        // Should be very high (90+) for small model on powerful hardware
        #expect(score > 90.0)
        #expect(score <= 100.0)
    }
    
    @Test("Calculate fit score for tight fit")
    func testTightFitScore() {
        let resources = createMockResources(ramGB: 8, gpuGB: 8)
        let model = createMockModel(name: "Qwen3.6-27B", ramMB: 14500, recommendedRAM: 32)
        
        let score = ModelFitAnalyzer.calculateFitScore(
            model: model,
            resources: resources
        )
        
        // Should be low (< 50) for large model on limited hardware
        #expect(score < 50.0)
        #expect(score >= 0.0)
    }
    
    @Test("Calculate fit score for impossible fit")
    func testImpossibleFitScore() {
        let resources = createMockResources(ramGB: 4, gpuGB: 4)
        let model = createMockModel(name: "Llama-70B", ramMB: 40000, recommendedRAM: 64)
        
        let score = ModelFitAnalyzer.calculateFitScore(
            model: model,
            resources: resources
        )
        
        // Should be very low (< 20) or even 0
        #expect(score < 20.0)
    }
    
    // MARK: - Fit Category Tests
    
    @Test("Categorize as Perfect fit")
    func testPerfectCategory() {
        let category = ModelFitAnalyzer.categorizeFit(score: 95.0)
        #expect(category == .perfect)
    }
    
    @Test("Categorize as Good fit")
    func testGoodCategory() {
        let category = ModelFitAnalyzer.categorizeFit(score: 75.0)
        #expect(category == .good)
    }
    
    @Test("Categorize as Tight fit")
    func testTightCategory() {
        let category = ModelFitAnalyzer.categorizeFit(score: 55.0)
        #expect(category == .tight)
    }
    
    @Test("Categorize as TooTight fit")
    func testTooTightCategory() {
        let category = ModelFitAnalyzer.categorizeFit(score: 30.0)
        #expect(category == .tooTight)
    }
    
    // MARK: - Model Analysis Tests
    
    @Test("Analyze multiple models and sort by fit score")
    func testAnalyzeModels() {
        let resources = createMockResources(ramGB: 16, gpuGB: 32)
        let models = [
            createMockModel(name: "Small", ramMB: 2000, recommendedRAM: 4),
            createMockModel(name: "Medium", ramMB: 8000, recommendedRAM: 16),
            createMockModel(name: "Large", ramMB: 30000, recommendedRAM: 64)
        ]
        
        let analysis = ModelFitAnalyzer.analyzeModels(models, resources: resources)
        
        // Should have 3 results
        #expect(analysis.count == 3)
        
        // Should be sorted by score (best first)
        for i in 0..<(analysis.count - 1) {
            #expect(analysis[i].fitScore >= analysis[i + 1].fitScore)
        }
    }
    
    @Test("Filter models by minimum fit score")
    func testFilterByMinScore() {
        let resources = createMockResources(ramGB: 16, gpuGB: 32)
        let models = [
            createMockModel(name: "Good", ramMB: 8000, recommendedRAM: 16),
            createMockModel(name: "Bad", ramMB: 50000, recommendedRAM: 128)
        ]
        
        let analysis = ModelFitAnalyzer.analyzeModels(
            models,
            resources: resources,
            minScore: 50.0
        )
        
        // Should filter out bad fits
        #expect(analysis.allSatisfy { $0.fitScore >= 50.0 })
    }
    
    // MARK: - Recommendation Tests
    
    @Test("Get top recommendations")
    func testGetTopRecommendations() {
        let resources = createMockResources(ramGB: 32, gpuGB: 64)
        let models = (1...10).map { i in
            createMockModel(name: "Model\(i)", ramMB: i * 2000, recommendedRAM: i * 4)
        }
        
        let recommendations = ModelFitAnalyzer.getTopRecommendations(
            models,
            resources: resources,
            limit: 3
        )
        
        #expect(recommendations.count == 3)
        
        // First should have highest score
        if recommendations.count >= 2 {
            #expect(recommendations[0].fitScore >= recommendations[1].fitScore)
        }
    }
    
    // MARK: - TPS Estimation Tests
    
    @Test("Estimate TPS based on model size and hardware")
    func testEstimateTPS() {
        let resources = createMockResources(ramGB: 32, gpuGB: 64)
        
        // Small model should have high TPS
        let smallTPS = ModelFitAnalyzer.estimateTPS(
            modelSizeMB: 2300,
            resources: resources
        )
        #expect(smallTPS > 50)
        
        // Large model should have lower TPS
        let largeTPS = ModelFitAnalyzer.estimateTPS(
            modelSizeMB: 30000,
            resources: resources
        )
        #expect(largeTPS < smallTPS)
    }
    
    // MARK: - Helper Functions
    
    private func createMockResources(ramGB: Int, gpuGB: Int) -> SystemProfiler.SystemResources {
        SystemProfiler.SystemResources(
            totalRAM: Int64(ramGB) * 1_073_741_824,
            availableRAM: Int64(ramGB) * 1_073_741_824 * 8 / 10, // 80% available
            cpuCores: 10,
            gpuName: "Apple M2 Max",
            gpuMemory: Int64(gpuGB) * 1_073_741_824,
            diskSpace: 500_000_000_000,
            osVersion: "14.0",
            chipModel: "M2 Max"
        )
    }
    
    private func createMockModel(name: String, ramMB: Int, recommendedRAM: Int) -> ModelInfo {
        ModelInfo(
            id: "mlx-community/\(name)",
            name: name,
            sizeMB: ramMB,
            recommendedRAMGB: recommendedRAM,
            description: "Test model"
        )
    }
}

@Suite("FitCommand Tests - Epic 16.7")
struct FitCommandTests {
    
    @Test("FitCommand initializes with default options")
    func testFitCommandInit() {
        let _ = FitCommand()
        // Just verify it compiles and initializes
    }
    
    @Test("Format fit category with emoji")
    func testFormatFitCategory() {
        #expect(FitCommand.formatCategory(.perfect).contains("🟢"))
        #expect(FitCommand.formatCategory(.good).contains("🟡"))
        #expect(FitCommand.formatCategory(.tight).contains("🟠"))
        #expect(FitCommand.formatCategory(.tooTight).contains("🔴"))
    }
    
    @Test("Format RAM size in human readable format")
    func testFormatRAMSize() {
        #expect(FitCommand.formatRAM(2300).contains("2.2 GB"))  // 2300/1024 = 2.246
        #expect(FitCommand.formatRAM(14500).contains("14.2 GB")) // 14500/1024 = 14.16
        #expect(FitCommand.formatRAM(500).contains("500 MB"))
    }
}
