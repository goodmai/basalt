import Foundation

public struct BenchmarkDataPoint: Codable {
    public let contextSize: Int
    public let avgTPS: Double
    public let avgTTFT: Double
    public let memoryActiveMB: Int
}

public struct ContextDegradationReport: Codable {
    public let modelPath: String?
    public let dataPoints: [BenchmarkDataPoint]
}

public struct ContextDegradationProfiler {
    public let orchestrator: ModelOrchestratorActor
    
    public init(orchestrator: ModelOrchestratorActor) {
        self.orchestrator = orchestrator
    }
    
    public func runBenchmark(contextSizes: [Int], tokensToGenerate: Int, iterations: Int = 1) async throws -> ContextDegradationReport {
        var points: [BenchmarkDataPoint] = []
        
        for size in contextSizes {
            // Generates a mock prompt containing approx `size` words.
            // Simple space separation to mimic tokens roughly for mock and baseline.
            // A more exact implementation would use the tokenizer.
            let prompt = String(repeating: "word ", count: size)
            
            var tpsSum: Double = 0
            var ttftSum: Double = 0
            var memoryMB: Int = 0
            
            // Optional warmup
            let reqWarmup = GenerationRequest(prompt: prompt, maxTokens: tokensToGenerate)
            _ = try await orchestrator.generate(request: reqWarmup)
            
            for _ in 0..<iterations {
                let req = GenerationRequest(prompt: prompt, maxTokens: tokensToGenerate)
                let r = try await orchestrator.generate(request: req)
                
                tpsSum += r.tokensPerSecond
                ttftSum += r.timeToFirstToken
                memoryMB = r.memory.activeBytes / 1_048_576
            }
            
            let avgTPS = tpsSum / Double(iterations)
            let avgTTFT = ttftSum / Double(iterations)
            
            points.append(BenchmarkDataPoint(contextSize: size, avgTPS: avgTPS, avgTTFT: avgTTFT, memoryActiveMB: memoryMB))
        }
        
        return ContextDegradationReport(modelPath: "Unknown", dataPoints: points) // TODO: expose modelPath from orchestrator
    }
}
