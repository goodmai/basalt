import Foundation
import ArgumentParser
import GemmaServerCore

struct Benchmark: AsyncParsableCommand {
    @Option(name: .shortAndLong, help: "Model path")
    var model: String

    @Option(name: .shortAndLong, help: "Number of iterations")
    var iterations: Int = 5

    @Option(name: .shortAndLong, help: "Prompt to use")
    var prompt: String = "Explain the importance of open source models."

    @Option(name: .shortAndLong, help: "Max tokens to generate")
    var tokens: Int = 100

    func run() async throws {
        print("🚀 Starting benchmark for model: \(model)")
        
        let engine = MLXInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine)
        
        print("⏳ Loading model...")
        try await orchestrator.loadModel(path: model)
        print("✅ Model loaded.\n")
        
        var totalTPS: Double = 0
        var totalTTFT: Double = 0
        var totalGenTime: Double = 0
        
        for i in 1...iterations {
            print("Iteration \(i)/\(iterations)... ", terminator: "")
            fflush(stdout)
            
            let request = GenerationRequest(prompt: prompt, maxTokens: tokens)
            let response = try await orchestrator.generate(request: request)
            
            totalTPS += response.tokensPerSecond
            totalTTFT += response.timeToFirstToken
            totalGenTime += response.generationTime
            
            print(String(format: "TPS: %.2f | TTFT: %.3fs | Total: %.2fs", 
                         response.tokensPerSecond, 
                         response.timeToFirstToken, 
                         response.generationTime))
        }
        
        let avgTPS = totalTPS / Double(iterations)
        let avgTTFT = totalTTFT / Double(iterations)
        let avgGenTime = totalGenTime / Double(iterations)
        
        print("\n📊 Results (Average over \(iterations) iterations):")
        print(String(format: "  Average TPS:  %.2f tokens/s", avgTPS))
        print(String(format: "  Average TTFT: %.3f s", avgTTFT))
        print(String(format: "  Average Time: %.2f s", avgGenTime))
        print("")
    }
}

Benchmark.main()
