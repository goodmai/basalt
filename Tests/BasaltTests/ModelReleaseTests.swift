import Foundation
import Testing
import MLX
@testable import BasaltCore
@testable import BasaltInference

@Suite("Loading, generating and dropping a real model leaves no MLX arrays behind (#251)", .serialized)
struct ModelReleaseTests {
    static var modelPath: String? {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1" else { return nil }
        return env["BASALT_E2E_RELEASE_MODEL_PATH"]
    }

    @Test("Three load-generate-drop rounds end with no live MLX memory",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_RELEASE_MODEL_PATH"))
    func roundsLeaveNothingBehind() async throws {
        let path = try #require(Self.modelPath)
        var residues: [Int] = []
        for _ in 1...3 {
            weak var released: MLXInferenceEngine?
            do {
                let engine = MLXInferenceEngine(promptCacheBytes: 1 << 30)
                let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
                try await orchestrator.loadModel(path: path)
                _ = try await orchestrator.generate(request: GenerationRequest(prompt: "Say hi.", maxTokens: 4, temperature: 0, cacheKey: "a"))
                await orchestrator.shutdown()
                released = engine
            }
            #expect(await GenerationGateTests.eventually(within: .seconds(10)) { released == nil })
            Memory.clearCache()
            residues.append(Memory.activeMemory)
        }
        #expect(residues.allSatisfy { $0 < 64 << 20 }, "live MLX memory after each round: \(residues.map { $0 >> 20 }) MB")
    }
}
