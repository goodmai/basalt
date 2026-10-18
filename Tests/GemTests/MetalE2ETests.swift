import Testing
import Foundation
@testable import GemCore

// MARK: — Golden Test Vector & Reproducibility Specification
// Golden Vector: ornith-ai/Ornith-1.5-9B-MLX-4bit on ARC-AGI Task 1

public struct GoldenTestVector {
    public static let modelId = "ornith-ai/Ornith-1.5-9B-MLX-4bit"
    public static let taskRelativePath = "benchmarks/arc-agi/compare-tasks/task_1.json"

    public struct ArcAgiTask: Codable, Sendable {
        public struct Example: Codable, Sendable {
            public let input: [[Int]]
            public let output: [[Int]]
        }
        public let train: [Example]
        public let test: [Example]
    }

    public static func loadTask() throws -> ArcAgiTask {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Tests/GemTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // repo root
        let taskURL = repoRoot.appendingPathComponent(taskRelativePath)
        let data = try Data(contentsOf: taskURL)
        return try JSONDecoder().decode(ArcAgiTask.self, from: data)
    }

    public static func formatPrompt(task: ArcAgiTask) -> String {
        var p = "Task: ARC-AGI pattern transformation grid reasoning.\n"
        for (idx, eg) in task.train.enumerated() {
            p += "Train \(idx + 1) Input: \(eg.input) -> Output: \(eg.output)\n"
        }
        if let firstTest = task.test.first {
            p += "Test Input: \(firstTest.input)\nTest Output:"
        }
        return p
    }

    public static let expectedOutputGrid: [[Int]] = [
        [3, 2, 3, 2, 3, 2],
        [7, 8, 7, 8, 7, 8],
        [2, 3, 2, 3, 2, 3],
        [8, 7, 8, 7, 8, 7],
        [3, 2, 3, 2, 3, 2],
        [7, 8, 7, 8, 7, 8]
    ]

    public static var expectedOutputString: String {
        " \(expectedOutputGrid)"
    }
}


// MARK: — SamplerAndGreedyParityTests Suite

@Suite("Sampler & Greedy Parity Tests")
struct SamplerAndGreedyParityTests {

    @Test("Golden Test Vector loads ARC-AGI Task 1 and matches expected grid")
    func testGoldenVectorArcAgi1Load() throws {
        let task = try GoldenTestVector.loadTask()
        #expect(task.train.count == 2)
        #expect(task.test.count == 1)
        
        let testEg = task.test[0]
        #expect(testEg.input == [[3, 2], [7, 8]])
        #expect(testEg.output == GoldenTestVector.expectedOutputGrid)

        let prompt = GoldenTestVector.formatPrompt(task: task)
        #expect(prompt.contains("ARC-AGI pattern transformation"))
        #expect(prompt.contains("Test Input: [[3, 2], [7, 8]]"))
    }

    @Test("Golden model ornith-ai/Ornith-1.5-9B-MLX-4bit is registered in ModelDatabase")
    func testOrnithModelDatabaseRegistration() {
        let models = ModelDatabase.allModels
        let ornith = models.first { $0.id == GoldenTestVector.modelId }
        #expect(ornith != nil)
        #expect(ornith?.name.contains("Ornith 1.5 9B") == true)
        #expect(ornith?.task == .code)
    }

    /// The real-engine proof of the acceptance criterion "two runs at temperature = 0 produce
    /// identical output". Opt-in: it loads the golden model (several GB) and runs on Metal, so it
    /// needs BASALT_E2E=1 and the model in the local HF cache (or BASALT_E2E_MODEL_PATH).
    static var goldenModelPath: String? {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1" else { return nil }
        if let path = env["BASALT_E2E_MODEL_PATH"] { return path }
        return ModelCache.isDownloaded(repoId: GoldenTestVector.modelId)
            ? ModelCache.cacheDir(for: GoldenTestVector.modelId).path : nil
    }

    @Test("MLX engine: two greedy runs on the golden vector are bit-identical",
          .enabled(if: SamplerAndGreedyParityTests.goldenModelPath != nil,
                   "set BASALT_E2E=1 and cache \(GoldenTestVector.modelId) or set BASALT_E2E_MODEL_PATH"))
    func testRealEngineGreedyParity() async throws {
        let path = try #require(Self.goldenModelPath)
        let prompt = GoldenTestVector.formatPrompt(task: try GoldenTestVector.loadTask())
        let orchestrator = ModelOrchestratorActor(engine: MLXInferenceEngine())
        try await orchestrator.loadModel(path: path)

        let request = GenerationRequest(prompt: prompt, maxTokens: 64, temperature: 0.0, topP: 1.0, enableThinking: false)
        let first = try await orchestrator.generate(request: request)
        let second = try await orchestrator.generate(request: request)

        #expect(!first.generatedText.isEmpty)
        #expect(first.generatedText == second.generatedText)
        #expect(first.completionTokens == second.completionTokens)
    }
}
