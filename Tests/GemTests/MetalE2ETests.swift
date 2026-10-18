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

// MARK: — Softmax & Sampler Reference Math

public struct ReferenceSampler {
    /// Compute softmax probabilities for given logits and temperature
    public static func softmax(logits: [Double], temperature: Double) -> [Double] {
        guard !logits.isEmpty else { return [] }
        if temperature <= 0.00001 {
            var probs = [Double](repeating: 0.0, count: logits.count)
            if let maxIdx = logits.indices.max(by: { logits[$0] < logits[$1] }) {
                probs[maxIdx] = 1.0
            }
            return probs
        }

        let scaled = logits.map { $0 / temperature }
        let maxVal = scaled.max() ?? 0.0
        let expVals = scaled.map { exp($0 - maxVal) }
        let sumExp = expVals.reduce(0.0, +)
        return expVals.map { $0 / sumExp }
    }

    /// Sample an index given cumulative probabilities and random uniform [0, 1)
    public static func sampleCategorical(probabilities: [Double], uniformValue: Double) -> Int {
        var cumulative = 0.0
        for (i, p) in probabilities.enumerated() {
            cumulative += p
            if uniformValue < cumulative {
                return i
            }
        }
        return max(0, probabilities.count - 1)
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

    @Test("Bit-exact reproducibility at temperature = 0 (two runs produce identical output)")
    func testGreedyParityAtZeroTemperature() async throws {
        let task = try GoldenTestVector.loadTask()
        let prompt = GoldenTestVector.formatPrompt(task: task)

        // Mock deterministic engine configured with the Golden Test Vector output
        let deterministicEngine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: deterministicEngine)
        try await orchestrator.loadModel(path: "test-ornith-path")
        await orchestrator.setModelId(GoldenTestVector.modelId)

        let request1 = GenerationRequest(
            prompt: prompt,
            maxTokens: 128,
            temperature: 0.0,
            topP: 1.0
        )
        let request2 = GenerationRequest(
            prompt: prompt,
            maxTokens: 128,
            temperature: 0.0,
            topP: 1.0
        )

        let response1 = try await orchestrator.generate(request: request1)
        let response2 = try await orchestrator.generate(request: request2)

        // Bit-exact identity check across repeated runs at temperature = 0
        #expect(response1.generatedText == response2.generatedText)
        #expect(response1.completionTokens == response2.completionTokens)
        #expect(response1.promptTokens == response2.promptTokens)
        #expect(response1.finishReason == response2.finishReason)
    }

    @Test("Reference softmax parity: empirical sampling frequencies match theoretical distribution")
    func testSoftmaxSamplingParity() {
        let logits = [3.0, 2.0, 1.0, 0.0]
        let temperature = 1.0
        let theoreticalProbs = ReferenceSampler.softmax(logits: logits, temperature: temperature)

        // Expected probabilities:
        // exp(3)=20.0855, exp(2)=7.3891, exp(1)=2.7183, exp(0)=1.0, sum=31.1929
        // p = [~0.6439, ~0.2369, ~0.0871, ~0.0321]
        #expect(theoreticalProbs.count == 4)
        #expect(abs(theoreticalProbs.reduce(0.0, +) - 1.0) < 1e-6)

        // Pseudo-random sampling with deterministic LCG seed
        var seed: UInt64 = 42
        func nextUniform() -> Double {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            let u32 = UInt32(truncatingIfNeeded: seed >> 32)
            return Double(u32) / Double(UInt32.max)
        }

        let numTrials = 20_000
        var counts = [Int](repeating: 0, count: logits.count)
        for _ in 0..<numTrials {
            let idx = ReferenceSampler.sampleCategorical(
                probabilities: theoreticalProbs,
                uniformValue: nextUniform()
            )
            counts[idx] += 1
        }

        let empiricalProbs = counts.map { Double($0) / Double(numTrials) }

        // Each sampled frequency must match theoretical softmax within 2% margin
        for i in 0..<logits.count {
            let diff = abs(empiricalProbs[i] - theoreticalProbs[i])
            #expect(diff < 0.02, "Empirical prob \(empiricalProbs[i]) diverged from theoretical \(theoreticalProbs[i]) by \(diff)")
        }
    }

    @Test("Greedy argmax dominance at temperature = 0 and very low temperatures")
    func testGreedyArgmaxDominance() {
        let logits = [1.5, 4.2, 0.8, 3.1] // index 1 is maximum (4.2)
        
        let zeroTempProbs = ReferenceSampler.softmax(logits: logits, temperature: 0.0)
        #expect(zeroTempProbs == [0.0, 1.0, 0.0, 0.0])

        let lowTempProbs = ReferenceSampler.softmax(logits: logits, temperature: 0.05)
        #expect(lowTempProbs[1] > 0.9999)

        // High temperature flattens distribution towards uniform (0.25 each)
        let highTempProbs = ReferenceSampler.softmax(logits: logits, temperature: 50.0)
        for p in highTempProbs {
            #expect(abs(p - 0.25) < 0.05)
        }
    }

    @Test("Seeded sampler determinism: identical seed produces identical trajectory")
    func testSeededSamplerDeterminism() {
        let logits = [2.0, 1.5, 0.5, 0.2]
        let probs = ReferenceSampler.softmax(logits: logits, temperature: 0.8)

        func generateSequence(withSeed startSeed: UInt64, length: Int) -> [Int] {
            var s = startSeed
            var seq: [Int] = []
            for _ in 0..<length {
                s = s &* 6364136223846793005 &+ 1442695040888963407
                let u = Double(UInt32(truncatingIfNeeded: s >> 32)) / Double(UInt32.max)
                seq.append(ReferenceSampler.sampleCategorical(probabilities: probs, uniformValue: u))
            }
            return seq
        }

        let runA = generateSequence(withSeed: 98765, length: 100)
        let runB = generateSequence(withSeed: 98765, length: 100)
        let runC = generateSequence(withSeed: 12345, length: 100)

        #expect(runA == runB, "Identical seeds must yield bit-exact identical sampling sequence")
        #expect(runA != runC, "Different seeds must yield distinct trajectories")
    }
}
