import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Adaptive thinking budget on real reasoning models: accuracy and thinking tokens with the flag off and on (#41)", .serialized)
struct AdaptiveThinkingRealModelTests {
    struct Problem: Sendable {
        let prompt: String
        let answer: String
    }

    struct Outcome: Sendable {
        let solved: Bool
        let thinkingTokens: Int
    }

    static let env = ProcessInfo.processInfo.environment

    static var models: [String] {
        guard env["BASALT_E2E"] == "1" else { return [] }
        return ["BASALT_E2E_THINKING_QWEN_MODEL_PATH", "BASALT_E2E_THINKING_LFM_MODEL_PATH"].compactMap { env[$0] }.filter { !$0.isEmpty }
    }

    static var maxTokens: Int { env["BASALT_E2E_THINKING_MAX_TOKENS"].flatMap { Int($0) } ?? 4096 }

    static let pool = [
        Problem(prompt: "What is 127 + 458?", answer: "585"),
        Problem(prompt: "What is 17 times 23?", answer: "391"),
        Problem(prompt: "How many days are there in a leap year?", answer: "366"),
        Problem(prompt: "What is the capital city of Australia?", answer: "canberra"),
        Problem(prompt: "A shop sells pencils at 3 for $1.20. How much do 15 pencils cost, in dollars?", answer: "6"),
        Problem(prompt: "A train travels 180 km in 2.5 hours. What is its average speed in km/h?", answer: "72"),
        Problem(prompt: "Solve this puzzle: if 5 machines make 5 widgets in 5 minutes, how many minutes do 100 machines need to make 100 widgets?", answer: "5"),
        Problem(prompt: "Alice is twice as old as Bob. In 10 years the sum of their ages will be 50. How old is Alice now?", answer: "20"),
        Problem(prompt: "Step by step, find the remainder when 2^100 is divided by 7.", answer: "2"),
        Problem(prompt: "Solve this puzzle step by step: a farmer has chickens and cows, 30 heads and 74 legs. How many cows are there?", answer: "7"),
        Problem(prompt: "Step by step, using inclusion-exclusion, derive how many integers n with 1 <= n <= 1000 are divisible by 3 or 5 but not by 15, and prove the count.", answer: "401"),
        Problem(prompt: "Implement an algorithm in your head and state its complexity: in how many ways can you climb a staircase of 20 steps taking 1 or 2 steps at a time?", answer: "10946"),
    ]

    static func answers(_ text: String, _ expected: String) -> Bool {
        let line = text.lowercased().replacingOccurrences(of: ",", with: "").split(separator: "\n").last { !$0.allSatisfy(\.isWhitespace) } ?? ""
        let words = line.split { !$0.isLetter && !$0.isNumber && $0 != "." }.map { $0.trimmingCharacters(in: CharacterSet(charactersIn: ".")) }
        return words.contains { $0 == expected || (Double($0) != nil && Double($0) == Double(expected)) }
    }

    @Test("A model solves the problems answered with the flag off as well with it on, and the log states the thinking tokens it saved",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_THINKING_QWEN_MODEL_PATH and/or BASALT_E2E_THINKING_LFM_MODEL_PATH"), arguments: models)
    func compare(path: String) async throws {
        let log = try await ModelRunLog("adaptive-thinking", modelPath: path)
        defer { log.releaseGPU() }
        let orchestrator = ModelOrchestratorActor(engine: MLXInferenceEngine(), maxTokens: Self.maxTokens)
        try await log.step("load") { try await orchestrator.loadModel(path: path) }
        var runs: [Bool: [Outcome]] = [:]
        for adaptive in [false, true] {
            await orchestrator.setAdaptiveThinkingBudget(adaptive)
            runs[adaptive] = try await Self.solve(Self.pool, on: orchestrator, adaptive: adaptive, log: log)
        }
        let off = try #require(runs[false]), on = try #require(runs[true])
        let lost = zip(Self.pool, zip(off, on)).filter { $1.0.solved && !$1.1.solved }.map(\.0.answer)
        let (spentOff, spentOn) = (off.map(\.thinkingTokens).reduce(0, +), on.map(\.thinkingTokens).reduce(0, +))
        log.record("summary", String(
            format: "solved off %d/%d on %d/%d, thinking tokens off %d on %d (%.1f%% saved), lost %@",
            off.count(where: \.solved), off.count, on.count(where: \.solved), on.count, spentOff, spentOn,
            spentOff > 0 ? (1 - Double(spentOn) / Double(spentOff)) * 100 : 0, lost.isEmpty ? "none" : lost.joined(separator: " ")))
        #expect(lost.isEmpty, "problems solved without the flag and lost with it: \(lost); see \(log.url.path)")
    }

    static func solve(_ pool: [Problem], on orchestrator: ModelOrchestratorActor, adaptive: Bool, log: ModelRunLog) async throws -> [Outcome] {
        var outcomes: [Outcome] = []
        for (index, problem) in pool.enumerated() {
            let request = GenerationRequest(
                prompt: "", maxTokens: maxTokens, temperature: 0, topP: 1,
                messages: [ChatMessageInput(role: .user, text: problem.prompt + " Give the final answer alone on the last line.")])
            let planned = ThinkingDifficulty.adaptiveBudget(for: request)
            let mode = adaptive ? "on" : "off"
            let response = try await log.step("\(mode) #\(index)") { try await orchestrator.generate(request: request) }
            let outcome = Outcome(solved: answers(response.generatedText, problem.answer), thinkingTokens: response.reasoningTokens ?? 0)
            log.record(mode, String(
                format: "#%d difficulty %@ budget %@ thinking %d completion %d finish %@ %.1f s %@",
                index, ThinkingDifficulty(request: request).rawValue, adaptive ? planned.map { String($0.budget) } ?? "none" : "none",
                outcome.thinkingTokens, response.completionTokens, response.finishReason.rawValue, response.generationTime,
                outcome.solved ? "solved" : "wrong (want \(problem.answer))"))
            outcomes.append(outcome)
        }
        return outcomes
    }
}
