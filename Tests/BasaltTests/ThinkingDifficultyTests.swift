import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltCLI

@Suite("Difficulty-based thinking budget behind --experimental-adaptive-thinking-budget (#41)")
struct ThinkingDifficultyTests {
    static func request(_ text: String, maxTokens: Int = 4096) -> GenerationRequest {
        GenerationRequest(prompt: "", maxTokens: maxTokens, messages: [ChatMessageInput(role: .user, text: text)])
    }

    static let longPrompt = Array(repeating: "word", count: 60).joined(separator: " ")

    @Test("The estimate is fixed by the text: no marker and a short prompt is easy, one marker or 50+ words is medium, three points are hard", arguments: [
        ("What is 127 + 458?", ThinkingDifficulty.easy),
        ("Solve this puzzle: which box holds the key?", .medium),
        (longPrompt, .medium),
        ("Prove the theorem step by step.", .hard),
        ("Implement the algorithm and state its complexity.", .hard),
    ])
    func estimateIsDeterministic(text: String, expected: ThinkingDifficulty) {
        #expect(ThinkingDifficulty(text: text) == expected)
        #expect(ThinkingDifficulty(text: text) == ThinkingDifficulty(text: text))
    }

    @Test("The last user message is scored, and a request with tools is hard")
    func requestScoring() {
        let turns = GenerationRequest(prompt: "", maxTokens: 4096, messages: [
            ChatMessageInput(role: .user, text: "Prove the theorem step by step."),
            ChatMessageInput(role: .assistant, text: "Done."),
            ChatMessageInput(role: .user, text: "Thanks, what is 2 + 2?"),
        ])
        #expect(ThinkingDifficulty(request: turns) == .easy)
        #expect(ThinkingDifficulty(request: GenerationRequest(prompt: "Prove it step by step by algorithm")) == .hard)
        let tool = ToolDefinition(name: "lookup")
        #expect(ThinkingDifficulty(request: GenerationRequest(prompt: "hi", tools: [tool])) == .hard)
    }

    @Test("Easy is 512 thinking tokens, medium 2048, hard is uncapped")
    func budgets() {
        #expect(ThinkingDifficulty.adaptiveBudget(for: Self.request("What is 127 + 458?"))?.budget == 512)
        #expect(ThinkingDifficulty.adaptiveBudget(for: Self.request("Solve this puzzle."))?.budget == 2048)
        #expect(ThinkingDifficulty.adaptiveBudget(for: Self.request("Prove the theorem step by step.")) == nil)
    }

    @Test("A request's own thinking_budget, reasoning_effort or disabled thinking wins, and a max_tokens too small for the budget leaves it unset")
    func explicitControlsWin() {
        var budgeted = Self.request("What is 127 + 458?")
        budgeted.thinkingBudget = 100
        var effort = Self.request("What is 127 + 458?")
        effort.reasoningEffort = "low"
        var disabled = Self.request("What is 127 + 458?")
        disabled.enableThinking = false
        for request in [budgeted, effort, disabled, Self.request("What is 127 + 458?", maxTokens: 766)] {
            #expect(ThinkingDifficulty.adaptiveBudget(for: request) == nil)
        }
        #expect(ThinkingDifficulty.adaptiveBudget(for: Self.request("What is 127 + 458?", maxTokens: 767))?.budget == 512)
    }

    @Test("The orchestrator sets the budget only with the flag on, and keeps a request's own thinking_budget")
    func orchestratorAppliesTheFlag() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine)
        try await orchestrator.loadModel(path: "test-model")
        _ = try await orchestrator.generate(request: Self.request("What is 127 + 458?"))
        #expect(await engine.lastRequest?.thinkingBudget == nil)

        await orchestrator.setAdaptiveThinkingBudget(true)
        _ = try await orchestrator.generate(request: Self.request("What is 127 + 458?"))
        #expect(await engine.lastRequest?.thinkingBudget == 512)
        var own = Self.request("What is 127 + 458?")
        own.thinkingBudget = 64
        _ = try await orchestrator.generate(request: own)
        #expect(await engine.lastRequest?.thinkingBudget == 64)
    }

    @Test("Off unless --experimental-adaptive-thinking-budget is given")
    func flagIsOptIn() throws {
        #expect(try !ServeCommand.parse([]).experimentalAdaptiveThinkingBudget)
        #expect(try ServeCommand.parse(["--experimental-adaptive-thinking-budget"]).experimentalAdaptiveThinkingBudget)
    }
}
