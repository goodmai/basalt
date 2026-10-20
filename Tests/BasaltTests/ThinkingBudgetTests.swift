import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

struct ThinkTokenizer: MLXLMCommon.Tokenizer {
    static let endOfTurn = 249
    static let open = 250
    static let close = 251
    static let markers = ["<think>": open, "</think>": close]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var ids: [Int] = []
        var rest = Substring(text)
        while let first = rest.first {
            if let (marker, id) = Self.markers.first(where: { rest.hasPrefix($0.key) }) {
                ids.append(id)
                rest = rest.dropFirst(marker.count)
            } else {
                ids += String(first).unicodeScalars.map { Int($0.value) % Self.endOfTurn }
                rest = rest.dropFirst()
            }
        }
        return ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { id in
            switch id {
            case Self.open: "<think>"
            case Self.close: "</think>"
            default: String(UnicodeScalar(UInt8(id)))
            }
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { Self.markers[token] }
    func convertIdToToken(_ id: Int) -> String? { decode(tokenIds: [id], skipSpecialTokens: false) }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap { encode(text: $0["content"] as? String ?? "", addSpecialTokens: false) }
            + [Self.endOfTurn, Self.open]
    }
}

struct ThinkInputProcessor: UserInputProcessor {
    func prepare(input: UserInput) throws -> LMInput {
        let messages = DefaultMessageGenerator().generate(from: input)
        let tokens = try ThinkTokenizer().applyChatTemplate(messages: messages, tools: nil, additionalContext: nil)
        return LMInput(tokens: MLXArray(tokens.map(Int32.init)))
    }
}

enum ThinkingModel {
    static let transition = QwenReasoningProtocol.qwen3.budgetTransition!.beforeEndDelimiter
    static let prompt = "Solve the puzzle and explain the rule in one line."

    static func engine(reasoning: ReasoningConfig? = QwenReasoningProtocol.tagged,
                       profile: any ModelProfile = QwenProfile()) async throws -> MLXInferenceEngine {
        let engine = MLXInferenceEngine()
        await engine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-qwen", reasoningConfig: reasoning),
            model: try withRandomState(MLXRandom.RandomState(seed: 56)) { try TinyModel.make() },
            processor: ThinkInputProcessor(),
            tokenizer: ThinkTokenizer())), profile: profile)
        return engine
    }

    static func request(budget: Int?, maxTokens: Int, key: String? = nil, effort: String? = nil) -> GenerationRequest {
        GenerationRequest(prompt: prompt, maxTokens: maxTokens, temperature: 0, reasoningEffort: effort,
                          cacheKey: key, thinkingBudget: budget)
    }
}

func scalars(_ text: String?) -> [Unicode.Scalar] {
    Array((text ?? "").unicodeScalars)
}

@Suite("Thinking budget on a tiny random model (#105)", .serialized)
struct ThinkingBudgetEngineTests {

    @Test("The forced Qwen transition lands in reasoning after exactly the budget, </think> closes it, the reasoning before it is the unbudgeted greedy one")
    func budgetForcesTheTransitionIntoReasoning() async throws {
        let engine = try await ThinkingModel.engine()
        let budget = 6
        let free = try await engine.generate(request: ThinkingModel.request(budget: nil, maxTokens: 200))
        let budgeted = try await engine.generate(request: ThinkingModel.request(budget: budget, maxTokens: 200))

        #expect(scalars(free.reasoningText).count > budget + 20)
        let reasoning = try #require(budgeted.reasoningText)
        #expect(scalars(reasoning) == Array(scalars(free.reasoningText).prefix(budget)) + scalars(ThinkingModel.transition))
        #expect(!budgeted.generatedText.isEmpty)
        #expect(!budgeted.generatedText.contains("Considering"))
        #expect(!budgeted.generatedText.contains("</think>"))
        #expect(budgeted.reasoningTokens == budget + scalars(ThinkingModel.transition).count)
    }

    @Test("The budget holds on the prompt-cache path, and a cached rerun answers the same")
    func budgetHoldsWithThePromptCache() async throws {
        let engine = try await ThinkingModel.engine()
        let first = try await engine.generate(request: ThinkingModel.request(budget: 6, maxTokens: 200, key: "ns"))
        let again = try await engine.generate(request: ThinkingModel.request(budget: 6, maxTokens: 200, key: "ns"))
        let uncached = try await engine.generate(request: ThinkingModel.request(budget: 6, maxTokens: 200))

        #expect(again.cachedPromptTokens > 0)
        #expect(first.reasoningText?.hasSuffix(ThinkingModel.transition) == true)
        #expect(again.reasoningText == first.reasoningText)
        #expect(again.generatedText == first.generatedText)
        #expect(first.reasoningText == uncached.reasoningText)
        #expect(first.generatedText == uncached.generatedText)
    }

    @Test("A budget that max_tokens cannot hold is clamped so the transition and an answer still fit")
    func oversizedBudgetIsClamped() async throws {
        let engine = try await ThinkingModel.engine()
        let result = try await engine.generate(request: ThinkingModel.request(budget: 31_999, maxTokens: 200))
        #expect(result.reasoningText?.hasSuffix(ThinkingModel.transition) == true)
        #expect(!result.generatedText.isEmpty)
    }

    @Test("max_tokens that cannot hold the transition is a 400")
    func tooFewTokensForTheTransitionIsRejected() async throws {
        let engine = try await ThinkingModel.engine()
        await #expect {
            _ = try await engine.generate(request: ThinkingModel.request(budget: 10, maxTokens: 100))
        } throws: { error in
            guard case .invalidRequestStructure(let details) = error as? BasaltError else { return false }
            return details.contains("thinking_budget")
        }
    }

    @Test("The transition headroom is checked only when the request leaves thinking on")
    func headroomIsValidatedOnlyWithThinkingOn() throws {
        let tight = { (effort: String?, thinking: Bool?) in
            GenerationRequest(prompt: "x", maxTokens: 50, enableThinking: thinking, reasoningEffort: effort, thinkingBudget: 10)
        }
        #expect(throws: BasaltError.self) { try tight(nil, true).validated() }
        #expect(throws: BasaltError.self) { try tight("low", nil).validated() }
        #expect(try tight("none", true).validated().maxTokens == 50)
        #expect(try tight(nil, false).validated().maxTokens == 50)
    }

    @Test("Thinking switched off ignores the budget, even one larger than max_tokens")
    func disabledThinkingIgnoresTheBudget() async throws {
        let engine = try await ThinkingModel.engine()
        let off = try await engine.generate(request: ThinkingModel.request(budget: 5000, maxTokens: 50, effort: "none"))
        let plain = try await engine.generate(request: ThinkingModel.request(budget: nil, maxTokens: 50, effort: "none"))
        #expect(off.reasoningText == plain.reasoningText)
        #expect(off.generatedText == plain.generatedText)
    }

    @Test("A model with no reasoning protocol generates as if no budget was asked")
    func modelWithoutReasoningIgnoresTheBudget() async throws {
        let engine = try await ThinkingModel.engine(reasoning: nil, profile: GenericProfile())
        let budgeted = try await engine.generate(request: ThinkingModel.request(budget: 6, maxTokens: 60))
        let plain = try await engine.generate(request: ThinkingModel.request(budget: nil, maxTokens: 60))
        #expect(budgeted.reasoningText == plain.reasoningText)
        #expect(budgeted.generatedText == plain.generatedText)
    }

    @Test("Qwen-tagged models without a declared transition get Qwen's, declared ones keep theirs, others get none")
    func transitionSelection() throws {
        let qwen = try #require(MLXInferenceEngine.budgetTransition(for: QwenReasoningProtocol.tagged, profile: QwenProfile()))
        #expect(qwen.beforeEndDelimiter
            == "\n\n Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n")
        #expect(qwen.afterEndDelimiter == "\n\n")

        var declared = QwenReasoningProtocol.tagged
        declared.budgetTransition = ReasoningBudgetTransition(beforeEndDelimiter: "enough.")
        #expect(MLXInferenceEngine.budgetTransition(for: declared, profile: GenericProfile())?.beforeEndDelimiter == "enough.")

        #expect(MLXInferenceEngine.budgetTransition(for: QwenReasoningProtocol.tagged, profile: GenericProfile()) == nil)
    }

    @Test("A non-Qwen reasoning model without a declared transition generates as if no budget was asked")
    func nonQwenReasoningModelIgnoresTheBudget() async throws {
        let engine = try await ThinkingModel.engine(profile: GenericProfile())
        let budgeted = try await engine.generate(request: ThinkingModel.request(budget: 6, maxTokens: 200))
        let plain = try await engine.generate(request: ThinkingModel.request(budget: nil, maxTokens: 200))
        #expect(budgeted.reasoningText == plain.reasoningText)
        #expect(budgeted.generatedText == plain.generatedText)
    }
}
