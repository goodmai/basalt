import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("LFM2.5 MoE out of the box (#113)", .serialized)
struct LiquidLFM25Tests {

    @Test("lfm2 and lfm2_moe checkpoints get the Liquid profile", arguments: ["lfm2", "lfm2_moe", "LFM2_MOE"])
    func registryPicksLiquid(modelType: String) {
        #expect(ModelProfileRegistry.profile(forModelType: modelType) is LiquidProfile)
    }

    @Test("Liquid templates have no thinking switch, so no request leaks template variables")
    func liquidTemplateContextIsEmpty() {
        for policy in [ReasoningPolicy.modelDefault, .disabled, .effort("high")] {
            #expect(LiquidProfile().templateContext(for: policy).isEmpty)
        }
    }

    @Test("thinking_budget forces the transition on an LFM2.5 checkpoint, which declares no reasoning protocol")
    func budgetAppliesWithoutADeclaredProtocol() async throws {
        let engine = try await ThinkingModel.engine(reasoning: nil, profile: LiquidProfile())
        let budget = 6
        let free = try await engine.generate(request: ThinkingModel.request(budget: nil, maxTokens: 200))
        let budgeted = try await engine.generate(request: ThinkingModel.request(budget: budget, maxTokens: 200))

        #expect(scalars(free.reasoningText).count > budget + 20)
        let reasoning = try #require(budgeted.reasoningText)
        #expect(scalars(reasoning) == Array(scalars(free.reasoningText).prefix(budget)) + scalars(ThinkingModel.transition))
        #expect(!budgeted.generatedText.isEmpty)
    }

    @Test("A Liquid tool call ends thinking, so a budget never forces the transition into the call; other profiles get no protocol")
    func liquidProtocolEndsAtToolCalls() throws {
        let liquid = try #require(MLXInferenceEngine.reasoningProtocol(for: LiquidProfile()))
        #expect(liquid.endDelimiter == "</think>")
        #expect(liquid.implicitEndDelimiters == ["<|tool_call_start|>"])
        #expect(MLXInferenceEngine.reasoningProtocol(for: GenericProfile()) == nil)
    }

    @Test("repetition_penalty in generation_config.json applies when neither the request nor --repetition-penalty sets one")
    func generationConfigRepetitionPenalty() async throws {
        let lfm24: [String: Any] = ["do_sample": true, "temperature": 0.1, "top_k": 50, "repetition_penalty": 1.05]
        let defaults = SamplingDefaults(generationConfig: lfm24)
        #expect(defaults == SamplingDefaults(temperature: 0.1, topK: 50, repetitionPenalty: 1.05))

        let engine = MLXInferenceEngine()
        await engine.attach(try TinyModel.container(), samplingDefaults: defaults)
        #expect(await engine.sampling(for: GenerationRequest(prompt: "p")).0.repetitionPenalty == 1.05)
        #expect(await engine.sampling(for: GenerationRequest(prompt: "p", repetitionPenalty: 1.2)).0.repetitionPenalty == 1.2)

        let flagged = MLXInferenceEngine(repetitionPenalty: 1.1)
        await flagged.attach(try TinyModel.container(), samplingDefaults: defaults)
        #expect(await flagged.sampling(for: GenerationRequest(prompt: "p")).0.repetitionPenalty == 1.1)
    }

    @Test("The answer after a closed reasoning block starts with its first visible character, not the newlines after </think>", arguments: [true, false])
    func answerStartsWithoutNewlines(liquid: Bool) async throws {
        let engine = try await ThinkingModel.engine(reasoning: liquid ? nil : QwenReasoningProtocol.tagged,
                                                    profile: liquid ? LiquidProfile() : QwenProfile())
        let budgeted = try await engine.generate(request: ThinkingModel.request(budget: 6, maxTokens: 200))
        let first = try #require(budgeted.generatedText.first)
        #expect(!first.isNewline)
        var streamed = ""
        for await chunk in try await engine.generateStream(request: ThinkingModel.request(budget: 6, maxTokens: 200)) {
            if case .text(let t) = chunk { streamed += t }
        }
        #expect(streamed == budgeted.generatedText)
    }
}
