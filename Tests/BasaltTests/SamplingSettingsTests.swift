import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Sampling defaults and penalties (#105)")
struct SamplingDefaultsTests {

    @Test("generation_config.json values are read; absent keys stay nil")
    func parsesGenerationConfig() throws {
        let qwen: [String: Any] = ["bos_token_id": 248044, "do_sample": true, "temperature": 1.0, "top_k": 20, "top_p": 0.95]
        #expect(SamplingDefaults(generationConfig: qwen)
            == SamplingDefaults(temperature: 1.0, topP: 0.95, topK: 20))
        #expect(SamplingDefaults(generationConfig: ["temperature": 0.6]) == SamplingDefaults(temperature: 0.6))
        #expect(SamplingDefaults(generationConfig: nil) == SamplingDefaults())

        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(#"{"temperature": 1.0, "top_p": 0.95, "top_k": 20,}"#.utf8)
            .write(to: directory.appendingPathComponent("generation_config.json"))
        #expect(SamplingDefaults(generationConfig: ModelCache.generationConfig(at: directory))
            == SamplingDefaults(temperature: 1.0, topP: 0.95, topK: 20))
    }

    private func engine(defaults: SamplingDefaults, topK: Int? = nil, repetitionPenalty: Float? = nil) async throws
        -> MLXInferenceEngine {
        let engine = MLXInferenceEngine(repetitionPenalty: repetitionPenalty, topK: topK)
        await engine.attach(try TinyModel.container(), samplingDefaults: defaults)
        return engine
    }

    private let modelDefaults = SamplingDefaults(temperature: 1.0, topP: 0.95, topK: 20)

    @Test("A request without temperature / top_p / top_k samples with the model's generation_config")
    func omittedFieldsFollowTheModel() async throws {
        let engine = try await engine(defaults: modelDefaults)
        let (params, _, _) = await engine.sampling(for: GenerationRequest(prompt: "p", temperature: nil, topP: nil))
        #expect(params.temperature == 1.0)
        #expect(params.topP == 0.95)
        #expect(params.topK == 20)
    }

    @Test("Without generation_config the old defaults hold")
    func noGenerationConfigKeepsOldDefaults() async throws {
        let engine = try await engine(defaults: SamplingDefaults())
        let (params, _, _) = await engine.sampling(for: GenerationRequest(prompt: "p", temperature: nil, topP: nil))
        #expect(params.temperature == 0.7)
        #expect(params.topP == 0.9)
        #expect(params.topK == 0)
    }

    @Test("Request values win over the model's, the server's --top-k wins over generation_config")
    func requestOverrides() async throws {
        let engine = try await engine(defaults: modelDefaults)
        let (params, _, _) = await engine.sampling(for: GenerationRequest(prompt: "p", temperature: 0.2, topP: 0.5, topK: 5))
        #expect(params.temperature == 0.2)
        #expect(params.topP == 0.5)
        #expect(params.topK == 5)

        let flagged = try await self.engine(defaults: modelDefaults, topK: 40)
        #expect(await flagged.sampling(for: GenerationRequest(prompt: "p")).0.topK == 40)
    }

    @Test("repetition_penalty from the request beats the server flag")
    func repetitionPenalty() async throws {
        let engine = try await engine(defaults: SamplingDefaults(), repetitionPenalty: 1.1)
        #expect(await engine.sampling(for: GenerationRequest(prompt: "p")).0.repetitionPenalty == 1.1)
        #expect(await engine.sampling(for: GenerationRequest(prompt: "p", repetitionPenalty: 1.05)).0.repetitionPenalty == 1.05)
    }

    @Test("presence_penalty counts only generated tokens: the prompt, fed through the prompt-anchored processor, is not penalised")
    func presencePenaltySkipsThePrompt() async throws {
        let engine = try await engine(defaults: SamplingDefaults())
        let (params, components, _) = await engine.sampling(for: GenerationRequest(prompt: "p", presencePenalty: 1.5))
        #expect(params.presencePenalty == nil)
        var processor = PromptAnchoredProcessor(
            base: try #require(components.logitProcessor(parameters: params)), prompt: MLXArray([Int32(2), 3]))
        processor.prompt(MLXArray([Int32(3)]))
        let logits = MLXArray.zeros([1, 6], type: Float32.self)
        #expect(processor.process(logits: logits).asArray(Float.self) == [0, 0, 0, 0, 0, 0])

        processor.didSample(token: MLXArray(Int32(4)))
        #expect(processor.process(logits: logits).asArray(Float.self) == [0, 0, 0, 0, -1.5, 0])

        let (plainParams, plain, _) = await engine.sampling(for: GenerationRequest(prompt: "p"))
        #expect(plain.logitProcessor(parameters: plainParams) == nil)
    }
}

@Suite("Generation request validation (#105)")
struct SamplingRequestValidationTests {

    @Test("validated() keeps omitted temperature and top_p unset so the engine can apply the model's defaults")
    func omittedSamplingStaysNil() throws {
        let v = try GenerationRequest(prompt: "p", temperature: nil, topP: nil).validated()
        #expect(v.temperature == nil)
        #expect(v.topP == nil)
    }

    @Test("New fields decode from the native DTO and survive withMaxTokens / withCacheKey / validated")
    func fieldsSurviveCopies() throws {
        let json = #"{"prompt": "p", "topK": 20, "presencePenalty": 1.5, "repetitionPenalty": 1.05, "thinkingBudget": 5000}"#
        let decoded = try JSONDecoder().decode(GenerationRequest.self, from: Data(json.utf8))
        for request in [decoded, decoded.withMaxTokens(9000), decoded.withCacheKey("k"), try decoded.validated()] {
            #expect(request.topK == 20)
            #expect(request.presencePenalty == 1.5)
            #expect(request.repetitionPenalty == 1.05)
            #expect(request.thinkingBudget == 5000)
            #expect(request.temperature == nil)
        }
    }

    @Test("Out-of-range sampling parameters are rejected as invalid requests", arguments: [
        GenerationRequest(prompt: "p", topK: -1),
        GenerationRequest(prompt: "p", presencePenalty: 2.5),
        GenerationRequest(prompt: "p", presencePenalty: -3),
        GenerationRequest(prompt: "p", repetitionPenalty: 0),
        GenerationRequest(prompt: "p", repetitionPenalty: 2.5),
        GenerationRequest(prompt: "p", thinkingBudget: -1),
        GenerationRequest(prompt: "p", thinkingBudget: GenerationRequest.maximumTokens + 1),
        GenerationRequest(prompt: "p", topP: 1.5),
    ])
    func rejectsOutOfRange(_ request: GenerationRequest) {
        #expect {
            _ = try request.validated()
        } throws: { error in
            guard case .invalidRequestStructure = error as? BasaltError else { return false }
            return true
        }
    }

    @Test("thinking_budget needs max_tokens to hold the transition headroom after the answer reserve")
    func thinkingBudgetNeedsRoomInMaxTokens() throws {
        _ = try GenerationRequest(prompt: "p", maxTokens: 85, thinkingBudget: 10).validated()
        _ = try GenerationRequest(prompt: "p", maxTokens: 84).validated()
        #expect {
            _ = try GenerationRequest(prompt: "p", maxTokens: 84, thinkingBudget: 10).validated()
        } throws: { error in
            guard case .invalidRequestStructure(let details) = error as? BasaltError else { return false }
            return details.contains("thinking_budget")
        }
    }

    @Test("Boundary values are accepted")
    func acceptsBoundaries() throws {
        _ = try GenerationRequest(prompt: "p", topK: 0, presencePenalty: -2, repetitionPenalty: 2, thinkingBudget: 0).validated()
        _ = try GenerationRequest(prompt: "p", presencePenalty: 2, thinkingBudget: GenerationRequest.maximumTokens).validated()
    }
}
