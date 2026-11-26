import Foundation
import MLX
import MLXLMCommon

struct DiffusionTokenIterator: TokenIteratorProtocol {
    let model: any DiffusionLanguageModel
    let adapter: LanguageModelBlockAdapter
    let promptTokens: [Int]
    let maxTokens: Int?
    let config: BlockDiffusionConfig
    let eosTokenId: Int?
    private(set) var tokenCount = 0
    private(set) var promptPrefillTime: TimeInterval = 0
    private var generated: [Int]?

    init(model: any DiffusionLanguageModel, input: LMInput, maxTokens: Int?, config: BlockDiffusionConfig, eosTokenId: Int?) throws {
        self.model = model
        adapter = try LanguageModelBlockAdapter(model)
        promptTokens = input.text.tokens.reshaped(-1).asArray(Int32.self).map(Int.init)
        self.maxTokens = maxTokens
        var config = config
        config.maskTokenId = model.maskTokenId
        self.config = config
        self.eosTokenId = eosTokenId
    }

    var state: LMOutput.State? { nil }
    var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? { nil }
    var evaluatedModel: (any LanguageModel)? { model }

    mutating func next() -> Int? {
        if generated == nil {
            let started = ContinuousClock.now
            let request = FastDLLMRequest(promptTokens: promptTokens, maxNewTokens: maxTokens ?? Self.defaultMaxTokens, eosTokenId: eosTokenId)
            generated = model.evaluating { FastDLLMRunner.generate(model: adapter, request: request, config: config).generatedTokens }
            promptPrefillTime = (ContinuousClock.now - started).inSeconds
        }
        guard let generated, tokenCount < generated.count else { return nil }
        defer { tokenCount += 1 }
        return generated[tokenCount]
    }

    mutating func discardGeneratedToken() {}

    static let defaultMaxTokens = 256
}
