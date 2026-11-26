import Foundation
import MLX
import MLXLMCommon

struct DiffusionTokenIterator: TokenIteratorProtocol {
    let model: any DiffusionLanguageModel
    let maxTokens: Int?
    let stopTokenIds: Set<Int>
    private var session: FastDLLMSession<LanguageModelBlockAdapter>
    private var pendingBlock: [Int] = []
    private var pendingCursor = 0
    private var startedPrefill = false
    private var finished = false
    private(set) var tokenCount = 0
    private(set) var promptPrefillTime: TimeInterval = 0

    init(model: any DiffusionLanguageModel, input: LMInput, maxTokens: Int?, config: BlockDiffusionConfig, stopTokenIds: Set<Int>) throws {
        self.model = model
        self.maxTokens = maxTokens
        self.stopTokenIds = stopTokenIds
        let adapter = try LanguageModelBlockAdapter(model)
        let promptTokens = input.text.tokens.reshaped(-1).asArray(Int32.self).map(Int.init)
        var resolved = config
        resolved.maskTokenId = model.maskTokenId
        resolved.bidirectionalPrompt = model.bidirectionalPrompt
        var request = FastDLLMRequest(
            promptTokens: promptTokens,
            maxNewTokens: maxTokens ?? Self.defaultMaxTokens,
            includeTrailingMaskSuffix: !resolved.prefixCacheOnly
        )
        request.stopTokenIds = stopTokenIds
        self.session = FastDLLMSession(model: adapter, request: request, config: resolved)
    }

    var state: LMOutput.State? { nil }
    var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? { nil }
    var evaluatedModel: (any LanguageModel)? { model }

    mutating func next() -> Int? {
        guard !finished else { return nil }
        if !startedPrefill {
            startedPrefill = true
            let started = ContinuousClock.now
            model.evaluating { session.prefillIfNeeded() }
            promptPrefillTime = (ContinuousClock.now - started).inSeconds
        }
        while pendingCursor >= pendingBlock.count {
            guard let block = model.evaluating({ session.nextBlock() }) else {
                finished = true
                return nil
            }
            pendingBlock = block
            pendingCursor = 0
            if block.isEmpty && session.isFinished {
                finished = true
                return nil
            }
        }
        let token = pendingBlock[pendingCursor]
        pendingCursor += 1
        if stopTokenIds.contains(token) {
            finished = true
            return nil
        }
        tokenCount += 1
        return token
    }

    mutating func discardGeneratedToken() {}

    static let defaultMaxTokens = 256

    static func stopTokens(_ configuration: ModelConfiguration, tokenizer: any Tokenizer, extra: [String] = []) -> Set<Int> {
        var stops = configuration.eosTokenIds
        if let eos = tokenizer.eosTokenId { stops.insert(eos) }
        for token in configuration.extraEOSTokens { if let id = tokenizer.convertTokenToId(token) { stops.insert(id) } }
        for token in extra { if let id = tokenizer.convertTokenToId(token) { stops.insert(id) } }
        return stops
    }
}
