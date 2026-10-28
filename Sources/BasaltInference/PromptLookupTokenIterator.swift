import Foundation
import MLX
import MLXLMCommon

struct PromptLookupTokenIterator: TokenIteratorProtocol {
    static let draftLimit = 8
    static let ngramSizes = [5, 4, 3]

    let model: any LanguageModel
    let cache: [KVCache]
    var state: LMOutput.State?
    var processor: LogitProcessor?
    let sampler: LogitSampler
    let maxTokens: Int?
    var tokenCount = 0
    var promptPrefillTime: TimeInterval = 0

    private var history: [Int]
    private var inFlight: MLXArray?
    private var backoff = DraftBackoff()
    private var pending: [Int] = []
    private var pendingIndex = 0
    private var rounds = 0
    private var drafted = 0
    private var accepted = 0
    private var verified = 0
    private var verifyCalls = 0

    static func supports(_ cache: [KVCache]) -> Bool {
        PromptCache(cache).isSharable
    }

    init(
        input: LMInput, context: [Int], model: any LanguageModel, cache: [KVCache], state: LMOutput.State? = nil,
        processor: LogitProcessor?, sampler: LogitSampler, prefill: PrefillParameters, maxTokens: Int?
    ) throws {
        self.model = model
        self.cache = cache
        self.state = state
        self.processor = processor
        self.sampler = sampler
        self.maxTokens = maxTokens
        self.history = context
        let start = Date()
        try prepare(input: input, prefill: prefill)
        promptPrefillTime = Date().timeIntervalSince(start)
    }

    var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? {
        rounds == 0 ? nil : SpeculativeDecodingTelemetry(
            roundCount: rounds, draftTokenCount: drafted, acceptedDraftTokenCount: accepted,
            targetModelCallCount: verifyCalls, draftModelCallCount: 0,
            targetVerifiedTokenCount: verified, emittedTokenCount: tokenCount)
    }

    private mutating func prepare(input: LMInput, prefill: PrefillParameters) throws {
        processor?.prompt(input.text.tokens)
        let logits: MLXArray
        switch try model.prepare(input, cache: cache, state: state, prefill: prefill) {
        case .tokens(let rest):
            let output = forward(rest)
            logits = output.logits
            prefill.progress?(input.text.tokens.size, input.text.tokens.size)
        case .logits(let output):
            state = output.state
            logits = output.logits
        }
        accept([sample(logits[0..., -1, 0...]).item(Int.self)])
    }

    private mutating func forward(_ input: LMInput.Text) -> LMOutput {
        let output = withPreparedCache(cache, lengths: input.sequenceLengths) {
            model(input[text: .newAxis], cache: cache, state: state)
        }
        state = output.state
        return output
    }

    private mutating func sample(_ logits: MLXArray) -> MLXArray {
        let processed = processor?.process(logits: logits) ?? logits
        let token = sampler.sample(logits: processed)
        processor?.didSample(token: token)
        return token
    }

    private mutating func accept(_ tokens: [Int]) {
        history += tokens
        pending = tokens
        pendingIndex = 0
    }

    static func draft(after history: [Int], limit: Int) -> [Int] {
        for n in ngramSizes where history.count > n {
            let suffix = history[(history.count - n)...]
            var match: Int?
            for start in stride(from: history.count - n - 1, through: 0, by: -1)
            where history[start ..< start + n].elementsEqual(suffix) {
                match = start
                if start + n + limit <= history.count { break }
            }
            if let match {
                return Array(history[(match + n) ..< Swift.min(match + n + limit, history.count)])
            }
        }
        return []
    }

    private mutating func advance() {
        let remaining = maxTokens.map { $0 - tokenCount } ?? Int.max
        let draft = backoff.allowsDraft && state == nil
            ? Self.draft(after: history, limit: Swift.min(Self.draftLimit, remaining - 1)) : []
        switch (inFlight, draft.isEmpty) {
        case (let token?, true):
            decodeAhead(from: token)
        case (let token?, false):
            inFlight = nil
            let confirmed = token.item(Int.self)
            if confirmed != draft[0] { backoff.record(acceptedDrafts: 0) }
            accept([confirmed])
        case (nil, true):
            decodeAhead(from: step(MLXArray([Int32(history[history.count - 1])])))
        case (nil, false):
            speculate(draft)
        }
    }

    private mutating func step(_ token: MLXArray) -> MLXArray {
        verifyCalls += 1
        verified += 1
        return sample(forward(LMInput.Text(tokens: token)).logits[0..., -1, 0...])
    }

    private mutating func decodeAhead(from token: MLXArray) {
        let next = step(token)
        asyncEval([next] + cache.flatMap { $0.state })
        inFlight = next
        backoff.tick()
        accept([token.item(Int.self)])
    }

    private mutating func speculate(_ draft: [Int]) {
        let fed = [history[history.count - 1]] + draft
        let logits = forward(LMInput.Text(tokens: MLXArray(fed.map(Int32.init))))
            .logits[0, 0..., 0...]

        var emitted: [Int] = []
        if processor == nil, sampler is ArgMaxSampler {
            let tokens = sampler.sample(logits: logits).asArray(Int.self)
            for (index, token) in tokens.enumerated() {
                emitted.append(token)
                if index == draft.count || token != draft[index] { break }
            }
        } else {
            for index in 0 ... draft.count {
                let token = sample(logits[index ..< index + 1]).item(Int.self)
                emitted.append(token)
                if index == draft.count || token != draft[index] { break }
            }
        }

        let acceptedDrafts = emitted.count - 1
        trimPromptCache(cache, numTokens: draft.count - acceptedDrafts)
        backoff.record(acceptedDrafts: acceptedDrafts)
        rounds += 1
        drafted += draft.count
        accepted += acceptedDrafts
        verifyCalls += 1
        verified += fed.count
        accept(emitted)
    }

    mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens { return nil }
        if pendingIndex == pending.count {
            autoreleasepool { advance() }
        }
        if tokenCount % 256 == 0 { Memory.clearCache() }
        tokenCount += 1
        pendingIndex += 1
        return pending[pendingIndex - 1]
    }
}

struct DraftBackoff {
    static let longestPause = 64

    private(set) var pause = 0
    private var nextPause = 1

    var allowsDraft: Bool { pause == 0 }

    mutating func tick() {
        pause = Swift.max(0, pause - 1)
    }

    mutating func record(acceptedDrafts: Int) {
        if acceptedDrafts > 0 {
            nextPause = 1
        } else {
            pause = nextPause
            nextPause = Swift.min(nextPause * 2, Self.longestPause)
        }
    }
}
