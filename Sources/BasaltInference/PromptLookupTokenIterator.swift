import Foundation
import MLX
import MLXLMCommon

struct PromptLookupTokenIterator: TokenIteratorProtocol {
    static let draftLimit = 16
    static let ngramSizes = [5, 4, 3]
    static let recycledDraftLimit = 6

    let model: any LanguageModel
    let cache: [KVCache]
    var state: LMOutput.State?
    var processor: LogitProcessor?
    let sampler: LogitSampler
    let maxTokens: Int?
    let recyclesTokens: Bool
    var tokenCount = 0
    var promptPrefillTime: TimeInterval = 0

    private var history: [Int]
    private var inFlight: DecodedStep?
    private(set) var recycling = TokenRecyclingTable(vocabularySize: 0)
    private(set) var recycledRounds = 0
    private(set) var recycledDraftTokens = 0
    private(set) var acceptedRecycledDrafts = 0
    private var backoffs = DrafterBackoffs()
    private var pending: [Int] = []
    private var pendingIndex = 0
    private var rounds = 0
    private var drafted = 0
    private var accepted = 0
    private var verified = 0
    private var verifyCalls = 0
    private(set) var replayRounds = 0
    private(set) var replayedTokens = 0

    static func supports(_ cache: [KVCache], model: any LanguageModel) -> Bool {
        PromptCache(cache).isSharable || (isReplayableHybrid(cache) && !routesExperts(model))
    }

    static func isReplayableHybrid(_ cache: [KVCache]) -> Bool {
        cache.contains { $0 is MambaCache }
            && cache.allSatisfy { $0 is MambaCache || type(of: $0) == KVCacheSimple.self }
    }

    static func routesExperts(_ model: any LanguageModel) -> Bool {
        model.modules().contains { $0 is SwitchLinear }
    }

    init(
        input: LMInput, context: [Int], model: any LanguageModel, cache: [KVCache], state: LMOutput.State? = nil,
        processor: LogitProcessor?, sampler: LogitSampler, prefill: PrefillParameters, maxTokens: Int?,
        recyclesTokens: Bool = false
    ) throws {
        self.recyclesTokens = recyclesTokens
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
        let last = logits[0..., -1, 0...]
        let token = sample(last).item(Int.self)
        if recyclesTokens {
            recycling = TokenRecyclingTable(vocabularySize: last.dim(-1))
            recycle(after: [history[history.count - 1]], successors: [argMax(last, axis: -1).item(Int.self)])
        }
        accept([token])
    }

    private mutating func recycle(after fed: [Int], successors: [Int]) {
        for (token, successor) in zip(fed, successors) { recycling.record(token, successor: successor) }
    }

    private mutating func confirm(_ step: DecodedStep) -> Int {
        let token = step.token.item(Int.self)
        if let successor = step.successor {
            recycle(after: [history[history.count - 1]], successors: [successor.item(Int.self)])
        }
        return token
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
        let looksUp = backoffs.lookup.allowsDraft && state == nil
        let recycles = recyclesTokens && backoffs.recycling.allowsDraft && state == nil
        let draft = looksUp ? Self.draft(after: history, limit: Swift.min(Self.draftLimit, remaining - 1)) : []
        switch (inFlight, draft.isEmpty) {
        case (let step?, true) where recycles && recycling.predicts(after: history[history.count - 1]):
            if recycling.chain(from: step.token.item(Int.self), limit: 1).isEmpty {
                emit(step, lookingAhead: remaining > 1)
            } else {
                inFlight = nil
                accept([confirm(step)])
            }
        case (let step?, true):
            emit(step, lookingAhead: remaining > 1)
        case (let step?, false):
            inFlight = nil
            let confirmed = confirm(step)
            if confirmed != draft[0] { backoffs.lookup.record(acceptedDrafts: 0) }
            accept([confirmed])
        case (nil, false):
            speculate(draft, recycled: false)
        case (nil, true):
            let recycled = recycles
                ? recycling.chain(from: history[history.count - 1], limit: Swift.min(Self.recycledDraftLimit, remaining - 1))
                : []
            if recycled.isEmpty {
                emit(step(MLXArray([Int32(history[history.count - 1])])), lookingAhead: remaining > 1)
            } else {
                speculate(recycled, recycled: true)
            }
        }
    }

    private mutating func step(_ token: MLXArray) -> DecodedStep {
        verifyCalls += 1
        verified += 1
        let logits = forward(LMInput.Text(tokens: token)).logits[0..., -1, 0...]
        return DecodedStep(token: sample(logits), successor: recyclesTokens ? argMax(logits, axis: -1) : nil)
    }

    private mutating func emit(_ current: DecodedStep, lookingAhead: Bool) {
        if lookingAhead {
            let next = step(current.token)
            asyncEval([next.token] + [next.successor].compactMap { $0 } + cache.flatMap { $0.state })
            inFlight = next
        } else {
            inFlight = nil
        }
        backoffs.tick()
        accept([confirm(current)])
    }

    private mutating func speculate(_ draft: [Int], recycled: Bool) {
        let fed = [history[history.count - 1]] + draft
        let recurrent = cache.compactMap { $0 as? MambaCache }
        let checkpoints = recurrent.map { (conv: $0[0], state: $0[1], offset: $0.offset) }
        let logits = forward(LMInput.Text(tokens: MLXArray(fed.map(Int32.init))))
            .logits[0, 0..., 0...]

        var emitted: [Int] = []
        if processor == nil, sampler is ArgMaxSampler {
            let tokens = sampler.sample(logits: logits).asArray(Int.self)
            for (index, token) in tokens.enumerated() {
                emitted.append(token)
                if index == draft.count || token != draft[index] { break }
            }
            if recyclesTokens { recycle(after: fed, successors: tokens) }
        } else {
            let successors = recyclesTokens ? argMax(logits, axis: -1) : nil
            if let successors { asyncEval(successors) }
            for index in 0 ... draft.count {
                let token = sample(logits[index ..< index + 1]).item(Int.self)
                emitted.append(token)
                if index == draft.count || token != draft[index] { break }
            }
            if let successors { recycle(after: fed, successors: successors.asArray(Int.self)) }
        }

        let acceptedDrafts = emitted.count - 1
        let replays = !recurrent.isEmpty && acceptedDrafts < draft.count
        if recurrent.isEmpty {
            trimPromptCache(cache, numTokens: draft.count - acceptedDrafts)
        } else if replays {
            for layer in cache where layer.isTrimmable { layer.trim(fed.count) }
            for (layer, checkpoint) in zip(recurrent, checkpoints) {
                layer[0] = checkpoint.conv
                layer[1] = checkpoint.state
                layer.offset = checkpoint.offset
            }
            _ = forward(LMInput.Text(tokens: MLXArray(fed[...acceptedDrafts].map(Int32.init))))
            replayRounds += 1
            replayedTokens += acceptedDrafts + 1
            verifyCalls += 1
            verified += acceptedDrafts + 1
        }
        if recycled {
            recycledRounds += 1
            recycledDraftTokens += draft.count
            acceptedRecycledDrafts += acceptedDrafts
        }
        backoffs.recordRound(recycled: recycled, acceptedDrafts: acceptedDrafts, replayed: replays)
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

struct DecodedStep {
    let token: MLXArray
    let successor: MLXArray?
}

struct TokenRecyclingTable {
    static let confirmationsToDraft: UInt8 = 2

    private var successors: [Int32]
    private var confirmations: [UInt8]

    init(vocabularySize: Int) {
        successors = Array(repeating: 0, count: vocabularySize)
        confirmations = Array(repeating: 0, count: vocabularySize)
    }

    var byteCount: Int { successors.count * MemoryLayout<Int32>.size + confirmations.count }

    mutating func record(_ token: Int, successor: Int) {
        guard successors.indices.contains(token) else { return }
        if successors[token] == Int32(successor) {
            confirmations[token] = Swift.min(confirmations[token] &+ 1, Self.confirmationsToDraft)
        } else {
            successors[token] = Int32(successor)
            confirmations[token] = 1
        }
    }

    func predicts(after token: Int) -> Bool {
        successors.indices.contains(token) && confirmations[token] >= Self.confirmationsToDraft
            && !chain(from: Int(successors[token]), limit: 1).isEmpty
    }

    func chain(from token: Int, limit: Int) -> [Int] {
        var chain: [Int] = []
        var current = token
        while chain.count < limit, successors.indices.contains(current), confirmations[current] >= Self.confirmationsToDraft {
            current = Int(successors[current])
            chain.append(current)
        }
        return chain
    }
}

struct DrafterBackoffs {
    var lookup = DraftBackoff()
    var recycling = DraftBackoff()

    mutating func tick() {
        lookup.tick()
        recycling.tick()
    }

    mutating func recordRound(recycled: Bool, acceptedDrafts: Int, replayed: Bool) {
        for _ in 0 ... acceptedDrafts {
            if recycled { lookup.tick() } else { recycling.tick() }
        }
        let paidDrafts = replayed ? acceptedDrafts - 1 : acceptedDrafts
        if recycled {
            recycling.record(acceptedDrafts: paidDrafts)
        } else {
            lookup.record(acceptedDrafts: paidDrafts)
        }
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
