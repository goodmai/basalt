import Foundation
import MLX
import MLXLMCommon

struct PromptLookupTokenIterator: TokenIteratorProtocol {
    static let draftLimit = 16
    static let replayedProbeLimit = 4
    static let ngramSizes = [5, 4, 3]
    static let recycledDraftLimit = 6

    let model: any LanguageModel
    let cache: [KVCache]
    var state: LMOutput.State?
    var processor: LogitProcessor?
    let sampler: LogitSampler
    let maxTokens: Int?
    let recyclesTokens: Bool
    let speculativeLookup: Bool
    let grammarJumpForward: Bool
    let replaysRejections: Bool
    var tokenCount = 0
    var promptPrefillTime: TimeInterval = 0

    private var history: [Int]
    private var inFlight: DecodedStep?
    private(set) var recycling = TokenRecyclingTable(vocabularySize: 0)
    private(set) var recycledRounds = 0
    private(set) var recycledDraftTokens = 0
    private(set) var acceptedRecycledDrafts = 0
    private(set) var forcedTokensCount = 0
    private var queuedForced: [Int] = []
    private var backoffs = DrafterBackoffs()
    private var pending: [Int] = []
    private var pendingIndex = 0
    private var rounds = 0
    private var drafted = 0
    private var accepted = 0
    private var verified = 0
    private var verifyCalls = 0
    private(set) var replayRounds = 0
    private var evalTargets: [MLXArray] = []

    static func supports(_ cache: [KVCache], routesExperts: Bool) -> Bool {
        PromptCache(cache).isSharable || (isReplayableHybrid(cache) && !routesExperts)
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
        recyclesTokens: Bool = false, speculativeLookup: Bool = true, grammarJumpForward: Bool = false
    ) throws {
        self.recyclesTokens = recyclesTokens
        self.speculativeLookup = speculativeLookup
        self.grammarJumpForward = grammarJumpForward
        self.replaysRejections = Self.isReplayableHybrid(cache)
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
        rounds == 0 && forcedTokensCount == 0 ? nil : SpeculativeDecodingTelemetry(
            roundCount: rounds, draftTokenCount: drafted, acceptedDraftTokenCount: accepted,
            targetModelCallCount: verifyCalls, draftModelCallCount: 0,
            targetVerifiedTokenCount: verified, emittedTokenCount: tokenCount)
    }

    private mutating func takeForcedTokens(limit: Int) -> [Int] {
        guard grammarJumpForward, limit > 0 else { return [] }
        if !queuedForced.isEmpty {
            defer { queuedForced.removeAll(keepingCapacity: true) }
            return Array(queuedForced.prefix(limit))
        }
        if var source = processor as? ForcedTokenSource {
            let tokens = source.takeForcedTokens(limit: limit)
            processor = source
            return tokens
        }
        return []
    }

    private mutating func didSampleForced(_ token: Int) {
        let arr = MLXArray([Int32(token)])
        if var source = processor as? ForcedTokenSource {
            source.didSampleForced(arr)
            processor = source
        } else {
            processor?.didSample(token: arr)
        }
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
        if recyclesTokens {
            recycling = TokenRecyclingTable(vocabularySize: last.dim(-1))
        }
        let needed = maxTokens ?? Int.max
        let forced = takeForcedTokens(limit: needed)
        if !forced.isEmpty {
            forcedTokensCount += forced.count
            for id in forced {
                didSampleForced(id)
            }
            if forced.count >= needed {
                accept(forced)
                return
            }
            verifyCalls += 1
            verified += forced.count
            let forcedLogits = forward(LMInput.Text(tokens: MLXArray(forced.map(Int32.init))))
                .logits[0, 0..., 0...]
            let nextToken = sample(forcedLogits[forced.count - 1 ..< forced.count]).item(Int.self)
            if recyclesTokens {
                recycle(after: forced, successors: argMax(forcedLogits, axis: -1).asArray(Int.self))
            }
            accept(forced + [nextToken])
            return
        }
        let token = sample(last).item(Int.self)
        if recyclesTokens {
            recycle(after: [history[history.count - 1]], successors: [argMax(last, axis: -1).item(Int.self)])
        }
        acceptOne(token)
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

    private mutating func acceptOne(_ token: Int) {
        history.append(token)
        pending.removeAll(keepingCapacity: true)
        pending.append(token)
        pendingIndex = 0
    }

    static func draft(after history: [Int], limit: Int) -> [Int] {
        guard limit > 0 else { return [] }
        let span = history.span
        let count = span.count
        for n in ngramSizes where count > n {
            let suffixStart = count - n
            let firstSuffix = span[suffixStart]
            let lastSuffix = span[count - 1]
            var match: Int?
            for start in stride(from: suffixStart - 1, through: 0, by: -1) {
                if span[start] != firstSuffix || span[start + n - 1] != lastSuffix {
                    continue
                }
                var matched = true
                for offset in 1 ..< n - 1 {
                    if span[start + offset] != span[suffixStart + offset] {
                        matched = false
                        break
                    }
                }
                if matched {
                    match = start
                    if start + n + limit <= count { break }
                }
            }
            if let match {
                let draftStart = match + n
                let draftEnd = Swift.min(draftStart + limit, count)
                return Array(history[draftStart ..< draftEnd])
            }
        }
        return []
    }

    private mutating func advance() {
        let remaining = maxTokens.map { $0 - tokenCount } ?? Int.max
        guard remaining > 0 else { return }

        let forced = takeForcedTokens(limit: remaining)
        if !forced.isEmpty {
            forcedTokensCount += forced.count
            for id in forced {
                didSampleForced(id)
            }
            if forced.count >= remaining {
                accept(forced)
                return
            }
            let fed = [history[history.count - 1]] + forced
            verifyCalls += 1
            verified += fed.count
            let logits = forward(LMInput.Text(tokens: MLXArray(fed.map(Int32.init))))
                .logits[0, 0..., 0...]
            let nextToken = sample(logits[fed.count - 1 ..< fed.count]).item(Int.self)
            if recyclesTokens {
                recycle(after: fed, successors: argMax(logits, axis: -1).asArray(Int.self))
            }
            accept(forced + [nextToken])
            return
        }

        let looksUp = speculativeLookup && backoffs.lookup.allowsDraft && state == nil
        let recycles = recyclesTokens && backoffs.recycling.allowsDraft && state == nil
        let limit = replaysRejections && backoffs.lookup.probing ? Self.replayedProbeLimit : backoffs.lookup.draftLimit
        let draft = looksUp ? Self.draft(after: history, limit: Swift.min(limit, remaining - 1)) : []
        switch (inFlight, draft.isEmpty) {
        case (let step?, true) where recycles && recycling.predicts(after: history[history.count - 1]):
            if recycling.chain(from: step.token.item(Int.self), limit: 1).isEmpty {
                emit(step, lookingAhead: remaining > 1)
            } else {
                inFlight = nil
                acceptOne(confirm(step))
            }
        case (let step?, true):
            emit(step, lookingAhead: remaining > 1)
        case (let step?, false):
            inFlight = nil
            let confirmed = confirm(step)
            if confirmed != draft[0] { backoffs.lookup.record(acceptedDrafts: 0) }
            acceptOne(confirmed)
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
        if lookingAhead && !grammarJumpForward {
            let next = step(current.token)
            evalTargets.removeAll(keepingCapacity: true)
            evalTargets.append(next.token)
            if let successor = next.successor { evalTargets.append(successor) }
            for layer in cache {
                for state in layer.state {
                    evalTargets.append(state)
                }
            }
            asyncEval(evalTargets)
            inFlight = next
        } else {
            inFlight = nil
        }
        backoffs.tick()
        acceptOne(confirm(current))
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
            let remaining = maxTokens.map { $0 - tokenCount } ?? Int.max
            for index in 0 ... draft.count {
                let token = sample(logits[index ..< index + 1]).item(Int.self)
                emitted.append(token)
                if index == draft.count || token != draft[index] { break }
                let forced = takeForcedTokens(limit: remaining - emitted.count)
                if !forced.isEmpty {
                    queuedForced = forced
                    break
                }
            }
            if let successors { recycle(after: Array(fed.prefix(emitted.count)), successors: Array(successors.asArray(Int.self).prefix(emitted.count))) }
        }

        let acceptedDrafts = emitted.count - 1
        let matchedDrafts = queuedForced.isEmpty ? acceptedDrafts : acceptedDrafts + 1
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
            verifyCalls += 1
        }
        if recycled {
            recycledRounds += 1
            recycledDraftTokens += draft.count
            acceptedRecycledDrafts += acceptedDrafts
        }
        backoffs.recordRound(recycled: recycled, acceptedDrafts: matchedDrafts, draftCount: draft.count, replayed: replays)
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

    mutating func recordRound(recycled: Bool, acceptedDrafts: Int, draftCount: Int = 0, replayed: Bool) {
        for _ in 0 ... acceptedDrafts {
            if recycled { lookup.tick() } else { recycling.tick() }
        }
        let paidDrafts = replayed ? acceptedDrafts - 1 : acceptedDrafts
        let pauseScale = replayed ? DraftBackoff.replayPauseScale : 1
        if recycled {
            recycling.record(acceptedDrafts: paidDrafts, drafted: draftCount, pauseScale: pauseScale)
        } else {
            lookup.record(acceptedDrafts: paidDrafts, drafted: draftCount, pauseScale: pauseScale)
        }
    }
}

struct DraftBackoff {
    static let longestPause = 64
    static let replayPauseScale = 4
    static let maxDraftLimit = 16
    static let minDraftLimit = 2

    private(set) var pause = 0
    private(set) var probing = true
    private(set) var draftLimit = maxDraftLimit
    private var nextPause = 1

    var allowsDraft: Bool { pause == 0 }

    mutating func tick() {
        pause = Swift.max(0, pause - 1)
    }

    mutating func record(acceptedDrafts: Int, drafted: Int = 0, pauseScale: Int = 1) {
        let isProfitable: Bool
        if drafted > 0 {
            isProfitable = acceptedDrafts >= 3
                || (drafted <= 2 && acceptedDrafts >= 1)
                || (Double(acceptedDrafts) / Double(drafted) >= 0.25)
            let matched = pauseScale > 1 ? acceptedDrafts + 1 : acceptedDrafts
            if !isProfitable || matched <= 1 {
                draftLimit = Self.minDraftLimit
            } else {
                draftLimit = Swift.min(Self.maxDraftLimit, Swift.max(Self.minDraftLimit, matched + 2))
            }
        } else {
            isProfitable = acceptedDrafts > 0
        }
        probing = !isProfitable
        if isProfitable {
            nextPause = 1
        } else {
            pause = nextPause * pauseScale
            nextPause = Swift.min(nextPause * 2, Self.longestPause)
        }
    }
}
