import Foundation
import MLXLMCommon
import Synchronization
import BasaltCore

struct OnyxReasoningStateMachine {
    struct Controls: Equatable, Sendable {
        let start: Int
        let message: Int
        let eom: Int
        let eot: Int
    }

    private enum Phase {
        case header(tokens: [Int], primed: Bool)
        case reasoning
        case nonReasoning
        case betweenFrames
    }

    private let tokenizer: any MLXLMCommon.Tokenizer
    private let controls: Controls
    private var detokenizer: NaiveStreamingDetokenizer
    private var phase: Phase = .header(tokens: [], primed: true)
    private let maxHeaderTokens = 64

    init?(tokenizer: any MLXLMCommon.Tokenizer) {
        guard let start = tokenizer.convertTokenToId("<|start|>"),
              let message = tokenizer.convertTokenToId("<|message|>"),
              let eom = tokenizer.convertTokenToId("<|eom|>"),
              let eot = tokenizer.convertTokenToId("<|eot|>"),
              Set([start, message, eom, eot]).count == 4
        else { return nil }
        self.tokenizer = tokenizer
        self.controls = Controls(start: start, message: message, eom: eom, eot: eot)
        self.detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
    }

    enum Step: Equatable, Sendable {
        case none
        case reasoningChunk(String)
        case enteredNonReasoning
    }

    mutating func push(_ token: Int) -> Step {
        if token == controls.start {
            phase = .header(tokens: [], primed: false)
            return .none
        }
        switch phase {
        case .header(let tokens, let primed):
            return pushHeader(token, tokens: tokens, primed: primed)
        case .reasoning:
            if isFrameBoundary(token) {
                phase = .betweenFrames
                return .none
            }
            detokenizer.append(token: token)
            guard let text = detokenizer.next(), !text.isEmpty else { return .none }
            return .reasoningChunk(text)
        case .nonReasoning:
            if isFrameBoundary(token) { phase = .betweenFrames }
            return .none
        case .betweenFrames:
            return .none
        }
    }

    private func isFrameBoundary(_ token: Int) -> Bool {
        token == controls.eom || token == controls.eot
    }

    private mutating func pushHeader(_ token: Int, tokens: [Int], primed: Bool) -> Step {
        if token == controls.message {
            if isReasoningHeader(tokens, primed: primed) {
                detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
                phase = .reasoning
                return .none
            }
            phase = .nonReasoning
            return .enteredNonReasoning
        }
        if isFrameBoundary(token) || tokens.count >= maxHeaderTokens {
            phase = .betweenFrames
            return .none
        }
        phase = .header(tokens: tokens + [token], primed: primed)
        return .none
    }

    private func isReasoningHeader(_ tokens: [Int], primed: Bool) -> Bool {
        let decoded = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
        let full = (primed ? "assistant" : "") + decoded
        let fields = full.split(whereSeparator: \.isWhitespace)
        guard fields.first == "assistant",
              let toField = fields.dropFirst().first(where: { $0.hasPrefix("to=") })
        else { return false }
        return toField.dropFirst(3) == "self"
    }
}

final class OnyxReasoningBox: Sendable {
    private struct State: Sendable {
        var initialPhase = true
        var initialEmitter: (@Sendable (String) -> Void)?
        var initialBufferedForEmitter: [String] = []
        var accountingQueue: [String] = []
        var laterChunks: [String] = []
    }

    private let state = Mutex(State())

    func record(_ step: OnyxReasoningStateMachine.Step) {
        switch step {
        case .none:
            return
        case .enteredNonReasoning:
            state.withLock {
                $0.initialPhase = false
                $0.initialEmitter = nil
            }
        case .reasoningChunk(let text):
            let emitter = state.withLock { state -> (@Sendable (String) -> Void)? in
                if state.initialPhase {
                    state.accountingQueue.append(text)
                    if let live = state.initialEmitter {
                        return live
                    }
                    state.initialBufferedForEmitter.append(text)
                    return nil
                }
                state.laterChunks.append(text)
                return nil
            }
            emitter?(text)
        }
    }

    func attachInitialEmitter(_ emit: @escaping @Sendable (String) -> Void) -> Bool {
        let buffered = state.withLock { state -> [String] in
            let pending = state.initialBufferedForEmitter
            state.initialBufferedForEmitter.removeAll()
            if state.initialPhase {
                state.initialEmitter = emit
            }
            return pending
        }
        for chunk in buffered { emit(chunk) }
        return !buffered.isEmpty
    }

    func drainRelayStep() -> (accountedInitial: [String], additionalChunks: [String]) {
        state.withLock { state in
            let accounted = state.accountingQueue
            let later = state.laterChunks
            state.accountingQueue.removeAll()
            state.laterChunks.removeAll()
            return (accounted, later)
        }
    }

    func drainUnemittedInitial() -> [String] {
        state.withLock { state in
            state.initialPhase = false
            state.initialEmitter = nil
            let pending = state.initialBufferedForEmitter
            state.initialBufferedForEmitter.removeAll()
            return pending
        }
    }
}

struct OnyxReasoningTapIterator<Base: TokenIteratorProtocol>: TokenIteratorProtocol {
    private var base: Base
    private var parser: OnyxReasoningStateMachine
    private let box: OnyxReasoningBox

    init(base: consuming Base, parser: OnyxReasoningStateMachine, box: OnyxReasoningBox) {
        self.base = base
        self.parser = parser
        self.box = box
    }

    var maxTokens: Int? { base.maxTokens }
    var tokenCount: Int { base.tokenCount }
    var promptPrefillTime: TimeInterval { base.promptPrefillTime }
    var state: LMOutput.State? { base.state }
    var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? { base.speculativeDecodingTelemetry }
    var evaluatedModel: (any LanguageModel)? { base.evaluatedModel }

    mutating func discardGeneratedToken() {
        base.discardGeneratedToken()
    }

    mutating func next() -> Int? {
        guard let token = base.next() else { return nil }
        box.record(parser.push(token))
        return token
    }
}

extension OnyxReasoningTapIterator: SelfSerializingTokenIterator where Base: SelfSerializingTokenIterator {}
