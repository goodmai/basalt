import BasaltCore
import Foundation

public struct VoiceSessionState: Sendable, Equatable {
    public enum CapturePhase: Sendable, Equatable {
        case idle
        case recording
        case transcribing
    }

    public var capture = CapturePhase.idle
    public var handsFree = false
    public var holdingThinkingPause = false
    public var enhancement = VoiceEnhancementOptions()
    public var speakingMessage: MessageID?
    public var synthesizingMessage: MessageID?
    public var voice = "alloy"
    public var speed = 1.0
    public var error: String?

    public init() {}

    public var isRecording: Bool { capture == .recording }

    public var isSpeaking: Bool { speakingMessage != nil || synthesizingMessage != nil }

    public var isActive: Bool { capture != .idle || isSpeaking || handsFree }
}

public enum VoiceAction: Sendable, Equatable {
    case pushToTalkPressed
    case pushToTalkReleased
    case recorded(Data)
    case transcribed(String)
    case readAloud(MessageID, String)
    case speechSynthesized(MessageID, Data)
    case playbackFinished(MessageID)
    case stopSpeaking
    case setHandsFree(Bool)
    case setEnhancement(VoiceEnhancementOptions)
    case turnEvaluated(TurnDecision)
    case assistantAnswered(MessageID, String)
    case failed(String)
}

public enum VoiceEffect: Sendable, Equatable {
    case startCapture
    case stopCapture
    case transcribe(Data)
    case submitTranscript(String)
    case synthesize(MessageID, text: String, voice: String, speed: Double)
    case playAudio(MessageID, Data)
    case stopPlayback
}

public enum VoiceReducer {
    public static func reduce(_ state: inout VoiceSessionState, _ action: VoiceAction) -> [VoiceEffect] {
        switch action {
        case .pushToTalkPressed, .pushToTalkReleased, .recorded, .transcribed, .turnEvaluated:
            return reduceCapture(&state, action)
        case .readAloud, .speechSynthesized, .playbackFinished, .stopSpeaking, .assistantAnswered:
            return reducePlayback(&state, action)
        case .setHandsFree(let enabled):
            state.handsFree = enabled
            guard !enabled, state.capture == .recording else { return [] }
            (state.capture, state.holdingThinkingPause) = (.idle, false)
            return [.stopCapture]
        case .setEnhancement(let options):
            state.enhancement = options
            return []
        case .failed(let message):
            let wasRecording = state.capture == .recording
            (state.capture, state.holdingThinkingPause, state.speakingMessage, state.synthesizingMessage, state.error) =
                (.idle, false, nil, nil, message)
            return wasRecording ? [.stopCapture] : []
        }
    }

    private static func reduceCapture(_ state: inout VoiceSessionState, _ action: VoiceAction) -> [VoiceEffect] {
        switch action {
        case .pushToTalkPressed:
            guard state.capture == .idle else { return [] }
            let wasSpeaking = state.isSpeaking
            (state.capture, state.holdingThinkingPause, state.speakingMessage, state.synthesizingMessage, state.error) =
                (.recording, false, nil, nil, nil)
            return wasSpeaking ? [.stopPlayback, .startCapture] : [.startCapture]
        case .pushToTalkReleased:
            guard state.capture == .recording else { return [] }
            (state.capture, state.holdingThinkingPause) = (.transcribing, false)
            return [.stopCapture]
        case .turnEvaluated(let decision):
            return reduceTurnDecision(&state, decision)
        case .recorded(let wav):
            guard state.capture != .idle else { return [] }
            guard !wav.isEmpty else {
                state.capture = .idle
                return []
            }
            state.capture = .transcribing
            return [.transcribe(wav)]
        case .transcribed(let raw):
            guard state.capture == .transcribing else { return [] }
            state.capture = .idle
            let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            return text.isEmpty ? [] : [.submitTranscript(text)]
        default:
            return []
        }
    }

    private static func reduceTurnDecision(_ state: inout VoiceSessionState, _ decision: TurnDecision) -> [VoiceEffect] {
        guard state.capture == .recording else { return [] }
        switch decision {
        case .listening:
            state.holdingThinkingPause = false
            return []
        case .holdThinkingPause:
            state.holdingThinkingPause = state.enhancement.semanticTurnTakingEnabled
            return []
        case .commitTurn:
            (state.capture, state.holdingThinkingPause) = (.transcribing, false)
            return [.stopCapture]
        }
    }

    private static func reducePlayback(_ state: inout VoiceSessionState, _ action: VoiceAction) -> [VoiceEffect] {
        switch action {
        case .readAloud(let id, let raw):
            return startOrToggleReadAloud(&state, id: id, raw: raw)
        case .speechSynthesized(let id, let wav):
            guard state.synthesizingMessage == id else { return [] }
            (state.synthesizingMessage, state.speakingMessage) = (nil, id)
            return [.playAudio(id, wav)]
        case .playbackFinished(let id):
            guard state.speakingMessage == id else { return [] }
            state.speakingMessage = nil
            guard state.handsFree, state.capture == .idle else { return [] }
            state.capture = .recording
            return [.startCapture]
        case .stopSpeaking:
            guard state.isSpeaking else { return [] }
            (state.speakingMessage, state.synthesizingMessage) = (nil, nil)
            return [.stopPlayback]
        case .assistantAnswered(let id, let raw):
            return state.handsFree ? beginSynthesis(&state, id: id, raw: raw) : []
        default:
            return []
        }
    }

    private static func startOrToggleReadAloud(_ state: inout VoiceSessionState, id: MessageID, raw: String) -> [VoiceEffect] {
        if state.speakingMessage == id || state.synthesizingMessage == id {
            (state.speakingMessage, state.synthesizingMessage) = (nil, nil)
            return [.stopPlayback]
        }
        return beginSynthesis(&state, id: id, raw: raw)
    }

    private static func beginSynthesis(_ state: inout VoiceSessionState, id: MessageID, raw: String) -> [VoiceEffect] {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return [] }
        let stopFirst = state.isSpeaking
        (state.synthesizingMessage, state.speakingMessage, state.error) = (id, nil, nil)
        let synth = VoiceEffect.synthesize(id, text: text, voice: state.voice, speed: state.speed)
        return stopFirst ? [.stopPlayback, synth] : [synth]
    }
}

public struct VoiceEnvironment: Sendable {
    public var engine: @Sendable () async -> EngineClient?
    public var startCapture: @Sendable () async throws -> Void
    public var stopCapture: @Sendable () async throws -> Data
    public var playAudio: @Sendable (Data) async throws -> Void
    public var stopPlayback: @Sendable () async -> Void
    public var submitTranscript: @Sendable (String) async -> Void

    public init(
        engine: @escaping @Sendable () async -> EngineClient?,
        startCapture: @escaping @Sendable () async throws -> Void,
        stopCapture: @escaping @Sendable () async throws -> Data,
        playAudio: @escaping @Sendable (Data) async throws -> Void,
        stopPlayback: @escaping @Sendable () async -> Void,
        submitTranscript: @escaping @Sendable (String) async -> Void
    ) {
        (self.engine, self.startCapture, self.stopCapture, self.playAudio, self.stopPlayback, self.submitTranscript) =
            (engine, startCapture, stopCapture, playAudio, stopPlayback, submitTranscript)
    }
}

public enum VoiceEffects {
    public static let captureID = "voice-capture"
    public static let transcribeID = "voice-transcribe"
    public static let playbackID = "voice-playback"

    public static func handler(_ environment: VoiceEnvironment) -> @Sendable (VoiceEffect) -> EffectRun<VoiceAction> {
        { effect in
            switch effect {
            case .startCapture:
                return .run(id: captureID) { send in
                    guard !Task.isCancelled else { return }
                    do {
                        try await environment.startCapture()
                    } catch {
                        if !Task.isCancelled { await send(.failed(describe(error))) }
                    }
                }
            case .stopCapture:
                return .run(id: captureID) { send in
                    do {
                        let wav = try await environment.stopCapture()
                        if !Task.isCancelled { await send(.recorded(wav)) }
                    } catch {
                        if !Task.isCancelled { await send(.failed(describe(error))) }
                    }
                }
            case .transcribe(let wav):
                return .run(id: transcribeID) { send in await transcribe(wav, environment, send) }
            case .submitTranscript(let text):
                return .run(id: "voice-submit") { _ in await environment.submitTranscript(text) }
            case .synthesize(let id, let text, let voice, let speed):
                return .run(id: playbackID) { send in await synthesize(id, text: text, voice: voice, speed: speed, environment, send) }
            case .playAudio(let id, let wav):
                return .run(id: playbackID) { send in await play(id, wav: wav, environment, send) }
            case .stopPlayback:
                return .run(id: playbackID) { _ in await environment.stopPlayback() }
            }
        }
    }

    private static func transcribe(_ wav: Data, _ environment: VoiceEnvironment, _ send: @escaping @Sendable (VoiceAction) async -> Void) async {
        guard let engine = await environment.engine() else { return await send(.failed("the engine is not running")) }
        do {
            let text = try await engine.transcribeAudio(AudioTranscriptionCall(wav: wav))
            guard !Task.isCancelled else { return }
            await send(.transcribed(text))
        } catch {
            guard !Task.isCancelled else { return }
            await send(.failed(describe(error)))
        }
    }

    private static func synthesize(
        _ id: MessageID, text: String, voice: String, speed: Double,
        _ environment: VoiceEnvironment, _ send: @escaping @Sendable (VoiceAction) async -> Void
    ) async {
        guard let engine = await environment.engine() else { return await send(.failed("the engine is not running")) }
        do {
            let wav = try await engine.synthesizeSpeech(SpeechSynthesisCall(input: text, voice: voice, speed: speed))
            guard !Task.isCancelled else { return }
            await send(.speechSynthesized(id, wav))
        } catch {
            guard !Task.isCancelled else { return }
            await send(.failed(describe(error)))
        }
    }

    private static func play(
        _ id: MessageID, wav: Data, _ environment: VoiceEnvironment, _ send: @escaping @Sendable (VoiceAction) async -> Void
    ) async {
        do {
            try await environment.playAudio(wav)
            guard !Task.isCancelled else { return }
            await send(.playbackFinished(id))
        } catch {
            guard !Task.isCancelled else { return }
            await send(.failed(describe(error)))
        }
    }

    private static func describe(_ error: any Error) -> String {
        if case .refused(_, let message) = error as? EngineError { return message }
        return (error as? LocalizedError)?.errorDescription ?? String(describing: error)
    }
}
