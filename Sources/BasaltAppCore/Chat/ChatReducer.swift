import Foundation

public struct ChatState: Sendable, Equatable {
    public enum Run: Sendable, Equatable {
        case idle
        case streaming(RunID)
        case failed(EngineError)
    }

    public var conversation: ConversationSnapshot
    public var draft = ""
    public var attachments: [ChatAttachment] = []
    public var run = Run.idle

    public init(conversation: ConversationSnapshot) { self.conversation = conversation }

    public var canSend: Bool {
        if case .streaming = run { return false }
        return !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !attachments.isEmpty
    }
}

public struct RunTiming: Sendable, Equatable {
    public var timeToFirstToken: Duration?
    public var streaming: Duration

    public init(timeToFirstToken: Duration?, streaming: Duration) {
        (self.timeToFirstToken, self.streaming) = (timeToFirstToken, streaming)
    }
}

public enum ChatAction: Sendable, Equatable {
    case draftChanged(String)
    case send(message: MessageID, reply: MessageID, run: RunID)
    case regenerate(reply: MessageID, run: RunID)
    case stop
    case delta(RunID, ChatDelta)
    case finished(RunID, RunTiming)
    case failed(RunID, EngineError)
    case attach(ChatAttachment)
    case detach(UUID)
    case perceived(RunID, PipelineResult)
}

public enum ChatEffect: Sendable, Equatable {
    case start(RunID, ChatRequest)
    case perceive(RunID, PipelineCall)
    case cancel(RunID)
    case persist(ConversationSnapshot)
}

public enum ChatReducer {
    public static let titleLength = 60

    public static func reduce(_ state: inout ChatState, _ action: ChatAction) -> [ChatEffect] {
        switch action {
        case .draftChanged(let text):
            state.draft = text
            return []
        case .send(let message, let reply, let run):
            return send(&state, message: message, reply: reply, run: run)
        case .regenerate(let reply, let run):
            guard currentRun(state) == nil, state.conversation.messages.last?.role == .assistant else { return [] }
            state.conversation.messages.removeLast()
            return begin(&state, reply: reply, run: run)
        case .stop:
            guard let run = currentRun(state) else { return [] }
            close(&state, as: .stopped, run: .idle)
            return [.cancel(run), .persist(state.conversation)]
        case .delta(let run, let delta):
            guard state.run == .streaming(run), let last = state.conversation.messages.indices.last else { return [] }
            apply(delta, to: &state.conversation.messages[last])
            return []
        case .finished(let run, let timing):
            guard state.run == .streaming(run), let last = state.conversation.messages.indices.last else { return [] }
            time(&state.conversation.messages[last], timing)
            close(&state, as: .complete, run: .idle)
            return [.persist(state.conversation)]
        case .failed(let run, let error):
            guard state.run == .streaming(run) else { return [] }
            close(&state, as: .failed, run: .failed(error))
            return [.persist(state.conversation)]
        case .attach, .detach, .perceived:
            return attachments(&state, action)
        }
    }

    static func attachments(_ state: inout ChatState, _ action: ChatAction) -> [ChatEffect] {
        switch action {
        case .attach(let attachment):
            if !state.attachments.contains(where: { $0.id == attachment.id }) { state.attachments.append(attachment) }
            return []
        case .detach(let id):
            state.attachments.removeAll { $0.id == id }
            return []
        case .perceived(let run, let result):
            guard state.run == .streaming(run), state.conversation.messages.count >= 2 else { return [] }
            let (question, answer) = (state.conversation.messages.count - 2, state.conversation.messages.count - 1)
            var index = 0
            for (offset, part) in state.conversation.messages[question].parts.enumerated() {
                guard case .attachment(var attachment) = part, index < result.perceived.count else { continue }
                attachment.perceived = result.perceived[index]
                state.conversation.messages[question].parts[offset] = .attachment(attachment)
                index += 1
            }
            (state.conversation.messages[answer].parts, state.conversation.messages[answer].receipt) = ([.text(result.answer)], result.receipt)
            close(&state, as: .complete, run: .idle)
            return [.persist(state.conversation)]
        default:
            return []
        }
    }

    static func send(_ state: inout ChatState, message: MessageID, reply: MessageID, run: RunID) -> [ChatEffect] {
        guard state.canSend else { return [] }
        let text = state.draft.trimmingCharacters(in: .whitespacesAndNewlines)
        let attached = state.attachments.map { MessagePart.attachment(AttachmentPart(name: $0.name, kind: $0.kind)) }
        state.conversation.messages.append(MessageSnapshot(id: message, role: .user, parts: (text.isEmpty ? [] : [.text(text)]) + attached))
        if state.conversation.title.isEmpty { state.conversation.title = String((text.isEmpty ? state.attachments.map(\.name).joined(separator: ", ") : text).prefix(titleLength)) }
        let call = PipelineCall(attachments: state.attachments, task: text.isEmpty ? "Summarize the attached files." : text)
        (state.draft, state.attachments) = ("", [])
        let effects = begin(&state, reply: reply, run: run)
        return call.attachments.isEmpty ? effects : effects.map { if case .start = $0 { .perceive(run, call) } else { $0 } }
    }

    static func currentRun(_ state: ChatState) -> RunID? {
        if case .streaming(let run) = state.run { run } else { nil }
    }

    static func begin(_ state: inout ChatState, reply: MessageID, run: RunID) -> [ChatEffect] {
        let request = ChatRequest(state.conversation)
        var answer = MessageSnapshot(id: reply, role: .assistant, state: .streaming)
        answer.receipt = Receipt(model: state.conversation.model)
        state.conversation.messages.append(answer)
        state.run = .streaming(run)
        return [.persist(state.conversation), .start(run, request)]
    }

    static func close(_ state: inout ChatState, as final: MessageSnapshot.State, run: ChatState.Run) {
        if let last = state.conversation.messages.indices.last, state.conversation.messages[last].state == .streaming {
            state.conversation.messages[last].state = final
        }
        state.run = run
    }

    static func apply(_ delta: ChatDelta, to message: inout MessageSnapshot) {
        switch (delta, message.parts.last) {
        case (.text(let piece), .text(let text)?): message.parts[message.parts.count - 1] = .text(text + piece)
        case (.reasoning(let piece), .reasoning(let text)?): message.parts[message.parts.count - 1] = .reasoning(text + piece)
        case (.text(let piece), _): message.parts.append(.text(piece))
        case (.reasoning(let piece), _): message.parts.append(.reasoning(piece))
        case (.toolCall(let call), _): message.parts.append(.toolCall(call))
        case (.usage(let prompt, let completion), _):
            message.receipt?.promptTokens = prompt
            message.receipt?.completionTokens = completion
        case (.finish, _): break
        }
    }

    static func time(_ message: inout MessageSnapshot, _ timing: RunTiming) {
        message.receipt?.timeToFirstToken = timing.timeToFirstToken
        let seconds = Double(timing.streaming.components.seconds) + Double(timing.streaming.components.attoseconds) / 1e18
        if let tokens = message.receipt?.completionTokens, tokens > 0, seconds > 0 { message.receipt?.tokensPerSecond = Double(tokens) / seconds }
    }
}
