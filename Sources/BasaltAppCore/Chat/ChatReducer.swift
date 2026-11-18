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
    public var imageMode = false
    public var beforeFilePath: String?
    public var run = Run.idle
    public var scrollPin = ScrollPinState.pinned
    public var branches = MessageBranches()

    public init(conversation: ConversationSnapshot) { self.conversation = conversation }

    public var usesImageComposer: Bool {
        imageMode || ChatReducer.hasImageCommandPrefix(draft)
    }

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
    case regenerateFrom(message: MessageID, reply: MessageID, run: RunID)
    case selectBranch(message: MessageID, index: Int)
    case stop
    case delta(RunID, ChatDelta)
    case finished(RunID, RunTiming)
    case failed(RunID, EngineError)
    case attach(ChatAttachment)
    case detach(UUID)
    case perceived(RunID, PipelineResult)
    case systemPromptChanged(String)
    case scrolled(atBottom: Bool)
    case jumpToBottom
    case toggleImageMode
    case imageStage(RunID, StageReceipt)
    case imagePartial(RunID, step: Int, totalSteps: Int, png: Data, width: Int, height: Int)
    case imageCompleted(RunID, GeneratedImagePart, StageReceipt)
    case editGeneratedImage(GeneratedImagePart)
}

public enum ChatEffect: Sendable, Equatable {
    case start(RunID, ChatRequest)
    case perceive(RunID, PipelineCall)
    case generateImage(RunID, ImageGenerationCall)
    case cancel(RunID)
    case persist(ConversationSnapshot)
    case answered(ConversationSnapshot)
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
        case .regenerate, .regenerateFrom, .selectBranch:
            return branch(&state, action)
        case .stop, .delta, .finished, .failed:
            return streamLifecycle(&state, action)
        case .attach, .detach, .perceived, .systemPromptChanged:
            return attachments(&state, action)
        case .scrolled, .jumpToBottom:
            return scroll(&state, action)
        case .toggleImageMode, .imageStage, .imagePartial, .imageCompleted, .editGeneratedImage:
            return reduceImages(&state, action)
        }
    }

    static func streamLifecycle(_ state: inout ChatState, _ action: ChatAction) -> [ChatEffect] {
        switch action {
        case .stop:
            guard let run = currentRun(state) else { return [] }
            close(&state, as: .stopped, run: .idle)
            return [.cancel(run), .persist(state.conversation)]
        case .delta(let run, let delta):
            guard state.run == .streaming(run), let last = state.conversation.messages.indices.last else { return [] }
            apply(delta, to: &state.conversation.messages[last])
            recordScrollDelta(delta, in: &state.scrollPin)
            return []
        case .finished(let run, let timing):
            guard state.run == .streaming(run), let last = state.conversation.messages.indices.last else { return [] }
            time(&state.conversation.messages[last], timing)
            close(&state, as: .complete, run: .idle)
            return [.persist(state.conversation), .answered(state.conversation)]
        case .failed(let run, let error):
            guard state.run == .streaming(run) else { return [] }
            close(&state, as: .failed, run: .failed(error))
            return [.persist(state.conversation)]
        default:
            return []
        }
    }

    static func branch(_ state: inout ChatState, _ action: ChatAction) -> [ChatEffect] {
        guard currentRun(state) == nil else { return [] }
        switch action {
        case .regenerate(let reply, let run):
            guard let last = state.conversation.messages.last, last.role == .assistant else { return [] }
            return regenerateFrom(&state, target: last.id, reply: reply, run: run)
        case .regenerateFrom(let message, let reply, let run):
            return regenerateFrom(&state, target: message, reply: reply, run: run)
        case .selectBranch(let message, let index):
            return state.branches.select(in: &state.conversation, message: message, index: index) ? [.persist(state.conversation)] : []
        default:
            return []
        }
    }

    static func regenerateFrom(_ state: inout ChatState, target: MessageID, reply: MessageID, run: RunID) -> [ChatEffect] {
        let imageCall = imageRegenerationCall(in: state.conversation, target: target)
        guard state.branches.fork(in: &state.conversation, target: target) else { return [] }
        let effects = if let imageCall {
            beginImage(&state, call: imageCall, reply: reply, run: run)
        } else {
            begin(&state, reply: reply, run: run)
        }
        if let created = state.conversation.messages.last {
            state.branches.recordHead(created, in: state.conversation)
        }
        return effects
    }

    static func scroll(_ state: inout ChatState, _ action: ChatAction) -> [ChatEffect] {
        switch action {
        case .scrolled(let atBottom):
            state.scrollPin.applyUserScroll(atBottom: atBottom)
        case .jumpToBottom:
            state.scrollPin.jumpToBottom()
        default:
            break
        }
        return []
    }

    static func recordScrollDelta(_ delta: ChatDelta, in pin: inout ScrollPinState) {
        switch delta {
        case .text, .reasoning, .toolCall:
            pin.recordStreamDelta()
        case .usage, .finish:
            break
        }
    }

    static func attachments(_ state: inout ChatState, _ action: ChatAction) -> [ChatEffect] {
        switch action {
        case .attach(let attachment):
            if !state.attachments.contains(where: { $0.id == attachment.id }) { state.attachments.append(attachment) }
            return []
        case .detach(let id):
            state.attachments.removeAll { $0.id == id }
            if let before = state.beforeFilePath, !state.attachments.contains(where: { $0.sourcePath == before }) {
                state.beforeFilePath = nil
            }
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
            return [.persist(state.conversation), .answered(state.conversation)]
        case .systemPromptChanged(let text):
            let prompt = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard (prompt.isEmpty ? nil : prompt) != state.conversation.systemPrompt else { return [] }
            state.conversation.systemPrompt = prompt.isEmpty ? nil : prompt
            return [.persist(state.conversation)]
        default:
            return []
        }
    }

    static func send(_ state: inout ChatState, message: MessageID, reply: MessageID, run: RunID) -> [ChatEffect] {
        guard state.canSend else { return [] }
        if let imageEffects = sendImageIfRequested(&state, message: message, reply: reply, run: run) {
            return imageEffects
        }
        let text = state.draft.trimmingCharacters(in: .whitespacesAndNewlines)
        let attached = state.attachments.map { MessagePart.attachment(AttachmentPart(name: $0.name, kind: $0.kind, filePath: $0.sourcePath)) }
        state.conversation.messages.append(MessageSnapshot(id: message, role: .user, parts: (text.isEmpty ? [] : [.text(text)]) + attached))
        if state.conversation.title.isEmpty { state.conversation.title = String((text.isEmpty ? state.attachments.map(\.name).joined(separator: ", ") : text).prefix(titleLength)) }
        let call = PipelineCall(attachments: state.attachments, task: text.isEmpty ? "Summarize the attached files." : text)
        (state.draft, state.attachments, state.beforeFilePath) = ("", [], nil)
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
        state.scrollPin = .pinned
        return [.persist(state.conversation), .start(run, request)]
    }

    static func close(_ state: inout ChatState, as final: MessageSnapshot.State, run: ChatState.Run) {
        if let last = state.conversation.messages.indices.last, state.conversation.messages[last].state == .streaming {
            state.conversation.messages[last].state = final
            state.branches.recordHead(state.conversation.messages[last], in: state.conversation)
        }
        state.run = run
    }

    static func apply(_ delta: ChatDelta, to message: inout MessageSnapshot) {
        switch (delta, message.parts.last) {
        case (.text(let piece), .text(var text)?):
            let index = message.parts.count - 1
            message.parts[index] = .text("")
            text.append(piece)
            message.parts[index] = .text(text)
        case (.reasoning(let piece), .reasoning(var text)?):
            let index = message.parts.count - 1
            message.parts[index] = .reasoning("")
            text.append(piece)
            message.parts[index] = .reasoning(text)
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
