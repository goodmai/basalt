import Foundation

enum ImageTurnPlanner {
    static let pngHeader = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])

    static func parseCommand(_ text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = trimmed.lowercased()
        for prefix in ["/imagine ", "/image "] {
            if lower.hasPrefix(prefix) {
                return String(trimmed.dropFirst(prefix.count)).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        if lower == "/image" || lower == "/imagine" { return "" }
        return nil
    }

    static func parseDimensions(_ size: String) -> (width: Int, height: Int) {
        let pieces = size.lowercased().split(separator: "x")
        guard pieces.count == 2, let w = Int(pieces[0]), let h = Int(pieces[1]), w > 0, h > 0 else {
            return (1024, 1024)
        }
        return (w, h)
    }

    static func regenerationCall(in conversation: ConversationSnapshot, target: MessageID) -> ImageGenerationCall? {
        guard let rawIndex = conversation.messages.firstIndex(where: { $0.id == target }) else { return nil }
        let forkIndex: Int
        if conversation.messages[rawIndex].role == .assistant {
            forkIndex = rawIndex
        } else if rawIndex + 1 < conversation.messages.count, conversation.messages[rawIndex + 1].role == .assistant {
            forkIndex = rawIndex + 1
        } else {
            return nil
        }
        guard let previous = conversation.messages[forkIndex].images.first else { return nil }
        var refs: [Data] = []
        if let beforePath = previous.beforeFilePath {
            let bytes = (try? Data(contentsOf: URL(fileURLWithPath: beforePath))) ?? pngHeader
            refs.append(bytes)
        } else if forkIndex > 0 {
            refs = conversation.messages[forkIndex - 1].attachments.compactMap { item in
                guard item.kind == "image", let path = item.filePath else { return nil }
                return (try? Data(contentsOf: URL(fileURLWithPath: path))) ?? pngHeader
            }
        }
        return ImageGenerationCall(
            prompt: previous.prompt,
            model: conversation.model ?? previous.model,
            size: "\(previous.width)x\(previous.height)",
            steps: max(previous.totalSteps, 1),
            referenceImages: refs,
            beforeFilePath: previous.beforeFilePath
        )
    }
}

extension ChatReducer {
    public static func hasImageCommandPrefix(_ text: String) -> Bool {
        ImageTurnPlanner.parseCommand(text) != nil
    }

    public static func parseImageCommand(_ text: String) -> String? {
        ImageTurnPlanner.parseCommand(text)
    }

    static func imageRegenerationCall(in conversation: ConversationSnapshot, target: MessageID) -> ImageGenerationCall? {
        ImageTurnPlanner.regenerationCall(in: conversation, target: target)
    }

    static func sendImageIfRequested(
        _ state: inout ChatState,
        message: MessageID,
        reply: MessageID,
        run: RunID
    ) -> [ChatEffect]? {
        let raw = state.draft.trimmingCharacters(in: .whitespacesAndNewlines)
        let commandPrompt = ImageTurnPlanner.parseCommand(raw)
        guard commandPrompt != nil || state.imageMode else { return nil }
        let imageAttachments = state.attachments.filter { $0.kind == "image" }
        let resolvedPrompt: String
        if let commandPrompt {
            resolvedPrompt = commandPrompt.isEmpty && !imageAttachments.isEmpty ? "Edit the reference image." : commandPrompt
        } else {
            resolvedPrompt = raw.isEmpty ? "Edit the reference image." : raw
        }
        guard !resolvedPrompt.isEmpty else { return [] }
        let before = state.beforeFilePath ?? imageAttachments.first?.sourcePath
        let call = ImageGenerationCall(
            prompt: resolvedPrompt,
            model: state.conversation.model,
            referenceImages: imageAttachments.map(\.data),
            beforeFilePath: before
        )
        let attached = state.attachments.map {
            MessagePart.attachment(AttachmentPart(name: $0.name, kind: $0.kind, filePath: $0.sourcePath))
        }
        let userParts: [MessagePart] = (raw.isEmpty ? [] : [.text(raw)]) + attached
        state.conversation.messages.append(MessageSnapshot(id: message, role: .user, parts: userParts))
        if state.conversation.title.isEmpty {
            state.conversation.title = String(resolvedPrompt.prefix(titleLength))
        }
        (state.draft, state.attachments, state.beforeFilePath) = ("", [], nil)
        return beginImage(&state, call: call, reply: reply, run: run)
    }

    static func beginImage(_ state: inout ChatState, call: ImageGenerationCall, reply: MessageID, run: RunID) -> [ChatEffect] {
        let dims = ImageTurnPlanner.parseDimensions(call.size)
        var placeholder = GeneratedImagePart(
            prompt: call.prompt,
            model: call.model ?? "z-image-turbo",
            width: dims.width,
            height: dims.height,
            totalSteps: call.steps
        )
        placeholder.beforeFilePath = call.beforeFilePath
        var answer = MessageSnapshot(id: reply, role: .assistant, parts: [.image(placeholder)], state: .streaming)
        answer.receipt = Receipt(model: call.model)
        state.conversation.messages.append(answer)
        state.run = .streaming(run)
        state.scrollPin = .pinned
        return [.persist(state.conversation), .generateImage(run, call)]
    }

    static func reduceImages(_ state: inout ChatState, _ action: ChatAction) -> [ChatEffect] {
        switch action {
        case .toggleImageMode:
            state.imageMode.toggle()
            return []
        case .imageStage(let run, let stage):
            guard state.run == .streaming(run), let last = state.conversation.messages.indices.last else { return [] }
            var receipt = state.conversation.messages[last].receipt ?? Receipt(model: state.conversation.model)
            receipt.stages = (receipt.stages ?? []) + [stage]
            state.conversation.messages[last].receipt = receipt
            return []
        case .imagePartial(let run, let step, let totalSteps, let png, let width, let height):
            guard state.run == .streaming(run), let last = state.conversation.messages.indices.last else { return [] }
            applyImagePartial(&state.conversation.messages[last], step: step, totalSteps: totalSteps, png: png, size: (width, height))
            state.scrollPin.recordStreamDelta()
            return []
        case .imageCompleted(let run, let part, let stage):
            guard state.run == .streaming(run), let last = state.conversation.messages.indices.last else { return [] }
            applyImageCompleted(&state.conversation.messages[last], part: part, stage: stage)
            close(&state, as: .complete, run: .idle)
            return [.persist(state.conversation), .answered(state.conversation)]
        case .editGeneratedImage(let part):
            attachGeneratedImageForEdit(&state, part: part)
            return []
        default:
            return []
        }
    }

    private static func applyImagePartial(
        _ message: inout MessageSnapshot,
        step: Int,
        totalSteps: Int,
        png: Data,
        size: (Int, Int)
    ) {
        if let idx = message.parts.lastIndex(where: { if case .image = $0 { true } else { false } }),
           case .image(var part) = message.parts[idx] {
            (part.currentStep, part.totalSteps, part.previewPNG, part.width, part.height) = (step, totalSteps, png, size.0, size.1)
            message.parts[idx] = .image(part)
        } else {
            var part = GeneratedImagePart(prompt: "", width: size.0, height: size.1, totalSteps: totalSteps)
            (part.currentStep, part.previewPNG) = (step, png)
            message.parts.append(.image(part))
        }
    }

    private static func applyImageCompleted(_ message: inout MessageSnapshot, part: GeneratedImagePart, stage: StageReceipt) {
        var cleaned = part.persisted
        if cleaned.beforeFilePath == nil, let existingBefore = message.images.last?.beforeFilePath {
            cleaned.beforeFilePath = existingBefore
        }
        if let idx = message.parts.lastIndex(where: { if case .image = $0 { true } else { false } }) {
            message.parts[idx] = .image(cleaned)
        } else {
            message.parts.append(.image(cleaned))
        }
        var receipt = message.receipt ?? Receipt(model: cleaned.model)
        receipt.model = cleaned.model
        receipt.stages = (receipt.stages ?? []) + [stage]
        message.receipt = receipt
    }

    private static func attachGeneratedImageForEdit(_ state: inout ChatState, part: GeneratedImagePart) {
        guard let path = part.filePath, !path.isEmpty else { return }
        let rawName = URL(fileURLWithPath: path).lastPathComponent
        let validName = ChatAttachment.accepts(rawName) ? rawName : "\(rawName.isEmpty ? "reference" : rawName).png"
        let data = (try? Data(contentsOf: URL(fileURLWithPath: path))) ?? part.previewPNG ?? ImageTurnPlanner.pngHeader
        guard let attachment = ChatAttachment(name: validName, data: data, sourcePath: path) else { return }
        state.attachments.removeAll { $0.sourcePath == path }
        state.attachments.append(attachment)
        state.beforeFilePath = path
        state.imageMode = true
    }
}
