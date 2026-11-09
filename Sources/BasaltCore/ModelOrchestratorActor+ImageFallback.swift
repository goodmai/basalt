import Foundation

struct MediaFallbacks: Sendable {
    var speech: (any AudioTranscriber)?
    var document: (any DocumentTranscriber)?
    var transcribesImagesForTextModels = false

    var imageFallback: (any DocumentTranscriber)? {
        transcribesImagesForTextModels ? document : nil
    }
}

extension ModelOrchestratorActor {
    public func setDocumentTranscriber(_ transcriber: (any DocumentTranscriber)?, imageFallback: Bool = false) {
        media.document = transcriber
        media.transcribesImagesForTextModels = imageFallback
    }

    @concurrent
    public nonisolated func transcribingImages(
        in request: GenerationRequest, modelId: String? = nil
    ) async throws(BasaltError) -> (request: GenerationRequest, images: Int) {
        guard let messages = request.messages, messages.contains(where: { !$0.images.isEmpty }),
              let transcriber = await media.imageFallback else { return (request, 0) }
        try await autoSwitch(to: modelId)
        guard await !engine.takesImages else { return (request, 0) }
        let images = messages.flatMap(\.images)
        try ImageInput.checkLimits(images)
        var transcripts: [String] = []
        for image in images {
            transcripts.append(try await transcriber.transcribe(image).markdown)
        }
        return (request.withMessages(ChatMessageInput.replacingImages(in: messages, with: transcripts)), images.count)
    }

    nonisolated func readWithVision(_ request: OCRRequest) async throws(BasaltError) -> OCRResult {
        guard let transcriber = await media.document else {
            throw .invalidRequestStructure(details: "\(OCRRequest.appleVision) is not available in this process")
        }
        guard request.task != .formula, request.task != .chart else {
            throw .invalidRequestStructure(details: "\(OCRRequest.appleVision) reads text, lists and tables, not formulas or charts; use task document, text or table")
        }
        var pages: [OCRPageResult] = []
        for page in try request.document.pages(request.pages, dpi: OCRRecipe.pageDPI) {
            let started = ContinuousClock.now
            let read = try await transcriber.transcribe(page.image)
            pages.append(OCRPageResult(page: page.number, markdown: read.markdown, blocks: read.blocks, tokens: 0, ms: Self.milliseconds(since: started)))
        }
        return OCRResult(model: OCRRequest.appleVision, family: OCRRequest.appleVisionFamily, task: request.task, pages: pages, format: request.format)
    }

    nonisolated static func milliseconds(since start: ContinuousClock.Instant) -> Int {
        let elapsed = (ContinuousClock.now - start).components
        return Int(elapsed.seconds * 1000 + elapsed.attoseconds / 1_000_000_000_000_000)
    }
}

extension ChatMessageInput {
    public static func replacingImages(in messages: [ChatMessageInput], with transcripts: [String]) -> [ChatMessageInput] {
        var number = 0
        return messages.map { message in
            guard !message.images.isEmpty else { return message }
            let blocks = message.images.map { _ in
                number += 1
                let text = number <= transcripts.count ? transcripts[number - 1] : ""
                return "[image \(number), OCR]\n\(text.isEmpty ? "(no text recognized)" : text)"
            }
            return ChatMessageInput(
                role: message.role, text: ([message.text] + blocks).filter { !$0.isEmpty }.joined(separator: "\n\n"),
                toolCalls: message.toolCalls, toolCallId: message.toolCallId, toolName: message.toolName)
        }
    }
}
