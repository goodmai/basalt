import Foundation
import BasaltCore

package struct PipelineStages: Sendable {
    package let ocr: ModelOrchestratorActor?
    package let audio: ModelOrchestratorActor?
    package let speech: (any AudioTranscriber)?
    package let abstainBelow: Double?
    package let minMargin: Double?
    package let questions: [String: SystemOneQuestion]

    package init(
        ocr: ModelOrchestratorActor? = nil, audio: ModelOrchestratorActor? = nil, speech: (any AudioTranscriber)? = nil,
        abstainBelow: Double? = nil, minMargin: Double? = nil, questions: [String: SystemOneQuestion] = [:]
    ) {
        (self.ocr, self.audio, self.speech, self.abstainBelow, self.minMargin) = (ocr, audio, speech, abstainBelow, minMargin)
        self.questions = questions
    }
}

struct PipelineRunner: Sendable {
    let chat: ModelOrchestratorActor
    let decisions: (any DecisionEngine)?
    let stages: PipelineStages

    func perceivers() async -> PipelineCore.Perceivers {
        PipelineCore.Perceivers(
            ocrModel: stages.ocr != nil, appleVision: await chat.readsDocuments, speechModel: stages.audio != nil,
            chatAudio: await chat.transcribesAudio, speechAnalyzer: stages.speech != nil)
    }

    func missingStage(_ request: PipelineRequest) async -> String? {
        PipelineCore.missingStage(request, perceivers: await perceivers(), hasDecisions: decisions != nil)
    }

    func run(_ request: PipelineRequest, origin: RequestOrigin, sessionId: String?) async throws(BasaltError) -> PipelineResponse {
        if let missing = await missingStage(request) { throw .invalidRequestStructure(details: missing) }
        let started = ContinuousClock.now
        let (perceived, perceiving) = try await perceive(request)
        let state = PipelineCore.state(perceived, text: request.text)
        let (decision, deciding) = try await decide(request.questions, state: state, origin: origin, sessionId: sessionId)
        let route = PipelineCore.route(questions: request.questions, decision: decision, task: request.task, state: state)
        var replies: [PipelineCore.System2Reply] = []
        let answering = perceived.pages == nil && perceived.transcript == nil ? origin : RequestOrigin(source: PipelineCore.perceivedSource, endpoint: origin.endpoint, stream: false)
        for call in route.calls { replies.append(try await answer(call, origin: answering, sessionId: sessionId)) }
        let (answers, system2) = PipelineCore.answers(route, replies: replies, questions: request.questions, decision: decision)
        let receipt = PipelineCore.receipt(
            perceiving + [deciding].compactMap { $0 }, replies: replies, answers: answers, ms: Self.elapsed(since: started))
        await chat.sessionLog?.logPipeline(origin: origin, sessionId: sessionId, receipt: receipt)
        return PipelineResponse(perceived: perceived, answers: answers, system2: system2, receipt: receipt)
    }

    func perceive(_ request: PipelineRequest) async throws(BasaltError) -> (PipelinePerceived, [PipelineStageReceipt]) {
        var (pages, transcripts, receipts) = ([PipelinePage](), [String](), [PipelineStageReceipt]())
        let perceivers = await perceivers()
        for (index, attachment) in request.attachments.enumerated() {
            guard let perceiver = perceivers.perceiver(for: attachment) else { throw .invalidRequestStructure(details: "pipeline: no perceiver for attachment \(index + 1)") }
            let started = ContinuousClock.now
            switch attachment {
            case .document(let document):
                let result = try await readPaged(document, pages: request.pages, with: perceiver)
                pages += result.pages.map { PipelinePage(attachment: index, page: $0.page, markdown: $0.markdown) }
                receipts.append(PipelineStageReceipt(
                    stage: .perceive, model: result.model, ms: Self.elapsed(since: started), tokens: result.pages.reduce(0) { $0 + $1.tokens }))
            case .audio(let clip):
                let heard = try await betweenGenerations(perceiver.betweenGenerations) { () async throws(BasaltError) in try await transcribe(clip, with: perceiver) }
                transcripts.append(heard.text)
                receipts.append(PipelineStageReceipt(stage: .perceive, model: heard.model, ms: Self.elapsed(since: started), tokens: 0))
            }
        }
        let transcript = transcripts.isEmpty ? nil : transcripts.joined(separator: "\n\n")
        return (PipelinePerceived(pages: pages.isEmpty ? nil : pages, transcript: transcript), receipts)
    }

    func betweenGenerations<T: Sendable>(_ gated: Bool, _ body: @Sendable () async throws(BasaltError) -> T) async throws(BasaltError) -> T {
        guard gated else { return try await body() }
        do {
            return try await chat.gate.boundedExclusively(body)
        } catch {
            throw error as? BasaltError ?? .cancelled
        }
    }

    func readPaged(_ document: OCRDocument, pages: String?, with perceiver: PipelineCore.Perceiver) async throws(BasaltError) -> (model: String, pages: [OCRPageResult]) {
        guard perceiver.betweenGenerations, let numbers = try document.pageNumbers(pages), numbers.count > 1 else {
            let result = try await betweenGenerations(perceiver.betweenGenerations) { () async throws(BasaltError) in try await read(document, pages: pages, with: perceiver) }
            return (result.model, result.pages)
        }
        var (model, read) = ("", [OCRPageResult]())
        for number in numbers {
            let page = try await betweenGenerations(true) { () async throws(BasaltError) in try await self.read(document, pages: String(number), with: perceiver) }
            (model, read) = (page.model, read + page.pages)
        }
        return (model, read)
    }

    func read(_ document: OCRDocument, pages: String?, with perceiver: PipelineCore.Perceiver) async throws(BasaltError) -> OCRResult {
        let selection = if case .pdf = document { pages } else { String?.none }
        guard perceiver == .ocrModel, let ocr = stages.ocr else {
            return try await chat.ocr(OCRRequest(document: document, pages: selection, task: nil, format: nil, model: OCRRequest.appleVision))
        }
        return try await ocr.ocr(OCRRequest(document: document, pages: selection, task: nil, format: nil, model: nil))
    }

    func transcribe(_ clip: AudioClip, with perceiver: PipelineCore.Perceiver) async throws(BasaltError) -> (text: String, model: String) {
        let request = try TranscriptionRequest(audio: clip)
        switch perceiver {
        case .speechModel:
            if let audio = stages.audio { return (try await audio.transcribe(request).text, await audio.modelInfo) }
        case .speechAnalyzer:
            if let speech = stages.speech { return (try await speech.transcribe(request).text, PipelineCore.speechAnalyzerModel) }
        case .chatModel, .ocrModel, .appleVision:
            break
        }
        return (try await chat.transcribe(request).text, await chat.modelInfo)
    }

    func decide(
        _ questions: [String: SystemOneQuestion], state: String, origin: RequestOrigin, sessionId: String?
    ) async throws(BasaltError) -> (SystemOneResponse?, PipelineStageReceipt?) {
        guard !questions.isEmpty, let decisions else { return (nil, nil) }
        let started = ContinuousClock.now
        guard !state.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, state.count <= SystemOneCore.maxStateCharacters else {
            return (nil, PipelineCore.skipped(decisions.modelID, ms: Self.elapsed(since: started)))
        }
        let request = SystemOneRequest(state: state, questions: questions, abstainBelow: stages.abstainBelow, minMargin: stages.minMargin)
        do {
            let response = try await SystemOneController.decide(decisions, request, log: chat.sessionLog, origin: origin, sessionId: sessionId)
            return (response, PipelineCore.decided(response, ms: Self.elapsed(since: started)))
        } catch .contextLengthExceeded, .invalidRequestStructure {
            return (nil, PipelineCore.skipped(decisions.modelID, ms: Self.elapsed(since: started)))
        }
    }

    func answer(_ call: PipelineCore.System2Call, origin: RequestOrigin, sessionId: String?) async throws(BasaltError) -> PipelineCore.System2Reply {
        let started = ContinuousClock.now
        let response = try await chat.generate(request: PipelineCore.request(for: call, origin: origin).withRouting(sessionId: sessionId, budgetKey: nil))
        return PipelineCore.System2Reply(
            call: call, model: await chat.modelInfo, text: response.generatedText,
            tokens: response.promptTokens + response.completionTokens, ms: Self.elapsed(since: started))
    }

    static func elapsed(since start: ContinuousClock.Instant) -> Int {
        PipelineCore.milliseconds(ContinuousClock.now - start)
    }
}
