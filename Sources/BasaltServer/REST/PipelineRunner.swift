import Foundation
import BasaltCore

package struct PipelineStages: Sendable {
    package let ocr: ModelOrchestratorActor?
    package let abstainBelow: Double?
    package let minMargin: Double?

    package init(ocr: ModelOrchestratorActor? = nil, abstainBelow: Double? = nil, minMargin: Double? = nil) {
        (self.ocr, self.abstainBelow, self.minMargin) = (ocr, abstainBelow, minMargin)
    }
}

struct PipelineRunner: Sendable {
    let chat: ModelOrchestratorActor
    let decisions: (any DecisionEngine)?
    let stages: PipelineStages

    func missingStage(_ request: PipelineRequest) -> String? {
        PipelineCore.missingStage(request, hasOCR: stages.ocr != nil, hasDecisions: decisions != nil)
    }

    func run(_ request: PipelineRequest, origin: RequestOrigin, sessionId: String?) async throws(BasaltError) -> PipelineResponse {
        if let missing = missingStage(request) { throw .invalidRequestStructure(details: missing) }
        let started = ContinuousClock.now
        let (perceived, perceiving) = try await perceive(request)
        let state = PipelineCore.state(perceived, text: request.text)
        let (decision, deciding) = try await decide(request.questions, state: state, origin: origin, sessionId: sessionId)
        let route = PipelineCore.route(questions: request.questions, decision: decision, task: request.task, state: state)
        var replies: [PipelineCore.System2Reply] = []
        for call in route.calls { replies.append(try await answer(call, origin: origin, sessionId: sessionId)) }
        let (answers, system2) = PipelineCore.answers(route, replies: replies, questions: request.questions, decision: decision)
        let receipt = PipelineCore.receipt(
            perceiving + [deciding].compactMap { $0 }, replies: replies, answers: answers, ms: Self.elapsed(since: started))
        await chat.sessionLog?.logPipeline(origin: origin, sessionId: sessionId, receipt: receipt)
        return PipelineResponse(perceived: perceived, answers: answers, system2: system2, receipt: receipt)
    }

    func perceive(_ request: PipelineRequest) async throws(BasaltError) -> (PipelinePerceived, [PipelineStageReceipt]) {
        var (pages, transcripts, receipts) = ([PipelinePage](), [String](), [PipelineStageReceipt]())
        for (index, attachment) in request.attachments.enumerated() {
            let started = ContinuousClock.now
            switch attachment {
            case .document(let document):
                let result = try await read(document, pages: request.pages)
                pages += result.pages.map { PipelinePage(attachment: index, page: $0.page, markdown: $0.markdown) }
                receipts.append(PipelineStageReceipt(
                    stage: .perceive, model: result.model, ms: Self.elapsed(since: started), tokens: result.pages.reduce(0) { $0 + $1.tokens }))
            case .audio(let clip):
                transcripts.append(try await chat.transcribe(TranscriptionRequest(audio: clip)).text)
                receipts.append(PipelineStageReceipt(stage: .perceive, model: await chat.modelInfo, ms: Self.elapsed(since: started), tokens: 0))
            }
        }
        let transcript = transcripts.isEmpty ? nil : transcripts.joined(separator: "\n\n")
        return (PipelinePerceived(pages: pages.isEmpty ? nil : pages, transcript: transcript), receipts)
    }

    func read(_ document: OCRDocument, pages: String?) async throws(BasaltError) -> OCRResult {
        guard let ocr = stages.ocr else { throw .invalidRequestStructure(details: "pipeline: no OCR model is configured") }
        let selection = if case .pdf = document { pages } else { String?.none }
        return try await ocr.ocr(OCRRequest(document: document, pages: selection, task: nil, format: nil, model: nil))
    }

    func decide(
        _ questions: [String: SystemOneQuestion], state: String, origin: RequestOrigin, sessionId: String?
    ) async throws(BasaltError) -> (SystemOneResponse?, PipelineStageReceipt?) {
        guard !questions.isEmpty, let decisions else { return (nil, nil) }
        let started = ContinuousClock.now
        let request = SystemOneRequest(state: state, questions: questions, abstainBelow: stages.abstainBelow, minMargin: stages.minMargin)
        let response = try await SystemOneController.decide(decisions, request, log: chat.sessionLog, origin: origin, sessionId: sessionId)
        return (response, PipelineCore.decided(response, ms: Self.elapsed(since: started)))
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
