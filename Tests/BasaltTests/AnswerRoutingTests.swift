import Testing
import Foundation
import Synchronization
@testable import BasaltCore
@testable import BasaltServer

final class CallCount: Sendable {
    let value = Mutex(0)
}

struct StubAnswer: AnswerTarget {
    let name: String
    var isCloud = false
    var reason: String?
    var failure: BasaltError?
    let calls = CallCount()

    func unavailable() async -> String? { reason }

    func answer(_ request: GenerationRequest) async throws(BasaltError) -> AnswerReply {
        calls.value.withLock { $0 += 1 }
        if let failure { throw failure }
        return AnswerReply(text: "\(name) says hi", model: name, tokens: 7, cost: isCloud ? 0.001 : nil)
    }
}

@Suite("Answer routes (MoM): the first matching rule picks the answer model, a fallback follows it, cloud models never see perceived text (#412 phase 2)", .timeLimit(.minutes(1)))
struct AnswerRoutingTests {
    static let routes = [
        AnswerRoute(when: .init(decide: ["route": "code"], costLeft: true), model: "frontier", fallback: "big"),
        AnswerRoute(when: .init(promptTokensOver: 8000), model: "big"),
        AnswerRoute(when: .init(modality: .documents, escalated: true), model: "big"),
        AnswerRoute(model: "chat"),
    ]

    static func situation(_ modalities: Set<AnswerRoute.Modality> = [.text], labels: [String: String] = [:], tokens: Int = 100, escalated: Bool = false, costLeft: Set<String> = ["frontier"]) -> AnswerSituation {
        AnswerSituation(modalities: modalities, labels: labels, promptTokens: tokens, escalated: escalated, costLeft: costLeft)
    }

    @Test("Each condition gates its rule; rules are tried in order; the default rule catches the rest")
    func choose() {
        #expect(AnswerRouting.choose(Self.routes, Self.situation(labels: ["route": "code"])) == .init(rule: 0, chain: ["frontier", "big"]))
        #expect(AnswerRouting.choose(Self.routes, Self.situation(labels: ["route": "code"], costLeft: []))?.rule == 3)
        #expect(AnswerRouting.choose(Self.routes, Self.situation(labels: ["route": "chat"], tokens: 9000))?.rule == 1)
        #expect(AnswerRouting.choose(Self.routes, Self.situation(tokens: 8000))?.rule == 3)
        #expect(AnswerRouting.choose(Self.routes, Self.situation([.documents], escalated: true))?.rule == 2)
        #expect(AnswerRouting.choose(Self.routes, Self.situation([.documents], escalated: false))?.rule == 3)
        #expect(AnswerRouting.choose(Array(Self.routes.prefix(1)), Self.situation()) == nil)
        #expect(AnswerSituation.estimatedTokens(String(repeating: "a", count: 9)) == 3)
    }

    static func router(frontier: StubAnswer, big: StubAnswer = StubAnswer(name: "big"), chat: StubAnswer = StubAnswer(name: "chat")) -> AnswerRouter {
        AnswerRouter(routes: Self.routes, targets: ["frontier": frontier, "big": big, "chat": chat], fallback: chat)
    }

    @Test("The router skips a cloud model for perceived text, one without budget and one that fails, naming why, and ends at the fallback")
    func router() async throws {
        let request = GenerationRequest(prompt: "Write a function.")
        let code = Self.situation(labels: ["route": "code"])
        let frontier = StubAnswer(name: "frontier", isCloud: true)
        let routed = try await Self.router(frontier: frontier).answer(request, situation: code)
        #expect(routed.reply.model == "frontier" && routed.rule == 0 && routed.skipped.isEmpty && routed.reply.cost == 0.001)
        let perceived = try await Self.router(frontier: frontier).answer(request, situation: Self.situation([.documents, .text], labels: ["route": "code"]))
        #expect(perceived.reply.model == "big" && perceived.skipped == ["frontier: documents and transcripts stay local"])
        let broke = StubAnswer(name: "frontier", isCloud: true, reason: "daily budget spent")
        #expect(try await Self.router(frontier: broke).answer(request, situation: code).skipped == ["frontier: daily budget spent"])
        let failing = try await Self.router(frontier: StubAnswer(name: "frontier", isCloud: true, failure: .modelInferenceError(details: "timeout")),
                                            big: StubAnswer(name: "big", failure: .modelNotLoaded)).answer(request, situation: code)
        #expect(failing.reply.model == "chat" && failing.skipped.count == 2 && failing.rule == 0)
        #expect(frontier.calls.value.withLock { $0 } == 1)
    }

    @Test("End to end on /v1/pipeline: a typed request decided as code goes to the cloud model, a document request with the same label stays local, and the receipt names the rule, the cost and what was skipped")
    func pipeline() async throws {
        let (_, chat) = try await PipelineSwapEdgeTests.chat()
        let frontier = StubAnswer(name: "frontier", isCloud: true)
        let routes = [AnswerRoute(when: .init(decide: ["route": "red"]), model: "frontier", fallback: "chat"), AnswerRoute(model: "chat")]
        let ocrEngine = MockInferenceEngine()
        await ocrEngine.setResponder { _ in "Invoice total: 42" }
        var stages = PipelineStages(ocr: try await OCRAPITests.orchestrator("GLM-OCR", modelType: "glm_ocr", engine: ocrEngine), abstainBelow: 0.5)
        stages.answers = AnswerRouter(routes: routes, targets: ["frontier": frontier, "chat": LocalAnswer(chat)], fallback: LocalAnswer(chat))
        let questions: [String: Any] = ["route": ["type": "choice", "instructions": "Which kind?", "criteria": ["red": "Code.", "blue": "Chat."]]]
        let typed = try await PipelineAPITests.post(chat, decisions: StubDecisionEngine(), stages: stages, ["text": "Write a sort.", "questions": questions, "task": "Answer it."])
        #expect(typed.status == .ok)
        let answer = PipelineAPITests.stages(typed.json).last
        #expect(answer?["model"] as? String == "frontier" && answer?["route"] as? Int == 0 && answer?["cost"] as? Double == 0.001)
        let document = try await PipelineAPITests.post(chat, decisions: StubDecisionEngine(), stages: stages, [
            "attachments": [PipelineSwapEdgeTests.attachments[1]], "questions": questions, "task": "Answer it."])
        #expect(document.status == .ok)
        let local = PipelineAPITests.stages(document.json).last
        #expect(local?["model"] as? String == "chat" && (local?["skipped"] as? [String])?.first?.contains("stay local") == true)
        #expect(frontier.calls.value.withLock { $0 } == 1)
    }
}
