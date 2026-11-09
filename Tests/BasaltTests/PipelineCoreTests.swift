import Testing
import Foundation
@testable import BasaltCore

@Suite("Pipeline routing core: confident answers stay with System One, abstained questions and tasks go to System 2, the receipt names every stage (#402)")
struct PipelineCoreTests {
    static let color = SystemOneQuestion(type: .choice, instructions: "Which color is the car?", criteria: .named(["red": "Red.", "blue": "Blue."]))
    static let rain = SystemOneQuestion(type: .noul, instructions: "It rains.")
    static let size = SystemOneQuestion(type: .score, instructions: "How big?", criteria: .levels(["Small.", "Medium.", "Large."]))
    static let usage = SystemOneUsage(prefixTokens: 40, candidateBranches: 4, prefixEvaluations: 1, branchForwardCalls: 1, suffixTokens: 12)
    static let origin = RequestOrigin(source: "pipeline", endpoint: "/v1/pipeline", stream: false)

    static func answer(_ question: SystemOneQuestion, _ scores: [Double], abstainBelow: Double? = nil) -> SystemOneAnswer {
        SystemOneCore.gated(SystemOneCore.answer(question, scores: scores, temperature: 1), abstainBelow: abstainBelow, minMargin: nil)
    }

    static func decision(_ answers: [String: SystemOneAnswer]) -> SystemOneResponse {
        SystemOneResponse(model: "org/jev", answers: answers, usage: usage, calibrated: true)
    }

    @Test("Confident answers are returned as they are, labelled by their option, with no System 2 call")
    func confidentStaysWithSystemOne() {
        let answers = ["color": Self.answer(Self.color, [0, 3]), "rain": Self.answer(Self.rain, [0, 3]), "size": Self.answer(Self.size, [0, 0, 4])]
        let route = PipelineCore.route(questions: ["color": Self.color, "rain": Self.rain, "size": Self.size], decision: Self.decision(answers), task: nil, state: "s")
        #expect(route.calls.isEmpty)
        #expect(route.answers.mapValues(\.label) == ["color": "red", "rain": "true", "size": "2"])
        #expect(route.answers.values.allSatisfy { $0.answeredBy == .decide && $0.text == nil })
        #expect(route.answers["color"]?.decision == answers["color"])
    }

    @Test("An abstained question, or one System One did not answer, goes to System 2 as a short constrained prompt over the state")
    func abstainEscalates() throws {
        let abstained = Self.answer(Self.color, [0, 0.1], abstainBelow: 0.75)
        #expect(abstained.abstain == true)
        let route = PipelineCore.route(questions: ["color": Self.color, "rain": Self.rain], decision: Self.decision(["color": abstained]), task: nil, state: "The car is blue.")
        #expect(route.answers.isEmpty)
        #expect(route.calls.map(\.question) == ["color", "rain"])
        let prompt = try #require(route.calls.first?.prompt)
        #expect(prompt == "The car is blue.\n\nQuestion:\nWhich color is the car?\n\nOptions:\nblue: Blue.\nred: Red.\n\nAnswer with exactly one option label (blue, red) and nothing else.")
        let request = PipelineCore.request(for: try #require(route.calls.first), origin: Self.origin)
        #expect(request.maxTokens == PipelineCore.questionMaxTokens && request.temperature == 0 && request.enableThinking == false)
        #expect(request.messages?.map(\.text) == [prompt] && request.origin == Self.origin)
    }

    @Test("A task goes to System 2 with the perceived text as context and the server's own sampling and token limit")
    func taskEscalates() {
        let route = PipelineCore.route(questions: [:], decision: nil, task: "Summarize it.", state: "[transcript]\nhello")
        #expect(route.calls == [.task(prompt: "Context:\n[transcript]\nhello\n\nTask:\nSummarize it.")])
        let request = PipelineCore.request(for: route.calls[0], origin: Self.origin)
        #expect(request.maxTokens == nil && request.temperature == nil && request.enableThinking == true)
        #expect(PipelineCore.taskPrompt("Do it.", state: "") == "Do it.")
    }

    @Test("System 2 replies are read back to option labels; a task reply becomes system2; an unreadable reply keeps its text with no label")
    func repliesBecomeAnswers() {
        let abstained = Self.answer(Self.color, [0, 0.1], abstainBelow: 0.75)
        let questions = ["color": Self.color, "rain": Self.rain]
        let decision = Self.decision(["color": abstained, "rain": Self.answer(Self.rain, [0, 0.1], abstainBelow: 0.75)])
        let route = PipelineCore.route(questions: questions, decision: decision, task: "Why?", state: "s")
        let replies = [
            PipelineCore.System2Reply(call: route.calls[0], model: "chat", text: "Blue.", tokens: 50, ms: 5),
            PipelineCore.System2Reply(call: route.calls[1], model: "chat", text: "I cannot tell", tokens: 50, ms: 5),
            PipelineCore.System2Reply(call: route.calls[2], model: "chat", text: "Because.", tokens: 80, ms: 9),
        ]
        let (answers, system2) = PipelineCore.answers(route, replies: replies, questions: questions, decision: decision)
        #expect(answers["color"] == PipelineAnswer(answeredBy: .answer, label: "blue", decision: abstained, text: "Blue."))
        #expect(answers["rain"]?.label == nil && answers["rain"]?.text == "I cannot tell")
        #expect(system2 == PipelineSystem2(model: "chat", text: "Because."))
    }

    @Test("Labels: the whole reply, else the first word that is an option, case-insensitive")
    func labels() {
        #expect(PipelineCore.label(in: " RED. ", keys: ["blue", "red"]) == "red")
        #expect(PipelineCore.label(in: "The answer is blue, not red", keys: ["blue", "red"]) == "blue")
        #expect(PipelineCore.label(in: "2", keys: ["0", "1", "2"]) == "2")
        #expect(PipelineCore.label(in: "green", keys: ["blue", "red"]) == nil)
        #expect(PipelineCore.label(of: Self.answer(Self.rain, [3, 0]), Self.rain) == "false")
    }

    @Test("The receipt lists perceive, decide and each System 2 call with model, ms, tokens, confidence and escalation, and which stage answered each question")
    func receipt() {
        let decision = Self.decision(["color": Self.answer(Self.color, [0, 0.1], abstainBelow: 0.75), "rain": Self.answer(Self.rain, [0, 3], abstainBelow: 0.75)])
        let decided = PipelineCore.decided(decision, ms: 7)
        #expect(decided.tokens == 52 && decided.escalated && decided.model == "org/jev")
        #expect(decided.confidence == decision.answers["color"]?.confidence)
        let route = PipelineCore.route(questions: ["color": Self.color, "rain": Self.rain], decision: decision, task: "t", state: "s")
        let replies = route.calls.map { PipelineCore.System2Reply(call: $0, model: "chat", text: "red", tokens: 30, ms: 4) }
        let (answers, _) = PipelineCore.answers(route, replies: replies, questions: ["color": Self.color, "rain": Self.rain], decision: decision)
        let perceive = PipelineStageReceipt(stage: .perceive, model: "GLM-OCR", ms: 100, tokens: 64)
        let receipt = PipelineCore.receipt([perceive, decided], replies: replies, answers: answers, ms: 200)
        #expect(receipt.stages.map(\.stage) == [.perceive, .decide, .answer, .answer])
        #expect(receipt.stages.map(\.question) == [nil, nil, "color", nil])
        #expect(receipt.answeredBy == ["color": .answer, "rain": .decide])
        #expect(receipt.escalated && receipt.system2Calls == 2 && receipt.ms == 200)
        let json = String(decoding: (try? JSONEncoder().encode(receipt)) ?? Data(), as: UTF8.self)
        #expect(json.contains("\"answered_by\"") && json.contains("\"system2_calls\":2"))
    }

    @Test("The state joins OCR pages, the transcript and the client text")
    func state() {
        let perceived = PipelinePerceived(pages: [PipelinePage(attachment: 0, page: 2, markdown: "# Report")], transcript: "hello")
        #expect(PipelineCore.state(perceived, text: "Note.") == "[document 1, page 2]\n# Report\n\n[transcript]\nhello\n\nNote.")
        #expect(PipelineCore.state(PipelinePerceived(), text: "only") == "only")
    }

    @Test("A request needs input and a question or task; questions, pages and limits are checked at the boundary; missing stages are named")
    func requestValidation() throws {
        func refused(_ build: () throws -> PipelineRequest) -> Bool { (try? build()) == nil }
        #expect(refused { try PipelineRequest(attachments: [], text: "s", questions: nil, task: "  ", pages: nil) })
        #expect(refused { try PipelineRequest(attachments: [], text: " ", questions: nil, task: "t", pages: nil) })
        #expect(refused { try PipelineRequest(attachments: [], text: "s", questions: ["q": SystemOneQuestion(type: .choice, instructions: "x")], task: nil, pages: nil) })
        #expect(refused { try PipelineRequest(attachments: [], text: "s", questions: nil, task: "t", pages: "0-2") })
        #expect(refused { try PipelineRequest(attachments: [], text: String(repeating: "a", count: SystemOneCore.maxStateCharacters + 1), questions: nil, task: "t", pages: nil) })
        #expect(throws: BasaltError.self) { try PipelineAttachment(kind: "video", encoded: "AA==", format: nil) }
        #expect(throws: BasaltError.self) { try PipelineAttachment(kind: "audio", data: Data([1]), format: nil) }
        let request = try PipelineRequest(attachments: [], text: "s", questions: ["q": Self.rain], task: nil, pages: "1-2")
        #expect(PipelineCore.missingStage(request, hasOCR: false, hasDecisions: false)?.contains("--decision-model") == true)
        #expect(PipelineCore.missingStage(request, hasOCR: false, hasDecisions: true) == nil)
        let image = try PipelineAttachment(kind: "image", data: try OCRFixture.png(width: 2, height: 2), format: nil)
        let document = try PipelineRequest(attachments: [image], text: nil, questions: nil, task: "t", pages: nil)
        #expect(PipelineCore.missingStage(document, hasOCR: false, hasDecisions: false)?.contains("--pipeline-ocr-model") == true)
    }
}
