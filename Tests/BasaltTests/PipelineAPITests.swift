import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltServer

struct GatingDecisionEngine: DecisionEngine {
    let modelID = "org/stub-jev"
    let calls = DecisionCalls()
    var flat: Set<String> = []

    func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        try SystemOneCore.validate(request)
        calls.requests.withLock { $0.append(request) }
        let answers = Dictionary(uniqueKeysWithValues: request.questions.map { id, question in
            let count = SystemOneCore.options(question).keys.count
            let scores = flat.contains(id) ? Array(repeating: 0.0, count: count) : (0 ..< count).map { Double($0) * 3 }
            return (id, SystemOneCore.gated(
                SystemOneCore.answer(question, scores: scores, temperature: 1), abstainBelow: request.abstainBelow, minMargin: request.minMargin))
        })
        return SystemOneResponse(
            model: modelID, answers: answers,
            usage: SystemOneUsage(prefixTokens: 10, candidateBranches: 2, prefixEvaluations: 1, branchForwardCalls: 1, suffixTokens: 8), calibrated: true)
    }
}

@Suite("/v1/pipeline and basalt_pipeline: perceive → decide → answer on stub engines, refusals, the receipt in the session log (#402)")
struct PipelineAPITests {
    static var questions: [String: Any] { [
        "color": ["type": "choice", "instructions": "Which color is the car?", "criteria": ["red": "Red.", "blue": "Blue."]],
        "hard": ["type": "choice", "instructions": "Which wheel is flat?", "criteria": ["left": "Left.", "right": "Right."]],
    ] }

    static func chat(_ text: @escaping @Sendable (GenerationRequest) -> String = { _ in "left" }, log: SessionLogWriter? = nil) async -> (MockInferenceEngine, ModelOrchestratorActor) {
        let engine = MockInferenceEngine()
        await engine.setResponder(text)
        return (engine, ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker(), sessionLog: log))
    }

    static func post(
        _ chat: ModelOrchestratorActor, decisions: (any DecisionEngine)?, stages: PipelineStages = PipelineStages(abstainBelow: 0.5), _ body: [String: Any]
    ) async throws -> (status: HTTPResponse.Status, json: [String: Any]) {
        let server = RESTServer(orchestrator: chat, config: try ServerConfig(modelPath: "org/mock-model")).withDecisions(decisions).withPipeline(stages)
        let data = try JSONSerialization.data(withJSONObject: body)
        return try await Application(router: server.makeRouter()).test(.router) { client in
            try await client.execute(uri: "/v1/pipeline", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(bytes: data)) {
                ($0.status, (try? JSONSerialization.jsonObject(with: Data(buffer: $0.body)) as? [String: Any]) ?? [:])
            }
        }
    }

    static func answers(_ json: [String: Any]) -> [String: [String: Any]] { json["answers"] as? [String: [String: Any]] ?? [:] }
    static func stages(_ json: [String: Any]) -> [[String: Any]] { (json["receipt"] as? [String: Any])?["stages"] as? [[String: Any]] ?? [] }

    @Test("Confident questions are answered by System One alone: the chat model gets no request")
    func confidentSkipsSystem2() async throws {
        let (engine, chat) = await Self.chat()
        let decisions = GatingDecisionEngine()
        let response = try await Self.post(chat, decisions: decisions, ["text": "The car is red.", "questions": Self.questions])
        #expect(response.status == .ok)
        #expect(Self.answers(response.json).mapValues { $0["answered_by"] as? String } == ["color": "decide", "hard": "decide"])
        #expect(Self.answers(response.json)["color"]?["label"] as? String == "red")
        #expect(response.json["system2"] == nil)
        #expect(Self.stages(response.json).map { $0["stage"] as? String } == ["decide"])
        #expect(await engine.getRequests().isEmpty)
        #expect(decisions.calls.requests.withLock { $0.first?.abstainBelow } == 0.5)
    }

    @Test("An abstained question goes to the chat model as a constrained prompt and its label comes back; a task gets the chat reply as system2")
    func abstainAndTaskReachSystem2() async throws {
        let (engine, chat) = await Self.chat { $0.prompt.contains("Options:") ? "Left." : "The car is red and a wheel is flat." }
        let response = try await Self.post(
            chat, decisions: GatingDecisionEngine(flat: ["hard"]), ["text": "The car is red.", "questions": Self.questions, "task": "Describe the car."])
        #expect(response.status == .ok)
        let hard = try #require(Self.answers(response.json)["hard"])
        #expect(hard["answered_by"] as? String == "answer" && hard["label"] as? String == "left" && hard["text"] as? String == "Left.")
        #expect((hard["decision"] as? [String: Any])?["abstain"] as? Bool == true)
        #expect(Self.answers(response.json)["color"]?["answered_by"] as? String == "decide")
        #expect((response.json["system2"] as? [String: Any])?["text"] as? String == "The car is red and a wheel is flat.")
        let receipt = try #require(response.json["receipt"] as? [String: Any])
        #expect(receipt["escalated"] as? Bool == true && receipt["system2_calls"] as? Int == 2)
        #expect(Self.stages(response.json).map { $0["question"] as? String } == [nil, "hard", nil])
        let requests = await engine.getRequests()
        #expect(requests.count == 2)
        #expect(requests.first?.prompt.hasPrefix("The car is red.\n\nQuestion:\nWhich wheel is flat?") == true && requests.first?.maxTokens == PipelineCore.questionMaxTokens)
        #expect(requests.last?.prompt == "Context:\nThe car is red.\n\nTask:\nDescribe the car.")
    }

    @Test("Perceive: an image is read by the OCR orchestrator, audio by the chat orchestrator's transcriber, and both reach System One's state")
    func perceiveFeedsDecide() async throws {
        let ocrEngine = MockInferenceEngine()
        await ocrEngine.setResponder { _ in "Invoice total: 42" }
        let ocr = try await OCRAPITests.orchestrator("GLM-OCR-bf16", modelType: "glm_ocr", engine: ocrEngine)
        let (engine, chat) = await Self.chat()
        await engine.setMockTranscription(Transcription(text: "pay it today"))
        let decisions = GatingDecisionEngine()
        let attachments: [[String: Any]] = [
            ["kind": "image", "data": try OCRFixture.png(width: 4, height: 4).base64EncodedString()],
            ["kind": "audio", "data": AudioFixture.sine(seconds: 0.2).base64EncodedString(), "format": "wav"],
        ]
        let response = try await Self.post(chat, decisions: decisions, stages: PipelineStages(ocr: ocr), ["attachments": attachments, "questions": Self.questions])
        #expect(response.status == .ok)
        let perceived = try #require(response.json["perceived"] as? [String: Any])
        #expect((perceived["pages"] as? [[String: Any]])?.first?["markdown"] as? String == "Invoice total: 42")
        #expect(perceived["transcript"] as? String == "pay it today")
        #expect(decisions.calls.requests.withLock { $0.first?.state } == "[document 1, page 1]\nInvoice total: 42\n\n[transcript]\npay it today")
        #expect(Self.stages(response.json).map { $0["stage"] as? String } == ["perceive", "perceive", "decide"])
        #expect(Self.stages(response.json).first?["model"] as? String == "GLM-OCR-bf16" && Self.stages(response.json).first?["tokens"] as? Int == 10)
        #expect(await ocrEngine.getRequests().count == 1)
        #expect(await engine.getRequests().isEmpty)
    }

    @Test("Refusals: an image without an OCR model and questions without a decision model are 404 naming the flag; malformed input is 400; nothing runs")
    func refusals() async throws {
        let (engine, chat) = await Self.chat()
        let png = try OCRFixture.png(width: 2, height: 2).base64EncodedString()
        let noOCR = try await Self.post(chat, decisions: GatingDecisionEngine(), ["attachments": [["kind": "image", "data": png]], "task": "t"])
        #expect(noOCR.status == .notFound)
        #expect(((noOCR.json["error"] as? [String: Any])?["message"] as? String)?.contains("--pipeline-ocr-model") == true)
        let noDecider = try await Self.post(chat, decisions: nil, ["text": "s", "questions": Self.questions])
        #expect(noDecider.status == .notFound)
        #expect(((noDecider.json["error"] as? [String: Any])?["message"] as? String)?.contains("--decision-model") == true)
        let bodies: [[String: Any]] = [
            ["text": "s"], ["task": "t"], ["text": "s", "task": "t", "attachments": [["kind": "video", "data": png]]],
            ["text": "s", "task": "t", "attachments": [["kind": "audio", "data": png]]], ["text": "s", "task": "t", "attachments": [["kind": "image", "path": "/etc/hosts"]]],
            ["text": "s", "questions": ["q": ["type": "choice", "instructions": "x", "criteria": ["a": "A"]]]], ["text": "s", "task": "t", "pages": "x"],
        ]
        for body in bodies {
            #expect(try await Self.post(chat, decisions: GatingDecisionEngine(), body).status == .badRequest, "\(body.keys.sorted())")
        }
        #expect(await engine.getRequests().isEmpty)
    }

    @Test("The route receipt is written to the session log as a pipeline record next to the decision record")
    func receiptInSessionLog() async throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "pipeline-log-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let writer = SessionLogWriter(mode: .meta, directory: dir, defaultSession: "d")
        let (_, chat) = await Self.chat(log: writer)
        #expect(try await Self.post(chat, decisions: GatingDecisionEngine(flat: ["hard"]), ["text": "The car is red.", "questions": Self.questions]).status == .ok)
        await writer.flush()
        await writer.close()
        let text = try String(contentsOf: try #require(SessionLogReader.file(for: "d", in: dir)), encoding: .utf8)
        let lines = text.split(separator: "\n")
        #expect(lines.contains { $0.contains("\"type\":\"decision\"") && $0.contains("/v1/pipeline") })
        let line = try #require(lines.first { $0.contains("\"type\":\"pipeline\"") })
        let record = try #require(JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
        let receipt = try #require(record["receipt"] as? [String: Any])
        #expect(receipt["answered_by"] as? [String: String] == ["color": "decide", "hard": "answer"])
        #expect(record["endpoint"] as? String == "/v1/pipeline")
    }

    @Test("basalt_pipeline is listed in the media group, uses the same parser and core, and honours the --image-dir sandbox")
    func mcpTool() async throws {
        let ocrEngine = MockInferenceEngine()
        await ocrEngine.setResponder { _ in "Scanned: the car is blue" }
        let ocr = try await OCRAPITests.orchestrator("GLM-OCR", modelType: "glm_ocr", engine: ocrEngine)
        let (_, chat) = await Self.chat()
        let inside = FileManager.default.temporaryDirectory.appendingPathComponent("pipeline-mcp-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: inside, withIntermediateDirectories: true)
        let image = inside.appendingPathComponent("scan.png")
        try OCRFixture.png(width: 4, height: 4).write(to: image)
        let lines = MCPLines()
        let server = MCPServer(orchestrator: chat, writer: { lines.append($0) })
            .withImageDirectories([inside]).withDecisions(GatingDecisionEngine()).withPipeline(PipelineStages(ocr: ocr))
        #expect(MCPTool(name: "basalt_pipeline") == .media(.pipeline))
        #expect(MCPToolCatalog.list.tools.contains { $0.name == "basalt_pipeline" })
        await server.dispatch(toolCall(1, "basalt_pipeline", ["attachments": [["kind": "image", "path": image.path]], "questions": Self.questions]))
        let result = try lines.toolText(0)
        #expect((result["perceived"] as? [String: Any]).map { ($0["pages"] as? [[String: Any]])?.first?["markdown"] as? String } == "Scanned: the car is blue")
        #expect((result["answers"] as? [String: [String: Any]])?["color"]?["answered_by"] as? String == "decide")
        await server.dispatch(toolCall(2, "basalt_pipeline", ["attachments": [["kind": "image", "path": "/etc/hosts"]], "task": "t"]))
        #expect(lines.all[1].contains("--image-dir"))
        await MCPServer(orchestrator: chat, writer: { lines.append($0) }).dispatch(toolCall(3, "basalt_pipeline", ["text": "s", "questions": Self.questions]))
        #expect(try lines.json(2)["error"] != nil && lines.all[2].contains("--decision-model"))
    }
}

@Suite("/v1/pipeline end to end on the tiny model: an abstaining decision engine sends the question to the real chat engine below BasaltServer (#402, #228)")
struct PipelineTinyModelTests {
    @Test("System 2 is the tiny MLX engine: the answer text equals its greedy reply to the constrained prompt, and the receipt counts its tokens")
    func abstainReachesTinyModel() async throws {
        let question = SystemOneQuestion(type: .noul, instructions: "The car is red.")
        let state = "The car is red."
        let call = PipelineCore.System2Call.question(id: "red", prompt: PipelineCore.prompt(question, state: state))
        let reference = try await TinyModel.engine().generate(request: PipelineCore.request(for: call, origin: .unspecified)).generatedText
        let chat = ModelOrchestratorActor(engine: try await TinyModel.engine(), telemetry: TelemetryTracker())
        let body: [String: Any] = ["text": state, "questions": ["red": ["type": "noul", "instructions": "The car is red."]]]
        let response = try await PipelineAPITests.post(chat, decisions: GatingDecisionEngine(flat: ["red"]), stages: PipelineStages(abstainBelow: 1), body)
        #expect(response.status == .ok)
        let answer = try #require(PipelineAPITests.answers(response.json)["red"])
        #expect(answer["answered_by"] as? String == "answer")
        #expect(answer["text"] as? String == reference)
        let system2 = try #require(PipelineAPITests.stages(response.json).last)
        #expect(system2["stage"] as? String == "answer" && system2["question"] as? String == "red")
        #expect((system2["tokens"] as? Int ?? 0) > ByteTokenizer().encode(text: state, addSpecialTokens: false).count)
    }
}
