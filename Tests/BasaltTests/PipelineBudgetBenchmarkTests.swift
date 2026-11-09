import Foundation
import Testing
import CoreGraphics
import CoreText
import Hummingbird
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer
@testable import BasaltSpeech

struct PipelineWorkItem {
    enum Input {
        case document(Data)
        case audio(Data)
        case typed(String)
    }

    let input: Input
    let decisions: [LabelledDecision]

    var kind: String {
        switch input {
        case .document: "document"
        case .audio: "audio"
        case .typed: "typed"
        }
    }

    var questions: [String: SystemOneQuestion] { Dictionary(uniqueKeysWithValues: decisions.map { ($0.name, $0.question) }) }
}

struct PipelineTally {
    var correct = 0
    var total = 0
    var llmCalls = 0
    var llmTokens = 0
    var escalated = 0
    var perceiveMs = 0
    var decideMs = 0
    var answerMs = 0
    var wallMs = 0
    var kinds: [String: (correct: Int, total: Int)] = [:]

    var accuracy: Double { Double(correct) / Double(max(total, 1)) }

    mutating func score(_ item: PipelineWorkItem, labels: [String: String?]) {
        for decision in item.decisions {
            let ok = labels[decision.name] == decision.label
            (total, correct) = (total + 1, correct + (ok ? 1 : 0))
            let kind = kinds[item.kind] ?? (0, 0)
            kinds[item.kind] = (kind.correct + (ok ? 1 : 0), kind.total + 1)
        }
    }

    func summary(states: Int) -> String {
        let per = { (ms: Int) in String(format: "%.0f", Double(ms) / Double(max(states, 1))) }
        let breakdown = kinds.keys.sorted().map { "\($0) \(kinds[$0]!.correct)/\(kinds[$0]!.total)" }.joined(separator: ", ")
        return "accuracy \(String(format: "%.4f", accuracy)) [\(breakdown)], escalated \(escalated)/\(total), \(llmCalls) LLM calls, \(llmTokens) LLM tokens, "
            + "per state ms: perceive \(per(perceiveMs)) decide \(per(decideMs)) answer \(per(answerMs)) end-to-end \(per(wallMs)); total \(String(format: "%.1f", Double(wallMs) / 1000)) s"
    }
}

@Suite("Pipeline budget: one /v1/pipeline call against the client chaining /v1/ocr or /v1/audio/transcriptions, /v1/systemone and chat, and against LLM-only (#402 phase 2)", .serialized)
struct PipelineBudgetBenchmarkTests {
    static let env = ProcessInfo.processInfo.environment
    static let chatModel = env["BASALT_E2E"] == "1" ? env["BASALT_E2E_PIPELINE_CHAT_MODEL"] : nil
    static let threshold = Double(env["BASALT_E2E_PIPELINE_ABSTAIN_BELOW"] ?? "") ?? 0.75
    static let decodeRule = 0.07

    static func prose(_ state: String) -> String {
        state.replacing(/Order: \{"id":(\d+),"promised_days":(\d+),"delivered_days":(\d+)\}/) { match in
            "Order \(match.1) was promised in \(match.2) days and delivered in \(match.3) days."
        }
    }

    static func page(_ text: String) throws -> Data {
        let lines = text.split(separator: "\n").map(String.init)
        let (width, height) = (1800, 120 + 70 * lines.count)
        let context = try #require(CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName("Helvetica" as CFString, 30, nil),
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(red: 0, green: 0, blue: 0, alpha: 1),
        ]
        for (index, line) in lines.enumerated() {
            context.textPosition = CGPoint(x: 60, y: height - 80 - 70 * index)
            CTLineDraw(CTLineCreateWithAttributedString(NSAttributedString(string: line, attributes: attributes)), context)
        }
        return try OCRImage.png(try #require(context.makeImage()))
    }

    static func workload(states: Int) throws -> [PipelineWorkItem] {
        let decisions = SystemOneDecisionSet.generate(states: states, seed: 402)
        return try stride(from: 0, to: decisions.count, by: 3).enumerated().map { index, start in
            let group = Array(decisions[start ..< start + 3])
            switch index % 3 {
            case 0: return PipelineWorkItem(input: .document(try page(prose(group[0].state))), decisions: group)
            case 1: return PipelineWorkItem(input: .audio(try NemotronTranscribeRealModelTests.synthesizeWAV(text: prose(group[0].state))), decisions: group)
            default: return PipelineWorkItem(input: .typed(group[0].state), decisions: group)
            }
        }
    }

    struct BenchClient: Sendable {
        let base: URL

        func post(_ uri: String, _ body: Data, type: String = "application/json") async throws -> Data {
            var request = URLRequest(url: base.appending(path: uri), timeoutInterval: 900)
            (request.httpMethod, request.httpBody) = ("POST", body)
            request.setValue(type, forHTTPHeaderField: "Content-Type")
            let (data, response) = try await URLSession.shared.data(for: request)
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            guard status == 200 else { throw BasaltError.inferenceError("\(uri) \(status): \(String(decoding: data.prefix(300), as: UTF8.self))") }
            return data
        }
    }

    static func serve<T>(_ server: RESTServer, _ body: (BenchClient) async throws -> T) async throws -> T {
        let port = Int(env["BASALT_E2E_PIPELINE_PORT"] ?? "") ?? 18402
        let (ready, signal) = AsyncStream<Void>.makeStream()
        let app = Application(router: server.makeRouter(), configuration: .init(address: .hostname("127.0.0.1", port: port)), onServerRunning: { _ in signal.yield() })
        let running = Task { try await app.runService(gracefulShutdownSignals: []) }
        defer { running.cancel() }
        for await _ in ready { break }
        return try await body(BenchClient(base: try #require(URL(string: "http://127.0.0.1:\(port)"))))
    }

    static func json(_ client: BenchClient, _ uri: String, _ body: [String: Any]) async throws -> [String: Any] {
        let data = try await client.post(uri, try JSONSerialization.data(withJSONObject: body))
        return try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    static func questionsJSON(_ item: PipelineWorkItem) throws -> Any {
        try JSONSerialization.jsonObject(with: try JSONEncoder().encode(item.questions))
    }

    static func milliseconds(since start: ContinuousClock.Instant) -> Int { PipelineCore.milliseconds(ContinuousClock.now - start) }

    static func longAnswer(_ chat: ModelOrchestratorActor) async throws -> GenerationResponse {
        var request = GenerationRequest(prompt: SystemOneBudgetBenchmarkTests.longPrompt, maxTokens: 1024, temperature: 0)
        request.reasoningEffort = "none"
        return try await chat.generate(request: request)
    }

    static func server(_ path: String, log: ModelRunLog) async throws -> (ModelOrchestratorActor, RESTServer) {
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        let chat = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await log.step("load chat") { try await chat.loadModel(path: path) }
        await chat.setSpeechFallback(SpeechAnalyzerTranscriber())
        await chat.setDocumentTranscriber(VisionDocumentTranscriber())
        let decisions = ReadoutDecisionEngine(engine: engine, modelID: "self", readout: .uncalibrated)
        let stages = PipelineStages(speech: SpeechAnalyzerTranscriber(), abstainBelow: threshold)
        return (chat, RESTServer(orchestrator: chat, config: try ServerConfig(modelPath: path)).withDecisions(decisions).withPipeline(stages))
    }

    struct Arms {
        var chained = PipelineTally()
        var piped = PipelineTally()
        var llm = PipelineTally()
    }

    static func arms(_ client: BenchClient, _ items: [PipelineWorkItem], log: ModelRunLog) async throws -> Arms {
        var (arms, states) = (Arms(), [String]())
        try await log.step("arms a+b, alternating order") {
            for (index, item) in items.enumerated() {
                if index.isMultiple(of: 2) {
                    states.append(try await chained(client, item, tally: &arms.chained))
                    try await piped(client, item, tally: &arms.piped)
                } else {
                    try await piped(client, item, tally: &arms.piped)
                    states.append(try await chained(client, item, tally: &arms.chained))
                }
            }
        }
        try await log.step("arm llm-only") {
            for (item, state) in zip(items, states) { try await llmOnly(client, item, state: state, tally: &arms.llm) }
        }
        return arms
    }

    static func overlap(_ client: BenchClient, _ chat: ModelOrchestratorActor, _ items: [PipelineWorkItem]) async throws -> (GenerationResponse, PipelineTally) {
        async let long = longAnswer(chat)
        var tally = PipelineTally()
        for item in items { try await piped(client, item, tally: &tally) }
        return (try await long, tally)
    }

    static func verdicts(_ arms: Arms, solo: GenerationResponse, overlapped: GenerationResponse) -> [String] {
        let saved = 1 - Double(arms.piped.llmTokens) / Double(max(arms.llm.llmTokens, 1))
        let drop = 1 - overlapped.tokensPerSecond / max(solo.tokensPerSecond, 1e-9)
        let mark = { (ok: Bool) in ok ? "PASS" : "FAIL" }
        return [
            "\(mark(arms.piped.wallMs < arms.chained.wallMs)) b faster than a: \(arms.piped.wallMs) ms vs \(arms.chained.wallMs) ms (\(String(format: "%+.1f", (Double(arms.piped.wallMs) / Double(max(arms.chained.wallMs, 1)) - 1) * 100))%)",
            "\(mark(saved >= 0.5)) LLM tokens b vs llm-only: \(arms.piped.llmTokens) vs \(arms.llm.llmTokens) (\(String(format: "%.1f", saved * 100))% fewer, need >= 50%)",
            "\(mark(abs(arms.piped.accuracy - arms.llm.accuracy) <= 0.02 || arms.piped.accuracy > arms.llm.accuracy)) accuracy b vs llm-only: \(String(format: "%.4f", arms.piped.accuracy)) vs \(String(format: "%.4f", arms.llm.accuracy)) (within 2 points)",
            "\(mark(drop <= decodeRule)) System 2 decode: \(String(format: "%.2f", solo.tokensPerSecond)) tok/s solo, \(String(format: "%.2f", overlapped.tokensPerSecond)) tok/s beside the pipeline (\(String(format: "%+.1f", -drop * 100))%, #273 rule <= \(Int(decodeRule * 100))%)",
        ]
    }

    @Test("Mixed documents, audio and typed questions: per-stage and end-to-end latency, LLM tokens, decision accuracy and System 2 decode tok/s for a (chained endpoints), b (/v1/pipeline) and LLM-only",
          .enabled(if: chatModel != nil, "set BASALT_E2E=1 and BASALT_E2E_PIPELINE_CHAT_MODEL; optional BASALT_E2E_PIPELINE_STATES (default 60) and BASALT_E2E_PIPELINE_ABSTAIN_BELOW (default 0.75)"))
    func budget() async throws {
        let path = NF4Run.path(try #require(Self.chatModel))
        let log = try await ModelRunLog("pipeline-budget", modelPath: path)
        defer { log.releaseGPU() }
        let count = Int(Self.env["BASALT_E2E_PIPELINE_STATES"] ?? "") ?? 60
        let items = try await log.step("workload: \(count) states, documents rendered, audio from say") { try Self.workload(states: count) }
        let (chat, server) = try await Self.server(path, log: log)
        try await Self.serve(server) { client in
            try await log.step("warm-up, one per modality, not counted") {
                var warm = PipelineTally()
                for item in items.prefix(3) { _ = try await Self.chained(client, item, tally: &warm) }
            }
            let solo = try await log.step("long answer solo") { try await Self.longAnswer(chat) }
            let arms = try await Self.arms(client, items, log: log)
            log.record("RESULT", "abstain_below \(Self.threshold), \(items.count) states (\(items.filter { $0.kind == "document" }.count) documents, \(items.filter { $0.kind == "audio" }.count) audio, \(items.filter { $0.kind == "typed" }.count) typed)")
            log.record("RESULT", "a chained endpoints: \(arms.chained.summary(states: items.count))")
            log.record("RESULT", "b /v1/pipeline: \(arms.piped.summary(states: items.count))")
            log.record("RESULT", "llm-only: \(arms.llm.summary(states: items.count))")
            let (overlapped, beside) = try await log.step("long answer beside b") { try await Self.overlap(client, chat, Array(items.prefix(6))) }
            log.record("RESULT", "b beside the long answer: \(beside.summary(states: 6))")
            for verdict in Self.verdicts(arms, solo: solo, overlapped: overlapped) { log.record("VERDICT", verdict) }
        }
    }
}

extension PipelineBudgetBenchmarkTests {
    static func perceive(_ client: BenchClient, _ item: PipelineWorkItem) async throws -> String {
        switch item.input {
        case .document(let png):
            let result = try await json(client, "/v1/ocr", ["file": png.base64EncodedString(), "model": OCRRequest.appleVision])
            let pages = (result["pages"] as? [[String: Any]] ?? []).enumerated().map { PipelinePage(attachment: 0, page: $0.offset + 1, markdown: $0.element["markdown"] as? String ?? "") }
            return PipelineCore.state(PipelinePerceived(pages: pages), text: nil)
        case .audio(let wav):
            let form = AudioFixture.form([], file: AudioFixture.File(name: "clip.wav", type: "audio/wav", data: wav))
            let data = try await client.post("/v1/audio/transcriptions", form, type: "multipart/form-data; boundary=XyZ-boundary")
            let text = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])["text"] as? String ?? ""
            return PipelineCore.state(PipelinePerceived(transcript: text), text: nil)
        case .typed(let text):
            return text
        }
    }

    static func ask(_ client: BenchClient, _ question: SystemOneQuestion, state: String, tally: inout PipelineTally) async throws -> String? {
        let body: [String: Any] = [
            "messages": [["role": "user", "content": PipelineCore.prompt(question, state: state)]],
            "max_tokens": PipelineCore.questionMaxTokens, "temperature": 0, "reasoning_effort": "none",
        ]
        let response = try await json(client, "/v1/chat/completions", body)
        let usage = response["usage"] as? [String: Any] ?? [:]
        tally.llmCalls += 1
        tally.llmTokens += (usage["prompt_tokens"] as? Int ?? 0) + (usage["completion_tokens"] as? Int ?? 0)
        let text = ((response["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])?["content"] as? String ?? ""
        return PipelineCore.label(in: text, keys: SystemOneCore.options(question).keys)
    }

    static func chained(_ client: BenchClient, _ item: PipelineWorkItem, tally: inout PipelineTally) async throws -> String {
        let start = ContinuousClock.now
        let state = try await perceive(client, item)
        let decided = ContinuousClock.now
        let body: [String: Any] = ["state": state, "questions": try questionsJSON(item), "abstain_below": threshold]
        let data = try await client.post("/v1/systemone", try JSONSerialization.data(withJSONObject: body))
        let decision = try JSONDecoder().decode(SystemOneResponse.self, from: data)
        let answering = ContinuousClock.now
        var labels: [String: String?] = [:]
        for (name, question) in item.questions.sorted(by: { $0.key < $1.key }) {
            if let answer = decision.answers[name], answer.abstain != true {
                labels[name] = PipelineCore.label(of: answer, question)
            } else {
                tally.escalated += 1
                labels[name] = try await ask(client, question, state: state, tally: &tally)
            }
        }
        tally.perceiveMs += PipelineCore.milliseconds(decided - start)
        tally.decideMs += PipelineCore.milliseconds(answering - decided)
        tally.answerMs += milliseconds(since: answering)
        tally.wallMs += milliseconds(since: start)
        tally.score(item, labels: labels)
        return state
    }

    static func pipelineBody(_ item: PipelineWorkItem) throws -> [String: Any] {
        var body: [String: Any] = ["questions": try questionsJSON(item)]
        switch item.input {
        case .document(let png): body["attachments"] = [["kind": "image", "data": png.base64EncodedString()]]
        case .audio(let wav): body["attachments"] = [["kind": "audio", "data": wav.base64EncodedString(), "format": "wav"]]
        case .typed(let text): body["text"] = text
        }
        return body
    }

    static func piped(_ client: BenchClient, _ item: PipelineWorkItem, tally: inout PipelineTally) async throws {
        let start = ContinuousClock.now
        let data = try await client.post("/v1/pipeline", try JSONSerialization.data(withJSONObject: try pipelineBody(item)))
        tally.wallMs += milliseconds(since: start)
        let response = try JSONDecoder().decode(PipelineResponse.self, from: data)
        for stage in response.receipt.stages {
            switch stage.stage {
            case .perceive: tally.perceiveMs += stage.ms
            case .decide: tally.decideMs += stage.ms
            case .answer: (tally.answerMs, tally.llmCalls, tally.llmTokens) = (tally.answerMs + stage.ms, tally.llmCalls + 1, tally.llmTokens + stage.tokens)
            }
        }
        tally.escalated += response.answers.values.filter { $0.answeredBy == .answer }.count
        tally.score(item, labels: response.answers.mapValues(\.label))
    }

    static func llmOnly(_ client: BenchClient, _ item: PipelineWorkItem, state: String, tally: inout PipelineTally) async throws {
        let start = ContinuousClock.now
        var labels: [String: String?] = [:]
        for (name, question) in item.questions { labels[name] = try await ask(client, question, state: state, tally: &tally) }
        (tally.answerMs, tally.wallMs) = (tally.answerMs + milliseconds(since: start), tally.wallMs + milliseconds(since: start))
        tally.score(item, labels: labels)
    }
}
