import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltServer

@Suite("/v1/pipeline perceive stage: the speech model, SpeechAnalyzer and Apple Vision fallbacks, and GPU stages gated between generations (#402 phase 2)", .timeLimit(.minutes(1)))
struct PipelinePerceiveAPITests {
    static func speechModel(_ text: String) async throws -> ModelOrchestratorActor {
        let engine = MockInferenceEngine()
        await engine.setMockTranscription(Transcription(text: text))
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "org/nemotron-transcribe")
        await orchestrator.setModelId("org/nemotron-transcribe")
        return orchestrator
    }

    static var audio: [String: Any] { ["kind": "audio", "data": AudioFixture.sine(seconds: 0.2).base64EncodedString(), "format": "wav"] }

    @Test("Audio goes to the speech model when one is loaded, else to SpeechAnalyzer; the transcript reaches System One and the receipt names the perceiver; with neither it is a 404 naming --pipeline-audio-model")
    func audioRouting() async throws {
        let (_, chat) = try await PipelineAPITests.chat()
        let fallback = ScriptedTranscriber()
        let decisions = GatingDecisionEngine()
        let nemotron = try await PipelineAPITests.post(
            chat, decisions: decisions, stages: PipelineStages(audio: try await Self.speechModel("the car is red"), speech: fallback),
            ["attachments": [Self.audio], "questions": PipelineAPITests.questions])
        #expect(nemotron.status == .ok)
        #expect((nemotron.json["perceived"] as? [String: Any])?["transcript"] as? String == "the car is red")
        #expect(PipelineAPITests.stages(nemotron.json).first?["model"] as? String == "org/nemotron-transcribe")
        #expect(decisions.calls.requests.withLock { $0.first?.state } == "[transcript]\nthe car is red")
        #expect(fallback.requests.withLock { $0.isEmpty })
        let analyzer = try await PipelineAPITests.post(chat, decisions: decisions, stages: PipelineStages(speech: fallback), ["attachments": [Self.audio], "text": "Note.", "questions": PipelineAPITests.questions])
        #expect(analyzer.status == .ok)
        #expect(PipelineAPITests.stages(analyzer.json).map { $0["model"] as? String } == [PipelineCore.speechAnalyzerModel, "org/stub-jev"])
        #expect(decisions.calls.requests.withLock { $0.last?.state } == "[transcript]\nhello world\n\nNote.")
        #expect(fallback.requests.withLock { $0.count } == 1)
        let none = try await PipelineAPITests.post(chat, decisions: decisions, ["attachments": [Self.audio], "task": "t"])
        #expect(none.status == .notFound)
        #expect(((none.json["error"] as? [String: Any])?["message"] as? String)?.contains("--pipeline-audio-model") == true)
    }

    @Test("Without an OCR model a document is read by Apple Vision on the chat orchestrator, and the receipt says so")
    func documentFallsBackToVision() async throws {
        let (engine, chat) = try await PipelineAPITests.chat()
        await chat.setDocumentTranscriber(ScriptedDocumentTranscriber(text: "Invoice total: 42"))
        let response = try await PipelineAPITests.post(
            chat, decisions: GatingDecisionEngine(), ["attachments": [["kind": "image", "data": try OCRFixture.png(width: 4, height: 4).base64EncodedString()]], "questions": PipelineAPITests.questions])
        #expect(response.status == .ok)
        #expect(((response.json["perceived"] as? [String: Any])?["pages"] as? [[String: Any]])?.first?["markdown"] as? String == "Invoice total: 42")
        #expect(PipelineAPITests.stages(response.json).first?["model"] as? String == OCRRequest.appleVision)
        #expect(await engine.getRequests().isEmpty)
    }

    struct TooLongDecisionEngine: DecisionEngine {
        let modelID = "org/short-context"

        func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
            throw .contextLengthExceeded(maxTokens: 512, requested: 513)
        }
    }

    @Test("A state too long for the decision model sends every question to System 2 instead of failing the request")
    func tooLongStateEscalates() async throws {
        let (_, chat) = try await PipelineAPITests.chat()
        let response = try await PipelineAPITests.post(chat, decisions: TooLongDecisionEngine(), ["text": "The car is red.", "questions": PipelineAPITests.questions])
        #expect(response.status == .ok)
        let answers = try #require(response.json["answers"] as? [String: [String: Any]])
        #expect(answers.count == PipelineAPITests.questions.count && answers.values.allSatisfy { $0["answered_by"] as? String == "answer" })
        let decide = PipelineAPITests.stages(response.json).first { $0["stage"] as? String == "decide" }
        #expect(decide?["model"] as? String == "org/short-context" && decide?["escalated"] as? Bool == true)
    }

    @Test("A PDF read by the OCR model holds the gate one page at a time, and chat calls that carry perceived pages are marked local-only")
    func ocrPerPageAndLocalAnswers() async throws {
        let ocrEngine = MockInferenceEngine()
        await ocrEngine.setResponder { _ in "Invoice total: 42" }
        let ocr = try await OCRAPITests.orchestrator("GLM-OCR", modelType: "glm_ocr", engine: ocrEngine)
        let (chatEngine, chat) = try await PipelineAPITests.chat()
        let pdf: [String: Any] = ["kind": "pdf", "data": OCRFixture.pdf(pages: ["One", "Two"]).base64EncodedString()]
        let response = try await PipelineAPITests.post(chat, decisions: nil, stages: PipelineStages(ocr: ocr), ["attachments": [pdf], "task": "Sum it."])
        #expect(response.status == .ok)
        #expect(await ocrEngine.getRequests().count == 2)
        #expect(((response.json["perceived"] as? [String: Any])?["pages"] as? [Any])?.count == 2)
        #expect(await chatEngine.getRequests().allSatisfy { $0.origin.source == PipelineCore.perceivedSource })
    }

    @Test("Audio length is summed over every attachment of a request")
    func audioSecondsAddUp() throws {
        let clip = try AudioClip(samples: Array(repeating: 0.1, count: Int(AudioClip.sampleRate)))
        #expect(PipelineRequest.audioSeconds([.audio(clip), .audio(clip)]) == 2)
    }

    struct BusyDecisionEngine: DecisionEngine {
        let modelID = "org/busy"

        func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
            throw .serverBusy(retryAfterSeconds: 1)
        }
    }

    @Test("A busy System One is a 503 for the pipeline, not a silent escalation of every question")
    func busyDecisionIsServiceUnavailable() async throws {
        let (_, chat) = try await PipelineAPITests.chat()
        let response = try await PipelineAPITests.post(chat, decisions: BusyDecisionEngine(), ["text": "The car is red.", "questions": PipelineAPITests.questions])
        #expect(response.status == .serviceUnavailable)
    }

    @Test("GPU stages are gated between generations: while a chat generation holds a slot, SpeechAnalyzer transcribes, but OCR in its own engine and the decision engine wait, and run once it ends")
    func gpuStagesWaitForGeneration() async throws {
        let ocrEngine = MockInferenceEngine()
        await ocrEngine.setResponder { _ in "Invoice total: 42" }
        let ocr = try await OCRAPITests.orchestrator("GLM-OCR", modelType: "glm_ocr", engine: ocrEngine)
        let (_, chat) = try await PipelineAPITests.chat()
        let (transcriber, decisions) = (ScriptedTranscriber(), GatingDecisionEngine())
        let served = GatedDecisionEngine.serving(decisions, gate: chat.gate)
        let generation = try await chat.gate.admit()
        let image: [String: Any] = ["kind": "image", "data": try OCRFixture.png(width: 4, height: 4).base64EncodedString()]
        async let heard = PipelineAPITests.post(chat, decisions: served, stages: PipelineStages(speech: transcriber), ["attachments": [Self.audio], "questions": PipelineAPITests.questions])
        async let read = PipelineAPITests.post(chat, decisions: served, stages: PipelineStages(ocr: ocr), ["attachments": [image], "questions": PipelineAPITests.questions])
        while await chat.gate.exclusiveWaiting < 2 { await Task.yield() }
        #expect(transcriber.requests.withLock { $0.count } == 1)
        #expect(await ocrEngine.getRequests().isEmpty)
        #expect(decisions.calls.requests.withLock { $0.isEmpty })
        await chat.gate.release(generation)
        #expect(try await heard.status == .ok)
        #expect(try await read.status == .ok)
        #expect(await ocrEngine.getRequests().count == 1)
        #expect(decisions.calls.requests.withLock { $0.count } == 2)
    }
}
