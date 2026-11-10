import Testing
import Foundation
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer
@testable import BasaltCLI

actor ReloadingEngine: InferenceEngine {
    let inner: MLXInferenceEngine
    let container: @Sendable () throws -> ModelContainer
    private(set) var loads = 0
    private(set) var unloads = 0
    private(set) var holding = false
    private var holdsNextLoad = false
    private var held: CheckedContinuation<Void, Never>?

    init(promptCacheBytes: Int? = nil, container: @escaping @Sendable () throws -> ModelContainer) {
        (inner, self.container) = (MLXInferenceEngine(EngineConfiguration(promptCacheBytes: promptCacheBytes)), container)
    }

    func holdNextLoad() { holdsNextLoad = true }

    func release() {
        holding = false
        held?.resume()
        held = nil
    }

    func load(modelPath: String) async throws(BasaltError) {
        loads += 1
        if holdsNextLoad {
            (holdsNextLoad, holding) = (false, true)
            await withCheckedContinuation { held = $0 }
        }
        do {
            await inner.attach(try container())
        } catch {
            throw .modelNotFound(identifier: modelPath)
        }
    }

    func unload() async {
        unloads += 1
        await inner.unload()
    }

    func generate(request: GenerationRequest) async throws(BasaltError) -> GenerationResponse { try await inner.generate(request: request) }
    func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk> { try await inner.generateStream(request: request) }
    func transcribe(_ request: TranscriptionRequest) async throws(BasaltError) -> Transcription { try await inner.transcribe(request) }
    var isLoaded: Bool { get async { await inner.isLoaded } }
    var transcribesAudio: Bool { get async { await inner.transcribesAudio } }
}

@Suite("Plan memory modes: resident sums the models, sequential keeps one stage at a time, auto refuses only a stage too big alone (#402)")
struct PipelineMemoryPlanTests {
    static let gb: Int64 = 1_000_000_000
    static let chat = Workload(name: "chat", role: "chat", unit: .gpu, kind: .decode, residentBytes: 11 * gb)
    static let speech = Workload(name: "speech", role: Workload.perceiveRole, unit: .gpu, kind: .encoder, residentBytes: 15 * gb)
    static let ocr = Workload(name: "ocr", role: Workload.perceiveRole, unit: .gpu, kind: .encoder, residentBytes: 1 * gb)
    static let decide = Workload(name: "decide", role: "decide", unit: .gpu, kind: .encoder, residentBytes: 1 * gb)

    @Test("Resident sums, sequential takes the largest stage, auto picks resident while the sum fits", .timeLimit(.minutes(1)))
    func modes() {
        let auto = ComputePlan.admit([Self.chat, Self.ocr], budgetBytes: 18 * Self.gb, memory: .auto)
        #expect(auto.memory == .resident && auto.residentBytes == 12 * Self.gb && auto.swapBytesPerRequest == 0 && auto.refusal == nil)
        let perceivers = ComputePlan.admit([Self.chat, Self.ocr], budgetBytes: 18 * Self.gb, memory: .sequential)
        #expect(perceivers.memory == .sequential && perceivers.keep(ofKind: .decode) == .resident && perceivers.keep(of: Workload.perceiveRole) == .sequential)
        #expect(perceivers.residentBytes == 12 * Self.gb && perceivers.swapBytesPerRequest == 1 * Self.gb)
        let all = ComputePlan.admit([Self.chat, Self.decide, Self.speech], budgetBytes: 19 * Self.gb, memory: .auto)
        #expect(all.memory == .sequential && all.refusal == nil)
        #expect(all.keep(ofKind: .decode) == .sequential && all.keep(of: "decide") == .resident && all.keep(of: Workload.perceiveRole) == .sequential)
        #expect(all.residentBytes == 16 * Self.gb && all.swapBytesPerRequest == 26 * Self.gb)
        #expect(all.expectedSwapSeconds == Double(26 * Self.gb) / ComputePlan.loadBytesPerSecond)
        #expect(ComputePlan.admit([Self.chat, Self.decide], budgetBytes: 18 * Self.gb, memory: .sequential).memory == .resident)
    }

    @Test("Refusal: resident refuses the sum, sequential and auto only a stage that does not fit alone with the KV reserve", .timeLimit(.minutes(1)))
    func refusals() throws {
        #expect(ComputePlan.admit([Self.chat, Self.speech], budgetBytes: 19 * Self.gb, memory: .resident).refusal != nil)
        #expect(ComputePlan.admit([Self.chat, Self.speech], budgetBytes: 17 * Self.gb, memory: .auto).refusal == nil)
        let tooBig = ComputePlan.admit([Self.chat, Self.speech], budgetBytes: 16 * Self.gb, memory: .auto)
        #expect(try #require(tooBig.refusal).contains("even one stage at a time the models need 15.0 GB resident plus 2.0 GB"))
        let pinned = ComputePlan.admit([Self.chat, Workload(name: "o", role: Workload.perceiveRole, unit: .gpu, kind: .encoder, residentBytes: Self.gb, keep: .sequential)], budgetBytes: 18 * Self.gb)
        #expect(pinned.memory == .sequential && pinned.keep(ofKind: .decode) == .resident)
    }
}

@Suite("Sequential pipeline on tiny models below BasaltServer: the speech perceiver loads for its stage and unloads after it, the chat model swaps around it, chat requests wait (#402, #228)", .serialized)
struct PipelineSequentialTests {
    static let instructions = "The speaker is calm."

    static func clip() -> Data { AudioFixture.sine(seconds: 0.5, rate: 16_000, channels: 1) }

    static func references() async throws -> (transcript: String, answer: String) {
        let speech = ModelOrchestratorActor(engine: try await NemotronTranscribeUnitTests.tinyEngine(seed: 3), telemetry: TelemetryTracker())
        let transcript = try await speech.transcribe(TranscriptionRequest(audio: AudioClip(encoded: clip(), format: "wav"))).text
        let state = PipelineCore.state(PipelinePerceived(transcript: transcript), text: nil)
        let call = PipelineCore.System2Call.question(id: "calm", prompt: PipelineCore.prompt(SystemOneQuestion(type: .noul, instructions: instructions), state: state))
        let answer = try await TinyModel.engine().generate(request: PipelineCore.request(for: call, origin: .unspecified)).generatedText
        return (transcript, answer)
    }

    static func served(unloadsChat: Bool) async throws -> (chat: ModelOrchestratorActor, chatEngine: ReloadingEngine, speechEngine: ReloadingEngine, stages: PipelineStages) {
        let chatEngine = ReloadingEngine { try TinyModel.container() }
        let chat = ModelOrchestratorActor(engine: chatEngine, telemetry: TelemetryTracker())
        try await chat.loadModel(path: "/tiny/chat")
        let speechEngine = ReloadingEngine(promptCacheBytes: 0) { try NemotronTranscribeUnitTests.tinyContainer(seed: 3) }
        let speech = ModelOrchestratorActor(engine: speechEngine, telemetry: TelemetryTracker())
        await speech.setModelId("tiny-nemotron-transcribe")
        let swap = PipelineSwap(ocrPath: nil, audioPath: "/tiny/speech", unloadsChat: unloadsChat)
        var stages = PipelineStages(audio: speech, abstainBelow: 1)
        stages.swap = swap
        return (chat, chatEngine, speechEngine, stages)
    }

    static var body: [String: Any] {
        ["attachments": [["kind": "audio", "data": clip().base64EncodedString(), "format": "wav"]], "questions": ["calm": ["type": "noul", "instructions": instructions]]]
    }

    @Test("Answers match the resident references; the speech engine is loaded once and unloaded after its stage; the chat model is reloaded only when it was swapped", .timeLimit(.minutes(1)), arguments: [true, false])
    func swapsPerceiver(unloadsChat: Bool) async throws {
        let reference = try await Self.references()
        let served = try await Self.served(unloadsChat: unloadsChat)
        let epoch = await served.chat.loaded.epoch
        let response = try await PipelineAPITests.post(served.chat, decisions: GatingDecisionEngine(flat: ["calm"]), stages: served.stages, Self.body)
        #expect(response.status == .ok)
        #expect((response.json["perceived"] as? [String: Any])?["transcript"] as? String == reference.transcript)
        #expect(PipelineAPITests.answers(response.json)["calm"]?["text"] as? String == reference.answer)
        #expect(((response.json["receipt"] as? [String: Any])?["swap_ms"] as? Int ?? -1) >= 0)
        let (speechLoads, speechUnloads, speechLoaded) = (await served.speechEngine.loads, await served.speechEngine.unloads, await served.speechEngine.isLoaded)
        #expect(speechLoads == 1 && speechUnloads == 1 && !speechLoaded)
        let (chatLoads, chatUnloads, chatLoaded) = (await served.chatEngine.loads, await served.chatEngine.unloads, await served.chatEngine.isLoaded)
        #expect(chatLoads == (unloadsChat ? 2 : 1) && chatUnloads == (unloadsChat ? 1 : 0) && chatLoaded)
        #expect(await served.chat.loaded.epoch == epoch + (unloadsChat ? 2 : 0))
    }

    @Test("A chat request that arrives while the perceiver is swapped in waits for the swap and then succeeds on the reloaded chat model", .timeLimit(.minutes(1)))
    func chatWaitsDuringSwap() async throws {
        let reference = try await TinyModel.engine().generate(request: GenerationRequest(prompt: "hello", maxTokens: 4, temperature: 0)).generatedText
        let served = try await Self.served(unloadsChat: true)
        await served.speechEngine.holdNextLoad()
        let stages = served.stages
        let chat = served.chat
        let pipeline = Task { try await PipelineAPITests.post(chat, decisions: GatingDecisionEngine(flat: ["calm"]), stages: stages, Self.body).status }
        while await !served.speechEngine.holding {
            try Task.checkCancellation()
            await Task.yield()
        }
        #expect(await !served.chatEngine.isLoaded)
        let reply = Task { try await chat.generate(request: GenerationRequest(prompt: "hello", maxTokens: 4, temperature: 0)) }
        while await chat.gate.queued == 0 {
            try Task.checkCancellation()
            await Task.yield()
        }
        await served.speechEngine.release()
        #expect(try await reply.value.generatedText == reference)
        #expect(try await pipeline.value == .ok)
        let (chatLoaded, speechLoaded) = (await served.chatEngine.isLoaded, await served.speechEngine.isLoaded)
        #expect(chatLoaded && !speechLoaded)
    }
}

@Suite("Sequential swap edge cases on stub engines: chat-encoder audio outside the swap, failed loads, stub mode, prompt caches, startup refusals (#402 review)")
struct PipelineSwapEdgeTests {
    static func chat(loaded: Bool = true) async throws -> (MockInferenceEngine, ModelOrchestratorActor) {
        let engine = MockInferenceEngine()
        await engine.setResponder { _ in "left" }
        await engine.setMockTranscription(Transcription(text: "pay it today"))
        let chat = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        if loaded { try await chat.loadModel(path: "/models/chat") }
        return (engine, chat)
    }

    static func sequentialOCR(_ engine: MockInferenceEngine, unloadsChat: Bool) async throws -> PipelineStages {
        let ocr = try await OCRAPITests.orchestrator("GLM-OCR", modelType: "glm_ocr", engine: engine)
        let path = await ocr.loaded.path
        await ocr.unloadModel()
        var stages = PipelineStages(ocr: ocr, abstainBelow: 0.5)
        stages.swap = PipelineSwap(ocrPath: path, audioPath: nil, unloadsChat: unloadsChat)
        return stages
    }

    static var attachments: [[String: Any]] { [
        ["kind": "audio", "data": AudioFixture.sine(seconds: 0.2).base64EncodedString(), "format": "wav"],
        ["kind": "image", "data": (try? OCRFixture.png(width: 4, height: 4).base64EncodedString()) ?? ""],
    ] }

    @Test("Audio for the chat model's own encoder runs outside the swap: no deadlock with the chat model resident, no missing encoder with it unloaded; results merge by attachment", .timeLimit(.minutes(1)), arguments: [false, true])
    func chatAudioOutsideSwap(unloadsChat: Bool) async throws {
        let ocrEngine = MockInferenceEngine()
        await ocrEngine.setResponder { _ in "Invoice total: 42" }
        let stages = try await Self.sequentialOCR(ocrEngine, unloadsChat: unloadsChat)
        let (engine, chat) = try await Self.chat()
        let response = try await PipelineAPITests.post(chat, decisions: GatingDecisionEngine(), stages: stages, ["attachments": Self.attachments, "questions": PipelineAPITests.questions])
        #expect(response.status == .ok)
        let perceived = try #require(response.json["perceived"] as? [String: Any])
        #expect(perceived["transcript"] as? String == "pay it today")
        #expect((perceived["pages"] as? [[String: Any]])?.first?["attachment"] as? Int == 1)
        #expect(PipelineAPITests.stages(response.json).map { $0["model"] as? String } == ["chat", "GLM-OCR", "org/stub-jev"])
        let (ocrLoaded, chatLoaded, released) = (await ocrEngine.isLoaded, await engine.isLoaded, await engine.releasedCaches)
        #expect(!ocrLoaded && chatLoaded)
        #expect(released == (unloadsChat ? 0 : 1))
    }

    @Test("A perceiver whose load fails after its weights arrived is unloaded again, and the chat model is reloaded", .timeLimit(.minutes(1)))
    func failedPerceiverLoadIsRolledBack() async throws {
        let ocrEngine = MockInferenceEngine()
        let stages = try await Self.sequentialOCR(ocrEngine, unloadsChat: true)
        await ocrEngine.failLoads(1, afterLoading: true)
        let (engine, chat) = try await Self.chat()
        let response = try await PipelineAPITests.post(chat, decisions: nil, stages: stages, ["attachments": [Self.attachments[1]], "task": "t"])
        #expect(response.status != .ok)
        let (ocrLoaded, chatLoaded) = (await ocrEngine.isLoaded, await engine.isLoaded)
        #expect(!ocrLoaded && chatLoaded)
    }

    @Test("Stub mode: with no chat model loaded the swap neither unloads nor reloads it", .timeLimit(.minutes(1)))
    func stubModeSkipsSwap() async throws {
        let (engine, chat) = try await Self.chat(loaded: false)
        let (value, swap) = try await chat.exclusively(unloadingModel: true) { () async throws(BasaltError) in 7 }
        let loads = await engine.loadCount
        #expect(value == 7 && swap == .zero && loads == 0)
    }

    @Test("A failed reload is retried once; two failures leave the request failed and the chat model unloaded", .timeLimit(.minutes(1)))
    func reloadRetries() async throws {
        let (engine, chat) = try await Self.chat()
        await engine.failLoads(1)
        let (value, _) = try await chat.exclusively(unloadingModel: true) { () async throws(BasaltError) in 1 }
        let (loads, loaded) = (await engine.loadCount, await engine.isLoaded)
        #expect(value == 1 && loads == 3 && loaded)
        await engine.failLoads(2)
        await #expect(throws: BasaltError.self) { _ = try await chat.exclusively(unloadingModel: true) { () async throws(BasaltError) in 1 } }
        let stillLoaded = await engine.isLoaded
        #expect(!stillLoaded)
    }

    @Test("Startup refuses an audio model without the Nemotron Transcribe architecture, and --adapter when the plan unloads the chat model", .timeLimit(.minutes(1)))
    func startupRefusals() async throws {
        let wrong = try OCRAPITests.modelDirectory("Qwen3-8B", modelType: "qwen3")
        let right = try OCRAPITests.modelDirectory("Transcribe", modelType: "nemotron_transcribe")
        #expect(ServeCommand.audioRefusal("org/qwen3", path: wrong.path)?.contains("model_type qwen3") == true)
        #expect(ServeCommand.audioRefusal("org/transcribe", path: right.path) == nil)
        let gb: Int64 = 1_000_000_000
        let swapsChat = ComputePlan.admit(
            [Workload(name: "c", role: "chat", unit: .gpu, kind: .decode, residentBytes: 11 * gb), Workload(name: "s", role: Workload.perceiveRole, unit: .gpu, kind: .encoder, residentBytes: 15 * gb)],
            budgetBytes: 19 * gb, memory: .auto)
        #expect(ServeCommand.adapterRefusal(["a=/x"], plan: swapsChat) != nil)
        #expect(ServeCommand.adapterRefusal([], plan: swapsChat) == nil && ServeCommand.adapterRefusal(["a=/x"], plan: nil) == nil)
        await #expect(throws: (any Error).self) { _ = try await ServeCommand.parse(["--adapter", "a=/x"]).loadPipelineStages(ServeCommand.PerceivePaths(), plan: swapsChat) }
    }
}
