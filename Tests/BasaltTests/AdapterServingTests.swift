import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

final class UnadaptableModel: Module, LanguageModel {
    @ModuleInfo var proj = Linear(64, 64)

    func prepare(_ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters) throws -> PrepareResult {
        .tokens(input.text)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [] }
}

actor AdapterEngine: InferenceEngine {
    let inner = MockInferenceEngine()
    private(set) var switches: [String?] = []
    private var active: String?

    func load(modelPath: String) async throws(BasaltError) { try await inner.load(modelPath: modelPath) }
    func generate(request: GenerationRequest) async throws(BasaltError) -> GenerationResponse { try await inner.generate(request: request) }
    func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk> {
        try await inner.generateStream(request: request)
    }
    var isLoaded: Bool { get async { await inner.isLoaded } }
    var adapterNames: [String] { ["a", "b"] }
    var activeAdapter: String? { active }

    func useAdapter(_ name: String?) async throws(BasaltError) {
        switches.append(name)
        active = name
    }
}

actor SwitchProbe {
    private(set) var runs = 0
    func ran() { runs += 1 }
}

@Suite("Per-request LoRA adapters on one resident model (#179)", .serialized)
struct AdapterServingTests {
    static let prompt = Conversation.firstTurn

    static func served() async throws -> (MLXInferenceEngine, ModelOrchestratorActor, [String: URL]) {
        let engine = try await TinyModel.engine(.qwen3)
        let directories = [
            "a": try LabGateTests.adapterDirectory(rank: 2, storedRank: 2),
            "b": try LabGateTests.adapterDirectory(rank: 2, storedRank: 2),
        ]
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orchestrator.loadAdapters(directories)
        return (engine, orchestrator, directories)
    }

    static func reference(_ directory: URL?) async throws -> String {
        let engine = try await TinyModel.engine(.qwen3)
        if let directory {
            let container = try #require(await engine.container)
            try await container.perform { context in try context.model.load(adapter: try LoRAContainer.from(directory: directory)) }
        }
        return try await engine.generate(request: greedy(prompt, key: nil)).generatedText
    }

    static func until(_ condition: () async -> Bool) async -> Bool {
        for _ in 0 ..< 1_000_000 {
            if await condition() { return true }
            await Task.yield()
        }
        return false
    }

    @Test("The adapter suffixes the prompt-cache namespace, so the in-memory tree, CacheBlend, MTP prefixes and the disk tier never mix adapters; no adapter leaves it as on main")
    func namespace() {
        var settings = GenerationPlan.Settings()
        let keyed = greedy(Self.prompt, key: "shared")
        #expect(GenerationPlan.cacheNamespace(keyed, settings, cacheEnabled: true, media: .text, sameEpoch: true) == "shared")
        settings.adapter = "a@123"
        #expect(GenerationPlan.cacheNamespace(keyed, settings, cacheEnabled: true, media: .text, sameEpoch: true) == "shared|adapter:a@123")
        #expect(GenerationPlan.cacheNamespace(keyed, settings, cacheEnabled: true, media: .images(digests: ["d"], end: 3), sameEpoch: true) == "shared#images:d|adapter:a@123")
        #expect(GenerationPlan.cacheNamespace(greedy(Self.prompt, key: nil), settings, cacheEnabled: true, media: .text, sameEpoch: true) == nil)
    }

    @Test("A → B → base → A through the orchestrator: each greedy output equals the adapter loaded once into a fresh model, base logits come back bit for bit, and the model is never reloaded")
    func switchesMatchStaticReferences() async throws {
        let (engine, orchestrator, directories) = try await Self.served()
        let container = try #require(await engine.container)
        let epoch = await engine.modelEpoch
        let logits = try await WeightDeltaTests.logits(container)
        let base = try await Self.reference(nil)
        let expected = ["a": try await Self.reference(directories["a"]), "b": try await Self.reference(directories["b"])]
        #expect(expected["a"] != base && expected["b"] != base && expected["a"] != expected["b"])
        for name in ["a", "b", nil, "a"] as [String?] {
            let response = try await orchestrator.generate(request: greedy(Self.prompt, key: nil).withAdapter(name))
            #expect(response.generatedText == name.flatMap { expected[$0] } ?? base, "\(name ?? "base")")
            #expect(await engine.activeAdapter == name)
            #expect(!response.modelModified)
        }
        try await engine.useAdapter(nil)
        #expect(try await WeightDeltaTests.logits(container) == logits)
        #expect(await LabGateTests.loraLayers(container) == 0)
        #expect(await engine.modelEpoch == epoch)
        #expect(await orchestrator.gate.checkInvariants(idle: true) == [])
    }

    @Test("A research call runs on the base model: a served adapter left active by the last request is switched off first")
    func researchRunsOnBase() async throws {
        let (engine, orchestrator, _) = try await Self.served()
        _ = try await orchestrator.generate(request: greedy(Self.prompt, key: nil).withAdapter("a"))
        #expect(await engine.activeAdapter == "a")
        let active = try await orchestrator.research { _ in await engine.activeAdapter }
        #expect(active == nil)
        #expect(await orchestrator.gate.checkInvariants(idle: true) == [])
    }

    @Test("The same prompt under A and then B never shares a lease; A again reuses A's prefix across the switch and answers the same")
    func cacheIsolatedPerAdapter() async throws {
        let (_, orchestrator, _) = try await Self.served()
        func run(_ name: String?) async throws -> GenerationResponse {
            try await orchestrator.generate(request: greedy(Self.prompt, key: "ns").withAdapter(name))
        }
        let first = try await run("a")
        #expect(first.cachedPromptTokens == 0)
        #expect(try await run("b").cachedPromptTokens == 0)
        #expect(try await run(nil).cachedPromptTokens == 0)
        let again = try await run("a")
        #expect(again.cachedPromptTokens > 0)
        #expect(again.generatedText == first.generatedText)
    }

    @Test("The gate runs a switch only once no generation runs, then hands back a running ticket")
    func gateAdmitsAfterSwitch() async throws {
        let gate = GenerationGate(maxConcurrent: 2)
        let running = try await gate.admit()
        let switched = Task { try await gate.admit(after: {}) }
        #expect(await Self.until { await gate.exclusiveWaiting == 1 })
        #expect(await gate.inFlight == 1)
        await gate.release(running)
        let ticket = try await switched.value
        #expect(await gate.inFlight == 1)
        await gate.release(ticket)
        #expect(await gate.checkInvariants(idle: true) == [])
    }

    @Test("A switch cancelled while it waits never runs and takes no ticket; one more switch than the queue holds is refused with 503")
    func switchCancelledWhileWaiting() async throws {
        let gate = GenerationGate(maxConcurrent: 2, maxQueue: 1)
        let probe = SwitchProbe()
        let running = try await gate.admit()
        let waiting = Task { try await gate.admit(after: { await probe.ran() }) }
        #expect(await Self.until { await gate.exclusiveWaiting == 1 })
        let refusal = await #expect(throws: BasaltError.self) { try await gate.admit(after: {}) }
        #expect(refusal?.localizedDescription == BasaltError.serverBusy(retryAfterSeconds: 1).localizedDescription)
        waiting.cancel()
        await gate.release(running)
        await #expect(throws: CancellationError.self) { try await waiting.value }
        #expect(await probe.runs == 0)
        #expect(await gate.checkInvariants(idle: true) == [])
    }

    @Test("A switch waits for in-flight generations of the other adapter; a second request for the active adapter runs at once without a switch")
    func switchWaitsForOtherAdapter() async throws {
        let engine = AdapterEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker(), gate: GenerationGate(maxConcurrent: 4))
        try await orchestrator.loadModel(path: "mock")
        await engine.inner.holdStreams()
        let first = try await orchestrator.generateStream(request: GenerationRequest(prompt: "1").withAdapter("a"))
        let second = try await orchestrator.generateStream(request: GenerationRequest(prompt: "2").withAdapter("a"))
        #expect(await engine.switches == ["a"])
        #expect(await orchestrator.gate.inFlight == 2)
        let other = Task { try await orchestrator.generate(request: GenerationRequest(prompt: "3").withAdapter("b")) }
        #expect(await Self.until { await orchestrator.gate.exclusiveWaiting == 1 })
        #expect(await engine.switches == ["a"])
        await engine.inner.releaseStreams()
        _ = await (GenerationGateTests.drain(first), GenerationGateTests.drain(second))
        _ = try await other.value
        #expect(await engine.switches == ["a", "b"])
        #expect(await orchestrator.gate.checkInvariants(idle: true) == [])
    }

    @Test("An unknown adapter is refused before it queues, and never reaches the engine or switches")
    func unknownAdapterRefused() async throws {
        let engine = AdapterEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "mock")
        let error = await #expect(throws: BasaltError.self) {
            try await orchestrator.generate(request: GenerationRequest(prompt: "p").withAdapter("zzz"))
        }
        #expect(error?.localizedDescription.contains("unknown adapter zzz") == true)
        #expect(await engine.switches.isEmpty)
        #expect(await engine.inner.getRequests().isEmpty)
    }

    @Test("X-Basalt-Adapter naming an adapter the server does not serve gets 400 on chat completions and never reaches the engine")
    func restRefusesUnknownAdapter() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, maxTokens: 65536, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "mock")
        let app = Application(router: RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "org/mock-model")).makeRouter())
        let body = #"{"model": "local", "messages": [{"role": "user", "content": "hi"}]}"#
        try await app.test(.router) { client in
            let headers: HTTPFields = [.contentType: "application/json", GenerationRequest.adapterHeader: "nope"]
            try await client.execute(uri: "/v1/chat/completions", method: .post, headers: headers, body: ByteBuffer(string: body)) { res in
                #expect(res.status == .badRequest)
                #expect(String(buffer: res.body).contains("unknown adapter nope"))
            }
        }
        #expect(await engine.getRequests().isEmpty)
    }

    @Test("At startup an adapter on a base without LoRA layers is refused naming the architecture; a rank mismatch is refused; nothing is registered or left applied")
    func startupRefusals() async throws {
        let directory = try LabGateTests.adapterDirectory(rank: 2, storedRank: 2)
        let plain = MLXInferenceEngine()
        await plain.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "unadaptable"), model: UnadaptableModel(), processor: ByteInputProcessor(), tokenizer: ByteTokenizer())))
        let refusal = await #expect(throws: BasaltError.self) { try await plain.loadAdapters(["a": directory]) }
        #expect(refusal?.localizedDescription.contains("UnadaptableModel") == true)
        #expect(await plain.adapterNames.isEmpty)

        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        let mismatched = try LabGateTests.adapterDirectory(rank: 2, storedRank: 3)
        await #expect(throws: BasaltError.self) { try await engine.loadAdapters(["a": directory, "m": mismatched]) }
        await #expect(throws: BasaltError.self) { try await engine.loadAdapters(["missing": URL(fileURLWithPath: "/nonexistent")]) }
        #expect(await engine.adapterNames.isEmpty)
        #expect(await LabGateTests.loraLayers(container) == 0)
        await #expect(throws: BasaltError.self) { try await engine.generate(request: greedy(Self.prompt, key: nil).withAdapter("a")) }
    }
}
