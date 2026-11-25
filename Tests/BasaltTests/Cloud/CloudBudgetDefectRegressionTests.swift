import Foundation
import HTTPTypes
import Hummingbird
import NIOCore
import Testing
@testable import BasaltCloud
@testable import BasaltCore

@Suite("Cloud budget, reservation, assess side-effect, and cascade stream metadata regressions (A-S4, A-S6, A-S10, A-S11)")
struct CloudBudgetDefectRegressionTests {
    actor RequestGate {
        private var entered = 0
        private var enteredWaiter: CheckedContinuation<Void, Never>?
        private var releaseWaiters: [CheckedContinuation<Void, Never>] = []
        private var released = false

        var hits: Int { entered }

        func arriveAndHold() async {
            entered += 1
            if let waiter = enteredWaiter {
                enteredWaiter = nil
                waiter.resume()
            }
            guard !released else { return }
            await withCheckedContinuation { releaseWaiters.append($0) }
        }

        func waitForFirstArrival() async {
            guard entered == 0 else { return }
            await withCheckedContinuation { enteredWaiter = $0 }
        }

        func releaseAll() {
            released = true
            let waiters = releaseWaiters
            releaseWaiters.removeAll()
            waiters.forEach { $0.resume() }
        }
    }

    private actor SegmentedLocalEngine: InferenceEngine {
        var isLoaded: Bool { true }
        private var turn = 0

        func load(modelPath: String) async throws(BasaltError) {}
        func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk> {
            AsyncStream { $0.finish() }
        }

        func generate(request: GenerationRequest) async throws(BasaltError) -> GenerationResponse {
            defer { turn += 1 }
            let confident = turn == 0
            let text = confident ? "part1 " : "unsure"
            let resp = GenerationResponse(
                generatedText: text, promptTokens: 12, completionTokens: 6,
                tokensPerSecond: 30, generationTime: 0.2, timeToFirstToken: 0.08,
                memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0),
                finishReason: .length
            )
            return resp.withConfidence(ResponseConfidence(min: confident ? 0.92 : 0.1, mean: confident ? 0.95 : 0.2))
        }
    }

    static func makeTracker(daily: Double = 1.0) throws -> (CostTracker, URL) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("cloud-defects-\(UUID().uuidString)")
        let file = dir.appendingPathComponent("cost.json")
        return (try CostTracker(storageURL: file, budget: CostTracker.Budget(dailyLimit: daily, monthlyLimit: daily * 10)), dir)
    }

    static func makeClient(host: String) throws -> OpenRouterClient {
        let config = OpenRouterClient.Config(
            apiKey: "k", baseURL: try #require(URL(string: "http://\(host)/api/v1")), timeout: 5, maxRetries: 0
        )
        return try OpenRouterClient(config: config, configuration: StubURLProtocol.configuration)
    }

    static func withGatedCloudClient(
        _ body: (OpenRouterClient, RequestGate) async throws -> Void
    ) async throws {
        let gate = RequestGate()
        let okPayload = #"{"id":"c1","model":"openai/gpt-4-turbo","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":10,"total_tokens":20,"cost":0.01}}"#
        let router = Router()
        router.post("/api/v1/chat/completions") { _, _ in
            await gate.arriveAndHold()
            return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(string: okPayload)))
        }
        let portReady = AsyncStream.makeStream(of: Int.self)
        let app = Application(
            router: router,
            configuration: .init(address: .hostname("127.0.0.1", port: 0)),
            onServerRunning: { channel in portReady.continuation.yield(channel.localAddress?.port ?? 0) }
        )
        let serverTask = Task { try await app.runService() }
        defer { serverTask.cancel() }
        var portIter = portReady.stream.makeAsyncIterator()
        let port = try #require(await portIter.next())
        let client = try OpenRouterClient(
            config: .init(apiKey: "k", baseURL: try #require(URL(string: "http://127.0.0.1:\(port)/api/v1")), timeout: 10, maxRetries: 0)
        )
        try await body(client, gate)
    }
}

extension CloudBudgetDefectRegressionTests {
    @Test("A-S4: ModelRouter includes usage.include on non-streaming requests and records provider-reported usage.cost")
    func nonStreamingChatIncludesUsageAndRecordsReportedCost() async throws {
        let req = ModelRouter.chatRequest(GenerationRequest(prompt: "hi", maxTokens: 16), model: "unlisted/model", stream: false)
        #expect(req.usage?.include == true)

        let payload: [String: Any] = [
            "id": "cmpl-cost",
            "model": "unlisted/model",
            "choices": [["message": ["role": "assistant", "content": "hello"], "finish_reason": "stop"]],
            "usage": ["prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.25],
        ]
        let data = try JSONSerialization.data(withJSONObject: payload)
        let host = StubURLProtocol.host { _, _ in (200, ["Content-Type": "application/json"], data) }
        let client = try Self.makeClient(host: host)
        let (tracker, dir) = try Self.makeTracker(daily: 5.0)
        defer { try? FileManager.default.removeItem(at: dir) }

        let router = ModelRouter(
            localOrchestrator: ModelOrchestratorActor(engine: MockInferenceEngine()),
            cloudClient: client,
            options: ModelRouter.Options(strategy: .cloudOnly, costTracker: tracker)
        )
        _ = try await router.route(request: GenerationRequest(prompt: "hi", maxTokens: 16), preferredModel: "unlisted/model")
        #expect(abs(await tracker.getStats().dailyUsage - 0.25) < 1e-6)
    }

    @Test("A-S6: CostTracker and CloudGate reservations prevent concurrent budget overspend and release cleanly")
    func budgetReservationsPreventConcurrentOverspend() async throws {
        let (tracker, dir) = try Self.makeTracker(daily: 0.10)
        defer { try? FileManager.default.removeItem(at: dir) }

        let t1 = try #require(await tracker.reserve(estimatedCost: 0.08, scopes: [.session("s1")]))
        #expect(await tracker.reserve(estimatedCost: 0.05, scopes: [.session("s1")]) == nil)
        #expect(abs(await tracker.remaining(scopes: [.session("s1")]) - 0.02) < 1e-9)
        await tracker.release(reservation: t1)
        let t2 = try #require(await tracker.reserve(estimatedCost: 0.08, scopes: [.session("s1")]))
        await tracker.release(reservation: t2)

        let gate = CloudGate(limits: ["m": 0.10], storage: nil)
        let c1 = await gate.reserve("m", estimatedCost: 0.08)
        #expect(c1.token != nil && c1.refusal == nil)
        let c2 = await gate.reserve("m", estimatedCost: 0.05)
        #expect(c2.token == nil && c2.refusal != nil)
        if let tok = c1.token { await gate.release(tok) }
        let c3 = await gate.reserve("m", estimatedCost: 0.08)
        #expect(c3.token != nil && c3.refusal == nil)
    }

    @Test("A-S6: concurrent CloudAnswer (priced & unpriced) and single-tier ModelRouter.route block overspend", .timeLimit(.minutes(1)))
    func concurrentCloudAnswerAndSingleTierRouteBlockOverspend() async throws {
        let req = GenerationRequest(prompt: "hi", maxTokens: 2_000)
        for model in ["openai/gpt-4-turbo", "custom/unpriced-model"] {
            try await Self.withGatedCloudClient { liveClient, gate in
                let cloudGate = CloudGate(limits: [model: 0.08], storage: nil)
                let answer = CloudAnswer(client: liveClient, model: model, gate: cloudGate)
                let first = Task { try await answer.answer(req) }
                await gate.waitForFirstArrival()
                await #expect(throws: BasaltError.self) { _ = try await answer.answer(req) }
                #expect(await gate.hits == 1)
                await gate.releaseAll()
                _ = try await first.value
            }
        }
        try await Self.withGatedCloudClient { liveClient, gate in
            let (tracker, dir) = try Self.makeTracker(daily: 0.08)
            defer { try? FileManager.default.removeItem(at: dir) }
            let modelRouter = ModelRouter(
                localOrchestrator: ModelOrchestratorActor(engine: MockInferenceEngine()),
                cloudClient: liveClient,
                options: ModelRouter.Options(strategy: .cloudOnly, costTracker: tracker)
            )
            let first = Task { try await modelRouter.route(request: req, preferredModel: "openai/gpt-4-turbo") }
            await gate.waitForFirstArrival()
            await #expect(throws: (any Error).self) { _ = try await modelRouter.route(request: req, preferredModel: "openai/gpt-4-turbo") }
            #expect(await gate.hits == 1)
            await gate.releaseAll()
            _ = try await first.value
        }
    }
}

extension CloudBudgetDefectRegressionTests {

    @Test("A-S10: assess is read-only & counts message prompt tokens; generate saves .strong and recordLocalOutcome saves .weak")
    func assessIsReadOnlyAndSessionTierTracksBothSides() async throws {
        let okPayload = #"{"id":"c1","model":"openai/gpt-4-turbo","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":10,"total_tokens":20,"cost":0.01}}"#
        let host = StubURLProtocol.host { _, _ in (200, ["Content-Type": "application/json"], Data(okPayload.utf8)) }
        let client = try Self.makeClient(host: host)
        let costRouting = ModelRouter.CostRouting(
            policy: StickyRoutingPolicy(strongModel: "gpt-4", weakModel: "local", threshold: 0.5, cacheStickinessPenalty: 1.0),
            scorer: try BradleyTerryRouterTests.scorer()
        )
        let router = ModelRouter(
            localOrchestrator: ModelOrchestratorActor(engine: MockInferenceEngine()),
            cloudClient: client,
            options: ModelRouter.Options(strategy: .hybrid, costRouting: costRouting, fallbackCloudModel: "gpt-4")
        )
        let bigMessage = Array(repeating: BradleyTerryRouterTests.hardWord, count: 200).joined(separator: " ")
        let req = GenerationRequest(
            prompt: "",
            maxTokens: 64,
            messages: [ChatMessageInput(role: .user, text: bigMessage)]
        ).withRouting(sessionId: "sess-ro", budgetKey: nil)
        let completionOnlyCost = await router.estimateCost(model: "gpt-4", promptTokens: 0, completionTokens: 64)
        let assessment = try #require(await router.assess(req, localModel: "local"))
        #expect(assessment.estimatedCost > completionOnlyCost)
        #expect(await router.sessionTier(for: "sess-ro") == nil)
        let dummyLocal = GenerationResponse(
            generatedText: "ok", promptTokens: 10, completionTokens: 5, tokensPerSecond: 20, generationTime: 0.1,
            timeToFirstToken: 0.05, memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0), finishReason: .stop
        )
        await router.recordLocalOutcome(req, response: dummyLocal, localModel: "local")
        #expect(await router.sessionTier(for: "sess-ro") == .weak)
        _ = try await router.generate(req)
        #expect(await router.sessionTier(for: "sess-ro") == .strong)
        await router.recordLocalOutcome(req, response: dummyLocal, localModel: "local")
        #expect(await router.sessionTier(for: "sess-ro") == .weak)
    }

    @Test("A-S11: CascadeRouting.runCascadeStream merges locally accepted tokens and TTFT into cloud stream metadata")
    func cascadeStreamMergesLocalTokensAndTTFT() async throws {
        let sse = """
        data: {"id":"1","model":"gpt-4","choices":[{"delta":{"content":"part2"},"finish_reason":null}]}

        data: {"id":"1","model":"gpt-4","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":4,"total_tokens":24}}

        data: [DONE]

        """
        let host = StubURLProtocol.host { _, _ in (200, ["Content-Type": "text/event-stream"], Data(sse.utf8)) }
        let client = try Self.makeClient(host: host)
        let cascade = CascadeExecutionMode(segmentTokens: 6, alpha: 0.2, referenceBudget: 10.0, cloudModel: "gpt-4")
        let router = ModelRouter(
            localOrchestrator: ModelOrchestratorActor(engine: SegmentedLocalEngine()),
            cloudClient: client,
            options: ModelRouter.Options(strategy: .localOnly).withCascade(cascade)
        )
        let stream = try await router.routeStream(request: GenerationRequest(prompt: "hi", maxTokens: 20), preferredModel: "local")
        var finalMeta: GenerationResponse?
        for try await chunk in stream {
            if case .metadata(let m) = chunk { finalMeta = m }
        }
        let meta = try #require(finalMeta)
        #expect(meta.completionTokens == 10)
        #expect(meta.timeToFirstToken >= 0.08 - 1e-6)
    }
}
