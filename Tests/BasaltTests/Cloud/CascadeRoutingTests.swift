import Foundation
import MLX
import Synchronization
import Testing
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCloud
@testable import BasaltServer

@Suite("Confidence probe and session-log confidence recording (#39)")
struct ConfidenceProbeTests {

    @Test("ConfidenceProbeRecorder.maxProbability equals softmax(row).max() and treats non-finite rows as missing")
    func maxProbabilityMatchesSoftmaxAndHandlesNonFinite() {
        let values: [Float] = [1.5, -0.5, 2.5, 0.0]
        let row = MLXArray(values)
        let maxVal = values.max() ?? 0
        let exps = values.map { exp(Double($0 - maxVal)) }
        let expected = (exps.max() ?? 0) / exps.reduce(0, +)
        let actual = ConfidenceProbeRecorder.maxProbability(logits: row)
        #expect(actual != nil)
        #expect(abs((actual ?? 0) - expected) < 1e-6)

        let nanRow = MLXArray([Float.nan, 1.0, 2.0])
        #expect(ConfidenceProbeRecorder.maxProbability(logits: nanRow) == nil)
        let infRow = MLXArray([Float.infinity, 1.0, 2.0])
        #expect(ConfidenceProbeRecorder.maxProbability(logits: infRow) == nil)
    }

    @Test("ConfidenceProbe aggregates min and mean across steps and merges segments weighted by token count")
    func confidenceAggregationAndMerge() throws {
        let probe = ConfidenceProbe()
        probe.record(0.9)
        probe.record(0.6)
        probe.record(0.3)
        probe.record(.nan)
        let summary = try #require(probe.summary)
        #expect(abs(summary.min - 0.3) < 1e-12)
        #expect(abs(summary.mean - 0.6) < 1e-12)

        let other = ResponseConfidence(min: 0.5, mean: 0.8)
        let merged = summary.merged(with: other, weightSelf: 3, weightOther: 1)
        #expect(abs(merged.min - 0.3) < 1e-12)
        #expect(abs(merged.mean - 0.65) < 1e-12)
    }

    @Test("SessionLog records confidence {min, mean} when present and omits it when off")
    func sessionLogRecordsConfidenceOptIn() async throws {
        let dir = SessionLogTests.tempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let writer = SessionLogWriter(mode: .meta, directory: dir, defaultSession: "test")
        let plain = GenerationResponse(
            generatedText: "ok", promptTokens: 4, completionTokens: 2, tokensPerSecond: 10,
            generationTime: 0.2, timeToFirstToken: 0.05,
            memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0), finishReason: .stop
        )
        let confident = plain.withConfidence(ResponseConfidence(min: 0.42, mean: 0.84))
        await writer.logResponse(GenerationRequest(prompt: "p1"), id: "req_plain", status: "ok", result: SessionLogResult(plain), queueWait: .zero)
        await writer.logResponse(GenerationRequest(prompt: "p2"), id: "req_conf", status: "ok", result: SessionLogResult(confident), queueWait: .zero)
        await writer.flush()

        let records = try SessionLogTests.lines(in: dir).filter { $0["type"] as? String == "response" }
        let plainRecord = try #require(records.first { $0["request_id"] as? String == "req_plain" })
        let confRecord = try #require(records.first { $0["request_id"] as? String == "req_conf" })
        #expect(plainRecord["confidence"] == nil)
        let loggedConf = try #require(confRecord["confidence"] as? [String: Double])
        #expect(abs((loggedConf["min"] ?? 0) - 0.42) < 1e-9)
        #expect(abs((loggedConf["mean"] ?? 0) - 0.84) < 1e-9)
    }

    @Test("Confidence recording on TinyModel(.qwen35) leaves greedy tokens and output bit-identical to when it is off")
    func confidenceProbePreservesBitIdenticalOutput() async throws {
        let engine = try await TinyModel.engine(.qwen35)
        let request = GenerationRequest(prompt: "cascade check", maxTokens: 8, temperature: 0)
        let baseline = try await engine.generate(request: request)
        #expect(baseline.confidence == nil)

        let recorded = try await engine.generate(request: request.withConfidenceRecording(true))
        #expect(recorded.generatedText == baseline.generatedText)
        #expect(recorded.completionTokens == baseline.completionTokens)
        let confidence = try #require(recorded.confidence)
        #expect(confidence.min > 0 && confidence.min <= confidence.mean && confidence.mean <= 1.0)
    }
}

@Suite("Segment-level speculative cascades (#39)")
struct CascadeRoutingTests {

    private func makeCloudClient(host: String) throws -> OpenRouterClient {
        let config = OpenRouterClient.Config(
            apiKey: "test-key", baseURL: URL(string: "http://\(host)/api/v1")!, timeout: 5, maxRetries: 1
        )
        return try OpenRouterClient(config: config, configuration: StubURLProtocol.configuration)
    }

    private func stubCloudChat(content: String = "cloud continuation", promptTokens: Int = 20, completionTokens: Int = 12) -> String {
        StubURLProtocol.host { _, _ in
            let message = OpenRouterClient.ChatResponse.Choice.Message(role: "assistant", content: content)
            let choice = OpenRouterClient.ChatResponse.Choice(message: message, finishReason: "stop")
            let usage = OpenRouterClient.ChatResponse.Usage(
                promptTokens: promptTokens, completionTokens: completionTokens, totalTokens: promptTokens + completionTokens
            )
            let response = OpenRouterClient.ChatResponse(id: "gen-1", model: "openai/gpt-4-turbo", choices: [choice], usage: usage)
            let data = (try? JSONEncoder().encode(response)) ?? Data()
            return (200, ["Content-Type": "application/json"], data)
        }
    }

    @Test("Confident local segments complete locally with zero cloud calls")
    func confidentSegmentsMakeNoCloudCall() async throws {
        let engine = MockInferenceEngine()
        try await engine.load(modelPath: "local-qwen")
        let countBox = SendableBox(0)
        await engine.setResponder { _ in
            countBox.next() == 1 ? "func add(a: Int, b: Int) -> Int {\n" : "    return a + b\n}"
        }
        await engine.setFinishReasonResponder { _ in
            countBox.current == 1 ? .length : .stop
        }
        await engine.setMockConfidence(ResponseConfidence(min: 0.92, mean: 0.96))

        let host = stubCloudChat()
        let client = try makeCloudClient(host: host)
        let cascade = CascadeExecutionMode(segmentTokens: 10, alpha: 0.2, referenceBudget: 10.0, cloudModel: "gpt-4")
        let router = ModelRouter(
            localOrchestrator: ModelOrchestratorActor(engine: engine),
            cloudClient: client,
            options: ModelRouter.Options(strategy: .localOnly).withCascade(cascade)
        )

        let response = try await router.route(
            request: GenerationRequest(prompt: "Write add in Swift", maxTokens: 20),
            preferredModel: "local-qwen"
        )
        #expect(response.generatedText == "func add(a: Int, b: Int) -> Int {\n    return a + b\n}")
        #expect(StubURLProtocol.requests(for: host).isEmpty)
        #expect(await router.cascadeDeferrals == 0)
        let conf = try #require(response.confidence)
        #expect(abs(conf.min - 0.92) < 1e-9)
    }

    @Test("A low-confidence segment after a confident segment defers to cloud with the accepted local prefix as a trailing assistant message")
    func lowConfidenceSegmentDefersWithAssistantPrefill() async throws {
        let engine = MockInferenceEngine()
        try await engine.load(modelPath: "local-qwen")
        let countBox = SendableBox(0)
        let confidenceBox = SendableBox(0)
        await engine.setResponder { _ in
            countBox.next() == 1 ? "func solve() {\n" : "    unsure draft"
        }
        await engine.setFinishReasonResponder { _ in .length }
        await engine.setConfidenceResponder { _ in
            confidenceBox.next() == 1
                ? ResponseConfidence(min: 0.91, mean: 0.95)
                : ResponseConfidence(min: 0.35, mean: 0.60)
        }

        let host = stubCloudChat(content: "    return 42\n}")
        let client = try makeCloudClient(host: host)
        let cascade = CascadeExecutionMode(segmentTokens: 10, alpha: 0.2, referenceBudget: 10.0, cloudModel: "gpt-4")
        let router = ModelRouter(
            localOrchestrator: ModelOrchestratorActor(engine: engine),
            cloudClient: client,
            options: ModelRouter.Options(strategy: .localOnly).withCascade(cascade)
        )

        let response = try await router.route(
            request: GenerationRequest(prompt: "Implement solve()", maxTokens: 20),
            preferredModel: "local-qwen",
            context: ModelRouter.RoutingContext(sessionId: "sess-1")
        )
        #expect(response.generatedText == "func solve() {\n    return 42\n}")
        #expect(await router.cascadeDeferrals == 1)

        let cloudRequests = StubURLProtocol.requests(for: host)
        #expect(cloudRequests.count == 1)
        let bodyData = try #require(cloudRequests.first?.1)
        let payload = try #require(try JSONSerialization.jsonObject(with: bodyData) as? [String: Any])
        let messages = try #require(payload["messages"] as? [[String: String]])
        #expect(messages.count == 2)
        #expect(messages[0]["role"] == "user")
        #expect(messages[0]["content"] == "Implement solve()")
        #expect(messages[1]["role"] == "assistant")
        #expect(messages[1]["content"] == "func solve() {\n")
    }

    @Test("Exhausted CostTracker budget prevents cascade deferral and finishes locally")
    func exhaustedBudgetPreventsDeferral() async throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("cascade-cost-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: file) }
        let budget = CostTracker.Budget(dailyLimit: 1.0, monthlyLimit: 10.0)
        let tracker = try CostTracker(storageURL: file, budget: budget)
        try await tracker.recordUsage(model: "openai/gpt-4-turbo", cost: 1.0)

        let engine = MockInferenceEngine()
        try await engine.load(modelPath: "local-qwen")
        await engine.setMockConfidence(ResponseConfidence(min: 0.25, mean: 0.50))

        let host = stubCloudChat()
        let client = try makeCloudClient(host: host)
        let cascade = CascadeExecutionMode(segmentTokens: 16, alpha: 0.2, referenceBudget: 1.0, cloudModel: "gpt-4")

        let brokeRouter = ModelRouter(
            localOrchestrator: ModelOrchestratorActor(engine: engine),
            cloudClient: client,
            options: ModelRouter.Options(strategy: .localOnly, costTracker: tracker).withCascade(cascade)
        )
        let localReply = try await brokeRouter.route(
            request: GenerationRequest(prompt: "hi", maxTokens: 10),
            preferredModel: "local-qwen"
        )
        #expect(localReply.generatedText == "Mock response")
        #expect(StubURLProtocol.requests(for: host).isEmpty)
    }

    @Test("High #34 stickiness penalty prevents cascade deferral on a warm local session")
    func stickinessPenaltyPreventsDeferral() async throws {
        let engine = MockInferenceEngine()
        try await engine.load(modelPath: "local-qwen")
        await engine.setMockConfidence(ResponseConfidence(min: 0.65, mean: 0.75))

        let host = stubCloudChat()
        let client = try makeCloudClient(host: host)
        let cascade = CascadeExecutionMode(segmentTokens: 16, alpha: 0.2, referenceBudget: 1.0, cloudModel: "gpt-4")
        let stickyRouting = ModelRouter.CostRouting(
            policy: StickyRoutingPolicy(
                strongModel: "gpt-4", weakModel: "local-qwen", threshold: 0.5,
                cacheStickinessPenalty: 0.6, switchCostTokenScale: 1_000
            ),
            scorer: try BradleyTerryRouterTests.scorer()
        )
        let stickyRouter = ModelRouter(
            localOrchestrator: ModelOrchestratorActor(engine: engine),
            cloudClient: client,
            options: ModelRouter.Options(strategy: .localOnly, costRouting: stickyRouting).withCascade(cascade)
        )
        let session = ModelRouter.RoutingContext(sessionId: "warm-agent")
        let warmPrompt = Array(repeating: BradleyTerryRouterTests.easyWord, count: 4_000).joined(separator: " ")
        _ = await stickyRouter.resolveModel(nil, request: GenerationRequest(prompt: warmPrompt), context: session)
        let keptReply = try await stickyRouter.route(
            request: GenerationRequest(prompt: "follow-up", maxTokens: 10),
            preferredModel: "local-qwen",
            context: session
        )
        #expect(keptReply.generatedText == "Mock response")
        #expect(StubURLProtocol.requests(for: host).isEmpty)
        #expect(await stickyRouter.cascadeDeferrals == 0)
    }
}

@Suite("Streaming and fallback cascade behavior (#39)")
struct CascadeStreamingTests {

    private func makeCloudClient(host: String) throws -> OpenRouterClient {
        let config = OpenRouterClient.Config(
            apiKey: "test-key", baseURL: URL(string: "http://\(host)/api/v1")!, timeout: 5, maxRetries: 1
        )
        return try OpenRouterClient(config: config, configuration: StubURLProtocol.configuration)
    }

    @Test("Streaming cascade yields confident local segments and streams cloud continuation on deferral")
    func streamingCascadeDefersAndStreamsContinuation() async throws {
        let engine = MockInferenceEngine()
        try await engine.load(modelPath: "local-qwen")
        let countBox = SendableBox(0)
        let confidenceBox = SendableBox(0)
        await engine.setResponder { _ in
            countBox.next() == 1 ? "let x = 1\n" : "draft?"
        }
        await engine.setFinishReasonResponder { _ in .length }
        await engine.setConfidenceResponder { _ in
            confidenceBox.next() == 1
                ? ResponseConfidence(min: 0.90, mean: 0.95)
                : ResponseConfidence(min: 0.30, mean: 0.55)
        }

        let sse = [
            #"data: {"id":"1","model":"openai/gpt-4-turbo","choices":[{"delta":{"content":"let y = 2\n"}}]}"#,
            #"data: {"id":"1","model":"openai/gpt-4-turbo","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":6,"total_tokens":21}}"#,
            "data: [DONE]",
            ""
        ].joined(separator: "\n\n")
        let host = StubURLProtocol.host { _, _ in
            (200, ["Content-Type": "text/event-stream"], Data(sse.utf8))
        }
        let client = try makeCloudClient(host: host)
        let cascade = CascadeExecutionMode(segmentTokens: 10, alpha: 0.2, referenceBudget: 10.0, cloudModel: "gpt-4")
        let router = ModelRouter(
            localOrchestrator: ModelOrchestratorActor(engine: engine),
            cloudClient: client,
            options: ModelRouter.Options(strategy: .localOnly).withCascade(cascade)
        )

        var chunks: [String] = []
        let stream = try await router.routeStream(
            request: GenerationRequest(prompt: "Write x and y", maxTokens: 20),
            preferredModel: "local-qwen"
        )
        for try await chunk in stream {
            if case .text(let t) = chunk { chunks.append(t) }
        }
        #expect(chunks.joined() == "let x = 1\nlet y = 2\n")
        #expect(await router.cascadeDeferrals == 1)
    }

    @Test("When every cloud model fails during a cascade deferral, the router finishes locally with the accepted prefix")
    func cascadeDeferralCloudFailureFallsBackLocally() async throws {
        let engine = MockInferenceEngine()
        try await engine.load(modelPath: "local-qwen")
        let countBox = SendableBox(0)
        let confidenceBox = SendableBox(0)
        await engine.setResponder { _ in
            let step = countBox.next()
            switch step {
            case 1: return "prefix\n"
            case 2: return "unsure"
            default: return "local fallback tail"
            }
        }
        await engine.setFinishReasonResponder { _ in
            countBox.current < 3 ? .length : .stop
        }
        await engine.setConfidenceResponder { _ in
            confidenceBox.next() == 1
                ? ResponseConfidence(min: 0.95, mean: 0.98)
                : ResponseConfidence(min: 0.20, mean: 0.40)
        }

        let downHost = StubURLProtocol.host { _, _ in (502, [:], Data("bad gateway".utf8)) }
        let client = try makeCloudClient(host: downHost)
        let cascade = CascadeExecutionMode(segmentTokens: 10, alpha: 0.2, referenceBudget: 10.0, cloudModel: "gpt-4")
        let router = ModelRouter(
            localOrchestrator: ModelOrchestratorActor(engine: engine),
            cloudClient: client,
            options: ModelRouter.Options(strategy: .localOnly).withCascade(cascade)
        )

        let response = try await router.route(
            request: GenerationRequest(prompt: "Write code", maxTokens: 20),
            preferredModel: "local-qwen"
        )
        #expect(response.generatedText == "prefix\nlocal fallback tail")
        #expect(await router.cascadeDeferrals == 1)
        #expect(StubURLProtocol.requests(for: downHost).count == 2)
    }
}

@Suite("Real-model segment confidence distribution (#39)", .serialized)
struct CascadeConfidenceRealModelTests {
    static let env = ProcessInfo.processInfo.environment

    static var models: [String] {
        guard env["BASALT_E2E"] == "1" else { return [] }
        return (env["BASALT_E2E_CASCADE_MODELS"] ?? "").split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
    }

    static let codingPrompts: [(name: String, prompt: String)] = [
        ("binary-search", "Write a Swift function binarySearch(_ array: [Int], target: Int) -> Int? using a while loop."),
        ("lru-cache", "Implement a concise generic LRUCache<Key: Hashable, Value> struct in Swift with get and put."),
        ("merge-intervals", "Write a Swift function mergeIntervals(_ intervals: [(Int, Int)]) -> [(Int, Int)] that merges overlapping intervals.")
    ]

    @Test("Measures per-segment Φ_q (min, mean) across a coding task set",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_CASCADE_MODELS"), arguments: models)
    func measureSegmentConfidence(entry: String) async throws {
        let path = entry.hasPrefix("/") ? entry : ModelCache.cacheDir(for: entry).path
        let log = try await ModelRunLog("cascade-confidence", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 512 << 20, recordConfidence: true))
        try await log.step("load") { try await engine.load(modelPath: path) }

        var allMins: [Double] = []
        var allMeans: [Double] = []
        for task in Self.codingPrompts {
            var prefix = ""
            for seg in 0..<2 {
                var base = GenerationRequest(
                    prompt: task.prompt, maxTokens: 16, temperature: 0,
                    messages: [ChatMessageInput(role: .user, text: task.prompt)]
                ).withRouting(sessionId: "cascade-\(task.name)", budgetKey: nil)
                base.enableThinking = false
                let req = CascadeExecutionMode.appendingAssistantPrefill(prefix, to: base)
                    .withMaxTokens(16)
                    .withConfidenceRecording(true)
                let resp = try await engine.generate(request: req)
                let conf = try #require(resp.confidence)
                allMins.append(conf.min)
                allMeans.append(conf.mean)
                prefix += resp.generatedText
                log.record(
                    "segment",
                    String(format: "%@ seg=%d tokens=%d phi_min=%.4f phi_mean=%.4f", task.name, seg, resp.completionTokens, conf.min, conf.mean)
                )
                if resp.finishReason != .length || resp.generatedText.isEmpty { break }
            }
        }
        let sortedMins = allMins.sorted()
        let p25 = sortedMins[sortedMins.count / 4]
        let medianMin = sortedMins[sortedMins.count / 2]
        let meanOfMeans = allMeans.reduce(0, +) / Double(max(1, allMeans.count))
        let chosenAlpha = ((1.0 - p25) * 100).rounded() / 100
        log.record(
            "distribution",
            String(format: "segments=%d phi_min_p25=%.4f phi_min_median=%.4f phi_mean_avg=%.4f recommended_alpha=%.2f",
                   allMins.count, p25, medianMin, meanOfMeans, chosenAlpha)
        )
        #expect(!allMins.isEmpty)
    }
}

private final class SendableBox: Sendable {
    private let value: Mutex<Int>

    init(_ initial: Int) {
        value = Mutex(initial)
    }

    func next() -> Int {
        value.withLock {
            $0 += 1
            return $0
        }
    }

    var current: Int {
        value.withLock { $0 }
    }
}
