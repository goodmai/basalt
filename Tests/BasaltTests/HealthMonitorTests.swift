import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXNN
@testable import BasaltCore
@testable import BasaltInference

private func response(
    tokens: Int = 64, tps: Double = 40, finish: GenerationResponse.FinishReason = .stop, drafted: Int? = nil, accepted: Int? = nil,
    source: DraftSource? = nil, degenerate: DegenerateOutput? = nil, text: String = ""
) -> GenerationResponse {
    GenerationResponse(
        generatedText: text, promptTokens: 3, completionTokens: tokens, tokensPerSecond: tps, generationTime: 1, timeToFirstToken: 0,
        memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0), finishReason: finish, draftTokens: drafted, acceptedDraftTokens: accepted
    ).reporting(degenerate: degenerate, draftSource: source)
}

@Suite("Non-finite logit guard (#325)")
struct NonFiniteLogitGuardTests {
    static let rows: [([Float], Bool)] = [
        ([1, 2, 3], false),
        ([1, -.infinity, 3], false),
        ([1, .nan, 3], true),
        ([1, .infinity, 3], true),
        ([-.infinity, -.infinity, -.infinity], true),
    ]

    @Test("Only NaN, +inf or a row with no finite logit trips; -inf masks do not", arguments: rows)
    func detects(_ row: [Float], trips: Bool) {
        let alarm = NonFiniteLogitAlarm()
        let logits = MLXArray(row).reshaped(1, -1)
        #expect(NonFiniteLogitGuard(alarm: alarm).process(logits: logits) === logits)
        #expect(alarm.isTripped == trips)
    }

    @Test("Reads the logits only every interval-th sampled step")
    func checksEveryInterval() {
        let alarm = NonFiniteLogitAlarm()
        var guarded = NonFiniteLogitGuard(alarm: alarm)
        let poisoned = MLXArray([Float(1), .nan]).reshaped(1, -1)
        guarded.didSample(token: MLXArray(0))
        for _ in 1 ..< NonFiniteLogitGuard.interval {
            _ = guarded.process(logits: poisoned)
            guarded.didSample(token: MLXArray(0))
        }
        #expect(!alarm.isTripped)
        _ = guarded.process(logits: poisoned)
        #expect(alarm.isTripped)
    }

    static let chains: [(Float, Bool, Bool)] = [(0, false, false), (0.7, false, true), (0, true, true)]

    @Test("Plain greedy requests keep an empty processor chain; sampled or processed ones get the guard", arguments: chains)
    func attaches(temperature: Float, hasProcessor: Bool, guarded: Bool) {
        let base = hasProcessor ? GenerationComponents().appendingBasaltProcessor { ExcludeTopChoices(probability: 0, seed: 1) } : GenerationComponents()
        let (components, _) = NonFiniteLogitGuard.guarding(base, temperature: temperature)
        let processor = components.logitProcessorFactory?()
        #expect((processor is NonFiniteLogitGuard || (processor as? BasaltChainedLogitProcessor)?.processors.last is NonFiniteLogitGuard) == guarded)
    }

    @Test("Non-finite logits end the request with a 500 server error")
    func wire() {
        let wire = BasaltError.nonFiniteLogits.wire
        #expect(wire.status == 500)
        #expect(wire.openAIType == "server_error")
        #expect(wire.anthropicType == "api_error")
    }
}

@Suite("Degenerate output detector (#325)")
struct DegenerateOutputDetectorTests {
    typealias Segment = ThinkingSplitter.Segment

    static let repeated = Array(repeating: Segment.answer(" the"), count: DegenerateOutputDetector.repeatLimit)
    static let cases: [([Segment], Bool, DegenerateOutput?)] = [
        ([.reasoning("think"), .answer("Paris.")], false, nil),
        ([.answer("Paris.")], false, nil),
        ([], false, nil),
        ([.reasoning("think")], false, .emptyAfterThinking),
        ([.reasoning("think")], true, nil),
        ([.reasoning("think"), .answer("\n\n  ")], false, .blankAnswer),
        ([.answer("<|im_end|>"), .answer(" \n<|endoftext|>")], false, .blankAnswer),
        ([.answer("<|im_"), .answer("end|>")], false, .blankAnswer),
        (repeated, false, .repeatedToken),
        (Array(repeated.dropLast()), false, nil),
        (repeated, true, .repeatedToken),
    ]

    @Test("Flags repetition, an empty answer after thinking and a blank answer", arguments: cases)
    func verdict(_ segments: [Segment], calledTools: Bool, expected: DegenerateOutput?) {
        var detector = DegenerateOutputDetector(controlTokens: ["<|im_end|>", "<|endoftext|>"])
        for segment in segments { detector.feed(segment) }
        #expect(detector.verdict(calledTools: calledTools) == expected)
    }

    @Test("A different token between repeats resets the run")
    func runResets() {
        var detector = DegenerateOutputDetector(controlTokens: [])
        for index in 0 ..< DegenerateOutputDetector.repeatLimit * 2 { detector.feed(.answer(index.isMultiple(of: 50) ? "x" : "a")) }
        #expect(detector.verdict(calledTools: false) == nil)
    }
}

@Suite("Drift watch (#325)")
struct DriftWatchTests {
    static let steady = Array(repeating: 40.0, count: DriftWatch.window + DriftWatch.warmup)
    static func run(_ value: Double, _ count: Int = DriftWatch.window) -> [Double] { Array(repeating: value, count: count) }
    static let sequences: [([Double], Int, Bool)] = [
        (steady + run(40), 0, false),
        (steady + run(30), 0, false),
        (steady + run(20), 1, true),
        (steady + run(20) + run(40), 1, false),
        (steady + run(20) + run(40) + run(10), 2, true),
        (run(40) + run(0, DriftWatch.warmup - 1), 0, false),
    ]

    @Test("Alerts once when the window mean falls below the floor of the baseline, and not before the warm-up", arguments: sequences)
    func alerts(_ samples: [Double], alerts: Int, drifting: Bool) {
        var watch = DriftWatch()
        let fired = samples.filter { watch.observe($0) }.count
        #expect(fired == alerts)
        #expect(watch.alerts == alerts)
        #expect(watch.drifting == drifting)
    }

    @Test("Samples: decode speed from long enough answers, acceptance by draft source, nothing from failed requests")
    func samples() {
        let keys = { (sampled: GenerationResponse) in DriftMetric.samples(sampled).map { $0.0 } }
        #expect(keys(response()) == [.decodeTokensPerSecond])
        #expect(keys(response(tokens: DriftMetric.minimumDecodeTokens - 1)).isEmpty)
        #expect(keys(response(finish: .error)).isEmpty)
        #expect(keys(response(drafted: 10, accepted: 7, source: .mtp)) == [.decodeTokensPerSecond, .mtpAcceptance])
        #expect(keys(response(tokens: 8, drafted: 10, accepted: 3, source: .lookup)) == [.lookupAcceptance])
        #expect(keys(response(drafted: 10, accepted: 7)) == [.decodeTokensPerSecond])
        #expect(DriftMetric.samples(response(tokens: 8, drafted: 10, accepted: 3, source: .lookup)).first?.1 == 0.3)
    }

    @Test("The degenerate verdict and the draft source survive a JSON round trip of the response")
    func responseCodable() throws {
        let original = response(drafted: 4, accepted: 2, source: .mtp, degenerate: .emptyAfterThinking)
        let decoded = try JSONDecoder().decode(GenerationResponse.self, from: JSONEncoder().encode(original))
        #expect(decoded.degenerate == .emptyAfterThinking)
        #expect(decoded.draftSource == .mtp)
        #expect(try JSONDecoder().decode(GenerationResponse.self, from: JSONEncoder().encode(response())).degenerate == nil)
    }

    @Test("Health telemetry keeps baselines per model, counts degenerate outputs and warns on drift")
    func perModel() {
        var health = HealthTelemetry()
        var warnings: [String] = []
        for _ in DriftWatchTests.steady { warnings += health.record(response(tps: 40), model: "a") }
        for _ in 0 ..< DriftWatch.window {
            warnings += health.record(response(tps: 10), model: "a")
            warnings += health.record(response(tps: 10), model: "b")
        }
        warnings += health.record(response(degenerate: .blankAnswer), model: "b")
        #expect(health.drift["a"]?["decodeTokensPerSecond"]?.drifting == true)
        #expect(health.drift["b"]?["decodeTokensPerSecond"]?.drifting == false)
        #expect(health.degenerateOutputs == ["blankAnswer": 1])
        #expect(warnings.count == 2)
        #expect(warnings.contains { $0.hasPrefix("decodeTokensPerSecond on a dropped to 25 ") })
    }
}

@Suite("Canary (#325)")
struct CanaryTests {
    static let start = ContinuousClock.now
    static func at(_ seconds: Int) -> ContinuousClock.Instant { start + .seconds(seconds) }

    static let timelines: [([(Int, Bool)], [CanarySchedule.Step])] = [
        ([(30, true)], [.wait(until: at(60))]),
        ([(60, true), (61, true)], [.run, .wait(until: at(120))]),
        ([(60, false), (70, true)], [.wait(until: at(70)), .run]),
        ([(60, false), (65, false), (75, true), (80, true)], [.wait(until: at(70)), .wait(until: at(75)), .run, .wait(until: at(135))]),
    ]

    @Test("Runs when due and idle, retries soon when busy, then waits a full interval", arguments: timelines)
    func schedule(_ ticks: [(Int, Bool)], expected: [CanarySchedule.Step]) {
        var schedule = CanarySchedule(interval: .seconds(60), retry: .seconds(10), start: Self.start)
        #expect(ticks.map { schedule.step(at: Self.at($0.0), idle: $0.1) } == expected)
    }

    @Test("The canary is a greedy background service request and passes only on the known answer")
    func request() {
        #expect(Canary.request.priority == .background)
        #expect(Canary.request.temperature == 0)
        #expect(Canary.request.requestDisablesThinking)
        #expect(Canary.passes(response(text: "42")))
        #expect(!Canary.passes(response(text: "41")))
        #expect(!Canary.passes(response(finish: .error, text: "42")))
    }

    @Test("Telemetry snapshot carries canary results and degenerate counts")
    func snapshot() async {
        let tracker = TelemetryTracker()
        await tracker.setModel("m")
        await tracker.recordCanary(passed: true, at: Date(timeIntervalSince1970: 1))
        await tracker.recordCanary(passed: false, at: Date(timeIntervalSince1970: 2))
        await tracker.requestFinished(with: response(degenerate: .repeatedToken))
        let health = await tracker.snapshot().health
        #expect(health?.canary?.passed == 1)
        #expect(health?.canary?.failed == 1)
        #expect(health?.canary?.lastPassed == false)
        #expect(health?.canary?.lastRun == Date(timeIntervalSince1970: 2))
        #expect(health?.degenerateOutputs == ["repeatedToken": 1])
        #expect(health?.drift["m"]?["decodeTokensPerSecond"]?.recent == [40])
    }
}

@Suite("Health monitors on tiny models (#325)", .serialized)
struct HealthMonitorTinyModelTests {
    static let families: [TinyModel.Family] = [.qwen3, .gemma4, .qwen35]

    static func engine(_ family: TinyModel.Family, poisoned: Bool) async throws -> MLXInferenceEngine {
        let model = try TinyModel.make(family)
        if poisoned {
            model.update(parameters: model.parameters().mapValues { $0.dtype.isFloatingPoint ? $0 * Float.nan : $0 })
            eval(model)
        }
        let engine = MLXInferenceEngine(promptCacheBytes: 1 << 26)
        await engine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-\(family.rawValue)"), model: model,
            processor: ByteInputProcessor(), tokenizer: ByteTokenizer())))
        return engine
    }

    static func request(temperature: Double, key: String? = nil) -> GenerationRequest {
        GenerationRequest(prompt: "Explain the build.", maxTokens: 40, temperature: temperature).with { $0.cacheKey = key }
    }

    static func metadata(_ engine: MLXInferenceEngine, _ request: GenerationRequest) async throws -> GenerationResponse {
        var metadata: GenerationResponse?
        for await chunk in try await engine.generateStream(request: request) {
            if case .metadata(let response) = chunk { metadata = response }
        }
        return try #require(metadata)
    }

    @Test("NaN weights end a sampled stream with an error and make generate throw nonFiniteLogits; healthy weights do not trip", arguments: families)
    func sampledRequestTrips(family: TinyModel.Family) async throws {
        let poisoned = try await Self.engine(family, poisoned: true)
        #expect(try await Self.metadata(poisoned, Self.request(temperature: 0.7)).finishReason == .error)
        let error = await #expect(throws: BasaltError.self) { try await poisoned.generate(request: Self.request(temperature: 0.7)) }
        #expect(error?.localizedDescription == BasaltError.nonFiniteLogits.localizedDescription)
        let healthy = try await Self.engine(family, poisoned: false)
        #expect(try await Self.metadata(healthy, Self.request(temperature: 0.7)).finishReason != .error)
    }

    @Test("A plain greedy request carries no guard, so even NaN weights do not end it with an error", arguments: families)
    func greedyRequestSkipsTheGuard(family: TinyModel.Family) async throws {
        let poisoned = try await Self.engine(family, poisoned: true)
        #expect(try await Self.metadata(poisoned, Self.request(temperature: 0)).finishReason != .error)
    }

    @Test("A request stopped on non-finite logits leaves no prompt cache behind")
    func noPromptCacheAfterTrip() async throws {
        let healthy = try await Self.engine(.qwen3, poisoned: false)
        _ = try await Self.metadata(healthy, Self.request(temperature: 0.7, key: "ns"))
        #expect(try await Self.metadata(healthy, Self.request(temperature: 0.7, key: "ns")).cachedPromptTokens > 0)
        let poisoned = try await Self.engine(.qwen3, poisoned: true)
        _ = try await Self.metadata(poisoned, Self.request(temperature: 0.7, key: "ns"))
        #expect(try await Self.metadata(poisoned, Self.request(temperature: 0.7, key: "ns")).cachedPromptTokens == 0)
    }

    @Test("The canary goes through the gate, fails on broken weights and leaves the gate idle", arguments: [TinyModel.Family.gemma4, .qwen35])
    func canaryProbe(family: TinyModel.Family) async throws {
        let healthy = ModelOrchestratorActor(engine: try await Self.engine(family, poisoned: false), telemetry: TelemetryTracker())
        #expect(await Canary.probe(healthy) != nil)
        #expect(await healthy.gate.checkInvariants(idle: true) == [])
        let broken = ModelOrchestratorActor(engine: try await Self.engine(family, poisoned: true), telemetry: TelemetryTracker())
        #expect(await Canary.probe(broken) == false)
        #expect(await broken.gate.checkInvariants(idle: true) == [])
    }

    @Test("A busy server counts as neither a pass nor a failure of the canary")
    func canaryOnBusyServer() async throws {
        let orchestrator = ModelOrchestratorActor(
            engine: try await Self.engine(.gemma4, poisoned: false), telemetry: TelemetryTracker(),
            gate: GenerationGate(maxConcurrent: 1, maxQueue: 0))
        let ticket = try await orchestrator.gate.admit()
        #expect(await Canary.probe(orchestrator) == nil)
        await orchestrator.gate.release(ticket)
    }
}
