import Foundation
import Testing
import MLX
import MLXLMCommon
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
