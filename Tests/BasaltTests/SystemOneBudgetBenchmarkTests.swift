import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("System One + LLM budget: decisions answered by the decision model save LLM tokens and time net of the decode slowdown (#234)", .serialized)
struct SystemOneBudgetBenchmarkTests {
    struct Tally {
        var promptTokens = 0
        var completionTokens = 0
        var llmCalls = 0
        var correct = 0
        var escalated = 0
        var total = 0
        var families: [String: (correct: Int, total: Int)] = [:]

        var tokens: Int { promptTokens + completionTokens }
        var accuracy: Double { Double(correct) / Double(max(total, 1)) }

        mutating func score(_ name: String, _ ok: Bool) {
            (total, correct) = (total + 1, correct + (ok ? 1 : 0))
            let family = families[name] ?? (0, 0)
            families[name] = (family.correct + (ok ? 1 : 0), family.total + 1)
        }

        var breakdown: String { families.keys.sorted().map { "\($0) \(families[$0]!.correct)/\(families[$0]!.total)" }.joined(separator: ", ") }

        mutating func add(_ response: GenerationResponse) {
            (promptTokens, completionTokens, llmCalls) = (promptTokens + response.promptTokens, completionTokens + response.completionTokens, llmCalls + 1)
        }
    }

    static let env = ProcessInfo.processInfo.environment
    static let chat = env["BASALT_E2E"] == "1" ? env["BASALT_E2E_BUDGET_CHAT_MODEL"] : nil
    static let thresholds = [0.0, 0.6, 0.75, 0.9]
    static let longPrompt = "Write a detailed, well-structured technical explanation of how unified memory on Apple Silicon affects running several machine learning models at once."

    static func llmDecision(_ engine: MLXInferenceEngine, _ item: LabelledDecision, _ tally: inout Tally) async throws -> String? {
        var request = GenerationRequest(prompt: SystemOneDecisionSet.llmPrompt(item), maxTokens: 8, temperature: 0)
        request.reasoningEffort = "none"
        let response = try await engine.generate(request: request)
        tally.add(response)
        return SystemOneDecisionSet.parse(response.generatedText, keys: SystemOneCore.options(item.question).keys)
    }

    static func llmOnly(_ engine: MLXInferenceEngine, _ items: [LabelledDecision]) async throws -> Tally {
        var tally = Tally()
        for item in items {
            let answer = try await llmDecision(engine, item, &tally)
            tally.score(item.name, answer == item.label)
        }
        return tally
    }

    static func withSystemOne(_ engine: MLXInferenceEngine, _ jev: any DecisionEngine, _ items: [LabelledDecision], threshold: Double) async throws -> (Tally, [Double]) {
        var (tally, latencies) = (Tally(), [Double]())
        for group in Dictionary(grouping: items, by: \.state).values {
            let start = ContinuousClock.now
            let response = try await jev.decide(SystemOneRequest(state: group[0].state, questions: Dictionary(uniqueKeysWithValues: group.map { ($0.name, $0.question) }), abstainBelow: threshold))
            let elapsed = ContinuousClock.now - start
            latencies.append(Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15)
            for item in group {
                guard let answer = response.answers[item.name] else { continue }
                var predicted = SystemOneDecisionSet.predicted(answer)
                if answer.abstain == true {
                    tally.escalated += 1
                    predicted = try await llmDecision(engine, item, &tally)
                }
                tally.score(item.name, predicted == item.label)
            }
        }
        return (tally, latencies.sorted())
    }

    static func longAnswer(_ engine: MLXInferenceEngine) async throws -> GenerationResponse {
        var request = GenerationRequest(prompt: longPrompt, maxTokens: 1024, temperature: 0)
        request.reasoningEffort = "none"
        return try await engine.generate(request: request)
    }

    @Test("LLM-only vs System One with escalation at four thresholds, then System One running during a long generation; tokens, time, accuracy and decode tok/s are logged",
          .enabled(if: chat != nil && env["BASALT_E2E_JEV_MODEL"] != nil && env["BASALT_E2E_JEV_BASE"] != nil, "set BASALT_E2E=1, BASALT_E2E_BUDGET_CHAT_MODEL, BASALT_E2E_JEV_MODEL and BASALT_E2E_JEV_BASE"))
    func budget() async throws {
        let path = NF4Run.path(try #require(Self.chat))
        let log = try await ModelRunLog("systemone-budget", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 1 << 30))
        try await log.step("load chat") { try await engine.load(modelPath: path) }
        let jev = try await log.step("load jev") {
            try await JevEvaluator(directory: URL(fileURLWithPath: try #require(Self.env["BASALT_E2E_JEV_MODEL"])), base: URL(fileURLWithPath: try #require(Self.env["BASALT_E2E_JEV_BASE"])))
        }
        let items = SystemOneDecisionSet.generate(states: Int(Self.env["BASALT_E2E_BUDGET_STATES"] ?? "") ?? 100, seed: 234)
        let solo = try await log.step("long answer solo") { try await Self.longAnswer(engine) }
        var clock = ContinuousClock.now
        let baseline = try await log.step("LLM only") { try await Self.llmOnly(engine, items) }
        let baselineSeconds = Self.seconds(since: clock)
        log.record("RESULT", "A llm-only: \(baseline.total) decisions, accuracy \(baseline.accuracy) [\(baseline.breakdown)], \(baseline.llmCalls) calls, \(baseline.promptTokens)+\(baseline.completionTokens) tokens, \(String(format: "%.1f", baselineSeconds)) s; long answer solo \(String(format: "%.2f", solo.tokensPerSecond)) tok/s")
        var deciders: [(String, any DecisionEngine)] = [("openjev-0.6b", jev), ("readout-self", ReadoutDecisionEngine(engine: engine, modelID: "self", readout: .uncalibrated))]
        if let flash = Self.env["BASALT_E2E_BUDGET_FLASH"] {
            let flashEngine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
            try await log.step("load flash") { try await flashEngine.load(modelPath: flash) }
            deciders.append(("openjev-flash-9b", ReadoutDecisionEngine(engine: flashEngine, modelID: "flash", readout: SystemOneReadout.known("openjev/OpenJev-Flash-9B") ?? .uncalibrated)))
        }
        for (label, decider) in deciders {
            for threshold in Self.thresholds {
                try await Self.arm(label, decider, threshold: threshold, run: Run(engine: engine, items: items, baseline: baseline, log: log))
            }
        }
        clock = ContinuousClock.now
        async let concurrent = Self.longAnswer(engine)
        let (overlapTally, _) = try await Self.withSystemOne(engine, jev, items, threshold: 0.75)
        let overlap = try await concurrent
        log.record("RESULT", "C concurrent @0.75: long answer \(String(format: "%.2f", overlap.tokensPerSecond)) tok/s (\(String(format: "%+.1f", (overlap.tokensPerSecond / max(solo.tokensPerSecond, 1e-9) - 1) * 100))% vs solo), decisions accuracy \(overlapTally.accuracy), wall \(String(format: "%.1f", Self.seconds(since: clock))) s vs A+long \(String(format: "%.1f", baselineSeconds + solo.generationTime)) s")
    }

    struct Run {
        let engine: MLXInferenceEngine
        let items: [LabelledDecision]
        let baseline: Tally
        let log: ModelRunLog
    }

    static func arm(_ label: String, _ decider: any DecisionEngine, threshold: Double, run: Run) async throws {
        let (engine, items, baseline, log) = (run.engine, run.items, run.baseline, run.log)
        let clock = ContinuousClock.now
        let (tally, latencies) = try await log.step("\(label) @\(threshold)") { try await Self.withSystemOne(engine, decider, items, threshold: threshold) }
        log.record("RESULT", "B \(label) abstain_below=\(threshold): accuracy \(tally.accuracy) [\(tally.breakdown)], escalated \(tally.escalated)/\(tally.total), \(tally.llmCalls) calls, \(tally.promptTokens)+\(tally.completionTokens) tokens (\(String(format: "%.1f", 100 * (1 - Double(tally.tokens) / Double(max(baseline.tokens, 1)))))% fewer), \(String(format: "%.1f", Self.seconds(since: clock))) s, decide p50 \(String(format: "%.1f", latencies[latencies.count / 2])) ms p95 \(String(format: "%.1f", latencies[latencies.count * 95 / 100])) ms")
    }

    static func seconds(since start: ContinuousClock.Instant) -> Double {
        let elapsed = ContinuousClock.now - start
        return Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
    }
}
