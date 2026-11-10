import Foundation
import Testing
import Synchronization
@testable import BasaltCore
@testable import BasaltCloud
@testable import BasaltInference

@Suite("Cascade System One on the decision set: jev-latest first, local readout below 0.6; accuracy, escalation, latency, cost and decode tok/s beside it (#407)", .serialized)
struct CascadeDecisionRealModelTests {
    struct Arm {
        var correct = 0
        var total = 0
        var escalated = 0
        var fallbacks = 0
        var cost = 0.0
        var latencies: [Double] = []

        var summary: String {
            let sorted = latencies.sorted()
            let p50 = sorted.isEmpty ? 0 : sorted[sorted.count / 2]
            let p95 = sorted.isEmpty ? 0 : sorted[min(sorted.count - 1, sorted.count * 95 / 100)]
            return String(
                format: "accuracy %.4f (%d/%d), escalated %d/%d (%.1f%%), fallbacks %d, p50 %.0f ms p95 %.0f ms per request, cost $%.6f ($%.5f per 1000 decisions)",
                Double(correct) / Double(max(total, 1)), correct, total, escalated, total, 100 * Double(escalated) / Double(max(total, 1)), fallbacks, p50, p95,
                cost, 1000 * cost / Double(max(total, 1)))
        }

        mutating func add(_ response: SystemOneResponse, items: [LabelledDecision], ms: Double) {
            latencies.append(ms)
            for item in items {
                total += 1
                correct += response.answers[item.name].flatMap(SystemOneDecisionSet.predicted) == item.label ? 1 : 0
            }
            let stages = response.usage.stages ?? []
            if stages.first?.fallback != nil { fallbacks += 1 } else if stages.count > 1 { escalated += stages[1].questions.count }
            cost += response.usage.cost ?? 0
        }
    }

    static let env = ProcessInfo.processInfo.environment
    static let chat = env["BASALT_E2E"] == "1" && env["OPENROUTER_API_KEY"] != nil ? env["BASALT_E2E_CASCADE_MODEL"] : nil
    static let longPrompt = "Write a detailed, well-structured technical explanation of how unified memory on Apple Silicon affects running several machine learning models at once."

    static func groups(_ items: [LabelledDecision]) -> [(request: SystemOneRequest, items: [LabelledDecision])] {
        Dictionary(grouping: items, by: \.state).values.map { group in
            (SystemOneRequest(state: group[0].state, questions: Dictionary(uniqueKeysWithValues: group.map { ($0.name, $0.question) })), group)
        }.sorted { $0.request.state < $1.request.state }
    }

    static func milliseconds(since start: ContinuousClock.Instant) -> Double {
        let elapsed = ContinuousClock.now - start
        return Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15
    }

    static func run(_ decider: any DecisionEngine, _ groups: [(request: SystemOneRequest, items: [LabelledDecision])]) async throws -> Arm {
        var arm = Arm()
        for group in groups {
            let start = ContinuousClock.now
            let response = try await decider.decide(group.request)
            arm.add(response, items: group.items, ms: Self.milliseconds(since: start))
        }
        return arm
    }

    static func longAnswer(_ engine: MLXInferenceEngine) async throws -> GenerationResponse {
        var request = GenerationRequest(prompt: longPrompt, maxTokens: 1024, temperature: 0)
        request.reasoningEffort = "none"
        return try await engine.generate(request: request)
    }

    final class Latencies: Sendable {
        let values = Mutex<[Double]>([])

        func decide(_ decider: any DecisionEngine, _ request: SystemOneRequest) async {
            let start = ContinuousClock.now
            guard (try? await decider.decide(request)) != nil else { return }
            values.withLock { $0.append(CascadeDecisionRealModelTests.milliseconds(since: start)) }
        }
    }

    static func workload(_ decider: any DecisionEngine, _ groups: [(request: SystemOneRequest, items: [LabelledDecision])], burst: Bool, into latencies: Latencies) async {
        guard burst else {
            while !Task.isCancelled {
                for group in groups where !Task.isCancelled { await latencies.decide(decider, group.request) }
            }
            return
        }
        await withTaskGroup(of: Void.self) { tasks in
            for group in groups { tasks.addTask { await latencies.decide(decider, group.request) } }
        }
    }

    static func beside(_ engine: MLXInferenceEngine, _ decider: any DecisionEngine, _ groups: [(request: SystemOneRequest, items: [LabelledDecision])], holding gate: GenerationGate?) async throws -> String {
        let slot = try await gate?.admit()
        let latencies = Latencies()
        let loop = Task { await Self.workload(decider, groups, burst: gate != nil, into: latencies) }
        let answer = try await Self.longAnswer(engine)
        let during = latencies.values.withLock { $0.sorted() }
        if let gate, let slot { await gate.release(slot) }
        loop.cancel()
        await loop.value
        let p50 = during.isEmpty ? 0 : during[during.count / 2]
        return String(format: "long answer %.2f tok/s (%d tokens), %d requests decided during it, p50 %.0f ms", answer.tokensPerSecond, answer.completionTokens, during.count, p50)
    }

    @Test("Local readout, Jev alone and the cascade on 100 states × 3 questions; then a long answer beside each: gated as serve runs it (all 100 requests at once) and ungated (requests in a loop)",
          .enabled(if: chat != nil, "set BASALT_E2E=1, OPENROUTER_API_KEY and BASALT_E2E_CASCADE_MODEL (chat model path or repo, read out as self)"))
    func decisionSet() async throws {
        let path = NF4Run.path(try #require(Self.chat))
        let log = try await ModelRunLog("systemone-cascade", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 1 << 30))
        try await log.step("load chat") { try await engine.load(modelPath: path) }
        let groups = Self.groups(SystemOneDecisionSet.generate(states: 100, seed: 234))
        let local = ReadoutDecisionEngine(engine: engine, modelID: "self", readout: .uncalibrated)
        let remote = try RemoteDecisionEngine(model: "~typesafe/jev-latest", apiKey: try #require(Self.env["OPENROUTER_API_KEY"]))
        let solo = try await log.step("long answer solo") { try await Self.longAnswer(engine) }
        log.record("RESULT", String(format: "solo long answer %.2f tok/s (%d tokens)", solo.tokensPerSecond, solo.completionTokens))
        for (label, decider) in [("local readout-self", local as any DecisionEngine), ("jev-latest alone", remote), ("cascade @0.6", Self.cascade(remote, local, GenerationGate()))] {
            let arm = try await log.step(label) { try await Self.run(decider, groups) }
            log.record("RESULT", "\(label): \(arm.summary)")
        }
        for gated in [true, false] {
            let gate = GenerationGate(maxConcurrent: 1, maxQueue: 128)
            let held = gated ? gate : nil
            let localOnly: any DecisionEngine = gated ? GatedDecisionEngine(inner: local, gate: gate) : local
            for (label, decider) in [("local-only", localOnly), ("cascade", Self.cascade(remote, local, gate))] {
                let line = try await log.step("\(label) beside a long answer, \(gated ? "gated" : "ungated")") { try await Self.beside(engine, decider, groups, holding: held) }
                log.record("RESULT", "beside \(gated ? "gated" : "ungated") \(label): \(line)")
            }
        }
    }

    static func cascade(_ remote: RemoteDecisionEngine, _ local: any DecisionEngine, _ gate: GenerationGate) -> CascadeDecisionEngine {
        CascadeDecisionEngine(remote: remote, local: local, escalateBelow: 0.6, gate: gate)
    }
}
