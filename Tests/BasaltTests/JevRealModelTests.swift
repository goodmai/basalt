import Foundation
import MLX
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("System One on a real OpenJev model against the PyTorch reference (#234)", .serialized)
struct JevRealModelTests {
    struct Case: Decodable {
        struct Expected: Decodable {
            let probabilities: [String: Double]
        }

        let state: String
        let questions: [String: SystemOneQuestion]
        let answers: [String: Expected]
    }

    static let env = ProcessInfo.processInfo.environment
    static let model = env["BASALT_E2E"] == "1" ? env["BASALT_E2E_JEV_MODEL"] : nil
    static let reference = env["BASALT_E2E_JEV_REFERENCE"]

    @Test("Calibrated probabilities match the fp32 reference within 1e-4, one prefix evaluation per request, latency logged",
          .enabled(if: model != nil && reference != nil && env["BASALT_E2E_JEV_BASE"] != nil, "set BASALT_E2E=1, BASALT_E2E_JEV_MODEL (OpenJev dir), BASALT_E2E_JEV_BASE (its Qwen3 base dir) and BASALT_E2E_JEV_REFERENCE"))
    func matchesReference() async throws {
        let directory = URL(fileURLWithPath: try #require(Self.model))
        let base = URL(fileURLWithPath: try #require(Self.env["BASALT_E2E_JEV_BASE"]))
        let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: URL(fileURLWithPath: try #require(Self.reference))))
        let evaluator = try await JevEvaluator(directory: directory, base: base, float32: Self.env["BASALT_E2E_JEV_BF16"] != "1")
        var worst = 0.0
        for item in cases {
            let start = ContinuousClock.now
            let response = try await evaluator.decide(SystemOneRequest(state: item.state, questions: item.questions))
            let elapsed = ContinuousClock.now - start
            #expect(response.usage.prefixEvaluations == 1)
            for (name, expected) in item.answers {
                let got = try #require(response.answers[name])
                let probabilities = got.probabilities ?? got.noul.map { ["false": 1 - $0, "true": $0] } ?? [:]
                let gap = expected.probabilities.map { abs((probabilities[$0.key] ?? -1) - $0.value) }.max() ?? 0
                print("RESULT jev question \(name): gap \(gap), got \(probabilities.sorted { $0.key < $1.key }.map { String(format: "%.4f", $0.value) }), want \(expected.probabilities.sorted { $0.key < $1.key }.map { String(format: "%.4f", $0.value) })")
                worst = max(worst, gap)
            }
            let ms = Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15
            print("RESULT jev \(evaluator.modelID): \(item.questions.count) questions, \(response.usage.candidateBranches) branches, \(response.usage.prefixTokens) prefix tokens, \(String(format: "%.1f", ms)) ms")
        }
        print("RESULT jev worst calibrated probability gap to the fp32 reference \(worst)")
        #expect(worst < 1e-4, "worst probability gap \(worst)")
    }

    struct Arm {
        let label: String
        let base: URL
        let float32: Bool
    }

    struct ArmResult {
        var probabilities: [String: [String: Double]] = [:]
        var correct = 0
        var milliseconds: [Double] = []
        var activeMB = 0
    }

    static func run(_ arm: Arm, model: URL, items: [LabelledDecision]) async throws -> ArmResult {
        Memory.peakMemory = 0
        let evaluator = try await JevEvaluator(directory: model, base: arm.base, float32: arm.float32)
        var result = ArmResult(activeMB: Memory.activeMemory >> 20)
        var seen: [String] = []
        for item in items where !seen.contains(item.state) { seen.append(item.state) }
        for (index, state) in seen.enumerated() {
            let group = items.filter { $0.state == state }
            let start = ContinuousClock.now
            let response = try await evaluator.decide(SystemOneRequest(state: state, questions: Dictionary(uniqueKeysWithValues: group.map { ($0.name, $0.question) })))
            let elapsed = ContinuousClock.now - start
            result.milliseconds.append(Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15)
            for item in group {
                let answer = try #require(response.answers[item.name])
                result.probabilities["\(index).\(item.name)"] = answer.probabilities ?? answer.noul.map { ["false": 1 - $0, "true": $0] } ?? [:]
                if SystemOneDecisionSet.predicted(answer) == item.label { result.correct += 1 }
            }
        }
        return result
    }

    @Test("Unmerged LoRA on the stored base and on a 4-bit base against the merged float32 path: label accuracy within 2% and top answers within 10% on bf16 and 8-bit bases, the 4-bit base logged, memory and latency (#423 J2)",
          .enabled(if: model != nil && env["BASALT_E2E_JEV_BASE"] != nil, "set BASALT_E2E=1, BASALT_E2E_JEV_MODEL, BASALT_E2E_JEV_BASE and optionally BASALT_E2E_JEV_BASE_8BIT, BASALT_E2E_JEV_BASE_4BIT"))
    func unmergedArms() async throws {
        let model = URL(fileURLWithPath: try #require(Self.model))
        let base = URL(fileURLWithPath: try #require(Self.env["BASALT_E2E_JEV_BASE"]))
        let log = try await ModelRunLog("jev-unmerged", modelPath: base.path)
        let items = SystemOneDecisionSet.generate(states: Int(Self.env["BASALT_E2E_JEV_STATES"] ?? "") ?? 60, seed: 423)
        var arms = [Arm(label: "merged-float32", base: base, float32: true), Arm(label: "unmerged-stored", base: base, float32: false)]
        for bits in [8, 4] {
            if let quantized = Self.env["BASALT_E2E_JEV_BASE_\(bits)BIT"] { arms.append(Arm(label: "unmerged-\(bits)bit", base: URL(fileURLWithPath: quantized), float32: false)) }
        }
        var reference: ArmResult?
        for arm in arms {
            let result = try await log.step(arm.label) { try await Self.run(arm, model: model, items: items) }
            let gaps = result.probabilities.map { key, got in got.map { abs($0.value - (reference?.probabilities[key]?[$0.key] ?? $0.value)) }.max() ?? 0 }
            let agree = result.probabilities.filter { key, got in got.max { $0.value < $1.value }?.key == (reference ?? result).probabilities[key]?.max { $0.value < $1.value }?.key }.count
            let sorted = result.milliseconds.sorted()
            log.record("RESULT \(arm.label)", String(format: "accuracy %d/%d, max gap %.4f, top answer agrees %d/%d, active %d MB, peak %d MB, median %.1f ms, p95 %.1f ms",
                result.correct, items.count, gaps.max() ?? 0, agree, result.probabilities.count, result.activeMB, Memory.peakMemory >> 20,
                sorted[sorted.count / 2], sorted[min(sorted.count - 1, sorted.count * 95 / 100)]))
            if let reference, arm.label != "unmerged-4bit" {
                #expect(Double(agree) >= 0.9 * Double(result.probabilities.count), "\(arm.label): top answer agrees on \(agree) of \(result.probabilities.count)")
                #expect(result.correct + items.count / 50 >= reference.correct, "\(arm.label): \(result.correct) correct against \(reference.correct)")
            } else if reference == nil {
                reference = result
            }
        }
    }
}
