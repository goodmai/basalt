import Foundation
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

    @Test("Calibrated probabilities match the fp32 reference within 1e-3, one prefix evaluation per request, latency logged",
          .enabled(if: model != nil && reference != nil && env["BASALT_E2E_JEV_BASE"] != nil, "set BASALT_E2E=1, BASALT_E2E_JEV_MODEL (OpenJev dir), BASALT_E2E_JEV_BASE (its Qwen3 base dir) and BASALT_E2E_JEV_REFERENCE"))
    func matchesReference() async throws {
        let directory = URL(fileURLWithPath: try #require(Self.model))
        let base = URL(fileURLWithPath: try #require(Self.env["BASALT_E2E_JEV_BASE"]))
        let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: URL(fileURLWithPath: try #require(Self.reference))))
        let evaluator = try await JevEvaluator(directory: directory, base: base)
        var worst = 0.0
        for item in cases {
            let start = ContinuousClock.now
            let response = try await evaluator.decide(SystemOneRequest(state: item.state, questions: item.questions))
            let elapsed = ContinuousClock.now - start
            #expect(response.usage.prefixEvaluations == 1)
            for (name, expected) in item.answers {
                let got = try #require(response.answers[name])
                let probabilities = got.probabilities ?? got.noul.map { ["false": 1 - $0, "true": $0] } ?? [:]
                worst = max(worst, expected.probabilities.map { abs((probabilities[$0.key] ?? -1) - $0.value) }.max() ?? 0)
            }
            let ms = Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15
            print("RESULT jev \(evaluator.modelID): \(item.questions.count) questions, \(response.usage.candidateBranches) branches, \(response.usage.prefixTokens) prefix tokens, \(String(format: "%.1f", ms)) ms")
        }
        print("RESULT jev worst calibrated probability gap to the fp32 reference \(worst)")
        #expect(worst < 1e-3, "worst probability gap \(worst)")
    }
}
