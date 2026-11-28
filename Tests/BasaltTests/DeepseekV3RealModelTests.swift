import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

struct DeepseekV3Reference: Decodable {
    let tokens: [Int]
    let argmax: [Int]
    let top: [[Double]]
    let targetLogprobs: [Double]
    let greedy: [Int]
}

@Suite("deepseek_v3 on real weights against an mlx-lm reference, within the band bfloat16 rounding order allows: mlx-lm in float32 against its own bfloat16 run agrees at 87 of 90 positions with a top-10 gap of 0.56 (#893)", .serialized)
struct DeepseekV3RealModelTests {
    static var env: [String: String] { ProcessInfo.processInfo.environment }
    static var enabled: Bool {
        env["BASALT_E2E"] == "1" && env["BASALT_E2E_DEEPSEEK_V3_MODEL_PATH"] != nil && env["BASALT_E2E_DEEPSEEK_V3_REFERENCE"] != nil
    }

    static func row(_ tokens: [Int]) -> MLXArray { MLXArray(tokens.map(Int32.init))[.newAxis] }

    @Test("The forward over the reference tokens gives the reference argmax at 95% of positions or more, its loss per token within 0.01, its top-10 log-probabilities within 0.6 and all of its greedy tokens",
          .enabled(if: enabled, "set BASALT_E2E=1, BASALT_E2E_DEEPSEEK_V3_MODEL_PATH and BASALT_E2E_DEEPSEEK_V3_REFERENCE (JSON written by an mlx-lm run)"))
    func matchesTheReference() async throws {
        let path = try #require(Self.env["BASALT_E2E_DEEPSEEK_V3_MODEL_PATH"])
        let data = try Data(contentsOf: URL(fileURLWithPath: try #require(Self.env["BASALT_E2E_DEEPSEEK_V3_REFERENCE"])))
        let reference = try JSONDecoder().decode(DeepseekV3Reference.self, from: data)
        let log = try await ModelRunLog("deepseek-v3-reference", modelPath: path)
        defer { log.releaseGPU() }
        await MLXInferenceEngine.registerAliases()
        let container = try await loadModelContainer(from: URL(fileURLWithPath: path), using: DefaultTokenizerLoader())
        let line = await container.perform { context -> String in
            let logits = context.model(Self.row(reference.tokens), cache: nil)[0].asType(.float32)
            let logp = logits - logSumExp(logits, axis: -1, keepDims: true)
            let argmax = argMax(logits, axis: -1).asArray(Int32.self).map(Int.init)
            let agree = zip(argmax, reference.argmax).count(where: { $0 == $1 })
            let targets = MLXArray(reference.tokens.dropFirst().map(Int32.init))[0..., .newAxis]
            let nll = -Double(takeAlong(logp[0 ..< (reference.tokens.count - 1)], targets, axis: -1).mean().item(Float.self))
            let referenceNLL = -reference.targetLogprobs.reduce(0, +) / Double(reference.targetLogprobs.count)
            let last = logp[reference.tokens.count - 1]
            let gaps = reference.top.map { abs(Double(last[Int($0[0])].item(Float.self)) - $0[1]) }
            var sequence = reference.tokens
            for _ in reference.greedy {
                sequence.append(Int(argMax(context.model(Self.row(sequence), cache: nil)[0, -1]).item(Int32.self)))
            }
            let greedy = Array(sequence.dropFirst(reference.tokens.count))
            let same = zip(greedy, reference.greedy).prefix(while: { $0 == $1 }).count
            #expect(Double(agree) >= 0.95 * Double(reference.argmax.count), "argmax agrees at \(agree) of \(reference.argmax.count) positions")
            #expect((gaps.max() ?? 1) < 0.6, "largest top-10 log-probability gap \(gaps.max() ?? 1)")
            #expect(abs(nll - referenceNLL) < 0.01, "nll per token \(nll) against \(referenceNLL)")
            #expect(greedy == reference.greedy, "greedy tokens equal for the first \(same) of \(reference.greedy.count)")
            return String(
                format: "tokens %d, argmax agree %d/%d, nll/token %.4f vs reference %.4f, top-10 max gap %.4f, greedy equal for %d of %d",
                reference.tokens.count, agree, reference.argmax.count, nll, referenceNLL, gaps.max() ?? -1, same, reference.greedy.count)
        }
        log.record("result", line)
    }
}
