import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

enum BatchProbe {
    struct Report: CustomStringConvertible {
        var oneRow: Float = 0
        var companions: [String: Float] = [:]
        var batchedVsSolo: Float = 0
        var batchedVsSoloKL: Float = 0
        var paddedVsUnpadded: Float = 0

        var description: String {
            let others = companions.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: ", ")
            return "one row vs solo \(oneRow); unpadded companions vs [A,A]: \(others); batched vs solo \(batchedVsSolo) (KL \(batchedVsSoloKL)); A padded vs unpadded \(paddedVsUnpadded)"
        }
    }

    static func prefilled(_ model: any LanguageModel, _ tokens: MLXArray) throws -> [KVCache] {
        let layers = try model.newCache(parameters: nil)
        eval([model(tokens[.newAxis], cache: layers)] + layers.flatMap { $0.innerState() })
        return layers
    }

    static func joined(_ rows: [[KVCache]]) -> [KVCache] {
        let layers = rows[0].map(BatchCore.batchLayer)
        for (index, layer) in layers.enumerated() { BatchCore.join(rows.map { $0[index] }, into: layer) }
        return layers
    }

    static func row(_ model: any LanguageModel, _ inputs: [MLXArray], target: Int) throws -> MLXArray {
        let layers = joined(try inputs.map { try prefilled(model, $0) })
        let tokens = MLXArray(Array(repeating: Int32(9), count: inputs.count)).reshaped(-1, 1)
        let logits = model(tokens, cache: layers)[target, -1].asType(.float32)
        eval(logits)
        return logits
    }

    static func solo(_ model: any LanguageModel, _ tokens: MLXArray) throws -> MLXArray {
        let layers = try prefilled(model, tokens)
        let logits = model(MLXArray([Int32(9)]).reshaped(1, 1), cache: layers)[0, -1].asType(.float32)
        eval(logits)
        return logits
    }

    static func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float { abs(a - b).max().item(Float.self) }

    static func kl(_ p: MLXArray, _ q: MLXArray) -> Float {
        let lp = p - logSumExp(p, axis: -1, keepDims: true)
        let lq = q - logSumExp(q, axis: -1, keepDims: true)
        return (exp(lp) * (lp - lq)).sum().item(Float.self)
    }

    static func run(_ model: any LanguageModel, _ inputs: [MLXArray]) throws -> Report {
        let ordered = inputs.sorted { $0.dim(0) > $1.dim(0) }
        let (a, b, c, shortest) = (ordered[0], ordered[1], ordered[2], ordered[ordered.count - 1])
        let reference = try row(model, [a, a], target: 0)
        let alone = try solo(model, a)
        var report = Report()
        report.oneRow = maxAbs(try row(model, [a], target: 0), alone)
        report.companions = [
            "[A,A,A,A]": maxAbs(try row(model, [a, a, a, a], target: 0), reference),
            "[A,B]": maxAbs(try row(model, [a, b], target: 0), reference),
            "[C,B,A]": maxAbs(try row(model, [c, b, a], target: 2), reference),
        ]
        report.batchedVsSolo = maxAbs(reference, alone)
        report.batchedVsSoloKL = kl(alone, reference)
        report.paddedVsUnpadded = maxAbs(try row(model, [a, shortest], target: 1), try row(model, [shortest, shortest], target: 0))
        return report
    }
}

@Suite("Batched decode keeps a row's logits independent of companions that add no padding (#319, #255)", .serialized)
struct BatchedOrderTests {
    @Test("Tiny qwen35, lfm2, gemma4 and gemma4Moe: one row through the batched path equals solo, and unpadded companions of any count leave the row bit for bit",
          arguments: [TinyModel.Family.qwen35, .lfm2, .gemma4, .gemma4Moe])
    func tiny(family: TinyModel.Family) throws {
        let model = try TinyModel.make(family)
        let report = try BatchProbe.run(model, BatchFixture.prompts.map { MLXArray($0.map(Int32.init)) })
        #expect(report.oneRow == 0, "\(report)")
        #expect(report.companions.values.allSatisfy { $0 == 0 }, "\(report)")
        #expect(report.batchedVsSolo < 1e-4 && report.paddedVsUnpadded < 1e-4, "\(report)")
    }
}

@Suite("Batched greedy decode on real hybrid and sliding-window models: what moves a row's logits (#319, #255)", .serialized)
struct BatchedOrderRealModelTests {
    static var modelPaths: [String] {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1" else { return [] }
        return (env["BASALT_E2E_BATCH_MODELS"] ?? "").split(separator: ",").map(String.init)
    }

    static let prompts = [
        "Write a short story about a lighthouse keeper during a storm.",
        "List three facts about Mars.",
        "Explain in two sentences why the sky is blue, and then give one everyday example that shows the same effect at work.",
        "Name a prime number above 100.",
    ]

    @Test("One row equals solo and unpadded companions leave a row bit for bit; batched-vs-solo and padding differences are reported",
          .enabled(if: !modelPaths.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_BATCH_MODELS"), arguments: modelPaths)
    func realModel(path: String) async throws {
        let engine = MLXInferenceEngine()
        try await engine.load(modelPath: path)
        let container = try #require(await engine.container)
        var inputs: [MLXArray] = []
        for prompt in Self.prompts { inputs.append(try await container.prepare(input: UserInput(chat: [.user(prompt)])).text.tokens.reshaped(-1)) }
        let report = try await container.perform(nonSendable: inputs) { context, inputs -> BatchProbe.Report? in
            guard BatchedTokenIterator.batchableLayers(try context.model.newCache(parameters: nil)) != nil else { return nil }
            return try withError { try BatchProbe.run(context.model, inputs) }
        }
        print("[#319] \(path): \(report.map(String.init(describing:)) ?? "not batchable")")
        guard let report else { return }
        #expect(report.oneRow == 0, "\(report)")
        #expect(report.companions.values.allSatisfy { $0 == 0 }, "\(report)")
    }
}
