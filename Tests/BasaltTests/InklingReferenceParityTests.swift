import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltInference

@Suite("Inkling 2 P1 tiny parity: basalt matches a golden from mlx-lm #2086 on the inkling2Loop TinyModel weights (#650, #228)")
struct InklingReferenceParityTests {
    struct Golden: Decodable {
        let tokens: [Int32]
        let argmax: [Int32]
        let positions: [Int]
        let logits: [[Float]]
        let greedy16: [Int32]
    }

    static func golden() throws -> Golden {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/Inkling2/mlxlm2086_tiny_golden.json")
        return try JSONDecoder().decode(Golden.self, from: Data(contentsOf: url))
    }

    @Test("A full forward gives the reference argmax at all 40 positions and logits within 1e-3 at three of them")
    func prefillMatchesReference() throws {
        let golden = try Self.golden()
        let model = try TinyModel.make(.inkling2Loop)
        let logits = model(MLXArray(golden.tokens)[.newAxis], cache: try model.newCache(parameters: nil))[0].asType(.float32)
        #expect(argMax(logits, axis: -1).asArray(Int32.self) == golden.argmax)
        for (position, expected) in zip(golden.positions, golden.logits) {
            #expect(abs(logits[position] - MLXArray(expected)).max().item(Float.self) < 1e-3, "position \(position)")
        }
    }

    @Test("Greedy decoding through the caches gives the reference's 16 tokens exactly")
    func greedyDecodeMatchesReference() throws {
        let golden = try Self.golden()
        let model = try TinyModel.make(.inkling2Loop)
        let cache = try model.newCache(parameters: nil)
        var next = argMax(model(MLXArray(golden.tokens)[.newAxis], cache: cache)[0, -1], axis: -1).item(Int32.self)
        var generated = [next]
        while generated.count < golden.greedy16.count {
            next = argMax(model(MLXArray([next])[.newAxis], cache: cache)[0, -1], axis: -1).item(Int32.self)
            generated.append(next)
        }
        #expect(generated == golden.greedy16)
    }

    struct RealReference: Decodable {
        struct Record: Decodable {
            let name: String
            let ids: [Int32]
            let top5: [[Int32]]
        }
        let reference: String
        let records: [Record]
    }

    static var realModelPath: String? {
        let env = ProcessInfo.processInfo.environment
        return env["BASALT_E2E"] == "1" ? env["BASALT_E2E_INKLING_MODEL_PATH"] : nil
    }

    @Test("Real oQ3e model, 2,560 teacher-forced positions: top-1 agrees with mlx-lm #2086 at 97% or more (mlx-lm bf16 agrees with its own fp32 at 97.97%) and is always in its top-5",
          .enabled(if: realModelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_INKLING_MODEL_PATH to the Inkling-2-Mini-oQ3e-mtp snapshot"))
    func realModelTop1MatchesReference() async throws {
        let path = try #require(Self.realModelPath)
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/Inkling2/mlxlm2086_oq3e_top5.json")
        let reference = try JSONDecoder().decode(RealReference.self, from: Data(contentsOf: url))
        let log = try await ModelRunLog("inkling-real-parity", modelPath: path)
        let engine = MLXInferenceEngine()
        try await engine.load(modelPath: path)
        let container = try #require(await engine.container)
        var (agree, inTop5, total) = (0, 0, 0)
        for record in reference.records {
            let top1 = try await container.perform(nonSendable: record.ids) { context, ids in
                let logits = context.model(MLXArray(ids)[.newAxis], cache: try context.model.newCache(parameters: nil))[0]
                return argMax(logits, axis: -1).asArray(Int32.self)
            }
            let same = zip(top1, record.top5).filter { $0 == $1[0] }.count
            let near = zip(top1, record.top5).filter { $1.contains($0) }.count
            (agree, inTop5, total) = (agree + same, inTop5 + near, total + top1.count)
            log.record("result", "\(record.name) tokens=\(top1.count) top1_agree=\(same) in_ref_top5=\(near)")
        }
        let rate = Double(agree) / Double(max(total, 1))
        log.record("result", "reference=\(reference.reference) positions=\(total) top1_agree=\(String(format: "%.4f", rate)) in_ref_top5=\(inTop5)/\(total)")
        #expect(total == 2_560)
        #expect(rate >= 0.97)
        #expect(inTop5 == total)
    }
}
