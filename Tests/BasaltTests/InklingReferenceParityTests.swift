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
}
