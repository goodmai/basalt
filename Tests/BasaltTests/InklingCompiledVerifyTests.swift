import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Inkling 2 P2b: verify rounds of up to 8 tokens run the compiled step, match the eager graph and still rewind (#650, #228)", .serialized)
struct InklingCompiledVerifyTests {
    static let prompt = (0 ..< 12).map { Int32(($0 * 29 + 5) % 250) }
    static let rounds: [Int: [Int32]] = [1: [17], 4: [17, 200, 41, 9], 8: [17, 200, 41, 9, 133, 61, 2, 99], 9: [17, 200, 41, 9, 133, 61, 2, 99, 250]]

    static func run(eager: Bool, length: Int = 4, keep: Int?) throws -> (logits: MLXArray, next: MLXArray) {
        let round = try #require(rounds[length])
        let model = try TinyModel.make(.inkling2Loop)
        let cache = try model.newCache(parameters: nil)
        _ = model(MLXArray(prompt)[.newAxis], cache: cache)
        (model as Module).train(eager)
        let logits = model(MLXArray(round)[.newAxis], cache: cache)
        (model as Module).train(false)
        if let keep {
            for layer in cache { #expect(layer.trim(round.count - keep) == round.count - keep) }
        }
        let next = model(MLXArray([Int32(7)])[.newAxis], cache: cache)
        return (logits, next)
    }

    @Test("Rounds of 1, 4 and 8 tokens through the compiled step, and 9 tokens past it on the eager path, give the eager logits within 1e-4, with the same greedy ids", arguments: [1, 4, 8, 9])
    func compiledMatchesEager(length: Int) throws {
        let compiled = try Self.run(eager: false, length: length, keep: nil)
        let eager = try Self.run(eager: true, length: length, keep: nil)
        #expect(abs(compiled.logits - eager.logits).max().item(Float.self) < 1e-4)
        #expect(argMax(compiled.logits, axis: -1).asArray(Int32.self) == argMax(eager.logits, axis: -1).asArray(Int32.self))
        #expect(abs(compiled.next - eager.next).max().item(Float.self) < 1e-4)
    }

    @Test("A compiled 4-token round keeps its conv windows, so trimming it back to 1, 2 or 3 positions continues like the eager round trimmed the same way", arguments: [1, 2, 3])
    func compiledRoundRewinds(keep: Int) throws {
        let compiled = try Self.run(eager: false, keep: keep)
        let eager = try Self.run(eager: true, keep: keep)
        #expect(abs(compiled.next - eager.next).max().item(Float.self) < 1e-4)
    }
}
