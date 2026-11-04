import Foundation
import Testing
import MLX
import MLXLMCommon

struct ByteTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer { ByteTokenizer() }
}

struct ReferenceLogits: Decodable {
    let tokens: [Int]
    let first: [Float]
    let last: [Float]
}

@Suite("Tiny random checkpoints load through the model factory and match mlx-lm's fp32 logits per family; Fixtures/ReferenceLogits/README.md regenerates them (#324)", .serialized)
struct ReferenceLogitsTests {
    static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("Fixtures/ReferenceLogits")

    static let tolerance: Float = 1e-4

    @Test("First and last position logits stay within 1e-4 of mlx-lm: ten times the Metal-vs-CPU fp32 noise, below the drift of a wrong eps, RoPE base or window",
          arguments: ["qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe", "gemma4", "gemma4_moe", "lfm2", "lfm2_moe", "nemotron_h", "nemotron_h_moe"])
    func matchesReference(family: String) async throws {
        let directory = Self.fixtures.appendingPathComponent(family)
        let reference = try JSONDecoder().decode(
            ReferenceLogits.self, from: Data(contentsOf: directory.appendingPathComponent("reference.json")))
        let container = try await loadModelContainer(from: directory, using: ByteTokenizerLoader())
        let tokens = reference.tokens
        let logits = try await container.perform { (context: ModelContext) throws -> [[Float]] in
            let inputs = MLXArray(tokens.map(Int32.init)).reshaped(1, -1)
            let output = context.model(inputs, cache: try context.model.newCache(parameters: nil)).asType(.float32)
            return [output[0, 0].asArray(Float.self), output[0, tokens.count - 1].asArray(Float.self)]
        }
        for (actual, expected) in zip(logits, [reference.first, reference.last]) {
            try #require(actual.count == expected.count)
            let drift = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
            #expect(drift <= Self.tolerance, "\(family): max |logit - mlx-lm| = \(drift)")
        }
    }
}
