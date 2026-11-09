import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Core AI embedding bundle: metadata, length buckets, bias and pooling inputs, Matryoshka truncation (#223)")
struct CoreAIEmbeddingBundleTests {
    static var metadata: [String: Any] { [
        "space": ["id": "google/embeddinggemma-2"],
        "dimension": 768,
        "matryoshka_dimensions": [768, 512, 256, 128],
        "components": [
            "token_table": ["asset": "token_table"],
            "encoder": [
                "asset": "encoder",
                "functions": [
                    ["function": "embeds_len512_b1", "batch": 1, "length": 512],
                    ["function": "embeds_len128_b1", "batch": 1, "length": 128],
                    ["function": "embeds_len1024_b1", "batch": 1, "length": 1024],
                    ["function": "embeds_len128_b4", "batch": 4, "length": 128],
                ],
                "bias": ["masked": -30000.0],
            ],
        ],
        "text": [
            "add_bos_token": true, "add_eos_token": true, "bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0,
            "truncation": ["max_tokens": 1024], "prompts": ["document": "title: none | text: "],
        ],
    ] }

    @Test("Batch-1 functions are kept in length order, and prompts and special tokens come from the metadata")
    func parses() throws {
        let bundle = try CoreAIEmbeddingBundle(embedding: Self.metadata, path: "/x")
        #expect(bundle.id == "google/embeddinggemma-2")
        #expect(bundle.functions.map(\.length) == [128, 512, 1024])
        #expect((bundle.bos, bundle.eos, bundle.pad, bundle.maxTokens) == (2, 1, 0, 1024))
        #expect(bundle.documentPrompt == "title: none | text: ")
        #expect(throws: BasaltError.self) { try CoreAIEmbeddingBundle(embedding: ["dimension": 768], path: "/x") }
    }

    @Test("Tokens get BOS and EOS, are cut to max_tokens with EOS kept, and pick the smallest bucket that fits")
    func bucketsAndTruncation() throws {
        let bundle = try CoreAIEmbeddingBundle(embedding: Self.metadata, path: "/x")
        #expect(bundle.tokens([7, 8]) == [2, 7, 8, 1])
        let long = bundle.tokens(Array(repeating: 5, count: 2000))
        #expect(long.count == 1024 && long.first == 2 && long.last == 1)
        #expect(bundle.function(for: 4).length == 128)
        #expect(bundle.function(for: 129).length == 512)
        #expect(bundle.function(for: 5000).length == 1024)
        #expect(bundle.padded([2, 7, 1], to: 5) == [2, 7, 1, 0, 0])
    }

    @Test("Bias masks padded keys for every query in the [1, key, 1, query] layout; pool weights average the real tokens")
    func maskAndPool() throws {
        let bundle = try CoreAIEmbeddingBundle(embedding: Self.metadata, path: "/x")
        #expect(bundle.bias(real: 2, length: 3) == [0, 0, 0, 0, 0, 0, -30000, -30000, -30000])
        #expect(bundle.poolWeights(real: 2, length: 4) == [0.5, 0.5, 0, 0])
    }

    @Test("Matryoshka truncation keeps the leading dimensions and renormalizes; other sizes are refused")
    func truncation() throws {
        let bundle = try CoreAIEmbeddingBundle(embedding: Self.metadata, path: "/x")
        let vector = [3, 4] + Array(repeating: Float(1), count: 766)
        #expect(try bundle.truncated(vector, to: nil) == vector)
        let head = try bundle.truncated(vector, to: 128)
        #expect(head.count == 128 && abs(head.reduce(0) { $0 + $1 * $1 } - 1) < 1e-5)
        #expect(throws: BasaltError.self) { try bundle.truncated(vector, to: 100) }
    }
}
