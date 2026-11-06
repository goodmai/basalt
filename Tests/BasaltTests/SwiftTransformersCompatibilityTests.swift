import Foundation
import Testing
import Tokenizers
@testable import BasaltCore
@testable import BasaltInference

@Suite("swift-transformers 1.4 behaviour basalt relies on: invalid tokenizer regexes refuse the load, BPE decoding keeps spaces")
struct SwiftTransformersCompatibilityTests {
    static let vocabulary = #""vocab": {"x": 0, "▁!=": 1, "▁y": 2, "▁": 3, "!": 4, "=": 5, "!=": 6, "y": 7}, "merges": ["! =", "▁ !=", "▁ y"]"#
    static let decoder = #""decoder": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": true, "prepend_scheme": "always"}"#

    static func directory(config: String, tokenizer: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appending(path: "st14-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(config.utf8).write(to: dir.appendingPathComponent("tokenizer_config.json"))
        try Data(tokenizer.utf8).write(to: dir.appendingPathComponent("tokenizer.json"))
        return dir
    }

    @Test("A Split pre-tokenizer regex Foundation cannot compile refuses the tokenizer, and the load reports the model configuration with the pattern, not corrupted weights")
    func invalidRegexRefusesTheLoad() async throws {
        let tokenizer = #"{"model": {"type": "BPE", \#(Self.vocabulary)}, "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "(?<"}, "behavior": "Isolated", "invert": false}, "added_tokens": []}"#
        let dir = try Self.directory(config: #"{"tokenizer_class": "LlamaTokenizerFast", "legacy": false}"#, tokenizer: tokenizer)
        defer { try? FileManager.default.removeItem(at: dir) }
        let error = await #expect(throws: Tokenizers.TokenizerError.self) { _ = try await MLXInferenceEngine.upstreamTokenizer(at: dir) }
        let failure = await MLXInferenceEngine.loadFailure(try #require(error), modelPath: dir.path)
        guard case .modelConfigurationInvalid(let path, let reason) = failure else {
            Issue.record("expected modelConfigurationInvalid, got \(failure)")
            return
        }
        #expect(path == dir.path)
        #expect(reason.hasPrefix("tokenizer: ") && reason.contains("(?<"))
    }

    @Test("A BPE tokenizer decodes x != y with its spaces whether clean_up_tokenization_spaces is absent or true, as Gemma 4's tokenizer omits it", arguments: ["", #", "clean_up_tokenization_spaces": true"#])
    func bpeDecodingKeepsSpaces(cleanup: String) async throws {
        let tokenizer = #"{"model": {"type": "BPE", \#(Self.vocabulary)}, \#(Self.decoder), "added_tokens": []}"#
        let dir = try Self.directory(config: #"{"tokenizer_class": "LlamaTokenizerFast", "legacy": false\#(cleanup)}"#, tokenizer: tokenizer)
        defer { try? FileManager.default.removeItem(at: dir) }
        let loaded = try await MLXInferenceEngine.upstreamTokenizer(at: dir)
        #expect(loaded.decode(tokens: [0, 1, 2], skipSpecialTokens: true) == "x != y")
    }
}
