import Foundation
import Hub
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Tokenizer regexes that HF tokenizers accept but ICU rejects compile after bare braces are escaped (#842)")
struct TokenizerRegexCompatibilityTests {
    static func compiles(_ pattern: String) -> Bool { (try? NSRegularExpression(pattern: pattern)) != nil }

    @Test("Huginn-0125's alternatives with bare { and } fail in ICU and compile once only the bare braces are escaped")
    func escapesBareBraces() {
        let huginn = #"\\[a-zA-Z]+|\\{|\\}|{|}|\[|\]|\p{L}+|\p{N}{1,1}|[{}]"#
        let fixed = TokenizerRegexCompatibility.compatible(huginn)
        #expect(!Self.compiles(huginn))
        #expect(fixed == #"\\[a-zA-Z]+|\\\{|\\\}|\{|\}|\[|\]|\p{L}+|\p{N}{1,1}|[{}]"#)
        #expect(Self.compiles(fixed))
    }

    @Test("A pattern ICU already compiles is returned unchanged, quantifiers and property escapes included",
          arguments: [#"'s|'t|'re| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#, "a{2,3}", #"\p{N}{1,3}"#, "x{2,}"])
    func leavesValidPatterns(pattern: String) {
        #expect(TokenizerRegexCompatibility.compatible(pattern) == pattern)
    }

    @Test("Only Regex fields under the pre-tokenizer and the other processing sections are rewritten; the vocabulary is not touched")
    func rewritesTokenizerData() {
        let split: Config = Config(["type": Config("Split"), "pattern": Config(["Regex": Config("a|{|}")])])
        let data = Config([
            "pre_tokenizer": Config(["type": Config("Sequence"), "pretokenizers": Config([split])]),
            "model": Config(["vocab": Config(["{": Config("1"), "Regex": Config("{|}")])]),
        ])
        let rewritten = MLXInferenceEngine.compatibleRegexes(in: data).dictionary()
        let pattern = rewritten?["pre_tokenizer"]?.dictionary()?["pretokenizers"]?.array()?.first?.dictionary()?["pattern"]
        #expect(pattern?.dictionary()?["Regex"]?.string() == #"a|\{|\}"#)
        #expect(rewritten?["model"]?.dictionary()?["vocab"]?.dictionary()?["Regex"]?.string() == "{|}")
    }
}
