import Foundation
import Testing
@testable import BasaltCore

@Suite("Token masks over a byte-level PDA (#42)")
struct GrammarMaskerTests {
    static let pieces = [
        "{", "}", "[", "]", ",", ":", "\"", " ", "\n", "a", "b", "x", "1", "2", "-", ".", "e", "\\", "n", "u", "0",
        "{\"", "\":", "\",", "\"}", "\"]", "\"b\":", "ab", "abc", "12", "1,", "1}", "true", "tr", "null", "\"a", "\"x",
        "},", "}]", "]}", " \"", "\n  ", "x\",\"", "é", "\u{1F600}",
    ]

    static func vocabulary(_ pieces: [String] = pieces, excluding excluded: Set<Int> = []) -> TokenVocabulary {
        TokenVocabulary(tokenBytes: pieces.enumerated().map { excluded.contains($0.offset) ? [] : Array($0.element.utf8) })
    }

    static func bruteForce(_ matcher: JSONGrammarMatcher, _ vocabulary: TokenVocabulary) -> TokenMask {
        var mask = TokenMask(size: vocabulary.size)
        for id in 0..<vocabulary.size where !vocabulary.bytes(of: id).isEmpty {
            var trial = matcher
            if trial.accept(vocabulary.bytes(of: id)) { mask.insert(id) }
        }
        return mask
    }

    static let schema = #"""
        {"type": "object", "properties": {
            "a": {"type": "string"}, "b": {"type": "array", "items": {"anyOf": [{"type": "integer"}, {"type": "null"}]}},
            "x": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}},
         "required": ["a", "x"]}
        """#

    @Test("At every state of a document the cached mask equals checking every token against the PDA")
    func maskMatchesBruteForceAlongADocument() throws {
        let vocabulary = Self.vocabulary()
        let masker = GrammarMasker(grammar: try grammar(Self.schema), vocabulary: vocabulary)
        let document = "{\"b\": [1, null,12],\n  \"x\":{\"a\":\"x\\n\u{1F600}é\"},\"a\":\"ab\"}"
        var matcher = JSONGrammarMatcher(grammar: masker.grammar)
        for byte in Array(document.utf8) {
            let expected = Self.bruteForce(matcher, vocabulary)
            #expect(masker.mask(for: matcher) == expected, "\(matcher.stack)")
            #expect(masker.mask(for: matcher) == expected)
            let step = matcher.advance(byte)
            #expect(step == .consumed)
        }
        #expect(matcher.isAccepting)
        #expect(masker.mask(for: matcher).isEmpty)
    }

    @Test("A token that closes a string is allowed by its context, not by the string frame it starts in")
    func closingTokenDependsOnTheStack() throws {
        let vocabulary = Self.vocabulary()
        let masker = GrammarMasker(grammar: try grammar(#"""
            {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}
            """#), vocabulary: vocabulary)
        let comma = try #require(Self.pieces.firstIndex(of: "\","))
        let brace = try #require(Self.pieces.firstIndex(of: "\"}"))
        var first = JSONGrammarMatcher(grammar: masker.grammar)
        let firstAccepted = first.accept(Array(#"{"a":"x"#.utf8))
        var last = JSONGrammarMatcher(grammar: masker.grammar)
        let lastAccepted = last.accept(Array(#"{"a":"x","b":"x"#.utf8))
        #expect(firstAccepted && lastAccepted)
        #expect(first.stack.last == last.stack.last)

        let inFirst = masker.mask(for: first)
        let inLast = masker.mask(for: last)
        #expect(inFirst.contains(comma) && !inFirst.contains(brace))
        #expect(!inLast.contains(comma) && inLast.contains(brace))
    }

    @Test("Excluded tokens are never allowed, even where their bytes fit")
    func excludedTokens() throws {
        let open = try #require(Self.pieces.firstIndex(of: "{"))
        let vocabulary = Self.vocabulary(excluding: [open])
        let masker = GrammarMasker(grammar: try grammar(Self.schema), vocabulary: vocabulary)
        let mask = masker.mask(for: JSONGrammarMatcher(grammar: masker.grammar))
        #expect(!mask.contains(open))
        #expect(mask.contains(try #require(Self.pieces.firstIndex(of: "{\""))))
    }

    @Test("tokenizer.json: byte-level BPE maps Ġ/Ċ to bytes, SentencePiece maps ▁ and <0x0A>, added tokens are excluded")
    func vocabularyFromTokenizerJSON() throws {
        let byteLevel = #"""
            {"model": {"type": "BPE", "vocab": {"{": 0, "Ġ\"": 1, "ĊĠĠ": 2, "Ã©": 3, "<|im_end|>": 4}},
             "added_tokens": [{"id": 4, "content": "<|im_end|>", "special": true}, {"id": 5, "content": "<think>", "special": false}],
             "decoder": {"type": "ByteLevel"}}
            """#
        let bpe = try TokenVocabulary.load(tokenizerJSON: Data(byteLevel.utf8))
        #expect(bpe.size == 6)
        #expect(bpe.bytes(of: 0) == Array("{".utf8))
        #expect(bpe.bytes(of: 1) == Array(" \"".utf8))
        #expect(bpe.bytes(of: 2) == Array("\n  ".utf8))
        #expect(bpe.bytes(of: 3) == Array("é".utf8))
        #expect(bpe.bytes(of: 4).isEmpty)
        #expect(bpe.bytes(of: 5).isEmpty)

        let sentencePiece = #"""
            {"model": {"type": "BPE", "vocab": {"▁{": 0, "<0x0A>": 1, "é": 2}},
             "added_tokens": [], "decoder": {"type": "Sequence", "decoders": [{"type": "Replace"}, {"type": "ByteFallback"}]}}
            """#
        let spm = try TokenVocabulary.load(tokenizerJSON: Data(sentencePiece.utf8))
        #expect(spm.bytes(of: 0) == Array(" {".utf8))
        #expect(spm.bytes(of: 1) == [0x0A])
        #expect(spm.bytes(of: 2) == Array("é".utf8))
    }
}
