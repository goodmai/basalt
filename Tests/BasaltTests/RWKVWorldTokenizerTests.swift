import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("RWKV World tokenizer: Python-literal vocabulary, greedy longest byte match, lossless decode (#825)")
struct RWKVWorldTokenizerTests {
    static let vocabulary = """
        1 '\\x00' 1
        2 'a' 1
        3 'b' 1
        4 ' ' 1
        5 'ab' 2
        6 'abc' 3
        7 'c' 1
        8 b'\\xe2' 1
        9 b'\\x80' 1
        10 b'\\x94' 1
        11 '\\u2014' 3
        12 '\\n\\n' 2
        13 "it's" 4
        14 '\\'' 1
        15 'i' 1
        16 't' 1
        17 's' 1
        18 '̀' 2
        """

    @Test("Every literal form parses to the bytes Python would produce, and lengths are checked")
    func parsesLiterals() throws {
        let v = try RWKVWorldVocabulary(contents: Self.vocabulary)
        #expect(v.bytes(of: 1) == [0])
        #expect(v.bytes(of: 9) == [0x80])
        #expect(v.bytes(of: 11) == [0xE2, 0x80, 0x94])
        #expect(v.bytes(of: 12) == [10, 10])
        #expect(v.bytes(of: 13) == Array("it's".utf8))
        #expect(v.bytes(of: 14) == [39])
        #expect(v.bytes(of: 18) == [0xCC, 0x80])
        #expect(throws: RWKVWorldVocabulary.ParseError.malformed(line: 1)) { try RWKVWorldVocabulary(contents: "1 'ab' 1") }
    }

    @Test("Encoding takes the longest match at each position, prefers a whole multi-byte token, and decodes back losslessly")
    func greedyLongestMatch() throws {
        let v = try RWKVWorldVocabulary(contents: Self.vocabulary)
        #expect(v.encode("abcab abca") == [6, 5, 4, 6, 2])
        #expect(v.encode("a\u{2014}b") == [2, 11, 3])
        #expect(v.encode("it's\n\nits") == [13, 12, 15, 16, 17])
        for text in ["abcab abca", "a\u{2014}b", "it's\n\nits", ""] { #expect(v.decode(v.encode(text)) == text) }
    }

    static var env: [String: String] { ProcessInfo.processInfo.environment }

    @Test("On the real rwkv_vocab_v20230424.txt, token ids equal an independent reference encoding of the same text",
          .enabled(if: env["BASALT_E2E_RWKV_VOCAB"] != nil, "set BASALT_E2E_RWKV_VOCAB, BASALT_E2E_RWKV_TEXT and BASALT_E2E_RWKV_IDS"))
    func realVocabularyParity() throws {
        let v = try RWKVWorldVocabulary(contents: String(contentsOfFile: try #require(Self.env["BASALT_E2E_RWKV_VOCAB"]), encoding: .utf8))
        let text = try String(contentsOfFile: try #require(Self.env["BASALT_E2E_RWKV_TEXT"]), encoding: .utf8)
        let expected = try String(contentsOfFile: try #require(Self.env["BASALT_E2E_RWKV_IDS"]), encoding: .utf8).split(separator: ",").compactMap { Int($0) }
        #expect(v.encode(text) == expected)
        #expect(v.decode(expected) == text)
    }

    @Test("The loader picks the RWKV tokenizer only when the vocabulary file is present and tokenizer.json is not")
    func loaderSelection() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("rwkv-tok-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(try RWKVWorldTokenizer.load(from: directory) == nil)
        try Data(Self.vocabulary.utf8).write(to: directory.appending(component: RWKVWorldVocabulary.fileName))
        let tokenizer = try #require(try RWKVWorldTokenizer.load(from: directory))
        #expect(tokenizer.encode(text: "abc", addSpecialTokens: true) == [6])
        #expect(tokenizer.convertTokenToId(RWKVWorldTokenizer.endOfText) == 0 && tokenizer.eosToken == "\n\n")
        #expect(try tokenizer.applyChatTemplate(messages: [["role": "user", "content": "ab"]], tools: nil, additionalContext: nil)
            == tokenizer.encode(text: "User: ab\n\nAssistant:", addSpecialTokens: false))
        try Data("{}".utf8).write(to: directory.appending(component: "tokenizer.json"))
        #expect(try RWKVWorldTokenizer.load(from: directory) == nil)
    }
}

@Suite("force_bos_token_insert in config.json puts BOS first on encode and chat templates (xLSTM-7b, #843)")
struct BOSInsertingTokenizerTests {
    @Test("BOS is prepended once when special tokens are requested, never when they are not, and the wrapper is used only when the flag is true")
    func prependsBOS() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("bos-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(RWKVWorldTokenizerTests.vocabulary.utf8).write(to: directory.appending(component: RWKVWorldVocabulary.fileName))
        let base = try #require(try RWKVWorldTokenizer.load(from: directory))
        try Data(#"{"model_type": "xlstm"}"#.utf8).write(to: directory.appending(component: "config.json"))
        #expect(BOSInsertingTokenizer.wrapping(base, directory: directory).encode(text: "ab", addSpecialTokens: true) == [5])
        try Data(#"{"model_type": "xlstm", "force_bos_token_insert": true}"#.utf8).write(to: directory.appending(component: "config.json"))
        let wrapped = BOSInsertingTokenizer.wrapping(base, directory: directory)
        #expect(wrapped.encode(text: "ab", addSpecialTokens: true) == [0, 5])
        #expect(wrapped.encode(text: "ab", addSpecialTokens: false) == [5])
        let chat = try wrapped.applyChatTemplate(messages: [["role": "user", "content": "ab"]], tools: nil, additionalContext: nil)
        #expect(chat.first == 0 && chat.dropFirst().first != 0)
        #expect((wrapped as? BOSInsertingTokenizer)?.prefixed([0, 5]) == [0, 5])
    }
}
