import Testing
import MLX
import BasaltCore
@testable import BasaltInference
@testable import BasaltHarness

enum ChunkSplits {
    static let otherFamilies = #"<tool_call>{"name":"f"}</tool_call> [TOOL_CALLS][{}] <|tool_call_start|>[f()]<|tool_call_end|> <|python_tag|>f()<|eom_id|>"#

    static func of(_ text: String) -> [[String]] {
        let scalars = Array(text.unicodeScalars)
        let interior = Array(1..<max(scalars.count, 1))
        var generator = SplitMix64(seed: 0x5EED)
        let random = (0..<32).map { _ in interior.filter { _ in Int.random(in: 0..<4, using: &generator) == 0 } }
        return (interior.map { [$0] } + random + [interior]).map { cuts in
            let bounds = [0] + cuts + [scalars.count]
            return zip(bounds, bounds.dropFirst()).map { String(String.UnicodeScalarView(scalars[$0..<$1])) }
        }
    }
}

@Suite("ThinkingSplitter gives the same segments however the stream is chunked")
struct ThinkingSplitterBoundaryTests {
    struct Case: Sendable, CustomTestStringConvertible {
        let text: String
        var startsThinking = false
        let expected: [ThinkingSplitter.Segment]
        var testDescription: String { text.debugDescription }
    }

    static let corpus: [Case] = [
        Case(text: "<think>\nweigh it up</think>\n\nThe answer.", expected: [.reasoning("weigh it up"), .answer("The answer.")]),
        Case(text: "<|channel>thought\nplan e\u{301} 🇺🇸🇩🇪 👩‍👩‍👧<channel|>Final 答案",
             expected: [.reasoning("plan e\u{301} 🇺🇸🇩🇪 👩‍👩‍👧"), .answer("Final 答案")]),
        Case(text: "<|channel|>thought plan<|channel|>final<|turn|>", expected: [.reasoning(" plan"), .answer("final<|turn|>")]),
        Case(text: "[THINK]x 🧑🏽‍💻[/THINK]\ny", expected: [.reasoning("x 🧑🏽‍💻"), .answer("y")]),
        Case(text: "still planning</think>Yes", startsThinking: true, expected: [.reasoning("still planning"), .answer("Yes")]),
        Case(text: "Yes</think> indeed", expected: [.answer("Yes indeed")]),
        Case(text: "<think>\n</think>\n\n\nAnswer", expected: [.answer("Answer")]),
        Case(text: "a<thi", expected: [.answer("a<thi")]),
        Case(text: "<think>\n", expected: []),
        Case(text: "Ünïcödé 🧑🏽‍💻 <think>नमस्ते</think>\r\nok", expected: [.answer("Ünïcödé 🧑🏽‍💻 "), .reasoning("नमस्ते"), .answer("ok")]),
        Case(text: "<think>x<|turn|>y", expected: [.reasoning("x"), .answer("y")]),
        Case(text: "<think></think><think>again</think>done", expected: [.reasoning("again"), .answer("done")]),
        Case(text: "plan\n<|channel|>\nanswer <thi", startsThinking: true, expected: [.reasoning("plan\n"), .answer("answer <thi")]),
        Case(text: "<think>\n\nreason\n</think>\nanswer\n", expected: [.reasoning("\nreason\n"), .answer("answer\n")]),
        Case(text: "<think>" + ChunkSplits.otherFamilies + "</think>ok", expected: [.reasoning(ChunkSplits.otherFamilies), .answer("ok")]),
    ]

    static func merged(_ chunks: [String], startsThinking: Bool) -> [ThinkingSplitter.Segment] {
        ThinkingSplitterTests.run(chunks, startsThinking: startsThinking).reduce(into: []) { merged, segment in
            switch (segment, merged.last) {
            case (.reasoning(let text), .reasoning(let held)?): merged[merged.count - 1] = .reasoning(held + text)
            case (.answer(let text), .answer(let held)?): merged[merged.count - 1] = .answer(held + text)
            default: merged.append(segment)
            }
        }
    }

    @Test("Every split of the stream matches the whole feed and the reference", arguments: corpus)
    func splitMatchesWhole(_ example: Case) {
        #expect(Self.merged([example.text], startsThinking: example.startsThinking) == example.expected)
        let diverging = ChunkSplits.of(example.text).first {
            Self.merged($0, startsThinking: example.startsThinking) != example.expected
        }
        #expect(diverging == nil)
    }

    @Test("An opener that ends a chunk still drops the newline that starts the next chunk")
    func newlineAfterOpenerInNextChunk() {
        #expect(Self.merged(["<think>", "\nweigh it up</think>", "\n", "The answer."], startsThinking: false)
            == [.reasoning("weigh it up"), .answer("The answer.")])
    }
}

@Suite("NemotronToolCallScanner gives the same text and calls however the stream is chunked")
struct NemotronScannerBoundaryTests {
    enum Piece: Equatable, Sendable {
        case text(String)
        case call(String, String)
    }

    struct Case: Sendable, CustomTestStringConvertible {
        let text: String
        let expected: [Piece]
        var testDescription: String { text.debugDescription }
    }

    static let corpus: [Case] = [
        Case(text: #"Sure.<TOOLCALL>[{"name":"get_weather","arguments":{"city":"Zürich 🌧"}}]</TOOLCALL>"#,
             expected: [.text("Sure."), .call("get_weather", #"{"city":"Zürich 🌧"}"#)]),
        Case(text: #"a<TOOLCALL>[{"name":"x","arguments":{}}]</TOOLCALL> mid é <TOOLCALL>{"name":"y","arguments":"{\"k\":1}"}</TOOLCALL>end"#,
             expected: [.text("a"), .call("x", "{}"), .text(" mid é "), .call("y", #"{"k":1}"#), .text("end")]),
        Case(text: "<TOOLCALL>not json</TOOLCALL> after", expected: [.text("<TOOLCALL>not json</TOOLCALL> after")]),
        Case(text: #"pre <TOOLCALL>[{"name":"open","arguments":{"q":"a</TOOL b"}}]"#,
             expected: [.text("pre "), .call("open", #"{"q":"a</TOOL b"}"#)]),
        Case(text: "tail <TOOL", expected: [.text("tail <TOOL")]),
        Case(text: "<<TOOLCALL>[]</TOOLCALL><", expected: [.text("<<TOOLCALL>[]</TOOLCALL><")]),
        Case(text: "plain 日本語 👩‍👩‍👧 e\u{301} text", expected: [.text("plain 日本語 👩‍👩‍👧 e\u{301} text")]),
        Case(text: ChunkSplits.otherFamilies, expected: [.text(ChunkSplits.otherFamilies)]),
    ]

    static func run(_ chunks: [String]) -> [Piece] {
        var scanner = NemotronToolCallScanner()
        return (chunks.flatMap { scanner.feed($0) } + scanner.finish()).reduce(into: []) { pieces, chunk in
            switch (chunk, pieces.last) {
            case (.text(let text), .text(let held)?): pieces[pieces.count - 1] = .text(held + text)
            case (.text(let text), _): pieces.append(.text(text))
            case (.toolCall(let call), _): pieces.append(.call(call.name, call.arguments))
            default: Issue.record("unexpected \(chunk)")
            }
        }
    }

    @Test("Every split of the stream matches the whole feed and the reference", arguments: corpus)
    func splitMatchesWhole(_ example: Case) {
        #expect(Self.run([example.text]) == example.expected)
        let diverging = ChunkSplits.of(example.text).first { Self.run($0) != example.expected }
        #expect(diverging == nil)
    }
}

@Suite("RepetitionGuard's verdict after a chunk depends only on the text so far")
struct RepetitionGuardBoundaryTests {
    static let corpus = [
        String(repeating: "the cat sat on the mat. ", count: 21),
        "Preamble that is not cyclic at all, 1 2 3 4 5. " + String(repeating: "日本語のテキスト🙂 e\u{301} ", count: 38),
        (0..<120).map { "n\($0) " }.joined(),
        String(repeating: "ab", count: 250),
        String(repeating: "🇺🇸 cycle text! ", count: 34),
    ]
    static let guards = [RepetitionGuard(), RepetitionGuard(probeSize: 8, threshold: 3, minLength: 40, retain: 100)]

    static func agreesWithWholePrefix(_ chunks: [String], prototype: RepetitionGuard) -> Bool {
        var streamed = prototype
        var prefix = ""
        return chunks.allSatisfy { chunk in
            prefix += chunk
            var whole = prototype
            return streamed.shouldStop(after: chunk) == whole.shouldStop(after: prefix) && streamed.score == whole.score
        }
    }

    @Test("Every split agrees, chunk by chunk, with a fresh guard fed the whole prefix", arguments: corpus, guards)
    func splitMatchesWhole(_ text: String, prototype: RepetitionGuard) {
        let diverging = ChunkSplits.of(text).first { !Self.agreesWithWholePrefix($0, prototype: prototype) }
        #expect(diverging == nil)
    }
}

@Suite("Constrained decoding leaves reasoning on exactly the token that completes a closer")
struct ConstraintPhaseBoundaryTests {
    static let reasoning = ["", "plan", "</thi", "<|turn|", String(repeating: "weigh each option, step by step; ", count: 3)]

    @Test("The phase flips on the closer's last byte and not before", arguments: MLXInferenceEngine.thinkingClosers, reasoning)
    func flipsOnTheLastByte(closer: String, before: String) throws {
        var processor = try JSONConstraintProcessorTests.processor(.reasoning, markers: false)
        let bytes = Array((before + closer).utf8)
        let flips = bytes.map { byte in
            processor.didSample(token: MLXArray(Int32(byte)))
            return processor.phase == .constrained
        }
        #expect(flips == bytes.indices.map { $0 == bytes.count - 1 })
    }
}
