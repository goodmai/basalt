import Testing
@testable import BasaltInference

@Suite("The stream splits into reasoning and answer at thinking markers, holding back a possible marker across chunks")
struct ThinkingSplitterTests {
    static func run(_ chunks: [String], startsThinking: Bool = false) -> [ThinkingSplitter.Segment] {
        var splitter = ThinkingSplitter(startsThinking: startsThinking)
        return chunks.flatMap { splitter.feed($0) } + splitter.feed("", final: true)
    }

    static func joined(_ segments: [ThinkingSplitter.Segment]) -> (reasoning: String, answer: String) {
        segments.reduce(into: ("", "")) { sum, segment in
            switch segment {
            case .reasoning(let text): sum.0 += text
            case .answer(let text): sum.1 += text
            }
        }
    }

    @Test("Plain text with no marker is all answer")
    func plainAnswer() {
        #expect(Self.joined(Self.run(["Hello, ", "world."])) == ("", "Hello, world."))
    }

    @Test("A think block becomes reasoning, and the newlines that open the answer after it are dropped")
    func thinkBlock() {
        #expect(Self.joined(Self.run(["<think>\nweigh it", " up</think>\n\n", "\nThe answer."])) == ("weigh it up", "The answer."))
    }

    @Test("A marker split across chunks is held back until it can be read whole")
    func splitMarker() {
        #expect(Self.joined(Self.run(["<thi", "nk>plan</th", "ink>done"])) == ("plan", "done"))
    }

    @Test("A prompt that ends inside thinking starts in reasoning; a stray closer in an answer is removed")
    func startsThinkingAndStrayCloser() {
        #expect(Self.joined(Self.run(["still planning</think>Yes"], startsThinking: true)) == ("still planning", "Yes"))
        #expect(Self.joined(Self.run(["Yes</think> indeed"])) == ("", "Yes indeed"))
    }

    @Test("Without the final flush, text that could start a marker stays buffered")
    func holdback() {
        var splitter = ThinkingSplitter(startsThinking: false)
        #expect(splitter.feed("ab<").isEmpty)
        #expect(splitter.feed("", final: true) == [.answer("ab<")])
    }
}
