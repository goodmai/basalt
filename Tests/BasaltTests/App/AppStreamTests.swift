import Foundation
import Testing
@testable import BasaltAppCore
@testable import BasaltCore
@testable import BasaltServer

@Suite("Mac app core: the SSE parser and the chat delta decoder read exactly what the server's chat stream encoder writes (#434 phase 1)")
struct AppStreamTests {
    static let wire = "event: note\ndata: one\ndata: two\n\n: keep-alive\r\ndata: {\"a\":1}\r\n\r\ndata: [DONE]\n\n"
    static let expected = [SSEEvent(name: "note", data: "one\ntwo"), SSEEvent(data: "{\"a\":1}"), SSEEvent(data: "[DONE]")]

    @Test("Events are the same wherever the bytes are split: one read, two reads at every offset, and byte by byte")
    func splits() {
        let bytes = Array(Self.wire.utf8)
        var whole = SSEParser()
        #expect(whole.feed(bytes) == Self.expected)
        for cut in 0 ... bytes.count {
            var parser = SSEParser()
            #expect(parser.feed(bytes[..<cut]) + parser.feed(bytes[cut...]) == Self.expected, "cut at \(cut)")
        }
        var single = SSEParser()
        #expect(bytes.flatMap { single.feed([$0]) } == Self.expected)
    }

    @Test("A frame without a blank line yet yields nothing, and an unknown field is ignored")
    func partial() {
        var parser = SSEParser()
        #expect(parser.feed(Array("id: 7\ndata: half".utf8)).isEmpty)
        #expect(parser.feed(Array("\n\n".utf8)) == [SSEEvent(data: "half")])
    }

    static func final(_ reason: GenerationResponse.FinishReason, prompt: Int, completion: Int) -> GenerationResponse {
        GenerationResponse(
            generatedText: "", promptTokens: prompt, completionTokens: completion, tokensPerSecond: 10, generationTime: 0.5, timeToFirstToken: 0.1,
            memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0), finishReason: reason)
    }

    static func deltas(_ frames: [String]) throws -> [ChatDelta] {
        var parser = SSEParser()
        var result: [ChatDelta] = []
        for event in parser.feed(Array(frames.joined().utf8)) {
            if case .deltas(let deltas) = try ChatDeltaDecoder.decode(event) { result += deltas }
        }
        return result
    }

    @Test("Reasoning, text, a tool call, the finish reason and usage decode from the server's own frames, then [DONE]")
    func serverFrames() throws {
        var stream = OpenAIChatStream(model: "org/model", includeUsage: true)
        var frames = stream.frames(for: .reasoning("thinking"))
        frames += stream.frames(for: .text("Hello"))
        frames += stream.frames(for: .text(" there"))
        frames += stream.frames(for: .toolCall(ToolCallOutput(id: "call_1", name: "get_weather", arguments: "{\"city\":\"Wien\"}")))
        frames += stream.frames(for: .metadata(Self.final(.stop, prompt: 12, completion: 5)))
        frames += stream.closing()
        #expect(try Self.deltas(frames) == [
            .reasoning("thinking"), .text("Hello"), .text(" there"),
            .toolCall(ToolCallOutput(id: "call_1", name: "get_weather", arguments: "{\"city\":\"Wien\"}")),
            .finish("tool_calls"), .usage(prompt: 12, completion: 5),
        ])
        var parser = SSEParser()
        let last = try #require(parser.feed(Array(frames.joined().utf8)).last)
        #expect(try ChatDeltaDecoder.decode(last) == .done)
    }

    @Test("The server's error frame is a refusal with its message; a frame that is not JSON is malformed")
    func failures() {
        var stream = OpenAIChatStream(model: "org/model", includeUsage: false)
        _ = stream.frames(for: .metadata(Self.final(.error, prompt: 1, completion: 0)))
        #expect(throws: EngineError.refused(status: 500, message: ModelsController.generationFailed)) { try Self.deltas(stream.closing()) }
        #expect(throws: EngineError.malformed("not json")) { try ChatDeltaDecoder.decode(SSEEvent(data: "not json")) }
    }

    @Test("A request carries the system prompt, the turns with text, stream and include_usage")
    func request() throws {
        var conversation = ConversationSnapshot(id: ConversationID(), model: "org/model", systemPrompt: "Be brief.")
        conversation.messages = [
            MessageSnapshot(id: MessageID(), role: .user, parts: [.text("Hi")]),
            MessageSnapshot(id: MessageID(), role: .assistant, parts: [.reasoning("hm"), .text("Hello")]),
            MessageSnapshot(id: MessageID(), role: .assistant, state: .streaming),
        ]
        let json = try #require(try JSONSerialization.jsonObject(with: JSONEncoder().encode(ChatRequest(conversation))) as? [String: Any])
        let messages = try #require(json["messages"] as? [[String: String]])
        #expect(messages == [["role": "system", "content": "Be brief."], ["role": "user", "content": "Hi"], ["role": "assistant", "content": "Hello"]])
        #expect(json["stream"] as? Bool == true && (json["stream_options"] as? [String: Bool])?["include_usage"] == true && json["model"] as? String == "org/model")
    }
}
