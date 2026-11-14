import Foundation
import HTTPTypes
import Hummingbird
import Testing
@testable import BasaltCore
@testable import BasaltServer

@Suite("SSEFastWriter ByteBuffer JSON framing and RFC 8259 escaping (#490)")
struct SSEFastWriterTests {
    static let textCases: [String] = [
        "",
        "hello world",
        "ASCII 0123456789 !@#$%^&*()_+-=[]{}|;:',.<>/?~`",
        "Quotes \"inside\" and \"\"adjacent\"\" and trailing \"",
        "Backslashes \\ single and \\\\ double and \\\" quote-backslash \\\\",
        "Whitespace \n newline \r carriage \t tab \u{08} backspace \u{0C} formfeed",
        "Multi-byte UTF-8: café, naïve, Київ, 日本語, 한국어, العربية",
        "Emoji & ZWJ: 🚀🔥✨ 👨‍👩‍👧‍👦 🏳️‍🌈 👩🏽‍🔬 𝕳𝖊𝖑𝖑𝖔",
        String((0...0x1F).map { Character(UnicodeScalar($0)!) }),
        "Mixed \u{00}null\u{01}soh\u{07}bel\u{0B}vt\u{0E}so\u{1B}esc\u{1F}us \"quote\" \\slash\\ 🚀\n\r\t end",
        String(repeating: "token-\"α\"-\\-🚀-\n", count: 128),
    ]

    private func decodeChunk(_ buffer: ByteBuffer) throws -> ChatCompletionChunk {
        let raw = String(decoding: buffer.readableBytesView, as: UTF8.self)
        #expect(raw.hasPrefix("data: "))
        #expect(raw.hasSuffix("\n\n"))
        let jsonText = String(raw.dropFirst(6).dropLast(2))
        return try JSONDecoder().decode(ChatCompletionChunk.self, from: Data(jsonText.utf8))
    }

    private func rawJSONObject(_ buffer: ByteBuffer) throws -> [String: Any] {
        let raw = String(decoding: buffer.readableBytesView, as: UTF8.self)
        let jsonText = String(raw.dropFirst(6).dropLast(2))
        let parsed = try JSONSerialization.jsonObject(with: Data(jsonText.utf8))
        return try #require(parsed as? [String: Any])
    }

    @Test("Text deltas round-trip through JSONDecoder to exact ChatCompletionChunk across ASCII, UTF-8, emoji, quotes, backslashes, and control bytes",
          arguments: textCases)
    func textDeltaRoundTrip(sample: String) throws {
        let writer = SSEFastWriter(id: "chatcmpl-test-\"1\"\\a", created: 1_763_120_000, model: "org/model-\"v1\"")
        let frame = writer.textDelta(sample)
        let decoded = try decodeChunk(frame)
        let expected = ChatCompletionChunk(
            id: "chatcmpl-test-\"1\"\\a",
            created: 1_763_120_000,
            model: "org/model-\"v1\"",
            choices: [
                ChatCompletionChunk.Choice(
                    index: 0,
                    delta: ChatCompletionChunk.Delta(role: "assistant", content: sample),
                    finishReason: nil
                )
            ],
            usage: nil
        )
        #expect(decoded == expected)
        let object = try rawJSONObject(frame)
        #expect(object["usage"] == nil)
        let choices = try #require(object["choices"] as? [[String: Any]])
        #expect(choices.first?["finish_reason"] is NSNull)
    }

    @Test("Every individual control byte U+0000 through U+001F escapes cleanly and decodes back byte-for-byte")
    func allControlCharactersIndividually() throws {
        let writer = SSEFastWriter(id: "chatcmpl-ctrl", created: 42, model: "local")
        for codePoint in 0...0x1F {
            let scalar = try #require(UnicodeScalar(codePoint))
            let token = "pre-\(Character(scalar))-post"
            let decoded = try decodeChunk(writer.textDelta(token))
            #expect(decoded.choices.first?.delta.content == token)
        }
    }

    @Test("Reasoning deltas emit both reasoning and reasoning_content with nil finish_reason and omitted usage",
          arguments: textCases)
    func reasoningDeltaRoundTrip(sample: String) throws {
        let writer = SSEFastWriter(id: "chatcmpl-r", created: 99, model: "qwen/qwen3")
        let frame = writer.reasoningDelta(sample)
        let decoded = try decodeChunk(frame)
        let expected = ChatCompletionChunk(
            id: "chatcmpl-r",
            created: 99,
            model: "qwen/qwen3",
            choices: [
                ChatCompletionChunk.Choice(
                    index: 0,
                    delta: ChatCompletionChunk.Delta(role: "assistant", content: nil, reasoning: sample, reasoningContent: sample),
                    finishReason: nil
                )
            ]
        )
        #expect(decoded == expected)
        let object = try rawJSONObject(frame)
        #expect(object["usage"] == nil)
        let choices = try #require(object["choices"] as? [[String: Any]])
        #expect(choices.first?["finish_reason"] is NSNull)
    }

    @Test("Finish, tool_calls, usage, generic encodeFrame, done, and error frames match OpenAI chunk wire schema",
          arguments: ["stop", "length", "tool_calls"])
    func finishUsageAndToolCallFrames(reason: String) throws {
        let writer = SSEFastWriter(id: "chatcmpl-fin", created: -123, model: "m")
        let finishFrame = writer.finish(reason)
        let finishDecoded = try decodeChunk(finishFrame)
        #expect(finishDecoded == ChatCompletionChunk(
            id: "chatcmpl-fin",
            created: -123,
            model: "m",
            choices: [ChatCompletionChunk.Choice(index: 0, delta: ChatCompletionChunk.Delta(), finishReason: reason)]
        ))
        #expect(try rawJSONObject(finishFrame)["usage"] == nil)

        let call = ToolCallOutput(id: "call_\"1\"", name: "fn_\\x", arguments: "{\"q\":\"a\\nb🚀\"}")
        let toolDecoded = try decodeChunk(writer.toolCallDelta(index: 2, call: call))
        let expectedTool = ChatCompletionChunk.ToolCall(
            index: 2, id: call.id, type: "function",
            function: ChatCompletionChunk.ToolCall.Function(name: call.name, arguments: call.arguments)
        )
        #expect(toolDecoded == ChatCompletionChunk(
            id: "chatcmpl-fin",
            created: -123,
            model: "m",
            choices: [ChatCompletionChunk.Choice(
                index: 0,
                delta: ChatCompletionChunk.Delta(role: "assistant", toolCalls: [expectedTool]),
                finishReason: nil
            )]
        ))

        let response = GenerationResponse(
            generatedText: "ok", reasoningText: "r", promptTokens: 15, cachedPromptTokens: 6, completionTokens: 9,
            tokensPerSecond: 10, generationTime: 1, timeToFirstToken: 0.1,
            memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0), finishReason: .stop, reasoningTokens: 4
        )
        let usageDecoded = try decodeChunk(writer.usage(response))
        let expectedUsage = ChatCompletionChunk(
            id: "chatcmpl-fin",
            created: -123,
            model: "m",
            choices: [],
            usage: ChatCompletionChunk.Usage(promptTokens: 15, completionTokens: 9, totalTokens: 24, cachedTokens: 6, reasoningTokens: 4)
        )
        #expect(usageDecoded == expectedUsage)
        #expect(String(decoding: SSEFastWriter.doneFrame().readableBytesView, as: UTF8.self) == "data: [DONE]\n\n")
        let errText = String(decoding: SSEFastWriter.errorFrame(message: "bad \"state\"\n").readableBytesView, as: UTF8.self)
        #expect(errText == "event: error\ndata: {\"error\":{\"message\":\"bad \\\"state\\\"\\n\",\"type\":\"server_error\"}}\n\n")
    }

    @Test("OpenAIChatStream live Response streams the exact same SSE frames as frames(for:) and closing()")
    func openAIChatStreamResponseParity() async throws {
        let meta = GenerationResponse(
            generatedText: "Hi!", reasoningText: "hm", promptTokens: 8, cachedPromptTokens: 3, completionTokens: 2,
            tokensPerSecond: 1, generationTime: 1, timeToFirstToken: 0,
            memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0), finishReason: .stop, reasoningTokens: 1
        )
        let inputs: [StreamChunk] = [.reasoning("hm\n"), .text("Hi"), .text(" \"!\""), .metadata(meta)]
        var direct = OpenAIChatStream(model: "local-m", includeUsage: true, id: "chatcmpl-fixed", created: 777)
        let expected = inputs.flatMap { direct.frames(for: $0) } + direct.closing()

        let stream = AsyncStream<StreamChunk> { continuation in
            for item in inputs { continuation.yield(item) }
            continuation.finish()
        }
        let response = OpenAIChatStream(model: "local-m", includeUsage: true, id: "chatcmpl-fixed", created: 777)
            .response(streaming: stream)
        let collector = ByteBufferCollector()
        try await response.body.write(collector)
        #expect(String(decoding: collector.collected.readableBytesView, as: UTF8.self) == expected.joined())
    }
}

private final class ByteBufferCollector: ResponseBodyWriter {
    var collected = ByteBuffer()

    func write(_ buffer: ByteBuffer) async throws {
        var copy = buffer
        collected.writeBuffer(&copy)
    }

    func finish(_ trailingHeaders: HTTPFields?) async throws {}
}

@Suite("JSONSchemaGate, OrderedJSON, and JSONNesting over Span<UInt8> (#490)")
struct SpanSchemaGateTests {
    static let validSchemas: [String] = [
        #"{"type": "object"}"#,
        #"{"type": "array"}"#,
        #"{"type": "object", "properties": {"z": {"type": "string", "description": "🚀 \"ok\" \\n"}, "a": {"type": "integer", "minimum": -10}}, "required": ["z", "a"]}"#,
        ##"{"$defs": {"item": {"type": "object", "properties": {"name": {"type": "string", "pattern": "^[a-z]+$"}}}}, "$ref": "#/$defs/item"}"##,
        #"{"type": ["string", "null"], "enum": ["alpha", "beta\u000a", "\uD83D\uDE80", null]}"#,
        #"{"type": "integer", "multipleOf": 4, "description": "lenient keyword"}"#,
        "\u{FEFF}" + #"{"type": "object", "properties": {"ok": {"type": "boolean"}}}"#,
    ]

    static let invalidPayloads: [[UInt8]] = [
        [],
        Array("   \n\t  ".utf8),
        Array("{".utf8),
        Array("}".utf8),
        Array(#"{"type": "object",}"#.utf8),
        Array(#"{,"type": "object"}"#.utf8),
        Array(#"{"type": [1, 2,]}"#.utf8),
        Array(#"{"type": [,1]}"#.utf8),
        Array(#"{"type": "object"} 123"#.utf8),
        Array(#"{"type": "object"}{}"#.utf8),
        Array(#"{"type": 01}"#.utf8),
        Array(#"{"type": -}"#.utf8),
        Array(#"{"type": 1.}"#.utf8),
        Array(#"{"type": 1e+}"#.utf8),
        Array(#"{"type": 1e500}"#.utf8),
        Array(#"{"type": tru}"#.utf8),
        Array(#"{"type": truefalse}"#.utf8),
        Array("{\"type\": \"unterminated".utf8),
        Array("{\"type\": \"bad\\xescape\"}".utf8),
        Array("{\"type\": \"lone\\uD800surrogate\"}".utf8),
        Array("{\"type\": \"lone\\uDC00low\"}".utf8),
        Array("{\"type\": \"unescaped\nnewline\"}".utf8),
        Array("{\"type\": \"null\u{00}byte\"}".utf8),
        [0xEF, 0xBB, 0xBF],
        [0xFF, 0xFE, 0x7B, 0x7D],
        [0x7B, 0x22, 0xC3, 0x28, 0x22, 0x3A, 0x31, 0x7D],
        Array((String(repeating: #"{"type": "array", "items": "#, count: 140) + "{}" + String(repeating: "}", count: 140)).utf8),
        Array(#"{"type": "integer", "multipleOf": 2}"#.utf8),
    ]

    @Test("JSONSchemaGate.prepare over Span<UInt8> matches Data and String across strict and lenient modes",
          arguments: validSchemas)
    func validSchemaSpanMatchesData(schema: String) {
        let data = Data(schema.utf8)
        let padded = Data([0xAA, 0xBB]) + data + Data([0xCC])
        for strict in [true, false] {
            let fromData = Result { () throws(UnsupportedJSONSchema) in try JSONSchemaGate.prepare(data, strict: strict) }
            let fromString = Result { () throws(UnsupportedJSONSchema) in try JSONSchemaGate.prepare(schema, strict: strict) }
            let fromSpan = Result { () throws(UnsupportedJSONSchema) in
                let span = data.span
                return try JSONSchemaGate.prepare(span, strict: strict)
            }
            let fromSubspan = Result { () throws(UnsupportedJSONSchema) in
                let outer = padded.span
                let slice = outer.extracting(2 ..< 2 + data.count)
                return try JSONSchemaGate.prepare(slice, strict: strict)
            }
            #expect(fromSpan == fromData)
            #expect(fromString == fromData)
            #expect(fromSubspan == fromData)
        }
    }

    @Test("JSONSchemaGate.prepare and OrderedJSON.parse over Span<UInt8> match Data across invalid and hostile payloads",
          arguments: invalidPayloads)
    func invalidPayloadSpanMatchesData(rawBytes: [UInt8]) {
        let data = Data(rawBytes)
        let bytesSpan = rawBytes.span
        #expect(OrderedJSON.parse(bytesSpan) == OrderedJSON.parse(data))
        #expect(JSONNesting.containsZero(bytesSpan) == JSONNesting.containsZero(data))
        #expect(JSONNesting.depth(of: bytesSpan) == JSONNesting.depth(of: data))
        for strict in [true, false] {
            let spanResult = Result { () throws(UnsupportedJSONSchema) in try JSONSchemaGate.prepare(bytesSpan, strict: strict) }
            let dataResult = Result { () throws(UnsupportedJSONSchema) in try JSONSchemaGate.prepare(data, strict: strict) }
            #expect(spanResult == dataResult)
        }
    }

    @Test("OrderedJSON.parse over Span<UInt8> keeps requested top-level subtrees while still validating skipped subtrees")
    func orderedJSONTopLevelKeepingAndValidation() throws {
        let valid = Data(#"{"messages":[{"content":"a \"[{\" \\uD83D\\uDE00"}],"response_format":{"type":"json_object","b":1,"a":2},"n":-0.25e+2}"#.utf8)
        let parsedSpan = OrderedJSON.parse(valid.span, keeping: ["response_format"])
        let keptFromSpan = try #require(parsedSpan)
        let keptFromData = try #require(OrderedJSON.parse(valid, keeping: ["response_format"]))
        #expect(keptFromSpan == keptFromData)
        #expect(keptFromSpan.text == #"{"response_format":{"type":"json_object","b":1,"a":2}}"#)

        let brokenSkipped = Data(#"{"messages":[{"content":01}],"response_format":{"type":"json_object"}}"#.utf8)
        #expect(OrderedJSON.parse(brokenSkipped.span, keeping: ["response_format"]) == nil)
    }
}
