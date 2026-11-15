import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdTesting
import MLX
import MLXLMCommon
import NIOCore
import Testing
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("ChatRequestFastReader yyjson parity and edge cases (#490)")
struct ChatRequestFastReaderParityTests {
    static let onePixelPNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

    private static func encodedDTO(_ dto: OpenAIController.ChatCompletionRequest) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(dto), as: UTF8.self)
    }

    private static func assertParity(_ json: String) throws {
        let data = Data(json.utf8)
        let fast = try ChatRequestFastReader.read(data)
        let ref = try JSONDecoder().decode(ChatPipeline.Body<OpenAIController.ChatCompletionRequest>.self, from: data)
        #expect(try encodedDTO(fast.dto) == encodedDTO(ref.dto))

        let fastAnti = try fast.antiRepetition.get()
        let refAnti = try ref.antiRepetition.get()
        #expect(fastAnti.dryMultiplier == refAnti.dryMultiplier)
        #expect(fastAnti.dryBase == refAnti.dryBase)
        #expect(fastAnti.dryAllowedLength == refAnti.dryAllowedLength)
        #expect(fastAnti.drySequenceBreakers == refAnti.drySequenceBreakers)
        #expect(fastAnti.xtcThreshold == refAnti.xtcThreshold)
        #expect(fastAnti.xtcProbability == refAnti.xtcProbability)

        let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let fastTools = try fast.tools.get()
        let refTools = try OpenAIController.requestedTools(in: raw)
        #expect(fastTools.tools == refTools.tools)
        #expect(fastTools.choice == refTools.choice)

        let fastFormat = try fast.responseFormat.get()
        let refFormat = try OpenAIController.responseFormat(in: raw, data: data)
        #expect(fastFormat == refFormat)

        let fastMessages = try OpenAIController.chatMessages(fast.dto.messages)
        let refMessages = try OpenAIController.chatMessages(ref.dto.messages)
        #expect(fastMessages == refMessages)
    }

    @Test("Full sampling, DRY/XTC anti-repetition, out-of-order keys, nulls and unknown fields match JSONDecoder")
    func samplingAndAntiRepetitionParity() throws {
        let rich = #"""
        {
          "stream_options": {"include_usage": true, "ignored": 1},
          "xtc_probability": 0.25,
          "xtc_threshold": 0.1,
          "dry_sequence_breakers": ["\n", ":", "\"", "*"],
          "dry_allowed_length": 2,
          "dry_base": 1.75,
          "dry_multiplier": 0.8,
          "grammar_jump_forward": true,
          "speculative_lookup": false,
          "thinking_budget": 512,
          "repetition_penalty": 1.1,
          "presence_penalty": 0.4,
          "top_k": 40,
          "reasoning_effort": "medium",
          "stream": false,
          "dynamic_temperature": true,
          "min_p": 0.05,
          "top_p": 0.92,
          "temperature": 0.6,
          "max_tokens": 256,
          "extra_object": {"nested": [1, true, null]},
          "messages": [{"role": "user", "content": "Hello \u2603 world"}],
          "model": "qwen3-4b"
        }
        """#
        let nulls = #"""
        {
          "model": null,
          "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
          "max_tokens": null,
          "temperature": null,
          "stream_options": null,
          "dry_multiplier": null,
          "dry_sequence_breakers": null,
          "stop": ["END"]
        }
        """#
        try Self.assertParity(rich)
        try Self.assertParity(nulls)
    }

    @Test("Multi-turn messages, tool calls, reasoning items and multimodal image_url variants match JSONDecoder")
    func messagesAndMultimodalParity() throws {
        let transcript = #"""
        {
          "messages": [
            {"role": "developer", "content": "Follow schema."},
            {
              "role": "user",
              "content": [
                {"type": "text", "text": "Inspect both images"},
                {"type": "image_url", "image_url": {"url": "\#(Self.onePixelPNG)"}},
                {"type": "image_url", "image_url": "\#(Self.onePixelPNG)"}
              ]
            },
            {
              "role": "assistant",
              "content": null,
              "reasoning": "Need weather tool",
              "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": "{\"city\":\"Paris\"}"}}
              ]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\":19}"},
            {"type": "reasoning", "content": "ignored"},
            {"type": "function_call", "call_id": "call_2", "name": "clock", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_2", "output": "12:00"}
          ]
        }
        """#
        try Self.assertParity(transcript)
    }

    @Test("Tools, additional_tools, tool_choice variants and sorted unescaped-slash parameters match reference")
    func toolsAndChoiceParity() throws {
        let forced = #"""
        {
          "messages": [{"role": "user", "content": "Search"}],
          "tools": [{
            "type": "function",
            "function": {
              "name": "fetch",
              "description": "Fetch URL",
              "parameters": {
                "z_url": {"type": "string", "default": "https://example.com/a/b"},
                "a_limit": {"type": "integer"}
              }
            }
          }],
          "additional_tools": [{
            "name": "mcp_lookup",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}
          }],
          "tool_choice": {"type": "function", "function": {"name": "fetch"}},
          "parallel_tool_calls": false
        }
        """#
        let disabled = #"""
        {
          "messages": [{"role": "user", "content": "No tools"}],
          "tools": [{"type": "function", "function": {"name": "ping", "parameters": {"b": 1, "a": 2}}}],
          "tool_choice": "none"
        }
        """#
        let disabledObj = #"""
        {
          "messages": [{"role": "user", "content": "Single"}],
          "tools": [{"type": "function", "function": {"name": "ping"}}],
          "tool_choice": {"disable_parallel_tool_use": true}
        }
        """#
        try Self.assertParity(forced)
        try Self.assertParity(disabled)
        try Self.assertParity(disabledObj)
        let parsed = try ChatRequestFastReader.read(Data(forced.utf8))
        let params = try #require(try parsed.tools.get().tools?.first?.parametersJSON)
        #expect(params.contains("https://example.com/a/b"))
        #expect(!params.contains("\\/"))
    }

    @Test("response_format, text.format, guided_regex, guided_grammar and insertion-ordered json_schema match reference")
    func responseFormatParity() throws {
        let orderedSchema = """
        {"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema","json_schema":{"name":"OrderCheck","strict":true,"schema":{"type":"object","properties":{"z_first":{"type":"string"},"a_second":{"type":"integer"}},"required":["z_first","a_second"],"additionalProperties":false}}}}
        """
        let textFormatSchema = """
        {"messages":[{"role":"user","content":"hi"}],"text":{"format":{"type":"json_schema","schema":{"type":"object","properties":{"k":{"type":"boolean"}}}}}}
        """
        let cases = [
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"response_format\":{\"type\":\"text\"}}",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"response_format\":{\"type\":\"json_object\"}}",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"response_format\":{\"type\":\"regex\",\"regex\":\"[0-9]{4}\"}}",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"response_format\":{\"type\":\"grammar\",\"grammar\":\"root ::= \\\"ok\\\"\"}}",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"guided_regex\":\"yes|no\"}",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"guided_grammar\":\"root ::= \\\"a\\\"\"}",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"response_format\":{\"type\":\"structural_tag\",\"structures\":[],\"triggers\":[\"<t>\"]}}",
            orderedSchema,
            textFormatSchema,
        ]
        for payload in cases { try Self.assertParity(payload) }
        let ordered = try ChatRequestFastReader.read(Data(orderedSchema.utf8))
        guard case .jsonSchema(let schemaText, let strict)? = try ordered.responseFormat.get() else {
            Issue.record("expected jsonSchema format")
            return
        }
        #expect(strict)
        let zIndex = try #require(schemaText.range(of: "z_first")?.lowerBound)
        let aIndex = try #require(schemaText.range(of: "a_second")?.lowerBound)
        #expect(zIndex < aIndex)
    }

    @Test("ByteBuffer non-zero readerIndex slice, RawSpan and UTF-8 BOM prefix read without copying")
    func zeroCopyBufferOffsetAndBOM() throws {
        let json = #"{"model":"m","messages":[{"role":"user","content":"slice ok"}],"max_tokens":12}"#
        var buffer = ByteBufferAllocator().buffer(capacity: 128)
        buffer.writeString("JUNK_PREFIX_")
        buffer.moveReaderIndex(forwardBy: 12)
        buffer.writeString(json)
        let fromSlice = try ChatRequestFastReader.read(buffer)
        #expect(fromSlice.dto.model == "m")
        #expect(fromSlice.dto.maxTokens == 12)
        #expect(fromSlice.dto.messages.first?.content?.textValue == "slice ok")

        var bomBytes: [UInt8] = [0xEF, 0xBB, 0xBF]
        bomBytes.append(contentsOf: json.utf8)
        let fromSpan = try ChatRequestFastReader.read(bomBytes.span)
        #expect(fromSpan.dto.maxTokens == 12)

        let loneBOM = Data([0xEF, 0xBB, 0xBF])
        #expect(throws: ChatRequestFastReader.DecodeError.self) { try ChatRequestFastReader.read(loneBOM) }
    }

    @Test("Malformed JSON, invalid UTF-8, wrong scalar types and semantic tool/schema errors fail cleanly")
    func rejectsMalformedAndInvalidInputs() throws {
        let decodeFailures: [Data] = [
            Data(),
            Data("[]".utf8),
            Data("42".utf8),
            Data(#"{"messages": "not-an-array"}"#.utf8),
            Data(#"{"model": "x"}"#.utf8),
            Data(#"{"messages":[{"role":"user","content":"hi"}"#.utf8),
            Data(#"{"messages":[{"role":"user","content":"hi"}]} trailing"#.utf8),
            Data([0x7B, 0x22, 0x6D, 0x6F, 0x64, 0x65, 0x6C, 0x22, 0x3A, 0x22, 0xFF, 0x22, 0x7D]),
            Data(#"{"messages":[{"role":"user","content":"hi"}],"speculative_lookup":"yes"}"#.utf8),
            Data(#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":"ten"}"#.utf8),
            Data(#"{"messages":[{"role":"user","content":"hi"}],"temperature":"hot"}"#.utf8),
            Data(#"{"messages":[{"role":"user","content":"hi"}],"stream_options":true}"#.utf8),
        ]
        for bad in decodeFailures {
            #expect(throws: ChatRequestFastReader.DecodeError.self) { try ChatRequestFastReader.read(bad) }
        }

        let badAnti = try ChatRequestFastReader.read(Data(
            #"{"messages":[{"role":"user","content":"hi"}],"dry_allowed_length":"two"}"#.utf8))
        #expect(throws: ChatRequestFastReader.DecodeError.self) { try badAnti.antiRepetition.get() }

        let badForced = try ChatRequestFastReader.read(Data(
            #"{"messages":[{"role":"user","content":"hi"}],"tool_choice":{"type":"function","function":{"name":"ghost"}}}"#.utf8))
        #expect(throws: OpenAIController.UndeclaredForcedTool.self) { try badForced.tools.get() }

        let badFormat = try ChatRequestFastReader.read(Data(
            #"{"messages":[{"role":"user","content":"hi"}],"guided_regex":"a+","response_format":{"type":"json_object"}}"#.utf8))
        #expect(throws: OpenAIController.InvalidResponseFormat.self) { try badFormat.responseFormat.get() }
    }
}

@Suite("Generic ModelOrchestratorActor static dispatch and /v1/chat/completions E2E (#490)")
struct GenericOrchestratorAndChatRouteTests {
    private static func makeApp() async throws -> (Application<RouterResponder<BasicRequestContext>>, ModelOrchestratorActor<MockInferenceEngine>, MockInferenceEngine) {
        let engine = MockInferenceEngine()
        await engine.setResponder { req in "reply:\(req.messages?.last?.text ?? req.prompt)" }
        let orchestrator = ModelOrchestratorActor(engine: engine, maxTokens: 65_536, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "mock-model")
        let server = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "org/mock-model"))
        return (Application(router: server.makeRouter()), orchestrator, engine)
    }

    @Test("Generic ModelOrchestratorActor<MockInferenceEngine> dispatches generate, stream and snapshots over concrete engine")
    func genericOrchestratorStaticDispatch() async throws {
        let (_, orchestrator, engine) = try await Self.makeApp()
        #expect(await orchestrator.engine.isLoaded)
        let single = try await orchestrator.generate(request: GenerationRequest(prompt: "ping", maxTokens: 16))
        #expect(single.generatedText == "reply:ping")

        var streamed = ""
        for await chunk in try await orchestrator.generateStream(request: GenerationRequest(prompt: "stream", maxTokens: 16)) {
            if case .text(let piece) = chunk { streamed += piece }
        }
        #expect(!streamed.isEmpty)
        let health = await orchestrator.healthSnapshot(modelId: nil)
        #expect(health.isReady)
        #expect(await engine.loadCount == 1)
        #expect(await engine.requests.count == 2)
    }

    @Test("/v1/chat/completions routes non-streaming, streaming and 400 validation errors through ChatRequestFastReader")
    func chatCompletionsRouteEndToEnd() async throws {
        let (app, _, _) = try await Self.makeApp()
        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/chat/completions", method: .post, headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"messages":[{"role":"user","content":"fast-path"}],"max_tokens":32}"#)
            ) { res in
                #expect(res.status == .ok)
                let obj = try #require(try JSONSerialization.jsonObject(with: Data(buffer: res.body)) as? [String: Any])
                let choices = try #require(obj["choices"] as? [[String: Any]])
                let message = try #require(choices.first?["message"] as? [String: Any])
                #expect(message["content"] as? String == "reply:fast-path")
            }

            try await client.execute(
                uri: "/v1/chat/completions", method: .post, headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"messages":[{"role":"user","content":"sse"}],"stream":true,"stream_options":{"include_usage":true}}"#)
            ) { res in
                #expect(res.status == .ok)
                let body = String(buffer: res.body)
                #expect(body.contains("data: "))
                #expect(body.contains("[DONE]"))
            }

            let badBodies = [
                #"{"messages":[]}"#,
                #"{"messages":[{"role":"user","content":"x"}],"speculative_lookup":"yes"}"#,
                #"{"messages":[{"role":"user","content":"x"}],"dry_allowed_length":"two"}"#,
                #"{"messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"function","function":{"name":"missing"}}}"#,
            ]
            for payload in badBodies {
                try await client.execute(
                    uri: "/v1/chat/completions", method: .post, headers: [.contentType: "application/json"],
                    body: ByteBuffer(string: payload)
                ) { res in
                    #expect(res.status == .badRequest)
                }
            }
        }
    }
}

@Suite("De-boxed HeadSplitKVCache RetrievalCache behavior (#490)", .serialized)
struct DeboxedKVCacheTests {
    @Test("HeadSplitKVCache updates, attends, copies and trims cleanly in both simple and zoned retrieval modes")
    func headSplitSimpleAndZonedModes() {
        let zones = HeadSplitKVCache.Zoning(
            zones: KVZones(sinkTokens: 2, windowTokens: 2, migrationStep: 2), precision: .q8)
        for zoning in [nil, zones] {
            let cache = HeadSplitKVCache(retrieval: [0, 2], zoning: zoning)
            let q = MLXArray.ones([1, 4, 8, 64], dtype: .float16)
            let k = MLXArray.ones([1, 4, 8, 64], dtype: .float16)
            let v = MLXArray.ones([1, 4, 8, 64], dtype: .float16)
            let attended = cache.updateAndAttend(queries: q, keys: k, values: v, scale: 0.125, mask: .causal)
            eval(attended)
            #expect(attended.shape == [1, 4, 8, 64])
            #expect(cache.offset == 8)
            #expect(cache.retrievalTokens == 8)
            #expect(cache.streamingTokens == 8)

            let cloned = cache.copy()
            let qStep = MLXArray.ones([1, 4, 1, 64], dtype: .float16)
            let kStep = MLXArray.ones([1, 4, 1, 64], dtype: .float16)
            let vStep = MLXArray.ones([1, 4, 1, 64], dtype: .float16)
            let stepOut = cache.updateAndAttend(queries: qStep, keys: kStep, values: vStep, scale: 0.125, mask: .none)
            eval(stepOut)
            #expect(cache.offset == 9)
            #expect(cloned.offset == 8)

            let removed = cache.trim(3)
            #expect(removed == 3)
            #expect(cache.offset == 6)
        }
    }

    @Test("KVHeadMask replaces KVCacheSimple layers with de-boxed HeadSplitKVCache while preserving other caches")
    func headMaskReplacement() {
        let mask = KVHeadMask(retrieval: [[0], [1]])
        let input: [any KVCache] = [KVCacheSimple(), RotatingKVCache(maxSize: 16), KVCacheSimple()]
        let replaced = mask.replacingFullAttention(input)
        #expect(replaced[0] is HeadSplitKVCache)
        #expect(replaced[1] is RotatingKVCache)
        #expect(replaced[2] is HeadSplitKVCache)
    }
}

@Suite("ChatRequestFastReader 10MB multimodal micro-benchmark (#490)", .serialized)
struct ChatRequestFastReaderBenchmarkTests {
    static var isEnabled: Bool { ProcessInfo.processInfo.environment["BASALT_BENCH"] == "1" }

    @Test("10MB multimodal request parsing throughput vs legacy JSONDecoder + JSONSerialization",
          .enabled(if: Self.isEnabled, "set BASALT_BENCH=1"))
    func tenMegabyteMultimodalThroughput() throws {
        let chunk = String(repeating: "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU2Nzg5", count: 52_000)
        let img = "data:image/png;base64," + chunk
        let json = """
        {"model":"test","stream":true,"max_tokens":512,"temperature":0.7,"top_p":0.9,"top_k":40,
        "dry_multiplier":0.8,"dry_base":1.75,"dry_allowed_length":2,"xtc_threshold":0.1,"xtc_probability":0.2,
        "tools":[{"type":"function","function":{"name":"lookup","description":"q","parameters":{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}}}],
        "response_format":{"type":"json_schema","json_schema":{"name":"ans","strict":true,"schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}}},
        "messages":[{"role":"user","content":[
        {"type":"text","text":"Describe all four frames in detail."},
        {"type":"image_url","image_url":{"url":"\(img)"}},
        {"type":"image_url","image_url":{"url":"\(img)"}},
        {"type":"image_url","image_url":{"url":"\(img)"}},
        {"type":"image_url","image_url":{"url":"\(img)"}}
        ]}]}
        """
        var buffer = ByteBufferAllocator().buffer(capacity: json.utf8.count)
        buffer.writeString(json)
        let mb = Double(buffer.readableBytes) / (1024.0 * 1024.0)
        let iterations = 20

        for _ in 0 ..< 3 {
            let p = try ChatRequestFastReader.read(buffer)
            _ = try p.tools.get()
        }
        let clock = ContinuousClock()
        let fastElapsed = try clock.measure {
            for _ in 0 ..< iterations {
                let p = try ChatRequestFastReader.read(buffer)
                _ = try p.antiRepetition.get()
                _ = try p.tools.get()
                _ = try p.responseFormat.get()
            }
        }
        let legacyElapsed = try clock.measure {
            let decoder = JSONDecoder()
            for _ in 0 ..< iterations {
                let data = Data(buffer: buffer)
                let ref = try decoder.decode(ChatPipeline.Body<OpenAIController.ChatCompletionRequest>.self, from: data)
                _ = try ref.antiRepetition.get()
                let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                _ = try OpenAIController.requestedTools(in: raw)
                _ = try OpenAIController.responseFormat(in: raw, data: data)
            }
        }
        let fastSec = Double(fastElapsed.components.seconds) + Double(fastElapsed.components.attoseconds) * 1e-18
        let legSec = Double(legacyElapsed.components.seconds) + Double(legacyElapsed.components.attoseconds) * 1e-18
        let fastMs = (fastSec / Double(iterations)) * 1000.0
        let legMs = (legSec / Double(iterations)) * 1000.0
        let speedup = legSec / max(1e-9, fastSec)
        let fastMBps = (mb * Double(iterations)) / max(1e-9, fastSec)
        let legMBps = (mb * Double(iterations)) / max(1e-9, legSec)
        let line = String(
            format: "BENCH_YYJSON payload_mb=%.2f fast_ms=%.3f legacy_ms=%.3f fast_mbps=%.1f legacy_mbps=%.1f speedup=%.2fx\n",
            mb, fastMs, legMs, fastMBps, legMBps, speedup)
        FileHandle.standardError.write(Data(line.utf8))
        #expect(speedup >= 3.0)
    }
}
