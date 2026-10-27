import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltServer

/// Controller tests for the REST surface on MockInferenceEngine, driven through the real router.
/// Streaming Anthropic and the Responses API are covered by CodingAgentAPITests; the perimeter
/// (auth, rate limit) by RateLimitTests and ConfigTests. The scenarios of scripts/test_rest.sh
/// (health, then POST /api/v1/generate) run here without a real model.
@Suite("REST controllers on the mock engine (#21)")
struct RESTServerTests {

    private struct Setup {
        let app: Application<RouterResponder<BasicRequestContext>>
        let engine: MockInferenceEngine
        let orchestrator: ModelOrchestratorActor
    }

    private func setup(load: Bool = true, reasoning: String? = nil, failGenerate: Bool = false) async throws -> Setup {
        let engine = MockInferenceEngine()
        await engine.setMockReasoning(reasoning)
        await engine.setShouldFailOnGenerate(failGenerate)
        // A budget above any request here, so the RAM-dependent cap never rewrites max_tokens.
        let orch = ModelOrchestratorActor(engine: engine, maxTokens: 65536, telemetry: TelemetryTracker())
        if load { try await orch.loadModel(path: "mock") }
        let server = RESTServer(orchestrator: orch, config: try ServerConfig(modelPath: "org/mock-model"))
        return Setup(app: Application(router: server.makeRouter()), engine: engine, orchestrator: orch)
    }

    private func json(_ buffer: ByteBuffer) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: Data(String(buffer: buffer).utf8)) as? [String: Any])
    }

    /// `data:` payloads of an SSE body, in order (raw strings, `[DONE]` included).
    private func sseData(_ buffer: ByteBuffer) -> [String] {
        String(buffer: buffer).components(separatedBy: "\n").compactMap {
            $0.hasPrefix("data: ") ? String($0.dropFirst(6)) : nil
        }
    }

    private func object(_ raw: String) -> [String: Any]? {
        try? JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any]
    }

    // MARK: — OpenAI /v1/chat/completions

    @Test("Chat completion: system and user turns reach the engine; text, finish_reason and usage come back")
    func chatCompletion() async throws {
        let s = try await setup()
        let request = #"{"model": "local", "max_tokens": 77, "messages": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": [{"type": "text", "text": "hi"}]}]}"#
        try await s.app.test(.router) { client in
            try await client.execute(uri: "/v1/chat/completions", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: request)) { res in
                #expect(res.status == .ok)
                let body = try json(res.body)
                #expect(body["object"] as? String == "chat.completion")
                #expect(body["model"] as? String == "local")
                let choice = try #require((body["choices"] as? [[String: Any]])?.first)
                #expect(choice["finish_reason"] as? String == "stop")
                let message = try #require(choice["message"] as? [String: Any])
                #expect(message["role"] as? String == "assistant")
                #expect(message["content"] as? String == "Mock response")
                let usage = try #require(body["usage"] as? [String: Any])
                let prompt = try #require(usage["prompt_tokens"] as? Int)
                #expect(usage["completion_tokens"] as? Int == 10)
                #expect(usage["total_tokens"] as? Int == prompt + 10)
            }
        }
        let sent = try #require(await s.engine.lastRequest)
        #expect(sent.messages == [ChatMessageInput(role: .system, text: "Be terse."), ChatMessageInput(role: .user, text: "hi")])
        #expect(sent.prompt == "Be terse.\n\nhi")
        #expect(sent.maxTokens == 77)
        #expect(sent.temperature == nil, "absent temperature leaves the model's generation_config in charge")
        #expect(sent.topP == nil, "absent top_p leaves the model's generation_config in charge")
        #expect(sent.minP == nil, "absent min_p leaves the server default in charge")
        #expect(sent.dynamicTemperature == nil)
    }

    @Test("min_p and dynamic_temperature reach the engine from every API", arguments: [
        ("/v1/chat/completions", #"{"messages": [{"role": "user", "content": "hi"}], "min_p": 0.05, "dynamic_temperature": true}"#),
        ("/v1/messages", #"{"messages": [{"role": "user", "content": "hi"}], "min_p": 0.05, "dynamic_temperature": true}"#),
        ("/v1/responses", #"{"input": "hi", "min_p": 0.05, "dynamic_temperature": true}"#),
    ])
    func samplingFieldsPerAPI(uri: String, request: String) async throws {
        let s = try await setup()
        try await s.app.test(.router) { client in
            try await client.execute(uri: uri, method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: request)) { res in
                #expect(res.status == .ok)
            }
        }
        let sent = try #require(await s.engine.lastRequest)
        #expect(sent.minP == 0.05)
        #expect(sent.dynamicTemperature == true)
    }

    @Test("min_p outside [0, 1] is refused with 400 on every API and never reaches the engine", arguments: [
        ("/v1/chat/completions", #"{"messages": [{"role": "user", "content": "hi"}], "min_p": 1.5}"#),
        ("/v1/messages", #"{"messages": [{"role": "user", "content": "hi"}], "min_p": 1.5}"#),
        ("/v1/responses", #"{"input": "hi", "min_p": 1.5}"#),
        ("/api/v1/generate", #"{"prompt": "hi", "minP": -0.1}"#),
    ])
    func minPOutOfRange(uri: String, request: String) async throws {
        let s = try await setup()
        try await s.app.test(.router) { client in
            try await client.execute(uri: uri, method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: request)) { res in
                #expect(res.status == .badRequest, "\(uri) -> \(res.status)")
            }
        }
        #expect(await s.engine.lastRequest == nil)
    }

    @Test("Chat completion stream: reasoning (also as reasoning_content) and text deltas, a finish chunk, then [DONE]")
    func chatCompletionStream() async throws {
        let s = try await setup(reasoning: "hmm")
        let request = #"{"model": "local", "stream": true, "messages": [{"role": "user", "content": "hi"}]}"#
        try await s.app.test(.router) { client in
            try await client.execute(uri: "/v1/chat/completions", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: request)) { res in
                #expect(res.status == .ok)
                #expect(res.headers[.contentType] == "text/event-stream")
                let data = sseData(res.body)
                #expect(data.last == "[DONE]")
                let chunks = data.dropLast().compactMap(object)
                #expect(chunks.count == data.count - 1, "every frame before [DONE] is JSON")
                #expect(chunks.allSatisfy { $0["object"] as? String == "chat.completion.chunk" })
                #expect(Set(chunks.compactMap { $0["id"] as? String }).count == 1, "one id for the whole stream")

                let choices = chunks.compactMap { ($0["choices"] as? [[String: Any]])?.first }
                let deltas = choices.compactMap { $0["delta"] as? [String: Any] }
                #expect(deltas.compactMap { $0["reasoning"] as? String } == ["hmm"])
                #expect(deltas.compactMap { $0["reasoning_content"] as? String } == ["hmm"])
                #expect(deltas.allSatisfy { ($0["reasoning"] as? String) == ($0["reasoning_content"] as? String) })
                #expect(deltas.compactMap { $0["content"] as? String }.joined() == "Mock response")
                #expect(choices.last?["finish_reason"] as? String == "stop")
                #expect(choices.dropLast().allSatisfy { $0["finish_reason"] is NSNull })
            }
        }
    }

    @Test("Non-streaming replies carry the reasoning the stream sends: chat message reasoning and reasoning_content, a Responses reasoning item, a Messages thinking block")
    func nonStreamingReasoning() async throws {
        let s = try await setup(reasoning: "hmm")
        let turn = #""messages": [{"role": "user", "content": "hi"}]"#
        try await s.app.test(.router) { client in
            try await client.execute(uri: "/v1/chat/completions", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: "{\(turn)}")) { res in
                let choice = try #require((try json(res.body)["choices"] as? [[String: Any]])?.first)
                let message = try #require(choice["message"] as? [String: Any])
                #expect(message["reasoning"] as? String == "hmm")
                #expect(message["reasoning_content"] as? String == "hmm")
                #expect(message["content"] as? String == "Mock response")
            }
            try await client.execute(uri: "/v1/responses", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: "{\(turn)}")) { res in
                let output = try #require(try json(res.body)["output"] as? [[String: Any]])
                #expect(output.map { $0["type"] as? String } == ["reasoning", "message"])
                let reasoning = try #require(output.first?["content"] as? [[String: Any]])
                #expect(reasoning.first?["type"] as? String == "reasoning_text")
                #expect(reasoning.first?["text"] as? String == "hmm")
            }
            try await client.execute(uri: "/v1/messages", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: "{\(turn)}")) { res in
                let content = try #require(try json(res.body)["content"] as? [[String: Any]])
                #expect(content.map { $0["type"] as? String } == ["thinking", "text"])
                #expect(content.first?["thinking"] as? String == "hmm")
                #expect(content.first?["signature"] as? String == "")
            }
        }
    }

    @Test("A reply without reasoning carries neither reasoning nor reasoning_content")
    func noReasoningNoAlias() async throws {
        let s = try await setup()
        try await s.app.test(.router) { client in
            try await client.execute(uri: "/v1/chat/completions", method: .post, headers: [.contentType: "application/json"],
                                     body: ByteBuffer(string: #"{"messages": [{"role": "user", "content": "hi"}]}"#)) { res in
                let choice = try #require((try json(res.body)["choices"] as? [[String: Any]])?.first)
                let message = try #require(choice["message"] as? [String: Any])
                #expect(message["reasoning"] == nil)
                #expect(message["reasoning_content"] == nil)
            }
        }
    }

    @Test("An assistant turn sent back with reasoning_content reaches the engine exactly as one sent with reasoning")
    func reasoningContentInputAlias() async throws {
        let s = try await setup()
        var seen: [[ChatMessageInput]?] = []
        for key in ["reasoning", "reasoning_content"] {
            let request = """
                {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello", "\(key)": "thought"}, {"role": "user", "content": "again"}]}
                """
            try await s.app.test(.router) { client in
                try await client.execute(uri: "/v1/chat/completions", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: request)) { res in
                    #expect(res.status == .ok, "\(key)")
                }
            }
            seen.append(await s.engine.lastRequest?.messages)
        }
        #expect(seen[0] != nil)
        #expect(seen[0] == seen[1])
    }

    @Test("Chat completion refuses an empty or undecodable body with 400; an engine failure keeps its own status in the error envelope")
    func chatCompletionErrors() async throws {
        let s = try await setup()
        try await s.app.test(.router) { client in
            for body in [#"{"messages": []}"#, "not json", #"{"messages": "hi"}"#] {
                try await client.execute(uri: "/v1/chat/completions", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: body)) { res in
                    #expect(res.status == .badRequest, "\(body)")
                    let err = try json(res.body)["error"] as? [String: Any]
                    #expect(err?["type"] as? String == "invalid_request_error")
                }
            }
        }
        let failing = try await setup(failGenerate: true)
        try await failing.app.test(.router) { client in
            try await client.execute(uri: "/v1/chat/completions", method: .post, headers: [.contentType: "application/json"],
                                     body: ByteBuffer(string: #"{"messages": [{"role": "user", "content": "hi"}]}"#)) { res in
                #expect(res.status.code == BasaltError.inferenceHardwareFailure(reason: "").httpStatus)
                let body = try json(res.body)
                #expect(body["type"] as? String == "error")
            }
        }
    }

    @Test("A three-turn conversation reaches the engine as role-tagged messages with no [System]:/[User]:/[Assistant]: text anywhere",
          arguments: [
            ("/v1/chat/completions", #"{"messages": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, {"role": "user", "content": "again"}]}"#),
            ("/v1/messages", #"{"system": "Be terse.", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}, {"role": "user", "content": "again"}]}"#),
            ("/v1/responses", #"{"instructions": "Be terse.", "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}, {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]}, {"role": "user", "content": "again"}]}"#),
          ])
    func threeTurnConversationIsStructured(uri: String, request: String) async throws {
        let s = try await setup()
        try await s.app.test(.router) { client in
            try await client.execute(uri: uri, method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: request)) { res in
                #expect(res.status == .ok)
            }
        }
        let sent = try #require(await s.engine.lastRequest)
        #expect(sent.messages == [
            ChatMessageInput(role: .system, text: "Be terse."),
            ChatMessageInput(role: .user, text: "hi"),
            ChatMessageInput(role: .assistant, text: "hello"),
            ChatMessageInput(role: .user, text: "again"),
        ])
        let texts = [sent.prompt] + (sent.messages ?? []).map(\.text)
        for marker in ["[System]:", "[User]:", "[Assistant]:"] {
            #expect(!texts.contains { $0.contains(marker) }, "\(marker) in \(texts)")
        }
    }

    @Test("A compacted service prompt replaces the conversation, so its messages are not sent alongside it")
    func compactedServicePromptDropsMessages() async throws {
        let s = try await setup()
        let history = String(repeating: "word ", count: 300)
        let request = #"{"messages": [{"role": "user", "content": "\#(history)"}, {"role": "user", "content": "Generate a title for this conversation"}]}"#
        try await s.app.test(.router) { client in
            try await client.execute(uri: "/v1/chat/completions", method: .post,
                                     headers: [.contentType: "application/json", HTTPField.Name("x-service-request")!: "true"],
                                     body: ByteBuffer(string: request)) { res in
                #expect(res.status == .ok)
            }
        }
        let sent = try #require(await s.engine.lastRequest)
        #expect(sent.prompt.hasPrefix("Generate a short 3-5 word title"))
        #expect(sent.messages == nil)
    }

    // MARK: — Anthropic /v1/messages (non-streaming)

    @Test("Messages: system prompt, default max_tokens and thinking budget reach the engine; reply is a text block")
    func anthropicMessage() async throws {
        let s = try await setup()
        let request = #"{"model": "claude-local/org--mock-model", "system": "You are Basalt.", "thinking": {"type": "enabled", "budget_tokens": 8100}, "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, {"role": "user", "content": "again"}]}"#
        try await s.app.test(.router) { client in
            try await client.execute(uri: "/v1/messages", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: request)) { res in
                #expect(res.status == .ok)
                let body = try json(res.body)
                #expect(body["type"] as? String == "message")
                #expect(body["role"] as? String == "assistant")
                #expect(body["stop_reason"] as? String == "end_turn")
                #expect((body["id"] as? String)?.hasPrefix("msg_") == true)
                let content = try #require(body["content"] as? [[String: Any]])
                #expect(content.count == 1)
                #expect(content.first?["type"] as? String == "text")
                #expect(content.first?["text"] as? String == "Mock response")
                #expect((body["usage"] as? [String: Any])?["output_tokens"] as? Int == 10)
            }
        }
        let sent = try #require(await s.engine.lastRequest)
        #expect(sent.messages?.map(\.role) == [.system, .user, .assistant, .user])
        #expect(sent.messages?.map(\.text) == ["You are Basalt.", "hi", "hello", "again"])
        #expect(sent.reasoningEffort == "high")
        #expect(sent.maxTokens == 8192, "the controller's default when max_tokens is absent")
    }

    @Test("Thinking budgets map to effort levels; disabled thinking turns it off; reasoning_effort wins")
    func anthropicThinkingMapping() async throws {
        let s = try await setup()
        let cases: [(String, String?)] = [
            (#""thinking": {"type": "disabled"}"#, "off"),
            (#""thinking": {"type": "enabled", "budget_tokens": 1000}"#, "low"),
            (#""thinking": {"type": "enabled", "budget_tokens": 5000}"#, "medium"),
            (#""thinking": {"type": "enabled"}"#, "medium"),
            (#""thinking": {"type": "enabled", "budget_tokens": 1000}, "reasoning_effort": "high""#, "high"),
            (#""max_tokens": 16"#, nil),
        ]
        try await s.app.test(.router) { client in
            for (fields, expected) in cases {
                let body = #"{"messages": [{"role": "user", "content": "hi"}], "# + fields + "}"
                let status = try await client.execute(uri: "/v1/messages", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: body)) { $0.status }
                #expect(status == .ok)
                #expect(await s.engine.lastRequest?.reasoningEffort == expected, "\(fields)")
            }
        }
    }

    @Test("Messages errors use the Anthropic envelope: 400 for a bad request, the engine's status for a failure")
    func anthropicErrors() async throws {
        let s = try await setup()
        try await s.app.test(.router) { client in
            for body in [#"{"messages": []}"#, #"{"model": "x"}"#] {
                try await client.execute(uri: "/v1/messages", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: body)) { res in
                    #expect(res.status == .badRequest, "\(body)")
                    let envelope = try json(res.body)
                    #expect(envelope["type"] as? String == "error")
                    #expect((envelope["error"] as? [String: Any])?["type"] as? String == "invalid_request_error")
                }
            }
        }
        let failing = try await setup(failGenerate: true)
        try await failing.app.test(.router) { client in
            try await client.execute(uri: "/v1/messages", method: .post, headers: [.contentType: "application/json"],
                                     body: ByteBuffer(string: #"{"messages": [{"role": "user", "content": "hi"}]}"#)) { res in
                #expect(res.status.code == BasaltError.inferenceHardwareFailure(reason: "").httpStatus)
                let body = try json(res.body)
                #expect((body["error"] as? [String: Any])?["type"] as? String == "api_error")
            }
        }
    }

    // MARK: — Native /api/v1 (scripts/test_rest.sh)

    @Test("test_rest.sh: health turns ready once the model is loaded, then /api/v1/generate answers")
    func nativeGenerate() async throws {
        let s = try await setup(load: false)
        try await s.app.test(.router) { client in
            try await client.execute(uri: "/api/v1/health", method: .get) { res in
                #expect(res.status == .ok)
                let health = try JSONDecoder().decode(HealthResponse.self, from: Data(String(buffer: res.body).utf8))
                #expect(health.isReady == false)
                #expect(health.status == "initializing")
                #expect(health.modelId == "mock-model")
            }
            try await s.orchestrator.loadModel(path: "mock")
            try await client.execute(uri: "/api/v1/health", method: .get) { res in
                let health = try JSONDecoder().decode(HealthResponse.self, from: Data(String(buffer: res.body).utf8))
                #expect(health.isReady)
                #expect(health.status == "ok")
            }
            let request = #"{"prompt": "What is the capital of France?", "maxTokens": 50}"#
            try await client.execute(uri: "/api/v1/generate", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: request)) { res in
                #expect(res.status == .ok)
                let result = try JSONDecoder().decode(GenerationResponse.self, from: Data(String(buffer: res.body).utf8))
                #expect(result.generatedText == "Mock response")
                #expect(result.completionTokens == 10)
                #expect(result.finishReason == .stop)
            }
        }
        let sent = try #require(await s.engine.lastRequest)
        #expect(sent.prompt == "What is the capital of France?")
        #expect(sent.maxTokens == 50)
    }

    @Test("Native generate: 400 on bad JSON, the engine's status on failure; /v1/generate returns the same body")
    func nativeGenerateErrors() async throws {
        let s = try await setup()
        try await s.app.test(.router) { client in
            for uri in ["/api/v1/generate", "/v1/generate"] {
                let status = try await client.execute(uri: uri, method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: #"{"maxTokens": 5}"#)) { $0.status }
                #expect(status == .badRequest, "\(uri)")
            }
            try await client.execute(uri: "/v1/generate", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: #"{"prompt": "hi"}"#)) { res in
                #expect(res.status == .ok)
                let result = try JSONDecoder().decode(GenerationResponse.self, from: Data(String(buffer: res.body).utf8))
                #expect(result.generatedText == "Mock response")
            }
        }
        let failing = try await setup(failGenerate: true)
        try await failing.app.test(.router) { client in
            try await client.execute(uri: "/api/v1/generate", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: #"{"prompt": "hi"}"#)) { res in
                #expect(res.status.code == 503)
                let envelope = try JSONDecoder().decode(ErrorResponse.self, from: Data(String(buffer: res.body).utf8))
                #expect(envelope.code == 503)
            }
        }
    }

    @Test("Native stream: StreamChunk frames (text before metadata), then [DONE]; a failure is an error event")
    func nativeGenerateStream() async throws {
        let s = try await setup()
        try await s.app.test(.router) { client in
            try await client.execute(uri: "/api/v1/generate/stream", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: #"{"prompt": "hi"}"#)) { res in
                #expect(res.status == .ok)
                #expect(res.headers[.contentType] == "text/event-stream")
                let data = sseData(res.body)
                #expect(data.last == "[DONE]")
                let chunks = try data.dropLast().map { try JSONDecoder().decode(StreamChunk.self, from: Data($0.utf8)) }
                guard case .text(let text)? = chunks.first, case .metadata(let meta)? = chunks.last else {
                    Issue.record("expected text first and metadata last, got \(chunks)")
                    return
                }
                #expect(text == "Mock response")
                #expect(meta.completionTokens == 10)
            }
        }
        let failing = try await setup(failGenerate: true)
        try await failing.app.test(.router) { client in
            try await client.execute(uri: "/api/v1/generate/stream", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: #"{"prompt": "hi"}"#)) { res in
                let raw = String(buffer: res.body)
                #expect(raw.hasPrefix("event: error\n"))
                #expect(!raw.contains("[DONE]"))
            }
        }
    }

    // MARK: — Models

    @Test("/v1/models lists claude-local IDs that round-trip to HF repo IDs, the loaded model first")
    func listModels() async throws {
        let s = try await setup()
        await s.orchestrator.setModelId("org/mock-model")
        try await s.app.test(.router) { client in
            try await client.execute(uri: "/v1/models", method: .get) { res in
                #expect(res.status == .ok)
                let body = try json(res.body)
                #expect(body["object"] as? String == "list")
                let data = try #require(body["data"] as? [[String: Any]])
                #expect(!data.isEmpty)
                for entry in data {
                    let id = try #require(entry["id"] as? String)
                    let display = try #require(entry["display_name"] as? String)
                    #expect(id.hasPrefix("claude-local/"), "Claude Code's discovery filter needs the prefix: \(id)")
                    #expect(ModelsController.hfRepoId(from: id) == display)
                }
                // The HF cache is the real one on this machine: either it is empty and the loaded
                // model is the single entry, or the loaded model (when cached) sorts first.
                let loaded = data.filter { $0["is_loaded"] as? Bool == true }
                if let first = loaded.first {
                    #expect(data.first?["id"] as? String == first["id"] as? String)
                }
            }
        }
    }

    @Test("/v1/models/current reports the loaded model's discovery ID and readiness")
    func currentModel() async throws {
        let s = try await setup()
        await s.orchestrator.setModelId("org/mock-model")
        try await s.app.test(.router) { client in
            try await client.execute(uri: "/v1/models/current", method: .get) { res in
                let body = try json(res.body)
                #expect(body["id"] as? String == "claude-local/org--mock-model")
                #expect(body["display_name"] as? String == "org/mock-model")
                #expect(body["is_ready"] as? Bool == true)
                #expect(body["status"] as? String == "ok")
            }
        }
    }

    @Test("/v1/models/load accepts the discovery ID form, 404s an uncached model, 400s a missing model field")
    func loadModel() async throws {
        let s = try await setup()
        await s.orchestrator.setModelId("org/mock-model")
        try await s.app.test(.router) { client in
            // Already loaded: resolves the discovery ID to the HF repo ID and is a no-op switch.
            try await client.execute(uri: "/v1/models/load", method: .post, headers: [.contentType: "application/json"],
                                     body: ByteBuffer(string: #"{"model": "claude-local/org--mock-model"}"#)) { res in
                #expect(res.status == .ok)
                let body = try json(res.body)
                #expect(body["status"] as? String == "loaded")
                #expect(body["display_name"] as? String == "org/mock-model")
            }
            try await client.execute(uri: "/v1/models/load", method: .post, headers: [.contentType: "application/json"],
                                     body: ByteBuffer(string: #"{"model": "nobody/not-cached-\#(UUID().uuidString)"}"#)) { res in
                #expect(res.status == .notFound)
                let body = try json(res.body)
                #expect((body["error"] as? [String: Any])?["type"] as? String == "invalid_request_error")
            }
            for body in [#"{}"#, #"{"model": ""}"#] {
                let status = try await client.execute(uri: "/v1/models/load", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: body)) { $0.status }
                #expect(status == .badRequest, "\(body)")
            }
        }
        #expect(await s.orchestrator.currentModelId == "org/mock-model", "a failed load leaves the current model in place")
    }

    private static let jsonHeaders: HTTPFields = [.contentType: "application/json"]
    private static let aliasModels = ["meta-llama/llama-3.1-8b-instruct:free", "org/name@rev", "org/model/variant"]
    private static let pathLikeModels = ["/tmp", "./x", "../x", "~/x", "claude-local/..--x", "org/../x", "/", " /tmp/x", " claude-local/..--x"]

    @Test("REST model ids are repo ids: HF charset, two components, no path forms (#72)")
    func repoIdShape() {
        for id in ["mlx-community/Qwen3.5-4B-4bit", "AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit", "org/mock-model"] {
            #expect(ModelsController.isRepoId(id), "\(id)")
        }
        for id in Self.pathLikeModels + ["basalt", "a/b/c", "org/", "/name", ".org/x", "org/.x", "org/x y", "org/x~", "-org/x"] {
            #expect(!ModelsController.isRepoId(ModelsController.hfRepoId(from: id)), "\(id)")
            #expect(ModelsController.restModel(from: id).repoId == nil, "\(id)")
        }
        for id in Self.pathLikeModels + [".hidden", "~user", "\t./x", "\n~/x", "org/x/../../y"] {
            #expect(ModelsController.restModel(from: id) == .path, "\(id)")
        }
        for id in Self.aliasModels + ["basalt", "gpt-4o", "claude-sonnet-4-5", "local", "org/x y"] {
            #expect(ModelsController.restModel(from: id) == .alias, "\(id)")
        }
        #expect(ModelsController.restModel(from: " org/mock-model\n") == .repoId("org/mock-model"))
        #expect(ModelsController.restModel(from: "claude-local/org--mock-model") == .repoId("org/mock-model"))
    }

    @Test("/v1/models/load refuses paths with 400, echoes no path, and the loaded model stays loaded (#72)")
    func loadModelRefusesPaths() async throws {
        let s = try await setup()
        await s.orchestrator.setModelId("org/mock-model")
        try await s.app.test(.router) { client in
            for model in Self.pathLikeModels {
                try await client.execute(uri: "/v1/models/load", method: .post, headers: Self.jsonHeaders,
                                         body: ByteBuffer(string: #"{"model": "\#(model)"}"#)) { res in
                    #expect(res.status == .badRequest, "\(model)")
                    let message = (try json(res.body)["error"] as? [String: Any])?["message"] as? String
                    #expect(message == ModelsController.repoIdRequired, "\(model)")
                }
            }
        }
        #expect(await s.orchestrator.currentModelId == "org/mock-model")
        #expect(await s.engine.isLoaded)
    }

    @Test("A path in a generation request's model field is refused before it can trigger a load (#72)")
    func generationRefusesPathModels() async throws {
        let s = try await setup(load: false)
        let requests = [
            ("/v1/messages", #"{"model": "MODEL", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}"#),
            ("/v1/chat/completions", #"{"model": "MODEL", "messages": [{"role": "user", "content": "hi"}]}"#),
            ("/v1/responses", #"{"model": "MODEL", "input": "hi"}"#),
        ]
        try await s.app.test(.router) { client in
            for (uri, template) in requests {
                for model in ["/tmp", "claude-local/..--x", " /tmp/x"] {
                    for stream in [false, true] {
                        let body = template.replacingOccurrences(of: "MODEL", with: model)
                            .replacingOccurrences(of: #""model""#, with: #""stream": \#(stream), "model""#)
                        try await client.execute(uri: uri, method: .post, headers: Self.jsonHeaders,
                                                 body: ByteBuffer(string: body)) { res in
                            #expect(res.status == .badRequest, "\(uri) \(model) stream=\(stream)")
                            #expect(!String(buffer: res.body).contains("/tmp"))
                        }
                    }
                }
            }
        }
        #expect(await !s.engine.isLoaded, "no request may load a local path")
    }

    @Test("REST error bodies carry no filesystem path; the path stays in the server log (#72)")
    func errorBodiesHideFilesystemPaths() async throws {
        let secret = "/Users/someone/private-models/x"
        #expect(ModelsController.publicMessage(for: BasaltError.weightsCorrupted(path: secret, reason: "r")) == ModelsController.modelUnavailable)
        #expect(ModelsController.publicMessage(for: BasaltError.modelNotFound(identifier: secret)) == ModelsController.modelUnavailable)
        let s = try await setup(failGenerate: true)
        await s.engine.setGenerateError(.modelNotFound(identifier: secret))
        let requests = [
            ("/v1/messages", #"{"max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}"#),
            ("/v1/chat/completions", #"{"messages": [{"role": "user", "content": "hi"}]}"#),
            ("/v1/responses", #"{"input": "hi"}"#),
            ("/api/v1/generate", #"{"prompt": "hi"}"#),
            ("/v1/generate", #"{"prompt": "hi"}"#),
        ]
        try await s.app.test(.router) { client in
            for (uri, body) in requests {
                try await client.execute(uri: uri, method: .post, headers: Self.jsonHeaders, body: ByteBuffer(string: body)) { res in
                    #expect(res.status != .ok, "\(uri)")
                    #expect(!String(buffer: res.body).contains("private-models"), "\(uri)")
                }
            }
        }
    }

    @Test("Generation treats a non-repo id that is not a path as an alias: served by the loaded model, no switch; load still 400s (#72)")
    func generationIgnoresAliasModels() async throws {
        let s = try await setup()
        await s.orchestrator.setModelId("org/mock-model")
        try await s.app.test(.router) { client in
            for model in Self.aliasModels {
                for (uri, body) in [
                    ("/v1/messages", #"{"model": "\#(model)", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}"#),
                    ("/v1/chat/completions", #"{"model": "\#(model)", "messages": [{"role": "user", "content": "hi"}]}"#),
                    ("/v1/responses", #"{"model": "\#(model)", "input": "hi"}"#),
                ] {
                    let status = try await client.execute(uri: uri, method: .post, headers: Self.jsonHeaders,
                                                          body: ByteBuffer(string: body)) { $0.status }
                    #expect(status == .ok, "\(uri) \(model)")
                }
                let status = try await client.execute(uri: "/v1/models/load", method: .post, headers: Self.jsonHeaders,
                                                      body: ByteBuffer(string: #"{"model": "\#(model)"}"#)) { $0.status }
                #expect(status == .badRequest, "load needs a strict repo id: \(model)")
            }
        }
        #expect(await s.orchestrator.currentModelId == "org/mock-model")
    }
}
