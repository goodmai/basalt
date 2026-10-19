import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltServer

@Suite("Prompt cache namespaces and cached-token usage on the wire (#32, #56)")
struct PromptCacheAPITests {

    private func app(cachedTokens: Int = 0) async throws -> (Application<RouterResponder<BasicRequestContext>>, MockInferenceEngine) {
        let engine = MockInferenceEngine()
        await engine.setMockCachedPromptTokens(cachedTokens)
        let orch = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orch.loadModel(path: "mock")
        let server = RESTServer(orchestrator: orch, config: try ServerConfig(modelPath: "mock"))
        return (Application(router: server.makeRouter()), engine)
    }

    private func post(
        _ uri: String, _ json: String, headers: HTTPFields = [:], cachedTokens: Int = 0
    ) async throws -> (body: String, sent: GenerationRequest?) {
        let (app, engine) = try await app(cachedTokens: cachedTokens)
        let body = try await app.test(.router) { client in
            try await client.execute(uri: uri, method: .post, headers: headers, body: ByteBuffer(string: json)) { res in
                #expect(res.status == .ok)
                return String(buffer: res.body)
            }
        }
        return (body, await engine.lastRequest)
    }

    private func json(_ text: String) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
    }

    private func sseData(_ text: String, event: String) throws -> [String: Any] {
        let frame = try #require(text.components(separatedBy: "\n\n").first { $0.contains("event: \(event)\n") })
        let line = try #require(frame.split(separator: "\n").first { $0.hasPrefix("data: ") })
        return try json(String(line.dropFirst(6)))
    }

    private func header(_ name: String, _ value: String) -> HTTPFields {
        var fields = HTTPFields()
        fields[HTTPField.Name(name)!] = value
        return fields
    }

    @Test("Every non-service request lands in its credential's namespace; conversation ids no longer split the cache")
    func conversationIdsShareOneNamespace() async throws {
        let requests = [
            ("/v1/chat/completions", #"{"prompt_cache_key": "codex-1", "messages": [{"role": "user", "content": "hi"}]}"#),
            ("/v1/responses", #"{"prompt_cache_key": "codex-2", "input": "hi"}"#),
            ("/v1/responses", #"{"instructions": "Translate.", "input": "first"}"#),
            ("/v1/messages", #"{"max_tokens": 64, "metadata": {"user_id": "u_session_1"}, "messages": [{"role": "user", "content": "hi"}]}"#),
            ("/v1/messages", #"{"max_tokens": 64, "system": "You are an Explore subagent.", "messages": [{"role": "user", "content": "hi"}]}"#),
        ]
        for (uri, body) in requests {
            let anonymous = try await post(uri, body)
            #expect(anonymous.sent?.cacheKey == PromptCacheKey.anonymous, "\(uri)")
            let session = try await post(uri, body, headers: header("x-session-id", "s-7"))
            #expect(session.sent?.cacheKey == PromptCacheKey.anonymous, "\(uri)")
            let keyed = try await post(uri, body, headers: header("authorization", "Bearer alice-key"))
            #expect(keyed.sent?.cacheKey == PromptCacheKey.resolve(
                headers: header("x-api-key", "alice-key"), isService: false), "\(uri)")
        }
    }

    @Test("Namespaces follow the credential a client presents, so clients with different keys never share a prefix")
    func keysAreNamespacedByCredential() async throws {
        let body = #"{"input": "hi"}"#
        let alice = try await post("/v1/responses", body, headers: header("authorization", "Bearer alice-key"))
        let aliceAgain = try await post("/v1/responses", body, headers: header("x-api-key", "alice-key"))
        let bob = try await post("/v1/responses", body, headers: header("authorization", "Bearer bob-key"))
        let anonymous = try await post("/v1/responses", body)
        #expect(alice.sent?.cacheKey?.hasPrefix("credential:") == true)
        #expect(alice.sent?.cacheKey == aliceAgain.sent?.cacheKey)
        #expect(alice.sent?.cacheKey != bob.sent?.cacheKey)
        #expect(alice.sent?.cacheKey != anonymous.sent?.cacheKey)
        #expect(alice.sent?.cacheKey?.contains("alice") == false)
    }

    @Test("A cacheKey in a raw generate body is ignored, so it cannot select another client's namespace")
    func rawGenerateCannotPickAForeignNamespace() async throws {
        let alice = header("authorization", "Bearer alice-key")
        let bob = header("authorization", "Bearer bob-key")
        let aliceNamespace = try #require(try await post("/v1/responses", #"{"input": "hi"}"#, headers: alice).sent?.cacheKey)
        let bobNamespace = try #require(try await post("/v1/responses", #"{"input": "hi"}"#, headers: bob).sent?.cacheKey)
        for uri in ["/api/v1/generate", "/v1/generate", "/api/v1/generate/stream"] {
            let probe = try await post(uri, #"{"prompt": "hi", "cacheKey": "\#(aliceNamespace)"}"#, headers: bob)
            #expect(probe.sent?.cacheKey == bobNamespace, "\(uri)")
            let anonymous = try await post(uri, #"{"prompt": "hi", "cacheKey": "\#(aliceNamespace)"}"#)
            #expect(anonymous.sent?.cacheKey == PromptCacheKey.anonymous, "\(uri)")
        }
    }

    @Test("A service request (thread title) never touches the prompt cache")
    func serviceRequestHasNoKey() async throws {
        let result = try await post("/v1/messages",
            #"{"max_tokens": 64, "metadata": {"user_id": "u"}, "messages": [{"role": "user", "content": "hi"}]}"#,
            headers: header("x-service-request", "true"))
        #expect(result.sent?.cacheKey == nil)
    }

    @Test("Chat Completions reports usage.prompt_tokens_details.cached_tokens")
    func chatCompletionsUsage() async throws {
        let result = try await post("/v1/chat/completions",
            #"{"messages": [{"role": "user", "content": "hi"}]}"#, cachedTokens: 30)
        let usage = try #require(try json(result.body)["usage"] as? [String: Any])
        #expect((usage["prompt_tokens_details"] as? [String: Any])?["cached_tokens"] as? Int == 30)
        #expect(usage["prompt_tokens"] as? Int == 32)
    }

    @Test("Responses reports usage.input_tokens_details.cached_tokens, streaming and not")
    func responsesUsage() async throws {
        let plain = try await post("/v1/responses", #"{"input": "hi"}"#, cachedTokens: 30)
        let usage = try #require(try json(plain.body)["usage"] as? [String: Any])
        #expect((usage["input_tokens_details"] as? [String: Any])?["cached_tokens"] as? Int == 30)
        #expect(usage["input_tokens"] as? Int == 31)

        let streamed = try await post("/v1/responses", #"{"input": "hi", "stream": true}"#, cachedTokens: 30)
        let completed = try sseData(streamed.body, event: "response.completed")
        let streamedUsage = try #require((completed["response"] as? [String: Any])?["usage"] as? [String: Any])
        #expect((streamedUsage["input_tokens_details"] as? [String: Any])?["cached_tokens"] as? Int == 30)
    }

    @Test("Anthropic reports cache_read_input_tokens and counts only uncached tokens as input_tokens")
    func anthropicUsage() async throws {
        let request = #"{"max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}"#
        let plain = try await post("/v1/messages", request, cachedTokens: 30)
        let usage = try #require(try json(plain.body)["usage"] as? [String: Any])
        #expect(usage["cache_read_input_tokens"] as? Int == 30)
        #expect(usage["input_tokens"] as? Int == 2)

        let streamed = try await post("/v1/messages",
            #"{"max_tokens": 64, "stream": true, "messages": [{"role": "user", "content": "hi"}]}"#, cachedTokens: 30)
        let delta = try sseData(streamed.body, event: "message_delta")
        let streamedUsage = try #require(delta["usage"] as? [String: Any])
        #expect(streamedUsage["cache_read_input_tokens"] as? Int == 30)
        #expect(streamedUsage["input_tokens"] as? Int == 2)
    }

    @Test("With nothing reused every API reports zero cached tokens")
    func noReuseReportsZero() async throws {
        let result = try await post("/v1/chat/completions", #"{"messages": [{"role": "user", "content": "hi"}]}"#)
        let usage = try #require(try json(result.body)["usage"] as? [String: Any])
        #expect((usage["prompt_tokens_details"] as? [String: Any])?["cached_tokens"] as? Int == 0)
    }
}
