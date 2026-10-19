import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltServer

@Suite("Prompt cache keys and cached-token usage on the wire (#56)")
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

    @Test("prompt_cache_key keys the slot on Chat Completions and Responses")
    func promptCacheKeyIsUsed() async throws {
        let chat = try await post("/v1/chat/completions",
            #"{"prompt_cache_key": "codex-1", "messages": [{"role": "user", "content": "hi"}]}"#)
        #expect(chat.sent?.cacheKey == "key:codex-1")
        let responses = try await post("/v1/responses", #"{"prompt_cache_key": "codex-2", "input": "hi"}"#)
        #expect(responses.sent?.cacheKey == "key:codex-2")
    }

    @Test("metadata.user_id keys the slot on the Anthropic Messages API")
    func anthropicUserIdIsUsed() async throws {
        let result = try await post("/v1/messages",
            #"{"max_tokens": 64, "metadata": {"user_id": "user_abc_session_42"}, "messages": [{"role": "user", "content": "hi"}]}"#)
        #expect(result.sent?.cacheKey?.hasPrefix("key:user_abc_session_42:") == true)
    }

    @Test("Under one metadata.user_id, turns of a conversation share a slot and a subagent with another system prompt gets its own")
    func anthropicSubagentsDoNotShareTheMainSlot() async throws {
        func request(system: String, messages: String) -> String {
            #"{"max_tokens": 64, "metadata": {"user_id": "u_session_1"}, "system": "\#(system)", "messages": [\#(messages)]}"#
        }
        let main = String(repeating: "You are Claude Code, the main agent. ", count: 20)
        let explorer = String(repeating: "You are an Explore subagent. ", count: 20)
        let firstTurn = #"{"role": "user", "content": "fix the bug"}"#
        let laterTurn = firstTurn + #", {"role": "assistant", "content": "done"}, {"role": "user", "content": "now test it"}"#
        let mainFirst = try await post("/v1/messages", request(system: main, messages: firstTurn))
        let mainLater = try await post("/v1/messages", request(system: main, messages: laterTurn))
        let subagent = try await post("/v1/messages", request(system: explorer, messages: firstTurn))
        #expect(mainFirst.sent?.cacheKey != nil)
        #expect(mainFirst.sent?.cacheKey == mainLater.sent?.cacheKey)
        #expect(mainFirst.sent?.cacheKey != subagent.sent?.cacheKey)
    }

    @Test("Keys are namespaced by the credential a client presents, so clients with different keys never share a slot")
    func keysAreNamespacedByCredential() async throws {
        let body = #"{"prompt_cache_key": "shared", "input": "hi"}"#
        let alice = try await post("/v1/responses", body, headers: header("authorization", "Bearer alice-key"))
        let aliceAgain = try await post("/v1/responses", body, headers: header("x-api-key", "alice-key"))
        let bob = try await post("/v1/responses", body, headers: header("authorization", "Bearer bob-key"))
        #expect(alice.sent?.cacheKey?.hasSuffix("/key:shared") == true)
        #expect(alice.sent?.cacheKey == aliceAgain.sent?.cacheKey)
        #expect(alice.sent?.cacheKey != bob.sent?.cacheKey)
        #expect(alice.sent?.cacheKey?.contains("alice") == false)
    }

    @Test("x-session-id keys the slot when the body names none; a body key wins over it")
    func sessionHeaderIsUsed() async throws {
        let headers = header("x-session-id", "s-7")
        let plain = try await post("/v1/messages",
            #"{"max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}"#, headers: headers)
        #expect(plain.sent?.cacheKey == "session:s-7")
        let keyed = try await post("/v1/responses", #"{"prompt_cache_key": "k", "input": "hi"}"#, headers: headers)
        #expect(keyed.sent?.cacheKey == "key:k")
    }

    @Test("Without a key, requests sharing the system prompt share a slot and different ones do not")
    func systemPrefixKeysTheSlot() async throws {
        let system = String(repeating: "You are a coding agent with tools. ", count: 20)
        let first = try await post("/v1/responses", #"{"instructions": "\#(system)", "input": "first"}"#)
        let second = try await post("/v1/responses", #"{"instructions": "\#(system)", "input": "second"}"#)
        let other = try await post("/v1/responses", #"{"instructions": "Translate.", "input": "first"}"#)
        #expect(first.sent?.cacheKey?.hasPrefix("prefix:") == true)
        #expect(first.sent?.cacheKey == second.sent?.cacheKey)
        #expect(first.sent?.cacheKey != other.sent?.cacheKey)
    }

    @Test("A service request (thread title) never touches a conversation slot")
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
