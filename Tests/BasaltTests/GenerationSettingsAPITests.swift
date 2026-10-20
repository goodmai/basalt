import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Generation settings over the wire (#105)")
struct GenerationSettingsAPITests {

    private func app(reasoning: String? = nil) async throws -> (Application<RouterResponder<BasicRequestContext>>, MockInferenceEngine) {
        let engine = MockInferenceEngine()
        await engine.setMockReasoning(reasoning)
        let orch = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orch.loadModel(path: "mock")
        let server = RESTServer(orchestrator: orch, config: try ServerConfig(modelPath: "mock"))
        return (Application(router: server.makeRouter()), engine)
    }

    private func post(_ app: Application<RouterResponder<BasicRequestContext>>, _ uri: String, _ json: String,
                      _ check: @escaping @Sendable (TestResponse) throws -> Void) async throws {
        try await app.test(.router) { client in
            try await client.execute(uri: uri, method: .post, headers: [.contentType: "application/json"],
                                     body: ByteBuffer(string: json)) { try check($0) }
        }
    }

    private func object(_ response: TestResponse) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: Data(String(buffer: response.body).utf8)) as? [String: Any])
    }

    private func dataFrames(_ response: TestResponse) -> [[String: Any]] {
        String(buffer: response.body).components(separatedBy: "\n\n").compactMap { frame in
            guard let line = frame.split(separator: "\n").first(where: { $0.hasPrefix("data: ") }) else { return nil }
            return try? JSONSerialization.jsonObject(with: Data(line.dropFirst(6).utf8)) as? [String: Any]
        }
    }

    @Test("Chat completions: top_k, presence_penalty, repetition_penalty and thinking_budget reach the engine; omitted temperature and top_p stay unset")
    func chatCompletionsMapsSettings() async throws {
        let (app, engine) = try await app()
        try await post(app, "/v1/chat/completions", #"""
            {"messages": [{"role": "user", "content": "hi"}], "top_k": 20, "presence_penalty": 1.5,
             "repetition_penalty": 1.05, "thinking_budget": 5000, "max_tokens": 8000}
            """#) { #expect($0.status == .ok) }
        let sent = try #require(await engine.lastRequest)
        #expect(sent.topK == 20)
        #expect(sent.presencePenalty == 1.5)
        #expect(sent.repetitionPenalty == 1.05)
        #expect(sent.thinkingBudget == 5000)
        #expect(sent.temperature == nil)
        #expect(sent.topP == nil)
    }

    @Test("Responses: the same settings reach the engine")
    func responsesMapsSettings() async throws {
        let (app, engine) = try await app()
        try await post(app, "/v1/responses", #"{"input": "hi", "top_k": 7, "presence_penalty": 0.5, "thinking_budget": 300}"#) {
            #expect($0.status == .ok)
        }
        let sent = try #require(await engine.lastRequest)
        #expect(sent.topK == 7)
        #expect(sent.presencePenalty == 0.5)
        #expect(sent.thinkingBudget == 300)
        #expect(sent.temperature == nil)
    }

    @Test("An out-of-range parameter is a 400 before a stream opens", arguments: [
        "/v1/chat/completions", "/v1/responses", "/v1/messages",
    ])
    func outOfRangeIsRejectedBeforeStreaming(_ uri: String) async throws {
        let (app, engine) = try await app()
        let body = uri == "/v1/messages"
            ? #"{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100, "top_k": -1, "stream": true}"#
            : #"{"messages": [{"role": "user", "content": "hi"}], "input": "hi", "presence_penalty": 3, "stream": true}"#
        try await post(app, uri, body) { #expect($0.status == .badRequest) }
        #expect(await engine.lastRequest == nil)
    }

    @Test("Anthropic: thinking.budget_tokens becomes the thinking budget only when thinking is enabled; top_k passes through")
    func anthropicThinkingBudget() async throws {
        let (app, engine) = try await app()
        try await post(app, "/v1/messages", #"""
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 6000, "top_k": 20,
             "thinking": {"type": "enabled", "budget_tokens": 4000}}
            """#) { #expect($0.status == .ok) }
        let enabled = try #require(await engine.lastRequest)
        #expect(enabled.thinkingBudget == 4000)
        #expect(enabled.topK == 20)
        #expect(enabled.temperature == nil)

        try await post(app, "/v1/messages", #"""
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 6000,
             "thinking": {"type": "disabled", "budget_tokens": 4000}}
            """#) { #expect($0.status == .ok) }
        #expect(await engine.lastRequest?.thinkingBudget == nil)
    }

    @Test("Usage reports reasoning tokens: chat completion_tokens_details, the include_usage stream chunk, Responses output_tokens_details")
    func usageReportsReasoningTokens() async throws {
        let (app, _) = try await app(reasoning: "one two three")
        try await post(app, "/v1/chat/completions", #"{"messages": [{"role": "user", "content": "hi"}]}"#) { res in
            let usage = try #require(try self.object(res)["usage"] as? [String: Any])
            #expect((usage["completion_tokens_details"] as? [String: Any])?["reasoning_tokens"] as? Int == 3)
        }
        try await post(app, "/v1/chat/completions", #"""
            {"messages": [{"role": "user", "content": "hi"}], "stream": true, "stream_options": {"include_usage": true}}
            """#) { res in
            let usage = try #require(self.dataFrames(res).compactMap { $0["usage"] as? [String: Any] }.last)
            #expect((usage["completion_tokens_details"] as? [String: Any])?["reasoning_tokens"] as? Int == 3)
            #expect(usage["completion_tokens"] as? Int == 10)
        }
        try await post(app, "/v1/chat/completions", #"{"messages": [{"role": "user", "content": "hi"}], "stream": true}"#) { res in
            #expect(self.dataFrames(res).allSatisfy { $0["usage"] == nil })
        }
        try await post(app, "/v1/responses", #"{"input": "hi"}"#) { res in
            let usage = try #require(try self.object(res)["usage"] as? [String: Any])
            #expect((usage["output_tokens_details"] as? [String: Any])?["reasoning_tokens"] as? Int == 3)
        }
        try await post(app, "/v1/responses", #"{"input": "hi", "stream": true}"#) { res in
            let completed = try #require(self.dataFrames(res).last { $0["type"] as? String == "response.completed" })
            let usage = try #require((completed["response"] as? [String: Any])?["usage"] as? [String: Any])
            #expect((usage["output_tokens_details"] as? [String: Any])?["reasoning_tokens"] as? Int == 3)
        }
    }
}
