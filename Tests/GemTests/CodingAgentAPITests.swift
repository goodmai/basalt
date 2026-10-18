import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
@testable import GemCore

/// Wire-format tests for the coding-agent endpoints, driven through the real router.
@Suite("Coding agent API wire format (#27)")
struct CodingAgentAPITests {

    private func app(reasoning: String? = nil) async throws -> (Application<RouterResponder<BasicRequestContext>>, MockInferenceEngine) {
        let engine = MockInferenceEngine()
        await engine.setMockReasoning(reasoning)
        let orch = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orch.loadModel(path: "mock")
        let server = RESTServer(orchestrator: orch, config: try ServerConfig(modelPath: "mock"))
        return (Application(router: server.makeRouter()), engine)
    }

    private func body(_ json: String) -> ByteBuffer { ByteBuffer(string: json) }

    /// Parses an SSE body into (event, JSON payload) pairs; `[DONE]` lines are returned as raw.
    private func events(_ buffer: ByteBuffer) -> [(event: String?, data: [String: Any]?, raw: String)] {
        String(buffer: buffer).components(separatedBy: "\n\n").compactMap { frame in
            guard !frame.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
            var event: String?
            var raw = ""
            for line in frame.split(separator: "\n") {
                if line.hasPrefix("event: ") { event = String(line.dropFirst(7)) }
                if line.hasPrefix("data: ") { raw = String(line.dropFirst(6)) }
            }
            let data = try? JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any]
            return (event, data, raw)
        }
    }

    @Test("Responses: instructions, reasoning.effort and tool-call history reach the engine")
    func responsesRequestMapping() async throws {
        let (app, engine) = try await app()
        let request = #"""
        {"model": "local", "instructions": "Be terse.", "reasoning": {"effort": "low"},
         "input": [
           {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list files"}]},
           {"type": "function_call", "name": "shell", "arguments": "{\"cmd\":\"ls\"}", "call_id": "c1"},
           {"type": "function_call_output", "call_id": "c1", "output": "README.md"}
         ]}
        """#
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/responses", method: .post, body: body(request)) { res in
                #expect(res.status == .ok)
                let json = try #require(try JSONSerialization.jsonObject(with: Data(String(buffer: res.body).utf8)) as? [String: Any])
                let output = try #require(json["output"] as? [[String: Any]])
                let content = try #require(output.first?["content"] as? [[String: Any]])
                #expect(content.first?["type"] as? String == "output_text")
            }
        }
        let sent = try #require(await engine.lastRequest)
        #expect(sent.prompt.hasPrefix("[System]: Be terse."))
        #expect(sent.prompt.contains("[User]: list files"))
        #expect(sent.prompt.contains(#"[Assistant called shell]: {"cmd":"ls"}"#))
        #expect(sent.prompt.contains("[Tool result]: README.md"))
        #expect(sent.reasoningEffort == "low")
    }

    @Test("Responses stream ends with response.completed carrying usage, and no [DONE]")
    func responsesStreamProtocol() async throws {
        let (app, _) = try await app(reasoning: "thinking…")
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/responses", method: .post,
                                     body: body(#"{"input": "hi", "stream": true}"#)) { res in
                #expect(res.status == .ok)
                let evs = events(res.body)
                let types = evs.compactMap(\.event)
                #expect(types.first == "response.created")
                #expect(types.last == "response.completed")
                #expect(types.contains("response.output_text.delta"))
                #expect(!types.contains("response.done"))
                #expect(!types.contains("response.text.delta"))
                #expect(!evs.contains { $0.raw == "[DONE]" })

                let seqs = evs.compactMap { $0.data?["sequence_number"] as? Int }
                #expect(seqs == Array(0..<seqs.count))

                let completed = try #require(evs.last?.data?["response"] as? [String: Any])
                #expect(completed["status"] as? String == "completed")
                let usage = try #require(completed["usage"] as? [String: Any])
                #expect(usage["output_tokens"] as? Int == 10)
                let item = try #require((completed["output"] as? [[String: Any]])?.first)
                let part = try #require((item["content"] as? [[String: Any]])?.first)
                #expect(part["type"] as? String == "output_text")
                #expect(part["text"] as? String == "Mock response")
            }
        }
    }

    @Test("Anthropic stream puts reasoning in a thinking block before the text block")
    func anthropicThinkingBlock() async throws {
        let (app, _) = try await app(reasoning: "let me think")
        let request = #"{"model": "local", "max_tokens": 64, "stream": true, "messages": [{"role": "user", "content": "hi"}]}"#
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/messages", method: .post, body: body(request)) { res in
                #expect(res.status == .ok)
                let evs = events(res.body).compactMap(\.data)
                let starts = evs.filter { $0["type"] as? String == "content_block_start" }
                #expect(starts.count == 2)
                #expect((starts.first?["content_block"] as? [String: Any])?["type"] as? String == "thinking")
                #expect(starts.first?["index"] as? Int == 0)
                #expect((starts.last?["content_block"] as? [String: Any])?["type"] as? String == "text")
                #expect(starts.last?["index"] as? Int == 1)

                let deltas = evs.filter { $0["type"] as? String == "content_block_delta" }
                for d in deltas {
                    let kind = (d["delta"] as? [String: Any])?["type"] as? String
                    #expect(d["index"] as? Int == (kind == "thinking_delta" ? 0 : 1))
                }
                let stops = evs.filter { $0["type"] as? String == "content_block_stop" }.compactMap { $0["index"] as? Int }
                #expect(stops == [0, 1])
            }
        }
    }

    @Test("Anthropic stream without reasoning has exactly one text block at index 0")
    func anthropicTextOnly() async throws {
        let (app, _) = try await app()
        let request = #"{"model": "local", "max_tokens": 64, "stream": true, "messages": [{"role": "user", "content": "hi"}]}"#
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/messages", method: .post, body: body(request)) { res in
                let evs = events(res.body).compactMap(\.data)
                let starts = evs.filter { $0["type"] as? String == "content_block_start" }
                #expect(starts.count == 1)
                #expect(starts.first?["index"] as? Int == 0)
                #expect((starts.first?["content_block"] as? [String: Any])?["type"] as? String == "text")
            }
        }
    }
}
