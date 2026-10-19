import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltUI
@testable import BasaltCLI

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
            try await client.execute(uri: "/v1/responses", method: .post, headers: [.contentType: "application/json"], body: body(request)) { res in
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
        #expect(sent.prompt.contains("[Tool result for shell]: README.md"))
        #expect(sent.reasoningEffort == "low")
    }

    @Test("Responses stream ends with response.completed carrying usage, and no [DONE]")
    func responsesStreamProtocol() async throws {
        let (app, _) = try await app(reasoning: "thinking…")
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/responses", method: .post, headers: [.contentType: "application/json"],
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

                // Reasoning is item 0, the message item 1 — in the events and in the final output.
                #expect(types.contains("response.reasoning_text.delta"))
                let output = try #require(completed["output"] as? [[String: Any]])
                #expect(output.map { $0["type"] as? String } == ["reasoning", "message"])
                let part = try #require((output[1]["content"] as? [[String: Any]])?.first)
                #expect(part["type"] as? String == "output_text")
                #expect(part["text"] as? String == "Mock response")
                let textDeltas = evs.filter { $0.event == "response.output_text.delta" }
                #expect(textDeltas.allSatisfy { $0.data?["output_index"] as? Int == 1 })
            }
        }
    }

    @Test("Anthropic stream puts reasoning in a thinking block before the text block")
    func anthropicThinkingBlock() async throws {
        let (app, _) = try await app(reasoning: "let me think")
        let request = #"{"model": "local", "max_tokens": 64, "stream": true, "messages": [{"role": "user", "content": "hi"}]}"#
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/messages", method: .post, headers: [.contentType: "application/json"], body: body(request)) { res in
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
            try await client.execute(uri: "/v1/messages", method: .post, headers: [.contentType: "application/json"], body: body(request)) { res in
                let evs = events(res.body).compactMap(\.data)
                let starts = evs.filter { $0["type"] as? String == "content_block_start" }
                #expect(starts.count == 1)
                #expect(starts.first?["index"] as? Int == 0)
                #expect((starts.first?["content_block"] as? [String: Any])?["type"] as? String == "text")
            }
        }
    }

    @Test("Responses stream without reasoning puts the message at index 0")
    func responsesStreamTextOnly() async throws {
        let (app, _) = try await app()
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/responses", method: .post, headers: [.contentType: "application/json"],
                                     body: body(#"{"input": "hi", "stream": true}"#)) { res in
                let evs = events(res.body)
                let types = evs.compactMap(\.event)
                #expect(!types.contains("response.reasoning_text.delta"))
                #expect(types.filter { $0 == "response.output_item.added" }.count == 1, "\(types)")
                let added = evs.first { $0.event == "response.output_item.added" }
                #expect(added?.data?["output_index"] as? Int == 0, "\(String(describing: added?.raw))")
            }
        }
    }

    @Test("A tool result made of content blocks (e.g. with an image) decodes and reaches the prompt")
    func toolOutputAsBlocks() async throws {
        let (app, engine) = try await app()
        let request = #"""
        {"input": [
           {"type": "function_call_output", "call_id": "c1",
            "output": [{"type": "input_text", "text": "screenshot taken"}, {"type": "input_image", "image_url": "data:x"}]},
           {"type": "custom_tool_call", "name": "apply_patch", "input": "*** Begin Patch", "call_id": "c2"}
         ]}
        """#
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/responses", method: .post, headers: [.contentType: "application/json"], body: body(request)) { res in
                #expect(res.status == .ok)
            }
        }
        let prompt = try #require(await engine.lastRequest?.prompt)
        #expect(prompt.contains("[Tool result]: screenshot taken\n[image omitted]"))
        #expect(prompt.contains("[Assistant called apply_patch]: *** Begin Patch"))
    }

    @Test("Effort levels are mapped per template: Qwen gets xhigh/low, unknown templates the name as sent")
    func effortMappingPerProfile() {
        let request = { (effort: String?) in GenerationRequest(prompt: "p", reasoningEffort: effort) }
        let qwen = QwenProfile()
        #expect(MLXInferenceEngine.buildTemplateContext(request: request("high"), profile: qwen)?["reasoning_effort"] as? String == "xhigh")
        #expect(MLXInferenceEngine.buildTemplateContext(request: request("minimal"), profile: qwen)?["reasoning_effort"] as? String == "low")
        #expect(MLXInferenceEngine.buildTemplateContext(request: request("off"), profile: qwen)?["enable_thinking"] as? Bool == false)

        let generic = GenericProfile()
        #expect(MLXInferenceEngine.buildTemplateContext(request: request("high"), profile: generic)?["reasoning_effort"] as? String == "high")
        #expect(MLXInferenceEngine.buildTemplateContext(request: request("none"), profile: generic)?["enable_thinking"] as? Bool == false)
        #expect(MLXInferenceEngine.buildTemplateContext(request: request(nil), profile: generic) == nil)
    }

    @Test("x-service-request declares a service request; the heuristic only reads the latest turn")
    func serviceDetection() {
        var headers = HTTPFields()
        #expect(ServiceRequestDetector.prepare(prompt: "[User]: hello", maxTokens: 500, headers: headers).isService == false)
        headers[HTTPField.Name("x-service-request")!] = "true"
        #expect(ServiceRequestDetector.prepare(prompt: "[User]: hello", maxTokens: 500, headers: headers).isService)

        let history = "[User]: read docs/README.md\n[Tool result]: ...give this a short title...\n" + String(repeating: "code ", count: 600)
        #expect(!ServiceRequestDetector.isTitleOrServiceRequest(prompt: history + "\n[User]: now fix the bug", maxTokens: 4096))
        #expect(ServiceRequestDetector.isTitleOrServiceRequest(prompt: history + "\n[User]: Write a 3-5 word title for this session", maxTokens: 4096))
    }

    @Test("Anthropic stream reports the engine's token count")
    func anthropicUsageFromMetadata() async throws {
        let (app, _) = try await app()
        let request = #"{"model": "local", "max_tokens": 64, "stream": true, "messages": [{"role": "user", "content": "hi"}]}"#
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/messages", method: .post, headers: [.contentType: "application/json"], body: body(request)) { res in
                let delta = events(res.body).compactMap(\.data).first { $0["type"] as? String == "message_delta" }
                #expect((delta?["usage"] as? [String: Any])?["output_tokens"] as? Int == 10)
                #expect((delta?["delta"] as? [String: Any])?["stop_reason"] as? String == "end_turn")
            }
        }
    }

    // MARK: — Tool calling

    private let shellTool = #"""
    "tools": [
      {"type": "function", "name": "shell", "description": "Run a command",
       "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}},
      {"type": "local_shell"}
    ]
    """#

    @Test("Function tools reach the engine with their schema; built-in tools are skipped")
    func toolsReachEngine() async throws {
        let (app, engine) = try await app()
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/responses", method: .post, headers: [.contentType: "application/json"],
                                     body: body("{\"input\": \"list files\", \(shellTool)}")) { res in
                #expect(res.status == .ok)
            }
        }
        let tools = try #require(await engine.lastRequest?.tools)
        #expect(tools.map(\.name) == ["shell"])
        #expect(tools.first?.description == "Run a command")
        #expect(tools.first?.parametersJSON?.contains("\"cmd\"") == true)

        let spec = try #require(MLXInferenceEngine.toolSpecs(tools)?.first)
        #expect(spec["type"] as? String == "function")
        let function = try #require(spec["function"] as? [String: any Sendable])
        #expect(function["name"] as? String == "shell")
        let parameters = try #require(function["parameters"] as? [String: any Sendable])
        #expect((parameters["required"] as? [any Sendable])?.first as? String == "cmd")
    }

    @Test("tool_choice none keeps the tools in the prompt and disables calls; a named tool keeps every tool and is enforced by the gate; parallel can be turned off")
    func toolChoice() throws {
        let tools = #"[{"type": "function", "name": "shell"}, {"type": "function", "name": "read"}, {"name": "edit", "input_schema": {"type": "object"}}]"#
        let parse = { (extra: String) throws in
            try OpenAIController.requestedTools(fromRequestBody: Data("{\"tools\": \(tools), \(extra)}".utf8))
        }
        #expect(try parse(#""tool_choice": "auto""#).tools?.map(\.name) == ["shell", "read", "edit"])
        #expect(try parse(#""tool_choice": "auto""#).choice == nil)
        for none in [#""tool_choice": "none""#, #""tool_choice": {"type": "none"}"#] {
            #expect(try parse(none).tools?.count == 3)
            #expect(try parse(none).choice == ToolChoice(callsDisabled: true))
        }
        for (choice, name) in [(#"{"type": "tool", "name": "edit"}"#, "edit"),
                               (#"{"type": "function", "function": {"name": "read"}}"#, "read"),
                               (#"{"type": "function", "name": "shell"}"#, "shell")] {
            let requested = try parse("\"tool_choice\": \(choice)")
            #expect(requested.tools?.count == 3)
            #expect(requested.choice == ToolChoice(forcedTool: name))
        }
        #expect(throws: OpenAIController.UndeclaredForcedTool.self) { try parse(#""tool_choice": {"type": "tool", "name": "rm"}"#) }
        #expect(try parse(#""parallel_tool_calls": false"#).choice == ToolChoice(parallel: false))
        #expect(try parse(#""tool_choice": {"type": "auto", "disable_parallel_tool_use": true}"#).choice == ToolChoice(parallel: false))
    }

    @Test("Tool schemas are serialized with sorted keys, so the same tools render the same prompt")
    func toolSchemaSortedKeys() throws {
        let body = #"{"tools": [{"type": "function", "name": "f", "parameters": {"type": "object", "required": ["b"], "properties": {"b": {}}}}]}"#
        let schema = try #require(try OpenAIController.requestedTools(fromRequestBody: Data(body.utf8)).tools?.first?.parametersJSON)
        #expect(schema == #"{"properties":{"b":{}},"required":["b"],"type":"object"}"#)
    }

    @Test("The tool-call gate passes declared calls, turns undeclared or non-forced ones into a visible note, and drops extra calls when parallel is off")
    func toolCallGate() {
        let tools = [ToolDefinition(name: "shell"), ToolDefinition(name: "read")]
        let call = { (name: String) in ToolCallOutput(id: "c", name: name, arguments: "{}") }
        func names(_ chunks: [StreamChunk?]) -> [String] {
            chunks.map { chunk in
                switch chunk {
                case .toolCall(let c): return c.name
                case .text(let t): return t
                case nil: return "dropped"
                default: return "other"
                }
            }
        }
        var open = ToolCallGate(request: GenerationRequest(prompt: "p", tools: tools))
        #expect(names([open.admit(call("shell")), open.admit(call("read")), open.admit(call("rm"))])
                == ["shell", "read", "\n[tool call to rm rejected: undeclared]"])
        var none = ToolCallGate(request: GenerationRequest(prompt: "p"))
        #expect(names([none.admit(call("shell"))]) == ["\n[tool call to shell rejected: undeclared]"])
        var disabled = ToolCallGate(request: GenerationRequest(prompt: "p", tools: tools, toolChoice: ToolChoice(callsDisabled: true)))
        #expect(names([disabled.admit(call("shell"))]) == ["\n[tool call to shell rejected: tool_choice is none]"])
        var forced = ToolCallGate(request: GenerationRequest(prompt: "p", tools: tools, toolChoice: ToolChoice(forcedTool: "read")))
        #expect(names([forced.admit(call("shell")), forced.admit(call("read"))])
                == ["\n[tool call to shell rejected: tool_choice is read]", "read"])
        var single = ToolCallGate(request: GenerationRequest(prompt: "p", tools: tools, toolChoice: ToolChoice(parallel: false)))
        #expect(names([single.admit(call("shell")), single.admit(call("read"))]) == ["shell", "dropped"])
        var passthrough = ToolCallGate(request: GenerationRequest(prompt: "p"))
        #expect(names([passthrough.admit(StreamChunk.text("hi"))]) == ["hi"])
    }

    @Test("A tool call streams as a function_call item and is the whole output when there is no text")
    func toolCallStream() async throws {
        let (app, engine) = try await app()
        await engine.setMockToolCall(ToolCallOutput(id: "call_1", name: "shell", arguments: #"{"cmd":"ls"}"#), text: "")
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/responses", method: .post, headers: [.contentType: "application/json"],
                                     body: body("{\"input\": \"list files\", \"stream\": true, \(shellTool)}")) { res in
                let evs = events(res.body)
                let types = evs.compactMap(\.event)
                #expect(types.contains("response.function_call_arguments.done"), "\(types)")
                let added = evs.first { $0.event == "response.output_item.added" }?.data?["item"] as? [String: Any]
                #expect(added?["type"] as? String == "function_call")
                #expect(added?["call_id"] as? String == "call_1")

                let completed = try #require(evs.last?.data?["response"] as? [String: Any])
                let output = try #require(completed["output"] as? [[String: Any]])
                #expect(output.count == 1, "no empty message next to a pure tool call")
                #expect(output.first?["type"] as? String == "function_call")
                #expect(output.first?["name"] as? String == "shell")
                #expect(output.first?["arguments"] as? String == #"{"cmd":"ls"}"#)
            }
        }
    }

    @Test("Non-streaming responses return text and tool calls as separate output items")
    func toolCallNonStream() async throws {
        let (app, engine) = try await app()
        await engine.setMockToolCall(ToolCallOutput(id: "call_9", name: "shell", arguments: #"{"cmd":"pwd"}"#), text: "Checking.")
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/responses", method: .post, headers: [.contentType: "application/json"],
                                     body: body("{\"input\": \"where am i\", \(shellTool)}")) { res in
                let json = try #require(try JSONSerialization.jsonObject(with: Data(String(buffer: res.body).utf8)) as? [String: Any])
                let output = try #require(json["output"] as? [[String: Any]])
                #expect(output.map { $0["type"] as? String } == ["message", "function_call"])
                #expect(output[1]["call_id"] as? String == "call_9")
                #expect(output[1]["arguments"] as? String == #"{"cmd":"pwd"}"#)
            }
        }
    }

    /// Output items of a streamed turn where the model says `before`, calls a tool, then emits `after`.
    private func outputAroundToolCall(after: String) async throws -> [[String: Any]] {
        let (app, engine) = try await app()
        await engine.setMockToolCall(ToolCallOutput(id: "c1", name: "shell", arguments: "{}"), text: "Let me look.", after: after)
        let request = body("{\"input\": \"x\", \"stream\": true, \(shellTool)}")
        let raw: String = try await app.test(.router) { client in
            try await client.execute(uri: "/v1/responses", method: .post, headers: [.contentType: "application/json"], body: request) { res in
                String(buffer: res.body)
            }
        }
        let completed = events(ByteBuffer(string: raw)).last?.data?["response"] as? [String: Any]
        return completed?["output"] as? [[String: Any]] ?? []
    }

    private func messageTexts(_ output: [[String: Any]]) -> [String] {
        output.filter { $0["type"] as? String == "message" }
            .compactMap { ($0["content"] as? [[String: Any]])?.first?["text"] as? String }
    }

    @Test("Text after a tool call is a new message with its own id, not a repeat of the first")
    func textAfterToolCall() async throws {
        let output = try await outputAroundToolCall(after: "Done.")
        let types = output.map { $0["type"] as? String ?? "" }
        #expect(types == ["message", "function_call", "message"])
        #expect(messageTexts(output) == ["Let me look.", "Done."])
        let ids = output.filter { $0["type"] as? String == "message" }.compactMap { $0["id"] as? String }
        #expect(Set(ids).count == 2)
    }

    @Test("Whitespace the template leaves after a tool call does not become a message")
    func whitespaceAfterToolCall() async throws {
        let output = try await outputAroundToolCall(after: "\n")
        let types = output.map { $0["type"] as? String ?? "" }
        #expect(types == ["message", "function_call"])
        #expect(messageTexts(output) == ["Let me look."])
    }
}

@Suite("Tool calling on /v1/messages and /v1/chat/completions (#73)")
struct ToolCallingEndpointsTests {

    private func app(call: ToolCallOutput? = nil, calls: [ToolCallOutput] = [], text: String = "", reasoning: String? = nil) async throws -> (Application<RouterResponder<BasicRequestContext>>, MockInferenceEngine) {
        let engine = MockInferenceEngine()
        await engine.setMockReasoning(reasoning)
        if call != nil || !calls.isEmpty { await engine.setMockToolCalls((call.map { [$0] } ?? []) + calls, text: text) }
        let orch = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orch.loadModel(path: "mock")
        let server = RESTServer(orchestrator: orch, config: try ServerConfig(modelPath: "mock"))
        return (Application(router: server.makeRouter()), engine)
    }

    private func post(_ app: Application<RouterResponder<BasicRequestContext>>, _ uri: String, _ json: String) async throws -> String {
        try await app.test(.router) { client in
            try await client.execute(uri: uri, method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: json)) { res in
                #expect(res.status == .ok)
                return String(buffer: res.body)
            }
        }
    }

    private func object(_ raw: String) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any])
    }

    private func dataFrames(_ raw: String) -> [[String: Any]] {
        raw.components(separatedBy: "\n\n").compactMap { frame in
            guard let line = frame.split(separator: "\n").first(where: { $0.hasPrefix("data: ") }) else { return nil }
            return try? JSONSerialization.jsonObject(with: Data(line.dropFirst(6).utf8)) as? [String: Any]
        }
    }

    private let call = ToolCallOutput(id: "call_abc", name: "Bash", arguments: #"{"command":"ls"}"#)
    private let second = ToolCallOutput(id: "call_def", name: "Read", arguments: #"{"file_path":"a.txt"}"#)

    private func status(_ app: Application<RouterResponder<BasicRequestContext>>, _ uri: String, _ json: String) async throws -> HTTPResponse.Status {
        try await app.test(.router) { client in
            try await client.execute(uri: uri, method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: json)) { $0.status }
        }
    }

    private let anthropicTools = #"""
    "tools": [{"name": "Bash", "description": "Run a shell command",
               "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
              {"name": "Read", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}}]
    """#

    @Test("Messages: tools reach the engine; tool_use and tool_result history is rendered into the prompt")
    func messagesHistory() async throws {
        let (app, engine) = try await app()
        let request = #"""
        {"model": "local", "max_tokens": 64, "messages": [
          {"role": "user", "content": "list files"},
          {"role": "assistant", "content": [
            {"type": "text", "text": "Listing."},
            {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}]},
          {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.txt\nb.txt"},
            {"type": "tool_result", "tool_use_id": "toolu_2", "is_error": true, "content": [{"type": "text", "text": "denied"}]},
            {"type": "text", "text": "how many?"}]}
        ], 
        """# + anthropicTools + "}"
        _ = try await post(app, "/v1/messages", request)
        let sent = try #require(await engine.lastRequest)
        #expect(sent.tools?.map(\.name) == ["Bash", "Read"])
        #expect(sent.tools?.first?.parametersJSON?.contains("\"command\"") == true)
        let expected = [
            "[User]: list files",
            "[Assistant]: Listing.",
            #"[Assistant called Bash]: {"command":"ls"}"#,
            "[Tool result for Bash]: a.txt\nb.txt",
            "[Tool error]: denied",
            "[User]: how many?"
        ].joined(separator: "\n\n")
        #expect(sent.prompt == expected)
    }

    @Test("Messages: tool_choice none keeps the tools in the prompt and disables calls")
    func messagesToolChoiceNone() async throws {
        let (app, engine) = try await app()
        let request = #"{"max_tokens": 64, "tool_choice": {"type": "none"}, "messages": [{"role": "user", "content": "hi"}], "# + anthropicTools + "}"
        _ = try await post(app, "/v1/messages", request)
        #expect(await engine.lastRequest?.tools?.count == 2)
        #expect(await engine.lastRequest?.toolChoice == ToolChoice(callsDisabled: true))
    }

    @Test("Messages non-streaming: a tool call is a tool_use block with stop_reason tool_use")
    func messagesNonStream() async throws {
        let (app, _) = try await app(call: call, text: "Let me check.")
        let json = try object(try await post(app, "/v1/messages",
            #"{"max_tokens": 64, "messages": [{"role": "user", "content": "ls"}], "# + anthropicTools + "}"))
        #expect(json["stop_reason"] as? String == "tool_use")
        let content = try #require(json["content"] as? [[String: Any]])
        #expect(content.map { $0["type"] as? String } == ["text", "tool_use"])
        #expect(content[0]["text"] as? String == "Let me check.")
        #expect(content[1]["id"] as? String == "toolu_abc")
        #expect(content[1]["name"] as? String == "Bash")
        #expect((content[1]["input"] as? [String: Any])?["command"] as? String == "ls")
    }

    @Test("Messages non-streaming: a pure tool call has no empty text block")
    func messagesNonStreamToolOnly() async throws {
        let (app, _) = try await app(call: call)
        let json = try object(try await post(app, "/v1/messages",
            #"{"max_tokens": 64, "messages": [{"role": "user", "content": "ls"}], "# + anthropicTools + "}"))
        #expect((json["content"] as? [[String: Any]])?.map { $0["type"] as? String } == ["tool_use"])
    }

    @Test("Messages stream: thinking, then tool_use with input_json_delta, then message_delta stop_reason tool_use")
    func messagesStream() async throws {
        let (app, _) = try await app(call: call, reasoning: "need ls")
        let raw = try await post(app, "/v1/messages",
            #"{"max_tokens": 64, "stream": true, "messages": [{"role": "user", "content": "ls"}], "# + anthropicTools + "}")
        let evs = dataFrames(raw)
        let types = evs.compactMap { $0["type"] as? String }
        #expect(types == ["message_start", "ping",
                          "content_block_start", "content_block_delta", "content_block_stop",
                          "content_block_start", "content_block_delta", "content_block_stop",
                          "message_delta", "message_stop"], "\(types)")
        let starts = evs.filter { $0["type"] as? String == "content_block_start" }
        let toolStart = try #require(starts.last)
        #expect(toolStart["index"] as? Int == 1)
        let block = try #require(toolStart["content_block"] as? [String: Any])
        #expect(block["type"] as? String == "tool_use")
        #expect(block["id"] as? String == "toolu_abc")
        #expect(block["name"] as? String == "Bash")
        #expect((block["input"] as? [String: Any])?.isEmpty == true)
        let delta = try #require(evs.filter { $0["type"] as? String == "content_block_delta" }.last?["delta"] as? [String: Any])
        #expect(delta["type"] as? String == "input_json_delta")
        #expect(delta["partial_json"] as? String == #"{"command":"ls"}"#)
        let stops = evs.filter { $0["type"] as? String == "content_block_stop" }.compactMap { $0["index"] as? Int }
        #expect(stops == [0, 1])
        let messageDelta = try #require(evs.first { $0["type"] as? String == "message_delta" })
        #expect((messageDelta["delta"] as? [String: Any])?["stop_reason"] as? String == "tool_use")
    }

    @Test("Messages stream: text before a tool call keeps its own block")
    func messagesStreamTextThenTool() async throws {
        let (app, _) = try await app(call: call, text: "Checking.")
        let raw = try await post(app, "/v1/messages",
            #"{"max_tokens": 64, "stream": true, "messages": [{"role": "user", "content": "ls"}], "# + anthropicTools + "}")
        let starts = dataFrames(raw).filter { $0["type"] as? String == "content_block_start" }
        #expect(starts.compactMap { ($0["content_block"] as? [String: Any])?["type"] as? String } == ["text", "tool_use"])
        #expect(starts.compactMap { $0["index"] as? Int } == [0, 1])
    }

    private let openAITools = #"""
    "tools": [{"type": "function", "function": {"name": "Bash", "description": "Run a shell command",
               "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}},
              {"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}}]
    """#

    @Test("Chat: tools reach the engine; assistant tool_calls and tool messages are rendered into the prompt")
    func chatHistory() async throws {
        let (app, engine) = try await app()
        let request = #"""
        {"messages": [
          {"role": "user", "content": "list files"},
          {"role": "assistant", "content": null,
           "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "Bash", "arguments": "{\"command\":\"ls\"}"}}]},
          {"role": "tool", "tool_call_id": "call_1", "content": "a.txt\nb.txt"},
          {"role": "user", "content": "how many?"}
        ], 
        """# + openAITools + "}"
        _ = try await post(app, "/v1/chat/completions", request)
        let sent = try #require(await engine.lastRequest)
        #expect(sent.tools?.map(\.name) == ["Bash", "Read"])
        let expected = [
            "[User]: list files",
            #"[Assistant called Bash]: {"command":"ls"}"#,
            "[Tool result for Bash]: a.txt\nb.txt",
            "[User]: how many?"
        ].joined(separator: "\n\n")
        #expect(sent.prompt == expected)
    }

    @Test("Chat: tool_choice none keeps the tools in the prompt and disables calls")
    func chatToolChoiceNone() async throws {
        let (app, engine) = try await app()
        _ = try await post(app, "/v1/chat/completions",
            #"{"tool_choice": "none", "messages": [{"role": "user", "content": "hi"}], "# + openAITools + "}")
        #expect(await engine.lastRequest?.tools?.count == 2)
        #expect(await engine.lastRequest?.toolChoice == ToolChoice(callsDisabled: true))
    }

    @Test("Chat non-streaming: message.tool_calls with finish_reason tool_calls")
    func chatNonStream() async throws {
        let (app, _) = try await app(call: call)
        let json = try object(try await post(app, "/v1/chat/completions",
            #"{"messages": [{"role": "user", "content": "ls"}], "# + openAITools + "}"))
        let choice = try #require((json["choices"] as? [[String: Any]])?.first)
        #expect(choice["finish_reason"] as? String == "tool_calls")
        let message = try #require(choice["message"] as? [String: Any])
        #expect(message.keys.contains("content"))
        #expect(message["content"] is NSNull)
        let calls = try #require(message["tool_calls"] as? [[String: Any]])
        #expect(calls.count == 1)
        #expect(calls[0]["id"] as? String == "call_abc")
        #expect(calls[0]["type"] as? String == "function")
        let function = try #require(calls[0]["function"] as? [String: Any])
        #expect(function["name"] as? String == "Bash")
        #expect(function["arguments"] as? String == #"{"command":"ls"}"#)
    }

    @Test("Chat stream: delta.tool_calls with index, id, name and arguments; final finish_reason tool_calls")
    func chatStream() async throws {
        let (app, _) = try await app(call: call, text: "Checking.")
        let raw = try await post(app, "/v1/chat/completions",
            #"{"stream": true, "messages": [{"role": "user", "content": "ls"}], "# + openAITools + "}")
        #expect(raw.hasSuffix("data: [DONE]\n\n"))
        let choices = dataFrames(raw).compactMap { ($0["choices"] as? [[String: Any]])?.first }
        let deltas = choices.compactMap { $0["delta"] as? [String: Any] }
        #expect(deltas.first?["content"] as? String == "Checking.")
        let toolCalls = deltas.compactMap { $0["tool_calls"] as? [[String: Any]] }
        let toolDelta = try #require(toolCalls.first?.first)
        #expect(toolDelta["index"] as? Int == 0)
        #expect(toolDelta["id"] as? String == "call_abc")
        #expect(toolDelta["type"] as? String == "function")
        let function = try #require(toolDelta["function"] as? [String: Any])
        #expect(function["name"] as? String == "Bash")
        #expect(function["arguments"] as? String == #"{"command":"ls"}"#)
        #expect(choices.last?["finish_reason"] as? String == "tool_calls")
        #expect(choices.dropLast().allSatisfy { $0["finish_reason"] is NSNull })
    }

    @Test("Chat stream reports finish_reason length when generation hit the cap")
    func chatStreamLength() async throws {
        let (app, engine) = try await app()
        await engine.setMockFinishReason(.length)
        let raw = try await post(app, "/v1/chat/completions", #"{"stream": true, "messages": [{"role": "user", "content": "hi"}]}"#)
        let last = dataFrames(raw).compactMap { ($0["choices"] as? [[String: Any]])?.first }.last
        #expect(last?["finish_reason"] as? String == "length")
    }

    @Test("Messages: parallel tool results are matched to their calls by tool_use_id, not by order; images become a visible note")
    func messagesParallelHistory() async throws {
        let (app, engine) = try await app()
        let request = #"""
        {"max_tokens": 64, "messages": [
          {"role": "user", "content": "go"},
          {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_a", "name": "Bash", "input": {"command": "ls"}},
            {"type": "tool_use", "id": "toolu_b", "name": "Read", "input": {"file_path": "a.png"}}]},
          {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_b", "content": [{"type": "image", "source": {"type": "base64", "data": "x"}}]},
            {"type": "tool_result", "tool_use_id": "toolu_a", "content": "a.png"}]}
        ], 
        """# + anthropicTools + "}"
        _ = try await post(app, "/v1/messages", request)
        let prompt = try #require(await engine.lastRequest?.prompt)
        #expect(prompt.hasSuffix("[Tool result for Read]: [image omitted]\n\n[Tool result for Bash]: a.png"), "\(prompt)")
    }

    @Test("Chat: parallel tool results are matched by tool_call_id")
    func chatParallelHistory() async throws {
        let (app, engine) = try await app()
        let request = #"""
        {"messages": [
          {"role": "user", "content": "go"},
          {"role": "assistant", "content": null, "tool_calls": [
            {"id": "call_a", "type": "function", "function": {"name": "Bash", "arguments": "{}"}},
            {"id": "call_b", "type": "function", "function": {"name": "Read", "arguments": "{}"}}]},
          {"role": "tool", "tool_call_id": "call_b", "content": "contents"},
          {"role": "tool", "tool_call_id": "call_a", "content": "a.txt"}
        ], 
        """# + openAITools + "}"
        _ = try await post(app, "/v1/chat/completions", request)
        let prompt = try #require(await engine.lastRequest?.prompt)
        #expect(prompt.hasSuffix("[Tool result for Read]: contents\n\n[Tool result for Bash]: a.txt"), "\(prompt)")
    }

    @Test("Messages non-streaming: two calls are two tool_use blocks")
    func messagesTwoCallsNonStream() async throws {
        let (app, _) = try await app(calls: [call, second])
        let json = try object(try await post(app, "/v1/messages",
            #"{"max_tokens": 64, "messages": [{"role": "user", "content": "ls"}], "# + anthropicTools + "}"))
        let content = try #require(json["content"] as? [[String: Any]])
        #expect(content.compactMap { $0["name"] as? String } == ["Bash", "Read"])
        #expect(content.compactMap { $0["id"] as? String } == ["toolu_abc", "toolu_def"])
        #expect(json["stop_reason"] as? String == "tool_use")
    }

    @Test("Messages stream: two calls are two tool_use blocks at indices 0 and 1")
    func messagesTwoCallsStream() async throws {
        let (app, _) = try await app(calls: [call, second])
        let raw = try await post(app, "/v1/messages",
            #"{"max_tokens": 64, "stream": true, "messages": [{"role": "user", "content": "ls"}], "# + anthropicTools + "}")
        let evs = dataFrames(raw)
        let starts = evs.filter { $0["type"] as? String == "content_block_start" }
        #expect(starts.compactMap { $0["index"] as? Int } == [0, 1])
        #expect(starts.compactMap { ($0["content_block"] as? [String: Any])?["name"] as? String } == ["Bash", "Read"])
        let deltas = evs.filter { $0["type"] as? String == "content_block_delta" }
        #expect(deltas.compactMap { $0["index"] as? Int } == [0, 1])
        #expect(deltas.compactMap { ($0["delta"] as? [String: Any])?["partial_json"] as? String } == [call.arguments, second.arguments])
    }

    @Test("Chat non-streaming: two calls are two message.tool_calls entries")
    func chatTwoCallsNonStream() async throws {
        let (app, _) = try await app(calls: [call, second])
        let json = try object(try await post(app, "/v1/chat/completions",
            #"{"messages": [{"role": "user", "content": "ls"}], "# + openAITools + "}"))
        let message = try #require(((json["choices"] as? [[String: Any]])?.first)?["message"] as? [String: Any])
        let calls = try #require(message["tool_calls"] as? [[String: Any]])
        #expect(calls.compactMap { $0["id"] as? String } == ["call_abc", "call_def"])
    }

    @Test("Chat stream: two calls are delta.tool_calls with indices 0 and 1")
    func chatTwoCallsStream() async throws {
        let (app, _) = try await app(calls: [call, second])
        let raw = try await post(app, "/v1/chat/completions",
            #"{"stream": true, "messages": [{"role": "user", "content": "ls"}], "# + openAITools + "}")
        let deltas = dataFrames(raw).compactMap { (($0["choices"] as? [[String: Any]])?.first)?["delta"] as? [String: Any] }
        let toolCalls = deltas.compactMap { ($0["tool_calls"] as? [[String: Any]])?.first }
        #expect(toolCalls.compactMap { $0["index"] as? Int } == [0, 1])
        #expect(toolCalls.compactMap { $0["id"] as? String } == ["call_abc", "call_def"])
        #expect(toolCalls.compactMap { ($0["function"] as? [String: Any])?["name"] as? String } == ["Bash", "Read"])
    }

    @Test("parallel_tool_calls false and disable_parallel_tool_use pass only the first call")
    func parallelOff() async throws {
        let (app, _) = try await app(calls: [call, second])
        let chat = try object(try await post(app, "/v1/chat/completions",
            #"{"parallel_tool_calls": false, "messages": [{"role": "user", "content": "ls"}], "# + openAITools + "}"))
        let message = try #require(((chat["choices"] as? [[String: Any]])?.first)?["message"] as? [String: Any])
        #expect((message["tool_calls"] as? [[String: Any]])?.count == 1)
        let messages = try object(try await post(app, "/v1/messages",
            #"{"max_tokens": 64, "tool_choice": {"type": "auto", "disable_parallel_tool_use": true}, "messages": [{"role": "user", "content": "ls"}], "# + anthropicTools + "}"))
        #expect((messages["content"] as? [[String: Any]])?.compactMap { $0["name"] as? String } == ["Bash"])
    }

    @Test("A call to a tool outside tool_choice reaches the client as the visible rejection note, not as a call")
    func undeclaredCallBecomesNote() async throws {
        let (app, engine) = try await app(call: call)
        let json = try object(try await post(app, "/v1/messages",
            #"{"max_tokens": 64, "tool_choice": {"type": "tool", "name": "Read"}, "messages": [{"role": "user", "content": "ls"}], "# + anthropicTools + "}"))
        #expect(await engine.lastRequest?.tools?.map(\.name) == ["Bash", "Read"])
        #expect(json["stop_reason"] as? String == "end_turn")
        let content = try #require(json["content"] as? [[String: Any]])
        #expect(content.map { $0["type"] as? String } == ["text"])
        #expect((content.first?["text"] as? String)?.hasSuffix("[tool call to Bash rejected: tool_choice is Read]") == true)

        let chat = try object(try await post(app, "/v1/chat/completions",
            #"{"tool_choice": "none", "messages": [{"role": "user", "content": "ls"}], "# + openAITools + "}"))
        let choice = try #require((chat["choices"] as? [[String: Any]])?.first)
        #expect(choice["finish_reason"] as? String == "stop")
        #expect(((choice["message"] as? [String: Any])?["content"] as? String)?.contains("[tool call to Bash rejected: tool_choice is none]") == true)
    }

    @Test("A forced tool that is not declared is a 400 on every endpoint")
    func forcedUndeclaredIs400() async throws {
        let (app, _) = try await app()
        #expect(try await status(app, "/v1/messages",
            #"{"max_tokens": 64, "tool_choice": {"type": "tool", "name": "rm"}, "messages": [{"role": "user", "content": "x"}], "# + anthropicTools + "}") == .badRequest)
        #expect(try await status(app, "/v1/chat/completions",
            #"{"tool_choice": {"type": "function", "function": {"name": "rm"}}, "messages": [{"role": "user", "content": "x"}], "# + openAITools + "}") == .badRequest)
        #expect(try await status(app, "/v1/responses",
            #"{"input": "x", "tool_choice": {"type": "function", "name": "rm"}}"#) == .badRequest)
    }

    @Test("Messages: arguments that are not a JSON object go out as input {} with the original text in a visible note")
    func messagesNonObjectArguments() async throws {
        let bad = ToolCallOutput(id: "call_x", name: "Bash", arguments: #"["ls"]"#)
        let note = #"[tool call to Bash: arguments are not an object, sent as {}: ["ls"]]"#
        let (app, _) = try await app(call: bad)
        let json = try object(try await post(app, "/v1/messages",
            #"{"max_tokens": 64, "messages": [{"role": "user", "content": "ls"}], "# + anthropicTools + "}"))
        let content = try #require(json["content"] as? [[String: Any]])
        #expect(content.map { $0["type"] as? String } == ["text", "tool_use"])
        #expect((content[0]["text"] as? String)?.hasSuffix(note) == true)
        #expect((content[1]["input"] as? [String: Any])?.isEmpty == true)

        let raw = try await post(app, "/v1/messages",
            #"{"max_tokens": 64, "stream": true, "messages": [{"role": "user", "content": "ls"}], "# + anthropicTools + "}")
        let deltas = dataFrames(raw).filter { $0["type"] as? String == "content_block_delta" }.compactMap { $0["delta"] as? [String: Any] }
        #expect((deltas.first?["text"] as? String)?.hasSuffix(note) == true)
        #expect(deltas.last?["type"] as? String == "input_json_delta")
        #expect(deltas.last?["partial_json"] as? String == "{}")
    }
}
