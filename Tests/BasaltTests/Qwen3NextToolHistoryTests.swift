import Foundation
import Testing
import Jinja
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Qwen3-Coder-Next history re-render: tool calls of a /v1/messages or /v1/chat/completions request reach the real chat template as decoded parameters and parse back to the same calls (#642 P4)")
struct Qwen3NextToolHistoryTests {
    enum Shape: String, CaseIterable, Sendable {
        case messages
        case chatCompletions
    }

    static let path = "Sources/App/Budget.swift"
    static let code = "    func budget(_ items: [Int]) -> Int {\n\tlet total = items.reduce(0, +) \n\n        return total * 2\n    }\n"
    static let readJSON = #"{"file_path": "Sources/App/Budget.swift", "offset": 42, "limit": 20}"#
    static let tools = [Qwen3NextTools.edit, Qwen3NextTools.read]

    static let expectedTail = [
        "<|im_start|>user\nFix the budget.<|im_end|>\n<|im_start|>assistant\nFixing it.\n\n",
        "<tool_call>\n<function=Edit>\n<parameter=file_path>\n\(path)\n</parameter>\n<parameter=new_string>\n\(code)\n</parameter>\n",
        "<parameter=old_string>\nreturn total\n</parameter>\n<parameter=replace_all>\ntrue\n</parameter>\n</function>\n</tool_call>\n",
        "<tool_call>\n<function=Read>\n<parameter=file_path>\n\(path)\n</parameter>\n<parameter=limit>\n20\n</parameter>\n",
        "<parameter=offset>\n42\n</parameter>\n</function>\n</tool_call><|im_end|>\n",
        "<|im_start|>user\n<tool_response>\nupdated\n</tool_response>\n<tool_response>\n42: return total\n</tool_response><|im_end|>\n",
        "<|im_start|>user\nNow run the tests.<|im_end|>\n<|im_start|>assistant\n",
    ].joined()

    static var cachedTemplate: URL? {
        ControlTokenGuardTests.cachedModel("mlx-community/Qwen3-Coder-Next-4bit")?.appendingPathComponent("chat_template.jinja")
    }

    static func literal(_ text: String) throws -> String {
        String(decoding: try JSONEncoder().encode(text), as: UTF8.self)
    }

    static func editJSON() throws -> String {
        let (file, text) = (try literal(path), try literal(code))
        return #"{"file_path": \#(file), "old_string": "return total", "new_string": \#(text), "replace_all": true}"#
    }

    static func toolsJSON(schemaKey: String, wrapped: Bool) -> String {
        tools.map { tool in
            let schema = tool.definition.parametersJSON ?? "{}"
            let spec = #"{"name": "\#(tool.name)", "description": "\#(tool.description)", "\#(schemaKey)": \#(schema)}"#
            return wrapped ? #"{"type": "function", "function": \#(spec)}"# : spec
        }.joined(separator: ", ")
    }

    static func messagesBody() throws -> String {
        let edit = try editJSON()
        return #"""
        {"system": "You are a coding agent.",
         "tools": [\#(toolsJSON(schemaKey: "input_schema", wrapped: false))],
         "messages": [
           {"role": "user", "content": "Fix the budget."},
           {"role": "assistant", "content": [{"type": "text", "text": "Fixing it."},
                                             {"type": "tool_use", "id": "toolu_1", "name": "Edit", "input": \#(edit)},
                                             {"type": "tool_use", "id": "toolu_2", "name": "Read", "input": \#(readJSON)}]},
           {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "updated"},
                                        {"type": "tool_result", "tool_use_id": "toolu_2", "content": "42: return total"},
                                        {"type": "text", "text": "Now run the tests."}]}]}
        """#
    }

    static func chatCompletionsBody() throws -> String {
        let (edit, read) = (try literal(try editJSON()), try literal(readJSON))
        return #"""
        {"tools": [\#(toolsJSON(schemaKey: "parameters", wrapped: true))],
         "messages": [
           {"role": "system", "content": "You are a coding agent."},
           {"role": "user", "content": "Fix the budget."},
           {"role": "assistant", "content": "Fixing it.", "tool_calls": [
             {"id": "call_1", "type": "function", "function": {"name": "Edit", "arguments": \#(edit)}},
             {"id": "call_2", "type": "function", "function": {"name": "Read", "arguments": \#(read)}}]},
           {"role": "tool", "tool_call_id": "call_1", "content": "updated"},
           {"role": "tool", "tool_call_id": "call_2", "content": "42: return total"},
           {"role": "user", "content": "Now run the tests."}]}
        """#
    }

    static func conversation(_ shape: Shape) throws -> (turns: [ChatMessageInput], tools: [ToolDefinition]?) {
        switch shape {
        case .messages:
            let data = Data(try messagesBody().utf8)
            let dto = try JSONDecoder().decode(AnthropicController.MessagesRequest.self, from: data)
            let body = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            let tools = try OpenAIController.requestedTools(in: body).tools
            return (try AnthropicController.chatMessages(system: dto.system, messages: dto.messages), tools)
        case .chatCompletions:
            let parsed = try ChatRequestFastReader.read(Data(try chatCompletionsBody().utf8))
            let tools = try parsed.tools.get().tools
            return (try OpenAIController.chatMessages(parsed.dto.messages), tools)
        }
    }

    static func prompt(_ turns: [ChatMessageInput], tools: [ToolDefinition]?) throws -> String {
        let specs = MLXInferenceEngine.toolSpecs(tools)
        let messages = DefaultMessageGenerator().generate(
            from: UserInput(chat: MLXInferenceEngine.chat(turns, lastText: turns[turns.count - 1].text), tools: specs))
        var context: [String: Value] = try ["messages": .array(messages.map { try Value(any: $0) }), "add_generation_prompt": .boolean(true)]
        if let specs { context["tools"] = try .array(specs.map { try Value(any: $0) }) }
        return try ChatTemplateFidelityTests.rendered(Qwen3CoderNextTemplate.jinja, context)
    }

    static func assistantTurn(in prompt: String) throws -> String {
        let afterHeader = try #require(prompt.components(separatedBy: "<|im_start|>assistant").dropFirst().first)
        return try #require(afterHeader.components(separatedBy: "<|im_end|>").first)
    }

    @Test("The fixture is the chat template of the cached checkpoint", .enabled(if: cachedTemplate != nil, "needs the cached mlx-community/Qwen3-Coder-Next-4bit"))
    func fixtureMatchesTheCachedTemplate() throws {
        let cached = try String(contentsOf: try #require(Self.cachedTemplate), encoding: .utf8)
        #expect(cached == Qwen3CoderNextTemplate.jinja)
    }

    @Test("An assistant turn with text and two tool calls, their results and a follow-up user turn render with one <parameter> per argument: the integer as 42, the boolean as true, the multi-line string with its tab, trailing space and blank line", arguments: Shape.allCases)
    func historyRendersDecodedParameters(shape: Shape) throws {
        let (turns, tools) = try Self.conversation(shape)
        #expect(turns.map(\.role) == [.system, .user, .assistant, .tool, .tool, .user])
        #expect(tools?.map(\.name) == ["Edit", "Read"])
        let prompt = try Self.prompt(turns, tools: tools)
        #expect(prompt.hasPrefix("<|im_start|>system\nYou are a coding agent.\n\n# Tools\n"))
        #expect(prompt.hasSuffix(Self.expectedTail), "\(prompt.suffix(Self.expectedTail.count + 40).debugDescription)")
        #expect(prompt.contains("<parameter=offset>\n42\n</parameter>"))
        #expect(prompt.contains("<parameter=new_string>\n\(Self.code)\n</parameter>"))
        #expect(!prompt.contains(#"{"file_path""#))
    }

    @Test("Both request shapes render the same prompt")
    func bothShapesRenderTheSamePrompt() throws {
        let prompts = try Shape.allCases.map { shape in
            let (turns, tools) = try Self.conversation(shape)
            return try Self.prompt(turns, tools: tools)
        }
        #expect(prompts[0] == prompts[1])
    }

    @Test("The rendered assistant turn parses back to the calls of the request, with the same argument types and the same bytes in the multi-line string", arguments: Shape.allCases)
    func renderedHistoryRoundTrips(shape: Shape) throws {
        let (turns, tools) = try Self.conversation(shape)
        let expected = try #require(turns[2].toolCalls).map { MLXInferenceEngine.toolCall($0).function }
        #expect(expected.map(\.name) == ["Edit", "Read"])
        let edit: [String: JSONValue] = [
            "file_path": .string(Self.path), "old_string": .string("return total"), "new_string": .string(Self.code), "replace_all": .bool(true),
        ]
        let read: [String: JSONValue] = ["file_path": .string(Self.path), "offset": .int(42), "limit": .int(20)]
        #expect(expected.map(\.arguments) == [edit, read])
        let rendered = try Self.assistantTurn(in: try Self.prompt(turns, tools: tools))
        let parsed = Qwen3NextTools.parse([rendered], tools: Self.tools)
        #expect(parsed.calls == expected)
        #expect(parsed.rejections.isEmpty)
        #expect(parsed.text.trimmingCharacters(in: .whitespacesAndNewlines) == "Fixing it.")
    }

    @Test("A number, an array, an object and a string with a newline in a history call render as their JSON text or raw text and parse back under the tool's schema")
    func typedValuesRoundTrip() throws {
        let plan = Qwen3NextTools.Tool(name: "Plan", description: "Plan the work.", parameters: [
            .init(name: "steps", type: "array", required: true), .init(name: "options", type: "object"),
            .init(name: "ratio", type: "number"), .init(name: "note", type: "string"),
        ])
        let call = ToolCallOutput(
            id: "call_1", name: "Plan",
            arguments: #"{"steps": ["read", "edit"], "options": {"depth": 2, "follow": true}, "ratio": 2.5, "note": "line one\n  line two"}"#)
        let turns = [
            ChatMessageInput(role: .user, text: "Plan it."), ChatMessageInput(role: .assistant, text: "", toolCalls: [call]),
            ChatMessageInput(role: .tool, text: "ok", toolCallId: "call_1", toolName: "Plan"), ChatMessageInput(role: .user, text: "Go on."),
        ]
        let prompt = try Self.prompt(turns, tools: [plan.definition])
        #expect(prompt.contains("<parameter=ratio>\n2.5\n</parameter>"))
        #expect(prompt.contains("<parameter=note>\nline one\n  line two\n</parameter>"))
        let rendered = try Self.assistantTurn(in: prompt)
        let parsed = Qwen3NextTools.parse([rendered], tools: [plan])
        #expect(parsed.calls == [MLXInferenceEngine.toolCall(call).function])
        #expect(parsed.rejections.isEmpty)
    }

    @Test("A history call whose optional argument is JSON null, at the top level, inside an object or inside an array, renders without that argument instead of failing the request")
    func nullArgumentRenders() throws {
        let call = ToolCallOutput(id: "call_1", name: "Read", arguments: #"{"file_path": "a.swift", "limit": null, "options": {"depth": 2, "follow": null}, "ranges": [null, {"from": 1, "to": null}]}"#)
        let turns = [
            ChatMessageInput(role: .user, text: "Read it."), ChatMessageInput(role: .assistant, text: "", toolCalls: [call]),
            ChatMessageInput(role: .tool, text: "ok", toolCallId: "call_1", toolName: "Read"), ChatMessageInput(role: .user, text: "Go on."),
        ]
        let prompt = try Self.prompt(turns, tools: [Qwen3NextTools.read.definition])
        let rendered = try Self.assistantTurn(in: prompt)
        #expect(rendered.contains("<parameter=file_path>\na.swift\n</parameter>"))
        #expect(!rendered.contains("<parameter=limit>") && !rendered.contains("follow"))
        #expect(rendered.contains("depth") && rendered.contains("from") && !rendered.contains("null"))
    }
}
