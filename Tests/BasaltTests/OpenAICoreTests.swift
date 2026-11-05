import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltServer

@Suite("OpenAI value core")
struct OpenAICoreTests {
    private func messages(_ json: String) throws -> [OpenAIController.ChatMessage] {
        try JSONDecoder().decode([OpenAIController.ChatMessage].self, from: Data(json.utf8))
    }

    private func result(
        text: String, reasoning: String? = nil, calls: [ToolCallOutput]? = nil, finish: GenerationResponse.FinishReason = .stop
    ) -> GenerationResponse {
        GenerationResponse(
            generatedText: text, reasoningText: reasoning, promptTokens: 7, cachedPromptTokens: 2, completionTokens: 5,
            tokensPerSecond: 1, generationTime: 1, timeToFirstToken: 0,
            memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0), finishReason: finish, toolCalls: calls,
            reasoningTokens: 3)
    }

    private func payloads(_ frames: [String]) -> [(event: String?, data: [String: Any]?)] {
        frames.map { frame in
            let lines = frame.split(separator: "\n")
            let event = lines.first { $0.hasPrefix("event: ") }.map { String($0.dropFirst(7)) }
            let data = lines.first { $0.hasPrefix("data: ") }.flatMap {
                try? JSONSerialization.jsonObject(with: Data($0.dropFirst(6).utf8)) as? [String: Any]
            }
            return (event, data)
        }
    }

    private let call = ToolCallOutput(id: "c1", name: "read", arguments: #"{"path":"a"}"#)

    @Test("Responses input items: consecutive calls merge into one assistant turn, outputs carry the tool name, reasoning is dropped")
    func responsesItemsToChat() throws {
        let chat = OpenAIController.chatMessages(try messages(#"""
        [{"role": "user", "content": "go"},
         {"type": "reasoning"},
         {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"},
         {"type": "custom_tool_call", "call_id": "c2", "name": "patch", "input": "diff"},
         {"type": "function_call_output", "call_id": "c1", "output": "file"}]
        """#))
        #expect(chat == [
            ChatMessageInput(role: .user, text: "go"),
            ChatMessageInput(role: .assistant, text: "", toolCalls: [
                ToolCallOutput(id: "c1", name: "read", arguments: "{}"), ToolCallOutput(id: "c2", name: "patch", arguments: "diff"),
            ]),
            ChatMessageInput(role: .tool, text: "file", toolCallId: "c1", toolName: "read"),
        ])
    }

    @Test("Chat messages: developer is system, tool results get the called name, empty turns without calls are dropped")
    func chatFormatToChat() throws {
        let chat = OpenAIController.chatMessages(try messages(#"""
        [{"role": "developer", "content": "rules"},
         {"role": "user", "content": ""},
         {"role": "assistant", "content": null, "tool_calls": [{"id": "t1", "function": {"name": "ls"}}]},
         {"role": "tool", "tool_call_id": "t1", "content": [{"type": "text", "text": "a"}, {"type": "image"}]}]
        """#))
        #expect(chat == [
            ChatMessageInput(role: .system, text: "rules"),
            ChatMessageInput(role: .assistant, text: "", toolCalls: [ToolCallOutput(id: "t1", name: "ls", arguments: "{}")]),
            ChatMessageInput(role: .tool, text: "a\n\(ToolTranscript.imageOmitted)", toolCallId: "t1", toolName: "ls"),
        ])
    }

    @Test("Responses request: text input is one user turn, empty text is none, instructions become a system turn")
    func responsesConversation() throws {
        let decode = { (json: String) in
            try JSONDecoder().decode(OpenAIController.ResponsesRequest.self, from: Data(json.utf8))
        }
        let request = try decode(#"{"input": "hi", "instructions": "be brief"}"#)
        #expect(request.conversation == [ChatMessageInput(role: .user, text: "hi")])
        #expect(request.instructionMessages == [ChatMessageInput(role: .system, text: "be brief")])
        #expect(try decode(#"{"input": "", "instructions": ""}"#).conversation.isEmpty)
        #expect(try decode(#"{"instructions": ""}"#).instructionMessages.isEmpty)
    }

    @Test("Chat completion body: a turn that only calls tools has null content and finish_reason tool_calls")
    func chatCompletionBody() {
        let response = OpenAIController.ChatCompletionResponse(result: result(text: " \n", reasoning: "hm", calls: [call]), model: "m")
        let choice = response.choices[0]
        #expect(choice.message.content == nil)
        #expect(choice.message.reasoning == "hm")
        #expect(choice.message.toolCalls?.first?.function.name == "read")
        #expect(choice.finishReason == "tool_calls")
        #expect(response.usage.totalTokens == 12)
        #expect(response.usage.promptTokensDetails.cachedTokens == 2)
        #expect(response.usage.completionTokensDetails?.reasoningTokens == 3)
        let truncated = OpenAIController.ChatCompletionResponse(result: result(text: "a", finish: .length), model: "m")
        #expect(truncated.choices[0].finishReason == "length")
        if case .text(let text) = truncated.choices[0].message.content { #expect(text == "a") } else { Issue.record("no text") }
    }

    @Test("Responses body: reasoning, then the message, then one function_call per tool call; a tool-only turn has no message")
    func responsesBodyOutput() {
        let output = OpenAIController.ResponsesResponse.output(for: result(text: "ok", reasoning: "r", calls: [call]))
        #expect(output.map(\.type) == ["reasoning", "message", "function_call"])
        #expect(output[2].callId == "c1")
        #expect(OpenAIController.ResponsesResponse.output(for: result(text: "", calls: [call])).map(\.type) == ["function_call"])
        #expect(OpenAIController.ResponsesResponse.output(for: result(text: "")).map(\.type) == ["message"])
    }

    @Test("Chat stream: whitespace after a tool call is dropped, the finish chunk says tool_calls, usage on request, then [DONE]")
    func chatStreamFrames() {
        var stream = OpenAIChatStream(model: "m", includeUsage: true)
        #expect(stream.opening().isEmpty)
        let toolFrames = stream.frames(for: .toolCall(call))
        #expect(stream.frames(for: .text("\n")).isEmpty)
        #expect(stream.frames(for: .metadata(result(text: ""))).isEmpty)
        let closing = stream.closing()
        #expect(closing.last == "data: [DONE]\n\n")
        let chunks = payloads(toolFrames + closing.dropLast()).compactMap(\.data)
        #expect(chunks.allSatisfy { $0["object"] as? String == "chat.completion.chunk" && $0["model"] as? String == "m" })
        let finish = (chunks[1]["choices"] as? [[String: Any]])?.first?["finish_reason"] as? String
        #expect(finish == "tool_calls")
        #expect((chunks[2]["usage"] as? [String: Any])?["total_tokens"] as? Int == 12)
    }

    @Test("Chat stream: a failed generation ends with an error event and no [DONE]")
    func chatStreamError() {
        var stream = OpenAIChatStream(model: "m", includeUsage: false)
        _ = stream.frames(for: .metadata(result(text: "", finish: .error)))
        let closing = stream.closing()
        #expect(closing.count == 1)
        #expect(closing[0].hasPrefix("event: error\n"))
        #expect((payloads(closing).first?.data?["error"] as? [String: Any])?["message"] as? String == ModelsController.generationFailed)
    }

    @Test("Responses stream: reasoning item, message item, numbered events, completed response with both items and usage")
    func responsesStreamFrames() {
        var stream = OpenAIResponsesStream(model: "m")
        var frames = stream.opening()
        frames += stream.frames(for: .reasoning("think"))
        frames += stream.frames(for: .text("hi"))
        frames += stream.frames(for: .metadata(result(text: "hi")))
        frames += stream.closing()
        let events = payloads(frames)
        #expect(events.map(\.event) == [
            "response.created", "response.output_item.added", "response.reasoning_text.delta",
            "response.reasoning_text.done", "response.output_item.done",
            "response.output_item.added", "response.content_part.added", "response.output_text.delta",
            "response.output_text.done", "response.content_part.done", "response.output_item.done", "response.completed",
        ])
        #expect(events.compactMap { $0.data?["sequence_number"] as? Int } == Array(0 ..< events.count))
        let completed = events.last?.data?["response"] as? [String: Any]
        #expect((completed?["output"] as? [[String: Any]])?.compactMap { $0["type"] as? String } == ["reasoning", "message"])
        #expect((completed?["usage"] as? [String: Any])?["total_tokens"] as? Int == 12)
    }

    @Test("Responses stream: a tool-only turn has one function_call item and no message; an error ends in response.failed")
    func responsesStreamToolsAndError() {
        var stream = OpenAIResponsesStream(model: "m")
        var frames = stream.frames(for: .toolCall(call))
        frames += stream.frames(for: .text(" "))
        frames += stream.closing()
        let events = payloads(frames)
        #expect(events.filter { $0.event == "response.output_item.added" }.count == 1)
        let output = (events.last?.data?["response"] as? [String: Any])?["output"] as? [[String: Any]]
        #expect(output?.map { $0["type"] as? String } == ["function_call"])
        #expect(output?.first?["arguments"] as? String == call.arguments)

        var failing = OpenAIResponsesStream(model: "m")
        _ = failing.frames(for: .metadata(result(text: "", finish: .error)))
        let failed = payloads(failing.closing())
        #expect(failed.map(\.event) == ["response.failed"])
        let error = (failed.first?.data?["response"] as? [String: Any])?["error"] as? [String: Any]
        #expect(error?["message"] as? String == ModelsController.generationFailed)
    }
}
