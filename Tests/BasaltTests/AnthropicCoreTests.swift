import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltServer

@Suite("Anthropic value core")
struct AnthropicCoreTests {
    private func request(_ json: String) throws -> AnthropicController.MessagesRequest {
        try JSONDecoder().decode(AnthropicController.MessagesRequest.self, from: Data(json.utf8))
    }

    private func result(
        text: String, reasoning: String? = nil, calls: [ToolCallOutput]? = nil, finish: GenerationResponse.FinishReason = .stop
    ) -> GenerationResponse {
        GenerationResponse(
            generatedText: text, reasoningText: reasoning, promptTokens: 7, cachedPromptTokens: 2, completionTokens: 5,
            tokensPerSecond: 1, generationTime: 1, timeToFirstToken: 0,
            memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0), finishReason: finish, toolCalls: calls)
    }

    private func events(_ frames: [String]) -> [(event: String, data: [String: Any])] {
        frames.compactMap { frame in
            let lines = frame.split(separator: "\n")
            guard let event = lines.first?.dropFirst(7), let data = lines.dropFirst().first?.dropFirst(6),
                  let json = try? JSONSerialization.jsonObject(with: Data(data.utf8)) as? [String: Any] else { return nil }
            return (String(event), json)
        }
    }

    @Test("Messages: system first, tool_use becomes a call, tool_result splits the turn and carries the tool name and error prefix")
    func messagesToChat() throws {
        let dto = try request(#"""
        {"system": [{"type": "text", "text": "rules"}],
         "messages": [
           {"role": "user", "content": "go"},
           {"role": "assistant", "content": [{"type": "text", "text": "ok"},
                                             {"type": "tool_use", "id": "t1", "name": "ls", "input": {"b": 1, "a": "x/y"}}]},
           {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": true, "content": "boom"},
                                        {"type": "image"}]},
           {"role": "assistant", "content": []}]}
        """#)
        #expect(AnthropicController.chatMessages(system: dto.system, messages: dto.messages) == [
            ChatMessageInput(role: .system, text: "rules"),
            ChatMessageInput(role: .user, text: "go"),
            ChatMessageInput(role: .assistant, text: "ok", toolCalls: [ToolCallOutput(id: "t1", name: "ls", arguments: #"{"a":"x/y","b":1}"#)]),
            ChatMessageInput(role: .tool, text: "Error: boom", toolCallId: "t1", toolName: "ls"),
            ChatMessageInput(role: .user, text: ToolTranscript.imageOmitted),
            ChatMessageInput(role: .assistant, text: ""),
        ])
    }

    @Test("Thinking: the budget must be below max_tokens, and it maps to a reasoning effort when none is given", arguments: [
        (#"{"messages": [], "max_tokens": 20000, "thinking": {"type": "enabled", "budget_tokens": 9000}}"#, false, "high"),
        (#"{"messages": [], "max_tokens": 100, "thinking": {"type": "enabled", "budget_tokens": 100}}"#, true, "low"),
        (#"{"messages": [], "thinking": {"type": "enabled", "budget_tokens": 0}}"#, true, "low"),
        (#"{"messages": [], "thinking": {"type": "disabled", "budget_tokens": 0}}"#, false, "off"),
        (#"{"messages": [], "thinking": {"type": "enabled"}, "reasoning_effort": "max"}"#, false, "max"),
    ])
    func thinking(json: String, rejected: Bool, effort: String) throws {
        let dto = try request(json)
        #expect((dto.thinkingBudgetProblem != nil) == rejected)
        #expect(dto.effectiveReasoningEffort == effort)
    }

    @Test("Response body: a call with non-object arguments is sent as {} with a note in the text block, stop_reason tool_use")
    func responseBody() {
        let calls = [ToolCallOutput(id: "call_9", name: "run", arguments: "[1]")]
        let response = AnthropicController.MessagesResponse(result: result(text: "", reasoning: "hm", calls: calls), model: "m")
        #expect(response.content.map(\.type) == ["thinking", "text", "tool_use"])
        #expect(response.content[1].text == AnthropicController.nonObjectNote(calls[0]))
        #expect(response.content[2].id == "toolu_9")
        #expect(response.stopReason == "tool_use")
        #expect(response.usage.inputTokens == 5)
        #expect(response.usage.cacheReadInputTokens == 2)
        #expect(response.id.hasPrefix("msg_") && response.id.count == 28)
    }

    @Test("Stream: thinking block, then text block, then message_delta with usage and message_stop")
    func streamThinkingThenText() {
        var stream = AnthropicMessagesStream(model: "m")
        var frames = stream.opening()
        frames += stream.frames(for: .reasoning("hm"))
        frames += stream.frames(for: .text("hi"))
        frames += stream.frames(for: .reasoning("late"))
        frames += stream.frames(for: .metadata(result(text: "hi")))
        frames += stream.closing()
        let evs = events(frames)
        #expect(evs.map(\.event) == [
            "message_start", "ping", "content_block_start", "content_block_delta", "content_block_stop",
            "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop",
        ])
        #expect((evs[2].data["content_block"] as? [String: Any])?["type"] as? String == "thinking")
        #expect(evs[5].data["index"] as? Int == 1)
        let delta = evs[8].data
        #expect((delta["delta"] as? [String: Any])?["stop_reason"] as? String == "end_turn")
        #expect((delta["usage"] as? [String: Any])?["input_tokens"] as? Int == 5)
    }

    @Test("Stream: a tool-only turn has no text block; whitespace after the call is dropped; usage falls back to the chunk count")
    func streamToolOnly() {
        var stream = AnthropicMessagesStream(model: "m")
        var frames = stream.frames(for: .toolCall(ToolCallOutput(id: "c1", name: "ls", arguments: "{}")))
        frames += stream.frames(for: .text("\n"))
        frames += stream.closing()
        let evs = events(frames)
        #expect(evs.map(\.event) == ["content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"])
        #expect((evs[0].data["content_block"] as? [String: Any])?["id"] as? String == "toolu_c1")
        let delta = evs[3].data
        #expect((delta["delta"] as? [String: Any])?["stop_reason"] as? String == "tool_use")
        #expect((delta["usage"] as? [String: Any])?["output_tokens"] as? Int == 2)
    }

    @Test("Stream: a failed generation ends with one error event")
    func streamError() {
        var stream = AnthropicMessagesStream(model: "m")
        _ = stream.frames(for: .metadata(result(text: "", finish: .error)))
        let evs = events(stream.closing())
        #expect(evs.map(\.event) == ["error"])
        #expect((evs[0].data["error"] as? [String: Any])?["type"] as? String == "api_error")
    }
}
