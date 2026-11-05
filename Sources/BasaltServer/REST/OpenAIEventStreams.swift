import Foundation
import BasaltCore

struct OpenAIChatStream: EventStreamEncoder {
    let id = "chatcmpl-\(UUID().uuidString)"
    let created = Int(Date().timeIntervalSince1970)
    let model: String
    let includeUsage: Bool
    var toolCallCount = 0
    var final: GenerationResponse?

    func opening() -> [String] { [] }

    mutating func frames(for chunk: StreamChunk) -> [String] {
        switch chunk {
        case .toolCall(let call):
            defer { toolCallCount += 1 }
            return frame(delta: ["role": "assistant", "tool_calls": [[
                "index": toolCallCount, "id": call.id, "type": "function",
                "function": ["name": call.name, "arguments": call.arguments]
            ] as [String: Any]]])
        case .text(let text):
            if toolCallCount > 0, text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return [] }
            return frame(delta: ["role": "assistant", "content": text])
        case .reasoning(let reasoning):
            return frame(delta: ["role": "assistant", "reasoning": reasoning, "reasoning_content": reasoning])
        case .metadata(let meta):
            final = meta
            return []
        }
    }

    func closing() -> [String] {
        if final?.finishReason == .error {
            let error: [String: Any] = ["error": ["message": ModelsController.generationFailed, "type": "server_error"]]
            return Self.json(error).map { ["event: error\ndata: \($0)\n\n"] } ?? []
        }
        var frames = frame(delta: [:], finish: OpenAIController.finishReason(final?.finishReason, calledTools: toolCallCount > 0))
        if includeUsage, let final {
            frames += frame(["choices": [] as [Any], "usage": OpenAIController.chatUsage(final)])
        }
        return frames + ["data: [DONE]\n\n"]
    }

    private func frame(delta: [String: Any], finish: String? = nil) -> [String] {
        frame(["choices": [["index": 0, "delta": delta, "finish_reason": finish ?? NSNull()] as [String: Any]]])
    }

    private func frame(_ fields: [String: Any]) -> [String] {
        var chunk: [String: Any] = ["id": id, "object": "chat.completion.chunk", "created": created, "model": model]
        chunk.merge(fields) { _, new in new }
        return Self.json(chunk).map { ["data: \($0)\n\n"] } ?? []
    }
}

struct OpenAIResponsesStream: EventStreamEncoder {
    enum Item: Sendable {
        case reasoning(id: String, text: String)
        case message(id: String, text: String)
        case functionCall(id: String, call: ToolCallOutput)

        var json: [String: Any] {
            switch self {
            case .reasoning(let id, let text):
                ["id": id, "type": "reasoning", "summary": [] as [Any], "content": [["type": "reasoning_text", "text": text]]]
            case .message(let id, let text):
                OpenAIResponsesStream.message(id: id, status: "completed", text: text)
            case .functionCall(let id, let call):
                OpenAIResponsesStream.functionCall(id: id, call: call, arguments: call.arguments, status: "completed")
            }
        }

        var isFunctionCall: Bool {
            if case .functionCall = self { true } else { false }
        }
    }

    let id = "resp_\(UUID().uuidString)"
    let reasoningId = "rs_\(UUID().uuidString)"
    let createdAt = Int(Date().timeIntervalSince1970)
    let model: String
    var messageId = "msg_\(UUID().uuidString)"
    var sequence = 0
    var completed: [Item] = []
    var reasoning: String?
    var messageIndex: Int?
    var text = ""
    var final: GenerationResponse?
    var out: [String] = []

    var calledTools: Bool { completed.contains(where: \.isFunctionCall) }

    mutating func opening() -> [String] {
        out = []
        send("response.created", ["response": responseObject(status: "in_progress", output: [])])
        return out
    }

    mutating func frames(for chunk: StreamChunk) -> [String] {
        out = []
        switch chunk {
        case .text(let delta): appendText(delta)
        case .reasoning(let delta): appendReasoning(delta)
        case .toolCall(let call): emitToolCall(call)
        case .metadata(let meta): final = meta
        }
        return out
    }

    mutating func closing() -> [String] {
        out = []
        if final?.finishReason == .error {
            send("response.failed", ["response": responseObject(status: "failed", output: [], extra: [
                "error": ["code": "server_error", "message": ModelsController.generationFailed]
            ])])
            return out
        }
        closeReasoning()
        if messageIndex != nil || !calledTools { _ = openMessage() }
        closeMessage()
        send("response.completed", ["response": responseObject(status: "completed", output: completed.map(\.json), extra: [
            "usage": usage
        ])])
        return out
    }

    var usage: [String: Any] {
        let inTokens = final?.promptTokens ?? 0
        let outTokens = final?.completionTokens ?? 0
        return [
            "input_tokens": inTokens, "output_tokens": outTokens, "total_tokens": inTokens + outTokens,
            "input_tokens_details": ["cached_tokens": final?.cachedPromptTokens ?? 0],
            "output_tokens_details": ["reasoning_tokens": final?.reasoningTokens ?? 0]]
    }

    static func message(id: String, status: String, text: String?) -> [String: Any] {
        let content: [Any] = text.map { [["type": "output_text", "text": $0, "annotations": [] as [Any]]] } ?? []
        return ["id": id, "type": "message", "status": status, "role": "assistant", "content": content]
    }

    static func functionCall(id: String, call: ToolCallOutput, arguments: String, status: String) -> [String: Any] {
        ["id": id, "type": "function_call", "call_id": call.id, "name": call.name, "arguments": arguments, "status": status]
    }

    private mutating func send(_ event: String, _ fields: [String: Any]) {
        var payload = fields
        payload["type"] = event
        payload["sequence_number"] = sequence
        sequence += 1
        if let json = Self.json(payload) { out.append("event: \(event)\ndata: \(json)\n\n") }
    }

    private func responseObject(status: String, output: [Any], extra: [String: Any] = [:]) -> [String: Any] {
        var response: [String: Any] = ["id": id, "object": "response", "created_at": createdAt,
                                       "status": status, "model": model, "output": output]
        response.merge(extra) { _, new in new }
        return response
    }

    private mutating func appendText(_ delta: String) {
        if messageIndex == nil, calledTools, delta.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return }
        let index = openMessage()
        text += delta
        send("response.output_text.delta", ["item_id": messageId, "output_index": index, "content_index": 0, "delta": delta])
    }

    private mutating func appendReasoning(_ delta: String) {
        guard messageIndex == nil, completed.isEmpty else { return }
        if reasoning == nil {
            send("response.output_item.added", ["output_index": 0, "item": [
                "id": reasoningId, "type": "reasoning", "summary": [] as [Any], "content": [] as [Any]
            ] as [String: Any]])
        }
        reasoning = (reasoning ?? "") + delta
        send("response.reasoning_text.delta", ["item_id": reasoningId, "output_index": 0, "content_index": 0, "delta": delta])
    }

    private mutating func closeReasoning() {
        guard let reasoning else { return }
        send("response.reasoning_text.done", [
            "item_id": reasoningId, "output_index": completed.count, "content_index": 0, "text": reasoning
        ])
        let done = Item.reasoning(id: reasoningId, text: reasoning)
        send("response.output_item.done", ["output_index": completed.count, "item": done.json])
        completed.append(done)
        self.reasoning = nil
    }

    private mutating func closeMessage() {
        guard let index = messageIndex else { return }
        send("response.output_text.done", ["item_id": messageId, "output_index": index, "content_index": 0, "text": text])
        send("response.content_part.done", [
            "item_id": messageId, "output_index": index, "content_index": 0,
            "part": ["type": "output_text", "text": text, "annotations": [] as [Any]]
        ])
        let done = Item.message(id: messageId, text: text)
        send("response.output_item.done", ["output_index": index, "item": done.json])
        completed.append(done)
        messageIndex = nil
        messageId = "msg_\(UUID().uuidString)"
        text = ""
    }

    private mutating func openMessage() -> Int {
        if let index = messageIndex { return index }
        closeReasoning()
        let index = completed.count
        send("response.output_item.added", ["output_index": index, "item": Self.message(id: messageId, status: "in_progress", text: nil)])
        send("response.content_part.added", [
            "item_id": messageId, "output_index": index, "content_index": 0,
            "part": ["type": "output_text", "text": "", "annotations": [] as [Any]]
        ])
        messageIndex = index
        return index
    }

    private mutating func emitToolCall(_ call: ToolCallOutput) {
        closeReasoning()
        closeMessage()
        let index = completed.count
        let itemId = "fc_\(UUID().uuidString)"
        send("response.output_item.added", [
            "output_index": index, "item": Self.functionCall(id: itemId, call: call, arguments: "", status: "in_progress")
        ])
        send("response.function_call_arguments.delta", ["item_id": itemId, "output_index": index, "delta": call.arguments])
        send("response.function_call_arguments.done", ["item_id": itemId, "output_index": index, "arguments": call.arguments])
        let done = Item.functionCall(id: itemId, call: call)
        send("response.output_item.done", ["output_index": index, "item": done.json])
        completed.append(done)
    }
}
