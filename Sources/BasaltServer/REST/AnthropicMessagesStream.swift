import Foundation
import BasaltCore

struct AnthropicMessagesStream: EventStreamEncoder {
    let id = AnthropicController.messageID()
    let model: String
    var openBlock: (index: Int, type: String)?
    var nextIndex = 0
    var calledTools = false
    var outputTokens = 0
    var final: GenerationResponse?
    var out: [String] = []

    mutating func opening() -> [String] {
        out = []
        send("message_start", [
            "type": "message_start",
            "message": [
                "id": id, "type": "message", "role": "assistant", "content": [] as [[String: Any]], "model": model,
                "stop_reason": NSNull(), "stop_sequence": NSNull(), "usage": ["input_tokens": 0, "output_tokens": 0]
            ] as [String: Any]
        ])
        send("ping", ["type": "ping"])
        return out
    }

    mutating func frames(for chunk: StreamChunk) -> [String] {
        out = []
        switch chunk {
        case .toolCall(let call):
            outputTokens += 1
            emitToolUse(call)
        case .text(let text):
            outputTokens += 1
            appendText(text)
        case .reasoning(let reasoning):
            outputTokens += 1
            appendThinking(reasoning)
        case .metadata(let meta):
            final = meta
        }
        return out
    }

    mutating func closing() -> [String] {
        out = []
        if final?.finishReason == .error {
            send("error", ["type": "error", "error": ["type": "api_error", "message": ModelsController.modelUnavailable]])
            return out
        }
        if openBlock != nil || !calledTools { stop(open("text")) }
        send("message_delta", [
            "type": "message_delta",
            "delta": ["stop_reason": AnthropicController.stopReason(final?.finishReason ?? .stop, calledTools: calledTools),
                      "stop_sequence": NSNull()],
            "usage": [
                "input_tokens": (final?.promptTokens ?? 0) - (final?.cachedPromptTokens ?? 0),
                "cache_read_input_tokens": final?.cachedPromptTokens ?? 0,
                "output_tokens": final?.completionTokens ?? outputTokens
            ]
        ])
        send("message_stop", ["type": "message_stop"])
        return out
    }

    private mutating func send(_ event: String, _ data: [String: Any]) {
        if let json = Self.json(data) { out.append("event: \(event)\ndata: \(json)\n\n") }
    }

    private mutating func delta(_ index: Int, _ delta: [String: Any]) {
        send("content_block_delta", ["type": "content_block_delta", "index": index, "delta": delta])
    }

    private mutating func stop(_ index: Int) {
        send("content_block_stop", ["type": "content_block_stop", "index": index])
    }

    private mutating func open(_ type: String) -> Int {
        if let current = openBlock {
            if current.type == type { return current.index }
            stop(current.index)
        }
        let block: [String: Any] = type == "thinking"
            ? ["type": "thinking", "thinking": "", "signature": ""]
            : ["type": "text", "text": ""]
        send("content_block_start", ["type": "content_block_start", "index": nextIndex, "content_block": block])
        openBlock = (nextIndex, type)
        nextIndex += 1
        return nextIndex - 1
    }

    private mutating func appendText(_ text: String) {
        if calledTools, openBlock == nil, text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return }
        delta(open("text"), ["type": "text_delta", "text": text])
    }

    private mutating func appendThinking(_ reasoning: String) {
        guard openBlock?.type != "text", !calledTools else { return }
        delta(open("thinking"), ["type": "thinking_delta", "thinking": reasoning])
    }

    private mutating func emitToolUse(_ call: ToolCallOutput) {
        let note = AnthropicController.objectInput(call.arguments) == nil ? AnthropicController.nonObjectNote(call) : nil
        if let note { delta(open("text"), ["type": "text_delta", "text": note]) }
        if let current = openBlock {
            stop(current.index)
            openBlock = nil
        }
        let block: [String: Any] = ["type": "tool_use", "id": AnthropicController.toolUseID(call.id), "name": call.name, "input": [String: Any]()]
        send("content_block_start", ["type": "content_block_start", "index": nextIndex, "content_block": block])
        delta(nextIndex, ["type": "input_json_delta", "partial_json": note == nil ? call.arguments : "{}"])
        stop(nextIndex)
        nextIndex += 1
        calledTools = true
    }
}
