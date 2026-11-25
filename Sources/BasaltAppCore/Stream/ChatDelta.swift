import BasaltCore
import Foundation

public enum ChatDelta: Sendable, Equatable {
    case text(String)
    case reasoning(String)
    case toolCall(ToolCallOutput)
    case usage(prompt: Int, completion: Int)
    case finish(String)
}

public enum ChatDeltaDecoder {
    public enum Decoded: Sendable, Equatable {
        case deltas([ChatDelta])
        case done
    }

    struct Frame: Decodable {
        struct Choice: Decodable {
            struct Delta: Decodable {
                struct ToolCall: Decodable {
                    struct Function: Decodable {
                        var name: String?
                        var arguments: String?
                    }

                    var index: Int?
                    var id: String?
                    var function: Function?
                }

                var content: String?
                var reasoning: String?
                var toolCalls: [ToolCall]?

                enum CodingKeys: String, CodingKey {
                    case content, reasoning
                    case toolCalls = "tool_calls"
                }
            }

            var delta: Delta?
            var finishReason: String?

            enum CodingKeys: String, CodingKey {
                case delta
                case finishReason = "finish_reason"
            }
        }

        struct Usage: Decodable {
            var promptTokens: Int
            var completionTokens: Int

            enum CodingKeys: String, CodingKey {
                case promptTokens = "prompt_tokens"
                case completionTokens = "completion_tokens"
            }
        }

        struct Failure: Decodable {
            var message: String
        }

        var choices: [Choice]?
        var usage: Usage?
        var error: Failure?
    }

    public static func decode(_ event: SSEEvent) throws(EngineError) -> Decoded {
        var ids: [Int: String] = [:]
        return try decode(event, toolCallIDs: &ids)
    }

    public static func decode(_ event: SSEEvent, toolCallIDs: inout [Int: String]) throws(EngineError) -> Decoded {
        guard event.data != "[DONE]" else { return .done }
        let frame: Frame
        do {
            frame = try JSONDecoder().decode(Frame.self, from: Data(event.data.utf8))
        } catch {
            throw .malformed(String(event.data.prefix(200)))
        }
        if let error = frame.error { throw .refused(status: 500, message: error.message) }
        var deltas: [ChatDelta] = []
        for choice in frame.choices ?? [] {
            if let reasoning = choice.delta?.reasoning, !reasoning.isEmpty { deltas.append(.reasoning(reasoning)) }
            if let text = choice.delta?.content, !text.isEmpty { deltas.append(.text(text)) }
            for call in choice.delta?.toolCalls ?? [] {
                let resolvedID = resolveToolCallID(call, in: &toolCallIDs)
                deltas.append(.toolCall(ToolCallOutput(id: resolvedID, name: call.function?.name ?? "", arguments: call.function?.arguments ?? "")))
            }
            if let reason = choice.finishReason { deltas.append(.finish(reason)) }
        }
        if let usage = frame.usage { deltas.append(.usage(prompt: usage.promptTokens, completion: usage.completionTokens)) }
        return .deltas(deltas)
    }

    private static func resolveToolCallID(_ call: Frame.Choice.Delta.ToolCall, in map: inout [Int: String]) -> String {
        if let rawID = call.id, !rawID.isEmpty {
            if let idx = call.index { map[idx] = rawID }
            return rawID
        }
        if let idx = call.index {
            if let known = map[idx] { return known }
            let generated = "call_\(idx)"
            map[idx] = generated
            return generated
        }
        return ""
    }
}
