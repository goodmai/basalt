import Foundation
import BasaltCore

extension OpenAIController {
    struct ResponsesResponse: Codable, Sendable {
        let id: String
        let object: String
        let createdAt: Int
        let status: String
        let model: String
        let output: [OutputItem]
        let usage: Usage

        struct OutputItem: Codable, Sendable {
            let id: String
            let type: String
            let status: String
            var role: String? = nil
            var content: [OutputContent]? = nil
            var callId: String? = nil
            var name: String? = nil
            var arguments: String? = nil
            var summary: [OutputContent]? = nil

            enum CodingKeys: String, CodingKey {
                case id, type, status, role, content, name, arguments, summary
                case callId = "call_id"
            }
        }

        struct OutputContent: Codable, Sendable {
            let type: String
            let text: String?
        }

        struct Usage: Codable, Sendable {
            let inputTokens: Int
            let inputTokensDetails: CachedTokens
            let outputTokens: Int
            let totalTokens: Int
            var outputTokensDetails: ReasoningTokens? = nil

            enum CodingKeys: String, CodingKey {
                case inputTokens = "input_tokens"
                case inputTokensDetails = "input_tokens_details"
                case outputTokens = "output_tokens"
                case totalTokens = "total_tokens"
                case outputTokensDetails = "output_tokens_details"
            }
        }

        enum CodingKeys: String, CodingKey {
            case id, object, status, model, output, usage
            case createdAt = "created_at"
        }
    }

    struct ChatCompletionResponse: Codable, Sendable {
        let id: String
        let object: String
        let created: Int
        let model: String
        let choices: [Choice]
        let usage: Usage

        struct Choice: Codable, Sendable {
            let index: Int
            let message: ChatMessage
            let finishReason: String
            enum CodingKeys: String, CodingKey {
                case index, message
                case finishReason = "finish_reason"
            }
        }

        struct Usage: Codable, Sendable {
            let promptTokens: Int
            let promptTokensDetails: CachedTokens
            let completionTokens: Int
            let totalTokens: Int
            var completionTokensDetails: ReasoningTokens? = nil
            enum CodingKeys: String, CodingKey {
                case promptTokens = "prompt_tokens"
                case promptTokensDetails = "prompt_tokens_details"
                case completionTokens = "completion_tokens"
                case totalTokens = "total_tokens"
                case completionTokensDetails = "completion_tokens_details"
            }
        }
    }

    struct ReasoningTokens: Codable, Sendable {
        let reasoningTokens: Int

        enum CodingKeys: String, CodingKey {
            case reasoningTokens = "reasoning_tokens"
        }
    }

    struct CachedTokens: Codable, Sendable {
        let cachedTokens: Int

        enum CodingKeys: String, CodingKey {
            case cachedTokens = "cached_tokens"
        }
    }

    static func finishReason(_ finish: GenerationResponse.FinishReason?, calledTools: Bool) -> String {
        calledTools ? "tool_calls" : (finish?.isTruncated == true ? "length" : "stop")
    }

    static func chatUsage(_ result: GenerationResponse) -> [String: Any] {
        [
            "prompt_tokens": result.promptTokens, "completion_tokens": result.completionTokens,
            "total_tokens": result.promptTokens + result.completionTokens,
            "prompt_tokens_details": ["cached_tokens": result.cachedPromptTokens],
            "completion_tokens_details": ["reasoning_tokens": result.reasoningTokens ?? 0],
        ]
    }
}

extension OpenAIController.ChatCompletionResponse {
    init(result: GenerationResponse, model: String) {
        let calls = result.toolCalls ?? []
        let onlyCalls = !calls.isEmpty && result.generatedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let message = OpenAIController.ChatMessage(
            role: "assistant", content: onlyCalls ? nil : .text(result.generatedText),
            toolCalls: calls.isEmpty ? nil : calls.map { OpenAIController.ChatToolCall($0) }, reasoning: result.reasoningText)
        let finishReason = OpenAIController.finishReason(result.finishReason, calledTools: !calls.isEmpty)
        self = Self(
            id: "chatcmpl-\(UUID().uuidString)", object: "chat.completion", created: Int(Date().timeIntervalSince1970),
            model: model, choices: [Choice(index: 0, message: message, finishReason: finishReason)],
            usage: Usage(
                promptTokens: result.promptTokens,
                promptTokensDetails: OpenAIController.CachedTokens(cachedTokens: result.cachedPromptTokens),
                completionTokens: result.completionTokens,
                totalTokens: result.promptTokens + result.completionTokens,
                completionTokensDetails: OpenAIController.ReasoningTokens(reasoningTokens: result.reasoningTokens ?? 0)))
    }
}

extension OpenAIController.ResponsesResponse {
    init(result: GenerationResponse, model: String) {
        self = Self(
            id: "resp_\(UUID().uuidString)", object: "response", createdAt: Int(Date().timeIntervalSince1970),
            status: "completed", model: model, output: Self.output(for: result),
            usage: Usage(
                inputTokens: result.promptTokens,
                inputTokensDetails: OpenAIController.CachedTokens(cachedTokens: result.cachedPromptTokens),
                outputTokens: result.completionTokens,
                totalTokens: result.promptTokens + result.completionTokens,
                outputTokensDetails: OpenAIController.ReasoningTokens(reasoningTokens: result.reasoningTokens ?? 0)))
    }

    static func output(for result: GenerationResponse) -> [OutputItem] {
        let reasoning = result.reasoningText.map { text in
            [OutputItem(id: "rs_\(UUID().uuidString)", type: "reasoning", status: "completed",
                        content: [OutputContent(type: "reasoning_text", text: text)], summary: [])]
        } ?? []
        let message = result.generatedText.isEmpty && result.toolCalls != nil ? [] : [
            OutputItem(id: "msg_\(UUID().uuidString)", type: "message", status: "completed", role: "assistant",
                       content: [OutputContent(type: "output_text", text: result.generatedText)])
        ]
        let calls = (result.toolCalls ?? []).map { call in
            OutputItem(id: "fc_\(UUID().uuidString)", type: "function_call", status: "completed",
                       callId: call.id, name: call.name, arguments: call.arguments)
        }
        return reasoning + message + calls
    }
}
