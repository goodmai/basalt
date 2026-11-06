import Foundation
import BasaltCore

extension OpenAIController {
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
