import Foundation
import BasaltCore

extension OpenRouterClient {
    public struct OpenRouterModel: Codable, Sendable {
        public let id: String
        public let name: String
        public let contextLength: Int
        public let pricing: Pricing

        public struct Pricing: Codable, Sendable {
            public let prompt: String
            public let completion: String
        }

        public enum CodingKeys: String, CodingKey {
            case id
            case name
            case contextLength = "context_length"
            case pricing
        }
    }

    public struct ModelsResponse: Codable, Sendable {
        public let data: [OpenRouterModel]
    }

    public struct ChatResponse: Codable, Sendable {
        public let id: String
        public let model: String
        public let choices: [Choice]
        public let usage: Usage?

        public struct Choice: Codable, Sendable {
            public let message: Message
            public let finishReason: String?

            public struct Message: Codable, Sendable {
                public let role: String
                public let content: String
            }

            public enum CodingKeys: String, CodingKey {
                case message
                case finishReason = "finish_reason"
            }
        }

        public struct Usage: Codable, Sendable {
            public let promptTokens: Int
            public let completionTokens: Int
            public let totalTokens: Int

            public enum CodingKeys: String, CodingKey {
                case promptTokens = "prompt_tokens"
                case completionTokens = "completion_tokens"
                case totalTokens = "total_tokens"
            }
        }
    }

    public struct ChatStreamResponse: Codable, Sendable {
        public typealias Usage = ChatResponse.Usage

        public let id: String
        public let model: String
        public let choices: [Choice]
        public let usage: Usage?

        public struct Choice: Codable, Sendable {
            public let delta: Delta
            public let finishReason: String?

            public struct Delta: Codable, Sendable {
                public let role: String?
                public let content: String?
            }

            public enum CodingKeys: String, CodingKey {
                case delta
                case finishReason = "finish_reason"
            }
        }
    }
}
