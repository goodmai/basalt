import Foundation

/// OpenRouter API client for cloud model inference (Refactored to use CloudAPIClient)
public actor OpenRouterClient: Sendable {
    
    // MARK: - Configuration
    
    public struct Config: Sendable {
        public let apiKey: String
        public let baseURL: URL
        public let timeout: TimeInterval
        public let maxRetries: Int
        
        public static let `default` = Config(
            apiKey: ProcessInfo.processInfo.environment["OPENROUTER_API_KEY"] ?? "",
            baseURL: URL(string: "https://openrouter.ai/api/v1")!,
            timeout: 60.0,
            maxRetries: 3
        )
        
        public init(apiKey: String, baseURL: URL, timeout: TimeInterval, maxRetries: Int) {
            self.apiKey = apiKey
            self.baseURL = baseURL
            self.timeout = timeout
            self.maxRetries = maxRetries
        }
    }
    
    // MARK: - DTOs (OpenAI-compatible)
    
    public struct ChatRequest: Codable, Sendable {
        public let model: String?
        public let models: [String]? // For OpenRouter auto-fallbacks
        public let messages: [Message]
        public let temperature: Double?
        public let maxTokens: Int?
        public let stream: Bool?
        public let provider: ProviderPreferences?
        
        public struct ProviderPreferences: Codable, Sendable {
            public let order: [String]?
            public let allowFallbacks: Bool?
            
            public enum CodingKeys: String, CodingKey {
                case order
                case allowFallbacks = "allow_fallbacks"
            }
            
            public init(order: [String]? = nil, allowFallbacks: Bool? = nil) {
                self.order = order
                self.allowFallbacks = allowFallbacks
            }
        }
        
        public struct Message: Codable, Sendable {
            public let role: String  // "system", "user", "assistant"
            public let content: String
            
            public init(role: String, content: String) {
                self.role = role
                self.content = content
            }
        }
        
        public enum CodingKeys: String, CodingKey {
            case model
            case models
            case messages
            case temperature
            case maxTokens = "max_tokens"
            case stream
            case provider
        }
        
        public init(
            model: String? = nil,
            models: [String]? = nil,
            messages: [Message],
            temperature: Double? = nil,
            maxTokens: Int? = nil,
            stream: Bool? = nil,
            provider: ProviderPreferences? = nil
        ) {
            self.model = model
            self.models = models
            self.messages = messages
            self.temperature = temperature
            self.maxTokens = maxTokens
            self.stream = stream
            self.provider = provider
        }
    }
    
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
    
    // MARK: - Properties
    
    private let apiClient: CloudAPIClient
    
    // MARK: - Initialization
    
    public init(config: Config = .default) throws {
        guard !config.apiKey.isEmpty else {
            throw GemmaServerError.invalidRequestStructure(
                details: "OPENROUTER_API_KEY not set. Get your key at https://openrouter.ai/keys"
            )
        }
        
        let apiConfig = CloudAPIConfig(
            apiKey: config.apiKey,
            baseURL: config.baseURL,
            timeout: config.timeout,
            maxRetries: config.maxRetries,
            extraHeaders: [
                "HTTP-Referer": "GemmaServer/0.2.0",
                "X-Title": "GemmaServer"
            ]
        )
        
        self.apiClient = CloudAPIClient(config: apiConfig)
    }
    
    // MARK: - API Methods
    
    /// Generate completion from cloud model
    public func chat(request: ChatRequest) async throws -> ChatResponse {
        let encoder = JSONEncoder()
        let body = try encoder.encode(request)
        
        return try await apiClient.performRequest(
            endpoint: "chat/completions",
            method: "POST",
            body: body,
            responseType: ChatResponse.self
        )
    }
    
    /// Stream completion from cloud model
    public func chatStream(request: ChatRequest) async throws -> AsyncThrowingStream<ChatStreamResponse, Error> {
        let encoder = JSONEncoder()
        let body = try encoder.encode(request)
        
        return try await apiClient.performStreamingRequest(
            endpoint: "chat/completions",
            method: "POST",
            body: body,
            responseType: ChatStreamResponse.self
        )
    }
    
    /// Fetch available models from OpenRouter
    public func getModels() async throws -> [OpenRouterModel] {
        let response = try await apiClient.performRequest(
            endpoint: "models",
            method: "GET",
            responseType: ModelsResponse.self
        )
        return response.data
    }
    
    // MARK: - Metrics
    
    public func getMetrics() async -> (requests: Int, errors: Int) {
        return await apiClient.getMetrics()
    }
}