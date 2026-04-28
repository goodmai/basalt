import Foundation

/// OpenRouter API client for cloud model inference
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
        public let model: String
        public let messages: [Message]
        public let temperature: Double?
        public let maxTokens: Int?
        public let stream: Bool?
        
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
            case messages
            case temperature
            case maxTokens = "max_tokens"
            case stream
        }
        
        public init(model: String, messages: [Message], temperature: Double? = nil, maxTokens: Int? = nil, stream: Bool? = nil) {
            self.model = model
            self.messages = messages
            self.temperature = temperature
            self.maxTokens = maxTokens
            self.stream = stream
        }
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
    
    // MARK: - Properties
    
    private let config: Config
    private let session: URLSession
    private var requestCount: Int = 0
    private var errorCount: Int = 0
    
    // MARK: - Initialization
    
    public init(config: Config = .default) throws {
        guard !config.apiKey.isEmpty else {
            throw GemmaServerError.invalidRequestStructure(
                details: "OPENROUTER_API_KEY not set. Get your key at https://openrouter.ai/keys"
            )
        }
        
        self.config = config
        
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = config.timeout
        configuration.timeoutIntervalForResource = config.timeout * 2
        self.session = URLSession(configuration: configuration)
    }
    
    // MARK: - API Methods
    
    /// Generate completion from cloud model
    public func chat(request: ChatRequest) async throws -> ChatResponse {
        let url = config.baseURL.appendingPathComponent("chat/completions")
        
        var httpRequest = URLRequest(url: url)
        httpRequest.httpMethod = "POST"
        httpRequest.setValue("Bearer \(config.apiKey)", forHTTPHeaderField: "Authorization")
        httpRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        httpRequest.setValue("GemmaServer/0.2.0", forHTTPHeaderField: "HTTP-Referer")
        httpRequest.setValue("GemmaServer", forHTTPHeaderField: "X-Title")
        
        let encoder = JSONEncoder()
        httpRequest.httpBody = try encoder.encode(request)
        
        // Retry logic with exponential backoff
        var lastError: Error?
        for attempt in 0..<config.maxRetries {
            do {
                let (data, response) = try await session.data(for: httpRequest)
                
                guard let httpResponse = response as? HTTPURLResponse else {
                    throw GemmaServerError.invalidRequestStructure(details: "Invalid response type")
                }
                
                // Handle HTTP errors
                guard (200...299).contains(httpResponse.statusCode) else {
                    let errorBody = String(data: data, encoding: .utf8) ?? "Unknown error"
                    
                    switch httpResponse.statusCode {
                    case 401:
                        throw GemmaServerError.authenticationFailed(
                            details: "Invalid OpenRouter API key"
                        )
                    case 429:
                        let retryAfter = httpResponse.value(forHTTPHeaderField: "Retry-After")
                        throw GemmaServerError.rateLimitExceeded(
                            retryAfter: retryAfter
                        )
                    case 500...599:
                        // Retry on server errors
                        if attempt < config.maxRetries - 1 {
                            let delay = pow(2.0, Double(attempt)) // Exponential backoff
                            try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                            continue
                        }
                        throw GemmaServerError.modelInferenceError(
                            details: "OpenRouter API error: \(errorBody)"
                        )
                    default:
                        throw GemmaServerError.invalidRequestStructure(
                            details: "HTTP \(httpResponse.statusCode): \(errorBody)"
                        )
                    }
                }
                
                // Parse response
                let decoder = JSONDecoder()
                let chatResponse = try decoder.decode(ChatResponse.self, from: data)
                
                recordSuccess()
                return chatResponse
                
            } catch let error as GemmaServerError {
                // Do not retry these errors
                if case .modelInferenceError = error {
                    // Handled above in 500-599
                } else {
                    recordError()
                    throw error
                }
                lastError = error
            } catch {
                lastError = error
                if attempt < config.maxRetries - 1 {
                    let delay = pow(2.0, Double(attempt))
                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                }
            }
        }
        
        recordError()
        throw lastError ?? GemmaServerError.modelInferenceError(
            details: "OpenRouter API request failed after \(config.maxRetries) attempts"
        )
    }
    
    // MARK: - Metrics
    
    private func recordSuccess() {
        requestCount += 1
    }
    
    private func recordError() {
        errorCount += 1
    }
    
    public func getMetrics() -> (requests: Int, errors: Int) {
        return (requestCount, errorCount)
    }
}
