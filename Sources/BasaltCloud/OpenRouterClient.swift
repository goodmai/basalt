import Foundation
import BasaltCore

public actor OpenRouterClient: Sendable {
    private let apiClient: CloudAPIClient

    public init(config: Config = .default, configuration: URLSessionConfiguration = .default) throws {
        self.apiClient = CloudAPIClient(config: try Self.apiConfig(config), configuration: configuration)
    }

    public static func isOpenRouter(_ url: URL) -> Bool {
        guard let host = url.host?.lowercased() else { return false }
        return host == "openrouter.ai" || host.hasSuffix(".openrouter.ai")
    }

    static func apiConfig(_ config: Config) throws(BasaltError) -> CloudAPIConfig {
        guard !config.apiKey.isEmpty || !isOpenRouter(config.baseURL) else {
            throw .invalidRequestStructure(
                details: "OPENROUTER_API_KEY not set. Get your key at https://openrouter.ai/keys"
            )
        }
        return CloudAPIConfig(
            apiKey: config.apiKey,
            baseURL: config.baseURL,
            timeout: config.timeout,
            maxRetries: config.maxRetries,
            extraHeaders: [
                "HTTP-Referer": "Basalt/\(HealthResponse.version)",
                "X-Title": "Basalt"
            ]
        )
    }

    public func chat(request: ChatRequest) async throws -> ChatResponse {
        try await apiClient.performRequest(
            endpoint: "chat/completions",
            method: "POST",
            body: try JSONEncoder().encode(request),
            responseType: ChatResponse.self
        )
    }

    public func chatStream(request: ChatRequest) async throws -> AsyncThrowingStream<ChatStreamResponse, Error> {
        try await apiClient.performStreamingRequest(
            endpoint: "chat/completions",
            method: "POST",
            body: try JSONEncoder().encode(request),
            responseType: ChatStreamResponse.self
        )
    }

    public func getModels() async throws -> [OpenRouterModel] {
        try await apiClient.performRequest(
            endpoint: "models",
            method: "GET",
            responseType: ModelsResponse.self
        ).data
    }

    public func getMetrics() async -> (requests: Int, errors: Int) {
        await apiClient.getMetrics()
    }
}

extension OpenRouterClient {
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

    public struct ChatRequest: Codable, Sendable {
        public let model: String?
        public let models: [String]?
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
            public let role: String
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
}
