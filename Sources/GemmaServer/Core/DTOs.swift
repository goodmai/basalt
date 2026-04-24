import Foundation

// MARK: — Request DTOs

/// Единый DTO для генерации — используется и в MCP, и в REST (A2A).
public struct GenerationRequest: Codable, Sendable {
    public let prompt: String
    public let maxTokens: Int?
    public let temperature: Double?
    public let topP: Double?

    public init(
        prompt: String,
        maxTokens: Int? = nil,
        temperature: Double? = 0.7,
        topP: Double? = 0.9
    ) {
        self.prompt = prompt
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
    }

    public func validated(defaultMaxTokens: Int = 4096) throws(GemmaServerError) -> GenerationRequest {
        guard !prompt.isEmpty else {
            throw .invalidRequestStructure(details: "prompt must not be empty")
        }
        let maxT = maxTokens ?? defaultMaxTokens
        guard maxT > 0 && maxT <= 65_536 else {
            throw .contextLengthExceeded(maxTokens: 65_536, requested: maxT)
        }
        let temp = temperature ?? 0.7
        guard temp >= 0.0 && temp <= 2.0 else {
            throw .invalidRequestStructure(details: "temperature must be in [0.0, 2.0]")
        }
        return GenerationRequest(prompt: prompt, maxTokens: maxT, temperature: temp, topP: topP ?? 0.9)
    }
}

// MARK: — Response DTOs

public struct GenerationResponse: Codable, Sendable {
    public let generatedText: String
    public let promptTokens: Int
    public let completionTokens: Int
    public let tokensPerSecond: Double
    public let generationTime: Double
    public let finishReason: FinishReason

    public enum FinishReason: String, Codable, Sendable {
        case stop
        case length
        case error
    }
}

// MARK: — Health / Info

public struct HealthResponse: Codable, Sendable {
    public let status: String
    public let modelId: String?
    public let isReady: Bool
    public let version: String

    public static let version = "0.1.0"
}

// MARK: — Error envelope (REST only)

public struct ErrorResponse: Codable, Sendable {
    public let error: String
    public let code: Int
}
