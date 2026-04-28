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
        guard 1...65_536 ~= maxT else {
            throw .contextLengthExceeded(maxTokens: 65_536, requested: maxT)
        }
        
        let temp = temperature ?? 0.7
        guard 0.0...2.0 ~= temp else {
            throw .invalidRequestStructure(details: "temperature must be in [0.0, 2.0]")
        }
        
        return GenerationRequest(prompt: prompt, maxTokens: maxT, temperature: temp, topP: topP ?? 0.9)
    }
}

// MARK: — Response DTOs

public enum StreamChunk: Sendable, Codable {
    case text(String)
    case metadata(GenerationResponse)
    
    enum CodingKeys: String, CodingKey {
        case type
        case text
        case metadata
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        
        switch type {
        case "text":
            let text = try container.decode(String.self, forKey: .text)
            self = .text(text)
        case "metadata":
            let metadata = try container.decode(GenerationResponse.self, forKey: .metadata)
            self = .metadata(metadata)
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type,
                in: container,
                debugDescription: "Unknown StreamChunk type: \(type)"
            )
        }
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        
        switch self {
        case .text(let text):
            try container.encode("text", forKey: .type)
            try container.encode(text, forKey: .text)
        case .metadata(let metadata):
            try container.encode("metadata", forKey: .type)
            try container.encode(metadata, forKey: .metadata)
        }
    }
}

public struct GenerationResponse: Codable, Sendable {
    public let generatedText: String
    public let promptTokens: Int
    public let completionTokens: Int
    public let tokensPerSecond: Double
    public let generationTime: Double
    public let timeToFirstToken: Double
    public let memory: MemoryUsage
    public let finishReason: FinishReason

    public struct MemoryUsage: Codable, Sendable {
        public let peakBytes: Int
        public let activeBytes: Int
        public let cacheBytes: Int
    }

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
