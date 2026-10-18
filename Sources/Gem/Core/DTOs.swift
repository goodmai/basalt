import Foundation

// MARK: — Request DTOs

/// Единый DTO для генерации — используется и в MCP, и в REST (A2A).
public struct GenerationRequest: Codable, Sendable {
    public let prompt: String
    public let maxTokens: Int?
    public let temperature: Double?
    public let topP: Double?
    public let enableThinking: Bool?
    public let reasoningEffort: String?
    public let isServiceRequest: Bool?
    public let prefillDelta: String?
    public let cachedTokensEstimate: Int?

    public init(
        prompt: String,
        maxTokens: Int? = nil,
        temperature: Double? = 0.7,
        topP: Double? = 0.9,
        enableThinking: Bool? = true,
        reasoningEffort: String? = nil,
        isServiceRequest: Bool? = false,
        prefillDelta: String? = nil,
        cachedTokensEstimate: Int? = nil
    ) {
        self.prompt = prompt
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
        self.isServiceRequest = isServiceRequest
        self.prefillDelta = prefillDelta
        self.cachedTokensEstimate = cachedTokensEstimate
    }

    public func validated(defaultMaxTokens: Int = 4096) throws(GemError) -> GenerationRequest {
        guard !prompt.isEmpty else {
            throw .invalidRequestStructure(details: "prompt must not be empty")
        }
        
        let maxT = maxTokens ?? defaultMaxTokens
        guard 1...131_072 ~= maxT else {
            throw .contextLengthExceeded(maxTokens: 131_072, requested: maxT)
        }
        
        let temp = temperature ?? 0.7
        guard 0.0...2.0 ~= temp else {
            throw .invalidRequestStructure(details: "temperature must be in [0.0, 2.0]")
        }
        
        return GenerationRequest(
            prompt: prompt,
            maxTokens: maxT,
            temperature: temp,
            topP: topP ?? 0.9,
            enableThinking: enableThinking,
            reasoningEffort: reasoningEffort,
            isServiceRequest: isServiceRequest,
            prefillDelta: prefillDelta,
            cachedTokensEstimate: cachedTokensEstimate
        )
    }
}

/// Heuristic detector for background service requests (such as thread title generation)
public enum ServiceRequestDetector: Sendable {
    public static func isTitleOrServiceRequest(prompt: String, maxTokens: Int? = nil) -> Bool {
        let p = prompt.lowercased()
        let keywords = [
            "thread title", "conversation title", "generate a title",
            "short title", "title for this", "title of this", "summarize in 3-5 words",
            "title for this session", "suggest a title", "name this conversation",
            "name this chat", "give this conversation a title"
        ]
        for kw in keywords {
            if p.contains(kw) { return true }
        }
        if let max = maxTokens, max <= 64, (p.contains("title") || p.contains("summarize")) {
            return true
        }
        return false
    }

    /// Fast path: truncates massive prompt context for lightweight title requests
    public static func compactTitlePrompt(_ fullPrompt: String) -> String {
        if fullPrompt.count > 1000 {
            let tail = String(fullPrompt.suffix(800))
            return "Generate a short 3-5 word title for this discussion:\n\(tail)"
        }
        return fullPrompt
    }
}

// MARK: — Response DTOs

public enum StreamChunk: Sendable, Codable {
    case text(String)
    case reasoning(String)
    case metadata(GenerationResponse)
    
    enum CodingKeys: String, CodingKey {
        case type
        case text
        case reasoning
        case metadata
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        
        switch type {
        case "text":
            let text = try container.decode(String.self, forKey: .text)
            self = .text(text)
        case "reasoning":
            let text = try container.decode(String.self, forKey: .reasoning)
            self = .reasoning(text)
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
        case .reasoning(let reasoning):
            try container.encode("reasoning", forKey: .type)
            try container.encode(reasoning, forKey: .reasoning)
        case .metadata(let metadata):
            try container.encode("metadata", forKey: .type)
            try container.encode(metadata, forKey: .metadata)
        }
    }
}

public struct GenerationResponse: Codable, Sendable {
    public let generatedText: String
    public let reasoningText: String?
    public let promptTokens: Int
    public let completionTokens: Int
    public let tokensPerSecond: Double
    public let generationTime: Double
    public let timeToFirstToken: Double
    public let memory: MemoryUsage
    public let finishReason: FinishReason

    public init(
        generatedText: String,
        reasoningText: String? = nil,
        promptTokens: Int,
        completionTokens: Int,
        tokensPerSecond: Double,
        generationTime: Double,
        timeToFirstToken: Double,
        memory: MemoryUsage,
        finishReason: FinishReason
    ) {
        self.generatedText = generatedText
        self.reasoningText = reasoningText
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.tokensPerSecond = tokensPerSecond
        self.generationTime = generationTime
        self.timeToFirstToken = timeToFirstToken
        self.memory = memory
        self.finishReason = finishReason
    }

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

    public static let version = "0.2.0"
}

// MARK: — Error envelope (REST only)

public struct ErrorResponse: Codable, Sendable {
    public let error: String
    public let code: Int
}
