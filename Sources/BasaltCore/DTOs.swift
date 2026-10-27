import Foundation

// MARK: — Request DTOs

/// A function the model may call (an OpenAI `function` tool). The JSON-Schema of its
/// parameters is kept as JSON text so the request stays Codable and Sendable.
public struct ToolDefinition: Codable, Sendable, Equatable {
    public let name: String
    public let description: String?
    public let parametersJSON: String?

    public init(name: String, description: String? = nil, parametersJSON: String? = nil) {
        self.name = name
        self.description = description
        self.parametersJSON = parametersJSON
    }
}

/// A call the model made: a call id, the function name, and its arguments as JSON text.
public struct ToolCallOutput: Codable, Sendable, Equatable {
    public let id: String
    public let name: String
    public let arguments: String

    public init(id: String, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}

public struct ToolChoice: Codable, Sendable, Equatable {
    public let forcedTool: String?
    public let parallel: Bool
    public let callsDisabled: Bool

    public init(forcedTool: String? = nil, parallel: Bool = true, callsDisabled: Bool = false) {
        self.forcedTool = forcedTool
        self.parallel = parallel
        self.callsDisabled = callsDisabled
    }
}

public enum ResponseFormat: Codable, Sendable, Equatable {
    case jsonObject
    case jsonSchema(String, strict: Bool = false)

    public func prepared() throws(UnsupportedJSONSchema) -> PreparedJSONSchema {
        switch self {
        case .jsonObject: try JSONSchemaGate.prepare(#"{"type": "object"}"#, strict: true)
        case .jsonSchema(let schema, let strict): try JSONSchemaGate.prepare(schema, strict: strict)
        }
    }
}

/// Единый DTO для генерации — используется и в MCP, и в REST (A2A).
public struct GenerationRequest: Codable, Sendable {
    public let prompt: String
    public private(set) var maxTokens: Int?
    public let temperature: Double?
    public let topP: Double?
    public let minP: Double?
    public let dynamicTemperature: Bool?
    public let enableThinking: Bool?
    public let reasoningEffort: String?
    public let isServiceRequest: Bool?
    /// Functions offered to the model; rendered into the chat template's tool section.
    public let tools: [ToolDefinition]?
    public let toolChoice: ToolChoice?
    public private(set) var cacheKey: String?
    public let messages: [ChatMessageInput]?
    public private(set) var assistantPrefix: String? = nil
    public private(set) var topK: Int? = nil
    public private(set) var presencePenalty: Double? = nil
    public private(set) var repetitionPenalty: Double? = nil
    public private(set) var thinkingBudget: Int? = nil
    public private(set) var responseFormat: ResponseFormat? = nil
    public private(set) var sessionId: String? = nil
    public private(set) var budgetKey: String? = nil

    enum CodingKeys: String, CodingKey {
        case prompt, maxTokens, temperature, topP, minP, dynamicTemperature, enableThinking, reasoningEffort
        case isServiceRequest, tools, toolChoice, cacheKey, messages
        case topK, presencePenalty, repetitionPenalty, thinkingBudget, responseFormat
    }

    public init(
        prompt: String,
        maxTokens: Int? = nil,
        temperature: Double? = 0.7,
        topP: Double? = 0.9,
        minP: Double? = nil,
        dynamicTemperature: Bool? = nil,
        enableThinking: Bool? = true,
        reasoningEffort: String? = nil,
        isServiceRequest: Bool? = false,
        tools: [ToolDefinition]? = nil,
        toolChoice: ToolChoice? = nil,
        cacheKey: String? = nil,
        messages: [ChatMessageInput]? = nil,
        assistantPrefix: String? = nil,
        topK: Int? = nil,
        presencePenalty: Double? = nil,
        repetitionPenalty: Double? = nil,
        thinkingBudget: Int? = nil,
        responseFormat: ResponseFormat? = nil
    ) {
        self.prompt = prompt
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.minP = minP
        self.dynamicTemperature = dynamicTemperature
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
        self.isServiceRequest = isServiceRequest
        self.tools = tools
        self.toolChoice = toolChoice
        self.cacheKey = cacheKey
        self.messages = messages
        self.assistantPrefix = assistantPrefix
        self.topK = topK
        self.presencePenalty = presencePenalty
        self.repetitionPenalty = repetitionPenalty
        self.thinkingBudget = thinkingBudget
        self.responseFormat = responseFormat
    }

    /// The same request with another token budget. Kept next to the stored properties so a new
    /// field cannot be forgotten by a caller that rebuilds the request field by field.
    public func withMaxTokens(_ maxTokens: Int) -> GenerationRequest {
        var copy = self
        copy.maxTokens = maxTokens
        return copy
    }

    public func withRouting(sessionId: String?, budgetKey: String?) -> GenerationRequest {
        var copy = self
        copy.sessionId = sessionId
        copy.budgetKey = budgetKey
        return copy
    }

    public func withCacheKey(_ cacheKey: String?) -> GenerationRequest {
        var copy = self
        copy.cacheKey = cacheKey
        return copy
    }

    public static let maximumTokens = 131_072
    public static let thinkingTransitionHeadroom = 64

    public static func thinkingAnswerReserve(maxTokens: Int) -> Int {
        min(512, maxTokens / 4)
    }

    public func validated(defaultMaxTokens: Int = 4096) throws(BasaltError) -> GenerationRequest {
        guard !prompt.isEmpty || messages?.isEmpty == false else {
            throw .invalidRequestStructure(details: "prompt must not be empty")
        }
        
        let maxT = maxTokens ?? defaultMaxTokens
        guard 1...Self.maximumTokens ~= maxT else {
            throw .contextLengthExceeded(maxTokens: Self.maximumTokens, requested: maxT)
        }

        try Self.require(temperature, in: 0.0...2.0, "temperature must be in [0.0, 2.0]")
        try Self.require(topP, in: 0.0...1.0, "top_p must be in [0.0, 1.0]")
        try Self.require(minP, in: 0.0...1.0, "min_p must be in [0.0, 1.0]")
        try Self.require(topK, in: 0...Int.max, "top_k must be >= 0")
        try Self.require(presencePenalty, in: -2.0...2.0, "presence_penalty must be in [-2.0, 2.0]")
        try Self.require(repetitionPenalty, in: Double.leastNonzeroMagnitude...2.0, "repetition_penalty must be in (0.0, 2.0]")
        try Self.require(thinkingBudget, in: 0...Self.maximumTokens, "thinking_budget must be in [0, \(Self.maximumTokens)]")
        if thinkingBudget != nil, !requestDisablesThinking, maxT - Self.thinkingAnswerReserve(maxTokens: maxT) < Self.thinkingTransitionHeadroom {
            throw .invalidRequestStructure(details:
                "max_tokens \(maxT) cannot hold the thinking_budget transition (\(Self.thinkingTransitionHeadroom) tokens) and an answer reserve of min(512, max_tokens/4)")
        }

        var validated = self
        validated.maxTokens = maxT
        return validated
    }

    public var requestDisablesThinking: Bool {
        enableThinking == false || ReasoningPolicy(rawValue: reasoningEffort) == .disabled
    }

    private static func require<Value: Comparable>(
        _ value: Value?, in range: ClosedRange<Value>, _ message: String
    ) throws(BasaltError) {
        if let value, !range.contains(value) { throw .invalidRequestStructure(details: message) }
    }
}

/// Heuristic detector for background service requests (such as thread title generation)
public enum ServiceRequestDetector: Sendable {
    public static func isTitleOrServiceRequest(prompt: String, maxTokens: Int? = nil) -> Bool {
        // Only the tail: the title instruction is the latest turn, while the history before it
        // can quote anything (a file mentioning "short title" would otherwise flag every turn).
        let p = String(prompt.suffix(1500)).lowercased()
        let keywords = [
            "thread title", "conversation title", "generate a title",
            "short title", "title for this", "title of this", "summarize in 3-5 words",
            "title for this session", "suggest a title", "name this conversation",
            "name this chat", "give this conversation a title",
            "word title", "summarize this coding conversation", "summarize this conversation in"
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
    /// The model called a function (parsed by the model's tool-call format).
    case toolCall(ToolCallOutput)
    
    enum CodingKeys: String, CodingKey {
        case type
        case text
        case reasoning
        case metadata
        case toolCall
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
        case "toolCall":
            self = .toolCall(try container.decode(ToolCallOutput.self, forKey: .toolCall))
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
        case .toolCall(let call):
            try container.encode("toolCall", forKey: .type)
            try container.encode(call, forKey: .toolCall)
        }
    }
}

public struct GenerationResponse: Codable, Sendable {
    public let generatedText: String
    public let reasoningText: String?
    public let promptTokens: Int
    public let cachedPromptTokens: Int
    public let completionTokens: Int
    public let tokensPerSecond: Double
    public let generationTime: Double
    public let timeToFirstToken: Double
    public let memory: MemoryUsage
    public let finishReason: FinishReason
    /// Functions the model called, in order; nil when it called none.
    public let toolCalls: [ToolCallOutput]?
    public let reasoningTokens: Int?

    public init(
        generatedText: String,
        reasoningText: String? = nil,
        promptTokens: Int,
        cachedPromptTokens: Int = 0,
        completionTokens: Int,
        tokensPerSecond: Double,
        generationTime: Double,
        timeToFirstToken: Double,
        memory: MemoryUsage,
        finishReason: FinishReason,
        toolCalls: [ToolCallOutput]? = nil,
        reasoningTokens: Int? = nil
    ) {
        self.generatedText = generatedText
        self.reasoningText = reasoningText
        self.promptTokens = promptTokens
        self.cachedPromptTokens = cachedPromptTokens
        self.completionTokens = completionTokens
        self.tokensPerSecond = tokensPerSecond
        self.generationTime = generationTime
        self.timeToFirstToken = timeToFirstToken
        self.memory = memory
        self.finishReason = finishReason
        self.toolCalls = toolCalls
        self.reasoningTokens = reasoningTokens
    }

    enum CodingKeys: String, CodingKey {
        case generatedText, reasoningText, promptTokens, cachedPromptTokens, completionTokens
        case tokensPerSecond, generationTime, timeToFirstToken, memory, finishReason, toolCalls, reasoningTokens
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        generatedText = try c.decode(String.self, forKey: .generatedText)
        reasoningText = try c.decodeIfPresent(String.self, forKey: .reasoningText)
        promptTokens = try c.decode(Int.self, forKey: .promptTokens)
        cachedPromptTokens = try c.decodeIfPresent(Int.self, forKey: .cachedPromptTokens) ?? 0
        completionTokens = try c.decode(Int.self, forKey: .completionTokens)
        tokensPerSecond = try c.decode(Double.self, forKey: .tokensPerSecond)
        generationTime = try c.decode(Double.self, forKey: .generationTime)
        timeToFirstToken = try c.decode(Double.self, forKey: .timeToFirstToken)
        memory = try c.decode(MemoryUsage.self, forKey: .memory)
        finishReason = try c.decode(FinishReason.self, forKey: .finishReason)
        toolCalls = try c.decodeIfPresent([ToolCallOutput].self, forKey: .toolCalls)
        reasoningTokens = try c.decodeIfPresent(Int.self, forKey: .reasoningTokens)
    }

    public struct MemoryUsage: Codable, Sendable {
        public let peakBytes: Int
        public let activeBytes: Int
        public let cacheBytes: Int

        package init(peakBytes: Int, activeBytes: Int, cacheBytes: Int) {
            self.peakBytes = peakBytes
            self.activeBytes = activeBytes
            self.cacheBytes = cacheBytes
        }
    }

    public enum FinishReason: String, Codable, Sendable {
        case stop
        case length
        case error
        case repetition

        public var isTruncated: Bool { self != .stop }
    }
}

// MARK: — Health / Info

public struct HealthResponse: Codable, Sendable {
    public let status: String
    public let modelId: String?
    public let isReady: Bool
    public let version: String

    public static let version = "0.3.0"

    package init(status: String, modelId: String?, isReady: Bool, version: String) {
        self.status = status
        self.modelId = modelId
        self.isReady = isReady
        self.version = version
    }
}

// MARK: — Error envelope (REST only)

public struct ErrorResponse: Codable, Sendable {
    public let error: String
    public let code: Int

    package init(error: String, code: Int) {
        self.error = error
        self.code = code
    }
}
