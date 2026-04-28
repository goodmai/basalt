import Foundation

public actor ModelRouter: Sendable {
    
    public enum RoutingStrategy: String, Sendable, Codable {
        case auto           // Smart routing based on resources
        case localOnly      // Never use cloud (privacy-first)
        case cloudOnly      // Always use cloud (no local inference)
        case hybrid         // Explicitly choose per request
    }
    
    public enum ModelLocation: Sendable {
        case local(path: String)
        case cloud(provider: CloudProvider, modelId: String)
        
        public enum CloudProvider: String, Sendable {
            case openrouter
            case openai
            case anthropic
        }
    }
    
    public struct ModelInfo: Sendable {
        public let id: String
        public let displayName: String
        public let location: ModelLocation
        public let contextWindow: Int
        public let estimatedRAM: Int64?       // MB (for local models)
        public let costPer1MTokens: Double?   // USD (for cloud models)
        
        public init(id: String, displayName: String, location: ModelLocation, contextWindow: Int, estimatedRAM: Int64?, costPer1MTokens: Double?) {
            self.id = id
            self.displayName = displayName
            self.location = location
            self.contextWindow = contextWindow
            self.estimatedRAM = estimatedRAM
            self.costPer1MTokens = costPer1MTokens
        }
    }
    
    private let localOrchestrator: ModelOrchestratorActor
    private let cloudClient: OpenRouterClient?
    private let strategy: RoutingStrategy
    private var modelRegistry: [String: ModelInfo] = [:]
    
    // For testing
    private let mockAvailableRAM: Int64?
    
    public init(
        localOrchestrator: ModelOrchestratorActor,
        cloudClient: OpenRouterClient?,
        strategy: RoutingStrategy = .auto,
        mockAvailableRAM: Int64? = nil
    ) {
        self.localOrchestrator = localOrchestrator
        self.cloudClient = cloudClient
        self.strategy = strategy
        self.mockAvailableRAM = mockAvailableRAM
        
        // Pre-populate registry with known models
        self.modelRegistry = Self.defaultModels()
    }
    
    // MARK: - Public API
    
    /// Route request to appropriate model (local or cloud)
    public func route(request: GenerationRequest, preferredModel: String?) async throws -> GenerationResponse {
        let modelId = preferredModel ?? "auto"
        
        // Determine routing decision
        let decision = try await makeRoutingDecision(
            modelId: modelId
        )
        
        switch decision {
        case .local(let path):
            return try await routeToLocal(path: path, request: request)
            
        case .cloud(_, let cloudModelId):
            guard let client = cloudClient else {
                throw GemmaServerError.invalidRequestStructure(
                    details: "Cloud models not configured. Set OPENROUTER_API_KEY"
                )
            }
            return try await routeToCloud(
                client: client,
                modelId: cloudModelId,
                request: request
            )
        }
    }
    
    // For tests and routing
    public func registerLocalModel(id: String, estimatedRAM: Int64) {
        modelRegistry[id] = ModelInfo(
            id: id,
            displayName: id,
            location: .local(path: id),
            contextWindow: 8192,
            estimatedRAM: estimatedRAM,
            costPer1MTokens: nil
        )
    }
    
    public func makeRoutingDecision(modelId: String) async throws -> ModelLocation {
        // Strategy: Local-only
        if strategy == .localOnly {
            return .local(path: modelId)
        }
        
        // Strategy: Cloud-only
        if strategy == .cloudOnly {
            let cloudModel = mapToCloudModel(modelId)
            return .cloud(provider: .openrouter, modelId: cloudModel)
        }
        
        // Strategy: Auto or Hybrid
        // Check if model explicitly cloud-only
        if isCloudOnlyModel(modelId) {
            let cloudModel = mapToCloudModel(modelId)
            return .cloud(provider: .openrouter, modelId: cloudModel)
        }
        
        // Check if local model available and sufficient RAM
        if let localInfo = modelRegistry[modelId],
           case .local = localInfo.location {
            
            let availableRAM = await getAvailableRAM()
            let requiredRAM = localInfo.estimatedRAM ?? 0
            
            // Sufficient RAM for local inference
            // Simple heuristic: we need at least requiredRAM * 1.5 safely
            if availableRAM > requiredRAM * 1_024 * 1_024 * 3 / 2 { 
                return .local(path: modelId)
            }
            
            // Insufficient RAM → fallback to cloud
            if strategy == .auto, cloudClient != nil {
                log("⚠️ Insufficient RAM for local model, falling back to cloud")
                let cloudModel = mapToCloudModel(modelId)
                return .cloud(provider: .openrouter, modelId: cloudModel)
            }
            
            // No cloud fallback available
            throw GemmaServerError.outOfMemory(
                requestedMB: Int(requiredRAM),
                availableMB: Int(availableRAM / 1_024 / 1_024)
            )
        }
        
        // Unknown model → try cloud
        if cloudClient != nil {
            let cloudModel = mapToCloudModel(modelId)
            return .cloud(provider: .openrouter, modelId: cloudModel)
        }
        
        throw GemmaServerError.invalidRequestStructure(
            details: "Model '\(modelId)' not found in local or cloud registry, and cloud fallback is disabled"
        )
    }
    
    public func estimateCost(model: String, promptTokens: Int, completionTokens: Int) -> Double {
        let mappedId = mapToCloudModel(model)
        
        // OpenRouter pricing (approximate)
        let pricing: [String: (input: Double, output: Double)] = [
            "openai/gpt-4-turbo": (0.01, 0.03),        // per 1K tokens
            "anthropic/claude-3.5-sonnet": (0.003, 0.015),
            "google/gemini-pro-1.5": (0.00125, 0.005)
        ]
        
        guard let price = pricing[mappedId] else {
            return 0.0 // Unknown model
        }
        
        let inputCost = Double(promptTokens) / 1000.0 * price.input
        let outputCost = Double(completionTokens) / 1000.0 * price.output
        
        return inputCost + outputCost
    }
    
    // MARK: - Private Implementations
    
    private func routeToLocal(path: String, request: GenerationRequest) async throws -> GenerationResponse {
        // Use existing local orchestrator
        return try await localOrchestrator.generate(request: request)
    }
    
    private func routeToCloud(
        client: OpenRouterClient,
        modelId: String,
        request: GenerationRequest
    ) async throws -> GenerationResponse {
        
        let chatRequest = OpenRouterClient.ChatRequest(
            model: modelId,
            messages: [
                .init(role: "user", content: request.prompt)
            ],
            temperature: request.temperature,
            maxTokens: request.maxTokens,
            stream: false
        )
        
        let startTime = ContinuousClock.now
        let chatResponse = try await client.chat(request: chatRequest)
        let endTime = ContinuousClock.now
        
        let duration = startTime.duration(to: endTime).components.seconds
        let durationSeconds = Double(duration) > 0 ? Double(duration) : 0.001
        
        guard let choice = chatResponse.choices.first else {
            throw GemmaServerError.modelInferenceError(
                details: "No response from cloud model"
            )
        }
        
        let usage = chatResponse.usage ?? .init(
            promptTokens: 0,
            completionTokens: 0,
            totalTokens: 0
        )
        
        return GenerationResponse(
            generatedText: choice.message.content,
            promptTokens: usage.promptTokens,
            completionTokens: usage.completionTokens,
            tokensPerSecond: Double(usage.completionTokens) / durationSeconds,
            generationTime: durationSeconds,
            timeToFirstToken: 0.0, // Not available from API without streaming
            memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0),
            finishReason: choice.finishReason == "length" ? .length : .stop
        )
    }
    
    private func isCloudOnlyModel(_ modelId: String) -> Bool {
        let cloudPrefixes = ["gpt-", "claude-", "gemini-", "o1-", "o3-"]
        return cloudPrefixes.contains { modelId.hasPrefix($0) }
    }
    
    private func mapToCloudModel(_ modelId: String) -> String {
        let mapping: [String: String] = [
            "gpt-4": "openai/gpt-4-turbo",
            "gpt-4o": "openai/gpt-4o",
            "claude": "anthropic/claude-3.5-sonnet",
            "claude-3.5": "anthropic/claude-3.5-sonnet",
            "gemini": "google/gemini-pro-1.5",
            "o1": "openai/o1-preview",
            "o3": "openai/o3-mini"
        ]
        
        return mapping[modelId] ?? modelId
    }
    
    private static func defaultModels() -> [String: ModelInfo] {
        var registry: [String: ModelInfo] = [:]
        
        registry["gpt-4"] = ModelInfo(
            id: "gpt-4",
            displayName: "GPT-4 Turbo",
            location: .cloud(provider: .openrouter, modelId: "openai/gpt-4-turbo"),
            contextWindow: 128_000,
            estimatedRAM: nil,
            costPer1MTokens: 10.0
        )
        
        registry["claude-3.5"] = ModelInfo(
            id: "claude-3.5",
            displayName: "Claude 3.5 Sonnet",
            location: .cloud(provider: .openrouter, modelId: "anthropic/claude-3.5-sonnet"),
            contextWindow: 200_000,
            estimatedRAM: nil,
            costPer1MTokens: 3.0
        )
        
        return registry
    }
    
    private func getAvailableRAM() async -> Int64 {
        if let mock = mockAvailableRAM {
            return mock
        }
        // In real app use SystemProfiler
        return Int64(ProcessInfo.processInfo.physicalMemory)
    }
    
    private func log(_ message: String) {
        fputs("[ModelRouter] \(message)\n", stderr)
    }
}
