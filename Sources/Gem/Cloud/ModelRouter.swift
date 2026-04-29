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
        case cloud(provider: CloudProvider, models: [String])
        
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
    
    /// Fetches models from OpenRouter and updates the local registry with exact pricing and context windows
    public func syncCloudModels() async throws {
        guard let client = cloudClient else { return }
        
        let fetchedModels = try await client.getModels()
        
        for model in fetchedModels {
            let promptPrice = Double(model.pricing.prompt) ?? 0.0
            let completionPrice = Double(model.pricing.completion) ?? 0.0
            
            // Average cost per 1M tokens as a unified simplified metric, 
            // though actual billing splits it.
            let avgCostPer1M = (promptPrice + completionPrice) / 2.0 * 1_000_000.0
            
            let info = ModelInfo(
                id: model.id,
                displayName: model.name,
                location: .cloud(provider: .openrouter, models: [model.id]),
                contextWindow: model.contextLength,
                estimatedRAM: nil,
                costPer1MTokens: avgCostPer1M > 0 ? avgCostPer1M : nil
            )
            
            modelRegistry[model.id] = info
        }
        
        log("Synced \(fetchedModels.count) models from OpenRouter")
    }
    
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
            
        case .cloud(_, let cloudModels):
            guard let client = cloudClient else {
                throw GemError.invalidRequestStructure(
                    details: "Cloud models not configured. Set OPENROUTER_API_KEY"
                )
            }
            return try await routeToCloud(
                client: client,
                models: cloudModels,
                request: request
            )
        }
    }
    
    /// Route stream request to appropriate model (local or cloud)
    public func routeStream(request: GenerationRequest, preferredModel: String?) async throws -> AsyncStream<StreamChunk> {
        let modelId = preferredModel ?? "auto"
        
        let decision = try await makeRoutingDecision(
            modelId: modelId
        )
        
        switch decision {
        case .local(let path):
            // Fallback for local models
            _ = path // local models have their own orchestrator stream logic, but we route it
            return try await localOrchestrator.generateStream(request: request)
            
        case .cloud(_, let cloudModels):
            guard let client = cloudClient else {
                throw GemError.invalidRequestStructure(
                    details: "Cloud models not configured. Set OPENROUTER_API_KEY"
                )
            }
            return try await routeToCloudStream(
                client: client,
                models: cloudModels,
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
            let cloudModels = mapToCloudModels(modelId)
            return .cloud(provider: .openrouter, models: cloudModels)
        }
        
        // Strategy: Auto or Hybrid
        // Check if model explicitly cloud-only
        if isCloudOnlyModel(modelId) {
            let cloudModels = mapToCloudModels(modelId)
            return .cloud(provider: .openrouter, models: cloudModels)
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
                let cloudModels = mapToCloudModels(modelId)
                return .cloud(provider: .openrouter, models: cloudModels)
            }
            
            // No cloud fallback available
            throw GemError.outOfMemory(
                requestedMB: Int(requiredRAM),
                availableMB: Int(availableRAM / 1_024 / 1_024)
            )
        }
        
        // Unknown model → try cloud
        if cloudClient != nil {
            let cloudModels = mapToCloudModels(modelId)
            return .cloud(provider: .openrouter, models: cloudModels)
        }
        
        throw GemError.invalidRequestStructure(
            details: "Model '\(modelId)' not found in local or cloud registry, and cloud fallback is disabled"
        )
    }
    
    public func estimateCost(model: String, promptTokens: Int, completionTokens: Int) -> Double {
        // Use the first mapped model for cost estimation
        let mappedId = mapToCloudModels(model).first ?? model
        
        // 1. Try to find precise cost from synced registry
        if let info = modelRegistry[mappedId], let cost = info.costPer1MTokens {
            return (Double(promptTokens + completionTokens) / 1_000_000.0) * cost
        }
        
        // 2. OpenRouter fallback pricing (approximate)
        let pricing: [String: (input: Double, output: Double)] = [
            "openai/gpt-4-turbo": (10.0, 30.0),        // per 1M tokens
            "anthropic/claude-3.5-sonnet": (3.0, 15.0),
            "google/gemini-pro-1.5": (1.25, 5.0)
        ]
        
        guard let price = pricing[mappedId] else {
            return 0.0 // Unknown model
        }
        
        let inputCost = Double(promptTokens) / 1_000_000.0 * price.input
        let outputCost = Double(completionTokens) / 1_000_000.0 * price.output
        
        return inputCost + outputCost
    }
    
    // MARK: - Private Implementations
    
    private func routeToLocal(path: String, request: GenerationRequest) async throws -> GenerationResponse {
        // Use existing local orchestrator
        return try await localOrchestrator.generate(request: request)
    }
    
    private func routeToCloud(
        client: OpenRouterClient,
        models: [String],
        request: GenerationRequest
    ) async throws -> GenerationResponse {
        
        let chatRequest: OpenRouterClient.ChatRequest
        
        if models.count == 1 {
            chatRequest = OpenRouterClient.ChatRequest(
                model: models[0],
                messages: [
                    .init(role: "user", content: request.prompt)
                ],
                temperature: request.temperature,
                maxTokens: request.maxTokens,
                stream: false
            )
        } else {
            chatRequest = OpenRouterClient.ChatRequest(
                models: models,
                messages: [
                    .init(role: "user", content: request.prompt)
                ],
                temperature: request.temperature,
                maxTokens: request.maxTokens,
                stream: false,
                provider: .init(allowFallbacks: true)
            )
        }
        
        let startTime = ContinuousClock.now
        let chatResponse = try await client.chat(request: chatRequest)
        let endTime = ContinuousClock.now
        
        let duration = startTime.duration(to: endTime).components.seconds
        let durationSeconds = Double(duration) > 0 ? Double(duration) : 0.001
        
        guard let choice = chatResponse.choices.first else {
            throw GemError.modelInferenceError(
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
    
    private func routeToCloudStream(
        client: OpenRouterClient,
        models: [String],
        request: GenerationRequest
    ) async throws -> AsyncStream<StreamChunk> {
        
        let chatRequest: OpenRouterClient.ChatRequest
        
        if models.count == 1 {
            chatRequest = OpenRouterClient.ChatRequest(
                model: models[0],
                messages: [
                    .init(role: "user", content: request.prompt)
                ],
                temperature: request.temperature,
                maxTokens: request.maxTokens,
                stream: true
            )
        } else {
            chatRequest = OpenRouterClient.ChatRequest(
                models: models,
                messages: [
                    .init(role: "user", content: request.prompt)
                ],
                temperature: request.temperature,
                maxTokens: request.maxTokens,
                stream: true,
                provider: .init(allowFallbacks: true)
            )
        }
        
        let startTime = ContinuousClock.now
        let sourceStream = try await client.chatStream(request: chatRequest)
        
        return AsyncStream { continuation in
            Task {
                do {
                    for try await response in sourceStream {
                        guard let choice = response.choices.first else { continue }
                        
                        if let text = choice.delta.content {
                            continuation.yield(.text(text))
                        }
                        
                        if let reason = choice.finishReason, reason != "null" {
                            let duration = startTime.duration(to: ContinuousClock.now).components.seconds
                            let durationSeconds = Double(duration) > 0 ? Double(duration) : 0.001
                            
                            let metadata = GenerationResponse(
                                generatedText: "", // Final metadata chunk doesn't carry text
                                promptTokens: response.usage?.promptTokens ?? 0,
                                completionTokens: response.usage?.completionTokens ?? 0,
                                tokensPerSecond: Double(response.usage?.completionTokens ?? 0) / durationSeconds,
                                generationTime: durationSeconds,
                                timeToFirstToken: 0.0,
                                memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0),
                                finishReason: reason == "length" ? .length : .stop
                            )
                            continuation.yield(.metadata(metadata))
                            break
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish()
                }
            }
        }
    }
    
    private func isCloudOnlyModel(_ modelId: String) -> Bool {
        let cloudPrefixes = ["gpt-", "claude-", "gemini-", "o1-", "o3-"]
        return cloudPrefixes.contains { modelId.hasPrefix($0) }
    }
    
    private func mapToCloudModels(_ modelId: String) -> [String] {
        let mapping: [String: [String]] = [
            "gpt-4": ["openai/gpt-4-turbo", "openai/gpt-4"],
            "gpt-4o": ["openai/gpt-4o", "openai/gpt-4o-2024-08-06"],
            "claude": ["anthropic/claude-3.5-sonnet", "anthropic/claude-3-opus"],
            "claude-3.5": ["anthropic/claude-3.5-sonnet", "anthropic/claude-3-opus"],
            "gemini": ["google/gemini-pro-1.5"],
            "o1": ["openai/o1-preview"],
            "o3": ["openai/o3-mini"]
        ]
        
        return mapping[modelId] ?? [modelId]
    }
    
    private static func defaultModels() -> [String: ModelInfo] {
        var registry: [String: ModelInfo] = [:]
        
        registry["gpt-4"] = ModelInfo(
            id: "gpt-4",
            displayName: "GPT-4 Turbo",
            location: .cloud(provider: .openrouter, models: ["openai/gpt-4-turbo", "openai/gpt-4"]),
            contextWindow: 128_000,
            estimatedRAM: nil,
            costPer1MTokens: 10.0
        )
        
        registry["claude-3.5"] = ModelInfo(
            id: "claude-3.5",
            displayName: "Claude 3.5 Sonnet",
            location: .cloud(provider: .openrouter, models: ["anthropic/claude-3.5-sonnet"]),
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
