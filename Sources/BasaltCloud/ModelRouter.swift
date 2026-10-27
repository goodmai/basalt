import Foundation
import BasaltCore

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
    
    public struct RoutingContext: Sendable {
        public let sessionId: String?
        public let virtualKey: String?

        public init(sessionId: String? = nil, virtualKey: String? = nil) {
            self.sessionId = sessionId
            self.virtualKey = virtualKey
        }

        public init(_ request: GenerationRequest) {
            self.init(sessionId: request.sessionId, virtualKey: request.budgetKey)
        }

        var budgetScopes: [CostTracker.BudgetScope] {
            [sessionId.map { .session($0) }, virtualKey.map { .virtualKey($0) }].compactMap { $0 }
        }
    }

    public struct CostRouting: Sendable {
        public let policy: StickyRoutingPolicy
        public let scorer: BradleyTerryRouter

        public init(policy: StickyRoutingPolicy, scorer: BradleyTerryRouter) {
            self.policy = policy
            self.scorer = scorer
        }
    }

    private struct SessionRoute {
        let tier: StickyRoutingPolicy.Tier
        let promptTokens: Int
    }

    private let localOrchestrator: ModelOrchestratorActor
    private let cloudClient: OpenRouterClient?
    private let strategy: RoutingStrategy
    private var modelRegistry: [String: ModelInfo] = [:]
    
    // For testing
    private let mockAvailableRAM: Int64?
    private let costRouting: CostRouting?
    private let costTracker: CostTracker?
    private let localFallbackModel: String?
    private let fallbackCloudModel: String?
    private var sessionRoutes: [String: SessionRoute] = [:]
    
    public init(
        localOrchestrator: ModelOrchestratorActor,
        cloudClient: OpenRouterClient?,
        strategy: RoutingStrategy = .auto,
        mockAvailableRAM: Int64? = nil,
        costRouting: CostRouting? = nil,
        costTracker: CostTracker? = nil,
        localFallbackModel: String? = nil,
        fallbackCloudModel: String? = nil
    ) {
        self.localOrchestrator = localOrchestrator
        self.cloudClient = cloudClient
        self.strategy = strategy
        self.mockAvailableRAM = mockAvailableRAM
        self.costRouting = costRouting
        self.costTracker = costTracker
        self.localFallbackModel = localFallbackModel
        self.fallbackCloudModel = fallbackCloudModel
        
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
    
    public func route(
        request: GenerationRequest, preferredModel: String?, context: RoutingContext = .init()
    ) async throws -> GenerationResponse {
        let modelId = resolveModel(preferredModel, request: request, context: context)
        switch try await makeRoutingDecision(modelId: modelId) {
        case .local(let path):
            return try await routeToLocal(path: path, request: request)
        case .cloud(_, let cloudModels):
            let client = try requireCloudClient()
            if let fallback = try await localFallbackIfBudgetExhausted(context) {
                return try await routeToLocal(path: fallback, request: request)
            }
            var lastError: Error?
            for model in cloudModels {
                let response: GenerationResponse
                do {
                    response = try await routeToCloud(client: client, model: model, request: request)
                } catch {
                    lastError = error
                    log("Cloud model \(model) failed, trying the next in the chain: \(error)")
                    continue
                }
                await recordCostLoggingFailure(model: model, response: response, context: context)
                return response
            }
            if let fallback = localFallbackModel {
                return try await routeToLocal(path: fallback, request: request)
            }
            throw lastError ?? BasaltError.modelInferenceError(details: "Empty cloud fallback chain")
        }
    }
    
    public func routeStream(
        request: GenerationRequest, preferredModel: String?, context: RoutingContext = .init()
    ) async throws -> AsyncThrowingStream<StreamChunk, Error> {
        let modelId = resolveModel(preferredModel, request: request, context: context)
        switch try await makeRoutingDecision(modelId: modelId) {
        case .local(let path):
            return try await routeToLocalStream(path: path, request: request)
        case .cloud(_, let cloudModels):
            let client = try requireCloudClient()
            if let fallback = try await localFallbackIfBudgetExhausted(context) {
                return try await routeToLocalStream(path: fallback, request: request)
            }
            var lastError: Error?
            for model in cloudModels {
                do {
                    return try await routeToCloudStream(client: client, model: model, request: request, context: context)
                } catch {
                    lastError = error
                    log("Cloud model \(model) failed to start streaming, trying the next in the chain: \(error)")
                }
            }
            if let fallback = localFallbackModel {
                return try await routeToLocalStream(path: fallback, request: request)
            }
            throw lastError ?? BasaltError.modelInferenceError(details: "Empty cloud fallback chain")
        }
    }

    func resolveModel(_ preferredModel: String?, request: GenerationRequest, context: RoutingContext) -> String {
        let modelId = preferredModel ?? "auto"
        guard modelId == "auto", let costRouting, let decision = stickyTier(request: request, context: context) else {
            return modelId
        }
        return costRouting.policy.model(for: decision.tier)
    }

    private func stickyTier(
        request: GenerationRequest, context: RoutingContext
    ) -> (tier: StickyRoutingPolicy.Tier, strongWinProbability: Double)? {
        guard let costRouting else { return nil }
        let previous = context.sessionId.flatMap { sessionRoutes[$0] }
        let probability = costRouting.scorer.strongWinProbability(for: request.prompt)
        let tier = costRouting.policy.tier(
            strongWinProbability: probability,
            previous: previous?.tier,
            warmPrefixTokens: previous?.promptTokens ?? 0
        )
        if let sessionId = context.sessionId {
            sessionRoutes[sessionId] = SessionRoute(tier: tier, promptTokens: request.prompt.utf8.count / 4)
        }
        return (tier, probability)
    }

    private func requireCloudClient() throws -> OpenRouterClient {
        guard let client = cloudClient else {
            throw BasaltError.invalidRequestStructure(
                details: "Cloud models not configured. Set OPENROUTER_API_KEY"
            )
        }
        return client
    }

    private func localFallbackIfBudgetExhausted(_ context: RoutingContext) async throws -> String? {
        guard let costTracker, await costTracker.isExhausted(scopes: context.budgetScopes) else { return nil }
        guard let localFallbackModel else {
            throw BasaltError.invalidRequestStructure(details: "Cloud budget exhausted and no local fallback model is configured")
        }
        log("Cloud budget exhausted, serving locally with \(localFallbackModel)")
        return localFallbackModel
    }

    private func recordCostLoggingFailure(model: String, response: GenerationResponse, context: RoutingContext) async {
        guard let costTracker else { return }
        let cost = estimateCost(model: model, promptTokens: response.promptTokens, completionTokens: response.completionTokens)
        do {
            try await costTracker.recordUsage(model: model, cost: cost, scopes: context.budgetScopes)
        } catch {
            log("Recording cloud cost for \(model) failed: \(error)")
        }
    }

    private func routeToLocalStream(path: String, request: GenerationRequest) async throws -> AsyncThrowingStream<StreamChunk, Error> {
        let local = try await localOrchestrator.generateStream(request: request, modelId: path)
        return AsyncThrowingStream { continuation in
            let task = Task {
                for await chunk in local { continuation.yield(chunk) }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
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
            throw BasaltError.outOfMemory(
                requestedMB: Int(requiredRAM),
                availableMB: Int(availableRAM / 1_024 / 1_024)
            )
        }
        
        // Unknown model → try cloud
        if cloudClient != nil {
            let cloudModels = mapToCloudModels(modelId)
            return .cloud(provider: .openrouter, models: cloudModels)
        }
        
        throw BasaltError.invalidRequestStructure(
            details: "Model '\(modelId)' not found in local or cloud registry, and cloud fallback is disabled"
        )
    }
    
    public func knowsPrice(of model: String) -> Bool {
        estimateCost(model: model, promptTokens: 1_000_000, completionTokens: 0) > 0
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
        return try await localOrchestrator.generate(request: request, modelId: path)
    }
    
    private func routeToCloud(
        client: OpenRouterClient,
        model: String,
        request: GenerationRequest
    ) async throws -> GenerationResponse {
        let chatRequest = OpenRouterClient.ChatRequest(
            model: model,
            messages: Self.cloudMessages(request),
            temperature: request.temperature,
            maxTokens: request.maxTokens,
            stream: false
        )
        let startTime = ContinuousClock.now
        let chatResponse = try await client.chat(request: chatRequest)
        let duration = startTime.duration(to: ContinuousClock.now).components.seconds
        let durationSeconds = Double(duration) > 0 ? Double(duration) : 0.001
        guard let choice = chatResponse.choices.first else {
            throw BasaltError.modelInferenceError(details: "No response from cloud model")
        }
        let usage = chatResponse.usage ?? .init(promptTokens: 0, completionTokens: 0, totalTokens: 0)
        return GenerationResponse(
            generatedText: choice.message.content,
            promptTokens: usage.promptTokens,
            completionTokens: usage.completionTokens,
            tokensPerSecond: Double(usage.completionTokens) / durationSeconds,
            generationTime: durationSeconds,
            timeToFirstToken: 0.0,
            memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0),
            finishReason: choice.finishReason == "length" ? .length : .stop
        )
    }
    
    private func routeToCloudStream(
        client: OpenRouterClient,
        model: String,
        request: GenerationRequest,
        context: RoutingContext
    ) async throws -> AsyncThrowingStream<StreamChunk, Error> {
        let chatRequest = OpenRouterClient.ChatRequest(
            model: model,
            messages: Self.cloudMessages(request),
            temperature: request.temperature,
            maxTokens: request.maxTokens,
            stream: true
        )
        let startTime = ContinuousClock.now
        let sourceStream = try await client.chatStream(request: chatRequest)
        return Self.relayCloudStream(sourceStream, startTime: startTime) { [weak self] metadata in
            await self?.recordCostLoggingFailure(model: model, response: metadata, context: context)
        }
    }

    static func relayCloudStream(
        _ source: AsyncThrowingStream<OpenRouterClient.ChatStreamResponse, Error>,
        startTime: ContinuousClock.Instant,
        onUsage: @escaping @Sendable (GenerationResponse) async -> Void = { _ in }
    ) -> AsyncThrowingStream<StreamChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await response in source {
                        guard let choice = response.choices.first else { continue }
                        if let text = choice.delta.content {
                            continuation.yield(.text(text))
                        }
                        if let reason = choice.finishReason, reason != "null" {
                            let duration = startTime.duration(to: ContinuousClock.now).components.seconds
                            let durationSeconds = Double(duration) > 0 ? Double(duration) : 0.001
                            let metadata = GenerationResponse(
                                generatedText: "",
                                promptTokens: response.usage?.promptTokens ?? 0,
                                completionTokens: response.usage?.completionTokens ?? 0,
                                tokensPerSecond: Double(response.usage?.completionTokens ?? 0) / durationSeconds,
                                generationTime: durationSeconds,
                                timeToFirstToken: 0.0,
                                memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0),
                                finishReason: reason == "length" ? .length : .stop
                            )
                            await onUsage(metadata)
                            continuation.yield(.metadata(metadata))
                            break
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
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
        return SystemProfiler.availableSystemMemory()
    }
    
    private func log(_ message: String) {
        fputs("[ModelRouter] \(message)\n", stderr)
    }
}

extension ModelRouter: CloudFallback {
    static let assumedCloudTokensPerSecond = 50.0

    static func cloudMessages(_ request: GenerationRequest) -> [OpenRouterClient.ChatRequest.Message] {
        guard let messages = request.messages, !messages.isEmpty else {
            return [.init(role: "user", content: request.prompt)]
        }
        return messages.map { message in
            switch message.role {
            case .tool:
                return .init(role: "user", content: "Result of \(message.toolName ?? "the tool"): \(message.text)")
            case .assistant:
                let calls = (message.toolCalls ?? []).map { "\($0.name)(\($0.arguments))" }
                return .init(role: "assistant", content: ([message.text] + calls).filter { !$0.isEmpty }.joined(separator: "\n"))
            case .system, .user:
                return .init(role: message.role.rawValue, content: message.text)
            }
        }
    }

    public func assess(_ request: GenerationRequest) async -> CloudAssessment? {
        guard cloudClient != nil, let cloudModel = fallbackCloudModel,
              request.tools?.isEmpty ?? true, request.responseFormat == nil
        else { return nil }
        let context = RoutingContext(request)
        if let costTracker, await costTracker.isExhausted(scopes: context.budgetScopes) { return nil }
        let completionTokens = request.maxTokens ?? 1_024
        let gap = stickyTier(request: request, context: context).map { $0.tier == .strong ? $0.strongWinProbability : 0 } ?? 0
        return CloudAssessment(
            capabilityGap: gap,
            estimatedCost: estimateCost(model: cloudModel, promptTokens: request.prompt.utf8.count / 4, completionTokens: completionTokens),
            remainingBudget: await costTracker?.remaining(scopes: context.budgetScopes),
            latencySeconds: Double(completionTokens) / Self.assumedCloudTokensPerSecond
        )
    }

    public func generate(_ request: GenerationRequest) async throws -> GenerationResponse {
        try await route(request: request, preferredModel: fallbackCloudModel, context: RoutingContext(request))
    }

    public func generateStream(_ request: GenerationRequest) async throws -> AsyncThrowingStream<StreamChunk, Error> {
        try await routeStream(request: request, preferredModel: fallbackCloudModel, context: RoutingContext(request))
    }
}
