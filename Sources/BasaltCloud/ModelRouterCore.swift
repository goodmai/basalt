import Foundation
import BasaltCore

extension ModelRouter {
    public enum RoutingStrategy: String, Sendable, Codable {
        case auto
        case localOnly
        case cloudOnly
        case hybrid
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
        public let estimatedRAM: Int64?
        public let costPer1MTokens: Double?

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
        public let outcomeBiasURL: URL?

        public init(policy: StickyRoutingPolicy, scorer: BradleyTerryRouter, outcomeBiasURL: URL? = nil) {
            self.policy = policy
            self.scorer = scorer
            self.outcomeBiasURL = outcomeBiasURL
        }
    }
}

struct ModelCatalog: Sendable {
    enum Route: Sendable, Equatable {
        case local(path: String)
        case cloud(models: [String])
        case cloudForLowRAM(models: [String])
    }

    static let cloudPrefixes = ["gpt-", "claude-", "gemini-", "o1-", "o3-"]

    static let cloudAliases: [String: [String]] = [
        "gpt-4": ["openai/gpt-4-turbo", "openai/gpt-4"],
        "gpt-4o": ["openai/gpt-4o", "openai/gpt-4o-2024-08-06"],
        "claude": ["anthropic/claude-3.5-sonnet", "anthropic/claude-3-opus"],
        "claude-3.5": ["anthropic/claude-3.5-sonnet", "anthropic/claude-3-opus"],
        "gemini": ["google/gemini-pro-1.5"],
        "o1": ["openai/o1-preview"],
        "o3": ["openai/o3-mini"]
    ]

    static let fallbackPricing: [String: (input: Double, output: Double)] = [
        "openai/gpt-4-turbo": (10.0, 30.0),
        "anthropic/claude-3.5-sonnet": (3.0, 15.0),
        "google/gemini-pro-1.5": (1.25, 5.0)
    ]

    static let defaultModels: [String: ModelRouter.ModelInfo] = [
        "gpt-4": ModelRouter.ModelInfo(
            id: "gpt-4", displayName: "GPT-4 Turbo",
            location: .cloud(provider: .openrouter, models: ["openai/gpt-4-turbo", "openai/gpt-4"]),
            contextWindow: 128_000, estimatedRAM: nil, costPer1MTokens: 10.0
        ),
        "claude-3.5": ModelRouter.ModelInfo(
            id: "claude-3.5", displayName: "Claude 3.5 Sonnet",
            location: .cloud(provider: .openrouter, models: ["anthropic/claude-3.5-sonnet"]),
            contextWindow: 200_000, estimatedRAM: nil, costPer1MTokens: 3.0
        )
    ]

    private(set) var models = Self.defaultModels

    static func cloudModels(for modelId: String) -> [String] {
        cloudAliases[modelId] ?? [modelId]
    }

    static func isCloudOnly(_ modelId: String) -> Bool {
        cloudPrefixes.contains { modelId.hasPrefix($0) }
    }

    mutating func registerLocal(id: String, estimatedRAM: Int64) {
        models[id] = ModelRouter.ModelInfo(
            id: id, displayName: id, location: .local(path: id),
            contextWindow: 8192, estimatedRAM: estimatedRAM, costPer1MTokens: nil
        )
    }

    mutating func merge(_ fetched: [OpenRouterClient.OpenRouterModel]) {
        for model in fetched {
            let promptPrice = Double(model.pricing.prompt) ?? 0.0
            let completionPrice = Double(model.pricing.completion) ?? 0.0
            let averagePer1M = (promptPrice + completionPrice) / 2.0 * 1_000_000.0
            models[model.id] = ModelRouter.ModelInfo(
                id: model.id, displayName: model.name,
                location: .cloud(provider: .openrouter, models: [model.id]),
                contextWindow: model.contextLength, estimatedRAM: nil,
                costPer1MTokens: averagePer1M > 0 ? averagePer1M : nil
            )
        }
    }

    func estimateCost(model: String, promptTokens: Int, completionTokens: Int) -> Double {
        let mappedId = Self.cloudModels(for: model).first ?? model
        if let cost = models[mappedId]?.costPer1MTokens {
            return (Double(promptTokens + completionTokens) / 1_000_000.0) * cost
        }
        guard let price = Self.fallbackPricing[mappedId] else { return 0.0 }
        let inputCost = Double(promptTokens) / 1_000_000.0 * price.input
        let outputCost = Double(completionTokens) / 1_000_000.0 * price.output
        return inputCost + outputCost
    }

    func route(
        _ modelId: String, strategy: ModelRouter.RoutingStrategy, cloudConfigured: Bool, availableRAM: () -> Int64
    ) throws(BasaltError) -> Route {
        switch strategy {
        case .localOnly: return .local(path: modelId)
        case .cloudOnly: return .cloud(models: Self.cloudModels(for: modelId))
        case .auto, .hybrid: break
        }
        if Self.isCloudOnly(modelId) {
            return .cloud(models: Self.cloudModels(for: modelId))
        }
        if let info = models[modelId], case .local = info.location {
            let canFallBack = strategy == .auto && cloudConfigured
            return try Self.localRoute(modelId, requiredMB: info.estimatedRAM ?? 0, availableBytes: availableRAM(), canFallBack: canFallBack)
        }
        guard cloudConfigured else {
            throw .invalidRequestStructure(
                details: "Model '\(modelId)' not found in local or cloud registry, and cloud fallback is disabled"
            )
        }
        return .cloud(models: Self.cloudModels(for: modelId))
    }

    static func localRoute(_ modelId: String, requiredMB: Int64, availableBytes: Int64, canFallBack: Bool) throws(BasaltError) -> Route {
        if availableBytes > requiredMB * 1_024 * 1_024 * 3 / 2 {
            return .local(path: modelId)
        }
        guard canFallBack else {
            throw .outOfMemory(requestedMB: Int(requiredMB), availableMB: Int(availableBytes / 1_024 / 1_024))
        }
        return .cloudForLowRAM(models: cloudModels(for: modelId))
    }
}

struct CostRoutingState: Sendable {
    struct TierChoice: Sendable, Equatable {
        let tier: StickyRoutingPolicy.Tier
        let strongWinProbability: Double

        var capabilityGap: Double { tier == .strong ? strongWinProbability : 0 }
    }

    private struct SessionRoute: Sendable {
        let tier: StickyRoutingPolicy.Tier
        let promptTokens: Int
    }

    static let maxTrackedSessions = 4_096

    let routing: ModelRouter.CostRouting
    private(set) var outcomeBias: RoutingOutcomeBias
    var lastLocalModel: String?
    private var sessions: [String: SessionRoute] = [:]

    init(routing: ModelRouter.CostRouting, outcomeBias: RoutingOutcomeBias = RoutingOutcomeBias()) {
        self.routing = routing
        self.outcomeBias = outcomeBias
    }

    var trackedSessionCount: Int { sessions.count }

    mutating func autoModel(for request: GenerationRequest, sessionId: String?, at now: Date) -> String {
        let localModel = lastLocalModel ?? routing.policy.weakModel
        return routing.policy.model(for: stickyTier(request, sessionId: sessionId, localModel: localModel, at: now).tier)
    }

    mutating func stickyTier(_ request: GenerationRequest, sessionId: String?, localModel: String, at now: Date) -> TierChoice {
        let previous = sessionId.flatMap { sessions[$0] }
        let bias = outcomeBias.beta(model: localModel, bucket: .init(request), at: now)
        let probability = routing.scorer.strongWinProbability(for: request.prompt, escalationBias: bias)
        let tier = routing.policy.tier(
            strongWinProbability: probability,
            previous: previous?.tier,
            warmPrefixTokens: previous?.promptTokens ?? 0
        )
        if let sessionId {
            if sessions[sessionId] == nil, sessions.count >= Self.maxTrackedSessions { sessions.removeAll() }
            sessions[sessionId] = SessionRoute(tier: tier, promptTokens: request.prompt.utf8.count / 4)
        }
        return TierChoice(tier: tier, strongWinProbability: probability)
    }

    mutating func recordOutcome(_ request: GenerationRequest, response: GenerationResponse, localModel: String, at now: Date) -> Bool {
        lastLocalModel = localModel
        guard request.isServiceRequest != true, let failed = RoutingOutcomeBias.failed(request, response) else { return false }
        let before = outcomeBias
        outcomeBias.record(failed: failed, model: localModel, bucket: .init(request), at: now)
        return outcomeBias != before
    }
}

extension GenerationResponse {
    static func cloudReply(
        text: String, promptTokens: Int, completionTokens: Int, finishReason: String?, elapsed: Duration
    ) -> GenerationResponse {
        let seconds = Double(elapsed.components.seconds)
        let durationSeconds = seconds > 0 ? seconds : 0.001
        return GenerationResponse(
            generatedText: text,
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            tokensPerSecond: Double(completionTokens) / durationSeconds,
            generationTime: durationSeconds,
            timeToFirstToken: 0.0,
            memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0),
            finishReason: finishReason == "length" ? .length : .stop
        )
    }
}
