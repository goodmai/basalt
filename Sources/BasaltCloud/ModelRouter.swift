import Foundation
import BasaltCore

public actor ModelRouter: Sendable {
    private let localOrchestrator: ModelOrchestratorActor
    private let cloudClient: OpenRouterClient?
    private let strategy: RoutingStrategy
    private let mockAvailableRAM: Int64?
    private let costTracker: CostTracker?
    private let localFallbackModel: String?
    private let fallbackCloudModel: String?
    private var catalog = ModelCatalog()
    private var costState: CostRoutingState?

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
        self.costTracker = costTracker
        self.localFallbackModel = localFallbackModel
        self.fallbackCloudModel = fallbackCloudModel
        self.costState = costRouting.map { routing in
            CostRoutingState(
                routing: routing,
                outcomeBias: routing.outcomeBiasURL.map(RoutingOutcomeBias.load(from:)) ?? RoutingOutcomeBias()
            )
        }
    }

    public func syncCloudModels() async throws {
        guard let client = cloudClient else { return }
        let fetchedModels = try await client.getModels()
        catalog.merge(fetchedModels)
        log("Synced \(fetchedModels.count) models from OpenRouter")
    }

    public func route(
        request: GenerationRequest, preferredModel: String?, context: RoutingContext = .init()
    ) async throws -> GenerationResponse {
        try await dispatch(request, preferredModel: preferredModel, context: context) { path in
            try await self.localOrchestrator.generate(request: request, modelId: path)
        } cloud: { client, model in
            let response = try await self.routeToCloud(client: client, model: model, request: request)
            await self.recordCloudCost(model: model, response: response, context: context)
            return response
        }
    }

    public func routeStream(
        request: GenerationRequest, preferredModel: String?, context: RoutingContext = .init()
    ) async throws -> AsyncThrowingStream<StreamChunk, Error> {
        try await dispatch(request, preferredModel: preferredModel, context: context) { path in
            try await self.routeToLocalStream(path: path, request: request)
        } cloud: { client, model in
            try await self.routeToCloudStream(client: client, model: model, request: request, context: context)
        }
    }

    func resolveModel(_ preferredModel: String?, request: GenerationRequest, context: RoutingContext) -> String {
        let modelId = preferredModel ?? "auto"
        guard modelId == "auto",
              let model = costState?.autoModel(for: request, sessionId: context.sessionId, at: Date())
        else { return modelId }
        return model
    }

    public func recordLocalOutcome(_ request: GenerationRequest, response: GenerationResponse, localModel: String) {
        guard costState?.recordOutcome(request, response: response, localModel: localModel, at: Date()) == true,
              let state = costState, let url = state.routing.outcomeBiasURL
        else { return }
        do {
            try state.outcomeBias.save(to: url)
        } catch {
            log("Saving the routing outcome bias to \(url.path) failed: \(error)")
        }
    }

    public var routingBias: [String: [String: Double]]? {
        costState?.outcomeBias.snapshot(at: Date())
    }

    public func registerLocalModel(id: String, estimatedRAM: Int64) {
        catalog.registerLocal(id: id, estimatedRAM: estimatedRAM)
    }

    public func makeRoutingDecision(modelId: String) async throws -> ModelLocation {
        let route = try catalog.route(modelId, strategy: strategy, cloudConfigured: cloudClient != nil) {
            mockAvailableRAM ?? SystemProfiler.availableSystemMemory()
        }
        switch route {
        case .local(let path):
            return .local(path: path)
        case .cloud(let models):
            return .cloud(provider: .openrouter, models: models)
        case .cloudForLowRAM(let models):
            log("⚠️ Insufficient RAM for local model, falling back to cloud")
            return .cloud(provider: .openrouter, models: models)
        }
    }

    public func knowsPrice(of model: String) -> Bool {
        catalog.estimateCost(model: model, promptTokens: 1_000_000, completionTokens: 0) > 0
    }

    public func estimateCost(model: String, promptTokens: Int, completionTokens: Int) -> Double {
        catalog.estimateCost(model: model, promptTokens: promptTokens, completionTokens: completionTokens)
    }

    private func dispatch<Output>(
        _ request: GenerationRequest, preferredModel: String?, context: RoutingContext,
        local: (String) async throws -> Output,
        cloud: (OpenRouterClient, String) async throws -> Output
    ) async throws -> Output {
        let modelId = resolveModel(preferredModel, request: request, context: context)
        switch try await makeRoutingDecision(modelId: modelId) {
        case .local(let path):
            return try await local(path)
        case .cloud(_, let cloudModels):
            let client = try requireCloudClient()
            if let fallback = try await localFallbackIfBudgetExhausted(context) {
                return try await local(fallback)
            }
            var lastError: Error?
            for model in cloudModels {
                do {
                    return try await cloud(client, model)
                } catch {
                    lastError = error
                    log("Cloud model \(model) failed, trying the next in the chain: \(error)")
                }
            }
            if let fallback = localFallbackModel {
                return try await local(fallback)
            }
            throw lastError ?? BasaltError.modelInferenceError(details: "Empty cloud fallback chain")
        }
    }

    private func log(_ message: String) {
        fputs("[ModelRouter] \(message)\n", stderr)
    }
}

extension ModelRouter {
    private func requireCloudClient() throws(BasaltError) -> OpenRouterClient {
        guard let client = cloudClient else {
            throw .invalidRequestStructure(details: "Cloud models not configured. Set OPENROUTER_API_KEY")
        }
        return client
    }

    private func localFallbackIfBudgetExhausted(_ context: RoutingContext) async throws(BasaltError) -> String? {
        guard let costTracker, await costTracker.isExhausted(scopes: context.budgetScopes) else { return nil }
        guard let localFallbackModel else {
            throw .invalidRequestStructure(details: "Cloud budget exhausted and no local fallback model is configured")
        }
        log("Cloud budget exhausted, serving locally with \(localFallbackModel)")
        return localFallbackModel
    }

    private func recordCloudCost(model: String, response: GenerationResponse, context: RoutingContext) async {
        guard let costTracker else { return }
        let cost = catalog.estimateCost(model: model, promptTokens: response.promptTokens, completionTokens: response.completionTokens)
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

    private func routeToCloud(
        client: OpenRouterClient, model: String, request: GenerationRequest
    ) async throws -> GenerationResponse {
        let startTime = ContinuousClock.now
        let chatResponse = try await client.chat(request: Self.chatRequest(request, model: model, stream: false))
        let elapsed = startTime.duration(to: ContinuousClock.now)
        guard let choice = chatResponse.choices.first else {
            throw BasaltError.modelInferenceError(details: "No response from cloud model")
        }
        return .cloudReply(
            text: choice.message.content,
            promptTokens: chatResponse.usage?.promptTokens ?? 0,
            completionTokens: chatResponse.usage?.completionTokens ?? 0,
            finishReason: choice.finishReason,
            elapsed: elapsed
        )
    }

    private func routeToCloudStream(
        client: OpenRouterClient, model: String, request: GenerationRequest, context: RoutingContext
    ) async throws -> AsyncThrowingStream<StreamChunk, Error> {
        let startTime = ContinuousClock.now
        let sourceStream = try await client.chatStream(request: Self.chatRequest(request, model: model, stream: true))
        return Self.relayCloudStream(sourceStream, startTime: startTime) { [weak self] metadata in
            await self?.recordCloudCost(model: model, response: metadata, context: context)
        }
    }

    private static func chatRequest(_ request: GenerationRequest, model: String, stream: Bool) -> OpenRouterClient.ChatRequest {
        OpenRouterClient.ChatRequest(
            model: model,
            messages: cloudMessages(request),
            temperature: request.temperature,
            maxTokens: request.maxTokens,
            stream: stream
        )
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
                            let metadata = GenerationResponse.cloudReply(
                                text: "",
                                promptTokens: response.usage?.promptTokens ?? 0,
                                completionTokens: response.usage?.completionTokens ?? 0,
                                finishReason: reason,
                                elapsed: startTime.duration(to: ContinuousClock.now)
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
}

extension ModelRouter: CloudFallback {
    static let assumedCloudTokensPerSecond = 50.0

    static func cloudMessages(_ request: GenerationRequest) -> [OpenRouterClient.ChatRequest.Message] {
        guard let messages = request.messages, !messages.isEmpty else {
            return [.init(role: "user", content: request.prompt)]
        }
        return (ChatMessageInput.splittingPrefill(messages)?.turns ?? messages).map { message in
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

    public func assess(_ request: GenerationRequest, localModel: String) async -> CloudAssessment? {
        costState?.lastLocalModel = localModel
        guard cloudClient != nil, let cloudModel = fallbackCloudModel,
              request.tools?.isEmpty ?? true, request.responseFormat == nil,
              ChatMessageInput.splittingPrefill(request.messages ?? [])?.prefill.isEmpty ?? true
        else { return nil }
        let context = RoutingContext(request)
        if let costTracker, await costTracker.isExhausted(scopes: context.budgetScopes) { return nil }
        let completionTokens = request.maxTokens ?? 1_024
        let gap = costState?.stickyTier(request, sessionId: context.sessionId, localModel: localModel, at: Date()).capabilityGap ?? 0
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
