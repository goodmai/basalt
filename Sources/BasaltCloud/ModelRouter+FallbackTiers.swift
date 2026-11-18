import Foundation
import BasaltCore

extension ModelRouter {
    static func isFallbackEligible(_ request: GenerationRequest) -> Bool {
        (request.tools?.isEmpty ?? true)
            && request.responseFormat == nil
            && request.messages?.contains(where: { !$0.images.isEmpty }) != true
            && (ChatMessageInput.splittingPrefill(request.messages ?? [])?.prefill.isEmpty ?? true)
    }

    func openRouterSyncClient() -> OpenRouterClient? {
        if let tier = state.fallback.tiers.first(where: { OpenRouterClient.isOpenRouter($0.baseURL) }) {
            return tier.client
        }
        return cloudClient
    }

    private static func isCancellation(_ error: any Error) -> Bool {
        Task.isCancelled || error is CancellationError || (error as? URLError)?.code == .cancelled
    }

    private func tierEstimatedCost(
        _ tier: FallbackTier,
        request: GenerationRequest,
        exhausted: Bool,
        remaining: Double?
    ) -> Double? {
        if tier.isFree { return 0 }
        guard !exhausted else { return nil }
        let text = (request.messages?.isEmpty == false)
            ? ChatMessageInput.plainText(request.messages ?? [])
            : request.prompt
        let promptTokens = text.utf8.count / 4
        let completionTokens = request.maxTokens ?? 1_024
        let cost = estimateCost(model: tier.model, promptTokens: promptTokens, completionTokens: completionTokens)
        if let remaining, cost > remaining { return nil }
        return cost
    }

    func assessFallbackTiers(_ request: GenerationRequest, localModel: String) async -> CloudAssessment? {
        guard Self.isFallbackEligible(request) else { return nil }
        let context = RoutingContext(request)
        let completionTokens = request.maxTokens ?? 1_024
        let exhausted = if let costTracker { await costTracker.isExhausted(scopes: context.budgetScopes) } else { false }
        let remaining = await costTracker?.remaining(scopes: context.budgetScopes)
        for tier in state.fallback.orderedTiers(sessionId: context.sessionId) {
            guard let estimatedCost = tierEstimatedCost(tier, request: request, exhausted: exhausted, remaining: remaining) else {
                continue
            }
            let tierRemaining: Double? = tier.isFree ? nil : remaining
            guard await isTierHealthy(tier) else { continue }
            let choice = state.cost?.stickyTier(request, sessionId: context.sessionId, localModel: localModel, at: Date())
            if let choice { state = .reduce(state, .tierChosen(choice.tier, sessionId: context.sessionId, request: request)) }
            return CloudAssessment(
                capabilityGap: choice?.capabilityGap ?? 0,
                estimatedCost: estimatedCost,
                remainingBudget: tierRemaining,
                latencySeconds: Double(completionTokens) / Self.assumedCloudTokensPerSecond
            )
        }
        return nil
    }

    func generateFromFallbackTiers(_ request: GenerationRequest) async throws -> GenerationResponse {
        let context = RoutingContext(request)
        let exhausted = if let costTracker { await costTracker.isExhausted(scopes: context.budgetScopes) } else { false }
        let remaining = await costTracker?.remaining(scopes: context.budgetScopes)
        var lastError: Error?
        for tier in state.fallback.orderedTiers(sessionId: context.sessionId) {
            guard tierEstimatedCost(tier, request: request, exhausted: exhausted, remaining: remaining) != nil else { continue }
            guard await isTierHealthy(tier) else {
                if Task.isCancelled { throw CancellationError() }
                continue
            }
            do {
                let response = try await routeToCloud(client: tier.client, model: tier.model, request: request)
                state = .reduce(state, .fallback(.sessionBound(sessionId: context.sessionId, tierId: tier.id)))
                if !tier.isFree {
                    await recordCloudCost(model: tier.model, response: response, context: context)
                }
                return response
            } catch {
                if Self.isCancellation(error) { throw error }
                lastError = error
                let instant = state.fallback.probeConfig.now()
                state = .reduce(state, .fallback(.failed(id: tier.id, at: instant)))
            }
        }
        throw lastError ?? BasaltError.modelInferenceError(details: "All fallback tiers unavailable")
    }

    func generateStreamFromFallbackTiers(_ request: GenerationRequest) async throws -> AsyncThrowingStream<StreamChunk, Error> {
        let context = RoutingContext(request)
        let exhausted = if let costTracker { await costTracker.isExhausted(scopes: context.budgetScopes) } else { false }
        let remaining = await costTracker?.remaining(scopes: context.budgetScopes)
        var lastError: Error?
        for tier in state.fallback.orderedTiers(sessionId: context.sessionId) {
            guard tierEstimatedCost(tier, request: request, exhausted: exhausted, remaining: remaining) != nil else { continue }
            guard await isTierHealthy(tier) else {
                if Task.isCancelled { throw CancellationError() }
                continue
            }
            do {
                let startTime = ContinuousClock.now
                let source = try await tier.client.chatStream(request: Self.chatRequest(request, model: tier.model, stream: true))
                state = .reduce(state, .fallback(.sessionBound(sessionId: context.sessionId, tierId: tier.id)))
                let isFree = tier.isFree
                let model = tier.model
                return Self.relayCloudStream(source, startTime: startTime) { [weak self] metadata in
                    guard !isFree else { return }
                    await self?.recordCloudCost(model: model, response: metadata, context: context)
                }
            } catch {
                if Self.isCancellation(error) { throw error }
                lastError = error
                let instant = state.fallback.probeConfig.now()
                state = .reduce(state, .fallback(.failed(id: tier.id, at: instant)))
            }
        }
        throw lastError ?? BasaltError.modelInferenceError(details: "All fallback tiers unavailable")
    }

    func isTierHealthy(_ tier: FallbackTier) async -> Bool {
        let clock = state.fallback.probeConfig.now
        let timeout = state.fallback.probeConfig.timeout
        switch state.fallback.verdict(for: tier.id, at: clock()) {
        case .healthy:
            return true
        case .unhealthy:
            return false
        case .needsProbe:
            state = .reduce(state, .fallback(.probeStarted(tier.id)))
            let generation = state.fallback.probesStarted[tier.id] ?? 1
            let healthy = await Self.probeTierModels(tier, timeout: timeout)
            if Task.isCancelled { return false }
            let instant = clock()
            state = .reduce(state, .fallback(.probed(id: tier.id, generation: generation, healthy: healthy, at: instant)))
            return state.fallback.verdict(for: tier.id, at: instant) == .healthy
        }
    }

    private static func probeTierModels(_ tier: FallbackTier, timeout: Duration) async -> Bool {
        guard let models = try? await tier.client.probeModelIDs(timeout: timeout) else { return false }
        return models.contains(tier.model)
    }
}
