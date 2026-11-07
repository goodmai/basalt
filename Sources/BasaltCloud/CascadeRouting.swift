import Foundation
import BasaltCore

public struct CascadeExecutionMode: Sendable, Equatable {
    public let segmentTokens: Int
    public let alpha: Double
    public let referenceBudget: Double
    public let cloudModel: String

    public init(
        segmentTokens: Int = 32,
        alpha: Double = 0.50,
        referenceBudget: Double = 10.0,
        cloudModel: String
    ) {
        self.segmentTokens = max(1, segmentTokens)
        self.alpha = min(max(alpha, 0.0), 1.0)
        self.referenceBudget = max(Double.leastNonzeroMagnitude, referenceBudget)
        self.cloudModel = cloudModel
    }

    public func effectiveAlpha(remainingBudget: Double?) -> Double {
        guard let remainingBudget else { return alpha }
        guard remainingBudget > 0 else { return 1.0 }
        let ratio = min(max(remainingBudget / referenceBudget, 0.0), 1.0)
        return 1.0 - (1.0 - alpha) * ratio
    }

    public func threshold(remainingBudget: Double?, switchPenalty: Double = 0.0) -> Double {
        let base = 1.0 - effectiveAlpha(remainingBudget: remainingBudget)
        return max(0.0, base - max(0.0, switchPenalty))
    }

    public func shouldDefer(
        confidence: ResponseConfidence?,
        remainingBudget: Double?,
        switchPenalty: Double = 0.0
    ) -> Bool {
        guard let confidence else { return false }
        if let remainingBudget, remainingBudget <= 0 { return false }
        return confidence.min < threshold(remainingBudget: remainingBudget, switchPenalty: switchPenalty)
    }

    public static func isEligible(_ request: GenerationRequest) -> Bool {
        (request.tools?.isEmpty ?? true)
            && request.responseFormat == nil
            && request.isServiceRequest != true
            && request.messages?.contains(where: { !$0.images.isEmpty }) != true
    }

    public static func appendingAssistantPrefill(_ prefix: String, to request: GenerationRequest) -> GenerationRequest {
        guard !prefix.isEmpty else { return request }
        let baseMessages: [ChatMessageInput]
        if let existing = request.messages, !existing.isEmpty {
            if let split = ChatMessageInput.splittingPrefill(existing) {
                baseMessages = split.turns + [ChatMessageInput(role: .assistant, text: split.prefill + prefix)]
            } else {
                baseMessages = existing + [ChatMessageInput(role: .assistant, text: prefix)]
            }
        } else {
            baseMessages = [
                ChatMessageInput(role: .user, text: request.prompt),
                ChatMessageInput(role: .assistant, text: prefix)
            ]
        }
        return request.withMessages(baseMessages)
    }
}

struct CascadeAccumulator {
    var acceptedText = ""
    var acceptedTokens = 0
    var promptTokens = 0
    var cachedPromptTokens = 0
    var totalTime = 0.0
    var ttft: Double?
    var lastMemory = GenerationResponse.MemoryUsage(peakBytes: 0, activeBytes: 0, cacheBytes: 0)
    var confidence: ResponseConfidence?
    var finishReason: GenerationResponse.FinishReason = .stop

    mutating func recordMetrics(_ segment: GenerationResponse) {
        if promptTokens == 0 {
            promptTokens = segment.promptTokens
            cachedPromptTokens = segment.cachedPromptTokens
        }
        if ttft == nil { ttft = segment.timeToFirstToken }
        totalTime += segment.generationTime
        lastMemory = segment.memory
    }

    mutating func accept(_ segment: GenerationResponse) -> Bool {
        let segmentTokens = max(1, segment.completionTokens)
        if let segConf = segment.confidence {
            confidence = confidence?.merged(
                with: segConf, weightSelf: max(1, acceptedTokens), weightOther: segmentTokens
            ) ?? segConf
        }
        if let reasoning = segment.reasoningText {
            acceptedText += "<think>\(reasoning)</think>"
        }
        acceptedText += segment.generatedText
        acceptedTokens += segmentTokens
        finishReason = segment.finishReason
        let empty = segment.generatedText.isEmpty && (segment.reasoningText?.isEmpty ?? true)
        return segment.finishReason == .length && !empty
    }

    func mergedResponse(cloud: GenerationResponse? = nil) -> GenerationResponse {
        let totalCompletion = acceptedTokens + (cloud?.completionTokens ?? 0)
        let duration = max(0.001, totalTime + (cloud?.generationTime ?? 0.0))
        return GenerationResponse(
            generatedText: acceptedText + (cloud?.generatedText ?? ""),
            promptTokens: promptTokens,
            cachedPromptTokens: cachedPromptTokens,
            completionTokens: totalCompletion,
            tokensPerSecond: Double(totalCompletion) / duration,
            generationTime: duration,
            timeToFirstToken: ttft ?? cloud?.timeToFirstToken ?? 0.0,
            memory: lastMemory,
            finishReason: cloud?.finishReason ?? finishReason
        ).withConfidence(confidence)
    }
}

extension ModelRouter {
    public var cascadeDeferrals: Int {
        state.cascadeDeferrals
    }

    func cascadeSwitchPenalty(for context: RoutingContext, acceptedTokens: Int) -> Double {
        guard let cost = state.cost, cost.sessionTier(sessionId: context.sessionId) == .weak else { return 0.0 }
        let policy = cost.routing.policy
        let warmTokens = cost.warmPrefixTokens(sessionId: context.sessionId) + max(0, acceptedTokens)
        return policy.cacheStickinessPenalty * policy.switchCost(warmPrefixTokens: warmTokens)
    }

    func shouldDeferSegment(
        _ cascade: CascadeExecutionMode,
        confidence: ResponseConfidence?,
        acceptedTokens: Int,
        context: RoutingContext
    ) async -> Bool {
        guard cloudClient != nil else { return false }
        if let costTracker, await costTracker.isExhausted(scopes: context.budgetScopes) { return false }
        let remaining = await costTracker?.remaining(scopes: context.budgetScopes)
        let penalty = cascadeSwitchPenalty(for: context, acceptedTokens: acceptedTokens)
        return cascade.shouldDefer(confidence: confidence, remainingBudget: remaining, switchPenalty: penalty)
    }

    func executeCascade(
        request: GenerationRequest,
        localPath: String,
        cascade: CascadeExecutionMode,
        context: RoutingContext
    ) async throws -> GenerationResponse {
        let totalBudget = max(1, request.maxTokens ?? 1_024)
        var acc = CascadeAccumulator()

        while acc.acceptedTokens < totalBudget {
            let remaining = totalBudget - acc.acceptedTokens
            let stepBudget = min(cascade.segmentTokens, remaining)
            let segmentRequest = CascadeExecutionMode.appendingAssistantPrefill(acc.acceptedText, to: request)
                .withMaxTokens(stepBudget)
                .withConfidenceRecording(true)
            let segment = try await localOrchestrator.generate(request: segmentRequest, modelId: localPath)
            try Task.checkCancellation()
            acc.recordMetrics(segment)

            if await shouldDeferSegment(cascade, confidence: segment.confidence, acceptedTokens: acc.acceptedTokens, context: context) {
                state = .reduce(state, .cascadeDeferred(sessionId: context.sessionId, request: request))
                let cloudRequest = CascadeExecutionMode.appendingAssistantPrefill(acc.acceptedText, to: request)
                    .withMaxTokens(remaining)
                let cloudResponse = try await routeCloudChain(
                    modelId: cascade.cloudModel, request: cloudRequest, context: context, localFallback: localPath
                )
                return acc.mergedResponse(cloud: cloudResponse)
            }
            if !acc.accept(segment) { break }
        }
        return acc.mergedResponse()
    }

    func executeCascadeStream(
        request: GenerationRequest,
        localPath: String,
        cascade: CascadeExecutionMode,
        context: RoutingContext
    ) -> AsyncThrowingStream<StreamChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.runCascadeStream(
                        request: request, localPath: localPath, cascade: cascade,
                        context: context, into: continuation
                    )
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func runCascadeStream(
        request: GenerationRequest,
        localPath: String,
        cascade: CascadeExecutionMode,
        context: RoutingContext,
        into continuation: AsyncThrowingStream<StreamChunk, Error>.Continuation
    ) async throws {
        let totalBudget = max(1, request.maxTokens ?? 1_024)
        var acc = CascadeAccumulator()

        while acc.acceptedTokens < totalBudget {
            try Task.checkCancellation()
            let remaining = totalBudget - acc.acceptedTokens
            let stepBudget = min(cascade.segmentTokens, remaining)
            let segmentRequest = CascadeExecutionMode.appendingAssistantPrefill(acc.acceptedText, to: request)
                .withMaxTokens(stepBudget)
                .withConfidenceRecording(true)
            let segment = try await localOrchestrator.generate(request: segmentRequest, modelId: localPath)
            try Task.checkCancellation()
            acc.recordMetrics(segment)

            if await shouldDeferSegment(cascade, confidence: segment.confidence, acceptedTokens: acc.acceptedTokens, context: context) {
                state = .reduce(state, .cascadeDeferred(sessionId: context.sessionId, request: request))
                let cloudRequest = CascadeExecutionMode.appendingAssistantPrefill(acc.acceptedText, to: request)
                    .withMaxTokens(remaining)
                let cloudStream = try await routeCloudChainStream(
                    modelId: cascade.cloudModel, request: cloudRequest, context: context, localFallback: localPath
                )
                for try await chunk in cloudStream {
                    continuation.yield(chunk)
                }
                continuation.finish()
                return
            }

            if let reasoning = segment.reasoningText, !reasoning.isEmpty {
                continuation.yield(.reasoning(reasoning))
            }
            if !segment.generatedText.isEmpty {
                continuation.yield(.text(segment.generatedText))
            }
            if !acc.accept(segment) { break }
        }

        continuation.yield(.metadata(acc.mergedResponse()))
        continuation.finish()
    }
}
