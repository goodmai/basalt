import Foundation
import BasaltCore

public struct FallbackTiers: CloudFallback {
    public let tiers: [ModelRouter]

    public init(_ tiers: [ModelRouter]) {
        self.tiers = tiers
    }

    public func assess(_ request: GenerationRequest, localModel: String) async -> CloudAssessment? {
        for tier in tiers {
            if let assessment = await tier.assess(request, localModel: localModel) { return assessment }
        }
        return nil
    }

    public func recordLocalOutcome(_ request: GenerationRequest, response: GenerationResponse, localModel: String) async {
        await tiers.first?.recordLocalOutcome(request, response: response, localModel: localModel)
    }

    public var routingBias: [String: [String: Double]]? {
        get async { await tiers.first?.routingBias }
    }

    public func generate(_ request: GenerationRequest) async throws -> GenerationResponse {
        try await firstAnswering { try await $0.generate(request) }
    }

    public func generateStream(_ request: GenerationRequest) async throws -> AsyncThrowingStream<StreamChunk, Error> {
        try await firstAnswering { try await $0.generateStream(request) }
    }

    static func failsOver(_ error: any Error) -> Bool {
        if error is URLError { return true }
        if case BasaltError.modelInferenceError = error { return true }
        return false
    }

    private func firstAnswering<Output>(_ call: (ModelRouter) async throws -> Output) async throws -> Output {
        var lastError: any Error = BasaltError.modelInferenceError(details: "No fallback tier is configured")
        for tier in tiers {
            do {
                return try await call(tier)
            } catch let error where Self.failsOver(error) {
                lastError = error
            }
        }
        throw lastError
    }
}
