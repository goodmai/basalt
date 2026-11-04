import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltCloud

@Suite("ModelRouterCoreTests")
struct ModelRouterCoreTests {
    static let gigabyte: Int64 = 1_024 * 1_024 * 1_024

    static func catalog(local id: String = "local-4g") -> ModelCatalog {
        var catalog = ModelCatalog()
        catalog.registerLocal(id: id, estimatedRAM: 4_000)
        return catalog
    }

    static func costRouting(_ scorer: BradleyTerryRouter) -> ModelRouter.CostRouting {
        .init(
            policy: StickyRoutingPolicy(strongModel: "gpt-4", weakModel: "small-local", threshold: 0.5, cacheStickinessPenalty: 1),
            scorer: scorer
        )
    }

    @Test("A strategy pins the route before the registry or RAM is consulted")
    func strategyPinsRoute() throws {
        let catalog = Self.catalog()
        let noRAM: () -> Int64 = { Issue.record("RAM read for a pinned strategy"); return 0 }
        #expect(try catalog.route("gpt-4", strategy: .localOnly, cloudConfigured: true, availableRAM: noRAM) == .local(path: "gpt-4"))
        #expect(try catalog.route("local-4g", strategy: .cloudOnly, cloudConfigured: false, availableRAM: noRAM) == .cloud(models: ["local-4g"]))
        #expect(try catalog.route("gpt-4", strategy: .auto, cloudConfigured: false, availableRAM: noRAM) == .cloud(models: ["openai/gpt-4-turbo", "openai/gpt-4"]))
    }

    @Test("A registered local model runs locally with 1.5× its RAM free, else falls back to cloud under auto, else throws")
    func localModelNeedsHeadroom() throws {
        let catalog = Self.catalog()
        let needed = Int64(4_000) * 1_024 * 1_024 * 3 / 2
        #expect(try catalog.route("local-4g", strategy: .auto, cloudConfigured: true) { needed + 1 } == .local(path: "local-4g"))
        #expect(try catalog.route("local-4g", strategy: .auto, cloudConfigured: true) { needed } == .cloudForLowRAM(models: ["local-4g"]))
        #expect(throws: BasaltError.self) {
            try catalog.route("local-4g", strategy: .hybrid, cloudConfigured: true) { needed }
        }
        #expect(throws: BasaltError.self) {
            try catalog.route("local-4g", strategy: .auto, cloudConfigured: false) { Self.gigabyte }
        }
    }

    @Test("An unknown model goes to the cloud only when a cloud client is configured")
    func unknownModel() throws {
        let catalog = ModelCatalog()
        #expect(try catalog.route("mystery", strategy: .hybrid, cloudConfigured: true) { 0 } == .cloud(models: ["mystery"]))
        #expect(throws: BasaltError.self) {
            try catalog.route("mystery", strategy: .auto, cloudConfigured: false) { 0 }
        }
    }

    @Test("Cost uses the registry's blended price first, then split input/output fallback pricing, else zero")
    func costEstimation() {
        var catalog = ModelCatalog()
        #expect(catalog.estimateCost(model: "gpt-4", promptTokens: 1_000_000, completionTokens: 1_000_000) == 10.0 + 30.0)
        #expect(catalog.estimateCost(model: "unpriced/model", promptTokens: 1_000_000, completionTokens: 0) == 0)
        catalog.merge([
            .init(id: "openai/gpt-4-turbo", name: "GPT-4 Turbo", contextLength: 128_000, pricing: .init(prompt: "0.000002", completion: "0.000004")),
            .init(id: "free/model", name: "Free", contextLength: 8_192, pricing: .init(prompt: "0", completion: "oops"))
        ])
        #expect(abs(catalog.estimateCost(model: "gpt-4", promptTokens: 500_000, completionTokens: 500_000) - 3.0) < 1e-9)
        #expect(catalog.models["free/model"]?.costPer1MTokens == nil)
        #expect(catalog.models["free/model"]?.contextWindow == 8_192)
    }

    @Test("With β = 0 the sticky tier carries exactly the scorer's win probability")
    func zeroBiasIsThePlainScorer() throws {
        let scorer = try BradleyTerryRouterTests.scorer()
        var state = CostRoutingState(routing: Self.costRouting(scorer))
        let request = GenerationRequest(prompt: BradleyTerryRouterTests.hardWord)
        let choice = state.stickyTier(request, sessionId: nil, localModel: "m", at: BradleyTerryRouterTests.start)
        #expect(choice.strongWinProbability == scorer.strongWinProbability(for: request.prompt))
        #expect(choice.tier == .strong)
        #expect(choice.capabilityGap == choice.strongWinProbability)
        #expect(state.trackedSessionCount == 0)
    }

    @Test("Session routes are remembered per session and the table is cleared once it reaches its cap")
    func sessionTableIsBounded() throws {
        var state = CostRoutingState(routing: Self.costRouting(try BradleyTerryRouterTests.scorer()))
        let request = GenerationRequest(prompt: BradleyTerryRouterTests.easyWord)
        for i in 0..<CostRoutingState.maxTrackedSessions {
            _ = state.stickyTier(request, sessionId: "s\(i)", localModel: "m", at: BradleyTerryRouterTests.start)
        }
        #expect(state.trackedSessionCount == CostRoutingState.maxTrackedSessions)
        _ = state.stickyTier(request, sessionId: "s0", localModel: "m", at: BradleyTerryRouterTests.start)
        #expect(state.trackedSessionCount == CostRoutingState.maxTrackedSessions)
        _ = state.stickyTier(request, sessionId: "new", localModel: "m", at: BradleyTerryRouterTests.start)
        #expect(state.trackedSessionCount == 1)
    }

    @Test("Outcomes update β under the reported local model, skip service requests and errors, and say when β changed")
    func outcomesUpdateBias() throws {
        var state = CostRoutingState(routing: Self.costRouting(try BradleyTerryRouterTests.borderlineScorer()))
        let tool = GenerationRequest(prompt: BradleyTerryRouterTests.hardWord, tools: [ToolDefinition(name: "t")])
        let service = GenerationRequest(prompt: "p", isServiceRequest: true, tools: [ToolDefinition(name: "t")])
        let now = BradleyTerryRouterTests.start
        #expect(state.autoModel(for: tool, sessionId: nil, at: now) == "small-local")
        #expect(!state.recordOutcome(service, response: BradleyTerryRouterTests.response(.repetition), localModel: "q", at: now))
        #expect(!state.recordOutcome(tool, response: BradleyTerryRouterTests.response(.error), localModel: "q", at: now))
        #expect(!state.recordOutcome(tool, response: BradleyTerryRouterTests.response(.stop), localModel: "q", at: now))
        #expect(state.lastLocalModel == "q")
        for _ in 0..<3 {
            #expect(state.recordOutcome(tool, response: BradleyTerryRouterTests.response(.repetition), localModel: "q", at: now))
        }
        #expect(state.autoModel(for: tool, sessionId: nil, at: now) == "gpt-4")
        #expect(state.outcomeBias.beta(model: "q", bucket: .toolUse, at: now) > 0)
    }

    @Test("A cloud reply floors sub-second durations at 1 ms and maps only \"length\" to a length stop")
    func cloudReplyMapping() {
        let quick = GenerationResponse.cloudReply(text: "a", promptTokens: 3, completionTokens: 2, finishReason: "length", elapsed: .milliseconds(400))
        #expect(quick.generationTime == 0.001)
        #expect(quick.tokensPerSecond == 2_000)
        #expect(quick.finishReason == .length)
        let slow = GenerationResponse.cloudReply(text: "b", promptTokens: 0, completionTokens: 10, finishReason: nil, elapsed: .seconds(2))
        #expect(slow.generationTime == 2)
        #expect(slow.tokensPerSecond == 5)
        #expect(slow.finishReason == .stop)
    }
}
