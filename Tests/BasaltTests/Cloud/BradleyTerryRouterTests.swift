import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltCloud

@Suite("BradleyTerryRouterTests")
struct BradleyTerryRouterTests {

    static func wordInSlot(_ slot: Int) -> String {
        (0..<200).map { "w\($0)" }.first { BradleyTerryRouter.hashedFeatures($0, count: 2)[slot] > 0 }!
    }

    static let hardWord = wordInSlot(0)
    static let easyWord = wordInSlot(1)

    static func scorer() throws -> BradleyTerryRouter {
        try BradleyTerryRouter(strongEmbedding: [2], weakEmbedding: [0], weights: [1], projection: [[1, -1]])
    }

    static func borderlineScorer() throws -> BradleyTerryRouter {
        try BradleyTerryRouter(strongEmbedding: [log(0.45 / 0.55)], weakEmbedding: [0], weights: [1], projection: [[1, -1]])
    }

    static func response(_ reason: GenerationResponse.FinishReason, repairs: Int = 0, rejections: Int = 0) -> GenerationResponse {
        GenerationResponse(
            generatedText: "", promptTokens: 1, completionTokens: 1, tokensPerSecond: 1, generationTime: 1, timeToFirstToken: 0,
            memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0), finishReason: reason,
            toolCallRepairs: repairs, toolCallRejections: rejections)
    }

    static let start = Date(timeIntervalSinceReferenceDate: 0)

    @Test("Win probability is σ(wᵀ(e_strong − e_weak) ⊙ W_q·v(q))")
    func winProbabilityFollowsTheFormula() throws {
        let scorer = try Self.scorer()
        #expect(abs(scorer.strongWinProbability(for: Self.hardWord) - 1 / (1 + exp(-2.0))) < 1e-12)
        #expect(abs(scorer.strongWinProbability(for: Self.easyWord) - 1 / (1 + exp(2.0))) < 1e-12)
    }

    @Test("Identical strong and weak embeddings give a coin flip for any prompt")
    func identicalEmbeddingsGiveHalf() throws {
        let scorer = try BradleyTerryRouter(strongEmbedding: [1, 3], weakEmbedding: [1, 3], weights: [1, 1], projection: [[1, 0, 2], [0, 1, 1]])
        #expect(scorer.strongWinProbability(for: "prove the Riemann hypothesis") == 0.5)
    }

    @Test("Parameters with mismatched shapes are rejected, also when decoded from JSON")
    func shapeMismatchIsRejected() {
        #expect(throws: BradleyTerryError.shapeMismatch) {
            _ = try BradleyTerryRouter(strongEmbedding: [1, 2], weakEmbedding: [0], weights: [1], projection: [[1]])
        }
        let json = #"{"strongEmbedding":[1],"weakEmbedding":[0],"weights":[1],"projection":[[1],[2]]}"#
        #expect(throws: BradleyTerryError.shapeMismatch) {
            _ = try JSONDecoder().decode(BradleyTerryRouter.self, from: Data(json.utf8))
        }
    }

    @Test("Hashed features are deterministic and L2-normalised")
    func hashedFeaturesAreStable() {
        let a = BradleyTerryRouter.hashedFeatures("Sort this list, then sort it again", count: 16)
        #expect(a == BradleyTerryRouter.hashedFeatures("sort THIS list then sort it again", count: 16))
        #expect(abs(a.reduce(0) { $0 + $1 * $1 } - 1) < 1e-12)
        #expect(BradleyTerryRouter.hashedFeatures("", count: 4) == [0, 0, 0, 0])
    }

    @Test("A calibrated threshold sends the requested share of calibration prompts to the strong model")
    func calibrationHitsTheStrongFraction() {
        let scores = (0..<100).map { Double($0) / 100 }
        for fraction in [0.0, 0.1, 0.3, 0.5, 1.0] {
            let threshold = BradleyTerryRouter.calibrateThreshold(scores: scores, strongFraction: fraction)
            #expect(scores.filter { $0 > threshold }.count == Int(fraction * 100))
        }
        let tiny = [1e-20, 2e-20]
        #expect(tiny.allSatisfy { $0 > BradleyTerryRouter.calibrateThreshold(scores: tiny, strongFraction: 1) })
        #expect(BradleyTerryRouter.calibrateThreshold(scores: tiny, strongFraction: 1) > 0)
    }

    @Test("A cold session picks the tier by the threshold alone")
    func coldSessionUsesThreshold() {
        let policy = StickyRoutingPolicy(strongModel: "s", weakModel: "w", threshold: 0.5, cacheStickinessPenalty: 1)
        #expect(policy.tier(strongWinProbability: 0.51, previous: nil, warmPrefixTokens: 0) == .strong)
        #expect(policy.tier(strongWinProbability: 0.49, previous: nil, warmPrefixTokens: 0) == .weak)
    }

    @Test("A warm session keeps its model until the margin beats the cache-switch penalty")
    func warmSessionIsSticky() {
        let policy = StickyRoutingPolicy(strongModel: "s", weakModel: "w", threshold: 0.5, cacheStickinessPenalty: 1, switchCostTokenScale: 4_096)
        #expect(policy.switchCost(warmPrefixTokens: 4_096) == 0.5)
        #expect(policy.tier(strongWinProbability: 0.9, previous: .weak, warmPrefixTokens: 4_096) == .weak)
        #expect(policy.tier(strongWinProbability: 0.1, previous: .strong, warmPrefixTokens: 4_096) == .strong)
        #expect(policy.tier(strongWinProbability: 0.9, previous: .weak, warmPrefixTokens: 1_000) == .strong)
        #expect(policy.tier(strongWinProbability: 0.9, previous: .weak, warmPrefixTokens: 0) == .strong)
    }

    @Test("Three failing local tool-use turns move a prompt at θ − 0.05 to the cloud, leave general prompts local, and wear off after the decay window")
    func failingToolTurnsEscalate() throws {
        let policy = StickyRoutingPolicy(strongModel: "s", weakModel: "w", threshold: 0.5)
        let scorer = try Self.borderlineScorer()
        #expect(abs(scorer.strongWinProbability(for: Self.hardWord) - (policy.threshold - 0.05)) < 1e-12)
        var bias = RoutingOutcomeBias()
        func tier(_ bucket: RoutingOutcomeBias.Bucket, at date: Date) -> StickyRoutingPolicy.Tier {
            let beta = bias.beta(model: "qwen@abc", bucket: bucket, at: date)
            return policy.tier(strongWinProbability: scorer.strongWinProbability(for: Self.hardWord, escalationBias: beta), previous: nil, warmPrefixTokens: 0)
        }
        #expect(tier(.toolUse, at: Self.start) == .weak)
        for turn in 0..<3 { bias.record(failed: true, model: "qwen@abc", bucket: .toolUse, at: Self.start + Double(turn)) }
        let now = Self.start + 3
        #expect(tier(.toolUse, at: now) == .strong)
        #expect(tier(.general, at: now) == .weak)
        #expect(bias.beta(model: "qwen@def", bucket: .toolUse, at: now) == 0)
        #expect(tier(.toolUse, at: now + RoutingOutcomeBias.halfLife) == .weak)
    }

    @Test("β stays within [0, βmax]; a clean turn takes back η/4 and an entry that reaches 0 is dropped")
    func biasIsClamped() {
        var bias = RoutingOutcomeBias()
        for _ in 0..<100 { bias.record(failed: true, model: "m", bucket: .toolUse, at: Self.start) }
        #expect(bias.beta(model: "m", bucket: .toolUse, at: Self.start) == RoutingOutcomeBias.betaMax)
        bias.record(failed: false, model: "m", bucket: .toolUse, at: Self.start)
        #expect(abs(bias.beta(model: "m", bucket: .toolUse, at: Self.start) - (RoutingOutcomeBias.betaMax - RoutingOutcomeBias.eta / 4)) < 1e-12)
        for _ in 0..<100 { bias.record(failed: false, model: "m", bucket: .toolUse, at: Self.start) }
        #expect(bias == RoutingOutcomeBias())
    }

    @Test("Repetition, a cut at the server's token ceiling and a repaired or rejected tool call fail a turn; a client max_tokens cut does not; errors are not counted")
    func outcomeClassification() {
        let open = GenerationRequest(prompt: "p")
        let capped = GenerationRequest(prompt: "p", maxTokens: 64)
        #expect(RoutingOutcomeBias.failed(open, Self.response(.repetition)) == true)
        #expect(RoutingOutcomeBias.failed(open, Self.response(.length)) == true)
        #expect(RoutingOutcomeBias.failed(capped, Self.response(.length)) == false)
        #expect(RoutingOutcomeBias.failed(capped, Self.response(.stop, repairs: 1)) == true)
        #expect(RoutingOutcomeBias.failed(capped, Self.response(.stop, rejections: 1)) == true)
        #expect(RoutingOutcomeBias.failed(open, Self.response(.stop)) == false)
        #expect(RoutingOutcomeBias.failed(open, Self.response(.error)) == nil)
        #expect(RoutingOutcomeBias.Bucket(GenerationRequest(prompt: "p", tools: [ToolDefinition(name: "t")])) == .toolUse)
        #expect(RoutingOutcomeBias.Bucket(open) == .general)
    }
}
