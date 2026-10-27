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
}
