import ArgumentParser
import Foundation
import Testing
@testable import BasaltCLI
@testable import BasaltCore

@Suite("Quantized AdaLoRA (LR/40), Two-Stage SFT->DPO, Self-Appraisal, and RLVR Anti-Shortcut Audit (#617)")
struct AdaLoRAAndShortcutAuditTests {
    @Test("QuantizedLRPolicy enforces 40x LR reduction on INT4/INT8 bases and scales batch size by parameter count (Derkach 2025)")
    func quantizedLRPolicyAndBatchSizing() throws {
        let fp16LR = try QuantizedLRPolicy.effectiveLearningRate(baseLR: 1e-3, weightBits: 16)
        let q4LR = try QuantizedLRPolicy.effectiveLearningRate(baseLR: 1e-3, weightBits: 4)
        let q8LR = try QuantizedLRPolicy.effectiveLearningRate(baseLR: 1e-3, weightBits: 8)
        let customLR = try QuantizedLRPolicy.effectiveLearningRate(baseLR: 1e-3, weightBits: 4, overrideDivisor: 20)

        #expect(abs(fp16LR - 1e-3) < 1e-7)
        #expect(abs(q4LR - 2.5e-5) < 1e-8)
        #expect(abs(q8LR - 2.5e-5) < 1e-8)
        #expect(abs(customLR - 5e-5) < 1e-8)

        #expect(QuantizedLRPolicy.recommendedBatchSize(parameterMillions: 39, quantized: true) == 8)
        #expect(QuantizedLRPolicy.recommendedBatchSize(parameterMillions: 244, quantized: true) == 4)
        #expect(QuantizedLRPolicy.recommendedBatchSize(parameterMillions: 769, quantized: true) == 2)
        #expect(QuantizedLRPolicy.recommendedBatchSize(parameterMillions: 1550, quantized: true) == 1)
        #expect(QuantizedLRPolicy.recommendedBatchSize(parameterMillions: 50, quantized: false) == 8)
        #expect(QuantizedLRPolicy.recommendedBatchSize(parameterMillions: -5, quantized: true) == 1)

        #expect(throws: BasaltError.self) { _ = try QuantizedLRPolicy.effectiveLearningRate(baseLR: 0, weightBits: 4) }
        #expect(throws: BasaltError.self) { _ = try QuantizedLRPolicy.effectiveLearningRate(baseLR: 1e-4, weightBits: 1) }
        #expect(throws: BasaltError.self) { _ = try QuantizedLRPolicy.effectiveLearningRate(baseLR: 1e-4, weightBits: 4, overrideDivisor: 0) }
    }

    @Test("AdaLoRARankScheduler updates EMA singular importance and reallocates ranks toward high-importance projections")
    func adaLoRARankPruningAndScheduling() throws {
        let cfg = AdaLoRARankScheduleConfig(initRank: 16, targetAverageRank: 8, warmupSteps: 5, pruneInterval: 5, importanceBeta: 0.85)
        try cfg.validate()
        #expect(throws: BasaltError.self) { try AdaLoRARankScheduleConfig(initRank: 0).validate() }
        #expect(throws: BasaltError.self) { try AdaLoRARankScheduleConfig(initRank: 8, targetAverageRank: 12).validate() }
        #expect(throws: BasaltError.self) { try AdaLoRARankScheduleConfig(warmupSteps: -1).validate() }
        #expect(throws: BasaltError.self) { try AdaLoRARankScheduleConfig(importanceBeta: 1.0).validate() }

        let updated = AdaLoRARankScheduler.updateImportance(previous: [1.0, 0.2], gradientSensitivity: [0.5, 0.8], beta: 0.8)
        #expect(updated.count == 2)
        #expect(updated[0] > updated[1])
        #expect(AdaLoRARankScheduler.updateImportance(previous: [1.0], gradientSensitivity: []).count == 1)

        let matrices = [
            AdaLoRAMatrixBudget(modulePath: "layers.0.self_attn.q_proj", currentRank: 4, minRank: 1, maxRank: 4, singularImportances: [0.95, 0.88, 0.80, 0.72]),
            AdaLoRAMatrixBudget(modulePath: "layers.0.self_attn.k_proj", currentRank: 4, minRank: 1, maxRank: 4, singularImportances: [0.15, 0.05, 0.02, 0.01]),
            AdaLoRAMatrixBudget(modulePath: "layers.0.self_attn.v_proj", currentRank: 4, minRank: 1, maxRank: 4, singularImportances: [0.90, 0.60, 0.10, 0.04]),
        ]
        let pruned = try AdaLoRARankScheduler.pruneToBudget(matrices: matrices, totalRankBudget: 7)
        #expect(pruned.reduce(0) { $0 + $1.currentRank } == 7)
        #expect(pruned[0].currentRank == 4)
        #expect(pruned[1].currentRank == 1)
        #expect(pruned[2].currentRank == 2)

        #expect(try AdaLoRARankScheduler.pruneToBudget(matrices: [], totalRankBudget: 4).isEmpty)
        #expect(throws: BasaltError.self) {
            _ = try AdaLoRARankScheduler.pruneToBudget(matrices: matrices, totalRankBudget: 2)
        }
    }

    @Test("TwoStageSFTDPOPolicy derives quantized SFT and DPO learning rates and integrates with TrainSubcommand")
    func twoStageSFTDPOPolicyAndCLI() throws {
        let policy = TwoStageSFTDPOPolicy(baseLearningRate: 2e-4, dpoLearningRateRatio: 0.2, dpoBeta: 0.1, dpoPairsPath: "pairs.jsonl")
        let stages = try policy.resolvedStages(weightBits: 4)
        #expect(abs(stages.sftLR - 5e-6) < 1e-9)
        #expect(abs(stages.dpoLR - 1e-6) < 1e-9)
        #expect(abs(stages.beta - 0.1) < 1e-6)

        #expect(throws: BasaltError.self) { _ = try TwoStageSFTDPOPolicy(dpoLearningRateRatio: 0).resolvedStages(weightBits: 4) }
        #expect(throws: BasaltError.self) { _ = try TwoStageSFTDPOPolicy(dpoBeta: -0.1).resolvedStages(weightBits: 4) }

        let cmd = try TrainSubcommand.parse(["org/model", "--dataset", "local.jsonl", "--quantized-base-bits", "4", "--learning-rate", "0.0004"])
        #expect(cmd.quantizedBaseBits == 4)
        let dpoCmd = try TrainSubcommand.parse(["org/model", "--dataset", "local.jsonl", "--dpo-pairs", "pairs.jsonl"])
        #expect(dpoCmd.dpoPairs == "pairs.jsonl")
        #expect(throws: (any Error).self) { _ = try TrainSubcommand.parse(["org/model", "--dataset", "local.jsonl", "--adalora-target-rank", "8"]) }

        var req = CorpusTrainRequest(corpus: CorpusSourceSpec(source: "local.jsonl"))
        req.learningRate = 4e-4
        req.quantizedBaseBits = 4
        #expect(abs(req.effectiveLearningRate - 1e-5) < 1e-9)
    }

    @Test("SelfAppraisalCore selects verified, novel, high-scoring episodes for iterative self-improvement (KIT)")
    func selfAppraisalEpisodeSelection() throws {
        let episodes = [
            AppraisalEpisode(id: "e1", reward: 0.9, selfScore: 0.85, noveltyScore: 0.7, tokenCount: 120, verifiedSuccess: true),
            AppraisalEpisode(id: "e2", reward: 0.95, selfScore: 0.95, noveltyScore: 0.8, tokenCount: 80, verifiedSuccess: false),
            AppraisalEpisode(id: "e3", reward: 0.8, selfScore: 0.8, noveltyScore: 0.05, tokenCount: 60, verifiedSuccess: true),
            AppraisalEpisode(id: "e4", reward: 0.9, selfScore: 0.85, noveltyScore: 0.7, tokenCount: 90, verifiedSuccess: true),
            AppraisalEpisode(id: "e5", reward: 0.3, selfScore: 0.3, noveltyScore: 0.5, tokenCount: 50, verifiedSuccess: true),
        ]
        let selected = try SelfAppraisalCore.selectEpisodes(
            episodes, config: SelfAppraisalConfig(minCombinedScore: 0.65, minNovelty: 0.2, maxSelectedEpisodes: 2, requireVerifiedSuccess: true)
        )
        #expect(selected.map(\.id) == ["e4", "e1"])

        #expect(throws: BasaltError.self) { _ = try SelfAppraisalCore.selectEpisodes(episodes, config: .init(minCombinedScore: 1.5)) }
        #expect(throws: BasaltError.self) { _ = try SelfAppraisalCore.selectEpisodes(episodes, config: .init(minNovelty: -0.1)) }
        #expect(throws: BasaltError.self) { _ = try SelfAppraisalCore.selectEpisodes(episodes, config: .init(maxSelectedEpisodes: 0)) }
    }

    @Test("ShortcutAuditCore flags RLVR verifier gaming and reward hacking under adversarial perturbations (DFKI ICLR 2026)")
    func shortcutAuditDetectsRewardHacking() throws {
        let clean = [
            RLVRVerificationSample(taskId: "t1", verifierPassed: true, adversarialPerturbationPassed: true, intermediateStepsValid: true, reasoningTokenCount: 64),
            RLVRVerificationSample(taskId: "t2", verifierPassed: true, adversarialPerturbationPassed: true, intermediateStepsValid: true, reasoningTokenCount: 72),
            RLVRVerificationSample(taskId: "t3", verifierPassed: false, adversarialPerturbationPassed: false, intermediateStepsValid: false, reasoningTokenCount: 50),
        ]
        let cleanReport = try ShortcutAuditCore.evaluate(clean, maxAllowedShortcutRate: 0.10)
        #expect(!cleanReport.shortcutFlagged)
        #expect(cleanReport.shortcutRate == 0.0)
        #expect(cleanReport.flaggedTaskIds.isEmpty)

        let hacked = [
            RLVRVerificationSample(taskId: "h1", verifierPassed: true, adversarialPerturbationPassed: false, intermediateStepsValid: true),
            RLVRVerificationSample(taskId: "h2", verifierPassed: true, adversarialPerturbationPassed: true, intermediateStepsValid: false),
            RLVRVerificationSample(taskId: "h3", verifierPassed: true, adversarialPerturbationPassed: true, intermediateStepsValid: true),
        ]
        let hackReport = try ShortcutAuditCore.evaluate(hacked, maxAllowedShortcutRate: 0.10)
        #expect(hackReport.shortcutFlagged)
        #expect(abs(hackReport.shortcutRate - (2.0 / 3.0)) < 1e-6)
        #expect(hackReport.flaggedTaskIds == ["h1", "h2"])

        let empty = try ShortcutAuditCore.evaluate([])
        #expect(empty.sampleCount == 0 && !empty.shortcutFlagged)
        #expect(throws: BasaltError.self) { _ = try ShortcutAuditCore.evaluate(clean, maxAllowedShortcutRate: 1.2) }

        let decoded = try JSONDecoder().decode(ShortcutAuditReport.self, from: JSONEncoder().encode(hackReport))
        #expect(decoded == hackReport)
    }
}
