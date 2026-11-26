import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference

@Suite("Offline DPO and GRPO value core: JSONL pair loading, group advantage normalization, DPO loss, and Schulman KL (#822)")
struct OfflineDPOAndGRPOCoreTests {
    @Test("PreferencePairsLoader parses prompt and messages JSONL schemas from strings and files")
    func preferencePairsLoaderValidSchemas() throws {
        let jsonl = """
        {"prompt": "What is 2 + 2?", "chosen": "4", "rejected": "5"}

        {"messages": [{"role": "user", "content": "Say hello."}], "chosen": "Hello!", "rejected": "Goodbye!"}
        {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Sky colour?"}], "chosen": [{"role": "assistant", "content": "Blue"}], "rejected": [{"role": "assistant", "content": "Green"}]}
        """
        let parsed = try PreferencePairsLoader.parse(jsonl: jsonl)
        #expect(parsed.count == 3)
        #expect(parsed[0] == PreferencePair(prompt: "What is 2 + 2?", chosen: "4", rejected: "5"))
        #expect(parsed[1] == PreferencePair(prompt: "Say hello.", chosen: "Hello!", rejected: "Goodbye!"))
        #expect(parsed[2].prompt == "system: Be brief.\nuser: Sky colour?")
        #expect(parsed[2].chosen == "Blue" && parsed[2].rejected == "Green")

        let scratch = FileManager.default.temporaryDirectory.appending(path: "dpo-pairs-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: scratch) }
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        let file = scratch.appending(component: "pairs.jsonl")
        try Data(jsonl.utf8).write(to: file)
        #expect(try PreferencePairsLoader.load(from: file.path) == parsed)
    }

    @Test("PreferencePairsLoader refuses empty files, missing fields, malformed JSON, and identical chosen/rejected responses")
    func preferencePairsLoaderRefusals() throws {
        let invalidInputs = [
            "",
            "   \n\t\n",
            "not valid json",
            #"{"chosen": "a", "rejected": "b"}"#,
            #"{"messages": [], "chosen": "a", "rejected": "b"}"#,
            #"{"prompt": "q", "rejected": "b"}"#,
            #"{"prompt": "q", "chosen": "a"}"#,
            #"{"prompt": "q", "chosen": "   ", "rejected": "b"}"#,
            #"{"prompt": "q", "chosen": "same", "rejected": "same"}"#,
            #"{"prompt": "q", "chosen": "same ", "rejected": " same"}"#,
        ]
        for bad in invalidInputs {
            #expect(throws: BasaltError.self) {
                _ = try PreferencePairsLoader.parse(jsonl: bad)
            }
        }
        #expect(throws: BasaltError.self) {
            _ = try PreferencePairsLoader.load(from: "/nonexistent-\(UUID().uuidString)/pairs.jsonl")
        }
    }

    @Test("GRPOAdvantage.normalize computes zero-mean unit-std advantages and returns zeros on degenerate or empty groups")
    func grpoAdvantageNormalization() {
        #expect(GRPOAdvantage.normalize(rewards: []).isEmpty)
        #expect(GRPOAdvantage.normalize(rewards: [1.0]) == [0.0])
        #expect(GRPOAdvantage.normalize(rewards: [1.0, 1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0, 0.0])
        #expect(GRPOAdvantage.normalize(rewards: [0.0, 0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0, 0.0])
        #expect(GRPOAdvantage.normalize(rewards: [1.0, .nan, 0.0]) == [0.0, 0.0, 0.0])

        let balanced = GRPOAdvantage.normalize(rewards: [1.0, 0.0, 1.0, 0.0])
        #expect(balanced.count == 4)
        for (actual, expected) in zip(balanced, [Float(1), Float(-1), Float(1), Float(-1)]) {
            #expect(Swift.abs(actual - expected) < 1e-5)
        }

        let skewed = GRPOAdvantage.normalize(rewards: [1.0, 0.0, 0.0, 0.0])
        let mean = skewed.reduce(Float(0), +) / Float(skewed.count)
        let variance = skewed.reduce(Float(0)) { $0 + ($1 - mean) * ($1 - mean) } / Float(skewed.count)
        #expect(Swift.abs(mean) < 1e-5)
        #expect(Swift.abs(variance - 1.0) < 1e-4)
        #expect(skewed[0] > 0 && skewed[1] < 0)
    }

    @Test("DPOMath.loss matches closed-form Rafailov objective with label smoothing and Schulman KL is always non-negative")
    func dpoLossAndSchulmanKL() {
        let zero = DPOMath.loss(
            policyChosenLogp: -5.0,
            policyRejectedLogp: -5.0,
            refChosenLogp: -5.0,
            refRejectedLogp: -5.0,
            beta: 0.1,
            labelSmoothing: 0.0
        )
        #expect(Swift.abs(zero.loss - Float(Foundation.log(2.0))) < 1e-6)
        #expect(Swift.abs(zero.margin) < 1e-6)
        #expect(Swift.abs(zero.chosenReward) < 1e-6 && Swift.abs(zero.rejectedReward) < 1e-6)

        let preferred = DPOMath.loss(
            policyChosenLogp: -2.0,
            policyRejectedLogp: -8.0,
            refChosenLogp: -4.0,
            refRejectedLogp: -4.0,
            beta: 0.5,
            labelSmoothing: 0.0
        )
        #expect(Swift.abs(preferred.chosenReward - 1.0) < 1e-6)
        #expect(Swift.abs(preferred.rejectedReward - (-2.0)) < 1e-6)
        #expect(Swift.abs(preferred.margin - 3.0) < 1e-6)
        #expect(preferred.loss < zero.loss && preferred.loss > 0)

        let smoothed = DPOMath.loss(
            policyChosenLogp: -2.0,
            policyRejectedLogp: -8.0,
            refChosenLogp: -4.0,
            refRejectedLogp: -4.0,
            beta: 0.5,
            labelSmoothing: 0.1
        )
        #expect(smoothed.loss > preferred.loss && smoothed.loss.isFinite)

        #expect(DPOMath.schulmanKL(logPolicy: -3.5, logReference: -3.5) == 0.0)
        for delta in [Float(-5.0), -1.0, -1e-6, 0.0, 1e-6, 0.5, 2.0, 5.0] {
            let kl = DPOMath.schulmanKL(logPolicy: -2.0, logReference: -2.0 + delta)
            #expect(kl.isFinite && kl >= 0.0)
        }
    }

    @Test("OfflineDPOConfig and GRPOConfig validate bounds and round-trip Codable")
    func configValidationAndCodable() throws {
        let dpo = OfflineDPOConfig(pairsPath: "pairs.jsonl")
        try dpo.validate()
        let decodedDPO = try JSONDecoder().decode(OfflineDPOConfig.self, from: JSONEncoder().encode(dpo))
        #expect(decodedDPO == dpo)

        var badBeta = dpo
        badBeta.beta = 0
        #expect(throws: BasaltError.self) { try badBeta.validate() }

        var badSmooth = dpo
        badSmooth.labelSmoothing = 0.5
        #expect(throws: BasaltError.self) { try badSmooth.validate() }

        let grpo = GRPOConfig()
        try grpo.validate()
        let decodedGRPO = try JSONDecoder().decode(GRPOConfig.self, from: JSONEncoder().encode(grpo))
        #expect(decodedGRPO == grpo)

        var badGroup = grpo
        badGroup.groupSize = 1
        #expect(throws: BasaltError.self) { try badGroup.validate() }

        var badClip = grpo
        badClip.clipEpsilon = 1.0
        #expect(throws: BasaltError.self) { try badClip.validate() }
    }
}

@Suite("CLI TrainSubcommand parsing and validation for standalone DPO, two-stage SFT -> DPO, and GRPO (#822)")
struct OfflineDPOAndGRPOCLITests {
    @Test("TrainSubcommand parses standalone --dpo-pairs, two-stage --dataset + --dpo-pairs, and --grpo")
    func parsesValidModes() throws {
        let standaloneDPO = try TrainSubcommand.parse([
            "org/model", "--dpo-pairs", "pairs.jsonl", "--dpo-beta", "0.25", "--iterations", "40",
        ])
        #expect(standaloneDPO.dataset == nil)
        #expect(standaloneDPO.dpoPairs == "pairs.jsonl")
        #expect(Swift.abs(standaloneDPO.dpoBeta - 0.25) < 1e-6)
        #expect(!standaloneDPO.grpo)

        let twoStage = try TrainSubcommand.parse([
            "org/model", "--dataset", "sft.jsonl", "--dpo-pairs", "pairs.jsonl", "--dpo-beta", "0.15",
        ])
        #expect(twoStage.dataset == "sft.jsonl")
        #expect(twoStage.dpoPairs == "pairs.jsonl")
        #expect(Swift.abs(twoStage.dpoBeta - 0.15) < 1e-6)
        #expect(!twoStage.grpo)

        let grpo = try TrainSubcommand.parse([
            "org/model", "--grpo", "--grpo-group-size", "6", "--grpo-beta-kl", "0.08",
            "--grpo-rounds", "5", "--grpo-tasks", "12",
        ])
        #expect(grpo.grpo)
        #expect(grpo.dataset == nil && grpo.dpoPairs == nil)
        #expect(grpo.grpoGroupSize == 6)
        #expect(Swift.abs(grpo.grpoBetaKL - 0.08) < 1e-6)
        #expect(grpo.grpoRounds == 5 && grpo.grpoTasks == 12)
    }

    @Test("TrainSubcommand rejects missing modes, incompatible flags, and invalid DPO/GRPO parameters")
    func rejectsInvalidModesAndArguments() {
        let invalidInvocations: [[String]] = [
            ["org/model"],
            ["org/model", "--dpo-pairs", "pairs.jsonl", "--dpo-beta", "0"],
            ["org/model", "--dpo-pairs", "pairs.jsonl", "--dpo-beta=-0.1"],
            ["org/model", "--grpo", "--grpo-group-size", "1"],
            ["org/model", "--grpo", "--grpo-beta-kl=-0.01"],
            ["org/model", "--grpo", "--grpo-rounds", "0"],
            ["org/model", "--grpo", "--grpo-tasks", "1"],
            ["org/model", "--grpo", "--dpo-pairs", "pairs.jsonl"],
            ["org/model", "--grpo", "--dataset", "sft.jsonl"],
            ["org/model", "--dpo-pairs", "pairs.jsonl", "--resume"],
        ]
        for args in invalidInvocations {
            #expect(throws: (any Error).self, "\(args)") {
                _ = try TrainSubcommand.parse(args)
            }
        }
    }
}

@Suite("End-to-end Offline DPO and GRPO LoRA training on TinyModel Qwen3 (#822)", .serialized)
struct OfflineDPOAndGRPOEngineTests {
    private static func parameterBits(_ container: ModelContainer) async -> [String: [UInt32]] {
        await container.perform { context in
            let flat = context.model.parameters().flattened()
            eval(flat.map(\.1))
            var out: [String: [UInt32]] = [:]
            for (key, array) in flat {
                if array.dtype == .uint32 {
                    out[key] = array.asArray(UInt32.self)
                } else {
                    out[key] = array.asType(.float32).asArray(Float.self).map(\.bitPattern)
                }
            }
            return out
        }
    }

    private static func writePreferencePairs(to directory: URL) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let file = directory.appending(component: "pairs.jsonl")
        let lines = [
            #"{"prompt":"Choose a calm greeting.","chosen":"hello warm morning friend","rejected":"zzzzz qqqqq xxxxx"}"#,
            #"{"prompt":"Pick a short colour.","chosen":"bright clear azure blue","rejected":"xxxxx vvvvv kkkkk"}"#,
            #"{"messages":[{"role":"user","content":"Name a fruit."}],"chosen":"sweet crisp green apple","rejected":"wwwww jjjjj qqqqq"}"#,
            #"{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Pick a season."}],"chosen":"cool golden autumn wind","rejected":"kkkkk zzzzz vvvvv"}"#,
        ].joined(separator: "\n")
        try Data(lines.utf8).write(to: file)
        return file
    }

    @Test("Offline DPO on TinyModel Qwen3 increases margin and preference accuracy, preserves base weights bit-identically, and reloads LoRA")
    func offlineDPOEndToEnd() async throws {
        let scratch = FileManager.default.temporaryDirectory.appending(path: "offline-dpo-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: scratch) }
        let jsonl = try Self.writePreferencePairs(to: scratch)
        let outDir = scratch.appending(component: "dpo-adapter")

        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        let beforeBits = await Self.parameterBits(container)

        var config = OfflineDPOConfig(
            pairsPath: jsonl.path,
            beta: 0.2,
            iterations: 10,
            learningRate: 1e-3,
            batchSize: 2,
            maxSeqLen: 128
        )
        config.rank = 4
        config.layers = [2, 3]
        config.seed = 822
        config.output = outDir.path

        let report = try await engine.trainOfflineDPO(config)
        #expect(report.losses.count == 10)
        #expect(report.losses.allSatisfy { $0.isFinite })
        let firstLoss = try #require(report.losses.first)
        let lastLoss = try #require(report.losses.last)
        #expect(lastLoss < firstLoss)
        #expect(report.meanMarginAfter > report.meanMarginBefore)
        #expect(report.preferenceAccuracyAfter >= report.preferenceAccuracyBefore)
        #expect(report.preferenceAccuracyAfter > 0)
        #expect(report.klDivergence.isFinite && report.klDivergence >= 0)
        #expect(report.outputPath == outDir.path)
        #expect(await Self.parameterBits(container) == beforeBits)

        let loaded = try LoRAContainer.from(directory: outDir)
        try loaded.load(into: try TinyModel.make(.qwen3))
    }

    @Test("Two-stage SFT -> DPO initializes from the SFT LoRA adapter, improves preference margin, and restores base weights bit-identically")
    func twoStageSFTThenDPO() async throws {
        let scratch = FileManager.default.temporaryDirectory.appending(path: "two-stage-dpo-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: scratch) }
        let pairsFile = try Self.writePreferencePairs(to: scratch)
        let sftFile = scratch.appending(component: "sft.jsonl")
        let sftLines = (0 ..< 4).map { idx in
            #"{"messages":[{"role":"user","content":"Question \#(idx)"},{"role":"assistant","content":"Answer \#(idx) is clear."}]}"#
        }.joined(separator: "\n")
        try Data(sftLines.utf8).write(to: sftFile)

        let sftDir = scratch.appending(component: "sft-adapter")
        let dpoDir = scratch.appending(component: "dpo-adapter")
        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        let beforeBits = await Self.parameterBits(container)

        var sftReq = CorpusTrainRequest(corpus: CorpusSourceSpec(source: sftFile.path, maxRows: 4, includeReasoning: false))
        (sftReq.rank, sftReq.iterations, sftReq.batchSize, sftReq.learningRate, sftReq.maxSeqLen) = (4, 3, 2, 3e-3, 128)
        (sftReq.checkpointEvery, sftReq.output) = (0, sftDir.path)
        let sftReport = try await engine.trainCorpus(sftReq, lease: nil)

        var dpoCfg = OfflineDPOConfig(pairsPath: pairsFile.path, beta: 0.2, iterations: 6, learningRate: 1e-3, batchSize: 2, maxSeqLen: 128)
        dpoCfg.labelSmoothing = 0.05
        dpoCfg.initialAdapterPath = sftReport.adapterDirectory
        dpoCfg.output = dpoDir.path
        let dpoReport = try await engine.trainOfflineDPO(dpoCfg)

        #expect(dpoReport.losses.count == 6 && dpoReport.losses.allSatisfy { $0.isFinite })
        #expect(dpoReport.meanMarginAfter > dpoReport.meanMarginBefore)
        #expect(await Self.parameterBits(container) == beforeBits)

        let loaded = try LoRAContainer.from(directory: dpoDir)
        try loaded.load(into: try TinyModel.make(.qwen3))
    }

    @Test("GRPO on TinyModel Qwen3 handles degenerate zero-variance groups without NaN, passes quality gate, preserves base weights, and reloads LoRA")
    func grpoZeroVarianceAndGate() async throws {
        let scratch = FileManager.default.temporaryDirectory.appending(path: "grpo-zero-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: scratch) }
        let outDir = scratch.appending(component: "grpo-adapter")

        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        let beforeBits = await Self.parameterBits(container)

        var config = GRPOConfig(groupSize: 4, betaKL: 0.04, clipEpsilon: 0.2, rounds: 2, learningRate: 1e-3, seed: 42)
        config.maxTokens = 6
        config.tasks = .arithmetic(count: 4, seed: 7)
        config.rank = 4
        config.layers = [2, 3]
        config.output = outDir.path
        config.gate = GateRequest(heldOut: ["What is 2 + 2?"])

        let report = try await engine.trainGRPO(config)
        #expect(report.losses.allSatisfy { $0.isFinite })
        #expect(report.meanRewards.count == 2 && report.meanRewards.allSatisfy { $0.isFinite })
        #expect(report.meanKL.isFinite && report.meanKL >= 0)
        #expect(report.accuracyBefore.isFinite && report.accuracyAfter.isFinite)
        #expect(report.gatePassed)
        #expect(report.outputPath == outDir.path)
        #expect(await Self.parameterBits(container) == beforeBits)

        let loaded = try LoRAContainer.from(directory: outDir)
        try loaded.load(into: try TinyModel.make(.qwen3))
    }

    @Test("GRPO clipped surrogate and mixed-reward training run finite updates with non-negative KL and bit-identical base weight restore")
    func grpoMixedRewardAndSurrogateClipping() async throws {
        let logRef = MLXArray([Float(-1.5), -2.0, -1.2])
        let logOld = MLXArray([Float(-1.5), -2.0, -1.2])
        let exactZero = GRPOTrainer.surrogateLoss(
            logPolicy: logOld, logOld: logOld, logReference: logRef, advantage: 1.0, clipEpsilon: 0.2, betaKL: 0.04
        )
        let clippedPos = GRPOTrainer.surrogateLoss(
            logPolicy: logOld + MLXArray(Float(1.0)), logOld: logOld, logReference: logRef, advantage: 1.0, clipEpsilon: 0.2, betaKL: 0.0
        )
        let clippedNeg = GRPOTrainer.surrogateLoss(
            logPolicy: logOld - MLXArray(Float(1.0)), logOld: logOld, logReference: logRef, advantage: -1.0, clipEpsilon: 0.2, betaKL: 0.0
        )
        eval(exactZero.loss, exactZero.kl, clippedPos.loss, clippedPos.kl, clippedNeg.loss, clippedNeg.kl)
        #expect(Swift.abs(exactZero.kl.item(Float.self)) < 1e-6)
        #expect(clippedPos.kl.item(Float.self) > 0 && clippedNeg.kl.item(Float.self) > 0)
        #expect(Swift.abs(clippedPos.loss.item(Float.self) - Float(-1.2)) < 1e-5)
        #expect(Swift.abs(clippedNeg.loss.item(Float.self) - Float(0.8)) < 1e-5)

        let scratch = FileManager.default.temporaryDirectory.appending(path: "grpo-mixed-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: scratch) }
        let outDir = scratch.appending(component: "grpo-mixed-adapter")

        let container = try TinyModel.container(.qwen3)
        let beforeBits = await Self.parameterBits(container)
        var config = GRPOConfig(groupSize: 4, betaKL: 0.04, clipEpsilon: 0.2, rounds: 2, learningRate: 1e-3, seed: 822)
        config.maxTokens = 6
        config.rank = 4
        config.layers = [2, 3]
        config.gate = GateRequest(heldOut: ["Hello."], maxKL: 50.0, maxPerplexityRise: 50.0)
        let tasks = TrainingTask.arithmetic(count: 3, seed: 11)

        let report = try await container.performEvaluating(nonSendable: ()) { [config, tasks] context, _ in
            try GRPOTrainer.train(tasks: tasks, config: config, directory: outDir, family: nil, context: context) { _, output in
                let hash = output.unicodeScalars.reduce(0) { ($0 &* 31) &+ Int($1.value) }
                return Float(hash & 0xFF) / 255.0
            }
        }
        #expect(!report.losses.isEmpty && report.losses.allSatisfy { $0.isFinite })
        #expect(report.meanRewards.count == 2 && report.meanRewards.allSatisfy { $0.isFinite })
        #expect(report.meanKL.isFinite && report.meanKL >= 0)
        #expect(await Self.parameterBits(container) == beforeBits)

        let loaded = try LoRAContainer.from(directory: outDir)
        let loraBMax = loaded.parameters.flattened()
            .filter { $0.0.hasSuffix(".lora_b") }
            .map { MLX.abs($0.1).max().item(Float.self) }
            .max() ?? 0
        #expect(loraBMax > 0)
        try loaded.load(into: try TinyModel.make(.qwen3))
    }
}
