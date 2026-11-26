import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXOptimizers
import BasaltCore

struct GRPORollout: Sendable {
    let responseIds: [Int]
    let text: String
    let reward: Float
}

struct PreparedGRPORollout: Sendable {
    let responseIds: [Int]
    let oldLogProbs: [Float]
    let refLogProbs: [Float]
    let advantage: Float
}

struct PreparedGRPOGroup: Sendable {
    let promptIds: [Int]
    let rollouts: [PreparedGRPORollout]
}

struct GRPOTaskSlot: Sendable {
    let task: TrainingTask
    let promptIds: [Int]
    let round: Int
    let taskIndex: Int
}

struct GRPORoundsOutcome: Sendable {
    let losses: [Float]
    let meanRewards: [Float]
    let meanKL: Float
    let updatesApplied: Int
    let skippedGroups: Int
}

public enum GRPOTrainer {
    typealias Scorer = (TrainingTask, String) -> Float

    static func makeOptimizer(_ kind: TrainingOptimizerKind = .adamw, learningRate: Float, weightDecay: Float) -> any Optimizer {
        TrainingOptimizerFactory.make(kind, learningRate: learningRate, weightDecay: weightDecay)
    }

    static func defaultScore(_ task: TrainingTask, _ output: String) -> Float {
        task.accepts(output) ? 1.0 : 0.0
    }

    public static func surrogateLoss(
        logPolicy: MLXArray,
        logOld: MLXArray,
        logReference: MLXArray,
        advantage: Float,
        clipEpsilon: Float,
        betaKL: Float
    ) -> (loss: MLXArray, kl: MLXArray) {
        let logRatio = clip(logPolicy - stopGradient(logOld), min: Float(-20.0), max: Float(20.0))
        let ratio = exp(logRatio)
        let adv = MLXArray(advantage)
        let clippedRatio = clip(ratio, min: Float(1.0) - clipEpsilon, max: Float(1.0) + clipEpsilon)
        let surrogate = minimum(ratio * adv, clippedRatio * adv)
        let klDiff = clip(stopGradient(logReference) - logPolicy, min: Float(-20.0), max: Float(20.0))
        let klPerToken = maximum(exp(klDiff) - klDiff - MLXArray(Float(1.0)), MLXArray(Float(0.0)))
        let loss = (-surrogate + MLXArray(betaKL) * klPerToken).mean()
        return (loss, stopGradient(klPerToken.mean()))
    }

    static func train(
        tasks: [TrainingTask],
        config: GRPOConfig,
        directory: URL,
        family: String?,
        context: ModelContext,
        scorer: Scorer? = nil
    ) throws -> GRPOReport {
        let score = scorer ?? Self.defaultScore
        let heldOut = Array(tasks.suffix(max(1, tasks.count / 5)))
        let practice = Array(tasks.dropLast(heldOut.count))
        let accuracyBefore = try evaluateHeldOut(heldOut, context: context, score: score)
        let method = MLXInferenceEngine.configuredMethod(explicit: config.method, fallbackRank: config.rank)
        let configuration = try withRandomState(MLXRandom.RandomState(seed: config.seed)) {
            try MLXInferenceEngine.attachConfiguredAdapter(to: context.model, layers: config.layers, method: method)
        }
        defer { _ = PHMAdapterApplicator.revert(in: context.model) }
        let optimizer = makeOptimizer(config.optimizer, learningRate: config.learningRate, weightDecay: config.weightDecay)
        let outcome = try runRounds(practice: practice, config: config, optimizer: optimizer, context: context, score: score)
        let accuracyAfter = try evaluateHeldOut(heldOut, context: context, score: score)
        let gateReport = try MLXInferenceEngine.gate(config.gate, context: context, live: [:], family: family)
        try MLXInferenceEngine.saveConfiguredAdapter(configuration, model: context.model, to: directory)
        var report = GRPOReport(outputPath: directory.path)
        report.losses = outcome.losses
        report.meanRewards = outcome.meanRewards
        report.accuracyBefore = accuracyBefore
        report.accuracyAfter = accuracyAfter
        report.meanKL = outcome.meanKL
        report.updatesApplied = outcome.updatesApplied
        report.skippedGroups = outcome.skippedGroups
        report.gatePassed = gateReport.passed
        return report
    }
}

extension GRPOTrainer {
    static func rolloutSeed(base: UInt64, round: Int, taskIndex: Int, slot: Int) -> UInt64 {
        let mixed = base
            ^ (UInt64(bitPattern: Int64(round)) &* 0x9E37_79B9_7F4A_7C15)
            ^ (UInt64(bitPattern: Int64(taskIndex)) &* 0xBF58_476D_1CE4_E5B9)
            ^ (UInt64(bitPattern: Int64(slot)) &* 0x94D0_49BB_1331_11EB)
        var rng = SplitMix64(seed: mixed)
        return rng.next()
    }

    static func evaluateHeldOut(
        _ tasks: [TrainingTask],
        context: ModelContext,
        score: Scorer
    ) throws -> Float {
        guard !tasks.isEmpty else { return 0 }
        var hits: Float = 0
        for task in tasks {
            let answer = try MLXInferenceEngine.labAnswer(task.prompt, context)
            if score(task, answer) > 0 { hits += 1 }
        }
        return hits / Float(tasks.count)
    }

    static func sampleGroup(
        _ slot: GRPOTaskSlot,
        config: GRPOConfig,
        stops: Set<Int>,
        context: ModelContext,
        score: Scorer
    ) throws -> [GRPORollout] {
        var rollouts: [GRPORollout] = []
        rollouts.reserveCapacity(config.groupSize)
        for index in 0 ..< config.groupSize {
            try Task.checkCancellation()
            let seed = rolloutSeed(base: config.seed, round: slot.round, taskIndex: slot.taskIndex, slot: index)
            let sample = try sampleOne(promptIds: slot.promptIds, seed: seed, config: config, stops: stops, context: context)
            let reward = score(slot.task, sample.text)
            rollouts.append(GRPORollout(responseIds: sample.tokenIds, text: sample.text, reward: reward))
        }
        return rollouts
    }

    static func sampleOne(
        promptIds: [Int],
        seed: UInt64,
        config: GRPOConfig,
        stops: Set<Int>,
        context: ModelContext
    ) throws -> (tokenIds: [Int], text: String) {
        let (generated, stopped) = try withRandomState(MLXRandom.RandomState(seed: seed)) {
            let params = GenerateParameters(maxTokens: config.maxTokens, temperature: config.temperature, seed: seed)
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(promptIds.map(Int32.init))),
                model: context.model,
                parameters: params
            )
            var out: [Int] = []
            var stopId: Int?
            while let token = iterator.next() {
                if stops.contains(token) {
                    stopId = token
                    break
                }
                out.append(token)
            }
            return (out, stopId)
        }
        let text = context.tokenizer.decode(tokenIds: generated, skipSpecialTokens: true)
        let vocab = MLXInferenceEngine.vocabularySize(context.model)
        let rawIds = generated.isEmpty ? (stopped.map { [$0] } ?? []) : generated
        let validIds = rawIds.filter { $0 >= 0 && $0 < vocab }
        return (validIds, text)
    }

    static func prepareGroup(
        promptIds: [Int],
        rollouts: [GRPORollout],
        advantages: [Float],
        model: any LanguageModel
    ) -> PreparedGRPOGroup {
        let active = zip(rollouts, advantages).filter { !$0.0.responseIds.isEmpty }
        guard !active.isEmpty else {
            return PreparedGRPOGroup(promptIds: promptIds, rollouts: [])
        }
        let oldList: [[Float]] = active.map { rollout, _ in
            let lp = ADPOTrainer.tokenLogProbs(model: model, promptIds: promptIds, responseIds: rollout.responseIds)
            eval(lp)
            return lp.asArray(Float.self)
        }
        let refList: [[Float]] = ADPOTrainer.withReferencePolicy(in: model) {
            active.map { rollout, _ in
                let lp = ADPOTrainer.tokenLogProbs(model: model, promptIds: promptIds, responseIds: rollout.responseIds)
                eval(lp)
                return lp.asArray(Float.self)
            }
        }
        var prepared: [PreparedGRPORollout] = []
        prepared.reserveCapacity(active.count)
        for idx in active.indices {
            prepared.append(PreparedGRPORollout(
                responseIds: active[idx].0.responseIds,
                oldLogProbs: oldList[idx],
                refLogProbs: refList[idx],
                advantage: active[idx].1
            ))
        }
        return PreparedGRPOGroup(promptIds: promptIds, rollouts: prepared)
    }
}

extension GRPOTrainer {
    static func runRounds(
        practice: [TrainingTask],
        config: GRPOConfig,
        optimizer: any Optimizer,
        context: ModelContext,
        score: Scorer
    ) throws -> GRPORoundsOutcome {
        let stops = MLXInferenceEngine.stopTokenIds(context)
        let vocab = MLXInferenceEngine.vocabularySize(context.model)
        let promptCache = try practice.map { try MLXInferenceEngine.labPrompt($0.prompt, context) }
        var (losses, meanRewards) = ([Float](), [Float]())
        meanRewards.reserveCapacity(config.rounds)
        var (klSum, klCount, updatesApplied, skippedGroups) = (Float(0), 0, 0, 0)
        for round in 0 ..< config.rounds {
            try Task.checkCancellation()
            var (roundRewardSum, roundRewardCount) = (Float(0), 0)
            for (taskIndex, task) in practice.enumerated() {
                let promptIds = promptCache[taskIndex]
                guard !promptIds.isEmpty, promptIds.allSatisfy({ $0 >= 0 && $0 < vocab }) else {
                    skippedGroups += 1; continue
                }
                let slot = GRPOTaskSlot(task: task, promptIds: promptIds, round: round, taskIndex: taskIndex)
                let rollouts = try sampleGroup(slot, config: config, stops: stops, context: context, score: score)
                let rewards = rollouts.map(\.reward)
                roundRewardSum += rewards.reduce(Float(0), +)
                roundRewardCount += rewards.count
                let advantages = GRPOAdvantage.normalize(rewards: rewards)
                guard advantages.contains(where: { $0 != 0 }) else { skippedGroups += 1; continue }
                let group = prepareGroup(promptIds: promptIds, rollouts: rollouts, advantages: advantages, model: context.model)
                guard !group.rollouts.isEmpty else { skippedGroups += 1; continue }
                let step = try stepGroup(group, config: config, optimizer: optimizer, model: context.model)
                losses.append(step.loss)
                klSum += max(0, step.kl); klCount += 1; updatesApplied += step.updates
            }
            meanRewards.append(roundRewardCount > 0 ? roundRewardSum / Float(roundRewardCount) : 0)
        }
        return GRPORoundsOutcome(
            losses: losses, meanRewards: meanRewards,
            meanKL: klCount > 0 ? klSum / Float(klCount) : 0,
            updatesApplied: updatesApplied, skippedGroups: skippedGroups
        )
    }

    static func stepGroup(
        _ group: PreparedGRPOGroup,
        config: GRPOConfig,
        optimizer: any Optimizer,
        model: any LanguageModel
    ) throws -> (loss: Float, kl: Float, updates: Int) {
        let epochs = max(1, config.ppoEpochs)
        let stepFn = valueAndGrad(model: model as Module) { _, activeGroup in
            groupObjective(model: model, group: activeGroup, config: config)
        }
        var lastLoss: Float = 0
        var lastKL: Float = 0
        for _ in 0 ..< epochs {
            try Task.checkCancellation()
            let (values, gradients) = stepFn(model as Module, group)
            let (clipped, _) = clipGradNorm(gradients: gradients, maxNorm: config.maxGradNorm)
            optimizer.update(model: model as Module, gradients: clipped)
            eval(model as Module, optimizer, values)
            let loss = values[0].item(Float.self)
            let kl = values[1].item(Float.self)
            guard loss.isFinite, kl.isFinite else {
                throw BasaltError.inferenceError("grpo: policy gradient loss or KL became non-finite")
            }
            lastLoss = loss
            lastKL = kl
        }
        return (lastLoss, lastKL, epochs)
    }

    static func groupObjective(
        model: any LanguageModel,
        group: PreparedGRPOGroup,
        config: GRPOConfig
    ) -> [MLXArray] {
        var losses: [MLXArray] = []
        var kls: [MLXArray] = []
        losses.reserveCapacity(group.rollouts.count)
        kls.reserveCapacity(group.rollouts.count)
        for item in group.rollouts {
            let logPi = ADPOTrainer.tokenLogProbs(model: model, promptIds: group.promptIds, responseIds: item.responseIds)
            let (loss, kl) = surrogateLoss(
                logPolicy: logPi,
                logOld: MLXArray(item.oldLogProbs),
                logReference: MLXArray(item.refLogProbs),
                advantage: item.advantage,
                clipEpsilon: config.clipEpsilon,
                betaKL: config.betaKL
            )
            losses.append(loss)
            kls.append(kl)
        }
        return [stacked(losses).mean(), stacked(kls).mean()]
    }
}
