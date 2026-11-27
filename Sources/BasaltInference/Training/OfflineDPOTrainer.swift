import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXOptimizers
import BasaltCore

struct CachedDPOPair: Sendable {
    let pair: TokenizedPreferencePair
    let refChosenLogp: Float
    let refRejectedLogp: Float
}

struct DPOEvalMetrics: Sendable {
    let meanMargin: Float
    let accuracy: Float
    let klDivergence: Float
}

public enum OfflineDPOTrainer {
    static func makeOptimizer(_ kind: TrainingOptimizerKind = .adamw, learningRate: Float, weightDecay: Float) -> any Optimizer {
        TrainingOptimizerFactory.make(kind, learningRate: learningRate, weightDecay: weightDecay)
    }

    static func makeOptimizer(_ options: TrainingOptimizerOptions, learningRate: Float) -> TrainingOptimizer {
        TrainingOptimizerFactory.make(options: options, learningRate: learningRate)
    }

    public static func pairLoss(
        chosenLogProbs: MLXArray,
        rejectedLogProbs: MLXArray,
        refChosenLogp: Float,
        refRejectedLogp: Float,
        beta: Float,
        labelSmoothing: Float
    ) -> (loss: MLXArray, margin: MLXArray) {
        let deltaW = chosenLogProbs.sum() - MLXArray(refChosenLogp)
        let deltaL = rejectedLogProbs.sum() - MLXArray(refRejectedLogp)
        let margin = MLXArray(beta) * (deltaW - deltaL)
        let primary = ADPOTrainer.negLogSigmoid(margin)
        if labelSmoothing > 0 {
            let smooth = ADPOTrainer.negLogSigmoid(-margin)
            let eps = MLXArray(labelSmoothing)
            let loss = (MLXArray(Float(1)) - eps) * primary + eps * smooth
            return (loss, margin)
        }
        return (primary, margin)
    }

    static func train(
        pairs: [PreferencePair],
        config: OfflineDPOConfig,
        directory: URL,
        family: String?,
        context: ModelContext
    ) throws -> OfflineDPOReport {
        try config.effectiveOptimizerOptions.validateAdapterCompatible(prefix: "dpo")
        let tokenized = try tokenize(pairs, maxSeqLen: config.maxSeqLen, context: context)
        guard !tokenized.isEmpty else {
            throw BasaltError.invalidRequestStructure(details: "dpo: no valid tokenized preference pairs within maxSeqLen \(config.maxSeqLen)")
        }
        defer { _ = PHMAdapterApplicator.revert(in: context.model) }
        let (configuration, cached) = try attachAndCacheReference(tokenized: tokenized, config: config, model: context.model)
        let before = evaluate(model: context.model, cached: cached, config: config)
        let (losses, stabilitySummary) = try runSteps(model: context.model, cached: cached, config: config)
        let after = evaluate(model: context.model, cached: cached, config: config)
        var gatePassed: Bool?
        if let gate = config.gate {
            gatePassed = try MLXInferenceEngine.gate(gate, context: context, live: [:], family: family).passed
        }
        try MLXInferenceEngine.saveConfiguredAdapter(configuration, model: context.model, to: directory)
        var report = OfflineDPOReport(outputPath: directory.path)
        report.losses = losses
        report.meanMarginBefore = before.meanMargin
        report.meanMarginAfter = after.meanMargin
        report.preferenceAccuracyBefore = before.accuracy
        report.preferenceAccuracyAfter = after.accuracy
        report.klDivergence = after.klDivergence
        report.gatePassed = gatePassed
        report.stabilitySummary = stabilitySummary
        return report
    }
}

extension OfflineDPOTrainer {
    static func tokenize(
        _ pairs: [PreferencePair],
        maxSeqLen: Int,
        context: ModelContext
    ) throws -> [TokenizedPreferencePair] {
        let vocab = MLXInferenceEngine.vocabularySize(context.model)
        let eos = context.tokenizer.eosTokenId.map { [$0] } ?? []
        var tokenized: [TokenizedPreferencePair] = []
        tokenized.reserveCapacity(pairs.count)
        for pair in pairs {
            try Task.checkCancellation()
            let promptIds = try MLXInferenceEngine.labPrompt(pair.prompt, context)
            guard promptIds.count < maxSeqLen, isValidTokens(promptIds, vocab: vocab) else { continue }
            let budget = maxSeqLen - promptIds.count
            let chosenIds = Array((context.tokenizer.encode(text: pair.chosen, addSpecialTokens: false) + eos).prefix(budget))
            let rejectedIds = Array((context.tokenizer.encode(text: pair.rejected, addSpecialTokens: false) + eos).prefix(budget))
            guard chosenIds != rejectedIds,
                  isValidTokens(chosenIds, vocab: vocab),
                  isValidTokens(rejectedIds, vocab: vocab) else {
                continue
            }
            tokenized.append(TokenizedPreferencePair(promptIds: promptIds, chosenIds: chosenIds, rejectedIds: rejectedIds))
        }
        return tokenized
    }

    private static func isValidTokens(_ ids: [Int], vocab: Int) -> Bool {
        !ids.isEmpty && ids.allSatisfy { $0 >= 0 && $0 < vocab }
    }

    static func attachAndCacheReference(
        tokenized: [TokenizedPreferencePair],
        config: OfflineDPOConfig,
        model: any LanguageModel
    ) throws -> (ConfiguredAdapter, [CachedDPOPair]) {
        if let initPath = config.initialAdapterPath {
            let initURL = URL(fileURLWithPath: (initPath as NSString).expandingTildeInPath)
            let configuration = try attachInitialAdapter(at: initURL, to: model)
            let cached = try cacheReference(tokenized, referenceFree: config.referenceFree, model: model)
            return (configuration, cached)
        }
        let cached = try cacheReference(tokenized, referenceFree: config.referenceFree, model: model)
        let method = MLXInferenceEngine.configuredMethod(explicit: config.method, fallbackRank: config.rank)
        let configuration = try withRandomState(MLXRandom.RandomState(seed: config.seed)) {
            try MLXInferenceEngine.attachConfiguredAdapter(to: model, layers: config.layers, method: method)
        }
        return (configuration, cached)
    }

    static func attachInitialAdapter(at directory: URL, to model: any LanguageModel) throws -> ConfiguredAdapter {
        func refused(_ reason: String) -> BasaltError {
            .invalidRequestStructure(details: "dpo: initial adapter \(directory.path): \(reason)")
        }
        switch MLXInferenceEngine.probeFineTuneType(directory) {
        case "phm":
            let container = try PHMAdapterContainer.from(directory: directory)
            try container.load(into: model)
            return .phm(container.configuration)
        case "oft":
            let container = try OFTAdapterContainer.from(directory: directory)
            try container.load(into: model)
            return .oft(container.configuration)
        default:
            let container = try MLXInferenceEngine.loraContainer(at: directory, for: model, refused)
            model.freeze()
            _ = try MLXInferenceEngine.attachContainer(container, to: model, refused)
            let trainable = model.trainableParameters().flattened()
            guard !trainable.isEmpty else { throw refused("loaded adapter exposes no trainable parameters") }
            eval(trainable.map(\.1))
            return .lora(container.configuration)
        }
    }

    static func cacheReference(
        _ pairs: [TokenizedPreferencePair],
        referenceFree: Bool,
        model: any LanguageModel
    ) throws -> [CachedDPOPair] {
        if referenceFree {
            return pairs.map { CachedDPOPair(pair: $0, refChosenLogp: 0, refRejectedLogp: 0) }
        }
        var cached: [CachedDPOPair] = []
        cached.reserveCapacity(pairs.count)
        for pair in pairs {
            try Task.checkCancellation()
            let cw = ADPOTrainer.tokenLogProbs(model: model, promptIds: pair.promptIds, responseIds: pair.chosenIds).sum()
            let cl = ADPOTrainer.tokenLogProbs(model: model, promptIds: pair.promptIds, responseIds: pair.rejectedIds).sum()
            eval(cw, cl)
            cached.append(CachedDPOPair(pair: pair, refChosenLogp: cw.item(Float.self), refRejectedLogp: cl.item(Float.self)))
        }
        return cached
    }
}

extension OfflineDPOTrainer {
    static func evaluate(
        model: any LanguageModel,
        cached: [CachedDPOPair],
        config: OfflineDPOConfig
    ) -> DPOEvalMetrics {
        guard !cached.isEmpty else { return DPOEvalMetrics(meanMargin: 0, accuracy: 0, klDivergence: 0) }
        var marginSum: Float = 0
        var correct = 0
        var klSum: Float = 0
        for item in cached {
            let cw = ADPOTrainer.tokenLogProbs(model: model, promptIds: item.pair.promptIds, responseIds: item.pair.chosenIds).sum()
            let cl = ADPOTrainer.tokenLogProbs(model: model, promptIds: item.pair.promptIds, responseIds: item.pair.rejectedIds).sum()
            eval(cw, cl)
            let piW = cw.item(Float.self)
            let piL = cl.item(Float.self)
            let stats = DPOMath.loss(
                policyChosenLogp: piW,
                policyRejectedLogp: piL,
                refChosenLogp: item.refChosenLogp,
                refRejectedLogp: item.refRejectedLogp,
                beta: config.beta,
                labelSmoothing: config.labelSmoothing
            )
            marginSum += stats.margin
            if stats.margin > 0 { correct += 1 }
            if !config.referenceFree {
                let n = Float(max(1, item.pair.chosenIds.count))
                klSum += DPOMath.schulmanKL(logPolicy: piW / n, logReference: item.refChosenLogp / n)
            }
        }
        let denom = Float(cached.count)
        return DPOEvalMetrics(
            meanMargin: marginSum / denom,
            accuracy: Float(correct) / denom,
            klDivergence: klSum / denom
        )
    }

    static func runSteps(
        model: any LanguageModel,
        cached: [CachedDPOPair],
        config: OfflineDPOConfig
    ) throws -> (losses: [Float], stabilitySummary: TrainingStabilitySummary?) {
        let optimizer = makeOptimizer(config.effectiveOptimizerOptions, learningRate: config.learningRate)
        if optimizer.usesQKClip {
            try QKClipController.validateTrainableSupport(for: model, prefix: "dpo")
        }
        let stepFn = valueAndGrad(model: model as Module) { _, batch in
            batchObjective(model: model, batch: batch, config: config)
        }
        let monitor = TrainingStabilityMonitor(sampleEvery: config.stabilityTelemetryEvery ?? 0)
        var stability = TrainingStabilityAccumulator()
        var rng = SplitMix64(seed: config.seed)
        var order: [Int] = []
        var cursor = 0
        var losses: [Float] = []
        losses.reserveCapacity(config.iterations)
        for step in 0 ..< config.iterations {
            try Task.checkCancellation()
            let batch = nextBatch(from: cached, batchSize: config.batchSize, order: &order, cursor: &cursor, rng: &rng)
            let before = monitor.snapshotWeights(model.trainableParameters(), step: step)
            let (values, gradients) = stepFn(model as Module, batch)
            let (clipped, _) = clipGradNorm(gradients: gradients, maxNorm: config.maxGradNorm)
            optimizer.update(model: model as Module, gradients: clipped)
            eval(model as Module, optimizer, values)
            let loss = values[0].item(Float.self)
            guard loss.isFinite else {
                throw BasaltError.inferenceError("dpo: loss became non-finite at step \(step)")
            }
            if optimizer.usesQKClip {
                let seqs = batch.compactMap { item -> MLXArray? in
                    let ids = item.pair.promptIds + item.pair.chosenIds
                    return ids.count > 1 ? MLXArray(ids.dropLast().map(Int32.init)).reshaped([1, ids.count - 1]) : nil
                }
                QKClipController.measureAndClipSequences(model: model, sequences: seqs, options: optimizer.options)
            }
            monitor.recordStep(step: step, before: before, after: model.trainableParameters(), gradients: gradients, into: &stability)
            losses.append(loss)
        }
        return (losses, monitor.isEnabled ? stability.finalize() : nil)
    }

    static func batchObjective(
        model: any LanguageModel,
        batch: [CachedDPOPair],
        config: OfflineDPOConfig
    ) -> [MLXArray] {
        var pairLosses: [MLXArray] = []
        pairLosses.reserveCapacity(batch.count)
        for item in batch {
            let cw = ADPOTrainer.tokenLogProbs(model: model, promptIds: item.pair.promptIds, responseIds: item.pair.chosenIds)
            let cl = ADPOTrainer.tokenLogProbs(model: model, promptIds: item.pair.promptIds, responseIds: item.pair.rejectedIds)
            let (loss, _) = pairLoss(
                chosenLogProbs: cw,
                rejectedLogProbs: cl,
                refChosenLogp: item.refChosenLogp,
                refRejectedLogp: item.refRejectedLogp,
                beta: config.beta,
                labelSmoothing: config.labelSmoothing
            )
            pairLosses.append(loss)
        }
        return [stacked(pairLosses).mean()]
    }

    static func nextBatch(
        from cached: [CachedDPOPair],
        batchSize: Int,
        order: inout [Int],
        cursor: inout Int,
        rng: inout SplitMix64
    ) -> [CachedDPOPair] {
        let count = min(batchSize, cached.count)
        var batch: [CachedDPOPair] = []
        batch.reserveCapacity(count)
        for _ in 0 ..< count {
            if cursor >= order.count {
                order = Array(cached.indices)
                order.shuffle(using: &rng)
                cursor = 0
            }
            batch.append(cached[order[cursor]])
            cursor += 1
        }
        return batch
    }
}
