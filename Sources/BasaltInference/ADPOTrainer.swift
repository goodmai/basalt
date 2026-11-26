import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXOptimizers
import BasaltCore

public struct TokenizedPreferencePair: Sendable, Equatable {
    public var promptIds: [Int]
    public var chosenIds: [Int]
    public var rejectedIds: [Int]

    public init(promptIds: [Int], chosenIds: [Int], rejectedIds: [Int]) {
        self.promptIds = promptIds
        self.chosenIds = chosenIds
        self.rejectedIds = rejectedIds
    }
}

public struct ADPOReferenceLogProbs: Sendable, Equatable {
    public var chosen: [Float]
    public var rejected: [Float]

    public init(chosen: [Float], rejected: [Float]) {
        self.chosen = chosen
        self.rejected = rejected
    }
}

public struct ADPOOptimizeOptions: Sendable, Equatable {
    public var iterations: Int
    public var learningRate: Float
    public var batchSize: Int
    public var emaDecay: Float?
    public var optimizer: TrainingOptimizerKind

    public init(
        iterations: Int,
        learningRate: Float,
        batchSize: Int = 4,
        emaDecay: Float? = nil,
        optimizer: TrainingOptimizerKind = .adamw
    ) {
        self.iterations = iterations
        self.learningRate = learningRate
        self.batchSize = max(1, batchSize)
        self.emaDecay = emaDecay
        self.optimizer = optimizer
    }
}

public struct ADPOTrainOutcome: Sendable, Equatable {
    public var losses: [Float]
    public var margins: [Float]
    public var marginBefore: Float
    public var marginAfter: Float
    public var entropyBefore: Float
    public var entropyAfter: Float
    public var sharpeningTax: Float
    public var passAtKDiversity: Float
    public var sharpeningGuardTriggered: Bool
    public var earlyStoppedStep: Int?
    public var emaApplied: Bool
    public var emaUpdates: Int
}

public struct EMAWeightShadow {
    public let decay: Float
    public private(set) var updates: Int = 0
    private var shadow: [(String, MLXArray)]
    private let dtypes: [String: DType]

    public init(decay: Float, model: Module) {
        self.decay = decay
        let flat = model.trainableParameters().flattened()
        var types: [String: DType] = [:]
        var initial: [(String, MLXArray)] = []
        initial.reserveCapacity(flat.count)
        for (key, array) in flat {
            types[key] = array.dtype
            initial.append((key, array.asType(.float32) + MLXArray(Float(0))))
        }
        if !initial.isEmpty { eval(initial.map(\.1)) }
        self.shadow = initial
        self.dtypes = types
    }

    public mutating func update(from model: Module) {
        let current = Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened())
        let mu = MLXArray(decay)
        let oneMinusMu = MLXArray(1.0 - decay)
        var next: [(String, MLXArray)] = []
        next.reserveCapacity(shadow.count)
        for (key, prev) in shadow {
            guard let live = current[key] else {
                next.append((key, prev))
                continue
            }
            next.append((key, mu * prev + oneMinusMu * live.asType(.float32)))
        }
        if !next.isEmpty { eval(next.map(\.1)) }
        shadow = next
        updates += 1
    }

    public func values(for key: String) -> [Float]? {
        guard let match = shadow.first(where: { $0.0 == key })?.1 else { return nil }
        eval(match)
        return match.asArray(Float.self)
    }

    public func apply(to model: Module) {
        guard !shadow.isEmpty else { return }
        let casted: [(String, MLXArray)] = shadow.map { key, array in
            let dtype = dtypes[key] ?? .float32
            return (key, (array + MLXArray(Float(0))).asType(dtype))
        }
        eval(casted.map(\.1))
        model.update(parameters: .unflattened(casted))
        eval(model.trainableParameters().flattened().map(\.1))
        model.invalidateCompiledTraces()
    }

    public func withShadowWeights<R>(in model: Module, _ body: () throws -> R) rethrows -> R {
        let saved = ADPOTrainer.cloneTrainable(model)
        apply(to: model)
        defer { ADPOTrainer.restoreTrainable(saved, into: model) }
        return try body()
    }
}

struct ADPOSegmentMasks: Sendable {
    let chosenSum: [Float]
    let rejectedSum: [Float]
    let rejectedMean: [Float]
    let count: Int
}

struct PreparedPreferencePair: Sendable {
    let pair: TokenizedPreferencePair
    let ref: ADPOReferenceLogProbs
    let masks: ADPOSegmentMasks
}

public enum ADPOTrainer {
    static func cloneTrainable(_ model: Module) -> [(String, MLXArray)] {
        let flat = model.trainableParameters().flattened()
        let saved: [(String, MLXArray)] = flat.map { key, array in
            (key, (array + MLXArray(0, dtype: array.dtype)).asType(array.dtype))
        }
        if !saved.isEmpty { eval(saved.map(\.1)) }
        return saved
    }

    static func restoreTrainable(_ saved: [(String, MLXArray)], into model: Module) {
        guard !saved.isEmpty else { return }
        let restored: [(String, MLXArray)] = saved.map { key, array in
            (key, (array + MLXArray(0, dtype: array.dtype)).asType(array.dtype))
        }
        eval(restored.map(\.1))
        model.update(parameters: .unflattened(restored))
        eval(model.trainableParameters().flattened().map(\.1))
        model.invalidateCompiledTraces()
    }

    public static func withReferencePolicy<R>(in model: Module, _ body: () throws -> R) rethrows -> R {
        let saved = cloneTrainable(model)
        let toggleStates: [(LoRALayer, Bool)] = model.namedModules().compactMap { _, module in
            (module as? LoRALayer).map { ($0, $0.loraEnabled) }
        }
        let zeroed: [(String, MLXArray)] = saved.compactMap { key, array in
            guard !key.hasSuffix(".m") else { return nil }
            return (key, MLXArray.zeros(array.shape, dtype: array.dtype))
        }
        if !zeroed.isEmpty {
            model.update(parameters: .unflattened(zeroed))
            eval(model.trainableParameters().flattened().map(\.1))
        }
        for (layer, _) in toggleStates { layer.loraEnabled = false }
        model.invalidateCompiledTraces()
        defer {
            restoreTrainable(saved, into: model)
            for (layer, wasEnabled) in toggleStates { layer.loraEnabled = wasEnabled }
            model.invalidateCompiledTraces()
        }
        return try body()
    }

    public static func referenceLogProbs(
        for pair: TokenizedPreferencePair,
        in model: any LanguageModel
    ) -> ADPOReferenceLogProbs {
        withReferencePolicy(in: model) {
            let cw = tokenLogProbs(model: model, promptIds: pair.promptIds, responseIds: pair.chosenIds)
            let cl = tokenLogProbs(model: model, promptIds: pair.promptIds, responseIds: pair.rejectedIds)
            eval(cw, cl)
            return ADPOReferenceLogProbs(chosen: cw.asArray(Float.self), rejected: cl.asArray(Float.self))
        }
    }

    public static func negLogSigmoid(_ u: MLXArray) -> MLXArray {
        logAddExp(MLXArray(Float(0)), -u)
    }

    public static func standardDPOLoss(
        chosenLogProbs: MLXArray,
        rejectedLogProbs: MLXArray,
        refChosenLogProbs: MLXArray,
        refRejectedLogProbs: MLXArray,
        beta: Float
    ) -> MLXArray {
        let deltaW = chosenLogProbs.sum() - stopGradient(refChosenLogProbs).sum()
        let deltaL = rejectedLogProbs.sum() - stopGradient(refRejectedLogProbs).sum()
        return negLogSigmoid(MLXArray(beta) * (deltaW - deltaL))
    }

    public static func segmentLoss(
        chosenLogProbs: MLXArray,
        rejectedLogProbs: MLXArray,
        refChosenLogProbs: MLXArray,
        refRejectedLogProbs: MLXArray,
        config: ADPOConfig
    ) -> (loss: MLXArray, margin: MLXArray) {
        let requested = config.objective == .dpo ? 1 : config.segments
        let cwLen = chosenLogProbs.dim(0)
        let clLen = rejectedLogProbs.dim(0)
        let (cwRanges, clRanges) = ADPOSegments.aligned(
            chosenLength: cwLen,
            rejectedLength: clLen,
            segments: requested
        )
        guard !cwRanges.isEmpty else {
            let zero = MLXArray(Float(0))
            return (zero, zero)
        }
        let (cwSum, _) = ADPOSegments.indicatorMatrices(ranges: cwRanges, length: cwLen)
        let (clSum, clMean) = ADPOSegments.indicatorMatrices(ranges: clRanges, length: clLen)
        return segmentLossWithMasks(
            chosenLogProbs: chosenLogProbs,
            rejectedLogProbs: rejectedLogProbs,
            refChosenLogProbs: refChosenLogProbs,
            refRejectedLogProbs: refRejectedLogProbs,
            masks: ADPOSegmentMasks(chosenSum: cwSum, rejectedSum: clSum, rejectedMean: clMean, count: cwRanges.count),
            config: config
        )
    }

    static func segmentLossWithMasks(
        chosenLogProbs: MLXArray,
        rejectedLogProbs: MLXArray,
        refChosenLogProbs: MLXArray,
        refRejectedLogProbs: MLXArray,
        masks: ADPOSegmentMasks,
        config: ADPOConfig
    ) -> (loss: MLXArray, margin: MLXArray) {
        let count = masks.count
        let cwLen = chosenLogProbs.dim(0)
        let clLen = rejectedLogProbs.dim(0)
        let mW = MLXArray(masks.chosenSum, [count, cwLen])
        let mL = MLXArray(masks.rejectedSum, [count, clLen])
        let refW = stopGradient(refChosenLogProbs)
        let refL = stopGradient(refRejectedLogProbs)
        let segPiW = matmul(mW, chosenLogProbs.reshaped([cwLen, 1])).reshaped([count])
        let segRefW = matmul(mW, refW.reshaped([cwLen, 1])).reshaped([count])
        let segPiL = matmul(mL, rejectedLogProbs.reshaped([clLen, 1])).reshaped([count])
        let segRefL = matmul(mL, refL.reshaped([clLen, 1])).reshaped([count])
        let deltaW = segPiW - segRefW
        let deltaL = segPiL - segRefL
        let beta = MLXArray(config.beta)
        let margins: MLXArray
        if config.objective == .cadpo {
            let meanL = MLXArray(masks.rejectedMean, [count, clLen])
            let ratio = exp(stopGradient(rejectedLogProbs - refL))
            let weights = clip(ratio, min: config.minImportanceWeight, max: Float(1.0)) * MLXArray(config.conservatism)
            let sBar = matmul(meanL, weights.reshaped([clLen, 1])).reshaped([count])
            margins = beta * deltaW - beta * sBar * deltaL
        } else {
            margins = beta * (deltaW - deltaL)
        }
        return (negLogSigmoid(margins).mean(), margins.mean())
    }
}

extension ADPOTrainer {
    public static func tokenLogProbs(
        model: any LanguageModel,
        promptIds: [Int],
        responseIds: [Int]
    ) -> MLXArray {
        let pLen = promptIds.count
        let rLen = responseIds.count
        let full = promptIds + responseIds
        let inputs = MLXArray(full.dropLast().map(Int32.init)).reshaped([1, pLen + rLen - 1])
        let logits = model(inputs, cache: nil).asType(.float32)
        let responseLogits = logits[0..., (pLen - 1) ..< (pLen - 1 + rLen), 0...].reshaped([rLen, logits.dim(-1)])
        let logProbs = responseLogits - logSumExp(responseLogits, axis: -1, keepDims: true)
        let targets = MLXArray(responseIds.map(Int32.init)).reshaped([rLen, 1])
        return takeAlong(logProbs, targets, axis: -1).reshaped([rLen])
    }

    public static func policyEntropy(
        model: any LanguageModel,
        pairs: [TokenizedPreferencePair]
    ) -> Float {
        guard !pairs.isEmpty else { return 0 }
        var total: Float = 0
        for pair in pairs {
            let pLen = pair.promptIds.count
            let rLen = pair.chosenIds.count
            let full = pair.promptIds + pair.chosenIds
            let inputs = MLXArray(full.dropLast().map(Int32.init)).reshaped([1, pLen + rLen - 1])
            let logits = model(inputs, cache: nil).asType(.float32)
            let responseLogits = logits[0..., (pLen - 1) ..< (pLen - 1 + rLen), 0...].reshaped([rLen, logits.dim(-1)])
            let logProbs = responseLogits - logSumExp(responseLogits, axis: -1, keepDims: true)
            let probs = exp(logProbs)
            let safeLogProbs = maximum(logProbs, MLXArray(Float(-100.0)))
            let meanEnt = (-(probs * safeLogProbs).sum(axis: -1)).mean()
            eval(meanEnt)
            total += meanEnt.item(Float.self)
        }
        return total / Float(pairs.count)
    }

    public static func measurePassAtKDiversity(
        model: any LanguageModel,
        prompts: [[Int]],
        samples: Int,
        stops: Set<Int> = []
    ) throws -> Float {
        guard samples >= 2, !prompts.isEmpty else { return 0 }
        var allSamples: [[[Int]]] = []
        for (pIndex, prompt) in prompts.prefix(4).enumerated() where !prompt.isEmpty {
            var promptDraws: [[Int]] = []
            for draw in 0 ..< samples {
                try Task.checkCancellation()
                let seed = UInt64(0x5710_0000) ^ (UInt64(pIndex) << 16) ^ UInt64(draw)
                let generated = try withRandomState(MLXRandom.RandomState(seed: seed)) {
                    let params = GenerateParameters(maxTokens: 12, temperature: 0.8, seed: seed)
                    var iterator = try TokenIterator(
                        input: LMInput(tokens: MLXArray(prompt.map(Int32.init))),
                        model: model,
                        parameters: params
                    )
                    var ids: [Int] = []
                    while let token = iterator.next() {
                        ids.append(token)
                        if stops.contains(token) { break }
                    }
                    return ids
                }
                promptDraws.append(generated)
            }
            allSamples.append(promptDraws)
        }
        return ADPOMetrics.passAtKDiversity(allSamples)
    }

    static func prepare(
        _ pairs: [TokenizedPreferencePair],
        in model: any LanguageModel,
        config: ADPOConfig
    ) -> [PreparedPreferencePair] {
        let requested = config.objective == .dpo ? 1 : config.segments
        return withReferencePolicy(in: model) {
            pairs.map { pair in
                let cw = tokenLogProbs(model: model, promptIds: pair.promptIds, responseIds: pair.chosenIds)
                let cl = tokenLogProbs(model: model, promptIds: pair.promptIds, responseIds: pair.rejectedIds)
                eval(cw, cl)
                let ref = ADPOReferenceLogProbs(chosen: cw.asArray(Float.self), rejected: cl.asArray(Float.self))
                let (cwRanges, clRanges) = ADPOSegments.aligned(
                    chosenLength: pair.chosenIds.count,
                    rejectedLength: pair.rejectedIds.count,
                    segments: requested
                )
                let (cwSum, _) = ADPOSegments.indicatorMatrices(ranges: cwRanges, length: pair.chosenIds.count)
                let (clSum, clMean) = ADPOSegments.indicatorMatrices(ranges: clRanges, length: pair.rejectedIds.count)
                let masks = ADPOSegmentMasks(chosenSum: cwSum, rejectedSum: clSum, rejectedMean: clMean, count: cwRanges.count)
                return PreparedPreferencePair(pair: pair, ref: ref, masks: masks)
            }
        }
    }

    static func meanPreparedMargin(
        model: any LanguageModel,
        prepared: [PreparedPreferencePair],
        config: ADPOConfig
    ) -> Float {
        guard !prepared.isEmpty else { return 0 }
        var sum: Float = 0
        for item in prepared {
            let cw = tokenLogProbs(model: model, promptIds: item.pair.promptIds, responseIds: item.pair.chosenIds)
            let cl = tokenLogProbs(model: model, promptIds: item.pair.promptIds, responseIds: item.pair.rejectedIds)
            let (_, margin) = segmentLossWithMasks(
                chosenLogProbs: cw,
                rejectedLogProbs: cl,
                refChosenLogProbs: MLXArray(item.ref.chosen),
                refRejectedLogProbs: MLXArray(item.ref.rejected),
                masks: item.masks,
                config: config
            )
            eval(margin)
            sum += margin.item(Float.self)
        }
        return sum / Float(prepared.count)
    }
}

extension ADPOTrainer {
    public static func optimize(
        model: any LanguageModel,
        pairs: [TokenizedPreferencePair],
        config: ADPOConfig,
        options: ADPOOptimizeOptions,
        stops: Set<Int> = []
    ) throws -> ADPOTrainOutcome {
        let prepared = prepare(pairs, in: model, config: config)
        let evalSubset = Array(pairs.prefix(min(2, pairs.count)))
        let entropyBefore = policyEntropy(model: model, pairs: evalSubset)
        let marginBefore = meanPreparedMargin(model: model, prepared: prepared, config: config)
        let effectiveEMA = options.emaDecay ?? config.emaDecay
        let loop = try runSteps(
            model: model,
            prepared: prepared,
            evalSubset: evalSubset,
            entropyBefore: entropyBefore,
            config: config,
            options: ADPOOptimizeOptions(
                iterations: options.iterations,
                learningRate: options.learningRate,
                batchSize: options.batchSize,
                emaDecay: effectiveEMA,
                optimizer: options.optimizer
            )
        )
        var emaApplied = false
        if let ema = loop.ema, config.swapEMAOnFinish, ema.updates > 0 {
            ema.apply(to: model)
            emaApplied = true
        }
        let marginAfter = meanPreparedMargin(model: model, prepared: prepared, config: config)
        let entropyAfter = policyEntropy(model: model, pairs: evalSubset)
        let tax = ADPOMetrics.sharpeningTax(before: entropyBefore, after: entropyAfter)
        let diversity = try measurePassAtKDiversity(
            model: model,
            prompts: evalSubset.map(\.promptIds),
            samples: config.diversitySamples,
            stops: stops
        )
        return ADPOTrainOutcome(
            losses: loop.losses,
            margins: loop.margins,
            marginBefore: marginBefore,
            marginAfter: marginAfter,
            entropyBefore: entropyBefore,
            entropyAfter: entropyAfter,
            sharpeningTax: tax,
            passAtKDiversity: diversity,
            sharpeningGuardTriggered: loop.guardTriggered,
            earlyStoppedStep: loop.earlyStep,
            emaApplied: emaApplied,
            emaUpdates: loop.ema?.updates ?? 0
        )
    }

    private static func batchObjective(
        model: any LanguageModel,
        batch: [PreparedPreferencePair],
        config: ADPOConfig
    ) -> [MLXArray] {
        var pairLosses: [MLXArray] = []
        var pairMargins: [MLXArray] = []
        pairLosses.reserveCapacity(batch.count)
        pairMargins.reserveCapacity(batch.count)
        for item in batch {
            let cw = tokenLogProbs(model: model, promptIds: item.pair.promptIds, responseIds: item.pair.chosenIds)
            let cl = tokenLogProbs(model: model, promptIds: item.pair.promptIds, responseIds: item.pair.rejectedIds)
            let (loss, margin) = segmentLossWithMasks(
                chosenLogProbs: cw,
                rejectedLogProbs: cl,
                refChosenLogProbs: MLXArray(item.ref.chosen),
                refRejectedLogProbs: MLXArray(item.ref.rejected),
                masks: item.masks,
                config: config
            )
            pairLosses.append(loss)
            pairMargins.append(stopGradient(margin))
        }
        return [stacked(pairLosses).mean(), stacked(pairMargins).mean()]
    }

    private static func runSteps(
        model: any LanguageModel,
        prepared: [PreparedPreferencePair],
        evalSubset: [TokenizedPreferencePair],
        entropyBefore: Float,
        config: ADPOConfig,
        options: ADPOOptimizeOptions
    ) throws -> (losses: [Float], margins: [Float], ema: EMAWeightShadow?, guardTriggered: Bool, earlyStep: Int?) {
        let optimizer = TrainingOptimizerFactory.make(options.optimizer, learningRate: options.learningRate)
        var ema = options.emaDecay.map { EMAWeightShadow(decay: $0, model: model) }
        let stepFn = valueAndGrad(model: model as Module) { _, batch in
            batchObjective(model: model, batch: batch, config: config)
        }
        let batchCount = min(options.batchSize, prepared.count)
        var losses: [Float] = []
        var margins: [Float] = []
        var guardTriggered = false
        var earlyStep: Int?
        for step in 0 ..< options.iterations {
            try Task.checkCancellation()
            let safeWeights = cloneTrainable(model)
            let batch = (0 ..< batchCount).map { prepared[(step * batchCount + $0) % prepared.count] }
            let (values, gradients) = stepFn(model as Module, batch)
            optimizer.update(model: model as Module, gradients: gradients)
            eval(model as Module, optimizer, values)
            let loss = values[0].item(Float.self)
            let margin = values[1].item(Float.self)
            guard loss.isFinite, margin.isFinite else {
                restoreTrainable(safeWeights, into: model)
                throw BasaltError.inferenceError("train: preference optimization loss became non-finite at step \(step)")
            }
            losses.append(loss)
            margins.append(margin)
            let currentEntropy = policyEntropy(model: model, pairs: evalSubset)
            if ADPOMetrics.isEntropyCollapsed(before: entropyBefore, current: currentEntropy, config: config) {
                restoreTrainable(safeWeights, into: model)
                guardTriggered = true
                earlyStep = step + 1
                break
            }
            ema?.update(from: model)
        }
        return (losses, margins, ema, guardTriggered, earlyStep)
    }
}
