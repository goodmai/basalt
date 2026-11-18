import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXOptimizers
import BasaltCore

final class BankLinearHead: Module {
    @ModuleInfo var linear: Linear

    init(inputDim: Int, outputDim: Int) {
        self._linear.wrappedValue = Linear(
            weight: MLXArray.zeros([outputDim, inputDim]),
            bias: MLXArray.zeros([outputDim])
        )
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        linear(hidden)
    }
}

public struct MultiHeadBankDataset: Sendable, Equatable {
    public var sampleCount: Int
    public var hiddenSize: Int
    public var totalOutputs: Int
    public var hiddenStates: [Float]
    public var targets: [Float]

    public init(
        sampleCount: Int,
        hiddenSize: Int,
        totalOutputs: Int,
        hiddenStates: [Float],
        targets: [Float]
    ) {
        self.sampleCount = sampleCount
        self.hiddenSize = hiddenSize
        self.totalOutputs = totalOutputs
        self.hiddenStates = hiddenStates
        self.targets = targets
    }
}

public struct MultiHeadBankTrainConfig: Sendable, Equatable {
    public var iterations: Int
    public var batchSize: Int
    public var learningRate: Float
    public var weightDecay: Float
    public var brierWeight: Float
    public var seed: UInt64

    public init(
        iterations: Int = 200,
        batchSize: Int = 64,
        learningRate: Float = 0.05,
        weightDecay: Float = 1e-4,
        brierWeight: Float = 0.1,
        seed: UInt64 = 42
    ) {
        self.iterations = iterations
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.brierWeight = brierWeight
        self.seed = seed
    }
}

public struct MultiHeadBankTrainReport: Sendable, Equatable {
    public var losses: [Float]
    public var initialLoss: Float
    public var finalLoss: Float
    public var weight: [Float]
    public var bias: [Float]

    public init(losses: [Float], initialLoss: Float, finalLoss: Float, weight: [Float], bias: [Float]) {
        self.losses = losses
        self.initialLoss = initialLoss
        self.finalLoss = finalLoss
        self.weight = weight
        self.bias = bias
    }
}

public enum MultiHeadBankTrainer {
    public static func score64Target(score: Double, sigma: Double = 0.35) -> [Float] {
        let clamped = min(max(score, 0.0), BankSlot.scoreMax)
        let width = max(sigma, 1e-3)
        let denom = Double(BankSlot.scoreBins - 1)
        var unnormalized = [Double](repeating: 0.0, count: BankSlot.scoreBins)
        var top = -Double.infinity
        for index in 0 ..< BankSlot.scoreBins {
            let binValue = BankSlot.scoreMax * Double(index) / denom
            let diff = (binValue - clamped) / width
            let logWeight = -0.5 * diff * diff
            unnormalized[index] = logWeight
            if logWeight > top { top = logWeight }
        }
        var weights = [Double](repeating: 0.0, count: BankSlot.scoreBins)
        var sum = 0.0
        for index in 0 ..< BankSlot.scoreBins {
            let w = Foundation::exp(unnormalized[index] - top)
            weights[index] = w
            sum += w
        }
        guard sum > 0 else { return [Float](repeating: Float(1.0 / Double(BankSlot.scoreBins)), count: BankSlot.scoreBins) }
        return weights.map { Float($0 / sum) }
    }

    public static func extractHiddenStates(
        states: [String],
        container: ModelContainer
    ) async throws(BasaltError) -> (hiddenStates: [Float], hiddenSize: Int) {
        guard !states.isEmpty else {
            throw .invalidRequestStructure(details: "bank_train: states cannot be empty")
        }
        do {
            return try await container.performEvaluating(nonSendable: ()) { context, _ in
                var flat: [Float] = []
                var dim = 0
                for state in states {
                    try Task.checkCancellation()
                    let tokens = MultiHeadDecisionEngine.encodeState(state, tokenizer: context.tokenizer)
                    let hidden = try MultiHeadDecisionEngine.lastHidden(context.model, tokens: tokens)
                    eval(hidden)
                    if dim == 0 {
                        dim = hidden.dim(0)
                        flat.reserveCapacity(states.count * dim)
                    }
                    flat.append(contentsOf: hidden.asArray(Float.self))
                }
                return (flat, dim)
            }
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .inferenceError("bank_train: \(error.localizedDescription)")
        }
    }

    public static func train(
        manifest: SystemOneBankManifest,
        dataset: MultiHeadBankDataset,
        config: MultiHeadBankTrainConfig = .init(),
        saveTo directory: URL? = nil
    ) throws(BasaltError) -> MultiHeadBankTrainReport {
        try validate(manifest: manifest, dataset: dataset, config: config)
        let slots = manifest.slots.sorted { $0.key < $1.key }.map(\.value)
        let head = BankLinearHead(inputDim: manifest.hiddenSize, outputDim: manifest.totalOutputs)
        head.train()
        let fullHidden = MLXArray(dataset.hiddenStates, [dataset.sampleCount, manifest.hiddenSize])
        let fullTargets = MLXArray(dataset.targets, [dataset.sampleCount, manifest.totalOutputs])
        let initial = lossValue(head: head, hidden: fullHidden, targets: fullTargets, slots: slots, brierWeight: config.brierWeight)
        eval(initial)
        let initialScalar = initial.item(Float.self)
        guard initialScalar.isFinite else {
            throw .inferenceError("bank_train: initial loss is non-finite")
        }
        let losses = try runSteps(
            head: head,
            hidden: fullHidden,
            targets: fullTargets,
            slots: slots,
            sampleCount: dataset.sampleCount,
            config: config
        )
        let finalEval = lossValue(head: head, hidden: fullHidden, targets: fullTargets, slots: slots, brierWeight: config.brierWeight)
        let wTensor = head.linear.weight.asType(.float32).reshaped(-1)
        let bTensor = (head.linear.bias ?? MLXArray.zeros([manifest.totalOutputs])).asType(.float32).reshaped(-1)
        eval(finalEval, wTensor, bTensor)
        let finalScalar = finalEval.item(Float.self)
        guard finalScalar.isFinite else {
            throw .inferenceError("bank_train: final loss is non-finite")
        }
        let trainedWeight = wTensor.asArray(Float.self)
        let trainedBias = bTensor.asArray(Float.self)
        if let directory {
            try withTaskCancellationShield { () throws(BasaltError) in
                try SystemOneBankBundle.save(manifest: manifest, weight: trainedWeight, bias: trainedBias, to: directory)
            }
        }
        return MultiHeadBankTrainReport(
            losses: losses,
            initialLoss: initialScalar,
            finalLoss: finalScalar,
            weight: trainedWeight,
            bias: trainedBias
        )
    }
}

extension MultiHeadBankTrainer {
    static func validate(
        manifest: SystemOneBankManifest,
        dataset: MultiHeadBankDataset,
        config: MultiHeadBankTrainConfig
    ) throws(BasaltError) {
        try manifest.validate(path: "bank_train")
        guard dataset.sampleCount > 0 else {
            throw .invalidRequestStructure(details: "bank_train: sampleCount must be positive")
        }
        guard dataset.hiddenSize == manifest.hiddenSize, dataset.totalOutputs == manifest.totalOutputs else {
            throw .invalidRequestStructure(details: "bank_train: dataset dimensions do not match bank_manifest")
        }
        guard dataset.hiddenStates.count == dataset.sampleCount * manifest.hiddenSize else {
            throw .invalidRequestStructure(details: "bank_train: hiddenStates count mismatch")
        }
        guard dataset.targets.count == dataset.sampleCount * manifest.totalOutputs else {
            throw .invalidRequestStructure(details: "bank_train: targets count mismatch")
        }
        guard config.iterations > 0, config.batchSize > 0 else {
            throw .invalidRequestStructure(details: "bank_train: iterations and batchSize must be positive")
        }
        guard config.learningRate.isFinite, config.learningRate > 0,
              config.weightDecay.isFinite, config.weightDecay >= 0,
              config.brierWeight.isFinite, config.brierWeight >= 0 else {
            throw .invalidRequestStructure(details: "bank_train: invalid optimizer hyperparameters")
        }
    }

    static func runSteps(
        head: BankLinearHead,
        hidden: MLXArray,
        targets: MLXArray,
        slots: [BankSlot],
        sampleCount: Int,
        config: MultiHeadBankTrainConfig
    ) throws(BasaltError) -> [Float] {
        let optimizer = AdamW(learningRate: config.learningRate, weightDecay: config.weightDecay)
        let brier = config.brierWeight
        let stepFn = valueAndGrad(model: head) { model, arrays in
            [lossValue(head: model, hidden: arrays[0], targets: arrays[1], slots: slots, brierWeight: brier)]
        }
        let batchCount = min(config.batchSize, sampleCount)
        var losses: [Float] = []
        losses.reserveCapacity(config.iterations)
        for step in 0 ..< config.iterations {
            if Task.isCancelled { throw .cancelled }
            var rng = SplitMix64(seed: config.seed &+ UInt64(step))
            let indices = Array(Array(0 ..< sampleCount).shuffled(using: &rng).prefix(batchCount)).map(Int32.init)
            let batchIdx = MLXArray(indices)
            let (values, grads) = stepFn(head, [hidden[batchIdx], targets[batchIdx]])
            optimizer.update(model: head, gradients: grads)
            eval(head, optimizer, values)
            let scalar = values[0].item(Float.self)
            guard scalar.isFinite else {
                throw .inferenceError("bank_train: loss became non-finite at step \(step)")
            }
            losses.append(scalar)
        }
        return losses
    }

    static func lossValue(
        head: BankLinearHead,
        hidden: MLXArray,
        targets: MLXArray,
        slots: [BankSlot],
        brierWeight: Float
    ) -> MLXArray {
        let logits = head(hidden)
        var total = MLXArray(Float(0))
        let brierScale = MLXArray(brierWeight)
        for slot in slots {
            let z = logits[0..., slot.slice.lowerBound ..< slot.slice.upperBound]
            let y = targets[0..., slot.slice.lowerBound ..< slot.slice.upperBound]
            if slot.kind == .noul && slot.slice.count == 1 {
                let eps = MLXArray(Float(1e-7))
                let one = MLXArray(Float(1))
                let bce = MLX::maximum(z, MLXArray(Float(0))) - z * y + MLX::log(one + MLX::exp(-MLX::abs(z)))
                let ent = -(y * MLX::log(MLX::maximum(y, eps)) + (one - y) * MLX::log(MLX::maximum(one - y, eps)))
                let prob = MLX.sigmoid(z)
                let brier = MLX::square(prob - y).mean()
                total = total + (bce - ent).mean() + brierScale * brier
            } else {
                let eps = MLXArray(Float(1e-7))
                let logProbs = z - logSumExp(z, axis: -1, keepDims: true)
                let kl = (y * (MLX::log(MLX::maximum(y, eps)) - logProbs)).sum(axis: -1).mean()
                let probs = MLX::exp(logProbs)
                let brier = MLX::square(probs - y).sum(axis: -1).mean()
                total = total + kl + brierScale * brier
            }
        }
        return total / MLXArray(Float(max(1, slots.count)))
    }
}
