import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXOptimizers

public struct DiLoCoRunReport: Sendable, Equatable {
    public let outerRounds: Int
    public let innerStepsPerRound: Int
    public let initialLoss: Float
    public let finalLoss: Float
    public let compressionRatio: Double
}

public struct DiLoCoScheduleSpec: Sendable, Equatable {
    public var outerRounds: Int
    public var outerConfig: DiLoCoOuterConfig
    public var innerLearningRate: Float
    public var powerSGDRank: Int

    public init(
        outerRounds: Int = 4,
        outerConfig: DiLoCoOuterConfig = DiLoCoOuterConfig(innerSteps: 8, outerLearningRate: 0.7, nesterovMomentum: 0.9),
        innerLearningRate: Float = 5e-3,
        powerSGDRank: Int = 2
    ) {
        self.outerRounds = max(1, outerRounds)
        self.outerConfig = outerConfig
        self.innerLearningRate = innerLearningRate
        self.powerSGDRank = max(1, powerSGDRank)
    }
}

public enum DiLoCoTrainer {
    public static func runSimulatedCluster(
        workerModels: [any LanguageModel],
        workerBatches: [[[[Int32]]]],
        evalBatch: [[Int32]],
        schedule: DiLoCoScheduleSpec = DiLoCoScheduleSpec()
    ) throws -> DiLoCoRunReport {
        guard !workerModels.isEmpty, workerModels.count == workerBatches.count else {
            throw BasaltError.invalidRequestStructure(details: "diloco: workerModels and workerBatches must match and be non-empty")
        }
        let config = schedule.outerConfig
        let initLoss = evaluateMeanCE(model: workerModels[0], batch: evalBatch)
        var globalParams = extractTrainableMap(from: workerModels[0])
        let shapes = extractShapes(from: workerModels[0])
        for wIdx in 1 ..< workerModels.count {
            applyTrainableMap(globalParams, shapes: shapes, to: workerModels[wIdx])
        }
        var outerOpt = DiLoCoOuterOptimizer(config: config)
        var efStates: [[String: EF21ResidualState]] = Array(repeating: [:], count: workerModels.count)
        var rawFloatCount = 0
        var transmittedFloats = 0

        for round in 0 ..< schedule.outerRounds {
            var workerDeltas: [(tokenCount: Int, delta: [String: [Float]])] = []
            for wIdx in workerModels.indices {
                let res = runInnerWorkerRound(
                    model: workerModels[wIdx],
                    batches: workerBatches[wIdx],
                    round: round,
                    globalParams: globalParams,
                    shapes: shapes,
                    schedule: schedule
                )
                var compressedDelta: [String: [Float]] = [:]
                for (key, rawDelta) in res.rawDeltas {
                    guard let shape = shapes[key] else { continue }
                    rawFloatCount += rawDelta.count
                    let (rows, cols) = factor2D(shape: shape, total: rawDelta.count)
                    var st = efStates[wIdx][key] ?? EF21ResidualState(dimension: rawDelta.count)
                    let stepped = st.stepPowerSGD(target: rawDelta, rows: rows, cols: cols, rank: schedule.powerSGDRank)
                    efStates[wIdx][key] = st
                    transmittedFloats += stepped.packet.wireFloats
                    compressedDelta[key] = stepped.estimate
                }
                workerDeltas.append((res.tokensUsed, compressedDelta))
            }
            globalParams = outerOpt.step(globalParameters: globalParams, workerPseudoGradients: workerDeltas)
        }
        for model in workerModels {
            applyTrainableMap(globalParams, shapes: shapes, to: model)
        }
        let finalLoss = evaluateMeanCE(model: workerModels[0], batch: evalBatch)
        let ratio = transmittedFloats > 0 ? Double(rawFloatCount) / Double(transmittedFloats) : 1.0
        return DiLoCoRunReport(
            outerRounds: schedule.outerRounds,
            innerStepsPerRound: config.innerSteps,
            initialLoss: initLoss,
            finalLoss: finalLoss,
            compressionRatio: ratio
        )
    }

    private static func runInnerWorkerRound(
        model: any LanguageModel,
        batches: [[[Int32]]],
        round: Int,
        globalParams: [String: [Float]],
        shapes: [String: [Int]],
        schedule: DiLoCoScheduleSpec
    ) -> (tokensUsed: Int, rawDeltas: [String: [Float]]) {
        applyTrainableMap(globalParams, shapes: shapes, to: model)
        let innerOpt = AdamW(learningRate: schedule.innerLearningRate, weightDecay: 0.01)
        var tokensUsed = 0
        let innerSteps = schedule.outerConfig.innerSteps
        for step in 0 ..< innerSteps {
            let curBatch = batches[(round * innerSteps + step) % batches.count]
            tokensUsed += curBatch.reduce(0) { $0 + $1.count }
            trainSingleStep(model: model, optimizer: innerOpt, samples: curBatch)
        }
        let localTheta = extractTrainableMap(from: model)
        var rawDeltas: [String: [Float]] = [:]
        for (key, gTheta) in globalParams {
            if let lTheta = localTheta[key] {
                rawDeltas[key] = zip(gTheta, lTheta).map { $0 - $1 }
            }
        }
        return (tokensUsed, rawDeltas)
    }

    private static func extractShapes(from model: any LanguageModel) -> [String: [Int]] {
        Dictionary(uniqueKeysWithValues: (model as Module).trainableParameters().flattened().map { ($0.0, $0.1.shape) })
    }

    private static func extractTrainableMap(from model: any LanguageModel) -> [String: [Float]] {
        let flat = (model as Module).trainableParameters().flattened()
        eval(flat.map(\.1))
        return Dictionary(uniqueKeysWithValues: flat.map { key, arr in
            (key, arr.asType(.float32).reshaped(-1).asArray(Float.self))
        })
    }

    private static func applyTrainableMap(
        _ params: [String: [Float]],
        shapes: [String: [Int]],
        to model: any LanguageModel
    ) {
        let pairs = params.compactMap { key, vals -> (String, MLXArray)? in
            guard let shape = shapes[key] else { return nil }
            return (key, MLXArray(vals, shape))
        }
        (model as Module).update(parameters: ModuleParameters.unflattened(pairs))
        eval(pairs.map(\.1))
    }

    private static func trainSingleStep(
        model: any LanguageModel,
        optimizer: AdamW,
        samples: [[Int32]]
    ) {
        let root = model as Module
        let b = samples.count
        let seq = samples[0].count
        let arr = MLXArray(samples.flatMap { $0 }, [b, seq])
        let inputs = arr[0..., ..<(seq - 1)]
        let targets = arr[0..., 1...]
        let vg = valueAndGrad(model: root) { (mod: Module, x: MLXArray, y: MLXArray) -> MLXArray in
            let logits = (mod as! any LanguageModel)(LMInput.Text(tokens: x), cache: nil, state: nil).logits
            return crossEntropy(logits: logits, targets: y, reduction: .mean)
        }
        let (loss, grads) = vg(root, inputs, targets)
        optimizer.update(model: root, gradients: grads)
        eval(root, optimizer, loss)
    }

    private static func evaluateMeanCE(model: any LanguageModel, batch: [[Int32]]) -> Float {
        let b = batch.count
        let seq = batch[0].count
        let arr = MLXArray(batch.flatMap { $0 }, [b, seq])
        let logits = model(LMInput.Text(tokens: arr[0..., ..<(seq - 1)]), cache: nil, state: nil).logits
        let loss = crossEntropy(logits: logits, targets: arr[0..., 1...], reduction: .mean)
        eval(loss)
        return loss.item(Float.self)
    }

    private static func factor2D(shape: [Int], total: Int) -> (Int, Int) {
        if shape.count == 2, shape[0] * shape[1] == total {
            return (shape[0], shape[1])
        }
        var r = Int(Double(total).squareRoot())
        while r > 1 && !total.isMultiple(of: r) { r -= 1 }
        return (r, max(1, total / r))
    }
}
