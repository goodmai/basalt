import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Heterogeneous DDP + ZeRO-2 Adapter Training & Rollouts (#566)", .serialized)
struct ZeRO2DistributedAdapterTests {
    private static func computeLoRALossAndGrads(
        model: any LanguageModel,
        samples: [[Int32]]
    ) -> (loss: Float, grads: [String: [Float]]) {
        let root = model as Module
        let count = samples.count
        let seqLen = samples[0].count
        let flat = samples.flatMap { $0 }
        let inputBatch = MLXArray(flat, [count, seqLen])
        let inputs = inputBatch[0..., ..<(seqLen - 1)]
        let targets = inputBatch[0..., 1...]
        let vg = valueAndGrad(model: root) { (mod: Module, x: MLXArray, y: MLXArray) -> MLXArray in
            let logits = (mod as! any LanguageModel)(LMInput.Text(tokens: x), cache: nil, state: nil).logits
            return crossEntropy(logits: logits, targets: y, reduction: .mean)
        }
        let (lossArr, gradTree) = vg(root, inputs, targets)
        var toEval: [MLXArray] = [lossArr]
        let flatGrads = gradTree.flattened()
        for (_, g) in flatGrads { toEval.append(g) }
        eval(toEval)
        var outGrads: [String: [Float]] = [:]
        for (key, g) in flatGrads {
            outGrads[key] = g.asType(.float32).reshaped(-1).asArray(Float.self)
        }
        return (lossArr.item(Float.self), outGrads)
    }

    private static func applyParameterSnapshot(
        _ snapshot: [String: [Float]],
        shapes: [String: [Int]],
        to model: any LanguageModel
    ) {
        let updates = snapshot.compactMap { key, vals -> (String, MLXArray)? in
            guard let shape = shapes[key] else { return nil }
            return (key, MLXArray(vals, shape))
        }
        (model as Module).update(parameters: ModuleParameters.unflattened(updates))
        eval(updates.map(\.1))
    }

    private static func syncedLoRATrio(
        seed: UInt64
    ) throws -> (single: any LanguageModel, rank0: any LanguageModel, rank1: any LanguageModel) {
        let singleModel = try TinyModel.make(.qwen3Small)
        let rank0Model = try TinyModel.make(.qwen3Small)
        let rank1Model = try TinyModel.make(.qwen3Small)
        try withRandomState(MLXRandom.RandomState(seed: seed)) {
            _ = try MLXInferenceEngine.attachLoRA(to: singleModel, layers: [0, 1], rank: 4)
            _ = try MLXInferenceEngine.attachLoRA(to: rank0Model, layers: [0, 1], rank: 4)
            _ = try MLXInferenceEngine.attachLoRA(to: rank1Model, layers: [0, 1], rank: 4)
        }
        let baseURL = FileManager.default.temporaryDirectory.appending(path: "zero2-init-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: baseURL) }
        eval(singleModel as Module)
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: (singleModel as Module).parameters().flattened()), url: baseURL)
        let syncedAll = try MLX.loadArrays(url: baseURL)
        (rank0Model as Module).update(parameters: ModuleParameters.unflattened(syncedAll))
        (rank1Model as Module).update(parameters: ModuleParameters.unflattened(syncedAll))
        eval(rank0Model as Module, rank1Model as Module)
        return (singleModel, rank0Model, rank1Model)
    }

    @Test("Single-node batch=8 and 2-rank heterogeneous (b0=5, b1=3) LoRA + ZeRO-2 match loss trajectory within 1e-4")
    func singleNodeVsTwoRankZeRO2LossParity() async throws {
        let (singleModel, rank0Model, rank1Model) = try Self.syncedLoRATrio(seed: 566)

        let initTrainable = (singleModel as Module).trainableParameters().flattened().sorted { $0.0 < $1.0 }
        eval(initTrainable.map(\.1))
        var shapesMap: [String: [Int]] = [:]
        let initialTriples = initTrainable.map { key, arr -> (key: String, shape: [Int], values: [Float]) in
            shapesMap[key] = arr.shape
            return (key, arr.shape, arr.asType(.float32).reshaped(-1).asArray(Float.self))
        }

        let cfg = ZeRO2AdamWConfig(learningRate: 5e-3, weightDecay: 0.01, emaDecay: 0.95)
        let singleCoord = DistributedZeRO2CoordinatorActor(initialParameters: initialTriples, worldSize: 1, config: cfg)
        let twoRankCoord = DistributedZeRO2CoordinatorActor(initialParameters: initialTriples, worldSize: 2, config: cfg)
        defer { await singleCoord.shutdown(); await twoRankCoord.shutdown() }

        let momentCounts = await twoRankCoord.perRankMomentElementCounts()
        let totalElems = initialTriples.reduce(0) { $0 + $1.values.count }
        #expect(momentCounts.count == 2)
        #expect(momentCounts.reduce(0, +) == totalElems)
        #expect(abs(momentCounts[0] - momentCounts[1]) <= 1)

        var dataset: [[Int32]] = []
        for row in 0 ..< 8 {
            var tokens: [Int32] = []
            for col in 0 ..< 12 {
                tokens.append(Int32((row * 17 + col * 7 + 3) % 240))
            }
            dataset.append(tokens)
        }

        for _ in 0 ..< 4 {
            let singleEval = Self.computeLoRALossAndGrads(model: singleModel, samples: dataset)
            let r0Eval = Self.computeLoRALossAndGrads(model: rank0Model, samples: Array(dataset[0 ..< 5]))
            let r1Eval = Self.computeLoRALossAndGrads(model: rank1Model, samples: Array(dataset[5 ..< 8]))
            let weightedClusterLoss = (5.0 * r0Eval.loss + 3.0 * r1Eval.loss) / 8.0
            #expect(abs(singleEval.loss - weightedClusterLoss) < 1e-4)

            let singleOut = await singleCoord.stepHeterogeneousBatch(rankGradients: [(8, singleEval.grads)])
            let clusterOut = await twoRankCoord.stepHeterogeneousBatch(rankGradients: [(5, r0Eval.grads), (3, r1Eval.grads)])
            #expect(clusterOut.emaWeights != nil)

            Self.applyParameterSnapshot(singleOut.weights, shapes: shapesMap, to: singleModel)
            Self.applyParameterSnapshot(clusterOut.weights, shapes: shapesMap, to: rank0Model)
            Self.applyParameterSnapshot(clusterOut.weights, shapes: shapesMap, to: rank1Model)
        }

        let finalSingle = Self.computeLoRALossAndGrads(model: singleModel, samples: dataset).loss
        let finalCluster = Self.computeLoRALossAndGrads(model: rank0Model, samples: dataset).loss
        #expect(abs(finalSingle - finalCluster) < 1e-4)
    }

    @Test("StragglerThermalPacer, QJL1GradientCodec error feedback, and ADPO/GRPO rollout collation uphold invariants")
    func pacerCompressionAndRollouts() {
        let balanced = StragglerThermalPacer.rebalanceMicrobatches(
            globalBatchSize: 12,
            telemetry: [
                PeerStepTelemetry(rank: 0, stepLatencyMs: 20.0, thermal: .nominal, baseWeight: 1.0),
                PeerStepTelemetry(rank: 1, stepLatencyMs: 20.0, thermal: .critical, baseWeight: 1.0),
            ]
        )
        #expect(balanced.reduce(0, +) == 12)
        #expect(balanced[0] > balanced[1] && balanced[1] >= 1)

        let trueGrad = (0 ..< 128).map { Float(sin(Double($0) * 0.17)) }
        var errBuf: [Float] = []
        var accumRecon = [Float](repeating: 0, count: 128)
        for step in 0 ..< 16 {
            let pkt = QJL1GradientCodec.compressWithFeedback(gradient: trueGrad, errorBuffer: &errBuf, seed: UInt64(100 + step))
            let dec = QJL1GradientCodec.decompress(pkt)
            for i in 0 ..< 128 { accumRecon[i] += dec[i] / 16.0 }
        }
        let mse = zip(trueGrad, accumRecon).reduce(Float(0)) { $0 + ($1.0 - $1.1) * ($1.0 - $1.1) } / 128.0
        #expect(mse < 0.02)

        let pairs = ZeRO2CollectiveCore.collateRolloutGroup([
            CandidateRolloutWire(rank: 0, promptID: "p1", tokenIDs: [10, 11, 12], reward: 0.9, sumLogProb: -1.4),
            CandidateRolloutWire(rank: 0, promptID: "p1", tokenIDs: [10, 15], reward: 0.4, sumLogProb: -2.1),
            CandidateRolloutWire(rank: 1, promptID: "p1", tokenIDs: [10, 99], reward: -0.2, sumLogProb: -3.5),
        ])
        #expect(pairs.count == 1)
        #expect(pairs[0].chosenTokens == [10, 11, 12])
        #expect(pairs[0].rejectedTokens == [10, 99])
        #expect(abs(pairs[0].rewardMargin - 1.1) < 1e-5)
    }

    @Test("ZeRO-2 reduce-scatter accumulates weighted heterogeneous rank gradients in Double and matches 1-rank vs 3-rank AdamW bit-identically")
    func exactWeightedReductionAndShardParity() async {
        let initial: [(key: String, shape: [Int], values: [Float])] = [
            ("w0", [2, 4], [0.25, -0.5, 0.75, -0.125, 0.3, -0.2, 0.4, -0.6]),
            ("w1", [5], [-0.4, 0.1, 0.8, -0.3, 0.2]),
        ]
        let cfg = ZeRO2AdamWConfig(learningRate: 5e-3, weightDecay: 0.01, emaDecay: 0.9)
        let single = DistributedZeRO2CoordinatorActor(initialParameters: initial, worldSize: 1, config: cfg)
        let threeRank = DistributedZeRO2CoordinatorActor(initialParameters: initial, worldSize: 3, config: cfg)
        defer {
            await single.shutdown()
            await threeRank.shutdown()
        }
        let g0 = ["w0": [Float](repeating: 0.3, count: 8), "w1": [Float](repeating: -0.7, count: 5)]
        let g1 = ["w0": [Float](repeating: -0.1, count: 8), "w1": [Float](repeating: 0.2, count: 5)]
        let g2 = ["w0": [Float](repeating: 0.5, count: 8), "w1": [Float](repeating: 0.4, count: 5)]
        let w0Expected = Float((2.0 * Double(Float(0.3)) + 3.0 * Double(Float(-0.1)) + 2.0 * Double(Float(0.5))) / 7.0)
        let w1Expected = Float((2.0 * Double(Float(-0.7)) + 3.0 * Double(Float(0.2)) + 2.0 * Double(Float(0.4))) / 7.0)
        let combined: [String: [Float]] = [
            "w0": [Float](repeating: w0Expected, count: 8),
            "w1": [Float](repeating: w1Expected, count: 5),
        ]
        for _ in 0 ..< 8 {
            let sOut = await single.stepHeterogeneousBatch(rankGradients: [(7, combined)])
            let mOut = await threeRank.stepHeterogeneousBatch(rankGradients: [(2, g0), (3, g1), (2, g2)])
            #expect(sOut.weights == mOut.weights)
            #expect(sOut.emaWeights == mOut.emaWeights)
        }
    }
}
