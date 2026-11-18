import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Layer-Split Zero-Bubble Pipeline (ZB-H1), Expert-Parallel MoE & Cluster Spec (#568)")
struct ZeroBubbleAndExpertParallelTests {
    private static func singleDeviceGradients(
        model: any LanguageModel,
        microbatches: [[[Int32]]]
    ) -> (meanLoss: Float, gradients: [String: [Float]]) {
        let root = model as Module
        let scaleMB = Float(1.0) / Float(microbatches.count)
        var totalLoss: Float = 0
        var accum: [String: MLXArray] = [:]
        for mb in microbatches {
            let b = mb.count
            let seq = mb[0].count
            let arr = MLXArray(mb.flatMap { $0 }, [b, seq])
            let inputs = arr[0..., ..<(seq - 1)]
            let targets = arr[0..., 1...]
            let vg = valueAndGrad(model: root) { (mod: Module, x: MLXArray, y: MLXArray) -> MLXArray in
                let logits = (mod as! any LanguageModel)(LMInput.Text(tokens: x), cache: nil, state: nil).logits
                return crossEntropy(logits: logits, targets: y, reduction: .mean) * scaleMB
            }
            let (lossArr, grads) = vg(root, inputs, targets)
            eval(lossArr)
            totalLoss += lossArr.item(Float.self)
            for (k, g) in grads.flattened() {
                accum[k] = accum[k].map { $0 + g } ?? g
            }
            eval(Array(accum.values))
        }
        var out: [String: [Float]] = [:]
        for (k, arr) in accum {
            out[k] = arr.asType(.float32).reshaped(-1).asArray(Float.self)
        }
        return (totalLoss, out)
    }

    @Test("2-stage ZB-H1 matches single-device LoRA gradients within 1e-4 on TinyModel qwen3Small and qwen3Moe")
    func twoStageZBH1GradientParity() throws {
        for family in [TinyModel.Family.qwen3Small, TinyModel.Family.qwen3Moe] {
            let single = try TinyModel.make(family)
            let piped = try TinyModel.make(family)
            let layerCount = family == .qwen3Small ? 2 : 4
            let targetLayers = Array(0 ..< layerCount)
            _ = try MLXInferenceEngine.attachLoRA(to: single, layers: targetLayers, rank: 4)
            _ = try MLXInferenceEngine.attachLoRA(to: piped, layers: targetLayers, rank: 4)

            let syncURL = FileManager.default.temporaryDirectory.appending(path: "zbh1-init-\(UUID().uuidString).safetensors")
            defer { try? FileManager.default.removeItem(at: syncURL) }
            eval(single as Module)
            try MLX.save(arrays: Dictionary(uniqueKeysWithValues: (single as Module).parameters().flattened()), url: syncURL)
            (piped as Module).update(parameters: ModuleParameters.unflattened(try MLX.loadArrays(url: syncURL)))
            eval(piped as Module)

            var microbatches: [[[Int32]]] = []
            for mb in 0 ..< 4 {
                var batchRows: [[Int32]] = []
                for row in 0 ..< 2 {
                    var cols: [Int32] = []
                    for col in 0 ..< 10 {
                        cols.append(Int32((mb * 23 + row * 11 + col * 3 + 5) % 220))
                    }
                    batchRows.append(cols)
                }
                microbatches.append(batchRows)
            }
            let mid = layerCount / 2
            let ranges = [0 ..< mid, mid ..< layerCount]

            let ref = Self.singleDeviceGradients(model: single, microbatches: microbatches)
            let zb = try ZBH1LayerSplitExecutor.computePipelineGradients(
                model: piped,
                microbatches: microbatches,
                stageRanges: ranges
            )
            #expect(abs(ref.meanLoss - zb.meanLoss) < 1e-4)
            #expect(!zb.gradients.isEmpty)
            for (key, refGrad) in ref.gradients {
                let ppGrad = try #require(zb.gradients[key], "Missing pipeline gradient for \(key)")
                let maxDiff = zip(refGrad, ppGrad).map { abs($0 - $1) }.max() ?? 1.0
                #expect(maxDiff < 1e-4)
            }
        }
    }

    @Test("LayerStagePartitioner respects 0.82 RAM ceiling, ZB-H1 reduces bubble ratio, and Expert-Parallel All-to-All matches local MoE")
    func stagePartitionerScheduleAndAllToAll() {
        let layers = [Int64](repeating: 2_000_000_000, count: 12)
        let peers: [Int64] = [12_000_000_000, 28_000_000_000]
        let part = LayerStagePartitioner.partition(layerMemoryBytes: layers, peerRAMBytes: peers)
        #expect(part.fitsSafetyCeiling)
        #expect(part.stageRanges.count == 2)
        #expect(part.stageRanges[0].count < part.stageRanges[1].count)

        let sched = ZBH1SchedulePlanner.build(numStages: 3, numMicrobatches: 6)
        #expect(sched.zbBubbleRatio < sched.standard1F1BBubbleRatio)

        let dispatcher = ExpertParallelMoEDispatcher(numExperts: 8, worldSize: 4)
        let tokens: [[Float]] = [[1.0, -0.5, 0.25, 2.0], [-1.0, 0.75, 0.5, -0.25]]
        let topK = [[0, 5], [2, 7]]
        let weights: [[Float]] = [[0.7, 0.3], [0.6, 0.4]]
        let epOut = dispatcher.executeAllToAll(tokens: tokens, topKIndices: topK, topKWeights: weights) { exp, x in
            x.map { $0 * Float(exp + 1) * 0.1 }
        }
        for t in tokens.indices {
            for d in 0 ..< 4 {
                let expected = weights[t][0] * tokens[t][d] * Float(topK[t][0] + 1) * 0.1
                    + weights[t][1] * tokens[t][d] * Float(topK[t][1] + 1) * 0.1
                #expect(abs(epOut[t][d] - expected) < 1e-6)
            }
        }

        let validSpec = ClusterOrchestrationSpec(peers: ["192.168.1.10:51200", "192.168.1.11:51200"], topology: .pipeline, compression: .powersgd)
        #expect(validSpec.validate().isEmpty)
        let badSpec = ClusterOrchestrationSpec(peers: ["0.0.0.0:51200"])
        #expect(!badSpec.validate().isEmpty)
        let metrics = ClusterOrchestrationSpec.renderPrometheusMetrics(activeRanks: 2, linkGbps: 80.0, bubbleRatio: sched.zbBubbleRatio)
        #expect(metrics.contains("basalt_cluster_ranks_active 2"))
    }
}
