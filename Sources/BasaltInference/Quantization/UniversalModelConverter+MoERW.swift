import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

package struct MoERWConversionSummary: Sendable, Equatable {
    package var plan: DeadZoneBitAllocator.Plan
    package var quantizedLayers: Int
    package var totalFlips: Int
    package var meanLossRatio: Float
    package var calibrationTokens: Int
    package var wallSeconds: Double
}

final class MoERouterGateBox {
    let numExperts: Int
    let inFeatures: Int
    var lastInput: MLXArray?
    var lastLogits: MLXArray?
    var pooledInputs: [MLXArray] = []
    var pooledLogits: [MLXArray] = []
    var pooledRows = 0

    init(numExperts: Int, inFeatures: Int) {
        self.numExperts = max(1, numExperts)
        self.inFeatures = max(1, inFeatures)
    }

    func record(input: MLXArray, logits: MLXArray) {
        let rows = input.size / inFeatures
        guard rows > 0, logits.size == rows * numExperts else { return }
        let xFlat = input.reshaped(rows, inFeatures).asType(.float32)
        let zFlat = logits.reshaped(rows, numExperts).asType(.float32)
        eval(xFlat, zFlat)
        lastInput = xFlat
        lastLogits = zFlat
        if pooledRows < 128 {
            let take = min(rows, 128 - pooledRows)
            pooledInputs.append(xFlat[0 ..< take])
            pooledLogits.append(zFlat[0 ..< take])
            pooledRows += take
        }
    }
}

final class MoERouterGateTap: Linear {
    let wrapped: Module
    let box: MoERouterGateBox

    init(wrapping module: Module, box: MoERouterGateBox) {
        self.wrapped = module
        self.box = box
        super.init(weight: MLXArray.zeros([1, 1], dtype: .float16), bias: nil)
        freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out: MLXArray
        if let nf4 = wrapped as? NF4Linear {
            out = nf4(x)
        } else if let ql = wrapped as? QuantizedLinear {
            out = ql(x)
        } else if let lin = wrapped as? Linear {
            out = lin(x)
        } else {
            preconditionFailure("Unsupported router gate module: \(type(of: wrapped))")
        }
        box.record(input: x, logits: out)
        return out
    }
}

final class MoERWSwitchReservoir {
    let inFeatures: Int
    let numExperts: Int
    let maxRows: Int
    let gateBox: MoERouterGateBox?
    var routedX: [[MLXArray]]
    var routedC: [[Float]]
    var countsByExpert: [Int]
    var poolX: [MLXArray] = []
    var poolProbs: [MLXArray] = []
    var poolCount = 0
    var totalAssignments = 0

    init(inFeatures: Int, numExperts: Int, maxRows: Int, gateBox: MoERouterGateBox?) {
        self.inFeatures = inFeatures
        self.numExperts = numExperts
        self.maxRows = maxRows
        self.gateBox = gateBox
        self.routedX = Array(repeating: [], count: numExperts)
        self.routedC = Array(repeating: [], count: numExperts)
        self.countsByExpert = Array(repeating: 0, count: numExperts)
    }

    func record(_ x: MLXArray, indices: MLXArray, sortedIndices: Bool) {
        guard inFeatures > 0, x.size >= inFeatures, indices.size > 0 else { return }
        let xRows = x.size / inFeatures
        let xFlat = x.reshaped(xRows, inFeatures).asType(.bfloat16)
        let idxFlat = indices.reshaped(-1).asType(.int32)
        eval(xFlat, idxFlat)
        let ids = idxFlat.asArray(Int32.self)
        let gates = resolveGates(idCount: ids.count, xRows: xRows, xFlat: xFlat, sortedIndices: sortedIndices)
        totalAssignments += ids.count
        if xRows == ids.count {
            appendDirect(xFlat: xFlat, ids: ids, gates: gates)
        } else if indices.ndim >= 2, xRows * indices.dim(-1) == ids.count {
            appendBroadcast(xFlat: xFlat, ids: ids, gates: gates, topK: indices.dim(-1))
        }
    }

    private func resolveGates(idCount: Int, xRows: Int, xFlat: MLXArray, sortedIndices: Bool) -> [Float] {
        guard let z = gateBox?.lastLogits else {
            recordFallbackPool(xFlat: xFlat, probs: nil)
            return Array(repeating: 1.0, count: idCount)
        }
        let zRows = max(1, z.size / max(1, numExperts))
        let topK = max(1, idCount / zRows)
        guard zRows * topK == idCount else {
            recordFallbackPool(xFlat: xFlat, probs: nil)
            return Array(repeating: 1.0, count: idCount)
        }
        let top = MoERouterWeightedHessian.topKGating(logits: z, topK: topK)
        recordFallbackPool(xFlat: xFlat, probs: xRows == zRows ? top.fullProbs : nil)
        if sortedIndices {
            let order = argSort(top.indices.reshaped(-1))
            let sortedW = top.weights.reshaped(-1).take(order, axis: 0)
            eval(sortedW)
            return sortedW.asArray(Float.self)
        }
        return top.weights.reshaped(-1).asArray(Float.self)
    }

    private func recordFallbackPool(xFlat: MLXArray, probs: MLXArray?) {
        guard poolCount < 128 else { return }
        let take = min(xFlat.dim(0), 128 - poolCount)
        guard take > 0 else { return }
        poolX.append(xFlat[0 ..< take])
        if let probs, probs.dim(0) >= take {
            poolProbs.append(probs[0 ..< take])
        }
        poolCount += take
    }

    private func appendDirect(xFlat: MLXArray, ids: [Int32], gates: [Float]) {
        for (rowIdx, rawExp) in ids.enumerated() {
            let e = Int(rawExp)
            guard e >= 0, e < numExperts, countsByExpert[e] < maxRows else { continue }
            routedX[e].append(xFlat[rowIdx ..< (rowIdx + 1)])
            routedC[e].append(rowIdx < gates.count ? gates[rowIdx] : 1.0)
            countsByExpert[e] += 1
        }
    }

    private func appendBroadcast(xFlat: MLXArray, ids: [Int32], gates: [Float], topK: Int) {
        let tokens = ids.count / max(1, topK)
        for t in 0 ..< tokens {
            for k in 0 ..< topK {
                let flatIdx = t * topK + k
                let e = Int(ids[flatIdx])
                guard e >= 0, e < numExperts, countsByExpert[e] < maxRows else { continue }
                routedX[e].append(xFlat[t ..< (t + 1)])
                routedC[e].append(flatIdx < gates.count ? gates[flatIdx] : 1.0)
                countsByExpert[e] += 1
            }
        }
    }

    func expertHessian(
        expert: Int,
        estimator: MoERouterWeightedHessian,
        routerWeight: MLXArray?
    ) -> (hessian: MLXArray, mass: Double)? {
        guard expert >= 0, expert < numExperts else { return nil }
        let mass = Double(routedC[expert].reduce(0, +)) / Double(max(1, totalAssignments))
        if countsByExpert[expert] >= estimator.minRoutedTokens {
            let xExp = concatenated(routedX[expert], axis: 0)
            let cExp = MLXArray(routedC[expert])
            return (estimator.hessian(activations: xExp, gatingWeights: cExp), max(1e-4, mass))
        }
        guard !poolX.isEmpty else { return nil }
        let xp = concatenated(poolX, axis: 0).asType(.float32)
        if poolProbs.count == poolX.count {
            let probs = concatenated(poolProbs, axis: 0)[0..., expert]
            return (estimator.hessian(activations: xp, gatingWeights: probs), max(1e-4, mass))
        }
        if let rw = routerWeight, rw.ndim == 2, rw.dim(0) == numExperts, rw.dim(1) == inFeatures {
            let probs = softmax(matmul(xp, rw.asType(.float32).T), axis: -1)[0..., expert]
            return (estimator.hessian(activations: xp, gatingWeights: probs), max(1e-4, mass))
        }
        let ones = MLXArray.ones([xp.dim(0)], dtype: .float32)
        return (estimator.hessian(activations: xp, gatingWeights: ones), max(1e-4, mass))
    }
}

final class MoERWSwitchTap: SwitchLinear {
    let wrapped: Module
    let reservoir: MoERWSwitchReservoir

    init(wrapping module: Module, reservoir: MoERWSwitchReservoir) {
        self.wrapped = module
        self.reservoir = reservoir
        let dummy = MLXArray.zeros([1, 1, 1], dtype: .float16)
        super.init(inputDims: 1, outputDims: 1, numExperts: 1, weight: dummy, bias: nil)
        freeze()
    }

    override func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false) -> MLXArray {
        reservoir.record(x, indices: indices, sortedIndices: sortedIndices)
        if let nf4 = wrapped as? NF4SwitchLinear { return nf4(x, indices, sortedIndices: sortedIndices) }
        if let msw = wrapped as? MACaQSwitchLinear { return msw(x, indices, sortedIndices: sortedIndices) }
        if let qsw = wrapped as? QuantizedSwitchLinear { return qsw(x, indices, sortedIndices: sortedIndices) }
        if let sw = wrapped as? SwitchLinear { return sw(x, indices, sortedIndices: sortedIndices) }
        preconditionFailure("Unsupported MoERW SwitchLinear: \(type(of: wrapped))")
    }
}

extension UniversalModelConverter {
    private enum MoEMACaQStrategy: Sendable {
        case budget(Double)
        case uniform(MACaQTargetMode)
    }

    package static func encodeMoERWBudgetDirectory(
        request: ModelConversionRequest,
        workingSource: URL,
        outputURL: URL,
        targetBPW: Double,
        onProgress: @escaping @Sendable (BasaltCore::ModelConversionProgress) -> Void
    ) async throws -> (shards: Int, bytesAfter: Int64) {
        try await runMoEMACaQDirectory(
            request: request, workingSource: workingSource, outputURL: outputURL,
            strategy: .budget(targetBPW), onProgress: onProgress)
    }

    package static func encodeMACaQWithMoERWDirectory(
        request: ModelConversionRequest,
        workingSource: URL,
        outputURL: URL,
        targetMode: MACaQTargetMode,
        onProgress: @escaping @Sendable (BasaltCore::ModelConversionProgress) -> Void
    ) async throws -> (shards: Int, bytesAfter: Int64) {
        try await runMoEMACaQDirectory(
            request: request, workingSource: workingSource, outputURL: outputURL,
            strategy: .uniform(targetMode), onProgress: onProgress)
    }

    private static func runMoEMACaQDirectory(
        request: ModelConversionRequest,
        workingSource: URL,
        outputURL: URL,
        strategy: MoEMACaQStrategy,
        onProgress: @escaping @Sendable (BasaltCore::ModelConversionProgress) -> Void
    ) async throws -> (shards: Int, bytesAfter: Int64) {
        try QTIPNativeLoader.validateOutput(source: workingSource, output: outputURL)
        try await MACaQRefiner.validateSourceDirectory(workingSource)
        var cfg = MACaQConfig()
        cfg.ridge = 0.0
        if let blk = request.hadamardBlock { cfg.hadamardBlock = blk }
        if let alpha = request.alpha { cfg.alpha = alpha }
        let finalCfg = cfg
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0, disableExpertStreaming: true))
        defer { await withTaskCancellationShield { await engine.unload(); Memory.clearCache() } }
        try await engine.load(modelPath: workingSource.path)
        guard let container = await engine.container else { throw BasaltError.modelNotLoaded }
        let corpus = (request.calibrationText?.isEmpty == false ? request.calibrationText : nil)
            ?? MultilingualMoECalibrationCorpus.defaultText
        let batches = await GSQRCOConverter.tokenizeCalibration(
            container, corpus: corpus, count: finalCfg.calibrationBatches, seqLen: finalCfg.calibrationSeqLen)
        guard !batches.isEmpty else {
            throw BasaltError.invalidRequestStructure(details: "macaq-moe: calibration text produced zero token batches")
        }
        do {
            return try await container.perform { ctx in
                let manifest: [String: MACaQLayerSpec]
                let moeBPW: Double?
                switch strategy {
                case .budget(let bpw):
                    let res = try quantizeModelMoERW(
                        ctx.model, calibrationTokenBatches: batches, targetBPW: bpw, config: finalCfg
                    ) { onProgress(.init(phase: "quantizing", completed: 2, total: 4, message: $0)) }
                    manifest = res.manifest
                    moeBPW = res.summary.plan.achievedBPW
                case .uniform(let targetMode):
                    let res = try MoERWPipeline.quantizeUniformMACaQ(
                        ctx.model, targetMode: targetMode, batches: batches, estimator: .init(), config: finalCfg
                    ) { onProgress(.init(phase: "quantizing", completed: 2, total: 4, message: $0)) }
                    manifest = res.manifest
                    moeBPW = nil
                }
                let written = try MACaQRefiner.writeMACaQDirectory(
                    model: ctx.model as Module, manifest: manifest, alpha: finalCfg.alpha,
                    moeBPW: moeBPW, source: workingSource, output: outputURL)
                return (written.shards, Int64(written.bytes))
            }
        } catch {
            try? FileManager.default.removeItem(at: outputURL)
            throw error
        }
    }

    package static func refineStandardDirectoryWithMoERW(
        output: URL,
        teacher: URL,
        calibrationText: String? = nil,
        estimator: MoERouterWeightedHessian = .init(),
        config: MACaQConfig = MACaQConfig()
    ) async throws -> MACaQSummary {
        try await MACaQRefiner.validateSourceDirectory(output)
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0, disableExpertStreaming: true))
        defer { await withTaskCancellationShield { await engine.unload(); Memory.clearCache() } }
        try await engine.load(modelPath: output.path)
        guard let container = await engine.container else { throw BasaltError.modelNotLoaded }
        let corpus = (calibrationText?.isEmpty == false ? calibrationText : nil)
            ?? MultilingualMoECalibrationCorpus.defaultText
        var localCfg = config
        localCfg.ridge = 0.0
        let finalCfg = localCfg
        let batches = await GSQRCOConverter.tokenizeCalibration(
            container, corpus: corpus, count: finalCfg.calibrationBatches, seqLen: finalCfg.calibrationSeqLen)
        guard !batches.isEmpty else {
            throw BasaltError.invalidRequestStructure(details: "moe-rw: calibration text produced zero token batches")
        }
        return try await container.perform { ctx in
            let reader = try MACaQShardTeacherReader(directory: teacher)
            var summary = try MoERWPipeline.refineStandardModel(
                ctx.model, batches: batches, estimator: estimator, config: finalCfg,
                teacherWeight: { try reader.denseWeight(for: $0) })
            summary.rewrittenTensors = try PVTuner.writeTunedShards(from: ctx.model as Module, into: output)
            return summary
        }
    }

    package static func quantizeModelMoERW(
        _ model: any LanguageModel,
        calibrationTokenBatches: [[Int32]],
        targetBPW: Double,
        estimator: MoERouterWeightedHessian = .init(),
        config: MACaQConfig = MACaQConfig(),
        onStep: ((String) -> Void)? = nil
    ) throws -> (summary: MoERWConversionSummary, manifest: [String: MACaQLayerSpec]) {
        try MoERWPipeline.quantizeBudgetModel(
            model, batches: calibrationTokenBatches, targetBPW: targetBPW,
            estimator: estimator, config: config, onStep: onStep)
    }
}
