import BasaltCore
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

private struct DecoderLayerOps {
    let inputNorm: UnaryLayer
    let postNorm: UnaryLayer
    let q: UnaryLayer
    let k: UnaryLayer
    let v: UnaryLayer
    let o: UnaryLayer
    let qNorm: UnaryLayer?
    let kNorm: UnaryLayer?
    let rope: any OffsetLayer
    let mlp: UnaryLayer
    let kvHeads: Int
    let headDim: Int

    func forward(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let normed = inputNorm(x)
        var qArr = q(normed).reshaped(b, l, -1, headDim)
        var kArr = k(normed).reshaped(b, l, kvHeads, headDim)
        let vArr = v(normed).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        if let qNorm { qArr = qNorm(qArr) }
        if let kNorm { kArr = kNorm(kArr) }
        qArr = rope(qArr.transposed(0, 2, 1, 3), offset: 0)
        kArr = rope(kArr.transposed(0, 2, 1, 3), offset: 0)
        let scale = pow(Float(headDim), -0.5)
        let attnOut = MLXFast.scaledDotProductAttention(
            queries: qArr, keys: kArr, values: vArr, scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(b, l, -1)
        let h = x + o(attnOut)
        return h + mlp(postNorm(h))
    }
}

private struct StageEndpoints {
    let isFirst: Bool
    let isLast: Bool
    let embedTokens: Embedding?
    let norm: RMSNorm?
    let lmHead: Linear?
}

private final class PipelineStageModule: Module {
    let stageIndex: Int
    let isFirst: Bool
    let isLast: Bool
    let layerOps: [DecoderLayerOps]
    let fallbackEmbed: Embedding?
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?
    @ModuleInfo(key: "layers") var layers: [Module]
    @ModuleInfo(key: "norm") var norm: RMSNorm?
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(stageIndex: Int, layers: [Module], layerOps: [DecoderLayerOps], endpoints: StageEndpoints) {
        self.stageIndex = stageIndex
        self.isFirst = endpoints.isFirst
        self.isLast = endpoints.isLast
        self.layerOps = layerOps
        self.fallbackEmbed = endpoints.embedTokens
        self._embedTokens.wrappedValue = endpoints.isFirst ? endpoints.embedTokens : nil
        self._layers.wrappedValue = layers
        self._norm.wrappedValue = endpoints.isLast ? endpoints.norm : nil
        self._lmHead.wrappedValue = endpoints.isLast ? endpoints.lmHead : nil
    }

    func forwardHidden(_ x: MLXArray) -> MLXArray {
        var h = x
        let mask = createAttentionMask(h: h, cache: nil as KVCache?)
        for op in layerOps {
            h = op.forward(h, mask: mask)
        }
        if isLast, let norm {
            h = norm(h)
        }
        return h
    }

    func forwardLogits(_ hiddenOrEmbed: MLXArray) -> MLXArray {
        let h = forwardHidden(hiddenOrEmbed)
        if let lmHead { return lmHead(h) }
        if let fallbackEmbed { return fallbackEmbed.asLinear(h) }
        return h
    }
}

public enum ZBH1LayerSplitExecutor {
    fileprivate struct ExecutionWorkspace {
        let stages: [PipelineStageModule]
        let stageRanges: [Range<Int>]
        let mbData: [(inputs: MLXArray, targets: MLXArray)]
        let scaleMB: Float
        var savedStageIn: [[MLXArray?]]
        var savedUpstreamCotangent: [[MLXArray?]]
        var losses: [Float]
        var accumGrads: [String: MLXArray] = [:]
    }

    public static func computePipelineGradients(
        model: any LanguageModel,
        microbatches: [[[Int32]]],
        stageRanges: [Range<Int>]
    ) throws -> (meanLoss: Float, gradients: [String: [Float]]) {
        let numStages = stageRanges.count
        let numMB = microbatches.count
        guard numStages >= 2, numMB >= numStages else {
            throw BasaltError.invalidRequestStructure(details: "zb-h1: requires >=2 stages and microbatches >= stages")
        }
        let stages = try PipelineStageBuilder.extractStages(from: model, ranges: stageRanges)
        let schedule = ZBH1SchedulePlanner.build(numStages: numStages, numMicrobatches: numMB)
        var ws = ExecutionWorkspace(
            stages: stages,
            stageRanges: stageRanges,
            mbData: microbatches.map { prepareMicrobatch($0) },
            scaleMB: Float(1.0) / Float(numMB),
            savedStageIn: Array(repeating: Array(repeating: nil, count: numMB), count: numStages),
            savedUpstreamCotangent: Array(repeating: Array(repeating: nil, count: numMB), count: numStages),
            losses: [Float](repeating: 0, count: numMB)
        )
        let orderedOps = schedule.perStageOps.flatMap { $0 }.sorted {
            if $0.clockTick != $1.clockTick { return $0.clockTick < $1.clockTick }
            return $0.stage < $1.stage
        }
        for op in orderedOps {
            try executeSlotOp(op, in: &ws)
        }
        let meanLoss = ws.losses.reduce(0, +)
        var outMap: [String: [Float]] = [:]
        for (k, arr) in ws.accumGrads {
            outMap[k] = arr.asType(.float32).reshaped(-1).asArray(Float.self)
        }
        return (meanLoss, outMap)
    }
}

extension ZBH1LayerSplitExecutor {
    fileprivate static func executeSlotOp(_ op: ZBH1SlotOp, in ws: inout ExecutionWorkspace) throws {
        switch op.kind {
        case .forward:
            try runForwardSlot(stageIdx: op.stage, mb: op.microbatch, in: &ws)
        case .backwardInputB:
            try runBackwardInputSlot(stageIdx: op.stage, mb: op.microbatch, in: &ws)
        case .backwardWeightW:
            try runBackwardWeightSlot(stageIdx: op.stage, mb: op.microbatch, in: &ws)
        }
    }

    private static func runForwardSlot(stageIdx s: Int, mb: Int, in ws: inout ExecutionWorkspace) throws {
        let stageIn: MLXArray
        if s == 0 {
            guard let emb = ws.stages[0].embedTokens else {
                throw BasaltError.inferenceError("zb-h1: missing stage 0 embedding")
            }
            stageIn = stopGradient(emb(ws.mbData[mb].inputs))
        } else {
            guard let prevIn = ws.savedStageIn[s - 1][mb] else {
                throw BasaltError.inferenceError("zb-h1: missing upstream activation")
            }
            stageIn = stopGradient(ws.stages[s - 1].forwardHidden(prevIn))
        }
        eval(stageIn)
        ws.savedStageIn[s][mb] = stageIn
    }

    private static func runBackwardInputSlot(stageIdx s: Int, mb: Int, in ws: inout ExecutionWorkspace) throws {
        guard let stageIn = ws.savedStageIn[s][mb] else {
            throw BasaltError.inferenceError("zb-h1: missing stage input for B")
        }
        let numStages = ws.stages.count
        let stg = ws.stages[s]
        if s == numStages - 1 {
            let targets = ws.mbData[mb].targets
            let scaleArr = MLXArray(ws.scaleMB, dtype: .float32)
            let (lossVal, cotangents) = vjp(
                { primals in [crossEntropy(logits: stg.forwardLogits(primals[0]), targets: targets, reduction: .mean)] },
                primals: [stageIn],
                cotangents: [scaleArr]
            )
            eval(lossVal[0], cotangents[0])
            ws.losses[mb] = lossVal[0].item(Float.self) * ws.scaleMB
            ws.savedUpstreamCotangent[s][mb] = scaleArr
            if s > 0 { ws.savedUpstreamCotangent[s - 1][mb] = cotangents[0] }
        } else if s > 0 {
            guard let dOut = ws.savedUpstreamCotangent[s][mb] else {
                throw BasaltError.inferenceError("zb-h1: missing downstream cotangent")
            }
            let (_, cotangents) = vjp(
                { primals in [stg.forwardHidden(primals[0])] },
                primals: [stageIn],
                cotangents: [dOut]
            )
            eval(cotangents[0])
            ws.savedUpstreamCotangent[s - 1][mb] = cotangents[0]
        }
    }

    private static func runBackwardWeightSlot(stageIdx s: Int, mb: Int, in ws: inout ExecutionWorkspace) throws {
        guard let stageIn = ws.savedStageIn[s][mb] else {
            throw BasaltError.inferenceError("zb-h1: missing stage input for W")
        }
        let stg = ws.stages[s]
        let mbInputs = ws.mbData[mb].inputs
        let targets = ws.mbData[mb].targets
        let scaleMB = ws.scaleMB
        let upstream = ws.savedUpstreamCotangent[s][mb]
        let vg = valueAndGrad(model: stg) { (mod: PipelineStageModule, x: MLXArray, inIds: MLXArray) -> MLXArray in
            let inAct = mod.isFirst && mod.embedTokens != nil ? mod.embedTokens!(inIds) : x
            if mod.isLast {
                return crossEntropy(logits: mod.forwardLogits(inAct), targets: targets, reduction: .mean) * scaleMB
            }
            let outAct = mod.forwardHidden(inAct)
            let cot = upstream ?? MLXArray.zeros(like: outAct)
            return sum(outAct * stopGradient(cot))
        }
        let (_, gradTree) = vg(stg, stageIn, mbInputs)
        for (localKey, g) in gradTree.flattened() {
            let globalKey = PipelineStageBuilder.remapStageKey(localKey, stageRange: ws.stageRanges[s])
            ws.accumGrads[globalKey] = ws.accumGrads[globalKey].map { $0 + g } ?? g
        }
        eval(Array(ws.accumGrads.values))
    }

    fileprivate static func prepareMicrobatch(_ samples: [[Int32]]) -> (inputs: MLXArray, targets: MLXArray) {
        let b = samples.count
        let seq = samples[0].count
        let arr = MLXArray(samples.flatMap { $0 }, [b, seq])
        return (arr[0..., ..<(seq - 1)], arr[0..., 1...])
    }
}

private enum PipelineStageBuilder {
    static func extractStages(
        from model: any LanguageModel,
        ranges: [Range<Int>]
    ) throws -> [PipelineStageModule] {
        guard let loraModel = model as? LoRAModel,
              let kvProvider = model as? KVCacheDimensionProvider,
              !kvProvider.kvHeads.isEmpty
        else {
            throw BasaltError.invalidRequestStructure(details: "zb-h1: unsupported model type for layer-split pipeline")
        }
        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        guard let embed = modules["model.embed_tokens"] as? Embedding,
              let norm = modules["model.norm"] as? RMSNorm
        else {
            throw BasaltError.invalidRequestStructure(details: "zb-h1: missing model.embed_tokens or model.norm")
        }
        let lmHead = modules["lm_head"] as? Linear
        let allLayers = loraModel.loraLayers
        var allOps: [DecoderLayerOps] = []
        for idx in allLayers.indices {
            guard let op = resolveLayerOps(modules: modules, layerIdx: idx, kvHeads: kvProvider.kvHeads[idx]) else {
                throw BasaltError.invalidRequestStructure(details: "zb-h1: unsupported decoder layer \(idx)")
            }
            allOps.append(op)
        }
        return ranges.enumerated().map { idx, r in
            PipelineStageModule(
                stageIndex: idx,
                layers: Array(allLayers[r]),
                layerOps: Array(allOps[r]),
                endpoints: StageEndpoints(
                    isFirst: idx == 0, isLast: idx == ranges.count - 1,
                    embedTokens: embed, norm: norm, lmHead: lmHead
                )
            )
        }
    }

    private static func resolveLayerOps(
        modules: [String: Module],
        layerIdx: Int,
        kvHeads: Int
    ) -> DecoderLayerOps? {
        let pfx = "model.layers.\(layerIdx)"
        guard let inputNorm = modules["\(pfx).input_layernorm"] as? UnaryLayer,
              let postNorm = modules["\(pfx).post_attention_layernorm"] as? UnaryLayer,
              let q = modules["\(pfx).self_attn.q_proj"] as? UnaryLayer,
              let k = modules["\(pfx).self_attn.k_proj"] as? UnaryLayer,
              let v = modules["\(pfx).self_attn.v_proj"] as? UnaryLayer,
              let o = modules["\(pfx).self_attn.o_proj"] as? UnaryLayer,
              let rope = modules["\(pfx).self_attn.rope"] as? any OffsetLayer,
              let mlp = modules["\(pfx).mlp"] as? UnaryLayer
        else { return nil }
        let kOut = (modules["\(pfx).self_attn.k_proj"] as? Linear)?.weight.dim(0) ?? (kvHeads * 16)
        let headDim = max(1, kOut / max(1, kvHeads))
        return DecoderLayerOps(
            inputNorm: inputNorm, postNorm: postNorm,
            q: q, k: k, v: v, o: o,
            qNorm: modules["\(pfx).self_attn.q_norm"] as? UnaryLayer,
            kNorm: modules["\(pfx).self_attn.k_norm"] as? UnaryLayer,
            rope: rope, mlp: mlp, kvHeads: kvHeads, headDim: headDim
        )
    }

    static func remapStageKey(_ localKey: String, stageRange: Range<Int>) -> String {
        if localKey.hasPrefix("layers.") {
            let parts = localKey.split(separator: ".", maxSplits: 2).map(String.init)
            if parts.count == 3, let localIdx = Int(parts[1]) {
                return "model.layers.\(stageRange.lowerBound + localIdx).\(parts[2])"
            }
        }
        if localKey.hasPrefix("embed_tokens.") || localKey.hasPrefix("norm.") {
            return "model.\(localKey)"
        }
        return localKey
    }
}

public struct ExpertParallelMoEDispatcher: Sendable {
    public let numExperts: Int
    public let worldSize: Int

    public init(numExperts: Int, worldSize: Int) {
        self.numExperts = max(1, numExperts)
        self.worldSize = max(1, min(numExperts, worldSize))
    }

    public func ownerRank(forExpert expert: Int) -> Int {
        min(worldSize - 1, (expert * worldSize) / numExperts)
    }

    public func executeAllToAll(
        tokens: [[Float]],
        topKIndices: [[Int]],
        topKWeights: [[Float]],
        expertLinear: (Int, [Float]) -> [Float]
    ) -> [[Float]] {
        guard !tokens.isEmpty else { return [] }
        let dim = tokens[0].count
        var rankBuckets: [[(tokenIdx: Int, slotIdx: Int, expert: Int, x: [Float])]] = Array(repeating: [], count: worldSize)
        for tIdx in tokens.indices {
            for sIdx in topKIndices[tIdx].indices {
                let exp = topKIndices[tIdx][sIdx]
                let rank = ownerRank(forExpert: exp)
                rankBuckets[rank].append((tIdx, sIdx, exp, tokens[tIdx]))
            }
        }
        var combined = Array(repeating: [Float](repeating: 0, count: dim), count: tokens.count)
        for rank in 0 ..< worldSize {
            for item in rankBuckets[rank] {
                let y = expertLinear(item.expert, item.x)
                let w = topKWeights[item.tokenIdx][item.slotIdx]
                for d in 0 ..< dim {
                    combined[item.tokenIdx][d] += w * y[d]
                }
            }
        }
        return combined
    }
}
