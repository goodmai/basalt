import Foundation
import MLX
import MLXLMCommon
import MLXNN

package final class DSALatentKVCache: KVCache {
    package private(set) var cachedLatent: MLXArray?
    package private(set) var cachedKPe: MLXArray?
    package private(set) var cachedKIndex: MLXArray?
    package private(set) var cachedValueGate: MLXArray?
    package private(set) var offset: Int = 0

    package init() {}

    package var maxSize: Int? { nil }
    package var isTrimmable: Bool { true }

    package func isTrimmable(after positions: Int) -> Bool { positions >= 0 }

    package func innerState() -> [MLXArray] {
        [cachedLatent, cachedKPe, cachedKIndex, cachedValueGate].compactMap { $0 }
    }

    package var state: [MLXArray] {
        get { innerState() }
        set {
            cachedLatent = newValue.first
            let rest = Array(newValue.dropFirst())
            cachedKPe = rest.first { $0.ndim == 4 && $0.dim(1) == 1 }
            cachedKIndex = rest.first { $0.ndim == 3 }
            cachedValueGate = rest.first { $0.ndim == 4 && $0.dim(1) > 1 }
            offset = cachedLatent?.dim(2) ?? 0
        }
    }

    package var metaState: [String] {
        get {
            [
                String(offset),
                cachedKPe != nil ? "1" : "0",
                cachedKIndex != nil ? "1" : "0",
                cachedValueGate != nil ? "1" : "0",
            ]
        }
        set {
            if let first = newValue.first, let stored = Int(first) {
                offset = max(0, stored)
            }
        }
    }

    package func updateLatent(latent: MLXArray, kPe: MLXArray?) -> (latent: MLXArray, kPe: MLXArray?) {
        let nextLatent = cachedLatent.map { concatenated([$0, latent], axis: 2) } ?? latent
        cachedLatent = nextLatent
        let nextKPe: MLXArray?
        if let kPe {
            let merged = cachedKPe.map { concatenated([$0, kPe], axis: 2) } ?? kPe
            cachedKPe = merged
            nextKPe = merged
        } else {
            nextKPe = nil
        }
        offset = nextLatent.dim(2)
        return (nextLatent, nextKPe)
    }

    package func updateIndexerKeys(_ kIndex: MLXArray) -> MLXArray {
        let next = cachedKIndex.map { concatenated([$0, kIndex], axis: 1) } ?? kIndex
        cachedKIndex = next
        return next
    }

    package func updateValueGate(_ gate: MLXArray) -> MLXArray {
        let next = cachedValueGate.map { concatenated([$0, gate], axis: 2) } ?? gate
        cachedValueGate = next
        return next
    }

    package func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let pe: MLXArray? = values.dim(-1) > 0 ? values : nil
        let (latent, updatedPe) = updateLatent(latent: keys, kPe: pe)
        return (latent, updatedPe ?? values)
    }

    @discardableResult
    package func trim(_ n: Int) -> Int {
        guard n > 0, offset > 0 else { return 0 }
        let removable = min(n, offset)
        let remaining = offset - removable
        if remaining == 0 {
            cachedLatent = nil
            cachedKPe = nil
            cachedKIndex = nil
            cachedValueGate = nil
        } else {
            cachedLatent = cachedLatent?[0..., 0..., ..<remaining, 0...]
            cachedKPe = cachedKPe?[0..., 0..., ..<remaining, 0...]
            cachedKIndex = cachedKIndex?[0..., ..<remaining, 0...]
            cachedValueGate = cachedValueGate?[0..., 0..., ..<remaining, 0...]
        }
        offset = remaining
        return removable
    }

    package func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        n > 1 ? .causal : .none
    }

    package func copy() -> any KVCache {
        let copied = DSALatentKVCache()
        copied.cachedLatent = cachedLatent
        copied.cachedKPe = cachedKPe
        copied.cachedKIndex = cachedKIndex
        copied.cachedValueGate = cachedValueGate
        copied.offset = offset
        return copied
    }
}

extension GatedMLAAttention {
    func attend(
        _ ctx: AttendContext,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        qOffset: Int,
        forceExpanded: Bool
    ) -> MLXArray {
        if !forceExpanded, usesAbsorbedPath {
            if ctx.topIndices != nil, let sparse = attendSparseAbsorbed(ctx, mask: mask, qOffset: qOffset) {
                return sparse
            }
            if ctx.topIndices == nil, let dense = attendDenseAbsorbed(ctx, mask: mask) {
                return dense
            }
        }
        if ctx.topIndices != nil {
            return attendSparseExpanded(ctx, mask: mask, qOffset: qOffset)
        }
        return attendDenseExpanded(ctx, mask: mask)
    }

    private func attendDenseAbsorbed(
        _ ctx: AttendContext,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray? {
        guard let (queries, keys) = absorbedQueriesAndKeys(ctx) else { return nil }
        let latentOut: MLXArray
        if let vGate = ctx.cachedVGate {
            let lQ = queries.dim(2)
            let lK = keys.dim(2)
            let rawLogits = matmul(queries.asType(.float32), keys.asType(.float32).swappedAxes(-1, -2)) * scale
            let maskedLogits = Self.applyMask(rawLogits, mask: mask, lQ: lQ, lK: lK)
            let weights = softmax(maskedLogits, axis: -1) * vGate.asType(.float32).swappedAxes(-1, -2)
            latentOut = matmul(weights, ctx.cachedLatent.asType(.float32)).asType(queries.dtype)
        } else {
            latentOut = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: ctx.cachedLatent, scale: scale, mask: mask)
        }
        return BailingHybridMLAAbsorption.unembedOutput(
            latentOut, kvBProj: kvBProj, heads: config.numHeads,
            nopeDim: config.qkNopeHeadDim, vDim: config.vHeadDim)
    }

    private func attendSparseAbsorbed(
        _ ctx: AttendContext,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        qOffset: Int
    ) -> MLXArray? {
        guard let topIndices = ctx.topIndices else { return nil }
        let qAbsorbed: MLXArray
        if let qNope = ctx.qNope, config.qkNopeHeadDim > 0 {
            guard let embedded = BailingHybridMLAAbsorption.embedQuery(
                qNope, kvBProj: kvBProj, heads: config.numHeads,
                nopeDim: config.qkNopeHeadDim, vDim: config.vHeadDim
            ) else { return nil }
            qAbsorbed = embedded
        } else if let qPe = ctx.qPe {
            qAbsorbed = MLXArray.zeros([qPe.dim(0), qPe.dim(1), qPe.dim(2), config.kvLoraRank], dtype: qPe.dtype)
        } else {
            return nil
        }
        let useMetal = (indexer?.config.useMetal ?? true) && !training
        let latentOut = DSALightningIndexerMetal.sparseLatentAttention(
            DSALightningIndexerMetal.SparseLatentInputs(
                qAbsorbed: qAbsorbed, qPe: ctx.qPe, cKV: ctx.cachedLatent,
                kPe: ctx.cachedKPe, topIndices: topIndices, valueHeadGate: ctx.cachedVGate),
            options: DSALightningIndexerMetal.SparseLatentOptions(
                scale: scale, causal: Self.isCausalMask(mask), qOffset: qOffset, useMetal: useMetal))
        return BailingHybridMLAAbsorption.unembedOutput(
            latentOut, kvBProj: kvBProj, heads: config.numHeads,
            nopeDim: config.qkNopeHeadDim, vDim: config.vHeadDim)
    }

    private func attendDenseExpanded(
        _ ctx: AttendContext,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let (queries, keys, values) = expandedQKV(ctx)
        return MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: scale, mask: mask)
    }

    private func attendSparseExpanded(
        _ ctx: AttendContext,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        qOffset: Int
    ) -> MLXArray {
        guard let topIndices = ctx.topIndices else { return attendDenseExpanded(ctx, mask: mask) }
        let (queries, keys, values) = expandedQKV(ctx)
        let (batch, heads, lQ, qkDim) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
        let lK = keys.dim(2)
        let topK = topIndices.dim(-1)
        let idx = topIndices.asType(.int32)
        var valid = (idx .>= Int32(0)) .&& (idx .< Int32(lK))
        if Self.isCausalMask(mask) {
            let qPos = MLXArray(Int32(qOffset) ..< Int32(qOffset + lQ)).reshaped(1, lQ, 1)
            valid = valid .&& (idx .<= qPos)
        }
        let safeIdx = MLX.where(valid, idx, MLXArray(Int32(0)))
        let flatIdx = stopGradient(safeIdx.reshaped(batch, 1, lQ * topK, 1))
        let kSel = takeAlong(keys, flatIdx, axis: 2).reshaped(batch, heads, lQ, topK, qkDim)
        let vSel = takeAlong(values, flatIdx, axis: 2).reshaped(batch, heads, lQ, topK, config.vHeadDim)
        let logits = matmul(queries.asType(.float32).expandedDimensions(axis: -2), kSel.asType(.float32).swappedAxes(-1, -2))
            .squeezed(axis: -2) * scale
        let activeValid = valid.expandedDimensions(axis: 1) .&& (logits .!= MLXArray(-Float.infinity))
        let anyValid = activeValid.any(axis: -1).expandedDimensions(axis: -1)
        let masked = MLX.where(activeValid, logits, MLXArray(-Float.infinity))
        let probs = MLX.where(anyValid, softmax(MLX.where(anyValid, masked, MLXArray(Float(0))), axis: -1), MLXArray(Float(0)))
        return matmul(probs.expandedDimensions(axis: -2), vSel.asType(.float32)).squeezed(axis: -2).asType(queries.dtype)
    }
}

extension GatedMLAAttention {
    private func absorbedQueriesAndKeys(_ ctx: AttendContext) -> (MLXArray, MLXArray)? {
        if let qNope = ctx.qNope, config.qkNopeHeadDim > 0 {
            guard let qAbs = BailingHybridMLAAbsorption.embedQuery(
                qNope, kvBProj: kvBProj, heads: config.numHeads,
                nopeDim: config.qkNopeHeadDim, vDim: config.vHeadDim
            ) else { return nil }
            if let qPe = ctx.qPe, let cachedKPe = ctx.cachedKPe {
                return (concatenated([qAbs, qPe], axis: -1), concatenated([ctx.cachedLatent, cachedKPe], axis: -1))
            }
            return (qAbs, ctx.cachedLatent)
        }
        guard let qPe = ctx.qPe, let cachedKPe = ctx.cachedKPe else { return nil }
        return (qPe, cachedKPe)
    }

    private func expandedQKV(_ ctx: AttendContext) -> (queries: MLXArray, keys: MLXArray, values: MLXArray) {
        let batch = ctx.cachedLatent.dim(0)
        let total = ctx.cachedLatent.dim(2)
        let kv = kvBProj(ctx.cachedLatent.squeezed(axis: 1))
            .reshaped(batch, total, config.numHeads, config.qkNopeHeadDim + config.vHeadDim)
            .transposed(0, 2, 1, 3)
        let kNope: MLXArray?
        var values: MLXArray
        if config.qkNopeHeadDim > 0 {
            let parts = split(kv, indices: [config.qkNopeHeadDim], axis: -1)
            kNope = parts[0]
            values = parts[1]
        } else {
            kNope = nil
            values = kv
        }
        if let vGate = ctx.cachedVGate {
            values = (values.asType(.float32) * vGate.asType(.float32)).asType(values.dtype)
        }
        let queries: MLXArray
        let keys: MLXArray
        if let qPe = ctx.qPe, let cachedKPe = ctx.cachedKPe {
            let kPe = broadcast(cachedKPe, to: [batch, config.numHeads, total, config.qkRopeHeadDim])
            if let qNope = ctx.qNope, let kNope {
                queries = concatenated([qNope, qPe], axis: -1)
                keys = concatenated([kNope, kPe], axis: -1)
            } else {
                queries = qPe
                keys = kPe
            }
        } else if let qNope = ctx.qNope, let kNope {
            queries = qNope
            keys = kNope
        } else {
            preconditionFailure("GatedMLAAttention requires positive qkNopeHeadDim or qkRopeHeadDim")
        }
        return (queries, keys, values)
    }

    static func isCausalMask(_ mask: MLXFast.ScaledDotProductAttentionMaskMode) -> Bool {
        if case .causal = mask { return true }
        return false
    }

    static func applyMask(
        _ logits: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        lQ: Int,
        lK: Int
    ) -> MLXArray {
        if case .causal = mask, lQ > 1 {
            let qOffset = max(0, lK - lQ)
            let qPos = MLXArray(Int32(qOffset) ..< Int32(qOffset + lQ)).reshaped(1, 1, lQ, 1)
            let kPos = MLXArray(Int32(0) ..< Int32(lK)).reshaped(1, 1, 1, lK)
            return MLX.where(kPos .<= qPos, logits, MLXArray(-Float.infinity))
        }
        if case .array(let maskArr) = mask {
            if maskArr.dtype == .bool {
                return MLX.where(maskArr, logits, MLXArray(-Float.infinity))
            }
            return logits + maskArr.asType(.float32)
        }
        return logits
    }
}
