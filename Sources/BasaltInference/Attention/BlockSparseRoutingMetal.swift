import BasaltCore
import Foundation
import MLX

package enum BlockSparseRoutingMetal {
    package static let supportedDTypes: Set<DType> = [.bfloat16, .float16, .float32]
    package static let maxHeadDim = 256

    package struct Config: Sendable {
        package static let defaultGates: [Float] = [0.2, 0.5, 0.3]

        package var blockSize: Int
        package var topBlocks: Int
        package var slidingWindow: Int
        package var scale: Float
        package var compressionGate: Float
        package var selectionGate: Float
        package var windowGate: Float
        package var causal: Bool

        package init(
            blockSize: Int = 16,
            topBlocks: Int = 2,
            slidingWindow: Int = 16,
            scale: Float,
            gates: [Float] = Self.defaultGates,
            causal: Bool = false
        ) {
            self.blockSize = max(1, blockSize)
            self.topBlocks = max(1, topBlocks)
            self.slidingWindow = max(1, slidingWindow)
            self.scale = scale
            self.compressionGate = gates.count > 0 ? gates[0] : 0.2
            self.selectionGate = gates.count > 1 ? gates[1] : 0.5
            self.windowGate = gates.count > 2 ? gates[2] : 0.3
            self.causal = causal
        }
    }

    package struct SparseOptions: Sendable {
        package var blockSize: Int
        package var scale: Float
        package var causal: Bool
        package var qOffset: Int?

        package init(blockSize: Int, scale: Float, causal: Bool = false, qOffset: Int? = nil) {
            self.blockSize = max(1, blockSize)
            self.scale = scale
            self.causal = causal
            self.qOffset = qOffset
        }
    }

    package struct PrecomputedCentroids {
        package var keys: MLXArray
        package var values: MLXArray
        package var blockSize: Int
        package var qOffset: Int?

        package init(keys: MLXArray, values: MLXArray, blockSize: Int, qOffset: Int? = nil) {
            self.keys = keys
            self.values = values
            self.blockSize = max(1, blockSize)
            self.qOffset = qOffset
        }
    }

    struct AttentionGeometry: Sendable {
        let batch: Int
        let qHeads: Int
        let kvHeads: Int
        let gqaGroup: Int
        let lQ: Int
        let lK: Int
        let d: Int

        var totalQHeads: Int { batch * qHeads }
        var totalKVHeads: Int { batch * kvHeads }

        static func resolve(q: MLXArray, k: MLXArray) -> AttentionGeometry {
            let is4D = q.ndim == 4
            let batch = is4D ? q.dim(0) : 1
            let qHeads = is4D ? q.dim(1) : q.dim(0)
            let kvHeads = k.ndim == 4 ? k.dim(1) : k.dim(0)
            let lQ = q.dim(-2)
            let lK = k.dim(-2)
            let d = q.dim(-1)
            precondition(batch >= 0 && qHeads >= 0 && kvHeads >= 0 && (kvHeads == 0 || qHeads.isMultiple(of: kvHeads)))
            return AttentionGeometry(
                batch: batch, qHeads: qHeads, kvHeads: kvHeads,
                gqaGroup: kvHeads > 0 ? qHeads / kvHeads : 1, lQ: lQ, lK: lK, d: d)
        }
    }

    private static func runOnCPUIfFloat64(_ arrays: [MLXArray?], _ body: () -> MLXArray) -> MLXArray {
        guard Device.defaultDevice() == .gpu, arrays.contains(where: { $0?.dtype == .float64 }) else {
            return body()
        }
        return Device.withDefaultDevice(.cpu, body)
    }

    package static func deterministicTopIndices(_ scores: MLXArray, k: Int) -> MLXArray {
        guard scores.ndim > 0 else { return MLXArray.zeros([0], type: Int32.self) }
        return runOnCPUIfFloat64([scores]) {
            let n = scores.dim(-1)
            let kSel = min(max(0, k), n)
            guard kSel > 0, scores.size > 0 else {
                return MLXArray.zeros(Array(scores.shape.dropLast()) + [kSel], type: Int32.self)
            }
            let f32 = stopGradient(scores).asType(.float32)
            let clean = MLX.where(isNaN(f32), MLXArray(-Float.infinity), f32 + Float(0.0))
            let rawBits = contiguous(clean).view(dtype: .uint32)
            let orderKey = MLX.where(
                rawBits .>= MLXArray(UInt32(0x8000_0000)),
                rawBits,
                MLXArray(UInt32(0x7FFF_FFFF)) - rawBits)
            let cols = MLXArray(Int32(0) ..< Int32(n)).asType(.uint64)
            let packed = orderKey.asType(.uint64) * MLXArray(UInt64(1) << 32) + cols
            return argSort(packed, axis: -1)[.ellipsis, 0 ..< kSel].asType(.int32)
        }
    }

    package static func blockCentroids(_ tensor: MLXArray, blockSize: Int) -> MLXArray {
        runOnCPUIfFloat64([tensor]) {
            let bs = max(1, blockSize)
            let heads = tensor.dim(0)
            let lK = tensor.dim(1)
            let d = tensor.dim(2)
            guard lK > 0 else {
                return MLXArray.zeros([heads, 0, d], dtype: tensor.dtype)
            }
            let fullBlocks = lK / bs
            let remainder = lK % bs
            if remainder == 0 {
                return mean(tensor.reshaped(heads, fullBlocks, bs, d), axis: 2)
            }
            let tail = mean(tensor[0..., (fullBlocks * bs) ..< lK, 0...], axis: 1, keepDims: true)
            if fullBlocks == 0 {
                return tail
            }
            let full = mean(tensor[0..., 0 ..< (fullBlocks * bs), 0...].reshaped(heads, fullBlocks, bs, d), axis: 2)
            return concatenated([full, tail], axis: 1)
        }
    }

    package static func selectMoBABlocks(
        q: MLXArray,
        k: MLXArray,
        blockSize: Int,
        topBlocks: Int
    ) -> MLXArray {
        runOnCPUIfFloat64([q, k]) {
            let bs = max(1, blockSize)
            let geo = AttentionGeometry.resolve(q: q, k: k)
            let numBlocks = (geo.lK + bs - 1) / bs
            guard numBlocks > 0, geo.lQ > 0, geo.totalQHeads > 0, geo.totalKVHeads > 0, geo.d > 0 else {
                return MLXArray.zeros([geo.totalQHeads, geo.lQ, 0], type: Int32.self)
            }
            let kCentroids = blockCentroids(k.asType(.float32).reshaped(geo.totalKVHeads, geo.lK, geo.d), blockSize: bs)
            let qGrouped = q.asType(.float32).reshaped(geo.totalKVHeads, geo.gqaGroup, geo.lQ, geo.d)
            let blockScores = matmul(qGrouped, kCentroids.expandedDimensions(axis: 1).swappedAxes(-1, -2))
                .reshaped(geo.totalQHeads, geo.lQ, numBlocks)
            return deterministicTopIndices(blockScores, k: topBlocks)
        }
    }
}

extension BlockSparseRoutingMetal {
    package static func selectNSABlocks(
        compWeights: MLXArray,
        blockSize: Int,
        topBlocks: Int,
        lK: Int,
        causal: Bool,
        qOffset: Int
    ) -> MLXArray {
        runOnCPUIfFloat64([compWeights]) {
            let bs = max(1, blockSize)
            let gqaGroup = compWeights.dim(1)
            let lQ = compWeights.dim(2)
            let numBlocks = compWeights.dim(3)
            var importance = sum(compWeights, axis: 1)
            guard causal else {
                return deterministicTopIndices(importance, k: topBlocks)
            }
            let qPos = MLXArray(Int32(qOffset) ..< Int32(qOffset + lQ)).reshaped(1, lQ, 1)
            let blockIds = MLXArray(Int32(0) ..< Int32(numBlocks)).reshaped(1, 1, numBlocks)
            let blockStart = blockIds * Int32(bs)
            let blockEnd = MLX.minimum((blockIds + Int32(1)) * Int32(bs), MLXArray(Int32(lK))) - Int32(1)
            let reachable = blockStart .<= qPos
            let isCurrentPartial = reachable .&& (blockEnd .> qPos)
            importance = MLX.where(isCurrentPartial, MLXArray(Float(gqaGroup) + 1.0), importance)
            importance = MLX.where(reachable, importance, MLXArray(-Float.infinity))
            let picked = deterministicTopIndices(importance, k: topBlocks)
            return MLX.where(picked * Int32(bs) .<= qPos, picked, MLXArray(Int32(-1)))
        }
    }

    package static func sparseSelectedAttention(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        selectedBlocks: MLXArray,
        blockSize: Int,
        scale: Float
    ) -> MLXArray {
        sparseSelectedAttention(
            q: q, k: k, v: v, selectedBlocks: selectedBlocks,
            options: SparseOptions(blockSize: blockSize, scale: scale))
    }

    package static func sparseSelectedAttention(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        selectedBlocks: MLXArray,
        options: SparseOptions
    ) -> MLXArray {
        runOnCPUIfFloat64([q, k, v, selectedBlocks]) {
            let origType = q.dtype
            let geo = AttentionGeometry.resolve(q: q, k: k)
            let topK = selectedBlocks.dim(-1)
            guard topK > 0, geo.lQ > 0, geo.lK > 0, geo.totalQHeads > 0, geo.totalKVHeads > 0, geo.d > 0 else {
                return MLXArray.zeros(q.shape, dtype: origType)
            }
            let selHeads = selectedBlocks.size / max(1, geo.lQ * topK)
            precondition(selHeads == geo.totalQHeads || selHeads == geo.totalKVHeads)
            let selGroup = geo.totalQHeads / selHeads
            let effectiveOffset = options.qOffset ?? max(0, geo.lK - geo.lQ)
            guard Device.defaultDevice() == .gpu,
                  geo.d.isMultiple(of: 32), geo.d <= maxHeadDim,
                  supportedDTypes.contains(origType),
                  k.dtype == origType,
                  v.dtype == origType
            else {
                return sparseSelectedAttentionMLX(
                    q: q, k: k, v: v, selectedBlocks: selectedBlocks, options: options)
            }
            let params = MLXArray([options.scale])
            let shapes = MLXArray([
                Int32(geo.totalQHeads), Int32(geo.lQ), Int32(geo.lK), Int32(geo.d),
                Int32(options.blockSize), Int32(topK), Int32(geo.gqaGroup), Int32(selGroup),
                Int32(options.causal ? 1 : 0), Int32(effectiveOffset),
            ])
            let numBlocks = (geo.lK + options.blockSize - 1) / max(1, options.blockSize)
            let safeBlocks = sanitizeInt32Indices(selectedBlocks, upperBound: numBlocks).reshaped(selHeads, geo.lQ, topK)
            let out = mobaSparseKernel(
                [
                    q.reshaped(geo.totalQHeads, geo.lQ, geo.d),
                    k.reshaped(geo.totalKVHeads, geo.lK, geo.d),
                    v.reshaped(geo.totalKVHeads, geo.lK, geo.d),
                    safeBlocks,
                    params,
                    shapes,
                ],
                template: [("T", origType), ("BLOCKS", geo.d / 32)],
                grid: (32, geo.lQ, geo.totalQHeads),
                threadGroup: (32, 1, 1),
                outputShapes: [[geo.totalQHeads, geo.lQ, geo.d]],
                outputDTypes: [origType]
            )[0]
            return out.reshaped(q.shape)
        }
    }
}

extension BlockSparseRoutingMetal {
    package static func nsaThreeBranchAttention(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        config: Config,
        gateTensor: MLXArray? = nil,
        qOffset: Int? = nil
    ) -> MLXArray {
        runOnCPUIfFloat64([q, k, v, gateTensor]) {
            let geo = AttentionGeometry.resolve(q: q, k: k)
            guard geo.lQ > 0, geo.lK > 0, geo.totalQHeads > 0, geo.totalKVHeads > 0, geo.d > 0 else {
                return MLXArray.zeros(q.shape, dtype: q.dtype)
            }
            let k3D = k.asType(.float32).reshaped(geo.totalKVHeads, geo.lK, geo.d)
            let v3D = v.asType(.float32).reshaped(geo.totalKVHeads, geo.lK, geo.d)
            let centroids = PrecomputedCentroids(
                keys: blockCentroids(k3D, blockSize: config.blockSize),
                values: blockCentroids(v3D, blockSize: config.blockSize),
                blockSize: config.blockSize,
                qOffset: qOffset)
            return nsaThreeBranchAttentionWithCentroids(
                q: q, k: k, v: v, centroids: centroids, config: config, gateTensor: gateTensor)
        }
    }

    package static func nsaThreeBranchAttentionWithCentroids(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        centroids: PrecomputedCentroids,
        config: Config,
        gateTensor: MLXArray? = nil
    ) -> MLXArray {
        runOnCPUIfFloat64([q, k, v, centroids.keys, centroids.values, gateTensor]) {
            let origType = q.dtype
            let geo = AttentionGeometry.resolve(q: q, k: k)
            guard geo.lQ > 0, geo.lK > 0, geo.totalQHeads > 0, geo.totalKVHeads > 0, geo.d > 0 else {
                return MLXArray.zeros(q.shape, dtype: origType)
            }
            let effectiveOffset = centroids.qOffset ?? max(0, geo.lK - geo.lQ)
            let qGrouped = q.asType(.float32).reshaped(geo.totalKVHeads, geo.gqaGroup, geo.lQ, geo.d)
            let kComp = centroids.keys.asType(.float32).reshaped(geo.totalKVHeads, -1, geo.d)
            let vComp = centroids.values.asType(.float32).reshaped(geo.totalKVHeads, -1, geo.d)

            var compConfig = config
            compConfig.blockSize = centroids.blockSize
            let (outComp, compWeights, hasCompressed) = compressedBranch(
                qGrouped: qGrouped, kComp: kComp, vComp: vComp,
                config: compConfig, lK: geo.lK, qOffset: effectiveOffset)
            let selected = selectNSABlocks(
                compWeights: compWeights, blockSize: centroids.blockSize, topBlocks: config.topBlocks,
                lK: geo.lK, causal: config.causal, qOffset: effectiveOffset)
            let outSel = sparseSelectedAttention(
                q: q, k: k, v: v, selectedBlocks: selected,
                options: SparseOptions(
                    blockSize: centroids.blockSize, scale: config.scale,
                    causal: config.causal, qOffset: effectiveOffset)
            ).asType(.float32).reshaped(geo.totalQHeads, geo.lQ, geo.d)
            let outWin = windowBranch(
                qGrouped: qGrouped, k: k, v: v,
                config: config, qOffset: effectiveOffset)
            let combined = combineBranches(
                comp: outComp, sel: outSel, win: outWin,
                hasCompressed: hasCompressed, config: config, gateTensor: gateTensor)
            return combined.reshaped(q.shape).asType(origType)
        }
    }
}

extension BlockSparseRoutingMetal {
    private static func maskedSoftmax(_ scores: MLXArray, valid: MLXArray) -> MLXArray {
        let active = valid .&& (scores .!= MLXArray(-Float.infinity))
        let hasValid = active.any(axis: -1, keepDims: true)
        let masked = MLX.where(active, scores, MLXArray(-Float.infinity))
        let safe = MLX.where(hasValid, masked, MLXArray(Float(0)))
        return MLX.where(hasValid, softmax(safe, axis: -1), MLXArray(Float(0)))
    }

    private static func compressedBranch(
        qGrouped: MLXArray,
        kComp: MLXArray,
        vComp: MLXArray,
        config: Config,
        lK: Int,
        qOffset: Int
    ) -> (output: MLXArray, weights: MLXArray, hasCompressed: MLXArray) {
        let totalQHeads = qGrouped.dim(0) * qGrouped.dim(1)
        let lQ = qGrouped.dim(2)
        let d = qGrouped.dim(3)
        let numBlocks = kComp.dim(1)
        let logits = matmul(qGrouped, kComp.expandedDimensions(axis: 1).swappedAxes(-1, -2)) * config.scale
        let weights: MLXArray
        let hasCompressed: MLXArray
        if config.causal {
            let qPos = MLXArray(Int32(qOffset) ..< Int32(qOffset + lQ)).reshaped(1, 1, lQ, 1)
            let blockIds = MLXArray(Int32(0) ..< Int32(numBlocks)).reshaped(1, 1, 1, numBlocks)
            let blockEnd = MLX.minimum((blockIds + Int32(1)) * Int32(config.blockSize), MLXArray(Int32(lK))) - Int32(1)
            let valid = blockEnd .<= qPos
            hasCompressed = valid.any(axis: -1, keepDims: true).reshaped(1, lQ, 1)
            weights = maskedSoftmax(logits, valid: valid)
        } else {
            hasCompressed = MLXArray(numBlocks > 0).reshaped(1, 1, 1)
            weights = maskedSoftmax(logits, valid: logits .!= MLXArray(-Float.infinity))
        }
        let output = matmul(weights, vComp.expandedDimensions(axis: 1)).reshaped(totalQHeads, lQ, d)
        return (output, weights, hasCompressed)
    }

    private static func windowBranch(
        qGrouped: MLXArray,
        k: MLXArray,
        v: MLXArray,
        config: Config,
        qOffset: Int
    ) -> MLXArray {
        let kvHeads = qGrouped.dim(0)
        let totalQHeads = kvHeads * qGrouped.dim(1)
        let lQ = qGrouped.dim(2)
        let d = qGrouped.dim(3)
        let lK = k.dim(-2)
        let winStart = config.causal ? max(0, qOffset - config.slidingWindow + 1) : max(0, lK - config.slidingWindow)
        let winEnd = config.causal ? min(lK, qOffset + lQ) : lK
        guard winStart < winEnd else { return MLXArray.zeros([totalQHeads, lQ, d], dtype: .float32) }
        let winLen = winEnd - winStart
        let kWin = k[.ellipsis, winStart ..< winEnd, 0...].asType(.float32).reshaped(kvHeads, 1, winLen, d)
        let vWin = v[.ellipsis, winStart ..< winEnd, 0...].asType(.float32).reshaped(kvHeads, 1, winLen, d)
        let scores = matmul(qGrouped, kWin.swappedAxes(-1, -2)) * config.scale
        let weights: MLXArray
        if config.causal {
            let qPos = MLXArray(Int32(qOffset) ..< Int32(qOffset + lQ)).reshaped(1, 1, lQ, 1)
            let kPos = MLXArray(Int32(winStart) ..< Int32(winEnd)).reshaped(1, 1, 1, winLen)
            let valid = (kPos .<= qPos) .&& (kPos .> (qPos - Int32(clamping: config.slidingWindow)))
            weights = maskedSoftmax(scores, valid: valid)
        } else {
            weights = maskedSoftmax(scores, valid: scores .!= MLXArray(-Float.infinity))
        }
        return matmul(weights, vWin).reshaped(totalQHeads, lQ, d)
    }

    private static func combineBranches(
        comp: MLXArray,
        sel: MLXArray,
        win: MLXArray,
        hasCompressed: MLXArray,
        config: Config,
        gateTensor: MLXArray?
    ) -> MLXArray {
        let gCompBase: MLXArray
        let gSel: MLXArray
        let gWin: MLXArray
        if let gateTensor {
            let totalQHeads = comp.dim(0)
            let lQ = comp.dim(1)
            precondition(gateTensor.dim(-1) == 3 && gateTensor.size == totalQHeads * lQ * 3)
            let gates = gateTensor.asType(.float32).reshaped(totalQHeads, lQ, 3)
            (gCompBase, gSel, gWin) = (gates[0..., 0..., 0 ..< 1], gates[0..., 0..., 1 ..< 2], gates[0..., 0..., 2 ..< 3])
        } else {
            gCompBase = MLXArray(config.compressionGate)
            gSel = MLXArray(config.selectionGate)
            gWin = MLXArray(config.windowGate)
        }
        let gComp = MLX.where(hasCompressed, gCompBase, MLXArray(Float(0)))
        let baseSum = gCompBase + gSel + gWin
        let activeSum = gComp + gSel + gWin
        let safeDenom = MLX.where(activeSum .> 0, activeSum, MLXArray(Float(1)))
        let scale = MLX.where(activeSum .> 0, baseSum / safeDenom, MLXArray(Float(0)))
        return (comp * gComp + sel * gSel + win * gWin) * scale
    }
}

extension BlockSparseRoutingMetal {
    package static func sanitizeInt32Indices(_ raw: MLXArray, upperBound: Int) -> MLXArray {
        guard raw.dtype.isInteger, upperBound > 0 else {
            return MLXArray.full(raw.shape, values: MLXArray(Int32(-1)))
        }
        let ok = (raw .>= 0) .&& (raw .< upperBound)
        return MLX.where(ok, raw.asType(.int32), MLXArray(Int32(-1)))
    }

    private static func sparseSelectedAttentionMLX(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        selectedBlocks: MLXArray,
        options: SparseOptions
    ) -> MLXArray {
        let origType = q.dtype
        let geo = AttentionGeometry.resolve(q: q, k: k)
        let topK = selectedBlocks.dim(-1)
        let selHeads = selectedBlocks.size / max(1, geo.lQ * topK)
        let selGroup = geo.totalQHeads / selHeads
        let effectiveOffset = options.qOffset ?? max(0, geo.lK - geo.lQ)
        let qF = q.asType(.float32).reshaped(geo.totalQHeads, geo.lQ, geo.d)
        let kExp = MLX.broadcast(
            k.asType(.float32).reshaped(geo.totalKVHeads, 1, geo.lK, geo.d),
            to: [geo.totalKVHeads, geo.gqaGroup, geo.lK, geo.d]
        ).reshaped(geo.totalQHeads, geo.lK, geo.d)
        let vExp = MLX.broadcast(
            v.asType(.float32).reshaped(geo.totalKVHeads, 1, geo.lK, geo.d),
            to: [geo.totalKVHeads, geo.gqaGroup, geo.lK, geo.d]
        ).reshaped(geo.totalQHeads, geo.lK, geo.d)
        let numBlocks = (geo.lK + options.blockSize - 1) / max(1, options.blockSize)
        let selExp = MLX.broadcast(
            sanitizeInt32Indices(selectedBlocks, upperBound: numBlocks).reshaped(selHeads, 1, geo.lQ, topK),
            to: [selHeads, selGroup, geo.lQ, topK]
        ).reshaped(geo.totalQHeads, geo.lQ, 1, topK)
        let kPos = MLXArray(Int32(0) ..< Int32(geo.lK))
        let kBlock = floorDivide(kPos, Int32(options.blockSize)).reshaped(1, 1, geo.lK, 1)
        var valid = ((selExp .>= Int32(0)) .&& (selExp .== kBlock)).any(axis: -1)
        if options.causal {
            let qPos = MLXArray(Int32(effectiveOffset) ..< Int32(effectiveOffset + geo.lQ)).reshaped(1, geo.lQ, 1)
            valid = valid .&& (kPos.reshaped(1, 1, geo.lK) .<= qPos)
        }
        let scores = matmul(qF, kExp.swappedAxes(-1, -2)) * options.scale
        let weights = maskedSoftmax(scores, valid: valid)
        return matmul(weights, vExp).reshaped(q.shape).asType(origType)
    }
}

private let mobaSparseSource = """
    const int H = int(shapes[0]);
    const int LQ = int(shapes[1]);
    const int LK = int(shapes[2]);
    const int D = int(shapes[3]);
    const int BSIZE = int(shapes[4]);
    const int TOPK = int(shapes[5]);
    const int GQA_G = int(shapes[6]);
    const int SEL_G = int(shapes[7]);
    const int CAUSAL = int(shapes[8]);
    const int Q_OFF = int(shapes[9]);
    const int lane = int(thread_index_in_simdgroup);
    const int qi = int(threadgroup_position_in_grid.y);
    const int h = int(threadgroup_position_in_grid.z);
    if (h >= H || qi >= LQ) { return; }
    const int kv_h = h / GQA_G;
    const int sel_h = h / SEL_G;
    const int max_tok = (CAUSAL != 0) ? min(LK - 1, Q_OFF + qi) : (LK - 1);
    const float attn_scale = params[0];

    float q_reg[BLOCKS];
    float o_acc[BLOCKS];
    auto q_ptr = q_in + (size_t(h) * size_t(LQ) + size_t(qi)) * size_t(D);
    for (int b = 0; b < BLOCKS; ++b) {
        q_reg[b] = float(q_ptr[b * 32 + lane]);
        o_acc[b] = 0.0f;
    }

    auto sel_ptr = sel_blocks + (size_t(sel_h) * size_t(LQ) + size_t(qi)) * size_t(TOPK);
    float m_run = -INFINITY;
    float l_run = 0.0f;
    for (int sb = 0; sb < TOPK; ++sb) {
        const int blk_id = int(sel_ptr[sb]);
        if (blk_id < 0) { continue; }
        bool dup = false;
        for (int prev = 0; prev < sb; ++prev) {
            if (int(sel_ptr[prev]) == blk_id) { dup = true; break; }
        }
        if (dup) { continue; }
        const int64_t start_tok = int64_t(blk_id) * int64_t(BSIZE);
        if (start_tok > int64_t(max_tok)) { continue; }
        const int end_tok = int(min(int64_t(max_tok) + 1, min(int64_t(LK), start_tok + int64_t(BSIZE))));
        for (int tok = int(start_tok); tok < end_tok; ++tok) {
            auto k_ptr = k_in + (size_t(kv_h) * size_t(LK) + size_t(tok)) * size_t(D);
            auto v_ptr = v_in + (size_t(kv_h) * size_t(LK) + size_t(tok)) * size_t(D);
            float dot_part = 0.0f;
            for (int b = 0; b < BLOCKS; ++b) {
                dot_part = fma(q_reg[b], float(k_ptr[b * 32 + lane]), dot_part);
            }
            const float s = simd_sum(dot_part) * attn_scale;
            const bool bad = isnan(m_run) || isnan(s);
            const float m_new = bad ? NAN : max(m_run, s);
            const bool active = bad || (m_new != -INFINITY);
            const float alpha = bad ? NAN : ((m_run != -INFINITY && active) ? exp(m_run - m_new) : 0.0f);
            const float beta = bad ? NAN : ((active && s != -INFINITY) ? exp(s - m_new) : 0.0f);
            l_run = l_run * alpha + beta;
            m_run = m_new;
            for (int b = 0; b < BLOCKS; ++b) {
                o_acc[b] = fma(beta, float(v_ptr[b * 32 + lane]), o_acc[b] * alpha);
            }
        }
    }

    const float inv_l = (isnan(l_run) || isnan(m_run)) ? NAN : ((l_run > 0.0f) ? (1.0f / l_run) : 0.0f);
    device T* out_ptr = out + (size_t(h) * size_t(LQ) + size_t(qi)) * size_t(D);
    for (int b = 0; b < BLOCKS; ++b) {
        out_ptr[b * 32 + lane] = static_cast<T>(o_acc[b] * inv_l);
    }
    """

private let mobaSparseKernel = MLXFast.metalKernel(
    name: "moba_nsa_sparse_attn_metal",
    inputNames: ["q_in", "k_in", "v_in", "sel_blocks", "params", "shapes"],
    outputNames: ["out"],
    source: mobaSparseSource,
    header: BasaltMetalSubstrate.combinedHeader
)
