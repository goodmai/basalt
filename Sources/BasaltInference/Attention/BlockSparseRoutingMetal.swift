import BasaltCore
import Foundation
import MLX

package enum BlockSparseRoutingMetal {
    package static let supportedDTypes: Set<DType> = [.bfloat16, .float16, .float32]
    package static let maxHeadDim = 256

    package struct Config: Sendable {
        package var blockSize: Int
        package var topBlocks: Int
        package var slidingWindow: Int
        package var scale: Float
        package var compressionGate: Float
        package var selectionGate: Float
        package var windowGate: Float

        package init(
            blockSize: Int = 16,
            topBlocks: Int = 2,
            slidingWindow: Int = 16,
            scale: Float,
            gates: [Float] = [0.2, 0.5, 0.3]
        ) {
            self.blockSize = max(1, blockSize)
            self.topBlocks = max(1, topBlocks)
            self.slidingWindow = max(1, slidingWindow)
            self.scale = scale
            self.compressionGate = gates.count > 0 ? gates[0] : 0.2
            self.selectionGate = gates.count > 1 ? gates[1] : 0.5
            self.windowGate = gates.count > 2 ? gates[2] : 0.3
        }
    }

    package static func selectMoBABlocks(
        q: MLXArray,
        k: MLXArray,
        blockSize: Int,
        topBlocks: Int
    ) -> MLXArray {
        let lQ = q.dim(-2)
        let lK = k.dim(-2)
        let d = q.dim(-1)
        let heads = q.size / max(1, lQ * d)
        let numBlocks = lK / blockSize
        let kCentroids = mean(k.asType(.float32).reshaped(heads, numBlocks, blockSize, d), axis: 2)
        let q3D = q.asType(.float32).reshaped(heads, lQ, d)
        let blockScores = matmul(q3D, kCentroids.swappedAxes(-1, -2))
        let kSel = min(topBlocks, numBlocks)
        return argSort(-blockScores, axis: -1)[.ellipsis, 0 ..< kSel].asType(.int32)
    }

    package static func sparseSelectedAttention(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        selectedBlocks: MLXArray,
        blockSize: Int,
        scale: Float
    ) -> MLXArray {
        let origType = q.dtype
        let lQ = q.dim(-2)
        let lK = k.dim(-2)
        let d = q.dim(-1)
        let heads = q.size / max(1, lQ * d)
        let topK = selectedBlocks.dim(-1)
        precondition(d.isMultiple(of: 32) && d <= maxHeadDim)
        let params = MLXArray([scale])
        let shapes = MLXArray([Int32(heads), Int32(lQ), Int32(lK), Int32(d), Int32(blockSize), Int32(topK)])
        let out = mobaSparseKernel(
            [
                q.reshaped(heads, lQ, d),
                k.reshaped(heads, lK, d),
                v.reshaped(heads, lK, d),
                selectedBlocks.asType(.int32).reshaped(heads, lQ, topK),
                params,
                shapes,
            ],
            template: [("T", origType), ("BLOCKS", d / 32)],
            grid: (32, lQ, heads),
            threadGroup: (32, 1, 1),
            outputShapes: [[heads, lQ, d]],
            outputDTypes: [origType]
        )[0]
        return out.reshaped(q.shape)
    }

    package static func nsaThreeBranchAttention(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        config: Config
    ) -> MLXArray {
        let origType = q.dtype
        let lQ = q.dim(-2)
        let lK = k.dim(-2)
        let d = q.dim(-1)
        let heads = q.size / max(1, lQ * d)
        let numBlocks = lK / config.blockSize
        let q3D = q.asType(.float32).reshaped(heads, lQ, d)
        let k3D = k.asType(.float32).reshaped(heads, lK, d)
        let v3D = v.asType(.float32).reshaped(heads, lK, d)

        let kComp = mean(k3D.reshaped(heads, numBlocks, config.blockSize, d), axis: 2)
        let vComp = mean(v3D.reshaped(heads, numBlocks, config.blockSize, d), axis: 2)
        let compWeights = softmax(matmul(q3D, kComp.swappedAxes(-1, -2)) * config.scale, axis: -1)
        let outComp = matmul(compWeights, vComp)

        let selected = selectMoBABlocks(q: q3D, k: k3D, blockSize: config.blockSize, topBlocks: config.topBlocks)
        let outSel = sparseSelectedAttention(
            q: q.reshaped(heads, lQ, d), k: k.reshaped(heads, lK, d), v: v.reshaped(heads, lK, d),
            selectedBlocks: selected, blockSize: config.blockSize, scale: config.scale
        ).asType(.float32)

        let winStart = max(0, lK - config.slidingWindow)
        let kWin = k3D[0..., winStart ..< lK, 0...]
        let vWin = v3D[0..., winStart ..< lK, 0...]
        let winWeights = softmax(matmul(q3D, kWin.swappedAxes(-1, -2)) * config.scale, axis: -1)
        let outWin = matmul(winWeights, vWin)

        let combined = outComp * config.compressionGate + outSel * config.selectionGate + outWin * config.windowGate
        return combined.reshaped(q.shape).asType(origType)
    }
}

private let mobaSparseSource = """
    const int H = int(shapes[0]);
    const int LQ = int(shapes[1]);
    const int LK = int(shapes[2]);
    const int D = int(shapes[3]);
    const int BSIZE = int(shapes[4]);
    const int TOPK = int(shapes[5]);
    const int lane = int(thread_index_in_simdgroup);
    const int qi = int(threadgroup_position_in_grid.y);
    const int h = int(threadgroup_position_in_grid.z);
    if (h >= H || qi >= LQ) { return; }
    const float attn_scale = params[0];

    float q_reg[BLOCKS];
    float o_acc[BLOCKS];
    const device T* q_ptr = q_in + (size_t(h) * size_t(LQ) + size_t(qi)) * size_t(D);
    for (int b = 0; b < BLOCKS; ++b) {
        q_reg[b] = float(q_ptr[b * 32 + lane]);
        o_acc[b] = 0.0f;
    }

    auto sel_ptr = sel_blocks + (size_t(h) * size_t(LQ) + size_t(qi)) * size_t(TOPK);
    float m_run = -INFINITY;
    float l_run = 0.0f;
    for (int sb = 0; sb < TOPK; ++sb) {
        const int blk_id = int(sel_ptr[sb]);
        if (blk_id < 0) { continue; }
        const int start_tok = blk_id * BSIZE;
        const int end_tok = min(LK, start_tok + BSIZE);
        for (int tok = start_tok; tok < end_tok; ++tok) {
            const device T* k_ptr = k_in + (size_t(h) * size_t(LK) + size_t(tok)) * size_t(D);
            const device T* v_ptr = v_in + (size_t(h) * size_t(LK) + size_t(tok)) * size_t(D);
            float dot_part = 0.0f;
            for (int b = 0; b < BLOCKS; ++b) {
                dot_part = fma(q_reg[b], float(k_ptr[b * 32 + lane]), dot_part);
            }
            const float s = simd_sum(dot_part) * attn_scale;
            const float m_new = max(m_run, s);
            const float alpha = (m_run == -INFINITY) ? 0.0f : exp(m_run - m_new);
            const float beta = exp(s - m_new);
            l_run = l_run * alpha + beta;
            m_run = m_new;
            for (int b = 0; b < BLOCKS; ++b) {
                o_acc[b] = fma(beta, float(v_ptr[b * 32 + lane]), o_acc[b] * alpha);
            }
        }
    }

    const float inv_l = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
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
