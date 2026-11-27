import Foundation
import MLX
import MLXLMCommon

package enum DualBlockFlashAttentionKernel {
    package static let supportedHeadDims: Set<Int> = [64, 128]
    package static let supportedDTypes: Set<DType> = [.bfloat16, .float16, .float32]
    package static let blockQ = 16
    package static let maxThreadgroupMemoryBytes = 32_768
    package static let minDispatchQueryLen = 16
    package static let minDispatchTotalK = 512
    package static let disableEnvironmentKey = "BASALT_DISABLE_DUAL_BLOCK_FLASH_ATTN"
    package static let forceEnvironmentKey = "BASALT_FORCE_DUAL_BLOCK_FLASH_ATTN"

    package struct Segments {
        package let prefixKeys: MLXArray?
        package let prefixValues: MLXArray?
        package let activeKeys: MLXArray
        package let activeValues: MLXArray
        package let suffixKeys: MLXArray?
        package let suffixValues: MLXArray?

        package init(
            prefixKeys: MLXArray?,
            prefixValues: MLXArray?,
            activeKeys: MLXArray,
            activeValues: MLXArray,
            suffixKeys: MLXArray?,
            suffixValues: MLXArray?
        ) {
            self.prefixKeys = prefixKeys
            self.prefixValues = prefixValues
            self.activeKeys = activeKeys
            self.activeValues = activeValues
            self.suffixKeys = suffixKeys
            self.suffixValues = suffixValues
        }
    }

    package static let environmentDefaultMode: DualBlockFlashKernelMode = {
        let env = ProcessInfo.processInfo.environment
        if env[disableEnvironmentKey] == "1" { return .disabled }
        if env[forceEnvironmentKey] == "1" { return .forced }
        return .auto
    }()

    package static func blockK(for dtype: DType) -> Int {
        dtype == .float32 ? 8 : 16
    }

    package static func padKV(for dtype: DType) -> Int {
        dtype == .float32 ? 4 : 8
    }

    package static func blockQ(for qLen: Int) -> Int {
        qLen <= 8 ? 8 : blockQ
    }

    package static func threadgroupMemoryBytes(headDim: Int, dtype: DType, qLen: Int = blockQ) -> Int {
        let bk = blockK(for: dtype)
        let bq = blockQ(for: qLen)
        let pad = padKV(for: dtype)
        let elemBytes = dtype == .float32 ? 4 : 2
        let kvBytes = bk * (headDim + pad) * elemBytes
        let xchgBytes = (bq / 8) * 2 * (bk / 8) * 64 * MemoryLayout<Float>.size
        return kvBytes + xchgBytes
    }

    package static func isEligible(
        queries: MLXArray,
        segments: Segments,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        mode: DualBlockFlashKernelMode = environmentDefaultMode
    ) -> Bool {
        guard mode != .disabled, Device.defaultDevice() == .gpu else { return false }
        guard case .none = mask else { return false }
        guard let dims = validated(queries: queries, segments: segments) else { return false }
        let totalK = dims.prefixLen + dims.activeLen + dims.suffixLen
        guard dims.qLen > 0, totalK > 0 else { return false }
        if mode == .forced { return true }
        return dims.qLen >= minDispatchQueryLen || totalK >= minDispatchTotalK
    }

    package static func attention(
        queries: MLXArray,
        segments: Segments,
        scale: Float
    ) -> MLXArray {
        if queries.shape.contains(0) {
            let dV = segments.activeValues.ndim >= 1 ? segments.activeValues.dim(-1) : (queries.ndim >= 1 ? queries.dim(-1) : 0)
            let outShape = queries.ndim >= 1 ? Array(queries.shape.dropLast()) + [dV] : queries.shape
            return MLXArray.zeros(outShape, dtype: queries.dtype, stream: queries.dtype == .float64 ? .cpu : .default)
        }
        guard Device.defaultDevice() == .gpu,
              let dims = validated(queries: queries, segments: segments)
        else {
            return referenceAttention(queries: queries, segments: segments, scale: scale)
        }
        let totalK = dims.prefixLen + dims.activeLen + dims.suffixLen
        guard totalK > 0 else {
            return MLXArray.zeros([dims.batch, dims.qHeads, dims.qLen, dims.headDim], dtype: dims.dtype)
        }
        return launchKernel(queries: queries, segments: segments, dims: dims, scale: scale)
    }
}

extension DualBlockFlashAttentionKernel {
    private struct ValidatedDims {
        let batch: Int
        let qHeads: Int
        let kvHeads: Int
        let qLen: Int
        let prefixLen: Int
        let activeLen: Int
        let suffixLen: Int
        let headDim: Int
        let dtype: DType
    }

    private static func validated(queries: MLXArray, segments: Segments) -> ValidatedDims? {
        guard queries.ndim == 4, segments.activeKeys.ndim == 4, segments.activeValues.ndim == 4 else { return nil }
        let headDim = queries.dim(3)
        let dtype = queries.dtype
        guard supportedHeadDims.contains(headDim), supportedDTypes.contains(dtype) else { return nil }
        let (batch, qHeads, qLen) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let (kvHeads, activeLen) = (segments.activeKeys.dim(1), segments.activeKeys.dim(2))
        guard batch > 0, qHeads > 0, kvHeads > 0, qHeads.isMultiple(of: kvHeads), qLen >= 0, activeLen >= 0 else { return nil }
        guard matchesSegment(segments.activeKeys, segments.activeValues, batch: batch, kvHeads: kvHeads, headDim: headDim, dtype: dtype) else { return nil }
        guard let prefixLen = optionalSegmentLength(segments.prefixKeys, segments.prefixValues, batch: batch, kvHeads: kvHeads, headDim: headDim, dtype: dtype),
              let suffixLen = optionalSegmentLength(segments.suffixKeys, segments.suffixValues, batch: batch, kvHeads: kvHeads, headDim: headDim, dtype: dtype)
        else { return nil }
        return ValidatedDims(
            batch: batch, qHeads: qHeads, kvHeads: kvHeads, qLen: qLen,
            prefixLen: prefixLen, activeLen: activeLen, suffixLen: suffixLen,
            headDim: headDim, dtype: dtype
        )
    }

    private static func matchesSegment(
        _ keys: MLXArray,
        _ values: MLXArray,
        batch: Int,
        kvHeads: Int,
        headDim: Int,
        dtype: DType
    ) -> Bool {
        keys.ndim == 4 && values.ndim == 4
            && keys.dim(0) == batch && values.dim(0) == batch
            && keys.dim(1) == kvHeads && values.dim(1) == kvHeads
            && keys.dim(2) == values.dim(2)
            && keys.dim(3) == headDim && values.dim(3) == headDim
            && keys.dtype == dtype && values.dtype == dtype
    }

    private static func launchKernel(
        queries: MLXArray,
        segments: Segments,
        dims: ValidatedDims,
        scale: Float
    ) -> MLXArray {
        let dummy = MLXArray.zeros([1, 1, 1, dims.headDim], dtype: dims.dtype)
        let kPre = dims.prefixLen > 0 ? (segments.prefixKeys ?? dummy) : dummy
        let vPre = dims.prefixLen > 0 ? (segments.prefixValues ?? dummy) : dummy
        let kAct = dims.activeLen > 0 ? segments.activeKeys : dummy
        let vAct = dims.activeLen > 0 ? segments.activeValues : dummy
        let kSuf = dims.suffixLen > 0 ? (segments.suffixKeys ?? dummy) : dummy
        let vSuf = dims.suffixLen > 0 ? (segments.suffixValues ?? dummy) : dummy
        let bq = blockQ(for: dims.qLen)
        let threads = (bq / 8) * 64
        let nQ = (dims.qLen + bq - 1) / bq
        let shapes = MLXArray([
            Int32(dims.batch), Int32(dims.qHeads), Int32(dims.kvHeads), Int32(dims.qLen),
            Int32(dims.prefixLen), Int32(dims.activeLen), Int32(dims.suffixLen), Int32(nQ),
        ])
        return dualBlockFlashAttnMetalKernel(
            [queries, kPre, vPre, kAct, vAct, kSuf, vSuf, MLXArray([scale]), shapes],
            template: [
                ("T", dims.dtype),
                ("D", dims.headDim),
                ("BQ", bq),
                ("BK", blockK(for: dims.dtype)),
                ("PAD_KV", padKV(for: dims.dtype)),
            ],
            grid: (threads, nQ, dims.batch * dims.qHeads),
            threadGroup: (threads, 1, 1),
            outputShapes: [[dims.batch, dims.qHeads, dims.qLen, dims.headDim]],
            outputDTypes: [dims.dtype]
        )[0]
    }

    private static func referenceAttention(
        queries: MLXArray,
        segments: Segments,
        scale: Float
    ) -> MLXArray {
        let usesFloat64 = [
            queries, segments.activeKeys, segments.activeValues,
            segments.prefixKeys, segments.prefixValues, segments.suffixKeys, segments.suffixValues,
        ].compactMap { $0 }.contains { $0.dtype == .float64 }
        if usesFloat64 && Device.defaultDevice() == .gpu {
            return Device.withDefaultDevice(.cpu) {
                referenceAttention(queries: queries, segments: segments, scale: scale)
            }
        }
        let keyParts = [segments.prefixKeys, segments.activeKeys, segments.suffixKeys]
            .compactMap { $0 }
            .filter { $0.dim(2) > 0 }
        let valueParts = [segments.prefixValues, segments.activeValues, segments.suffixValues]
            .compactMap { $0 }
            .filter { $0.dim(2) > 0 }
        guard !keyParts.isEmpty, !valueParts.isEmpty, !queries.shape.contains(0) else {
            let dV = segments.activeValues.ndim >= 1 ? segments.activeValues.dim(-1) : (queries.ndim >= 1 ? queries.dim(-1) : 0)
            let outShape = queries.ndim >= 1 ? Array(queries.shape.dropLast()) + [dV] : queries.shape
            return MLXArray.zeros(outShape, dtype: queries.dtype)
        }
        let keys = keyParts.count == 1 ? keyParts[0] : concatenated(keyParts, axis: 2)
        let values = valueParts.count == 1 ? valueParts[0] : concatenated(valueParts, axis: 2)
        let isRealFloat: (DType) -> Bool = { $0.isFloatingPoint && !$0.isComplex }
        if isRealFloat(queries.dtype) && isRealFloat(keys.dtype) && isRealFloat(values.dtype) {
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: .none
            )
        }
        let cType: DType = usesFloat64 ? .float64 : .float32
        return MLXFast.scaledDotProductAttention(
            queries: queries.asType(cType), keys: keys.asType(cType), values: values.asType(cType),
            scale: scale, mask: .none
        ).asType(queries.dtype)
    }

    private static func optionalSegmentLength(
        _ keys: MLXArray?,
        _ values: MLXArray?,
        batch: Int,
        kvHeads: Int,
        headDim: Int,
        dtype: DType
    ) -> Int? {
        switch (keys, values) {
        case (nil, nil):
            return 0
        case let (k?, v?):
            guard matchesSegment(k, v, batch: batch, kvHeads: kvHeads, headDim: headDim, dtype: dtype) else { return nil }
            return k.dim(2)
        default:
            return nil
        }
    }
}

private enum DualBlockFlashAttentionMSL {
    static let header = """
        #include <metal_simdgroup>
        #include <metal_simdgroup_matrix>

        static inline float dual_nan_max(float a, float b) {
            return (isnan(a) || isnan(b)) ? NAN : max(a, b);
        }
        """

    static let source = """
        constexpr int WM = BQ / 8;
        constexpr int WN = 2;
        constexpr int DH = D / WN;
        constexpr int TDH = DH / 8;
        constexpr int TK = BK / 8;
        constexpr int LD_KV = D + PAD_KV;
        constexpr float LOG2E = 1.4426950408889634f;

        const int B = shapes[0];
        const int H_q = shapes[1];
        const int H_kv = shapes[2];
        const int N_q = shapes[3];
        const int N_pre = shapes[4];
        const int N_act = shapes[5];
        const int N_suf = shapes[6];
        const int NQ = shapes[7];
        const int BH = B * H_q;

        const int tg_q = int(threadgroup_position_in_grid.y);
        const int bh = int(threadgroup_position_in_grid.z);
        if (tg_q >= NQ || bh >= BH) { return; }

        const int sg = int(simdgroup_index_in_threadgroup);
        const int lane = int(thread_index_in_simdgroup);
        const int row_group = sg / WN;
        const int d_part = sg % WN;
        const int qid = lane / 4;
        const int sm = (qid & 4) + ((lane / 2) % 4);
        const int sn = (qid & 2) * 2 + (lane % 2) * 2;

        const int b = bh / H_q;
        const int h_q = bh % H_q;
        const int h_kv = h_q / (H_q / H_kv);

        const int q_base = tg_q * BQ + row_group * 8;
        const int q_row = q_base + sm;
        const bool valid_q = (q_row < N_q);

        const int64_t q_head_base = int64_t(b) * q_strides[0] + int64_t(h_q) * q_strides[1];
        const int64_t q_s2 = q_strides[2];
        const int64_t q_s3 = q_strides[3];

        T q_reg[TDH][2];
        simdgroup_float8x8 o_acc[TDH];
        for (int dd = 0; dd < TDH; ++dd) {
            o_acc[dd] = simdgroup_float8x8(0.0f);
            const int d_col = d_part * DH + dd * 8 + sn;
            if (valid_q) {
                const int64_t q_off = q_head_base + int64_t(q_row) * q_s2 + int64_t(d_col) * q_s3;
                q_reg[dd][0] = q[q_off];
                q_reg[dd][1] = q[q_off + q_s3];
            } else {
                q_reg[dd][0] = static_cast<T>(0.0f);
                q_reg[dd][1] = static_cast<T>(0.0f);
            }
        }

        threadgroup T kv_smem[BK * LD_KV];
        threadgroup float s_xchg[WM * WN * TK * 64];

        const float scale_log2 = scale_arr[0] * LOG2E;
        float m_run = -INFINITY;
        float l_run = 0.0f;
        const int xchg_base = (row_group * WN) * TK * 64 + lane * 2;
        const int my_xchg = (row_group * WN + d_part) * TK * 64 + lane * 2;

        for (int seg = 0; seg < 3; ++seg) {
            const int seg_len = (seg == 0) ? N_pre : ((seg == 1) ? N_act : N_suf);
            if (seg_len <= 0) { continue; }

            const int64_t k_head_base = (seg == 0)
                ? (int64_t(b) * k_pre_strides[0] + int64_t(h_kv) * k_pre_strides[1])
                : ((seg == 1)
                    ? (int64_t(b) * k_act_strides[0] + int64_t(h_kv) * k_act_strides[1])
                    : (int64_t(b) * k_suf_strides[0] + int64_t(h_kv) * k_suf_strides[1]));
            const int64_t k_s2 = (seg == 0) ? k_pre_strides[2] : ((seg == 1) ? k_act_strides[2] : k_suf_strides[2]);
            const int64_t k_s3 = (seg == 0) ? k_pre_strides[3] : ((seg == 1) ? k_act_strides[3] : k_suf_strides[3]);
            const int64_t v_head_base = (seg == 0)
                ? (int64_t(b) * v_pre_strides[0] + int64_t(h_kv) * v_pre_strides[1])
                : ((seg == 1)
                    ? (int64_t(b) * v_act_strides[0] + int64_t(h_kv) * v_act_strides[1])
                    : (int64_t(b) * v_suf_strides[0] + int64_t(h_kv) * v_suf_strides[1]));
            const int64_t v_s2 = (seg == 0) ? v_pre_strides[2] : ((seg == 1) ? v_act_strides[2] : v_suf_strides[2]);
            const int64_t v_s3 = (seg == 0) ? v_pre_strides[3] : ((seg == 1) ? v_act_strides[3] : v_suf_strides[3]);
            const int64_t k_step = int64_t(32) * k_s3;
            const int64_t v_step = int64_t(32) * v_s3;
            const int NK = (seg_len + BK - 1) / BK;

            for (int kb = 0; kb < NK; ++kb) {
                const int k_base = kb * BK;
                for (int r = sg; r < BK; r += WM * WN) {
                    const int k_idx = k_base + r;
                    const int dst_row = r * LD_KV;
                    if (k_idx < seg_len) {
                        int64_t src = k_head_base + int64_t(k_idx) * k_s2 + int64_t(lane) * k_s3;
                        for (int c = lane; c < D; c += 32) {
                            kv_smem[dst_row + c] = (seg == 0) ? k_pre[src] : ((seg == 1) ? k_act[src] : k_suf[src]);
                            src += k_step;
                        }
                    } else {
                        for (int c = lane; c < D; c += 32) {
                            kv_smem[dst_row + c] = static_cast<T>(0.0f);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_float8x8 s_mat[TK];
                for (int ik = 0; ik < TK; ++ik) {
                    s_mat[ik] = simdgroup_float8x8(0.0f);
                }
                for (int dd = 0; dd < TDH; ++dd) {
                    simdgroup_float8x8 q_mat;
                    q_mat.thread_elements()[0] = float(q_reg[dd][0]);
                    q_mat.thread_elements()[1] = float(q_reg[dd][1]);
                    const int d_off = d_part * DH + dd * 8 + sm;
                    for (int ik = 0; ik < TK; ++ik) {
                        const int k0 = ik * 8 + sn;
                        simdgroup_float8x8 k_mat;
                        k_mat.thread_elements()[0] = float(kv_smem[k0 * LD_KV + d_off]);
                        k_mat.thread_elements()[1] = float(kv_smem[(k0 + 1) * LD_KV + d_off]);
                        simdgroup_multiply_accumulate(s_mat[ik], q_mat, k_mat, s_mat[ik]);
                    }
                }

                for (int ik = 0; ik < TK; ++ik) {
                    s_xchg[my_xchg + ik * 64] = s_mat[ik].thread_elements()[0];
                    s_xchg[my_xchg + ik * 64 + 1] = s_mat[ik].thread_elements()[1];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float s_full[TK][2];
                float tile_max = -INFINITY;
                for (int ik = 0; ik < TK; ++ik) {
                    const int idx0 = xchg_base + ik * 64;
                    const int idx1 = xchg_base + (TK + ik) * 64;
                    float v0 = (s_xchg[idx0] + s_xchg[idx1]) * scale_log2;
                    float v1 = (s_xchg[idx0 + 1] + s_xchg[idx1 + 1]) * scale_log2;
                    const int k_col0 = k_base + ik * 8 + sn;
                    if (k_col0 >= seg_len) { v0 = -INFINITY; }
                    if (k_col0 + 1 >= seg_len) { v1 = -INFINITY; }
                    s_full[ik][0] = v0;
                    s_full[ik][1] = v1;
                    tile_max = dual_nan_max(tile_max, dual_nan_max(v0, v1));
                }

                tile_max = dual_nan_max(tile_max, simd_shuffle_xor(tile_max, ushort(1)));
                tile_max = dual_nan_max(tile_max, simd_shuffle_xor(tile_max, ushort(8)));
                const float m_new = dual_nan_max(m_run, tile_max);
                const bool bad_row = isnan(m_new);
                const bool active_row = bad_row || (m_new != -INFINITY);
                const float alpha = bad_row ? NAN : ((m_run != -INFINITY && active_row) ? exp2(m_run - m_new) : 0.0f);
                float tile_sum = 0.0f;
                simdgroup_float8x8 p_mat[TK];
                for (int ik = 0; ik < TK; ++ik) {
                    const float p0 = bad_row ? NAN : ((active_row && s_full[ik][0] != -INFINITY) ? exp2(s_full[ik][0] - m_new) : 0.0f);
                    const float p1 = bad_row ? NAN : ((active_row && s_full[ik][1] != -INFINITY) ? exp2(s_full[ik][1] - m_new) : 0.0f);
                    p_mat[ik].thread_elements()[0] = p0;
                    p_mat[ik].thread_elements()[1] = p1;
                    tile_sum += p0 + p1;
                }
                tile_sum += simd_shuffle_xor(tile_sum, ushort(1));
                tile_sum += simd_shuffle_xor(tile_sum, ushort(8));
                l_run = l_run * alpha + tile_sum;
                m_run = m_new;
                for (int dd = 0; dd < TDH; ++dd) {
                    o_acc[dd].thread_elements()[0] *= alpha;
                    o_acc[dd].thread_elements()[1] *= alpha;
                }

                for (int r = sg; r < BK; r += WM * WN) {
                    const int k_idx = k_base + r;
                    const int dst_row = r * LD_KV;
                    if (k_idx < seg_len) {
                        int64_t src = v_head_base + int64_t(k_idx) * v_s2 + int64_t(lane) * v_s3;
                        for (int c = lane; c < D; c += 32) {
                            kv_smem[dst_row + c] = (seg == 0) ? v_pre[src] : ((seg == 1) ? v_act[src] : v_suf[src]);
                            src += v_step;
                        }
                    } else {
                        for (int c = lane; c < D; c += 32) {
                            kv_smem[dst_row + c] = static_cast<T>(0.0f);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (int ik = 0; ik < TK; ++ik) {
                    const int v_row_off = (ik * 8 + sm) * LD_KV + d_part * DH + sn;
                    for (int dd = 0; dd < TDH; ++dd) {
                        simdgroup_float8x8 v_mat;
                        const int off = v_row_off + dd * 8;
                        v_mat.thread_elements()[0] = float(kv_smem[off]);
                        v_mat.thread_elements()[1] = float(kv_smem[off + 1]);
                        simdgroup_multiply_accumulate(o_acc[dd], p_mat[ik], v_mat, o_acc[dd]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        if (valid_q) {
            const float inv_l = (isnan(l_run) || isnan(m_run)) ? NAN : ((l_run > 0.0f) ? (1.0f / l_run) : 0.0f);
            const size_t o_base = (size_t(bh) * size_t(N_q) + size_t(q_row)) * size_t(D);
            for (int dd = 0; dd < TDH; ++dd) {
                const int d_col = d_part * DH + dd * 8 + sn;
                out[o_base + size_t(d_col)] = static_cast<T>(o_acc[dd].thread_elements()[0] * inv_l);
                out[o_base + size_t(d_col + 1)] = static_cast<T>(o_acc[dd].thread_elements()[1] * inv_l);
            }
        }
        """
}

private let dualBlockFlashAttnMetalKernel = MLXFast.metalKernel(
    name: "basalt_dual_block_flash_attn",
    inputNames: ["q", "k_pre", "v_pre", "k_act", "v_act", "k_suf", "v_suf", "scale_arr", "shapes"],
    outputNames: ["out"],
    source: DualBlockFlashAttentionMSL.source,
    header: DualBlockFlashAttentionMSL.header,
    ensureRowContiguous: false
)
