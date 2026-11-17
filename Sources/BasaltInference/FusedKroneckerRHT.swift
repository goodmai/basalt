import Foundation
import MLX

package enum FusedKroneckerRHT {
    package static let dtypes: Set<DType> = [.bfloat16, .float16, .float32]
    package static let maxMetalHadamardWidth = 4096
    package static let maxMetalOuter = 64

    package struct Spec: Sendable, Equatable {
        package let dimension: Int
        package let seed: UInt64
        package let outer: Int
        package let hadamardWidth: Int
        package let hadamardScale: Float
        package let signsValues: [Float]
        package let packedSignsValues: [UInt32]
        package let qValues: [Float]?

        package static func decompose(_ d: Int) -> (outer: Int, hadamardWidth: Int)? {
            guard d >= 2, d.isMultiple(of: 2) else { return nil }
            let tz = d.trailingZeroBitCount
            return (d >> tz, 1 << tz)
        }

        package init?(dimension: Int, seed: UInt64) {
            guard let (outer, hadamardWidth) = Self.decompose(dimension) else { return nil }
            let generated = KroneckerRHT.generateFactors(dimension: dimension, outer: outer, seed: seed)
            self.dimension = dimension
            self.seed = seed
            self.outer = outer
            self.hadamardWidth = hadamardWidth
            self.hadamardScale = 1.0 / Float(hadamardWidth).squareRoot()
            self.signsValues = generated.signs
            self.packedSignsValues = Self.packSigns(generated.signs)
            self.qValues = generated.qValues
        }

        package init(_ rht: KroneckerRHT) {
            self.dimension = rht.dimension
            self.seed = rht.seed
            self.outer = rht.outer
            self.hadamardWidth = rht.hadamardWidth
            self.hadamardScale = rht.hadamardScale
            self.signsValues = rht.signsValues
            self.packedSignsValues = rht.packedSignsValues
            self.qValues = rht.qValues
        }

        package static func packSigns(_ signs: [Float]) -> [UInt32] {
            var words = [UInt32](repeating: 0, count: max(1, (signs.count + 31) / 32))
            for (idx, val) in signs.enumerated() where val < 0 {
                words[idx >> 5] |= (UInt32(1) << UInt32(idx & 31))
            }
            return words
        }
    }

    package struct Buffers {
        package let spec: Spec
        package let usesMetalKernel: Bool
        package let packedSigns: MLXArray
        package let qMatrix: MLXArray
        let scaleParams: MLXArray
        let placeholder: MLXArray
        let fallbackSigns: MLXArray?
        let fallbackQ: MLXArray?

        package var dimension: Int { spec.dimension }
        package var seed: UInt64 { spec.seed }
        package var outer: Int { spec.outer }
        package var hadamardWidth: Int { spec.hadamardWidth }

        package init(spec: Spec) {
            self.spec = spec
            let canUseMetal = spec.hadamardWidth >= 2
                && spec.hadamardWidth <= FusedKroneckerRHT.maxMetalHadamardWidth
                && spec.outer >= 1
                && spec.outer <= FusedKroneckerRHT.maxMetalOuter
            self.usesMetalKernel = canUseMetal
            let pSigns = MLXArray(spec.packedSignsValues)
            let qMat = spec.qValues.map { MLXArray($0, [spec.outer, spec.outer]) } ?? MLXArray([Float(1)])
            let sParams = MLXArray([spec.hadamardScale, Float(0)])
            let dummy = MLXArray([Float(0)])
            self.packedSigns = pSigns
            self.qMatrix = qMat
            self.scaleParams = sParams
            self.placeholder = dummy
            if canUseMetal {
                self.fallbackSigns = nil
                self.fallbackQ = nil
                eval(pSigns, qMat, sParams, dummy)
            } else {
                let fSigns = MLXArray(spec.signsValues)
                let fQ = spec.qValues.map { MLXArray($0, [spec.outer, spec.outer]) }
                self.fallbackSigns = fSigns
                self.fallbackQ = fQ
                if let fQ { eval(pSigns, qMat, sParams, dummy, fSigns, fQ) }
                else { eval(pSigns, qMat, sParams, dummy, fSigns) }
            }
        }

        package init(_ rht: KroneckerRHT) {
            self.init(spec: rht.spec)
        }
    }

    package struct Pair {
        package let inBuffers: Buffers
        package let outBuffers: Buffers

        package init(inBuffers: Buffers, outBuffers: Buffers) {
            self.inBuffers = inBuffers
            self.outBuffers = outBuffers
        }
    }
}

extension FusedKroneckerRHT {
    package static func threadsPerRow(hadamardWidth: Int) -> Int {
        if hadamardWidth <= 32 { return 32 }
        if hadamardWidth == 64 { return 64 }
        return hadamardWidth == 128 ? 128 : 256
    }

    private static func resolvedOutDType(_ x: MLXArray, requested: DType?) -> DType {
        let candidate = requested ?? x.dtype
        return dtypes.contains(candidate) ? candidate : .float32
    }

    package static func forward(_ x: MLXArray, buffers: Buffers, outputDType: DType? = nil) -> MLXArray {
        let outType = resolvedOutDType(x, requested: outputDType)
        precondition(x.ndim > 0 && x.dim(-1) == buffers.dimension, "FusedKroneckerRHT.forward shape mismatch")
        let rows = x.size / buffers.dimension
        guard rows > 0 else { return x.asType(outType) }
        guard buffers.usesMetalKernel, rows <= Int(Int32.max / 2) else {
            return referenceForward(x, spec: buffers.spec, signs: buffers.fallbackSigns, qMatrix: buffers.fallbackQ, outputDType: outType)
        }
        let input = dtypes.contains(x.dtype) ? x : x.asType(.float32)
        let threads = threadsPerRow(hadamardWidth: buffers.hadamardWidth)
        let out = kroneckerForwardKernel(
            [input.reshaped(rows, buffers.dimension), buffers.packedSigns, buffers.qMatrix, buffers.placeholder, buffers.scaleParams],
            template: [
                ("T", input.dtype), ("TO", outType), ("TW", DType.float32),
                ("N", buffers.hadamardWidth), ("P", buffers.outer), ("THREADS", threads), ("HAS_NORM", false)
            ],
            grid: (threads, buffers.outer, rows), threadGroup: (threads, 1, 1),
            outputShapes: [[rows, buffers.dimension]], outputDTypes: [outType]
        )[0]
        return out.reshaped(x.shape)
    }

    package static func rmsNormForward(
        _ x: MLXArray, normWeight: MLXArray, eps: Float, buffers: Buffers, outputDType: DType? = nil
    ) -> MLXArray {
        let outType = resolvedOutDType(x, requested: outputDType)
        precondition(x.ndim > 0 && x.dim(-1) == buffers.dimension && normWeight.size == buffers.dimension)
        let rows = x.size / buffers.dimension
        guard rows > 0 else { return x.asType(outType) }
        guard buffers.usesMetalKernel, rows <= Int(Int32.max / 2) else {
            return referenceRMSNormForward(x, normWeight: normWeight, eps: eps, spec: buffers.spec, outputDType: outType)
        }
        let input = dtypes.contains(x.dtype) ? x : x.asType(.float32)
        let nw = dtypes.contains(normWeight.dtype) ? normWeight : normWeight.asType(.float32)
        let params = MLXArray([buffers.spec.hadamardScale, eps])
        let threads = threadsPerRow(hadamardWidth: buffers.hadamardWidth)
        let out = kroneckerForwardKernel(
            [input.reshaped(rows, buffers.dimension), buffers.packedSigns, buffers.qMatrix, nw.reshaped(buffers.dimension), params],
            template: [
                ("T", input.dtype), ("TO", outType), ("TW", nw.dtype),
                ("N", buffers.hadamardWidth), ("P", buffers.outer), ("THREADS", threads), ("HAS_NORM", true)
            ],
            grid: (threads, buffers.outer, rows), threadGroup: (threads, 1, 1),
            outputShapes: [[rows, buffers.dimension]], outputDTypes: [outType]
        )[0]
        return out.reshaped(x.shape)
    }

    package static func inverse(
        _ y: MLXArray, buffers: Buffers, bias: MLXArray? = nil, outputDType: DType? = nil
    ) -> MLXArray {
        let outType = resolvedOutDType(y, requested: outputDType)
        precondition(y.ndim > 0 && y.dim(-1) == buffers.dimension, "FusedKroneckerRHT.inverse shape mismatch")
        if let bias { precondition(bias.size == buffers.dimension, "FusedKroneckerRHT.inverse bias mismatch") }
        let rows = y.size / buffers.dimension
        guard rows > 0 else { return y.asType(outType) }
        guard buffers.usesMetalKernel, rows <= Int(Int32.max / 2) else {
            return referenceInverse(y, spec: buffers.spec, bias: bias, outputDType: outType)
        }
        let input = dtypes.contains(y.dtype) ? y : y.asType(.float32)
        let biasBuf = bias.map { (dtypes.contains($0.dtype) ? $0 : $0.asType(.float32)).reshaped(buffers.dimension) } ?? buffers.placeholder
        let threads = threadsPerRow(hadamardWidth: buffers.hadamardWidth)
        let out = kroneckerInverseKernel(
            [input.reshaped(rows, buffers.dimension), buffers.packedSigns, buffers.qMatrix, biasBuf, buffers.scaleParams],
            template: [
                ("T", input.dtype), ("TO", outType), ("TB", biasBuf.dtype),
                ("N", buffers.hadamardWidth), ("P", buffers.outer), ("THREADS", threads), ("HAS_BIAS", bias != nil)
            ],
            grid: (threads, buffers.outer, rows), threadGroup: (threads, 1, 1),
            outputShapes: [[rows, buffers.dimension]], outputDTypes: [outType]
        )[0]
        return out.reshaped(y.shape)
    }

    package static func weightedInverseCollapse(
        _ rotatedOutputs: MLXArray, weights: MLXArray, buffers: Buffers, bias: MLXArray? = nil, outputDType: DType? = nil
    ) -> MLXArray {
        let outType = resolvedOutDType(rotatedOutputs, requested: outputDType)
        let weighted = (rotatedOutputs.asType(.float32) * expandedDimensions(weights.asType(.float32), axis: -1)).sum(axis: -2)
        return inverse(weighted, buffers: buffers, bias: bias, outputDType: outType)
    }
}

extension FusedKroneckerRHT {
    package static func blockHadamardForward(_ x: MLXArray, signs: MLXArray, blockSize: Int) -> MLXArray {
        blockHadamard(x, signs: signs, blockSize: blockSize, isInverse: false)
    }

    package static func blockHadamardInverse(_ y: MLXArray, signs: MLXArray, blockSize: Int) -> MLXArray {
        blockHadamard(y, signs: signs, blockSize: blockSize, isInverse: true)
    }

    private static func blockHadamard(_ x: MLXArray, signs: MLXArray, blockSize: Int, isInverse: Bool) -> MLXArray {
        if MACaQFusedKernels.supportsSingleSimdHadamard(blockSize: blockSize, dtype: x.dtype) {
            return MACaQFusedKernels.blockHadamard(x, scaleVector: signs, blockSize: blockSize, isInverse: isInverse)
        }
        return blockHadamardFallback(x, signs: signs, blockSize: blockSize, isInverse: isInverse)
    }

    package static func blockHadamardFallback(_ x: MLXArray, signs: MLXArray, blockSize: Int, isInverse: Bool) -> MLXArray {
        let origType = x.dtype
        guard x.ndim > 0, blockSize > 0 else { return x }
        let width = x.dim(-1)
        let rows = width > 0 ? x.size / width : 0
        guard rows > 0, width > 0, width.isMultiple(of: blockSize) else { return x }
        if blockSize == 1 { return (x.asType(.float32) * signs.asType(.float32)).asType(origType) }
        let numBlocks = width / blockSize
        let totalBlocks = rows * numBlocks
        guard blockSize <= maxMetalHadamardWidth,
              blockSize & (blockSize - 1) == 0,
              dtypes.contains(origType),
              totalBlocks <= Int(Int32.max / 2) else {
            let scale = 1.0 / Float(blockSize).squareRoot()
            if isInverse {
                let h = hadamardTransform(x.asType(.float32).reshaped(-1, blockSize), scale: scale).reshaped(x.shape)
                return (h * signs.asType(.float32)).asType(origType)
            }
            let xs = x.asType(.float32) * signs.asType(.float32)
            return hadamardTransform(xs.reshaped(-1, blockSize), scale: scale).reshaped(x.shape).asType(origType)
        }
        let cleanSigns = dtypes.contains(signs.dtype) ? signs : signs.asType(.float32)
        let scaleParam = MLXArray([1.0 / Float(blockSize).squareRoot()])
        let threads = threadsPerRow(hadamardWidth: blockSize)
        let out = blockHadamardKernel(
            [x.reshaped(totalBlocks, blockSize), cleanSigns.reshaped(width), scaleParam],
            template: [
                ("T", origType), ("TS", cleanSigns.dtype),
                ("N", blockSize), ("NUM_BLOCKS", numBlocks), ("THREADS", threads), ("IS_INV", isInverse)
            ],
            grid: (threads, totalBlocks, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[totalBlocks, blockSize]], outputDTypes: [origType]
        )[0]
        return out.reshaped(x.shape)
    }

    package static func referenceForward(
        _ x: MLXArray, spec: Spec, signs: MLXArray? = nil, qMatrix: MLXArray? = nil, outputDType: DType? = nil
    ) -> MLXArray {
        let outType = resolvedOutDType(x, requested: outputDType)
        let sArr = signs ?? MLXArray(spec.signsValues)
        let qArr = qMatrix ?? spec.qValues.map { MLXArray($0, [spec.outer, spec.outer]) }
        var y = hadamardTransform((x.asType(.float32) * sArr).reshaped(-1, spec.hadamardWidth), scale: spec.hadamardScale)
        if let qArr, spec.outer > 1 {
            let batch = y.size / spec.dimension
            y = matmul(y.reshaped(batch, spec.outer, spec.hadamardWidth).transposed(0, 2, 1), qArr.T).transposed(0, 2, 1)
        }
        return y.reshaped(x.shape).asType(outType)
    }

    package static func referenceRMSNormForward(
        _ x: MLXArray, normWeight: MLXArray, eps: Float, spec: Spec, outputDType: DType? = nil
    ) -> MLXArray {
        let outType = resolvedOutDType(x, requested: outputDType)
        let xf = x.asType(.float32)
        let normed = xf * rsqrt((xf * xf).mean(axis: -1, keepDims: true) + eps) * normWeight.asType(.float32)
        return referenceForward(normed, spec: spec, outputDType: outType)
    }

    package static func referenceInverse(
        _ y: MLXArray, spec: Spec, bias: MLXArray? = nil, outputDType: DType? = nil
    ) -> MLXArray {
        let outType = resolvedOutDType(y, requested: outputDType)
        let sArr = MLXArray(spec.signsValues)
        let qArr = spec.qValues.map { MLXArray($0, [spec.outer, spec.outer]) }
        var z = y.asType(.float32)
        if let qArr, spec.outer > 1 {
            let batch = z.size / spec.dimension
            z = matmul(z.reshaped(batch, spec.outer, spec.hadamardWidth).transposed(0, 2, 1), qArr).transposed(0, 2, 1)
        }
        var res = hadamardTransform(z.reshaped(-1, spec.hadamardWidth), scale: spec.hadamardScale).reshaped(y.shape) * sArr
        if let bias { res = res + bias.asType(.float32) }
        return res.asType(outType)
    }
}

private enum FusedKroneckerRHTMetal {
    static let header = """
        #include <metal_simdgroup>

        template <int N, int THREADS>
        inline int rht_out_col(int tid, int lane, int sg, int u) {
            if (N <= 32) { return tid; }
            constexpr int SIMDGROUPS = THREADS / 32;
            constexpr int M = (N >= 64) ? (N / 32) : 1;
            constexpr int SUB = (M < 32) ? M : 32;
            constexpr int Q_RAD = M / SUB;
            constexpr int PACK = 32 / SUB;
            constexpr int COLS_STEP = SIMDGROUPS * PACK;
            const int ei = u / Q_RAD;
            const int q = u % Q_RAD;
            const int m_sub = lane & (SUB - 1);
            const int ell = ei * COLS_STEP + sg * PACK + (lane / SUB);
            return (q * 32 + m_sub) * 32 + ell;
        }

        template <int N, int THREADS>
        inline void rht_butterfly(thread float* vals, threadgroup float* tg_buf, int lane, int sg, float scale) {
            if (N <= 32) {
                float v = vals[0];
                for (int h = 1; h < N; h <<= 1) {
                    const float other = simd_shuffle_xor(v, ushort(h));
                    v = other + (((lane & h) != 0) ? -v : v);
                }
                vals[0] = v * scale;
                return;
            }
            constexpr int SIMDGROUPS = THREADS / 32;
            constexpr int E = (N > THREADS) ? (N / THREADS) : 1;
            constexpr int M = (N >= 64) ? (N / 32) : 1;
            constexpr int SUB = (M < 32) ? M : 32;
            constexpr int Q_RAD = M / SUB;
            constexpr int PACK = 32 / SUB;
            constexpr int COLS_STEP = SIMDGROUPS * PACK;
            constexpr int EI = 32 / COLS_STEP;
            for (int e = 0; e < E; ++e) {
                float v = vals[e];
                for (int h = 1; h < 32; h <<= 1) {
                    const float other = simd_shuffle_xor(v, ushort(h));
                    v = other + (((lane & h) != 0) ? -v : v);
                }
                tg_buf[(e * SIMDGROUPS + sg) * 33 + lane] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const int m_sub = lane & (SUB - 1);
            const int c_sub = lane / SUB;
            for (int ei = 0; ei < EI; ++ei) {
                const int ell = ei * COLS_STEP + sg * PACK + c_sub;
                float s[Q_RAD];
                for (int q = 0; q < Q_RAD; ++q) { s[q] = tg_buf[(q * 32 + m_sub) * 33 + ell]; }
                if (Q_RAD == 2) {
                    const float a0 = s[0], a1 = s[1 % Q_RAD];
                    s[0] = a0 + a1;
                    s[1 % Q_RAD] = a0 - a1;
                } else if (Q_RAD == 4) {
                    const float t0 = s[0] + s[1 % Q_RAD], t1 = s[0] - s[1 % Q_RAD];
                    const float t2 = s[2 % Q_RAD] + s[3 % Q_RAD], t3 = s[2 % Q_RAD] - s[3 % Q_RAD];
                    s[0] = t0 + t2;
                    s[1 % Q_RAD] = t1 + t3;
                    s[2 % Q_RAD] = t0 - t2;
                    s[3 % Q_RAD] = t1 - t3;
                }
                for (int h = 1; h < SUB; h <<= 1) {
                    for (int q = 0; q < Q_RAD; ++q) {
                        const float other = simd_shuffle_xor(s[q], ushort(h));
                        s[q] = other + (((m_sub & h) != 0) ? -s[q] : s[q]);
                    }
                }
                for (int q = 0; q < Q_RAD; ++q) { vals[ei * Q_RAD + q] = s[q] * scale; }
            }
        }
        """

    static let forwardSource = """
        constexpr int D = P * N;
        constexpr int SIMDGROUPS = THREADS / 32;
        constexpr int E = (N > THREADS) ? (N / THREADS) : 1;
        constexpr int TG_M = (N >= 64) ? (N / 32) : 1;
        const int r_blk = int(threadgroup_position_in_grid.y);
        const int row = int(threadgroup_position_in_grid.z);
        const int tid = int(thread_position_in_threadgroup.x);
        const int lane = int(thread_index_in_simdgroup);
        const int sg = int(simdgroup_index_in_threadgroup);
        const size_t row_base = size_t(row) * size_t(D);
        const float scale = float(params[0]);
        threadgroup float tg_buf[TG_M * 33];
        threadgroup float tg_red[SIMDGROUPS];
        float inv_rms = 1.0f;
        if (HAS_NORM) {
            float sq = 0.0f;
            for (int idx = tid; idx < D; idx += THREADS) {
                const float xv = float(inp[row_base + size_t(idx)]);
                sq = fma(xv, xv, sq);
            }
            sq = simd_sum(sq);
            if (SIMDGROUPS > 1) {
                if (lane == 0) { tg_red[sg] = sq; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg == 0) {
                    const float total = simd_sum((lane < SIMDGROUPS) ? tg_red[lane] : 0.0f);
                    if (lane == 0) { tg_red[0] = total; }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                sq = tg_red[0];
            }
            inv_rms = metal::precise::rsqrt(sq * (1.0f / float(D)) + float(params[1]));
        }
        float vals[E];
        if (P == 1) {
            for (int e = 0; e < E; ++e) {
                const int col = e * THREADS + tid;
                if (col < N) {
                    const uint sw = packed_signs[col >> 5];
                    const float sgn = (((sw >> uint(col & 31)) & 1u) != 0u) ? -1.0f : 1.0f;
                    float xv = float(inp[row_base + size_t(col)]);
                    if (HAS_NORM) { xv = xv * inv_rms * float(norm_w[col]); }
                    vals[e] = xv * sgn;
                } else {
                    vals[e] = 0.0f;
                }
            }
        } else {
            for (int e = 0; e < E; ++e) { vals[e] = 0.0f; }
            for (int c = 0; c < P; ++c) {
                const float qrc = float(q_mat[r_blk * P + c]);
                for (int e = 0; e < E; ++e) {
                    const int col = e * THREADS + tid;
                    if (col < N) {
                        const int d = c * N + col;
                        const uint sw = packed_signs[d >> 5];
                        const float sgn = (((sw >> uint(d & 31)) & 1u) != 0u) ? -1.0f : 1.0f;
                        float xv = float(inp[row_base + size_t(d)]);
                        if (HAS_NORM) { xv = xv * inv_rms * float(norm_w[d]); }
                        vals[e] = fma(qrc, xv * sgn, vals[e]);
                    }
                }
            }
        }
        rht_butterfly<N, THREADS>(vals, tg_buf, lane, sg, scale);
        for (int u = 0; u < E; ++u) {
            const int out_col = rht_out_col<N, THREADS>(tid, lane, sg, u);
            if (out_col < N) {
                out[row_base + size_t(r_blk * N + out_col)] = static_cast<TO>(vals[u]);
            }
        }
        """

    static let inverseSource = """
        constexpr int D = P * N;
        constexpr int E = (N > THREADS) ? (N / THREADS) : 1;
        constexpr int TG_M = (N >= 64) ? (N / 32) : 1;
        const int r_blk = int(threadgroup_position_in_grid.y);
        const int row = int(threadgroup_position_in_grid.z);
        const int tid = int(thread_position_in_threadgroup.x);
        const int lane = int(thread_index_in_simdgroup);
        const int sg = int(simdgroup_index_in_threadgroup);
        const size_t row_base = size_t(row) * size_t(D);
        const float scale = float(params[0]);
        threadgroup float tg_buf[TG_M * 33];
        float vals[E];
        if (P == 1) {
            for (int e = 0; e < E; ++e) {
                const int col = e * THREADS + tid;
                vals[e] = (col < N) ? float(inp[row_base + size_t(col)]) : 0.0f;
            }
        } else {
            for (int e = 0; e < E; ++e) { vals[e] = 0.0f; }
            for (int c = 0; c < P; ++c) {
                const float qcr = float(q_mat[c * P + r_blk]);
                for (int e = 0; e < E; ++e) {
                    const int col = e * THREADS + tid;
                    const float yv = (col < N) ? float(inp[row_base + size_t(c * N + col)]) : 0.0f;
                    vals[e] = fma(qcr, yv, vals[e]);
                }
            }
        }
        rht_butterfly<N, THREADS>(vals, tg_buf, lane, sg, scale);
        for (int u = 0; u < E; ++u) {
            const int out_col = rht_out_col<N, THREADS>(tid, lane, sg, u);
            if (out_col < N) {
                const int d_out = r_blk * N + out_col;
                const uint sw = packed_signs[d_out >> 5];
                const float sgn = (((sw >> uint(d_out & 31)) & 1u) != 0u) ? -1.0f : 1.0f;
                float res = vals[u] * sgn;
                if (HAS_BIAS) { res += float(bias_buf[d_out]); }
                out[row_base + size_t(d_out)] = static_cast<TO>(res);
            }
        }
        """

    static let blockSource = """
        constexpr int E = (N > THREADS) ? (N / THREADS) : 1;
        constexpr int TG_M = (N >= 64) ? (N / 32) : 1;
        const int blk_idx = int(threadgroup_position_in_grid.y);
        const int tid = int(thread_position_in_threadgroup.x);
        const int lane = int(thread_index_in_simdgroup);
        const int sg = int(simdgroup_index_in_threadgroup);
        const size_t base = size_t(blk_idx) * size_t(N);
        const size_t sign_base = size_t(blk_idx % NUM_BLOCKS) * size_t(N);
        const float scale = float(params[0]);
        threadgroup float tg_buf[TG_M * 33];
        float vals[E];
        for (int e = 0; e < E; ++e) {
            const int col = e * THREADS + tid;
            if (col < N) {
                float xv = float(inp[base + size_t(col)]);
                if (!IS_INV) { xv *= float(signs[sign_base + size_t(col)]); }
                vals[e] = xv;
            } else {
                vals[e] = 0.0f;
            }
        }
        rht_butterfly<N, THREADS>(vals, tg_buf, lane, sg, scale);
        for (int u = 0; u < E; ++u) {
            const int out_col = rht_out_col<N, THREADS>(tid, lane, sg, u);
            if (out_col < N) {
                float res = vals[u];
                if (IS_INV) { res *= float(signs[sign_base + size_t(out_col)]); }
                out[base + size_t(out_col)] = static_cast<T>(res);
            }
        }
        """
}

private let kroneckerForwardKernel = MLXFast.metalKernel(
    name: "rht_kronecker_fwd",
    inputNames: ["inp", "packed_signs", "q_mat", "norm_w", "params"],
    outputNames: ["out"],
    source: FusedKroneckerRHTMetal.forwardSource,
    header: FusedKroneckerRHTMetal.header
)

private let kroneckerInverseKernel = MLXFast.metalKernel(
    name: "rht_kronecker_inv",
    inputNames: ["inp", "packed_signs", "q_mat", "bias_buf", "params"],
    outputNames: ["out"],
    source: FusedKroneckerRHTMetal.inverseSource,
    header: FusedKroneckerRHTMetal.header
)

private let blockHadamardKernel = MLXFast.metalKernel(
    name: "rht_block_hadamard",
    inputNames: ["inp", "signs", "params"],
    outputNames: ["out"],
    source: FusedKroneckerRHTMetal.blockSource,
    header: FusedKroneckerRHTMetal.header
)
