import Foundation
import MLX
import MLXNN

private let qtipKernelsHeader = """
    #include <metal_simdgroup>

    inline ushort qtip_fast_code_idx(uint edge) {
        uint h = edge;
        h = ((h ^ (h >> 8u)) * 0x9E37u) & 0xFFFFu;
        h = ((h ^ (h >> 7u)) * 0xC2B5u) & 0xFFFFu;
        h ^= (h >> 9u);
        return ushort(h & 2047u);
    }

    inline float2 qtip_fast_hyb_lookup(ushort edge, const threadgroup half2* lut) {
        ushort code = qtip_fast_code_idx(uint(edge));
        float sx = (code & 512u) != 0u ? -1.0f : 1.0f;
        float sy = (code & 1024u) != 0u ? -1.0f : 1.0f;
        float2 base = float2(lut[code & 511u]);
        return float2(base.x * sx, base.y * sy);
    }
    """

private let qtipMatvecSource = """
    const uint TROWS = uint(packed_shape[0]);
    const uint IN_DIM = uint(x_shape[1]);
    const uint TCOLS = IN_DIM >> 4u;
    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;

    threadgroup half2 s_tlut[SG][512];
    for (uint idx = tid; idx < 512u; idx += 32u * uint(SG)) {
        half2 v = half2(half(float(tlut[2u * idx])), half(float(tlut[2u * idx + 1u])));
        #pragma clang loop unroll(full)
        for (uint s = 0u; s < uint(SG); ++s) {
            s_tlut[s][idx] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint tr = threadgroup_position_in_grid.y * uint(SG) + sg;
    if (tr >= TROWS) {
        return;
    }

    const uint colPair = lane & 7u;
    const uint rowSub = lane >> 3u;
    const threadgroup half2* my_tlut = s_tlut[sg];
    const size_t rowWordBase = size_t(tr) * size_t(TCOLS) * size_t(WORDS);

    ushort wIdx0[4];
    ushort wIdx1[4];
    uint bShift[4];
    #pragma clang loop unroll(full)
    for (uint rg = 0u; rg < 4u; ++rg) {
        uint bitPos = ((rg << 5u) | lane) * uint(K);
        uint wi = bitPos >> 5u;
        wIdx0[rg] = ushort(wi);
        wIdx1[rg] = ushort((wi + 1u == uint(WORDS)) ? 0u : (wi + 1u));
        bShift[rg] = bitPos & 31u;
    }

    float acc[M][4];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < uint(M); ++m) {
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 4u; ++r) {
            acc[m][r] = 0.0f;
        }
    }

    for (uint tc = 0u; tc < TCOLS; ++tc) {
        size_t tileBase = rowWordBase + size_t(tc) * size_t(WORDS);
        uint myWord = (lane < uint(WORDS)) ? packed[tileBase + size_t(lane)] : 0u;

        float2 xm[M];
        size_t xCol = size_t(tc) * 16u + size_t(colPair << 1u);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < uint(M); ++m) {
            size_t xIdx = size_t(m) * size_t(IN_DIM) + xCol;
            xm[m] = float2(float(x[xIdx]), float(x[xIdx + 1u]));
        }

        #pragma clang loop unroll(full)
        for (uint rg = 0u; rg < 4u; ++rg) {
            ulong lo = ulong(simd_shuffle(myWord, wIdx0[rg]));
            ulong hi = ulong(simd_shuffle(myWord, wIdx1[rg]));
            ushort edge = ushort(((lo | (hi << 32u)) >> bShift[rg]) & 0xFFFFul);
            float2 w = qtip_fast_hyb_lookup(edge, my_tlut);
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < uint(M); ++m) {
                acc[m][rg] = fma(w.x, xm[m].x, fma(w.y, xm[m].y, acc[m][rg]));
            }
        }
    }

    const float scale = float(row_scales[tr]);
    const size_t outDim = size_t(TROWS) * 16u;
    #pragma clang loop unroll(full)
    for (uint rg = 0u; rg < 4u; ++rg) {
        uint outRow = tr * 16u + (rg << 2u) + rowSub;
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < uint(M); ++m) {
            float v = acc[m][rg];
            v += simd_shuffle_xor(v, ushort(4));
            v += simd_shuffle_xor(v, ushort(2));
            v += simd_shuffle_xor(v, ushort(1));
            if (colPair == 0u) {
                out[size_t(m) * outDim + size_t(outRow)] = static_cast<T>(v * scale);
            }
        }
    }
    """

private let qtipGatherQmvSource = """
    const uint NUM_EXPERTS = uint(packed_shape[0]);
    const uint TROWS = uint(packed_shape[1]);
    const uint X_ROWS = uint(x_shape[0]);
    const uint IN_DIM = uint(x_shape[1]);
    const uint TCOLS = IN_DIM >> 4u;
    const uint A = uint(indices_shape[0]);
    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;

    threadgroup half2 s_tlut[SG][512];
    for (uint idx = tid; idx < 512u; idx += 32u * uint(SG)) {
        half2 v = half2(half(float(tlut[2u * idx])), half(float(tlut[2u * idx + 1u])));
        #pragma clang loop unroll(full)
        for (uint s = 0u; s < uint(SG); ++s) {
            s_tlut[s][idx] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint a = threadgroup_position_in_grid.z;
    const uint tr = threadgroup_position_in_grid.y * uint(SG) + sg;
    if (a >= A || tr >= TROWS) {
        return;
    }

    const uint rawExpert = uint(indices[a]);
    const uint e = (rawExpert < NUM_EXPERTS) ? rawExpert : (NUM_EXPERTS - 1u);
    const uint xRow = (X_ROWS == A) ? a : (a / (A / X_ROWS));
    const size_t xRowBase = size_t(xRow) * size_t(IN_DIM);
    const size_t expertTileRow = size_t(e) * size_t(TROWS) + size_t(tr);
    const size_t rowWordBase = expertTileRow * size_t(TCOLS) * size_t(WORDS);

    const uint colPair = lane & 7u;
    const uint rowSub = lane >> 3u;
    const threadgroup half2* my_tlut = s_tlut[sg];

    ushort wIdx0[4];
    ushort wIdx1[4];
    uint bShift[4];
    #pragma clang loop unroll(full)
    for (uint rg = 0u; rg < 4u; ++rg) {
        uint bitPos = ((rg << 5u) | lane) * uint(K);
        uint wi = bitPos >> 5u;
        wIdx0[rg] = ushort(wi);
        wIdx1[rg] = ushort((wi + 1u == uint(WORDS)) ? 0u : (wi + 1u));
        bShift[rg] = bitPos & 31u;
    }

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint tc = 0u; tc < TCOLS; ++tc) {
        size_t tileBase = rowWordBase + size_t(tc) * size_t(WORDS);
        uint myWord = (lane < uint(WORDS)) ? packed[tileBase + size_t(lane)] : 0u;
        size_t xIdx = xRowBase + size_t(tc) * 16u + size_t(colPair << 1u);
        float x0 = float(x[xIdx]);
        float x1 = float(x[xIdx + 1u]);

        #pragma clang loop unroll(full)
        for (uint rg = 0u; rg < 4u; ++rg) {
            ulong lo = ulong(simd_shuffle(myWord, wIdx0[rg]));
            ulong hi = ulong(simd_shuffle(myWord, wIdx1[rg]));
            ushort edge = ushort(((lo | (hi << 32u)) >> bShift[rg]) & 0xFFFFul);
            float2 w = qtip_fast_hyb_lookup(edge, my_tlut);
            acc[rg] = fma(w.x, x0, fma(w.y, x1, acc[rg]));
        }
    }

    const float scale = float(row_scales[expertTileRow]);
    const size_t outBase = size_t(a) * size_t(TROWS * 16u) + size_t(tr * 16u + rowSub);
    #pragma clang loop unroll(full)
    for (uint rg = 0u; rg < 4u; ++rg) {
        float v = acc[rg];
        v += simd_shuffle_xor(v, ushort(4));
        v += simd_shuffle_xor(v, ushort(2));
        v += simd_shuffle_xor(v, ushort(1));
        if (colPair == 0u) {
            out[outBase + size_t(rg << 2u)] = static_cast<T>(v * scale);
        }
    }
    """

private let qtipMatvecKernel = MLXFast.metalKernel(
    name: "qtip_matvec",
    inputNames: ["x", "packed", "row_scales", "tlut"],
    outputNames: ["out"],
    source: qtipMatvecSource,
    header: qtipKernelsHeader
)

private let qtipGatherQmvKernel = MLXFast.metalKernel(
    name: "qtip_gather_qmv",
    inputNames: ["x", "packed", "row_scales", "tlut", "indices"],
    outputNames: ["out"],
    source: qtipGatherQmvSource,
    header: qtipKernelsHeader
)

package enum QTIPKernels {
    package static let matvecRows = 1 ... 8
    package static let simdgroups = 4
    package static let tileBytes = 64 << 20
    package static let dtypes: Set<DType> = [.float32, .float16, .bfloat16]

    package static func linear(
        _ xRot: MLXArray,
        packed: MLXArray,
        rowScales: MLXArray,
        tlut: MLXArray,
        rate: QTIPRate,
        inFeatures: Int? = nil
    ) -> MLXArray {
        let width = inFeatures ?? (xRot.ndim >= 1 ? xRot.dim(-1) : 0)
        let rows = width > 0 ? xRot.size / width : 0
        if rows == 1 {
            return qmv(xRot, packed: packed, rowScales: rowScales, tlut: tlut, rate: rate)
        }
        if matvecRows.contains(rows) {
            return smallM(xRot, packed: packed, rowScales: rowScales, tlut: tlut, rate: rate)
        }
        return qmmTiled(xRot, packed: packed, rowScales: rowScales, tlut: tlut, rate: rate)
    }

    package static func qmv(
        _ xRot: MLXArray, packed: MLXArray, rowScales: MLXArray, tlut: MLXArray, rate: QTIPRate
    ) -> MLXArray {
        precondition(xRot.ndim >= 1 && xRot.dim(-1) > 0 && xRot.size / xRot.dim(-1) == 1, "qmv requires M == 1")
        return runMatvec(xRot, packed: packed, rowScales: rowScales, tlut: tlut, rate: rate)
    }

    package static func smallM(
        _ xRot: MLXArray, packed: MLXArray, rowScales: MLXArray, tlut: MLXArray, rate: QTIPRate
    ) -> MLXArray {
        let rows = (xRot.ndim >= 1 && xRot.dim(-1) > 0) ? xRot.size / xRot.dim(-1) : 0
        precondition(matvecRows.contains(rows), "smallM requires M in 1...8")
        return runMatvec(xRot, packed: packed, rowScales: rowScales, tlut: tlut, rate: rate)
    }

    private static func cleanFloat(_ array: MLXArray) -> MLXArray {
        dtypes.contains(array.dtype) ? array : array.asType(.float32, stream: array.dtype == .float64 ? .cpu : .default)
    }

    private static func runOnCPUIfFloat64(_ arrays: [MLXArray], _ body: () -> MLXArray) -> MLXArray {
        guard Device.defaultDevice() == .gpu, arrays.contains(where: { $0.dtype == .float64 }) else { return body() }
        return Device.withDefaultDevice(.cpu, body)
    }

    private static func runMatvec(
        _ xRot: MLXArray, packed: MLXArray, rowScales: MLXArray, tlut: MLXArray, rate: QTIPRate
    ) -> MLXArray {
        precondition(xRot.ndim >= 1 && packed.ndim >= 2, "QTIPKernels.runMatvec requires ndim >= 1 input and ndim >= 2 packed")
        return runOnCPUIfFloat64([xRot, packed, rowScales, tlut]) {
            let inFeatures = xRot.dim(-1)
            let tileRows = packed.dim(0)
            let outFeatures = tileRows * 16
            let leading = Array(xRot.shape.dropLast())
            let rows = inFeatures > 0 ? xRot.size / inFeatures : 0
            guard rows > 0, tileRows > 0, inFeatures > 0 else {
                return MLXArray.zeros(leading + [outFeatures], dtype: xRot.dtype)
            }
            precondition(inFeatures.isMultiple(of: 16), "QTIPKernels.runMatvec requires inFeatures multiple of 16")
            let tileCols = inFeatures / 16
            precondition(packed.size == tileRows * tileCols * rate.wordsPerTile, "QTIPKernels.runMatvec packed size mismatch")
            precondition(rowScales.size == tileRows, "QTIPKernels.runMatvec rowScales size mismatch")
            precondition(tlut.size == 1024, "QTIPKernels.runMatvec requires 1024-element tlut")
            let packed2D = packed.ndim == 2 ? packed : packed.reshaped(tileRows, tileCols * rate.wordsPerTile)
            let scales1D = rowScales.reshaped(tileRows)
            let rowGroups = (tileRows + simdgroups - 1) / simdgroups
            guard Device.defaultDevice() == .gpu,
                  packed2D.dtype == .uint32,
                  simdgroups * rowGroups <= Int(Int32.max)
            else {
                return qmmTiled(xRot, packed: packed2D, rowScales: scales1D, tlut: tlut, rate: rate)
            }
            let computeDType = dtypes.contains(xRot.dtype) ? xRot.dtype : .float32
            let xFlat = (xRot.dtype == computeDType ? xRot : xRot.asType(computeDType)).reshaped(rows, inFeatures)
            let out = qtipMatvecKernel(
                [xFlat, packed2D, cleanFloat(scales1D), cleanFloat(tlut)],
                template: [
                    ("K", rate.stepBits), ("WORDS", rate.wordsPerTile),
                    ("M", rows), ("SG", simdgroups), ("T", computeDType)
                ],
                grid: (32, simdgroups * rowGroups, 1),
                threadGroup: (32, simdgroups, 1),
                outputShapes: [[rows, outFeatures]],
                outputDTypes: [computeDType]
            )[0]
            let casted = out.dtype == xRot.dtype ? out : out.asType(xRot.dtype)
            return casted.reshaped(leading + [outFeatures])
        }
    }

    package static func qmmTiled(
        _ xRot: MLXArray, packed: MLXArray, rowScales: MLXArray, tlut: MLXArray,
        rate: QTIPRate, tileBytes: Int = tileBytes
    ) -> MLXArray {
        precondition(xRot.ndim >= 1 && packed.ndim >= 2, "QTIPKernels.qmmTiled requires ndim >= 1 input and ndim >= 2 packed")
        return runOnCPUIfFloat64([xRot, packed, rowScales, tlut]) {
            let inFeatures = xRot.dim(-1)
            let tileRows = packed.dim(0)
            let outFeatures = tileRows * 16
            let leading = Array(xRot.shape.dropLast())
            let rows = inFeatures > 0 ? xRot.size / inFeatures : 0
            guard rows > 0, tileRows > 0, inFeatures > 0 else {
                return MLXArray.zeros(leading + [outFeatures], dtype: xRot.dtype)
            }
            precondition(inFeatures.isMultiple(of: 16), "QTIPKernels.qmmTiled requires inFeatures multiple of 16")
            let tileCols = inFeatures / 16
            let packed2D = packed.ndim == 2 ? packed : packed.reshaped(tileRows, tileCols * rate.wordsPerTile)
            let scales1D = rowScales.reshaped(tileRows)
            let computeDType = dtypes.contains(xRot.dtype) ? xRot.dtype : .float32
            let xCompute = xRot.dtype == computeDType ? xRot : xRot.asType(computeDType)
            let bytesPerTileRow = max(1, 16 * inFeatures * computeDType.size)
            let rowsPerChunk = max(1, max(1, tileBytes) / bytesPerTileRow)
            var parts: [MLXArray] = []
            for start in stride(from: 0, to: tileRows, by: rowsPerChunk) {
                let end = min(tileRows, start + rowsPerChunk)
                let wChunk = QTIPMetal.decode(
                    packed: packed2D[start ..< end], rowScales: scales1D[start ..< end],
                    tlut: tlut, rate: rate, tileCols: tileCols, dtype: computeDType)
                parts.append(matmul(xCompute, wChunk.T))
            }
            let out = parts.count == 1 ? parts[0] : concatenated(parts, axis: -1)
            return out.dtype == xRot.dtype ? out : out.asType(xRot.dtype)
        }
    }
}

extension QTIPKernels {
    package static func gatherQmv(
        _ xRot: MLXArray, packedExperts: MLXArray, rowScales: MLXArray,
        tlut: MLXArray, indices: MLXArray, rate: QTIPRate
    ) -> MLXArray {
        gatherMatmul(x: xRot, packedExperts: packedExperts, rowScales: rowScales, tlut: tlut, indices: indices, rate: rate)
    }

    package static func gatherMatmul(
        x: MLXArray, packedExperts: MLXArray, rowScales: MLXArray,
        tlut: MLXArray, indices: MLXArray, rate: QTIPRate
    ) -> MLXArray {
        precondition(x.ndim >= 1 && packedExperts.ndim >= 3, "QTIPKernels.gatherMatmul requires ndim >= 1 input and ndim >= 3 packedExperts")
        return runOnCPUIfFloat64([x, packedExperts, rowScales, tlut, indices]) {
            let inFeatures = x.dim(-1)
            let numExperts = packedExperts.dim(0)
            let tileRows = packedExperts.dim(1)
            let outFeatures = tileRows * 16
            guard indices.size > 0, x.size > 0, inFeatures > 0, numExperts > 0, tileRows > 0 else {
                return MLXArray.zeros(gatherOutputShape(x: x, indices: indices, outFeatures: outFeatures), dtype: x.dtype)
            }
            precondition(inFeatures.isMultiple(of: 16), "QTIPKernels.gatherMatmul requires inFeatures multiple of 16")
            let expectedRowWords = (inFeatures / 16) * rate.wordsPerTile
            precondition(packedExperts.size == numExperts * tileRows * expectedRowWords, "QTIPKernels.gatherMatmul packedExperts size mismatch")
            precondition(rowScales.size == numExperts * tileRows, "QTIPKernels.gatherMatmul rowScales size mismatch")
            precondition(tlut.size == 1024, "QTIPKernels.gatherMatmul requires 1024-element tlut")
            let prep = prepareGatherInputs(x: x, indices: indices, inFeatures: inFeatures, outFeatures: outFeatures)
            let assignments = prep.flatIndices.size
            let packed3D = packedExperts.ndim == 3 ? packedExperts : packedExperts.reshaped(numExperts, tileRows, expectedRowWords)
            let scales2D = rowScales.reshaped(numExperts, tileRows)
            let rowGroups = (tileRows + simdgroups - 1) / simdgroups
            guard Device.defaultDevice() == .gpu, packed3D.dtype == .uint32,
                  simdgroups * rowGroups <= Int(Int32.max), assignments <= Int(Int32.max)
            else {
                let idx64 = prep.flatIndices.asType(.int64)
                let nExp = Int64(numExperts)
                let safeFlatIndices = which((idx64 .>= Int64(0)) .&& (idx64 .< nExp), idx64, Int64(nExp - 1))
                let cpuOut = cpuGatherMatmul(
                    flatX: prep.flatX, packed3D: packed3D, scales2D: scales2D,
                    tlut: tlut, flatIndices: safeFlatIndices, rate: rate)
                return cpuOut.asType(x.dtype).reshaped(prep.outShape)
            }
            let computeDType = dtypes.contains(x.dtype) ? x.dtype : .float32
            let xCompute = prep.flatX.dtype == computeDType ? prep.flatX : prep.flatX.asType(computeDType)
            let uintIndices = prep.flatIndices.asType(.uint32)
            let out = qtipGatherQmvKernel(
                [xCompute, packed3D, cleanFloat(scales2D), cleanFloat(tlut), uintIndices],
                template: [("K", rate.stepBits), ("WORDS", rate.wordsPerTile), ("SG", simdgroups), ("T", computeDType)],
                grid: (32, simdgroups * rowGroups, assignments), threadGroup: (32, simdgroups, 1),
                outputShapes: [[assignments, outFeatures]], outputDTypes: [computeDType])[0]
            return (out.dtype == x.dtype ? out : out.asType(x.dtype)).reshaped(prep.outShape)
        }
    }

    static func broadcastDim(_ a: Int, _ b: Int) -> Int {
        (a == 0 || b == 0) ? 0 : max(a, b)
    }

    static func gatherOutputShape(x: MLXArray, indices: MLXArray, outFeatures: Int) -> [Int] {
        if x.ndim == indices.ndim + 2 {
            return zip(Array(x.shape.dropLast(2)), indices.shape).map(broadcastDim) + [x.dim(-2), outFeatures]
        }
        if x.ndim == indices.ndim + 1 {
            return zip(Array(x.shape.dropLast()), indices.shape).map(broadcastDim) + [outFeatures]
        }
        guard indices.ndim >= 1 else { return [outFeatures] }
        let idxLead = Array(indices.shape.dropLast())
        let xLead = Array(x.shape.dropLast())
        let pad = max(0, idxLead.count - xLead.count)
        let b = Array(idxLead.prefix(pad)) + zip(idxLead.dropFirst(pad), xLead).map(broadcastDim)
        return b + [indices.dim(-1), outFeatures]
    }

    static func prepareGatherInputs(
        x: MLXArray, indices: MLXArray, inFeatures: Int, outFeatures: Int
    ) -> (flatX: MLXArray, flatIndices: MLXArray, outShape: [Int]) {
        let assignments = indices.size
        if x.ndim == indices.ndim + 2 {
            let lhsBatch = Array(x.shape.dropLast(2))
            let mInner = x.dim(-2)
            let bShape = zip(lhsBatch, indices.shape).map(broadcastDim)
            if mInner == 1, lhsBatch == indices.shape {
                return (x.reshaped(assignments, inFeatures), indices.reshaped(assignments), bShape + [1, outFeatures])
            }
            if mInner == 1, lhsBatch.last == 1, Array(lhsBatch.dropLast()) == Array(indices.shape.dropLast()) {
                return (x.reshaped(x.size / inFeatures, inFeatures), indices.reshaped(assignments), bShape + [1, outFeatures])
            }
            let xBroad = broadcast(x, to: bShape + [mInner, inFeatures])
            let idxBroad = repeated(expandedDimensions(broadcast(indices, to: bShape), axis: -1), count: mInner, axis: -1)
            return (xBroad.reshaped(idxBroad.size, inFeatures), idxBroad.reshaped(idxBroad.size), bShape + [mInner, outFeatures])
        }
        let xLead = Array(x.shape.dropLast())
        if x.ndim == indices.ndim + 1 {
            let bShape = zip(xLead, indices.shape).map(broadcastDim)
            if xLead == indices.shape {
                return (x.reshaped(assignments, inFeatures), indices.reshaped(assignments), bShape + [outFeatures])
            }
            if xLead.last == 1, Array(xLead.dropLast()) == Array(indices.shape.dropLast()) {
                return (x.reshaped(x.size / inFeatures, inFeatures), indices.reshaped(assignments), bShape + [outFeatures])
            }
            let total = bShape.reduce(1, *)
            return (
                broadcast(x, to: bShape + [inFeatures]).reshaped(total, inFeatures),
                broadcast(indices, to: bShape).reshaped(total),
                bShape + [outFeatures])
        }
        precondition(x.ndim <= indices.ndim && indices.ndim >= 1, "QTIPKernels.gatherMatmul incompatible input and index ranks")
        let idxLead = Array(indices.shape.dropLast())
        let pad = idxLead.count - xLead.count
        let bShape = Array(idxLead.prefix(pad)) + zip(idxLead.dropFirst(pad), xLead).map(broadcastDim)
        let kDim = indices.dim(-1)
        let xBroad = x.shape == bShape + [inFeatures] ? x : broadcast(x, to: bShape + [inFeatures])
        let idxBroad = indices.shape == bShape + [kDim] ? indices : broadcast(indices, to: bShape + [kDim])
        return (xBroad.reshaped(bShape.reduce(1, *), inFeatures), idxBroad.reshaped(idxBroad.size), bShape + [kDim, outFeatures])
    }
}

private extension QTIPKernels {
    static func cpuGatherMatmul(
        flatX: MLXArray,
        packed3D: MLXArray,
        scales2D: MLXArray,
        tlut: MLXArray,
        flatIndices: MLXArray,
        rate: QTIPRate
    ) -> MLXArray {
        let numExperts = packed3D.dim(0)
        let tileCols = flatX.dim(1) / 16
        let assignments = flatIndices.size
        let xRows = flatX.dim(0)
        let xF32 = flatX.asType(.float32)
        let idxArr = flatIndices.asType(.int32)
        eval(xF32, idxArr)
        let idxHost = idxArr.asArray(Int32.self)
        var decodedCache: [Int: MLXArray] = [:]
        var rowsOut: [MLXArray] = []
        rowsOut.reserveCapacity(assignments)
        let groupSize = max(1, assignments / max(1, xRows))
        for a in 0 ..< assignments {
            let rawE = Int(idxHost[a])
            let e = (rawE >= 0 && rawE < numExperts) ? rawE : (numExperts - 1)
            let wExpert: MLXArray
            if let cached = decodedCache[e] {
                wExpert = cached
            } else {
                let dec = QTIPMetal.decode(
                    packed: packed3D[e], rowScales: scales2D[e], tlut: tlut,
                    rate: rate, tileCols: tileCols, dtype: .float32)
                decodedCache[e] = dec
                wExpert = dec
            }
            let xRow = xRows == assignments ? a : (a / groupSize)
            rowsOut.append(matmul(xF32[xRow ..< (xRow + 1)], wExpert.T)[0])
        }
        return stacked(rowsOut, axis: 0)
    }
}

package final class QTIPSwitchLinear: Module, Quantized {
    package let rate: QTIPRate
    package let expertCount: Int
    package let outFeatures: Int
    package let inFeatures: Int
    package let rhtIn: KroneckerRHT?
    package let rhtOut: KroneckerRHT?
    package let inBuffers: FusedKroneckerRHT.Buffers?
    package let outBuffers: FusedKroneckerRHT.Buffers?
    @ParameterInfo(key: "weight") package var weight: MLXArray
    @ParameterInfo(key: "row_scales") package var rowScales: MLXArray
    @ParameterInfo(key: "tlut") package var tlut: MLXArray
    @ParameterInfo(key: "bias") package var bias: MLXArray?

    public var groupSize: Int { 16 }
    public var bits: Int { max(2, rate.stepBits / 2) }
    public var mode: QuantizationMode { .affine }

    package init(
        packedExperts: MLXArray,
        rowScales: MLXArray,
        tlut: MLXArray,
        bias: MLXArray? = nil,
        rate: QTIPRate,
        spec: QTIPLayerSpec? = nil,
        buffers: FusedKroneckerRHT.Pair? = nil
    ) {
        let resolvedRate = spec?.rate ?? rate
        let experts = packedExperts.dim(0)
        let tileRows = packedExperts.dim(1)
        let wordsPerExpert = max(1, experts * tileRows * resolvedRate.wordsPerTile)
        let tileCols = spec.map { $0.rhtIn.dimension / 16 } ?? (packedExperts.size / wordsPerExpert)
        self.rate = resolvedRate
        self.expertCount = experts
        self.outFeatures = spec?.rhtOut.dimension ?? (tileRows * 16)
        self.inFeatures = spec?.rhtIn.dimension ?? (tileCols * 16)
        self.rhtIn = spec?.rhtIn
        self.rhtOut = spec?.rhtOut
        self.inBuffers = buffers?.inBuffers ?? spec.map { FusedKroneckerRHT.Buffers($0.rhtIn) }
        self.outBuffers = buffers?.outBuffers ?? spec.map { FusedKroneckerRHT.Buffers($0.rhtOut) }
        self._weight.wrappedValue = packedExperts
        self._rowScales.wrappedValue = rowScales
        self._tlut.wrappedValue = tlut
        self._bias.wrappedValue = bias
        super.init()
        freeze()
        train(false)
    }
}

extension QTIPSwitchLinear {
    package func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false) -> MLXArray {
        _ = sortedIndices
        if Device.defaultDevice() == .gpu,
           [x, indices, weight, rowScales, tlut, bias].contains(where: { $0?.dtype == .float64 }) {
            return Device.withDefaultDevice(.cpu) { self(x, indices, sortedIndices: sortedIndices) }
        }
        precondition(x.ndim >= 1 && weight.ndim >= 3, "QTIPSwitchLinear requires ndim >= 1 input and ndim >= 3 weight")
        var y: MLXArray
        if x.size > 0, indices.size > 0, weight.size > 0 {
            let xRot = inBuffers.map { FusedKroneckerRHT.forward(x, buffers: $0, outputDType: x.dtype) } ?? x
            y = QTIPKernels.gatherMatmul(
                x: xRot, packedExperts: weight, rowScales: rowScales, tlut: tlut, indices: indices, rate: rate)
            if let outBuffers { y = FusedKroneckerRHT.inverse(y, buffers: outBuffers, outputDType: x.dtype) }
            if y.dtype != x.dtype { y = y.asType(x.dtype) }
        } else {
            let outFeat = weight.ndim >= 2 ? weight.dim(1) * 16 : 0
            y = MLXArray.zeros(QTIPKernels.gatherOutputShape(x: x, indices: indices, outFeatures: outFeat), dtype: x.dtype)
        }
        return addBiasIfNeeded(y, x: x, indices: indices)
    }

    private func addBiasIfNeeded(_ y: MLXArray, x: MLXArray, indices: MLXArray) -> MLXArray {
        guard let bias, y.size > 0, weight.dim(0) > 0, bias.ndim >= 1, bias.dim(0) > 0, bias.size > 0 else { return y }
        let castBias = bias.dtype == x.dtype ? bias : bias.asType(x.dtype)
        if bias.ndim == 1 {
            precondition(bias.dim(0) == y.dim(-1), "QTIPSwitchLinear 1D bias size mismatch")
            return y + castBias
        }
        let idx64 = indices.asType(.int64)
        let nBias = Int64(bias.dim(0))
        let safeIdx = which((idx64 .>= Int64(0)) .&& (idx64 .< nBias), idx64, Int64(nBias - 1)).asType(.int32)
        if x.ndim == indices.ndim + 2 {
            let bShape = zip(Array(x.shape.dropLast(2)), indices.shape).map(QTIPKernels.broadcastDim)
            return y + expandedDimensions(castBias[broadcast(safeIdx, to: bShape)], axis: -2)
        }
        let targetLead = Array(y.shape.dropLast())
        let alignedIdx = safeIdx.shape == targetLead ? safeIdx : broadcast(safeIdx, to: targetLead)
        return y + castBias[alignedIdx]
    }
}
