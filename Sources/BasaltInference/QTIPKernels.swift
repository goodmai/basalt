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
        let width = inFeatures ?? xRot.dim(-1)
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
        precondition(xRot.dim(-1) > 0 && xRot.size / xRot.dim(-1) == 1, "qmv requires M == 1")
        return runMatvec(xRot, packed: packed, rowScales: rowScales, tlut: tlut, rate: rate)
    }

    package static func smallM(
        _ xRot: MLXArray, packed: MLXArray, rowScales: MLXArray, tlut: MLXArray, rate: QTIPRate
    ) -> MLXArray {
        let rows = xRot.dim(-1) > 0 ? xRot.size / xRot.dim(-1) : 0
        precondition(matvecRows.contains(rows), "smallM requires M in 1...8")
        return runMatvec(xRot, packed: packed, rowScales: rowScales, tlut: tlut, rate: rate)
    }

    private static func runMatvec(
        _ xRot: MLXArray, packed: MLXArray, rowScales: MLXArray, tlut: MLXArray, rate: QTIPRate
    ) -> MLXArray {
        let inFeatures = xRot.dim(-1)
        let packed2D = packed.ndim == 2 ? packed : packed.reshaped(packed.dim(0), -1)
        let scales1D = rowScales.reshaped(packed2D.dim(0))
        let tileRows = packed2D.dim(0)
        let outFeatures = tileRows * 16
        let leading = Array(xRot.shape.dropLast())
        let rows = xRot.size / inFeatures
        guard tileRows > 0 else {
            return MLXArray.zeros(leading + [outFeatures], dtype: xRot.dtype)
        }
        let computeDType = dtypes.contains(xRot.dtype) ? xRot.dtype : .float32
        let xFlat = (xRot.dtype == computeDType ? xRot : xRot.asType(computeDType)).reshaped(rows, inFeatures)
        let rowGroups = (tileRows + simdgroups - 1) / simdgroups
        let out = qtipMatvecKernel(
            [xFlat, packed2D, scales1D, tlut],
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

    package static func qmmTiled(
        _ xRot: MLXArray,
        packed: MLXArray,
        rowScales: MLXArray,
        tlut: MLXArray,
        rate: QTIPRate,
        tileBytes: Int = tileBytes
    ) -> MLXArray {
        let inFeatures = xRot.dim(-1)
        let packed2D = packed.ndim == 2 ? packed : packed.reshaped(packed.dim(0), -1)
        let scales1D = rowScales.reshaped(packed2D.dim(0))
        let tileRows = packed2D.dim(0)
        let outFeatures = tileRows * 16
        let leading = Array(xRot.shape.dropLast())
        let rows = inFeatures > 0 ? xRot.size / inFeatures : 0
        guard rows > 0, tileRows > 0 else {
            return MLXArray.zeros(leading + [outFeatures], dtype: xRot.dtype)
        }
        let computeDType = dtypes.contains(xRot.dtype) ? xRot.dtype : .float32
        let xCompute = xRot.dtype == computeDType ? xRot : xRot.asType(computeDType)
        let tileCols = inFeatures / 16
        let bytesPerTileRow = max(1, 16 * inFeatures * computeDType.size)
        let rowsPerChunk = max(1, max(1, tileBytes) / bytesPerTileRow)
        var parts: [MLXArray] = []
        for start in stride(from: 0, to: tileRows, by: rowsPerChunk) {
            let end = min(tileRows, start + rowsPerChunk)
            let wChunk = QTIPMetal.decode(
                packed: packed2D[start ..< end],
                rowScales: scales1D[start ..< end],
                tlut: tlut,
                rate: rate,
                tileCols: tileCols,
                dtype: computeDType
            )
            parts.append(matmul(xCompute, wChunk.T))
        }
        let out = parts.count == 1 ? parts[0] : concatenated(parts, axis: -1)
        return out.dtype == xRot.dtype ? out : out.asType(xRot.dtype)
    }
}

extension QTIPKernels {
    package static func gatherQmv(
        _ xRot: MLXArray,
        packedExperts: MLXArray,
        rowScales: MLXArray,
        tlut: MLXArray,
        indices: MLXArray,
        rate: QTIPRate
    ) -> MLXArray {
        gatherMatmul(
            x: xRot, packedExperts: packedExperts, rowScales: rowScales,
            tlut: tlut, indices: indices, rate: rate)
    }

    package static func gatherMatmul(
        x: MLXArray,
        packedExperts: MLXArray,
        rowScales: MLXArray,
        tlut: MLXArray,
        indices: MLXArray,
        rate: QTIPRate
    ) -> MLXArray {
        let inFeatures = x.dim(-1)
        let numExperts = packedExperts.dim(0)
        let tileRows = packedExperts.dim(1)
        let outFeatures = tileRows * 16
        guard indices.size > 0, x.size > 0, inFeatures > 0, numExperts > 0, tileRows > 0 else {
            return MLXArray.zeros(gatherOutputShape(x: x, indices: indices, outFeatures: outFeatures), dtype: x.dtype)
        }
        let prep = prepareGatherInputs(x: x, indices: indices, inFeatures: inFeatures, outFeatures: outFeatures)
        let assignments = prep.flatIndices.size
        let computeDType = dtypes.contains(x.dtype) ? x.dtype : .float32
        let xCompute = prep.flatX.dtype == computeDType ? prep.flatX : prep.flatX.asType(computeDType)
        let packed3D = packedExperts.ndim == 3 ? packedExperts : packedExperts.reshaped(numExperts, tileRows, -1)
        let scales2D = rowScales.reshaped(numExperts, tileRows)
        let uintIndices = prep.flatIndices.dtype == .uint32 ? prep.flatIndices : prep.flatIndices.asType(.uint32)
        let rowGroups = (tileRows + simdgroups - 1) / simdgroups
        let out = qtipGatherQmvKernel(
            [xCompute, packed3D, scales2D, tlut, uintIndices],
            template: [
                ("K", rate.stepBits), ("WORDS", rate.wordsPerTile),
                ("SG", simdgroups), ("T", computeDType)
            ],
            grid: (32, simdgroups * rowGroups, assignments),
            threadGroup: (32, simdgroups, 1),
            outputShapes: [[assignments, outFeatures]],
            outputDTypes: [computeDType]
        )[0]
        let casted = out.dtype == x.dtype ? out : out.asType(x.dtype)
        return casted.reshaped(prep.outShape)
    }

    private static func gatherOutputShape(x: MLXArray, indices: MLXArray, outFeatures: Int) -> [Int] {
        if x.ndim == indices.ndim + 2 {
            let bShape = zip(Array(x.shape.dropLast(2)), indices.shape).map(max)
            return bShape + [x.dim(-2), outFeatures]
        }
        return indices.shape + [outFeatures]
    }

    private static func prepareGatherInputs(
        x: MLXArray,
        indices: MLXArray,
        inFeatures: Int,
        outFeatures: Int
    ) -> (flatX: MLXArray, flatIndices: MLXArray, outShape: [Int]) {
        let assignments = indices.size
        if x.ndim == indices.ndim + 2 {
            let lhsBatch = Array(x.shape.dropLast(2))
            let mInner = x.dim(-2)
            let bShape = zip(lhsBatch, indices.shape).map(max)
            if mInner == 1, lhsBatch == indices.shape {
                return (x.reshaped(assignments, inFeatures), indices.reshaped(assignments), bShape + [1, outFeatures])
            }
            if mInner == 1, lhsBatch.last == 1, Array(lhsBatch.dropLast()) == Array(indices.shape.dropLast()) {
                return (
                    x.reshaped(x.size / inFeatures, inFeatures),
                    indices.reshaped(assignments),
                    bShape + [1, outFeatures]
                )
            }
            let xBroad = broadcast(x, to: bShape + [mInner, inFeatures])
            let idxBroad = repeated(expandedDimensions(broadcast(indices, to: bShape), axis: -1), count: mInner, axis: -1)
            let total = idxBroad.size
            return (xBroad.reshaped(total, inFeatures), idxBroad.reshaped(total), bShape + [mInner, outFeatures])
        }
        let outShape = indices.shape + [outFeatures]
        let xLead = Array(x.shape.dropLast())
        if xLead == indices.shape {
            return (x.reshaped(assignments, inFeatures), indices.reshaped(assignments), outShape)
        }
        if x.size == inFeatures {
            return (x.reshaped(1, inFeatures), indices.reshaped(assignments), outShape)
        }
        let idxLead = Array(indices.shape.dropLast())
        let xAligned = xLead == idxLead ? x : broadcast(x, to: idxLead + [inFeatures])
        return (xAligned.reshaped(xAligned.size / inFeatures, inFeatures), indices.reshaped(assignments), outShape)
    }
}

package final class QTIPSwitchLinear: Module, Quantized {
    package let rate: QTIPRate
    package let expertCount: Int
    package let outFeatures: Int
    package let inFeatures: Int
    package let rhtIn: KroneckerRHT?
    package let rhtOut: KroneckerRHT?
    let inSigns: MLXArray?
    let outSigns: MLXArray?
    let inQ: MLXArray?
    let outQ: MLXArray?
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
        spec: QTIPLayerSpec? = nil
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
        self.inSigns = spec.map { MLXArray($0.rhtIn.signsValues) }
        self.outSigns = spec.map { MLXArray($0.rhtOut.signsValues) }
        self.inQ = spec.flatMap { s in s.rhtIn.qValues.map { MLXArray($0, [s.rhtIn.outer, s.rhtIn.outer]) } }
        self.outQ = spec.flatMap { s in s.rhtOut.qValues.map { MLXArray($0, [s.rhtOut.outer, s.rhtOut.outer]) } }
        self._weight.wrappedValue = packedExperts
        self._rowScales.wrappedValue = rowScales
        self._tlut.wrappedValue = tlut
        self._bias.wrappedValue = bias
        super.init()
        freeze()
    }
}

extension QTIPSwitchLinear {
    package func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false) -> MLXArray {
        _ = sortedIndices
        guard x.size > 0, indices.size > 0 else {
            return QTIPKernels.gatherMatmul(
                x: x, packedExperts: weight, rowScales: rowScales, tlut: tlut, indices: indices, rate: rate)
        }
        let xRot: MLXArray
        if let rhtIn, let inSigns {
            xRot = rhtIn.forwardFloat32(x, signs: inSigns, qMatrix: inQ).asType(x.dtype)
        } else {
            xRot = x
        }
        var y = QTIPKernels.gatherMatmul(
            x: xRot, packedExperts: weight, rowScales: rowScales,
            tlut: tlut, indices: indices, rate: rate)
        if let rhtOut, let outSigns {
            y = rhtOut.inverseFloat32(y, signs: outSigns, qMatrix: outQ).asType(x.dtype)
        }
        if let bias {
            let castBias = bias.dtype == x.dtype ? bias : bias.asType(x.dtype)
            if x.ndim == indices.ndim + 2 {
                let bShape = zip(Array(x.shape.dropLast(2)), indices.shape).map(max)
                y = y + expandedDimensions(castBias[broadcast(indices, to: bShape)], axis: -2)
            } else {
                y = y + castBias[indices]
            }
        }
        return y
    }
}
