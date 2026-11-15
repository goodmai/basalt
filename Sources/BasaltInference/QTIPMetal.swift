import Foundation
import MLX
import MLXNN

private let qtipMetalHeader = """
    inline ushort qtip_code_idx(uint edge) {
        uint h = edge;
        h = ((h ^ (h >> 8u)) * 0x9E37u) & 0xFFFFu;
        h = ((h ^ (h >> 7u)) * 0xC2B5u) & 0xFFFFu;
        h ^= (h >> 9u);
        return ushort(h & 2047u);
    }

    inline uint qtip_swz(uint s) {
        return s ^ (((s >> 5u) ^ (s >> 7u)) & 31u);
    }

    inline float2 qtip_hyb_lookup(ushort edge, const threadgroup half2* tlut) {
        ushort code = qtip_code_idx(uint(edge));
        float sx = (code & 512u) != 0u ? -1.0f : 1.0f;
        float sy = (code & 1024u) != 0u ? -1.0f : 1.0f;
        float2 base = float2(tlut[code & 511u]);
        return float2(base.x * sx, base.y * sy);
    }

    inline ushort qtip_read_edge_tg(const threadgroup uint* words, uint step, uint stepBits, uint numWords) {
        uint bitPos = step * stepBits;
        uint wIdx = bitPos >> 5u;
        uint shift = bitPos & 31u;
        ulong lo = ulong(words[wIdx]);
        ulong hi = ulong(words[(wIdx + 1u) % numWords]);
        return ushort(((lo | (hi << 32u)) >> shift) & 0xFFFFul);
    }

    inline ushort qtip_read_edge_dev(device const uint* words, uint step, uint stepBits, uint numWords) {
        uint bitPos = step * stepBits;
        uint wIdx = bitPos >> 5u;
        uint shift = bitPos & 31u;
        ulong lo = ulong(words[wIdx]);
        ulong hi = ulong(words[(wIdx + 1u) % numWords]);
        return ushort(((lo | (hi << 32u)) >> shift) & 0xFFFFul);
    }
    """

private let qtipEncodeSource = """
    const uint Q = uint(S) / 256u;
    const uint FANOUT = 1u << uint(K);
    const uint M = 16u - 2u * uint(K);
    const uint MID_MASK = (1u << M) - 1u;
    const uint HI_SHIFT = 16u - uint(K);
    const uint SMASK = uint(S) - 1u;
    uint tid = thread_index_in_threadgroup;
    uint tileIdx = threadgroup_position_in_grid.x;
    device const float* tilePtr = tiles + tileIdx * 256u;
    uint mid = tid & MID_MASK;
    uint uBase = (tid >> M) * Q;
    uint srcBase = mid << uint(K);

    threadgroup half2 s_tlut[512];
    threadgroup float s_cost[S];
    threadgroup float s_dist[2048];
    threadgroup ushort s_minIdx[256];
    threadgroup uint s_words[WORDS];

    for (uint idx = tid; idx < 512u; idx += 256u) {
        s_tlut[idx] = half2(half(tlut[2u * idx]), half(tlut[2u * idx + 1u]));
    }
    #pragma clang loop unroll(full)
    for (uint q = 0u; q < Q; ++q) {
        s_cost[q * 256u + tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = 128u - uint(BURNIN); step < 128u; ++step) {
        float w0 = tilePtr[2u * step];
        float w1 = tilePtr[2u * step + 1u];
        for (uint idx = tid; idx < 512u; idx += 256u) {
            float2 u = float2(s_tlut[idx]);
            float norm2 = 0.5f * (u.x * u.x + u.y * u.y);
            float dx = w0 * u.x;
            float dy = w1 * u.y;
            s_dist[idx] = norm2 - dx - dy;
            s_dist[idx + 512u] = norm2 + dx - dy;
            s_dist[idx + 1024u] = norm2 - dx + dy;
            s_dist[idx + 1536u] = norm2 + dx + dy;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float best[Q];
        #pragma clang loop unroll(full)
        for (uint ui = 0u; ui < Q; ++ui) {
            best[ui] = 1e20f;
        }
        for (uint b = 0u; b < FANOUT; ++b) {
            uint lowEdge = srcBase | b;
            float sPrevVal = s_cost[qtip_swz(lowEdge)];
            #pragma clang loop unroll(full)
            for (uint ui = 0u; ui < Q; ++ui) {
                uint edge = ((uBase + ui) << HI_SHIFT) | lowEdge;
                best[ui] = min(best[ui], sPrevVal + s_dist[qtip_code_idx(edge)]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint ui = 0u; ui < Q; ++ui) {
            uint sNext = ((uBase + ui) << M) | mid;
            s_cost[qtip_swz(sNext)] = best[ui];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float localBest = 1e20f;
    ushort localState = 0u;
    #pragma clang loop unroll(full)
    for (uint q = 0u; q < Q; ++q) {
        uint s = q * 256u + tid;
        float v = s_cost[qtip_swz(s)];
        if (v < localBest || (v == localBest && ushort(s) < localState)) {
            localBest = v;
            localState = ushort(s);
        }
    }
    s_dist[tid] = localBest;
    s_minIdx[tid] = localState;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            float otherV = s_dist[tid + stride];
            ushort otherS = s_minIdx[tid + stride];
            float curV = s_dist[tid];
            ushort curS = s_minIdx[tid];
            if (otherV < curV || (otherV == curV && otherS < curS)) {
                s_dist[tid] = otherV;
                s_minIdx[tid] = otherS;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint s0 = uint(s_minIdx[0]);

    #pragma clang loop unroll(full)
    for (uint q = 0u; q < Q; ++q) {
        uint s = q * 256u + tid;
        s_cost[qtip_swz(s)] = (s == s0) ? 0.0f : 1e20f;
    }
    if (tid < uint(WORDS)) {
        s_words[tid] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device uchar* tileBP = bp_out + tileIdx * (128u * uint(S));
    for (uint step = 0u; step < 128u; ++step) {
        float w0 = tilePtr[2u * step];
        float w1 = tilePtr[2u * step + 1u];
        for (uint idx = tid; idx < 512u; idx += 256u) {
            float2 u = float2(s_tlut[idx]);
            float norm2 = 0.5f * (u.x * u.x + u.y * u.y);
            float dx = w0 * u.x;
            float dy = w1 * u.y;
            s_dist[idx] = norm2 - dx - dy;
            s_dist[idx + 512u] = norm2 + dx - dy;
            s_dist[idx + 1024u] = norm2 - dx + dy;
            s_dist[idx + 1536u] = norm2 + dx + dy;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float best[Q];
        uchar bestB[Q];
        #pragma clang loop unroll(full)
        for (uint ui = 0u; ui < Q; ++ui) {
            best[ui] = 1e20f;
            bestB[ui] = 0u;
        }
        for (uint b = 0u; b < FANOUT; ++b) {
            uint lowEdge = srcBase | b;
            float sPrevVal = s_cost[qtip_swz(lowEdge)];
            #pragma clang loop unroll(full)
            for (uint ui = 0u; ui < Q; ++ui) {
                uint edge = ((uBase + ui) << HI_SHIFT) | lowEdge;
                float cand = sPrevVal + s_dist[qtip_code_idx(edge)];
                bool better = cand < best[ui];
                best[ui] = better ? cand : best[ui];
                bestB[ui] = better ? uchar(b) : bestB[ui];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint ui = 0u; ui < Q; ++ui) {
            uint sNext = ((uBase + ui) << M) | mid;
            s_cost[qtip_swz(sNext)] = best[ui];
            tileBP[step * uint(S) + sNext] = bestB[ui];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        uint cur = s0;
        for (int step = 127; step >= 0; --step) {
            uint b = uint(tileBP[uint(step) * uint(S) + cur]);
            uint bitPos = uint(step) * uint(K);
            uint wIdx = bitPos >> 5u;
            uint shift = bitPos & 31u;
            s_words[wIdx] |= (b << shift);
            if (shift + uint(K) > 32u) {
                s_words[wIdx + 1u] |= (b >> (32u - shift));
            }
            cur = ((cur << uint(K)) | b) & SMASK;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(WORDS)) {
        packed_out[tileIdx * uint(WORDS) + tid] = s_words[tid];
    }
    if (tid < 128u) {
        ushort edge = qtip_read_edge_tg(s_words, tid, uint(K), uint(WORDS));
        float2 dec = qtip_hyb_lookup(edge, s_tlut);
        hat_out[tileIdx * 256u + 2u * tid] = dec.x;
        hat_out[tileIdx * 256u + 2u * tid + 1u] = dec.y;
    }
    """

private let qtipDecodeSource = """
    uint tid = thread_index_in_threadgroup;
    uint tileIdx = threadgroup_position_in_grid.x;
    uint tr = tileIdx / uint(TCOLS);
    uint tc = tileIdx % uint(TCOLS);

    threadgroup half2 s_tlut[512];
    for (uint idx = tid; idx < 512u; idx += 128u) {
        s_tlut[idx] = half2(half(tlut[2u * idx]), half(tlut[2u * idx + 1u]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const uint* tileWords = packed + tileIdx * uint(WORDS);
    ushort edge = qtip_read_edge_dev(tileWords, tid, uint(K), uint(WORDS));
    float scale = float(row_scales[tr]);
    float2 dec = qtip_hyb_lookup(edge, s_tlut) * scale;

    uint localRow = tid >> 3u;
    uint localCol = (tid & 7u) << 1u;
    uint globalRow = tr * 16u + localRow;
    uint globalCol = tc * 16u + localCol;
    uint outOffset = globalRow * (uint(TCOLS) * 16u) + globalCol;
    out[outOffset] = static_cast<T>(dec.x);
    out[outOffset + 1u] = static_cast<T>(dec.y);
    """

package enum QTIPMetal {
    static let dequantTileBytes = 256 << 20
    static let maxEncodeTilesPerLaunch = 2048
    static let burnInSteps = 24

    package static func encodeTiles(
        _ tiles: MLXArray,
        tlut: MLXArray,
        rate: QTIPRate,
        maxTilesPerLaunch: Int = maxEncodeTilesPerLaunch
    ) -> (packed: MLXArray, decoded: MLXArray) {
        let cap = max(1, maxTilesPerLaunch)
        let numTiles = tiles.dim(0)
        if numTiles <= cap {
            return encodeLaunch(tiles, count: numTiles, tlut: tlut, rate: rate)
        }
        var packedParts: [MLXArray] = []
        var hatParts: [MLXArray] = []
        for start in stride(from: 0, to: numTiles, by: cap) {
            let end = min(numTiles, start + cap)
            let (pChunk, hChunk) = encodeLaunch(tiles[start ..< end], count: end - start, tlut: tlut, rate: rate)
            eval(pChunk, hChunk)
            packedParts.append(pChunk)
            hatParts.append(hChunk)
        }
        return (concatenated(packedParts, axis: 0), concatenated(hatParts, axis: 0))
    }

    private static func encodeLaunch(
        _ chunk: MLXArray, count: Int, tlut: MLXArray, rate: QTIPRate
    ) -> (packed: MLXArray, decoded: MLXArray) {
        let outputs = encodeKernel(
            [chunk, tlut],
            template: [
                ("K", rate.stepBits), ("S", rate.numStates), ("WORDS", rate.wordsPerTile),
                ("BURNIN", burnInSteps), ("TL", tlut.dtype)
            ],
            grid: (256 * count, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [
                [count, rate.wordsPerTile],
                [count, 256],
                [count, 128 * rate.numStates]
            ],
            outputDTypes: [.uint32, .float32, .uint8]
        )
        return (outputs[0], outputs[1])
    }

    package static func decode(
        packed: MLXArray, rowScales: MLXArray, tlut: MLXArray, rate: QTIPRate, tileCols: Int, dtype: DType
    ) -> MLXArray {
        let tileRows = packed.dim(0)
        let totalTiles = tileRows * tileCols
        return decodeKernel(
            [packed, rowScales, tlut],
            template: [
                ("K", rate.stepBits), ("WORDS", rate.wordsPerTile), ("TCOLS", tileCols),
                ("T", dtype), ("TS", rowScales.dtype), ("TL", tlut.dtype)
            ],
            grid: (128 * totalTiles, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[tileRows * 16, tileCols * 16]],
            outputDTypes: [dtype]
        )[0]
    }
}

private let encodeKernel = MLXFast.metalKernel(
    name: "qtip_encode_tiles",
    inputNames: ["tiles", "tlut"],
    outputNames: ["packed_out", "hat_out", "bp_out"],
    source: qtipEncodeSource,
    header: qtipMetalHeader
)

private let decodeKernel = MLXFast.metalKernel(
    name: "qtip_decode_tiles",
    inputNames: ["packed", "row_scales", "tlut"],
    outputNames: ["out"],
    source: qtipDecodeSource,
    header: qtipMetalHeader
)

package struct QTIPLayerSpec: Sendable {
    package let rate: QTIPRate
    package let rhtIn: KroneckerRHT
    package let rhtOut: KroneckerRHT

    package init(rate: QTIPRate, rhtIn: KroneckerRHT, rhtOut: KroneckerRHT) {
        self.rate = rate
        self.rhtIn = rhtIn
        self.rhtOut = rhtOut
    }
}

package final class QTIPLinear: Linear, Quantized {
    private static let legacyParameterKeys = ["inSigns", "outSigns", "inQ", "outQ"]

    package let rate: QTIPRate
    package let outFeatures: Int
    package let inFeatures: Int
    package let rhtIn: KroneckerRHT
    package let rhtOut: KroneckerRHT
    package let inBuffers: FusedKroneckerRHT.Buffers
    package let outBuffers: FusedKroneckerRHT.Buffers
    @ParameterInfo(key: "row_scales") package var rowScales: MLXArray
    @ParameterInfo(key: "tlut") package var tlut: MLXArray

    public var groupSize: Int { 16 }
    public var bits: Int { max(2, rate.stepBits / 2) }
    public var mode: QuantizationMode { .affine }
    public override var shape: (Int, Int) { (outFeatures, inFeatures) }

    package init(
        packedWeight: MLXArray,
        rowScales: MLXArray,
        tlut: MLXArray,
        bias: MLXArray?,
        spec: QTIPLayerSpec,
        buffers: FusedKroneckerRHT.Pair? = nil
    ) {
        self.rate = spec.rate
        self.outFeatures = spec.rhtOut.dimension
        self.inFeatures = spec.rhtIn.dimension
        self.rhtIn = spec.rhtIn
        self.rhtOut = spec.rhtOut
        self.inBuffers = buffers?.inBuffers ?? FusedKroneckerRHT.Buffers(spec.rhtIn)
        self.outBuffers = buffers?.outBuffers ?? FusedKroneckerRHT.Buffers(spec.rhtOut)
        self._rowScales.wrappedValue = rowScales
        self._tlut.wrappedValue = tlut
        super.init(weight: packedWeight, bias: bias)
        freeze()
    }

    @discardableResult
    public override func update(
        parameters: ModuleParameters,
        verify: VerifyUpdate,
        path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        var cleaned = parameters
        for key in Self.legacyParameterKeys {
            cleaned[key] = nil
        }
        try super.update(parameters: cleaned, verify: verify, path: path, modulePath: modulePath)
        return self
    }

    package func decodedRotatedWeight(dtype: DType = .float32) -> MLXArray {
        QTIPMetal.decode(
            packed: weight, rowScales: rowScales, tlut: tlut, rate: rate,
            tileCols: inFeatures / 16, dtype: dtype)
    }

    package func decodedWeight(dtype: DType = .float32) -> MLXArray {
        let wRot = decodedRotatedWeight(dtype: .float32)
        let wUnrotOut = FusedKroneckerRHT.inverse(wRot.T, buffers: outBuffers, outputDType: .float32).T
        return FusedKroneckerRHT.inverse(wUnrotOut, buffers: inBuffers, outputDType: dtype)
    }

    package func rotateInput(_ x: MLXArray) -> MLXArray {
        FusedKroneckerRHT.forward(x, buffers: inBuffers, outputDType: x.dtype)
    }

    package func rotateNormalizedInput(_ x: MLXArray, normWeight: MLXArray, eps: Float) -> MLXArray {
        FusedKroneckerRHT.rmsNormForward(x, normWeight: normWeight, eps: eps, buffers: inBuffers, outputDType: x.dtype)
    }

    package func matmulRotated(_ xRot: MLXArray) -> MLXArray {
        let tileRows = weight.dim(0)
        let tileCols = inFeatures / 16
        let rowsPerChunk = max(1, QTIPMetal.dequantTileBytes / max(1, 16 * inFeatures * xRot.dtype.size))
        var parts: [MLXArray] = []
        for start in stride(from: 0, to: tileRows, by: rowsPerChunk) {
            let end = min(tileRows, start + rowsPerChunk)
            let wChunk = QTIPMetal.decode(
                packed: weight[start ..< end],
                rowScales: rowScales[start ..< end],
                tlut: tlut,
                rate: rate,
                tileCols: tileCols,
                dtype: xRot.dtype
            )
            parts.append(matmul(xRot, wChunk.T))
        }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: -1)
    }

    package func callPreRotated(_ xRot: MLXArray) -> MLXArray {
        FusedKroneckerRHT.inverse(matmulRotated(xRot), buffers: outBuffers, bias: bias, outputDType: xRot.dtype)
    }

    override package func callAsFunction(_ x: MLXArray) -> MLXArray {
        callPreRotated(rotateInput(x))
    }
}

extension QTIPLinear {
    package func sharesInputRotation(with other: QTIPLinear) -> Bool {
        inFeatures == other.inFeatures && rhtIn.seed == other.rhtIn.seed
    }

    package func sharesOutputRotation(with other: QTIPLinear) -> Bool {
        outFeatures == other.outFeatures && rhtOut.seed == other.rhtOut.seed
    }

    package static func applySiblings(
        _ x: MLXArray,
        layers: [QTIPLinear],
        norm: (weight: MLXArray, eps: Float)? = nil
    ) -> [MLXArray] {
        guard !layers.isEmpty else { return [] }
        var rotatedByKey: [String: MLXArray] = [:]
        rotatedByKey.reserveCapacity(layers.count)
        return layers.map { layer in
            let key = "\(layer.inFeatures):\(layer.rhtIn.seed)"
            let xRot: MLXArray
            if let cached = rotatedByKey[key] {
                xRot = cached
            } else {
                let computed = norm.map {
                    layer.rotateNormalizedInput(x, normWeight: $0.weight, eps: $0.eps)
                } ?? layer.rotateInput(x)
                rotatedByKey[key] = computed
                xRot = computed
            }
            return layer.callPreRotated(xRot)
        }
    }

    package static func applyMoECollapse(
        _ expertInputs: [MLXArray],
        experts: [QTIPLinear],
        weights: MLXArray
    ) -> MLXArray {
        precondition(!experts.isEmpty && expertInputs.count == experts.count, "QTIPLinear.applyMoECollapse count mismatch")
        let first = experts[0]
        let outType = expertInputs[0].dtype
        if experts.allSatisfy({ $0.sharesOutputRotation(with: first) && $0.bias == nil }) {
            let yRots = zip(expertInputs, experts).map { input, expert in
                expert.matmulRotated(expert.rotateInput(input))
            }
            return FusedKroneckerRHT.weightedInverseCollapse(
                stacked(yRots, axis: -2),
                weights: weights,
                buffers: first.outBuffers,
                outputDType: outType
            )
        }
        let ys = zip(expertInputs, experts).map { $1($0) }
        let weighted = stacked(ys, axis: -2).asType(.float32) * expandedDimensions(weights.asType(.float32), axis: -1)
        return weighted.sum(axis: -2).asType(outType)
    }
}
