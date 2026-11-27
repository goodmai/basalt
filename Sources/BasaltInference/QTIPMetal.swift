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

    template <typename Ptr>
    inline ushort qtip_read_edge_dev(Ptr words, size_t base, uint step, uint stepBits, uint numWords) {
        uint bitPos = step * stepBits;
        uint wIdx = bitPos >> 5u;
        uint shift = bitPos & 31u;
        ulong lo = ulong(words[base + size_t(wIdx)]);
        ulong hi = ulong(words[base + size_t((wIdx + 1u) % numWords)]);
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
    const size_t tileIdx = size_t(threadgroup_position_in_grid.y);
    const size_t tileBase = tileIdx * size_t(256);
    uint mid = tid & MID_MASK;
    uint uBase = (tid >> M) * Q;
    uint srcBase = mid << uint(K);

    threadgroup half2 s_tlut[512];
    threadgroup float s_cost[S];
    threadgroup float s_dist[2048];
    threadgroup ushort s_minIdx[256];
    threadgroup uint s_words[WORDS];

    for (uint idx = tid; idx < 512u; idx += 256u) {
        s_tlut[idx] = half2(half(float(tlut[2u * idx])), half(float(tlut[2u * idx + 1u])));
    }
    #pragma clang loop unroll(full)
    for (uint q = 0u; q < Q; ++q) {
        s_cost[q * 256u + tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = 128u - uint(BURNIN); step < 128u; ++step) {
        float w0 = float(tiles[tileBase + size_t(2u * step)]);
        float w1 = float(tiles[tileBase + size_t(2u * step + 1u)]);
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

    const size_t bpBase = tileIdx * size_t(128) * size_t(S);
    for (uint step = 0u; step < 128u; ++step) {
        float w0 = float(tiles[tileBase + size_t(2u * step)]);
        float w1 = float(tiles[tileBase + size_t(2u * step + 1u)]);
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
            bp_out[bpBase + size_t(step) * size_t(S) + size_t(sNext)] = bestB[ui];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0u) {
        uint cur = s0;
        for (int step = 127; step >= 0; --step) {
            uint b = uint(bp_out[bpBase + size_t(step) * size_t(S) + size_t(cur)]);
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
        packed_out[tileIdx * size_t(WORDS) + size_t(tid)] = s_words[tid];
    }
    if (tid < 128u) {
        ushort edge = qtip_read_edge_tg(s_words, tid, uint(K), uint(WORDS));
        float2 dec = qtip_hyb_lookup(edge, s_tlut);
        hat_out[tileBase + size_t(2u * tid)] = dec.x;
        hat_out[tileBase + size_t(2u * tid + 1u)] = dec.y;
    }
    """

private let qtipDecodeSource = """
    const uint tid = thread_index_in_threadgroup;
    const uint tc = threadgroup_position_in_grid.y;
    const uint tr = threadgroup_position_in_grid.z;
    const uint tcols = uint(packed_shape[1]);

    threadgroup half2 s_tlut[512];
    for (uint idx = tid; idx < 512u; idx += 128u) {
        s_tlut[idx] = half2(half(float(tlut[2u * idx])), half(float(tlut[2u * idx + 1u])));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const size_t tileIdx = size_t(tr) * size_t(tcols) + size_t(tc);
    const size_t wordBase = tileIdx * size_t(WORDS);
    ushort edge = qtip_read_edge_dev(packed, wordBase, tid, uint(K), uint(WORDS));
    float scale = float(row_scales[tr]);
    float2 dec = qtip_hyb_lookup(edge, s_tlut) * scale;

    const size_t globalRow = size_t(tr) * size_t(16) + size_t(tid >> 3u);
    const size_t globalCol = size_t(tc) * size_t(16) + size_t((tid & 7u) << 1u);
    const size_t outOffset = globalRow * (size_t(tcols) * size_t(16)) + globalCol;
    out[outOffset] = static_cast<T>(dec.x);
    out[outOffset + 1u] = static_cast<T>(dec.y);
    """

package enum QTIPMetal {
    static let maxEncodeTilesPerLaunch = 2048
    static let burnInSteps = 24
    static let supportedDTypes: Set<DType> = [.float32, .float16, .bfloat16]

    private static func cleanFloat(_ array: MLXArray) -> MLXArray {
        if supportedDTypes.contains(array.dtype) { return array }
        return array.asType(.float32, stream: array.dtype == .float64 ? .cpu : .default)
    }

    package static func encodeTiles(
        _ tiles: MLXArray,
        tlut: MLXArray,
        rate: QTIPRate,
        maxTilesPerLaunch: Int = maxEncodeTilesPerLaunch
    ) -> (packed: MLXArray, decoded: MLXArray) {
        precondition(tiles.ndim >= 1, "QTIPMetal.encodeTiles requires ndim >= 1 tiles")
        if Device.defaultDevice() == .gpu, (tiles.dtype == .float64 || tlut.dtype == .float64) {
            return Device.withDefaultDevice(.cpu) {
                encodeTiles(tiles, tlut: tlut, rate: rate, maxTilesPerLaunch: maxTilesPerLaunch)
            }
        }
        let numTiles = tiles.dim(0)
        guard numTiles > 0 else {
            return (
                MLXArray.zeros([0, rate.wordsPerTile], dtype: .uint32),
                MLXArray.zeros([0, 256], dtype: .float32)
            )
        }
        precondition(tiles.size == numTiles * 256, "QTIPMetal.encodeTiles requires 256 elements per tile")
        precondition(tlut.size == 1024, "QTIPMetal.encodeTiles requires 1024-element tlut")
        let flatTiles = tiles.reshaped(numTiles, 256)
        guard Device.defaultDevice() == .gpu else {
            return cpuEncodeTiles(flatTiles, count: numTiles, tlut: tlut, rate: rate)
        }
        let cap = min(max(1, maxTilesPerLaunch), Int(Int32.max))
        if numTiles <= cap {
            return encodeLaunch(flatTiles, count: numTiles, tlut: tlut, rate: rate)
        }
        var packedParts: [MLXArray] = []
        var hatParts: [MLXArray] = []
        for start in stride(from: 0, to: numTiles, by: cap) {
            let end = min(numTiles, start + cap)
            let (pChunk, hChunk) = encodeLaunch(flatTiles[start ..< end], count: end - start, tlut: tlut, rate: rate)
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
            [cleanFloat(chunk), cleanFloat(tlut)],
            template: [
                ("K", rate.stepBits), ("S", rate.numStates), ("WORDS", rate.wordsPerTile),
                ("BURNIN", burnInSteps)
            ],
            grid: (256, count, 1),
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

    private static func cpuEncodeTiles(
        _ tiles: MLXArray, count: Int, tlut: MLXArray, rate: QTIPRate
    ) -> (packed: MLXArray, decoded: MLXArray) {
        let tilesF32 = tiles.asType(.float32)
        let tlutF16 = tlut.asType(.float16).asType(.float32)
        eval(tilesF32, tlutF16)
        let flatHost = tilesF32.asArray(Float.self)
        let tlutHost = tlutF16.asArray(Float.self)
        let edgeCodes = QTIPCodebook.edgeCodeTableValues()
        var allWords: [UInt32] = []
        allWords.reserveCapacity(count * rate.wordsPerTile)
        var allDecoded: [Float] = []
        allDecoded.reserveCapacity(count * 256)
        for t in 0 ..< count {
            let slice = Array(flatHost[(t * 256) ..< ((t + 1) * 256)])
            let (words, decoded) = QTIPReference.encodeTile(
                tile: slice, tlut: tlutHost, edgeCodes: edgeCodes, rate: rate, burnInSteps: burnInSteps)
            allWords.append(contentsOf: words)
            allDecoded.append(contentsOf: decoded)
        }
        return (MLXArray(allWords, [count, rate.wordsPerTile]), MLXArray(allDecoded, [count, 256]))
    }

    package static func decode(
        packed: MLXArray, rowScales: MLXArray, tlut: MLXArray, rate: QTIPRate, tileCols: Int, dtype: DType
    ) -> MLXArray {
        precondition(packed.ndim >= 2, "QTIPMetal.decode requires ndim >= 2 packed weights")
        if Device.defaultDevice() == .gpu,
           dtype == .float64 || [packed, rowScales, tlut].contains(where: { $0.dtype == .float64 }) {
            return Device.withDefaultDevice(.cpu) {
                decode(packed: packed, rowScales: rowScales, tlut: tlut, rate: rate, tileCols: tileCols, dtype: dtype)
            }
        }
        let tileRows = packed.dim(0)
        guard tileRows > 0, tileCols > 0 else {
            return MLXArray.zeros([max(0, tileRows) * 16, max(0, tileCols) * 16], dtype: dtype)
        }
        precondition(packed.size == tileRows * tileCols * rate.wordsPerTile, "QTIPMetal.decode packed size mismatch")
        precondition(rowScales.size == tileRows, "QTIPMetal.decode rowScales size mismatch")
        precondition(tlut.size == 1024, "QTIPMetal.decode requires 1024-element tlut")
        guard Device.defaultDevice() == .gpu,
              packed.dtype == .uint32,
              supportedDTypes.contains(dtype),
              tileRows <= Int(Int32.max),
              tileCols <= Int(Int32.max)
        else {
            return cpuDecode(packed: packed, rowScales: rowScales, tlut: tlut, rate: rate, tileCols: tileCols, dtype: dtype)
        }
        let packed3D = packed.reshaped(tileRows, tileCols, rate.wordsPerTile)
        let scalesClean = cleanFloat(rowScales).reshaped(tileRows)
        let tlutClean = cleanFloat(tlut)
        return decodeKernel(
            [packed3D, scalesClean, tlutClean],
            template: [("K", rate.stepBits), ("WORDS", rate.wordsPerTile), ("T", dtype)],
            grid: (128, tileCols, tileRows),
            threadGroup: (128, 1, 1),
            outputShapes: [[tileRows * 16, tileCols * 16]],
            outputDTypes: [dtype]
        )[0]
    }

    private static func cpuDecode(
        packed: MLXArray, rowScales: MLXArray, tlut: MLXArray, rate: QTIPRate, tileCols: Int, dtype: DType
    ) -> MLXArray {
        let tileRows = packed.dim(0)
        let wpt = rate.wordsPerTile
        let packedU32 = packed.asType(.uint32).reshaped(tileRows, tileCols, wpt)
        let scalesF32 = rowScales.reshaped(tileRows).asType(.float32)
        let tlutF16 = tlut.asType(.float16).asType(.float32)
        eval(packedU32, scalesF32, tlutF16)
        let words = packedU32.asArray(UInt32.self)
        let scales = scalesF32.asArray(Float.self)
        let tlutHost = tlutF16.asArray(Float.self)
        let (outFeatures, inFeatures) = (tileRows * 16, tileCols * 16)
        var host = [Float](repeating: 0, count: outFeatures * inFeatures)
        for tr in 0 ..< tileRows {
            let scale = scales[tr]
            for tc in 0 ..< tileCols {
                let tIdx = tr * tileCols + tc
                let slice = Array(words[(tIdx * wpt) ..< ((tIdx + 1) * wpt)])
                let tile = QTIPReference.decodeTile(words: slice, tlut: tlutHost, rate: rate, scale: scale)
                for r in 0 ..< 16 {
                    let dstBase = (tr * 16 + r) * inFeatures + tc * 16
                    for c in 0 ..< 16 { host[dstBase + c] = tile[r * 16 + c] }
                }
            }
        }
        return MLXArray(host, [outFeatures, inFeatures]).asType(dtype)
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
        packedWeight: MLXArray, rowScales: MLXArray, tlut: MLXArray,
        bias: MLXArray?, spec: QTIPLayerSpec, buffers: FusedKroneckerRHT.Pair? = nil
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
        train(false)
    }

    @discardableResult
    public override func update(
        parameters: ModuleParameters, verify: VerifyUpdate,
        path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        var cleaned = parameters
        for key in Self.legacyParameterKeys { cleaned[key] = nil }
        try super.update(parameters: cleaned, verify: verify, path: path, modulePath: modulePath)
        return self
    }

    package func decodedRotatedWeight(dtype: DType = .float32) -> MLXArray {
        QTIPMetal.decode(packed: weight, rowScales: rowScales, tlut: tlut, rate: rate, tileCols: inFeatures / 16, dtype: dtype)
    }

    private static func onCPUIfFloat64(_ dtypes: [DType?], _ body: () -> MLXArray) -> MLXArray {
        Device.defaultDevice() == .gpu && dtypes.contains(.float64) ? Device.withDefaultDevice(.cpu, body) : body()
    }

    package func decodedWeight(dtype: DType = .float32) -> MLXArray {
        Self.onCPUIfFloat64([dtype, weight.dtype, rowScales.dtype, tlut.dtype]) {
            let wRot = decodedRotatedWeight(dtype: .float32)
            let wUnrotOut = FusedKroneckerRHT.inverse(wRot.T, buffers: outBuffers, outputDType: .float32).T
            let res = FusedKroneckerRHT.inverse(wUnrotOut, buffers: inBuffers, outputDType: dtype)
            return res.dtype == dtype ? res : res.asType(dtype)
        }
    }

    package func rotateInput(_ x: MLXArray) -> MLXArray {
        precondition(x.ndim > 0 && x.dim(-1) == inFeatures, "QTIPLinear input shape mismatch")
        return Self.onCPUIfFloat64([x.dtype]) {
            guard x.size > 0 else { return x }
            let res = training
                ? FusedKroneckerRHT.referenceForward(x, spec: inBuffers.spec, signs: inBuffers.fallbackSigns, qMatrix: inBuffers.fallbackQ, outputDType: x.dtype)
                : FusedKroneckerRHT.forward(x, buffers: inBuffers, outputDType: x.dtype)
            return res.dtype == x.dtype ? res : res.asType(x.dtype)
        }
    }

    package func rotateNormalizedInput(_ x: MLXArray, normWeight: MLXArray, eps: Float) -> MLXArray {
        precondition(x.ndim > 0 && x.dim(-1) == inFeatures, "QTIPLinear input shape mismatch")
        precondition(normWeight.size == inFeatures, "QTIPLinear normWeight size mismatch")
        return Self.onCPUIfFloat64([x.dtype, normWeight.dtype]) {
            guard x.size > 0 else { return x }
            let res = training
                ? FusedKroneckerRHT.referenceRMSNormForward(x, normWeight: normWeight, eps: eps, spec: inBuffers.spec, outputDType: x.dtype)
                : FusedKroneckerRHT.rmsNormForward(x, normWeight: normWeight, eps: eps, buffers: inBuffers, outputDType: x.dtype)
            return res.dtype == x.dtype ? res : res.asType(x.dtype)
        }
    }

    package func matmulRotated(_ xRot: MLXArray) -> MLXArray {
        QTIPKernels.linear(xRot, packed: weight, rowScales: rowScales, tlut: tlut, rate: rate, inFeatures: inFeatures)
    }

    package func callPreRotated(_ xRot: MLXArray) -> MLXArray {
        Self.onCPUIfFloat64([xRot.dtype, weight.dtype, rowScales.dtype, tlut.dtype, bias?.dtype]) {
            let mat = matmulRotated(xRot)
            guard mat.size > 0 else { return mat }
            let res = training
                ? FusedKroneckerRHT.referenceInverse(mat, spec: outBuffers.spec, bias: bias, signs: outBuffers.fallbackSigns, qMatrix: outBuffers.fallbackQ, outputDType: xRot.dtype)
                : FusedKroneckerRHT.inverse(mat, buffers: outBuffers, bias: bias, outputDType: xRot.dtype)
            return res.dtype == xRot.dtype ? res : res.asType(xRot.dtype)
        }
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

    package static func applySiblings(_ x: MLXArray, layers: [QTIPLinear], norm: (weight: MLXArray, eps: Float)? = nil) -> [MLXArray] {
        guard !layers.isEmpty else { return [] }
        var rotatedByKey: [String: MLXArray] = [:]
        rotatedByKey.reserveCapacity(layers.count)
        return layers.map { layer in
            let key = "\(layer.inFeatures):\(layer.rhtIn.seed):\(layer.training)"
            let xRot = rotatedByKey[key] ?? {
                let c = norm.map { layer.rotateNormalizedInput(x, normWeight: $0.weight, eps: $0.eps) } ?? layer.rotateInput(x)
                rotatedByKey[key] = c
                return c
            }()
            return layer.callPreRotated(xRot)
        }
    }

    package static func applyMoECollapse(_ expertInputs: [MLXArray], experts: [QTIPLinear], weights: MLXArray) -> MLXArray {
        precondition(!experts.isEmpty && expertInputs.count == experts.count, "QTIPLinear.applyMoECollapse count mismatch")
        let inputTypes: [DType?] = expertInputs.map { Optional($0.dtype) } + [Optional(weights.dtype)]
        let expertTypes: [DType?] = experts.flatMap { e -> [DType?] in [e.weight.dtype, e.rowScales.dtype, e.tlut.dtype, e.bias?.dtype] }
        let dtypes = inputTypes + expertTypes
        return onCPUIfFloat64(dtypes) {
            let first = experts[0]
            let outType = expertInputs[0].dtype
            let anyTraining = experts.contains(where: \.training)
            if !anyTraining, experts.allSatisfy({ $0.sharesOutputRotation(with: first) && $0.bias == nil }) {
                let yRots = zip(expertInputs, experts).map { $1.matmulRotated($1.rotateInput($0)) }
                let res = FusedKroneckerRHT.weightedInverseCollapse(
                    stacked(yRots, axis: -2), weights: weights, buffers: first.outBuffers, outputDType: outType)
                return res.dtype == outType ? res : res.asType(outType)
            }
            let ys = zip(expertInputs, experts).map { $1($0) }
            let weighted = stacked(ys, axis: -2).asType(.float32) * expandedDimensions(weights.asType(.float32), axis: -1)
            return weighted.sum(axis: -2).asType(outType)
        }
    }
}
