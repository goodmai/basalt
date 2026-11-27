import BasaltCore
import Foundation
import MLX
import MLXNN

package enum E8PLattice {
    package static let vectorDim = 8
    package static let baseEntries = 256

    package static let baseShellHalfUnits: [UInt8] = buildBaseShell()

    private static func buildBaseShell() -> [UInt8] {
        var rows: [[UInt8]] = []
        rows.reserveCapacity(baseEntries)
        rows.append([0, 0, 0, 0, 0, 0, 0, 0])
        for i in 0 ..< 8 {
            for j in (i + 1) ..< 8 {
                var v = [UInt8](repeating: 0, count: 8)
                v[i] = 2
                v[j] = 2
                rows.append(v)
            }
        }
        rows.append([1, 1, 1, 1, 1, 1, 1, 1])
        for i in 0 ..< 8 {
            var v = [UInt8](repeating: 0, count: 8)
            v[i] = 4
            rows.append(v)
        }
        for i in 0 ..< 8 {
            var v = [UInt8](repeating: 1, count: 8)
            v[i] = 3
            rows.append(v)
        }
        appendCombinations4Of2(into: &rows)
        for i in 0 ..< 8 {
            for j in (i + 1) ..< 8 {
                var v = [UInt8](repeating: 1, count: 8)
                v[i] = 3
                v[j] = 3
                rows.append(v)
            }
        }
        for i in 0 ..< 8 {
            for j in (i + 1) ..< 8 {
                var v = [UInt8](repeating: 2, count: 8)
                v[i] = 0
                v[j] = 0
                rows.append(v)
            }
        }
        appendPermutations422(into: &rows, limit: baseEntries)
        precondition(rows.count == baseEntries)
        return rows.flatMap { $0 }
    }

    private static func appendCombinations4Of2(into rows: inout [[UInt8]]) {
        for a in 0 ..< 5 {
            for b in (a + 1) ..< 6 {
                for c in (b + 1) ..< 7 {
                    for d in (c + 1) ..< 8 {
                        var v = [UInt8](repeating: 0, count: 8)
                        v[a] = 2
                        v[b] = 2
                        v[c] = 2
                        v[d] = 2
                        rows.append(v)
                    }
                }
            }
        }
    }

    private static func appendPermutations422(into rows: inout [[UInt8]], limit: Int) {
        for i in 0 ..< 8 {
            for j in 0 ..< 8 where j != i {
                for k in (j + 1) ..< 8 where k != i {
                    if rows.count >= limit { return }
                    var v = [UInt8](repeating: 0, count: 8)
                    v[i] = 4
                    v[j] = 2
                    v[k] = 2
                    rows.append(v)
                }
            }
        }
    }

    package static func cpuDecodeCode(_ code: UInt16) -> [Float] {
        let base = Int(code & 0xff)
        let signs = Int((code >> 8) & 0xff)
        let table = baseShellHalfUnits
        return (0 ..< 8).map { j in
            let mag = 0.5 * Float(table[base * 8 + j])
            let sgn: Float = ((signs >> j) & 1) != 0 ? -1.0 : 1.0
            return mag * sgn
        }
    }

    package static func isValidE8Point(_ point: [Float]) -> Bool {
        guard point.count == 8, point.allSatisfy({ $0.isFinite && abs($0) < 1e15 }) else { return false }
        let doubled = point.map { Int(rintf($0 * 2.0)) }
        for j in 0 ..< 8 where abs(Float(doubled[j]) * 0.5 - point[j]) > 1e-5 {
            return false
        }
        let mod = doubled[0] & 1
        for j in 1 ..< 8 where (doubled[j] & 1) != mod {
            return false
        }
        let sumDoubled = doubled.reduce(0, +)
        return sumDoubled.isMultiple(of: 4)
    }

    package static func nearestCode(_ target: [Float]) -> UInt16 {
        let table = baseShellHalfUnits
        var bestCode: UInt16 = 0
        var bestDist: Float = .greatestFiniteMagnitude
        for b in 0 ..< baseEntries {
            let off = b * 8
            let isHalfCoset = (table[off] & 1) == 1
            var signs = 0
            var halfSum = 0
            var dist: Float = 0
            var flipCost = [Float](repeating: .greatestFiniteMagnitude, count: 8)
            var minFlipIdx = 0
            for j in 0 ..< 8 {
                let u = Int(table[off + j])
                let mag = 0.5 * Float(u)
                let t = target[j]
                let neg = t < 0 && u > 0
                if neg { signs |= (1 << j) }
                let signedHalf = neg ? -u : u
                halfSum += signedHalf
                let chosen = neg ? -mag : mag
                let diff = t - chosen
                dist += diff * diff
                if isHalfCoset {
                    let altDiff = t + chosen
                    let extra = altDiff * altDiff - diff * diff
                    flipCost[j] = extra
                    if extra < flipCost[minFlipIdx] { minFlipIdx = j }
                }
            }
            if isHalfCoset && ((halfSum & 3) != 0) {
                signs ^= (1 << minFlipIdx)
                dist += flipCost[minFlipIdx]
            }
            if dist < bestDist {
                bestDist = dist
                bestCode = UInt16(b | (signs << 8))
            }
        }
        return bestCode
    }
}

package enum E8PMetalKernels {
    package static let supportedDTypes: Set<DType> = [.bfloat16, .float16, .float32]
    static let rowsPerThreadgroup = 4

    package struct QuantizedPack {
        package var packedCodes: MLXArray
        package var rowScale: MLXArray
        package var rhtSigns: MLXArray
    }

    package static func quantizeWeight(_ weight: MLXArray, seed: UInt64 = 549) -> QuantizedPack {
        let wf = weight.dtype == .float64 && Device.defaultDevice() == .gpu
            ? weight.asType(.float32, stream: .cpu)
            : weight.asType(.float32)
        precondition(wf.ndim == 2, "E8PMetalKernels.quantizeWeight requires 2D weight, got ndim=\(wf.ndim)")
        let nOut = wf.dim(0)
        let k = wf.dim(1)
        precondition(k > 0 && k.isMultiple(of: 32), "E8PMetalKernels.quantizeWeight requires k > 0 and multiple of 32")
        var rng = seed == 0 ? 0xE850_2026_0549 : seed
        let signsHost: [Float] = (0 ..< k).map { _ in
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return ((rng >> 63) & 1) == 0 ? 1.0 : -1.0
        }
        let signsArr = MLXArray(signsHost)
        guard nOut > 0 else {
            return QuantizedPack(
                packedCodes: MLXArray.zeros([0, k / 16], dtype: .uint32),
                rowScale: MLXArray.zeros([0], dtype: .float32),
                rhtSigns: signsArr)
        }
        let wRot: MLXArray
        if Device.defaultDevice() == .gpu {
            wRot = MACaQFusedKernels.blockHadamard(wf, scaleVector: signsArr, blockSize: 32, isInverse: false)
        } else {
            let scale = 1.0 / Float(32).squareRoot()
            wRot = hadamardTransform((wf * signsArr).reshaped(-1, 32), scale: scale).reshaped(wf.shape)
        }
        let rowRms = maximum(sqrt(mean(square(wRot), axis: 1)), Float(1e-6)) * Float(0.9)
        let wNorm = wRot / rowRms.reshaped(nOut, 1)
        eval(wNorm, rowRms)
        let normHost = wNorm.asArray(Float.self)
        let wordsPerRow = k / 16
        var packedWords = [UInt32](repeating: 0, count: nOut * wordsPerRow)
        for r in 0 ..< nOut {
            for pair in 0 ..< wordsPerRow {
                let b0 = r * k + pair * 16
                let c0 = UInt32(E8PLattice.nearestCode(Array(normHost[b0 ..< (b0 + 8)])))
                let c1 = UInt32(E8PLattice.nearestCode(Array(normHost[(b0 + 8) ..< (b0 + 16)])))
                packedWords[r * wordsPerRow + pair] = c0 | (c1 << 16)
            }
        }
        return QuantizedPack(
            packedCodes: MLXArray(packedWords, [nOut, wordsPerRow]),
            rowScale: rowRms,
            rhtSigns: signsArr)
    }

    package struct CPUPack: Sendable {
        package var packedWords: [UInt32]
        package var nOut: Int
        package var rhtSigns: [Float]
        package var rowScale: [Float]
        package var bias: [Float]?
    }

    package static func cpuMatmul(_ x: [Float], rows: Int, pack: CPUPack) -> [Float] {
        let k = pack.rhtSigns.count
        let wordsPerRow = k / 16
        var wDeq = [Float](repeating: 0, count: pack.nOut * k)
        for n in 0 ..< pack.nOut {
            for pair in 0 ..< wordsPerRow {
                let word = pack.packedWords[n * wordsPerRow + pair]
                let d0 = E8PLattice.cpuDecodeCode(UInt16(truncatingIfNeeded: word))
                let d1 = E8PLattice.cpuDecodeCode(UInt16(truncatingIfNeeded: word >> 16))
                let cBase = n * k + pair * 16
                for j in 0 ..< 8 {
                    wDeq[cBase + j] = d0[j]
                    wDeq[cBase + 8 + j] = d1[j]
                }
            }
        }
        var out = [Float](repeating: 0, count: rows * pack.nOut)
        for m in 0 ..< rows {
            let xRot = BasaltMetalSubstrate.cpuHadamardBlock(
                Array(x[(m * k) ..< ((m + 1) * k)]), scale: pack.rhtSigns, blockSize: 32)
            for n in 0 ..< pack.nOut {
                var acc: Float = 0
                for c in 0 ..< k { acc += xRot[c] * wDeq[n * k + c] }
                var val = acc * pack.rowScale[n]
                if let bias = pack.bias { val += bias[n] }
                out[m * pack.nOut + n] = val
            }
        }
        return out
    }

    private static func cleanDType(_ array: MLXArray) -> MLXArray {
        if supportedDTypes.contains(array.dtype) { return array }
        return array.asType(.float32, stream: array.dtype == .float64 ? .cpu : .default)
    }

    private static func canUseMetal(dtype: DType, pack: QuantizedPack, rows: Int, nOut: Int) -> Bool {
        let k = pack.rhtSigns.size
        let maxDim = Int(Int32.max) - rowsPerThreadgroup
        return Device.defaultDevice() == .gpu
            && supportedDTypes.contains(dtype)
            && pack.packedCodes.dtype == .uint32
            && rows <= maxDim
            && nOut <= maxDim
            && k <= maxDim
    }

    private static func cpuMatmulFallback(
        _ x: MLXArray,
        pack: QuantizedPack,
        bias: MLXArray?,
        rows: Int,
        outShape: [Int]
    ) -> MLXArray {
        let nOut = pack.packedCodes.dim(0)
        let xf = x.asType(.float32)
        let codes = pack.packedCodes.asType(.uint32)
        let signs = pack.rhtSigns.asType(.float32)
        let scales = pack.rowScale.asType(.float32)
        let biasF = bias?.asType(.float32)
        if let biasF { eval(xf, codes, signs, scales, biasF) } else { eval(xf, codes, signs, scales) }
        let cpuPack = CPUPack(
            packedWords: codes.asArray(UInt32.self),
            nOut: nOut,
            rhtSigns: signs.asArray(Float.self),
            rowScale: scales.asArray(Float.self),
            bias: biasF?.asArray(Float.self))
        let outHost = cpuMatmul(xf.asArray(Float.self), rows: rows, pack: cpuPack)
        return MLXArray(outHost, outShape).asType(x.dtype)
    }

    package static func matmul(
        _ x: MLXArray,
        pack: QuantizedPack,
        bias: MLXArray? = nil
    ) -> MLXArray {
        precondition(x.ndim >= 1 && pack.packedCodes.ndim == 2, "E8PMetalKernels.matmul requires ndim >= 1 input and 2D packedCodes")
        if Device.defaultDevice() == .gpu,
           [x, pack.packedCodes, pack.rhtSigns, pack.rowScale, bias].contains(where: { $0?.dtype == .float64 }) {
            return Device.withDefaultDevice(.cpu) { matmul(x, pack: pack, bias: bias) }
        }
        let origType = x.dtype
        let k = pack.rhtSigns.size
        let nOut = pack.packedCodes.dim(0)
        let wordsPerRow = pack.packedCodes.dim(1)
        let outShape = Array(x.shape.dropLast()) + [nOut]
        precondition(x.dim(-1) == k, "E8PMetalKernels.matmul input width \(x.dim(-1)) != rhtSigns size \(k)")
        let rows = k > 0 ? (x.size / k) : x.shape.dropLast().reduce(1, *)
        guard rows > 0, nOut > 0 else {
            return MLXArray.zeros(outShape, dtype: origType)
        }
        if k == 0 {
            let zeros = MLXArray.zeros(outShape, dtype: origType)
            guard let bias else { return zeros }
            precondition(bias.size == nOut, "E8PMetalKernels.matmul bias size mismatch")
            return (zeros.asType(.float32) + bias.reshaped(nOut).asType(.float32)).asType(origType)
        }
        precondition(k.isMultiple(of: 32) && wordsPerRow == k / 16, "E8PMetalKernels.matmul requires k%32==0 and wordsPerRow==k/16")
        precondition(pack.rowScale.size == nOut, "E8PMetalKernels.matmul rowScale size mismatch")
        if let bias { precondition(bias.size == nOut, "E8PMetalKernels.matmul bias size mismatch") }
        guard canUseMetal(dtype: origType, pack: pack, rows: rows, nOut: nOut) else {
            return cpuMatmulFallback(x, pack: pack, bias: bias, rows: rows, outShape: outShape)
        }
        let mTile = MACaQFusedMatmul.selectMTile(rows: rows)
        let mBlocks = (rows + mTile - 1) / mTile
        let nGroups = (nOut + rowsPerThreadgroup - 1) / rowsPerThreadgroup
        let dummy = MLXArray.zeros([2], dtype: origType)
        let signsClean = cleanDType(pack.rhtSigns)
        let rsClean = cleanDType(pack.rowScale)
        let biasClean = bias.map(cleanDType) ?? dummy
        let shapes = MLXArray([Int32(rows), Int32(nOut), Int32(k), Int32(wordsPerRow)])
        let out = e8pFusedKernel(
            [x.reshaped(rows, k), pack.packedCodes, signsClean.reshaped(k), rsClean.reshaped(nOut), biasClean.reshaped(biasClean.size), shapes],
            template: [("T", origType), ("HAS_BIAS", bias != nil), ("MTILE", mTile)],
            grid: (32, nGroups * rowsPerThreadgroup, mBlocks),
            threadGroup: (32, rowsPerThreadgroup, 1),
            outputShapes: [[rows, nOut]],
            outputDTypes: [origType]
        )[0]
        return out.reshaped(outShape)
    }
}

package final class E8PLinear: Linear, Quantized {
    public let groupSize: Int = 32
    public let bits: Int = 2
    public let mode: QuantizationMode = .affine
    public override var shape: (Int, Int) { (weight.dim(0), rhtSigns.size) }
    @ParameterInfo(key: "row_scale") package var rowScale: MLXArray
    @ParameterInfo(key: "rht_signs") package var rhtSigns: MLXArray

    package init(pack: E8PMetalKernels.QuantizedPack, bias: MLXArray? = nil) {
        self._rowScale.wrappedValue = pack.rowScale
        self._rhtSigns.wrappedValue = pack.rhtSigns
        super.init(weight: pack.packedCodes, bias: bias)
        freeze()
        train(false)
    }

    override public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let pack = E8PMetalKernels.QuantizedPack(packedCodes: weight, rowScale: rowScale, rhtSigns: rhtSigns)
        return E8PMetalKernels.matmul(x, pack: pack, bias: bias)
    }
}

private let e8pShellHeader: String = {
    let values = E8PLattice.baseShellHalfUnits.map(String.init).joined(separator: ",")
    return "constant uint8_t e8p_shell_const[2048] = {\(values)};\n"
}()

private let e8pSource = """
    const int M = int(shapes[0]);
    const int N = int(shapes[1]);
    const int K = int(shapes[2]);
    const int WPR = int(shapes[3]);
    const int lane = int(thread_index_in_simdgroup);
    const int sg = int(simdgroup_index_in_threadgroup);
    const int tid = sg * 32 + lane;
    threadgroup uint8_t tg_shell[2048];
    for (int idx = tid; idx < 2048; idx += 128) {
        tg_shell[idx] = e8p_shell_const[idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int n = int(threadgroup_position_in_grid.y) * 4 + sg;
    const int m0 = int(threadgroup_position_in_grid.z) * MTILE;
    if (n >= N || m0 >= M) { return; }
    const size_t row_base = size_t(n) * size_t(WPR);
    float acc[MTILE];
    for (int r = 0; r < MTILE; ++r) { acc[r] = 0.0f; }
    for (int base = 0; base < K; base += 32) {
        const int col = base + lane;
        const int vec_idx = col >> 3;
        const int elem_in_vec = col & 7;
        const uint32_t word = packed_codes[row_base + size_t(vec_idx >> 1)];
        const uint code16 = ((vec_idx & 1) == 0) ? (word & 0xffffu) : (word >> 16u);
        const uint base_id = code16 & 255u;
        const uint sign_mask = (code16 >> 8u) & 255u;
        const float mag = 0.5f * float(tg_shell[base_id * 8u + uint(elem_in_vec)]);
        const float wv = (((sign_mask >> uint(elem_in_vec)) & 1u) != 0u) ? -mag : mag;
        const float sgn = float(rht_signs[col]);
        for (int r = 0; r < MTILE; ++r) {
            const int mr = m0 + r;
            const float xv = (mr < M) ? (float(x[size_t(mr) * size_t(K) + size_t(col)]) * sgn) : 0.0f;
            const float xr = basalt_wht32_simd(xv, lane);
            acc[r] = fma(xr, wv, acc[r]);
        }
    }
    for (int r = 0; r < MTILE; ++r) {
        const float total = simd_sum(acc[r]);
        const int mr = m0 + r;
        if (lane == 0 && mr < M) {
            out[size_t(mr) * size_t(N) + size_t(n)] = basalt_dual_scale_epilogue<T, true, HAS_BIAS>(
                total, row_scale, bias, n);
        }
    }
    """

private let e8pFusedKernel = MLXFast.metalKernel(
    name: "iq_e8p_fused_qmm",
    inputNames: ["x", "packed_codes", "rht_signs", "row_scale", "bias", "shapes"],
    outputNames: ["out"],
    source: e8pSource,
    header: BasaltMetalSubstrate.combinedHeader + e8pShellHeader
)
