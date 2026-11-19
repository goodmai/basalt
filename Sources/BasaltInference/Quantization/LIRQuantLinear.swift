import BasaltCore
import Foundation
import MLX
import MLXNN

package struct LIRGivensSchedule: Sendable, Equatable {
    package var numLayers: Int
    package var pairsPerLayer: Int
    package var flatPairs: [Int32]
    package var flatAngles: [Float]

    package init(numLayers: Int, pairsPerLayer: Int, flatPairs: [Int32], flatAngles: [Float]) {
        self.numLayers = numLayers
        self.pairsPerLayer = pairsPerLayer
        self.flatPairs = flatPairs
        self.flatAngles = flatAngles
    }
}

package struct LIRBasisParameters {
    package var perm: MLXArray
    package var givensPairs: MLXArray
    package var givensAngles: MLXArray
    package var diagScale: MLXArray
    package var numLayers: Int
    package var pairsPerLayer: Int

    package init(
        perm: MLXArray,
        givensPairs: MLXArray,
        givensAngles: MLXArray,
        diagScale: MLXArray,
        numLayers: Int,
        pairsPerLayer: Int
    ) {
        self.perm = perm
        self.givensPairs = givensPairs
        self.givensAngles = givensAngles
        self.diagScale = diagScale
        self.numLayers = numLayers
        self.pairsPerLayer = pairsPerLayer
    }
}

package enum LIRQuantKernels {
    package static let maxThreadgroupK = 4096
    package static let supportedDTypes: Set<DType> = [.bfloat16, .float16, .float32]

    package static func cpuTransform(
        _ x: [Float],
        rows: Int,
        perm: [Int32],
        schedule: LIRGivensSchedule,
        diagScale: [Float]
    ) -> [Float] {
        let k = perm.count
        precondition(k.isMultiple(of: 32) && diagScale.count == k)
        var out = [Float](repeating: 0, count: rows * k)
        let ones = [Float](repeating: 1.0, count: 32)
        for m in 0 ..< rows {
            let base = m * k
            var buf = (0 ..< k).map { x[base + Int(perm[$0])] }
            let pCount = schedule.pairsPerLayer
            for l in 0 ..< schedule.numLayers {
                for p in 0 ..< pCount {
                    let off = (l * pCount + p) * 2
                    let i = Int(schedule.flatPairs[off])
                    let j = Int(schedule.flatPairs[off + 1])
                    if i >= 0, j >= 0, i != j {
                        let c = schedule.flatAngles[off]
                        let s = schedule.flatAngles[off + 1]
                        let xi = buf[i]
                        let xj = buf[j]
                        buf[i] = c * xi + s * xj
                        buf[j] = -s * xi + c * xj
                    }
                }
            }
            for col in 0 ..< k { buf[col] *= diagScale[col] }
            let wht = BasaltMetalSubstrate.cpuHadamardBlock(buf, scale: ones, blockSize: 32)
            for col in 0 ..< k { out[base + col] = wht[col] }
        }
        return out
    }

    package static func transform(_ x: MLXArray, basis: LIRBasisParameters) -> MLXArray {
        let origType = x.dtype
        let k = basis.perm.size
        guard x.ndim >= 1, x.dim(-1) == k, k.isMultiple(of: 32) else { return x }
        let rows = x.size / max(1, k)
        guard rows > 0 else { return x }
        if k <= maxThreadgroupK, supportedDTypes.contains(origType) {
            let scaleClean = supportedDTypes.contains(basis.diagScale.dtype)
                ? basis.diagScale : basis.diagScale.asType(.float32)
            let pSafe = basis.givensPairs.size > 0 ? basis.givensPairs.asType(.int32) : MLXArray([Int32(0), Int32(0)])
            let aSafe = basis.givensAngles.size > 0 ? basis.givensAngles.asType(.float32) : MLXArray([Float(1), Float(0)])
            let shapes = MLXArray([Int32(rows), Int32(k), Int32(basis.numLayers), Int32(basis.pairsPerLayer)])
            let out = lirGivensKernel(
                [x.reshaped(rows, k), basis.perm.asType(.int32).reshaped(k), pSafe, aSafe, scaleClean.reshaped(k), shapes],
                template: [("T", origType), ("TD", scaleClean.dtype)],
                grid: (128, rows, 1),
                threadGroup: (128, 1, 1),
                outputShapes: [[rows, k]],
                outputDTypes: [origType]
            )[0]
            return out.reshaped(x.shape)
        }
        return differentiableTransform(x, basis: basis)
    }

    package static func differentiableTransform(_ x: MLXArray, basis: LIRBasisParameters) -> MLXArray {
        let origType = x.dtype
        let k = basis.perm.size
        guard x.ndim >= 1, x.dim(-1) == k, k.isMultiple(of: 32) else { return x }
        let rows = x.size / max(1, k)
        guard rows > 0 else { return x }
        var cur = x.asType(.float32).reshaped(rows, k).take(basis.perm.asType(.int32), axis: -1)
        if let givens = givensMatrix(basis) {
            let rotatedSub = matmul(cur[0..., 0 ..< givens.width], givens.matrix)
            cur = givens.width < k ? concatenated([rotatedSub, cur[0..., givens.width...]], axis: -1) : rotatedSub
        }
        cur = cur * basis.diagScale.asType(.float32).reshaped(1, k)
        return hadamardTransform(cur.reshaped(-1, 32), scale: 1.0 / Float(32.0).squareRoot())
            .reshaped(x.shape)
            .asType(origType)
    }

    private static func givensMatrix(_ basis: LIRBasisParameters) -> (matrix: MLXArray, width: Int)? {
        guard basis.numLayers > 0, basis.pairsPerLayer > 0 else { return nil }
        let pFlat = basis.givensPairs.asType(.int32).asArray(Int32.self)
        let aFlat = basis.givensAngles.asType(.float32).asArray(Float.self)
        let maxIdx = pFlat.max().map(Int.init) ?? -1
        guard maxIdx >= 1 else { return nil }
        let width = min(basis.perm.size, maxIdx + 1)
        var mat = [Float](repeating: 0, count: width * width)
        for r in 0 ..< width { mat[r * width + r] = 1.0 }
        for l in 0 ..< basis.numLayers {
            for p in 0 ..< basis.pairsPerLayer {
                let off = (l * basis.pairsPerLayer + p) * 2
                guard off + 1 < pFlat.count, off + 1 < aFlat.count else { continue }
                let (i, j) = (Int(pFlat[off]), Int(pFlat[off + 1]))
                guard i >= 0, j >= 0, i < width, j < width, i != j else { continue }
                let (c, s) = (aFlat[off], aFlat[off + 1])
                for r in 0 ..< width {
                    let (mi, mj) = (mat[r * width + i], mat[r * width + j])
                    mat[r * width + i] = c * mi + s * mj
                    mat[r * width + j] = -s * mi + c * mj
                }
            }
        }
        return (MLXArray(mat, [width, width]), width)
    }
}

package struct LIRQuantLayerSpec: Codable, Sendable, Equatable {
    package var bits: Int
    package var groupSize: Int
    package var mode: String
    package var subspaceDim: Int
    package var numLayers: Int
    package var pairsPerLayer: Int
    package var scaleDType: String?
    package var hasBias: Bool?

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
        case mode
        case subspaceDim = "subspace_dim"
        case numLayers = "num_layers"
        case pairsPerLayer = "pairs_per_layer"
        case scaleDType = "scale_dtype"
        case hasBias = "has_bias"
    }

    package init(
        bits: Int,
        groupSize: Int,
        mode: QuantizationMode,
        subspaceDim: Int,
        numLayers: Int,
        pairsPerLayer: Int
    ) {
        self.bits = bits
        self.groupSize = groupSize
        self.mode = mode.rawValue
        self.subspaceDim = subspaceDim
        self.numLayers = numLayers
        self.pairsPerLayer = pairsPerLayer
    }

    package var quantizationMode: QuantizationMode {
        QuantizationMode(rawValue: mode) ?? .affine
    }

    package var resolvedScaleDType: DType {
        if quantizationMode == .mxfp4 { return .uint8 }
        if let scaleDType, let match = DType.allCases.first(where: { "\($0)" == scaleDType }) {
            return match
        }
        return .bfloat16
    }
}

package final class LIRQuantLinear: QuantizedLinear {
    package let numLayers: Int
    package let pairsPerLayer: Int
    @ParameterInfo(key: "perm") package var perm: MLXArray
    @ParameterInfo(key: "givens_pairs") package var givensPairs: MLXArray
    @ParameterInfo(key: "givens_angles") package var givensAngles: MLXArray
    @ParameterInfo(key: "diag_scale") package var diagScale: MLXArray

    package init(
        packed: MACaQPackedWeights,
        bias: MLXArray?,
        basis: LIRBasisParameters,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    ) {
        self.numLayers = basis.numLayers
        self.pairsPerLayer = basis.pairsPerLayer
        self._perm.wrappedValue = basis.perm
        self._givensPairs.wrappedValue = basis.givensPairs
        self._givensAngles.wrappedValue = basis.givensAngles
        self._diagScale.wrappedValue = basis.diagScale
        if mode == .affine {
            super.init(
                weight: packed.weight, bias: bias, scales: packed.scales,
                biases: packed.biases, groupSize: groupSize, bits: bits)
        } else if bias == nil {
            super.init(
                weight: packed.weight, scales: packed.scales, biases: nil,
                groupSize: groupSize, bits: bits, mode: mode)
        } else {
            let stub = MLXArray.zeros([1, groupSize], dtype: .float16)
            super.init(weight: stub, bias: bias, groupSize: groupSize, bits: bits, mode: mode)
            update(parameters: ModuleParameters.unflattened([("weight", packed.weight), ("scales", packed.scales)]))
        }
        freeze()
    }

    package static func placeholder(replacing linear: Linear, spec: LIRQuantLayerSpec) -> LIRQuantLinear {
        let (outputs, inputs) = linear.shape
        let groups = inputs / spec.groupSize
        let qMode = spec.quantizationMode
        let sType = spec.resolvedScaleDType
        let factorType: DType = sType.isFloatingPoint ? sType : .bfloat16
        let weight = MLXArray.zeros([outputs, (inputs * spec.bits) / 32], dtype: .uint32)
        let scales = MLXArray.zeros([outputs, groups], dtype: sType)
        let biases = qMode == .affine ? MLXArray.zeros([outputs, groups], dtype: sType) : nil
        let hasBias = spec.hasBias ?? (linear.bias != nil)
        let bias = hasBias ? (linear.bias ?? MLXArray.zeros([outputs], dtype: factorType)) : nil
        let pairLen = max(2, spec.numLayers * spec.pairsPerLayer * 2)
        let basis = LIRBasisParameters(
            perm: MLXArray(Array(0 ..< Int32(inputs))),
            givensPairs: MLXArray.zeros([pairLen], dtype: .int32),
            givensAngles: MLXArray.zeros([pairLen], dtype: .float32),
            diagScale: MLXArray.ones([inputs], dtype: factorType),
            numLayers: spec.numLayers,
            pairsPerLayer: spec.pairsPerLayer)
        let packed = MACaQPackedWeights(weight: weight, scales: scales, biases: biases)
        return LIRQuantLinear(
            packed: packed, bias: bias, basis: basis,
            groupSize: spec.groupSize, bits: spec.bits, mode: qMode)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let basis = LIRBasisParameters(
            perm: perm, givensPairs: givensPairs, givensAngles: givensAngles,
            diagScale: diagScale, numLayers: numLayers, pairsPerLayer: pairsPerLayer)
        let transformed = training
            ? LIRQuantKernels.differentiableTransform(x, basis: basis)
            : LIRQuantKernels.transform(x, basis: basis)
        return super.callAsFunction(transformed)
    }

    package func unrotatedDenseWeight() -> MLXArray {
        let wRot = dequantized(
            weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits, mode: mode
        ).asType(.float32)
        let rows = wRot.dim(0)
        let k = wRot.dim(1)
        let unWht = MACaQMath.rotateBlocks(wRot, blockSize: 32) * diagScale.asType(.float32).reshaped(1, k)
        eval(unWht, perm, givensPairs, givensAngles)
        var arr = unWht.asArray(Float.self)
        let pFlat = givensPairs.asType(.int32).asArray(Int32.self)
        let aFlat = givensAngles.asType(.float32).asArray(Float.self)
        for m in 0 ..< rows {
            let base = m * k
            for l in stride(from: numLayers - 1, through: 0, by: -1) where pairsPerLayer > 0 {
                for p in 0 ..< pairsPerLayer {
                    let off = (l * pairsPerLayer + p) * 2
                    let i = Int(pFlat[off])
                    let j = Int(pFlat[off + 1])
                    if i >= 0, j >= 0, i != j {
                        let c = aFlat[off]
                        let s = aFlat[off + 1]
                        let yi = arr[base + i]
                        let yj = arr[base + j]
                        arr[base + i] = c * yi - s * yj
                        arr[base + j] = s * yi + c * yj
                    }
                }
            }
        }
        let permHost = perm.asType(.int32).asArray(Int32.self)
        var restored = [Float](repeating: 0, count: rows * k)
        for m in 0 ..< rows {
            let base = m * k
            for q in 0 ..< k {
                restored[base + Int(permHost[q])] = arr[base + q]
            }
        }
        return MLXArray(restored, [rows, k])
    }
}

package enum LIRQuantCalibrator {
    package static func buildSchedule(
        subspaceCov: [Float],
        subspaceWidth kSub: Int,
        layers: Int = 2
    ) -> LIRGivensSchedule {
        guard kSub >= 2, layers > 0 else {
            return LIRGivensSchedule(numLayers: 0, pairsPerLayer: 0, flatPairs: [], flatAngles: [])
        }
        let pairsPerLayer = kSub / 2
        var cov = subspaceCov
        var flatPairs: [Int32] = []
        var flatAngles: [Float] = []
        for l in 0 ..< layers {
            let strideStep = 1 << (l % max(1, Int(log2(Double(kSub)))))
            var used = [Bool](repeating: false, count: kSub)
            var count = 0
            for i in 0 ..< kSub where !used[i] {
                let j = i ^ strideStep
                guard j > i, j < kSub, !used[j] else { continue }
                used[i] = true
                used[j] = true
                let hii = cov[i * kSub + i]
                let hjj = cov[j * kSub + j]
                let hij = cov[i * kSub + j]
                let theta = 0.5 * atan2f(2.0 * hij, hii - hjj)
                let c = cosf(theta)
                let s = sinf(theta)
                flatPairs.append(Int32(i))
                flatPairs.append(Int32(j))
                flatAngles.append(c)
                flatAngles.append(s)
                applyGivensToSymmetric(&cov, dim: kSub, i: i, j: j, c: c, s: s)
                count += 1
            }
            while count < pairsPerLayer {
                flatPairs.append(-1)
                flatPairs.append(-1)
                flatAngles.append(1.0)
                flatAngles.append(0.0)
                count += 1
            }
        }
        return LIRGivensSchedule(
            numLayers: layers, pairsPerLayer: pairsPerLayer,
            flatPairs: flatPairs, flatAngles: flatAngles)
    }

    private static func applyGivensToSymmetric(_ cov: inout [Float], dim: Int, i: Int, j: Int, c: Float, s: Float) {
        for r in 0 ..< dim where r != i && r != j {
            let cri = cov[r * dim + i]
            let crj = cov[r * dim + j]
            let ni = c * cri - s * crj
            let nj = s * cri + c * crj
            cov[r * dim + i] = ni
            cov[i * dim + r] = ni
            cov[r * dim + j] = nj
            cov[j * dim + r] = nj
        }
        let cii = cov[i * dim + i]
        let cjj = cov[j * dim + j]
        let cij = cov[i * dim + j]
        cov[i * dim + i] = c * c * cii - 2.0 * c * s * cij + s * s * cjj
        cov[j * dim + j] = s * s * cii + 2.0 * c * s * cij + c * c * cjj
        cov[i * dim + j] = (c * c - s * s) * cij + c * s * (cii - cjj)
        cov[j * dim + i] = cov[i * dim + j]
    }

    package struct QuantizeOptions: Sendable {
        package var subspaceBlocks: Int
        package var givensLayers: Int
        package var bits: Int
        package var groupSize: Int
        package var mode: QuantizationMode

        package init(
            subspaceBlocks: Int = 1,
            givensLayers: Int = 2,
            bits: Int = 4,
            groupSize: Int = 32,
            mode: QuantizationMode = .affine
        ) {
            self.subspaceBlocks = subspaceBlocks
            self.givensLayers = givensLayers
            self.bits = bits
            self.groupSize = groupSize
            self.mode = mode
        }
    }

    package static func quantizeLayer(
        linear: Linear,
        activations: MLXArray? = nil,
        options: QuantizeOptions = .init()
    ) -> LIRQuantLinear {
        let wf = linear.weight.asType(.float32)
        let k = wf.dim(1)
        let actFlat = activations?.asType(.float32).reshaped(-1, k)
        let sens: MLXArray = if let actFlat {
            mean(square(actFlat), axis: 0) * mean(square(wf), axis: 0)
        } else {
            mean(square(wf), axis: 0)
        }
        let permArr = argSort(-sens, axis: 0).asType(.int32)
        let kSub = min(k, max(32, options.subspaceBlocks * 32))
        let xPerm: MLXArray = if let actFlat {
            actFlat.take(permArr, axis: -1)[0..., 0 ..< kSub]
        } else {
            wf.take(permArr, axis: -1)[0..., 0 ..< kSub]
        }
        let covArr = matmul(xPerm.T, xPerm) / Float(max(1, xPerm.dim(0)))
        let diagScale = MACaQMath.balancePreScale(
            weight: wf.take(permArr, axis: -1),
            meanSquareActivation: actFlat.map { mean(square($0.take(permArr, axis: -1)), axis: 0) } ?? MLXArray.ones([k], dtype: .float32))
        eval(permArr, covArr, diagScale)
        let schedule = buildSchedule(
            subspaceCov: covArr.asArray(Float.self), subspaceWidth: kSub, layers: options.givensLayers)
        let pairsTensor = schedule.flatPairs.isEmpty
            ? MLXArray([Int32(0), Int32(0)]) : MLXArray(schedule.flatPairs)
        let anglesTensor = schedule.flatAngles.isEmpty
            ? MLXArray([Float(1), Float(0)]) : MLXArray(schedule.flatAngles)
        let invBasis = LIRBasisParameters(
            perm: permArr, givensPairs: pairsTensor, givensAngles: anglesTensor,
            diagScale: (Float(1.0) / diagScale).asType(.float32),
            numLayers: schedule.numLayers, pairsPerLayer: schedule.pairsPerLayer)
        let wConj = LIRQuantKernels.transform(wf, basis: invBasis).asType(linear.weight.dtype)
        let (wq, scales, biases) = MLX.quantized(
            wConj, groupSize: options.groupSize, bits: options.bits, mode: options.mode)
        let packed = MACaQPackedWeights(weight: wq, scales: scales, biases: biases)
        let basis = LIRBasisParameters(
            perm: permArr, givensPairs: pairsTensor, givensAngles: anglesTensor,
            diagScale: diagScale.asType(linear.weight.dtype),
            numLayers: schedule.numLayers, pairsPerLayer: schedule.pairsPerLayer)
        return LIRQuantLinear(
            packed: packed, bias: linear.bias, basis: basis,
            groupSize: options.groupSize, bits: options.bits, mode: options.mode)
    }
}

private let lirGivensSource = """
    const int M = int(shapes[0]);
    const int K = int(shapes[1]);
    const int L = int(shapes[2]);
    const int P = int(shapes[3]);
    const int row = int(threadgroup_position_in_grid.y);
    const int tid = int(thread_position_in_threadgroup.x);
    const int sg = int(simdgroup_index_in_threadgroup);
    const int lane = int(thread_index_in_simdgroup);
    if (row >= M) { return; }
    threadgroup float tg_row[4096];
    const device T* x_row = x + size_t(row) * size_t(K);
    device T* out_row = out + size_t(row) * size_t(K);
    for (int col = tid; col < K; col += 128) {
        tg_row[col] = float(x_row[perm[col]]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int l = 0; l < L; ++l) {
        for (int p = tid; p < P; p += 128) {
            const int off = (l * P + p) * 2;
            const int i = int(pairs[off]);
            const int j = int(pairs[off + 1]);
            if (i >= 0 && j >= 0 && i != j) {
                float yi = 0.0f;
                float yj = 0.0f;
                basalt_givens_pair(tg_row[i], tg_row[j], angles[off], angles[off + 1], yi, yj);
                tg_row[i] = yi;
                tg_row[j] = yj;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int num_blocks = K >> 5;
    for (int blk = sg; blk < num_blocks; blk += 4) {
        const int col = (blk << 5) + lane;
        const float v = tg_row[col] * float(diag_scale[col]);
        out_row[col] = static_cast<T>(basalt_wht32_simd(v, lane));
    }
    """

private let lirGivensKernel = MLXFast.metalKernel(
    name: "lirquant_givens_prologue",
    inputNames: ["x", "perm", "pairs", "angles", "diag_scale", "shapes"],
    outputNames: ["out"],
    source: lirGivensSource,
    header: BasaltMetalSubstrate.combinedHeader
)
