import BasaltCore
import Foundation
import MLX
import MLXNN

package enum FlatQuantKernels {
    package static let maxThreadgroupK = 4096
    package static let supportedDTypes: Set<DType> = [.bfloat16, .float16, .float32]

    package static func factorizeDimension(_ k: Int) -> (d1: Int, d2: Int) {
        let root = Int(Double(k).squareRoot().rounded(.down))
        for cand in stride(from: min(64, max(1, root)), through: 1, by: -1) where k.isMultiple(of: cand) {
            return (cand, k / cand)
        }
        return (1, k)
    }

    package struct CPUFactors: Sendable {
        package var d1: Int
        package var d2: Int
        package var p1: [Float]
        package var p2: [Float]
        package var clipBounds: (Float, Float)?

        package init(d1: Int, d2: Int, p1: [Float], p2: [Float], clipBounds: (Float, Float)? = nil) {
            self.d1 = d1
            self.d2 = d2
            self.p1 = p1
            self.p2 = p2
            self.clipBounds = clipBounds
        }
    }

    package static func cpuTransform(
        _ x: [Float],
        rows: Int,
        factors: CPUFactors
    ) -> [Float] {
        let (d1, d2) = (factors.d1, factors.d2)
        let k = d1 * d2
        var out = [Float](repeating: 0, count: rows * k)
        for m in 0 ..< rows {
            let rowBase = m * k
            var tmp = [Float](repeating: 0, count: k)
            for a in 0 ..< d1 {
                for j in 0 ..< d2 {
                    var s: Float = 0
                    for i in 0 ..< d1 {
                        s += factors.p1[i * d1 + a] * x[rowBase + i * d2 + j]
                    }
                    tmp[a * d2 + j] = s
                }
            }
            for a in 0 ..< d1 {
                for b in 0 ..< d2 {
                    var s: Float = 0
                    for j in 0 ..< d2 {
                        s += tmp[a * d2 + j] * factors.p2[j * d2 + b]
                    }
                    if let (lo, hi) = factors.clipBounds {
                        s = min(max(s, lo), hi)
                    }
                    out[rowBase + a * d2 + b] = s
                }
            }
        }
        return out
    }

    package static func transform(
        _ x: MLXArray,
        p1: MLXArray,
        p2: MLXArray,
        clipBounds: MLXArray? = nil
    ) -> MLXArray {
        let origType = x.dtype
        let d1 = p1.dim(0)
        let d2 = p2.dim(0)
        let k = d1 * d2
        guard x.ndim >= 1, x.dim(-1) == k else { return x }
        let rows = x.size / max(1, k)
        guard rows > 0 else { return x }
        if k <= maxThreadgroupK, supportedDTypes.contains(origType) {
            let p1Clean = supportedDTypes.contains(p1.dtype) ? p1 : p1.asType(.float32)
            let p2Clean = supportedDTypes.contains(p2.dtype) ? p2 : p2.asType(.float32)
            let clipClean = clipBounds?.asType(.float32) ?? MLXArray([Float(-1e9), Float(1e9)])
            let shapes = MLXArray([Int32(rows), Int32(d1), Int32(d2)])
            let out = kroneckerKernel(
                [x.reshaped(rows, k), p1Clean, p2Clean, clipClean.reshaped(2), shapes],
                template: [
                    ("T", origType),
                    ("TP1", p1Clean.dtype),
                    ("TP2", p2Clean.dtype),
                    ("HAS_CLIP", clipBounds != nil),
                ],
                grid: (128, rows, 1),
                threadGroup: (128, 1, 1),
                outputShapes: [[rows, k]],
                outputDTypes: [origType]
            )[0]
            return out.reshaped(x.shape)
        }
        let mat = x.asType(.float32).reshaped(rows, d1, d2)
        let left = matmul(transposed(mat, axes: [0, 2, 1]), p1.asType(.float32))
        var prod = matmul(transposed(left, axes: [0, 2, 1]), p2.asType(.float32)).reshaped(x.shape)
        if let clipBounds {
            let b = clipBounds.asType(.float32).reshaped(2)
            prod = clip(prod, min: b[0], max: b[1])
        }
        return prod.asType(origType)
    }
}

package struct FlatQuantBasis {
    package var p1: MLXArray
    package var p2: MLXArray
    package var clipBounds: MLXArray?
    package var rowScale: MLXArray?

    package init(p1: MLXArray, p2: MLXArray, clipBounds: MLXArray? = nil, rowScale: MLXArray? = nil) {
        self.p1 = p1
        self.p2 = p2
        self.clipBounds = clipBounds
        self.rowScale = rowScale
    }
}

package final class FlatQuantLinear: QuantizedLinear {
    @ParameterInfo(key: "p1") package var p1: MLXArray
    @ParameterInfo(key: "p2") package var p2: MLXArray
    @ParameterInfo(key: "clip_bounds") package var clipBounds: MLXArray?
    @ParameterInfo(key: "row_scale") package var rowScale: MLXArray?

    package init(
        packed: MACaQPackedWeights,
        bias: MLXArray?,
        basis: FlatQuantBasis,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    ) {
        self._p1.wrappedValue = basis.p1
        self._p2.wrappedValue = basis.p2
        self._clipBounds.wrappedValue = basis.clipBounds
        self._rowScale.wrappedValue = basis.rowScale
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

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let transformed = FlatQuantKernels.transform(x, p1: p1, p2: p2, clipBounds: clipBounds)
        guard let rowScale else {
            return super.callAsFunction(transformed)
        }
        let raw = quantizedMM(
            transformed, weight, scales: scales, biases: biases,
            transpose: true, groupSize: groupSize, bits: bits, mode: mode)
        let scaled = (raw.asType(.float32) * rowScale.asType(.float32)).asType(x.dtype)
        return bias.map { scaled + $0 } ?? scaled
    }

    package func unrotatedDenseWeight() -> MLXArray {
        var wTilde = dequantized(
            weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits, mode: mode
        ).asType(.float32)
        if let rowScale {
            wTilde = wTilde * rowScale.asType(.float32).expandedDimensions(axis: -1)
        }
        let nOut = wTilde.dim(0)
        let d1 = p1.dim(0)
        let d2 = p2.dim(0)
        let mat = wTilde.reshaped(nOut, d1, d2)
        let p1f = p1.asType(.float32)
        let p2f = p2.asType(.float32)
        let left = matmul(transposed(mat, axes: [0, 2, 1]), p1f.T)
        return matmul(transposed(left, axes: [0, 2, 1]), p2f.T).reshaped(nOut, d1 * d2)
    }
}

package enum FlatQuantCalibrator {
    package struct Factors {
        package var p1: MLXArray
        package var p2: MLXArray
        package var p1Inv: MLXArray
        package var p2Inv: MLXArray
        package var clipBounds: MLXArray?
    }

    package static func calibrateFactors(
        weight: MLXArray,
        activations: MLXArray?,
        alpha: Float = 0.5,
        clipQuantile: Float? = nil
    ) -> Factors {
        let k = weight.dim(-1)
        let (d1, d2) = FlatQuantKernels.factorizeDimension(k)
        let wf = weight.asType(.float32)
        let wRms = sqrt(maximum(mean(square(wf), axis: 0), Float(1e-10)))
        let aRms: MLXArray = if let activations {
            sqrt(maximum(mean(square(activations.asType(.float32).reshaped(-1, k)), axis: 0), Float(1e-10)))
        } else {
            MLXArray.ones([k], dtype: .float32)
        }
        let sVec = clip(pow(wRms, MLXArray(alpha)) / maximum(pow(aRms, MLXArray(Float(1.0) - alpha)), Float(1e-8)), min: Float(1e-3), max: Float(1e3))
        let sMat = sVec.reshaped(d1, d2)
        let s1 = clip(exp(mean(log(sMat), axis: 1)), min: Float(1e-2), max: Float(1e2))
        let s2 = clip(exp(mean(log(sMat / s1.reshaped(d1, 1)), axis: 0)), min: Float(1e-2), max: Float(1e2))
        let h1 = hadamardOrEye(d1)
        let h2 = hadamardOrEye(d2)
        let p1 = s1.reshaped(d1, 1) * h1
        let p2 = s2.reshaped(d2, 1) * h2
        let p1Inv = h1.T * (Float(1.0) / s1).reshaped(1, d1)
        let p2Inv = h2.T * (Float(1.0) / s2).reshaped(1, d2)
        var bounds: MLXArray?
        if let q = clipQuantile, let activations, q > 0, q < 1 {
            let xTrans = FlatQuantKernels.transform(activations.asType(.float32), p1: p1, p2: p2)
            let maxVal = abs(xTrans).max().item(Float.self) * q
            bounds = MLXArray([-maxVal, maxVal])
        }
        eval(p1, p2, p1Inv, p2Inv)
        return Factors(p1: p1, p2: p2, p1Inv: p1Inv, p2Inv: p2Inv, clipBounds: bounds)
    }

    package static func conjugateWeight(_ weight: MLXArray, factors: Factors) -> MLXArray {
        let nOut = weight.dim(0)
        let d1 = factors.p1.dim(0)
        let d2 = factors.p2.dim(0)
        let mat = weight.asType(.float32).reshaped(nOut, d1, d2)
        let left = matmul(transposed(mat, axes: [0, 2, 1]), factors.p1Inv.T)
        return matmul(transposed(left, axes: [0, 2, 1]), factors.p2Inv.T).reshaped(nOut, d1 * d2)
    }

    package static func quantizeLayer(
        linear: Linear,
        activations: MLXArray? = nil,
        bits: Int = 4,
        groupSize: Int = 32,
        mode: QuantizationMode = .affine,
        clipQuantile: Float? = nil
    ) -> FlatQuantLinear {
        let factors = calibrateFactors(weight: linear.weight, activations: activations, clipQuantile: clipQuantile)
        let wConj = conjugateWeight(linear.weight, factors: factors).asType(linear.weight.dtype)
        let (wq, scales, biases) = MLX.quantized(wConj, groupSize: groupSize, bits: bits, mode: mode)
        let packed = MACaQPackedWeights(weight: wq, scales: scales, biases: biases)
        let basis = FlatQuantBasis(
            p1: factors.p1.asType(linear.weight.dtype),
            p2: factors.p2.asType(linear.weight.dtype),
            clipBounds: factors.clipBounds)
        return FlatQuantLinear(
            packed: packed, bias: linear.bias, basis: basis,
            groupSize: groupSize, bits: bits, mode: mode)
    }

    private static func hadamardOrEye(_ n: Int) -> MLXArray {
        guard n > 1, (n & (n - 1)) == 0 else {
            return eye(n, dtype: .float32)
        }
        let scale = 1.0 / Float(n).squareRoot()
        return hadamardTransform(eye(n, dtype: .float32), scale: scale)
    }
}

private let kroneckerSource = """
    const int M = int(shapes[0]);
    const int D1 = int(shapes[1]);
    const int D2 = int(shapes[2]);
    const int K = D1 * D2;
    const int row = int(threadgroup_position_in_grid.y);
    const int tid = int(thread_position_in_threadgroup.x);
    if (row >= M) { return; }
    threadgroup float tg_tmp[4096];
    const device T* x_row = x + size_t(row) * size_t(K);
    device T* out_row = out + size_t(row) * size_t(K);
    for (int idx = tid; idx < K; idx += 128) {
        const int a = idx / D2;
        const int j = idx % D2;
        float s = 0.0f;
        for (int i = 0; i < D1; ++i) {
            s = fma(float(p1[i * D1 + a]), float(x_row[i * D2 + j]), s);
        }
        tg_tmp[idx] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float c_lo = HAS_CLIP ? float(clip_bounds[0]) : 0.0f;
    const float c_hi = HAS_CLIP ? float(clip_bounds[1]) : 0.0f;
    for (int idx = tid; idx < K; idx += 128) {
        const int a = idx / D2;
        const int b = idx % D2;
        float s = 0.0f;
        const int a_off = a * D2;
        for (int j = 0; j < D2; ++j) {
            s = fma(tg_tmp[a_off + j], float(p2[j * D2 + b]), s);
        }
        if (HAS_CLIP) {
            s = clamp(s, c_lo, c_hi);
        }
        out_row[idx] = static_cast<T>(s);
    }
    """

private let kroneckerKernel = MLXFast.metalKernel(
    name: "flatquant_kronecker_prologue",
    inputNames: ["x", "p1", "p2", "clip_bounds", "shapes"],
    outputNames: ["out"],
    source: kroneckerSource,
    header: BasaltMetalSubstrate.combinedHeader
)
