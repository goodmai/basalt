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
                        if !s.isNaN { s = s > lo ? s : lo }
                        if !s.isNaN { s = s < hi ? s : hi }
                    }
                    out[rowBase + a * d2 + b] = s
                }
            }
        }
        return out
    }

    private static func validatedShape(
        _ x: MLXArray,
        p1: MLXArray,
        p2: MLXArray,
        clipBounds: MLXArray?
    ) -> (rows: Int, d1: Int, d2: Int, k: Int)? {
        guard x.ndim >= 1,
              p1.ndim == 2, p2.ndim == 2,
              p1.dim(0) > 0, p1.dim(0) == p1.dim(1),
              p2.dim(0) > 0, p2.dim(0) == p2.dim(1)
        else { return nil }
        let d1 = p1.dim(0)
        let d2 = p2.dim(0)
        let (k, overflow) = d1.multipliedReportingOverflow(by: d2)
        guard !overflow, k > 0, x.dim(-1) == k else { return nil }
        let rows = x.size / k
        guard rows > 0, clipBounds.map({ $0.size == 2 }) ?? true else { return nil }
        return (rows, d1, d2, k)
    }

    package static func transform(
        _ x: MLXArray,
        p1: MLXArray,
        p2: MLXArray,
        clipBounds: MLXArray? = nil
    ) -> MLXArray {
        let origType = x.dtype
        guard let shape = validatedShape(x, p1: p1, p2: p2, clipBounds: clipBounds) else { return x }
        if Device.defaultDevice() == .gpu,
           shape.k <= maxThreadgroupK,
           shape.rows <= Int(Int32.max),
           supportedDTypes.contains(origType) {
            let p1Clean = supportedDTypes.contains(p1.dtype) ? p1 : p1.asType(.float32, stream: p1.dtype == .float64 ? .cpu : .default)
            let p2Clean = supportedDTypes.contains(p2.dtype) ? p2 : p2.asType(.float32, stream: p2.dtype == .float64 ? .cpu : .default)
            let clipClean = clipBounds.map { $0.asType(.float32, stream: $0.dtype == .float64 ? .cpu : .default) }
                ?? MLXArray([Float(-1e9), Float(1e9)])
            let shapes = MLXArray([Int32(shape.rows), Int32(shape.d1), Int32(shape.d2)])
            let out = kroneckerKernel(
                [x.reshaped(shape.rows, shape.k), p1Clean, p2Clean, clipClean.reshaped(2), shapes],
                template: [
                    ("T", origType),
                    ("TP1", p1Clean.dtype),
                    ("TP2", p2Clean.dtype),
                    ("HAS_CLIP", clipBounds != nil),
                ],
                grid: (128, shape.rows, 1),
                threadGroup: (128, 1, 1),
                outputShapes: [[shape.rows, shape.k]],
                outputDTypes: [origType]
            )[0]
            return out.reshaped(x.shape)
        }
        return differentiableTransform(x, p1: p1, p2: p2, clipBounds: clipBounds)
    }

    package static func differentiableTransform(
        _ x: MLXArray,
        p1: MLXArray,
        p2: MLXArray,
        clipBounds: MLXArray? = nil
    ) -> MLXArray {
        let origType = x.dtype
        guard let shape = validatedShape(x, p1: p1, p2: p2, clipBounds: clipBounds) else { return x }
        if Device.defaultDevice() == .gpu,
           origType == .float64 || p1.dtype == .float64 || p2.dtype == .float64 || clipBounds?.dtype == .float64 {
            return Device.withDefaultDevice(.cpu) {
                differentiableTransform(x, p1: p1, p2: p2, clipBounds: clipBounds)
            }
        }
        let mat = x.asType(.float32).reshaped(shape.rows, shape.d1, shape.d2)
        let left = matmul(transposed(mat, axes: [0, 2, 1]), p1.asType(.float32))
        var prod = matmul(transposed(left, axes: [0, 2, 1]), p2.asType(.float32)).reshaped(x.shape)
        if let clipBounds {
            let b = clipBounds.asType(.float32).reshaped(2)
            prod = clip(prod, min: b[0], max: b[1])
        }
        return prod.asType(origType)
    }
}

package struct FlatQuantLayerSpec: Codable, Sendable, Equatable {
    package var bits: Int
    package var groupSize: Int
    package var mode: String
    package var d1: Int
    package var d2: Int
    package var hasClipBounds: Bool
    package var hasRowScale: Bool
    package var scaleDType: String?
    package var hasBias: Bool?

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
        case mode, d1, d2
        case hasClipBounds = "has_clip_bounds"
        case hasRowScale = "has_row_scale"
        case scaleDType = "scale_dtype"
        case hasBias = "has_bias"
    }

    package init(
        bits: Int,
        groupSize: Int,
        mode: QuantizationMode,
        d1: Int,
        d2: Int,
        hasClipBounds: Bool = false
    ) {
        self.bits = bits
        self.groupSize = groupSize
        self.mode = mode.rawValue
        self.d1 = d1
        self.d2 = d2
        self.hasClipBounds = hasClipBounds
        self.hasRowScale = false
    }

    package var quantizationMode: QuantizationMode {
        QuantizationMode(rawValue: mode) ?? .affine
    }

    package var resolvedScaleDType: DType {
        if quantizationMode != .affine { return .uint8 }
        if let scaleDType, let match = DType.allCases.first(where: { "\($0)" == scaleDType }) {
            return match
        }
        return .bfloat16
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
        let wf = weight.asType(.float32, stream: weight.dtype == .float64 ? .cpu : .default)
        let wRms = sqrt(maximum(mean(square(wf), axis: 0), Float(1e-10)))
        let validActs = activations.flatMap { k > 0 && $0.size > 0 && $0.size.isMultiple(of: k) ? $0 : nil }
        let actsF32 = validActs.map { $0.asType(.float32, stream: $0.dtype == .float64 ? .cpu : .default).reshaped(-1, k) }
        let aRms: MLXArray = if let actsF32 {
            sqrt(maximum(mean(square(actsF32), axis: 0), Float(1e-10)))
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
        if let q = clipQuantile, let actsF32, q > 0, q < 1 {
            let xTrans = FlatQuantKernels.transform(actsF32, p1: p1, p2: p2)
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
        let mat = weight.asType(.float32, stream: weight.dtype == .float64 ? .cpu : .default).reshaped(nOut, d1, d2)
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
        let qType: DType = linear.weight.dtype == .float64 ? .float32 : linear.weight.dtype
        let factors = calibrateFactors(weight: linear.weight, activations: activations, clipQuantile: clipQuantile)
        let wConj = conjugateWeight(linear.weight, factors: factors).asType(qType)
        let (wq, scales, biases) = MLX.quantized(wConj, groupSize: groupSize, bits: bits, mode: mode)
        let packed = MACaQPackedWeights(weight: wq, scales: scales, biases: biases)
        let basis = FlatQuantBasis(
            p1: factors.p1.asType(qType),
            p2: factors.p2.asType(qType),
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
    if (row >= M || K <= 0 || K > 4096) { return; }
    threadgroup float tg_tmp[4096];
    const size_t row_off = size_t(row) * size_t(K);
    for (int idx = tid; idx < K; idx += 128) {
        const int a = idx / D2;
        const int j = idx % D2;
        float s = 0.0f;
        for (int i = 0; i < D1; ++i) {
            s = fma(float(p1[i * D1 + a]), float(x[row_off + size_t(i * D2 + j)]), s);
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
            s = isnan(s) ? s : ((s > c_lo) ? s : c_lo);
            s = isnan(s) ? s : ((s < c_hi) ? s : c_hi);
        }
        out[row_off + size_t(idx)] = static_cast<T>(s);
    }
    """

private let kroneckerKernel = MLXFast.metalKernel(
    name: "flatquant_kronecker_prologue",
    inputNames: ["x", "p1", "p2", "clip_bounds", "shapes"],
    outputNames: ["out"],
    source: kroneckerSource,
    header: BasaltMetalSubstrate.combinedHeader
)

package final class FlatQuantLinear: QuantizedLinear {
    @ParameterInfo(key: "p1") package var p1: MLXArray
    @ParameterInfo(key: "p2") package var p2: MLXArray
    @ParameterInfo(key: "clip_bounds") package var clipBounds: MLXArray?
    @ParameterInfo(key: "row_scale") package var rowScale: MLXArray?

    package static func placeholder(replacing linear: Linear, spec: FlatQuantLayerSpec) -> FlatQuantLinear {
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
        let basis = FlatQuantBasis(
            p1: eye(spec.d1, dtype: factorType),
            p2: eye(spec.d2, dtype: factorType),
            clipBounds: spec.hasClipBounds ? MLXArray([Float(-1e9), Float(1e9)]) : nil,
            rowScale: spec.hasRowScale ? MLXArray.ones([outputs], dtype: factorType) : nil)
        let packed = MACaQPackedWeights(weight: weight, scales: scales, biases: biases)
        return FlatQuantLinear(
            packed: packed, bias: bias, basis: basis,
            groupSize: spec.groupSize, bits: spec.bits, mode: qMode)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let transformed = training
            ? FlatQuantKernels.differentiableTransform(x, p1: p1, p2: p2, clipBounds: clipBounds)
            : FlatQuantKernels.transform(x, p1: p1, p2: p2, clipBounds: clipBounds)
        guard let rowScale else {
            return FusedTransformerKernels.stockMatmul(transformed, weights: AffineQuantizedWeights(self))
        }
        let raw = FusedTransformerKernels.stockMatmul(transformed, weights: AffineQuantizedWeights(self).withoutBias())
        let rowScaleF = rowScale.asType(.float32, stream: rowScale.dtype == .float64 ? .cpu : .default)
        let rawF = raw.asType(.float32, stream: raw.dtype == .float64 ? .cpu : .default)
        let scaled = (rawF * rowScaleF).asType(raw.dtype, stream: raw.dtype == .float64 ? .cpu : .default)
        guard let bias else { return scaled }
        if Device.defaultDevice() == .gpu, (scaled.dtype == .float64 || bias.dtype == .float64) {
            return Device.withDefaultDevice(.cpu) { scaled + bias }
        }
        return scaled + bias
    }

    package func unrotatedDenseWeight() -> MLXArray {
        if weight.ndim == 2, weight.dim(0) == 0 || weight.dim(1) == 0 {
            return MLXArray.zeros([weight.dim(0), (weight.dim(1) * 32) / max(1, bits)], dtype: .float32)
        }
        let deq: () -> MLXArray = {
            dequantized(self.weight, scales: self.scales, biases: self.biases, groupSize: self.groupSize, bits: self.bits, mode: self.mode).asType(.float32)
        }
        var wTilde = (Device.defaultDevice() == .gpu && (scales.dtype == .float64 || biases?.dtype == .float64))
            ? Device.withDefaultDevice(.cpu, deq) : deq()
        if let rowScale {
            wTilde = wTilde * rowScale.asType(.float32, stream: rowScale.dtype == .float64 ? .cpu : .default).expandedDimensions(axis: -1)
        }
        let nOut = wTilde.dim(0)
        guard p1.ndim == 2, p2.ndim == 2,
              p1.dim(0) > 0, p1.dim(0) == p1.dim(1),
              p2.dim(0) > 0, p2.dim(0) == p2.dim(1),
              p1.dim(0) * p2.dim(0) == wTilde.dim(1)
        else { return wTilde }
        let d1 = p1.dim(0)
        let d2 = p2.dim(0)
        let mat = wTilde.reshaped(nOut, d1, d2)
        let p1f = p1.asType(.float32, stream: p1.dtype == .float64 ? .cpu : .default)
        let p2f = p2.asType(.float32, stream: p2.dtype == .float64 ? .cpu : .default)
        let left = matmul(transposed(mat, axes: [0, 2, 1]), p1f.T)
        return matmul(transposed(left, axes: [0, 2, 1]), p2f.T).reshaped(nOut, d1 * d2)
    }

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
        super.init(
            weight: packed.weight, bias: bias, scales: packed.scales,
            biases: mode == .affine ? packed.biases : nil, groupSize: groupSize, bits: bits, mode: mode)
        freeze()
    }
}

