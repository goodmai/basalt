import BasaltCore
import Foundation
import MLX
import MLXNN

public enum GatedActivation: Int, Sendable, CaseIterable {
    case silu = 0
    case geluApprox = 1
}

public struct AffineQuantizedWeights {
    public let weight: MLXArray
    public let scales: MLXArray
    public let biases: MLXArray?
    public private(set) var bias: MLXArray?
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    public let globalScale: MLXArray?

    public init(
        weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        bias: MLXArray? = nil, groupSize: Int = 64, bits: Int = 4
    ) {
        (self.weight, self.scales, self.biases, self.bias) = (weight, scales, biases, bias)
        (self.groupSize, self.bits, self.mode, self.globalScale) = (groupSize, bits, .affine, nil)
    }

    public init(_ linear: QuantizedLinear) {
        (self.weight, self.scales, self.biases, self.bias) = (linear.weight, linear.scales, linear.biases, linear.bias)
        (self.groupSize, self.bits, self.mode, self.globalScale) = (linear.groupSize, linear.bits, linear.mode, linear.globalScale)
    }

    package func withoutBias() -> AffineQuantizedWeights {
        var copy = self
        copy.bias = nil
        return copy
    }
}

public enum FusedTransformerKernels {
    public static let rows = 1 ... 8
    public static let supportedBits: Set<Int> = [4, 8]
    public static let supportedGroupSizes: Set<Int> = [32, 64, 128]
    public static let dtypes: Set<DType> = [.float16, .bfloat16, .float32]
    public static let supportedDTypes: Set<DType> = dtypes

    public static func supports(rows m: Int, bits: Int, groupSize: Int, columns k: Int, dtype: DType) -> Bool {
        rows.contains(m) && supportedBits.contains(bits) && supportedGroupSizes.contains(groupSize)
            && supportedDTypes.contains(dtype) && k >= groupSize && k % groupSize == 0
    }

    private static func hasValidAffineShapes(_ weights: AffineQuantizedWeights, columns k: Int) -> Bool {
        guard weights.weight.ndim == 2, weights.scales.ndim == 2,
              let biases = weights.biases, biases.ndim == 2
        else { return false }
        let outputs = weights.scales.dim(0)
        guard outputs > 0, outputs <= Int(Int32.max / 2), k <= Int(Int32.max - 256) else { return false }
        let expectedGroups = k / weights.groupSize
        let expectedPacked = (k * weights.bits) / 32
        return weights.scales.dim(1) == expectedGroups
            && biases.shape == [outputs, expectedGroups]
            && weights.weight.shape == [outputs, expectedPacked]
    }

    private static func handles(_ weights: AffineQuantizedWeights, rows m: Int, columns k: Int, dtype: DType) -> Bool {
        guard Device.defaultDevice() == .gpu,
              weights.mode == .affine, weights.globalScale == nil,
              weights.weight.dtype == .uint32,
              supportedDTypes.contains(weights.scales.dtype),
              weights.biases?.dtype == weights.scales.dtype,
              supports(rows: m, bits: weights.bits, groupSize: weights.groupSize, columns: k, dtype: dtype)
        else { return false }
        return hasValidAffineShapes(weights, columns: k)
    }

    private static func isRealFloat(_ dtype: DType) -> Bool {
        dtype == .float16 || dtype == .bfloat16 || dtype == .float32 || dtype == .float64
    }

    private static func promotedDType(_ a: DType, _ b: DType) -> DType {
        if a == b { return a }
        if a == .complex64 || b == .complex64 { return .complex64 }
        if a == .float64 || b == .float64 { return .float64 }
        if isRealFloat(a) && isRealFloat(b) { return .float32 }
        if isRealFloat(a) { return a }
        if isRealFloat(b) { return b }
        if (a == .uint64 && b.isSignedInteger) || (b == .uint64 && a.isSignedInteger) { return .float32 }
        return .int64
    }

    private static func addBias(_ out: MLXArray, bias: MLXArray?) -> MLXArray {
        guard let bias else { return out }
        if Device.defaultDevice() == .gpu && (out.dtype == .float64 || bias.dtype == .float64) {
            return Device.withDefaultDevice(.cpu) { out + bias }
        }
        return out + bias
    }

    private static func outputDType(input: DType, weights: AffineQuantizedWeights) -> DType {
        guard weights.mode == .affine else { return input }
        let withScales = promotedDType(input, weights.scales.dtype)
        return weights.biases.map { promotedDType(withScales, $0.dtype) } ?? withScales
    }

    private static func hasValidEmptyWeights(_ weights: AffineQuantizedWeights, inputs: Int, outputs: Int) -> Bool {
        guard weights.weight.dtype == .uint32, weights.groupSize > 0, weights.bits > 0,
              inputs.isMultiple(of: weights.groupSize),
              weights.weight.dim(0) == outputs,
              weights.weight.dim(1) * 32 == inputs * weights.bits,
              weights.scales.dim(1) == inputs / weights.groupSize
        else { return false }
        if weights.mode == .affine {
            guard let biases = weights.biases else { return false }
            return biases.shape == weights.scales.shape && isRealFloat(promotedDType(weights.scales.dtype, biases.dtype))
        }
        return weights.biases == nil && weights.scales.dtype == .uint8
    }

    private static func emptyOutputIfNeeded(
        _ x: MLXArray, weights: AffineQuantizedWeights, outputDType: DType
    ) -> MLXArray? {
        guard isRealFloat(outputDType), x.ndim > 0, weights.weight.ndim == 2, weights.scales.ndim == 2 else { return nil }
        let inputs = x.dim(-1)
        let outputs = weights.scales.dim(0)
        guard x.size == 0 || outputs == 0,
              hasValidEmptyWeights(weights, inputs: inputs, outputs: outputs)
        else { return nil }
        let make = {
            let zeros = MLXArray.zeros(Array(x.shape.dropLast()) + [outputs], dtype: outputDType)
            return weights.bias.map { zeros + $0 } ?? zeros
        }
        let needsCPU = (outputDType == .float64 || weights.bias?.dtype == .float64) && Device.defaultDevice() == .gpu
        return needsCPU ? Device.withDefaultDevice(.cpu, make) : make()
    }

    public static func rmsNormQuantizedMatmul(
        _ x: MLXArray, normWeight: MLXArray, eps: Float = 1e-5, weights: AffineQuantizedWeights
    ) -> MLXArray {
        let inputs = x.ndim > 0 ? x.dim(-1) : 0
        let m = inputs > 0 ? x.size / inputs : 0
        let normOutType = promotedDType(x.dtype, normWeight.dtype)
        let outType = outputDType(input: normOutType, weights: weights)
        guard normWeight.ndim == 1, normWeight.dim(0) == inputs, supportedDTypes.contains(normWeight.dtype),
              let biases = weights.biases, handles(weights, rows: m, columns: inputs, dtype: x.dtype)
        else {
            if isRealFloat(normOutType), normWeight.ndim == 1, normWeight.dim(0) == inputs,
               let empty = emptyOutputIfNeeded(x, weights: weights, outputDType: outType) {
                return empty
            }
            let normed = (Device.defaultDevice() == .gpu && (x.dtype == .float64 || normWeight.dtype == .float64))
                ? Device.withDefaultDevice(.cpu) { MLXFast.rmsNorm(x, weight: normWeight, eps: eps) }
                : MLXFast.rmsNorm(x, weight: normWeight, eps: eps)
            return stockMatmul(normed, weights: weights)
        }
        let outputs = weights.scales.dim(0)
        let out = rmsNormQmvKernel(
            [x.reshaped(m, inputs), normWeight, weights.weight, weights.scales, biases, MLXArray([eps])],
            template: [("T", outType), ("M", m), ("B", weights.bits), ("G", weights.groupSize)],
            grid: (32, 4 * ((outputs + 7) / 8), 1),
            threadGroup: (32, 4, 1),
            outputShapes: [[m, outputs]],
            outputDTypes: [outType])[0]
        return addBias(out.reshaped(Array(x.shape.dropLast()) + [outputs]), bias: weights.bias)
    }

    public static func rmsNormQuantizedMatmul(_ x: MLXArray, norm: RMSNorm, linear: QuantizedLinear) -> MLXArray {
        guard type(of: norm) == RMSNorm.self, type(of: linear) == QuantizedLinear.self else {
            return linear(norm(x))
        }
        return rmsNormQuantizedMatmul(x, normWeight: norm.weight, eps: norm.eps, weights: AffineQuantizedWeights(linear))
    }

    public static func gatedQuantizedMatmul(
        gate: MLXArray, up: MLXArray, activation: GatedActivation = .silu, weights: AffineQuantizedWeights
    ) -> MLXArray {
        let inputs = gate.ndim > 0 ? gate.dim(-1) : 0
        let m = inputs > 0 ? gate.size / inputs : 0
        let gateActType: DType = (isRealFloat(gate.dtype) || gate.dtype == .complex64) ? gate.dtype : .float32
        let actType = promotedDType(gateActType, up.dtype)
        let outType = outputDType(input: actType, weights: weights)
        guard gate.shape == up.shape, gate.dtype == up.dtype,
              let biases = weights.biases, handles(weights, rows: m, columns: inputs, dtype: gate.dtype)
        else {
            if isRealFloat(actType), gate.shape == up.shape,
               let empty = emptyOutputIfNeeded(gate, weights: weights, outputDType: outType) {
                return empty
            }
            let act = (Device.defaultDevice() == .gpu && (gate.dtype == .float64 || up.dtype == .float64))
                ? Device.withDefaultDevice(.cpu) { activated(gate: gate, up: up, activation: activation) }
                : activated(gate: gate, up: up, activation: activation)
            return stockMatmul(act, weights: weights)
        }
        let outputs = weights.scales.dim(0)
        let out = gatedQmvKernel(
            [gate.reshaped(m, inputs), up.reshaped(m, inputs), weights.weight, weights.scales, biases],
            template: [("T", outType), ("M", m), ("B", weights.bits), ("G", weights.groupSize), ("ACT", activation.rawValue)],
            grid: (32, 4 * ((outputs + 7) / 8), 1),
            threadGroup: (32, 4, 1),
            outputShapes: [[m, outputs]],
            outputDTypes: [outType])[0]
        return addBias(out.reshaped(Array(gate.shape.dropLast()) + [outputs]), bias: weights.bias)
    }

    public static func gatedQuantizedMatmul(
        gate: MLXArray, up: MLXArray, activation: GatedActivation = .silu, down: QuantizedLinear
    ) -> MLXArray {
        guard type(of: down) == QuantizedLinear.self else {
            return down(activated(gate: gate, up: up, activation: activation))
        }
        return gatedQuantizedMatmul(gate: gate, up: up, activation: activation, weights: AffineQuantizedWeights(down))
    }

    private static func activated(gate: MLXArray, up: MLXArray, activation: GatedActivation) -> MLXArray {
        switch activation {
        case .silu: silu(gate) * up
        case .geluApprox: geluApproximate(gate) * up
        }
    }

    static func stockMatmul(_ x: MLXArray, weights: AffineQuantizedWeights) -> MLXArray {
        let outType = outputDType(input: x.dtype, weights: weights)
        if let empty = emptyOutputIfNeeded(x, weights: weights, outputDType: outType) {
            return empty
        }
        var out: MLXArray
        let validAffineScales = weights.biases.map { isRealFloat(promotedDType(weights.scales.dtype, $0.dtype)) } ?? false
        if outType == .float64, (weights.mode != .affine || validAffineScales) {
            let runF64: () -> MLXArray = {
                let dense: MLXArray
                if weights.mode == .affine {
                    dense = dequantized(
                        weights.weight, scales: weights.scales, biases: weights.biases,
                        groupSize: weights.groupSize, bits: weights.bits, mode: .affine).asType(.float64)
                } else {
                    dense = dequantized(
                        weights.weight, scales: weights.scales, biases: weights.biases,
                        groupSize: weights.groupSize, bits: weights.bits, mode: weights.mode, dtype: .float32).asType(.float64)
                }
                return matmul(x.asType(.float64), dense.swappedAxes(-1, -2))
            }
            out = Device.defaultDevice() == .gpu ? Device.withDefaultDevice(.cpu, runF64) : runF64()
        } else {
            out = quantizedMM(
                x, weights.weight, scales: weights.scales, biases: weights.biases,
                transpose: true, groupSize: weights.groupSize, bits: weights.bits, mode: weights.mode)
        }
        if let globalScale = weights.globalScale {
            let scaleOut: () -> MLXArray = { (out * (globalScale / Float(448 * 6))).asType(out.dtype) }
            out = (Device.defaultDevice() == .gpu && (out.dtype == .float64 || globalScale.dtype == .float64))
                ? Device.withDefaultDevice(.cpu, scaleOut) : scaleOut()
        }
        return addBias(out, bias: weights.bias)
    }
}

public final class PassthroughRMSNorm: RMSNorm {
    public init(replacing norm: RMSNorm) {
        super.init(dimensions: norm.weight.dim(0), eps: norm.eps)
        update(parameters: ModuleParameters.unflattened(["weight": norm.weight]))
        freeze()
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray { x }
}

public final class FusedRMSNormQuantizedLinear: QuantizedLinear {
    public let normWeight: MLXArray
    public let normEps: Float

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        if training {
            return super.callAsFunction(MLXFast.rmsNorm(x, weight: normWeight, eps: normEps))
        }
        return FusedTransformerKernels.rmsNormQuantizedMatmul(
            x, normWeight: normWeight, eps: normEps, weights: AffineQuantizedWeights(self))
    }

    @discardableResult
    public static func install(in model: Module) -> Int {
        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        let pairs: [(normSuffix: String, projSuffixes: [String])] = [
            ("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
            ("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
        ]
        var updates: [(String, Module)] = []
        var fusedCount = 0
        for (path, module) in modules {
            for pair in pairs where path.hasSuffix("." + pair.normSuffix) {
                guard let norm = module as? RMSNorm, type(of: norm) == RMSNorm.self else { continue }
                let prefix = String(path.dropLast(pair.normSuffix.count))
                guard isSafeToFuse(pair.normSuffix, prefix: prefix, modules: modules) else { continue }
                let targets = pair.projSuffixes.compactMap { suffix -> (String, QuantizedLinear)? in
                    guard let q = modules[prefix + suffix] as? QuantizedLinear,
                          type(of: q) == QuantizedLinear.self,
                          q.mode == .affine, q.biases != nil, q.globalScale == nil,
                          FusedTransformerKernels.supportedBits.contains(q.bits),
                          FusedTransformerKernels.supportedGroupSizes.contains(q.groupSize)
                    else { return nil }
                    return (prefix + suffix, q)
                }
                guard targets.count == pair.projSuffixes.count else { continue }
                updates.append((path, PassthroughRMSNorm(replacing: norm)))
                for (projPath, proj) in targets {
                    updates.append((projPath, FusedRMSNormQuantizedLinear(replacing: proj, normWeight: norm.weight, eps: norm.eps)))
                    fusedCount += 1
                }
            }
        }
        PHMAdapterApplicator.updateLeafModules(in: model, updates: updates)
        return fusedCount
    }

    private static let standardLayerChildren: Set<String> = [
        "self_attn", "mlp", "input_layernorm", "post_attention_layernorm",
        "pre_feedforward_layernorm", "post_feedforward_layernorm",
    ]
    private static let standardSelfAttnChildren: Set<String> = [
        "q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm", "rope",
    ]
    private static let standardMLPChildren: Set<String> = [
        "gate_proj", "up_proj", "down_proj",
    ]

    private static func isSafeToFuse(_ normSuffix: String, prefix: String, modules: [String: Module]) -> Bool {
        guard !hasNonStandardChild(prefix: prefix, allowed: standardLayerChildren, modules: modules, allowNested: true) else {
            return false
        }
        if normSuffix == "post_attention_layernorm" {
            return modules[prefix + "pre_feedforward_layernorm"] == nil
                && modules[prefix + "post_feedforward_layernorm"] == nil
                && !hasNonStandardChild(prefix: prefix + "mlp.", allowed: standardMLPChildren, modules: modules, allowNested: false)
        }
        let hasMLP = modules[prefix + "mlp"] != nil || modules.keys.contains { $0.hasPrefix(prefix + "mlp.") }
        let hasMLPNorm = modules[prefix + "post_attention_layernorm"] != nil || modules[prefix + "pre_feedforward_layernorm"] != nil
        guard !hasMLP || hasMLPNorm else { return false }
        guard !(modules[prefix + "self_attn"] is DFlashAttention) else { return false }
        if let selfAttn = modules[prefix + "self_attn"],
           !selfAttn.parameters().flattened().allSatisfy({ $0.0.contains(".") }) {
            return false
        }
        return !hasNonStandardChild(prefix: prefix + "self_attn.", allowed: standardSelfAttnChildren, modules: modules, allowNested: false)
    }

    private static func hasNonStandardChild(
        prefix: String, allowed: Set<String>, modules: [String: Module], allowNested: Bool
    ) -> Bool {
        modules.keys.contains { path in
            guard path.hasPrefix(prefix) else { return false }
            let rest = path.dropFirst(prefix.count)
            guard !rest.isEmpty else { return false }
            if let dot = rest.firstIndex(of: ".") {
                return !allowNested || !allowed.contains(String(rest[..<dot]))
            }
            return !allowed.contains(String(rest))
        }
    }

    public init(replacing layer: QuantizedLinear, normWeight: MLXArray, eps: Float) {
        self.normWeight = normWeight
        self.normEps = eps
        super.init(
            weight: layer.weight, bias: layer.bias, scales: layer.scales, biases: layer.biases,
            groupSize: layer.groupSize, bits: layer.bits, mode: layer.mode, globalScale: layer.globalScale)
        freeze()
        train(false)
    }
}

public final class FusedGatedQuantizedMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") public var gateProj: Linear
    @ModuleInfo(key: "up_proj") public var upProj: Linear
    @ModuleInfo(key: "down_proj") public var downProj: Linear
    public let activation: GatedActivation

    public init(gateProj: Linear, upProj: Linear, downProj: Linear, activation: GatedActivation = .silu) {
        self._gateProj.wrappedValue = gateProj
        self._upProj.wrappedValue = upProj
        self._downProj.wrappedValue = downProj
        self.activation = activation
        super.init()
        train(false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = gateProj(x)
        let u = upProj(x)
        if !training, let down = downProj as? QuantizedLinear, type(of: down) == QuantizedLinear.self {
            return FusedTransformerKernels.gatedQuantizedMatmul(gate: g, up: u, activation: activation, down: down)
        }
        let act = activation == .silu ? silu(g) * u : geluApproximate(g) * u
        return downProj(act)
    }
}

private enum FusedTransformerMetal {
    static let preamble = """
        constexpr int SIMD_W = 32, SG_COUNT = 4, THREADS = SIMD_W * SG_COUNT;
        constexpr int COLS_PER_SG = 2, COLS_PER_TG = 8, TILE_K = 256, PACK = 32 / B;
        const int lane = int(thread_index_in_simdgroup), sg = int(simdgroup_index_in_threadgroup);
        const int tid = sg * SIMD_W + lane, col_base = int(threadgroup_position_in_grid.y) * COLS_PER_TG;
        const int N = int(scales_shape[0]), KG = K / G, KW = K / PACK;
        float acc0[M], acc1[M];
        for (int m = 0; m < M; m++) { acc0[m] = 0.0f; acc1[m] = 0.0f; }
        threadgroup float x_tile[M * TILE_K];
        """

    static let qmvDotBody = """
            const int k_local = lane * 8;
            if (k_local < tile_len) {
                const int k_global = k_base + k_local, g_idx = k_global / G;
                for (int r = 0; r < COLS_PER_SG; r++) {
                    const int col = col_base + sg * COLS_PER_SG + r;
                    if (col < N) {
                        const float sc = float(scales[size_t(col) * size_t(KG) + size_t(g_idx)]);
                        const float bi = float(biases[size_t(col) * size_t(KG) + size_t(g_idx)]);
                        float w_deq[8];
                        if (B == 4) {
                            const uint32_t word = w[size_t(col) * size_t(KW) + size_t(k_global / 8)];
                            for (int j = 0; j < 8; j++) w_deq[j] = fma(sc, float((word >> (4 * j)) & 0xFu), bi);
                        } else {
                            const size_t w_idx = size_t(col) * size_t(KW) + size_t(k_global / 4);
                            const uint32_t w0 = w[w_idx], w1 = w[w_idx + 1];
                            for (int j = 0; j < 4; j++) {
                                w_deq[j] = fma(sc, float((w0 >> (8 * j)) & 0xFFu), bi);
                                w_deq[j + 4] = fma(sc, float((w1 >> (8 * j)) & 0xFFu), bi);
                            }
                        }
                        for (int m = 0; m < M; m++) {
                            float dot = 0.0f;
                            for (int j = 0; j < 8; j++) dot = fma(w_deq[j], x_tile[m * TILE_K + k_local + j], dot);
                            if (r == 0) acc0[m] += dot; else acc1[m] += dot;
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const int col0 = col_base + sg * COLS_PER_SG, col1 = col0 + 1;
        for (int m = 0; m < M; m++) {
            const float sum0 = simd_sum(acc0[m]), sum1 = simd_sum(acc1[m]);
            if (lane == 0) {
                if (col0 < N) out[size_t(m) * size_t(N) + size_t(col0)] = static_cast<T>(sum0);
                if (col1 < N) out[size_t(m) * size_t(N) + size_t(col1)] = static_cast<T>(sum1);
            }
        }
        """

    static let rmsNormQmvSource = """
        const int K = int(x_shape[1]);
        \(preamble)
        float sq[M];
        for (int m = 0; m < M; m++) sq[m] = 0.0f;
        for (int k = tid; k < K; k += THREADS) {
            for (int m = 0; m < M; m++) {
                const float xv = float(x[size_t(m) * size_t(K) + size_t(k)]);
                sq[m] = fma(xv, xv, sq[m]);
            }
        }
        threadgroup float tg_sq[SG_COUNT * M];
        threadgroup float inv_rms[M];
        for (int m = 0; m < M; m++) {
            const float sg_sum = simd_sum(sq[m]);
            if (lane == 0) tg_sq[sg * M + m] = sg_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            for (int m = 0; m < M; m++) {
                const float part = (lane < SG_COUNT) ? tg_sq[lane * M + m] : 0.0f;
                const float total_sq = simd_sum(part);
                if (lane == 0) inv_rms[m] = metal::precise::rsqrt(total_sq / float(K) + float(eps_val[0]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k_base = 0; k_base < K; k_base += TILE_K) {
            const int tile_len = min(TILE_K, K - k_base);
            for (int t = tid; t < tile_len; t += THREADS) {
                const int k = k_base + t;
                const float nw = float(norm_w[k]);
                for (int m = 0; m < M; m++) {
                    const float xv = float(x[size_t(m) * size_t(K) + size_t(k)]);
                    x_tile[m * TILE_K + t] = xv * inv_rms[m] * nw;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        \(qmvDotBody)
        """

    static let gatedQmvSource = """
        const int K = int(gate_shape[1]);
        \(preamble)
        for (int k_base = 0; k_base < K; k_base += TILE_K) {
            const int tile_len = min(TILE_K, K - k_base);
            for (int t = tid; t < tile_len; t += THREADS) {
                const int k = k_base + t;
                for (int m = 0; m < M; m++) {
                    const size_t idx = size_t(m) * size_t(K) + size_t(k);
                    const float g = float(gate[idx]), u = float(up[idx]);
                    float act = 0.0f;
                    if (ACT == 0) {
                        act = (g / (1.0f + metal::precise::exp(-g))) * u;
                    } else {
                        const float inner = 0.79788456f * fma(0.044715f, g * g * g, g);
                        act = 0.5f * g * (1.0f + metal::precise::tanh(inner)) * u;
                    }
                    x_tile[m * TILE_K + t] = act;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        \(qmvDotBody)
        """
}

private let rmsNormQmvKernel = MLXFast.metalKernel(
    name: "fused_rmsnorm_qmv",
    inputNames: ["x", "norm_w", "w", "scales", "biases", "eps_val"],
    outputNames: ["out"],
    source: FusedTransformerMetal.rmsNormQmvSource,
    header: "#include <metal_simdgroup>\n#include <metal_math>\n")

private let gatedQmvKernel = MLXFast.metalKernel(
    name: "fused_gated_qmv",
    inputNames: ["gate", "up", "w", "scales", "biases"],
    outputNames: ["out"],
    source: FusedTransformerMetal.gatedQmvSource,
    header: "#include <metal_simdgroup>\n#include <metal_math>\n")
