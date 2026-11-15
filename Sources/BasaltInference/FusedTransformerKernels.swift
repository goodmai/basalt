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
    public let bias: MLXArray?
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
}

public enum FusedTransformerKernels {
    public static let rows = 1 ... 8
    public static let supportedBits: Set<Int> = [4, 8]
    public static let supportedGroupSizes: Set<Int> = [32, 64, 128]
    public static let dtypes: Set<DType> = [.float16, .bfloat16, .float32]

    public static func supports(rows m: Int, bits: Int, groupSize: Int, columns k: Int, dtype: DType) -> Bool {
        rows.contains(m) && supportedBits.contains(bits) && supportedGroupSizes.contains(groupSize)
            && dtypes.contains(dtype) && k >= groupSize && k % groupSize == 0
    }

    private static func handles(_ weights: AffineQuantizedWeights, rows m: Int, columns k: Int, dtype: DType) -> Bool {
        weights.mode == .affine && weights.biases != nil && weights.globalScale == nil
            && weights.weight.ndim == 2 && weights.weight.dtype == .uint32
            && weights.scales.ndim == 2 && dtypes.contains(weights.scales.dtype)
            && weights.biases?.dtype == weights.scales.dtype
            && supports(rows: m, bits: weights.bits, groupSize: weights.groupSize, columns: k, dtype: dtype)
    }

    public static func rmsNormQuantizedMatmul(
        _ x: MLXArray, normWeight: MLXArray, eps: Float = 1e-5, weights: AffineQuantizedWeights
    ) -> MLXArray {
        let inputs = x.ndim > 0 ? x.dim(-1) : 0
        let m = inputs > 0 ? x.size / inputs : 0
        guard normWeight.ndim == 1, normWeight.dim(0) == inputs, dtypes.contains(normWeight.dtype),
              let biases = weights.biases, handles(weights, rows: m, columns: inputs, dtype: x.dtype)
        else {
            return stockMatmul(MLXFast.rmsNorm(x, weight: normWeight, eps: eps), weights: weights)
        }
        let outputs = weights.scales.dim(0)
        let out = rmsNormQmvKernel(
            [x.reshaped(m, inputs), normWeight, weights.weight, weights.scales, biases, MLXArray([eps])],
            template: [("T", x.dtype), ("M", m), ("B", weights.bits), ("G", weights.groupSize)],
            grid: (32, 4 * ((outputs + 7) / 8), 1),
            threadGroup: (32, 4, 1),
            outputShapes: [[m, outputs]],
            outputDTypes: [x.dtype])[0]
        let reshaped = out.reshaped(Array(x.shape.dropLast()) + [outputs])
        return weights.bias.map { reshaped + $0 } ?? reshaped
    }

    public static func rmsNormQuantizedMatmul(_ x: MLXArray, norm: RMSNorm, linear: QuantizedLinear) -> MLXArray {
        guard linear.globalScale == nil else { return linear(norm(x)) }
        return rmsNormQuantizedMatmul(x, normWeight: norm.weight, eps: norm.eps, weights: AffineQuantizedWeights(linear))
    }

    public static func gatedQuantizedMatmul(
        gate: MLXArray, up: MLXArray, activation: GatedActivation = .silu, weights: AffineQuantizedWeights
    ) -> MLXArray {
        let inputs = gate.ndim > 0 ? gate.dim(-1) : 0
        let m = inputs > 0 ? gate.size / inputs : 0
        guard gate.shape == up.shape, gate.dtype == up.dtype,
              let biases = weights.biases, handles(weights, rows: m, columns: inputs, dtype: gate.dtype)
        else {
            return stockMatmul(activated(gate: gate, up: up, activation: activation), weights: weights)
        }
        let outputs = weights.scales.dim(0)
        let out = gatedQmvKernel(
            [gate.reshaped(m, inputs), up.reshaped(m, inputs), weights.weight, weights.scales, biases],
            template: [("T", gate.dtype), ("M", m), ("B", weights.bits), ("G", weights.groupSize), ("ACT", activation.rawValue)],
            grid: (32, 4 * ((outputs + 7) / 8), 1),
            threadGroup: (32, 4, 1),
            outputShapes: [[m, outputs]],
            outputDTypes: [gate.dtype])[0]
        let reshaped = out.reshaped(Array(gate.shape.dropLast()) + [outputs])
        return weights.bias.map { reshaped + $0 } ?? reshaped
    }

    public static func gatedQuantizedMatmul(
        gate: MLXArray, up: MLXArray, activation: GatedActivation = .silu, down: QuantizedLinear
    ) -> MLXArray {
        guard down.globalScale == nil else { return down(activated(gate: gate, up: up, activation: activation)) }
        return gatedQuantizedMatmul(gate: gate, up: up, activation: activation, weights: AffineQuantizedWeights(down))
    }

    private static func activated(gate: MLXArray, up: MLXArray, activation: GatedActivation) -> MLXArray {
        switch activation {
        case .silu: silu(gate) * up
        case .geluApprox: geluApproximate(gate) * up
        }
    }

    private static func stockMatmul(_ x: MLXArray, weights: AffineQuantizedWeights) -> MLXArray {
        let out = quantizedMM(
            x, weights.weight, scales: weights.scales, biases: weights.biases,
            transpose: true, groupSize: weights.groupSize, bits: weights.bits, mode: weights.mode)
        return weights.bias.map { out + $0 } ?? out
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

    public init(replacing layer: QuantizedLinear, normWeight: MLXArray, eps: Float) {
        self.normWeight = normWeight
        self.normEps = eps
        super.init(
            weight: layer.weight, bias: layer.bias, scales: layer.scales, biases: layer.biases,
            groupSize: layer.groupSize, bits: layer.bits)
        freeze()
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        FusedTransformerKernels.rmsNormQuantizedMatmul(
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
                if pair.normSuffix == "post_attention_layernorm", modules[prefix + "pre_feedforward_layernorm"] != nil { continue }
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
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = gateProj(x)
        let u = upProj(x)
        if let down = downProj as? QuantizedLinear {
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
