import Foundation
import MLX
import MLXLMCommon
import MLXNN

package final class RicursiveMLP: Module, UnaryLayer {
    static let rowChunkLimit = 2048

    let clampThreshold: Float?
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    package init(_ args: RicursiveConfiguration) {
        self.clampThreshold = args.ffnClampThreshold
        self._gate.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
        self._up.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
        self._down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: false)
    }

    private func forwardRows(_ x: MLXArray) -> MLXArray {
        var g = gate(x)
        var u = up(x)
        if let threshold = clampThreshold, threshold > 0 {
            let hi = MLXArray(threshold, dtype: g.dtype)
            let lo = MLXArray(-threshold, dtype: u.dtype)
            g = minimum(g, hi)
            u = clip(u, min: lo, max: hi)
        }
        return down(silu(g) * u)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard x.ndim == 2 || x.ndim == 3 else { return forwardRows(x) }
        let hidden = x.dim(-1)
        let rows = hidden > 0 ? x.size / hidden : 0
        guard rows > Self.rowChunkLimit else { return forwardRows(x) }
        let flat = x.reshaped(rows, hidden)
        var pieces: [MLXArray] = []
        var start = 0
        while start < rows {
            let end = min(start + Self.rowChunkLimit, rows)
            pieces.append(forwardRows(flat[start ..< end]))
            start = end
        }
        let joined = pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 0)
        return joined.reshaped(Array(x.shape.dropLast()) + [joined.dim(-1)])
    }
}

package final class RicursiveAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let usesRotary: Bool
    let gateFullPrecision: Bool
    let rope: RoPE?

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "g_proj") var wg: Linear?
    @ParameterInfo(key: "value_res_lambda") var valueResLambda: MLXArray?

    package init(_ args: RicursiveConfiguration, layerIdx: Int) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDim = args.headDim
        self.scale = 1.0 / Float(args.headDim).squareRoot()
        let isLocal = args.layerTypes[layerIdx] == "sliding_attention"
        self.usesRotary = isLocal
        self.gateFullPrecision = args.gateFullPrecision

        self._wq.wrappedValue = Linear(args.hiddenSize, args.attentionHeads * args.headDim, bias: false)
        self._wk.wrappedValue = Linear(args.hiddenSize, args.kvHeads * args.headDim, bias: false)
        self._wv.wrappedValue = Linear(args.hiddenSize, args.kvHeads * args.headDim, bias: false)
        self._wo.wrappedValue = Linear(args.attentionHeads * args.headDim, args.hiddenSize, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        if args.useGatedAttention {
            self._wg.wrappedValue = Linear(args.hiddenSize, args.attentionHeads * args.headDim, bias: false)
        }
        if args.useValres {
            self._valueResLambda.wrappedValue = MLXArray.zeros([1], dtype: .float32)
        }
        self.rope = isLocal ? RoPE(dimensions: args.headDim, traditional: false, base: args.ropeTheta) : nil
    }

    package func callAsFunction(
        _ x: MLXArray,
        valueResidual: MLXArray?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let valueInput: MLXArray
        if let valueResidual, let lambda = valueResLambda {
            let blend = sigmoid(lambda.asType(.float32))[0].asType(x.dtype)
            let base = valueResidual.asType(x.dtype)
            valueInput = base + blend * (x - base)
        } else {
            valueInput = x
        }

        var queries = qNorm(wq(x).reshaped(b, l, heads, headDim)).transposed(0, 2, 1, 3)
        var keys = kNorm(wk(x).reshaped(b, l, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let values = wv(valueInput).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)

        if usesRotary, let rope {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries.asType(.float32), offset: offset).asType(x.dtype)
            keys = applyRotaryPosition(rope, to: keys.asType(.float32), offset: offset).asType(x.dtype)
        }

        var attended = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(b, l, -1)

        if let wg {
            let rawGate = wg(x)
            let gate = gateFullPrecision
                ? sigmoid(rawGate.asType(.float32)).asType(attended.dtype)
                : sigmoid(rawGate)
            attended = attended * gate
        }
        return wo(attended)
    }
}
