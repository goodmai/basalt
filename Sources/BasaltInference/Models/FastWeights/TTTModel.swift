import BasaltCore
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

struct TTTNormParams {
    let weight: MLXArray
    let bias: MLXArray
    let eps: Float
}

struct TTTSliceInputs {
    let q: MLXArray
    let k: MLXArray
    let v: MLXArray
    let eta: MLXArray
    let tau: MLXArray
}

struct TTTState {
    var w1: MLXArray
    var b1: MLXArray
    var w2: MLXArray?
    var b2: MLXArray?
    var accW1: MLXArray
    var accB1: MLXArray
    var accW2: MLXArray?
    var accB2: MLXArray?

    static func slotCount(variant: TTTConfiguration.Variant) -> Int {
        variant == .linear ? 6 : 10
    }

    static func convSlots(variant: TTTConfiguration.Variant) -> (q: Int, k: Int) {
        variant == .linear ? (4, 5) : (8, 9)
    }

    static func load(
        from cache: FastWeightLayerCache?,
        batch: Int,
        mixer: TTTSequenceMixer
    ) -> TTTState {
        let h = mixer.numHeads
        let d = mixer.headDim
        let inner = mixer.innerDim
        let w1 = cache?.slot(0)?.asType(.float32)
            ?? broadcast(mixer.w1Init.asType(.float32), to: [batch, h, d, inner])
        let b1 = cache?.slot(1)?.asType(.float32)
            ?? broadcast(mixer.b1Init.asType(.float32), to: [batch, h, 1, inner])
        if mixer.config.variant == .linear {
            let accW1 = cache?.slot(2)?.asType(.float32) ?? MLXArray.zeros([batch, h, d, inner], dtype: .float32)
            let accB1 = cache?.slot(3)?.asType(.float32) ?? MLXArray.zeros([batch, h, 1, inner], dtype: .float32)
            return TTTState(w1: w1, b1: b1, w2: nil, b2: nil, accW1: accW1, accB1: accB1, accW2: nil, accB2: nil)
        }
        let w2Init = mixer.w2Init ?? MLXArray.zeros([1, h, inner, d], dtype: .float32)
        let b2Init = mixer.b2Init ?? MLXArray.zeros([1, h, 1, d], dtype: .float32)
        let w2 = cache?.slot(2)?.asType(.float32) ?? broadcast(w2Init.asType(.float32), to: [batch, h, inner, d])
        let b2 = cache?.slot(3)?.asType(.float32) ?? broadcast(b2Init.asType(.float32), to: [batch, h, 1, d])
        let accW1 = cache?.slot(4)?.asType(.float32) ?? MLXArray.zeros([batch, h, d, inner], dtype: .float32)
        let accB1 = cache?.slot(5)?.asType(.float32) ?? MLXArray.zeros([batch, h, 1, inner], dtype: .float32)
        let accW2 = cache?.slot(6)?.asType(.float32) ?? MLXArray.zeros([batch, h, inner, d], dtype: .float32)
        let accB2 = cache?.slot(7)?.asType(.float32) ?? MLXArray.zeros([batch, h, 1, d], dtype: .float32)
        return TTTState(w1: w1, b1: b1, w2: w2, b2: b2, accW1: accW1, accB1: accB1, accW2: accW2, accB2: accB2)
    }

    func commit(to cache: FastWeightLayerCache?, variant: TTTConfiguration.Variant) {
        guard let cache else { return }
        cache.commit(0, w1)
        cache.commit(1, b1)
        if variant == .linear {
            cache.commit(2, accW1)
            cache.commit(3, accB1)
        } else {
            cache.commit(2, w2)
            cache.commit(3, b2)
            cache.commit(4, accW1)
            cache.commit(5, accB1)
            cache.commit(6, accW2)
            cache.commit(7, accB2)
        }
    }
}

enum TTTDualStep {
    static func dualProjection(
        q: MLXArray,
        k: MLXArray,
        grad: MLXArray,
        eta: MLXArray,
        causal: MLXArray
    ) -> MLXArray {
        let etaRow = eta.swappedAxes(-1, -2)
        let qk = matmul(q, k.swappedAxes(-1, -2))
        let weighted = ((qk * causal) + causal) * etaRow
        return matmul(weighted, grad)
    }

    static func stepLinear(
        _ inputs: TTTSliceInputs,
        norm: TTTNormParams,
        slice: FastWeightChunkSlice,
        state: inout TTTState
    ) -> MLXArray {
        let z1 = matmul(inputs.k, state.w1) + state.b1
        let target = inputs.v - inputs.k
        let grad = FastWeightMath.lnFusedL2Backward(
            x: z1,
            target: target,
            weight: norm.weight,
            bias: norm.bias,
            eps: norm.eps
        )
        let causal = FastWeightMath.causalLowerTriangular(length: slice.count)
        let zBase = matmul(inputs.q, state.w1) + state.b1
        let zPrev = matmul(inputs.q, state.accW1) + state.accB1
        let zDual = dualProjection(q: inputs.q, k: inputs.k, grad: grad, eta: inputs.eta, causal: causal)
        let zBar = zBase - inputs.tau * (zPrev + zDual)
        let weightedGrad = grad * inputs.eta
        state.accW1 = state.accW1 + matmul(inputs.k.swappedAxes(-1, -2), weightedGrad)
        state.accB1 = state.accB1 + weightedGrad.sum(axis: -2, keepDims: true)
        if slice.completesChunk {
            let tauLast = Float(1.0) / Float(slice.chunkSize)
            state.w1 = state.w1 - tauLast * state.accW1
            state.b1 = state.b1 - tauLast * state.accB1
            state.accW1 = MLXArray.zeros(like: state.accW1)
            state.accB1 = MLXArray.zeros(like: state.accB1)
        }
        return zBar
    }

    static func stepMLP(
        _ inputs: TTTSliceInputs,
        norm: TTTNormParams,
        slice: FastWeightChunkSlice,
        state: inout TTTState
    ) -> MLXArray {
        guard let w2 = state.w2, let b2 = state.b2, let accW2 = state.accW2, let accB2 = state.accB2 else {
            return stepLinear(inputs, norm: norm, slice: slice, state: &state)
        }
        let z1 = matmul(inputs.k, state.w1) + state.b1
        let x2 = FastWeightMath.geluTanh(z1)
        let z2 = matmul(x2, w2) + b2
        let grad2 = FastWeightMath.lnFusedL2Backward(
            x: z2,
            target: inputs.v - inputs.k,
            weight: norm.weight,
            bias: norm.bias,
            eps: norm.eps
        )
        let grad1 = FastWeightMath.geluTanhBackward(z: z1, gradOut: matmul(grad2, w2.swappedAxes(-1, -2)))
        let causal = FastWeightMath.causalLowerTriangular(length: slice.count)
        let z1Base = matmul(inputs.q, state.w1) + state.b1
        let z1Prev = matmul(inputs.q, state.accW1) + state.accB1
        let z1Bar = z1Base - inputs.tau * (z1Prev + dualProjection(q: inputs.q, k: inputs.k, grad: grad1, eta: inputs.eta, causal: causal))
        let x2Bar = FastWeightMath.geluTanh(z1Bar)
        let z2Base = matmul(x2Bar, w2) + b2
        let z2Prev = matmul(x2Bar, accW2) + accB2
        let z2Bar = z2Base - inputs.tau * (z2Prev + dualProjection(q: x2Bar, k: x2, grad: grad2, eta: inputs.eta, causal: causal))
        let wg1 = grad1 * inputs.eta
        let wg2 = grad2 * inputs.eta
        let nextAccW1 = state.accW1 + matmul(inputs.k.swappedAxes(-1, -2), wg1)
        let nextAccB1 = state.accB1 + wg1.sum(axis: -2, keepDims: true)
        let nextAccW2 = accW2 + matmul(x2.swappedAxes(-1, -2), wg2)
        let nextAccB2 = accB2 + wg2.sum(axis: -2, keepDims: true)
        if slice.completesChunk {
            let tauLast = Float(1.0) / Float(slice.chunkSize)
            state.w1 = state.w1 - tauLast * nextAccW1
            state.b1 = state.b1 - tauLast * nextAccB1
            state.w2 = w2 - tauLast * nextAccW2
            state.b2 = b2 - tauLast * nextAccB2
            state.accW1 = MLXArray.zeros(like: nextAccW1)
            state.accB1 = MLXArray.zeros(like: nextAccB1)
            state.accW2 = MLXArray.zeros(like: w2)
            state.accB2 = MLXArray.zeros(like: b2)
        } else {
            state.accW1 = nextAccW1
            state.accB1 = nextAccB1
            state.accW2 = nextAccW2
            state.accB2 = nextAccB2
        }
        return z2Bar
    }
}

final class TTTSequenceMixer: Module {
    let config: TTTConfiguration
    let numHeads: Int
    let headDim: Int
    let innerDim: Int
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear?
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    @ModuleInfo(key: "conv_q") var convQ: CausalDepthwiseConv1d?
    @ModuleInfo(key: "conv_k") var convK: CausalDepthwiseConv1d?
    @ModuleInfo(key: "ttt_lr_gate") var tttLRGate: Linear
    @ModuleInfo(key: "post_norm") var postNorm: LayerNorm
    @ParameterInfo(key: "ttt_norm_weight") var tttNormWeight: MLXArray
    @ParameterInfo(key: "ttt_norm_bias") var tttNormBias: MLXArray
    @ParameterInfo(key: "w1_init") var w1Init: MLXArray
    @ParameterInfo(key: "b1_init") var b1Init: MLXArray
    @ParameterInfo(key: "w2_init") var w2Init: MLXArray?
    @ParameterInfo(key: "b2_init") var b2Init: MLXArray?

    init(_ config: TTTConfiguration) {
        self.config = config
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.innerDim = config.variant == .linear ? config.headDim : config.headDim * config.mlpExpansion
        self.rope = RoPE(dimensions: config.headDim, traditional: false, base: config.ropeTheta)
        let d = config.hiddenSize
        _qProj.wrappedValue = Linear(d, d, bias: false)
        _kProj.wrappedValue = config.shareQK ? nil : Linear(d, d, bias: false)
        _vProj.wrappedValue = Linear(d, d, bias: false)
        _oProj.wrappedValue = Linear(d, d, bias: false)
        _gProj.wrappedValue = config.useGate ? Linear(d, d, bias: false) : nil
        if config.convKernel > 1 {
            _convQ.wrappedValue = CausalDepthwiseConv1d(channels: d, kernelSize: config.convKernel, bias: true)
            _convK.wrappedValue = CausalDepthwiseConv1d(channels: d, kernelSize: config.convKernel, bias: true)
        }
        _tttLRGate.wrappedValue = Linear(d, config.numAttentionHeads, bias: true)
        _postNorm.wrappedValue = LayerNorm(dimensions: d, eps: config.rmsNormEps, bias: true)
        _tttNormWeight.wrappedValue = MLXArray.ones([1, numHeads, 1, headDim], dtype: .float32)
        _tttNormBias.wrappedValue = MLXArray.zeros([1, numHeads, 1, headDim], dtype: .float32)
        _w1Init.wrappedValue = MLXRandom.normal([1, numHeads, headDim, innerDim]) * Float(0.02)
        _b1Init.wrappedValue = MLXArray.zeros([1, numHeads, 1, innerDim], dtype: .float32)
        if config.variant == .mlp {
            _w2Init.wrappedValue = MLXRandom.normal([1, numHeads, innerDim, headDim]) * Float(0.02)
            _b2Init.wrappedValue = MLXArray.zeros([1, numHeads, 1, headDim], dtype: .float32)
        }
    }

    func callAsFunction(_ x: MLXArray, cache: FastWeightLayerCache?) -> MLXArray {
        let (batch, length, hidden) = (x.dim(0), x.dim(1), x.dim(2))
        guard length > 0 else { return MLXArray.zeros([batch, 0, hidden], dtype: x.dtype) }
        let (q, k, v) = projectQKV(x, cache: cache)
        let lrScale = config.tttBaseLR / Float(headDim)
        let eta = (sigmoid(tttLRGate(x).asType(.float32)) * lrScale).transposed(0, 2, 1).expandedDimensions(axis: -1)
        let (slices, nextStep) = FastWeightSchedule.slices(
            chunkSize: config.miniBatchSize,
            stepInChunk: cache?.stepInChunk ?? 0,
            tokenCount: length
        )
        var state = TTTState.load(from: cache, batch: batch, mixer: self)
        let norm = TTTNormParams(weight: tttNormWeight, bias: tttNormBias, eps: config.tttNormEps)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(slices.count)
        for slice in slices {
            let positions = (slice.stepStart ..< slice.stepEnd).map { Float(1.0) / Float($0 + 1) }
            let tau = MLXArray(positions).reshaped(1, 1, slice.count, 1)
            let sliceInputs = TTTSliceInputs(
                q: q[0..., 0..., slice.range, 0...],
                k: k[0..., 0..., slice.range, 0...],
                v: v[0..., 0..., slice.range, 0...],
                eta: eta[0..., 0..., slice.range, 0...],
                tau: tau
            )
            let zBar = config.variant == .linear
                ? TTTDualStep.stepLinear(sliceInputs, norm: norm, slice: slice, state: &state)
                : TTTDualStep.stepMLP(sliceInputs, norm: norm, slice: slice, state: &state)
            let normed = FastWeightMath.layerNormForward(zBar, weight: norm.weight, bias: norm.bias, eps: norm.eps)
            outputs.append(sliceInputs.q + normed)
        }
        state.commit(to: cache, variant: config.variant)
        cache?.advance(tokens: length, nextStepInChunk: nextStep)
        let merged = concatenated(outputs, axis: 2).transposed(0, 2, 1, 3).reshaped(batch, length, hidden).asType(x.dtype)
        let normedOut = postNorm(merged)
        let gated = gProj.map { geluApproximate($0(x)) * normedOut } ?? normedOut
        return oProj(gated)
    }

    private func projectQKV(_ x: MLXArray, cache: FastWeightLayerCache?) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        let (batch, length) = (x.dim(0), x.dim(1))
        var qRaw = qProj(x)
        var kRaw = kProj.map { $0(x) } ?? qRaw
        let convSlots = TTTState.convSlots(variant: config.variant)
        if let convQ, let convK {
            let (qConv, qTail) = convQ(qRaw, state: cache?.slot(convSlots.q))
            let (kConv, kTail) = convK(kRaw, state: cache?.slot(convSlots.k))
            (qRaw, kRaw) = (qConv, kConv)
            cache?.commit(convSlots.q, qTail)
            cache?.commit(convSlots.k, kTail)
        }
        let posOffset = cache?.offset ?? 0
        let q = rope(qRaw.reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3), offset: posOffset).asType(.float32)
        let k = rope(kRaw.reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3), offset: posOffset).asType(.float32)
        let v = vProj(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3).asType(.float32)
        return (q, k, v)
    }
}

final class TTTBlock: Module {
    @ModuleInfo(key: "seq_norm") var seqNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "seq_modeling_block") var mixer: TTTSequenceMixer
    @ModuleInfo(key: "mlp") var mlp: FastWeightSwiGLUMLP

    init(_ config: TTTConfiguration) {
        _seqNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _mixer.wrappedValue = TTTSequenceMixer(config)
        _mlp.wrappedValue = FastWeightSwiGLUMLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
    }

    func callAsFunction(_ x: MLXArray, cache: FastWeightLayerCache?) -> MLXArray {
        let h = x + mixer(seqNorm(x), cache: cache)
        return h + mlp(ffnNorm(h))
    }
}

final class TTTModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [TTTBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: TTTConfiguration) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in TTTBlock(config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        for (index, layer) in layers.enumerated() {
            let layerCache = cache.flatMap { index < $0.count ? ($0[index] as? FastWeightLayerCache) : nil }
            h = layer(h, cache: layerCache)
        }
        return norm(h)
    }
}

public final class TTTModel: Module, LLMModel, LoRAModel {
    public static let modelType = "ttt"
    public let vocabularySize: Int
    public let configuration: TTTConfiguration
    @ModuleInfo(key: "model") var model: TTTModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: TTTConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabSize
        _model.wrappedValue = TTTModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public convenience init(configurationData data: Data) throws {
        self.init(try JSONDecoder.json5().decode(TTTConfiguration.self, from: data))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return lmHead.map { $0(hidden) } ?? model.embedTokens.asLinear(hidden)
    }

    public var loraLayers: [Module] { model.layers }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let slots = TTTState.slotCount(variant: configuration.variant)
        return (0 ..< configuration.numHiddenLayers).map { _ in
            FastWeightLayerCache(chunkSize: configuration.miniBatchSize, slotCount: slots)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        TTTCheckpoint.sanitize(weights, config: configuration)
    }
}

enum TTTCheckpoint {
    private static let infixReplacements: [(String, String)] = [
        (".input_layernorm.", ".seq_norm."),
        (".post_attention_layernorm.", ".ffn_norm."),
        (".self_attn.", ".seq_modeling_block."),
    ]

    private static let suffixReplacements: [(String, String)] = [
        (".learnable_ttt_lr_weight", ".ttt_lr_gate.weight"),
        (".learnable_ttt_lr_bias", ".ttt_lr_gate.bias"),
        (".W1", ".w1_init"),
        (".b1", ".b1_init"),
        (".W2", ".w2_init"),
        (".b2", ".b2_init"),
        (".conv_q.weight", ".conv_q.conv.weight"),
        (".conv_q.bias", ".conv_q.conv.bias"),
        (".conv_k.weight", ".conv_k.conv.weight"),
        (".conv_k.bias", ".conv_k.conv.bias"),
    ]

    static func sanitize(_ weights: [String: MLXArray], config: TTTConfiguration) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (rawKey, value) in weights {
            guard !shouldDrop(rawKey, config: config) else { continue }
            var key = rawKey
            for (from, to) in infixReplacements where key.contains(from) {
                key = key.replacingOccurrences(of: from, with: to)
            }
            for (from, to) in suffixReplacements where key.hasSuffix(from) {
                key = String(key.dropLast(from.count)) + to
                break
            }
            out[key] = normalizeTensor(value, key: key, config: config)
        }
        return out
    }

    private static func shouldDrop(_ key: String, config: TTTConfiguration) -> Bool {
        if config.tieWordEmbeddings && key.hasPrefix("lm_head.") { return true }
        if key.contains("rotary_emb.inv_freq") || key.hasSuffix(".token_idx") || key.hasSuffix(".learnable_token_idx") {
            return true
        }
        if config.convKernel <= 1 && (key.contains(".conv_q.") || key.contains(".conv_k.")) { return true }
        if !config.useGate && key.contains(".g_proj.") { return true }
        if config.shareQK && key.contains(".k_proj.") { return true }
        return false
    }

    private static func normalizeTensor(_ value: MLXArray, key: String, config: TTTConfiguration) -> MLXArray {
        let h = config.numAttentionHeads
        let d = config.headDim
        let inner = config.variant == .linear ? d : d * config.mlpExpansion
        if key.hasSuffix(".conv_q.conv.weight") || key.hasSuffix(".conv_k.conv.weight") {
            return FastWeightMath.normalizeConvWeight(value, kernelSize: config.convKernel)
        }
        if key.hasSuffix(".ttt_lr_gate.weight"), value.ndim == 3 {
            return value.reshaped(h, config.hiddenSize)
        }
        if key.hasSuffix(".ttt_lr_gate.bias"), value.ndim != 1 {
            return value.reshaped(h)
        }
        if key.hasSuffix(".ttt_norm_weight") || key.hasSuffix(".ttt_norm_bias") {
            return value.reshaped(1, h, 1, d)
        }
        if key.hasSuffix(".w1_init") { return value.reshaped(1, h, d, inner) }
        if key.hasSuffix(".b1_init") { return value.reshaped(1, h, 1, inner) }
        if key.hasSuffix(".w2_init") { return value.reshaped(1, h, inner, d) }
        if key.hasSuffix(".b2_init") { return value.reshaped(1, h, 1, d) }
        return value
    }
}
