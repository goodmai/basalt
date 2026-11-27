import BasaltCore
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

struct LaCTTargetNorms {
    let w1: MLXArray?
    let w2: MLXArray?
    let w3: MLXArray?
}

struct LaCTSliceInputs {
    let q: MLXArray
    let k: MLXArray
    let v: MLXArray
    let lr1: MLXArray
    let lr2: MLXArray?
    let lr3: MLXArray?
}

package struct LaCTMemoryState {
    package var w1: MLXArray
    package var w2: MLXArray?
    package var w3: MLXArray?
    package var g1: MLXArray
    package var g2: MLXArray?
    package var g3: MLXArray?
    package var m1: MLXArray?
    package var m2: MLXArray?
    package var m3: MLXArray?

    package static func slotCount(mode: LaCTConfiguration.MemoryMode) -> Int {
        mode == .linear ? 6 : 12
    }

    static func convSlots(mode: LaCTConfiguration.MemoryMode) -> (q: Int, k: Int, v: Int) {
        mode == .linear ? (3, 4, 5) : (9, 10, 11)
    }

    static func load(from cache: FastWeightLayerCache?, batch: Int, memory: LaCTMemory) -> LaCTMemoryState {
        memory.config.memoryMode == .linear
            ? loadLinear(from: cache, batch: batch, memory: memory)
            : loadSwiGLU(from: cache, batch: batch, memory: memory)
    }

    private static func slotOrZero(_ cache: FastWeightLayerCache?, _ index: Int, shape: [Int]) -> MLXArray {
        cache?.slot(index)?.asType(.float32) ?? MLXArray.zeros(shape, dtype: .float32)
    }

    private static func slotOrBroadcast(_ cache: FastWeightLayerCache?, _ index: Int, initTensor: MLXArray?, shape: [Int]) -> MLXArray {
        if let stored = cache?.slot(index) { return stored.asType(.float32) }
        guard let initTensor else { return MLXArray.zeros(shape, dtype: .float32) }
        return broadcast(initTensor.asType(.float32), to: shape)
    }

    private static func loadLinear(from cache: FastWeightLayerCache?, batch: Int, memory: LaCTMemory) -> LaCTMemoryState {
        let shape = [batch, memory.numHeads, memory.headDim, memory.headDim]
        let w1 = slotOrBroadcast(cache, 0, initTensor: memory.w1Init, shape: shape)
        let g1 = slotOrZero(cache, 1, shape: shape)
        let m1 = memory.config.updateRule == .sgd ? nil : slotOrZero(cache, 2, shape: shape)
        return LaCTMemoryState(w1: w1, w2: nil, w3: nil, g1: g1, g2: nil, g3: nil, m1: m1, m2: nil, m3: nil)
    }

    private static func loadSwiGLU(from cache: FastWeightLayerCache?, batch: Int, memory: LaCTMemory) -> LaCTMemoryState {
        let (h, d, m) = (memory.numHeads, memory.headDim, memory.hiddenDim)
        let (md, dm) = ([batch, h, m, d], [batch, h, d, m])
        let w1 = slotOrBroadcast(cache, 0, initTensor: memory.w1Init, shape: md)
        let w2 = slotOrBroadcast(cache, 1, initTensor: memory.w2Init, shape: dm)
        let w3 = slotOrBroadcast(cache, 2, initTensor: memory.w3Init, shape: md)
        let (g1, g2, g3) = (slotOrZero(cache, 3, shape: md), slotOrZero(cache, 4, shape: dm), slotOrZero(cache, 5, shape: md))
        let usesMom = memory.config.updateRule != .sgd
        let m1 = usesMom ? slotOrZero(cache, 6, shape: md) : nil
        let m2 = usesMom ? slotOrZero(cache, 7, shape: dm) : nil
        let m3 = usesMom ? slotOrZero(cache, 8, shape: md) : nil
        return LaCTMemoryState(w1: w1, w2: w2, w3: w3, g1: g1, g2: g2, g3: g3, m1: m1, m2: m2, m3: m3)
    }

    func commit(to cache: FastWeightLayerCache?, mode: LaCTConfiguration.MemoryMode) {
        guard let cache else { return }
        if mode == .linear {
            cache.commit(0, w1)
            cache.commit(1, g1)
            cache.commit(2, m1)
            return
        }
        cache.commit(0, w1)
        cache.commit(1, w2)
        cache.commit(2, w3)
        cache.commit(3, g1)
        cache.commit(4, g2)
        cache.commit(5, g3)
        cache.commit(6, m1)
        cache.commit(7, m2)
        cache.commit(8, m3)
    }
}

enum LaCTUpdateOps {
    static func cotangent(pred: MLXArray, target: MLXArray, objective: LaCTConfiguration.Objective) -> MLXArray {
        switch objective {
        case .dotProduct: -target
        case .l2: pred - target
        }
    }

    static func stepLinear(
        _ inputs: LaCTSliceInputs,
        config: LaCTConfiguration,
        targetNorm: MLXArray?,
        slice: FastWeightChunkSlice,
        state: inout LaCTMemoryState
    ) -> MLXArray {
        let out = matmul(inputs.q, state.w1.swappedAxes(-1, -2))
        let pred = matmul(inputs.k, state.w1.swappedAxes(-1, -2))
        let delta = cotangent(pred: pred, target: inputs.v, objective: config.objective)
        let weighted = delta * inputs.lr1
        state.g1 = state.g1 + matmul(weighted.swappedAxes(-1, -2), inputs.k)
        if slice.completesChunk {
            let (nextW, nextM) = flushMatrix(
                w: state.w1,
                gradSum: state.g1,
                momentum: state.m1,
                config: config,
                targetNorm: targetNorm
            )
            state.w1 = nextW
            state.m1 = nextM
            state.g1 = MLXArray.zeros(like: state.g1)
        }
        return out
    }

    static func stepSwiGLU(
        _ inputs: LaCTSliceInputs,
        config: LaCTConfiguration,
        norms: LaCTTargetNorms,
        slice: FastWeightChunkSlice,
        state: inout LaCTMemoryState
    ) -> MLXArray {
        guard let w2 = state.w2, let w3 = state.w3, let g2 = state.g2, let g3 = state.g3 else {
            return stepLinear(inputs, config: config, targetNorm: norms.w1, slice: slice, state: &state)
        }
        let w1T = state.w1.swappedAxes(-1, -2)
        let w2T = w2.swappedAxes(-1, -2)
        let w3T = w3.swappedAxes(-1, -2)
        let out = matmul(silu(matmul(inputs.q, w1T)) * matmul(inputs.q, w3T), w2T)
        let gateK = matmul(inputs.k, w1T)
        let upK = matmul(inputs.k, w3T)
        let siluK = silu(gateK)
        let hiddenK = siluK * upK
        let pred = matmul(hiddenK, w2T)
        let deltaV = cotangent(pred: pred, target: inputs.v, objective: config.objective)
        let deltaH = matmul(deltaV, w2)
        let deltaU = deltaH * siluK
        let deltaG = FastWeightMath.siluBackward(z: gateK, gradOut: deltaH * upK)
        let lr2 = inputs.lr2 ?? inputs.lr1
        let lr3 = inputs.lr3 ?? inputs.lr1
        let nextG1 = state.g1 + matmul((deltaG * inputs.lr1).swappedAxes(-1, -2), inputs.k)
        let nextG2 = g2 + matmul((deltaV * lr2).swappedAxes(-1, -2), hiddenK)
        let nextG3 = g3 + matmul((deltaU * lr3).swappedAxes(-1, -2), inputs.k)
        if slice.completesChunk {
            let (nw1, nm1) = flushMatrix(w: state.w1, gradSum: nextG1, momentum: state.m1, config: config, targetNorm: norms.w1)
            let (nw2, nm2) = flushMatrix(w: w2, gradSum: nextG2, momentum: state.m2, config: config, targetNorm: norms.w2)
            let (nw3, nm3) = flushMatrix(w: w3, gradSum: nextG3, momentum: state.m3, config: config, targetNorm: norms.w3)
            (state.w1, state.w2, state.w3) = (nw1, nw2, nw3)
            (state.m1, state.m2, state.m3) = (nm1, nm2, nm3)
            (state.g1, state.g2, state.g3) = (MLXArray.zeros(like: nw1), MLXArray.zeros(like: nw2), MLXArray.zeros(like: nw3))
        } else {
            (state.g1, state.g2, state.g3) = (nextG1, nextG2, nextG3)
        }
        return out
    }

    static func flushMatrix(
        w: MLXArray,
        gradSum: MLXArray,
        momentum: MLXArray?,
        config: LaCTConfiguration,
        targetNorm: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        guard config.baseLR > 0 else { return (w, momentum) }
        let avgGrad = gradSum / Float(config.chunkSize)
        let nextMomentum: MLXArray?
        let direction: MLXArray
        switch config.updateRule {
        case .sgd:
            nextMomentum = nil
            direction = avgGrad
        case .momentum:
            let prev = momentum ?? MLXArray.zeros(like: avgGrad)
            let m = config.momentumBeta * prev + avgGrad
            nextMomentum = m
            direction = m
        case .muon:
            let prev = momentum ?? MLXArray.zeros(like: avgGrad)
            let m = config.momentumBeta * prev + avgGrad
            nextMomentum = m
            direction = FastWeightMath.newtonSchulz5(m, steps: config.muonSteps)
        }
        let updated = w - config.baseLR * direction
        if config.useWeightNorm, let targetNorm {
            return (FastWeightMath.preserveRowNorm(updated, targetNorm: targetNorm), nextMomentum)
        }
        return (updated, nextMomentum)
    }
}

package final class LaCTMemory: Module {
    let config: LaCTConfiguration
    let numHeads: Int
    let headDim: Int
    let hiddenDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "lr_proj") var lrProj: Linear?
    @ModuleInfo(key: "q_conv") var qConv: CausalDepthwiseConv1d?
    @ModuleInfo(key: "k_conv") var kConv: CausalDepthwiseConv1d?
    @ModuleInfo(key: "v_conv") var vConv: CausalDepthwiseConv1d?
    @ModuleInfo(key: "fw_out_norm") var fwOutNorm: RMSNorm
    @ParameterInfo(key: "w1_init") var w1Init: MLXArray
    @ParameterInfo(key: "w2_init") var w2Init: MLXArray?
    @ParameterInfo(key: "w3_init") var w3Init: MLXArray?

    init(_ config: LaCTConfiguration) {
        self.config = config
        self.numHeads = config.numFastWeightHeads
        self.headDim = config.fastWeightHeadDim
        self.hiddenDim = config.fastWeightHiddenDim
        let d = config.hiddenSize
        let fwDim = numHeads * headDim
        let scale = Float(1.0) / Float(headDim).squareRoot()
        _qProj.wrappedValue = Linear(d, fwDim, bias: false)
        _kProj.wrappedValue = Linear(d, fwDim, bias: false)
        _vProj.wrappedValue = Linear(d, fwDim, bias: false)
        _oProj.wrappedValue = Linear(fwDim, d, bias: false)
        _lrProj.wrappedValue = config.memoryMode == .swiglu ? Linear(d, 3 * numHeads, bias: true) : nil
        if config.convKernel > 1 {
            _qConv.wrappedValue = CausalDepthwiseConv1d(channels: fwDim, kernelSize: config.convKernel, bias: true)
            _kConv.wrappedValue = CausalDepthwiseConv1d(channels: fwDim, kernelSize: config.convKernel, bias: true)
            _vConv.wrappedValue = CausalDepthwiseConv1d(channels: fwDim, kernelSize: config.convKernel, bias: true)
        }
        _fwOutNorm.wrappedValue = RMSNorm(dimensions: fwDim, eps: config.rmsNormEps)
        if config.memoryMode == .linear {
            let eyeInit = MLXArray.eye(headDim, dtype: .float32) * Float(0.1)
            _w1Init.wrappedValue = broadcast(eyeInit.reshaped(1, 1, headDim, headDim), to: [1, numHeads, headDim, headDim])
        } else {
            _w1Init.wrappedValue = MLXRandom.normal([1, numHeads, hiddenDim, headDim]) * scale
            _w2Init.wrappedValue = MLXRandom.normal([1, numHeads, headDim, hiddenDim]) * scale
            _w3Init.wrappedValue = MLXRandom.normal([1, numHeads, hiddenDim, headDim]) * scale
        }
    }

    func callAsFunction(_ x: MLXArray, cache: FastWeightLayerCache?) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let fwDim = numHeads * headDim
        guard length > 0 else { return MLXArray.zeros([batch, 0, config.hiddenSize], dtype: x.dtype) }
        let (q, k, v) = projectQKV(x, cache: cache)
        let (lr1, lr2, lr3) = projectLR(x, batch: batch, length: length)
        let (slices, nextStep) = FastWeightSchedule.slices(
            chunkSize: config.chunkSize,
            stepInChunk: cache?.stepInChunk ?? 0,
            tokenCount: length
        )
        var state = LaCTMemoryState.load(from: cache, batch: batch, memory: self)
        let norms = targetNorms()
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(slices.count)
        for slice in slices {
            let sliceInputs = LaCTSliceInputs(
                q: q[0..., 0..., slice.range, 0...],
                k: k[0..., 0..., slice.range, 0...],
                v: v[0..., 0..., slice.range, 0...],
                lr1: lr1[0..., 0..., slice.range, 0...],
                lr2: lr2.map { $0[0..., 0..., slice.range, 0...] },
                lr3: lr3.map { $0[0..., 0..., slice.range, 0...] }
            )
            let out = config.memoryMode == .linear
                ? LaCTUpdateOps.stepLinear(sliceInputs, config: config, targetNorm: norms.w1, slice: slice, state: &state)
                : LaCTUpdateOps.stepSwiGLU(sliceInputs, config: config, norms: norms, slice: slice, state: &state)
            outputs.append(out)
        }
        state.commit(to: cache, mode: config.memoryMode)
        cache?.advance(tokens: length, nextStepInChunk: nextStep)
        let merged = concatenated(outputs, axis: 2).transposed(0, 2, 1, 3).reshaped(batch, length, fwDim).asType(x.dtype)
        return oProj(fwOutNorm(merged))
    }

    private func targetNorms() -> LaCTTargetNorms {
        guard config.useWeightNorm else { return LaCTTargetNorms(w1: nil, w2: nil, w3: nil) }
        return LaCTTargetNorms(
            w1: FastWeightMath.rowL2Norm(w1Init),
            w2: w2Init.map { FastWeightMath.rowL2Norm($0) },
            w3: w3Init.map { FastWeightMath.rowL2Norm($0) }
        )
    }

    private func projectLR(_ x: MLXArray, batch: Int, length: Int) -> (MLXArray, MLXArray?, MLXArray?) {
        guard let lrProj else {
            return (MLXArray.ones([batch, numHeads, length, 1], dtype: .float32), nil, nil)
        }
        let gates = sigmoid(lrProj(x).asType(.float32)).reshaped(batch, length, 3, numHeads).transposed(2, 0, 3, 1)
        return (
            gates[0].expandedDimensions(axis: -1),
            gates[1].expandedDimensions(axis: -1),
            gates[2].expandedDimensions(axis: -1)
        )
    }

    private func projectQKV(_ x: MLXArray, cache: FastWeightLayerCache?) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        let (batch, length) = (x.dim(0), x.dim(1))
        var qRaw = qProj(x)
        var kRaw = kProj(x)
        var vRaw = vProj(x)
        let slots = LaCTMemoryState.convSlots(mode: config.memoryMode)
        if let qConv, let kConv, let vConv {
            let (qc, qTail) = qConv(qRaw, state: cache?.slot(slots.q))
            let (kc, kTail) = kConv(kRaw, state: cache?.slot(slots.k))
            let (vc, vTail) = vConv(vRaw, state: cache?.slot(slots.v))
            (qRaw, kRaw, vRaw) = (qc, kc, vc)
            cache?.commit(slots.q, qTail)
            cache?.commit(slots.k, kTail)
            cache?.commit(slots.v, vTail)
        }
        let qHeads = qRaw.reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        let kHeads = kRaw.reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        let vHeads = vRaw.reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3).asType(.float32)
        let q = config.memoryMode == .swiglu ? FastWeightMath.l2Normalize(qHeads) : qHeads.asType(.float32)
        let k = FastWeightMath.l2Normalize(kHeads)
        return (q, k, vHeads)
    }
}

final class LaCTSlidingAttention: Module {
    let config: LaCTConfiguration
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: LaCTConfiguration) {
        self.config = config
        self.scale = pow(Float(config.headDim), -0.5)
        self.rope = RoPE(dimensions: config.headDim, traditional: false, base: config.ropeTheta)
        let d = config.hiddenSize
        let kvDim = config.numKeyValueHeads * config.headDim
        _qProj.wrappedValue = Linear(d, d, bias: false)
        _kProj.wrappedValue = Linear(d, kvDim, bias: false)
        _vProj.wrappedValue = Linear(d, kvDim, bias: false)
        _oProj.wrappedValue = Linear(d, d, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let q = applyRotaryPosition(
            rope,
            to: qProj(x).reshaped(batch, length, config.numAttentionHeads, config.headDim).transposed(0, 2, 1, 3),
            offset: cache?.ropeOffset
        )
        let k = applyRotaryPosition(
            rope,
            to: kProj(x).reshaped(batch, length, config.numKeyValueHeads, config.headDim).transposed(0, 2, 1, 3),
            offset: cache?.ropeOffset
        )
        let v = vProj(x).reshaped(batch, length, config.numKeyValueHeads, config.headDim).transposed(0, 2, 1, 3)
        let out = attentionWithCacheUpdate(queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class LaCTBlock: Module {
    let config: LaCTConfiguration
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "fw_layernorm") var fwNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: LaCTSlidingAttention
    @ModuleInfo(key: "fast_weight") var fastWeight: LaCTMemory
    @ModuleInfo(key: "mlp") var mlp: FastWeightSwiGLUMLP

    init(_ config: LaCTConfiguration) {
        self.config = config
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _fwNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _selfAttn.wrappedValue = LaCTSlidingAttention(config)
        _fastWeight.wrappedValue = LaCTMemory(config)
        _mlp.wrappedValue = FastWeightSwiGLUMLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        swaCache: KVCache?,
        fwCache: FastWeightLayerCache?
    ) -> MLXArray {
        guard x.dim(1) > 0 else { return x }
        let afterAttn = x + selfAttn(inputNorm(x), mask: mask, cache: swaCache)
        let afterFW = afterAttn + fastWeight(fwNorm(afterAttn), cache: fwCache)
        return afterFW + mlp(postAttentionNorm(afterFW))
    }
}

final class LaCTModelInner: Module {
    let config: LaCTConfiguration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [LaCTBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: LaCTConfiguration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in LaCTBlock(config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let firstSWA = cache.flatMap { $0.isEmpty ? nil : $0[0] }
        let swaMask = makeAttentionMask(n: h.dim(1), cache: firstSWA, windowSize: config.windowSize)
        for (index, layer) in layers.enumerated() {
            let swaCache = cache.flatMap { 2 * index < $0.count ? $0[2 * index] : nil }
            let fwCache = cache.flatMap { 2 * index + 1 < $0.count ? ($0[2 * index + 1] as? FastWeightLayerCache) : nil }
            h = layer(h, mask: swaMask, swaCache: swaCache, fwCache: fwCache)
        }
        return norm(h)
    }
}

public final class LaCTModel: Module, LLMModel, LoRAModel {
    public static let modelType = "lact_swiglu"
    public static let aliasModelType = "lact"
    public static let modelTypes = [modelType, aliasModelType]
    public let vocabularySize: Int
    public let configuration: LaCTConfiguration
    @ModuleInfo(key: "model") var model: LaCTModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: LaCTConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabSize
        _model.wrappedValue = LaCTModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public convenience init(configurationData data: Data) throws {
        self.init(try JSONDecoder.json5().decode(LaCTConfiguration.self, from: data))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return lmHead.map { $0(hidden) } ?? model.embedTokens.asLinear(hidden)
    }

    public var loraLayers: [Module] { model.layers }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let slots = LaCTMemoryState.slotCount(mode: configuration.memoryMode)
        return (0 ..< configuration.numHiddenLayers).flatMap { _ -> [any KVCache] in
            [
                RotatingKVCache(maxSize: configuration.windowSize, keep: 0),
                FastWeightLayerCache(chunkSize: configuration.chunkSize, slotCount: slots),
            ]
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        LaCTCheckpoint.sanitize(weights, config: configuration)
    }
}

enum LaCTCheckpoint {
    private static let replacements: [(String, String)] = [
        ("model.embeddings.weight", "model.embed_tokens.weight"),
        (".attn_norm.", ".input_layernorm."),
        (".mlp_norm.", ".post_attention_layernorm."),
        (".attn.w0", ".fast_weight.w1_init"),
        (".attn.w1", ".fast_weight.w2_init"),
        (".attn.w2", ".fast_weight.w3_init"),
        (".attn.lr_proj.", ".fast_weight.lr_proj."),
        (".attn.q_conv.", ".fast_weight.q_conv."),
        (".attn.k_conv.", ".fast_weight.k_conv."),
        (".attn.v_conv.", ".fast_weight.v_conv."),
        (".attn.q_proj.", ".self_attn.q_proj."),
        (".attn.k_proj.", ".self_attn.k_proj."),
        (".attn.v_proj.", ".self_attn.v_proj."),
        (".attn.o_proj.", ".self_attn.o_proj."),
        (".q_conv.weight", ".q_conv.conv.weight"),
        (".q_conv.bias", ".q_conv.conv.bias"),
        (".k_conv.weight", ".k_conv.conv.weight"),
        (".k_conv.bias", ".k_conv.conv.bias"),
        (".v_conv.weight", ".v_conv.conv.weight"),
        (".v_conv.bias", ".v_conv.conv.bias"),
    ]

    static func sanitize(_ weights: [String: MLXArray], config: LaCTConfiguration) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (rawKey, value) in weights {
            if config.tieWordEmbeddings && rawKey.hasPrefix("lm_head.") { continue }
            if rawKey.contains("rotary_emb.inv_freq") || rawKey.contains(".momentum_proj.") { continue }
            if rawKey.hasSuffix(".attn.qkv.weight") {
                splitFusedQKV(rawKey, value: value, config: config, into: &out)
                continue
            }
            var key = rawKey
            for (from, to) in replacements where key.contains(from) {
                key = key.replacingOccurrences(of: from, with: to)
            }
            out[key] = normalizeTensor(value, key: key, config: config)
        }
        return out
    }

    private static func splitFusedQKV(
        _ rawKey: String,
        value: MLXArray,
        config: LaCTConfiguration,
        into out: inout [String: MLXArray]
    ) {
        let prefix = String(rawKey.dropLast(".attn.qkv.weight".count)) + ".self_attn."
        let qDim = config.hiddenSize
        let kvDim = config.numKeyValueHeads * config.headDim
        guard value.ndim == 2, value.dim(0) == qDim + 2 * kvDim else {
            out[rawKey] = value
            return
        }
        out[prefix + "q_proj.weight"] = value[0 ..< qDim, 0...]
        out[prefix + "k_proj.weight"] = value[qDim ..< (qDim + kvDim), 0...]
        out[prefix + "v_proj.weight"] = value[(qDim + kvDim)..., 0...]
    }

    private static func normalizeTensor(_ value: MLXArray, key: String, config: LaCTConfiguration) -> MLXArray {
        let h = config.numFastWeightHeads
        let d = config.fastWeightHeadDim
        let m = config.fastWeightHiddenDim
        if key.hasSuffix("_conv.conv.weight") {
            return FastWeightMath.normalizeConvWeight(value, kernelSize: config.convKernel)
        }
        if key.hasSuffix(".w1_init") {
            return config.memoryMode == .linear ? value.reshaped(1, h, d, d) : value.reshaped(1, h, m, d)
        }
        if key.hasSuffix(".w2_init") { return value.reshaped(1, h, d, m) }
        if key.hasSuffix(".w3_init") { return value.reshaped(1, h, m, d) }
        return value
    }
}
