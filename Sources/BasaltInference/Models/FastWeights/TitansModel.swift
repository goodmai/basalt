import BasaltCore
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

package struct TitansMemoryWeights {
    package var w1: MLXArray
    package var b1: MLXArray
    package var w2: MLXArray?
    package var b2: MLXArray?

    package static func zeros(batch: Int, heads: Int, headDim: Int, hiddenDim: Int, depth: Int) -> TitansMemoryWeights {
        let inner = depth == 1 ? headDim : hiddenDim
        return TitansMemoryWeights(
            w1: MLXArray.zeros([batch, heads, headDim, inner], dtype: .float32),
            b1: MLXArray.zeros([batch, heads, 1, inner], dtype: .float32),
            w2: depth == 2 ? MLXArray.zeros([batch, heads, inner, headDim], dtype: .float32) : nil,
            b2: depth == 2 ? MLXArray.zeros([batch, heads, 1, headDim], dtype: .float32) : nil
        )
    }

    package func evalMemory(_ x: MLXArray) -> MLXArray {
        let z1 = matmul(x, w1) + b1
        if let w2, let b2 {
            return matmul(silu(z1), w2) + b2 + x
        }
        return z1 + x
    }
}

package struct TitansMemoryState {
    package var base: TitansMemoryWeights
    package var current: TitansMemoryWeights
    package var momentum: TitansMemoryWeights

    package static func slotCount(depth: Int) -> Int {
        depth == 1 ? 9 : 15
    }

    static func convSlots(depth: Int) -> (q: Int, k: Int, v: Int) {
        depth == 1 ? (6, 7, 8) : (12, 13, 14)
    }

    static func load(from cache: FastWeightLayerCache?, batch: Int, memory: TitansNeuralMemory) -> TitansMemoryState {
        let depth = memory.config.memoryDepth
        let stride = depth == 1 ? 2 : 4
        let fallback = memory.initialWeights(batch: batch)
        let zeroMom = TitansMemoryWeights.zeros(
            batch: batch,
            heads: memory.numHeads,
            headDim: memory.headDim,
            hiddenDim: memory.hiddenDim,
            depth: depth
        )
        let base = loadGroup(cache, start: 0, depth: depth, fallback: fallback)
        let current = loadGroup(cache, start: stride, depth: depth, fallback: base)
        let momentum = loadGroup(cache, start: 2 * stride, depth: depth, fallback: zeroMom)
        return TitansMemoryState(base: base, current: current, momentum: momentum)
    }

    func commit(to cache: FastWeightLayerCache?, depth: Int) {
        guard let cache else { return }
        let stride = depth == 1 ? 2 : 4
        commitGroup(base, to: cache, start: 0, depth: depth)
        commitGroup(current, to: cache, start: stride, depth: depth)
        commitGroup(momentum, to: cache, start: 2 * stride, depth: depth)
    }

    private static func loadGroup(
        _ cache: FastWeightLayerCache?,
        start: Int,
        depth: Int,
        fallback: TitansMemoryWeights
    ) -> TitansMemoryWeights {
        let w1 = cache?.slot(start)?.asType(.float32) ?? fallback.w1
        let b1 = cache?.slot(start + 1)?.asType(.float32) ?? fallback.b1
        guard depth == 2 else { return TitansMemoryWeights(w1: w1, b1: b1, w2: nil, b2: nil) }
        let w2 = cache?.slot(start + 2)?.asType(.float32) ?? fallback.w2
        let b2 = cache?.slot(start + 3)?.asType(.float32) ?? fallback.b2
        return TitansMemoryWeights(w1: w1, b1: b1, w2: w2, b2: b2)
    }

    private func commitGroup(_ group: TitansMemoryWeights, to cache: FastWeightLayerCache, start: Int, depth: Int) {
        cache.commit(start, group.w1)
        cache.commit(start + 1, group.b1)
        if depth == 2 {
            cache.commit(start + 2, group.w2)
            cache.commit(start + 3, group.b2)
        }
    }
}

struct TitansStepWeights {
    let mem: MLXArray
    let mom: MLXArray
}

package struct TitansChunkUpdateInputs {
    package let keys: MLXArray
    package let values: MLXArray
    package let alpha: MLXArray
    package let eta: MLXArray
    package let theta: MLXArray

    package init(keys: MLXArray, values: MLXArray, alpha: MLXArray, eta: MLXArray, theta: MLXArray) {
        self.keys = keys
        self.values = values
        self.alpha = alpha
        self.eta = eta
        self.theta = theta
    }
}

package final class TitansNeuralMemory: Module {
    let config: TitansConfiguration
    let numHeads: Int
    let headDim: Int
    let hiddenDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "alpha_proj") var alphaProj: Linear
    @ModuleInfo(key: "eta_proj") var etaProj: Linear
    @ModuleInfo(key: "theta_proj") var thetaProj: Linear
    @ModuleInfo(key: "q_conv") var qConv: CausalDepthwiseConv1d?
    @ModuleInfo(key: "k_conv") var kConv: CausalDepthwiseConv1d?
    @ModuleInfo(key: "v_conv") var vConv: CausalDepthwiseConv1d?
    @ModuleInfo(key: "mem_norm") var memNorm: RMSNorm
    @ParameterInfo(key: "w1_init") var w1Init: MLXArray
    @ParameterInfo(key: "b1_init") var b1Init: MLXArray
    @ParameterInfo(key: "w2_init") var w2Init: MLXArray?
    @ParameterInfo(key: "b2_init") var b2Init: MLXArray?

    init(_ config: TitansConfiguration) {
        self.config = config
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.hiddenDim = config.memoryHiddenDim
        let d = config.hiddenSize
        let inner = config.memoryDepth == 1 ? headDim : hiddenDim
        _qProj.wrappedValue = Linear(d, d, bias: false)
        _kProj.wrappedValue = Linear(d, d, bias: false)
        _vProj.wrappedValue = Linear(d, d, bias: false)
        _oProj.wrappedValue = Linear(d, d, bias: false)
        _alphaProj.wrappedValue = Linear(d, numHeads, bias: true)
        _etaProj.wrappedValue = Linear(d, numHeads, bias: true)
        _thetaProj.wrappedValue = Linear(d, numHeads, bias: true)
        if config.convKernel > 1 {
            _qConv.wrappedValue = CausalDepthwiseConv1d(channels: d, kernelSize: config.convKernel, bias: true, applySiLU: true)
            _kConv.wrappedValue = CausalDepthwiseConv1d(channels: d, kernelSize: config.convKernel, bias: true, applySiLU: true)
            _vConv.wrappedValue = CausalDepthwiseConv1d(channels: d, kernelSize: config.convKernel, bias: true, applySiLU: true)
        }
        _memNorm.wrappedValue = RMSNorm(dimensions: d, eps: config.rmsNormEps)
        _w1Init.wrappedValue = MLXRandom.normal([1, numHeads, headDim, inner]) * Float(0.02)
        _b1Init.wrappedValue = MLXArray.zeros([1, numHeads, 1, inner], dtype: .float32)
        if config.memoryDepth == 2 {
            _w2Init.wrappedValue = MLXRandom.normal([1, numHeads, inner, headDim]) * Float(0.02)
            _b2Init.wrappedValue = MLXArray.zeros([1, numHeads, 1, headDim], dtype: .float32)
        }
    }

    func initialWeights(batch: Int) -> TitansMemoryWeights {
        let inner = config.memoryDepth == 1 ? headDim : hiddenDim
        let w1 = broadcast(w1Init.asType(.float32), to: [batch, numHeads, headDim, inner])
        let b1 = broadcast(b1Init.asType(.float32), to: [batch, numHeads, 1, inner])
        guard config.memoryDepth == 2, let w2Init, let b2Init else {
            return TitansMemoryWeights(w1: w1, b1: b1, w2: nil, b2: nil)
        }
        let w2 = broadcast(w2Init.asType(.float32), to: [batch, numHeads, inner, headDim])
        let b2 = broadcast(b2Init.asType(.float32), to: [batch, numHeads, 1, headDim])
        return TitansMemoryWeights(w1: w1, b1: b1, w2: w2, b2: b2)
    }

    func callAsFunction(_ x: MLXArray, cache: FastWeightLayerCache?) -> MLXArray {
        let (batch, length, hidden) = (x.dim(0), x.dim(1), x.dim(2))
        guard length > 0 else { return MLXArray.zeros([batch, 0, hidden], dtype: x.dtype) }
        let q = projectQuery(x, cache: cache, useConv: true)
        let (k, v, alpha, eta, theta) = projectWriteInputs(x, cache: cache)
        let (slices, nextStep) = FastWeightSchedule.slices(
            chunkSize: config.chunkSize,
            stepInChunk: cache?.stepInChunk ?? 0,
            tokenCount: length
        )
        var state = TitansMemoryState.load(from: cache, batch: batch, memory: self)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(slices.count)
        for slice in slices {
            let qs = q[0..., 0..., slice.range, 0...]
            outputs.append(state.base.evalMemory(qs))
            let update = TitansChunkUpdateInputs(
                keys: k[0..., 0..., slice.range, 0...],
                values: v[0..., 0..., slice.range, 0...],
                alpha: alpha[0..., 0..., slice.range],
                eta: eta[0..., 0..., slice.range],
                theta: theta[0..., 0..., slice.range]
            )
            Self.applyChunkUpdate(update, state: &state, completesChunk: slice.completesChunk)
        }
        state.commit(to: cache, depth: config.memoryDepth)
        cache?.advance(tokens: length, nextStepInChunk: nextStep)
        let merged = concatenated(outputs, axis: 2).transposed(0, 2, 1, 3).reshaped(batch, length, hidden).asType(x.dtype)
        return memNorm(oProj(merged))
    }

    func retrieve(_ x: MLXArray, cache: FastWeightLayerCache?, useConv: Bool) -> MLXArray {
        let (batch, length, hidden) = (x.dim(0), x.dim(1), x.dim(2))
        let q = projectQuery(x, cache: cache, useConv: useConv)
        let state = TitansMemoryState.load(from: cache, batch: batch, memory: self)
        let out = state.base.evalMemory(q).transposed(0, 2, 1, 3).reshaped(batch, length, hidden).asType(x.dtype)
        return memNorm(oProj(out))
    }

    func updateSlice(_ x: MLXArray, slice: FastWeightChunkSlice, cache: FastWeightLayerCache?) {
        let batch = x.dim(0)
        let (k, v, alpha, eta, theta) = projectWriteInputs(x, cache: cache)
        var state = TitansMemoryState.load(from: cache, batch: batch, memory: self)
        let update = TitansChunkUpdateInputs(keys: k, values: v, alpha: alpha, eta: eta, theta: theta)
        Self.applyChunkUpdate(update, state: &state, completesChunk: slice.completesChunk)
        state.commit(to: cache, depth: config.memoryDepth)
        cache?.advance(tokens: slice.count, nextStepInChunk: slice.completesChunk ? 0 : slice.stepEnd)
    }

    private func projectQuery(_ x: MLXArray, cache: FastWeightLayerCache?, useConv: Bool) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        var raw = qProj(x)
        if useConv, let qConv {
            let slot = TitansMemoryState.convSlots(depth: config.memoryDepth).q
            let (convOut, tail) = qConv(raw, state: cache?.slot(slot))
            raw = convOut
            cache?.commit(slot, tail)
        } else {
            raw = silu(raw)
        }
        return FastWeightMath.l2Normalize(raw.reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3))
    }

    private func projectWriteInputs(
        _ x: MLXArray,
        cache: FastWeightLayerCache?
    ) -> (k: MLXArray, v: MLXArray, alpha: MLXArray, eta: MLXArray, theta: MLXArray) {
        let (batch, length) = (x.dim(0), x.dim(1))
        var kRaw = kProj(x)
        var vRaw = vProj(x)
        let slots = TitansMemoryState.convSlots(depth: config.memoryDepth)
        if let kConv, let vConv {
            let (kc, kTail) = kConv(kRaw, state: cache?.slot(slots.k))
            let (vc, vTail) = vConv(vRaw, state: cache?.slot(slots.v))
            (kRaw, vRaw) = (kc, vc)
            cache?.commit(slots.k, kTail)
            cache?.commit(slots.v, vTail)
        } else {
            kRaw = silu(kRaw)
            vRaw = silu(vRaw)
        }
        let k = FastWeightMath.l2Normalize(kRaw.reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3))
        let v = vRaw.reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3).asType(.float32)
        let alpha = sigmoid(alphaProj(x).asType(.float32)).transposed(0, 2, 1)
        let eta = sigmoid(etaProj(x).asType(.float32)).transposed(0, 2, 1)
        let theta = (sigmoid(thetaProj(x).asType(.float32)) * config.baseLR).transposed(0, 2, 1)
        return (k, v, alpha, eta, theta)
    }
}

extension TitansNeuralMemory {
    package static func applyChunkUpdate(
        _ inputs: TitansChunkUpdateInputs,
        state: inout TitansMemoryState,
        completesChunk: Bool
    ) {
        let beta = Float(1.0) - inputs.alpha.asType(.float32)
        let trans = FastWeightMath.titansTransition(beta: beta, eta: inputs.eta.asType(.float32))
        let theta = inputs.theta.asType(.float32).expandedDimensions(axis: -1)
        let stepWeights = TitansStepWeights(mem: trans.uLast * theta, mom: trans.eLast * theta)
        let k = inputs.keys.asType(.float32)
        let v = inputs.values.asType(.float32)
        let z1 = matmul(k, state.base.w1) + state.base.b1
        if let baseW2 = state.base.w2, let baseB2 = state.base.b2,
           let curW2 = state.current.w2, let curB2 = state.current.b2,
           let momW2 = state.momentum.w2, let momB2 = state.momentum.b2 {
            let h1 = silu(z1)
            let delta2 = matmul(h1, baseW2) + baseB2 + k - v
            let delta1 = FastWeightMath.siluBackward(z: z1, gradOut: matmul(delta2, baseW2.swappedAxes(-1, -2)))
            (state.current.w1, state.momentum.w1) = advanceMatrix(cur: state.current.w1, mom: state.momentum.w1, act: k, delta: delta1, trans: trans, weights: stepWeights)
            (state.current.b1, state.momentum.b1) = advanceBias(cur: state.current.b1, mom: state.momentum.b1, delta: delta1, trans: trans, weights: stepWeights)
            let (nw2, nmw2) = advanceMatrix(cur: curW2, mom: momW2, act: h1, delta: delta2, trans: trans, weights: stepWeights)
            let (nb2, nmb2) = advanceBias(cur: curB2, mom: momB2, delta: delta2, trans: trans, weights: stepWeights)
            (state.current.w2, state.momentum.w2, state.current.b2, state.momentum.b2) = (nw2, nmw2, nb2, nmb2)
        } else {
            let delta1 = z1 + k - v
            (state.current.w1, state.momentum.w1) = advanceMatrix(cur: state.current.w1, mom: state.momentum.w1, act: k, delta: delta1, trans: trans, weights: stepWeights)
            (state.current.b1, state.momentum.b1) = advanceBias(cur: state.current.b1, mom: state.momentum.b1, delta: delta1, trans: trans, weights: stepWeights)
        }
        if completesChunk {
            state.base = state.current
        }
    }

    private static func advanceMatrix(
        cur: MLXArray,
        mom: MLXArray,
        act: MLXArray,
        delta: MLXArray,
        trans: TitansTransition,
        weights: TitansStepWeights
    ) -> (MLXArray, MLXArray) {
        let actT = act.swappedAxes(-1, -2)
        let gradMom = matmul(actT, delta * weights.mom)
        let gradMem = matmul(actT, delta * weights.mem)
        let nextMom = trans.pEta * mom - gradMom
        let nextCur = trans.pBeta * cur + trans.cS * mom - gradMem
        return (nextCur, nextMom)
    }

    private static func advanceBias(
        cur: MLXArray,
        mom: MLXArray,
        delta: MLXArray,
        trans: TitansTransition,
        weights: TitansStepWeights
    ) -> (MLXArray, MLXArray) {
        let gradMom = (delta * weights.mom).sum(axis: -2, keepDims: true)
        let gradMem = (delta * weights.mem).sum(axis: -2, keepDims: true)
        let nextMom = trans.pEta * mom - gradMom
        let nextCur = trans.pBeta * cur + trans.cS * mom - gradMem
        return (nextCur, nextMom)
    }
}

final class TitansSlidingAttention: Module {
    let config: TitansConfiguration
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ParameterInfo(key: "persistent_tokens") var persistentTokens: MLXArray?

    init(_ config: TitansConfiguration) {
        self.config = config
        self.scale = pow(Float(config.headDim), -0.5)
        self.rope = RoPE(dimensions: config.headDim, traditional: false, base: config.ropeTheta)
        let d = config.hiddenSize
        let kvDim = config.numKeyValueHeads * config.headDim
        _qProj.wrappedValue = Linear(d, d, bias: false)
        _kProj.wrappedValue = Linear(d, kvDim, bias: false)
        _vProj.wrappedValue = Linear(d, kvDim, bias: false)
        _oProj.wrappedValue = Linear(d, d, bias: false)
        if config.numPersistentTokens > 0 {
            _persistentTokens.wrappedValue = MLXRandom.normal([1, config.numPersistentTokens, d]) * Float(0.02)
        }
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
        guard let persistentTokens, config.numPersistentTokens > 0 else {
            let out = attentionWithCacheUpdate(queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask)
            return oProj(out.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
        }
        let (kWin, vWin) = cache?.update(keys: k, values: v) ?? (k, v)
        let p = broadcast(persistentTokens.asType(x.dtype), to: [batch, config.numPersistentTokens, config.hiddenSize])
        let kP = kProj(p).reshaped(batch, config.numPersistentTokens, config.numKeyValueHeads, config.headDim).transposed(0, 2, 1, 3)
        let vP = vProj(p).reshaped(batch, config.numPersistentTokens, config.numKeyValueHeads, config.headDim).transposed(0, 2, 1, 3)
        let kAll = concatenated([kP, kWin], axis: 2)
        let vAll = concatenated([vP, vWin], axis: 2)
        let fullMask = FastWeightMath.prependPersistentMask(
            mask,
            queryLength: length,
            keyLength: kWin.dim(2),
            persistentCount: config.numPersistentTokens
        )
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: kAll, values: vAll, scale: scale, mask: fullMask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class TitansBlock: Module {
    let config: TitansConfiguration
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_memory_layernorm") var postMemoryNorm: RMSNorm?
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: TitansSlidingAttention
    @ModuleInfo(key: "memory") var memory: TitansNeuralMemory
    @ModuleInfo(key: "mac_ctx_proj") var macCtxProj: Linear?
    @ModuleInfo(key: "mlp") var mlp: FastWeightSwiGLUMLP

    init(_ config: TitansConfiguration) {
        self.config = config
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postMemoryNorm.wrappedValue = config.variant == .mal ? RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps) : nil
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _selfAttn.wrappedValue = TitansSlidingAttention(config)
        _memory.wrappedValue = TitansNeuralMemory(config)
        _macCtxProj.wrappedValue = config.variant == .mac ? Linear(config.hiddenSize, config.hiddenSize, bias: false) : nil
        _mlp.wrappedValue = FastWeightSwiGLUMLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
    }

    func callAsFunction(_ x: MLXArray, swaCache: KVCache?, memCache: FastWeightLayerCache?) -> MLXArray {
        guard x.dim(1) > 0 else { return x }
        let mixed: MLXArray = switch config.variant {
        case .mac: forwardMAC(x, swaCache: swaCache, memCache: memCache)
        case .mag: forwardMAG(x, swaCache: swaCache, memCache: memCache)
        case .mal: forwardMAL(x, swaCache: swaCache, memCache: memCache)
        }
        return mixed + mlp(postAttentionNorm(mixed))
    }

    private func forwardMAG(_ x: MLXArray, swaCache: KVCache?, memCache: FastWeightLayerCache?) -> MLXArray {
        let normed = inputNorm(x)
        let mask = makeAttentionMask(n: x.dim(1), cache: swaCache, windowSize: config.windowSize)
        let attnOut = selfAttn(normed, mask: mask, cache: swaCache)
        let memOut = memory(normed, cache: memCache)
        return x + attnOut * sigmoid(memOut.asType(.float32)).asType(attnOut.dtype)
    }

    private func forwardMAL(_ x: MLXArray, swaCache: KVCache?, memCache: FastWeightLayerCache?) -> MLXArray {
        let afterMem = x + memory(inputNorm(x), cache: memCache)
        let attnIn = postMemoryNorm.map { $0(afterMem) } ?? afterMem
        let mask = makeAttentionMask(n: x.dim(1), cache: swaCache, windowSize: config.windowSize)
        return afterMem + selfAttn(attnIn, mask: mask, cache: swaCache)
    }

    private func forwardMAC(_ x: MLXArray, swaCache: KVCache?, memCache: FastWeightLayerCache?) -> MLXArray {
        let normed = inputNorm(x)
        let (slices, _) = FastWeightSchedule.slices(
            chunkSize: config.chunkSize,
            stepInChunk: memCache?.stepInChunk ?? 0,
            tokenCount: x.dim(1)
        )
        let activeSWA: KVCache = swaCache ?? RotatingKVCache(maxSize: config.windowSize, keep: 0)
        let activeMem = memCache ?? FastWeightLayerCache(
            chunkSize: config.chunkSize,
            slotCount: TitansMemoryState.slotCount(depth: config.memoryDepth)
        )
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(slices.count)
        for slice in slices {
            let xSlice = normed[0..., slice.range, 0...]
            let hist = memory.retrieve(xSlice, cache: activeMem, useConv: true)
            let attnIn = xSlice + (macCtxProj.map { $0(hist) } ?? hist)
            let mask = makeAttentionMask(n: slice.count, cache: activeSWA, windowSize: config.windowSize)
            let yAttn = selfAttn(attnIn, mask: mask, cache: activeSWA)
            let gate = memory.retrieve(yAttn, cache: activeMem, useConv: false)
            memory.updateSlice(yAttn, slice: slice, cache: activeMem)
            outputs.append(yAttn * sigmoid(gate.asType(.float32)).asType(yAttn.dtype))
        }
        return x + concatenated(outputs, axis: 1)
    }
}

final class TitansModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [TitansBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: TitansConfiguration) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in TitansBlock(config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        for (index, layer) in layers.enumerated() {
            let swaCache = cache.flatMap { 2 * index < $0.count ? $0[2 * index] : nil }
            let memCache = cache.flatMap { 2 * index + 1 < $0.count ? ($0[2 * index + 1] as? FastWeightLayerCache) : nil }
            h = layer(h, swaCache: swaCache, memCache: memCache)
        }
        return norm(h)
    }
}

public final class TitansModel: Module, LLMModel, LoRAModel {
    public static let modelTypes = ["titans", "titans_mac", "titans_mag", "titans_mal"]
    public let vocabularySize: Int
    public let configuration: TitansConfiguration
    @ModuleInfo(key: "model") var model: TitansModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: TitansConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabSize
        _model.wrappedValue = TitansModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public convenience init(configurationData data: Data) throws {
        self.init(try JSONDecoder.json5().decode(TitansConfiguration.self, from: data))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return lmHead.map { $0(hidden) } ?? model.embedTokens.asLinear(hidden)
    }

    public var loraLayers: [Module] { model.layers }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let slots = TitansMemoryState.slotCount(depth: configuration.memoryDepth)
        return (0 ..< configuration.numHiddenLayers).flatMap { _ -> [any KVCache] in
            [
                RotatingKVCache(maxSize: configuration.windowSize, keep: 0),
                FastWeightLayerCache(chunkSize: configuration.chunkSize, slotCount: slots),
            ]
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        TitansCheckpoint.sanitize(weights, config: configuration)
    }
}

enum TitansCheckpoint {
    private static let replacements: [(String, String)] = [
        (".neural_memory.", ".memory."),
        (".attn.", ".self_attn."),
        (".q_conv.weight", ".q_conv.conv.weight"),
        (".q_conv.bias", ".q_conv.conv.bias"),
        (".k_conv.weight", ".k_conv.conv.weight"),
        (".k_conv.bias", ".k_conv.conv.bias"),
        (".v_conv.weight", ".v_conv.conv.weight"),
        (".v_conv.bias", ".v_conv.conv.bias"),
    ]

    static func sanitize(_ weights: [String: MLXArray], config: TitansConfiguration) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (rawKey, value) in weights {
            if config.tieWordEmbeddings && rawKey.hasPrefix("lm_head.") { continue }
            if rawKey.contains("rotary_emb.inv_freq") { continue }
            var key = rawKey
            for (from, to) in replacements where key.contains(from) {
                key = key.replacingOccurrences(of: from, with: to)
            }
            if key.hasSuffix("_conv.conv.weight") {
                out[key] = FastWeightMath.normalizeConvWeight(value, kernelSize: config.convKernel)
            } else {
                out[key] = value
            }
        }
        return out
    }
}
