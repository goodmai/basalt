import Foundation
import MLX
import MLXLMCommon
import MLXNN

package struct BailingHybridKDAInputs {
    package let q: MLXArray
    package let k: MLXArray
    package let v: MLXArray
    package let decay: MLXArray
    package let beta: MLXArray

    package init(q: MLXArray, k: MLXArray, v: MLXArray, decay: MLXArray, beta: MLXArray) {
        self.q = q
        self.k = k
        self.v = v
        self.decay = decay
        self.beta = beta
    }

    func slice(at step: Int) -> BailingHybridKDAInputs {
        BailingHybridKDAInputs(
            q: q[0..., step],
            k: k[0..., step],
            v: v[0..., step],
            decay: decay[0..., step],
            beta: beta[0..., step]
        )
    }
}

package enum BailingHybridKDAOps {
    package static func decayLogGate(
        f: MLXArray,
        aLog: MLXArray,
        dtBias: MLXArray,
        safeGate: Bool,
        lowerBound: Float
    ) -> MLXArray {
        let pre = f.asType(.float32) + dtBias.asType(.float32)
        let alpha = exp(aLog.asType(.float32)).expandedDimensions(axis: -1)
        if safeGate {
            return MLXArray(lowerBound, dtype: .float32) * sigmoid(alpha * pre)
        }
        return -alpha * softplus(pre)
    }

    package static func l2Normalize(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        let xf = x.asType(.float32)
        let norm = sqrt((xf * xf).sum(axis: -1, keepDims: true))
        return xf / (norm + eps)
    }

    package static func recurrentUpdate(
        _ inputs: BailingHybridKDAInputs,
        state: MLXArray?,
        mask: MLXArray? = nil,
        useKernel: Bool = true
    ) -> (MLXArray, MLXArray) {
        if Device.defaultDevice() == .gpu,
           [inputs.q, inputs.k, inputs.v, inputs.decay, inputs.beta, state, mask].contains(where: { $0?.dtype == .float64 }) {
            return Device.withDefaultDevice(.cpu) {
                recurrentUpdate(inputs, state: state, mask: mask, useKernel: useKernel)
            }
        }
        if useKernel, BailingHybridKDAKernel.canUseKernel(inputs, state: state, mask: mask) {
            return BailingHybridKDAKernel.run(inputs, state: state, mask: mask)
        }
        return recurrentUpdateOps(inputs, state: state, mask: mask)
    }

    package static func recurrentUpdateOps(
        _ inputs: BailingHybridKDAInputs,
        state: MLXArray?,
        mask: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let batch = inputs.q.dim(0)
        let steps = inputs.q.dim(1)
        let hk = inputs.q.dim(2)
        let hv = inputs.v.dim(2)
        let headVDim = inputs.v.dim(3)
        let headKDim = inputs.k.dim(3)
        var current = (state ?? MLXArray.zeros([batch, hv, headVDim, headKDim], dtype: .float32)).asType(.float32)
        guard steps > 0 else {
            return (MLXArray.zeros([batch, 0, hv, headVDim], dtype: inputs.q.dtype), current)
        }
        let repeatFactor = hv / max(hk, 1)
        let effective = repeatFactor > 1
            ? BailingHybridKDAInputs(
                q: repeated(inputs.q, count: repeatFactor, axis: 2),
                k: repeated(inputs.k, count: repeatFactor, axis: 2),
                v: inputs.v,
                decay: inputs.decay,
                beta: inputs.beta
            )
            : inputs
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(steps)
        for step in 0 ..< steps {
            let stepMask = mask.map { $0[0..., step] }
            let (y, nextState) = recurrentStep(effective.slice(at: step), state: current, mask: stepMask)
            outputs.append(y)
            current = nextState
        }
        return (MLX.stacked(outputs, axis: 1).asType(inputs.q.dtype), current)
    }

    private static func recurrentStep(
        _ step: BailingHybridKDAInputs,
        state: MLXArray,
        mask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let previous = state
        let g = step.decay.ndim == 2
            ? step.decay.expandedDimensions(axes: [2, 3])
            : step.decay.expandedDimensions(axis: -2)
        var updated = state * g.asType(.float32)
        let kExpanded = step.k.asType(.float32).expandedDimensions(axis: -2)
        let kvMem = (updated * kExpanded).sum(axis: -1)
        let delta = (step.v.asType(.float32) - kvMem) * step.beta.asType(.float32).expandedDimensions(axis: -1)
        updated = updated + kExpanded * delta.expandedDimensions(axis: -1)
        var y = (updated * step.q.asType(.float32).expandedDimensions(axis: -2)).sum(axis: -1)
        if let mask {
            y = MLX.where(mask.expandedDimensions(axes: [1, 2]), y, MLXArray.zeros(like: y))
            updated = MLX.where(mask.expandedDimensions(axes: [1, 2, 3]), updated, previous)
        }
        return (y, updated)
    }
}

package final class BailingHybridShortConv1d: Module {
    let kernelSize: Int
    @ModuleInfo(key: "conv") var conv: Conv1d

    package init(channels: Int, kernelSize: Int) {
        self.kernelSize = kernelSize
        self._conv.wrappedValue = Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: kernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: channels,
            bias: false
        )
    }

    package func callAsFunction(_ x: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        let batch = x.dim(0)
        let steps = x.dim(1)
        let channels = x.dim(2)
        let history = max(kernelSize - 1, 0)
        let previous = state ?? MLXArray.zeros([batch, history, channels], dtype: x.dtype)
        guard steps > 0 else {
            return (MLXArray.zeros([batch, 0, channels], dtype: x.dtype), previous)
        }
        let convInput = concatenated([previous, x], axis: 1)
        let out = silu(conv(convInput))
        let newState: MLXArray
        if history == 0 {
            newState = MLXArray.zeros([batch, 0, channels], dtype: x.dtype)
        } else {
            let total = convInput.dim(1)
            newState = contiguous(convInput[0..., (total - history)..., 0...])
        }
        return (out, newState)
    }
}

package final class BailingHybridLinearAttention: Module {
    let layerIdx: Int
    let numHeads: Int
    let kvHeads: Int
    let headDim: Int
    let safeGate: Bool
    let lowerBound: Float
    let noKDALoRA: Bool
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "q_conv1d") var qConv1d: BailingHybridShortConv1d
    @ModuleInfo(key: "k_conv1d") var kConv1d: BailingHybridShortConv1d
    @ModuleInfo(key: "v_conv1d") var vConv1d: BailingHybridShortConv1d
    @ModuleInfo(key: "f_proj") var fProj: Linear?
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    @ModuleInfo(key: "f_a_proj") var fAProj: Linear?
    @ModuleInfo(key: "f_b_proj") var fBProj: Linear?
    @ModuleInfo(key: "g_a_proj") var gAProj: Linear?
    @ModuleInfo(key: "g_b_proj") var gBProj: Linear?
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ModuleInfo(key: "o_norm") var oNorm: RMSNorm
    @ModuleInfo(key: "o_proj") var oProj: Linear

    package init(_ config: BailingHybridConfiguration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.numHeads = config.numAttentionHeads
        self.kvHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.safeGate = config.kdaSafeGate
        self.lowerBound = config.kdaLowerBound
        self.noKDALoRA = config.noKDALoRA
        self.scale = pow(Float(config.headDim), -0.5)

        let hidden = config.hiddenSize
        let projDim = config.numAttentionHeads * config.headDim
        self._qProj.wrappedValue = Linear(hidden, projDim, bias: false)
        self._kProj.wrappedValue = Linear(hidden, projDim, bias: false)
        self._vProj.wrappedValue = Linear(hidden, projDim, bias: false)
        self._qConv1d.wrappedValue = BailingHybridShortConv1d(channels: projDim, kernelSize: config.convKernelSize)
        self._kConv1d.wrappedValue = BailingHybridShortConv1d(channels: projDim, kernelSize: config.convKernelSize)
        self._vConv1d.wrappedValue = BailingHybridShortConv1d(channels: projDim, kernelSize: config.convKernelSize)

        if config.noKDALoRA {
            self._fProj.wrappedValue = Linear(hidden, projDim, bias: false)
            self._gProj.wrappedValue = Linear(hidden, projDim, bias: false)
        } else {
            self._fAProj.wrappedValue = Linear(hidden, config.headDim, bias: false)
            self._fBProj.wrappedValue = Linear(config.headDim, projDim, bias: false)
            self._gAProj.wrappedValue = Linear(hidden, config.headDim, bias: false)
            self._gBProj.wrappedValue = Linear(config.headDim, projDim, bias: false)
        }

        self._bProj.wrappedValue = Linear(hidden, config.numAttentionHeads, bias: false)
        self._aLog.wrappedValue = MLXArray.zeros([config.numAttentionHeads], dtype: .float32)
        self._dtBias.wrappedValue = MLXArray.zeros([projDim], dtype: .float32)
        self._oNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        self._oProj.wrappedValue = Linear(projDim, hidden, bias: false)
    }

    package func callAsFunction(
        _ x: MLXArray,
        cache: ArraysCache?,
        probe: (any ForwardProbe)? = nil
    ) -> MLXArray {
        let batch = x.dim(0)
        let steps = x.dim(1)
        let dtype = x.dtype
        let ssmMask = cache?.makeMask(N: steps)

        let (qConv, kConv, vConv) = convolveQKV(x, cache: cache, mask: ssmMask)
        let q = BailingHybridKDAOps.l2Normalize(qConv.reshaped(batch, steps, numHeads, headDim)) * scale
        let k = BailingHybridKDAOps.l2Normalize(kConv.reshaped(batch, steps, numHeads, headDim))
        let v = vConv.reshaped(batch, steps, numHeads, headDim).asType(.float32)

        let (fRaw, gRaw) = projectGates(x)
        let f = fRaw.reshaped(batch, steps, numHeads, headDim)
        let gLog = BailingHybridKDAOps.decayLogGate(
            f: f,
            aLog: aLog,
            dtBias: dtBias.reshaped(numHeads, headDim),
            safeGate: safeGate,
            lowerBound: lowerBound
        )
        let beta = sigmoid(bProj(x).asType(.float32))
        let inputs = BailingHybridKDAInputs(q: q, k: k, v: v, decay: exp(gLog), beta: beta)
        let (recurrenceOut, nextState) = BailingHybridKDAOps.recurrentUpdate(
            inputs, state: cache?[3], mask: ssmMask, useKernel: !training)
        if let cache {
            cache[3] = nextState
            cache.advance(steps)
        }

        let gate = sigmoid(gRaw.reshaped(batch, steps, numHeads, headDim).asType(.float32)).asType(dtype)
        var gated = oNorm(recurrenceOut.asType(dtype)) * gate
        if let probe {
            var perHead = gated.transposed(0, 2, 1, 3)
            if let headMask = probe.headMask(layer: layerIdx) {
                perHead = perHead * headMask.reshaped(1, numHeads, 1, 1).asType(perHead.dtype)
            }
            probe.headOutputs(layer: layerIdx, perHead)
            gated = perHead.transposed(0, 2, 1, 3)
        }
        return oProj(gated.reshaped(batch, steps, -1))
    }

    private func convolveQKV(_ x: MLXArray, cache: ArraysCache?, mask: MLXArray?) -> (MLXArray, MLXArray, MLXArray) {
        var qIn = qProj(x)
        var kIn = kProj(x)
        var vIn = vProj(x)
        if let mask {
            let expanded = mask.expandedDimensions(axis: -1)
            qIn = MLX.where(expanded, qIn, MLXArray.zeros(like: qIn))
            kIn = MLX.where(expanded, kIn, MLXArray.zeros(like: kIn))
            vIn = MLX.where(expanded, vIn, MLXArray.zeros(like: vIn))
        }
        let (qOut, qState) = qConv1d(qIn, state: cache?[0])
        let (kOut, kState) = kConv1d(kIn, state: cache?[1])
        let (vOut, vState) = vConv1d(vIn, state: cache?[2])
        if let cache {
            cache[0] = qState
            cache[1] = kState
            cache[2] = vState
        }
        return (qOut, kOut, vOut)
    }

    private func projectGates(_ x: MLXArray) -> (MLXArray, MLXArray) {
        if noKDALoRA, let fProj, let gProj {
            return (fProj(x), gProj(x))
        }
        guard let fAProj, let fBProj, let gAProj, let gBProj else {
            preconditionFailure("BailingHybridLinearAttention gate projections not initialized")
        }
        return (fBProj(fAProj(x)), gBProj(gAProj(x)))
    }
}
