import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

package final class Mamba3Mixer: Module {
    let layerIdx: Int
    let numHeads: Int
    let headDim: Int
    let dInner: Int
    let dState: Int
    let nGroups: Int
    let mimoRank: Int
    let chunkSize: Int
    let heavyTail: Bool
    let aFloor: Float
    let numRopeAngles: Int

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm?
    @ModuleInfo(key: "B_norm") var bNorm: RMSNorm
    @ModuleInfo(key: "C_norm") var cNorm: RMSNorm
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "D") var dSkip: MLXArray
    @ParameterInfo(key: "B_bias") var bBias: MLXArray
    @ParameterInfo(key: "C_bias") var cBias: MLXArray
    @ParameterInfo(key: "mimo_x") var mimoX: MLXArray?
    @ParameterInfo(key: "mimo_z") var mimoZ: MLXArray?
    @ParameterInfo(key: "mimo_o") var mimoO: MLXArray?

    package init(_ config: Mamba3Configuration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.numHeads = config.numHeads
        self.headDim = config.headDim
        self.dInner = config.dInner
        self.dState = config.dState
        self.nGroups = config.nGroups
        self.mimoRank = config.mimoRank
        self.chunkSize = config.chunkSize
        self.heavyTail = config.heavyTail
        self.aFloor = config.aFloor
        self.numRopeAngles = config.numRopeAngles

        self._inProj.wrappedValue = Linear(config.hiddenSize, config.inProjDim, bias: false)
        self._outProj.wrappedValue = Linear(config.dInner, config.hiddenSize, bias: false)
        if config.isOutprojNorm {
            self._norm.wrappedValue = RMSNorm(dimensions: config.dInner, eps: config.rmsNormEps)
        }
        self._bNorm.wrappedValue = RMSNorm(dimensions: config.dState, eps: config.rmsNormEps)
        self._cNorm.wrappedValue = RMSNorm(dimensions: config.dState, eps: config.rmsNormEps)
        self._dtBias.wrappedValue = MLXArray.zeros([config.numHeads], dtype: .float32)
        self._dSkip.wrappedValue = MLXArray.ones([config.numHeads], dtype: .float32)
        self._bBias.wrappedValue = MLXArray.ones([config.numHeads, config.mimoRank, config.dState], dtype: .float32)
        self._cBias.wrappedValue = MLXArray.ones([config.numHeads, config.mimoRank, config.dState], dtype: .float32)
        if config.isMIMO {
            let invRank = Float(1.0) / Float(config.mimoRank)
            self._mimoX.wrappedValue = MLXArray.ones([config.numHeads, config.mimoRank, config.headDim], dtype: .float32) * invRank
            self._mimoZ.wrappedValue = MLXArray.ones([config.numHeads, config.mimoRank, config.headDim], dtype: .float32) * invRank
            self._mimoO.wrappedValue = MLXArray.ones([config.numHeads, config.mimoRank, config.headDim], dtype: .float32) * invRank
        }
    }

    package func callAsFunction(
        _ x: MLXArray,
        cache: ArraysCache?,
        useChunked: Bool = true
    ) -> MLXArray {
        let (batch, steps, dtype) = (x.dim(0), x.dim(1), x.dtype)
        guard steps > 0 else {
            return MLXArray.zeros([batch, 0, x.dim(2)], dtype: dtype)
        }
        let ssmMask = cache?.makeMask(N: steps)
        let proj = splitProjection(inProj(x), batch: batch, steps: steps)

        let dt = softplus(proj.ddDt.asType(.float32) + dtBias.asType(.float32))
        let aScale = heavyTail ? Mamba3Ops.heavyTail(proj.ddA) : softplus(proj.ddA.asType(.float32))
        let aNeg = minimum(-aScale, MLXArray(-aFloor, dtype: .float32))
        let logAlpha = aNeg * dt
        let trap = sigmoid(proj.trapRaw.asType(.float32))

        let kPre = prepareBC(proj.bRaw, norm: bNorm, bias: bBias)
        let qPre = prepareBC(proj.cRaw, norm: cNorm, bias: cBias)
        let vIn = expandMIMOValue(proj.xIn.asType(.float32))

        let recurrenceInputs = Mamba3RecurrenceInputs(
            q: qPre, k: kPre, v: vIn, dt: dt, logAlpha: logAlpha, trap: trap
        ).withAngles(proj.anglesRaw.asType(.float32), splitHalfRoPE: mimoRank > 1)

        let initialState = readCacheState(cache)
        let (ssmOut, nextState) = Mamba3Ops.forward(
            recurrenceInputs,
            state: initialState,
            mask: ssmMask,
            chunkSize: useChunked ? chunkSize : 1
        )
        commitCacheState(nextState, steps: steps, to: cache)

        var withSkip = ssmOut + dSkip.asType(.float32).reshaped([1, 1, numHeads, 1, 1]) * vIn
        if let ssmMask {
            withSkip = MLX.where(ssmMask.expandedDimensions(axes: [2, 3, 4]), withSkip, MLXArray.zeros(like: withSkip))
        }
        let gated = gateAndReduce(withSkip, z: proj.z.asType(.float32))
        return outProj(gated.reshaped(batch, steps, dInner).asType(dtype))
    }

    private struct SplitProjection {
        let z: MLXArray
        let xIn: MLXArray
        let bRaw: MLXArray
        let cRaw: MLXArray
        let ddDt: MLXArray
        let ddA: MLXArray
        let trapRaw: MLXArray
        let anglesRaw: MLXArray
    }

    private func splitProjection(_ projected: MLXArray, batch: Int, steps: Int) -> SplitProjection {
        let bcWidth = nGroups * mimoRank * dState
        let o1 = dInner
        let o2 = o1 + dInner
        let o3 = o2 + bcWidth
        let o4 = o3 + bcWidth
        let o5 = o4 + numHeads
        let o6 = o5 + numHeads
        let o7 = o6 + numHeads
        let o8 = o7 + numRopeAngles
        let sharedAngles = numRopeAngles > 0
            ? broadcast(projected[0..., 0..., o7 ..< o8].expandedDimensions(axis: 2), to: [batch, steps, numHeads, numRopeAngles])
            : MLXArray.zeros([batch, steps, numHeads, 0], dtype: projected.dtype)
        return SplitProjection(
            z: projected[0..., 0..., ..<o1].reshaped(batch, steps, numHeads, headDim),
            xIn: projected[0..., 0..., o1 ..< o2].reshaped(batch, steps, numHeads, headDim),
            bRaw: projected[0..., 0..., o2 ..< o3].reshaped(batch, steps, nGroups, mimoRank, dState),
            cRaw: projected[0..., 0..., o3 ..< o4].reshaped(batch, steps, nGroups, mimoRank, dState),
            ddDt: projected[0..., 0..., o4 ..< o5],
            ddA: projected[0..., 0..., o5 ..< o6],
            trapRaw: projected[0..., 0..., o6 ..< o7],
            anglesRaw: sharedAngles
        )
    }

    private func prepareBC(_ raw: MLXArray, norm: RMSNorm, bias: MLXArray) -> MLXArray {
        var normalized = norm(raw).asType(.float32)
        let repeatFactor = numHeads / max(nGroups, 1)
        if repeatFactor > 1 {
            normalized = repeated(normalized, count: repeatFactor, axis: 2)
        }
        return normalized + bias.asType(.float32).reshaped(1, 1, numHeads, mimoRank, dState)
    }

    private func expandMIMOValue(_ xIn: MLXArray) -> MLXArray {
        if mimoRank > 1, let mimoX {
            return xIn.expandedDimensions(axis: 3) * mimoX.asType(.float32).reshaped(1, 1, numHeads, mimoRank, headDim)
        }
        return xIn.expandedDimensions(axis: 3)
    }

    private func gateAndReduce(_ y: MLXArray, z: MLXArray) -> MLXArray {
        var values = y
        if let norm {
            let variance = mean(values * values, axis: -1, keepDims: true)
            let weight = norm.weight.asType(.float32).reshaped(1, 1, numHeads, 1, headDim)
            values = values * rsqrt(variance + MLXArray(norm.eps, dtype: .float32)) * weight
        }
        if mimoRank > 1, let mimoZ, let mimoO {
            let zExpanded = z.expandedDimensions(axis: 3) * mimoZ.asType(.float32).reshaped(1, 1, numHeads, mimoRank, headDim)
            let gated = values * silu(zExpanded)
            return (gated * mimoO.asType(.float32).reshaped(1, 1, numHeads, mimoRank, headDim)).sum(axis: 3)
        }
        return values.squeezed(axis: 3) * silu(z)
    }

    private func readCacheState(_ cache: ArraysCache?) -> Mamba3State? {
        guard let cache,
              let angle = cache[0],
              let ssm = cache[1],
              let k = cache[2],
              let v = cache[3]
        else { return nil }
        return Mamba3State(angleState: angle, ssmState: ssm, kState: k, vState: v)
    }

    private func commitCacheState(_ state: Mamba3State, steps: Int, to cache: ArraysCache?) {
        guard let cache else { return }
        let detached = state.detached()
        cache[0] = detached.angleState
        cache[1] = detached.ssmState
        cache[2] = detached.kState
        cache[3] = detached.vState
        cache.offset += steps
        cache.advance(steps)
    }
}

package final class Mamba3GatedMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    package init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(upProj(x) * silu(gateProj(x)))
    }
}
