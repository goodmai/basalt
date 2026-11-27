import Foundation
import MLX
import MLXNN

public final class XLSTMRMSNorm: Module, UnaryLayer {
    let eps: Float
    let forceFloat32: Bool
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?

    public init(dimensions: Int, eps: Float = 1e-6, useBias: Bool = false, forceFloat32: Bool = true) {
        self.eps = eps
        self.forceFloat32 = forceFloat32
        _weight.wrappedValue = MLXArray.ones([dimensions], dtype: .float32)
        if useBias {
            _bias.wrappedValue = MLXArray.zeros([dimensions], dtype: .float32)
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let inDType = x.dtype
        let work = forceFloat32 ? x.asType(.float32) : x
        let normed = (work * rsqrt(mean(work * work, axis: -1, keepDims: true) + eps)).asType(inDType)
        let scaled = normed * weight.asType(inDType)
        return bias.map { scaled + $0.asType(inDType) } ?? scaled
    }
}

public final class MultiHeadLayerNorm: Module {
    let numHeads: Int
    let headDim: Int
    let eps: Float
    let forceFloat32: Bool
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?

    public init(numHeads: Int, headDim: Int, eps: Float = 1e-6, useBias: Bool = false, forceFloat32: Bool = true) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.eps = eps
        self.forceFloat32 = forceFloat32
        let features = numHeads * headDim
        _weight.wrappedValue = MLXArray.ones([features], dtype: .float32)
        if useBias {
            _bias.wrappedValue = MLXArray.zeros([features], dtype: .float32)
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let inDType = x.dtype
        let work = forceFloat32 ? x.asType(.float32) : x
        let centered = work - mean(work, axis: -1, keepDims: true)
        let variance = mean(centered * centered, axis: -1, keepDims: true)
        let normed = (centered * rsqrt(variance + eps)).asType(inDType).reshaped(b, l, numHeads * headDim)
        let scaled = normed * weight.asType(inDType)
        return bias.map { scaled + $0.asType(inDType) } ?? scaled
    }
}

public struct MLSTMState {
    public let c: MLXArray
    public let n: MLXArray
    public let m: MLXArray

    public init(c: MLXArray, n: MLXArray, m: MLXArray) {
        self.c = c
        self.n = n
        self.m = m
    }
}

public struct SLSTMState {
    public let c: MLXArray
    public let n: MLXArray
    public let m: MLXArray
    public let h: MLXArray

    public init(c: MLXArray, n: MLXArray, m: MLXArray, h: MLXArray) {
        self.c = c
        self.n = n
        self.m = m
        self.h = h
    }
}

struct MLSTMInputs {
    let q: MLXArray
    let k: MLXArray
    let v: MLXArray
    let iGate: MLXArray
    let fGate: MLXArray
}

struct MLSTMChunkTensors {
    let qC: MLXArray
    let kC: MLXArray
    let vC: MLXArray
    let vecI: MLXArray
    let vecB: MLXArray
    let cStarts: MLXArray
    let nStarts: MLXArray
    let mStarts: MLXArray
}

struct SLSTMInputs {
    let zIn: MLXArray
    let iIn: MLXArray
    let fIn: MLXArray
    let oIn: MLXArray
}

struct SLSTMRecurrentWeights {
    let rZ: MLXArray
    let rI: MLXArray
    let rF: MLXArray
    let rO: MLXArray
}

public enum XLSTMOps {
    public static func softCap(_ x: MLXArray, cap: Float?) -> MLXArray {
        guard let cap, cap > 0 else { return x }
        let capArray = MLXArray(cap, dtype: x.dtype)
        return capArray * tanh(x / capArray)
    }

    static func logSigmoid(_ x: MLXArray) -> MLXArray {
        -logAddExp(MLXArray(Float(0), dtype: .float32), -x)
    }

    static func mlstmForward(
        _ inputs: MLSTMInputs,
        state: MLSTMState?,
        chunkSize: Int,
        eps: Float
    ) -> (output: MLXArray, state: MLSTMState) {
        let (b, h, l, dQK) = (inputs.q.dim(0), inputs.q.dim(1), inputs.q.dim(2), inputs.q.dim(3))
        let dV = inputs.v.dim(3)
        let initial = initialMLSTMState(state, batch: b, heads: h, qkHeadDim: dQK, vHeadDim: dV)
        if l == 0 {
            return (MLXArray.zeros([b, h, 0, dV], dtype: inputs.q.dtype), initial)
        }
        if l == 1 {
            return mlstmStep(inputs, state: initial, eps: eps)
        }
        return mlstmChunkwise(inputs, state: initial, chunkSize: max(1, chunkSize), eps: eps)
    }

    static func mlstmStep(
        _ inputs: MLSTMInputs,
        state: MLSTMState,
        eps: Float
    ) -> (output: MLXArray, state: MLSTMState) {
        let q = inputs.q.squeezed(axis: 2).asType(.float32)
        let k = inputs.k.squeezed(axis: 2).asType(.float32)
        let v = inputs.v.squeezed(axis: 2).asType(.float32)
        let iGate = inputs.iGate.asType(.float32)
        let fLog = logSigmoid(inputs.fGate.asType(.float32))
        let mNext = maximum(fLog + state.m, iGate)
        let fAct = exp(fLog + state.m - mNext)
        let iAct = exp(iGate - mNext)
        let kv = matmul(k.expandedDimensions(axis: -1), v.expandedDimensions(axis: -2))
        let cNext = fAct.expandedDimensions(axis: -1) * state.c + iAct.expandedDimensions(axis: -1) * kv
        let nNext = fAct * state.n + iAct * k
        let qScale = MLXArray(Float(1.0) / Float(q.dim(-1)).squareRoot(), dtype: .float32)
        let qScaled = q * qScale
        let hNum = matmul(qScaled.expandedDimensions(axis: -2), cNext).squeezed(axis: -2)
        let qnDot = (qScaled * nNext).sum(axis: -1, keepDims: true)
        let denom = maximum(MLX.abs(qnDot), exp(-mNext)) + MLXArray(eps, dtype: .float32)
        let hOut = (hNum / denom).expandedDimensions(axis: 2).asType(inputs.q.dtype)
        return (hOut, MLSTMState(c: cNext, n: nNext, m: mNext))
    }

    static func mlstmChunkwise(
        _ inputs: MLSTMInputs,
        state: MLSTMState,
        chunkSize: Int,
        eps: Float
    ) -> (output: MLXArray, state: MLSTMState) {
        let (b, h, l, dQK) = (inputs.q.dim(0), inputs.q.dim(1), inputs.q.dim(2), inputs.q.dim(3))
        let dV = inputs.v.dim(3)
        let rem = l % chunkSize
        let pad = rem == 0 ? 0 : (chunkSize - rem)
        let paddedLen = l + pad
        let numChunks = paddedLen / chunkSize
        var qF = inputs.q.asType(.float32)
        var kF = inputs.k.asType(.float32)
        var vF = inputs.v.asType(.float32)
        var iF = inputs.iGate.asType(.float32)
        var fLog = logSigmoid(inputs.fGate.asType(.float32))
        if pad > 0 {
            let widths4 = [IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((0, pad)), IntOrPair((0, 0))]
            qF = MLX.padded(qF, widths: widths4)
            kF = MLX.padded(kF, widths: widths4)
            vF = MLX.padded(vF, widths: widths4)
            fLog = concatenated([fLog, MLXArray.zeros([b, h, pad], dtype: .float32)], axis: 2)
            iF = concatenated([iF, MLXArray.full([b, h, pad], values: MLXArray(-Float.infinity, dtype: .float32))], axis: 2)
        }
        let qC = qF.reshaped(b, h, numChunks, chunkSize, dQK)
        let kC = kF.reshaped(b, h, numChunks, chunkSize, dQK)
        let vC = vF.reshaped(b, h, numChunks, chunkSize, dV)
        let vecI = iF.reshaped(b, h, numChunks, chunkSize)
        let vecB = cumsum(fLog.reshaped(b, h, numChunks, chunkSize), axis: -1)
        let (cStarts, nStarts, mStarts, finalState) = mlstmInterChunkStates(
            kChunks: kC, vChunks: vC, vecB: vecB, vecI: vecI, initial: state
        )
        let tensors = MLSTMChunkTensors(
            qC: qC, kC: kC, vC: vC, vecI: vecI, vecB: vecB,
            cStarts: cStarts, nStarts: nStarts, mStarts: mStarts
        )
        let hFlat = mlstmParallelChunkOutputs(tensors, eps: eps).reshaped(b, h, paddedLen, dV)
        let hOut = (pad > 0 ? hFlat[0..., 0..., ..<l, 0...] : hFlat).asType(inputs.q.dtype)
        return (hOut, finalState)
    }

    static func slstmRecurrence(
        _ inputs: SLSTMInputs,
        weights: SLSTMRecurrentWeights,
        state: SLSTMState?,
        config: XLSTMConfiguration
    ) -> (output: MLXArray, state: SLSTMState) {
        let (b, l, h, dH) = (inputs.zIn.dim(0), inputs.zIn.dim(1), inputs.zIn.dim(2), inputs.zIn.dim(3))
        var current = initialSLSTMState(state, batch: b, heads: h, headDim: dH)
        guard l > 0 else {
            return (MLXArray.zeros([b, 0, h, dH], dtype: inputs.zIn.dtype), current)
        }
        let zF = inputs.zIn.asType(.float32)
        let iF = inputs.iIn.asType(.float32)
        let fF = inputs.fIn.asType(.float32)
        let oF = inputs.oIn.asType(.float32)
        let rZ = weights.rZ.asType(.float32).expandedDimensions(axis: 0)
        let rI = weights.rI.asType(.float32).expandedDimensions(axis: 0)
        let rF = weights.rF.asType(.float32).expandedDimensions(axis: 0)
        let rO = weights.rO.asType(.float32).expandedDimensions(axis: 0)
        let epsArr = MLXArray(config.eps, dtype: .float32)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(l)
        for step in 0 ..< l {
            let hRow = current.h.expandedDimensions(axis: -2)
            let zTilde = zF[0..., step, 0..., 0...] + matmul(hRow, rZ).squeezed(axis: -2)
            let iTilde = softCap(iF[0..., step, 0..., 0...] + matmul(hRow, rI).squeezed(axis: -2), cap: config.gateSoftCap)
            let fTilde = softCap(fF[0..., step, 0..., 0...] + matmul(hRow, rF).squeezed(axis: -2), cap: config.gateSoftCap)
            let oTilde = oF[0..., step, 0..., 0...] + matmul(hRow, rO).squeezed(axis: -2)
            let logF = config.slstmGateMode == .logSigmoid ? logSigmoid(fTilde) : fTilde
            let mNext = maximum(logF + current.m, iTilde)
            let iAct = exp(iTilde - mNext)
            let fAct = exp(logF + current.m - mNext)
            let cNext = fAct * current.c + iAct * tanh(zTilde)
            let nNext = fAct * current.n + iAct
            let denom = nNext + epsArr
            let hNext = sigmoid(oTilde) * (cNext / denom)
            current = SLSTMState(c: cNext, n: nNext, m: mNext, h: hNext)
            outputs.append(hNext)
        }
        return (MLX.stacked(outputs, axis: 1).asType(inputs.zIn.dtype), current)
    }
}

extension XLSTMOps {
    private static func mlstmInterChunkStates(
        kChunks: MLXArray,
        vChunks: MLXArray,
        vecB: MLXArray,
        vecI: MLXArray,
        initial: MLSTMState
    ) -> (cStarts: MLXArray, nStarts: MLXArray, mStarts: MLXArray, finalState: MLSTMState) {
        let numChunks = vecB.dim(2)
        let chunkSize = vecB.dim(3)
        let lastB = vecB[0..., 0..., 0..., (chunkSize - 1)...]
        let scaG = lastB.squeezed(axis: -1)
        let vecA = lastB - vecB + vecI
        let scaAMax = vecA.max(axis: -1)
        var cK = initial.c
        var nK = initial.n
        var mK = initial.m.squeezed(axis: -1)
        var cList: [MLXArray] = []
        var nList: [MLXArray] = []
        var mList: [MLXArray] = []
        cList.reserveCapacity(numChunks)
        nList.reserveCapacity(numChunks)
        mList.reserveCapacity(numChunks)
        for chunk in 0 ..< numChunks {
            cList.append(cK)
            nList.append(nK)
            mList.append(mK)
            let gK = scaG[0..., 0..., chunk]
            let mNext = maximum(gK + mK, scaAMax[0..., 0..., chunk])
            let aBarK = exp(vecA[0..., 0..., chunk, 0...] - mNext.expandedDimensions(axis: -1)).expandedDimensions(axis: -1)
            let kGated = kChunks[0..., 0..., chunk, 0..., 0...] * aBarK
            let vChunk = vChunks[0..., 0..., chunk, 0..., 0...]
            let gBarK = exp(gK + mK - mNext).expandedDimensions(axis: -1)
            cK = gBarK.expandedDimensions(axis: -1) * cK + matmul(kGated.swappedAxes(-2, -1), vChunk)
            nK = gBarK * nK + kGated.sum(axis: -2)
            mK = mNext
        }
        let finalState = MLSTMState(c: cK, n: nK, m: mK.expandedDimensions(axis: -1))
        return (MLX.stacked(cList, axis: 2), MLX.stacked(nList, axis: 2), MLX.stacked(mList, axis: 2), finalState)
    }

    private static func mlstmParallelChunkOutputs(_ t: MLSTMChunkTensors, eps: Float) -> MLXArray {
        let chunkSize = t.qC.dim(3)
        let dQK = t.qC.dim(4)
        let idx = MLXArray(0 ..< chunkSize)
        let ltr = idx.expandedDimensions(axis: 0) .<= idx.expandedDimensions(axis: 1)
        let fDiff = t.vecB.expandedDimensions(axis: -1) - t.vecB.expandedDimensions(axis: -2)
        let rawLogD = fDiff + t.vecI.expandedDimensions(axis: -2)
        let logD = MLX.where(ltr, rawLogD, MLXArray(-Float.infinity, dtype: .float32))
        let mIntra = logD.max(axis: -1)
        let mInter = t.vecB + t.mStarts.expandedDimensions(axis: -1)
        let mCombine = maximum(mInter, mIntra).expandedDimensions(axis: -1)
        let mInterExp = mInter.expandedDimensions(axis: -1)
        let matD = exp(logD - mCombine)
        let qScale = MLXArray(Float(1.0) / Float(dQK).squareRoot(), dtype: .float32)
        let matM = matmul(t.qC, t.kC.swappedAxes(-2, -1)) * qScale * matD
        let bBar = exp(mInterExp - mCombine)
        let qGated = t.qC * bBar * qScale
        let numerator = matmul(qGated, t.cStarts) + matmul(matM, t.vC)
        let denomLinear = matmul(qGated, t.nStarts.expandedDimensions(axis: -1)) + matM.sum(axis: -1, keepDims: true)
        let denom = maximum(MLX.abs(denomLinear), exp(-mCombine)) + MLXArray(eps, dtype: .float32)
        return numerator / denom
    }

    private static func initialMLSTMState(
        _ state: MLSTMState?,
        batch: Int,
        heads: Int,
        qkHeadDim: Int,
        vHeadDim: Int
    ) -> MLSTMState {
        guard let state else {
            return MLSTMState(
                c: MLXArray.zeros([batch, heads, qkHeadDim, vHeadDim], dtype: .float32),
                n: MLXArray.zeros([batch, heads, qkHeadDim], dtype: .float32),
                m: MLXArray.zeros([batch, heads, 1], dtype: .float32)
            )
        }
        return MLSTMState(c: state.c.asType(.float32), n: state.n.asType(.float32), m: state.m.asType(.float32))
    }

    private static func initialSLSTMState(
        _ state: SLSTMState?,
        batch: Int,
        heads: Int,
        headDim: Int
    ) -> SLSTMState {
        guard let state else {
            return SLSTMState(
                c: MLXArray.zeros([batch, heads, headDim], dtype: .float32),
                n: MLXArray.zeros([batch, heads, headDim], dtype: .float32),
                m: MLXArray.zeros([batch, heads, headDim], dtype: .float32),
                h: MLXArray.zeros([batch, heads, headDim], dtype: .float32)
            )
        }
        return SLSTMState(
            c: state.c.asType(.float32), n: state.n.asType(.float32),
            m: state.m.asType(.float32), h: state.h.asType(.float32)
        )
    }
}
