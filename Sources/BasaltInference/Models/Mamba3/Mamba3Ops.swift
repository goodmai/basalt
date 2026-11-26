import Foundation
import MLX

package struct Mamba3State {
    package var angleState: MLXArray
    package var ssmState: MLXArray
    package var kState: MLXArray
    package var vState: MLXArray

    package init(angleState: MLXArray, ssmState: MLXArray, kState: MLXArray, vState: MLXArray) {
        self.angleState = angleState
        self.ssmState = ssmState
        self.kState = kState
        self.vState = vState
    }

    package func detached() -> Mamba3State {
        Mamba3State(
            angleState: contiguous(angleState.asType(.float32)),
            ssmState: contiguous(ssmState.asType(.float32)),
            kState: contiguous(kState.asType(.float32)),
            vState: contiguous(vState.asType(.float32))
        )
    }

    package static func zeros(
        batch: Int, heads: Int, headDim: Int, dState: Int, mimoRank: Int, numRopeAngles: Int? = nil
    ) -> Mamba3State {
        let angles = numRopeAngles ?? (dState / 2)
        return Mamba3State(
            angleState: MLXArray.zeros([batch, heads, angles], dtype: .float32),
            ssmState: MLXArray.zeros([batch, heads, headDim, dState], dtype: .float32),
            kState: MLXArray.zeros([batch, heads, mimoRank, dState], dtype: .float32),
            vState: MLXArray.zeros([batch, heads, mimoRank, headDim], dtype: .float32)
        )
    }
}

package struct Mamba3RecurrenceInputs {
    package let q: MLXArray
    package let k: MLXArray
    package let v: MLXArray
    package let dt: MLXArray
    package let logAlpha: MLXArray
    package let trap: MLXArray
    package var angles: MLXArray
    package var splitHalfRoPE: Bool

    package init(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        dt: MLXArray,
        logAlpha: MLXArray,
        trap: MLXArray
    ) {
        self.q = q
        self.k = k
        self.v = v
        self.dt = dt
        self.logAlpha = logAlpha
        self.trap = trap
        let b = q.dim(0)
        let t = q.dim(1)
        let h = q.dim(2)
        let half = q.dim(4) / 2
        self.angles = MLXArray.zeros([b, t, h, half], dtype: .float32)
        self.splitHalfRoPE = q.dim(3) > 1
    }

    package func withAngles(_ angles: MLXArray, splitHalfRoPE: Bool) -> Mamba3RecurrenceInputs {
        var copy = self
        copy.angles = angles
        copy.splitHalfRoPE = splitHalfRoPE
        return copy
    }
}

package enum Mamba3Ops {
    package static func heavyTail(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let zero = MLXArray(Float(0), dtype: .float32)
        let pos = maximum(xf, zero)
        let neg = minimum(xf, zero)
        return MLX.where(xf .>= zero, 1 + pos, 1 / (1 - neg))
    }

    package static func applyRoPE(
        _ x: MLXArray,
        cos cosValues: MLXArray,
        sin sinValues: MLXArray,
        splitHalf: Bool
    ) -> MLXArray {
        let dState = x.dim(-1)
        let half = cosValues.dim(-1)
        let rotDim = half * 2
        guard rotDim > 0 else { return x }
        if splitHalf {
            let halfState = dState / 2
            let x1 = x[.ellipsis, ..<half]
            let x2 = x[.ellipsis, halfState ..< (halfState + half)]
            let r1 = x1 * cosValues - x2 * sinValues
            let r2 = x1 * sinValues + x2 * cosValues
            guard half < halfState else {
                return concatenated([r1, r2], axis: -1)
            }
            return concatenated(
                [r1, x[.ellipsis, half ..< halfState], r2, x[.ellipsis, (halfState + half)...]],
                axis: -1
            )
        }
        let head = rotDim < dState ? x[.ellipsis, ..<rotDim] : x
        let leading = Array(head.shape.dropLast())
        let pairs = head.reshaped(leading + [half, 2])
        let x1 = pairs[.ellipsis, 0]
        let x2 = pairs[.ellipsis, 1]
        let r1 = x1 * cosValues - x2 * sinValues
        let r2 = x1 * sinValues + x2 * cosValues
        let rotated = MLX.stacked([r1, r2], axis: -1).reshaped(leading + [rotDim])
        return rotDim < dState ? concatenated([rotated, x[.ellipsis, rotDim...]], axis: -1) : rotated
    }

    package static func forward(
        _ inputs: Mamba3RecurrenceInputs,
        state: Mamba3State?,
        mask: MLXArray? = nil,
        chunkSize: Int = 64
    ) -> (output: MLXArray, state: Mamba3State) {
        let (batch, steps, heads, rank, dState) = (
            inputs.q.dim(0), inputs.q.dim(1), inputs.q.dim(2), inputs.q.dim(3), inputs.q.dim(4)
        )
        let headDim = inputs.v.dim(4)
        let initial = state ?? Mamba3State.zeros(
            batch: batch, heads: heads, headDim: headDim, dState: dState,
            mimoRank: rank, numRopeAngles: inputs.angles.dim(-1)
        )
        guard steps > 0 else {
            return (MLXArray.zeros([batch, 0, heads, rank, headDim], dtype: .float32), initial.detached())
        }
        if steps > 1 && mask == nil && chunkSize > 1 {
            return chunkedSSD(inputs, state: initial, chunkSize: chunkSize)
        }
        return recurrentScan(inputs, state: initial, mask: mask)
    }

    package static func chunkedSSD(
        _ inputs: Mamba3RecurrenceInputs,
        state: Mamba3State,
        chunkSize: Int
    ) -> (output: MLXArray, state: Mamba3State) {
        let steps = inputs.q.dim(1)
        var current = state
        var chunks: [MLXArray] = []
        let stepSize = max(1, chunkSize)
        var start = 0
        while start < steps {
            let end = min(steps, start + stepSize)
            let (chunkOut, nextState) = ssdChunk(inputs, range: start ..< end, state: current)
            chunks.append(chunkOut)
            current = nextState
            start = end
        }
        let combined = chunks.count == 1 ? chunks[0] : concatenated(chunks, axis: 1)
        return (combined, current.detached())
    }

    package static func recurrentScan(
        _ inputs: Mamba3RecurrenceInputs,
        state: Mamba3State,
        mask: MLXArray? = nil
    ) -> (output: MLXArray, state: Mamba3State) {
        let steps = inputs.q.dim(1)
        var current = state
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(steps)
        for step in 0 ..< steps {
            let stepMask = mask.map { $0[0..., step] }
            let (yStep, nextState) = recurrentStep(inputs, step: step, state: current, mask: stepMask)
            outputs.append(yStep)
            current = nextState
        }
        return (MLX.stacked(outputs, axis: 1), current.detached())
    }
}

extension Mamba3Ops {
    private struct ChunkInputs {
        let qRot: MLXArray
        let kRot: MLXArray
        let vT: MLXArray
        let dtT: MLXArray
        let logAT: MLXArray
        let trapT: MLXArray
        let phiEnd: MLXArray
    }

    private static func prepareChunk(
        _ inputs: Mamba3RecurrenceInputs,
        range: Range<Int>,
        angleState: MLXArray
    ) -> ChunkInputs {
        let l = range.count
        let dtC = inputs.dt[0..., range, 0...].asType(.float32)
        let logAC = inputs.logAlpha[0..., range, 0...].asType(.float32)
        let trapC = inputs.trap[0..., range, 0...].asType(.float32)
        let angC = inputs.angles[0..., range, 0..., 0...].asType(.float32)

        let deltaPhi = Float.pi * MLX.tanh(angC) * dtC.expandedDimensions(axis: -1)
        let phi = angleState.asType(.float32).expandedDimensions(axis: 1) + cumsum(deltaPhi, axis: 1)
        let cosPhi = MLX.cos(phi).expandedDimensions(axis: 3)
        let sinPhi = MLX.sin(phi).expandedDimensions(axis: 3)
        let qRot = applyRoPE(inputs.q[0..., range, 0..., 0..., 0...].asType(.float32), cos: cosPhi, sin: sinPhi, splitHalf: inputs.splitHalfRoPE)
            .transposed(0, 2, 1, 3, 4)
        let kRot = applyRoPE(inputs.k[0..., range, 0..., 0..., 0...].asType(.float32), cos: cosPhi, sin: sinPhi, splitHalf: inputs.splitHalfRoPE)
            .transposed(0, 2, 1, 3, 4)
        let vT = inputs.v[0..., range, 0..., 0..., 0...].asType(.float32).transposed(0, 2, 1, 3, 4)
        return ChunkInputs(
            qRot: qRot, kRot: kRot, vT: vT,
            dtT: dtC.transposed(0, 2, 1), logAT: logAC.transposed(0, 2, 1), trapT: trapC.transposed(0, 2, 1),
            phiEnd: phi[0..., l - 1, 0..., 0...]
        )
    }

    private static func ssdChunk(
        _ inputs: Mamba3RecurrenceInputs,
        range: Range<Int>,
        state: Mamba3State
    ) -> (MLXArray, Mamba3State) {
        let (b, l, h, r, n, p) = (
            inputs.q.dim(0), range.count, inputs.q.dim(2), inputs.q.dim(3), inputs.q.dim(4), inputs.v.dim(4)
        )
        let c = prepareChunk(inputs, range: range, angleState: state.angleState)
        let cumA = cumsum(c.logAT, axis: -1)
        let gamma = c.trapT * c.dtT
        let oneMinusTrapDt = (1 - c.trapT) * c.dtT
        let effWeight: MLXArray
        if l > 1 {
            let zeroTail = MLXArray.zeros([b, h, 1], dtype: .float32)
            effWeight = gamma + concatenated([oneMinusTrapDt[0..., 0..., 1...], zeroTail], axis: -1)
        } else {
            effWeight = gamma
        }

        let uPrev = matmul(state.vState.asType(.float32).swappedAxes(-1, -2), state.kState.asType(.float32))
        let hTilde = state.ssmState.asType(.float32) + oneMinusTrapDt[0..., 0..., 0].reshaped(b, h, 1, 1) * uPrev
        let weightMatrix = trapezoidalWeightMatrix(cumA: cumA, effWeight: effWeight, gamma: gamma, length: l)

        let qFlat = c.qRot.reshaped([b, h, l * r, n])
        let kFlat = c.kRot.reshaped([b, h, l * r, n])
        let vFlat = c.vT.reshaped([b, h, l * r, p])
        let qk = matmul(qFlat, kFlat.swappedAxes(-1, -2)).reshaped([b, h, l, r, l, r])
        let attn = (qk * weightMatrix.reshaped([b, h, l, 1, l, 1])).reshaped([b, h, l * r, l * r])
        let yIntra = matmul(attn, vFlat).reshaped([b, h, l, r, p])

        let qScaled = (c.qRot * MLX.exp(cumA).reshaped([b, h, l, 1, 1])).reshaped([b, h, l * r, n])
        let yState = matmul(qScaled, hTilde.swappedAxes(-1, -2)).reshaped([b, h, l, r, p])
        let yChunk = (yState + yIntra).transposed(0, 2, 1, 3, 4)

        let cumEnd = cumA[0..., 0..., l - 1]
        let toEnd = MLX.exp(cumEnd.expandedDimensions(axis: -1) - cumA) * effWeight
        let kWeighted = (c.kRot * toEnd.reshaped([b, h, l, 1, 1])).reshaped([b, h, l * r, n])
        let nextSSM = MLX.exp(cumEnd).reshaped(b, h, 1, 1) * hTilde + matmul(vFlat.swappedAxes(-1, -2), kWeighted)
        let nextState = Mamba3State(
            angleState: c.phiEnd,
            ssmState: nextSSM,
            kState: c.kRot[0..., 0..., l - 1, 0..., 0...],
            vState: c.vT[0..., 0..., l - 1, 0..., 0...]
        )
        return (yChunk, nextState)
    }

    private static func trapezoidalWeightMatrix(
        cumA: MLXArray,
        effWeight: MLXArray,
        gamma: MLXArray,
        length l: Int
    ) -> MLXArray {
        let idx = MLXArray((0 ..< l).map(Int32.init))
        let strictLower = idx.expandedDimensions(axis: 1) .> idx.expandedDimensions(axis: 0)
        let diagMask = idx.expandedDimensions(axis: 1) .== idx.expandedDimensions(axis: 0)
        let diff = cumA.expandedDimensions(axis: -1) - cumA.expandedDimensions(axis: -2)
        let negFloor = MLXArray(Float(-1e9), dtype: .float32)
        let zero = MLXArray(Float(0), dtype: .float32)
        let safeDecay = MLX.where(strictLower, MLX.exp(MLX.where(strictLower, diff, negFloor)), zero)
        let lowerWeighted = safeDecay * effWeight.expandedDimensions(axis: -2)
        let diagWeighted = MLX.where(diagMask, gamma.expandedDimensions(axis: -2), zero)
        return lowerWeighted + diagWeighted
    }

    private static func recurrentStep(
        _ inputs: Mamba3RecurrenceInputs,
        step: Int,
        state: Mamba3State,
        mask: MLXArray?
    ) -> (MLXArray, Mamba3State) {
        let b = inputs.q.dim(0)
        let h = inputs.q.dim(2)
        let dtT = inputs.dt[0..., step, 0...].asType(.float32)
        let logAT = inputs.logAlpha[0..., step, 0...].asType(.float32)
        let trapT = inputs.trap[0..., step, 0...].asType(.float32)
        let alphaT = MLX.exp(logAT)
        let betaT = (1 - trapT) * dtT * alphaT
        let gammaT = trapT * dtT

        let deltaPhi = Float.pi * MLX.tanh(inputs.angles[0..., step, 0..., 0...].asType(.float32))
            * dtT.expandedDimensions(axis: -1)
        let candAngle = state.angleState.asType(.float32) + deltaPhi
        let cosPhi = MLX.cos(candAngle).expandedDimensions(axis: 2)
        let sinPhi = MLX.sin(candAngle).expandedDimensions(axis: 2)

        let kT = applyRoPE(inputs.k[0..., step, 0..., 0..., 0...].asType(.float32), cos: cosPhi, sin: sinPhi, splitHalf: inputs.splitHalfRoPE)
        let qT = applyRoPE(inputs.q[0..., step, 0..., 0..., 0...].asType(.float32), cos: cosPhi, sin: sinPhi, splitHalf: inputs.splitHalfRoPE)
        let vT = inputs.v[0..., step, 0..., 0..., 0...].asType(.float32)

        let uPrev = matmul(state.vState.asType(.float32).swappedAxes(-1, -2), state.kState.asType(.float32))
        let uCurr = matmul(vT.swappedAxes(-1, -2), kT)
        let candSSM = alphaT.reshaped(b, h, 1, 1) * state.ssmState.asType(.float32)
            + betaT.reshaped(b, h, 1, 1) * uPrev
            + gammaT.reshaped(b, h, 1, 1) * uCurr
        var yT = matmul(qT, candSSM.swappedAxes(-1, -2))

        if let mask {
            let m2 = mask.expandedDimensions(axes: [1, 2])
            let m3 = mask.expandedDimensions(axes: [1, 2, 3])
            yT = MLX.where(m3, yT, MLXArray.zeros(like: yT))
            let masked = Mamba3State(
                angleState: MLX.where(m2, candAngle, state.angleState.asType(.float32)),
                ssmState: MLX.where(m3, candSSM, state.ssmState.asType(.float32)),
                kState: MLX.where(m3, kT, state.kState.asType(.float32)),
                vState: MLX.where(m3, vT, state.vState.asType(.float32))
            )
            return (yT, masked)
        }
        return (yT, Mamba3State(angleState: candAngle, ssmState: candSSM, kState: kT, vState: vT))
    }
}
