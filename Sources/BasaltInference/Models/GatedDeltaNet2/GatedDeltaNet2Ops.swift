import Foundation
import MLX

package enum GatedDeltaNet2Ops {
    package static func applyBetaGuard(
        erase: MLXArray,
        key: MLXArray,
        decay: MLXArray
    ) -> MLXArray {
        let zero = MLXArray(Float(0), dtype: .float32)
        let one = MLXArray(Float(1), dtype: .float32)
        let eps = MLXArray(Float(1e-12), dtype: .float32)
        let k2 = key * key
        let aSum = k2.sum(axis: -1, keepDims: true)
        let sSum = (erase * k2).sum(axis: -1, keepDims: true)
        let dMax = minimum(decay.max(axis: -1, keepDims: true), one)
        let bMean = sSum / maximum(aSum, eps)
        let centered = erase - bMean
        let delta2 = aSum * (centered * centered * k2).sum(axis: -1, keepDims: true)
        let d2 = dMax * dMax
        let lhs = d2 * delta2
        let oneMinusS = one - sSum
        let factor = maximum(zero, one - d2 * oneMinusS * oneMinusS)
        let rhs = maximum(zero, (one - d2) * factor)
        let safeLhs = maximum(lhs, MLXArray(Float(1e-30), dtype: .float32))
        let rho = MLX.sqrt(minimum(rhs / safeLhs, one))
        let shrink = (aSum .> eps) .&& (lhs .> rhs) .&& (lhs .> zero)
        return MLX.where(shrink, bMean + rho * centered, erase)
    }

    package static func forward(
        _ inputs: GatedDeltaNet2Inputs,
        state: MLXArray?,
        mask: MLXArray? = nil,
        chunkSize: Int = 64,
        useMetal: Bool = true
    ) -> (output: MLXArray, state: MLXArray) {
        if Device.defaultDevice() == .gpu,
           [inputs.q, inputs.k, inputs.v, inputs.gLog, inputs.erase, inputs.write, state, mask].contains(where: { $0?.dtype == .float64 }) {
            return Device.withDefaultDevice(.cpu) {
                forward(inputs, state: state, mask: mask, chunkSize: chunkSize, useMetal: useMetal)
            }
        }
        let steps = inputs.q.dim(1)
        guard steps > 0 else {
            let stateIn = contiguous(
                (state ?? MLXArray.zeros([inputs.q.dim(0), inputs.v.dim(2), inputs.v.dim(3), inputs.k.dim(3)], dtype: .float32))
                    .asType(.float32)
            )
            return (MLXArray.zeros([inputs.q.dim(0), 0, inputs.v.dim(2), inputs.v.dim(3)], dtype: inputs.q.dtype), stateIn)
        }
        if steps > 1 && mask == nil && chunkSize > 1 {
            return chunkedWY(inputs, state: state, chunkSize: chunkSize)
        }
        if useMetal, GatedDeltaNet2Kernel.canUseKernel(inputs, state: state, mask: mask) {
            return GatedDeltaNet2Kernel.run(inputs, state: state, mask: mask)
        }
        return recurrentOps(inputs, state: state, mask: mask)
    }

    package static func recurrentOps(
        _ inputs: GatedDeltaNet2Inputs,
        state: MLXArray?,
        mask: MLXArray? = nil
    ) -> (output: MLXArray, state: MLXArray) {
        let (batch, steps, hk, dk) = (inputs.q.dim(0), inputs.q.dim(1), inputs.q.dim(2), inputs.q.dim(3))
        let (hv, dv) = (inputs.v.dim(2), inputs.v.dim(3))
        var current = (state ?? MLXArray.zeros([batch, hv, dv, dk], dtype: .float32)).asType(.float32)
        guard steps > 0 else {
            return (MLXArray.zeros([batch, 0, hv, dv], dtype: inputs.q.dtype), contiguous(current))
        }
        let repeatFactor = hv / max(hk, 1)
        let qFull = (repeatFactor > 1 ? repeated(inputs.q, count: repeatFactor, axis: 2) : inputs.q).asType(.float32)
        let kFull = (repeatFactor > 1 ? repeated(inputs.k, count: repeatFactor, axis: 2) : inputs.k).asType(.float32)
        let vFull = inputs.v.asType(.float32)
        let decayFull = MLX.exp(inputs.gLog.asType(.float32))
        let rawErase = inputs.erase.asType(.float32)
        let eraseFull = inputs.betaGuard ? applyBetaGuard(erase: rawErase, key: kFull, decay: decayFull) : rawErase
        let writeFull = inputs.write.asType(.float32)

        var outputs: [MLXArray] = []
        outputs.reserveCapacity(steps)
        for step in 0 ..< steps {
            let stepMask = mask.map { $0[0..., step] }
            let slice = RecurrentSlice(
                q: qFull[0..., step],
                k: kFull[0..., step],
                v: vFull[0..., step],
                decay: decayFull[0..., step],
                erase: eraseFull[0..., step],
                write: writeFull[0..., step]
            )
            let (yStep, nextState) = recurrentStep(slice, state: current, mask: stepMask)
            outputs.append(yStep)
            current = nextState
        }
        return (MLX.stacked(outputs, axis: 1).asType(inputs.q.dtype), contiguous(current))
    }

    package static func chunkedWY(
        _ inputs: GatedDeltaNet2Inputs,
        state: MLXArray?,
        chunkSize: Int = 64
    ) -> (output: MLXArray, state: MLXArray) {
        let (batch, steps, hk, dk) = (inputs.q.dim(0), inputs.q.dim(1), inputs.q.dim(2), inputs.q.dim(3))
        let (hv, dv) = (inputs.v.dim(2), inputs.v.dim(3))
        var current = (state ?? MLXArray.zeros([batch, hv, dv, dk], dtype: .float32)).asType(.float32)
        guard steps > 0 else {
            return (MLXArray.zeros([batch, 0, hv, dv], dtype: inputs.q.dtype), contiguous(current))
        }
        let repeatFactor = hv / max(hk, 1)
        let qFull = (repeatFactor > 1 ? repeated(inputs.q, count: repeatFactor, axis: 2) : inputs.q).asType(.float32)
        let kFull = (repeatFactor > 1 ? repeated(inputs.k, count: repeatFactor, axis: 2) : inputs.k).asType(.float32)
        let vWrite = inputs.v.asType(.float32) * inputs.write.asType(.float32)
        let gFull = inputs.gLog.asType(.float32)
        let rawErase = inputs.erase.asType(.float32)
        let effErase = inputs.betaGuard ? applyBetaGuard(erase: rawErase, key: kFull, decay: MLX.exp(gFull)) : rawErase
        let eraseKey = effErase * kFull

        let prepared = PreparedWY(
            q: qFull.transposed(0, 2, 1, 3),
            k: kFull.transposed(0, 2, 1, 3),
            eraseKey: eraseKey.transposed(0, 2, 1, 3),
            vWrite: vWrite.transposed(0, 2, 1, 3),
            gLog: gFull.transposed(0, 2, 1, 3)
        )
        let stepSize = max(1, chunkSize)
        var chunks: [MLXArray] = []
        var start = 0
        while start < steps {
            let end = min(steps, start + stepSize)
            let (chunkOut, nextState) = wyChunk(prepared, range: start ..< end, state: current)
            chunks.append(chunkOut)
            current = nextState
            start = end
        }
        let combined = chunks.count == 1 ? chunks[0] : concatenated(chunks, axis: 2)
        return (combined.transposed(0, 2, 1, 3).asType(inputs.q.dtype), contiguous(current))
    }
}

extension GatedDeltaNet2Ops {
    private struct RecurrentSlice {
        let q: MLXArray
        let k: MLXArray
        let v: MLXArray
        let decay: MLXArray
        let erase: MLXArray
        let write: MLXArray
    }

    private struct PreparedWY {
        let q: MLXArray
        let k: MLXArray
        let eraseKey: MLXArray
        let vWrite: MLXArray
        let gLog: MLXArray
    }

    private static func recurrentStep(
        _ slice: RecurrentSlice,
        state: MLXArray,
        mask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let previous = state
        var updated = state * slice.decay.expandedDimensions(axis: -2)
        let eraseKey = (slice.erase * slice.k).expandedDimensions(axis: -2)
        let kvMem = (updated * eraseKey).sum(axis: -1)
        let delta = slice.write * slice.v - kvMem
        updated = updated + slice.k.expandedDimensions(axis: -2) * delta.expandedDimensions(axis: -1)
        var y = (updated * slice.q.expandedDimensions(axis: -2)).sum(axis: -1)
        if let mask {
            y = MLX.where(mask.expandedDimensions(axes: [1, 2]), y, MLXArray.zeros(like: y))
            updated = MLX.where(mask.expandedDimensions(axes: [1, 2, 3]), updated, previous)
        }
        return (y, updated)
    }

    private static func wyChunk(
        _ p: PreparedWY,
        range: Range<Int>,
        state: MLXArray
    ) -> (MLXArray, MLXArray) {
        let qC = p.q[0..., 0..., range, 0...]
        let kC = p.k[0..., 0..., range, 0...]
        let aC = p.eraseKey[0..., 0..., range, 0...]
        let vC = p.vWrite[0..., 0..., range, 0...]
        let gC = p.gLog[0..., 0..., range, 0...]
        let cumG = cumsum(gC, axis: 2)
        let expCumG = MLX.exp(cumG)

        let aMat = tiledDecayInnerProducts(lhs: aC, rhs: kC, cumG: cumG, strictLower: true)
        let pMat = tiledDecayInnerProducts(lhs: qC, rhs: kC, cumG: cumG, strictLower: false)
        let tInv = invertUnitLowerTriangular(aMat)

        let stateT = state.swappedAxes(-1, -2)
        let uState = matmul(aC * expCumG, stateT)
        let delta = matmul(tInv, vC - uState)
        let yState = matmul(qC * expCumG, stateT)
        let yChunk = yState + matmul(pMat, delta)

        let l = range.count
        let gEnd = cumG[0..., 0..., l - 1, 0...]
        let kEnd = kC * MLX.exp(gEnd.expandedDimensions(axis: -2) - cumG)
        let nextState = state * MLX.exp(gEnd).expandedDimensions(axis: -2)
            + matmul(delta.swappedAxes(-1, -2), kEnd)
        return (yChunk, nextState)
    }

    private static func tiledDecayInnerProducts(
        lhs: MLXArray,
        rhs: MLXArray,
        cumG: MLXArray,
        strictLower: Bool
    ) -> MLXArray {
        let (b, h, l) = (lhs.dim(0), lhs.dim(1), lhs.dim(2))
        let tile = 16
        let negFloor = MLXArray(Float(-1e9), dtype: .float32)
        let zero = MLXArray(Float(0), dtype: .float32)
        var rowBlocks: [MLXArray] = []
        var rowStart = 0
        while rowStart < l {
            let rowEnd = min(l, rowStart + tile)
            let rLen = rowEnd - rowStart
            let cLen = rowEnd
            let rowIdx = MLXArray((rowStart ..< rowEnd).map(Int32.init)).expandedDimensions(axis: 1)
            let colIdx = MLXArray((0 ..< cLen).map(Int32.init)).expandedDimensions(axis: 0)
            let mask2d = strictLower ? (rowIdx .> colIdx) : (rowIdx .>= colIdx)
            let mask5d = mask2d.reshaped(1, 1, rLen, cLen, 1)
            let gRow = cumG[0..., 0..., rowStart ..< rowEnd, 0...].expandedDimensions(axis: 3)
            let gCol = cumG[0..., 0..., 0 ..< cLen, 0...].expandedDimensions(axis: 2)
            let decayTile = MLX.where(mask5d, MLX.exp(MLX.where(mask5d, gRow - gCol, negFloor)), zero)
            let lhsTile = lhs[0..., 0..., rowStart ..< rowEnd, 0...].expandedDimensions(axis: 3)
            let rhsTile = rhs[0..., 0..., 0 ..< cLen, 0...].expandedDimensions(axis: 2)
            var block = (lhsTile * rhsTile * decayTile).sum(axis: -1)
            if cLen < l {
                block = concatenated([block, MLXArray.zeros([b, h, rLen, l - cLen], dtype: .float32)], axis: -1)
            }
            rowBlocks.append(block)
            rowStart = rowEnd
        }
        return rowBlocks.count == 1 ? rowBlocks[0] : concatenated(rowBlocks, axis: 2)
    }

    private static func invertUnitLowerTriangular(_ strictLower: MLXArray) -> MLXArray {
        let (b, h, l) = (strictLower.dim(0), strictLower.dim(1), strictLower.dim(2))
        var p = 1
        while p < l { p *= 2 }
        var padded = strictLower
        if p > l {
            let rightPad = MLXArray.zeros([b, h, l, p - l], dtype: .float32)
            let wide = concatenated([padded, rightPad], axis: -1)
            let botPad = MLXArray.zeros([b, h, p - l, p], dtype: .float32)
            padded = concatenated([wide, botPad], axis: -2)
        }
        var invDiag = MLXArray.ones([b, h, p, 1, 1], dtype: .float32)
        var m = 1
        while m < p {
            let halfBlocks = p / (2 * m)
            let paired = invDiag.reshaped([b, h, halfBlocks, 2, m, m])
            let inv11 = paired[0..., 0..., 0..., 0, 0..., 0...]
            let inv22 = paired[0..., 0..., 0..., 1, 0..., 0...]
            let l21Slices = (0 ..< halfBlocks).map { blk in
                padded[0..., 0..., (2 * blk + 1) * m ..< (2 * blk + 2) * m, (2 * blk) * m ..< (2 * blk + 1) * m]
            }
            let l21 = MLX.stacked(l21Slices, axis: 2)
            let inv21 = -matmul(matmul(inv22, l21), inv11)
            let zeros = MLXArray.zeros(like: inv11)
            let top = concatenated([inv11, zeros], axis: -1)
            let bot = concatenated([inv21, inv22], axis: -1)
            invDiag = concatenated([top, bot], axis: -2)
            m *= 2
        }
        return invDiag.squeezed(axis: 2)[0..., 0..., ..<l, ..<l]
    }
}
