import BasaltCore
import Foundation
import MLX

package struct MACaQFlipResult: Sendable, Equatable {
    package var flips: Int
    package var sweeps: Int
    package var lossBefore: Float
    package var lossAfter: Float

    package init(flips: Int, sweeps: Int, lossBefore: Float, lossAfter: Float) {
        self.flips = flips
        self.sweeps = sweeps
        self.lossBefore = lossBefore
        self.lossAfter = lossAfter
    }
}

package struct MACaQGridState {
    package var codes: MLXArray
    package var scaleElem: MLXArray
    package var biasElem: MLXArray?
    package var levelTable: MLXArray?
    package var maxLevel: Int32

    package init(
        codes: MLXArray,
        scaleElem: MLXArray,
        biasElem: MLXArray?,
        levelTable: MLXArray?,
        maxLevel: Int32
    ) {
        self.codes = codes
        self.scaleElem = scaleElem
        self.biasElem = biasElem
        self.levelTable = levelTable
        self.maxLevel = maxLevel
    }

    package func dequantizedValues() -> MLXArray {
        let base: MLXArray
        if let levelTable {
            base = levelTable.take(codes.asType(.int32))
        } else {
            base = codes.asType(.float32)
        }
        let scaled = base * scaleElem
        guard let biasElem else { return scaled }
        return scaled + biasElem
    }
}

package enum MACaQMath {
    package static let mxfp4BlockSize = 32
    package static let mxfp4SignedLevels: [Float] = [
        -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    ]
    package static let mxfp4MagLUT: [UInt32] = [7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7]
    package static let mxfp4SignLUT: [UInt32] = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    package static func balancePreScale(
        weight: MLXArray,
        meanSquareActivation: MLXArray,
        alpha: Float = 0.75
    ) -> MLXArray {
        let wf = weight.asType(.float32)
        let ddof = wf.dim(0) > 1 ? 1 : 0
        let wStd = maximum(std(wf, axis: 0, ddof: ddof), Float(1e-8))
        let actRMS = sqrt(maximum(meanSquareActivation.asType(.float32), Float(1e-12)))
        let numer = pow(wStd, MLXArray(alpha))
        let denom = pow(actRMS, MLXArray(Float(1.0) - alpha))
        let raw = numer / maximum(denom, Float(1e-12))
        return clip(raw, min: Float(1e-4), max: Float(1e4))
    }

    package static func rotateBlocks(_ matrix: MLXArray, blockSize: Int) -> MLXArray {
        guard blockSize > 1 else { return matrix.asType(.float32) }
        let ones = MLXArray.ones([matrix.dim(-1)], dtype: .float32)
        return MACaQFusedKernels.blockHadamard(
            matrix.asType(.float32), scaleVector: ones, blockSize: blockSize, isInverse: false)
    }

    package static func transformFrame(
        weight: MLXArray,
        hessian: MLXArray,
        preScale: MLXArray?,
        blockSize: Int?,
        ridge: Float = 1.0
    ) -> (weight: MLXArray, hessian: MLXArray) {
        var wf = weight.asType(.float32)
        var hf = Float(0.5) * (hessian.asType(.float32) + hessian.asType(.float32).T)
        if let p = preScale?.asType(.float32) {
            wf = wf / p.expandedDimensions(axis: 0)
            hf = hf * p.expandedDimensions(axis: 0) * p.expandedDimensions(axis: 1)
        }
        if let b = blockSize, b > 1 {
            wf = rotateBlocks(wf, blockSize: b)
            hf = rotateBlocks(rotateBlocks(hf, blockSize: b).T, blockSize: b).T
        }
        if ridge > 0 {
            let cols = hf.dim(0)
            let diagMean = mean(diagonal(hf))
            let reg = ridge * maximum(diagMean, Float(1e-12))
            hf = hf + eye(cols, dtype: .float32) * reg
        }
        return (wf, hf)
    }

    package static func hessianLoss(values: MLXArray, target: MLXArray, hessian: MLXArray) -> Float {
        let diff = values.asType(.float32) - target.asType(.float32)
        let loss = (matmul(diff, hessian.asType(.float32)) * diff).sum()
        eval(loss)
        return loss.item(Float.self)
    }
}

extension MACaQMath {
    package static func mxfp4RTNSignedIndex(
        weight: MLXArray,
        exponentOffset: Int = 0
    ) -> (sidx: MLXArray, e8m0: MLXArray, scaleElem: MLXArray) {
        let rows = weight.dim(0)
        let cols = weight.dim(1)
        let blocks = cols / mxfp4BlockSize
        let grouped = weight.asType(.float32).reshaped(rows, blocks, mxfp4BlockSize)
        let amax = maximum(abs(grouped).max(axis: -1, keepDims: true), Float(1e-20))
        let rawExp = floor(log2(amax)) - Float(2.0) + Float(exponentOffset)
        let clampedExp = clip(rawExp, min: Float(-127.0), max: Float(127.0))
        let scale = pow(MLXArray(Float(2.0)), clampedExp)
        let scaledAbs = abs(grouped) / scale
        let c1 = (scaledAbs .> Float(0.25)).asType(.int32)
        let c2 = (scaledAbs .> Float(0.75)).asType(.int32)
        let c3 = (scaledAbs .> Float(1.25)).asType(.int32)
        let c4 = (scaledAbs .> Float(1.75)).asType(.int32)
        let c5 = (scaledAbs .> Float(2.5)).asType(.int32)
        let c6 = (scaledAbs .> Float(3.5)).asType(.int32)
        let c7 = (scaledAbs .> Float(5.0)).asType(.int32)
        let mag = c1 + c2 + c3 + c4 + c5 + c6 + c7
        let neg = grouped .< Float(0.0)
        let sidx = which(
            mag .== Int32(0),
            MLXArray(Int32(7)),
            which(neg, Int32(7) - mag, Int32(7) + mag)
        ).reshaped(rows, cols)
        let e8m0 = clip(clampedExp.squeezed(axis: -1) + Float(127.0), min: Float(0), max: Float(254)).asType(.uint8)
        let scaleElem = broadcast(scale, to: [rows, blocks, mxfp4BlockSize]).reshaped(rows, cols)
        return (sidx, e8m0, scaleElem)
    }

    package static func screenMXFP4Exponents(
        weight: MLXArray,
        hessianDiag: MLXArray,
        offsets: [Int] = [1, 0, -1]
    ) -> (sidx: MLXArray, e8m0: MLXArray, scaleElem: MLXArray) {
        guard let first = offsets.first else {
            return mxfp4RTNSignedIndex(weight: weight, exponentOffset: 0)
        }
        let rows = weight.dim(0)
        let cols = weight.dim(1)
        let blocks = cols / mxfp4BlockSize
        let wf = weight.asType(.float32)
        let levels = MLXArray(mxfp4SignedLevels)
        let hdBlock = hessianDiag.asType(.float32).reshaped(1, blocks, mxfp4BlockSize)
        var (bestSidx, bestE8, bestScale) = mxfp4RTNSignedIndex(weight: wf, exponentOffset: first)
        let initDeq = levels.take(bestSidx) * bestScale
        var bestErr = ((wf - initDeq).square().reshaped(rows, blocks, mxfp4BlockSize) * hdBlock).sum(axis: -1)
        for off in offsets.dropFirst() {
            let (candSidx, candE8, candScale) = mxfp4RTNSignedIndex(weight: wf, exponentOffset: off)
            let candDeq = levels.take(candSidx) * candScale
            let candErr = ((wf - candDeq).square().reshaped(rows, blocks, mxfp4BlockSize) * hdBlock).sum(axis: -1)
            let better = candErr .< bestErr
            let betterElem = better.expandedDimensions(axis: -1)
            bestErr = which(better, candErr, bestErr)
            bestE8 = which(better, candE8, bestE8)
            bestSidx = which(
                betterElem,
                candSidx.reshaped(rows, blocks, mxfp4BlockSize),
                bestSidx.reshaped(rows, blocks, mxfp4BlockSize)
            ).reshaped(rows, cols)
            bestScale = which(
                betterElem,
                candScale.reshaped(rows, blocks, mxfp4BlockSize),
                bestScale.reshaped(rows, blocks, mxfp4BlockSize)
            ).reshaped(rows, cols)
        }
        eval(bestSidx, bestE8, bestScale)
        return (bestSidx, bestE8, bestScale)
    }

    package static func packMXFP4(sidx: MLXArray, e8m0: MLXArray) -> (weight: MLXArray, scales: MLXArray) {
        let magTable = MLXArray(mxfp4MagLUT)
        let signTable = MLXArray(mxfp4SignLUT)
        let idx = sidx.asType(.int32)
        let nibbles = ((signTable.take(idx) << UInt32(3)) | magTable.take(idx)).asType(.uint8)
        let packed = PVCodeGrid.packCodes(nibbles, bits: 4)
        return (packed, e8m0.asType(.uint8))
    }

    package static func unpackMXFP4(
        weight: MLXArray,
        scales: MLXArray
    ) -> (sidx: MLXArray, e8m0: MLXArray, scaleElem: MLXArray) {
        let nibbles = PVCodeGrid.unpackCodes(weight, bits: 4).asType(.int32)
        let rows = nibbles.dim(0)
        let cols = nibbles.dim(1)
        let blocks = cols / mxfp4BlockSize
        let mag = nibbles & Int32(7)
        let neg = ((nibbles >> Int32(3)) & Int32(1)) .== Int32(1)
        let sidx = which(
            mag .== Int32(0),
            MLXArray(Int32(7)),
            which(neg, Int32(7) - mag, Int32(7) + mag))
        let e8 = scales.asType(.uint8)
        let blockScale = pow(MLXArray(Float(2.0)), e8.asType(.float32) - Float(127.0)).expandedDimensions(axis: -1)
        let scaleElem = broadcast(blockScale, to: [rows, blocks, mxfp4BlockSize]).reshaped(rows, cols)
        return (sidx, e8, scaleElem)
    }
}

extension MACaQMath {
    package static func makeGridState(
        packedWeight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        spec: PVCodeGrid.Spec
    ) -> (state: MACaQGridState, mxfp4E8M0: MLXArray?) {
        let rows = packedWeight.dim(0)
        if spec.encoding == .affine(.mxfp4) {
            let unp = unpackMXFP4(weight: packedWeight, scales: scales)
            let state = MACaQGridState(
                codes: unp.sidx, scaleElem: unp.scaleElem, biasElem: nil,
                levelTable: MLXArray(mxfp4SignedLevels), maxLevel: 14)
            return (state, unp.e8m0)
        }
        let codes = PVCodeGrid.unpackCodes(packedWeight, bits: spec.bits).asType(.int32)
        let cols = codes.dim(1)
        let groups = cols / spec.groupSize
        let scaleElem = broadcast(
            scales.asType(.float32).expandedDimensions(axis: -1),
            to: [rows, groups, spec.groupSize]
        ).reshaped(rows, cols)
        if spec.encoding == .nf4 {
            let state = MACaQGridState(
                codes: codes, scaleElem: scaleElem, biasElem: nil,
                levelTable: MLXArray(PVCodeGrid.nf4TableValues), maxLevel: 15)
            return (state, nil)
        }
        let biasElem = biases.map {
            broadcast(
                $0.asType(.float32).expandedDimensions(axis: -1),
                to: [rows, groups, spec.groupSize]
            ).reshaped(rows, cols)
        }
        let maxLevel = Int32((1 << spec.bits) - 1)
        let state = MACaQGridState(
            codes: codes, scaleElem: scaleElem, biasElem: biasElem,
            levelTable: nil, maxLevel: maxLevel)
        return (state, nil)
    }

    package static func flipRefine(
        state: inout MACaQGridState,
        targetWeight: MLXArray,
        hessian: MLXArray,
        maxIterations: Int = 128,
        rowChunkElements: Int = 36_000_000
    ) -> MACaQFlipResult {
        let totalRows = state.codes.dim(0)
        let cols = state.codes.dim(1)
        guard totalRows > 0, cols > 0, maxIterations > 0 else {
            let loss = hessianLoss(values: state.dequantizedValues(), target: targetWeight, hessian: hessian)
            return MACaQFlipResult(flips: 0, sweeps: 0, lossBefore: loss, lossAfter: loss)
        }
        let lossBefore = hessianLoss(values: state.dequantizedValues(), target: targetWeight, hessian: hessian)
        let rowsPerChunk = max(256, min(totalRows, rowChunkElements / max(1, 3 * cols)))
        var updatedChunks: [MLXArray] = []
        var totalFlips = 0
        var maxSweeps = 0
        var startRow = 0
        while startRow < totalRows {
            let endRow = min(totalRows, startRow + rowsPerChunk)
            let subState = MACaQGridState(
                codes: state.codes[startRow ..< endRow],
                scaleElem: state.scaleElem[startRow ..< endRow],
                biasElem: state.biasElem.map { $0[startRow ..< endRow] },
                levelTable: state.levelTable,
                maxLevel: state.maxLevel)
            let chunkRes = MACaQFlipSweep.runChunk(
                subState, target: targetWeight[startRow ..< endRow],
                hessian: hessian, maxIterations: maxIterations)
            updatedChunks.append(chunkRes.codes)
            totalFlips += chunkRes.flips
            maxSweeps = max(maxSweeps, chunkRes.sweeps)
            startRow = endRow
        }
        state.codes = updatedChunks.count == 1 ? updatedChunks[0] : concatenated(updatedChunks, axis: 0)
        eval(state.codes)
        let lossAfter = hessianLoss(values: state.dequantizedValues(), target: targetWeight, hessian: hessian)
        return MACaQFlipResult(flips: totalFlips, sweeps: maxSweeps, lossBefore: lossBefore, lossAfter: lossAfter)
    }
}

private enum MACaQFlipSweep {
    struct SweepContext {
        var hdRow: MLXArray
        var hf: MLXArray
        var table: MLXArray?
        var maxLevel: Int32
        var infVal: MLXArray
    }

    static func runChunk(
        _ initial: MACaQGridState,
        target: MLXArray,
        hessian: MLXArray,
        maxIterations: Int
    ) -> (codes: MLXArray, flips: Int, sweeps: Int) {
        let rows = initial.codes.dim(0)
        let cols = initial.codes.dim(1)
        let hf = hessian.asType(.float32)
        let ctx = SweepContext(
            hdRow: diagonal(hf).reshaped(1, cols), hf: hf,
            table: initial.levelTable, maxLevel: initial.maxLevel, infVal: MLXArray(Float.infinity))
        var fullQ = initial.codes.asType(.int32)
        var activeIds = (0 ..< rows).map(Int32.init)
        var activeArr = MLXArray(activeIds)
        var (q, se) = (fullQ, initial.scaleElem)
        var g = matmul(target.asType(.float32) - initial.dequantizedValues(), hf)
        eval(fullQ, q, se, g, ctx.hdRow)
        var (totalFlips, iter) = (0, 0)
        while iter < maxIterations && !activeIds.isEmpty {
            let steps = min(4, maxIterations - iter)
            var flipsInBatch = MLXArray(Int32(0))
            var lastImproving = MLXArray.zeros([activeIds.count], dtype: .bool)
            for _ in 0 ..< steps {
                let step = singleSweepStep(q: q, se: se, g: g, ctx: ctx)
                (q, g, lastImproving) = (step.q, step.g, step.improving)
                flipsInBatch = flipsInBatch + step.improving.asType(.int32).sum()
            }
            iter += steps
            eval(q, g, lastImproving, flipsInBatch)
            let count = Int(flipsInBatch.item(Int32.self))
            totalFlips += count
            let idx2D = broadcast(activeArr.expandedDimensions(axis: 1), to: [activeIds.count, cols])
            fullQ = putAlong(fullQ, idx2D, values: q, axis: 0)
            if count == 0 || !compactActive(lastImproving, ids: &activeIds, arr: &activeArr, q: &q, se: &se, g: &g) {
                eval(fullQ)
                break
            }
            eval(fullQ, q, se, g)
        }
        return (fullQ, totalFlips, iter)
    }

    private static func compactActive(
        _ improving: MLXArray,
        ids: inout [Int32],
        arr: inout MLXArray,
        q: inout MLXArray,
        se: inout MLXArray,
        g: inout MLXArray
    ) -> Bool {
        let flags = improving.asArray(Bool.self)
        var (keptLocal, nextGlobal) = ([Int32](), [Int32]())
        for (localIdx, flag) in flags.enumerated() where flag {
            keptLocal.append(Int32(localIdx))
            nextGlobal.append(ids[localIdx])
        }
        guard !keptLocal.isEmpty else { return false }
        if keptLocal.count < ids.count {
            let keepArr = MLXArray(keptLocal)
            ids = nextGlobal
            arr = MLXArray(nextGlobal)
            q = q.take(keepArr, axis: 0)
            se = se.take(keepArr, axis: 0)
            g = g.take(keepArr, axis: 0)
        }
        return true
    }

    private static func singleSweepStep(
        q: MLXArray,
        se: MLXArray,
        g: MLXArray,
        ctx: SweepContext
    ) -> (q: MLXArray, g: MLXArray, improving: MLXArray) {
        let upOk = q .< ctx.maxLevel
        let dnOk = q .> Int32(0)
        let cUp: MLXArray
        let cDn: MLXArray
        if let table = ctx.table {
            let cur = table.take(q)
            cUp = (table.take(minimum(q + Int32(1), ctx.maxLevel)) - cur) * se
            cDn = (table.take(maximum(q - Int32(1), Int32(0))) - cur) * se
        } else {
            cUp = se
            cDn = -se
        }
        let dlUp = which(upOk, cUp * cUp * ctx.hdRow - Float(2.0) * cUp * g, ctx.infVal)
        let dlDn = which(dnOk, cDn * cDn * ctx.hdRow - Float(2.0) * cDn * g, ctx.infVal)
        let upBetter = dlUp .<= dlDn
        let dl = which(upBetter, dlUp, dlDn)
        let bestJ = argMin(dl, axis: 1)
        let bestIdx2D = bestJ.expandedDimensions(axis: 1)
        let bestDL = takeAlong(dl, bestIdx2D, axis: 1).squeezed(axis: 1)
        let improving = bestDL .< Float(-1e-12)
        let upSel = takeAlong(upBetter, bestIdx2D, axis: 1).squeezed(axis: 1)
        let deltaQ = which(
            improving,
            which(upSel, MLXArray(Int32(1)), MLXArray(Int32(-1))),
            MLXArray(Int32(0)))
        let chosenUp = takeAlong(cUp, bestIdx2D, axis: 1).squeezed(axis: 1)
        let chosenDn = takeAlong(cDn, bestIdx2D, axis: 1).squeezed(axis: 1)
        let cChosen = which(improving, which(upSel, chosenUp, chosenDn), MLXArray(Float(0.0)))
        let nextQ = putAlong(
            q, bestIdx2D,
            values: takeAlong(q, bestIdx2D, axis: 1) + deltaQ.expandedDimensions(axis: 1),
            axis: 1)
        let nextG = g - cChosen.expandedDimensions(axis: 1) * ctx.hf.take(bestJ, axis: 0)
        return (nextQ, nextG, improving)
    }
}
