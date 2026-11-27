import BasaltCore
import Foundation
import MLX

package struct BrickIndex3D: Sendable, Equatable {
    package let t: Int, h: Int, w: Int
    package init(_ t: Int, _ h: Int, _ w: Int) { (self.t, self.h, self.w) = (t, h, w) }
}

package struct SpatioTemporalGrid: Sendable, Equatable {
    package let frames: Int, height: Int, width: Int
    package var tokenCount: Int { frames * height * width }

    package init(frames: Int, height: Int, width: Int) throws(BasaltError) {
        guard frames >= 1, height >= 1, width >= 1 else {
            throw .invalidRequestStructure(details: "SpatioTemporalGrid requires positive dimensions")
        }
        (self.frames, self.height, self.width) = (frames, height, width)
    }
}

package struct BrickShape: Sendable, Equatable {
    package static let appleDefault = BrickShape(uncheckedT: 4, h: 8, w: 8)
    package let t: Int, h: Int, w: Int
    package var volume: Int { t * h * w }

    package init(t: Int, h: Int, w: Int) throws(BasaltError) {
        guard t >= 1, h >= 1, w >= 1 else {
            throw .invalidRequestStructure(details: "BrickShape requires positive dimensions")
        }
        (self.t, self.h, self.w) = (t, h, w)
    }

    private init(uncheckedT t: Int, h: Int, w: Int) { (self.t, self.h, self.w) = (t, h, w) }
}

package struct WindowRadiusBricks: Sendable, Equatable {
    package let t: Int, h: Int, w: Int
    package init(t: Int, h: Int, w: Int) { (self.t, self.h, self.w) = (max(0, t), max(0, h), max(0, w)) }
}

package enum BrickDistanceMetric: Sendable, Equatable {
    case manhattan
    case euclidean
}

package struct RadialDecaySchedule: Sendable, Equatable {
    package static let disabled = RadialDecaySchedule(metric: .euclidean, decayRate: 0.0, energyThreshold: 0.0)
    package let metric: BrickDistanceMetric
    package let decayRate: Float
    package let energyThreshold: Float
    package let temporalWeight: Float
    package let spatialWeight: Float
    package let logarithmicShells: Bool

    package init(
        metric: BrickDistanceMetric = .euclidean,
        decayRate: Float,
        energyThreshold: Float,
        temporalWeight: Float = 1.0,
        spatialWeight: Float = 1.0,
        logarithmicShells: Bool = false
    ) {
        self.metric = metric
        self.decayRate = max(0.0, decayRate)
        self.energyThreshold = max(0.0, min(1.0, energyThreshold))
        self.temporalWeight = max(0.0, temporalWeight)
        self.spatialWeight = max(0.0, spatialWeight)
        self.logarithmicShells = logarithmicShells
    }

    package func distance(dt: Int, dh: Int, dw: Int) -> Float {
        switch metric {
        case .manhattan:
            return temporalWeight * Float(abs(dt)) + spatialWeight * Float(abs(dh) + abs(dw))
        case .euclidean:
            let ft = temporalWeight * Float(dt), fh = spatialWeight * Float(dh), fw = spatialWeight * Float(dw)
            return (ft * ft + fh * fh + fw * fw).squareRoot()
        }
    }

    package func energy(dt: Int, dh: Int, dw: Int) -> Float {
        Foundation.exp(-decayRate * distance(dt: dt, dh: dh, dw: dw))
    }

    package func retains(queryBrick q: BrickIndex3D, keyBrick k: BrickIndex3D) -> Bool {
        if q == k { return true }
        let dt = k.t - q.t, dh = k.h - q.h, dw = k.w - q.w
        guard energy(dt: dt, dh: dh, dw: dw) >= energyThreshold else { return false }
        if logarithmicShells {
            let l1 = abs(dt) + abs(dh) + abs(dw)
            if l1 >= 2 {
                let shell = Int(Foundation.log2(Double(l1)))
                return (k.t + k.h + k.w).isMultiple(of: 1 << max(0, shell - 1))
            }
        }
        return true
    }
}

package struct BrickSparsePlan: Sendable, Equatable {
    package let grid: SpatioTemporalGrid
    package let brick: BrickShape
    package let window: WindowRadiusBricks
    package let radial: RadialDecaySchedule
    package let bricksT: Int, bricksH: Int, bricksW: Int
    package let csrRowOffsets: [Int32]
    package let csrColIndices: [Int32]
    package let validTokenFlags: [UInt8]
    package let coversFullGrid: Bool

    package var brickCount: Int { bricksT * bricksH * bricksW }
    package var brickVolume: Int { brick.volume }

    package var isExactGrid: Bool {
        grid.frames == bricksT * brick.t && grid.height == bricksH * brick.h && grid.width == bricksW * brick.w
    }

    package var sparsityRatio: Double {
        1.0 - Double(csrColIndices.count) / Double(max(1, brickCount * brickCount))
    }

    package func packedSlot(t: Int, y: Int, x: Int) -> Int {
        let brickIdx = ((t / brick.t) * bricksH + (y / brick.h)) * bricksW + (x / brick.w)
        let localIdx = ((t % brick.t) * brick.h + (y % brick.h)) * brick.w + (x % brick.w)
        return brickIdx * brick.volume + localIdx
    }
}

package struct GlobalConditioningKV {
    package let keys: MLXArray, values: MLXArray
    package init(keys: MLXArray, values: MLXArray) { (self.keys, self.values) = (keys, values) }
}

package struct CPUGlobalKV: Sendable, Equatable {
    package let keys: [Float], values: [Float]
    package init(keys: [Float], values: [Float]) { (self.keys, self.values) = (keys, values) }
}

package struct STAExecutionOptions: Sendable, Equatable {
    package let scale: Float?, forceMetalKernel: Bool
    package init(scale: Float? = nil, forceMetalKernel: Bool = false) {
        (self.scale, self.forceMetalKernel) = (scale, forceMetalKernel)
    }
}

package struct CPUAttentionShape: Sendable, Equatable {
    package let batch: Int, qHeads: Int, kvHeads: Int, headDim: Int, sinkTokens: Int, scale: Float?
    package init(batch: Int, qHeads: Int, kvHeads: Int, headDim: Int, sinkTokens: Int = 0, scale: Float? = nil) {
        (self.batch, self.qHeads, self.kvHeads, self.headDim, self.sinkTokens, self.scale) = (batch, qHeads, kvHeads, headDim, sinkTokens, scale)
    }
}

package enum SlidingTileAttention3D {
    package static let supportedDTypes: Set<DType> = [.float16, .bfloat16, .float32]
    package static let maxKernelHeadDim = 256

    package static func buildPlan(
        grid: SpatioTemporalGrid, brick: BrickShape = .appleDefault,
        window: WindowRadiusBricks, radial: RadialDecaySchedule = .disabled
    ) -> BrickSparsePlan {
        let extents = BrickIndex3D(
            (grid.frames + brick.t - 1) / brick.t,
            (grid.height + brick.h - 1) / brick.h,
            (grid.width + brick.w - 1) / brick.w
        )
        let totalBricks = extents.t * extents.h * extents.w
        let csr = buildCSRNeighborhood(extents: extents, window: window, radial: radial)
        let flags = buildValidTokenFlags(grid: grid, brick: brick, extents: extents)
        return BrickSparsePlan(
            grid: grid, brick: brick, window: window, radial: radial,
            bricksT: extents.t, bricksH: extents.h, bricksW: extents.w,
            csrRowOffsets: csr.offsets, csrColIndices: csr.cols,
            validTokenFlags: flags, coversFullGrid: csr.cols.count == totalBricks * totalBricks
        )
    }

    package static func packIntoBricks(_ x: MLXArray, plan: BrickSparsePlan) -> MLXArray {
        if x.dtype == .float64 && Device.defaultDevice() == .gpu { return Device.withDefaultDevice(.cpu) { packIntoBricks(x, plan: plan) } }
        let b = x.dim(0), h = x.dim(1), d = x.dim(3), bh = b * h
        var gridView = x.reshaped(bh, plan.grid.frames, plan.grid.height, plan.grid.width, d)
        if !plan.isExactGrid {
            let padT = plan.bricksT * plan.brick.t - plan.grid.frames
            let padH = plan.bricksH * plan.brick.h - plan.grid.height
            let padW = plan.bricksW * plan.brick.w - plan.grid.width
            gridView = padded(gridView, widths: [IntOrPair(0), IntOrPair((0, padT)), IntOrPair((0, padH)), IntOrPair((0, padW)), IntOrPair(0)])
        }
        return gridView
            .reshaped([bh, plan.bricksT, plan.brick.t, plan.bricksH, plan.brick.h, plan.bricksW, plan.brick.w, d])
            .transposed(axes: [0, 1, 3, 5, 2, 4, 6, 7])
            .reshaped(b, h, plan.brickCount, plan.brickVolume, d)
    }

    package static func unpackFromBricks(_ packed: MLXArray, plan: BrickSparsePlan) -> MLXArray {
        if packed.dtype == .float64 && Device.defaultDevice() == .gpu { return Device.withDefaultDevice(.cpu) { unpackFromBricks(packed, plan: plan) } }
        let b = packed.dim(0), h = packed.dim(1), d = packed.dim(-1), bh = b * h
        var gridView = packed
            .reshaped([bh, plan.bricksT, plan.bricksH, plan.bricksW, plan.brick.t, plan.brick.h, plan.brick.w, d])
            .transposed(axes: [0, 1, 4, 2, 5, 3, 6, 7])
            .reshaped(bh, plan.bricksT * plan.brick.t, plan.bricksH * plan.brick.h, plan.bricksW * plan.brick.w, d)
        if !plan.isExactGrid {
            gridView = gridView[0..., ..<plan.grid.frames, ..<plan.grid.height, ..<plan.grid.width, 0...]
        }
        return gridView.reshaped(b, h, plan.grid.tokenCount, d)
    }

    package static func attention(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        plan: BrickSparsePlan, globalKV: GlobalConditioningKV? = nil, options: STAExecutionOptions = .init()
    ) -> MLXArray {
        let usesFloat64 = [queries, keys, values, globalKV?.keys, globalKV?.values].compactMap { $0 }.contains { $0.dtype == .float64 }
        if usesFloat64 && Device.defaultDevice() == .gpu {
            return Device.withDefaultDevice(.cpu) { attention(queries: queries, keys: keys, values: values, plan: plan, globalKV: globalKV, options: options) }
        }
        if queries.shape.contains(0) { return MLXArray.zeros(queries.shape, dtype: queries.dtype) }
        let scale = options.scale ?? (1.0 / Float(max(1, queries.dim(3))).squareRoot())
        if plan.coversFullGrid && !options.forceMetalKernel {
            return denseAttention(queries: queries, keys: keys, values: values, globalKV: globalKV, scale: scale)
        }
        guard supportsMetalKernel(queries: queries, keys: keys, values: values, globalKV: globalKV) else {
            return sparseFallback(queries: queries, keys: keys, values: values, plan: plan, globalKV: globalKV, scale: scale)
        }
        return metalBrickAttention(queries: queries, keys: keys, values: values, plan: plan, globalKV: globalKV, scale: scale)
    }

    package static func evaluateStepWithTeaCache(
        step: Int, modulatedInput: MLXArray, teaCache: inout TeaCacheState, computeVelocity: () -> MLXArray
    ) -> MLXArray {
        teaCache.resolveVelocity(step: step, modulatedInput: modulatedInput, compute: computeVelocity)
    }
}

extension SlidingTileAttention3D {
    private static func buildCSRNeighborhood(
        extents: BrickIndex3D, window: WindowRadiusBricks, radial: RadialDecaySchedule
    ) -> (offsets: [Int32], cols: [Int32]) {
        var offsets: [Int32] = [0]
        offsets.reserveCapacity(extents.t * extents.h * extents.w + 1)
        var cols: [Int32] = []
        for qbt in 0 ..< extents.t {
            for qbh in 0 ..< extents.h {
                for qbw in 0 ..< extents.w {
                    appendBrickNeighbors(
                        query: BrickIndex3D(qbt, qbh, qbw), extents: extents, window: window, radial: radial, into: &cols
                    )
                    offsets.append(Int32(cols.count))
                }
            }
        }
        return (offsets, cols)
    }

    private static func appendBrickNeighbors(
        query q: BrickIndex3D, extents: BrickIndex3D,
        window: WindowRadiusBricks, radial: RadialDecaySchedule, into cols: inout [Int32]
    ) {
        let tRange = max(0, q.t - window.t) ... min(extents.t - 1, q.t + min(window.t, extents.t))
        let hRange = max(0, q.h - window.h) ... min(extents.h - 1, q.h + min(window.h, extents.h))
        let wRange = max(0, q.w - window.w) ... min(extents.w - 1, q.w + min(window.w, extents.w))
        for kbt in tRange {
            for kbh in hRange {
                for kbw in wRange where radial.retains(queryBrick: q, keyBrick: BrickIndex3D(kbt, kbh, kbw)) {
                    cols.append(Int32((kbt * extents.h + kbh) * extents.w + kbw))
                }
            }
        }
    }

    private static func buildValidTokenFlags(
        grid: SpatioTemporalGrid, brick: BrickShape, extents: BrickIndex3D
    ) -> [UInt8] {
        var flags = [UInt8](repeating: 0, count: extents.t * extents.h * extents.w * brick.volume)
        var idx = 0
        for bt in 0 ..< extents.t {
            for bh in 0 ..< extents.h {
                for bw in 0 ..< extents.w {
                    for lt in 0 ..< brick.t {
                        let tValid = (bt * brick.t + lt) < grid.frames
                        for lh in 0 ..< brick.h {
                            let thValid = tValid && (bh * brick.h + lh) < grid.height
                            for lw in 0 ..< brick.w {
                                flags[idx] = (thValid && (bw * brick.w + lw) < grid.width) ? 1 : 0
                                idx += 1
                            }
                        }
                    }
                }
            }
        }
        return flags
    }

    package static func supportsMetalKernel(
        queries: MLXArray, keys: MLXArray, values: MLXArray, globalKV: GlobalConditioningKV?
    ) -> Bool {
        guard Device.defaultDevice() == .gpu, queries.ndim == 4, keys.ndim == 4, values.ndim == 4 else { return false }
        let batch = queries.dim(0), qHeads = queries.dim(1), tokens = queries.dim(2), headDim = queries.dim(3)
        let kvHeads = keys.dim(1), dtype = queries.dtype
        guard batch >= 1, tokens >= 1, headDim >= 1, headDim <= maxKernelHeadDim,
              supportedDTypes.contains(dtype), keys.dtype == dtype, values.dtype == dtype,
              keys.shape == [batch, kvHeads, tokens, headDim], values.shape == keys.shape,
              kvHeads >= 1, qHeads >= kvHeads, qHeads.isMultiple(of: kvHeads)
        else { return false }
        guard let globalKV else { return true }
        guard globalKV.keys.ndim == 4, globalKV.keys.dtype == dtype, globalKV.values.dtype == dtype,
              globalKV.keys.dim(0) == batch, globalKV.keys.dim(1) == kvHeads,
              globalKV.keys.dim(3) == headDim, globalKV.values.shape == globalKV.keys.shape
        else { return false }
        return true
    }

    private static func alignedBatch(_ x: MLXArray, to batch: Int) -> MLXArray {
        (x.dim(0) == 1 && batch > 1) ? broadcast(x, to: [batch, x.dim(1), x.dim(2), x.dim(3)]) : x
    }

    private static func alignedGlobalKV(_ globalKV: GlobalConditioningKV?, batch: Int) -> GlobalConditioningKV? {
        guard let g = globalKV, g.keys.dim(2) > 0 else { return nil }
        return GlobalConditioningKV(keys: alignedBatch(g.keys, to: batch), values: alignedBatch(g.values, to: batch))
    }

    private static func denseAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, globalKV: GlobalConditioningKV?, scale: Float
    ) -> MLXArray {
        let batch = queries.dim(0), qT = queries.dtype
        var (k, v) = (alignedBatch(keys, to: batch), alignedBatch(values, to: batch))
        if let s = alignedGlobalKV(globalKV, batch: batch) {
            (k, v) = (concatenated([k, s.keys], axis: 2), concatenated([v, s.values], axis: 2))
        }
        let sameReal = qT.isFloatingPoint && !qT.isComplex && k.dtype == qT && v.dtype == qT
        let cT: DType = sameReal ? qT : ([queries, k, v].contains { $0.dtype == .float64 } ? .float64 : .float32)
        let out = MLXFast.scaledDotProductAttention(
            queries: qT == cT ? queries : queries.asType(cT), keys: k.dtype == cT ? k : k.asType(cT),
            values: v.dtype == cT ? v : v.asType(cT), scale: scale, mask: .none)
        return out.dtype == qT ? out : out.asType(qT)
    }

    private static func metalBrickAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        plan: BrickSparsePlan, globalKV: GlobalConditioningKV?, scale: Float
    ) -> MLXArray {
        let batch = queries.dim(0), qHeads = queries.dim(1), kvHeads = keys.dim(1)
        let headDim = queries.dim(3), dtype = queries.dtype
        let sinkCount = globalKV?.keys.dim(2) ?? 0
        let qPacked = packIntoBricks(queries, plan: plan)
        let kPacked = packIntoBricks(keys, plan: plan)
        let vPacked = packIntoBricks(values, plan: plan)
        let sinkK = (sinkCount > 0 ? globalKV?.keys : nil) ?? MLXArray.zeros([1], dtype: dtype)
        let sinkV = (sinkCount > 0 ? globalKV?.values : nil) ?? MLXArray.zeros([1], dtype: dtype)
        let csrOffsets = MLXArray(plan.csrRowOffsets)
        let csrIndices = MLXArray(plan.csrColIndices.isEmpty ? [Int32(0)] : plan.csrColIndices)
        let validFlags = MLXArray(plan.validTokenFlags)
        let meta = MLXArray([
            Int32(qHeads), Int32(kvHeads), Int32(plan.brickCount), Int32(plan.brickVolume), Int32(sinkCount),
        ])
        let qTilesPerBrick = (plan.brickVolume + 3) / 4
        let outPacked = sta3DBrickAttentionKernel(
            [qPacked, kPacked, vPacked, sinkK, sinkV, csrOffsets, csrIndices, validFlags, meta, MLXArray([scale])],
            template: [("T", dtype), ("D", headDim), ("HAS_SINKS", sinkCount > 0), ("EXACT_GRID", plan.isExactGrid)],
            grid: (128 * plan.brickCount * qTilesPerBrick, batch * qHeads, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[batch, qHeads, plan.brickCount, plan.brickVolume, headDim]],
            outputDTypes: [dtype]
        )[0]
        return unpackFromBricks(outPacked, plan: plan)
    }
}

private struct CPUHeadTask {
    let bqh: Int, bkvh: Int, sinkTokens: Int, headDim: Int, scale: Float
}

private struct CPUPackedQKV {
    let q: [Float], k: [Float], v: [Float]
}

private struct CPUOnlineSoftmaxState {
    var rowMax: Float = -.infinity
    var rowSum: Float = 0.0
    var acc: [Float]

    init(dim: Int) { self.acc = [Float](repeating: 0, count: dim) }

    mutating func update(q: [Float], qOff: Int, k: [Float], v: [Float], kvOff: Int, scale: Float) {
        var dot: Float = 0.0
        for c in 0 ..< acc.count { dot += q[qOff + c] * k[kvOff + c] }
        let score = dot * scale
        let bad = rowMax.isNaN || score.isNaN
        let nextMax: Float = bad ? .nan : max(rowMax, score)
        let active = bad || nextMax != -.infinity
        let alpha: Float = bad ? .nan : ((rowMax != -.infinity && active) ? Foundation.exp(rowMax - nextMax) : 0.0)
        let beta: Float = bad ? .nan : ((active && score != -.infinity) ? Foundation.exp(score - nextMax) : 0.0)
        rowMax = nextMax
        rowSum = rowSum * alpha + beta
        for c in 0 ..< acc.count { acc[c] = acc[c] * alpha + beta * v[kvOff + c] }
    }
}

extension SlidingTileAttention3D {
    private static func sparseFallback(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        plan: BrickSparsePlan, globalKV: GlobalConditioningKV?, scale: Float
    ) -> MLXArray {
        let batch = queries.dim(0), headDim = queries.dim(3)
        let kAligned = alignedBatch(keys, to: batch), vAligned = alignedBatch(values, to: batch)
        let q32 = queries.asType(.float32), k32 = kAligned.asType(.float32), v32 = vAligned.asType(.float32)
        let sinks = alignedGlobalKV(globalKV, batch: batch)
        let cpuSinks = sinks.map {
            CPUGlobalKV(keys: $0.keys.asType(.float32).asArray(Float.self), values: $0.values.asType(.float32).asArray(Float.self))
        }
        let shape = CPUAttentionShape(
            batch: batch, qHeads: queries.dim(1), kvHeads: kAligned.dim(1), headDim: headDim, sinkTokens: sinks?.keys.dim(2) ?? 0, scale: scale
        )
        let out = cpuReferenceBrickAttention(
            queries: q32.asArray(Float.self), keys: k32.asArray(Float.self), values: v32.asArray(Float.self),
            shape: shape, plan: plan, globalKV: cpuSinks
        )
        return MLXArray(out, queries.shape).asType(queries.dtype)
    }

    package static func cpuReferenceBrickAttention(
        queries: [Float], keys: [Float], values: [Float],
        shape: CPUAttentionShape, plan: BrickSparsePlan, globalKV: CPUGlobalKV? = nil
    ) -> [Float] {
        let packed = CPUPackedQKV(
            q: cpuTransferBricks(queries, batch: shape.batch, heads: shape.qHeads, headDim: shape.headDim, plan: plan, pack: true),
            k: cpuTransferBricks(keys, batch: shape.batch, heads: shape.kvHeads, headDim: shape.headDim, plan: plan, pack: true),
            v: cpuTransferBricks(values, batch: shape.batch, heads: shape.kvHeads, headDim: shape.headDim, plan: plan, pack: true)
        )
        let packedTokens = plan.brickCount * plan.brickVolume
        var outPacked = [Float](repeating: 0, count: shape.batch * shape.qHeads * packedTokens * shape.headDim)
        let groupSize = max(1, shape.qHeads / max(1, shape.kvHeads))
        let scale = shape.scale ?? (1.0 / Float(max(1, shape.headDim)).squareRoot())
        for b in 0 ..< shape.batch {
            for qh in 0 ..< shape.qHeads {
                let task = CPUHeadTask(
                    bqh: b * shape.qHeads + qh, bkvh: b * shape.kvHeads + (qh / groupSize),
                    sinkTokens: shape.sinkTokens, headDim: shape.headDim, scale: scale
                )
                cpuAttendSingleHead(packed: packed, globalKV: globalKV, task: task, plan: plan, outPacked: &outPacked)
            }
        }
        return cpuTransferBricks(outPacked, batch: shape.batch, heads: shape.qHeads, headDim: shape.headDim, plan: plan, pack: false)
    }

    private static func cpuAttendSingleHead(
        packed: CPUPackedQKV, globalKV: CPUGlobalKV?,
        task: CPUHeadTask, plan: BrickSparsePlan, outPacked: inout [Float]
    ) {
        let d = task.headDim, bv = plan.brickVolume, packedLen = plan.brickCount * bv
        for qb in 0 ..< plan.brickCount {
            let nbrStart = Int(plan.csrRowOffsets[qb]), nbrEnd = Int(plan.csrRowOffsets[qb + 1])
            for qt in 0 ..< bv where plan.validTokenFlags[qb * bv + qt] != 0 {
                let qOff = (task.bqh * packedLen + qb * bv + qt) * d
                var state = CPUOnlineSoftmaxState(dim: d)
                for pos in nbrStart ..< nbrEnd {
                    let kb = Int(plan.csrColIndices[pos])
                    for kt in 0 ..< bv where plan.validTokenFlags[kb * bv + kt] != 0 {
                        let kvOff = (task.bkvh * packedLen + kb * bv + kt) * d
                        state.update(q: packed.q, qOff: qOff, k: packed.k, v: packed.v, kvOff: kvOff, scale: task.scale)
                    }
                }
                if let globalKV, task.sinkTokens > 0 {
                    for st in 0 ..< task.sinkTokens {
                        let kvOff = (task.bkvh * task.sinkTokens + st) * d
                        state.update(q: packed.q, qOff: qOff, k: globalKV.keys, v: globalKV.values, kvOff: kvOff, scale: task.scale)
                    }
                }
                let inv: Float = (state.rowSum.isNaN || state.rowMax.isNaN) ? .nan : (state.rowSum > 0 ? (1.0 / state.rowSum) : 0.0)
                for c in 0 ..< d { outPacked[qOff + c] = state.acc[c] * inv }
            }
        }
    }

    package static func cpuTransferBricks(
        _ src: [Float], batch: Int, heads: Int, headDim: Int, plan: BrickSparsePlan, pack: Bool
    ) -> [Float] {
        let bhCount = batch * heads, packedTokens = plan.brickCount * plan.brickVolume
        let gridTokens = plan.grid.tokenCount
        var out = [Float](repeating: 0, count: bhCount * (pack ? packedTokens : gridTokens) * headDim)
        for bh in 0 ..< bhCount {
            for t in 0 ..< plan.grid.frames {
                for y in 0 ..< plan.grid.height {
                    for x in 0 ..< plan.grid.width {
                        let gridTok = (t * plan.grid.height + y) * plan.grid.width + x
                        let brickSlot = plan.packedSlot(t: t, y: y, x: x)
                        let srcOff = (pack ? (bh * gridTokens + gridTok) : (bh * packedTokens + brickSlot)) * headDim
                        let dstOff = (pack ? (bh * packedTokens + brickSlot) : (bh * gridTokens + gridTok)) * headDim
                        for c in 0 ..< headDim { out[dstOff + c] = src[srcOff + c] }
                    }
                }
            }
        }
        return out
    }
}

private enum STA3DMetal {
    static let header = """
        #include <metal_simdgroup>
        template <int D, int E>
        inline void sta_attend_tile(
            int tile_len, int lane, bool check_valid, const threadgroup uint8_t* valid_tile,
            const thread float* qr, const threadgroup float* k_tile, const threadgroup float* v_tile,
            float attn_scale, thread float& row_max, thread float& row_sum, thread float* acc
        ) {
            for (int t_local = 0; t_local < tile_len; ++t_local) {
                if (check_valid && valid_tile[t_local] == 0) { continue; }
                float score = 0.0f;
                const int base_off = t_local * D;
                for (int e = 0; e < E; ++e) {
                    const int d = e * 32 + lane;
                    if (d < D) { score += qr[e] * k_tile[base_off + d]; }
                }
                score = simd_sum(score) * attn_scale;
                const bool bad = isnan(row_max) || isnan(score);
                const float next_max = bad ? NAN : max(row_max, score);
                const bool active = bad || (next_max != -INFINITY);
                const float alpha = bad ? NAN : ((row_max != -INFINITY && active) ? exp(row_max - next_max) : 0.0f);
                const float beta = bad ? NAN : ((active && score != -INFINITY) ? exp(score - next_max) : 0.0f);
                row_max = next_max;
                row_sum = row_sum * alpha + beta;
                for (int e = 0; e < E; ++e) {
                    const int d = e * 32 + lane;
                    if (d < D) { acc[e] = acc[e] * alpha + beta * v_tile[base_off + d]; }
                }
            }
        }
        """

    static let source = """
        constexpr int SG = 4, E = (D + 31) / 32, TILE_KV = (D <= 128) ? 16 : 8;
        const int tg_x = int(threadgroup_position_in_grid.x), bqh = int(threadgroup_position_in_grid.y);
        const int tid = int(thread_position_in_threadgroup.x);
        const int lane = int(thread_index_in_simdgroup), sg = int(simdgroup_index_in_threadgroup);
        const int q_heads = meta[0], kv_heads = meta[1];
        const int brick_count = meta[2], brick_vol = meta[3], sink_count = meta[4];
        const float attn_scale = scale_arr[0];
        const int q_tiles = (brick_vol + SG - 1) / SG;
        const int qb = tg_x / q_tiles, qt = (tg_x % q_tiles) * SG + sg;
        if (qb >= brick_count) { return; }
        const int bkvh = (bqh / q_heads) * kv_heads + ((bqh % q_heads) / (q_heads / kv_heads));
        const bool active_q = (qt < brick_vol) && (EXACT_GRID || (valid_flags[size_t(qb) * size_t(brick_vol) + size_t(qt)] != 0));
        const size_t q_base = ((size_t(bqh) * size_t(brick_count) + size_t(qb)) * size_t(brick_vol) + size_t(min(qt, brick_vol - 1))) * size_t(D);
        float qr[E], acc[E];
        for (int e = 0; e < E; ++e) {
            const int d = e * 32 + lane;
            qr[e] = (active_q && d < D) ? float(q[q_base + size_t(d)]) : 0.0f;
            acc[e] = 0.0f;
        }
        float row_max = -INFINITY, row_sum = 0.0f;
        threadgroup float k_tile[TILE_KV * D], v_tile[TILE_KV * D];
        threadgroup uint8_t valid_tile[TILE_KV];
        const int row_start = csr_offsets[qb], row_end = csr_offsets[qb + 1];
        for (int nbr_pos = row_start; nbr_pos < row_end; ++nbr_pos) {
            const int kb = csr_indices[nbr_pos];
            const size_t kv_brick_base = (size_t(bkvh) * size_t(brick_count) + size_t(kb)) * size_t(brick_vol);
            for (int kt_base = 0; kt_base < brick_vol; kt_base += TILE_KV) {
                const int tile_len = min(TILE_KV, brick_vol - kt_base);
                for (int idx = tid; idx < tile_len * D; idx += SG * 32) {
                    const size_t src = (kv_brick_base + size_t(kt_base + idx / D)) * size_t(D) + size_t(idx % D);
                    k_tile[idx] = float(k[src]);
                    v_tile[idx] = float(v[src]);
                }
                if (!EXACT_GRID && tid < tile_len) { valid_tile[tid] = valid_flags[size_t(kb) * size_t(brick_vol) + size_t(kt_base + tid)]; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (active_q) { sta_attend_tile<D, E>(tile_len, lane, !EXACT_GRID, valid_tile, qr, k_tile, v_tile, attn_scale, row_max, row_sum, acc); }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (HAS_SINKS) {
            const size_t sink_head_base = size_t(bkvh) * size_t(sink_count);
            for (int st_base = 0; st_base < sink_count; st_base += TILE_KV) {
                const int tile_len = min(TILE_KV, sink_count - st_base);
                for (int idx = tid; idx < tile_len * D; idx += SG * 32) {
                    const size_t src = (sink_head_base + size_t(st_base + idx / D)) * size_t(D) + size_t(idx % D);
                    k_tile[idx] = float(sink_k[src]);
                    v_tile[idx] = float(sink_v[src]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (active_q) { sta_attend_tile<D, E>(tile_len, lane, false, valid_tile, qr, k_tile, v_tile, attn_scale, row_max, row_sum, acc); }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (qt < brick_vol) {
            const float inv_sum = active_q ? ((isnan(row_sum) || isnan(row_max)) ? NAN : ((row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f)) : 0.0f;
            for (int e = 0; e < E; ++e) {
                const int d = e * 32 + lane;
                if (d < D) { out[q_base + size_t(d)] = static_cast<T>(active_q ? (acc[e] * inv_sum) : 0.0f); }
            }
        }
        """
}

private let sta3DBrickAttentionKernel = MLXFast.metalKernel(
    name: "sta_3d_brick_attention",
    inputNames: ["q", "k", "v", "sink_k", "sink_v", "csr_offsets", "csr_indices", "valid_flags", "meta", "scale_arr"],
    outputNames: ["out"],
    source: STA3DMetal.source,
    header: STA3DMetal.header
)
