import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

package final class PVQTIPLinear: Linear {
    let lutIndices: MLXArray
    let lutSigns: MLXArray
    let spec: QTIPLayerSpec
    let buffers: FusedKroneckerRHT.Pair
    let inSigns: MLXArray
    let inQ: MLXArray?
    let outSigns: MLXArray
    let outQ: MLXArray?
    let tileRows: Int
    let tileCols: Int
    let scaleDType: DType
    let tlutDType: DType
    @ParameterInfo(key: "row_scales") package var rowScales: MLXArray
    @ParameterInfo(key: "tlut") package var tlut: MLXArray

    package init(from layer: QTIPLinear) {
        let tRows = layer.outFeatures / 16
        let tCols = layer.inFeatures / 16
        let unpacked = Self.unpackTrellis(packed: layer.weight, rate: layer.rate, tileRows: tRows, tileCols: tCols)
        self.lutIndices = unpacked.indices
        self.lutSigns = unpacked.signs
        self.spec = QTIPLayerSpec(rate: layer.rate, rhtIn: layer.rhtIn, rhtOut: layer.rhtOut)
        self.buffers = FusedKroneckerRHT.Pair(inBuffers: layer.inBuffers, outBuffers: layer.outBuffers)
        self.inSigns = MLXArray(layer.rhtIn.signsValues)
        self.inQ = layer.rhtIn.qValues.map { MLXArray($0, [layer.rhtIn.outer, layer.rhtIn.outer]) }
        self.outSigns = MLXArray(layer.rhtOut.signsValues)
        self.outQ = layer.rhtOut.qValues.map { MLXArray($0, [layer.rhtOut.outer, layer.rhtOut.outer]) }
        self.tileRows = tRows
        self.tileCols = tCols
        self.scaleDType = layer.rowScales.dtype
        self.tlutDType = layer.tlut.dtype
        self._rowScales.wrappedValue = layer.rowScales
        self._tlut.wrappedValue = layer.tlut
        let frozenWeight = layer.weight + MLXArray(UInt32(0))
        eval(frozenWeight, unpacked.indices, unpacked.signs)
        super.init(weight: frozenWeight, bias: layer.bias)
        freeze()
        try? unfreeze(keys: ["row_scales", "tlut"], strict: false)
    }

    static func unpackTrellis(
        packed: MLXArray, rate: QTIPRate, tileRows: Int, tileCols: Int
    ) -> (indices: MLXArray, signs: MLXArray) {
        eval(packed)
        let rawWords = packed.asArray(UInt32.self)
        let totalTiles = tileRows * tileCols
        let words = rate.wordsPerTile
        let k = rate.stepBits
        var indices = [Int32](repeating: 0, count: totalTiles * 128)
        var signs = [Float](repeating: 1.0, count: totalTiles * 256)
        for tile in 0 ..< totalTiles {
            let base = tile * words
            let outBase = tile * 128
            for step in 0 ..< 128 {
                let bitPos = step * k
                let wIdx = bitPos >> 5
                let shift = bitPos & 31
                let lo = UInt64(rawWords[base + wIdx])
                let hi = UInt64(rawWords[base + ((wIdx + 1) % words)])
                let edge = UInt16(((lo | (hi << 32)) >> UInt64(shift)) & 0xFFFF)
                let code = QTIPCodebook.edgeCodeIndex(edge)
                indices[outBase + step] = Int32(code & 511)
                let sIdx = (outBase + step) * 2
                signs[sIdx] = (code & 512) != 0 ? -1.0 : 1.0
                signs[sIdx + 1] = (code & 1024) != 0 ? -1.0 : 1.0
            }
        }
        return (
            MLXArray(indices, [tileRows, tileCols, 128]),
            MLXArray(signs, [tileRows, tileCols, 128, 2])
        )
    }

    package func reconstructedRotatedWeight() -> MLXArray {
        let pairs = tlut.asType(.float32).take(lutIndices, axis: 0) * lutSigns
        let tiles = pairs.reshaped(tileRows, tileCols, 16, 16).transposed(0, 2, 1, 3)
        let scaled = tiles * rowScales.asType(.float32).reshaped(tileRows, 1, 1, 1)
        return scaled.reshaped(spec.rhtOut.dimension, spec.rhtIn.dimension)
    }

    override package func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xRot = spec.rhtIn.forwardFloat32(x, signs: inSigns, qMatrix: inQ)
        let yRot = matmul(xRot, reconstructedRotatedWeight().T)
        var y = spec.rhtOut.inverseFloat32(yRot, signs: outSigns, qMatrix: outQ).asType(x.dtype)
        if let bias {
            y = y + bias.asType(x.dtype)
        }
        return y
    }

    package func packedLayer() -> QTIPLinear {
        let clonedScales = (rowScales + MLXArray(0, dtype: rowScales.dtype)).asType(scaleDType)
        let clonedTlut = (tlut + MLXArray(0, dtype: tlut.dtype)).asType(tlutDType)
        eval(clonedScales, clonedTlut)
        return QTIPLinear(
            packedWeight: weight, rowScales: clonedScales, tlut: clonedTlut,
            bias: bias, spec: spec, buffers: buffers)
    }
}

private final class PVQTIPRotatedProbe: Linear {
    let spec: QTIPLayerSpec
    let inSigns: MLXArray
    let inQ: MLXArray?
    let outSigns: MLXArray
    let outQ: MLXArray?

    init(from layer: QTIPLinear) {
        self.spec = QTIPLayerSpec(rate: layer.rate, rhtIn: layer.rhtIn, rhtOut: layer.rhtOut)
        self.inSigns = MLXArray(layer.rhtIn.signsValues)
        self.inQ = layer.rhtIn.qValues.map { MLXArray($0, [layer.rhtIn.outer, layer.rhtIn.outer]) }
        self.outSigns = MLXArray(layer.rhtOut.signsValues)
        self.outQ = layer.rhtOut.qValues.map { MLXArray($0, [layer.rhtOut.outer, layer.rhtOut.outer]) }
        let denseRot = layer.decodedRotatedWeight(dtype: .float32)
        eval(denseRot)
        super.init(weight: denseRot, bias: layer.bias)
        freeze()
        try? unfreeze(keys: ["weight"], strict: false)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xRot = spec.rhtIn.forwardFloat32(x, signs: inSigns, qMatrix: inQ)
        let yRot = matmul(xRot, weight.T)
        var y = spec.rhtOut.inverseFloat32(yRot, signs: outSigns, qMatrix: outQ).asType(x.dtype)
        if let bias { y = y + bias.asType(x.dtype) }
        return y
    }
}

extension PVTuner {
    static let qtipDiscreteMinRelativeKLGain: Float = 0.02
    static let qtipDiscreteStepSizes: [Float] = [0.05, 0.015]

    static func runQTIPDiscreteGroup(
        _ model: any LanguageModel,
        blocks: Set<Int>,
        train: [PVTeacherBatch],
        val: [PVTeacherBatch],
        stepSeed: Int,
        config: PVTuneConfig
    ) throws -> Int {
        guard !train.isEmpty, !config.discreteFractions.isEmpty else { return 0 }
        let root = model as Module
        let qtipTargets: [(path: String, layer: QTIPLinear)] = root.namedModules().compactMap { path, mod in
            guard let idx = textBlockIndex(path), blocks.contains(idx), !path.contains("router"),
                  let ql = mod as? QTIPLinear else { return nil }
            return (path, ql)
        }
        guard !qtipTargets.isEmpty else { return 0 }
        let bCount = min(config.batchesPerDiscreteStep, train.count)
        let slice = (0 ..< bCount).map { train[(stepSeed * bCount + $0) % train.count] }
        let gateBatches = val.isEmpty ? slice : val
        let grads = try qtipRotatedGradients(model, targets: qtipTargets, minBlock: blocks.min() ?? 0, batches: slice)
        let baseValKL = meanKL(model, gateBatches)
        guard baseValKL > 1e-7 else { return 0 }
        let savedWeights = qtipTargets.map { ($0.path, $0.layer.weight + MLXArray(UInt32(0))) }
        eval(savedWeights.map(\.1))
        for stepSize in qtipDiscreteStepSizes {
            var updates: [(String, MLXArray)] = []
            var reencodedTiles = 0
            for (path, ql) in qtipTargets {
                guard let grad = grads[path] else { continue }
                let updated = reencodeQTIPLayer(ql, rotatedGrad: grad, stepSize: stepSize)
                updates.append(("\(path).weight", updated.packed))
                reencodedTiles += updated.tiles
            }
            guard !updates.isEmpty else { continue }
            root.update(parameters: ModuleParameters.unflattened(updates))
            eval(root)
            root.invalidateCompiledTraces()
            let candValKL = meanKL(model, gateBatches)
            if candValKL.isFinite && candValKL <= (1.0 - qtipDiscreteMinRelativeKLGain) * baseValKL {
                return reencodedTiles
            }
        }
        root.update(parameters: ModuleParameters.unflattened(savedWeights.map { ("\($0.0).weight", $0.1) }))
        eval(root)
        root.invalidateCompiledTraces()
        return 0
    }

    private static func reencodeQTIPLayer(
        _ layer: QTIPLinear, rotatedGrad: MLXArray, stepSize: Float
    ) -> (packed: MLXArray, tiles: Int) {
        let tRows = layer.outFeatures / 16
        let tCols = layer.inFeatures / 16
        let wRot = layer.decodedRotatedWeight(dtype: .float32) - stepSize * rotatedGrad
        let rawScales = layer.rowScales.asType(.float32)
        let mag = maximum(abs(rawScales), Float(1e-6))
        let scales = which(rawScales .< Float(0), -mag, mag).reshaped(tRows, 1, 1, 1)
        let normTiles = (wRot.reshaped(tRows, 16, tCols, 16) / scales)
            .transposed(0, 2, 1, 3)
            .reshaped(tRows * tCols, 256)
        eval(normTiles)
        let encoded = QTIPMetal.encodeTiles(normTiles, tlut: layer.tlut, rate: layer.rate)
        let reshaped = encoded.packed.reshaped(tRows, tCols * layer.rate.wordsPerTile)
        eval(reshaped)
        return (reshaped, tRows * tCols)
    }

    private static func qtipRotatedGradients(
        _ model: any LanguageModel,
        targets: [(path: String, layer: QTIPLinear)],
        minBlock: Int,
        batches: [PVTeacherBatch]
    ) throws -> [String: MLXArray] {
        let root = model as Module
        root.freeze()
        defer {
            for t in targets { try? swapModule(t.path, with: t.layer, in: root) }
            root.freeze()
            root.train(false)
        }
        for t in targets {
            try swapModule(t.path, with: PVQTIPRotatedProbe(from: t.layer), in: root)
        }
        setTrainingSuffix(root, from: minBlock)
        var accum: [String: MLXArray] = [:]
        let gradFn = valueAndGrad(model: root) { _, arrays in
            [klLossArrays(model, inputs: arrays[0], ids: arrays[1], tLogp: arrays[2])]
        }
        for batch in batches {
            try Task.checkCancellation()
            let tok = MLXArray(batch.tokens)[.newAxis]
            let ids = MLXArray(batch.topIDs).reshaped(batch.seqLen, batch.topK)
            let lp = MLXArray(batch.topLogprobs).reshaped(batch.seqLen, batch.topK)
            let (_, grads) = gradFn(root, [tok, ids, lp])
            let flat = Dictionary(uniqueKeysWithValues: grads.flattened())
            for t in targets {
                guard let g = flat["\(t.path).weight"]?.asType(.float32) else { continue }
                accum[t.path] = accum[t.path].map { $0 + g } ?? g
            }
            eval(Array(accum.values))
        }
        let inv = Float(1.0 / Double(max(batches.count, 1)))
        let averaged = accum.mapValues { $0 * inv }
        eval(Array(averaged.values))
        return averaged
    }
}
