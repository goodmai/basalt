import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

package struct QTIPConfig: Sendable {
    package var rate: QTIPRate
    package var damping: Float = 0.01
    package var calibrationBatches = 8
    package var calibrationSeqLen = 256
    package var blockGroupSize = 8
    package var minDimension = 64
    package var maxSourceBits = 8
    package var seed: UInt64 = 0x5154_4950_2026_0409

    package init(rate: QTIPRate) {
        self.rate = rate
    }
}

package struct QTIPQuantizeSummary: Sendable {
    package var rate: QTIPRate
    package var quantizedLayers: Int
    package var quantizedWeights: Int
    package var calibrationTokens: Int
    package var wallSeconds: Double
}

final class QTIPActivationBox {
    var samples: [MLXArray] = []
    var capturedTokens = 0
    let maxTokens: Int
    let inFeatures: Int

    init(inFeatures: Int, maxTokens: Int) {
        self.inFeatures = inFeatures
        self.maxTokens = maxTokens
    }

    func record(_ x: MLXArray) {
        guard capturedTokens < maxTokens else { return }
        let rows = x.size / inFeatures
        guard rows > 0 else { return }
        let takeRows = min(rows, maxTokens - capturedTokens)
        let flat = x.reshaped(rows, inFeatures)[0 ..< takeRows].asType(.bfloat16)
        eval(flat)
        samples.append(flat)
        capturedTokens += takeRows
    }
}

final class QTIPCalibrationTap: Linear {
    let wrapped: Module
    let box: QTIPActivationBox

    init(wrapping module: Module, inFeatures: Int, box: QTIPActivationBox) {
        self.wrapped = module
        self.box = box
        super.init(weight: MLXArray.zeros([1, 1], dtype: .float16), bias: nil)
        freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        box.record(x)
        if let ql = wrapped as? QuantizedLinear { return ql(x) }
        if let lin = wrapped as? Linear { return lin(x) }
        preconditionFailure("Unsupported tapped module: \(type(of: wrapped))")
    }
}

struct QTIPTargetEntry {
    var path: String
    var groupKey: String
    var module: Module
    var outFeatures: Int
    var inFeatures: Int
}

package enum QTIPQuantizer {
    package static func upperCholesky(_ hessianRot: MLXArray, damping: Float) -> (u: MLXArray, diagInv: MLXArray) {
        let n = hessianRot.dim(0)
        let meanDiag = diagonal(hessianRot).mean()
        let damp = maximum(Float(1e-6), damping * meanDiag)
        let damped = hessianRot + eye(n, dtype: .float32) * damp
        let rev = MLXArray((0 ..< n).reversed().map(Int32.init))
        let hRev = damped.take(rev, axis: 0).take(rev, axis: 1)
        eval(hRev)
        let lRev = MLXLinalg.cholesky(hRev, upper: false, stream: .cpu)
        eval(lRev)
        let u = lRev.take(rev, axis: 0).take(rev, axis: 1)
        let strips = n / 16
        let diagBlocks = stacked((0 ..< strips).map { g in
            let c0 = g * 16
            return u[c0 ..< (c0 + 16), c0 ..< (c0 + 16)]
        }, axis: 0)
        eval(diagBlocks)
        let diagInv = MLXLinalg.inv(diagBlocks, stream: .cpu)
        eval(u, diagInv)
        return (u, diagInv)
    }

    package static func rotatedHessian(samples: [MLXArray], rhtIn: KroneckerRHT) -> MLXArray? {
        guard !samples.isEmpty else { return nil }
        let d = rhtIn.dimension
        let signs = MLXArray(rhtIn.signsValues)
        let qMat = rhtIn.qValues.map { MLXArray($0, [rhtIn.outer, rhtIn.outer]) }
        var h = MLXArray.zeros([d, d], dtype: .float32)
        var count = 0
        for sample in samples {
            let xRot = rhtIn.forwardFloat32(sample, signs: signs, qMatrix: qMat)
            h = h + matmul(xRot.T, xRot)
            count += sample.dim(0)
            eval(h)
        }
        guard count > 0 else { return nil }
        let scaled = h * (Float(1.0) / Float(count))
        eval(scaled)
        return scaled
    }

    package static func prepareNormalizedWeight(
        weight: MLXArray, rhtIn: KroneckerRHT, rhtOut: KroneckerRHT
    ) -> (wNorm: MLXArray, rowScales: MLXArray) {
        let (m, n) = (rhtOut.dimension, rhtIn.dimension)
        let tileRows = m / 16
        let inSigns = MLXArray(rhtIn.signsValues)
        let outSigns = MLXArray(rhtOut.signsValues)
        let inQ = rhtIn.qValues.map { MLXArray($0, [rhtIn.outer, rhtIn.outer]) }
        let outQ = rhtOut.qValues.map { MLXArray($0, [rhtOut.outer, rhtOut.outer]) }
        let wIn = rhtIn.forwardFloat32(weight, signs: inSigns, qMatrix: inQ)
        let wRot = rhtOut.forwardFloat32(wIn.T, signs: outSigns, qMatrix: outQ).T
        let grouped = wRot.reshaped(tileRows, 16 * n)
        let rowScalesF16 = maximum(Float(1e-8), sqrt((grouped * grouped).mean(axis: -1))).asType(.float16)
        let scaleDiv = maximum(Float(1e-8), rowScalesF16.asType(.float32)).reshaped(tileRows, 1, 1)
        let wNorm = (wRot.reshaped(tileRows, 16, n) / scaleDiv).reshaped(m, n)
        eval(wNorm, rowScalesF16)
        return (wNorm, rowScalesF16)
    }

    package static func encodeNormalizedWeight(
        wNorm: MLXArray,
        choleskyFactors: (u: MLXArray, diagInv: MLXArray)?,
        tlut: MLXArray,
        rate: QTIPRate
    ) -> MLXArray {
        let (tileRows, tileCols) = (wNorm.dim(0) / 16, wNorm.dim(1) / 16)
        if let factors = choleskyFactors {
            return encodeBlockLDLQ(wNorm: wNorm, factors: factors, tlut: tlut, rate: rate)
        }
        let tiles = wNorm.reshaped(tileRows, 16, tileCols, 16).transposed(0, 2, 1, 3).reshaped(tileRows * tileCols, 256)
        eval(tiles)
        let (pAll, _) = QTIPMetal.encodeTiles(tiles, tlut: tlut, rate: rate)
        let packed = pAll.reshaped(tileRows, tileCols * rate.wordsPerTile)
        eval(packed)
        return packed
    }

    package static func quantizeMatrix(
        weight: MLXArray,
        bias: MLXArray?,
        spec: QTIPLayerSpec,
        choleskyFactors: (u: MLXArray, diagInv: MLXArray)?,
        tlut: MLXArray
    ) -> QTIPLinear {
        let (wNorm, rowScalesF16) = prepareNormalizedWeight(weight: weight, rhtIn: spec.rhtIn, rhtOut: spec.rhtOut)
        let tlutF16 = tlut.asType(.float16)
        eval(tlutF16)
        let packedWeight = encodeNormalizedWeight(
            wNorm: wNorm, choleskyFactors: choleskyFactors, tlut: tlutF16, rate: spec.rate)
        let layer = QTIPLinear(packedWeight: packedWeight, rowScales: rowScalesF16, tlut: tlutF16, bias: bias, spec: spec)
        eval(layer)
        return layer
    }

    static func encodeBlockLDLQ(
        wNorm: MLXArray,
        factors: (u: MLXArray, diagInv: MLXArray),
        tlut: MLXArray,
        rate: QTIPRate,
        panelStrips: Int = 16
    ) -> MLXArray {
        let (m, n) = (wNorm.dim(0), wNorm.dim(1))
        let (tileRows, tileCols) = (m / 16, n / 16)
        let stepStrips = max(1, panelStrips)
        var remAccum = MLXArray.zeros([m, n], dtype: .float32)
        var strips: [MLXArray] = []
        strips.reserveCapacity(tileCols)
        for pStart in stride(from: 0, to: tileCols, by: stepStrips) {
            let pEnd = min(tileCols, pStart + stepStrips)
            let (colStart, colEnd) = (pStart * 16, pEnd * 16)
            let panelCols = colEnd - colStart
            var remPanel = remAccum[0..., 0 ..< panelCols]
            let uPanel = contiguous(factors.u[colStart ..< colEnd, colStart ..< colEnd])
            eval(remPanel, uPanel)
            var errStrips: [MLXArray] = []
            errStrips.reserveCapacity(pEnd - pStart)
            for k in 0 ..< (pEnd - pStart) {
                let g = pStart + k
                let (lc0, lc1) = (k * 16, (k + 1) * 16)
                let wOrigStrip = wNorm[0..., (colStart + lc0) ..< (colStart + lc1)]
                let wEffStrip = wOrigStrip + matmul(remPanel[0..., 0 ..< 16], factors.diagInv[g])
                let (pStrip, hTiles) = QTIPMetal.encodeTiles(wEffStrip.reshaped(tileRows, 256), tlut: tlut, rate: rate)
                strips.append(pStrip)
                let errStrip = wOrigStrip - hTiles.reshaped(m, 16)
                errStrips.append(errStrip)
                if lc1 < panelCols {
                    remPanel = remPanel[0..., 16...] + matmul(errStrip, uPanel[lc0 ..< lc1, lc1 ..< panelCols])
                    if (k + 1).isMultiple(of: 4) { eval(remPanel, pStrip) }
                }
            }
            if colEnd < n {
                let errPanel = concatenated(errStrips, axis: 1)
                remAccum = remAccum[0..., panelCols...] + matmul(errPanel, factors.u[colStart ..< colEnd, colEnd ..< n])
                eval(remAccum)
            }
        }
        let packed = concatenated(strips, axis: 1)
        eval(packed)
        return packed
    }
}

extension QTIPQuantizer {
    static func sharedInputGroupKey(path: String, inFeatures: Int) -> String {
        let parts = path.split(separator: ".")
        guard let leaf = parts.last else { return path }
        let parent = parts.dropLast().joined(separator: ".")
        switch leaf {
        case "q_proj", "k_proj", "v_proj": return "\(parent)._qkv_in_\(inFeatures)"
        case "in_proj_qkv", "in_proj_z": return "\(parent)._gdn_in_\(inFeatures)"
        case "gate_proj", "up_proj": return "\(parent)._mlp_in_\(inFeatures)"
        default: return "\(path)_\(inFeatures)"
        }
    }

    static func hashString(_ text: String, seed: UInt64) -> UInt64 {
        var h = seed ^ 0xCBF2_9CE4_8422_2325
        for byte in text.utf8 {
            h ^= UInt64(byte)
            h &*= 0x0000_0100_0000_01B3
        }
        return h
    }

    static func collectBlockTargets(_ root: Module, blocks: Set<Int>, config: QTIPConfig) -> [QTIPTargetEntry] {
        var out: [QTIPTargetEntry] = []
        for (path, mod) in root.namedModules() {
            guard let bIdx = PVTuner.textBlockIndex(path), blocks.contains(bIdx),
                  !path.contains("router"), !path.hasSuffix("in_proj_b"), !path.hasSuffix("in_proj_a") else { continue }
            let dims: (out: Int, inp: Int)?
            if let ql = mod as? QuantizedLinear, ql.globalScale == nil, ql.bits <= config.maxSourceBits {
                dims = (ql.weight.dim(0), ql.weight.dim(1) * 32 / ql.bits)
            } else if type(of: mod) == Linear.self, let lin = mod as? Linear {
                dims = (lin.weight.dim(0), lin.weight.dim(1))
            } else {
                dims = nil
            }
            guard let dims, dims.out >= config.minDimension, dims.inp >= config.minDimension,
                  KroneckerRHT.supports(dims.out), KroneckerRHT.supports(dims.inp) else { continue }
            out.append(QTIPTargetEntry(
                path: path, groupKey: sharedInputGroupKey(path: path, inFeatures: dims.inp),
                module: mod, outFeatures: dims.out, inFeatures: dims.inp))
        }
        return out
    }

    static func extractDenseWeight(_ module: Module) -> (weight: MLXArray, bias: MLXArray?) {
        if let ql = module as? QuantizedLinear {
            let dense = dequantized(
                ql.weight, scales: ql.scales, biases: ql.biases,
                groupSize: ql.groupSize, bits: ql.bits, mode: ql.mode)
            eval(dense)
            return (dense, ql.bias)
        }
        if let lin = module as? Linear {
            return (lin.weight, lin.bias)
        }
        preconditionFailure("Unsupported QTIP target module: \(type(of: module))")
    }
}

extension QTIPQuantizer {
    package static func quantizeModel(
        _ model: any LanguageModel,
        calibrationTokenBatches: [[Int32]],
        config: QTIPConfig,
        onStep: ((String) -> Void)? = nil
    ) throws -> QTIPQuantizeSummary {
        let startClock = ContinuousClock.now
        let root = model as Module
        root.train(false)
        let blockGroups = groupedBlocks(in: root, stepSize: max(1, config.blockGroupSize))
        let tlut = MLXArray(QTIPCodebook.baseLUTValues(rate: config.rate), [512, 2]).asType(.float16)
        eval(tlut)
        let maxTokens = max(1, calibrationTokenBatches.reduce(0) { $0 + $1.count })
        var (totalLayers, totalWeights) = (0, 0)
        for (gIdx, group) in blockGroups.enumerated() {
            try Task.checkCancellation()
            let targets = collectBlockTargets(root, blocks: Set(group), config: config)
            guard !targets.isEmpty else { continue }
            let boxes = try captureGroupActivations(
                model: model, root: root, targets: targets, batches: calibrationTokenBatches, maxTokens: maxTokens)
            let groupStats = try quantizeCapturedTargets(
                root: root, targets: targets, boxes: boxes, config: config, tlut: tlut, onStep: onStep)
            totalLayers += groupStats.layers
            totalWeights += groupStats.weights
            Memory.clearCache()
            onStep?("qtip(\(config.rate.label)bpw) group \(gIdx + 1)/\(blockGroups.count) layers=\(totalLayers)")
        }
        MLXInferenceEngine.settle(root)
        Memory.clearCache()
        let elapsed = ContinuousClock.now - startClock
        let secs = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        return QTIPQuantizeSummary(
            rate: config.rate, quantizedLayers: totalLayers, quantizedWeights: totalWeights,
            calibrationTokens: calibrationTokenBatches.reduce(0) { $0 + $1.count }, wallSeconds: secs)
    }

    static func groupedBlocks(in root: Module, stepSize: Int) -> [[Int]] {
        let allBlocks = Array(Set(root.namedModules().compactMap { PVTuner.textBlockIndex($0.0) }).sorted())
        return stride(from: 0, to: allBlocks.count, by: stepSize).map {
            Array(allBlocks[$0 ..< min(allBlocks.count, $0 + stepSize)])
        }
    }

    static func captureGroupActivations(
        model: any LanguageModel,
        root: Module,
        targets: [QTIPTargetEntry],
        batches: [[Int32]],
        maxTokens: Int
    ) throws -> [String: QTIPActivationBox] {
        guard !batches.isEmpty else { return [:] }
        var boxes: [String: QTIPActivationBox] = [:]
        var tapped: [QTIPTargetEntry] = []
        defer {
            for entry in tapped { try? PVTuner.swapModule(entry.path, with: entry.module, in: root) }
        }
        for entry in targets where boxes[entry.groupKey] == nil {
            let box = QTIPActivationBox(inFeatures: entry.inFeatures, maxTokens: maxTokens)
            try PVTuner.swapModule(
                entry.path, with: QTIPCalibrationTap(wrapping: entry.module, inFeatures: entry.inFeatures, box: box), in: root)
            boxes[entry.groupKey] = box
            tapped.append(entry)
        }
        for tokens in batches {
            let out = model(LMInput.Text(tokens: MLXArray(tokens)[.newAxis]), cache: nil, state: nil)
            eval(out.logits)
        }
        return boxes
    }

    private static func quantizeCapturedTargets(
        root: Module,
        targets: [QTIPTargetEntry],
        boxes: [String: QTIPActivationBox],
        config: QTIPConfig,
        tlut: MLXArray,
        onStep: ((String) -> Void)?
    ) throws -> (layers: Int, weights: Int) {
        var factorCache: [String: (rhtIn: KroneckerRHT, cholesky: (u: MLXArray, diagInv: MLXArray)?)] = [:]
        var remaining = Dictionary(targets.map { ($0.groupKey, 1) }, uniquingKeysWith: +)
        var (layers, weights) = (0, 0)
        for entry in targets {
            try Task.checkCancellation()
            guard let inEntry = resolveInputFactor(entry, boxes: boxes, config: config, cache: &factorCache) else { continue }
            remaining[entry.groupKey, default: 1] -= 1
            let releaseFactor = remaining[entry.groupKey, default: 0] <= 0
            if releaseFactor { factorCache.removeValue(forKey: entry.groupKey) }
            let outSeed = hashString(entry.path + ".out", seed: config.seed)
            guard let rhtOut = KroneckerRHT(dimension: entry.outFeatures, seed: outSeed) else { continue }
            let (dense, bias) = extractDenseWeight(entry.module)
            let spec = QTIPLayerSpec(rate: config.rate, rhtIn: inEntry.rhtIn, rhtOut: rhtOut)
            let qLayer = quantizeMatrix(
                weight: dense, bias: bias, spec: spec, choleskyFactors: inEntry.cholesky, tlut: tlut)
            try PVTuner.swapModule(entry.path, with: qLayer, in: root)
            layers += 1
            weights += entry.outFeatures * entry.inFeatures
            if releaseFactor { Memory.clearCache() }
            if layers.isMultiple(of: 8) {
                Memory.clearCache()
                onStep?("qtip(\(config.rate.label)bpw) progress \(layers)/\(targets.count) in-group (\(entry.path))")
            }
        }
        factorCache.removeAll()
        return (layers, weights)
    }

    static func resolveInputFactor(
        _ entry: QTIPTargetEntry,
        boxes: [String: QTIPActivationBox],
        config: QTIPConfig,
        cache: inout [String: (rhtIn: KroneckerRHT, cholesky: (u: MLXArray, diagInv: MLXArray)?)]
    ) -> (rhtIn: KroneckerRHT, cholesky: (u: MLXArray, diagInv: MLXArray)?)? {
        if let cached = cache[entry.groupKey] { return cached }
        Memory.clearCache()
        let inSeed = hashString(entry.groupKey, seed: config.seed)
        guard let rhtIn = KroneckerRHT(dimension: entry.inFeatures, seed: inSeed) else { return nil }
        let hRot = boxes[entry.groupKey].flatMap { rotatedHessian(samples: $0.samples, rhtIn: rhtIn) }
        boxes[entry.groupKey]?.samples.removeAll()
        let chol = hRot.map { upperCholesky($0, damping: config.damping) }
        let resolved = (rhtIn, chol)
        cache[entry.groupKey] = resolved
        return resolved
    }
}
