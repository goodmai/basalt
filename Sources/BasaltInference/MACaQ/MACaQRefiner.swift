import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

package enum MACaQTargetMode: Sendable, Equatable {
    case affine(bits: Int, groupSize: Int)
    case mxfp4

    package var bits: Int {
        switch self {
        case .affine(let bits, _): bits
        case .mxfp4: 4
        }
    }

    package var groupSize: Int {
        switch self {
        case .affine(_, let groupSize): groupSize
        case .mxfp4: 32
        }
    }

    package var quantizationMode: QuantizationMode {
        switch self {
        case .affine: .affine
        case .mxfp4: .mxfp4
        }
    }

    package var label: String {
        switch self {
        case .affine(let bits, let groupSize): "affine:\(bits):\(groupSize)"
        case .mxfp4: "mxfp4"
        }
    }

    package static func parse(_ raw: String) throws(WeightConverter.Failure) -> MACaQTargetMode {
        if raw == "mxfp4" { return .mxfp4 }
        let parts = raw.split(separator: ":")
        guard parts.count == 3, parts[0] == "affine",
              let bits = Int(parts[1]), [2, 3, 4].contains(bits),
              let group = Int(parts[2]), [32, 64, 128].contains(group)
        else {
            throw .options("--experimental-macaq takes affine:<2|3|4>:<32|64|128> or mxfp4, got '\(raw)'")
        }
        return .affine(bits: bits, groupSize: group)
    }
}

package struct MACaQConfig: Sendable, Equatable {
    package var calibrationBatches = 8
    package var calibrationSeqLen = 256
    package var blockGroupSize = 8
    package var maxHessianBytes = 2_400_000_000
    package var maxFlipIterations = 128
    package var ridge: Float = 1.0
    package var alpha: Float = 0.75
    package var hadamardBlock = 64
    package var minExpertTokens = 8
    package var maxSwitchSamples = 512
    package var screenMXFP4 = true

    package init() {}
}

package struct MACaQSummary: Sendable, Equatable {
    package var refinedLayers: Int
    package var totalFlips: Int
    package var calibrationTokens: Int
    package var meanLossRatio: Float
    package var rewrittenTensors: Int
    package var wallSeconds: Double

    package init(
        refinedLayers: Int,
        totalFlips: Int,
        calibrationTokens: Int,
        meanLossRatio: Float,
        rewrittenTensors: Int = 0,
        wallSeconds: Double
    ) {
        self.refinedLayers = refinedLayers
        self.totalFlips = totalFlips
        self.calibrationTokens = calibrationTokens
        self.meanLossRatio = meanLossRatio
        self.rewrittenTensors = rewrittenTensors
        self.wallSeconds = wallSeconds
    }
}

final class MACaQHessianBox {
    let inFeatures: Int
    var sumXXT: MLXArray
    var tokenCount = 0

    init(inFeatures: Int) {
        self.inFeatures = inFeatures
        self.sumXXT = MLXArray.zeros([inFeatures, inFeatures], dtype: .float32)
        eval(self.sumXXT)
    }

    func record(_ x: MLXArray) {
        let rows = inFeatures > 0 ? x.size / inFeatures : 0
        guard rows > 0 else { return }
        let xf = x.reshaped(rows, inFeatures).asType(.float32)
        sumXXT = sumXXT + matmul(xf.T, xf)
        tokenCount += rows
        eval(sumXXT)
    }

    func meanHessian() -> MLXArray? {
        guard tokenCount > 0 else { return nil }
        let meanH = sumXXT * (Float(1.0) / Float(tokenCount))
        eval(meanH)
        return meanH
    }
}

final class MACaQLinearTap: Linear {
    let wrapped: Module
    let box: MACaQHessianBox

    init(wrapping module: Module, box: MACaQHessianBox) {
        self.wrapped = module
        self.box = box
        super.init(weight: MLXArray.zeros([1, 1], dtype: .float16), bias: nil)
        freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        box.record(x)
        if let nf4 = wrapped as? NF4Linear { return nf4(x) }
        if let ql = wrapped as? QuantizedLinear { return ql(x) }
        if let lin = wrapped as? Linear { return lin(x) }
        preconditionFailure("Unsupported MACaQ tapped Linear: \(type(of: wrapped))")
    }
}

final class MACaQSwitchReservoir {
    let inFeatures: Int
    let numExperts: Int
    let maxRows: Int
    var routedByExpert: [[MLXArray]]
    var countsByExpert: [Int]
    var pool: [MLXArray] = []
    var poolCount = 0

    init(inFeatures: Int, numExperts: Int, maxRows: Int) {
        self.inFeatures = inFeatures
        self.numExperts = numExperts
        self.maxRows = maxRows
        self.routedByExpert = Array(repeating: [], count: numExperts)
        self.countsByExpert = Array(repeating: 0, count: numExperts)
    }

    func record(_ x: MLXArray, indices: MLXArray) {
        guard inFeatures > 0, x.size >= inFeatures, indices.size > 0 else { return }
        let xRows = x.size / inFeatures
        let xFlat = x.reshaped(xRows, inFeatures).asType(.bfloat16)
        let idxFlat = indices.reshaped(-1).asType(.int32)
        eval(xFlat, idxFlat)
        if poolCount < 128 {
            let take = min(xRows, 128 - poolCount)
            pool.append(xFlat[0 ..< take])
            poolCount += take
        }
        let ids = idxFlat.asArray(Int32.self)
        if xRows == ids.count {
            appendDirect(xFlat: xFlat, ids: ids)
        } else if indices.ndim >= 2, xRows * indices.dim(-1) == ids.count {
            appendBroadcast(xFlat: xFlat, ids: ids, topK: indices.dim(-1))
        }
    }

    private func appendDirect(xFlat: MLXArray, ids: [Int32]) {
        for (rowIdx, rawExp) in ids.enumerated() {
            let e = Int(rawExp)
            guard e >= 0, e < numExperts, countsByExpert[e] < maxRows else { continue }
            routedByExpert[e].append(xFlat[rowIdx ..< (rowIdx + 1)])
            countsByExpert[e] += 1
        }
    }

    private func appendBroadcast(xFlat: MLXArray, ids: [Int32], topK: Int) {
        let tokens = ids.count / max(1, topK)
        for t in 0 ..< tokens {
            for k in 0 ..< topK {
                let e = Int(ids[t * topK + k])
                guard e >= 0, e < numExperts, countsByExpert[e] < maxRows else { continue }
                routedByExpert[e].append(xFlat[t ..< (t + 1)])
                countsByExpert[e] += 1
            }
        }
    }

    func expertHessian(expert: Int, minTokens: Int, routerWeight: MLXArray?) -> MLXArray? {
        if expert >= 0, expert < numExperts, countsByExpert[expert] >= minTokens {
            let cat = concatenated(routedByExpert[expert], axis: 0).asType(.float32)
            let h = matmul(cat.T, cat) * (Float(1.0) / Float(max(1, cat.dim(0))))
            eval(h)
            return h
        }
        guard !pool.isEmpty else { return nil }
        let xp = concatenated(pool, axis: 0).asType(.float32)
        let rows = xp.dim(0)
        if let rw = routerWeight, rw.ndim == 2, rw.dim(0) == numExperts, rw.dim(1) == inFeatures {
            let probs = softmax(matmul(xp, rw.asType(.float32).T), axis: -1)[0..., expert]
            let total = maximum(probs.sum(), Float(1e-12))
            let weights = sqrt(probs * (Float(rows) / total)).expandedDimensions(axis: 1)
            let xw = xp * weights
            let h = matmul(xw.T, xw) * (Float(1.0) / Float(max(1, rows)))
            eval(h)
            return h
        }
        let h = matmul(xp.T, xp) * (Float(1.0) / Float(max(1, rows)))
        eval(h)
        return h
    }
}

final class MACaQSwitchTap: SwitchLinear {
    let wrapped: Module
    let reservoir: MACaQSwitchReservoir

    init(wrapping module: Module, reservoir: MACaQSwitchReservoir) {
        self.wrapped = module
        self.reservoir = reservoir
        let dummy = MLXArray.zeros([1, 1, 1], dtype: .float16)
        super.init(inputDims: 1, outputDims: 1, numExperts: 1, weight: dummy, bias: nil)
        freeze()
    }

    override func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false) -> MLXArray {
        reservoir.record(x, indices: indices)
        if let nf4 = wrapped as? NF4SwitchLinear { return nf4(x, indices, sortedIndices: sortedIndices) }
        if let msw = wrapped as? MACaQSwitchLinear { return msw(x, indices, sortedIndices: sortedIndices) }
        if let qsw = wrapped as? QuantizedSwitchLinear { return qsw(x, indices, sortedIndices: sortedIndices) }
        if let sw = wrapped as? SwitchLinear { return sw(x, indices, sortedIndices: sortedIndices) }
        preconditionFailure("Unsupported MACaQ tapped SwitchLinear: \(type(of: wrapped))")
    }
}

package enum MACaQRefiner {
    struct BlockTargets {
        var linears: [LinearEntry]
        var switches: [SwitchEntry]
    }
    struct LinearEntry {
        var path: String
        var groupKey: String
        var module: Module
        var outFeatures: Int
        var inFeatures: Int
    }

    struct SwitchEntry {
        var path: String
        var routerPath: String
        var module: Module
        var numExperts: Int
        var outFeatures: Int
        var inFeatures: Int
    }

    static func groupedBlocks(in root: Module, stepSize: Int, maxHessianBytes: Int = 3_600_000_000) -> [[Int]] {
        var widthsByBlock: [Int: Set<String>] = [:]
        var bytesByBlock: [Int: Int] = [:]
        for (path, mod) in root.namedModules() {
            guard let bIdx = PVTuner.textBlockIndex(path), !path.contains("router") else { continue }
            let inF: Int
            if let ql = mod as? QuantizedLinear {
                inF = (ql.weight.dim(1) * 32) / max(1, ql.bits)
            } else if let nf4 = mod as? NF4Linear {
                inF = nf4.shape.1
            } else if let lin = mod as? Linear, type(of: mod) == Linear.self {
                inF = lin.shape.1
            } else {
                continue
            }
            let gKey = QTIPQuantizer.sharedInputGroupKey(path: path, inFeatures: inF, sharedMoERHT: false)
            if widthsByBlock[bIdx, default: []].insert(gKey).inserted {
                bytesByBlock[bIdx, default: 0] += inF * inF * 4
            }
        }
        let blocks = Array(Set(root.namedModules().compactMap { PVTuner.textBlockIndex($0.0) }).sorted())
        let maxStep = max(1, stepSize)
        var groups: [[Int]] = []
        var cur: [Int] = []
        var curBytes = 0
        for b in blocks {
            let cost = bytesByBlock[b] ?? 0
            if !cur.isEmpty, cur.count >= maxStep || (curBytes + cost > maxHessianBytes) {
                groups.append(cur)
                cur = []
                curBytes = 0
            }
            cur.append(b)
            curBytes += cost
        }
        if !cur.isEmpty { groups.append(cur) }
        return groups
    }

    static func routerWeight(in root: Module, for switchPath: String) -> MLXArray? {
        let parts = switchPath.split(separator: ".")
        guard let mlpIdx = parts.lastIndex(of: "mlp") else { return nil }
        let prefix = parts[0 ... mlpIdx].joined(separator: ".")
        let named = Dictionary(root.namedModules(), uniquingKeysWith: { a, _ in a })
        for cand in ["\(prefix).gate", "\(prefix).router"] {
            if let mod = named[cand] {
                if let lin = mod as? Linear, !(mod is QuantizedLinear) { return lin.weight }
                if let ql = mod as? QuantizedLinear { return QTIPQuantizer.extractDenseWeight(ql).weight }
            }
        }
        return nil
    }

    static func captureBlockHessians(
        model: any LanguageModel,
        root: Module,
        targets: BlockTargets,
        batches: [[Int32]],
        maxSwitchRows: Int
    ) throws -> (linear: [String: MLXArray], switches: [String: MACaQSwitchReservoir]) {
        var linearBoxes: [String: MACaQHessianBox] = [:]
        var switchBoxes: [String: MACaQSwitchReservoir] = [:]
        var swapped: [(String, Module)] = []
        defer {
            for (path, orig) in swapped { try? PVTuner.swapModule(path, with: orig, in: root) }
        }
        var tappedGroups: Set<String> = []
        for item in targets.linears where tappedGroups.insert(item.groupKey).inserted {
            let box = MACaQHessianBox(inFeatures: item.inFeatures)
            linearBoxes[item.groupKey] = box
            try PVTuner.swapModule(item.path, with: MACaQLinearTap(wrapping: item.module, box: box), in: root)
            swapped.append((item.path, item.module))
        }
        for item in targets.switches {
            let res = MACaQSwitchReservoir(
                inFeatures: item.inFeatures, numExperts: item.numExperts, maxRows: maxSwitchRows)
            switchBoxes[item.path] = res
            try PVTuner.swapModule(item.path, with: MACaQSwitchTap(wrapping: item.module, reservoir: res), in: root)
            swapped.append((item.path, item.module))
        }
        guard !swapped.isEmpty else { return ([:], [:]) }
        for tokens in batches where !tokens.isEmpty {
            let out = model(LMInput.Text(tokens: MLXArray(tokens)[.newAxis]), cache: nil, state: nil)
            eval(out.logits)
        }
        var resolved: [String: MLXArray] = [:]
        for (key, box) in linearBoxes {
            if let h = box.meanHessian() { resolved[key] = h }
        }
        return (resolved, switchBoxes)
    }
}
