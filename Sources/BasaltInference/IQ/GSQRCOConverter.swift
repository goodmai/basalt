import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

package struct GSQRCOConfig: Sendable {
    package var targetBPW: Double
    package var calibrationBatches = 4
    package var calibrationSeqLen = 128
    package var minDimension = 32
    package var allowedTypes: [GGMLType] = GSQRCOBlockEncoder.encodableTypes
    package var minFloorForSensitiveRoles: GGMLType? = .iq4XS

    package init(targetBPW: Double) {
        self.targetBPW = targetBPW
    }
}

package struct GSQRCOSummary: Sendable, Equatable {
    package var plan: RCOSensitivityAllocator.AllocationPlan
    package var quantizedLayers: Int
    package var quantizedWeights: Int
    package var formatCounts: [String: Int]
    package var wallSeconds: Double
}

package struct QTIPBudgetSummary: Sendable, Equatable {
    package var plan: RCOSensitivityAllocator.RateAllocationPlan
    package var qtipLayers: Int
    package var affineProtectedLayers: Int
    package var skippedProtectedLayers: Int
    package var rateCounts: [String: Int]
    package var affinePaths: Set<String>
    package var wallSeconds: Double
}

final class ActivationMeanSquareBox {
    var sumSquares: Double = 0
    var elementCount = 0

    func record(_ x: MLXArray) {
        guard x.size > 0 else { return }
        let xf = x.asType(.float32)
        let sq = (xf * xf).sum()
        eval(sq)
        let val = Double(sq.item(Float.self))
        if val.isFinite {
            sumSquares += val
            elementCount += x.size
        }
    }

    var meanSquare: Double {
        elementCount > 0 ? max(sumSquares / Double(elementCount), 1e-8) : 1.0
    }
}

final class SensitivityTraceTap: Linear {
    let wrapped: Module
    let box: ActivationMeanSquareBox

    init(wrapping module: Module, box: ActivationMeanSquareBox) {
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

struct SensitivityTarget {
    var path: String
    var groupKey: String
    var module: Module
    var outFeatures: Int
    var inFeatures: Int
    var role: RCOSensitivityAllocator.LayerRole
}

package enum GSQRCOConverter {
    static let maxShardBytes = 4_500_000_000

    package static func measureSensitivities(
        _ model: any LanguageModel,
        calibrationTokenBatches: [[Int32]],
        minDimension: Int = 32,
        requireNoBias: Bool = false,
        requireRHTSupport: Bool = false
    ) throws -> [RCOSensitivityAllocator.LayerSensitivity] {
        let root = model as Module
        let targets = collectTargets(
            root, minDimension: minDimension, requireNoBias: requireNoBias, requireRHTSupport: requireRHTSupport)
        return try scoreTargets(model: model, root: root, targets: targets, batches: calibrationTokenBatches)
    }

    static func collectTargets(
        _ root: Module,
        minDimension: Int,
        requireNoBias: Bool,
        requireRHTSupport: Bool
    ) -> [SensitivityTarget] {
        var out: [SensitivityTarget] = []
        for (path, mod) in root.namedModules() {
            guard PVTuner.textBlockIndex(path) != nil,
                  !path.contains("router"), !path.hasSuffix("in_proj_b"), !path.hasSuffix("in_proj_a") else { continue }
            guard let info = linearGeometry(mod) else { continue }
            if requireNoBias, info.hasBias { continue }
            let align = requireRHTSupport ? 16 : 32
            guard info.out >= minDimension, info.inp >= minDimension, info.inp.isMultiple(of: align) else { continue }
            if requireRHTSupport, !(KroneckerRHT.supports(info.out) && KroneckerRHT.supports(info.inp)) { continue }
            let role = RCOSensitivityAllocator.LayerRole.infer(from: path)
            let groupKey = QTIPQuantizer.sharedInputGroupKey(path: path, inFeatures: info.inp, sharedMoERHT: true)
            out.append(SensitivityTarget(
                path: path, groupKey: groupKey, module: mod,
                outFeatures: info.out, inFeatures: info.inp, role: role))
        }
        return out
    }

    private static func linearGeometry(_ mod: Module) -> (out: Int, inp: Int, hasBias: Bool)? {
        if let ql = mod as? QuantizedLinear, ql.globalScale == nil {
            return (ql.weight.dim(0), ql.weight.dim(1) * 32 / ql.bits, ql.bias != nil)
        }
        if type(of: mod) == Linear.self, let lin = mod as? Linear {
            return (lin.weight.dim(0), lin.weight.dim(1), lin.bias != nil)
        }
        return nil
    }

    static func scoreTargets(
        model: any LanguageModel,
        root: Module,
        targets: [SensitivityTarget],
        batches: [[Int32]]
    ) throws -> [RCOSensitivityAllocator.LayerSensitivity] {
        guard !targets.isEmpty else { return [] }
        let boxes = try captureMeanSquares(model: model, root: root, targets: targets, batches: batches)
        return targets.map { item in
            let (dense, _) = QTIPQuantizer.extractDenseWeight(item.module)
            let wf = dense.asType(.float32)
            let wMeanSq = (wf * wf).mean()
            eval(wMeanSq)
            let wEnergy = max(Double(wMeanSq.item(Float.self)), 1e-10)
            let xEnergy = boxes[item.groupKey]?.meanSquare ?? 1.0
            let roleWeight: Double = item.role.requiresSensitiveFloor ? 1.25 : 1.0
            let sigma = max(1e-12, xEnergy * wEnergy * roleWeight)
            return RCOSensitivityAllocator.LayerSensitivity(
                name: item.path,
                weightCount: item.outFeatures * item.inFeatures,
                sensitivity: sigma,
                role: item.role,
                columns: item.inFeatures)
        }
    }

    private static func captureMeanSquares(
        model: any LanguageModel,
        root: Module,
        targets: [SensitivityTarget],
        batches: [[Int32]]
    ) throws -> [String: ActivationMeanSquareBox] {
        guard !batches.isEmpty else { return [:] }
        var boxes: [String: ActivationMeanSquareBox] = [:]
        var tappedKeys: Set<String> = []
        var swapped: [SensitivityTarget] = []
        defer {
            for entry in swapped { try? PVTuner.swapModule(entry.path, with: entry.module, in: root) }
        }
        for entry in targets {
            let tapKey = QTIPQuantizer.sharedInputGroupKey(path: entry.path, inFeatures: entry.inFeatures, sharedMoERHT: false)
            guard tappedKeys.insert(tapKey).inserted else { continue }
            let box = boxes[entry.groupKey] ?? ActivationMeanSquareBox()
            boxes[entry.groupKey] = box
            try PVTuner.swapModule(entry.path, with: SensitivityTraceTap(wrapping: entry.module, box: box), in: root)
            swapped.append(entry)
        }
        for tokens in batches where !tokens.isEmpty {
            let out = model(LMInput.Text(tokens: MLXArray(tokens)[.newAxis]), cache: nil, state: nil)
            eval(out.logits)
        }
        return boxes
    }
}

extension GSQRCOConverter {
    package static func quantizeModel(
        _ model: any LanguageModel,
        calibrationTokenBatches: [[Int32]],
        config: GSQRCOConfig,
        onStep: ((String) -> Void)? = nil
    ) throws -> GSQRCOSummary {
        let startClock = ContinuousClock.now
        let root = model as Module
        root.train(false)
        let targets = collectTargets(root, minDimension: config.minDimension, requireNoBias: true, requireRHTSupport: false)
        let sensitivities = try scoreTargets(model: model, root: root, targets: targets, batches: calibrationTokenBatches)
        let allowed = config.allowedTypes.filter(GSQRCOBlockEncoder.supports)
        let budgetCfg = RCOSensitivityAllocator.BudgetConfig(
            targetBPW: config.targetBPW,
            allowedTypes: allowed.isEmpty ? GSQRCOBlockEncoder.encodableTypes : allowed,
            minFloorForSensitiveRoles: config.minFloorForSensitiveRoles)
        let plan = RCOSensitivityAllocator().allocate(layers: sensitivities, config: budgetCfg)
        let byName = Dictionary(uniqueKeysWithValues: targets.map { ($0.path, $0) })
        var formatCounts: [String: Int] = [:]
        var (layers, weights) = (0, 0)
        for alloc in plan.allocations {
            try Task.checkCancellation()
            guard let target = byName[alloc.name] else { continue }
            let (dense, _) = QTIPQuantizer.extractDenseWeight(target.module)
            let flat = dense.asType(.float32).reshaped(target.outFeatures * target.inFeatures)
            eval(flat)
            let bytes = GSQRCOBlockEncoder.encodeMatrix(
                flat.asArray(Float.self), rows: target.outFeatures, columns: target.inFeatures, type: alloc.ggmlType)
            let rowBytes = bytes.count / target.outFeatures
            let packed = MLXArray(bytes, [target.outFeatures, rowBytes])
            let gLayer = GGUFLinear(type: alloc.ggmlType, outputs: target.outFeatures, inputs: target.inFeatures)
            try gLayer.update(parameters: ModuleParameters.unflattened(["weight": packed]), verify: .all)
            eval(gLayer)
            try PVTuner.swapModule(target.path, with: gLayer, in: root)
            formatCounts[alloc.ggmlType.name, default: 0] += 1
            layers += 1
            weights += target.outFeatures * target.inFeatures
            if layers.isMultiple(of: 8) {
                Memory.clearCache()
                onStep?("gsq-rco progress \(layers)/\(plan.allocations.count) (\(alloc.name) -> \(alloc.ggmlType.name))")
            }
        }
        MLXInferenceEngine.settle(root)
        Memory.clearCache()
        let elapsed = ContinuousClock.now - startClock
        let secs = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        return GSQRCOSummary(
            plan: plan, quantizedLayers: layers, quantizedWeights: weights,
            formatCounts: formatCounts, wallSeconds: secs)
    }

    package static func convertDirectory(
        source: URL,
        output: URL,
        targetBPW: Double,
        calibrationText: String? = nil,
        config: GSQRCOConfig? = nil,
        onStep: (@Sendable (String) -> Void)? = nil
    ) async throws -> (summary: GSQRCOSummary, shards: Int, bytesAfter: Int) {
        try QTIPNativeLoader.validateOutput(source: source, output: output)
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        defer { await withTaskCancellationShield { await engine.unload(); Memory.clearCache() } }
        try await engine.load(modelPath: source.path)
        guard let container = await engine.container else { throw BasaltError.modelNotLoaded }
        let corpus = (calibrationText?.isEmpty == false ? calibrationText : nil) ?? PVTuner.defaultCalibrationCorpus
        var rcoCfg = config ?? GSQRCOConfig(targetBPW: targetBPW)
        rcoCfg.targetBPW = targetBPW
        let finalCfg = rcoCfg
        let batches = await tokenizeCalibration(container, corpus: corpus, count: finalCfg.calibrationBatches, seqLen: finalCfg.calibrationSeqLen)
        do {
            return try await container.perform { ctx in
                let summary = try quantizeModel(ctx.model, calibrationTokenBatches: batches, config: finalCfg, onStep: onStep)
                let written = try writeGSQDirectory(model: ctx.model as Module, source: source, output: output)
                return (summary, written.shards, written.bytes)
            }
        } catch {
            try? FileManager.default.removeItem(at: output)
            throw error
        }
    }

    package static func writeGSQDirectory(
        model: Module, source: URL, output: URL
    ) throws -> (shards: Int, bytes: Int) {
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        var entries: [String: GGUFNativeLoader.Entry] = [:]
        for (path, mod) in model.namedModules() {
            guard let g = mod as? GGUFLinear else { continue }
            entries[path] = GGUFNativeLoader.Entry(type: g.type.name, inputOrder: g.inputOrder == nil ? nil : true)
        }
        var writer = ShardWriter(directory: output, maxBytes: maxShardBytes, metadata: ["format": "mlx"])
        var totalBytes = 0
        for (key, arr) in model.parameters().flattened().sorted(by: { $0.0 < $1.0 }) {
            totalBytes += arr.nbytes
            try writer.add([key: arr])
        }
        let shards = try writer.finish()
        let cfgURL = source.appending(component: "config.json")
        var cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: cfgURL)) as? [String: Any] ?? [:]
        cfg["quantization"] = nil
        cfg["quantization_config"] = nil
        cfg["gguf_quantization"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(entries))
        try JSONSerialization.data(withJSONObject: cfg, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appending(component: "config.json"))
        for name in try FileManager.default.contentsOfDirectory(atPath: source.path) where !name.hasPrefix(".") {
            let lowered = name.lowercased()
            guard lowered != "config.json", !lowered.hasSuffix(".safetensors"),
                  !lowered.hasSuffix(".safetensors.index.json") else { continue }
            try FileManager.default.copyItem(
                at: source.appending(component: name).resolvingSymlinksInPath(),
                to: output.appending(component: name))
        }
        return (shards, totalBytes)
    }

    static func tokenizeCalibration(
        _ container: ModelContainer, corpus: String, count: Int, seqLen: Int
    ) async -> [[Int32]] {
        await container.perform { ctx in
            let encoded = ctx.tokenizer.encode(text: corpus, addSpecialTokens: false)
            let len = max(1, seqLen)
            let usable = min(count, encoded.count / len)
            return (0 ..< usable).map { idx in
                encoded[(idx * len) ..< ((idx + 1) * len)].map(Int32.init)
            }
        }
    }
}

extension GSQRCOConverter {
    package static func quantizeModelQTIPBudget(
        _ model: any LanguageModel,
        calibrationTokenBatches: [[Int32]],
        targetBPW: Double,
        baseConfig: QTIPConfig? = nil,
        onStep: ((String) -> Void)? = nil
    ) throws -> QTIPBudgetSummary {
        let startClock = ContinuousClock.now
        let root = model as Module
        root.train(false)
        let cfg = baseConfig ?? QTIPConfig(rate: .twoHalfBPW)
        let skippedProtected = countSkippedProtectedProjections(root)
        let sensTargets = collectTargets(root, minDimension: max(16, cfg.minDimension), requireNoBias: false, requireRHTSupport: true)
        let sensitivities = try scoreTargets(model: model, root: root, targets: sensTargets, batches: calibrationTokenBatches)
        let plan = RCOSensitivityAllocator().allocate(
            layers: sensitivities,
            targetBitsPerWeight: targetBPW,
            candidates: RCOSensitivityAllocator.qtipBudgetCandidates)
        let assignedByPath = Dictionary(uniqueKeysWithValues: plan.allocations.map { ($0.name, $0.candidate) })
        let applied = try applyQTIPBudgetPlan(
            model: model, root: root, assigned: assignedByPath,
            batches: calibrationTokenBatches, config: cfg, onStep: onStep)
        MLXInferenceEngine.settle(root)
        Memory.clearCache()
        let elapsed = ContinuousClock.now - startClock
        let secs = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        return QTIPBudgetSummary(
            plan: plan,
            qtipLayers: applied.qtipCount,
            affineProtectedLayers: applied.affinePaths.count,
            skippedProtectedLayers: skippedProtected,
            rateCounts: applied.rateCounts,
            affinePaths: applied.affinePaths,
            wallSeconds: secs)
    }

    private static func countSkippedProtectedProjections(_ root: Module) -> Int {
        var count = 0
        for (path, mod) in root.namedModules() where PVTuner.textBlockIndex(path) != nil {
            guard mod is Linear || mod is QuantizedLinear else { continue }
            if path.contains("router") || path.hasSuffix("in_proj_b") || path.hasSuffix("in_proj_a")
                || RCOSensitivityAllocator.LayerRole.infer(from: path) == .router {
                count += 1
            }
        }
        return count
    }

    private static func applyQTIPBudgetPlan(
        model: any LanguageModel,
        root: Module,
        assigned: [String: RCOSensitivityAllocator.RateCandidate],
        batches: [[Int32]],
        config: QTIPConfig,
        onStep: ((String) -> Void)?
    ) throws -> (qtipCount: Int, affinePaths: Set<String>, rateCounts: [String: Int]) {
        var tluts: [QTIPRate: MLXArray] = [:]
        for r in QTIPRate.allCases {
            let t = MLXArray(QTIPCodebook.baseLUTValues(rate: r), [512, 2]).asType(.float16)
            eval(t)
            tluts[r] = t
        }
        let maxTokens = max(1, batches.reduce(0) { $0 + $1.count })
        let blockGroups = QTIPQuantizer.groupedBlocks(in: root, stepSize: max(1, config.blockGroupSize))
        var qtipCount = 0
        var affinePaths: Set<String> = []
        var rateCounts: [String: Int] = [:]
        for (gIdx, group) in blockGroups.enumerated() {
            try Task.checkCancellation()
            let targets = QTIPQuantizer.collectBlockTargets(root, blocks: Set(group), config: config)
                .filter { assigned[$0.path] != nil }
            guard !targets.isEmpty else { continue }
            let boxes = try QTIPQuantizer.captureGroupActivations(
                model: model, root: root, targets: targets, batches: batches, maxTokens: maxTokens)
            let groupRes = try quantizeMixedGroup(
                root: root, targets: targets, assigned: assigned, boxes: boxes, config: config, tluts: tluts)
            qtipCount += groupRes.qtip
            affinePaths.formUnion(groupRes.affine)
            for (k, v) in groupRes.counts { rateCounts[k, default: 0] += v }
            Memory.clearCache()
            onStep?("qtip-budget group \(gIdx + 1)/\(blockGroups.count) qtip=\(qtipCount) affine4=\(affinePaths.count)")
        }
        return (qtipCount, affinePaths, rateCounts)
    }

    private static func quantizeMixedGroup(
        root: Module,
        targets: [QTIPTargetEntry],
        assigned: [String: RCOSensitivityAllocator.RateCandidate],
        boxes: [String: QTIPActivationBox],
        config: QTIPConfig,
        tluts: [QTIPRate: MLXArray]
    ) throws -> (qtip: Int, affine: Set<String>, counts: [String: Int]) {
        var factorCache: [String: (rhtIn: KroneckerRHT, cholesky: (u: MLXArray, diagInv: MLXArray)?)] = [:]
        var bufferCache: [String: FusedKroneckerRHT.Buffers] = [:]
        var (qtip, affine, counts) = (0, Set<String>(), [String: Int]())
        for entry in targets {
            try Task.checkCancellation()
            guard let cand = assigned[entry.path] else { continue }
            let (dense, bias) = QTIPQuantizer.extractDenseWeight(entry.module)
            if cand.bitsPerWeight >= 4.0, entry.inFeatures.isMultiple(of: 32) {
                let gSize = entry.inFeatures.isMultiple(of: 64) ? 64 : 32
                let qLayer = QuantizedLinear(weight: dense, bias: bias, groupSize: gSize, bits: 4, mode: .affine)
                eval(qLayer)
                try PVTuner.swapModule(entry.path, with: qLayer, in: root)
                affine.insert(entry.path)
                counts[cand.id, default: 0] += 1
                continue
            }
            let rate = QTIPRate(bpw: cand.bitsPerWeight) ?? .threeBPW
            guard let tlut = tluts[rate],
                  let inEntry = QTIPQuantizer.resolveInputFactor(entry, boxes: boxes, config: config, cache: &factorCache) else {
                continue
            }
            let outSeed = QTIPQuantizer.hashString(
                QTIPQuantizer.sharedOutputKey(path: entry.path, sharedMoERHT: config.sharedMoERHT), seed: config.seed)
            guard let rhtOut = KroneckerRHT(dimension: entry.outFeatures, seed: outSeed) else { continue }
            let (wNorm, rowScales) = QTIPQuantizer.prepareNormalizedWeight(weight: dense, rhtIn: inEntry.rhtIn, rhtOut: rhtOut)
            let packed = QTIPQuantizer.encodeNormalizedWeight(wNorm: wNorm, choleskyFactors: inEntry.cholesky, tlut: tlut, rate: rate)
            let spec = QTIPLayerSpec(rate: rate, rhtIn: inEntry.rhtIn, rhtOut: rhtOut)
            let pair = FusedKroneckerRHT.Pair(
                inBuffers: QTIPQuantizer.cachedBuffers(for: inEntry.rhtIn, cache: &bufferCache),
                outBuffers: QTIPQuantizer.cachedBuffers(for: rhtOut, cache: &bufferCache))
            let ownedTlut = tlut + MLXArray(Float16(0))
            let qLayer = QTIPLinear(packedWeight: packed, rowScales: rowScales, tlut: ownedTlut, bias: bias, spec: spec, buffers: pair)
            eval(qLayer)
            try PVTuner.swapModule(entry.path, with: qLayer, in: root)
            qtip += 1
            counts[cand.id, default: 0] += 1
        }
        factorCache.removeAll()
        return (qtip, affine, counts)
    }
}

extension GSQRCOConverter {
    package static func convertQTIPBudgetDirectory(
        source: URL,
        output: URL,
        targetBPW: Double,
        calibrationText: String? = nil,
        config: QTIPConfig? = nil,
        onStep: (@Sendable (String) -> Void)? = nil
    ) async throws -> (summary: QTIPBudgetSummary, shards: Int, bytesAfter: Int) {
        try QTIPNativeLoader.validateOutput(source: source, output: output)
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        defer { await withTaskCancellationShield { await engine.unload(); Memory.clearCache() } }
        try await engine.load(modelPath: source.path)
        guard let container = await engine.container else { throw BasaltError.modelNotLoaded }
        let corpus = (calibrationText?.isEmpty == false ? calibrationText : nil) ?? PVTuner.defaultCalibrationCorpus
        let finalCfg = config ?? QTIPConfig(rate: .twoHalfBPW)
        let batches = await tokenizeCalibration(container, corpus: corpus, count: finalCfg.calibrationBatches, seqLen: finalCfg.calibrationSeqLen)
        do {
            return try await container.perform { ctx in
                let summary = try quantizeModelQTIPBudget(
                    ctx.model, calibrationTokenBatches: batches, targetBPW: targetBPW, baseConfig: finalCfg, onStep: onStep)
                let written = try QTIPNativeLoader.writeMixedDirectory(
                    model: ctx.model as Module, achievedBPW: summary.plan.achievedBPW,
                    seed: finalCfg.seed, source: source, output: output, affinePaths: summary.affinePaths)
                return (summary, written.shards, written.bytes)
            }
        } catch {
            try? FileManager.default.removeItem(at: output)
            throw error
        }
    }
}
