import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

package struct QTIPStagedEntry: Sendable {
    package var index: Int
    package var path: String
    package var groupKey: String
    package var outFeatures: Int
    package var inFeatures: Int
    package var hasBias: Bool
}

package struct QTIPStagedPlan: Sendable {
    package var directory: URL
    package var seed: UInt64
    package var groups: [[QTIPStagedEntry]]
    package var calibrationTokens: Int
    package var sharedMoERHT = true
}

private struct QTIPPreparedLayer {
    var index: Int
    var wNorm: MLXArray
    var rowScales: MLXArray
    var bias: MLXArray?
    var cholesky: (u: MLXArray, diagInv: MLXArray)?
}

private struct QTIPMultiRateContext {
    var primaryRate: QTIPRate
    var secondaryRates: [QTIPRate]
    var config: QTIPConfig
    var stageDirectory: URL
    var tluts: [QTIPRate: MLXArray]
    var onStep: ((String) -> Void)?
}

extension QTIPQuantizer {
    package static func quantizeModelMultiRate(
        _ model: any LanguageModel,
        calibrationTokenBatches: [[Int32]],
        rates: [QTIPRate],
        baseConfig: QTIPConfig,
        stageDirectory: URL,
        onStep: ((String) -> Void)? = nil
    ) throws -> (primary: QTIPQuantizeSummary, plan: QTIPStagedPlan) {
        guard let primaryRate = rates.first else {
            throw BasaltError.invalidRequestStructure(details: "QTIP multi-rate requires at least one rate")
        }
        let secondaryRates = Array(rates.dropFirst())
        try FileManager.default.createDirectory(at: stageDirectory, withIntermediateDirectories: true)
        let startClock = ContinuousClock.now
        let root = model as Module
        root.train(false)
        var tluts: [QTIPRate: MLXArray] = [:]
        for r in rates {
            let t = MLXArray(QTIPCodebook.baseLUTValues(rate: r), [512, 2]).asType(.float16)
            eval(t)
            tluts[r] = t
        }
        var cfg = baseConfig
        cfg.rate = primaryRate
        let mctx = QTIPMultiRateContext(
            primaryRate: primaryRate, secondaryRates: secondaryRates, config: cfg,
            stageDirectory: stageDirectory, tluts: tluts, onStep: onStep)
        let blockGroups = groupedBlocks(in: root, stepSize: max(1, cfg.blockGroupSize))
        let maxTokens = max(1, calibrationTokenBatches.reduce(0) { $0 + $1.count })
        var (totalLayers, totalWeights) = (0, 0)
        var stagedGroups: [[QTIPStagedEntry]] = []
        for (gIdx, group) in blockGroups.enumerated() {
            try Task.checkCancellation()
            let targets = collectBlockTargets(root, blocks: Set(group), config: cfg)
            guard !targets.isEmpty else { continue }
            let boxes = try captureGroupActivations(
                model: model, root: root, targets: targets, batches: calibrationTokenBatches, maxTokens: maxTokens)
            let res = try quantizeGroupMultiRate(
                root: root, groupIndex: stagedGroups.count, targets: targets, boxes: boxes, ctx: mctx)
            totalLayers += res.layers
            totalWeights += res.weights
            stagedGroups.append(res.entries)
            Memory.clearCache()
            onStep?("qtip-multi group \(gIdx + 1)/\(blockGroups.count) layers=\(totalLayers)")
        }
        MLXInferenceEngine.settle(root)
        Memory.clearCache()
        let elapsed = ContinuousClock.now - startClock
        let secs = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        let calTokens = calibrationTokenBatches.reduce(0) { $0 + $1.count }
        let summary = QTIPQuantizeSummary(
            rate: primaryRate, quantizedLayers: totalLayers, quantizedWeights: totalWeights,
            calibrationTokens: calTokens, wallSeconds: secs)
        let plan = QTIPStagedPlan(
            directory: stageDirectory, seed: cfg.seed, groups: stagedGroups,
            calibrationTokens: calTokens, sharedMoERHT: cfg.sharedMoERHT)
        return (summary, plan)
    }

    private static func quantizeGroupMultiRate(
        root: Module,
        groupIndex: Int,
        targets: [QTIPTargetEntry],
        boxes: [String: QTIPActivationBox],
        ctx: QTIPMultiRateContext
    ) throws -> (layers: Int, weights: Int, entries: [QTIPStagedEntry]) {
        var factorCache: [String: (rhtIn: KroneckerRHT, cholesky: (u: MLXArray, diagInv: MLXArray)?)] = [:]
        var bufferCache: [String: FusedKroneckerRHT.Buffers] = [:]
        var remaining = Dictionary(targets.map { ($0.groupKey, 1) }, uniquingKeysWith: +)
        var stagedArrays: [QTIPRate: [String: MLXArray]] = [:]
        for r in ctx.secondaryRates { stagedArrays[r] = [:] }
        var entries: [QTIPStagedEntry] = []
        var (layers, weights) = (0, 0)
        guard let primaryTLUT = ctx.tluts[ctx.primaryRate] else { return (0, 0, []) }
        for entry in targets {
            try Task.checkCancellation()
            guard let inEntry = resolveInputFactor(entry, boxes: boxes, config: ctx.config, cache: &factorCache) else { continue }
            remaining[entry.groupKey, default: 1] -= 1
            let releaseFactor = remaining[entry.groupKey, default: 0] <= 0
            if releaseFactor { factorCache.removeValue(forKey: entry.groupKey) }
            let outKey = sharedOutputKey(path: entry.path, sharedMoERHT: ctx.config.sharedMoERHT)
            let outSeed = hashString(outKey, seed: ctx.config.seed)
            guard let rhtOut = KroneckerRHT(dimension: entry.outFeatures, seed: outSeed) else { continue }
            let (dense, bias) = extractDenseWeight(entry.module)
            let (wNorm, rowScales) = prepareNormalizedWeight(weight: dense, rhtIn: inEntry.rhtIn, rhtOut: rhtOut)
            let pPrimary = encodeNormalizedWeight(
                wNorm: wNorm, choleskyFactors: inEntry.cholesky, tlut: primaryTLUT, rate: ctx.primaryRate)
            let primarySpec = QTIPLayerSpec(rate: ctx.primaryRate, rhtIn: inEntry.rhtIn, rhtOut: rhtOut)
            let inBuf = cachedBuffers(for: inEntry.rhtIn, cache: &bufferCache)
            let outBuf = cachedBuffers(for: rhtOut, cache: &bufferCache)
            let pair = FusedKroneckerRHT.Pair(inBuffers: inBuf, outBuffers: outBuf)
            let qLayer = QTIPLinear(
                packedWeight: pPrimary, rowScales: rowScales, tlut: primaryTLUT, bias: bias,
                spec: primarySpec, buffers: pair)
            eval(qLayer)
            try PVTuner.swapModule(entry.path, with: qLayer, in: root)
            let localIdx = entries.count
            let prep = QTIPPreparedLayer(
                index: localIdx, wNorm: wNorm, rowScales: rowScales, bias: bias, cholesky: inEntry.cholesky)
            stageSecondaryRates(prep, ctx: ctx, into: &stagedArrays)
            entries.append(QTIPStagedEntry(
                index: localIdx, path: entry.path, groupKey: entry.groupKey,
                outFeatures: entry.outFeatures, inFeatures: entry.inFeatures, hasBias: bias != nil))
            layers += 1
            weights += entry.outFeatures * entry.inFeatures
            if releaseFactor { Memory.clearCache() }
            if layers.isMultiple(of: 8) {
                Memory.clearCache()
                ctx.onStep?("qtip-multi progress \(layers)/\(targets.count) in-group (\(entry.path))")
            }
        }
        factorCache.removeAll()
        try flushStagedGroup(stagedArrays, groupIndex: groupIndex, ctx: ctx)
        stagedArrays.removeAll()
        return (layers, weights, entries)
    }
}

extension QTIPQuantizer {
    private static func stageSecondaryRates(
        _ prep: QTIPPreparedLayer,
        ctx: QTIPMultiRateContext,
        into stagedArrays: inout [QTIPRate: [String: MLXArray]]
    ) {
        for secRate in ctx.secondaryRates {
            guard let secTLUT = ctx.tluts[secRate] else { continue }
            let pSec = encodeNormalizedWeight(
                wNorm: prep.wNorm, choleskyFactors: prep.cholesky, tlut: secTLUT, rate: secRate)
            stagedArrays[secRate]?["t\(prep.index).packed"] = pSec
            stagedArrays[secRate]?["t\(prep.index).scales"] = prep.rowScales
            if let bias = prep.bias { stagedArrays[secRate]?["t\(prep.index).bias"] = bias }
        }
    }

    private static func flushStagedGroup(
        _ stagedArrays: [QTIPRate: [String: MLXArray]],
        groupIndex: Int,
        ctx: QTIPMultiRateContext
    ) throws {
        for secRate in ctx.secondaryRates {
            guard let dict = stagedArrays[secRate], !dict.isEmpty else { continue }
            let url = ctx.stageDirectory.appending(component: "rate-\(secRate.label)-group-\(groupIndex).safetensors")
            try save(arrays: dict, metadata: ["format": "mlx"], url: url)
        }
    }

    package static func applyStagedRate(_ root: Module, plan: QTIPStagedPlan, rate: QTIPRate) throws -> Int {
        let tlut = MLXArray(QTIPCodebook.baseLUTValues(rate: rate), [512, 2]).asType(.float16)
        eval(tlut)
        var rhtCache: [String: KroneckerRHT] = [:]
        var bufferCache: [String: FusedKroneckerRHT.Buffers] = [:]
        var applied = 0
        for (gIdx, entries) in plan.groups.enumerated() where !entries.isEmpty {
            let url = plan.directory.appending(component: "rate-\(rate.label)-group-\(gIdx).safetensors")
            let arrays = try loadArrays(url: url)
            for item in entries {
                guard let packed = arrays["t\(item.index).packed"],
                      let scales = arrays["t\(item.index).scales"] else {
                    throw BasaltError.inferenceError("QTIP staged missing tensor for \(item.path)")
                }
                let bias = item.hasBias ? arrays["t\(item.index).bias"] : nil
                let inKey = "\(item.inFeatures):\(hashString(item.groupKey, seed: plan.seed))"
                let outName = sharedOutputKey(path: item.path, sharedMoERHT: plan.sharedMoERHT)
                let outKey = "\(item.outFeatures):\(hashString(outName, seed: plan.seed))"
                let rhtIn = try cachedRHT(dimension: item.inFeatures, seedKey: inKey, cache: &rhtCache)
                let rhtOut = try cachedRHT(dimension: item.outFeatures, seedKey: outKey, cache: &rhtCache)
                let spec = QTIPLayerSpec(rate: rate, rhtIn: rhtIn, rhtOut: rhtOut)
                let inBuf = cachedBuffers(for: rhtIn, cache: &bufferCache)
                let outBuf = cachedBuffers(for: rhtOut, cache: &bufferCache)
                let pair = FusedKroneckerRHT.Pair(inBuffers: inBuf, outBuffers: outBuf)
                let layer = QTIPLinear(
                    packedWeight: packed, rowScales: scales, tlut: tlut, bias: bias, spec: spec, buffers: pair)
                eval(layer)
                try PVTuner.swapModule(item.path, with: layer, in: root)
                applied += 1
            }
            Memory.clearCache()
        }
        MLXInferenceEngine.settle(root)
        Memory.clearCache()
        return applied
    }

    static func cachedRHT(dimension: Int, seedKey: String, cache: inout [String: KroneckerRHT]) throws(BasaltError) -> KroneckerRHT {
        if let existing = cache[seedKey] { return existing }
        let rawSeed = UInt64(seedKey.split(separator: ":").last ?? "0") ?? 0
        guard let built = KroneckerRHT(dimension: dimension, seed: rawSeed) else {
            throw .inferenceError("QTIP RHT unsupported dimension \(dimension)")
        }
        cache[seedKey] = built
        return built
    }
}

package enum QTIPNativeLoader {
    package struct LayerEntry: Codable, Sendable, Equatable {
        package var outFeatures: Int
        package var inFeatures: Int
        package var hasBias: Bool
        package var bpw: Double?

        package init(outFeatures: Int, inFeatures: Int, hasBias: Bool, bpw: Double? = nil) {
            self.outFeatures = outFeatures
            self.inFeatures = inFeatures
            self.hasBias = hasBias
            self.bpw = bpw
        }
    }

    package struct AffineLayerEntry: Codable, Sendable, Equatable {
        package var outFeatures: Int
        package var inFeatures: Int
        package var groupSize: Int
        package var bits: Int
        package var hasBias: Bool
        package var dtype: String

        package init(
            outFeatures: Int, inFeatures: Int, groupSize: Int, bits: Int, hasBias: Bool, dtype: String
        ) {
            self.outFeatures = outFeatures
            self.inFeatures = inFeatures
            self.groupSize = groupSize
            self.bits = bits
            self.hasBias = hasBias
            self.dtype = dtype
        }

        var scaleDType: DType {
            DType.allCases.first { "\($0)" == dtype } ?? .float16
        }
    }

    package struct Manifest: Codable, Sendable, Equatable {
        package var bpw: Double
        package var seed: UInt64
        package var layers: [String: LayerEntry]
        package var sharedMoERHT: Bool?
        package var affineLayers: [String: AffineLayerEntry]?

        enum CodingKeys: String, CodingKey {
            case bpw, seed, layers
            case sharedMoERHT = "shared_moe_rht"
            case affineLayers = "affine_layers"
        }

        package init(
            bpw: Double,
            seed: UInt64,
            layers: [String: LayerEntry],
            sharedMoERHT: Bool? = nil,
            affineLayers: [String: AffineLayerEntry]? = nil
        ) {
            self.bpw = bpw
            self.seed = seed
            self.layers = layers
            self.sharedMoERHT = sharedMoERHT
            self.affineLayers = affineLayers
        }
    }

    private struct Configuration: Decodable {
        let qtipQuantization: Manifest?

        enum CodingKeys: String, CodingKey { case qtipQuantization = "qtip_quantization" }
    }

    package static func manifest(in directory: URL) -> Manifest? {
        guard let data = try? Data(contentsOf: directory.appending(component: "config.json")) else { return nil }
        return (try? JSONDecoder().decode(Configuration.self, from: data))?.qtipQuantization
    }

    package static func loadContainer(from directory: URL, tokenizerLoader: any TokenizerLoader) async throws -> ModelContainer? {
        guard let manifest = manifest(in: directory) else { return nil }
        let configuration = try Data(contentsOf: directory.appending(component: "config.json"))
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configuration)
        var failure: (any Error)?
        for family in ExpertStreamingLoader.families where await family.types.contains(base.modelType) {
            do {
                let model = try await family.types.createModel(configuration: configuration, modelType: base.modelType)
                try install(manifest, in: model)
                let registry = ModelTypeRegistry<any LanguageModel>(creators: [base.modelType: { _ in model }])
                return try await family.factory(registry).loadContainer(from: directory, using: tokenizerLoader)
            } catch {
                failure = error
            }
        }
        throw failure ?? BasaltError.inferenceError("QTIP native: no model factory knows model_type \(base.modelType)")
    }

    package static func install(_ manifest: Manifest, in model: Module) throws(BasaltError) {
        if manifest.layers.isEmpty, (manifest.affineLayers ?? [:]).isEmpty, QTIPRate(bpw: manifest.bpw) == nil {
            throw .inferenceError("QTIP native: unsupported rate \(manifest.bpw) bpw")
        }
        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        let sharedMoE = manifest.sharedMoERHT ?? false
        var rhtCache: [String: KroneckerRHT] = [:]
        var bufferCache: [String: FusedKroneckerRHT.Buffers] = [:]
        var updates: [(String, Module)] = []
        for (path, entry) in manifest.layers.sorted(by: { $0.key < $1.key }) {
            let layerBPW = entry.bpw ?? manifest.bpw
            guard let rate = QTIPRate(bpw: layerBPW) else {
                throw .inferenceError("QTIP native: unsupported rate \(layerBPW) bpw for \(path)")
            }
            guard modules[path] is Linear else {
                throw .inferenceError("QTIP native: \(path) is not a linear layer of this model")
            }
            let groupKey = QTIPQuantizer.sharedInputGroupKey(path: path, inFeatures: entry.inFeatures, sharedMoERHT: sharedMoE)
            let outName = QTIPQuantizer.sharedOutputKey(path: path, sharedMoERHT: sharedMoE)
            let inKey = "\(entry.inFeatures):\(QTIPQuantizer.hashString(groupKey, seed: manifest.seed))"
            let outKey = "\(entry.outFeatures):\(QTIPQuantizer.hashString(outName, seed: manifest.seed))"
            let rhtIn = try QTIPQuantizer.cachedRHT(dimension: entry.inFeatures, seedKey: inKey, cache: &rhtCache)
            let rhtOut = try QTIPQuantizer.cachedRHT(dimension: entry.outFeatures, seedKey: outKey, cache: &rhtCache)
            let spec = QTIPLayerSpec(rate: rate, rhtIn: rhtIn, rhtOut: rhtOut)
            let inBuf = QTIPQuantizer.cachedBuffers(for: rhtIn, cache: &bufferCache)
            let outBuf = QTIPQuantizer.cachedBuffers(for: rhtOut, cache: &bufferCache)
            let pair = FusedKroneckerRHT.Pair(inBuffers: inBuf, outBuffers: outBuf)
            let packed = MLXArray.zeros([entry.outFeatures / 16, (entry.inFeatures / 16) * rate.wordsPerTile], dtype: .uint32)
            let scales = MLXArray.zeros([entry.outFeatures / 16], dtype: .float16)
            let layerTLUT = MLXArray.zeros([512, 2], dtype: .float16)
            let bias = entry.hasBias ? MLXArray.zeros([entry.outFeatures], dtype: .float16) : nil
            updates.append((path, QTIPLinear(
                packedWeight: packed, rowScales: scales, tlut: layerTLUT, bias: bias, spec: spec, buffers: pair)))
        }
        try appendAffinePlaceholders(manifest.affineLayers ?? [:], modules: modules, into: &updates)
        model.update(modules: ModuleChildren.unflattened(updates))
    }
}

extension QTIPNativeLoader {
    private static func appendAffinePlaceholders(
        _ affineLayers: [String: AffineLayerEntry],
        modules: [String: Module],
        into updates: inout [(String, Module)]
    ) throws(BasaltError) {
        for (path, aff) in affineLayers.sorted(by: { $0.key < $1.key }) {
            guard modules[path] is Linear else {
                throw .inferenceError("QTIP native: \(path) is not a linear layer of this model")
            }
            let dt = aff.scaleDType
            let packed = MLXArray.zeros([aff.outFeatures, (aff.inFeatures * aff.bits) / 32], dtype: .uint32)
            let groups = aff.inFeatures / aff.groupSize
            let scales = MLXArray.zeros([aff.outFeatures, groups], dtype: dt)
            let biases = MLXArray.zeros([aff.outFeatures, groups], dtype: dt)
            let bias = aff.hasBias ? MLXArray.zeros([aff.outFeatures], dtype: dt) : nil
            updates.append((path, QuantizedLinear(
                weight: packed, bias: bias, scales: scales, biases: biases,
                groupSize: aff.groupSize, bits: aff.bits, mode: .affine, globalScale: nil)))
        }
    }

    package static func writeMixedDirectory(
        model: Module,
        achievedBPW: Double,
        seed: UInt64,
        source: URL,
        output: URL,
        affinePaths: Set<String> = []
    ) throws -> (shards: Int, bytes: Int) {
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        var layerMap: [String: LayerEntry] = [:]
        var affineMap: [String: AffineLayerEntry] = [:]
        for (path, mod) in model.namedModules() {
            if let q = mod as? QTIPLinear {
                layerMap[path] = LayerEntry(
                    outFeatures: q.outFeatures, inFeatures: q.inFeatures,
                    hasBias: q.bias != nil, bpw: q.rate.bpw)
            } else if affinePaths.contains(path), let ql = mod as? QuantizedLinear {
                affineMap[path] = AffineLayerEntry(
                    outFeatures: ql.weight.dim(0),
                    inFeatures: ql.weight.dim(1) * 32 / ql.bits,
                    groupSize: ql.groupSize,
                    bits: ql.bits,
                    hasBias: ql.bias != nil,
                    dtype: "\(ql.scales.dtype)")
            }
        }
        let manifest = Manifest(
            bpw: achievedBPW, seed: seed, layers: layerMap,
            sharedMoERHT: true, affineLayers: affineMap.isEmpty ? nil : affineMap)
        let shardInfo = try writeShards(from: model, into: output)
        try writeConfigAndAssets(source: source, output: output, manifest: manifest)
        return shardInfo
    }
}

extension QTIPNativeLoader {
    static let maxShardBytes = 4_500_000_000

    package static func convertDirectory(
        source: URL,
        output: URL,
        rate: QTIPRate,
        calibrationText: String? = nil,
        config: QTIPConfig? = nil,
        onStep: (@Sendable (String) -> Void)? = nil
    ) async throws -> (summary: QTIPQuantizeSummary, shards: Int, bytesAfter: Int) {
        try validateOutput(source: source, output: output)
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0, disableExpertStreaming: true))
        defer { await withTaskCancellationShield { await engine.unload(); Memory.clearCache() } }
        try await engine.load(modelPath: source.path)
        guard let container = await engine.container else { throw BasaltError.modelNotLoaded }
        let corpus = (calibrationText?.isEmpty == false ? calibrationText : nil) ?? PVTuner.defaultCalibrationCorpus
        var qConfig = config ?? QTIPConfig(rate: rate)
        qConfig.rate = rate
        let finalCfg = qConfig
        let batches = await container.performEvaluating(nonSendable: ()) { ctx, _ in
            let encoded = ctx.tokenizer.encode(text: corpus, addSpecialTokens: false)
            let seqLen = max(1, finalCfg.calibrationSeqLen)
            let usable = min(finalCfg.calibrationBatches, encoded.count / seqLen)
            return (0 ..< usable).map { idx in
                encoded[(idx * seqLen) ..< ((idx + 1) * seqLen)].map(Int32.init)
            }
        }
        do {
            return try await container.performEvaluating(nonSendable: ()) { ctx, _ in
                let summary = try QTIPQuantizer.quantizeModel(
                    ctx.model, calibrationTokenBatches: batches, config: finalCfg, onStep: onStep)
                let written = try writeDirectory(
                    model: ctx.model as Module, rate: rate, seed: finalCfg.seed,
                    source: source, output: output, sharedMoERHT: finalCfg.sharedMoERHT)
                return (summary, written.shards, written.bytes)
            }
        } catch {
            try? FileManager.default.removeItem(at: output)
            throw error
        }
    }

    static func validateOutput(source: URL, output: URL) throws(BasaltError) {
        let srcPath = source.standardizedFileURL.path
        let outPath = output.standardizedFileURL.path
        if FileManager.default.fileExists(atPath: outPath) {
            throw .invalidRequestStructure(details: "output directory \(outPath) already exists")
        }
        if outPath == srcPath || outPath.hasPrefix(srcPath + "/") {
            throw .invalidRequestStructure(details: "output directory \(outPath) must not be inside \(srcPath)")
        }
    }

    package static func writeDirectory(
        model: Module, rate: QTIPRate, seed: UInt64, source: URL, output: URL, sharedMoERHT: Bool = true
    ) throws -> (shards: Int, bytes: Int) {
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        var layerMap: [String: LayerEntry] = [:]
        for (path, mod) in model.namedModules() {
            guard let q = mod as? QTIPLinear else { continue }
            layerMap[path] = LayerEntry(outFeatures: q.outFeatures, inFeatures: q.inFeatures, hasBias: q.bias != nil)
        }
        let manifest = Manifest(bpw: rate.bpw, seed: seed, layers: layerMap, sharedMoERHT: sharedMoERHT)
        let shardInfo = try writeShards(from: model, into: output)
        try writeConfigAndAssets(source: source, output: output, manifest: manifest)
        return shardInfo
    }

    private static func writeShards(from model: Module, into output: URL) throws -> (shards: Int, bytes: Int) {
        let sorted = model.parameters().flattened().sorted { $0.0 < $1.0 }
        eval(sorted.map(\.1))
        var groups: [[(String, MLXArray)]] = []
        var current: [(String, MLXArray)] = []
        var currentBytes = 0
        var totalBytes = 0
        for (key, arr) in sorted {
            if !current.isEmpty, currentBytes + arr.nbytes > maxShardBytes {
                groups.append(current)
                current = []
                currentBytes = 0
            }
            current.append((key, arr))
            currentBytes += arr.nbytes
            totalBytes += arr.nbytes
        }
        if !current.isEmpty { groups.append(current) }
        var weightMap: [String: String] = [:]
        for (idx, group) in groups.enumerated() {
            let name = groups.count == 1
                ? "model.safetensors"
                : String(format: "model-%05d-of-%05d.safetensors", idx + 1, groups.count)
            let dict = Dictionary(uniqueKeysWithValues: group)
            try save(arrays: dict, metadata: ["format": "mlx"], url: output.appending(component: name))
            for (k, _) in group { weightMap[k] = name }
        }
        if groups.count > 1 {
            let indexObj: [String: Any] = ["metadata": ["total_size": totalBytes], "weight_map": weightMap]
            let data = try JSONSerialization.data(withJSONObject: indexObj, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: output.appending(component: "model.safetensors.index.json"))
        }
        return (groups.count, totalBytes)
    }

    private static func writeConfigAndAssets(source: URL, output: URL, manifest: Manifest) throws {
        let cfgURL = source.appending(component: "config.json")
        var config = try JSONSerialization.jsonObject(with: Data(contentsOf: cfgURL)) as? [String: Any] ?? [:]
        if manifest.affineLayers != nil {
            config["quantization"] = nil
            config["quantization_config"] = nil
        }
        config["qtip_quantization"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(manifest))
        let cfgData = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
        try cfgData.write(to: output.appending(component: "config.json"))
        for name in try FileManager.default.contentsOfDirectory(atPath: source.path) where !name.hasPrefix(".") {
            let lowered = name.lowercased()
            guard lowered != "config.json", !lowered.hasSuffix(".safetensors"),
                  !lowered.hasSuffix(".safetensors.index.json") else { continue }
            try FileManager.default.copyItem(
                at: source.appending(component: name).resolvingSymlinksInPath(),
                to: output.appending(component: name))
        }
    }
}
