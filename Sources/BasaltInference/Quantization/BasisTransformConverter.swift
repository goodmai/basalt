import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

package enum BasisTransformConverter {
    static let maxShardBytes = 4_500_000_000
    static let calibrationBatches = 6
    static let calibrationSeqLen = 128
    static let blockGroupSize = 6

    package static func validateSourceDirectory(_ directory: URL) async throws {
        try await MACaQRefiner.validateSourceDirectory(directory)
    }

    package static func convertDirectory(
        source: URL,
        output: URL,
        target: QuantizationTarget,
        calibrationText: String? = nil,
        alpha: Float? = nil,
        onStep: (@Sendable (String) -> Void)? = nil
    ) async throws -> (quantizedLayers: Int, shards: Int, bytesAfter: Int) {
        try QTIPNativeLoader.validateOutput(source: source, output: output)
        try await validateSourceDirectory(source)
        let container = try await loadSourceContainer(from: source)
        let corpus = (calibrationText?.isEmpty == false ? calibrationText : nil) ?? PVTuner.defaultCalibrationCorpus
        let rawBatches = await GSQRCOConverter.tokenizeCalibration(
            container, corpus: corpus, count: calibrationBatches, seqLen: calibrationSeqLen)
        let vocab = vocabSize(in: source)
        let batches: [[Int32]] = rawBatches.isEmpty
            ? [(0 ..< 32).map { Int32(($0 * 11 + 3) % max(8, vocab)) }]
            : rawBatches
        do {
            return try await container.perform { ctx in
                switch target {
                case .flatQuant(let bits, let groupSize):
                    let fqAlpha = alpha ?? 0.5
                    let built = try quantizeFlatQuant(
                        ctx.model, bits: bits, groupSize: groupSize, alpha: fqAlpha, batches: batches, onStep: onStep)
                    let manifest = BasisTransformNativeLoader.FlatQuantManifest(
                        bits: bits, groupSize: groupSize, alpha: fqAlpha, modules: built.manifest)
                    let written = try writeDirectory(
                        model: ctx.model as Module, configKey: BasisTransformNativeLoader.flatQuantConfigKey,
                        manifest: manifest, source: source, output: output)
                    return (built.layers, written.shards, written.bytes)
                case .lirQuant(let bits, let groupSize, let subspaceDim):
                    let built = try quantizeLIRQuant(
                        ctx.model, bits: bits, groupSize: groupSize, subspaceDim: subspaceDim, batches: batches, onStep: onStep)
                    let manifest = BasisTransformNativeLoader.LIRQuantManifest(
                        bits: bits, groupSize: groupSize, subspaceDim: subspaceDim, modules: built.manifest)
                    let written = try writeDirectory(
                        model: ctx.model as Module, configKey: BasisTransformNativeLoader.lirQuantConfigKey,
                        manifest: manifest, source: source, output: output)
                    return (built.layers, written.shards, written.bytes)
                default:
                    throw BasaltError.invalidRequestStructure(details: "basis convert: unsupported target \(target.specString)")
                }
            }
        } catch {
            try? FileManager.default.removeItem(at: output)
            throw error
        }
    }

    private static func loadSourceContainer(from source: URL) async throws -> ModelContainer {
        let loader = FallbackCalibrationTokenizerLoader(vocabSize: vocabSize(in: source))
        if let nf4 = try await NF4NativeLoader.loadContainer(from: source, tokenizerLoader: loader) {
            return nf4
        }
        return try await loadModelContainer(from: source, using: loader)
    }

    private static func vocabSize(in directory: URL) -> Int {
        guard let data = try? Data(contentsOf: directory.appending(component: "config.json")),
              let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return 256
        }
        if let v = json["vocab_size"] as? Int, v > 0 { return v }
        if let tc = json["text_config"] as? [String: Any], let v = tc["vocab_size"] as? Int, v > 0 { return v }
        return 256
    }
}

extension BasisTransformConverter {
    private static func collectTargets(_ root: Module, blocks: Set<Int>, align: Int) -> [QTIPTargetEntry] {
        var out: [QTIPTargetEntry] = []
        for (path, mod) in root.namedModules() {
            guard let bIdx = PVTuner.textBlockIndex(path), blocks.contains(bIdx),
                  !path.contains("router"), !path.hasSuffix("in_proj_b"), !path.hasSuffix("in_proj_a")
            else { continue }
            if let ql = mod as? QuantizedLinear, type(of: ql) == QuantizedLinear.self, ql.globalScale == nil {
                let inF = (ql.weight.dim(1) * 32) / ql.bits
                guard inF.isMultiple(of: align) else { continue }
                let gKey = QTIPQuantizer.sharedInputGroupKey(path: path, inFeatures: inF, sharedMoERHT: false)
                out.append(QTIPTargetEntry(path: path, groupKey: gKey, module: ql, outFeatures: ql.weight.dim(0), inFeatures: inF))
            } else if type(of: mod) == Linear.self, let lin = mod as? Linear, lin.shape.1.isMultiple(of: align) {
                let gKey = QTIPQuantizer.sharedInputGroupKey(path: path, inFeatures: lin.shape.1, sharedMoERHT: false)
                out.append(QTIPTargetEntry(path: path, groupKey: gKey, module: lin, outFeatures: lin.shape.0, inFeatures: lin.shape.1))
            }
        }
        return out
    }

    private static func quantizeFlatQuant(
        _ model: any LanguageModel,
        bits: Int,
        groupSize: Int,
        alpha: Float,
        batches: [[Int32]],
        onStep: ((String) -> Void)?
    ) throws -> (layers: Int, manifest: [String: FlatQuantLayerSpec]) {
        let root = model as Module
        root.train(false)
        let blockGroups = QTIPQuantizer.groupedBlocks(in: root, stepSize: blockGroupSize)
        let maxTokens = max(1, batches.reduce(0) { $0 + $1.count })
        var manifest: [String: FlatQuantLayerSpec] = [:]
        for (gIdx, group) in blockGroups.enumerated() {
            try Task.checkCancellation()
            let targets = collectTargets(root, blocks: Set(group), align: groupSize)
            guard !targets.isEmpty else { continue }
            let boxes = try QTIPQuantizer.captureGroupActivations(
                model: model, root: root, targets: targets, batches: batches, maxTokens: maxTokens)
            for item in targets {
                let (dense, bias) = QTIPQuantizer.extractDenseWeight(item.module)
                let saveType: DType = dense.dtype == .float16 ? .float16 : .bfloat16
                let acts = boxes[item.groupKey].flatMap { $0.samples.isEmpty ? nil : concatenated($0.samples, axis: 0) }
                let factors = FlatQuantCalibrator.calibrateFactors(weight: dense, activations: acts, alpha: alpha)
                let wConj = FlatQuantCalibrator.conjugateWeight(dense, factors: factors).asType(saveType)
                let (wq, scales, biases) = MLX.quantized(wConj, groupSize: groupSize, bits: bits, mode: .affine)
                let basis = FlatQuantBasis(
                    p1: factors.p1.asType(saveType), p2: factors.p2.asType(saveType), clipBounds: factors.clipBounds)
                let layer = FlatQuantLinear(
                    packed: MACaQPackedWeights(weight: wq, scales: scales, biases: biases),
                    bias: bias, basis: basis, groupSize: groupSize, bits: bits, mode: .affine)
                eval(layer)
                try PVTuner.swapModule(item.path, with: layer, in: root)
                var spec = FlatQuantLayerSpec(
                    bits: bits, groupSize: groupSize, mode: .affine,
                    d1: factors.p1.dim(0), d2: factors.p2.dim(0), hasClipBounds: factors.clipBounds != nil)
                spec.scaleDType = "\(scales.dtype)"
                spec.hasBias = bias != nil
                manifest[item.path] = spec
            }
            Memory.clearCache()
            onStep?("flatquant group \(gIdx + 1)/\(blockGroups.count) quantized=\(manifest.count)")
        }
        MLXInferenceEngine.settle(root)
        return (manifest.count, manifest)
    }

    private static func quantizeLIRQuant(
        _ model: any LanguageModel,
        bits: Int,
        groupSize: Int,
        subspaceDim: Int,
        batches: [[Int32]],
        onStep: ((String) -> Void)?
    ) throws -> (layers: Int, manifest: [String: LIRQuantLayerSpec]) {
        let root = model as Module
        root.train(false)
        let blockGroups = QTIPQuantizer.groupedBlocks(in: root, stepSize: blockGroupSize)
        let maxTokens = max(1, batches.reduce(0) { $0 + $1.count })
        var manifest: [String: LIRQuantLayerSpec] = [:]
        for (gIdx, group) in blockGroups.enumerated() {
            try Task.checkCancellation()
            let targets = collectTargets(root, blocks: Set(group), align: max(32, groupSize))
            guard !targets.isEmpty else { continue }
            let boxes = try QTIPQuantizer.captureGroupActivations(
                model: model, root: root, targets: targets, batches: batches, maxTokens: maxTokens)
            for item in targets {
                let (dense, bias) = QTIPQuantizer.extractDenseWeight(item.module)
                let saveType: DType = dense.dtype == .float16 ? .float16 : .bfloat16
                let acts = boxes[item.groupKey].flatMap { $0.samples.isEmpty ? nil : concatenated($0.samples, axis: 0) }
                let kSub = min(item.inFeatures, max(32, subspaceDim))
                let opts = LIRQuantCalibrator.QuantizeOptions(
                    subspaceBlocks: max(1, kSub / 32), givensLayers: 2, bits: bits, groupSize: groupSize, mode: .affine)
                let layer = LIRQuantCalibrator.quantizeLayer(
                    linear: Linear(weight: dense.asType(saveType), bias: bias), activations: acts, options: opts)
                eval(layer)
                try PVTuner.swapModule(item.path, with: layer, in: root)
                var spec = LIRQuantLayerSpec(
                    bits: bits, groupSize: groupSize, mode: .affine,
                    subspaceDim: kSub, numLayers: layer.numLayers, pairsPerLayer: layer.pairsPerLayer)
                spec.scaleDType = "\(layer.scales.dtype)"
                spec.hasBias = bias != nil
                manifest[item.path] = spec
            }
            Memory.clearCache()
            onStep?("lirquant group \(gIdx + 1)/\(blockGroups.count) quantized=\(manifest.count)")
        }
        MLXInferenceEngine.settle(root)
        return (manifest.count, manifest)
    }
}

extension BasisTransformConverter {
    fileprivate static func writeDirectory<M: Encodable>(
        model: Module,
        configKey: String,
        manifest: M,
        source: URL,
        output: URL
    ) throws -> (shards: Int, bytes: Int) {
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        var writer = ShardWriter(directory: output, maxBytes: maxShardBytes, metadata: ["format": "mlx"])
        var totalBytes = 0
        var handledKeys: Set<String> = []
        for (leafPath, leaf) in model.leafModules().flattened().sorted(by: { $0.0 < $1.0 }) {
            let prefix = leafPath.isEmpty ? "" : "\(leafPath)."
            var group: [String: MLXArray] = [:]
            for (k, arr) in leaf.parameters().flattened() {
                let full = "\(prefix)\(k)"
                group[full] = arr
                handledKeys.insert(full)
                totalBytes += arr.nbytes
            }
            if !group.isEmpty { try writer.add(group) }
        }
        for (key, arr) in model.parameters().flattened().sorted(by: { $0.0 < $1.0 }) where !handledKeys.contains(key) {
            totalBytes += arr.nbytes
            try writer.add([key: arr])
        }
        let shards = try writer.finish()
        let cfgURL = source.appending(component: "config.json")
        var cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: cfgURL)) as? [String: Any] ?? [:]
        cfg[configKey] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(manifest))
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

    package static func exportCanonicalShards(
        from source: URL,
        to output: URL,
        dtype: DType,
        prepareDenseSidecars: (URL, URL, DType) throws -> Void
    ) throws -> WeightConverter.Summary {
        let fqManifest = try BasisTransformNativeLoader.flatQuantManifest(in: source)
        let lirManifest = try BasisTransformNativeLoader.lirQuantManifest(in: source)
        let rawCfg = try Data(contentsOf: source.appending(component: "config.json"))
        let perLayer = (try? JSONDecoder.json5().decode(BaseConfiguration.self, from: rawCfg))?.perLayerQuantization
        try prepareDenseSidecars(source, output, dtype)
        let shardNames = try FileManager.default.contentsOfDirectory(atPath: source.path)
            .filter { $0.hasSuffix(".safetensors") }
            .sorted()
        var writer = ShardWriter(directory: output, maxBytes: 5 << 30, metadata: ["format": "mlx"])
        var summary = WeightConverter.Summary()
        summary.bytesBefore = Int(UniversalSourceReader.directoryWeightBytes(source))
        let decodeCtx = ShardDecodeContext(
            fqModules: fqManifest?.modules,
            lirModules: lirManifest?.modules,
            perLayer: perLayer)
        for name in shardNames {
            let arrays = try loadArrays(url: source.appending(component: name))
            for key in arrays.keys.sorted() where !isCompanionSuffix(key) {
                guard let tensor = arrays[key] else { continue }
                let (outTensor, consumed) = decodeShardEntry(
                    key: key, tensor: tensor, arrays: arrays, dtype: dtype, context: decodeCtx)
                eval(outTensor)
                summary.tensors += consumed
                summary.outputTensors += 1
                summary.bytesAfter += outTensor.nbytes
                try writer.add([key: outTensor])
            }
            Memory.clearCache()
        }
        summary.shards = try writer.finish()
        return summary
    }

    private struct ShardDecodeContext {
        var fqModules: [String: FlatQuantLayerSpec]?
        var lirModules: [String: LIRQuantLayerSpec]?
        var perLayer: BaseConfiguration.PerLayerQuantization?
    }

    private static func isCompanionSuffix(_ key: String) -> Bool {
        let suffixes = [
            ".scales", ".biases", ".p1", ".p2", ".clip_bounds",
            ".row_scale", ".perm", ".givens_pairs", ".givens_angles", ".diag_scale",
        ]
        return suffixes.contains { key.hasSuffix($0) }
    }

    private static func decodeShardEntry(
        key: String,
        tensor: MLXArray,
        arrays: [String: MLXArray],
        dtype: DType,
        context: ShardDecodeContext
    ) -> (MLXArray, Int) {
        guard key.hasSuffix(".weight") else {
            return (tensor.dtype.isFloatingPoint ? tensor.asType(dtype) : tensor, 1)
        }
        let base = String(key.dropLast(".weight".count))
        let alt = base.hasPrefix("language_model.") ? String(base.dropFirst("language_model.".count)) : "language_model.\(base)"
        if let spec = context.fqModules?[base] ?? context.fqModules?[alt],
           let scales = arrays["\(base).scales"], let p1 = arrays["\(base).p1"], let p2 = arrays["\(base).p2"] {
            let biases = arrays["\(base).biases"]
            let clip = arrays["\(base).clip_bounds"]
            let row = arrays["\(base).row_scale"]
            let layer = FlatQuantLinear(
                packed: MACaQPackedWeights(weight: tensor, scales: scales, biases: biases),
                bias: nil, basis: FlatQuantBasis(p1: p1, p2: p2, clipBounds: clip, rowScale: row),
                groupSize: spec.groupSize, bits: spec.bits, mode: spec.quantizationMode)
            let count = 4 + (biases != nil ? 1 : 0) + (clip != nil ? 1 : 0) + (row != nil ? 1 : 0)
            return (layer.unrotatedDenseWeight().asType(dtype), count)
        }
        if let spec = context.lirModules?[base] ?? context.lirModules?[alt],
           let scales = arrays["\(base).scales"], let perm = arrays["\(base).perm"],
           let pairs = arrays["\(base).givens_pairs"], let angles = arrays["\(base).givens_angles"],
           let diag = arrays["\(base).diag_scale"] {
            let biases = arrays["\(base).biases"]
            let basis = LIRBasisParameters(
                perm: perm, givensPairs: pairs, givensAngles: angles, diagScale: diag,
                numLayers: spec.numLayers, pairsPerLayer: spec.pairsPerLayer)
            let layer = LIRQuantLinear(
                packed: MACaQPackedWeights(weight: tensor, scales: scales, biases: biases),
                bias: nil, basis: basis, groupSize: spec.groupSize, bits: spec.bits, mode: spec.quantizationMode)
            return (layer.unrotatedDenseWeight().asType(dtype), 6 + (biases != nil ? 1 : 0))
        }
        if let scales = arrays["\(base).scales"] {
            let biases = arrays["\(base).biases"]
            let q = context.perLayer?.quantization(layer: base) ?? context.perLayer?.quantization(layer: alt)
            let bits = q?.bits ?? ((tensor.dim(-1) * 32) / max(1, scales.dim(-1) * 64))
            let group = q?.groupSize ?? 64
            let mode = q?.mode ?? (scales.dtype == .uint8 && biases == nil ? .mxfp4 : .affine)
            let deq = MLX.dequantized(tensor, scales: scales, biases: biases, groupSize: group, bits: bits, mode: mode, dtype: dtype)
            return (deq, 2 + (biases != nil ? 1 : 0))
        }
        return (tensor.dtype.isFloatingPoint ? tensor.asType(dtype) : tensor, 1)
    }
}
