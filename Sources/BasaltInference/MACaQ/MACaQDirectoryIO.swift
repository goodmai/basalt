import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class MACaQShardTeacherReader {
    let directory: URL
    let shardOf: [String: URL]
    let isNF4: Bool
    let quantization: BaseConfiguration.PerLayerQuantization?
    var cachedShard: URL?
    var cachedArrays: [String: MLXArray] = [:]

    init(directory: URL) throws {
        self.directory = directory
        let cfgURL = directory.appending(component: "config.json")
        let cfgData = try Data(contentsOf: cfgURL)
        let cfgJSON = (try JSONSerialization.jsonObject(with: cfgData) as? [String: Any]) ?? [:]
        self.isNF4 = NF4NativeLoader.isNF4Config(cfgJSON)
        let index = (try? Data(contentsOf: directory.appending(component: "model.safetensors.index.json")))
            .flatMap { (try? JSONSerialization.jsonObject(with: $0)) as? [String: Any] }
        let files = Set(try FileManager.default.contentsOfDirectory(atPath: directory.path))
        let shards = ModelDeepCheck.weightFiles(in: files, prefix: "", weightMap: index?["weight_map"] as? [String: String])
            .map { directory.appending(component: $0) }
        var map: [String: URL] = [:]
        for shard in shards {
            for key in try loadArrays(url: shard).keys { map[key] = shard }
        }
        self.shardOf = map
        if !isNF4, cfgJSON["quantization"] != nil {
            self.quantization = try? JSONDecoder.json5().decode(BaseConfiguration.self, from: cfgData).perLayerQuantization
        } else {
            self.quantization = nil
        }
    }

    func denseWeight(for modulePath: String) throws -> MLXArray? {
        for base in candidateBases(for: modulePath) {
            if let tensor = try fetch(base + ".weight") {
                return try decode(base: base, weight: tensor)
            }
        }
        return nil
    }

    private func candidateBases(for path: String) -> [String] {
        var out = [path]
        if path.hasPrefix("language_model.") {
            out.append(String(path.dropFirst("language_model.".count)))
        } else {
            out.append("language_model." + path)
        }
        return out
    }

    private func fetch(_ key: String) throws -> MLXArray? {
        guard let url = shardOf[key] else { return nil }
        if cachedShard != url {
            cachedArrays = try loadArrays(url: url)
            cachedShard = url
        }
        return cachedArrays[key]
    }

    private func decode(base: String, weight: MLXArray) throws -> MLXArray {
        guard let scales = try fetch(base + ".scales") else {
            return weight.asType(.float32)
        }
        let biases = try fetch(base + ".biases")
        if isNF4, biases == nil {
            let groupSize = (weight.dim(-1) * 8) / max(1, scales.dim(-1))
            let deq = NF4Metal.dequantized(weight, scales: scales, groupSize: groupSize).asType(.float32)
            eval(deq)
            return deq
        }
        if scales.dtype == .uint8, biases == nil {
            let deq = MLX.dequantized(weight, scales: scales, biases: nil, groupSize: 32, bits: 4, mode: .mxfp4, dtype: .float32)
            eval(deq)
            return deq
        }
        if let q = quantization?.quantization(layer: base) {
            let deq = MLX.dequantized(
                weight, scales: scales, biases: biases,
                groupSize: q.groupSize, bits: q.bits, mode: q.mode, dtype: .float32)
            eval(deq)
            return deq
        }
        return weight.asType(.float32)
    }
}

extension MACaQRefiner {
    static let maxShardBytes = 4_500_000_000

    package static func validateSourceDirectory(_ directory: URL) async throws {
        let cfgURL = directory.appending(component: "config.json")
        guard let raw = try? Data(contentsOf: cfgURL),
              let json = try? JSONSerialization.jsonObject(with: raw) as? [String: Any]
        else {
            throw BasaltError.invalidRequestStructure(details: "macaq: missing or invalid config.json in \(directory.path)")
        }
        for key in [MACaQNativeLoader.configKey, "qtip_quantization", "gguf_quantization"] where json[key] != nil {
            throw BasaltError.invalidRequestStructure(details: "macaq: source checkpoint already uses custom block layout (\(key))")
        }
        guard let modelType = (json["model_type"] as? String)
            ?? ((json["text_config"] as? [String: Any])?["model_type"] as? String)
        else {
            throw BasaltError.invalidRequestStructure(details: "macaq: missing model_type in config.json")
        }
        if modelType == PrismHadamard.modelType {
            throw BasaltError.invalidRequestStructure(details: "macaq: source checkpoint is already Hadamard-rotated (\(modelType))")
        }
        var supported = false
        for family in ExpertStreamingLoader.families where await family.types.contains(modelType) {
            supported = true
            break
        }
        guard supported else {
            throw BasaltError.invalidRequestStructure(details: "macaq: model_type '\(modelType)' is not a supported causal LanguageModel")
        }
    }

    package static func refineConvertedDirectory(
        output: URL,
        teacher: URL,
        calibrationText: String? = nil,
        config: MACaQConfig = MACaQConfig(),
        onStep: (@Sendable (String) -> Void)? = nil
    ) async throws -> MACaQSummary {
        try await validateSourceDirectory(output)
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        defer { await withTaskCancellationShield { await engine.unload(); Memory.clearCache() } }
        try await engine.load(modelPath: output.path)
        guard let container = await engine.container else { throw BasaltError.modelNotLoaded }
        let corpus = (calibrationText?.isEmpty == false ? calibrationText : nil) ?? PVTuner.defaultCalibrationCorpus
        let batches = await GSQRCOConverter.tokenizeCalibration(
            container, corpus: corpus, count: config.calibrationBatches, seqLen: config.calibrationSeqLen)
        guard !batches.isEmpty else {
            throw BasaltError.invalidRequestStructure(details: "macaq-refine: calibration text produced zero token batches")
        }
        return try await container.perform { ctx in
            let reader = try MACaQShardTeacherReader(directory: teacher)
            var summary = try refineModel(
                ctx.model, calibrationTokenBatches: batches, config: config,
                teacherWeight: { try reader.denseWeight(for: $0) }, onStep: onStep)
            summary.rewrittenTensors = try PVTuner.writeTunedShards(from: ctx.model as Module, into: output)
            return summary
        }
    }

    package static func convertDirectory(
        source: URL,
        output: URL,
        targetMode: MACaQTargetMode,
        calibrationText: String? = nil,
        config: MACaQConfig = MACaQConfig(),
        onStep: (@Sendable (String) -> Void)? = nil
    ) async throws -> (summary: MACaQSummary, shards: Int, bytesAfter: Int) {
        try QTIPNativeLoader.validateOutput(source: source, output: output)
        try await validateSourceDirectory(source)
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        defer { await withTaskCancellationShield { await engine.unload(); Memory.clearCache() } }
        try await engine.load(modelPath: source.path)
        guard let container = await engine.container else { throw BasaltError.modelNotLoaded }
        let corpus = (calibrationText?.isEmpty == false ? calibrationText : nil) ?? PVTuner.defaultCalibrationCorpus
        let batches = await GSQRCOConverter.tokenizeCalibration(
            container, corpus: corpus, count: config.calibrationBatches, seqLen: config.calibrationSeqLen)
        guard !batches.isEmpty else {
            throw BasaltError.invalidRequestStructure(details: "macaq: calibration text produced zero token batches")
        }
        do {
            return try await container.perform { ctx in
                let result = try quantizeModel(
                    ctx.model, targetMode: targetMode, calibrationTokenBatches: batches, config: config, onStep: onStep)
                let written = try writeMACaQDirectory(
                    model: ctx.model as Module, manifest: result.manifest, alpha: config.alpha,
                    source: source, output: output)
                var summary = result.summary
                summary.rewrittenTensors = result.manifest.count
                return (summary, written.shards, written.bytes)
            }
        } catch {
            try? FileManager.default.removeItem(at: output)
            throw error
        }
    }

    package static func writeMACaQDirectory(
        model: Module,
        manifest: [String: MACaQLayerSpec],
        alpha: Float,
        source: URL,
        output: URL
    ) throws -> (shards: Int, bytes: Int) {
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        var writer = ShardWriter(directory: output, maxBytes: maxShardBytes, metadata: ["format": "mlx"])
        var totalBytes = 0
        for (key, arr) in model.parameters().flattened().sorted(by: { $0.0 < $1.0 }) {
            totalBytes += arr.nbytes
            try writer.add([key: arr])
        }
        let shards = try writer.finish()
        let cfgURL = source.appending(component: "config.json")
        var cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: cfgURL)) as? [String: Any] ?? [:]
        let manifestObj = MACaQNativeLoader.Manifest(version: 1, alpha: alpha, modules: manifest)
        cfg[MACaQNativeLoader.configKey] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(manifestObj))
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
}
