import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension ModelOptImporter {
    typealias ExpertParts = [String: [Int: [String: MLXArray]]]

    package func run(output: URL, model: any LanguageModel) throws -> Summary {
        let arrays = try Self.loadShards(in: source)
        var summary = Summary()
        var plain: [String: MLXArray] = [:]
        var experts: ExpertParts = [:]
        for key in arrays.keys.sorted() {
            if Self.isCompanion(key) {
                summary.dropped += 1
                continue
            }
            guard key.hasSuffix(".weight") else {
                plain[key] = arrays[key]
                continue
            }
            let stem = String(key.dropLast(".weight".count))
            let converted = try convert(stem: stem, arrays: arrays, summary: &summary)
            if let expert = Self.expertKey(stem) {
                let parts = Dictionary(uniqueKeysWithValues: converted.map { (String($0.key.dropFirst(stem.count + 1)), $0.value) })
                experts["\(expert.prefix).\(expert.projection)", default: [:]][expert.index] = parts
            } else {
                plain.merge(converted) { _, new in new }
            }
        }
        var weights = model.sanitize(weights: plain, metadata: [:])
        let stacked = try Self.stack(experts, targets: Self.switchTargets(model))
        summary.stackedExperts = experts.count
        weights.merge(stacked) { _, new in new }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        summary.bytes = try Self.write(weights, to: output)
        try writeConfig(output: output, weights: weights)
        try Self.copySidecars(from: source, to: output)
        return summary
    }

    func convert(stem: String, arrays: [String: MLXArray], summary: inout Summary) throws(Refused) -> [String: MLXArray] {
        guard let weight = arrays[stem + ".weight"] else { return [:] }
        let kind = Self.kind(stem: stem, in: arrays)
        if kind != .plain, excluded(stem) {
            throw Refused(description: "\(stem) matches exclude_modules but carries quantization scales")
        }
        switch kind {
        case .nvfp4:
            summary.nvfp4 += 1
            return Self.nvfp4(stem: stem, weight: weight, scale: arrays[stem + ".weight_scale"]!, scale2: arrays[stem + ".weight_scale_2"]!)
        case .fp8:
            summary.fp8 += 1
            let (converted, rmse) = Self.fp8(stem: stem, weight: weight, scale: arrays[stem + ".weight_scale"]!)
            summary.fp8MaxRelativeRMSE = max(summary.fp8MaxRelativeRMSE, rmse)
            return converted
        case .plain:
            summary.plain += 1
            var out = [stem + ".weight": weight]
            out[stem + ".bias"] = arrays[stem + ".bias"]
            return out
        }
    }

    static func switchTargets(_ model: any LanguageModel) -> [String: String] {
        var targets: [String: String] = [:]
        for (path, module) in model.leafModules().flattened() where module is SwitchLinear {
            guard let layer = path.firstMatch(of: /layers\.(\d+)\./)?.1, let projection = path.split(separator: ".").last else { continue }
            targets["\(layer).\(projection)"] = path
        }
        return targets
    }

    static func stack(_ experts: ExpertParts, targets: [String: String]) throws(Refused) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (group, byIndex) in experts {
            guard let layer = group.firstMatch(of: /layers\.(\d+)\./)?.1, let projection = group.split(separator: ".").last,
                  let target = targets["\(layer).\(projection)"]
            else { throw Refused(description: "the model has no switch layer for ModelOpt experts \(group)") }
            let indices = byIndex.keys.sorted()
            guard indices == Array(0 ..< indices.count), let parts = byIndex[0]?.keys else {
                throw Refused(description: "ModelOpt experts \(group) are not numbered 0..<\(indices.count)")
            }
            for part in parts {
                let rows = try indices.map { index throws(Refused) -> MLXArray in
                    guard let array = byIndex[index]?[part] else { throw Refused(description: "expert \(index) of \(group) has no \(part)") }
                    return array
                }
                let joined = stacked(rows)
                out["\(target).\(part)"] = part == "global_scale" ? joined.reshaped([-1]) : joined
            }
        }
        return out
    }

    static func loadShards(in directory: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        var arrays: [String: MLXArray] = [:]
        for file in files {
            arrays.merge(try loadArrays(url: file)) { _, new in new }
        }
        return arrays
    }

    static func write(_ weights: [String: MLXArray], to output: URL) throws -> Int {
        var writer = ShardWriter(directory: output, maxBytes: 4 << 30, metadata: ["format": "mlx"])
        var bytes = 0
        for key in weights.keys.sorted() {
            guard let array = weights[key] else { continue }
            bytes += array.nbytes
            try writer.add([key: array])
        }
        _ = try writer.finish()
        return bytes
    }

    func writeConfig(output: URL, weights: [String: MLXArray]) throws {
        let data = try Data(contentsOf: source.appending(component: "config.json"))
        guard var config = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Refused(description: "config.json is not a JSON object")
        }
        config["quantization_config"] = nil
        var quantization: [String: Any] = ["group_size": groupSize, "bits": 4, "mode": "nvfp4"]
        for key in weights.keys where key.hasSuffix(".scales") {
            let stem = String(key.dropLast(".scales".count))
            if weights[stem + ".global_scale"] == nil {
                quantization[stem] = ["group_size": Self.fp8GroupSize, "bits": 8, "mode": "affine"]
            }
        }
        config["quantization"] = quantization
        config["basalt_import"] = ["format": "modelopt", "quant_algo": algorithm, "source": source.lastPathComponent]
        try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appending(component: "config.json"))
    }

    static func copySidecars(from source: URL, to output: URL) throws {
        let skip: Set<String> = ["config.json", "hf_quant_config.json", "model.safetensors.index.json"]
        for file in try FileManager.default.contentsOfDirectory(at: source, includingPropertiesForKeys: nil)
        where file.pathExtension != "safetensors" && !skip.contains(file.lastPathComponent) {
            let target = output.appending(component: file.lastPathComponent)
            if !FileManager.default.fileExists(atPath: target.path) {
                try FileManager.default.copyItem(at: file, to: target)
            }
        }
    }
}

extension ModelOptImporter {
    package static func skeleton(source: URL) async throws -> any LanguageModel {
        await MLXInferenceEngine.registerAliases()
        let configuration = try Data(contentsOf: source.appending(component: "config.json"))
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configuration)
        for family in ExpertStreamingLoader.families where await family.types.contains(base.modelType) {
            return try await family.types.createModel(configuration: configuration, modelType: base.modelType)
        }
        throw Refused(description: "no model factory knows model_type \(base.modelType)")
    }
}
