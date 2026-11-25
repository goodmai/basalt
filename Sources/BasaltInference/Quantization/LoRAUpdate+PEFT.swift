import Foundation
import MLX

extension LoRAUpdate {
    init(peftDirectory directory: URL, keys: [String: URL]) throws {
        let configURL = directory.appending(component: "adapter_config.json")
        let configuration: PEFTLoRAConfiguration
        do {
            configuration = try JSONDecoder().decode(PEFTLoRAConfiguration.self, from: Data(contentsOf: configURL))
        } catch {
            throw WeightConverter.Failure.malformed("adapter_config.json is not a PEFT LoRA config: \(error.localizedDescription)")
        }
        if let refusal = configuration.refusal {
            throw WeightConverter.Failure.malformed(refusal)
        }
        let raw: [String: MLXArray]
        do {
            raw = try loadArrays(url: directory.appending(component: PEFTLoRAReader.weightsFile))
        } catch {
            throw WeightConverter.Failure.malformed("\(PEFTLoRAReader.weightsFile) is unreadable: \(error.localizedDescription)")
        }
        var aByLayer: [String: MLXArray] = [:]
        var bByLayer: [String: MLXArray] = [:]
        for (key, tensor) in raw {
            let isA = key.hasSuffix(".lora_A.weight")
            let isB = key.hasSuffix(".lora_B.weight")
            guard isA || isB else {
                throw WeightConverter.Failure.malformed("\(key) is neither a lora_A nor a lora_B weight")
            }
            let suffixLen = isA ? ".lora_A.weight".count : ".lora_B.weight".count
            let peftPath = String(key.dropLast(suffixLen).trimmingPrefix("base_model.model."))
            guard let matched = Self.matchedWeightBase(peftPath, in: keys) else {
                throw WeightConverter.Failure.malformed("\(key) belongs to no linear weight in the source model")
            }
            if isA {
                aByLayer[matched] = tensor.transposed()
            } else {
                bByLayer[matched] = tensor.transposed()
            }
        }
        let allLayers = Set(aByLayer.keys).union(bByLayer.keys)
        guard !allLayers.isEmpty else {
            throw WeightConverter.Failure.malformed("\(directory.path) holds no PEFT lora_A/lora_B pairs")
        }
        var resolved: [String: (a: MLXArray, b: MLXArray)] = [:]
        for layer in allLayers {
            guard let a = aByLayer[layer], let b = bByLayer[layer] else {
                throw WeightConverter.Failure.malformed("PEFT adapter layer \(layer) has only one of lora_A and lora_B")
            }
            resolved[layer] = (a, b)
        }
        self.scale = configuration.scale
        self.pairs = resolved
    }

    static func matchedWeightBase(_ peftPath: String, in keys: [String: URL]) -> String? {
        PEFTLoRAReader.prefixRewrites.lazy.compactMap { rewrite -> String? in
            guard peftPath.hasPrefix(rewrite.from) else { return nil }
            let candidate = rewrite.to + peftPath.dropFirst(rewrite.from.count)
            return keys[candidate + ".weight"] == nil ? nil : candidate
        }.first
    }
}
