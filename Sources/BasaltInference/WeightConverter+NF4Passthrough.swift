import BasaltCore
import Foundation
import MLX
import MLXLMCommon

extension WeightConverter {
    static func recordingNF4Passthrough(_ config: [String: Any], _ layout: SourceLayout) -> [String: Any] {
        guard let quantization = layout.quantization else { return config }
        var block = config["quantization"] as? [String: Any] ?? [:]
        for key in layout.shardOf.keys where key.hasSuffix(".biases") {
            let base = String(key.dropLast(".biases".count))
            guard layout.shardOf[base + ".scales"] != nil, !isLinearPath(base + ".weight"),
                  let layer = quantization.quantization(layer: base) else { continue }
            for name in MixedBits.loaderNames(base) { block[name] = ["bits": layer.bits, "group_size": layer.groupSize, "mode": "affine"] }
        }
        var next = config
        next["quantization"] = block
        return next
    }
}

enum NF4Passthrough {
    struct Layer: Sendable {
        let bits: Int
        let groupSize: Int
    }

    static func affineLayers(in config: [String: Any]?) -> [String: Layer] {
        let block = config?["quantization"] as? [String: Any] ?? [:]
        var layers: [String: Layer] = [:]
        for (key, value) in block {
            guard let entry = value as? [String: Any], entry["mode"] as? String == "affine",
                  let bits = entry["bits"] as? Int, let group = entry["group_size"] as? Int else { continue }
            layers[key] = Layer(bits: bits, groupSize: group)
        }
        return layers
    }

    static func dequantized(
        _ weight: MLXArray, scales: MLXArray, biases: MLXArray, base: String,
        layers: [String: Layer], dtype: DType
    ) throws -> MLXArray {
        guard let layer = MixedBits.loaderNames(base).lazy.compactMap({ layers[$0] }).first else {
            throw BasaltError.modelConfigurationInvalid(path: base, reason: "an affine layer in an nf4 pack has no recorded bits and group size; reconvert it with a current basalt")
        }
        return MLX.dequantized(weight, scales: scales, biases: biases, groupSize: layer.groupSize, bits: layer.bits, mode: .affine, dtype: dtype)
    }
}
