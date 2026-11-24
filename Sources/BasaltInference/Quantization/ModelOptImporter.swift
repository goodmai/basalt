import BasaltCore
import Foundation
import MLX

package struct ModelOptImporter: Sendable {
    package struct Refused: Error, Equatable, CustomStringConvertible {
        package let description: String
    }

    package struct Summary: Sendable, Equatable {
        package var nvfp4 = 0
        package var fp8 = 0
        package var plain = 0
        package var stackedExperts = 0
        package var dropped = 0
        package var fp8MaxRelativeRMSE: Float = 0
        package var bytes = 0
    }

    package enum Kind: Equatable {
        case nvfp4, fp8, plain
    }

    static let supported = ["NVFP4", "MIXED_PRECISION"]
    static let companions = [".weight_scale", ".weight_scale_2", ".input_scale", ".k_scale", ".v_scale"]
    static let globalScaleFactor: Float = 448 * 6
    static let fp8GroupSize = 64

    package let source: URL
    package let algorithm: String
    package let groupSize: Int
    package let excludes: [String]

    package init(source: URL) throws(Refused) {
        let (algorithm, groupSize, excludes) = try Self.readQuantConfig(source)
        guard Self.supported.contains(algorithm) else {
            throw Refused(description: "ModelOpt quant_algo \(algorithm) is not imported; only \(Self.supported.joined(separator: " and ")) are (#677)")
        }
        guard groupSize == 16 else { throw Refused(description: "ModelOpt NVFP4 group_size \(groupSize) is not 16") }
        (self.source, self.algorithm, self.groupSize, self.excludes) = (source, algorithm, groupSize, excludes)
    }

    static func readQuantConfig(_ source: URL) throws(Refused) -> (String, Int, [String]) {
        let hf = source.appending(component: "hf_quant_config.json")
        if let data = try? Data(contentsOf: hf), let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            let quantization = json["quantization"] as? [String: Any] ?? json
            guard let algorithm = quantization["quant_algo"] as? String else { throw Refused(description: "hf_quant_config.json has no quant_algo") }
            return (algorithm, quantization["group_size"] as? Int ?? 16, quantization["exclude_modules"] as? [String] ?? [])
        }
        guard let config = try? Data(contentsOf: source.appending(component: "config.json")),
              let json = try? JSONSerialization.jsonObject(with: config) as? [String: Any],
              let quantization = json["quantization_config"] as? [String: Any],
              (quantization["quant_method"] as? String) == "modelopt"
        else { throw Refused(description: "\(source.lastPathComponent) is not a ModelOpt checkpoint (no hf_quant_config.json, no quant_method modelopt)") }
        guard let algorithm = quantization["quant_algo"] as? String else { throw Refused(description: "quantization_config has no quant_algo") }
        return (algorithm, quantization["group_size"] as? Int ?? 16, quantization["exclude_modules"] as? [String] ?? quantization["ignore"] as? [String] ?? [])
    }

    static func kind(stem: String, in arrays: [String: MLXArray]) -> Kind {
        if arrays[stem + ".weight_scale_2"] != nil, arrays[stem + ".weight_scale"] != nil { return .nvfp4 }
        if let scale = arrays[stem + ".weight_scale"], scale.size == 1, arrays[stem + ".weight"]?.dtype == .uint8 { return .fp8 }
        return .plain
    }

    static func isCompanion(_ key: String) -> Bool {
        companions.contains { key.hasSuffix($0) }
    }

    static func nvfp4(stem: String, weight: MLXArray, scale: MLXArray, scale2: MLXArray) -> [String: MLXArray] {
        [
            stem + ".weight": weight.view(dtype: .uint32),
            stem + ".scales": scale.asType(.uint8),
            stem + ".global_scale": (scale2.asType(.float32) * globalScaleFactor).reshaped([]),
        ]
    }

    static func fp8(stem: String, weight: MLXArray, scale: MLXArray) -> (arrays: [String: MLXArray], relativeRMSE: Float) {
        let dense = fromFP8(weight, dtype: .float32) * scale.asType(.float32).reshaped([])
        let (wq, scales, biases) = MLX.quantized(dense, groupSize: fp8GroupSize, bits: 8)
        let back = MLX.dequantized(wq, scales: scales, biases: biases, groupSize: fp8GroupSize, bits: 8, dtype: .float32)
        let rmse = sqrt(mean(square(back - dense))) / maximum(sqrt(mean(square(dense))), MLXArray(Float(1e-12)))
        var arrays = [stem + ".weight": wq, stem + ".scales": scales.asType(.bfloat16)]
        arrays[stem + ".biases"] = biases?.asType(.bfloat16)
        return (arrays, rmse.item(Float.self))
    }

    static func expertKey(_ stem: String) -> (prefix: String, index: Int, projection: String)? {
        guard let match = stem.firstMatch(of: /^(.*\.experts)\.(\d+)\.(\w+)$/), let index = Int(match.2) else { return nil }
        return (String(match.1), index, String(match.3))
    }

    static func matches(_ pattern: String, _ name: String) -> Bool {
        guard pattern.hasSuffix("*") else { return name == pattern }
        return name.hasPrefix(String(pattern.dropLast()))
    }

    func excluded(_ stem: String) -> Bool {
        excludes.contains { Self.matches($0, stem) }
    }
}
