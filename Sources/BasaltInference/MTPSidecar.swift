import BasaltCore
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

final class MTPDrafter: @unchecked Sendable {
    let model: any MTPDrafterModel
    let source: String

    init(model: any MTPDrafterModel, source: String) {
        self.model = model
        self.source = source
    }
}

enum MTPSidecarError: LocalizedError, Equatable {
    case malformed(String)

    var errorDescription: String? {
        switch self {
        case .malformed(let reason): "MTP head not usable: \(reason)"
        }
    }
}

enum MTPSidecar {
    static func file(in directory: URL) -> URL? {
        let config = ModelCache.config(at: directory)
        let declared = (config?["mlx_lm_extra_tensors"] as? [String: Any])?["mtp_file"] as? String
        return ([declared].compactMap { $0 } + ["mtp.safetensors", "mtp/weights.safetensors"])
            .map { directory.appending(path: $0) }
            .first { FileManager.default.fileExists(atPath: $0.path) }
    }

    static func quantization(of weights: [String: MLXArray], hiddenSize: Int) throws -> (groupSize: Int, bits: Int)? {
        guard let scales = weights["mtp.fc.scales"] else { return nil }
        guard let packed = weights["mtp.fc.weight"], packed.dtype == .uint32, packed.ndim == 2, scales.ndim == 2 else {
            throw MTPSidecarError.malformed("mtp.fc has scales but no packed uint32 weight")
        }
        let inputs = 2 * hiddenSize
        let bits = packed.dim(1) * 32 / inputs
        guard scales.dim(1) > 0, inputs % scales.dim(1) == 0, [2, 3, 4, 5, 6, 8].contains(bits), packed.dim(1) * 32 == inputs * bits else {
            throw MTPSidecarError.malformed("mtp.fc shapes \(packed.shape) and \(scales.shape) do not match hidden size \(hiddenSize)")
        }
        return (inputs / scales.dim(1), bits)
    }

    static func load(directory: URL, target: any LanguageModel) throws -> MTPDrafter? {
        guard target is Qwen35Model, let url = file(in: directory) else { return nil }
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let configuration = try JSONDecoder.json5().decode(Qwen35Configuration.self, from: data)
        let textConfig = (try JSONSerialization.jsonObject(with: data) as? [String: Any])?["text_config"] as? [String: Any]
        guard let hiddenSize = textConfig?["hidden_size"] as? Int else { throw MTPSidecarError.malformed("config.json has no text_config.hidden_size") }
        var weights = try loadArrays(url: url)
        guard let norm = weights["mtp.norm.weight"] else { throw MTPSidecarError.malformed("\(url.lastPathComponent) has no mtp.norm.weight") }
        let preconvertedNorms = norm.asType(.float32).mean().item(Float.self) > 0.5
        let model = Qwen35MTPDraftModel(configuration, preconvertedNorms: preconvertedNorms)
        weights = model.sanitize(weights: weights)
        if let quantization = try quantization(of: weights, hiddenSize: hiddenSize) {
            quantize(model: model) { path, _ in
                weights["\(path).scales"] == nil ? nil : (quantization.groupSize, quantization.bits, .affine)
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return MTPDrafter(model: model, source: url.path)
    }
}
