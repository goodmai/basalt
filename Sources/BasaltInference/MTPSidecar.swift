import BasaltCore
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

struct MTPPrefix: Sendable {
    let layers: [[MaterializedArray]]
    let nextPosition: Int
    let carryHidden: MaterializedArray

    init?(_ state: MTPDrafterState) {
        guard let carryHidden = state.carryHidden, state.cache.allSatisfy({ type(of: $0) == KVCacheSimple.self }) else {
            return nil
        }
        let layers = state.cache.map(\.state)
        eval(layers.joined() + [carryHidden])
        self.layers = layers.map { $0.map { $0.materialized() } }
        nextPosition = state.nextPosition
        self.carryHidden = carryHidden.materialized()
    }

    var state: MTPDrafterState {
        MTPDrafterState(cache: layers.map(Self.cache), nextPosition: nextPosition, carryHidden: carryHidden[.ellipsis])
    }

    private static func cache(_ state: [MaterializedArray]) -> any KVCache {
        let cache = KVCacheSimple()
        if !state.isEmpty { cache.state = state.map { $0[.ellipsis] } }
        return cache
    }
}

final class GreedyDraftingMTPDrafter: Module, StatefulMTPDrafterModel {
    let inner: any StatefulMTPDrafterModel
    private let argMax = ArgMaxSampler()

    init(_ inner: any StatefulMTPDrafterModel) {
        self.inner = inner
        super.init()
    }

    var maximumBlockSize: Int? { inner.maximumBlockSize }
    var requiresSharedTargetKV: Bool { inner.requiresSharedTargetKV }
    var requiresPromptPrefill: Bool { inner.requiresPromptPrefill }
    var requiresGreedySampling: Bool { false }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] { inner.sanitize(weights: weights) }

    func makeState(parameters: GenerateParameters?) -> MTPDrafterState { inner.makeState(parameters: parameters) }

    func prepareDrafterState(target: any LanguageModel, input: MTPPrefillInput, state: inout MTPDrafterState, sampler _: any LogitSampler) {
        inner.prepareDrafterState(target: target, input: input, state: &state, sampler: argMax)
    }

    func draftBlock(target: any LanguageModel, input: MTPDraftInput, sampler _: any LogitSampler) -> MLXArray {
        inner.draftBlock(target: target, input: input, sampler: argMax)
    }

    func draftBlock(target: any LanguageModel, input: MTPDraftInput, state: inout MTPDrafterState, sampler _: any LogitSampler) -> MLXArray {
        inner.draftBlock(target: target, input: input, state: &state, sampler: argMax)
    }

    func commitDrafterState(target: any LanguageModel, input: MTPCommitInput, state: inout MTPDrafterState, sampler _: any LogitSampler) {
        inner.commitDrafterState(target: target, input: input, state: &state, sampler: argMax)
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
        let root = directory.standardizedFileURL.path + "/"
        return ([declared].compactMap { $0 } + ["mtp.safetensors", "mtp/weights.safetensors"])
            .map { directory.appending(path: $0).standardizedFileURL }
            .first { $0.path.hasPrefix(root) && FileManager.default.fileExists(atPath: $0.path) }
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

    static func load(directory: URL, target: any LanguageModel, draftHead: Bool = false) throws -> (any MTPDrafterModel)? {
        guard target is Qwen35Model || target is Qwen35TextModel, let url = file(in: directory) else { return nil }
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let textConfig = root?["text_config"] as? [String: Any] ?? root
        guard let hiddenSize = textConfig?["hidden_size"] as? Int else { throw MTPSidecarError.malformed("config.json has no hidden_size") }
        var weights = try loadArrays(url: url)
        guard let norm = weights["mtp.norm.weight"] else { throw MTPSidecarError.malformed("\(url.lastPathComponent) has no mtp.norm.weight") }
        let preconvertedNorms = norm.asType(.float32).mean().item(Float.self) > 0.5
        let model = target is Qwen35Model
            ? Qwen35MTPDraftModel(try JSONDecoder.json5().decode(Qwen35Configuration.self, from: data), preconvertedNorms: preconvertedNorms)
            : Qwen35MTPDraftModel(try JSONDecoder.json5().decode(Qwen35TextConfiguration.self, from: data), preconvertedNorms: preconvertedNorms)
        weights = model.sanitize(weights: weights)
        if let quantization = try quantization(of: weights, hiddenSize: hiddenSize) {
            quantize(model: model) { path, _ in
                weights["\(path).scales"] == nil ? nil : (quantization.groupSize, quantization.bits, .affine)
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        if draftHead, !model.useDraftHead(of: target) {
            BasaltLogger(module: "MTPSidecar").warn("--experimental-mtp-draft-head: the target head is 4-bit or narrower, so the MTP head drafts through it unchanged")
        }
        return GreedyDraftingMTPDrafter(model)
    }
}
