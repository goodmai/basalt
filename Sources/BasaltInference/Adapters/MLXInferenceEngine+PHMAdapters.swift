import Foundation
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

enum ConfiguredAdapter: Sendable {
    case lora(LoRAConfiguration)
    case phm(PHMAdapterConfiguration)
    case oft(OFTAdapterConfiguration)
}

private struct AdapterTypeProbe: Decodable {
    let fineTuneType: String?

    enum CodingKeys: String, CodingKey {
        case fineTuneType = "fine_tune_type"
    }
}

extension MLXInferenceEngine {
    static func configuredMethod(explicit: AdapterMethod?, fallbackRank: Int) -> AdapterMethod {
        explicit ?? .lora(rank: fallbackRank, scale: 10.0)
    }

    static func attachConfiguredAdapter(
        to model: Module,
        layers: [Int]?,
        method: AdapterMethod,
        keys: [String]? = nil
    ) throws(BasaltError) -> ConfiguredAdapter {
        switch method {
        case .lora(let rank, let scale):
            if scale == 10.0 {
                return .lora(try attachLoRA(to: model, layers: layers, rank: rank, keys: keys))
            }
            return .lora(try attachScaledLoRA(to: model, layers: layers, rank: rank, scale: scale, keys: keys))
        case .phm(let hypercomplexDim, let factorRank, let sharedBasis, let scale):
            let sel = try resolveAdapterSelection(in: model, layers: layers, keys: keys)
            let spec = AdapterSpec(
                method: .phm(hypercomplexDim: hypercomplexDim, factorRank: factorRank, sharedBasis: sharedBasis, scale: scale),
                layers: sel.chosen,
                targetSuffixes: sel.targetKeys
            )
            _ = try PHMAdapterApplicator.apply(to: model, spec: spec, targetSuffixes: sel.targetKeys)
            let params = PHMAdapterConfiguration.Parameters(
                hypercomplexDim: hypercomplexDim,
                factorRank: factorRank,
                sharedBasis: sharedBasis,
                scale: scale,
                layers: sel.chosen.sorted(),
                keys: sel.targetKeys
            )
            return .phm(PHMAdapterConfiguration(numLayers: sel.blocks.count - sel.first, phmParameters: params))
        case .oft(let blockSize, let parameterization, let blockShared):
            let sel = try resolveAdapterSelection(in: model, layers: layers, keys: keys)
            let spec = AdapterSpec(
                method: .oft(blockSize: blockSize, parameterization: parameterization, blockShared: blockShared),
                layers: sel.chosen,
                targetSuffixes: sel.targetKeys
            )
            _ = try PHMAdapterApplicator.apply(to: model, spec: spec, targetSuffixes: sel.targetKeys)
            let params = OFTAdapterConfiguration.Parameters(
                blockSize: blockSize,
                parameterization: parameterization,
                blockShared: blockShared,
                layers: sel.chosen.sorted(),
                keys: sel.targetKeys
            )
            return .oft(OFTAdapterConfiguration(numLayers: sel.blocks.count - sel.first, oftParameters: params))
        }
    }

    private static func resolveAdapterSelection(
        in model: Module,
        layers: [Int]?,
        keys: [String]?
    ) throws(BasaltError) -> (blocks: [Module], chosen: [Int], first: Int, targetKeys: [String]) {
        let blocks = loraBlocks(of: model)
        guard !blocks.isEmpty else {
            throw .invalidRequestStructure(details: "train: this architecture has no LoRA layers in the fork")
        }
        guard !model.namedModules().contains(where: { $0.1 is LoRALayer }) else {
            throw .invalidRequestStructure(details: "train: an adapter is applied; undo it before training another")
        }
        let chosen = layers ?? Array(blocks.indices)
        guard let first = chosen.min(), chosen.allSatisfy(blocks.indices.contains) else {
            throw .invalidRequestStructure(details: "train: layers must be a non-empty subset of 0..<\(blocks.count)")
        }
        let defaultKeys = (model as? LoRAModel)?.loraDefaultKeys
            ?? Array(Set(blocks.flatMap { $0.namedModules() }.compactMap { $0.1 is Linear ? $0.0 : nil }))
        return (blocks, chosen, first, (keys ?? defaultKeys).sorted())
    }

    private static func attachScaledLoRA(
        to model: Module,
        layers: [Int]?,
        rank: Int,
        scale: Float,
        keys: [String]?
    ) throws(BasaltError) -> LoRAConfiguration {
        let sel = try resolveAdapterSelection(in: model, layers: layers, keys: keys)
        let parameters = LoRAConfiguration.LoRAParameters(rank: rank, scale: scale, keys: sel.targetKeys)
        model.freeze()
        replaceLoRA(in: sel.blocks, indices: Set(sel.chosen), parameters: parameters)
        model.invalidateCompiledTraces()
        guard !model.trainableParameters().flattened().isEmpty else {
            detachLoRA(from: model, blocks: sel.blocks)
            throw .invalidRequestStructure(details: "train: the chosen layers have no plain or quantized linear projection to adapt")
        }
        return LoRAConfiguration(numLayers: sel.blocks.count - sel.first, loraParameters: parameters)
    }
}

extension MLXInferenceEngine {
    static func saveConfiguredAdapter(_ configuration: ConfiguredAdapter, model: Module, to directory: URL) throws {
        try withTaskCancellationShield {
            switch configuration {
            case .lora(let loraConfig):
                try saveAdapter(loraConfig, model: model, to: directory)
            case .phm(let phmConfig):
                try PHMAdapterContainer.save(configuration: phmConfig, model: model, to: directory)
            case .oft(let oftConfig):
                try OFTAdapterContainer.save(configuration: oftConfig, model: model, to: directory)
            }
        }
    }

    static func probeFineTuneType(_ directory: URL) -> String? {
        let configURL = directory.appending(component: "adapter_config.json")
        guard let data = try? Data(contentsOf: configURL),
              let probe = try? JSONDecoder().decode(AdapterTypeProbe.self, from: data) else {
            return nil
        }
        return probe.fineTuneType
    }

    static func isPHMAdapterDirectory(_ directory: URL) -> Bool {
        probeFineTuneType(directory) == "phm"
    }

    static func isOFTAdapterDirectory(_ directory: URL) -> Bool {
        probeFineTuneType(directory) == "oft"
    }

    private static func loadCustomAdapter(
        at directory: URL,
        refused: (String) -> BasaltError
    ) throws(BasaltError) -> (any ModelAdapter)? {
        if EditAdapter.isEditAdapterDirectory(directory) {
            let adapter: EditAdapter
            do {
                adapter = try EditAdapter.from(directory: directory)
            } catch {
                throw refused("invalid edit adapter directory: \(briefly(error))")
            }
            guard adapter.program.approval == .approved else {
                throw refused("edit adapter '\(directory.lastPathComponent)' has approval status '\(adapter.program.approval.rawValue)'; approve explicitly via `basalt selfedit --accept-edit \(directory.path)` before serving")
            }
            return adapter
        }
        switch probeFineTuneType(directory) {
        case "phm":
            do {
                return try PHMAdapterContainer.from(directory: directory)
            } catch {
                throw refused("invalid PHM adapter directory: \(briefly(error))")
            }
        case "oft":
            do {
                return try OFTAdapterContainer.from(directory: directory)
            } catch {
                throw refused("invalid OFT adapter directory: \(briefly(error))")
            }
        default:
            return nil
        }
    }

    static func attachAnyAdapterDirectory(_ directory: URL, to model: any LanguageModel) throws(BasaltError) -> [String] {
        func refused(_ reason: String) -> BasaltError { .invalidRequestStructure(details: "adapter_apply \(directory.path): \(reason)") }
        if EditAdapter.isEditAdapterDirectory(directory) {
            throw refused("edit adapters are served through --adapter / loadAdapters or applied to a WeightCopy, not live adapter_apply")
        }
        guard let custom = try loadCustomAdapter(at: directory, refused: refused) else {
            return try attachContainer(try loraContainer(at: directory, for: model, refused), to: model, refused)
        }
        guard loraModules(of: model).isEmpty else {
            throw refused("another adapter is applied; undo it first")
        }
        do {
            try model.load(adapter: custom)
        } catch {
            _ = PHMAdapterApplicator.revert(in: model)
            throw refused(briefly(error))
        }
        let modules = loraModules(of: model)
        guard !modules.isEmpty else { throw refused("it matches no linear layer of this model") }
        return modules
    }

    static func loadServedAdapter(
        at directory: URL,
        into model: any LanguageModel,
        refused: (String) -> BasaltError
    ) throws(BasaltError) -> any ModelAdapter {
        guard let custom = try loadCustomAdapter(at: directory, refused: refused) else {
            let weights = try loraContainer(at: directory, for: model, refused)
            _ = try attachContainer(weights, to: model, refused)
            weights.unload(from: model)
            settle(model)
            return weights
        }
        do {
            try model.load(adapter: custom)
        } catch {
            custom.unload(from: model)
            settle(model)
            throw refused(briefly(error))
        }
        custom.unload(from: model)
        settle(model)
        return custom
    }
}
