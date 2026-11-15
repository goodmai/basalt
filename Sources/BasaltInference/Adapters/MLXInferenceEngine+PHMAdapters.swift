import Foundation
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

enum ConfiguredAdapter: Sendable {
    case lora(LoRAConfiguration)
    case phm(PHMAdapterConfiguration)
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
            let targetKeys = (keys ?? defaultKeys).sorted()
            let spec = AdapterSpec(
                method: .phm(hypercomplexDim: hypercomplexDim, factorRank: factorRank, sharedBasis: sharedBasis, scale: scale),
                layers: chosen,
                targetSuffixes: targetKeys
            )
            _ = try PHMAdapterApplicator.apply(to: model, spec: spec, targetSuffixes: targetKeys)
            let params = PHMAdapterConfiguration.Parameters(
                hypercomplexDim: hypercomplexDim,
                factorRank: factorRank,
                sharedBasis: sharedBasis,
                scale: scale,
                layers: chosen.sorted(),
                keys: targetKeys
            )
            return .phm(PHMAdapterConfiguration(numLayers: blocks.count - first, phmParameters: params))
        }
    }

    private static func attachScaledLoRA(
        to model: Module,
        layers: [Int]?,
        rank: Int,
        scale: Float,
        keys: [String]?
    ) throws(BasaltError) -> LoRAConfiguration {
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
        let parameters = LoRAConfiguration.LoRAParameters(rank: rank, scale: scale, keys: (keys ?? defaultKeys).sorted())
        model.freeze()
        replaceLoRA(in: blocks, indices: Set(chosen), parameters: parameters)
        model.invalidateCompiledTraces()
        guard !model.trainableParameters().flattened().isEmpty else {
            detachLoRA(from: model, blocks: blocks)
            throw .invalidRequestStructure(details: "train: the chosen layers have no plain or quantized linear projection to adapt")
        }
        return LoRAConfiguration(numLayers: blocks.count - first, loraParameters: parameters)
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
            }
        }
    }

    static func isPHMAdapterDirectory(_ directory: URL) -> Bool {
        let configURL = directory.appending(component: "adapter_config.json")
        guard let data = try? Data(contentsOf: configURL),
              let probe = try? JSONDecoder().decode(AdapterTypeProbe.self, from: data) else {
            return false
        }
        return probe.fineTuneType == "phm"
    }

    static func attachAnyAdapterDirectory(_ directory: URL, to model: any LanguageModel) throws(BasaltError) -> [String] {
        func refused(_ reason: String) -> BasaltError { .invalidRequestStructure(details: "adapter_apply \(directory.path): \(reason)") }
        guard isPHMAdapterDirectory(directory) else {
            return try attachContainer(try loraContainer(at: directory, refused), to: model, refused)
        }
        guard loraModules(of: model).isEmpty else {
            throw refused("another adapter is applied; undo it first")
        }
        let phm: PHMAdapterContainer
        do {
            phm = try PHMAdapterContainer.from(directory: directory)
        } catch {
            throw refused("invalid PHM adapter directory: \(briefly(error))")
        }
        do {
            try model.load(adapter: phm)
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
        guard isPHMAdapterDirectory(directory) else {
            let weights = try loraContainer(at: directory, refused)
            _ = try attachContainer(weights, to: model, refused)
            weights.unload(from: model)
            settle(model)
            return weights
        }
        let phm: PHMAdapterContainer
        do {
            phm = try PHMAdapterContainer.from(directory: directory)
        } catch {
            throw refused("invalid PHM adapter directory: \(briefly(error))")
        }
        do {
            try model.load(adapter: phm)
        } catch {
            phm.unload(from: model)
            settle(model)
            throw refused(briefly(error))
        }
        phm.unload(from: model)
        settle(model)
        return phm
    }
}
