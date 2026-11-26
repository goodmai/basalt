import Foundation
import MLX
import MLXLMCommon
import MLXNN
import BasaltCore

public struct SuperAdapterSpec: Sendable, Equatable {
    public struct Source: Sendable, Equatable {
        public let name: String
        public let directory: URL
        public let weight: Float

        public init(name: String, directory: URL, weight: Float = 1.0) {
            self.name = name
            self.directory = directory
            self.weight = weight
        }
    }

    public let sources: [Source]

    public init(sources: [Source]) {
        self.sources = sources
    }
}

public enum SuperAdapterComposer: Sendable {
    fileprivate struct LoadedSource {
        let rank: Int
        let effectiveScale: Float
        let pairs: [String: (a: MLXArray, b: MLXArray)]
    }

    public static func compose(_ spec: SuperAdapterSpec, for model: Module) throws(BasaltError) -> LoRAContainer {
        try compose(sources: spec.sources, for: model)
    }

    public static func compose(sources: [SuperAdapterSpec.Source], for model: Module) throws(BasaltError) -> LoRAContainer {
        try validateSources(sources)
        guard let loraModel = model as? LoRAModel else {
            throw .invalidRequestStructure(details: "super-adapter: the \(type(of: model)) architecture takes no LoRA adapter")
        }
        let linears = Dictionary(
            model.namedModules().compactMap { path, module in (module as? Linear).map { (path, $0) } },
            uniquingKeysWith: { first, _ in first }
        )
        var loaded: [LoadedSource] = []
        loaded.reserveCapacity(sources.count)
        for source in sources {
            loaded.append(try loadSource(source, linears: linears, for: model))
        }
        let totalRank = try combinedRank(of: loaded)
        var moduleSet = Set<String>()
        for item in loaded {
            moduleSet.formUnion(item.pairs.keys)
        }
        let allModules = moduleSet.sorted()
        guard !allModules.isEmpty else {
            throw .invalidRequestStructure(details: "super-adapter: sources contain no LoRA weights")
        }
        let (parameters, relativeKeys) = try composeParameters(loaded: loaded, modules: allModules, linears: linears)
        let configuration = LoRAConfiguration(
            numLayers: loraModel.loraLayers.count,
            fineTuneType: .lora,
            loraParameters: .init(rank: totalRank, scale: 1.0, keys: relativeKeys)
        )
        return LoRAContainer(
            configuration: configuration,
            parameters: ModuleParameters.unflattened(parameters)
        )
    }

    public static func saveSuperAdapter(_ container: LoRAContainer, to directory: URL) throws(BasaltError) {
        let weights = Dictionary(uniqueKeysWithValues: container.parameters.flattened())
        guard !weights.isEmpty else {
            throw .invalidRequestStructure(details: "super-adapter: cannot save an adapter with no LoRA weights")
        }
        guard container.configuration.loraParameters.rank > 0, container.configuration.loraParameters.scale.isFinite else {
            throw .invalidRequestStructure(details: "super-adapter: rank must be positive and scale must be finite")
        }
        do {
            try withTaskCancellationShield {
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                let configData = try encoder.encode(container.configuration)
                try configData.write(to: directory.appending(component: "adapter_config.json"), options: .atomic)
                try MLX.save(arrays: weights, url: directory.appending(component: "adapters.safetensors"))
            }
        } catch {
            throw .inferenceError("super-adapter: could not save to \(directory.path): \(error.localizedDescription)")
        }
    }
}

extension SuperAdapterComposer {
    private static func validateSources(_ sources: [SuperAdapterSpec.Source]) throws(BasaltError) {
        guard !sources.isEmpty else {
            throw .invalidRequestStructure(details: "super-adapter: at least one source adapter is required")
        }
        var seenNames = Set<String>()
        for source in sources {
            let name = source.name.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else {
                throw .invalidRequestStructure(details: "super-adapter: source adapter name must not be empty")
            }
            guard seenNames.insert(name).inserted else {
                throw .invalidRequestStructure(details: "super-adapter: duplicate source adapter name '\(name)'")
            }
            guard source.weight.isFinite else {
                throw .invalidRequestStructure(details: "super-adapter: weight for '\(name)' must be finite, got \(source.weight)")
            }
        }
    }

    private static func combinedRank(of loaded: [LoadedSource]) throws(BasaltError) -> Int {
        var total = 0
        for item in loaded {
            let (next, overflow) = total.addingReportingOverflow(item.rank)
            guard !overflow, next > 0 else {
                throw .invalidRequestStructure(details: "super-adapter: combined rank overflows Int")
            }
            total = next
        }
        return total
    }

    private static func loadSource(
        _ source: SuperAdapterSpec.Source,
        linears: [String: Linear],
        for model: Module
    ) throws(BasaltError) -> LoadedSource {
        func refused(_ reason: String) -> BasaltError {
            .invalidRequestStructure(details: "super-adapter \(source.name): \(reason)")
        }
        let container = try MLXInferenceEngine.loraContainer(at: source.directory, for: model, refused)
        guard container.configuration.fineTuneType == .lora else {
            throw refused("fine_tune_type \(container.configuration.fineTuneType.rawValue) is not lora; super-adapter only composes LoRA adapters")
        }
        let rank = container.configuration.loraParameters.rank
        let scale = container.configuration.loraParameters.scale
        guard rank > 0, scale.isFinite else {
            throw refused("rank \(rank) must be positive and scale \(scale) must be finite")
        }
        let effectiveScale = source.weight * scale
        guard effectiveScale.isFinite else {
            throw refused("effective scale \(source.weight) * \(scale) is not finite")
        }
        let pairs = try extractPairs(from: container, rank: rank, linears: linears, refused: refused)
        return LoadedSource(rank: rank, effectiveScale: effectiveScale, pairs: pairs)
    }

    private static func extractPairs(
        from container: LoRAContainer,
        rank: Int,
        linears: [String: Linear],
        refused: (String) -> BasaltError
    ) throws(BasaltError) -> [String: (a: MLXArray, b: MLXArray)] {
        let suffixCount = ".lora_a".count
        var aByPath: [String: MLXArray] = [:]
        var bByPath: [String: MLXArray] = [:]
        for (key, tensor) in container.parameters.flattened() {
            let isA = key.hasSuffix(".lora_a")
            let isB = key.hasSuffix(".lora_b")
            guard isA || isB else { throw refused("\(key) is neither a lora_a nor a lora_b weight") }
            let modulePath = String(key.dropLast(suffixCount))
            guard let linear = linears[modulePath], PEFTLoRAReader.layerRelative(modulePath) != nil else {
                throw refused("\(key) belongs to no linear layer of this model")
            }
            let (outputs, inputs) = linear.shape
            let expected = isA ? [inputs, rank] : [rank, outputs]
            guard tensor.shape == expected else {
                throw refused("\(key) has shape \(tensor.shape), the layer needs \(expected)")
            }
            if isA { aByPath[modulePath] = tensor } else { bByPath[modulePath] = tensor }
        }
        let paths = Set(aByPath.keys).union(bByPath.keys)
        guard !paths.isEmpty else { throw refused("adapter holds no LoRA weights") }
        var pairs: [String: (a: MLXArray, b: MLXArray)] = [:]
        for path in paths {
            guard let a = aByPath[path], let b = bByPath[path] else {
                throw refused("\(path) has only one of lora_a and lora_b")
            }
            pairs[path] = (a, b)
        }
        return pairs
    }

    private static func composeParameters(
        loaded: [LoadedSource],
        modules: [String],
        linears: [String: Linear]
    ) throws(BasaltError) -> ([String: MLXArray], [String]) {
        var parameters: [String: MLXArray] = [:]
        var relativeKeys = Set<String>()
        for modulePath in modules {
            guard let linear = linears[modulePath], let relative = PEFTLoRAReader.layerRelative(modulePath) else {
                throw .invalidRequestStructure(details: "super-adapter: \(modulePath) belongs to no linear layer of this model")
            }
            let (outputs, inputs) = linear.shape
            let dtype = PEFTLoRAReader.computeType(of: linear)
            var aBlocks: [MLXArray] = []
            var bBlocks: [MLXArray] = []
            aBlocks.reserveCapacity(loaded.count)
            bBlocks.reserveCapacity(loaded.count)
            for source in loaded {
                if let pair = source.pairs[modulePath] {
                    aBlocks.append(pair.a.asType(dtype))
                    bBlocks.append((source.effectiveScale * pair.b.asType(.float32)).asType(dtype))
                } else {
                    aBlocks.append(MLXArray.zeros([inputs, source.rank], dtype: dtype))
                    bBlocks.append(MLXArray.zeros([source.rank, outputs], dtype: dtype))
                }
            }
            parameters["\(modulePath).lora_a"] = concatenated(aBlocks, axis: 1)
            parameters["\(modulePath).lora_b"] = concatenated(bBlocks, axis: 0)
            relativeKeys.insert(relative)
        }
        return (parameters, relativeKeys.sorted())
    }
}
