import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import BasaltCore

enum ModelSkeletonBuilder {
    static let gdnFusedLeafNames = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]

    static func buildCopy(
        from original: any LanguageModel,
        configuration: ModelConfiguration? = nil,
        makeSkeleton: (() throws -> any LanguageModel)? = nil
    ) async throws -> any LanguageModel {
        let skeleton: any LanguageModel
        if let makeSkeleton {
            skeleton = try makeSkeleton()
        } else if let directory = modelDirectory(from: configuration),
                  let fromDisk = try await buildFromDiskConfig(directory: directory) {
            skeleton = fromDisk
        } else {
            skeleton = try buildFromMirror(original)
        }
        return try shareWeights(from: original, into: skeleton)
    }

    static func buildCopySync(
        from original: any LanguageModel,
        makeSkeleton: (() throws -> any LanguageModel)? = nil
    ) throws -> any LanguageModel {
        let skeleton = try makeSkeleton?() ?? buildFromMirror(original)
        return try shareWeights(from: original, into: skeleton)
    }

    static func modelDirectory(from configuration: ModelConfiguration?) -> URL? {
        guard let configuration else { return nil }
        if case .directory(let url) = configuration.id {
            return url
        }
        return nil
    }

    static func buildFromDiskConfig(directory: URL) async throws -> (any LanguageModel)? {
        let configURL = directory.appending(component: "config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else { return nil }
        let rawConfig = try Data(contentsOf: configURL)
        await MLXInferenceEngine.registerAliases()
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: rawConfig)
        for registry in [VLMTypeRegistry.shared, LLMTypeRegistry.shared] {
            guard await registry.contains(base.modelType) else { continue }
            if let model = try? await registry.createModel(configuration: rawConfig, modelType: base.modelType) {
                return model
            }
        }
        return nil
    }

    static func buildFromMirror(_ original: any LanguageModel) throws -> any LanguageModel {
        if let qwen35 = original as? Qwen35TextModel,
           let config: Qwen35TextConfiguration = reflectedChild(of: qwen35, matching: Qwen35TextConfiguration.self) {
            return Qwen35TextModel(config)
        }
        if let qwen35Wrapper = original as? MLXLLM.Qwen35Model,
           let textModel = qwen35Wrapper.namedModules().first(where: { $0.0 == "language_model" })?.1 as? Qwen35TextModel,
           let textConfig: Qwen35TextConfiguration = reflectedChild(of: textModel, matching: Qwen35TextConfiguration.self) {
            let encoded = try JSONEncoder().encode(["text_config": textConfig])
            let wrapperConfig = try JSONDecoder().decode(MLXLLM.Qwen35Configuration.self, from: encoded)
            return original is MLXLLM.Qwen35MoEModel ? MLXLLM.Qwen35MoEModel(wrapperConfig) : MLXLLM.Qwen35Model(wrapperConfig)
        }
        if let qwen3 = original as? Qwen3Model,
           let config: Qwen3Configuration = reflectedChild(of: qwen3, matching: Qwen3Configuration.self) {
            return Qwen3Model(config)
        }
        if let gemma4 = try buildGemma4FromMirror(original) {
            return gemma4
        }
        if let lfm2 = original as? LFM2Model,
           let config: LFM2Configuration = reflectedChild(of: lfm2, matching: LFM2Configuration.self) {
            return LFM2Model(config)
        }
        throw WeightCopyError.unsupportedModel(String(describing: type(of: original)))
    }

    static func reflectedChild<T>(of object: Any, matching _: T.Type) -> T? {
        var mirror: Mirror? = Mirror(reflecting: object)
        while let current = mirror {
            for child in current.children {
                if let match = child.value as? T {
                    return match
                }
            }
            mirror = current.superclassMirror
        }
        return nil
    }
}

extension ModelSkeletonBuilder {
    static func shareWeights(
        from original: any LanguageModel,
        into copy: any LanguageModel
    ) throws -> any LanguageModel {
        let origModules = original.namedModules()
        var moduleUpdates: [(String, Module)] = []
        for (path, module) in origModules where !path.isEmpty {
            if let cloned = cloneLeafSharingArrays(module) {
                moduleUpdates.append((path, cloned))
            }
        }
        PHMAdapterApplicator.updateLeafModules(in: copy, updates: moduleUpdates)

        let origParams = Dictionary(original.parameters().flattened(), uniquingKeysWith: { first, _ in first })
        let copyParams = Dictionary(copy.parameters().flattened(), uniquingKeysWith: { first, _ in first })
        var remaining: [(String, MLXArray)] = []
        for (name, origArray) in origParams {
            if let existing = copyParams[name], existing === origArray {
                continue
            }
            remaining.append((name, origArray))
        }
        if !remaining.isEmpty {
            try copy.update(parameters: ModuleParameters.unflattened(remaining), verify: [.noUnusedKeys])
        }
        copy.train(original.training)
        MLXInferenceEngine.settle(copy)
        eval(copy)
        return copy
    }

    static func extractLoRAScales(from model: Module) -> [String: Float] {
        var scales: [String: Float] = [:]
        for (path, module) in model.namedModules() where module is LoRALayer {
            if let edit = module as? EditRankOneLinear {
                scales[path] = edit.scale
            } else if let scale: Float = reflectedChild(of: module, matching: Float.self) {
                scales[path] = scale
            }
        }
        return scales
    }

    static func fusableGroupInfo(in model: Module) -> (groups: [String: Int64], memberToGroup: [String: String]) {
        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        var groups: [String: Int64] = [:]
        var memberToGroup: [String: String] = [:]
        for (path, _) in modules where path.hasSuffix(".linear_attn") || path == "linear_attn" {
            var projections: [QuantizedLinear] = []
            var paths: [String] = []
            for leaf in gdnFusedLeafNames {
                let full = path.isEmpty ? leaf : "\(path).\(leaf)"
                guard let q = modules[full] as? QuantizedLinear,
                      ObjectIdentifier(type(of: q)) == ObjectIdentifier(QuantizedLinear.self),
                      q.bias == nil else { break }
                projections.append(q)
                paths.append(full)
            }
            guard projections.count == gdnFusedLeafNames.count,
                  let first = projections.first,
                  projections.allSatisfy({
                      $0.bits == first.bits && $0.groupSize == first.groupSize && $0.mode == first.mode
                          && $0.shape.1 == first.shape.1 && ($0.biases != nil) == (first.biases != nil)
                  }) else { continue }
            let groupBytes = projections.reduce(Int64(0)) { $0 + Int64(TensorPacker.bytes($1)) }
            groups[path] = groupBytes
            for full in paths {
                memberToGroup[full] = path
            }
        }
        return (groups, memberToGroup)
    }
}

extension ModelSkeletonBuilder {
    static func cloneLeafSharingArrays(_ module: Module) -> Module? {
        if let cloned = cloneLoRASharingBase(module) { return cloned }
        if let cloned = cloneLinearSharingArrays(module) { return cloned }
        if let cloned = cloneSwitchSharingArrays(module) { return cloned }
        return cloneEmbeddingSharingArrays(module)
    }

    private static func cloneLoRASharingBase(_ module: Module) -> Module? {
        if let edit = module as? EditRankOneLinear {
            let base = (cloneLinearSharingArrays(edit.baseLinear) as? Linear) ?? edit.baseLinear
            let cloned = EditRankOneLinear(base: base, loraA: edit.loraA, loraB: edit.loraB, scale: edit.scale)
            cloned.loraEnabled = edit.loraEnabled
            cloned.train(edit.training)
            return cloned
        }
        guard module is LoRALinear || module is QLoRALinear,
              let lora = module as? LoRALayer,
              let base = lora.reverted() as? Linear else { return nil }
        let params = Dictionary(lora.parameters().flattened(), uniquingKeysWith: { first, _ in first })
        guard let loraA = params["lora_a"], let loraB = params["lora_b"] else { return nil }
        let scale = reflectedChild(of: lora, matching: Float.self) ?? 1.0
        let cloned = LoRALinear.from(linear: base, rank: loraA.dim(1), scale: scale)
        cloned.update(parameters: ModuleParameters.unflattened(["lora_a": loraA, "lora_b": loraB]))
        cloned.loraEnabled = lora.loraEnabled
        return cloned
    }

    private static func cloneLinearSharingArrays(_ module: Module) -> Module? {
        if let nf4 = module as? NF4Linear {
            let cloned = try? NF4Linear(weight: nf4.weight, bias: nf4.bias, scales: nf4.scales, groupSize: nf4.groupSize)
            cloned?.train(nf4.training)
            return cloned
        }
        if let mac = module as? MACaQQuantizedLinear {
            let packed = MACaQPackedWeights(weight: mac.weight, scales: mac.scales, biases: mac.biases)
            return EditOpExecutor.makeMACaQ(from: mac, packed: packed, bias: mac.bias)
        }
        if let smallM = module as? SmallMQuantizedLinear {
            let cloned = SmallMQuantizedLinear(replacing: smallM)
            cloned.train(smallM.training)
            return cloned
        }
        if let quantized = module as? QuantizedLinear,
           ObjectIdentifier(type(of: quantized)) == ObjectIdentifier(QuantizedLinear.self) {
            let cloned = QuantizedLinear(
                weight: quantized.weight, bias: quantized.bias, scales: quantized.scales, biases: quantized.biases,
                groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode, globalScale: quantized.globalScale
            )
            cloned.train(quantized.training)
            cloned.freeze()
            return cloned
        }
        if let linear = module as? Linear, ObjectIdentifier(type(of: linear)) == ObjectIdentifier(Linear.self) {
            let cloned = Linear(weight: linear.weight, bias: linear.bias)
            cloned.train(linear.training)
            return cloned
        }
        return nil
    }

    private static func cloneEmbeddingSharingArrays(_ module: Module) -> Module? {
        if let quantizedEmbed = module as? QuantizedEmbedding,
           ObjectIdentifier(type(of: quantizedEmbed)) == ObjectIdentifier(QuantizedEmbedding.self) {
            let cloned = QuantizedEmbedding(
                weight: quantizedEmbed.weight, scales: quantizedEmbed.scales, biases: quantizedEmbed.biases,
                groupSize: quantizedEmbed.groupSize, bits: quantizedEmbed.bits, mode: quantizedEmbed.mode, globalScale: quantizedEmbed.globalScale
            )
            cloned.train(quantizedEmbed.training)
            cloned.freeze()
            return cloned
        }
        if let embed = module as? Embedding, ObjectIdentifier(type(of: embed)) == ObjectIdentifier(Embedding.self) {
            let cloned = Embedding(weight: embed.weight)
            cloned.train(embed.training)
            return cloned
        }
        return nil
    }
}

extension ModelSkeletonBuilder {
    static func buildGemma4FromMirror(_ original: any LanguageModel) throws -> (any LanguageModel)? {
        if let gemma4 = original as? MLXLLM::Gemma4TextModel,
           let config: MLXLLM::Gemma4TextConfiguration = reflectedChild(of: gemma4, matching: MLXLLM::Gemma4TextConfiguration.self) {
            return MLXLLM::Gemma4TextModel(config)
        }
        if let gemma4Wrapper = original as? MLXLLM::Gemma4Model,
           let textModel = gemma4Wrapper.namedModules().first(where: { $0.0 == "language_model" })?.1 as? MLXLLM::Gemma4TextModel,
           let textConfig: MLXLLM::Gemma4TextConfiguration = reflectedChild(of: textModel, matching: MLXLLM::Gemma4TextConfiguration.self) {
            let encoded = try JSONEncoder().encode(textConfig)
            let wrapperConfig = try JSONDecoder().decode(MLXLLM::Gemma4Configuration.self, from: encoded)
            return MLXLLM::Gemma4Model(wrapperConfig)
        }
        if let gemma4VLM = original as? MLXVLM::Gemma4 {
            return MLXVLM::Gemma4(gemma4VLM.config)
        }
        if let gemma4Unified = original as? MLXVLM::Gemma4Unified {
            return MLXVLM::Gemma4Unified(gemma4Unified.config)
        }
        return nil
    }

    static func cloneSwitchSharingArrays(_ module: Module) -> Module? {
        if let mac = module as? MACaQSwitchLinear {
            let params = Dictionary(mac.parameters().flattened(), uniquingKeysWith: { first, _ in first })
            guard let weight = params["weight"] else { return nil }
            let geometry = MACaQSwitchGeometry(experts: weight.dim(0), outputs: weight.dim(1), inputs: mac.scales.dim(2) * mac.groupSize)
            var spec = MACaQLayerSpec(bits: mac.bits, groupSize: mac.groupSize, mode: mac.mode, hadamardBlock: mac.hadamardBlock)
            spec.scaleDType = "\(mac.scales.dtype)"
            spec.hasBias = params["bias"] != nil
            spec.perExpertPreScale = mac.preScale.ndim == 2
            let packed = MACaQPackedWeights(weight: weight, scales: mac.scales, biases: mac.biases)
            let cloned = MACaQSwitchLinear(geometry: geometry, packed: packed, bias: params["bias"], preScale: mac.preScale, spec: spec)
            cloned.activationDType = mac.activationDType
            cloned.useQ3Kernel = mac.useQ3Kernel
            cloned.train(mac.training)
            return cloned
        }
        if let nf4 = module as? NF4SwitchLinear {
            let params = Dictionary(nf4.parameters().flattened(), uniquingKeysWith: { first, _ in first })
            guard let weight = params["weight"] else { return nil }
            let cloned = try? NF4SwitchLinear(
                inputDims: weight.dim(2) * 8, outputDims: weight.dim(1), numExperts: weight.dim(0),
                weight: weight, bias: params["bias"], scales: nf4.scales, groupSize: nf4.groupSize
            )
            cloned?.train(nf4.training)
            return cloned
        }
        if let quantized = module as? QuantizedSwitchLinear,
           ObjectIdentifier(type(of: quantized)) == ObjectIdentifier(QuantizedSwitchLinear.self) {
            let params = Dictionary(quantized.parameters().flattened(), uniquingKeysWith: { first, _ in first })
            guard let weight = params["weight"], let scales = params["scales"] else { return nil }
            let cloned = QuantizedSwitchLinear(
                weight: weight, scales: scales, biases: params["biases"], bias: params["bias"],
                inputDims: scales.dim(2) * quantized.groupSize, outputDims: weight.dim(1), numExperts: weight.dim(0),
                groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode, globalScale: quantized.globalScale
            )
            cloned.train(quantized.training)
            return cloned
        }
        if let sw = module as? SwitchLinear,
           ObjectIdentifier(type(of: sw)) == ObjectIdentifier(SwitchLinear.self) {
            let params = Dictionary(sw.parameters().flattened(), uniquingKeysWith: { first, _ in first })
            guard let weight = params["weight"] else { return nil }
            let cloned = SwitchLinear(
                inputDims: weight.dim(2), outputDims: weight.dim(1), numExperts: weight.dim(0),
                weight: weight, bias: params["bias"]
            )
            cloned.train(sw.training)
            return cloned
        }
        return nil
    }
}
