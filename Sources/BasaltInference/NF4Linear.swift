import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

open class NF4Linear: Linear {
    public let groupSize: Int
    @ParameterInfo(key: "scales") public var scales: MLXArray

    open override var shape: (Int, Int) { (weight.dim(0), weight.dim(1) * 8) }

    public init(weight: MLXArray, bias: MLXArray? = nil, scales: MLXArray, groupSize: Int? = nil) throws(NF4Error) {
        let columns = weight.dim(-1) * 8
        guard weight.ndim == 2, scales.ndim == 2, scales.dim(0) == weight.dim(0), scales.dim(1) > 0,
            columns % scales.dim(1) == 0
        else {
            throw .shape("NF4 codes \(weight.shape) do not fit scales \(scales.shape)")
        }
        let resolvedGroup = groupSize ?? (columns / scales.dim(1))
        guard NF4Metal.groupSizes.contains(resolvedGroup), columns / scales.dim(1) == resolvedGroup else {
            throw .groupSize(resolvedGroup)
        }
        self.groupSize = resolvedGroup
        self._scales.wrappedValue = scales
        super.init(weight: weight, bias: bias)
        self.freeze()
    }

    public init(dense: MLXArray, bias: MLXArray? = nil, groupSize: Int = 32) {
        let (codes, scales) = NF4Metal.quantized(dense, groupSize: groupSize)
        self.groupSize = groupSize
        self._scales.wrappedValue = scales
        super.init(weight: codes, bias: bias)
        self.freeze()
    }

    open override func unfreeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false) throws {
        try super.unfreeze(recursive: recursive, keys: keys, strict: strict)
        try self.freeze(recursive: false, keys: ["weight"], strict: false)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out = NF4Metal.matmul(x, codes: weight, scales: scales, groupSize: groupSize)
        guard let bias else { return out }
        return out + bias
    }
}

open class NF4SwitchLinear: SwitchLinear {
    public let groupSize: Int
    @ParameterInfo(key: "scales") public var scales: MLXArray

    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil, scales: MLXArray, groupSize: Int? = nil
    ) throws(NF4Error) {
        let columns = weight.dim(-1) * 8
        guard weight.ndim == 3, scales.ndim == 3,
            weight.dim(0) == numExperts, weight.dim(1) == outputDims, columns == inputDims,
            scales.dim(0) == numExperts, scales.dim(1) == outputDims, scales.dim(2) > 0,
            columns % scales.dim(2) == 0
        else {
            throw .shape("NF4 expert codes \(weight.shape) do not fit scales \(scales.shape)")
        }
        let resolvedGroup = groupSize ?? (columns / scales.dim(2))
        guard NF4Metal.groupSizes.contains(resolvedGroup), columns / scales.dim(2) == resolvedGroup else {
            throw .groupSize(resolvedGroup)
        }
        self.groupSize = resolvedGroup
        self._scales.wrappedValue = scales
        super.init(inputDims: inputDims, outputDims: outputDims, numExperts: numExperts, weight: weight, bias: bias)
        self.freeze()
    }

    public init(_ other: SwitchLinear, groupSize: Int = 32) {
        let params = Dictionary(uniqueKeysWithValues: other.parameters().flattened())
        let dense = params["weight"] ?? MLXArray.zeros([1, 1, groupSize])
        let (codes, scales) = NF4Metal.quantized(dense, groupSize: groupSize)
        self.groupSize = groupSize
        self._scales.wrappedValue = scales
        super.init(
            inputDims: dense.dim(2), outputDims: dense.dim(1), numExperts: dense.dim(0),
            weight: codes, bias: params["bias"])
        self.freeze()
    }

    open override func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false) -> MLXArray {
        let params = parameters()
        guard case .value(let codes)? = params["weight"] else { return x }
        var out = NF4Metal.gatherMatmul(x, codes: codes, scales: scales, rhsIndices: indices, groupSize: groupSize)
        if case .value(let bias)? = params["bias"] {
            out = out + expandedDimensions(bias[indices], axis: -2)
        }
        return out
    }
}

package enum NF4NativeLoader {
    package static func isNF4Config(_ json: [String: Any]) -> Bool {
        for key in ["quantization", "quantization_config"] {
            guard let block = json[key] as? [String: Any] else { continue }
            if (block["mode"] as? String) == NF4Metal.method || (block["quant_method"] as? String) == NF4Metal.method {
                return true
            }
        }
        return false
    }

    package static func strippedConfigData(_ raw: Data) throws -> (modelType: String, stripped: Data)? {
        guard var json = (try? JSONSerialization.jsonObject(with: raw)) as? [String: Any],
            isNF4Config(json),
            let modelType = json["model_type"] as? String
        else { return nil }
        json["quantization"] = nil
        json["quantization_config"] = nil
        let stripped = try JSONSerialization.data(withJSONObject: json, options: [.sortedKeys])
        return (modelType, stripped)
    }

    package static func loadContainer(
        from directory: URL, tokenizerLoader: any TokenizerLoader
    ) async throws -> ModelContainer? {
        let configURL = directory.appending(component: "config.json")
        guard let raw = try? Data(contentsOf: configURL),
            let info = try strippedConfigData(raw)
        else { return nil }
        let staging = FileManager.default.temporaryDirectory.appending(component: "basalt-nf4-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: staging) }
        try stageDirectory(source: directory, staging: staging, strippedConfig: info.stripped)
        var failure: (any Error)?
        for family in ExpertStreamingLoader.families where await family.types.contains(info.modelType) {
            do {
                let model = try await family.types.createModel(configuration: info.stripped, modelType: info.modelType)
                let loaded = try readSanitizedWeights(in: directory, model: model)
                try installModules(in: model, weights: loaded)
                let registry = ModelTypeRegistry<any LanguageModel>(creators: [info.modelType: { _ in model }])
                return try await family.factory(registry).loadContainer(from: staging, using: tokenizerLoader)
            } catch {
                failure = error
            }
        }
        throw failure ?? BasaltError.inferenceError("NF4 native: no model factory knows model_type \(info.modelType)")
    }

    package static func loadWeights(modelDirectory: URL, model: any LanguageModel) throws {
        let weights = try readSanitizedWeights(in: modelDirectory, model: model)
        try installModules(in: model, weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        try model.prepare()
        eval(model)
    }

    package static func quantize(model: Module, groupSize: Int = 32, filter: (String, Module) -> Bool) {
        var updates: [(String, Module)] = []
        for (path, mod) in model.namedModules() where filter(path, mod) {
            if let linear = mod as? Linear, !(mod is QuantizedLinear), !(mod is NF4Linear), linear.shape.1 % groupSize == 0 {
                updates.append((path, NF4Linear(dense: linear.weight, bias: linear.bias, groupSize: groupSize)))
            } else if let sw = mod as? SwitchLinear, !(mod is QuantizedSwitchLinear), !(mod is NF4SwitchLinear) {
                let inDims = sw.parameters().flattened().first(where: { $0.0 == "weight" })?.1.dim(2) ?? 0
                if inDims > 0, inDims % groupSize == 0 {
                    updates.append((path, NF4SwitchLinear(sw, groupSize: groupSize)))
                }
            }
        }
        if !updates.isEmpty {
            model.update(modules: ModuleChildren.unflattened(updates))
        }
    }
}

extension NF4NativeLoader {
    private static func stageDirectory(source: URL, staging: URL, strippedConfig: Data) throws {
        let fm = FileManager.default
        try strippedConfig.write(to: staging.appending(component: "config.json"))
        for name in try fm.contentsOfDirectory(atPath: source.path) where name != "config.json" && !name.hasPrefix(".") {
            try fm.createSymbolicLink(
                at: staging.appending(component: name),
                withDestinationURL: source.appending(component: name).resolvingSymlinksInPath())
        }
    }

    private static func readSanitizedWeights(in directory: URL, model: any LanguageModel) throws -> [String: MLXArray] {
        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
            .filter { $0.hasSuffix(".safetensors") }
            .sorted()
        var raw: [String: MLXArray] = [:]
        var metadata: [String: String] = [:]
        for name in names {
            let (arrays, meta) = try loadArraysAndMetadata(url: directory.appending(component: name))
            raw.merge(arrays) { _, new in new }
            if metadata.isEmpty { metadata = meta }
        }
        return model.sanitize(weights: raw, metadata: metadata)
    }

    package static func installModules(in model: Module, weights: [String: MLXArray]) throws {
        var updates: [(String, Module)] = []
        for (path, module) in model.namedModules() {
            guard let w = weights["\(path).weight"], let s = weights["\(path).scales"] else { continue }
            let b = weights["\(path).biases"]
            let bias = weights["\(path).bias"]
            if let linear = module as? Linear, !(module is QuantizedLinear), !(module is NF4Linear) {
                updates.append((path, try replacementLinear(linear, weight: w, scales: s, biases: b, bias: bias)))
            } else if let sw = module as? SwitchLinear, !(module is QuantizedSwitchLinear), !(module is NF4SwitchLinear) {
                updates.append((path, try replacementSwitch(sw, weight: w, scales: s, biases: b, bias: bias)))
            } else if let embedding = module as? Embedding, !(module is QuantizedEmbedding), let b {
                updates.append((path, affineEmbedding(embedding, weight: w, scales: s, biases: b)))
            }
        }
        if !updates.isEmpty {
            model.update(modules: ModuleChildren.unflattened(updates))
        }
    }

    private static func replacementLinear(
        _ linear: Linear, weight: MLXArray, scales: MLXArray, biases: MLXArray?, bias: MLXArray?
    ) throws -> Module {
        if let biases {
            let inDims = linear.shape.1
            let bits = (weight.dim(-1) * 32) / max(1, inDims)
            let groupSize = inDims / max(1, scales.dim(-1))
            return QuantizedLinear(
                weight: weight, bias: bias, scales: scales, biases: biases,
                groupSize: groupSize, bits: bits, mode: .affine)
        }
        return try NF4Linear(weight: weight, bias: bias, scales: scales)
    }

    private static func affineEmbedding(_ embedding: Embedding, weight: MLXArray, scales: MLXArray, biases: MLXArray) -> Module {
        let dims = embedding.weight.dim(-1)
        return QuantizedEmbedding(
            weight: weight, scales: scales, biases: biases,
            groupSize: dims / max(1, scales.dim(-1)), bits: (weight.dim(-1) * 32) / max(1, dims), mode: .affine)
    }

    private static func replacementSwitch(
        _ sw: SwitchLinear, weight: MLXArray, scales: MLXArray, biases: MLXArray?, bias: MLXArray?
    ) throws -> Module {
        let oldShape = sw.parameters().flattened().first(where: { $0.0 == "weight" })?.1.shape ?? [weight.dim(0), weight.dim(1), weight.dim(2) * 8]
        let (experts, outDims, inDims) = (oldShape[0], oldShape[1], oldShape[2])
        if let biases {
            let bits = (weight.dim(-1) * 32) / max(1, inDims)
            let groupSize = inDims / max(1, scales.dim(-1))
            let dummy = SwitchLinear(inputDims: inDims, outputDims: outDims, numExperts: experts, bias: bias != nil)
            let q = QuantizedSwitchLinear(dummy, groupSize: groupSize, bits: bits, mode: .affine)
            var params: [(String, MLXArray)] = [("weight", weight), ("scales", scales), ("biases", biases)]
            if let bias { params.append(("bias", bias)) }
            q.update(parameters: ModuleParameters.unflattened(params))
            return q
        }
        return try NF4SwitchLinear(
            inputDims: inDims, outputDims: outDims, numExperts: experts,
            weight: weight, bias: bias, scales: scales)
    }
}
