import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

package struct MACaQLayerSpec: Codable, Sendable, Equatable {
    package var bits: Int
    package var groupSize: Int
    package var mode: String
    package var hadamardBlock: Int
    package var perExpertPreScale: Bool?
    package var scaleDType: String?
    package var hasBias: Bool?

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
        case mode
        case hadamardBlock = "hadamard_block"
        case perExpertPreScale = "per_expert_pre_scale"
        case scaleDType = "scale_dtype"
        case hasBias = "has_bias"
    }

    package init(
        bits: Int,
        groupSize: Int,
        mode: QuantizationMode,
        hadamardBlock: Int
    ) {
        self.bits = bits
        self.groupSize = groupSize
        self.mode = mode.rawValue
        self.hadamardBlock = hadamardBlock
    }

    package var quantizationMode: QuantizationMode {
        QuantizationMode(rawValue: mode) ?? .affine
    }

    package var resolvedScaleDType: DType {
        if quantizationMode == .mxfp4 { return .uint8 }
        if let scaleDType, let match = DType.allCases.first(where: { "\($0)" == scaleDType }) {
            return match
        }
        return .bfloat16
    }
}

package struct MACaQSwitchGeometry: Sendable, Equatable {
    package var experts: Int
    package var outputs: Int
    package var inputs: Int

    package init(experts: Int, outputs: Int, inputs: Int) {
        self.experts = experts
        self.outputs = outputs
        self.inputs = inputs
    }
}

package final class MACaQQuantizedLinear: QuantizedLinear {
    package let hadamardBlock: Int
    @ParameterInfo(key: "pre_scale") package var preScale: MLXArray

    package init(
        packed: MACaQPackedWeights,
        bias: MLXArray?,
        preScale: MLXArray,
        spec: MACaQLayerSpec
    ) {
        self.hadamardBlock = spec.hadamardBlock
        self._preScale.wrappedValue = preScale
        let qMode = spec.quantizationMode
        if qMode == .affine {
            super.init(
                weight: packed.weight, bias: bias, scales: packed.scales,
                biases: packed.biases, groupSize: spec.groupSize, bits: spec.bits)
        } else if bias == nil {
            super.init(
                weight: packed.weight, scales: packed.scales, biases: nil,
                groupSize: spec.groupSize, bits: spec.bits, mode: qMode)
        } else {
            let stub = MLXArray.zeros([1, spec.groupSize], dtype: .float16)
            super.init(weight: stub, bias: bias, groupSize: spec.groupSize, bits: spec.bits, mode: qMode)
            update(parameters: ModuleParameters.unflattened([("weight", packed.weight), ("scales", packed.scales)]))
        }
        freeze()
    }

    package convenience init(replacing linear: Linear, spec: MACaQLayerSpec) {
        let (outputs, inputs) = linear.shape
        let groups = inputs / spec.groupSize
        let qMode = spec.quantizationMode
        let sType = spec.resolvedScaleDType
        let weight = MLXArray.zeros([outputs, (inputs * spec.bits) / 32], dtype: .uint32)
        let scales = MLXArray.zeros([outputs, groups], dtype: sType)
        let biases = qMode == .affine ? MLXArray.zeros([outputs, groups], dtype: sType) : nil
        let hasBias = spec.hasBias ?? (linear.bias != nil)
        let bias = hasBias ? (linear.bias ?? MLXArray.zeros([outputs], dtype: .bfloat16)) : nil
        let pre = MLXArray.ones([inputs], dtype: .bfloat16)
        let packed = MACaQPackedWeights(weight: weight, scales: scales, biases: biases)
        self.init(packed: packed, bias: bias, preScale: pre, spec: spec)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let rotated = training
            ? MACaQFusedKernels.differentiableBlockHadamard(x, scaleVector: preScale, blockSize: hadamardBlock)
            : MACaQFusedKernels.blockHadamard(x, scaleVector: preScale, blockSize: hadamardBlock, isInverse: false)
        return super.callAsFunction(rotated)
    }

    package func unrotatedDenseWeight() -> MLXArray {
        let wRot = dequantized(weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits, mode: mode).asType(.float32)
        let unrotated = MACaQMath.rotateBlocks(wRot, blockSize: hadamardBlock)
        return unrotated * preScale.asType(.float32).expandedDimensions(axis: 0)
    }
}

package final class MACaQSwitchLinear: SwitchLinear, Quantized {
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    package let hadamardBlock: Int
    @ParameterInfo(key: "scales") package var scales: MLXArray
    @ParameterInfo(key: "biases") package var biases: MLXArray?
    @ParameterInfo(key: "pre_scale") package var preScale: MLXArray

    package init(
        geometry: MACaQSwitchGeometry,
        packed: MACaQPackedWeights,
        bias: MLXArray?,
        preScale: MLXArray,
        spec: MACaQLayerSpec
    ) {
        self.groupSize = spec.groupSize
        self.bits = spec.bits
        self.mode = spec.quantizationMode
        self.hadamardBlock = spec.hadamardBlock
        self._scales.wrappedValue = packed.scales
        self._biases.wrappedValue = packed.biases
        self._preScale.wrappedValue = preScale
        super.init(
            inputDims: geometry.inputs,
            outputDims: geometry.outputs,
            numExperts: geometry.experts,
            weight: packed.weight,
            bias: bias)
        freeze()
    }

    package convenience init(replacing sw: SwitchLinear, spec: MACaQLayerSpec) {
        let flat = Dictionary(uniqueKeysWithValues: sw.parameters().flattened())
        let shape = flat["weight"]?.shape ?? [1, 1, spec.groupSize]
        let geo = MACaQSwitchGeometry(experts: shape[0], outputs: shape[1], inputs: shape[2])
        let groups = geo.inputs / spec.groupSize
        let qMode = spec.quantizationMode
        let sType = spec.resolvedScaleDType
        let weight = MLXArray.zeros([geo.experts, geo.outputs, (geo.inputs * spec.bits) / 32], dtype: .uint32)
        let scales = MLXArray.zeros([geo.experts, geo.outputs, groups], dtype: sType)
        let biases = qMode == .affine ? MLXArray.zeros([geo.experts, geo.outputs, groups], dtype: sType) : nil
        let hasBias = spec.hasBias ?? (flat["bias"] != nil)
        let bias = hasBias ? (flat["bias"] ?? MLXArray.zeros([geo.experts, geo.outputs], dtype: .bfloat16)) : nil
        let preShape = (spec.perExpertPreScale == true) ? [geo.experts, geo.inputs] : [geo.inputs]
        let pre = MLXArray.ones(preShape, dtype: .bfloat16)
        let packed = MACaQPackedWeights(weight: weight, scales: scales, biases: biases)
        self.init(geometry: geo, packed: packed, bias: bias, preScale: pre, spec: spec)
    }

    public override func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let params = parameters()
        guard case .value(let w)? = params["weight"] else { return x }
        let rotated: MLXArray
        if training {
            rotated = MACaQFusedKernels.differentiableSwitchHadamard(
                x, indices: indices, preScale: preScale, blockSize: hadamardBlock)
        } else if preScale.ndim == 1 {
            rotated = MACaQFusedKernels.blockHadamard(
                x, scaleVector: preScale, blockSize: hadamardBlock, isInverse: false)
        } else {
            let gathered = expandedDimensions(preScale.take(indices, axis: 0), axis: -2)
            let scaled = (x.asType(.float32) * gathered.asType(.float32)).asType(x.dtype)
            let ones = MLXArray.ones([x.dim(-1)], dtype: x.dtype)
            rotated = MACaQFusedKernels.blockHadamard(
                scaled, scaleVector: ones, blockSize: hadamardBlock, isInverse: false)
        }
        var out = MLX.gatherQuantizedMM(
            rotated,
            w,
            scales: scales,
            biases: biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: mode,
            sortedIndices: sortedIndices)
        if case .value(let b)? = params["bias"] {
            out = out + expandedDimensions(b[indices], axis: -2)
        }
        return out
    }
}

package enum MACaQNativeLoader {
    package static let configKey = "macaq_quantization"

    package struct Manifest: Codable, Sendable {
        package var version: Int = 1
        package var alpha: Float
        package var modules: [String: MACaQLayerSpec]

        package init(version: Int = 1, alpha: Float, modules: [String: MACaQLayerSpec]) {
            self.version = version
            self.alpha = alpha
            self.modules = modules
        }
    }

    struct WrapperConfig: Decodable {
        let macaqQuantization: Manifest?

        enum CodingKeys: String, CodingKey {
            case macaqQuantization = "macaq_quantization"
        }
    }

    package static func manifest(in directory: URL) throws(BasaltError) -> Manifest? {
        let configURL = directory.appending(component: "config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              json[configKey] != nil else {
            return nil
        }
        guard let manifest = (try? JSONDecoder().decode(WrapperConfig.self, from: data))?.macaqQuantization,
              manifest.version == 1 else {
            throw .modelConfigurationInvalid(path: configURL.path, reason: "invalid macaq_quantization manifest or unsupported version")
        }
        return manifest
    }

    package static func loadContainer(
        from directory: URL, tokenizerLoader: any TokenizerLoader
    ) async throws -> ModelContainer? {
        guard let manifest = try manifest(in: directory) else { return nil }
        let rawConfig = try Data(contentsOf: directory.appending(component: "config.json"))
        guard let json = try JSONSerialization.jsonObject(with: rawConfig) as? [String: Any],
              let modelType = (json["model_type"] as? String)
                  ?? ((json["text_config"] as? [String: Any])?["model_type"] as? String)
        else {
            throw BasaltError.inferenceError("MACaQ native: missing model_type in config.json")
        }
        var failure: (any Error)?
        await MLXInferenceEngine.registerAliases()
        let scaleKeys = safetensorsScaleKeys(in: directory)
        let perLayer = (try? JSONDecoder.json5().decode(BaseConfiguration.self, from: rawConfig))?.perLayerQuantization
        for family in ExpertStreamingLoader.families where await family.types.contains(modelType) {
            do {
                let model = try await family.types.createModel(configuration: rawConfig, modelType: modelType)
                try install(manifest.modules, in: model)
                if let perLayer {
                    try installUntouchedQuantized(perLayer: perLayer, scaleKeys: scaleKeys, in: model)
                }
                let registry = ModelTypeRegistry<any LanguageModel>(creators: [modelType: { _ in model }])
                return try await family.factory(registry).loadContainer(from: directory, using: tokenizerLoader)
            } catch {
                failure = error
            }
        }
        throw failure ?? BasaltError.inferenceError("MACaQ native: no model factory knows model_type \(modelType)")
    }

    package static func install(_ modules: [String: MACaQLayerSpec], in model: Module) throws(BasaltError) {
        let named = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        for (path, spec) in modules.sorted(by: { $0.key < $1.key }) {
            guard spec.hadamardBlock > 0, (spec.hadamardBlock & (spec.hadamardBlock - 1)) == 0,
                  [2, 3, 4, 8].contains(spec.bits), [32, 64, 128].contains(spec.groupSize) else {
                throw .modelConfigurationInvalid(
                    path: path,
                    reason: "invalid MACaQ spec (block=\(spec.hadamardBlock), bits=\(spec.bits), group=\(spec.groupSize))")
            }
            let replacement: Module
            switch named[path] {
            case let linear as Linear where !(linear is QuantizedLinear):
                guard linear.shape.1.isMultiple(of: spec.hadamardBlock),
                      linear.shape.1.isMultiple(of: spec.groupSize)
                else {
                    throw .modelConfigurationInvalid(path: path, reason: "width \(linear.shape.1) incompatible with block \(spec.hadamardBlock) / group \(spec.groupSize)")
                }
                replacement = MACaQQuantizedLinear(replacing: linear, spec: spec)
            case let sw as SwitchLinear where !(sw is QuantizedSwitchLinear):
                let inDim = sw.parameters().flattened().first(where: { $0.0 == "weight" })?.1.dim(-1) ?? 0
                guard inDim > 0, inDim.isMultiple(of: spec.hadamardBlock), inDim.isMultiple(of: spec.groupSize) else {
                    throw .modelConfigurationInvalid(path: path, reason: "SwitchLinear width \(inDim) incompatible with block \(spec.hadamardBlock) / group \(spec.groupSize)")
                }
                replacement = MACaQSwitchLinear(replacing: sw, spec: spec)
            default:
                throw .modelConfigurationInvalid(path: path, reason: "not a replaceable Linear or SwitchLinear")
            }
            try swapChild(path: path, replacement: replacement, named: named, root: model)
        }
        model.invalidateCompiledTraces()
    }

    private static func installUntouchedQuantized(
        perLayer: BaseConfiguration.PerLayerQuantization,
        scaleKeys: Set<String>,
        in model: Module
    ) throws(BasaltError) {
        let named = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        for (path, leaf) in model.leafModules().flattened() where !(leaf is Quantized) {
            guard let quantizable = leaf as? Quantizable else { continue }
            let stripped = path.hasPrefix("language_model.")
                ? String(path.dropFirst("language_model.".count)) : path
            guard scaleKeys.contains("\(path).scales") || scaleKeys.contains("\(stripped).scales"),
                  let q = perLayer.quantization(layer: path) ?? perLayer.quantization(layer: stripped)
            else { continue }
            let replacement = quantizable.toQuantized(groupSize: q.groupSize, bits: q.bits, mode: q.mode)
            try swapChild(path: path, replacement: replacement, named: named, root: model)
        }
        model.invalidateCompiledTraces()
    }

    private static func swapChild(
        path: String,
        replacement: Module,
        named: [String: Module],
        root: Module
    ) throws(BasaltError) {
        var parts = path.split(separator: ".").map(String.init)
        guard let leafKey = parts.popLast() else {
            throw .inferenceError("MACaQ native: empty module path")
        }
        let parentPath = parts.joined(separator: ".")
        let parent = parentPath.isEmpty ? root : named[parentPath]
        guard let parent else {
            throw .inferenceError("MACaQ native: missing parent module '\(parentPath)' for \(path)")
        }
        do {
            try parent.update(modules: ModuleChildren.unflattened([(leafKey, replacement)]), verify: .noUnusedKeys)
        } catch {
            throw .inferenceError("MACaQ native: failed swapping \(path): \(error.localizedDescription)")
        }
    }

    private static func safetensorsScaleKeys(in directory: URL) -> Set<String> {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: directory.path) else { return [] }
        var out: Set<String> = []
        for name in names where name.lowercased().hasSuffix(".safetensors") {
            let url = directory.appending(component: name)
            let size = (try? fm.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
            if case .success(let hdr) = ModelInspector.readSafetensors(at: url, fileSize: size) {
                for key in hdr.tensors.keys where key.hasSuffix(".scales") { out.insert(key) }
            }
        }
        return out
    }
}
