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
        let rotated = MACaQFusedKernels.blockHadamard(
            x, scaleVector: preScale, blockSize: hadamardBlock, isInverse: false)
        return super.callAsFunction(rotated)
    }
}

package final class MACaQSwitchLinear: SwitchLinear {
    package let groupSize: Int
    package let bits: Int
    package let mode: QuantizationMode
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
        if preScale.ndim == 1 {
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

    package static func manifest(in directory: URL) -> Manifest? {
        guard let data = try? Data(contentsOf: directory.appending(component: "config.json")) else {
            return nil
        }
        return (try? JSONDecoder().decode(WrapperConfig.self, from: data))?.macaqQuantization
    }

    package static func loadContainer(
        from directory: URL, tokenizerLoader: any TokenizerLoader
    ) async throws -> ModelContainer? {
        guard let manifest = manifest(in: directory) else { return nil }
        let rawConfig = try Data(contentsOf: directory.appending(component: "config.json"))
        guard var json = try JSONSerialization.jsonObject(with: rawConfig) as? [String: Any],
              let modelType = (json["model_type"] as? String)
                  ?? ((json["text_config"] as? [String: Any])?["model_type"] as? String)
        else {
            throw BasaltError.inferenceError("MACaQ native: missing model_type in config.json")
        }
        json["quantization"] = nil
        json["quantization_config"] = nil
        json[configKey] = nil
        let stripped = try JSONSerialization.data(withJSONObject: json, options: [.sortedKeys])
        let staging = FileManager.default.temporaryDirectory.appending(component: "basalt-macaq-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: staging) }
        try stageDirectory(source: directory, staging: staging, strippedConfig: stripped)
        var failure: (any Error)?
        for family in ExpertStreamingLoader.families where await family.types.contains(modelType) {
            do {
                let model = try await family.types.createModel(configuration: stripped, modelType: modelType)
                try install(manifest.modules, in: model)
                let registry = ModelTypeRegistry<any LanguageModel>(creators: [modelType: { _ in model }])
                return try await family.factory(registry).loadContainer(from: staging, using: tokenizerLoader)
            } catch {
                failure = error
            }
        }
        throw failure ?? BasaltError.inferenceError("MACaQ native: no model factory knows model_type \(modelType)")
    }

    package static func install(_ modules: [String: MACaQLayerSpec], in model: Module) throws(BasaltError) {
        let named = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        var updates: [(String, Module)] = []
        for (path, spec) in modules.sorted(by: { $0.key < $1.key }) {
            guard spec.hadamardBlock > 0, (spec.hadamardBlock & (spec.hadamardBlock - 1)) == 0 else {
                throw .inferenceError("MACaQ native: \(path) has invalid hadamard_block \(spec.hadamardBlock)")
            }
            switch named[path] {
            case let linear as Linear where !(linear is QuantizedLinear):
                guard linear.shape.1.isMultiple(of: spec.hadamardBlock),
                      linear.shape.1.isMultiple(of: spec.groupSize)
                else {
                    throw .inferenceError("MACaQ native: \(path) width \(linear.shape.1) incompatible with block \(spec.hadamardBlock) / group \(spec.groupSize)")
                }
                updates.append((path, MACaQQuantizedLinear(replacing: linear, spec: spec)))
            case let sw as SwitchLinear where !(sw is QuantizedSwitchLinear):
                updates.append((path, MACaQSwitchLinear(replacing: sw, spec: spec)))
            default:
                throw .inferenceError("MACaQ native: \(path) is not a replaceable Linear or SwitchLinear")
            }
        }
        if !updates.isEmpty {
            model.update(modules: ModuleChildren.unflattened(updates))
        }
    }

    private static func stageDirectory(source: URL, staging: URL, strippedConfig: Data) throws {
        let fm = FileManager.default
        try strippedConfig.write(to: staging.appending(component: "config.json"))
        for name in try fm.contentsOfDirectory(atPath: source.path) where name != "config.json" && !name.hasPrefix(".") {
            try fm.createSymbolicLink(
                at: staging.appending(component: name),
                withDestinationURL: source.appending(component: name).resolvingSymlinksInPath())
        }
    }
}
