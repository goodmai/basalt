import Foundation
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

public protocol ParameterEfficientLayer: LoRALayer {
    var adapterScale: Float { get }
    func materializeDeltaWeight() -> MLXArray
}

public final class PHMBasisBank {
    public let hypercomplexDim: Int
    public let basis: MLXArray
    var ownerBasis: (() -> MLXArray?)?

    public init(hypercomplexDim: Int) {
        let n = max(1, hypercomplexDim)
        self.hypercomplexDim = n
        let scale = 1.0 / sqrt(Float(n))
        let eyePart = (MLXArray.eye(n) * (0.5 * scale)).reshaped(1, n, n)
        let jitter = MLXRandom.uniform(low: -scale, high: scale, [n, n, n])
        self.basis = eyePart + jitter
    }

    public init(basis: MLXArray) {
        self.hypercomplexDim = basis.dim(0)
        self.basis = basis
    }
}

public enum PHMBasisMode {
    case independent
    case owned(PHMBasisBank)
    case borrowed(PHMBasisBank)
}

public struct PHMAdapterConfiguration: Sendable, Codable, Equatable {
    public struct Parameters: Sendable, Codable, Equatable {
        public var hypercomplexDim: Int
        public var factorRank: Int
        public var sharedBasis: Bool
        public var scale: Float
        public var layers: [Int]?
        public var keys: [String]?

        public init(
            hypercomplexDim: Int,
            factorRank: Int = 1,
            sharedBasis: Bool = true,
            scale: Float = 20.0,
            layers: [Int]? = nil,
            keys: [String]? = nil
        ) {
            self.hypercomplexDim = hypercomplexDim
            self.factorRank = factorRank
            self.sharedBasis = sharedBasis
            self.scale = scale
            self.layers = layers
            self.keys = keys
        }
    }

    public var fineTuneType: String
    public var numLayers: Int
    public var phmParameters: Parameters

    enum CodingKeys: String, CodingKey {
        case fineTuneType = "fine_tune_type"
        case numLayers = "num_layers"
        case phmParameters = "phm_parameters"
    }

    public init(numLayers: Int, phmParameters: Parameters) {
        self.fineTuneType = "phm"
        self.numLayers = numLayers
        self.phmParameters = phmParameters
    }

    public var spec: AdapterSpec {
        AdapterSpec(
            method: .phm(
                hypercomplexDim: phmParameters.hypercomplexDim,
                factorRank: phmParameters.factorRank,
                sharedBasis: phmParameters.sharedBasis,
                scale: phmParameters.scale
            ),
            layers: phmParameters.layers,
            targetSuffixes: phmParameters.keys
        )
    }
}

public struct PHMAdapterContainer: ModelAdapter, Sendable, Equatable {
    public struct SavedTensor: Sendable, Equatable {
        public let shape: [Int]
        public let values: [Float]

        public init(shape: [Int], values: [Float]) {
            self.shape = shape
            self.values = values
        }

        public init(_ array: MLXArray) {
            eval(array)
            self.shape = array.shape
            self.values = array.asType(.float32).asArray(Float.self)
        }

        public func makeArray() -> MLXArray {
            MLXArray(values, shape)
        }
    }

    public let configuration: PHMAdapterConfiguration
    public let tensors: [String: SavedTensor]

    public init(configuration: PHMAdapterConfiguration, tensors: [String: SavedTensor]) {
        self.configuration = configuration
        self.tensors = tensors
    }

    public init(configuration: PHMAdapterConfiguration, parameters: ModuleParameters) {
        let flat = parameters.flattened()
        eval(flat.map(\.1))
        self.configuration = configuration
        self.tensors = Dictionary(uniqueKeysWithValues: flat.map { ($0.0, SavedTensor($0.1)) })
    }

    public static func from(directory: URL) throws -> PHMAdapterContainer {
        let configURL = directory.appending(component: "adapter_config.json")
        let data = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(PHMAdapterConfiguration.self, from: data)
        guard config.fineTuneType == "phm" else {
            throw ModelAdapterError.unsupportedAdapterType(config.fineTuneType)
        }
        let weightsURL = directory.appending(component: "adapters.safetensors")
        let arrays = try MLX.loadArrays(url: weightsURL)
        eval(Array(arrays.values))
        let saved = arrays.mapValues { SavedTensor($0) }
        return PHMAdapterContainer(configuration: config, tensors: saved)
    }

    public static func save(configuration: PHMAdapterConfiguration, model: Module, to directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        try JSONEncoder().encode(configuration).write(to: directory.appending(component: "adapter_config.json"))
        let weights = Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened())
        try MLX.save(arrays: weights, url: directory.appending(component: "adapters.safetensors"))
    }

    public func load(into model: any LanguageModel) throws {
        _ = try PHMAdapterApplicator.apply(to: model, spec: configuration.spec, targetSuffixes: configuration.phmParameters.keys)
        let expected = Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened().map { ($0.0, $0.1.shape) })
        let actual = tensors.mapValues(\.shape)
        guard actual == expected else {
            _ = PHMAdapterApplicator.revert(in: model)
            throw BasaltError.invalidRequestStructure(details: "phm adapter weights do not match target model shapes")
        }
        do {
            let arrays = tensors.mapValues { $0.makeArray() }
            try model.update(parameters: .unflattened(arrays), verify: [.noUnusedKeys, .shapeMismatch])
            eval(model.trainableParameters().flattened().map(\.1))
        } catch {
            _ = PHMAdapterApplicator.revert(in: model)
            throw error
        }
    }

    public func fuse(with model: any LanguageModel) throws {
        try load(into: model)
        _ = PHMAdapterApplicator.fuse(in: model)
    }

    public func unload(from model: any LanguageModel) {
        _ = PHMAdapterApplicator.revert(in: model)
    }
}
