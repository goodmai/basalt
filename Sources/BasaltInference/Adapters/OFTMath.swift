import Foundation
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

public final class OFTSkewBasis {
    public let blockSize: Int
    public let freePerBlock: Int
    public let upperIndices: MLXArray
    public let identity: MLXArray

    public init(blockSize: Int) {
        let b = max(2, blockSize)
        let free = b * (b - 1) / 2
        var indices = [Int32](repeating: 0, count: b * b)
        var cursor: Int32 = 1
        for row in 0 ..< b {
            for col in (row + 1) ..< b {
                indices[row * b + col] = cursor
                cursor += 1
            }
        }
        let idxArray = MLXArray(indices)
        let eyeArray = MLXArray.eye(b, dtype: .float32).reshaped(1, b, b)
        eval(idxArray, eyeArray)
        self.blockSize = b
        self.freePerBlock = free
        self.upperIndices = idxArray
        self.identity = eyeArray
    }
}

public enum OFTMath {
    public static let trainableKeys: Set<String> = ["oft_r"]

    public static func validate(
        inputDims: Int,
        outputDims: Int,
        blockSize: Int,
        parameterization: OrthogonalParameterization,
        blockShared: Bool
    ) throws(BasaltError) {
        let method = AdapterMethod.oft(
            blockSize: blockSize,
            parameterization: parameterization,
            blockShared: blockShared
        )
        try method.validate(inputDim: inputDims, outputDim: outputDims, prefix: "oft")
    }

    public static func skewSymmetric(params: MLXArray, basis: OFTSkewBasis) -> MLXArray {
        let p = params.asType(.float32)
        let k = p.dim(0)
        let b = basis.blockSize
        let zeroCol = MLXArray.zeros([k, 1], dtype: .float32)
        let padded = concatenated([zeroCol, p], axis: 1)
        let upper = take(padded, basis.upperIndices, axis: 1).reshaped(k, b, b)
        return upper - upper.swappedAxes(-1, -2)
    }

    public static func matrixExp(_ skew: MLXArray, identity: MLXArray) -> MLXArray {
        let a = skew.asType(.float32)
        let maxNorm: Float = 32.0
        let normF = ((a * a).sum(axes: [-2, -1], keepDims: true) + Float(1e-12)).sqrt()
        let bounded = a * (maxNorm / maximum(normF, maxNorm))
        let scaled = bounded * Float(0.015625)
        var poly = identity + scaled * Float(0.1)
        for k in stride(from: 9, through: 2, by: -1) {
            poly = identity + matmul(scaled, poly) * (Float(1.0) / Float(k))
        }
        var expm1 = matmul(scaled, poly)
        for _ in 0 ..< 6 {
            expm1 = expm1 * Float(2.0) + matmul(expm1, expm1)
        }
        return identity + expm1
    }

    public static func cayley(_ skew: MLXArray, identity: MLXArray) -> MLXArray {
        let a = skew.asType(.float32)
        let maxNorm: Float = 0.75
        let normF = ((a * a).sum(axes: [-2, -1], keepDims: true) + Float(1e-12)).sqrt()
        let bounded = a * (maxNorm / maximum(normF, maxNorm))
        let a2 = matmul(bounded, bounded)
        let a4 = matmul(a2, a2)
        let a8 = matmul(a4, a4)
        let a16 = matmul(a8, a8)
        let plusA = identity + bounded
        var inv = matmul(plusA, identity + a2)
        inv = matmul(inv, identity + a4)
        inv = matmul(inv, identity + a8)
        inv = matmul(inv, identity + a16)
        return matmul(inv, plusA)
    }

    public static func orthogonalBlocks(
        params: MLXArray,
        basis: OFTSkewBasis,
        parameterization: OrthogonalParameterization
    ) -> MLXArray {
        let skew = skewSymmetric(params: params, basis: basis)
        switch parameterization {
        case .matrixExp:
            return matrixExp(skew, identity: basis.identity)
        case .cayley:
            return cayley(skew, identity: basis.identity)
        }
    }

    public static func rotateInput(
        _ x: MLXArray,
        blocks: MLXArray,
        totalBlocks: Int,
        targetDType: DType
    ) -> MLXArray {
        let b = blocks.dim(-1)
        let inputDims = totalBlocks * b
        let mCount = x.size / max(1, inputDims)
        guard mCount > 0 else { return x.asType(targetDType) }
        let xf = x.asType(.float32)
        let rotated: MLXArray
        if blocks.dim(0) == 1 {
            rotated = matmul(xf.reshaped(mCount * totalBlocks, b), blocks[0].T).reshaped(x.shape)
        } else {
            let batched = xf.reshaped(mCount, totalBlocks, b).swappedAxes(0, 1)
            let out = matmul(batched, blocks.swappedAxes(-1, -2))
            rotated = out.swappedAxes(0, 1).reshaped(x.shape)
        }
        return rotated.asType(targetDType)
    }

    public static func rotateWeight(_ weight: MLXArray, blocks: MLXArray) -> MLXArray {
        let wf = weight.asType(.float32)
        let dOut = wf.dim(0)
        let dIn = wf.dim(1)
        let b = blocks.dim(-1)
        let totalBlocks = dIn / b
        if blocks.dim(0) == 1 {
            return matmul(wf.reshaped(dOut * totalBlocks, b), blocks[0]).reshaped(dOut, dIn)
        }
        let batched = wf.reshaped(dOut, totalBlocks, b).swappedAxes(0, 1)
        return matmul(batched, blocks).swappedAxes(0, 1).reshaped(dOut, dIn)
    }

    public static func blockOrthogonalityDefects(_ r: MLXArray) -> MLXArray {
        let rf = r.asType(.float32)
        let batched = rf.ndim == 2 ? rf.reshaped(1, rf.dim(0), rf.dim(1)) : rf
        let b = batched.dim(-1)
        let eye = MLXArray.eye(b, dtype: .float32).reshaped(1, b, b)
        let diff = matmul(batched.swappedAxes(-1, -2), batched) - eye
        return (diff * diff).sum(axes: [-2, -1]).sqrt()
    }

    public static func orthogonalityDefect(_ r: MLXArray) -> Float {
        let worst = blockOrthogonalityDefects(r).max()
        eval(worst)
        return worst.item(Float.self)
    }
}

public struct OFTAdapterConfiguration: Sendable, Codable, Equatable {
    public struct Parameters: Sendable, Codable, Equatable {
        public var blockSize: Int
        public var parameterization: OrthogonalParameterization
        public var blockShared: Bool
        public var layers: [Int]?
        public var keys: [String]?

        public init(
            blockSize: Int,
            parameterization: OrthogonalParameterization = .cayley,
            blockShared: Bool = false,
            layers: [Int]? = nil,
            keys: [String]? = nil
        ) {
            self.blockSize = blockSize
            self.parameterization = parameterization
            self.blockShared = blockShared
            self.layers = layers
            self.keys = keys
        }
    }

    public var fineTuneType: String
    public var numLayers: Int
    public var oftParameters: Parameters

    enum CodingKeys: String, CodingKey {
        case fineTuneType = "fine_tune_type"
        case numLayers = "num_layers"
        case oftParameters = "oft_parameters"
    }

    public init(numLayers: Int, oftParameters: Parameters) {
        self.fineTuneType = "oft"
        self.numLayers = numLayers
        self.oftParameters = oftParameters
    }

    public var spec: AdapterSpec {
        AdapterSpec(
            method: .oft(
                blockSize: oftParameters.blockSize,
                parameterization: oftParameters.parameterization,
                blockShared: oftParameters.blockShared
            ),
            layers: oftParameters.layers,
            targetSuffixes: oftParameters.keys
        )
    }
}

public struct OFTAdapterContainer: ModelAdapter, Sendable, Equatable {
    public typealias SavedTensor = PHMAdapterContainer.SavedTensor

    public let configuration: OFTAdapterConfiguration
    public let tensors: [String: SavedTensor]

    public init(configuration: OFTAdapterConfiguration, tensors: [String: SavedTensor]) {
        self.configuration = configuration
        self.tensors = tensors
    }

    public init(configuration: OFTAdapterConfiguration, parameters: ModuleParameters) {
        let flat = parameters.flattened()
        eval(flat.map(\.1))
        self.configuration = configuration
        self.tensors = Dictionary(uniqueKeysWithValues: flat.map { ($0.0, SavedTensor($0.1)) })
    }

    public static func from(directory: URL) throws -> OFTAdapterContainer {
        let configURL = directory.appending(component: "adapter_config.json")
        let data = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(OFTAdapterConfiguration.self, from: data)
        guard config.fineTuneType == "oft" else {
            throw ModelAdapterError.unsupportedAdapterType(config.fineTuneType)
        }
        let weightsURL = directory.appending(component: "adapters.safetensors")
        let arrays = try MLX.loadArrays(url: weightsURL)
        eval(Array(arrays.values))
        let saved = arrays.mapValues { SavedTensor($0) }
        return OFTAdapterContainer(configuration: config, tensors: saved)
    }

    public static func save(configuration: OFTAdapterConfiguration, model: Module, to directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        try JSONEncoder().encode(configuration).write(to: directory.appending(component: "adapter_config.json"))
        let weights = Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened())
        try MLX.save(arrays: weights, url: directory.appending(component: "adapters.safetensors"))
    }

    public func load(into model: any LanguageModel) throws {
        _ = try PHMAdapterApplicator.apply(to: model, spec: configuration.spec, targetSuffixes: configuration.oftParameters.keys)
        let expected = Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened().map { ($0.0, $0.1.shape) })
        let actual = tensors.mapValues(\.shape)
        guard actual == expected else {
            _ = PHMAdapterApplicator.revert(in: model)
            throw BasaltError.invalidRequestStructure(details: "oft adapter weights do not match target model shapes")
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
