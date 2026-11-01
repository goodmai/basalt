import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

struct SignedBlockHadamard: Sendable, Equatable {
    let blockSize: Int
    private let scale: Float

    init?(blockSize: Int, width: Int) {
        guard blockSize > 0, blockSize & (blockSize - 1) == 0, width % blockSize == 0 else { return nil }
        self.blockSize = blockSize
        scale = 1 / Float(blockSize).squareRoot()
    }

    func forward(_ x: MLXArray, signs: MLXArray) -> MLXArray {
        transform(x.asType(.float32) * signs).asType(x.dtype)
    }

    func inverse(_ x: MLXArray, signs: MLXArray) -> MLXArray {
        (transform(x.asType(.float32)) * signs).asType(x.dtype)
    }

    private func transform(_ x: MLXArray) -> MLXArray {
        hadamardTransform(x.reshaped(-1, blockSize), scale: scale).reshaped(x.shape)
    }
}

final class HadamardQuantizedLinear: QuantizedLinear {
    let rotation: SignedBlockHadamard
    let signs: MLXArray

    init(replacing linear: Linear, quantization: PrismHadamardManifest.Quantization, rotation: SignedBlockHadamard) {
        let (outputs, inputs) = linear.shape
        let groups = inputs / quantization.groupSize
        self.rotation = rotation
        signs = MLXArray.ones([inputs], dtype: .float32)
        super.init(
            weight: MLXArray.zeros([outputs, inputs * quantization.bits / 32], dtype: .uint32), bias: linear.bias,
            scales: MLXArray.zeros([outputs, groups], dtype: .float16),
            biases: MLXArray.zeros([outputs, groups], dtype: .float16),
            groupSize: quantization.groupSize, bits: quantization.bits, mode: .affine)
        freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        super.callAsFunction(rotation.forward(x, signs: signs))
    }
}

final class HadamardQuantizedEmbedding: QuantizedEmbedding {
    let rotation: SignedBlockHadamard
    let signs: MLXArray

    init(replacing embedding: Embedding, quantization: PrismHadamardManifest.Quantization, rotation: SignedBlockHadamard) {
        let (count, dimensions) = embedding.shape
        let groups = dimensions / quantization.groupSize
        self.rotation = rotation
        signs = MLXArray.ones([dimensions], dtype: .float32)
        super.init(
            weight: MLXArray.zeros([count, dimensions * quantization.bits / 32], dtype: .uint32),
            scales: MLXArray.zeros([count, groups], dtype: .float16),
            biases: MLXArray.zeros([count, groups], dtype: .float16),
            groupSize: quantization.groupSize, bits: quantization.bits, mode: .affine)
        freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        rotation.inverse(super.callAsFunction(x), signs: signs)
    }

    override func asLinear(_ x: MLXArray) -> MLXArray {
        super.asLinear(rotation.forward(x, signs: signs))
    }
}

struct PrismHadamardManifest: Decodable {
    struct Entry: Decodable {
        var path: String
        var block: Int
        var embedding: Bool?
        var dtype: String?
    }

    struct Quantization: Decodable {
        var groupSize: Int
        var bits: Int
        var mode: String?

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size", bits, mode
        }
    }

    var schemaVersion: Int
    var baseModelType: String?
    var tensorNamespace: String?
    var gdnActivationLayout: String?
    var modules: [Entry]
    var quantization: Quantization

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version", baseModelType = "base_model_type", tensorNamespace = "tensor_namespace"
        case gdnActivationLayout = "gdn_activation_layout", modules, quantization
    }
}

enum PrismHadamardError: LocalizedError, Equatable {
    case unsupported(String)
    case module(String, String)
    case invalidSigns(String)

    var errorDescription: String? {
        switch self {
        case .unsupported(let reason): "Hadamard-rotated checkpoint not supported: \(reason)"
        case .module(let path, let reason): "Hadamard-rotated checkpoint module \(path): \(reason)"
        case .invalidSigns(let path): "Hadamard-rotated checkpoint module \(path) has signs other than -1 and +1"
        }
    }
}

enum PrismHadamard {
    static let modelType = "prism_hadamard_qwen35"

    static func model(configuration data: Data) throws -> Qwen35Model {
        let decoder = JSONDecoder.json5()
        let manifest = try decoder.decode(PrismHadamardManifest.self, from: data)
        try validate(manifest)
        let model = Qwen35Model(try decoder.decode(Qwen35Configuration.self, from: data))
        try substitute(manifest, in: model, prefix: "language_model.")
        return model
    }

    static func validate(_ manifest: PrismHadamardManifest) throws {
        guard manifest.schemaVersion == 2 else { throw PrismHadamardError.unsupported("schema_version \(manifest.schemaVersion), expected 2") }
        guard manifest.baseModelType == "qwen3_5" else { throw PrismHadamardError.unsupported("base_model_type \(manifest.baseModelType ?? "nil")") }
        guard manifest.tensorNamespace == "mlx-vlm-qwen3_5" else { throw PrismHadamardError.unsupported("tensor_namespace \(manifest.tensorNamespace ?? "nil")") }
        guard (manifest.gdnActivationLayout ?? "grouped") == "grouped" else { throw PrismHadamardError.unsupported("gdn_activation_layout \(manifest.gdnActivationLayout ?? "")") }
        guard (manifest.quantization.mode ?? "affine") == "affine" else { throw PrismHadamardError.unsupported("quantization mode \(manifest.quantization.mode ?? "")") }
        guard !manifest.modules.isEmpty else { throw PrismHadamardError.unsupported("the manifest lists no modules") }
        guard Set(manifest.modules.map(\.path)).count == manifest.modules.count else { throw PrismHadamardError.unsupported("a module is listed twice") }
        guard Set(manifest.modules.compactMap(\.dtype)).isSubset(of: ["float16"]) else {
            throw PrismHadamardError.unsupported("activation dtypes \(Set(manifest.modules.compactMap(\.dtype)).sorted()), expected float16")
        }
    }

    static func substitute(_ manifest: PrismHadamardManifest, in model: Module, prefix: String) throws {
        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        let replacements = try manifest.modules.map { entry -> (String, Module) in
            let path = prefix + entry.path
            guard let module = leaves[path] else { throw PrismHadamardError.module(path, "not in the model") }
            switch module {
            case let embedding as Embedding where entry.embedding == true && type(of: module) == Embedding.self:
                guard let rotation = SignedBlockHadamard(blockSize: entry.block, width: embedding.shape.1) else {
                    throw PrismHadamardError.module(path, "block \(entry.block) does not divide width \(embedding.shape.1)")
                }
                return (path, HadamardQuantizedEmbedding(replacing: embedding, quantization: manifest.quantization, rotation: rotation))
            case let linear as Linear where entry.embedding != true && type(of: module) == Linear.self:
                guard let rotation = SignedBlockHadamard(blockSize: entry.block, width: linear.shape.1) else {
                    throw PrismHadamardError.module(path, "block \(entry.block) does not divide width \(linear.shape.1)")
                }
                return (path, HadamardQuantizedLinear(replacing: linear, quantization: manifest.quantization, rotation: rotation))
            default:
                throw PrismHadamardError.module(path, "is a \(type(of: module)), not a plain Linear or Embedding")
            }
        }
        model.update(modules: ModuleChildren.unflattened(replacements))
    }

    static func prepareLoaded(_ model: Module) throws {
        let leaves = model.leafModules().flattened()
        let rotated = leaves.filter { $0.1 is HadamardQuantizedLinear || $0.1 is HadamardQuantizedEmbedding }
        guard !rotated.isEmpty else { return }
        let rotatedPaths = Set(rotated.map(\.0))
        let signs = rotated.compactMap { path, module in
            (module as? HadamardQuantizedLinear)?.signs ?? (module as? HadamardQuantizedEmbedding)?.signs
        }
        if !stacked(signs.map { (abs($0) .== 1).all() }).all().item(Bool.self) {
            let invalid = rotated.first { _, module in
                let values = (module as? HadamardQuantizedLinear)?.signs ?? (module as? HadamardQuantizedEmbedding)?.signs
                return values.map { !(abs($0) .== 1).all().item(Bool.self) } ?? false
            }
            throw PrismHadamardError.invalidSigns(invalid?.0 ?? "")
        }
        let cast = model.parameters().flattened().compactMap { key, value -> (String, MLXArray)? in
            let owner = key.split(separator: ".").dropLast().joined(separator: ".")
            guard value.dtype.isFloatingPoint, value.dtype != .float16, !key.hasSuffix("A_log"),
                  !rotatedPaths.contains(owner) else { return nil }
            return (key, value.asType(.float16))
        }
        guard !cast.isEmpty else { return }
        try model.update(parameters: ModuleParameters.unflattened(cast), verify: [.noUnusedKeys])
        eval(cast.map(\.1))
    }
}
