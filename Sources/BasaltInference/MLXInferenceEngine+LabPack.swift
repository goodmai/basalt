import Foundation
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

struct PackedLayer: Sendable {
    let path: String
    let weight: MaterializedArray
    let bias: MaterializedArray?
    let scales: MaterializedArray?
    let biases: MaterializedArray?
    let globalScale: MaterializedArray?
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode

    init(path: String, layer: Linear) {
        let quantized = layer as? QuantizedLinear
        self.path = path
        weight = materialize(layer.weight)
        bias = layer.bias.map { materialize($0) }
        scales = quantized.map { materialize($0.scales) }
        biases = quantized?.biases.map { materialize($0) }
        globalScale = quantized?.globalScale.map { materialize($0) }
        groupSize = quantized?.groupSize ?? 0
        bits = quantized?.bits ?? 0
        mode = quantized?.mode ?? .affine
    }

    var layer: Linear {
        func live(_ array: MaterializedArray?) -> MLXArray? { array.map { $0.reshaped($0.shape) } }
        let packed = self.weight.reshaped(self.weight.shape)
        guard let scales = live(self.scales) else { return Linear(weight: packed, bias: live(bias)) }
        return QuantizedLinear(
            weight: packed, bias: live(bias), scales: scales, biases: live(biases), groupSize: groupSize, bits: bits, mode: mode,
            globalScale: live(globalScale))
    }

    func restore(in model: Module) throws(BasaltError) {
        let layer = self.layer
        try MLXInferenceEngine.replace(path, with: layer, in: model)
        eval(layer)
    }
}

enum TensorPacker {
    static let ternaryGroupSize = 64
    static let affineBits: Set<Int> = [2, 3, 4, 5, 6, 8]
    static let groupSizes: Set<Int> = [32, 64, 128]

    struct Outcome: Sendable {
        var original: PackedLayer
        var from: String
        var bytesBefore: Int
        var bytesAfter: Int
        var rmse: Float
        var maxAbsError: Float
        var relativeError: Float
    }

    static func pack(_ request: PackRequest, in model: Module) throws(BasaltError) -> Outcome {
        let (path, layer) = try linear(request.tensor, in: model)
        let original = dense(layer)
        let packed = try converted(original, bias: layer.bias, to: request.format)
        let reference = original.asType(.float32)
        let difference = reference - dense(packed).asType(.float32)
        let squared = (difference * difference).sum()
        let norm = (reference * reference).sum()
        let rmse = sqrt(squared / Float(max(reference.size, 1)))
        let maxAbsError = abs(difference).max()
        let relativeError = which(norm .> 0, sqrt(squared / norm), 0)
        eval(packed, rmse, maxAbsError, relativeError)
        let outcome = Outcome(
            original: PackedLayer(path: path, layer: layer), from: describe(layer), bytesBefore: bytes(layer), bytesAfter: bytes(packed),
            rmse: rmse.item(Float.self), maxAbsError: maxAbsError.item(Float.self), relativeError: relativeError.item(Float.self))
        try MLXInferenceEngine.replace(path, with: packed, in: model)
        return outcome
    }

    static func linear(_ name: String, in model: Module) throws(BasaltError) -> (String, Linear) {
        let names = model.parameters().flattened().map(\.0)
        guard names.contains(name) else {
            let matches = MLXInferenceEngine.closeMatches(name, in: names).joined(separator: ", ")
            throw refused("the model has no tensor named \(name); close matches: \(matches)")
        }
        let path = String(name.dropLast(".weight".count))
        let module = name.hasSuffix(".weight") ? model.namedModules().first(where: { $0.0 == path })?.1 : nil
        guard let layer = module as? Linear, type(of: layer) == Linear.self || type(of: layer) == QuantizedLinear.self else {
            let owner = module.map { " (its module is a \(type(of: $0)))" } ?? ""
            throw refused("\(name) is not the weight of a Linear or QuantizedLinear layer\(owner); a pack swaps one whole linear layer")
        }
        return (path, layer)
    }

    static func converted(_ dense: MLXArray, bias: MLXArray?, to format: PackFormat) throws(BasaltError) -> Linear {
        let columns = dense.dim(-1)
        switch format {
        case .float16: return Linear(weight: dense.asType(.float16), bias: bias)
        case .bfloat16: return Linear(weight: dense.asType(.bfloat16), bias: bias)
        case .affine(let bits, let groupSize):
            guard affineBits.contains(bits), groupSizes.contains(groupSize) else {
                throw refused("affine packs take 2, 3, 4, 5, 6 or 8 bits and a group size of 32, 64 or 128; got \(bits) bits, group size \(groupSize)")
            }
            try divisible(columns, by: groupSize)
            return QuantizedLinear(weight: dense, bias: bias, groupSize: groupSize, bits: bits, mode: .affine)
        case .mxfp4:
            try divisible(columns, by: 32)
            return QuantizedLinear(weight: dense, bias: bias, groupSize: 32, bits: 4, mode: .mxfp4)
        case .ternary:
            try divisible(columns, by: ternaryGroupSize)
            return ternary(dense, bias: bias)
        }
    }

    static func ternary(_ dense: MLXArray, bias: MLXArray?) -> Linear {
        let rows = dense.dim(0), columns = dense.dim(1)
        let groups = dense.asType(.float32).reshaped(rows, columns / ternaryGroupSize, ternaryGroupSize)
        let scale = abs(groups).mean(axis: -1, keepDims: true)
        let signs = clip(round(groups / which(scale .> 0, scale, 1)), min: Float(-1), max: Float(1))
        let codes = (signs + 1).asType(.uint32).reshaped(rows, columns / 16, 16)
        let shifts = MLXArray((0 ..< 16).map { UInt32(1) << UInt32(2 * $0) })
        let packed = (codes * shifts).sum(axis: -1).asType(.uint32)
        let scales = scale.squeezed(axis: -1).asType(dense.dtype)
        return QuantizedLinear(weight: packed, bias: bias, scales: scales, biases: -scales, groupSize: ternaryGroupSize, bits: 2, mode: .affine)
    }

    static func dense(_ layer: Linear) -> MLXArray {
        guard let layer = layer as? QuantizedLinear else { return layer.weight }
        return dequantized(
            layer.weight, scales: layer.scales, biases: layer.biases, groupSize: layer.groupSize, bits: layer.bits, mode: layer.mode,
            globalScale: layer.globalScale)
    }

    static func bytes(_ layer: Module) -> Int { layer.parameters().flattened().reduce(0) { $0 + $1.1.nbytes } }

    static func describe(_ layer: Linear) -> String {
        (layer as? QuantizedLinear).map { "\($0.bits)-bit \($0.mode.rawValue) g\($0.groupSize)" } ?? "\(layer.weight.dtype)"
    }

    static func describe(_ format: PackFormat) -> String {
        switch format {
        case .affine(let bits, let groupSize): "\(bits)-bit affine g\(groupSize)"
        case .mxfp4: "4-bit mxfp4 g32"
        case .ternary: "ternary g\(ternaryGroupSize)"
        case .float16: "float16"
        case .bfloat16: "bfloat16"
        }
    }

    static func divisible(_ columns: Int, by groupSize: Int) throws(BasaltError) {
        guard columns % groupSize == 0 else { throw refused("the layer's input dimension \(columns) is not a multiple of the group size \(groupSize)") }
    }

    static func refused(_ reason: String) -> BasaltError { .invalidRequestStructure(details: "tensor_pack: \(reason)") }
}

extension MLXInferenceEngine {
    public func pack(_ request: PackRequest) async throws(BasaltError) -> PackReport {
        guard let container else { throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first") }
        let owner = String(request.tensor.dropLast(".weight".count)) + "."
        guard !modelFacts.deltas.keys.contains(where: { $0 == request.tensor || $0.hasPrefix(owner) }) else {
            throw .invalidRequestStructure(details: "tensor_pack: \(request.tensor) or its layer already carries an active delta or pack; undo it first")
        }
        record(.learned(modelEpoch) { $0.deltas[request.tensor] = AppliedDelta() })
        let outcome: TensorPacker.Outcome
        do throws(BasaltError) {
            outcome = try await Self.labRun(container, "the pack") { context in try TensorPacker.pack(request, in: context.model) }
        } catch {
            record(.learned(modelEpoch) { $0.deltas[request.tensor] = nil })
            throw error
        }
        guard self.container === container else {
            throw .inferenceHardwareFailure(reason: "the model was replaced while the tensor was packed; nothing is applied")
        }
        let info = register(nil, description: "tensor_pack \(request.tensor): \(outcome.from) → \(TensorPacker.describe(request.format))")
        weightsChanged { $0.deltas[request.tensor] = AppliedDelta(info: info, pack: outcome.original) }
        scheduleUndo(of: info)
        return PackReport(
            overlay: info, bytesBefore: ByteCount(Int64(outcome.bytesBefore)), bytesAfter: ByteCount(Int64(outcome.bytesAfter)),
            rmse: outcome.rmse, maxAbsError: outcome.maxAbsError, relativeError: outcome.relativeError)
    }
}
