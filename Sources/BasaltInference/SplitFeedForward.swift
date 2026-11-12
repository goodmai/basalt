import BasaltCore
import CoreAI
import Darwin
import Foundation
import MLX
import MLXLMCommon
import MLXNN

protocol FeedForwardShare: Sendable {
    var columns: Int { get }
    func supports(layer: Int, tokens: Int) -> Bool
    func run(layer: Int, planar: NDArray, tokens: Int) async throws -> NDArray
}

enum FeedForwardActivation: String, Sendable {
    case geluTanh, silu

    init(_ name: String) {
        self = name.contains("gelu") ? .geluTanh : .silu
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch self {
        case .geluTanh: MLXNN.geluApproximate(x)
        case .silu: MLXNN.silu(x)
        }
    }
}

struct SplitFeedForward {
    struct Rest {
        let gate: QuantizedLinear
        let up: QuantizedLinear
        let down: QuantizedLinear
    }

    let share: any FeedForwardShare
    let activation: FeedForwardActivation
    let rest: [Int: Rest]
    let full: [Int: any UnaryLayer]

    init(model: any SplitFeedForwardPrefill & Module, share: any FeedForwardShare, activation: FeedForwardActivation) throws(BasaltError) {
        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        var (rest, full) = ([Int: Rest](), [Int: any UnaryLayer]())
        for layer in model.denseFeedForwardLayers {
            let path = model.feedForwardPath(layer: layer)
            guard let mlp = modules[path] as? any UnaryLayer, let gate = modules[path + ".gate_proj"] as? QuantizedLinear,
                  let up = modules[path + ".up_proj"] as? QuantizedLinear, let down = modules[path + ".down_proj"] as? QuantizedLinear
            else { throw .modelConfigurationInvalid(path: path, reason: "the feed-forward split needs a quantized dense MLP at \(path)") }
            rest[layer] = Rest(
                gate: try Self.rows(gate, from: share.columns), up: try Self.rows(up, from: share.columns), down: try Self.inputs(down, from: share.columns))
            full[layer] = mlp
        }
        (self.share, self.activation, self.rest, self.full) = (share, activation, rest, full)
    }

    static func rows(_ linear: QuantizedLinear, from start: Int) throws(BasaltError) -> QuantizedLinear {
        guard start > 0, start < linear.weight.dim(0) else { throw .modelConfigurationInvalid(path: "feed-forward split", reason: "\(start) columns out of \(linear.weight.dim(0))") }
        return QuantizedLinear(
            weight: linear.weight[start...], scales: linear.scales[start...], biases: linear.biases?[start...], groupSize: linear.groupSize, bits: linear.bits, mode: linear.mode)
    }

    static func inputs(_ linear: QuantizedLinear, from start: Int) throws(BasaltError) -> QuantizedLinear {
        guard start % linear.groupSize == 0, start * linear.bits % 32 == 0 else {
            throw .modelConfigurationInvalid(path: "feed-forward split", reason: "\(start) columns do not fall on a \(linear.groupSize)-wide group of \(linear.bits)-bit weights")
        }
        let (packed, group) = (start * linear.bits / 32, start / linear.groupSize)
        let (weight, scales, biases) = (contiguous(linear.weight[0..., packed...]), contiguous(linear.scales[0..., group...]), linear.biases.map { contiguous($0[0..., group...]) })
        eval(weight, scales)
        if let biases { eval(biases) }
        return QuantizedLinear(weight: weight, scales: scales, biases: biases, groupSize: linear.groupSize, bits: linear.bits, mode: linear.mode)
    }

    func callAsFunction(_ layer: Int, _ x: MLXArray) async throws -> MLXArray {
        let tokens = x.dim(1)
        guard let rest = rest[layer], share.supports(layer: layer, tokens: tokens) else {
            guard let mlp = full[layer] else { throw BasaltError.inferenceError("layer \(layer) has no dense feed-forward to split") }
            return mlp(x)
        }
        let (input, share) = (Self.planar(x), share)
        async let shared = share.run(layer: layer, planar: input, tokens: tokens)
        let local = rest.down(activation(rest.gate(x)) * rest.up(x))
        eval(local)
        return local + Self.unplanar(try await shared, tokens: tokens, hidden: x.dim(2)).asType(local.dtype)
    }

    static func planar(_ x: MLXArray) -> NDArray {
        let (tokens, hidden) = (x.dim(1), x.dim(2))
        let planar = contiguous(x.reshaped([tokens, hidden]).transposed().asType(.float16))
        eval(planar)
        let bytes = planar.asData(access: .noCopyIfContiguous).data
        var array = NDArray(shape: [1, hidden, 1, tokens], scalarType: .float16)
        array.mutableView(as: Float16.self).withUnsafeMutablePointer { pointer, _, _ in
            bytes.withUnsafeBytes { source in _ = memcpy(pointer, source.baseAddress!, source.count) }
        }
        return array
    }

    static func unplanar(_ y: NDArray, tokens: Int, hidden: Int) -> MLXArray {
        let planar = y.view(as: Float16.self).withUnsafePointer { pointer, _, _ in
            MLXArray(UnsafeBufferPointer(start: pointer, count: hidden * tokens), [hidden, tokens])
        }
        return planar.transposed().reshaped([1, tokens, hidden])
    }
}
