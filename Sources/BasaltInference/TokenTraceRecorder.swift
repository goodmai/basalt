import Foundation
import MLX
import MLXLMCommon
import BasaltCore

final class TokenTraceRecorder: ForwardProbe {
    let request: TokenTraceRequest
    private let traced: Set<Int>?
    private let directions: [Int: MLXArray]
    private let readout: ((MLXArray) -> MLXArray)?
    private let edits: (any ForwardProbe)?
    private var pendingFloats: [(segment: TraceSegment, values: MLXArray)] = []
    private var pendingInts: [(segment: TraceSegment, values: MLXArray)] = []
    private var floatRows: [MLXArray] = []
    private var intRows: [MLXArray] = []
    private(set) var layout: TokenTraceLayout?
    private(set) var promptResiduals: [Int: MLXArray] = [:]
    var capturesPromptResiduals = false

    init(_ request: TokenTraceRequest, readout: ((MLXArray) -> MLXArray)?, edits: (any ForwardProbe)? = nil) {
        self.request = request
        traced = request.layers.map(Set.init)
        directions = Dictionary(request.projections.map { ($0.layer, MLXArray($0.vector)) }, uniquingKeysWith: { first, _ in first })
        self.readout = readout
        self.edits = edits
    }

    private func traces(_ layer: Int) -> Bool { traced?.contains(layer) ?? true }

    private func record(_ field: TraceField, _ layer: Int, _ values: MLXArray) {
        pendingFloats.append((TraceSegment(layer: layer, field: field, width: values.dim(1)), values.asType(.float32)))
    }

    private func record(ints field: TraceField, _ layer: Int, _ values: MLXArray) {
        pendingInts.append((TraceSegment(layer: layer, field: field, width: values.dim(1)), values.asType(.int32)))
    }

    func residual(layer: Int, _ hidden: MLXArray) -> MLXArray {
        let edited = edits?.residual(layer: layer, hidden) ?? hidden
        guard traces(layer) else { return edited }
        let states = edited[0].asType(.float32)
        if capturesPromptResiduals { promptResiduals[layer] = states[states.dim(0) - 1] }
        if let direction = directions[layer] { record(.projection, layer, matmul(states, direction).reshaped(-1, 1)) }
        if request.lensTopK > 0, let readout { recordLens(layer, readout(edited[0])) }
        return edited
    }

    private func recordLens(_ layer: Int, _ logits: MLXArray) {
        let k = request.lensTopK
        let probabilities = softmax(logits.asType(.float32), axis: -1)
        let ids = argPartition(-probabilities, kth: k - 1, axis: -1)[0..., ..<k]
        let picked = takeAlong(probabilities, ids, axis: -1)
        let order = argSort(-picked, axis: -1)
        record(ints: .lens, layer, takeAlong(ids, order, axis: -1))
        record(.lensProbabilities, layer, takeAlong(picked, order, axis: -1))
    }

    func headOutputs(layer: Int, _ perHead: MLXArray) {
        edits?.headOutputs(layer: layer, perHead)
        guard request.heads, traces(layer) else { return }
        record(.heads, layer, sqrt(square(perHead[0].asType(.float32)).sum(axis: -1)).transposed())
    }

    func wantsAttention(layer: Int) -> Bool { request.attentionEntropy && traces(layer) }

    func attentionWeights(layer: Int, _ weights: MLXArray) {
        guard wantsAttention(layer: layer) else { return }
        let p = weights[0].asType(.float32)
        record(.attentionEntropy, layer, -which(p .> 0, p * log(maximum(p, Float.leastNormalMagnitude)), 0).sum(axis: -1).transposed())
    }

    func headMask(layer: Int) -> MLXArray? { edits?.headMask(layer: layer) }

    func routerLogits(layer: Int, _ logits: MLXArray) {
        edits?.routerLogits(layer: layer, logits)
        guard request.experts, traces(layer) else { return }
        let p = softmax(logits.reshaped(-1, logits.dim(-1)).asType(.float32), axis: -1)
        record(.routerEntropy, layer, -which(p .> 0, p * log(maximum(p, Float.leastNormalMagnitude)), 0).sum(axis: -1, keepDims: true))
    }

    func routerTopK(layer: Int, indices: MLXArray, gates: MLXArray) {
        edits?.routerTopK(layer: layer, indices: indices, gates: gates)
        guard request.experts, traces(layer) else { return }
        let k = indices.dim(-1)
        let weights = gates.reshaped(-1, k).asType(.float32)
        record(ints: .experts, layer, indices.reshaped(-1, k))
        record(.gates, layer, weights)
        let sorted = MLX.sorted(weights, axis: -1)
        let margin = k > 1 ? sorted[0..., k - 1] - sorted[0..., k - 2] : sorted[0..., 0]
        record(.routerMargin, layer, margin.reshaped(-1, 1))
    }

    func expertMask(layer: Int) -> MLXArray? { edits?.expertMask(layer: layer) }

    func collect(positions: Int) throws(BasaltError) -> [MLXArray] {
        defer { (pendingFloats, pendingInts, capturesPromptResiduals) = ([], [], false) }
        let step = TokenTraceLayout(floats: pendingFloats.map(\.segment), ints: pendingInts.map(\.segment))
        guard layout == nil || layout == step else {
            throw .inferenceError("the traced hooks changed between forwards; trace fewer options or layers")
        }
        guard pendingFloats.allSatisfy({ $0.values.dim(0) == positions }), pendingInts.allSatisfy({ $0.values.dim(0) == positions }) else {
            throw .inferenceError("a traced hook reported a different number of positions than the forward ran")
        }
        layout = step
        let floats = pendingFloats.isEmpty ? MLXArray.zeros([positions, 0], type: Float.self) : concatenated(pendingFloats.map(\.values), axis: 1)
        let ints = pendingInts.isEmpty ? MLXArray.zeros([positions, 0], type: Int32.self) : concatenated(pendingInts.map(\.values), axis: 1)
        floatRows.append(floats)
        intRows.append(ints)
        return [floats, ints] + Array(promptResiduals.values)
    }

    func table() -> (floats: MLXArray, ints: MLXArray)? {
        guard !floatRows.isEmpty else { return nil }
        return (concatenated(floatRows, axis: 0), concatenated(intRows, axis: 0))
    }
}
