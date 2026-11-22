import Foundation
import MLX
import MLXLMCommon
import MLXNN

private let compiledRouting = MLXLMCommon.compile { (x: MLXArray, weight: MLXArray, correction: MLXArray, settings: MLXArray) -> (MLXArray, MLXArray) in
    let topK = settings.dim(0) - 1
    let scores = sigmoid(matmul(x.asType(.float32), weight.asType(.float32).transposed()))
    let indices = argPartition(-(scores + correction.asType(.float32)), kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
    let selected = takeAlong(scores, indices, axis: -1)
    return (indices, selected / (selected.sum(axis: -1, keepDims: true) + 1e-20) * settings[0])
}

final class Inkling2Router: Module {
    let topK: Int
    let routeScale: Float
    var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correction: MLXArray

    init(_ config: Inkling2Configuration) {
        (topK, routeScale) = (config.expertsPerToken, config.routeScale)
        weight = MLXArray.zeros([config.experts, config.hiddenSize], dtype: .float32)
        _correction.wrappedValue = MLXArray.zeros([config.experts], dtype: .float32)
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        compiledRouting(x, weight, correction, MLXArray.full([topK + 1], values: MLXArray(routeScale)))
    }
}

final class Inkling2Experts: Module {
    let gateCap: Float
    let upCap: Float
    @ModuleInfo(key: "gate_up_proj") var gateUp: SwitchLinear
    @ModuleInfo(key: "down_proj") var down: SwitchLinear

    init(_ config: Inkling2Configuration) {
        (gateCap, upCap) = (config.gateSoftcap, config.upSoftcap)
        _gateUp.wrappedValue = SwitchLinear(inputDims: config.hiddenSize, outputDims: 2 * config.expertWidth, numExperts: config.experts, bias: false)
        _down.wrappedValue = SwitchLinear(inputDims: config.expertWidth, outputDims: config.hiddenSize, numExperts: config.experts, bias: false)
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        var input = expandedDimensions(x, axes: [-2, -3])
        let sorts = indices.size >= 64
        var order = indices
        var inverse: MLXArray?
        if sorts { (input, order, inverse) = gatherSort(x: input, indices: indices) }
        let gateUpOut = gateUp(input, order, sortedIndices: sorts)
        let activated = Inkling2Ops.softcappedSwiGLU(
            gateUpOut[.ellipsis, .stride(by: 2)], gateUpOut[.ellipsis, .stride(from: 1, by: 2)], gateCap: gateCap, upCap: upCap)
        var out = down(activated, order, sortedIndices: sorts)
        if let inverse { out = scatterUnsort(x: out, invOrder: inverse, shape: indices.shape) }
        return out.squeezed(axis: -2)
    }
}

final class Inkling2SparseMoE: Module {
    @ModuleInfo(key: "gate") var router: Inkling2Router
    @ModuleInfo(key: "experts") var experts: Inkling2Experts

    init(_ config: Inkling2Configuration) {
        _router.wrappedValue = Inkling2Router(config)
        _experts.wrappedValue = Inkling2Experts(config)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = router(x)
        let out = experts(x, indices: indices).asType(.float32) * weights.expandedDimensions(axis: -1)
        return out.sum(axis: -2).asType(x.dtype)
    }
}

final class Inkling2Query: Module {
    var query: MLXArray

    init(rows: Int, dims: Int) {
        query = MLXArray.zeros([rows, dims], dtype: .float32)
    }

    func callAsFunction(_ blocks: MLXArray, _ partial: MLXArray) -> MLXArray {
        Inkling2Ops.depthAttention(blocks, partial, query: query.reshaped(-1, partial.dim(-1)))
    }
}

private let compiledMessageGate = compile { (arrays: [MLXArray]) -> [MLXArray] in
    let (hidden, message, down, forget, gain) = (arrays[0], arrays[1], arrays[2], arrays[3], arrays[4])
    func unit(_ x: MLXArray) -> MLXArray {
        let states = x.asType(.float32)
        return states * rsqrt((states * states).mean(axis: -1, keepDims: true) + 1e-6)
    }
    let joined = concatenated([unit(hidden), unit(message)], axis: -1).asType(down.dtype)
    let reduced = matmul(joined, down.transposed()).asType(.float32)
    let gates = matmul(reduced, concatenated([forget, gain], axis: 0).asType(.float32).transposed())
    let width = forget.dim(0)
    let blended = (1 - tanh(gates[.ellipsis, ..<width])) * hidden.asType(.float32) + tanh(gates[.ellipsis, width...]) * message.asType(.float32)
    return [blended]
}

final class Inkling2MessageGate: Module {
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "forget_gate_proj") var forgetProj: Linear
    @ModuleInfo(key: "message_gate_proj") var messageProj: Linear

    init(dims: Int, rank: Int) {
        _downProj.wrappedValue = Linear(2 * dims, rank, bias: false)
        _forgetProj.wrappedValue = Linear(rank, dims, bias: false)
        _messageProj.wrappedValue = Linear(rank, dims, bias: false)
    }

    func callAsFunction(_ hidden: MLXArray, message: MLXArray) -> MLXArray {
        compiledMessageGate([hidden, message, downProj.weight, forgetProj.weight, messageProj.weight])[0]
    }
}

final class Inkling2MessageInput: Module {
    var query: MLXArray

    init(looped: Int, dims: Int) {
        query = MLXArray.zeros([looped, 4, dims], dtype: .float32)
    }

    func callAsFunction(_ bank: (blocks: MLXArray, partial: MLXArray), layer: Int) -> MLXArray {
        Inkling2Ops.depthAttention(bank.blocks, bank.partial, query: query[layer])
    }
}
