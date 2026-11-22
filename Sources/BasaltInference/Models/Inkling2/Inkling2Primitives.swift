import Foundation
import MLX
import MLXLMCommon
import MLXNN

private let compiledUnitRMS = compile(shapeless: true) { (x: MLXArray, eps: MLXArray) -> MLXArray in
    let states = x.asType(.float32)
    return states * rsqrt((states * states).mean(axis: -1, keepDims: true) + eps)
}

private let compiledL2Clip = compile(shapeless: true) { (x: MLXArray, scale: MLXArray, threshold: MLXArray) -> MLXArray in
    let states = x.asType(.float32) * scale
    let norm = sqrt((states * states).sum(axis: -1, keepDims: true))
    return (states * minimum(threshold / (norm + 1e-6), 1)).asType(x.dtype)
}

private let compiledSwiGLU = compile(shapeless: true) { (gate: MLXArray, up: MLXArray, caps: MLXArray) -> MLXArray in
    let g = gate.asType(.float32)
    let u = up.asType(.float32)
    return (tanh(g / caps[0]) * caps[0] * sigmoid(g) * (tanh(u / caps[1]) * caps[1])).asType(gate.dtype)
}

private let compiledDepthAttention = compile { (blocks: MLXArray, partial: MLXArray, query: MLXArray) -> MLXArray in
    let stacked = concatenated([blocks, partial.expandedDimensions(axis: 0)], axis: 0)
    let sources = stacked.movedAxis(source: 0, destination: -2)
    let scale = 1 / Float(partial.dim(-1)).squareRoot()
    let weights = softmax(matmul(sources, query.asType(.float32).transposed()) * scale, axis: -2, precise: true)
    return matmul(weights.swappedAxes(-1, -2), sources)
}

private let compiledShortConvolution = compile { (window: MLXArray, x: MLXArray, weight: MLXArray) -> MLXArray in
    let (kernel, length) = (weight.dim(1), x.dim(1))
    let taps = weight[0..., 0..., 0].asType(.float32)
    var out = window[0..., 0 ..< length, 0...] * taps[0..., 0]
    for k in 1 ..< kernel {
        out = out + window[0..., k ..< (k + length), 0...] * taps[0..., k]
    }
    let mixed = (out.asType(weight.dtype).asType(.float32) + x.asType(.float32)) / Float(kernel + 1).squareRoot()
    return mixed.asType(x.dtype)
}

private let compiledDynamicConvolution = MLXLMCommon.compile { (window: MLXArray, x: MLXArray, staticWeight: MLXArray, dynamic: MLXArray) -> MLXArray in
    let (batch, length, channels) = (x.dim(0), x.dim(1), x.dim(2))
    let (kernel, heads) = (staticWeight.dim(0), dynamic.dim(-1) / staticWeight.dim(0))
    let weights = staticWeight.asType(.float32).transposed().reshaped(heads, -1, kernel)
        + dynamic.asType(.float32).reshaped(batch, length, heads, 1, kernel)
    let lags = (0 ..< kernel).map { lag in window[0..., (kernel - 1 - lag) ..< (kernel - 1 - lag + length), 0...] }
    let windows = stacked(lags, axis: -1).reshaped(batch, length, heads, -1, kernel)
    let out = (windows * weights).sum(axis: -1).reshaped(batch, length, channels)
    return ((out + x.asType(.float32)) / Float(2 * kernel + 1).squareRoot()).asType(x.dtype)
}

enum Inkling2Ops {
    static func unitRMS(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        compiledUnitRMS(x, MLXArray(eps))
    }

    static func l2Clip(_ x: MLXArray, scale: MLXArray?, threshold: Float) -> MLXArray {
        compiledL2Clip(x, scale?.asType(.float32) ?? MLXArray(Float(1)), MLXArray(threshold))
    }

    static func softcappedSwiGLU(_ gate: MLXArray, _ up: MLXArray, gateCap: Float, upCap: Float) -> MLXArray {
        compiledSwiGLU(gate, up, MLXArray([gateCap, upCap]))
    }

    static func depthAttention(_ blocks: MLXArray, _ partial: MLXArray, query: MLXArray) -> MLXArray {
        compiledDepthAttention(blocks, partial, query)
    }

    static func convWindow(_ x: MLXArray, state: MLXArray?, kernel: Int) -> MLXArray {
        let states = x.asType(.float32)
        let previous = state ?? MLXArray.zeros([states.dim(0), kernel - 1, states.dim(-1)], dtype: .float32)
        return concatenated([previous.asType(.float32), states], axis: 1)
    }

    static func convState(_ window: MLXArray, kernel: Int) -> MLXArray {
        let length = window.dim(1)
        return window[0..., (length - (kernel - 1))..., 0...] + 0
    }
}

final class Inkling2DepthwiseWeight: Module {
    var weight: MLXArray

    init(dims: Int, kernel: Int) {
        weight = MLXArray.zeros([dims, kernel, 1])
    }
}

final class Inkling2ShortConvolution: Module {
    let kernel: Int
    @ModuleInfo(key: "conv1d") var conv: Inkling2DepthwiseWeight

    init(dims: Int, kernel: Int) {
        self.kernel = kernel
        _conv.wrappedValue = Inkling2DepthwiseWeight(dims: dims, kernel: kernel)
    }

    func callAsFunction(_ x: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        Inkling2Ops.shortConvolution(x, weight: conv.weight, state: state, kernel: kernel)
    }
}

extension Inkling2Ops {
    static func shortConvolution(_ x: MLXArray, weight: MLXArray, state: MLXArray?, kernel: Int) -> (MLXArray, MLXArray) {
        let window = convWindow(x, state: state, kernel: kernel)
        return (compiledShortConvolution(window, x, weight), window)
    }

    static func dynamicConvolution(
        _ x: MLXArray, dynamic: MLXArray, staticWeight: MLXArray, heads: Int, state: MLXArray?, kernel: Int
    ) -> (MLXArray, MLXArray) {
        let window = convWindow(x, state: state, kernel: kernel)
        return (compiledDynamicConvolution(window, x, staticWeight, dynamic), window)
    }
}
