import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class Inkling2RelativeLogits: Module {
    var proj: MLXArray

    init(relativeDim: Int, extent: Int) {
        proj = MLXArray.zeros([relativeDim, extent])
    }
}

final class Inkling2ClipScale: Module {
    var scale: MLXArray

    init(heads: Int, dims: Int) {
        scale = MLXArray.ones([heads, dims])
    }
}

final class Inkling2KVConvolution: Module {
    @ModuleInfo(key: "conv1d") var conv: Inkling2DepthwiseWeight?
    @ParameterInfo(key: "static_weight") var staticWeight: MLXArray?
    @ModuleInfo(key: "weight_proj") var weightProj: Linear?
    let kernel: Int
    let heads: Int

    init(dims: Int, inputDims: Int, heads: Int, kernel: Int, dynamic: Bool) {
        self.kernel = kernel
        self.heads = heads
        if dynamic {
            _staticWeight.wrappedValue = MLXArray.zeros([kernel, dims])
            _weightProj.wrappedValue = Linear(inputDims, heads * kernel, bias: false)
        } else {
            _conv.wrappedValue = Inkling2DepthwiseWeight(dims: dims, kernel: kernel)
        }
    }

    func callAsFunction(_ x: MLXArray, input: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        if let conv { return Inkling2Ops.shortConvolution(x, weight: conv.weight, state: state, kernel: kernel) }
        guard let staticWeight, let weightProj else { return (x, Inkling2Ops.convWindow(x, state: state, kernel: kernel)) }
        return Inkling2Ops.dynamicConvolution(x, dynamic: weightProj(input), staticWeight: staticWeight, heads: heads, state: state, kernel: kernel)
    }
}

struct Inkling2Positions {
    var query: Int
    var key: Int
}

struct Inkling2AttentionInputs {
    var query: MLXArray
    var key: MLXArray
    var value: MLXArray
}

final class Inkling2Attention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let window: Int
    let scale: Float
    let config: Inkling2Configuration
    let global: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "r_proj") var rProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "sink_proj") var sinkProj: Linear
    @ModuleInfo(key: "rel_logits_proj") var relativeLogits: Inkling2RelativeLogits
    @ModuleInfo(key: "q_l2_clip") var queryClip: Inkling2ClipScale
    @ModuleInfo(key: "rel_l2_clip") var relativeClip: Inkling2ClipScale
    @ModuleInfo(key: "k_sconv") var keyConv: Inkling2KVConvolution
    @ModuleInfo(key: "v_sconv") var valueConv: Inkling2KVConvolution

    init(_ config: Inkling2Configuration, global: Bool) {
        self.config = config
        self.global = global
        (heads, kvHeads, headDim) = (config.heads, config.kvHeads, config.headDim)
        window = global ? 0 : config.slidingWindow
        scale = 1 / Float(config.headDim).squareRoot()
        let (dims, qDims, kvDims) = (config.hiddenSize, config.heads * config.headDim, config.kvHeads * config.headDim)
        _qProj.wrappedValue = Linear(dims, qDims, bias: false)
        _kProj.wrappedValue = Linear(dims, kvDims, bias: false)
        _vProj.wrappedValue = Linear(dims, kvDims, bias: false)
        _rProj.wrappedValue = Linear(dims, config.heads * config.relativeDim, bias: false)
        _oProj.wrappedValue = Linear(qDims, dims, bias: false)
        _gateProj.wrappedValue = Linear(dims, qDims, bias: false)
        _sinkProj.wrappedValue = Linear(dims, config.heads, bias: false)
        _relativeLogits.wrappedValue = Inkling2RelativeLogits(relativeDim: config.relativeDim, extent: global ? config.relativeExtent : config.slidingWindow)
        _queryClip.wrappedValue = Inkling2ClipScale(heads: config.heads, dims: config.headDim)
        _relativeClip.wrappedValue = Inkling2ClipScale(heads: config.heads, dims: config.relativeDim)
        _keyConv.wrappedValue = Inkling2KVConvolution(dims: kvDims, inputDims: dims, heads: config.kvHeads, kernel: config.shortConvKernel, dynamic: global)
        _valueConv.wrappedValue = Inkling2KVConvolution(dims: kvDims, inputDims: dims, heads: config.kvHeads, kernel: config.shortConvKernel, dynamic: global)
    }

    func callAsFunction(_ inputs: Inkling2AttentionInputs, cache: KVCache?, conv: inout [MLXArray?]) -> MLXArray {
        let (batch, length) = (inputs.query.dim(0), inputs.query.dim(1))
        let hidden = inputs.query
        let (keys, keyWindow) = keyConv(kProj(inputs.key), input: inputs.key, state: conv[0])
        let (values, valueWindow) = valueConv(vProj(inputs.value), input: inputs.value, state: conv[1])
        (conv[0], conv[1]) = (keyWindow, valueWindow)
        let start = cache?.offset ?? 0
        var queries = Inkling2Ops.l2Clip(qProj(hidden).reshaped(batch, length, heads, headDim), scale: queryClip.scale, threshold: config.qkClip)
        var relative = Inkling2Ops.l2Clip(rProj(hidden).reshaped(batch, length, heads, config.relativeDim), scale: relativeClip.scale, threshold: config.relativeClip)
        if global {
            let positions = (MLXArray(Int32(start) ..< Int32(start + length)).asType(.float32) + 1) / Float(config.logScalingFloor)
            let tau = (1 + config.logScalingAlpha * log(maximum(positions, 1))).reshaped(1, length, 1, 1)
            queries = (queries.asType(.float32) * tau).asType(queries.dtype)
            relative = (relative.asType(.float32) * tau).asType(relative.dtype)
        }
        let bias = matmul(relative, relativeLogits.proj.asType(relative.dtype))
        let k = Inkling2Ops.l2Clip(keys.reshaped(batch, length, kvHeads, headDim), scale: nil, threshold: config.qkClip).transposed(0, 2, 1, 3)
        let v = Inkling2Ops.unitRMS(values.reshaped(batch, length, kvHeads, headDim), eps: config.rmsNormEps).asType(values.dtype).transposed(0, 2, 1, 3)
        var (allKeys, allValues) = (k, v)
        var keyStart = 0
        if let window = cache as? Inkling2WindowKVCache {
            (allKeys, allValues) = window.update(keys: k, values: v)
            keyStart = window.lastKeyStart
        } else if let cache {
            (allKeys, allValues) = cache.update(keys: k, values: v)
        } else {
            keyStart = start
        }
        let out = attend(queries.transposed(0, 2, 1, 3), allKeys, allValues, bias: bias, sinks: sinkProj(hidden), positions: Inkling2Positions(query: start, key: keyStart))
        let gated = out.reshaped(batch, length, -1).asType(.float32) * sigmoid(gateProj(hidden).asType(.float32))
        return oProj(gated.asType(hidden.dtype))
    }

    func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, bias: MLXArray, sinks: MLXArray, positions: Inkling2Positions) -> MLXArray {
        let (queryStart, keyStart) = (positions.query, positions.key)
        let (batch, length, keysCount) = (q.dim(0), q.dim(2), k.dim(2))
        let extent = bias.dim(-1)
        let queryPositions = MLXArray(Int32(queryStart) ..< Int32(queryStart + length))
        let keyPositions = MLXArray(Int32(keyStart) ..< Int32(keyStart + keysCount))
        let distance = queryPositions.expandedDimensions(axis: 1) - keyPositions.expandedDimensions(axis: 0)
        var visible = distance .>= 0
        if window > 0 { visible = visible .&& (distance .< Int32(window)) }
        let index = broadcast(clip(distance, min: 0, max: extent - 1), to: [batch, heads, length, keysCount])
        let gathered = takeAlong(bias.transposed(0, 2, 1, 3), index, axis: -1).asType(.float32)
        let positional = MLX.where(distance .< Int32(extent), gathered, MLXArray(Float(0)))
        let mask = MLX.where(visible, positional, MLXArray(-Float.infinity)).reshaped(batch, kvHeads, -1, length, keysCount)
        let grouped = q.asType(.float32).reshaped(batch, kvHeads, -1, length, headDim)
        let scores = matmul(grouped, k.asType(.float32).expandedDimensions(axis: 2).swappedAxes(-1, -2)) * scale + mask
        let sinkLogits = sinks.asType(.float32).transposed(0, 2, 1).reshaped(batch, kvHeads, -1, length, 1)
        let weights = softmax(concatenated([scores, sinkLogits], axis: -1), axis: -1, precise: true)[.ellipsis, ..<keysCount]
        let out = matmul(weights.asType(v.dtype), v.expandedDimensions(axis: 2))
        return out.reshaped(batch, heads, length, headDim).transposed(0, 2, 1, 3)
    }
}
