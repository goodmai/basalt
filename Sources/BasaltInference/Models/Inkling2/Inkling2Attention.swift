import Foundation
import MLX
import MLXLMCommon
import MLXNN

private let compiledMask = compile { (bias: MLXArray, distance: MLXArray, limits: MLXArray) -> MLXArray in
    let (batch, heads, extent) = (bias.dim(0), bias.dim(2), bias.dim(-1))
    let (length, keys) = (distance.dim(0), distance.dim(1))
    let visible = (distance .>= 0) .&& (distance .< limits[0])
    let index = broadcast(clip(distance, min: 0, max: extent - 1), to: [batch, heads, length, keys])
    let gathered = takeAlong(bias.transposed(0, 2, 1, 3), index, axis: -1).asType(.float32)
    let positional = MLX.where(distance .< Int32(extent), gathered, MLXArray(Float(0)))
    return MLX.where(visible, positional, MLXArray(-Float.infinity))
}

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

struct Inkling2Prepared {
    var queries: MLXArray
    var keys: MLXArray
    var values: MLXArray
    var bias: MLXArray
    var sinks: MLXArray
    var gate: MLXArray
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
        let length = inputs.query.dim(1)
        let start = cache?.offset ?? 0
        let (prepared, keyWindow, valueWindow) = prepare(inputs, keyState: conv[0], valueState: conv[1], positions: Self.positions(start, length))
        (conv[0], conv[1]) = (keyWindow, valueWindow)
        let (allKeys, allValues, keyStart) = Self.store(prepared.keys, prepared.values, in: cache, start: start)
        return finish(prepared, keys: allKeys, values: allValues, distance: Self.distance(query: start, length: length, key: keyStart, count: allKeys.dim(2)))
    }

    static func positions(_ start: Int, _ length: Int) -> MLXArray {
        MLXArray(Int32(start) ..< Int32(start + length)).asType(.float32)
    }

    static func distance(query: Int, length: Int, key: Int, count: Int) -> MLXArray {
        MLXArray(Int32(query) ..< Int32(query + length)).expandedDimensions(axis: 1) - MLXArray(Int32(key) ..< Int32(key + count)).expandedDimensions(axis: 0)
    }

    static func store(_ k: MLXArray, _ v: MLXArray, in cache: KVCache?, start: Int) -> (MLXArray, MLXArray, Int) {
        if let window = cache as? Inkling2WindowKVCache {
            let (keys, values) = window.update(keys: k, values: v)
            return (keys, values, window.lastKeyStart)
        }
        if let cache {
            let (keys, values) = cache.update(keys: k, values: v)
            return (keys, values, 0)
        }
        return (k, v, start)
    }

    func prepare(_ inputs: Inkling2AttentionInputs, keyState: MLXArray?, valueState: MLXArray?, positions: MLXArray) -> (Inkling2Prepared, MLXArray, MLXArray) {
        let (batch, length) = (inputs.query.dim(0), inputs.query.dim(1))
        let hidden = inputs.query
        let (keys, keyWindow) = keyConv(kProj(inputs.key), input: inputs.key, state: keyState)
        let (values, valueWindow) = valueConv(vProj(inputs.value), input: inputs.value, state: valueState)
        var queries = Inkling2Ops.l2Clip(qProj(hidden).reshaped(batch, length, heads, headDim), scale: queryClip.scale, threshold: config.qkClip)
        var relative = Inkling2Ops.l2Clip(rProj(hidden).reshaped(batch, length, heads, config.relativeDim), scale: relativeClip.scale, threshold: config.relativeClip)
        if global {
            let tau = (1 + config.logScalingAlpha * log(maximum((positions + 1) / Float(config.logScalingFloor), 1))).reshaped(1, length, 1, 1)
            queries = (queries.asType(.float32) * tau).asType(queries.dtype)
            relative = (relative.asType(.float32) * tau).asType(relative.dtype)
        }
        let prepared = Inkling2Prepared(
            queries: queries.transposed(0, 2, 1, 3),
            keys: Inkling2Ops.l2Clip(keys.reshaped(batch, length, kvHeads, headDim), scale: nil, threshold: config.qkClip).transposed(0, 2, 1, 3),
            values: Inkling2Ops.unitRMS(values.reshaped(batch, length, kvHeads, headDim), eps: config.rmsNormEps).asType(values.dtype).transposed(0, 2, 1, 3),
            bias: matmul(relative, relativeLogits.proj.asType(relative.dtype)),
            sinks: sinkProj(hidden), gate: gateProj(hidden))
        return (prepared, keyWindow, valueWindow)
    }

    func finish(_ prepared: Inkling2Prepared, keys: MLXArray, values: MLXArray, distance: MLXArray) -> MLXArray {
        let (batch, length) = (prepared.queries.dim(0), prepared.queries.dim(2))
        let mask = compiledMask(prepared.bias, distance, MLXArray([window > 0 ? Int32(window) : Int32.max])).asType(prepared.queries.dtype)
        let out = sdpa(prepared.queries, keys, values, mask: mask, sinks: prepared.sinks)
        let gated = out.reshaped(batch, length, -1).asType(.float32) * sigmoid(prepared.gate.asType(.float32))
        return oProj(gated.asType(prepared.gate.dtype))
    }

    func sdpa(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, mask: MLXArray, sinks: MLXArray) -> MLXArray {
        if q.dim(2) == 1, q.dim(0) == 1 {
            return MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask, sinks: sinks[0, 0].asType(q.dtype)).transposed(0, 2, 1, 3)
        }
        let zero = MLXArray.zeros([k.dim(0), kvHeads, 1, headDim], dtype: k.dtype)
        let sinkColumn = sinks.transposed(0, 2, 1).expandedDimensions(axis: -1).asType(q.dtype)
        return MLXFast.scaledDotProductAttention(
            queries: q, keys: concatenated([k, zero], axis: 2), values: concatenated([v, zero.asType(v.dtype)], axis: 2),
            scale: scale, mask: concatenated([mask, sinkColumn], axis: -1)).transposed(0, 2, 1, 3)
    }

    func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, bias: MLXArray, sinks: MLXArray, positions: Inkling2Positions) -> MLXArray {
        let mask = additiveMask(bias: bias, length: q.dim(2), keysCount: k.dim(2), positions: positions).asType(q.dtype)
        return sdpa(q, k, v, mask: mask, sinks: sinks)
    }

    func additiveMask(bias: MLXArray, length: Int, keysCount: Int, positions: Inkling2Positions) -> MLXArray {
        let queryPositions = MLXArray(Int32(positions.query) ..< Int32(positions.query + length))
        let keyPositions = MLXArray(Int32(positions.key) ..< Int32(positions.key + keysCount))
        let distance = queryPositions.expandedDimensions(axis: 1) - keyPositions.expandedDimensions(axis: 0)
        return compiledMask(bias, distance, MLXArray([window > 0 ? Int32(window) : Int32.max]))
    }

    func attendReference(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, bias: MLXArray, sinks: MLXArray, positions: Inkling2Positions) -> MLXArray {
        let (batch, length, keysCount) = (q.dim(0), q.dim(2), k.dim(2))
        let mask = additiveMask(bias: bias, length: length, keysCount: keysCount, positions: positions).reshaped(batch, kvHeads, -1, length, keysCount)
        let grouped = q.asType(.float32).reshaped(batch, kvHeads, -1, length, headDim)
        let scores = matmul(grouped, k.asType(.float32).expandedDimensions(axis: 2).swappedAxes(-1, -2)) * scale + mask
        let sinkLogits = sinks.asType(.float32).transposed(0, 2, 1).reshaped(batch, kvHeads, -1, length, 1)
        let weights = softmax(concatenated([scores, sinkLogits], axis: -1), axis: -1, precise: true)[.ellipsis, ..<keysCount]
        let out = matmul(weights.asType(v.dtype), v.expandedDimensions(axis: 2))
        return out.reshaped(batch, heads, length, headDim).transposed(0, 2, 1, 3)
    }
}
