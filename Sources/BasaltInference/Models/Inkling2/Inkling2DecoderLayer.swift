import Foundation
import MLX
import MLXLMCommon
import MLXNN

private let compiledResidual = compile(shapeless: true) { (residual: MLXArray, update: MLXArray) -> MLXArray in
    (residual + update.asType(.float32)).asType(update.dtype)
}

final class Inkling2DecoderLayer: Module {
    let index: Int
    let global: Bool
    let blockBoundary: Bool
    let kernel: Int
    let steps = Inkling2CompiledSteps()

    @ModuleInfo(key: "attention_input") var attentionInput: Inkling2Query?
    @ModuleInfo(key: "mlp_input") var mlpInput: Inkling2Query
    @ModuleInfo(key: "self_attn") var attention: Inkling2Attention
    @ModuleInfo(key: "mlp") var mlp: Inkling2SparseMoE
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "post_attention_convolution") var postAttentionConv: Inkling2ShortConvolution
    @ModuleInfo(key: "post_mlp_convolution") var postMLPConv: Inkling2ShortConvolution
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm?
    @ModuleInfo(key: "q_input_layernorm") var queryNorm: RMSNorm?
    @ModuleInfo(key: "k_input_layernorm") var keyNorm: RMSNorm?
    @ModuleInfo(key: "v_input_layernorm") var valueNorm: RMSNorm?
    @ModuleInfo(key: "q_message_gate") var queryGate: Inkling2MessageGate?
    @ModuleInfo(key: "k_message_gate") var keyGate: Inkling2MessageGate?
    @ModuleInfo(key: "v_message_gate") var valueGate: Inkling2MessageGate?
    @ModuleInfo(key: "mlp_message_gate") var mlpGate: Inkling2MessageGate?

    init(_ config: Inkling2Configuration, index: Int) {
        self.index = index
        global = !config.localLayers.contains(index)
        blockBoundary = index.isMultiple(of: config.layersPerResidualBlock)
        kernel = config.shortConvKernel
        let dims = config.hiddenSize
        if index > 0 { _attentionInput.wrappedValue = Inkling2Query(rows: global ? 3 : 1, dims: dims) }
        _mlpInput.wrappedValue = Inkling2Query(rows: 1, dims: dims)
        _attention.wrappedValue = Inkling2Attention(config, global: global)
        _mlp.wrappedValue = Inkling2SparseMoE(config)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: dims, eps: config.rmsNormEps)
        _postAttentionConv.wrappedValue = Inkling2ShortConvolution(dims: dims, kernel: kernel)
        _postMLPConv.wrappedValue = Inkling2ShortConvolution(dims: dims, kernel: kernel)
        if global {
            _queryNorm.wrappedValue = RMSNorm(dimensions: dims, eps: config.rmsNormEps)
            _keyNorm.wrappedValue = RMSNorm(dimensions: dims, eps: config.rmsNormEps)
            _valueNorm.wrappedValue = RMSNorm(dimensions: dims, eps: config.rmsNormEps)
        } else {
            _inputNorm.wrappedValue = RMSNorm(dimensions: dims, eps: config.rmsNormEps)
        }
        if let loop = config.loop, (loop.coreStart ..< loop.coreEnd).contains(index) {
            _queryGate.wrappedValue = Inkling2MessageGate(dims: dims, rank: loop.gateRank)
            _keyGate.wrappedValue = Inkling2MessageGate(dims: dims, rank: loop.gateRank)
            _valueGate.wrappedValue = Inkling2MessageGate(dims: dims, rank: loop.gateRank)
            _mlpGate.wrappedValue = Inkling2MessageGate(dims: dims, rank: loop.gateRank)
        }
    }

    func callAsFunction(
        _ hidden: MLXArray, blocks: MLXArray?, cache: KVCache?, conv: ArraysCache?, messages: MLXArray?, probe: (any ForwardProbe)? = nil
    ) -> (MLXArray, MLXArray?) {
        if probe == nil, (1 ... Self.compiledRows).contains(hidden.dim(1)), !training,
           let out = compiledDecode(hidden, blocks: blocks, cache: cache, conv: conv, messages: messages) { return out }
        return eager(hidden, blocks: blocks, cache: cache, conv: conv, messages: messages, probe: probe)
    }

    static let compiledRows = 8

    func eager(
        _ hidden: MLXArray, blocks: MLXArray?, cache: KVCache?, conv: ArraysCache?, messages: MLXArray?, probe: (any ForwardProbe)? = nil
    ) -> (MLXArray, MLXArray?) {
        let dtype = hidden.dtype
        var states: [MLXArray?] = (0 ..< 4).map { conv?[$0] }
        let normedResidual = Inkling2Ops.unitRMS(hidden)
        let inputs = attentionInputs(hidden, mixed: attentionInput.map { $0(blocks!, normedResidual) }, messages: messages, dtype: dtype)
        let attended = attention(inputs, cache: cache, conv: &states)
        let (afterAttention, attentionWindow) = postAttentionConv(attended, state: states[2])
        states[2] = attentionWindow
        var blocks = blocks
        var residual = afterAttention.asType(.float32)
        if blockBoundary {
            let block = normedResidual.expandedDimensions(axis: 0)
            blocks = blocks.map { concatenated([$0, block], axis: 0) } ?? block
        } else {
            residual = hidden.asType(.float32) + residual
        }
        var x = mlpInput(blocks!, Inkling2Ops.unitRMS(residual))[.ellipsis, 0, 0...]
        if let mlpGate, let messages { x = mlpGate(x, message: messages[.ellipsis, 3, 0...]) }
        let (afterMLP, mlpWindow) = postMLPConv(mlp(postAttentionNorm(x).asType(dtype), probe: probe, layer: index), state: states[3])
        states[3] = mlpWindow
        if let conv {
            let windows = states.compactMap { $0 }
            Inkling2Ops.commit(windows.map { Inkling2Ops.convState($0, kernel: kernel) }, windows: windows, count: hidden.dim(1), to: conv)
        }
        return (compiledResidual(residual, afterMLP), blocks)
    }

    func attentionInputs(_ hidden: MLXArray, mixed: MLXArray?, messages: MLXArray?, dtype: DType) -> Inkling2AttentionInputs {
        var (q, k, v) = global
            ? (mixed?[.ellipsis, 0, 0...] ?? hidden, mixed?[.ellipsis, 1, 0...] ?? hidden, mixed?[.ellipsis, 2, 0...] ?? hidden)
            : (mixed?[.ellipsis, 0, 0...] ?? hidden, mixed?[.ellipsis, 0, 0...] ?? hidden, mixed?[.ellipsis, 0, 0...] ?? hidden)
        if let messages, let queryGate, let keyGate, let valueGate {
            q = queryGate(q, message: messages[.ellipsis, 0, 0...])
            k = keyGate(k, message: messages[.ellipsis, 1, 0...])
            v = valueGate(v, message: messages[.ellipsis, 2, 0...])
        }
        if let queryNorm, let keyNorm, let valueNorm {
            return Inkling2AttentionInputs(query: queryNorm(q).asType(dtype), key: keyNorm(k).asType(dtype), value: valueNorm(v).asType(dtype))
        }
        let norm = inputNorm!
        return Inkling2AttentionInputs(query: norm(q).asType(dtype), key: norm(k).asType(dtype), value: norm(v).asType(dtype))
    }
}
