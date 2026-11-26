import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

private struct TTTHeadSignals {
    let q: [[Float]]
    let k: [[Float]]
    let v: [[Float]]
    let eta: [Float]
}

private struct TTTLinearRefState {
    var w1: [Float]
    var b1: [Float]
}

private struct TTTMLPRefState {
    var w1: [Float]
    var b1: [Float]
    var w2: [Float]
    var b2: [Float]
}

private struct TTTReference {
    let weights: [String: [Float]]
    let config: TTTConfiguration

    init(_ model: TTTModel) {
        self.weights = FastWeightCPUMath.flatWeights(model)
        self.config = model.configuration
    }

    func forward(_ tokens: [Int32]) throws -> (logits: [[Float]], finalW1: [[[Float]]]) {
        var h = FastWeightCPUMath.embedding(tokens, weight: try w("model.embed_tokens.weight"), hiddenSize: config.hiddenSize)
        var finalW1: [[[Float]]] = []
        for layer in 0 ..< config.numHiddenLayers {
            let (nextH, layerW1) = try block(h, layer: layer)
            h = nextH
            finalW1.append(layerW1)
        }
        let normed = FastWeightCPUMath.rmsNorm(h, weight: try w("model.norm.weight"), eps: config.rmsNormEps)
        let head = try weights["lm_head.weight"] ?? w("model.embed_tokens.weight")
        return (FastWeightCPUMath.linear(normed, weight: head, outDim: config.vocabSize), finalW1)
    }

    private func block(_ input: [[Float]], layer: Int) throws -> ([[Float]], [[Float]]) {
        let p = "model.layers.\(layer)."
        let seqIn = FastWeightCPUMath.rmsNorm(input, weight: try w(p + "seq_norm.weight"), eps: config.rmsNormEps)
        let (mixed, layerW1) = try sequenceMixer(seqIn, prefix: p + "seq_modeling_block.")
        let mid = FastWeightCPUMath.add(input, mixed)
        let ffnIn = FastWeightCPUMath.rmsNorm(mid, weight: try w(p + "ffn_norm.weight"), eps: config.rmsNormEps)
        let ffnOut = FastWeightCPUMath.swiGLU(
            ffnIn,
            gateW: try w(p + "mlp.gate_proj.weight"),
            upW: try w(p + "mlp.up_proj.weight"),
            downW: try w(p + "mlp.down_proj.weight"),
            interDim: config.intermediateSize
        )
        return (FastWeightCPUMath.add(mid, ffnOut), layerW1)
    }

    private func sequenceMixer(_ x: [[Float]], prefix p: String) throws -> ([[Float]], [[Float]]) {
        let d = config.hiddenSize
        var qRaw = FastWeightCPUMath.linear(x, weight: try w(p + "q_proj.weight"), outDim: d)
        var kRaw = config.shareQK ? qRaw : FastWeightCPUMath.linear(x, weight: try w(p + "k_proj.weight"), outDim: d)
        if config.convKernel > 1 {
            qRaw = FastWeightCPUMath.depthwiseConv1d(qRaw, weight: try w(p + "conv_q.conv.weight"), bias: weights[p + "conv_q.conv.bias"], kernel: config.convKernel, applySiLU: false)
            kRaw = FastWeightCPUMath.depthwiseConv1d(kRaw, weight: try w(p + "conv_k.conv.weight"), bias: weights[p + "conv_k.conv.bias"], kernel: config.convKernel, applySiLU: false)
        }
        let vRaw = FastWeightCPUMath.linear(x, weight: try w(p + "v_proj.weight"), outDim: d)
        let q = FastWeightCPUMath.applyRoPE(FastWeightCPUMath.splitHeads(qRaw, heads: config.numAttentionHeads, headDim: config.headDim), theta: config.ropeTheta)
        let k = FastWeightCPUMath.applyRoPE(FastWeightCPUMath.splitHeads(kRaw, heads: config.numAttentionHeads, headDim: config.headDim), theta: config.ropeTheta)
        let v = FastWeightCPUMath.splitHeads(vRaw, heads: config.numAttentionHeads, headDim: config.headDim)
        let lrRaw = FastWeightCPUMath.linear(x, weight: try w(p + "ttt_lr_gate.weight"), bias: weights[p + "ttt_lr_gate.bias"], outDim: config.numAttentionHeads)
        let lrScale = config.tttBaseLR / Float(config.headDim)
        let etas: [[Float]] = (0 ..< config.numAttentionHeads).map { h in
            lrRaw.map { FastWeightCPUMath.sigmoidScalar($0[h]) * lrScale }
        }
        var headOuts: [[[Float]]] = []
        var layerW1: [[Float]] = []
        for h in 0 ..< config.numAttentionHeads {
            let sig = TTTHeadSignals(q: q[h], k: k[h], v: v[h], eta: etas[h])
            let (outH, finalW) = try runHead(sig, prefix: p, head: h)
            headOuts.append(outH)
            layerW1.append(finalW)
        }
        let merged = FastWeightCPUMath.mergeHeads(headOuts)
        var normed = FastWeightCPUMath.layerNorm(merged, weight: try w(p + "post_norm.weight"), bias: try w(p + "post_norm.bias"), eps: config.rmsNormEps)
        if config.useGate {
            let g = FastWeightCPUMath.linear(x, weight: try w(p + "g_proj.weight"), outDim: d)
            normed = zip(g, normed).map { gr, nr in zip(gr, nr).map { FastWeightCPUMath.geluTanhScalar($0) * $1 } }
        }
        return (FastWeightCPUMath.linear(normed, weight: try w(p + "o_proj.weight"), outDim: d), layerW1)
    }

    private func runHead(_ sig: TTTHeadSignals, prefix p: String, head: Int) throws -> ([[Float]], [Float]) {
        let d = config.headDim
        let inner = config.variant == .linear ? d : d * config.mlpExpansion
        let w1All = try w(p + "w1_init")
        let b1All = try w(p + "b1_init")
        let nwAll = try w(p + "ttt_norm_weight")
        let nbAll = try w(p + "ttt_norm_bias")
        let nw = Array(nwAll[(head * d) ..< ((head + 1) * d)])
        let nb = Array(nbAll[(head * d) ..< ((head + 1) * d)])
        let w1 = Array(w1All[(head * d * inner) ..< ((head + 1) * d * inner)])
        let b1 = Array(b1All[(head * inner) ..< ((head + 1) * inner)])
        if config.variant == .linear {
            var state = TTTLinearRefState(w1: w1, b1: b1)
            let out = runLinearHead(sig, norm: (nw, nb), state: &state)
            return (out, state.w1)
        }
        let w2All = try w(p + "w2_init")
        let b2All = try w(p + "b2_init")
        var state = TTTMLPRefState(
            w1: w1,
            b1: b1,
            w2: Array(w2All[(head * inner * d) ..< ((head + 1) * inner * d)]),
            b2: Array(b2All[(head * d) ..< ((head + 1) * d)])
        )
        let out = runMLPHead(sig, norm: (nw, nb), state: &state)
        return (out, state.w1)
    }

    private func runLinearHead(
        _ sig: TTTHeadSignals,
        norm: (w: [Float], b: [Float]),
        state: inout TTTLinearRefState
    ) -> [[Float]] {
        let d = config.headDim
        let mb = config.miniBatchSize
        var out: [[Float]] = []
        var accW = Array(repeating: Float(0), count: d * d)
        var accB = Array(repeating: Float(0), count: d)
        for t in 0 ..< sig.q.count {
            let stepIn = t % mb
            let z1 = matVec(sig.k[t], state.w1, state.b1, rows: d, cols: d)
            let grad = FastWeightCPUMath.lnFusedL2BackwardVec(
                x: z1, target: zip(sig.v[t], sig.k[t]).map(-), weight: norm.w, bias: norm.b, eps: config.tttNormEps
            )
            accumulateOuter(&accW, &accB, x: sig.k[t], g: grad, scale: sig.eta[t])
            let tau = Float(1.0) / Float(stepIn + 1)
            let wt = zip(state.w1, accW).map { $0 - tau * $1 }
            let bt = zip(state.b1, accB).map { $0 - tau * $1 }
            let zBar = matVec(sig.q[t], wt, bt, rows: d, cols: d)
            let y = FastWeightCPUMath.layerNormVec(zBar, weight: norm.w, bias: norm.b, eps: config.tttNormEps)
            out.append(zip(sig.q[t], y).map(+))
            if stepIn + 1 == mb {
                (state.w1, state.b1) = (wt, bt)
                (accW, accB) = (Array(repeating: Float(0), count: d * d), Array(repeating: Float(0), count: d))
            }
        }
        return out
    }

    private func runMLPHead(
        _ sig: TTTHeadSignals,
        norm: (w: [Float], b: [Float]),
        state: inout TTTMLPRefState
    ) -> [[Float]] {
        let d = config.headDim
        let m = d * config.mlpExpansion
        let mb = config.miniBatchSize
        var out: [[Float]] = []
        var acc = TTTMLPRefState(
            w1: Array(repeating: 0, count: d * m), b1: Array(repeating: 0, count: m),
            w2: Array(repeating: 0, count: m * d), b2: Array(repeating: 0, count: d)
        )
        for t in 0 ..< sig.q.count {
            let stepIn = t % mb
            let z1 = matVec(sig.k[t], state.w1, state.b1, rows: d, cols: m)
            let x2 = z1.map(FastWeightCPUMath.geluTanhScalar)
            let z2 = matVec(x2, state.w2, state.b2, rows: m, cols: d)
            let g2 = FastWeightCPUMath.lnFusedL2BackwardVec(
                x: z2, target: zip(sig.v[t], sig.k[t]).map(-), weight: norm.w, bias: norm.b, eps: config.tttNormEps
            )
            let g1 = backpropGelu(z1: z1, g2: g2, w2: state.w2, outDim: d)
            accumulateOuter(&acc.w1, &acc.b1, x: sig.k[t], g: g1, scale: sig.eta[t])
            accumulateOuter(&acc.w2, &acc.b2, x: x2, g: g2, scale: sig.eta[t])
            let tau = Float(1.0) / Float(stepIn + 1)
            let cur = TTTMLPRefState(
                w1: zip(state.w1, acc.w1).map { $0 - tau * $1 }, b1: zip(state.b1, acc.b1).map { $0 - tau * $1 },
                w2: zip(state.w2, acc.w2).map { $0 - tau * $1 }, b2: zip(state.b2, acc.b2).map { $0 - tau * $1 }
            )
            let x2Bar = matVec(sig.q[t], cur.w1, cur.b1, rows: d, cols: m).map(FastWeightCPUMath.geluTanhScalar)
            let z2Bar = matVec(x2Bar, cur.w2, cur.b2, rows: m, cols: d)
            let y = FastWeightCPUMath.layerNormVec(z2Bar, weight: norm.w, bias: norm.b, eps: config.tttNormEps)
            out.append(zip(sig.q[t], y).map(+))
            if stepIn + 1 == mb {
                state = cur
                acc = TTTMLPRefState(
                    w1: Array(repeating: 0, count: d * m), b1: Array(repeating: 0, count: m),
                    w2: Array(repeating: 0, count: m * d), b2: Array(repeating: 0, count: d)
                )
            }
        }
        return out
    }

    private func accumulateOuter(_ accW: inout [Float], _ accB: inout [Float], x: [Float], g: [Float], scale: Float) {
        let cols = g.count
        for r in 0 ..< x.count {
            let xr = scale * x[r]
            for c in 0 ..< cols { accW[r * cols + c] += xr * g[c] }
        }
        for c in 0 ..< cols { accB[c] += scale * g[c] }
    }

    private func backpropGelu(z1: [Float], g2: [Float], w2: [Float], outDim: Int) -> [Float] {
        (0 ..< z1.count).map { j in
            var back: Float = 0
            for c in 0 ..< outDim { back += g2[c] * w2[j * outDim + c] }
            return back * FastWeightCPUMath.geluTanhDerivScalar(z1[j])
        }
    }

    private func matVec(_ x: [Float], _ w: [Float], _ b: [Float], rows: Int, cols: Int) -> [Float] {
        (0 ..< cols).map { c in
            var acc = b[c]
            for r in 0 ..< rows { acc += x[r] * w[r * cols + c] }
            return acc
        }
    }

    private func w(_ key: String) throws -> [Float] {
        try #require(weights[key], "Missing TTT reference weight \(key)")
    }
}

@Suite("TTT-Linear and TTT-MLP fast-weight sequence models, parity, streaming invariance, and test-time adaptation (#848)", .serialized)
struct TTTModelTests {
    static let tokens: [Int32] = [5, 19, 42, 7, 88, 13, 64, 21, 33, 11]

    @Test("TTT-Linear and TTT-MLP match the independent explicit-weight scalar CPU reference within 1e-4 in logits, final fast weights, and exact argmax ids (#228)",
          arguments: [TinyModel.Family.tttLinear, TinyModel.Family.tttMlp])
    func matchesIndependentCPUReference(family: TinyModel.Family) throws {
        let model = try #require(try TinyModel.make(family) as? TTTModel)
        let cache = model.newCache(parameters: nil)
        let logits = model(MLXArray(Self.tokens)[.newAxis], cache: cache).asType(.float32)
        let reference = try TTTReference(model).forward(Self.tokens)
        #expect(FastWeightCPUMath.maxAbsDiff(logits, reference.logits) < 1e-4)
        #expect(argMax(logits[0], axis: -1).asArray(Int32.self) == FastWeightCPUMath.argMaxRows(reference.logits))
        for (layerIdx, layerCache) in cache.enumerated() {
            let fw = try #require(layerCache as? FastWeightLayerCache)
            let w1 = try #require(fw.slot(0)).asType(.float32).asArray(Float.self)
            let refW1 = reference.finalW1[layerIdx].flatMap { $0 }
            let diff = zip(w1, refW1).reduce(Float(0)) { Swift.max($0, abs($1.0 - $1.1)) }
            #expect(diff < 1e-4)
        }
    }

    @Test("Multi-chunk prefill (5 + 3 + 2) and token-by-token streaming match single-pass prefill within 1e-4 for TTT-Linear and TTT-MLP",
          arguments: [TinyModel.Family.tttLinear, TinyModel.Family.tttMlp])
    func chunkedAndStreamingMatchSinglePass(family: TinyModel.Family) throws {
        let model = try #require(try TinyModel.make(family) as? TTTModel)
        let singleCache = model.newCache(parameters: nil)
        let single = model(MLXArray(Self.tokens)[.newAxis], cache: singleCache).asType(.float32)

        let chunkCache = model.newCache(parameters: nil)
        let c1 = model(MLXArray(Array(Self.tokens[0 ..< 5]))[.newAxis], cache: chunkCache).asType(.float32)
        let c2 = model(MLXArray(Array(Self.tokens[5 ..< 8]))[.newAxis], cache: chunkCache).asType(.float32)
        let c3 = model(MLXArray(Array(Self.tokens[8 ..< 10]))[.newAxis], cache: chunkCache).asType(.float32)
        let chunked = concatenated([c1, c2, c3], axis: 1)
        #expect(MLX.abs(chunked - single).max().item(Float.self) < 1e-4)

        let streamCache = model.newCache(parameters: nil)
        let stepped = concatenated(Self.tokens.map { model(MLXArray([$0])[.newAxis], cache: streamCache).asType(.float32) }, axis: 1)
        #expect(MLX.abs(stepped - single).max().item(Float.self) < 1e-4)
        #expect(argMax(stepped, axis: -1).asArray(Int32.self) == argMax(single, axis: -1).asArray(Int32.self))

        for (s, st) in zip(singleCache, streamCache) {
            let (a, b) = (try #require(s as? FastWeightLayerCache), try #require(st as? FastWeightLayerCache))
            #expect(a.offset == Self.tokens.count && b.offset == Self.tokens.count)
            #expect(a.stepInChunk == 2 && b.stepInChunk == 2)
            for (ta, tb) in zip(a.innerState(), b.innerState()) {
                #expect(MLX.abs(ta - tb).max().item(Float.self) < 1e-4)
            }
        }
    }

    @Test("Inner-loop TTT reconstruction loss decreases monotonically across repeated mini-batches for both Linear and MLP",
          arguments: [TTTConfiguration.Variant.linear, TTTConfiguration.Variant.mlp])
    func innerLoopLossDecreasesMonotonically(variant: TTTConfiguration.Variant) throws {
        let json = """
            {"model_type": "ttt", "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
             "num_hidden_layers": 1, "num_attention_heads": 2, "head_dim": 8, "ttt_layer_type": "\(variant.rawValue)",
             "mini_batch_size": 4, "ttt_base_lr": 0.25, "conv_kernel": 0, "ttt_mlp_expansion": 2, "ttt_norm_eps": 1e-2}
            """
        let model = try withRandomState(MLXRandom.RandomState(seed: 8485)) { try TTTModel(configurationData: Data(json.utf8)) }
        let mixer = model.model.layers[0].mixer
        let (k, v) = withRandomState(MLXRandom.RandomState(seed: 8486)) {
            (MLXRandom.normal([1, 2, 4, 8]) * Float(0.5), MLXRandom.normal([1, 2, 4, 8]) * Float(0.5))
        }
        var state = TTTState.load(from: nil, batch: 1, mixer: mixer)
        let norm = TTTNormParams(weight: mixer.tttNormWeight, bias: mixer.tttNormBias, eps: mixer.config.tttNormEps)
        let slice = FastWeightChunkSlice(range: 0 ..< 4, stepStart: 0, stepEnd: 4, completesChunk: true, chunkSize: 4)
        let stepLR: Float = 1e-3
        let eta = MLXArray.ones([1, 2, 4, 1], dtype: .float32) * stepLR
        let tau = MLXArray([Float(1.0), 0.5, 1.0 / 3.0, 0.25]).reshaped(1, 1, 4, 1)
        let inputs = TTTSliceInputs(q: k, k: k, v: v, eta: eta, tau: tau)
        var losses: [Float] = []
        for _ in 0 ..< 5 {
            losses.append(Self.innerLoss(k: k, v: v, state: state, norm: norm, variant: variant))
            _ = variant == .linear
                ? TTTDualStep.stepLinear(inputs, norm: norm, slice: slice, state: &state)
                : TTTDualStep.stepMLP(inputs, norm: norm, slice: slice, state: &state)
        }
        for idx in 1 ..< losses.count {
            #expect(losses[idx] < losses[idx - 1])
        }
    }

    @Test("Test-time adaptation yields strictly lower next-token NLL on repeated context than frozen fast weights (lr = 0) for TTT-Linear and TTT-MLP",
          arguments: [TTTConfiguration.Variant.linear, TTTConfiguration.Variant.mlp])
    func adaptationLowersNextTokenNLLOnRepeatedContext(variant: TTTConfiguration.Variant) throws {
        let active = try Self.makeAssociativeTTT(variant: variant, baseLR: variant == .mlp ? 8.0 : 4.0)
        let frozen = try Self.makeAssociativeTTT(variant: variant, baseLR: 0.0)
        let seq: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8]
        let repeated = seq + seq
        let activeLogits = active(MLXArray(repeated)[.newAxis], cache: active.newCache(parameters: nil))
        let frozenLogits = frozen(MLXArray(repeated)[.newAxis], cache: frozen.newCache(parameters: nil))
        let activeNLL = FastWeightCPUMath.nextTokenNLL(logits: activeLogits, tokens: repeated, positions: 8 ..< 15)
        let frozenNLL = FastWeightCPUMath.nextTokenNLL(logits: frozenLogits, tokens: repeated, positions: 8 ..< 15)
        #expect(activeNLL < frozenNLL - 0.1)
    }

    private static func innerLoss(
        k: MLXArray,
        v: MLXArray,
        state: TTTState,
        norm: TTTNormParams,
        variant: TTTConfiguration.Variant
    ) -> Float {
        let z1 = matmul(k, state.w1) + state.b1
        let preNorm: MLXArray = if variant == .mlp, let w2 = state.w2, let b2 = state.b2 {
            matmul(FastWeightMath.geluTanh(z1), w2) + b2
        } else {
            z1
        }
        let pred = FastWeightMath.layerNormForward(preNorm, weight: norm.weight, bias: norm.bias, eps: norm.eps)
        let diff = pred - (v - k)
        return (Float(0.5) * (diff * diff).sum()).item(Float.self)
    }

    private static func makeAssociativeTTT(variant: TTTConfiguration.Variant, baseLR: Float) throws -> TTTModel {
        let json = """
            {"model_type": "ttt", "vocab_size": 16, "hidden_size": 32, "intermediate_size": 32,
             "num_hidden_layers": 1, "num_attention_heads": 1, "head_dim": 32, "ttt_layer_type": "\(variant.rawValue)",
             "mini_batch_size": 8, "ttt_base_lr": \(baseLR), "conv_kernel": 2, "use_gate": false,
             "ttt_mlp_expansion": 1, "rms_norm_eps": 1e-6, "ttt_norm_eps": 1e-2, "rope_theta": 100000000.0,
             "tie_word_embeddings": false}
            """
        let model = try TTTModel(configurationData: Data(json.utf8))
        var embed = Array(repeating: Float(0), count: 16 * 32)
        for v in 0 ..< 8 { embed[(v + 1) * 32 + 8 + v] = 1.0 }
        let embedArr = MLXArray(embed).reshaped(16, 32)
        let eye32 = MLXArray.eye(32, dtype: .float32)
        let convIdentity = MLXArray(Array(repeating: [Float(0), Float(1)], count: 32).flatMap { $0 }).reshaped(32, 2, 1)
        let convLag = MLXArray(Array(repeating: [Float(1), Float(0)], count: 32).flatMap { $0 }).reshaped(32, 2, 1)
        var updates: [(String, MLXArray)] = [
            ("model.embed_tokens.weight", embedArr),
            ("lm_head.weight", embedArr * Float(4.0)),
            ("model.norm.weight", MLXArray.ones([32], dtype: .float32)),
            ("model.layers.0.seq_norm.weight", MLXArray.ones([32], dtype: .float32)),
            ("model.layers.0.ffn_norm.weight", MLXArray.ones([32], dtype: .float32)),
            ("model.layers.0.mlp.down_proj.weight", MLXArray.zeros([32, 32], dtype: .float32)),
            ("model.layers.0.seq_modeling_block.q_proj.weight", eye32 * Float(0.25)),
            ("model.layers.0.seq_modeling_block.k_proj.weight", eye32 * Float(0.25)),
            ("model.layers.0.seq_modeling_block.v_proj.weight", eye32 * Float(0.5)),
            ("model.layers.0.seq_modeling_block.o_proj.weight", eye32),
            ("model.layers.0.seq_modeling_block.conv_q.conv.weight", convIdentity),
            ("model.layers.0.seq_modeling_block.conv_q.conv.bias", MLXArray.zeros([32], dtype: .float32)),
            ("model.layers.0.seq_modeling_block.conv_k.conv.weight", convLag),
            ("model.layers.0.seq_modeling_block.conv_k.conv.bias", MLXArray.zeros([32], dtype: .float32)),
            ("model.layers.0.seq_modeling_block.ttt_lr_gate.weight", MLXArray.zeros([1, 32], dtype: .float32)),
            ("model.layers.0.seq_modeling_block.ttt_lr_gate.bias", MLXArray([Float(6.0)])),
            ("model.layers.0.seq_modeling_block.post_norm.weight", MLXArray.ones([32], dtype: .float32)),
            ("model.layers.0.seq_modeling_block.post_norm.bias", MLXArray.zeros([32], dtype: .float32)),
            ("model.layers.0.seq_modeling_block.ttt_norm_weight", MLXArray.ones([1, 1, 1, 32], dtype: .float32)),
            ("model.layers.0.seq_modeling_block.ttt_norm_bias", MLXArray.zeros([1, 1, 1, 32], dtype: .float32)),
            ("model.layers.0.seq_modeling_block.w1_init", eye32.reshaped(1, 1, 32, 32) * Float(variant == .mlp ? 0.5 : 0.0)),
            ("model.layers.0.seq_modeling_block.b1_init", MLXArray.zeros([1, 1, 1, 32], dtype: .float32)),
        ]
        if variant == .mlp {
            updates.append(("model.layers.0.seq_modeling_block.w2_init", eye32.reshaped(1, 1, 32, 32) * Float(0.25)))
            updates.append(("model.layers.0.seq_modeling_block.b2_init", MLXArray.zeros([1, 1, 1, 32], dtype: .float32)))
        }
        model.update(parameters: ModuleParameters.unflattened(updates))
        eval(model)
        return model
    }
}

extension TTTModelTests {
    @Test("Invalid TTT configs throw DecodingError cleanly", arguments: [
        #"{"vocab_size": 0, "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4}"#,
        #"{"vocab_size": 256, "hidden_size": 65, "num_hidden_layers": 2, "num_attention_heads": 4}"#,
        #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "mini_batch_size": 0}"#,
        #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "ttt_base_lr": -0.1}"#,
    ])
    func invalidTTTConfigRefused(_ json: String) {
        #expect(throws: DecodingError.self) {
            _ = try TTTModel(configurationData: Data(json.utf8))
        }
    }

    @Test("TTT sanitize maps ttt-lm-pytorch weights, drops lm_head when tied, and LLMTypeRegistry + ModelCapabilities register ttt")
    func sanitizeAndRegistryForTTT() async throws {
        let tied = try #require(try TinyModel.make(.tttLinear, adding: #""tie_word_embeddings": true, "#) as? TTTModel)
        #expect(tied.lmHead == nil)
        let raw: [String: MLXArray] = [
            "lm_head.weight": MLXArray.ones([256, 64]),
            "model.layers.0.input_layernorm.weight": MLXArray.ones([64]),
            "model.layers.0.post_attention_layernorm.weight": MLXArray.ones([64]),
            "model.layers.0.self_attn.learnable_ttt_lr_weight": MLXArray.ones([4, 1, 64]),
            "model.layers.0.self_attn.learnable_ttt_lr_bias": MLXArray.ones([4, 1, 1]),
            "model.layers.0.self_attn.W1": MLXArray.ones([4, 16, 16]),
            "model.layers.0.self_attn.b1": MLXArray.ones([4, 1, 16]),
            "model.layers.0.self_attn.ttt_norm_weight": MLXArray.ones([4, 1, 16]),
            "model.layers.0.self_attn.conv_q.weight": MLXArray.ones([64, 1, 4]),
            "model.layers.0.self_attn.token_idx": MLXArray.ones([4]),
        ]
        let clean = tied.sanitize(weights: raw)
        #expect(clean["lm_head.weight"] == nil)
        #expect(clean["model.layers.0.seq_norm.weight"] != nil)
        #expect(clean["model.layers.0.ffn_norm.weight"] != nil)
        #expect(clean["model.layers.0.seq_modeling_block.ttt_lr_gate.weight"]?.shape == [4, 64])
        #expect(clean["model.layers.0.seq_modeling_block.ttt_lr_gate.bias"]?.shape == [4])
        #expect(clean["model.layers.0.seq_modeling_block.w1_init"]?.shape == [1, 4, 16, 16])
        #expect(clean["model.layers.0.seq_modeling_block.b1_init"]?.shape == [1, 4, 1, 16])
        #expect(clean["model.layers.0.seq_modeling_block.ttt_norm_weight"]?.shape == [1, 4, 1, 16])
        #expect(clean["model.layers.0.seq_modeling_block.conv_q.conv.weight"]?.shape == [64, 4, 1])
        #expect(clean["model.layers.0.seq_modeling_block.token_idx"] == nil)
        #expect(Set(tied.sanitize(weights: clean).keys) == Set(clean.keys))

        await MLXInferenceEngine.registerAliases()
        #expect(await LLMTypeRegistry.shared.contains("ttt"))
        let created = try await LLMTypeRegistry.shared.createModel(configuration: Data(TinyModel.Family.tttLinear.config.utf8), modelType: "ttt")
        #expect(created is TTTModel)
        let caps = try #require(ModelCapabilities.of(modelType: "ttt"))
        #expect(caps.allows(.promptCacheReuse) && caps.allows(.grammarJumpForward) && !caps.allows(.kvQuantization))
    }
}
