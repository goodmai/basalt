import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

private struct LaCTHeadLRs {
    let lr1: [Float]
    let lr2: [Float]
    let lr3: [Float]
}

private struct LaCTRefHeadState {
    var w1: [Float]
    var w2: [Float]
    var w3: [Float]
    var g1: [Float]
    var g2: [Float]
    var g3: [Float]
    var m1: [Float]
    var m2: [Float]
    var m3: [Float]
    let n1: [Float]
    let n2: [Float]
    let n3: [Float]
}

private struct LaCTReference {
    let weights: [String: [Float]]
    let config: LaCTConfiguration

    init(_ model: LaCTModel) {
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

    private func block(_ x: [[Float]], layer: Int) throws -> ([[Float]], [[Float]]) {
        let p = "model.layers.\(layer)."
        let attnIn = FastWeightCPUMath.rmsNorm(x, weight: try w(p + "input_layernorm.weight"), eps: config.rmsNormEps)
        let afterAttn = FastWeightCPUMath.add(x, try attention(attnIn, prefix: p + "self_attn."))
        let fwIn = FastWeightCPUMath.rmsNorm(afterAttn, weight: try w(p + "fw_layernorm.weight"), eps: config.rmsNormEps)
        let (fwOut, layerW1) = try runFastWeight(fwIn, prefix: p + "fast_weight.")
        let afterFW = FastWeightCPUMath.add(afterAttn, fwOut)
        let ffnIn = FastWeightCPUMath.rmsNorm(afterFW, weight: try w(p + "post_attention_layernorm.weight"), eps: config.rmsNormEps)
        let ffnOut = FastWeightCPUMath.swiGLU(
            ffnIn,
            gateW: try w(p + "mlp.gate_proj.weight"),
            upW: try w(p + "mlp.up_proj.weight"),
            downW: try w(p + "mlp.down_proj.weight"),
            interDim: config.intermediateSize
        )
        return (FastWeightCPUMath.add(afterFW, ffnOut), layerW1)
    }

    private func attention(_ x: [[Float]], prefix p: String) throws -> [[Float]] {
        let d = config.hiddenSize
        let kvDim = config.numKeyValueHeads * config.headDim
        let q = FastWeightCPUMath.applyRoPE(
            FastWeightCPUMath.splitHeads(FastWeightCPUMath.linear(x, weight: try w(p + "q_proj.weight"), outDim: d), heads: config.numAttentionHeads, headDim: config.headDim),
            theta: config.ropeTheta
        )
        let k = FastWeightCPUMath.applyRoPE(
            FastWeightCPUMath.splitHeads(FastWeightCPUMath.linear(x, weight: try w(p + "k_proj.weight"), outDim: kvDim), heads: config.numKeyValueHeads, headDim: config.headDim),
            theta: config.ropeTheta
        )
        let v = FastWeightCPUMath.splitHeads(FastWeightCPUMath.linear(x, weight: try w(p + "v_proj.weight"), outDim: kvDim), heads: config.numKeyValueHeads, headDim: config.headDim)
        let empty = Array(repeating: [[Float]](), count: config.numKeyValueHeads)
        let input = CPUSlidingAttentionInput(q: q, k: k, v: v, kPersistent: empty, vPersistent: empty, windowSize: config.windowSize)
        return FastWeightCPUMath.linear(FastWeightCPUMath.slidingAttention(input), weight: try w(p + "o_proj.weight"), outDim: d)
    }

    private func runFastWeight(_ x: [[Float]], prefix p: String) throws -> ([[Float]], [[Float]]) {
        let hCount = config.numFastWeightHeads
        let d = config.fastWeightHeadDim
        let fwDim = hCount * d
        var qRaw = FastWeightCPUMath.linear(x, weight: try w(p + "q_proj.weight"), outDim: fwDim)
        var kRaw = FastWeightCPUMath.linear(x, weight: try w(p + "k_proj.weight"), outDim: fwDim)
        var vRaw = FastWeightCPUMath.linear(x, weight: try w(p + "v_proj.weight"), outDim: fwDim)
        if config.convKernel > 1 {
            qRaw = FastWeightCPUMath.depthwiseConv1d(qRaw, weight: try w(p + "q_conv.conv.weight"), bias: weights[p + "q_conv.conv.bias"], kernel: config.convKernel, applySiLU: false)
            kRaw = FastWeightCPUMath.depthwiseConv1d(kRaw, weight: try w(p + "k_conv.conv.weight"), bias: weights[p + "k_conv.conv.bias"], kernel: config.convKernel, applySiLU: false)
            vRaw = FastWeightCPUMath.depthwiseConv1d(vRaw, weight: try w(p + "v_conv.conv.weight"), bias: weights[p + "v_conv.conv.bias"], kernel: config.convKernel, applySiLU: false)
        }
        let qSplit = FastWeightCPUMath.splitHeads(qRaw, heads: hCount, headDim: d)
        let q = config.memoryMode == .swiglu ? qSplit.map { $0.map { FastWeightCPUMath.l2NormVec($0) } } : qSplit
        let k = FastWeightCPUMath.splitHeads(kRaw, heads: hCount, headDim: d).map { $0.map { FastWeightCPUMath.l2NormVec($0) } }
        let v = FastWeightCPUMath.splitHeads(vRaw, heads: hCount, headDim: d)
        let lrs = try projectLR(x, prefix: p)
        var headOuts: [[[Float]]] = []
        var layerW1: [[Float]] = []
        for h in 0 ..< hCount {
            var state = try initHeadState(head: h, prefix: p)
            let headLR = LaCTHeadLRs(lr1: lrs.lr1[h], lr2: lrs.lr2[h], lr3: lrs.lr3[h])
            let outH = runHead(q: q[h], k: k[h], v: v[h], lrs: headLR, state: &state)
            headOuts.append(outH)
            layerW1.append(state.w1)
        }
        let merged = FastWeightCPUMath.mergeHeads(headOuts)
        let normed = FastWeightCPUMath.rmsNorm(merged, weight: try w(p + "fw_out_norm.weight"), eps: config.rmsNormEps)
        return (FastWeightCPUMath.linear(normed, weight: try w(p + "o_proj.weight"), outDim: config.hiddenSize), layerW1)
    }

    private func projectLR(_ x: [[Float]], prefix p: String) throws -> (lr1: [[Float]], lr2: [[Float]], lr3: [[Float]]) {
        let hCount = config.numFastWeightHeads
        if config.memoryMode == .linear {
            let ones = Array(repeating: Array(repeating: Float(1), count: x.count), count: hCount)
            return (ones, ones, ones)
        }
        let raw = FastWeightCPUMath.linear(x, weight: try w(p + "lr_proj.weight"), bias: weights[p + "lr_proj.bias"], outDim: 3 * hCount)
        let lr1 = (0 ..< hCount).map { h in raw.map { FastWeightCPUMath.sigmoidScalar($0[h]) } }
        let lr2 = (0 ..< hCount).map { h in raw.map { FastWeightCPUMath.sigmoidScalar($0[hCount + h]) } }
        let lr3 = (0 ..< hCount).map { h in raw.map { FastWeightCPUMath.sigmoidScalar($0[2 * hCount + h]) } }
        return (lr1, lr2, lr3)
    }

    private func initHeadState(head: Int, prefix p: String) throws -> LaCTRefHeadState {
        let d = config.fastWeightHeadDim
        let m = config.fastWeightHiddenDim
        let w1All = try w(p + "w1_init")
        if config.memoryMode == .linear {
            let w1 = Array(w1All[(head * d * d) ..< ((head + 1) * d * d)])
            return LaCTRefHeadState(
                w1: w1, w2: [], w3: [],
                g1: Array(repeating: 0, count: d * d), g2: [], g3: [],
                m1: Array(repeating: 0, count: d * d), m2: [], m3: [],
                n1: Self.rowNorms(w1, rows: d, cols: d), n2: [], n3: []
            )
        }
        let w2All = try w(p + "w2_init")
        let w3All = try w(p + "w3_init")
        let w1 = Array(w1All[(head * m * d) ..< ((head + 1) * m * d)])
        let w2 = Array(w2All[(head * d * m) ..< ((head + 1) * d * m)])
        let w3 = Array(w3All[(head * m * d) ..< ((head + 1) * m * d)])
        return LaCTRefHeadState(
            w1: w1, w2: w2, w3: w3,
            g1: Array(repeating: 0, count: m * d), g2: Array(repeating: 0, count: d * m), g3: Array(repeating: 0, count: m * d),
            m1: Array(repeating: 0, count: m * d), m2: Array(repeating: 0, count: d * m), m3: Array(repeating: 0, count: m * d),
            n1: Self.rowNorms(w1, rows: m, cols: d), n2: Self.rowNorms(w2, rows: d, cols: m), n3: Self.rowNorms(w3, rows: m, cols: d)
        )
    }

    private func w(_ key: String) throws -> [Float] {
        try #require(weights[key], "Missing LaCT reference weight \(key)")
    }
}

extension LaCTReference {
    private func runHead(
        q: [[Float]],
        k: [[Float]],
        v: [[Float]],
        lrs: LaCTHeadLRs,
        state: inout LaCTRefHeadState
    ) -> [[Float]] {
        let d = config.fastWeightHeadDim
        let m = config.fastWeightHiddenDim
        var out: [[Float]] = []
        for t in 0 ..< q.count {
            if config.memoryMode == .linear {
                out.append(Self.matVecT(q[t], state.w1, rows: d, cols: d))
                let pred = Self.matVecT(k[t], state.w1, rows: d, cols: d)
                let delta = cotangent(pred: pred, target: v[t])
                Self.addOuter(&state.g1, a: delta, b: k[t], scale: lrs.lr1[t])
            } else {
                out.append(evalSwiGLU(q[t], state: state, d: d, m: m))
                accumulateSwiGLU(k: k[t], v: v[t], lrs: (lrs.lr1[t], lrs.lr2[t], lrs.lr3[t]), state: &state)
            }
            if (t % config.chunkSize) + 1 == config.chunkSize {
                flushHead(&state)
            }
        }
        return out
    }

    private func evalSwiGLU(_ x: [Float], state: LaCTRefHeadState, d: Int, m: Int) -> [Float] {
        let g = Self.matVecT(x, state.w1, rows: m, cols: d)
        let u = Self.matVecT(x, state.w3, rows: m, cols: d)
        let h = zip(g, u).map { FastWeightCPUMath.siluScalar($0) * $1 }
        return Self.matVecT(h, state.w2, rows: d, cols: m)
    }

    private func accumulateSwiGLU(k: [Float], v: [Float], lrs: (Float, Float, Float), state: inout LaCTRefHeadState) {
        let d = config.fastWeightHeadDim
        let m = config.fastWeightHiddenDim
        let g = Self.matVecT(k, state.w1, rows: m, cols: d)
        let u = Self.matVecT(k, state.w3, rows: m, cols: d)
        let s = g.map(FastWeightCPUMath.siluScalar)
        let h = zip(s, u).map(*)
        let pred = Self.matVecT(h, state.w2, rows: d, cols: m)
        let deltaV = cotangent(pred: pred, target: v)
        let deltaH = (0 ..< m).map { j -> Float in
            var acc: Float = 0
            for a in 0 ..< d { acc += deltaV[a] * state.w2[a * m + j] }
            return acc
        }
        let deltaU = zip(deltaH, s).map(*)
        let deltaG = (0 ..< m).map { j in deltaH[j] * u[j] * FastWeightCPUMath.siluDerivScalar(g[j]) }
        Self.addOuter(&state.g1, a: deltaG, b: k, scale: lrs.0)
        Self.addOuter(&state.g2, a: deltaV, b: h, scale: lrs.1)
        Self.addOuter(&state.g3, a: deltaU, b: k, scale: lrs.2)
    }

    private func flushHead(_ state: inout LaCTRefHeadState) {
        let d = config.fastWeightHeadDim
        let m = config.fastWeightHiddenDim
        if config.memoryMode == .linear {
            (state.w1, state.m1) = flushMatrix(state.w1, g: state.g1, m: state.m1, shape: (d, d), targetNorm: state.n1)
            state.g1 = Array(repeating: 0, count: d * d)
            return
        }
        (state.w1, state.m1) = flushMatrix(state.w1, g: state.g1, m: state.m1, shape: (m, d), targetNorm: state.n1)
        (state.w2, state.m2) = flushMatrix(state.w2, g: state.g2, m: state.m2, shape: (d, m), targetNorm: state.n2)
        (state.w3, state.m3) = flushMatrix(state.w3, g: state.g3, m: state.m3, shape: (m, d), targetNorm: state.n3)
        state.g1 = Array(repeating: 0, count: m * d)
        state.g2 = Array(repeating: 0, count: d * m)
        state.g3 = Array(repeating: 0, count: m * d)
    }

    private func flushMatrix(
        _ w: [Float],
        g: [Float],
        m: [Float],
        shape: (rows: Int, cols: Int),
        targetNorm: [Float]
    ) -> ([Float], [Float]) {
        guard config.baseLR > 0 else { return (w, m) }
        let avg = g.map { $0 / Float(config.chunkSize) }
        let nextM: [Float]
        let dir: [Float]
        switch config.updateRule {
        case .sgd:
            nextM = m
            dir = avg
        case .momentum:
            nextM = zip(m, avg).map { config.momentumBeta * $0 + $1 }
            dir = nextM
        case .muon:
            nextM = zip(m, avg).map { config.momentumBeta * $0 + $1 }
            dir = Self.cpuNewtonSchulz5(nextM, rows: shape.rows, cols: shape.cols, steps: config.muonSteps)
        }
        var updated = zip(w, dir).map { $0 - config.baseLR * $1 }
        if config.useWeightNorm {
            let curNorms = Self.rowNorms(updated, rows: shape.rows, cols: shape.cols)
            for r in 0 ..< shape.rows {
                let scale = targetNorm[r] / curNorms[r]
                for c in 0 ..< shape.cols { updated[r * shape.cols + c] *= scale }
            }
        }
        return (updated, nextM)
    }

    private func cotangent(pred: [Float], target: [Float]) -> [Float] {
        switch config.objective {
        case .dotProduct: target.map { -$0 }
        case .l2: zip(pred, target).map(-)
        }
    }
}

extension LaCTReference {
    private static func matVecT(_ x: [Float], _ w: [Float], rows: Int, cols: Int) -> [Float] {
        (0 ..< rows).map { r in
            var acc: Float = 0
            for c in 0 ..< cols { acc += x[c] * w[r * cols + c] }
            return acc
        }
    }

    private static func addOuter(_ acc: inout [Float], a: [Float], b: [Float], scale: Float) {
        let cols = b.count
        for r in 0 ..< a.count {
            let ar = scale * a[r]
            for c in 0 ..< cols { acc[r * cols + c] += ar * b[c] }
        }
    }

    private static func rowNorms(_ w: [Float], rows: Int, cols: Int) -> [Float] {
        (0 ..< rows).map { r in
            var sumSq: Float = 0
            for c in 0 ..< cols {
                let val = w[r * cols + c]
                sumSq += val * val
            }
            return sqrt(sumSq) + 1e-6
        }
    }

    private static func cpuNewtonSchulz5(_ g: [Float], rows: Int, cols: Int, steps: Int) -> [Float] {
        let transposed = rows > cols
        let r = transposed ? cols : rows
        let c = transposed ? rows : cols
        var x = transposed ? transpose(g, rows: rows, cols: cols) : g
        let frob = sqrt(x.reduce(Float(0)) { $0 + $1 * $1 }) + 1e-7
        x = x.map { $0 / frob }
        for _ in 0 ..< steps {
            let gram = matMul(x, transpose(x, rows: r, cols: c), m: r, k: c, n: r)
            let gram2 = matMul(gram, gram, m: r, k: r, n: r)
            let poly = zip(gram, gram2).map { Float(-4.7750) * $0 + Float(2.0315) * $1 }
            let bx = matMul(poly, x, m: r, k: r, n: c)
            x = zip(x, bx).map { Float(3.4445) * $0 + $1 }
        }
        return transposed ? transpose(x, rows: r, cols: c) : x
    }

    private static func transpose(_ a: [Float], rows: Int, cols: Int) -> [Float] {
        var out = Array(repeating: Float(0), count: rows * cols)
        for r in 0 ..< rows {
            for c in 0 ..< cols { out[c * rows + r] = a[r * cols + c] }
        }
        return out
    }

    private static func matMul(_ a: [Float], _ b: [Float], m: Int, k: Int, n: Int) -> [Float] {
        var out = Array(repeating: Float(0), count: m * n)
        for i in 0 ..< m {
            for j in 0 ..< n {
                var acc: Float = 0
                for p in 0 ..< k { acc += a[i * k + p] * b[p * n + j] }
                out[i * n + j] = acc
            }
        }
        return out
    }
}

@Suite("LaCT large-chunk fast-weight sequence model (SwiGLU + Muon NS-5 + WeightNorm, Listing B.8 Linear, streaming, adaptation) (#848)", .serialized)
struct LaCTModelTests {
    static let tokens: [Int32] = [5, 19, 42, 7, 88, 13, 64, 21, 33, 11]

    static func makeVariant(mode: String, objective: String, rule: String, weightNorm: Bool, conv: Int) throws -> LaCTModel {
        let json = """
            {"model_type": "lact_swiglu", "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128,
             "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
             "num_fw_heads": 4, "fw_head_dim": 16, "inter_multi": 1.0, "lact_chunk_size": 4,
             "window_size": 8, "objective": "\(objective)", "update_rule": "\(rule)",
             "memory_mode": "\(mode)", "use_weight_norm": \(weightNorm), "muon_steps": 5, "base_lr": 0.08,
             "conv_kernel": \(conv), "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": false}
            """
        return try withRandomState(MLXRandom.RandomState(seed: 8490)) {
            let model = try LaCTModel(configurationData: Data(json.utf8))
            TinyModel.seedSpecialParameters(model, family: .lact)
            eval(model)
            return model
        }
    }

    @Test("LaCT SwiGLU+Muon+WeightNorm, SwiGLU+Momentum+L2, and Listing B.8 Linear+SGD+L2 match the independent scalar CPU reference within 1e-4 (#228)",
          arguments: [
              ("swiglu", "dot_product", "muon", true, 4),
              ("swiglu", "l2", "momentum", false, 0),
              ("linear", "l2", "sgd", false, 0),
          ])
    func matchesIndependentCPUReference(mode: String, objective: String, rule: String, weightNorm: Bool, conv: Int) throws {
        let model = try Self.makeVariant(mode: mode, objective: objective, rule: rule, weightNorm: weightNorm, conv: conv)
        let cache = model.newCache(parameters: nil)
        #expect(cache.count == 2 * model.configuration.numHiddenLayers)
        #expect(cache[0] is RotatingKVCache && cache[1] is FastWeightLayerCache)
        let logits = model(MLXArray(Self.tokens)[.newAxis], cache: cache).asType(.float32)
        let reference = try LaCTReference(model).forward(Self.tokens)
        #expect(FastWeightCPUMath.maxAbsDiff(logits, reference.logits) < 1e-4)
        #expect(argMax(logits[0], axis: -1).asArray(Int32.self) == FastWeightCPUMath.argMaxRows(reference.logits))
        for layerIdx in 0 ..< model.configuration.numHiddenLayers {
            let fw = try #require(cache[2 * layerIdx + 1] as? FastWeightLayerCache)
            let w1 = try #require(fw.slot(0)).asType(.float32).asArray(Float.self)
            let refW1 = reference.finalW1[layerIdx].flatMap { $0 }
            let diff = zip(w1, refW1).reduce(Float(0)) { Swift.max($0, abs($1.0 - $1.1)) }
            #expect(diff < 1e-4)
        }
    }

    @Test("Sub-chunk multi-chunk prefill (2 + 3 + 3 + 2) and token-by-token streaming match single-pass prefill for SwiGLU+Muon and Linear+SGD",
          arguments: [("swiglu", "dot_product", "muon", true), ("linear", "l2", "sgd", false)])
    func subChunkAndStreamingMatchSinglePass(mode: String, objective: String, rule: String, weightNorm: Bool) throws {
        let model = try Self.makeVariant(mode: mode, objective: objective, rule: rule, weightNorm: weightNorm, conv: 4)
        let singleCache = model.newCache(parameters: nil)
        let single = model(MLXArray(Self.tokens)[.newAxis], cache: singleCache).asType(.float32)

        let chunkCache = model.newCache(parameters: nil)
        let parts = [0 ..< 2, 2 ..< 5, 5 ..< 8, 8 ..< 10].map { r in
            model(MLXArray(Array(Self.tokens[r]))[.newAxis], cache: chunkCache).asType(.float32)
        }
        #expect(MLX.abs(concatenated(parts, axis: 1) - single).max().item(Float.self) < 1e-4)

        let streamCache = model.newCache(parameters: nil)
        let stepped = concatenated(Self.tokens.map { model(MLXArray([$0])[.newAxis], cache: streamCache).asType(.float32) }, axis: 1)
        #expect(MLX.abs(stepped - single).max().item(Float.self) < 1e-4)
        #expect(argMax(stepped, axis: -1).asArray(Int32.self) == argMax(single, axis: -1).asArray(Int32.self))
        #expect(PromptCache(streamCache).isSnapshot(of: Self.tokens.count))
    }

    @Test("LaCT inner-loop loss decreases monotonically for both L2 (Linear and SwiGLU) and DotProduct (SwiGLU+Muon), and lowers next-token NLL on repeated context vs frozen (lr = 0)",
          arguments: [
              (LaCTConfiguration.MemoryMode.linear, LaCTConfiguration.Objective.l2, LaCTConfiguration.UpdateRule.sgd, false),
              (.swiglu, .dotProduct, .muon, true),
          ])
    func adaptationReducesInnerLossAndNextTokenNLL(
        mode: LaCTConfiguration.MemoryMode,
        objective: LaCTConfiguration.Objective,
        rule: LaCTConfiguration.UpdateRule,
        weightNorm: Bool
    ) throws {
        let json = """
            {"model_type": "lact_swiglu", "vocab_size": 16, "hidden_size": 16, "intermediate_size": 16,
             "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 16,
             "num_fw_heads": 1, "fw_head_dim": 16, "inter_multi": 1.0, "lact_chunk_size": 4,
             "window_size": 4, "objective": "\(objective.rawValue)", "update_rule": "\(rule.rawValue)",
             "memory_mode": "\(mode.rawValue)", "use_weight_norm": \(weightNorm), "muon_steps": 5,
             "momentum_beta": 0.0, "base_lr": 0.05, "conv_kernel": 0}
            """
        let model = try withRandomState(MLXRandom.RandomState(seed: 9401)) { try LaCTModel(configurationData: Data(json.utf8)) }
        let mem = model.model.layers[0].fastWeight
        let (k, v) = withRandomState(MLXRandom.RandomState(seed: 9402)) {
            (FastWeightMath.l2Normalize(MLXRandom.normal([1, 1, 4, 16])), MLXRandom.normal([1, 1, 4, 16]) * Float(0.5))
        }
        var state = LaCTMemoryState.load(from: nil, batch: 1, memory: mem)
        let ones = MLXArray.ones([1, 1, 4, 1], dtype: .float32)
        let slice = FastWeightChunkSlice(range: 0 ..< 4, stepStart: 0, stepEnd: 4, completesChunk: true, chunkSize: 4)
        let inputs = LaCTSliceInputs(q: k, k: k, v: v, lr1: ones, lr2: ones, lr3: ones)
        let norms = weightNorm
            ? LaCTTargetNorms(w1: FastWeightMath.rowL2Norm(mem.w1Init), w2: mem.w2Init.map { FastWeightMath.rowL2Norm($0) }, w3: mem.w3Init.map { FastWeightMath.rowL2Norm($0) })
            : LaCTTargetNorms(w1: nil, w2: nil, w3: nil)
        var losses: [Float] = []
        for _ in 0 ..< 5 {
            losses.append(Self.innerObjectiveLoss(k: k, v: v, state: state, config: mem.config))
            _ = mode == .linear
                ? LaCTUpdateOps.stepLinear(inputs, config: mem.config, targetNorm: norms.w1, slice: slice, state: &state)
                : LaCTUpdateOps.stepSwiGLU(inputs, config: mem.config, norms: norms, slice: slice, state: &state)
        }
        for idx in 1 ..< losses.count { #expect(losses[idx] < losses[idx - 1]) }

        let active = try Self.makeAssociativeLaCT(mode: mode, objective: objective, rule: rule, weightNorm: weightNorm, baseLR: mode == .linear ? 4.0 : 0.8)
        let frozen = try Self.makeAssociativeLaCT(mode: mode, objective: objective, rule: rule, weightNorm: weightNorm, baseLR: 0.0)
        let repeated: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]
        let activeNLL = FastWeightCPUMath.nextTokenNLL(logits: active(MLXArray(repeated)[.newAxis], cache: active.newCache(parameters: nil)), tokens: repeated, positions: 8 ..< 15)
        let frozenNLL = FastWeightCPUMath.nextTokenNLL(logits: frozen(MLXArray(repeated)[.newAxis], cache: frozen.newCache(parameters: nil)), tokens: repeated, positions: 8 ..< 15)
        #expect(activeNLL < frozenNLL - 0.1)
    }
}

extension LaCTModelTests {
    static func innerObjectiveLoss(k: MLXArray, v: MLXArray, state: LaCTMemoryState, config: LaCTConfiguration) -> Float {
        let pred: MLXArray
        if config.memoryMode == .swiglu, let w2 = state.w2, let w3 = state.w3 {
            pred = matmul(silu(matmul(k, state.w1.swappedAxes(-1, -2))) * matmul(k, w3.swappedAxes(-1, -2)), w2.swappedAxes(-1, -2))
        } else {
            pred = matmul(k, state.w1.swappedAxes(-1, -2))
        }
        if config.objective == .dotProduct {
            return (-(pred * v).sum()).item(Float.self)
        }
        let diff = pred - v
        return (Float(0.5) * (diff * diff).sum()).item(Float.self)
    }

    static func makeAssociativeLaCT(
        mode: LaCTConfiguration.MemoryMode,
        objective: LaCTConfiguration.Objective,
        rule: LaCTConfiguration.UpdateRule,
        weightNorm: Bool,
        baseLR: Float
    ) throws -> LaCTModel {
        let json = """
            {"model_type": "lact_swiglu", "vocab_size": 16, "hidden_size": 16, "intermediate_size": 16,
             "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 16,
             "num_fw_heads": 1, "fw_head_dim": 16, "inter_multi": 1.0, "lact_chunk_size": 8,
             "window_size": 4, "objective": "\(objective.rawValue)", "update_rule": "\(rule.rawValue)",
             "memory_mode": "\(mode.rawValue)", "use_weight_norm": \(weightNorm), "muon_steps": 5,
             "momentum_beta": 0.0, "base_lr": \(baseLR), "conv_kernel": 2}
            """
        let model = try LaCTModel(configurationData: Data(json.utf8))
        let eye16 = MLXArray.eye(16, dtype: .float32)
        let convId = MLXArray(Array(repeating: [Float(0), Float(1)], count: 16).flatMap { $0 }).reshaped(16, 2, 1)
        let convLag = MLXArray(Array(repeating: [Float(1), Float(0)], count: 16).flatMap { $0 }).reshaped(16, 2, 1)
        var updates: [(String, MLXArray)] = [
            ("model.embed_tokens.weight", eye16),
            ("lm_head.weight", eye16 * Float(6.0)),
            ("model.norm.weight", MLXArray.ones([16], dtype: .float32)),
            ("model.layers.0.input_layernorm.weight", MLXArray.ones([16], dtype: .float32)),
            ("model.layers.0.fw_layernorm.weight", MLXArray.ones([16], dtype: .float32)),
            ("model.layers.0.post_attention_layernorm.weight", MLXArray.ones([16], dtype: .float32)),
            ("model.layers.0.self_attn.o_proj.weight", MLXArray.zeros([16, 16], dtype: .float32)),
            ("model.layers.0.mlp.down_proj.weight", MLXArray.zeros([16, 16], dtype: .float32)),
            ("model.layers.0.fast_weight.q_proj.weight", eye16 * Float(0.25)),
            ("model.layers.0.fast_weight.k_proj.weight", eye16 * Float(0.25)),
            ("model.layers.0.fast_weight.v_proj.weight", eye16 * Float(0.25)),
            ("model.layers.0.fast_weight.o_proj.weight", eye16),
            ("model.layers.0.fast_weight.fw_out_norm.weight", MLXArray.ones([16], dtype: .float32)),
            ("model.layers.0.fast_weight.q_conv.conv.weight", convId),
            ("model.layers.0.fast_weight.q_conv.conv.bias", MLXArray.zeros([16], dtype: .float32)),
            ("model.layers.0.fast_weight.k_conv.conv.weight", convLag),
            ("model.layers.0.fast_weight.k_conv.conv.bias", MLXArray.zeros([16], dtype: .float32)),
            ("model.layers.0.fast_weight.v_conv.conv.weight", convId),
            ("model.layers.0.fast_weight.v_conv.conv.bias", MLXArray.zeros([16], dtype: .float32)),
            ("model.layers.0.fast_weight.w1_init", eye16.reshaped(1, 1, 16, 16) * Float(mode == .swiglu ? 1.0 : 0.1)),
        ]
        if mode == .swiglu {
            updates.append(("model.layers.0.fast_weight.w2_init", eye16.reshaped(1, 1, 16, 16) * Float(0.2)))
            updates.append(("model.layers.0.fast_weight.w3_init", eye16.reshaped(1, 1, 16, 16)))
            updates.append(("model.layers.0.fast_weight.lr_proj.weight", MLXArray.zeros([3, 16], dtype: .float32)))
            updates.append(("model.layers.0.fast_weight.lr_proj.bias", MLXArray([Float(-10.0), Float(10.0), Float(-10.0)])))
        }
        model.update(parameters: ModuleParameters.unflattened(updates))
        eval(model)
        return model
    }

    @Test("LaCT config parses a1600012888/LaCT aliases, refuses invalid configs, sanitizes w0/w1/w2 and fused qkv, and registers both lact_swiglu and lact")
    func aliasesSanitizeAndRegistry() async throws {
        let upstreamJSON = """
            {"model_type": "lact_swiglu", "vocab_size": 128, "hidden_size": 64, "num_hidden_layers": 2,
             "num_attn_heads": 4, "num_fw_heads": 2, "fw_head_dim": 32, "inter_multi": 1.5,
             "lact_chunk_size": 512, "window_size": 128, "use_muon": false, "use_momentum": true,
             "norm_eps": 1e-5, "tie_word_embeddings": true}
            """
        let cfg = try JSONDecoder().decode(LaCTConfiguration.self, from: Data(upstreamJSON.utf8))
        #expect(cfg.numAttentionHeads == 4 && cfg.numFastWeightHeads == 2 && cfg.fastWeightHiddenDim == 48)
        #expect(cfg.chunkSize == 512 && cfg.windowSize == 128 && cfg.updateRule == .momentum)
        #expect(throws: DecodingError.self) {
            _ = try LaCTModel(configurationData: Data(#"{"vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2, "muon_steps": 0}"#.utf8))
        }

        let model = try LaCTModel(configurationData: Data(upstreamJSON.utf8))
        #expect(model.lmHead == nil)
        let raw: [String: MLXArray] = [
            "model.embeddings.weight": MLXArray.ones([128, 64]),
            "lm_head.weight": MLXArray.ones([128, 64]),
            "model.layers.0.attn_norm.weight": MLXArray.ones([64]),
            "model.layers.0.mlp_norm.weight": MLXArray.ones([64]),
            "model.layers.0.attn.qkv.weight": MLXArray.ones([192, 64]),
            "model.layers.0.attn.o_proj.weight": MLXArray.ones([64, 64]),
            "model.layers.0.attn.w0": MLXArray.ones([2, 48, 32]),
            "model.layers.0.attn.w1": MLXArray.ones([2, 32, 48]),
            "model.layers.0.attn.w2": MLXArray.ones([2, 48, 32]),
            "model.layers.0.attn.lr_proj.weight": MLXArray.ones([6, 64]),
            "model.layers.0.attn.momentum_proj.weight": MLXArray.ones([2, 64]),
        ]
        let clean = model.sanitize(weights: raw)
        #expect(clean["model.embed_tokens.weight"] != nil && clean["lm_head.weight"] == nil)
        #expect(clean["model.layers.0.input_layernorm.weight"] != nil && clean["model.layers.0.post_attention_layernorm.weight"] != nil)
        #expect(clean["model.layers.0.self_attn.q_proj.weight"]?.shape == [64, 64])
        #expect(clean["model.layers.0.self_attn.k_proj.weight"]?.shape == [64, 64])
        #expect(clean["model.layers.0.self_attn.v_proj.weight"]?.shape == [64, 64])
        #expect(clean["model.layers.0.fast_weight.w1_init"]?.shape == [1, 2, 48, 32])
        #expect(clean["model.layers.0.fast_weight.w2_init"]?.shape == [1, 2, 32, 48])
        #expect(clean["model.layers.0.fast_weight.w3_init"]?.shape == [1, 2, 48, 32])
        #expect(clean["model.layers.0.fast_weight.lr_proj.weight"] != nil)
        #expect(clean["model.layers.0.attn.momentum_proj.weight"] == nil)

        await MLXInferenceEngine.registerAliases()
        for alias in LaCTModel.modelTypes {
            #expect(await LLMTypeRegistry.shared.contains(alias))
            let caps = try #require(ModelCapabilities.of(modelType: alias))
            #expect(caps.allows(.promptCacheReuse) && caps.allows(.grammarJumpForward) && !caps.allows(.kvQuantization))
        }
        let created = try await LLMTypeRegistry.shared.createModel(configuration: Data(upstreamJSON.utf8), modelType: "lact")
        #expect(created is LaCTModel)
    }
}
