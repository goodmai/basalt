import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

private struct TitansRefHeadWeights {
    var w1: [Float]
    var b1: [Float]
    var w2: [Float]
    var b2: [Float]

    func eval(_ x: [Float], d: Int, m: Int, depth: Int) -> [Float] {
        let inner = depth == 1 ? d : m
        var z1 = (0 ..< inner).map { c -> Float in
            var acc = b1[c]
            for r in 0 ..< d { acc += x[r] * w1[r * inner + c] }
            return acc
        }
        guard depth == 2 else { return zip(z1, x).map(+) }
        z1 = z1.map(FastWeightCPUMath.siluScalar)
        return (0 ..< d).map { c in
            var acc = b2[c] + x[c]
            for j in 0 ..< m { acc += z1[j] * w2[j * d + c] }
            return acc
        }
    }

    static func zeros(d: Int, m: Int, depth: Int) -> TitansRefHeadWeights {
        let inner = depth == 1 ? d : m
        return TitansRefHeadWeights(
            w1: Array(repeating: 0, count: d * inner),
            b1: Array(repeating: 0, count: inner),
            w2: depth == 2 ? Array(repeating: 0, count: inner * d) : [],
            b2: depth == 2 ? Array(repeating: 0, count: d) : []
        )
    }
}

private struct TitansRefFactors {
    let beta: Float
    let eta: Float
    let theta: Float
}

private struct TitansRefStepInput {
    let k: [Float]
    let v: [Float]
    let alpha: Float
    let eta: Float
    let theta: Float
}

private struct TitansRefMemState {
    var base: [TitansRefHeadWeights]
    var cur: [TitansRefHeadWeights]
    var mom: [TitansRefHeadWeights]
}

private struct TitansReference {
    let weights: [String: [Float]]
    let config: TitansConfiguration

    init(_ model: TitansModel) {
        self.weights = FastWeightCPUMath.flatWeights(model)
        self.config = model.configuration
    }

    func forward(_ tokens: [Int32]) throws -> (logits: [[Float]], finalBaseW1: [[[Float]]]) {
        var h = FastWeightCPUMath.embedding(tokens, weight: try w("model.embed_tokens.weight"), hiddenSize: config.hiddenSize)
        var finalBase: [[[Float]]] = []
        for layer in 0 ..< config.numHiddenLayers {
            let (nextH, layerBase) = try block(h, layer: layer)
            h = nextH
            finalBase.append(layerBase)
        }
        let normed = FastWeightCPUMath.rmsNorm(h, weight: try w("model.norm.weight"), eps: config.rmsNormEps)
        let head = try weights["lm_head.weight"] ?? w("model.embed_tokens.weight")
        return (FastWeightCPUMath.linear(normed, weight: head, outDim: config.vocabSize), finalBase)
    }

    private func block(_ x: [[Float]], layer: Int) throws -> ([[Float]], [[Float]]) {
        let p = "model.layers.\(layer)."
        let mixed: [[Float]]
        let baseW1: [[Float]]
        switch config.variant {
        case .mag:
            let normed = FastWeightCPUMath.rmsNorm(x, weight: try w(p + "input_layernorm.weight"), eps: config.rmsNormEps)
            let attn = try attention(normed, prefix: p + "self_attn.")
            let (mem, finalW1) = try runMemoryFull(normed, prefix: p + "memory.")
            let gated = zip(attn, mem).map { ar, mr in zip(ar, mr).map { $0 * FastWeightCPUMath.sigmoidScalar($1) } }
            (mixed, baseW1) = (FastWeightCPUMath.add(x, gated), finalW1)
        case .mal:
            let normed = FastWeightCPUMath.rmsNorm(x, weight: try w(p + "input_layernorm.weight"), eps: config.rmsNormEps)
            let (mem, finalW1) = try runMemoryFull(normed, prefix: p + "memory.")
            let mid = FastWeightCPUMath.add(x, mem)
            let attnIn = FastWeightCPUMath.rmsNorm(mid, weight: try w(p + "post_memory_layernorm.weight"), eps: config.rmsNormEps)
            (mixed, baseW1) = (FastWeightCPUMath.add(mid, try attention(attnIn, prefix: p + "self_attn.")), finalW1)
        case .mac:
            (mixed, baseW1) = try runMAC(x, prefix: p)
        }
        let ffnIn = FastWeightCPUMath.rmsNorm(mixed, weight: try w(p + "post_attention_layernorm.weight"), eps: config.rmsNormEps)
        let ffnOut = FastWeightCPUMath.swiGLU(
            ffnIn,
            gateW: try w(p + "mlp.gate_proj.weight"),
            upW: try w(p + "mlp.up_proj.weight"),
            downW: try w(p + "mlp.down_proj.weight"),
            interDim: config.intermediateSize
        )
        return (FastWeightCPUMath.add(mixed, ffnOut), baseW1)
    }

    private func runMAC(_ x: [[Float]], prefix p: String) throws -> ([[Float]], [[Float]]) {
        let normed = FastWeightCPUMath.rmsNorm(x, weight: try w(p + "input_layernorm.weight"), eps: config.rmsNormEps)
        let mp = p + "memory."
        let qFull = try projectMemQ(normed, prefix: mp, useConv: true)
        var state = try initMemState(prefix: mp)
        var allAttnIn: [[Float]] = []
        var allYAttn: [[Float]] = []
        var allOut: [[Float]] = []
        var cursor = 0
        while cursor < x.count {
            let end = Swift.min(cursor + config.chunkSize, x.count)
            let qSlice = qFull.map { Array($0[cursor ..< end]) }
            let hist = try readout(qSlice, state: state, prefix: mp)
            let ctx = FastWeightCPUMath.linear(hist, weight: try w(p + "mac_ctx_proj.weight"), outDim: config.hiddenSize)
            let attnSlice = FastWeightCPUMath.add(Array(normed[cursor ..< end]), ctx)
            allAttnIn.append(contentsOf: attnSlice)
            let fullAttn = try attention(allAttnIn, prefix: p + "self_attn.")
            let yAttn = Array(fullAttn[cursor ..< end])
            allYAttn.append(contentsOf: yAttn)
            let qGate = try projectMemQ(yAttn, prefix: mp, useConv: false)
            let gate = try readout(qGate, state: state, prefix: mp)
            let gated = zip(yAttn, gate).map { yr, gr in zip(yr, gr).map { $0 * FastWeightCPUMath.sigmoidScalar($1) } }
            allOut.append(contentsOf: gated)
            try updateMACChunk(allYAttn: allYAttn, range: cursor ..< end, completes: (end - cursor) == config.chunkSize, prefix: mp, state: &state)
            cursor = end
        }
        return (FastWeightCPUMath.add(x, allOut), state.base.map(\.w1))
    }

    private func updateMACChunk(
        allYAttn: [[Float]],
        range: Range<Int>,
        completes: Bool,
        prefix mp: String,
        state: inout TitansRefMemState
    ) throws {
        let (kAll, vAll, alphaAll, etaAll, thetaAll) = try projectWriteAll(allYAttn, prefix: mp)
        for h in 0 ..< config.numAttentionHeads {
            for t in range {
                stepHeadRecurrence(
                    base: state.base[h],
                    cur: &state.cur[h],
                    mom: &state.mom[h],
                    input: TitansRefStepInput(k: kAll[h][t], v: vAll[h][t], alpha: alphaAll[h][t], eta: etaAll[h][t], theta: thetaAll[h][t])
                )
            }
            if completes { state.base[h] = state.cur[h] }
        }
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
        let (kP, vP) = try persistentKV(prefix: p, kvDim: kvDim)
        let input = CPUSlidingAttentionInput(q: q, k: k, v: v, kPersistent: kP, vPersistent: vP, windowSize: config.windowSize)
        return FastWeightCPUMath.linear(FastWeightCPUMath.slidingAttention(input), weight: try w(p + "o_proj.weight"), outDim: d)
    }

    private func persistentKV(prefix p: String, kvDim: Int) throws -> ([[[Float]]], [[[Float]]]) {
        guard config.numPersistentTokens > 0, let raw = weights[p + "persistent_tokens"] else {
            let empty = Array(repeating: [[Float]](), count: config.numKeyValueHeads)
            return (empty, empty)
        }
        let d = config.hiddenSize
        let tokens = (0 ..< config.numPersistentTokens).map { idx in Array(raw[(idx * d) ..< ((idx + 1) * d)]) }
        let kP = FastWeightCPUMath.splitHeads(FastWeightCPUMath.linear(tokens, weight: try w(p + "k_proj.weight"), outDim: kvDim), heads: config.numKeyValueHeads, headDim: config.headDim)
        let vP = FastWeightCPUMath.splitHeads(FastWeightCPUMath.linear(tokens, weight: try w(p + "v_proj.weight"), outDim: kvDim), heads: config.numKeyValueHeads, headDim: config.headDim)
        return (kP, vP)
    }

    private func runMemoryFull(_ x: [[Float]], prefix p: String) throws -> ([[Float]], [[Float]]) {
        let q = try projectMemQ(x, prefix: p, useConv: true)
        let (k, v, alpha, eta, theta) = try projectWriteAll(x, prefix: p)
        var state = try initMemState(prefix: p)
        var headOuts = Array(repeating: [[Float]](), count: config.numAttentionHeads)
        for t in 0 ..< x.count {
            let completes = (t % config.chunkSize) + 1 == config.chunkSize
            for h in 0 ..< config.numAttentionHeads {
                headOuts[h].append(state.base[h].eval(q[h][t], d: config.headDim, m: config.memoryHiddenDim, depth: config.memoryDepth))
                stepHeadRecurrence(
                    base: state.base[h],
                    cur: &state.cur[h],
                    mom: &state.mom[h],
                    input: TitansRefStepInput(k: k[h][t], v: v[h][t], alpha: alpha[h][t], eta: eta[h][t], theta: theta[h][t])
                )
                if completes { state.base[h] = state.cur[h] }
            }
        }
        let merged = FastWeightCPUMath.mergeHeads(headOuts)
        let proj = FastWeightCPUMath.linear(merged, weight: try w(p + "o_proj.weight"), outDim: config.hiddenSize)
        return (FastWeightCPUMath.rmsNorm(proj, weight: try w(p + "mem_norm.weight"), eps: config.rmsNormEps), state.base.map(\.w1))
    }

    private func readout(_ q: [[[Float]]], state: TitansRefMemState, prefix p: String) throws -> [[Float]] {
        let headOuts: [[[Float]]] = (0 ..< config.numAttentionHeads).map { h in
            q[h].map { state.base[h].eval($0, d: config.headDim, m: config.memoryHiddenDim, depth: config.memoryDepth) }
        }
        let proj = FastWeightCPUMath.linear(FastWeightCPUMath.mergeHeads(headOuts), weight: try w(p + "o_proj.weight"), outDim: config.hiddenSize)
        return FastWeightCPUMath.rmsNorm(proj, weight: try w(p + "mem_norm.weight"), eps: config.rmsNormEps)
    }

    private func w(_ key: String) throws -> [Float] {
        try #require(weights[key], "Missing Titans reference weight \(key)")
    }
}

extension TitansReference {
    private func initMemState(prefix p: String) throws -> TitansRefMemState {
        let d = config.headDim
        let m = config.memoryHiddenDim
        let inner = config.memoryDepth == 1 ? d : m
        let w1All = try w(p + "w1_init")
        let b1All = try w(p + "b1_init")
        let w2All = config.memoryDepth == 2 ? try w(p + "w2_init") : []
        let b2All = config.memoryDepth == 2 ? try w(p + "b2_init") : []
        let base: [TitansRefHeadWeights] = (0 ..< config.numAttentionHeads).map { h in
            TitansRefHeadWeights(
                w1: Array(w1All[(h * d * inner) ..< ((h + 1) * d * inner)]),
                b1: Array(b1All[(h * inner) ..< ((h + 1) * inner)]),
                w2: config.memoryDepth == 2 ? Array(w2All[(h * inner * d) ..< ((h + 1) * inner * d)]) : [],
                b2: config.memoryDepth == 2 ? Array(b2All[(h * d) ..< ((h + 1) * d)]) : []
            )
        }
        let mom = (0 ..< config.numAttentionHeads).map { _ in TitansRefHeadWeights.zeros(d: d, m: m, depth: config.memoryDepth) }
        return TitansRefMemState(base: base, cur: base, mom: mom)
    }

    private func projectMemQ(_ x: [[Float]], prefix p: String, useConv: Bool) throws -> [[[Float]]] {
        var raw = FastWeightCPUMath.linear(x, weight: try w(p + "q_proj.weight"), outDim: config.hiddenSize)
        if useConv, config.convKernel > 1 {
            raw = FastWeightCPUMath.depthwiseConv1d(raw, weight: try w(p + "q_conv.conv.weight"), bias: weights[p + "q_conv.conv.bias"], kernel: config.convKernel, applySiLU: true)
        } else {
            raw = raw.map { $0.map(FastWeightCPUMath.siluScalar) }
        }
        return FastWeightCPUMath.splitHeads(raw, heads: config.numAttentionHeads, headDim: config.headDim)
            .map { $0.map { FastWeightCPUMath.l2NormVec($0) } }
    }

    private func projectWriteAll(_ x: [[Float]], prefix p: String) throws -> ([[[Float]]], [[[Float]]], [[Float]], [[Float]], [[Float]]) {
        var kRaw = FastWeightCPUMath.linear(x, weight: try w(p + "k_proj.weight"), outDim: config.hiddenSize)
        var vRaw = FastWeightCPUMath.linear(x, weight: try w(p + "v_proj.weight"), outDim: config.hiddenSize)
        if config.convKernel > 1 {
            kRaw = FastWeightCPUMath.depthwiseConv1d(kRaw, weight: try w(p + "k_conv.conv.weight"), bias: weights[p + "k_conv.conv.bias"], kernel: config.convKernel, applySiLU: true)
            vRaw = FastWeightCPUMath.depthwiseConv1d(vRaw, weight: try w(p + "v_conv.conv.weight"), bias: weights[p + "v_conv.conv.bias"], kernel: config.convKernel, applySiLU: true)
        } else {
            kRaw = kRaw.map { $0.map(FastWeightCPUMath.siluScalar) }
            vRaw = vRaw.map { $0.map(FastWeightCPUMath.siluScalar) }
        }
        let k = FastWeightCPUMath.splitHeads(kRaw, heads: config.numAttentionHeads, headDim: config.headDim).map { $0.map { FastWeightCPUMath.l2NormVec($0) } }
        let v = FastWeightCPUMath.splitHeads(vRaw, heads: config.numAttentionHeads, headDim: config.headDim)
        let aRaw = FastWeightCPUMath.linear(x, weight: try w(p + "alpha_proj.weight"), bias: weights[p + "alpha_proj.bias"], outDim: config.numAttentionHeads)
        let eRaw = FastWeightCPUMath.linear(x, weight: try w(p + "eta_proj.weight"), bias: weights[p + "eta_proj.bias"], outDim: config.numAttentionHeads)
        let tRaw = FastWeightCPUMath.linear(x, weight: try w(p + "theta_proj.weight"), bias: weights[p + "theta_proj.bias"], outDim: config.numAttentionHeads)
        let alpha = (0 ..< config.numAttentionHeads).map { h in aRaw.map { FastWeightCPUMath.sigmoidScalar($0[h]) } }
        let eta = (0 ..< config.numAttentionHeads).map { h in eRaw.map { FastWeightCPUMath.sigmoidScalar($0[h]) } }
        let theta = (0 ..< config.numAttentionHeads).map { h in tRaw.map { FastWeightCPUMath.sigmoidScalar($0[h]) * config.baseLR } }
        return (k, v, alpha, eta, theta)
    }
}

extension TitansReference {
    private func stepHeadRecurrence(
        base: TitansRefHeadWeights,
        cur: inout TitansRefHeadWeights,
        mom: inout TitansRefHeadWeights,
        input: TitansRefStepInput
    ) {
        let d = config.headDim
        let m = config.memoryHiddenDim
        let factors = TitansRefFactors(beta: Float(1.0) - input.alpha, eta: input.eta, theta: input.theta)
        let inner = config.memoryDepth == 1 ? d : m
        let z1 = (0 ..< inner).map { c -> Float in
            var acc = base.b1[c]
            for r in 0 ..< d { acc += input.k[r] * base.w1[r * inner + c] }
            return acc
        }
        if config.memoryDepth == 1 {
            let delta = (0 ..< d).map { z1[$0] + input.k[$0] - input.v[$0] }
            applyParamStep(&cur.w1, &mom.w1, act: input.k, delta: delta, factors: factors)
            applyBiasStep(&cur.b1, &mom.b1, delta: delta, factors: factors)
            return
        }
        let h1 = z1.map(FastWeightCPUMath.siluScalar)
        let delta2 = (0 ..< d).map { c -> Float in
            var acc = base.b2[c] + input.k[c] - input.v[c]
            for j in 0 ..< m { acc += h1[j] * base.w2[j * d + c] }
            return acc
        }
        let delta1 = (0 ..< m).map { j -> Float in
            var back: Float = 0
            for c in 0 ..< d { back += delta2[c] * base.w2[j * d + c] }
            return back * FastWeightCPUMath.siluDerivScalar(z1[j])
        }
        applyParamStep(&cur.w1, &mom.w1, act: input.k, delta: delta1, factors: factors)
        applyBiasStep(&cur.b1, &mom.b1, delta: delta1, factors: factors)
        applyParamStep(&cur.w2, &mom.w2, act: h1, delta: delta2, factors: factors)
        applyBiasStep(&cur.b2, &mom.b2, delta: delta2, factors: factors)
    }

    private func applyParamStep(
        _ cur: inout [Float],
        _ mom: inout [Float],
        act: [Float],
        delta: [Float],
        factors: TitansRefFactors
    ) {
        let cols = delta.count
        for r in 0 ..< act.count {
            for c in 0 ..< cols {
                let idx = r * cols + c
                mom[idx] = factors.eta * mom[idx] - factors.theta * act[r] * delta[c]
                cur[idx] = factors.beta * cur[idx] + mom[idx]
            }
        }
    }

    private func applyBiasStep(
        _ cur: inout [Float],
        _ mom: inout [Float],
        delta: [Float],
        factors: TitansRefFactors
    ) {
        for c in 0 ..< delta.count {
            mom[c] = factors.eta * mom[c] - factors.theta * delta[c]
            cur[c] = factors.beta * cur[c] + mom[c]
        }
    }
}

@Suite("Titans Neural Long-Term Memory (MAC, MAG, MAL, depths 1 & 2, NaN-free edge cases, streaming, adaptation) (#848)", .serialized)
struct TitansModelTests {
    static let tokens: [Int32] = [5, 19, 42, 7, 88, 13, 64, 21, 33, 11]

    static func makeModel(variant: TitansConfiguration.Variant, depth: Int) throws -> TitansModel {
        let cfg = TinyModel.Family.titans.config
            .replacingOccurrences(of: #""titans_variant": "mag""#, with: #""titans_variant": "\#(variant.rawValue)""#)
            .replacingOccurrences(of: #""memory_depth": 2"#, with: #""memory_depth": \#(depth)"#)
        let variantId = variant == .mac ? 1 : (variant == .mag ? 2 : 3)
        return try withRandomState(MLXRandom.RandomState(seed: UInt64(9100 + depth * 10 + variantId))) {
            let model = try TitansModel(configurationData: Data(cfg.utf8))
            TinyModel.seedSpecialParameters(model, family: .titans)
            eval(model)
            return model
        }
    }

    @Test("Titans MAC, MAG, and MAL at memory depths 1 and 2 match the independent scalar CPU reference within 1e-4 and exact argmax ids (#228)",
          arguments: [
              (TitansConfiguration.Variant.mac, 1), (.mac, 2),
              (.mag, 1), (.mag, 2),
              (.mal, 1), (.mal, 2),
          ])
    func matchesIndependentCPUReference(variant: TitansConfiguration.Variant, depth: Int) throws {
        let model = try Self.makeModel(variant: variant, depth: depth)
        let cache = model.newCache(parameters: nil)
        #expect(cache.count == 2 * model.configuration.numHiddenLayers)
        #expect(cache[0] is RotatingKVCache && cache[1] is FastWeightLayerCache)
        let logits = model(MLXArray(Self.tokens)[.newAxis], cache: cache).asType(.float32)
        let reference = try TitansReference(model).forward(Self.tokens)
        #expect(FastWeightCPUMath.maxAbsDiff(logits, reference.logits) < 1e-4)
        #expect(argMax(logits[0], axis: -1).asArray(Int32.self) == FastWeightCPUMath.argMaxRows(reference.logits))
        for layerIdx in 0 ..< model.configuration.numHiddenLayers {
            let fw = try #require(cache[2 * layerIdx + 1] as? FastWeightLayerCache)
            let w1 = try #require(fw.slot(0)).asType(.float32).asArray(Float.self)
            let refW1 = reference.finalBaseW1[layerIdx].flatMap { $0 }
            let diff = zip(w1, refW1).reduce(Float(0)) { Swift.max($0, abs($1.0 - $1.1)) }
            #expect(diff < 1e-4)
        }
    }

    @Test("Multi-chunk prefill (5 + 3 + 2) and token-by-token streaming match single-pass prefill past windowSize across MAC, MAG, and MAL",
          arguments: TitansConfiguration.Variant.allCases)
    func chunkedAndStreamingMatchSinglePass(variant: TitansConfiguration.Variant) throws {
        let model = try Self.makeModel(variant: variant, depth: 2)
        let singleCache = model.newCache(parameters: nil)
        let single = model(MLXArray(Self.tokens)[.newAxis], cache: singleCache).asType(.float32)

        let chunkCache = model.newCache(parameters: nil)
        let c1 = model(MLXArray(Array(Self.tokens[0 ..< 5]))[.newAxis], cache: chunkCache).asType(.float32)
        let c2 = model(MLXArray(Array(Self.tokens[5 ..< 8]))[.newAxis], cache: chunkCache).asType(.float32)
        let c3 = model(MLXArray(Array(Self.tokens[8 ..< 10]))[.newAxis], cache: chunkCache).asType(.float32)
        #expect(MLX.abs(concatenated([c1, c2, c3], axis: 1) - single).max().item(Float.self) < 1e-4)

        let streamCache = model.newCache(parameters: nil)
        let stepped = concatenated(Self.tokens.map { model(MLXArray([$0])[.newAxis], cache: streamCache).asType(.float32) }, axis: 1)
        #expect(MLX.abs(stepped - single).max().item(Float.self) < 1e-4)
        #expect(argMax(stepped, axis: -1).asArray(Int32.self) == argMax(single, axis: -1).asArray(Int32.self))
        #expect(PromptCache(streamCache).isSnapshot(of: Self.tokens.count))
    }

    @Test("Titans chunk update is NaN-free and exact against scalar recurrence at exact alpha = 1 and eta = 0 for depths 1 and 2",
          arguments: [1, 2])
    func exactAlphaOneAndEtaZeroAreNaNFree(depth: Int) {
        let (k, v, w1, b1, w2, b2) = withRandomState(MLXRandom.RandomState(seed: UInt64(9200 + depth))) {
            (
                MLXRandom.normal([1, 2, 5, 4]) * Float(0.3),
                MLXRandom.normal([1, 2, 5, 4]) * Float(0.3),
                MLXRandom.normal([1, 2, 4, depth == 1 ? 4 : 6]) * Float(0.2),
                MLXRandom.normal([1, 2, 1, depth == 1 ? 4 : 6]) * Float(0.1),
                MLXRandom.normal([1, 2, 6, 4]) * Float(0.2),
                MLXRandom.normal([1, 2, 1, 4]) * Float(0.1)
            )
        }
        let initW = TitansMemoryWeights(w1: w1, b1: b1, w2: depth == 2 ? w2 : nil, b2: depth == 2 ? b2 : nil)
        let initMom = TitansMemoryWeights.zeros(batch: 1, heads: 2, headDim: 4, hiddenDim: 6, depth: depth)
        let alpha = MLXArray([Float(0.2), 1.0, 0.3, 1.0, 0.1, 1.0, 0.0, 1.0, 0.5, 0.0]).reshaped(1, 2, 5)
        let eta = MLXArray([Float(0.0), 0.7, 0.0, 0.4, 0.0, 0.5, 0.0, 0.0, 0.8, 0.0]).reshaped(1, 2, 5)
        let theta = MLXArray.ones([1, 2, 5], dtype: .float32) * Float(0.15)

        var batchState = TitansMemoryState(base: initW, current: initW, momentum: initMom)
        TitansNeuralMemory.applyChunkUpdate(
            TitansChunkUpdateInputs(keys: k, values: v, alpha: alpha, eta: eta, theta: theta),
            state: &batchState,
            completesChunk: true
        )
        var stepState = TitansMemoryState(base: initW, current: initW, momentum: initMom)
        for step in 0 ..< 5 {
            TitansNeuralMemory.applyChunkUpdate(
                TitansChunkUpdateInputs(
                    keys: k[0..., 0..., step ..< (step + 1), 0...],
                    values: v[0..., 0..., step ..< (step + 1), 0...],
                    alpha: alpha[0..., 0..., step ..< (step + 1)],
                    eta: eta[0..., 0..., step ..< (step + 1)],
                    theta: theta[0..., 0..., step ..< (step + 1)]
                ),
                state: &stepState,
                completesChunk: step == 4
            )
        }
        #expect(!isNaN(batchState.base.w1).any().item(Bool.self))
        #expect(!isNaN(batchState.momentum.w1).any().item(Bool.self))
        #expect(MLX.abs(batchState.base.w1 - stepState.base.w1).max().item(Float.self) < 1e-5)
        #expect(MLX.abs(batchState.momentum.w1 - stepState.momentum.w1).max().item(Float.self) < 1e-5)
        if depth == 2, let bw2 = batchState.base.w2, let sw2 = stepState.base.w2 {
            #expect(!isNaN(bw2).any().item(Bool.self))
            #expect(MLX.abs(bw2 - sw2).max().item(Float.self) < 1e-5)
        }
    }

    @Test("Titans inner-loop associative loss decreases monotonically and test-time adaptation lowers next-token NLL on repeated context vs frozen (lr = 0)",
          arguments: [1, 2])
    func adaptationReducesLossAndNextTokenNLL(depth: Int) throws {
        let (k, v) = withRandomState(MLXRandom.RandomState(seed: UInt64(9300 + depth))) {
            (FastWeightMath.l2Normalize(MLXRandom.normal([1, 2, 4, 8])), MLXRandom.normal([1, 2, 4, 8]) * Float(0.5))
        }
        let inner = depth == 1 ? 8 : 16
        let initW = TitansMemoryWeights(
            w1: withRandomState(MLXRandom.RandomState(seed: 9310)) { MLXRandom.normal([1, 2, 8, inner]) * Float(0.1) },
            b1: MLXArray.zeros([1, 2, 1, inner], dtype: .float32),
            w2: depth == 2 ? withRandomState(MLXRandom.RandomState(seed: 9311)) { MLXRandom.normal([1, 2, 16, 8]) * Float(0.1) } : nil,
            b2: depth == 2 ? MLXArray.zeros([1, 2, 1, 8], dtype: .float32) : nil
        )
        var state = TitansMemoryState(base: initW, current: initW, momentum: TitansMemoryWeights.zeros(batch: 1, heads: 2, headDim: 8, hiddenDim: 16, depth: depth))
        let update = TitansChunkUpdateInputs(
            keys: k,
            values: v,
            alpha: MLXArray.zeros([1, 2, 4], dtype: .float32),
            eta: MLXArray.zeros([1, 2, 4], dtype: .float32),
            theta: MLXArray.ones([1, 2, 4], dtype: .float32) * Float(0.05)
        )
        var losses: [Float] = []
        for _ in 0 ..< 5 {
            let diff = state.base.evalMemory(k) - v
            losses.append((Float(0.5) * (diff * diff).sum()).item(Float.self))
            TitansNeuralMemory.applyChunkUpdate(update, state: &state, completesChunk: true)
        }
        for idx in 1 ..< losses.count { #expect(losses[idx] < losses[idx - 1]) }

        let active = try Self.makeAssociativeTitans(depth: depth, baseLR: 1.5)
        let frozen = try Self.makeAssociativeTitans(depth: depth, baseLR: 0.0)
        let repeated: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]
        let activeNLL = FastWeightCPUMath.nextTokenNLL(logits: active(MLXArray(repeated)[.newAxis], cache: active.newCache(parameters: nil)), tokens: repeated, positions: 8 ..< 15)
        let frozenNLL = FastWeightCPUMath.nextTokenNLL(logits: frozen(MLXArray(repeated)[.newAxis], cache: frozen.newCache(parameters: nil)), tokens: repeated, positions: 8 ..< 15)
        #expect(activeNLL < frozenNLL - 0.1)
    }
}

extension TitansModelTests {
    static func makeAssociativeTitans(depth: Int, baseLR: Float) throws -> TitansModel {
        let json = """
            {"model_type": "titans_mag", "vocab_size": 16, "hidden_size": 16, "intermediate_size": 16,
             "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 16,
             "memory_depth": \(depth), "memory_hidden_multiplier": 1, "num_persistent_tokens": 0,
             "chunk_size": 8, "window_size": 4, "conv_kernel": 2, "base_lr": \(baseLR),
             "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": false}
            """
        let model = try TitansModel(configurationData: Data(json.utf8))
        let eye16 = MLXArray.eye(16, dtype: .float32)
        let convId = MLXArray(Array(repeating: [Float(0), Float(1)], count: 16).flatMap { $0 }).reshaped(16, 2, 1)
        let convLag = MLXArray(Array(repeating: [Float(1), Float(0)], count: 16).flatMap { $0 }).reshaped(16, 2, 1)
        var updates: [(String, MLXArray)] = [
            ("model.embed_tokens.weight", eye16),
            ("lm_head.weight", eye16 * Float(6.0)),
            ("model.norm.weight", MLXArray.ones([16], dtype: .float32)),
            ("model.layers.0.input_layernorm.weight", MLXArray.ones([16], dtype: .float32)),
            ("model.layers.0.post_attention_layernorm.weight", MLXArray.ones([16], dtype: .float32)),
            ("model.layers.0.mlp.down_proj.weight", MLXArray.zeros([16, 16], dtype: .float32)),
            ("model.layers.0.self_attn.q_proj.weight", MLXArray.zeros([16, 16], dtype: .float32)),
            ("model.layers.0.self_attn.k_proj.weight", MLXArray.zeros([16, 16], dtype: .float32)),
            ("model.layers.0.self_attn.v_proj.weight", MLXArray.ones([16, 16], dtype: .float32) * Float(0.25)),
            ("model.layers.0.self_attn.o_proj.weight", eye16),
            ("model.layers.0.memory.q_proj.weight", eye16 * Float(0.25)),
            ("model.layers.0.memory.k_proj.weight", eye16 * Float(0.25)),
            ("model.layers.0.memory.v_proj.weight", eye16 * Float(0.5)),
            ("model.layers.0.memory.o_proj.weight", eye16),
            ("model.layers.0.memory.mem_norm.weight", MLXArray.ones([16], dtype: .float32)),
            ("model.layers.0.memory.q_conv.conv.weight", convId),
            ("model.layers.0.memory.q_conv.conv.bias", MLXArray.zeros([16], dtype: .float32)),
            ("model.layers.0.memory.k_conv.conv.weight", convLag),
            ("model.layers.0.memory.k_conv.conv.bias", MLXArray.zeros([16], dtype: .float32)),
            ("model.layers.0.memory.v_conv.conv.weight", convId),
            ("model.layers.0.memory.v_conv.conv.bias", MLXArray.zeros([16], dtype: .float32)),
            ("model.layers.0.memory.alpha_proj.weight", MLXArray.zeros([1, 16], dtype: .float32)),
            ("model.layers.0.memory.alpha_proj.bias", MLXArray([Float(-10.0)])),
            ("model.layers.0.memory.eta_proj.weight", MLXArray.zeros([1, 16], dtype: .float32)),
            ("model.layers.0.memory.eta_proj.bias", MLXArray([Float(-10.0)])),
            ("model.layers.0.memory.theta_proj.weight", MLXArray.zeros([1, 16], dtype: .float32)),
            ("model.layers.0.memory.theta_proj.bias", MLXArray([Float(10.0)])),
            ("model.layers.0.memory.w1_init", eye16.reshaped(1, 1, 16, 16) * Float(depth == 2 ? 0.5 : 0.0)),
            ("model.layers.0.memory.b1_init", MLXArray.zeros([1, 1, 1, 16], dtype: .float32)),
        ]
        if depth == 2 {
            updates.append(("model.layers.0.memory.w2_init", eye16.reshaped(1, 1, 16, 16) * Float(0.5)))
            updates.append(("model.layers.0.memory.b2_init", MLXArray.zeros([1, 1, 1, 16], dtype: .float32)))
        }
        model.update(parameters: ModuleParameters.unflattened(updates))
        eval(model)
        return model
    }

    @Test("Titans config aliases infer variant, invalid configs throw DecodingError, and registry + capabilities cover all aliases")
    func configAliasesSanitizeAndRegistry() async throws {
        let macJSON = #"{"model_type": "titans_mac", "vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2}"#
        let malJSON = #"{"model_type": "titans_mal", "vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2}"#
        #expect(try JSONDecoder().decode(TitansConfiguration.self, from: Data(macJSON.utf8)).variant == .mac)
        #expect(try JSONDecoder().decode(TitansConfiguration.self, from: Data(malJSON.utf8)).variant == .mal)
        #expect(throws: DecodingError.self) {
            _ = try TitansModel(configurationData: Data(#"{"vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2, "memory_depth": 3}"#.utf8))
        }

        let model = try Self.makeModel(variant: .mag, depth: 2)
        let sanitized = model.sanitize(weights: [
            "model.layers.0.neural_memory.q_conv.weight": MLXArray.ones([64, 1, 4]),
            "model.layers.0.attn.q_proj.weight": MLXArray.ones([64, 64]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([8]),
        ])
        #expect(sanitized["model.layers.0.memory.q_conv.conv.weight"]?.shape == [64, 4, 1])
        #expect(sanitized["model.layers.0.self_attn.q_proj.weight"] != nil)
        #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)

        await MLXInferenceEngine.registerAliases()
        for alias in TitansModel.modelTypes {
            #expect(await LLMTypeRegistry.shared.contains(alias))
            let caps = try #require(ModelCapabilities.of(modelType: alias))
            #expect(caps.allows(.promptCacheReuse) && caps.allows(.grammarJumpForward) && !caps.allows(.kvQuantization))
        }
        let created = try await LLMTypeRegistry.shared.createModel(configuration: Data(macJSON.utf8), modelType: "titans_mac")
        #expect((created as? TitansModel)?.configuration.variant == .mac)
    }
}
