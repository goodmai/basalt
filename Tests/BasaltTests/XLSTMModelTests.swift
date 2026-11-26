import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

private struct XLSTMCPUTensor {
    let shape: [Int]
    let values: [Double]
}

private struct XLSTMCPULayerState {
    let c: [Float]
    let n: [Float]
    let m: [Float]
    let h: [Float]?
}

private struct XLSTMCPUOutput {
    let logits: [Float]
    let states: [XLSTMCPULayerState]
}

private struct XLSTMCPUMLSTMInputs {
    let q: [[Double]]
    let k: [[Double]]
    let v: [[Double]]
    let iGate: [[Double]]
    let fGate: [[Double]]
}

private struct XLSTMCPUSLSTMInputs {
    let z: [[Double]]
    let i: [[Double]]
    let f: [[Double]]
    let o: [[Double]]
    let rZ: [Double]
    let rI: [Double]
    let rF: [Double]
    let rO: [Double]
}

private enum XLSTMCPUMath {
    static func softCap(_ x: Double, cap: Float?) -> Double {
        guard let cap, cap > 0 else { return x }
        let c = Double(cap)
        return c * Foundation.tanh(x / c)
    }

    static func sigmoid(_ x: Double) -> Double {
        if x >= 0 {
            return 1.0 / (1.0 + Foundation.exp(-x))
        }
        let e = Foundation.exp(x)
        return e / (1.0 + e)
    }

    static func silu(_ x: Double) -> Double {
        x * sigmoid(x)
    }

    static func logSigmoid(_ x: Double) -> Double {
        x >= 0 ? -Foundation.log1p(Foundation.exp(-x)) : x - Foundation.log1p(Foundation.exp(x))
    }

    static func linear(_ x: [[Double]], weight: XLSTMCPUTensor, bias: XLSTMCPUTensor?) -> [[Double]] {
        let outDim = weight.shape[0]
        let inDim = weight.shape[1]
        return x.map { row in
            var out = [Double](repeating: 0, count: outDim)
            for o in 0 ..< outDim {
                var sum = bias?.values[o] ?? 0.0
                let base = o * inDim
                for i in 0 ..< inDim {
                    sum += row[i] * weight.values[base + i]
                }
                out[o] = sum
            }
            return out
        }
    }

    static func rmsNorm(_ x: [[Double]], weight: XLSTMCPUTensor, bias: XLSTMCPUTensor?, eps: Float) -> [[Double]] {
        let epsD = Double(eps)
        return x.map { row in
            let ms = row.reduce(0.0) { $0 + $1 * $1 } / Double(row.count)
            let inv = 1.0 / Foundation.sqrt(ms + epsD)
            return row.indices.map { i in
                row[i] * inv * weight.values[i] + (bias?.values[i] ?? 0.0)
            }
        }
    }

    static func multiHeadLayerNorm(
        _ x: [[Double]],
        numHeads: Int,
        headDim: Int,
        weight: XLSTMCPUTensor,
        bias: XLSTMCPUTensor?,
        eps: Float
    ) -> [[Double]] {
        let epsD = Double(eps)
        return x.map { row in
            var out = [Double](repeating: 0, count: numHeads * headDim)
            for h in 0 ..< numHeads {
                let base = h * headDim
                var mu = 0.0
                for j in 0 ..< headDim { mu += row[base + j] }
                mu /= Double(headDim)
                var variance = 0.0
                for j in 0 ..< headDim {
                    let centered = row[base + j] - mu
                    variance += centered * centered
                }
                variance /= Double(headDim)
                let inv = 1.0 / Foundation.sqrt(variance + epsD)
                for j in 0 ..< headDim {
                    let idx = base + j
                    out[idx] = (row[idx] - mu) * inv * weight.values[idx] + (bias?.values[idx] ?? 0.0)
                }
            }
            return out
        }
    }

    static func mlstmRecurrence(
        _ input: XLSTMCPUMLSTMInputs,
        numHeads: Int,
        qkHeadDim: Int,
        vHeadDim: Int,
        eps: Float
    ) -> ([[Double]], XLSTMCPULayerState) {
        let steps = input.q.count
        var c = [Double](repeating: 0.0, count: numHeads * qkHeadDim * vHeadDim)
        var n = [Double](repeating: 0.0, count: numHeads * qkHeadDim)
        var m = [Double](repeating: 0.0, count: numHeads)
        var outputs = [[Double]](repeating: [Double](repeating: 0.0, count: numHeads * vHeadDim), count: steps)
        let qScale = 1.0 / Foundation.sqrt(Double(qkHeadDim))
        let epsD = Double(eps)
        for t in 0 ..< steps {
            for h in 0 ..< numHeads {
                let fLog = logSigmoid(input.fGate[t][h])
                let iVal = input.iGate[t][h]
                let mNext = Swift.max(fLog + m[h], iVal)
                let fAct = Foundation.exp(fLog + m[h] - mNext)
                let iAct = Foundation.exp(iVal - mNext)
                m[h] = mNext
                let qkBase = h * qkHeadDim
                let vBase = h * vHeadDim
                let cBase = h * qkHeadDim * vHeadDim
                var qnDot = 0.0
                for r in 0 ..< qkHeadDim {
                    let kVal = input.k[t][qkBase + r]
                    n[qkBase + r] = fAct * n[qkBase + r] + iAct * kVal
                    qnDot += (input.q[t][qkBase + r] * qScale) * n[qkBase + r]
                    let rowBase = cBase + r * vHeadDim
                    for s in 0 ..< vHeadDim {
                        c[rowBase + s] = fAct * c[rowBase + s] + iAct * kVal * input.v[t][vBase + s]
                    }
                }
                let denom = Swift.max(Swift.abs(qnDot), Foundation.exp(-mNext)) + epsD
                for s in 0 ..< vHeadDim {
                    var num = 0.0
                    for r in 0 ..< qkHeadDim {
                        num += (input.q[t][qkBase + r] * qScale) * c[cBase + r * vHeadDim + s]
                    }
                    outputs[t][vBase + s] = num / denom
                }
            }
        }
        let state = XLSTMCPULayerState(c: c.map(Float.init), n: n.map(Float.init), m: m.map(Float.init), h: nil)
        return (outputs, state)
    }

    static func slstmRecurrence(
        _ input: XLSTMCPUSLSTMInputs,
        config: XLSTMConfiguration
    ) -> ([[Double]], XLSTMCPULayerState) {
        let (steps, heads, headDim) = (input.z.count, config.numHeads, config.slstmHeadDim)
        let total = heads * headDim
        var c = [Double](repeating: 0.0, count: total)
        var n = [Double](repeating: 0.0, count: total)
        var m = [Double](repeating: 0.0, count: total)
        var h = [Double](repeating: 0.0, count: total)
        var outputs = [[Double]](repeating: [Double](repeating: 0.0, count: total), count: steps)
        let epsD = Double(config.eps)
        for t in 0 ..< steps {
            var nextH = [Double](repeating: 0.0, count: total)
            for head in 0 ..< heads {
                let base = head * headDim
                let rBase = head * headDim * headDim
                for col in 0 ..< headDim {
                    var (recZ, recI, recF, recO) = (0.0, 0.0, 0.0, 0.0)
                    for row in 0 ..< headDim {
                        let prev = h[base + row]
                        let wIdx = rBase + row * headDim + col
                        recZ += prev * input.rZ[wIdx]
                        recI += prev * input.rI[wIdx]
                        recF += prev * input.rF[wIdx]
                        recO += prev * input.rO[wIdx]
                    }
                    let idx = base + col
                    let zTilde = input.z[t][idx] + recZ
                    let iTilde = softCap(input.i[t][idx] + recI, cap: config.gateSoftCap)
                    let fTilde = softCap(input.f[t][idx] + recF, cap: config.gateSoftCap)
                    let oTilde = input.o[t][idx] + recO
                    let logF = config.slstmGateMode == .logSigmoid ? logSigmoid(fTilde) : fTilde
                    let mNext = Swift.max(logF + m[idx], iTilde)
                    let iAct = Foundation.exp(iTilde - mNext)
                    let fAct = Foundation.exp(logF + m[idx] - mNext)
                    let cNext = fAct * c[idx] + iAct * Foundation.tanh(zTilde)
                    let nNext = fAct * n[idx] + iAct
                    let denom = nNext + epsD
                    c[idx] = cNext
                    n[idx] = nNext
                    m[idx] = mNext
                    nextH[idx] = sigmoid(oTilde) * (cNext / denom)
                }
            }
            h = nextH
            outputs[t] = h
        }
        let state = XLSTMCPULayerState(c: c.map(Float.init), n: n.map(Float.init), m: m.map(Float.init), h: h.map(Float.init))
        return (outputs, state)
    }
}

private struct XLSTMCPUReference {
    let weights: [String: XLSTMCPUTensor]
    let config: XLSTMConfiguration

    init(_ model: XLSTMModel) {
        var map: [String: XLSTMCPUTensor] = [:]
        for (key, array) in model.parameters().flattened() {
            let f32 = array.asType(.float32)
            eval(f32)
            map[key] = XLSTMCPUTensor(shape: f32.shape, values: f32.asArray(Float.self).map(Double.init))
        }
        self.weights = map
        self.config = model.configuration
    }

    func forward(_ tokens: [Int32]) throws -> XLSTMCPUOutput {
        let embed = try tensor("backbone.embeddings.weight")
        let d = config.hiddenSize
        var h: [[Double]] = tokens.map { token in
            let row = Int(token)
            return Array(embed.values[(row * d) ..< ((row + 1) * d)])
        }
        var states: [XLSTMCPULayerState] = []
        for layer in 0 ..< config.numHiddenLayers {
            let (nextH, state) = try runBlock(h, layer: layer)
            h = nextH
            states.append(state)
        }
        if config.addOutNorm {
            h = XLSTMCPUMath.rmsNorm(h, weight: try tensor("backbone.out_norm.weight"), bias: nil, eps: config.normEps)
        }
        let head = try weights["lm_head.weight"] ?? tensor("backbone.embeddings.weight")
        let rawLogits = XLSTMCPUMath.linear(h, weight: head, bias: nil)
        let capped = rawLogits.flatMap { row in
            row.map { Float(XLSTMCPUMath.softCap($0, cap: config.outputLogitSoftCap)) }
        }
        return XLSTMCPUOutput(logits: capped, states: states)
    }

    private func runBlock(_ x: [[Double]], layer: Int) throws -> ([[Double]], XLSTMCPULayerState) {
        let p = "backbone.blocks.\(layer)."
        let recurrent: [[Double]]
        let state: XLSTMCPULayerState
        if config.isSLSTM(layer: layer) {
            let normed = XLSTMCPUMath.rmsNorm(x, weight: try tensor(p + "norm_slstm.weight"), bias: weights[p + "norm_slstm.bias"], eps: config.normEps)
            (recurrent, state) = try runSLSTM(normed, prefix: p + "slstm_layer.")
        } else {
            let normed = XLSTMCPUMath.rmsNorm(x, weight: try tensor(p + "norm_mlstm.weight"), bias: weights[p + "norm_mlstm.bias"], eps: config.normEps)
            (recurrent, state) = try runMLSTM(normed, prefix: p + "mlstm_layer.")
        }
        let mid = add(x, recurrent)
        let ffnIn = XLSTMCPUMath.rmsNorm(mid, weight: try tensor(p + "norm_ffn.weight"), bias: weights[p + "norm_ffn.bias"], eps: config.normEps)
        let ffnOut = try runFFN(ffnIn, prefix: p + "ffn.")
        return (add(mid, ffnOut), state)
    }

    private func runMLSTM(_ x: [[Double]], prefix p: String) throws -> ([[Double]], XLSTMCPULayerState) {
        let (q, k, v, oPreact, iRaw, fRaw) = try projectMLSTM(x, prefix: p)
        let iCap = iRaw.map { $0.map { XLSTMCPUMath.softCap($0, cap: config.gateSoftCap) } }
        let fCap = fRaw.map { $0.map { XLSTMCPUMath.softCap($0, cap: config.gateSoftCap) } }
        let inputs = XLSTMCPUMLSTMInputs(q: q, k: k, v: v, iGate: iCap, fGate: fCap)
        let (h, state) = XLSTMCPUMath.mlstmRecurrence(
            inputs, numHeads: config.numHeads, qkHeadDim: config.qkHeadDim, vHeadDim: config.vHeadDim, eps: config.eps
        )
        let normed = XLSTMCPUMath.multiHeadLayerNorm(
            h, numHeads: config.numHeads, headDim: config.vHeadDim,
            weight: try tensor(p + "multihead_norm.weight"), bias: weights[p + "multihead_norm.bias"], eps: config.normEps
        )
        let gated = zip(oPreact, normed).map { oRow, nRow in
            zip(oRow, nRow).map { XLSTMCPUMath.sigmoid($0) * $1 }
        }
        let out = XLSTMCPUMath.linear(gated, weight: try tensor(p + "out_proj.weight"), bias: weights[p + "out_proj.bias"])
        return (out, state)
    }

    private func projectMLSTM(
        _ x: [[Double]],
        prefix p: String
    ) throws -> ([[Double]], [[Double]], [[Double]], [[Double]], [[Double]], [[Double]]) {
        switch config.weightMode {
        case .single:
            let q = XLSTMCPUMath.linear(x, weight: try tensor(p + "q.weight"), bias: weights[p + "q.bias"])
            let k = XLSTMCPUMath.linear(x, weight: try tensor(p + "k.weight"), bias: weights[p + "k.bias"])
            let v = XLSTMCPUMath.linear(x, weight: try tensor(p + "v.weight"), bias: weights[p + "v.bias"])
            let o = XLSTMCPUMath.linear(x, weight: try tensor(p + "ogate_preact.weight"), bias: weights[p + "ogate_preact.bias"])
            let i = XLSTMCPUMath.linear(x, weight: try tensor(p + "igate_preact.weight"), bias: weights[p + "igate_preact.bias"])
            let f = XLSTMCPUMath.linear(x, weight: try tensor(p + "fgate_preact.weight"), bias: weights[p + "fgate_preact.bias"])
            return (q, k, v, o, i, f)
        case .fused:
            let qkvo = XLSTMCPUMath.linear(x, weight: try tensor(p + "qkv_opreact.weight"), bias: weights[p + "qkv_opreact.bias"])
            let ifGates = XLSTMCPUMath.linear(x, weight: try tensor(p + "ifgate_preact.weight"), bias: weights[p + "ifgate_preact.bias"])
            let (qD, vD, h) = (config.qkDim, config.vDim, config.numHeads)
            let q = qkvo.map { Array($0[..<qD]) }
            let k = qkvo.map { Array($0[qD ..< (2 * qD)]) }
            let v = qkvo.map { Array($0[(2 * qD) ..< (2 * qD + vD)]) }
            let o = qkvo.map { Array($0[(2 * qD + vD)...]) }
            let i = ifGates.map { Array($0[..<h]) }
            let f = ifGates.map { Array($0[h...]) }
            return (q, k, v, o, i, f)
        }
    }

    private func runSLSTM(_ x: [[Double]], prefix p: String) throws -> ([[Double]], XLSTMCPULayerState) {
        let z = XLSTMCPUMath.linear(x, weight: try tensor(p + "z_proj.weight"), bias: weights[p + "z_proj.bias"])
        let i = XLSTMCPUMath.linear(x, weight: try tensor(p + "i_proj.weight"), bias: weights[p + "i_proj.bias"])
        let f = XLSTMCPUMath.linear(x, weight: try tensor(p + "f_proj.weight"), bias: weights[p + "f_proj.bias"])
        let o = XLSTMCPUMath.linear(x, weight: try tensor(p + "o_proj.weight"), bias: weights[p + "o_proj.bias"])
        let inputs = XLSTMCPUSLSTMInputs(
            z: z, i: i, f: f, o: o,
            rZ: try tensor(p + "r_z").values,
            rI: try tensor(p + "r_i").values,
            rF: try tensor(p + "r_f").values,
            rO: try tensor(p + "r_o").values
        )
        let (h, state) = XLSTMCPUMath.slstmRecurrence(inputs, config: config)
        let normed = XLSTMCPUMath.multiHeadLayerNorm(
            h, numHeads: config.numHeads, headDim: config.slstmHeadDim,
            weight: try tensor(p + "multihead_norm.weight"), bias: weights[p + "multihead_norm.bias"], eps: config.normEps
        )
        let out = XLSTMCPUMath.linear(normed, weight: try tensor(p + "out_proj.weight"), bias: weights[p + "out_proj.bias"])
        return (out, state)
    }

    private func runFFN(_ x: [[Double]], prefix p: String) throws -> [[Double]] {
        let hidden: [[Double]]
        switch config.weightMode {
        case .single:
            let gate = XLSTMCPUMath.linear(x, weight: try tensor(p + "proj_up_gate.weight"), bias: weights[p + "proj_up_gate.bias"])
            let up = XLSTMCPUMath.linear(x, weight: try tensor(p + "proj_up.weight"), bias: weights[p + "proj_up.bias"])
            hidden = zip(gate, up).map { gRow, uRow in zip(gRow, uRow).map { XLSTMCPUMath.silu($0) * $1 } }
        case .fused:
            let fused = XLSTMCPUMath.linear(x, weight: try tensor(p + "proj_up_gate_z.weight"), bias: weights[p + "proj_up_gate_z.bias"])
            let d = config.ffnHiddenDim
            hidden = fused.map { row in (0 ..< d).map { XLSTMCPUMath.silu(row[$0]) * row[d + $0] } }
        }
        return XLSTMCPUMath.linear(hidden, weight: try tensor(p + "proj_down.weight"), bias: weights[p + "proj_down.bias"])
    }

    private func tensor(_ key: String) throws -> XLSTMCPUTensor {
        try #require(weights[key], "Missing CPU reference tensor \(key)")
    }

    private func add(_ a: [[Double]], _ b: [[Double]]) -> [[Double]] {
        zip(a, b).map { zip($0, $1).map(+) }
    }
}

@Suite("xLSTM (mLSTM matrix memory + optional sLSTM scalar memory) model, cache, ops, and registration (#843)", .serialized)
struct XLSTMModelTests {
    static let tokens7: [Int32] = [5, 19, 42, 7, 88, 13, 64]
    static let tokens8: [Int32] = [5, 19, 42, 7, 88, 13, 64, 21]
    static let tokens13: [Int32] = [5, 19, 42, 7, 88, 13, 64, 21, 99, 3, 140, 51, 17]

    static func model(adding extra: String = "") throws -> XLSTMModel {
        try #require(try TinyModel.make(.xlstm, adding: extra) as? XLSTMModel)
    }

    static func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(Float(0)) { Swift.max($0, Swift.abs($1.0 - $1.1)) }
    }

    static func argMaxPerRow(_ logits: [Float], vocab: Int) -> [Int] {
        let rows = logits.count / vocab
        return (0 ..< rows).map { r in
            let base = r * vocab
            var bestIdx = 0
            var bestVal = logits[base]
            for v in 1 ..< vocab where logits[base + v] > bestVal {
                bestVal = logits[base + v]
                bestIdx = v
            }
            return bestIdx
        }
    }

    fileprivate static func assertStatesMatch(_ caches: [any KVCache], _ reference: [XLSTMCPULayerState]) throws {
        #expect(caches.count == reference.count)
        for (layerCache, ref) in zip(caches, reference) {
            let xCache = try #require(layerCache as? XLSTMLayerCache)
            let cArr = try #require(xCache.cState).asType(.float32)
            let nArr = try #require(xCache.nState).asType(.float32)
            let mArr = try #require(xCache.mState).asType(.float32)
            eval(cArr, nArr, mArr)
            #expect(maxAbsDiff(cArr.asArray(Float.self), ref.c) < 1e-4)
            #expect(maxAbsDiff(nArr.asArray(Float.self), ref.n) < 1e-4)
            #expect(maxAbsDiff(mArr.asArray(Float.self), ref.m) < 1e-4)
            if let refH = ref.h {
                let hArr = try #require(xCache.hState).asType(.float32)
                eval(hArr)
                #expect(maxAbsDiff(hArr.asArray(Float.self), refH) < 1e-4)
            }
        }
    }

    @Test("Full mLSTM forward matches independent plain-Swift CPU reference within 1e-4 for logits, states, and exact argmax ids (#228)")
    func fullMLSTMForwardMatchesCPUReference() throws {
        let model = try Self.model()
        let cache = model.newCache(parameters: nil)
        let logits = model(MLXArray(Self.tokens7)[.newAxis], cache: cache).asType(.float32)
        eval(logits)
        let ref = try XLSTMCPUReference(model).forward(Self.tokens7)
        let actual = logits.asArray(Float.self)
        #expect(Self.maxAbsDiff(actual, ref.logits) < 1e-4)
        #expect(Self.argMaxPerRow(actual, vocab: model.configuration.vocabSize) == Self.argMaxPerRow(ref.logits, vocab: model.configuration.vocabSize))
        try Self.assertStatesMatch(cache, ref.states)
    }

    @Test("Chunkwise parallel prefill (exact multiple L=8 and non-multiple tail L=7, L=13) matches single-step L=1 decode and split prefill L=5+8")
    func chunkwisePrefillMatchesSingleStepAndSplitPrefill() throws {
        let model = try Self.model()
        let cfg = model.configuration
        for sequence in [Self.tokens7, Self.tokens8, Self.tokens13] {
            let fullCache = model.newCache(parameters: nil)
            let fullLogits = model(MLXArray(sequence)[.newAxis], cache: fullCache).asType(.float32)
            let stepCache = model.newCache(parameters: nil)
            let stepped = concatenated(sequence.map { model(MLXArray([$0])[.newAxis], cache: stepCache).asType(.float32) }, axis: 1)
            #expect(MLX.abs(fullLogits - stepped).max().item(Float.self) < 1e-4)
            #expect(argMax(fullLogits, axis: -1).asArray(Int32.self) == argMax(stepped, axis: -1).asArray(Int32.self))
            for (a, b) in zip(fullCache, stepCache) {
                let (ca, cb) = (try #require(a as? XLSTMLayerCache), try #require(b as? XLSTMLayerCache))
                #expect(ca.offset == sequence.count && cb.offset == sequence.count)
                #expect(MLX.abs(try #require(ca.cState) - (try #require(cb.cState))).max().item(Float.self) < 1e-4)
                #expect(MLX.abs(try #require(ca.nState) - (try #require(cb.nState))).max().item(Float.self) < 1e-4)
                #expect(MLX.abs(try #require(ca.mState) - (try #require(cb.mState))).max().item(Float.self) < 1e-4)
            }
        }
        let full13Cache = model.newCache(parameters: nil)
        let full13 = model(MLXArray(Self.tokens13)[.newAxis], cache: full13Cache).asType(.float32)
        let splitCache = model.newCache(parameters: nil)
        let partA = model(MLXArray(Array(Self.tokens13[..<5]))[.newAxis], cache: splitCache).asType(.float32)
        let partB = model(MLXArray(Array(Self.tokens13[5...]))[.newAxis], cache: splitCache).asType(.float32)
        let splitCombined = concatenated([partA, partB], axis: 1)
        #expect(MLX.abs(full13 - splitCombined).max().item(Float.self) < 1e-4)
        #expect(argMax(full13, axis: -1).asArray(Int32.self) == argMax(splitCombined, axis: -1).asArray(Int32.self))
        for (a, b) in zip(full13Cache, splitCache) {
            let (ca, cb) = (try #require(a as? XLSTMLayerCache), try #require(b as? XLSTMLayerCache))
            #expect(ca.isTrimmable == false && ca.trim(2) == 0)
            #expect(ca.cState?.shape == [1, cfg.numHeads, cfg.qkHeadDim, cfg.vHeadDim])
            #expect(ca.nState?.shape == [1, cfg.numHeads, cfg.qkHeadDim])
            #expect(ca.mState?.shape == [1, cfg.numHeads, 1])
            let isNoneMask: Bool = if case .none = ca.makeMask(n: 4, windowSize: nil, returnArray: false) { true } else { false }
            #expect(isNoneMask)
            #expect(MLX.abs(try #require(ca.cState) - (try #require(cb.cState))).max().item(Float.self) < 1e-4)
            #expect(MLX.abs(try #require(ca.nState) - (try #require(cb.nState))).max().item(Float.self) < 1e-4)
            #expect(MLX.abs(try #require(ca.mState) - (try #require(cb.mState))).max().item(Float.self) < 1e-4)
        }
    }

    @Test("Extreme positive and negative gate pre-activations remain finite via gate_soft_cap, m_0=0, and exp(-m_t) denominator floor")
    func extremeGateValuesRemainFiniteAndMatchCPUReference() throws {
        let (b, h, l, dQK, dV) = (1, 2, 7, 4, 6)
        let (q, k, v) = withRandomState(MLXRandom.RandomState(seed: 4242)) {
            (MLXRandom.normal([b, h, l, dQK]), MLXRandom.normal([b, h, l, dQK]), MLXRandom.normal([b, h, l, dV]))
        }
        let rawIValues: [Float] = [80, -80, 50, -60, 75, -75, 40, -40, 80, -55, 65, -70, 85, -85]
        let rawFValues: [Float] = [-80, 80, -60, 70, -50, 80, -70, -40, -80, 60, -75, 80, -65, 75]
        let (rawI, rawF) = (MLXArray(rawIValues, [b, h, l]), MLXArray(rawFValues, [b, h, l]))
        let cappedI = XLSTMOps.softCap(rawI, cap: 15.0)
        let cappedF = XLSTMOps.softCap(rawF, cap: 15.0)
        #expect(MLX.abs(cappedI).max().item(Float.self) <= 15.0 + 1e-5)
        #expect(MLX.abs(XLSTMOps.softCap(rawI, cap: nil) - rawI).max().item(Float.self) == 0)

        for (iGate, fGate) in [(cappedI, cappedF), (rawI, rawF)] {
            let inputs = MLSTMInputs(q: q, k: k, v: v, iGate: iGate, fGate: fGate)
            let (chunkH, chunkState) = XLSTMOps.mlstmForward(inputs, state: nil, chunkSize: 4, eps: 1e-6)
            var stepState: MLSTMState?
            var stepOuts: [MLXArray] = []
            for t in 0 ..< l {
                let stepIn = MLSTMInputs(
                    q: q[0..., 0..., t ..< (t + 1), 0...],
                    k: k[0..., 0..., t ..< (t + 1), 0...],
                    v: v[0..., 0..., t ..< (t + 1), 0...],
                    iGate: iGate[0..., 0..., t ..< (t + 1)],
                    fGate: fGate[0..., 0..., t ..< (t + 1)]
                )
                let (hStep, next) = XLSTMOps.mlstmForward(stepIn, state: stepState, chunkSize: 4, eps: 1e-6)
                stepOuts.append(hStep)
                stepState = next
            }
            let steppedH = concatenated(stepOuts, axis: 2)
            #expect(MLX.isFinite(chunkH).all().item(Bool.self))
            #expect(MLX.isFinite(chunkState.c).all().item(Bool.self))
            #expect(MLX.isFinite(chunkState.n).all().item(Bool.self))
            #expect(MLX.isFinite(chunkState.m).all().item(Bool.self))
            #expect(MLX.abs(chunkH - steppedH).max().item(Float.self) < 1e-4)
            let finalStep = try #require(stepState)
            #expect(MLX.abs(chunkState.c - finalStep.c).max().item(Float.self) < 1e-4)
            #expect(MLX.abs(chunkState.n - finalStep.n).max().item(Float.self) < 1e-4)
            #expect(MLX.abs(chunkState.m - finalStep.m).max().item(Float.self) < 1e-4)
            Self.assertMLSTMMatchesCPU(inputs, out: chunkH, state: chunkState)
        }
    }

    private static func rows(_ array: MLXArray, steps: Int) -> [[Double]] {
        let flat = array.asType(.float32)
        eval(flat)
        let vals = flat.asArray(Float.self).map(Double.init)
        let width = vals.count / steps
        return (0 ..< steps).map { Array(vals[($0 * width) ..< (($0 + 1) * width)]) }
    }

    private static func assertMLSTMMatchesCPU(_ inputs: MLSTMInputs, out: MLXArray, state: MLSTMState) {
        let (h, l, dQK, dV) = (inputs.q.dim(1), inputs.q.dim(2), inputs.q.dim(3), inputs.v.dim(3))
        let cpuIn = XLSTMCPUMLSTMInputs(
            q: rows(inputs.q.transposed(0, 2, 1, 3), steps: l),
            k: rows(inputs.k.transposed(0, 2, 1, 3), steps: l),
            v: rows(inputs.v.transposed(0, 2, 1, 3), steps: l),
            iGate: rows(inputs.iGate.transposed(0, 2, 1), steps: l),
            fGate: rows(inputs.fGate.transposed(0, 2, 1), steps: l)
        )
        let (cpuH, cpuState) = XLSTMCPUMath.mlstmRecurrence(cpuIn, numHeads: h, qkHeadDim: dQK, vHeadDim: dV, eps: 1e-6)
        let outFlat = out.transposed(0, 2, 1, 3).asType(.float32)
        eval(outFlat, state.c, state.n, state.m)
        #expect(maxAbsDiff(outFlat.asArray(Float.self), cpuH.flatMap { $0.map(Float.init) }) < 1e-4)
        #expect(maxAbsDiff(state.c.asArray(Float.self), cpuState.c) < 1e-4)
        #expect(maxAbsDiff(state.n.asArray(Float.self), cpuState.n) < 1e-4)
        #expect(maxAbsDiff(state.m.asArray(Float.self), cpuState.m) < 1e-4)
    }
}

extension XLSTMModelTests {
    @Test("Fused and single weight modes produce identical logits via sanitize, and tie_word_embeddings matches CPU reference")
    func fusedAndSingleWeightModesAndTiedEmbeddingsMatch() throws {
        let single = try Self.model()
        let singleWeights = Dictionary(uniqueKeysWithValues: single.parameters().flattened())
        let fused = try Self.model(adding: #""weight_mode": "fused", "#)
        let fusedWeights = fused.sanitize(weights: singleWeights)
        #expect(fusedWeights["backbone.blocks.0.mlstm_layer.qkv_opreact.weight"] != nil)
        #expect(fusedWeights["backbone.blocks.0.mlstm_layer.ifgate_preact.weight"] != nil)
        #expect(fusedWeights["backbone.blocks.0.ffn.proj_up_gate_z.weight"] != nil)
        fused.update(parameters: ModuleParameters.unflattened(fusedWeights))
        eval(fused)
        let singleLogits = single(MLXArray(Self.tokens7)[.newAxis], cache: nil).asType(.float32)
        let fusedLogits = fused(MLXArray(Self.tokens7)[.newAxis], cache: nil).asType(.float32)
        #expect(MLX.abs(singleLogits - fusedLogits).max().item(Float.self) < 1e-4)
        #expect(argMax(singleLogits, axis: -1).asArray(Int32.self) == argMax(fusedLogits, axis: -1).asArray(Int32.self))

        let roundTripped = try Self.model()
        let backToSingle = roundTripped.sanitize(weights: fusedWeights)
        #expect(backToSingle["backbone.blocks.0.mlstm_layer.q.weight"] != nil)
        #expect(backToSingle["backbone.blocks.0.mlstm_layer.igate_preact.bias"] != nil)
        roundTripped.update(parameters: ModuleParameters.unflattened(backToSingle))
        eval(roundTripped)
        let rtLogits = roundTripped(MLXArray(Self.tokens7)[.newAxis], cache: nil).asType(.float32)
        #expect(MLX.abs(singleLogits - rtLogits).max().item(Float.self) < 1e-4)

        let cfg = single.configuration
        let aliasRaw: [String: MLXArray] = [
            "model.embed_tokens.weight": MLXArray.zeros([cfg.vocabSize, cfg.hiddenSize]),
            "xlstm.backbone.blocks.0.mlstm_layer.qkv.weight": MLXArray.ones([2 * cfg.qkDim + cfg.vDim, cfg.hiddenSize]),
            "model.norm.weight": MLXArray.ones([cfg.hiddenSize]),
            "lm_head.weight": MLXArray.ones([cfg.vocabSize, cfg.hiddenSize]),
        ]
        let tied = try Self.model(adding: #""tie_word_embeddings": true, "#)
        #expect(tied.lmHead == nil)
        let sanitizedTied = tied.sanitize(weights: aliasRaw)
        #expect(sanitizedTied["backbone.embeddings.weight"] != nil)
        #expect(sanitizedTied["backbone.blocks.0.mlstm_layer.q.weight"]?.shape == [cfg.qkDim, cfg.hiddenSize])
        #expect(sanitizedTied["backbone.blocks.0.mlstm_layer.v.weight"]?.shape == [cfg.vDim, cfg.hiddenSize])
        #expect(sanitizedTied["backbone.out_norm.weight"] != nil)
        #expect(sanitizedTied["lm_head.weight"] == nil)
        let tiedLogits = tied(MLXArray(Self.tokens7)[.newAxis], cache: nil).asType(.float32)
        eval(tiedLogits)
        let tiedRef = try XLSTMCPUReference(tied).forward(Self.tokens7)
        #expect(Self.maxAbsDiff(tiedLogits.asArray(Float.self), tiedRef.logits) < 1e-4)
        #expect(Self.argMaxPerRow(tiedLogits.asArray(Float.self), vocab: cfg.vocabSize) == Self.argMaxPerRow(tiedRef.logits, vocab: cfg.vocabSize))
    }

    @Test("Hybrid sLSTM blocks match CPU reference, single-step decode, and split prefill in both logsigmoid and exp gate modes",
          arguments: [XLSTMSLSTMGateMode.logSigmoid, .exp])
    func hybridSLSTMBlocksMatchCPUReferenceInBothGateModes(mode: XLSTMSLSTMGateMode) throws {
        let model = try Self.model(adding: #""slstm_at": [1], "slstm_gate_mode": "\#(mode.rawValue)", "#)
        let fullCache = model.newCache(parameters: nil)
        #expect((fullCache[0] as? XLSTMLayerCache)?.kind == .mlstm)
        #expect((fullCache[1] as? XLSTMLayerCache)?.kind == .slstm)
        #expect((fullCache[2] as? XLSTMLayerCache)?.kind == .mlstm)
        let fullLogits = model(MLXArray(Self.tokens7)[.newAxis], cache: fullCache).asType(.float32)
        eval(fullLogits)
        let ref = try XLSTMCPUReference(model).forward(Self.tokens7)
        let actual = fullLogits.asArray(Float.self)
        #expect(Self.maxAbsDiff(actual, ref.logits) < 1e-4)
        #expect(Self.argMaxPerRow(actual, vocab: model.configuration.vocabSize) == Self.argMaxPerRow(ref.logits, vocab: model.configuration.vocabSize))
        try Self.assertStatesMatch(fullCache, ref.states)

        let splitCache = model.newCache(parameters: nil)
        let p1 = model(MLXArray(Array(Self.tokens7[..<3]))[.newAxis], cache: splitCache).asType(.float32)
        let p2 = model(MLXArray(Array(Self.tokens7[3...]))[.newAxis], cache: splitCache).asType(.float32)
        #expect(MLX.abs(fullLogits - concatenated([p1, p2], axis: 1)).max().item(Float.self) < 1e-4)
        let stepCache = model.newCache(parameters: nil)
        let stepped = concatenated(Self.tokens7.map { model(MLXArray([$0])[.newAxis], cache: stepCache).asType(.float32) }, axis: 1)
        #expect(MLX.abs(fullLogits - stepped).max().item(Float.self) < 1e-4)
        let sFull = try #require(fullCache[1] as? XLSTMLayerCache)
        let sStep = try #require(stepCache[1] as? XLSTMLayerCache)
        #expect(sFull.innerState().count == 4)
        #expect(MLX.abs(try #require(sFull.hState) - (try #require(sStep.hState))).max().item(Float.self) < 1e-4)
    }

    @Test("sLSTM logsigmoid and exp gate modes diverge on identical weights")
    func slstmGateModesDiffer() throws {
        let logSig = try Self.model(adding: #""slstm_at": [1], "slstm_gate_mode": "logsigmoid", "#)
        let expModel = try Self.model(adding: #""slstm_at": [1], "slstm_gate_mode": "exp", "#)
        expModel.update(parameters: logSig.parameters())
        eval(expModel)
        let l1 = logSig(MLXArray(Self.tokens7)[.newAxis], cache: nil).asType(.float32)
        let l2 = expModel(MLXArray(Self.tokens7)[.newAxis], cache: nil).asType(.float32)
        #expect(MLX.abs(l1 - l2).max().item(Float.self) > 1e-4)

        let zeroR = MLXArray.zeros([1, 2, 2], dtype: .float32)
        let slstmW = SLSTMRecurrentWeights(rZ: zeroR, rI: zeroR, rF: zeroR, rO: zeroR)
        let slstmIn = SLSTMInputs(
            zIn: MLXArray([Float(0.7), -0.4], [1, 1, 1, 2]),
            iIn: MLXArray([Float(-3.0), -3.0], [1, 1, 1, 2]),
            fIn: MLXArray([Float(0.0), 0.0], [1, 1, 1, 2]),
            oIn: MLXArray([Float(0.5), -0.2], [1, 1, 1, 2])
        )
        let (step1Out, _) = XLSTMOps.slstmRecurrence(slstmIn, weights: slstmW, state: nil, config: logSig.configuration)
        let expectedStep1 = sigmoid(slstmIn.oIn) * tanh(slstmIn.zIn)
        eval(step1Out, expectedStep1)
        #expect(MLX.abs(step1Out - expectedStep1).max().item(Float.self) < 1e-5)
    }
}

extension XLSTMModelTests {
    @Test("XLSTMLayerCache copy isolates state, state/metaState round-trips, HybridPrefixSnapshot replays cleanly, and PromptCacheDiskStore skips custom cache")
    func cacheCopyIsolationHybridSnapshotAndDiskStore() throws {
        let model = try Self.model(adding: #""slstm_at": [1], "#)
        let prefix = Array(Self.tokens7[..<5])
        let cache = model.newCache(parameters: nil)
        _ = model(MLXArray(prefix)[.newAxis], cache: cache)
        let cloned = cache.map { $0.copy() }
        let branchA = model(MLXArray([Int32(91)])[.newAxis], cache: cache).asType(.float32)
        for layer in cache { (layer as? XLSTMLayerCache)?.reset() }
        #expect(cache.allSatisfy { $0.offset == 0 && $0.innerState().isEmpty })
        let branchB = model(MLXArray([Int32(44)])[.newAxis], cache: cloned).asType(.float32)
        let referenceB = model(MLXArray(prefix + [44])[.newAxis], cache: nil)[0..., 5..., 0...].asType(.float32)
        #expect(MLX.abs(branchB - referenceB).max().item(Float.self) < 1e-4)
        #expect(MLX.abs(branchA - branchB).max().item(Float.self) > 1e-4)

        let restored = try cloned.map { source -> any KVCache in
            let src = try #require(source as? XLSTMLayerCache)
            let target = XLSTMLayerCache(kind: src.kind)
            target.metaState = src.metaState
            target.state = src.state
            return target
        }
        let nextCloned = model(MLXArray([Int32(12)])[.newAxis], cache: cloned).asType(.float32)
        let nextRestored = model(MLXArray([Int32(12)])[.newAxis], cache: restored).asType(.float32)
        #expect(MLX.abs(nextCloned - nextRestored).max().item(Float.self) == 0)

        let promptCache = PromptCache(model.newCache(parameters: nil))
        #expect(promptCache.isHybrid)
        let tokens = MLXArray(Self.tokens13)
        let head = try HybridPrefixSnapshot.prefillHead(
            tokens, into: promptCache, plan: CachedPrompt(boundary: 5), model: model, prefill: GenerateParameters().prefill
        )
        let kept = try #require(head.kept)
        #expect(kept.isSnapshot(of: 5))
        let suffix = model(MLXArray(Array(Self.tokens13[5...]))[.newAxis], cache: kept.layers).asType(.float32)
        let direct = model(tokens[.newAxis], cache: nil)[0..., 5..., 0...].asType(.float32)
        #expect(MLX.abs(suffix - direct).max().item(Float.self) < 1e-4)

        let directory = FileManager.default.temporaryDirectory.appending(path: "xlstm-disk-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        var store = PromptCacheDiskStore(directory: directory, byteQuota: 1 << 30, retentionDays: 1)
        #expect(store.store(fingerprint: "fp", model: "tiny-xlstm", namespace: "ns", tokens: prefix.map(Int.init), cache: kept) == nil)
        #expect((try? FileManager.default.contentsOfDirectory(atPath: directory.path))?.isEmpty ?? true)
    }

    @Test("Configuration decodes HF xLSTM-7b and 247M checkpoints, computes exact 128.25 MiB fixedStateBytes, round-trips Codable, and integrates with KVCacheGeometry")
    func configurationAndKVCacheGeometryParity() throws {
        let xlstm7BJSON = """
            {"model_type": "xlstm", "vocab_size": 50304, "embedding_dim": 4096, "num_blocks": 32,
             "num_heads": 8, "qk_dim_factor": 0.5, "v_dim_factor": 1.0, "ffn_proj_factor": 2.667,
             "ffn_round_up_to_multiple_of": 64, "chunk_size": 64, "gate_soft_cap": 15.0,
             "output_logit_soft_cap": 30.0, "norm_eps": 1e-6, "eps": 1e-6, "tie_word_embeddings": false,
             "weight_mode": "single", "use_bias": false, "norm_reduction_force_float32": true}
            """
        let cfg7B = try JSONDecoder.json5().decode(XLSTMConfiguration.self, from: Data(xlstm7BJSON.utf8))
        let expectedBytes: Int64 = 32 * (8 * 256 * 512 + 8 * 256 + 8) * 4
        #expect(cfg7B.hiddenSize == 4096 && cfg7B.numHiddenLayers == 32 && cfg7B.numHeads == 8)
        #expect(cfg7B.qkDim == 2048 && cfg7B.qkHeadDim == 256 && cfg7B.vDim == 4096 && cfg7B.vHeadDim == 512)
        #expect(cfg7B.ffnHiddenDim == 10_944)
        #expect(cfg7B.fixedStateBytes == expectedBytes)
        #expect(Swift.abs(Double(cfg7B.fixedStateBytes) / Double(1 << 20) - 128.25) < 0.01)

        let raw7B = try #require(JSONSerialization.jsonObject(with: Data(xlstm7BJSON.utf8)) as? [String: Any])
        let geom = try #require(KVCacheGeometry(config: raw7B))
        #expect(geom.isFromConfig)
        #expect(geom.elementsPerToken == 0)
        #expect(geom.fixedStateBytes == expectedBytes)
        #expect(geom.bytes(tokens: TokenCount(0)).rawValue == expectedBytes)
        #expect(geom.bytes(tokens: TokenCount(100_000)).rawValue == expectedBytes)

        let xlstm247MJSON = """
            {"model_type": "xlstm", "vocab_size": 50257, "embedding_dim": 768, "num_blocks": 24,
             "num_heads": 4, "qk_dim_factor": 0.5, "v_dim_factor": 1.0, "ffn_proj_factor": 2.667,
             "ffn_round_up_to_multiple_of": 64, "gate_soft_cap": null, "output_logit_soft_cap": null}
            """
        let cfg247M = try JSONDecoder.json5().decode(XLSTMConfiguration.self, from: Data(xlstm247MJSON.utf8))
        #expect(cfg247M.ffnHiddenDim == 2048)
        #expect(cfg247M.gateSoftCap == nil && cfg247M.outputLogitSoftCap == nil)
        let encoded = try JSONEncoder().encode(cfg247M)
        let decoded = try JSONDecoder().decode(XLSTMConfiguration.self, from: encoded)
        #expect(decoded == cfg247M)
    }

    @Test("Invalid xLSTM configurations are refused with DecodingError", arguments: [
        #"{"vocab_size": 0, "hidden_size": 64, "num_hidden_layers": 2, "num_heads": 4}"#,
        #"{"vocab_size": 256, "hidden_size": 0, "num_hidden_layers": 2, "num_heads": 4}"#,
        #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 0, "num_heads": 4}"#,
        #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "num_heads": 0}"#,
        #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "num_heads": 6}"#,
        #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "num_heads": 4, "chunk_size": 0}"#,
        #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "num_heads": 4, "slstm_at": [2]}"#,
        #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "num_heads": 4, "gate_soft_cap": 0}"#,
        #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "num_heads": 4, "eps": 0}"#,
    ])
    func invalidConfigRefused(_ json: String) {
        #expect(throws: DecodingError.self) {
            _ = try XLSTMModel(configurationData: Data(json.utf8))
        }
    }

    @Test("LLMTypeRegistry registers xlstm, basalt models types reports it loadable, ModelCapabilities classifies it as recurrent, and TinyModel builds it")
    func registryCapabilitiesAndTinyModelSupportXLSTM() async throws {
        await MLXInferenceEngine.registerAliases()
        #expect(await LLMTypeRegistry.shared.contains("xlstm"))
        let loadable = await MLXInferenceEngine.loadableModelTypes(among: ["xlstm", "not_a_model"])
        #expect(loadable == ["xlstm"])
        let created = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(TinyModel.Family.xlstm.config.utf8),
            modelType: "xlstm"
        )
        #expect(created is XLSTMModel)
        #expect(try TinyModel.make(.xlstm) is XLSTMModel)
        let capabilities = try #require(ModelCapabilities.of(modelType: "xlstm"))
        #expect(capabilities.allows(.promptCacheReuse))
        #expect(capabilities.allows(.grammarJumpForward))
        #expect(!capabilities.allows(.kvQuantization))
        #expect(throws: BasaltError.self) {
            try ModelCapabilities.refuseUnverifiedKVBits(.q4, modelType: "xlstm", path: "tiny-xlstm")
        }
    }
}
