import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

private struct CPUModelWeights {
    let tensors: [String: [Float]]

    init(_ model: Module) {
        self.tensors = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map {
                ($0.0, $0.1.asType(.float32).asArray(Float.self))
            }
        )
    }

    subscript(_ key: String) -> [Float]? { tensors[key] }
}

private struct CPUGates {
    let pre: [[Float]]
    let post: [[Float]]
    let res: [[Float]]?
}

private struct TestLoRAPair {
    let path: String
    let a: MLXArray
    let b: MLXArray
}

private enum HyperConnectionCPUReference {
    static func forward(
        _ model: HyperConnectionModel,
        tokens: [Int],
        steerLayer: Int? = nil,
        steerDelta: [Float]? = nil
    ) -> [Float] {
        let w = CPUModelWeights(model)
        let c = model.configuration
        let hc = c.hyperConnection
        let seqLen = tokens.count
        let hidden = c.hiddenSize
        let n = hc.expansionRate
        let embed = w["model.embed_tokens.weight"] ?? []
        var streams = [Float](repeating: 0, count: seqLen * n * hidden)
        for t in 0 ..< seqLen {
            let tokBase = tokens[t] * hidden
            for lane in 0 ..< n {
                let dst = (t * n + lane) * hidden
                for d in 0 ..< hidden {
                    streams[dst + d] = embed[tokBase + d]
                }
            }
        }
        for layer in 0 ..< c.numHiddenLayers {
            streams = runLayer(streams, layer: layer, seqLen: seqLen, config: c, weights: w)
            if steerLayer == layer, let delta = steerDelta {
                streams = applySteering(streams, delta: delta, seqLen: seqLen, n: n, hidden: hidden, mode: hc.collapseMode)
            }
        }
        let collapsed = collapse(streams, seqLen: seqLen, n: n, hidden: hidden, mode: hc.collapseMode)
        let normed = rmsNorm(collapsed, weight: w["model.norm.weight"] ?? [], rows: seqLen, dims: hidden, eps: c.rmsNormEps)
        let headKey = c.tieWordEmbeddings ? "model.embed_tokens.weight" : "lm_head.weight"
        return linear(normed, weight: w[headKey] ?? [], rows: seqLen, inDims: hidden, outDims: c.vocabSize)
    }

    private static func runLayer(
        _ streams: [Float],
        layer: Int,
        seqLen: Int,
        config: HyperConnectionConfiguration,
        weights: CPUModelWeights
    ) -> [Float] {
        let p = "model.layers.\(layer)."
        let d = config.hiddenSize
        let n = config.hyperConnection.expansionRate
        let attnGates = computeGates(streams, prefix: p + "hc_attn.", seqLen: seqLen, config: config, weights: weights)
        let attnIn = preRead(streams, hPre: attnGates.pre, seqLen: seqLen, n: n, hidden: d)
        let attnNormed = rmsNorm(attnIn, weight: weights[p + "input_layernorm.weight"] ?? [], rows: seqLen, dims: d, eps: config.rmsNormEps)
        let attnOut = attention(attnNormed, prefix: p + "self_attn.", seqLen: seqLen, config: config, weights: weights)
        let afterAttn = mixStreams(streams, subOut: attnOut, gates: attnGates, seqLen: seqLen, n: n)
        let mlpGates = computeGates(afterAttn, prefix: p + "hc_mlp.", seqLen: seqLen, config: config, weights: weights)
        let mlpIn = preRead(afterAttn, hPre: mlpGates.pre, seqLen: seqLen, n: n, hidden: d)
        let mlpNormed = rmsNorm(mlpIn, weight: weights[p + "post_attention_layernorm.weight"] ?? [], rows: seqLen, dims: d, eps: config.rmsNormEps)
        let mlpOut = swiGLU(mlpNormed, prefix: p + "mlp.", seqLen: seqLen, config: config, weights: weights)
        return mixStreams(afterAttn, subOut: mlpOut, gates: mlpGates, seqLen: seqLen, n: n)
    }

    private static func computeGates(
        _ streams: [Float],
        prefix: String,
        seqLen: Int,
        config: HyperConnectionConfiguration,
        weights: CPUModelWeights
    ) -> CPUGates {
        let hc = config.hyperConnection
        let n = hc.expansionRate
        let nd = n * config.hiddenSize
        let usesRes = hc.variant != .ihc
        let projDim = usesRes ? (2 * n + n * n) : (2 * n)
        let phi = weights[prefix + "phi_weight"]
        let aPre = (weights[prefix + "alpha_pre"] ?? [0])[0]
        let aPost = (weights[prefix + "alpha_post"] ?? [0])[0]
        let aRes = (weights[prefix + "alpha_res"] ?? [0])[0]
        let bPre = weights[prefix + "bias_pre"] ?? [Float](repeating: 0, count: n)
        let bPost = weights[prefix + "bias_post"] ?? [Float](repeating: 0, count: n)
        let bRes = weights[prefix + "bias_res"] ?? [Float](repeating: 0, count: n * n)
        var preAll: [[Float]] = []
        var postAll: [[Float]] = []
        var resAll: [[Float]] = []
        for t in 0 ..< seqLen {
            let raw = tokenRawLogits(
                streams, token: t, nd: nd, projDim: projDim,
                phi: hc.dynamic ? phi : nil, eps: config.rmsNormEps
            )
            let tildePre = (0 ..< n).map { i in (raw.map { $0[i] * aPre } ?? 0) + bPre[i] }
            let tildePost = (0 ..< n).map { i in (raw.map { $0[n + i] * aPost } ?? 0) + bPost[i] }
            let tildeRes = usesRes ? (0 ..< n * n).map { i in (raw.map { $0[2 * n + i] * aRes } ?? 0) + bRes[i] } : []
            switch hc.variant {
            case .hc:
                preAll.append(tildePre)
                postAll.append(tildePost)
                resAll.append(tildeRes)
            case .mhc:
                let gated = gateVector(pre: tildePre, post: tildePost, mode: hc.gateMode)
                preAll.append(gated.pre)
                postAll.append(gated.post)
                resAll.append(BirkhoffMath.sinkhornCPU(logits: tildeRes, n: n, iterations: hc.sinkhornIterations, eps: hc.sinkhornEps))
            case .ihc:
                let gated = gateVector(pre: tildePre, post: tildePost, mode: hc.gateMode)
                preAll.append(gated.pre)
                postAll.append(gated.post)
            }
        }
        return CPUGates(pre: preAll, post: postAll, res: usesRes ? resAll : nil)
    }

    private static func tokenRawLogits(
        _ streams: [Float],
        token: Int,
        nd: Int,
        projDim: Int,
        phi: [Float]?,
        eps: Float
    ) -> [Float]? {
        guard let phi else { return nil }
        let base = token * nd
        var sqSum: Float = 0
        for k in 0 ..< nd {
            let v = streams[base + k]
            sqSum += v * v
        }
        let invRms = 1 / (sqSum / Float(nd) + eps).squareRoot()
        var out = [Float](repeating: 0, count: projDim)
        for j in 0 ..< projDim {
            var acc: Float = 0
            for k in 0 ..< nd {
                acc += streams[base + k] * phi[k * projDim + j]
            }
            out[j] = acc * invRms
        }
        return out
    }

    private static func gateVector(
        pre: [Float],
        post: [Float],
        mode: HyperConnectionGateMode
    ) -> (pre: [Float], post: [Float]) {
        switch mode {
        case .sigmoid:
            return (pre.map(sigmoidCPU), post.map { 2 * sigmoidCPU($0) })
        case .monographSoftmaxSoftplus:
            return (softmaxCPU(pre), post.map(softplusCPU))
        }
    }
}

extension HyperConnectionCPUReference {
    private static func preRead(
        _ streams: [Float],
        hPre: [[Float]],
        seqLen: Int,
        n: Int,
        hidden: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: seqLen * hidden)
        for t in 0 ..< seqLen {
            for d in 0 ..< hidden {
                var acc: Float = 0
                for lane in 0 ..< n {
                    acc += hPre[t][lane] * streams[(t * n + lane) * hidden + d]
                }
                out[t * hidden + d] = acc
            }
        }
        return out
    }

    private static func mixStreams(
        _ streams: [Float],
        subOut: [Float],
        gates: CPUGates,
        seqLen: Int,
        n: Int
    ) -> [Float] {
        let hidden = subOut.count / max(seqLen, 1)
        var next = [Float](repeating: 0, count: streams.count)
        for t in 0 ..< seqLen {
            let hRes = gates.res?[t]
            for i in 0 ..< n {
                let dst = (t * n + i) * hidden
                let postVal = gates.post[t][i]
                for d in 0 ..< hidden {
                    var mixed: Float = 0
                    if let hRes {
                        for j in 0 ..< n {
                            mixed += hRes[i * n + j] * streams[(t * n + j) * hidden + d]
                        }
                    } else {
                        mixed = streams[dst + d]
                    }
                    next[dst + d] = mixed + postVal * subOut[t * hidden + d]
                }
            }
        }
        return next
    }

    private static func applySteering(
        _ streams: [Float],
        delta: [Float],
        seqLen: Int,
        n: Int,
        hidden: Int,
        mode: HyperConnectionCollapseMode
    ) -> [Float] {
        let scale: Float = mode == .mean ? 1.0 : (1.0 / Float(n))
        var out = streams
        for t in 0 ..< seqLen {
            for lane in 0 ..< n {
                let base = (t * n + lane) * hidden
                for d in 0 ..< hidden {
                    out[base + d] += delta[d] * scale
                }
            }
        }
        return out
    }

    private static func collapse(
        _ streams: [Float],
        seqLen: Int,
        n: Int,
        hidden: Int,
        mode: HyperConnectionCollapseMode
    ) -> [Float] {
        let scale: Float = mode == .mean ? (1.0 / Float(n)) : 1.0
        var out = [Float](repeating: 0, count: seqLen * hidden)
        for t in 0 ..< seqLen {
            for d in 0 ..< hidden {
                var sum: Float = 0
                for lane in 0 ..< n {
                    sum += streams[(t * n + lane) * hidden + d]
                }
                out[t * hidden + d] = sum * scale
            }
        }
        return out
    }
}

extension HyperConnectionCPUReference {
    private static func attention(
        _ x: [Float],
        prefix: String,
        seqLen: Int,
        config: HyperConnectionConfiguration,
        weights: CPUModelWeights
    ) -> [Float] {
        let (d, heads, kvHeads, hd) = (config.hiddenSize, config.numAttentionHeads, config.numKeyValueHeads, config.headDim)
        var q = linear(x, weight: weights[prefix + "q_proj.weight"] ?? [], rows: seqLen, inDims: d, outDims: heads * hd)
        var k = linear(x, weight: weights[prefix + "k_proj.weight"] ?? [], rows: seqLen, inDims: d, outDims: kvHeads * hd)
        let v = linear(x, weight: weights[prefix + "v_proj.weight"] ?? [], rows: seqLen, inDims: d, outDims: kvHeads * hd)
        applyRoPE(&q, seqLen: seqLen, heads: heads, headDim: hd, base: config.ropeTheta)
        applyRoPE(&k, seqLen: seqLen, heads: kvHeads, headDim: hd, base: config.ropeTheta)
        let group = heads / kvHeads
        let scale = 1 / Float(hd).squareRoot()
        var ctx = [Float](repeating: 0, count: seqLen * heads * hd)
        for h in 0 ..< heads {
            let kvh = h / group
            for qi in 0 ..< seqLen {
                var scores = [Float](repeating: 0, count: qi + 1)
                for ki in 0 ... qi {
                    var dot: Float = 0
                    for dim in 0 ..< hd {
                        dot += q[(qi * heads + h) * hd + dim] * k[(ki * kvHeads + kvh) * hd + dim]
                    }
                    scores[ki] = dot * scale
                }
                let probs = softmaxCPU(scores)
                let dst = (qi * heads + h) * hd
                for ki in 0 ... qi {
                    let p = probs[ki]
                    let vBase = (ki * kvHeads + kvh) * hd
                    for dim in 0 ..< hd {
                        ctx[dst + dim] += p * v[vBase + dim]
                    }
                }
            }
        }
        return linear(ctx, weight: weights[prefix + "o_proj.weight"] ?? [], rows: seqLen, inDims: heads * hd, outDims: d)
    }

    private static func swiGLU(
        _ x: [Float],
        prefix: String,
        seqLen: Int,
        config: HyperConnectionConfiguration,
        weights: CPUModelWeights
    ) -> [Float] {
        let (d, inter) = (config.hiddenSize, config.intermediateSize)
        let gate = linear(x, weight: weights[prefix + "gate_proj.weight"] ?? [], rows: seqLen, inDims: d, outDims: inter)
        let up = linear(x, weight: weights[prefix + "up_proj.weight"] ?? [], rows: seqLen, inDims: d, outDims: inter)
        var prod = [Float](repeating: 0, count: seqLen * inter)
        for idx in 0 ..< prod.count {
            prod[idx] = gate[idx] * sigmoidCPU(gate[idx]) * up[idx]
        }
        return linear(prod, weight: weights[prefix + "down_proj.weight"] ?? [], rows: seqLen, inDims: inter, outDims: d)
    }

    private static func applyRoPE(_ tensor: inout [Float], seqLen: Int, heads: Int, headDim: Int, base: Float) {
        let half = headDim / 2
        let invFreq = (0 ..< half).map { idx in Float(Foundation.pow(Double(base), -Double(2 * idx) / Double(headDim))) }
        for t in 0 ..< seqLen {
            for h in 0 ..< heads {
                let baseIdx = (t * heads + h) * headDim
                for k in 0 ..< half {
                    let angle = Double(t) * Double(invFreq[k])
                    let c = Float(Foundation.cos(angle))
                    let s = Float(Foundation.sin(angle))
                    let x1 = tensor[baseIdx + k]
                    let x2 = tensor[baseIdx + half + k]
                    tensor[baseIdx + k] = x1 * c - x2 * s
                    tensor[baseIdx + half + k] = x2 * c + x1 * s
                }
            }
        }
    }
}

extension HyperConnectionCPUReference {
    private static func rmsNorm(_ x: [Float], weight: [Float], rows: Int, dims: Int, eps: Float) -> [Float] {
        var out = [Float](repeating: 0, count: rows * dims)
        for r in 0 ..< rows {
            let base = r * dims
            var sq: Float = 0
            for c in 0 ..< dims {
                let v = x[base + c]
                sq += v * v
            }
            let inv = 1 / (sq / Float(dims) + eps).squareRoot()
            for c in 0 ..< dims {
                out[base + c] = x[base + c] * inv * weight[c]
            }
        }
        return out
    }

    private static func linear(_ x: [Float], weight: [Float], rows: Int, inDims: Int, outDims: Int) -> [Float] {
        var out = [Float](repeating: 0, count: rows * outDims)
        for r in 0 ..< rows {
            let xBase = r * inDims
            let oBase = r * outDims
            for o in 0 ..< outDims {
                let wBase = o * inDims
                var acc: Float = 0
                for k in 0 ..< inDims {
                    acc += x[xBase + k] * weight[wBase + k]
                }
                out[oBase + o] = acc
            }
        }
        return out
    }

    private static func sigmoidCPU(_ x: Float) -> Float {
        Float(1.0 / (1.0 + Foundation.exp(Double(-x))))
    }

    private static func softplusCPU(_ x: Float) -> Float {
        if x > 20 { return x }
        return Float(Foundation.log(1.0 + Foundation.exp(Double(x))))
    }

    private static func softmaxCPU(_ values: [Float]) -> [Float] {
        let top = values.max() ?? 0
        let exps = values.map { Float(Foundation.exp(Double($0 - top))) }
        let sum = exps.reduce(0, +)
        return exps.map { $0 / sum }
    }
}

private final class CaptureSteerProbe: ForwardProbe {
    let steerLayer: Int?
    let steerDelta: MLXArray?
    let maskLayer: Int?
    let mask: MLXArray?
    private(set) var capturedResiduals: [Int: MLXArray] = [:]
    private(set) var capturedHeads: [Int: MLXArray] = [:]

    init(
        steerLayer: Int? = nil,
        steerDelta: MLXArray? = nil,
        maskLayer: Int? = nil,
        mask: MLXArray? = nil
    ) {
        self.steerLayer = steerLayer
        self.steerDelta = steerDelta
        self.maskLayer = maskLayer
        self.mask = mask
    }

    func residual(layer: Int, _ hidden: MLXArray) -> MLXArray {
        capturedResiduals[layer] = hidden
        if steerLayer == layer, let delta = steerDelta {
            return hidden + delta.asType(hidden.dtype)
        }
        return hidden
    }

    func attentionWeights(layer: Int, _ weights: MLXArray) {}
    func wantsAttention(layer: Int) -> Bool { false }
    func headMask(layer: Int) -> MLXArray? { maskLayer == layer ? mask : nil }
    func headOutputs(layer: Int, _ perHead: MLXArray) { capturedHeads[layer] = perHead }
    func routerLogits(layer: Int, _ logits: MLXArray) {}
    func expertMask(layer: Int) -> MLXArray? { nil }
}

@Suite("Hyper-Connections (HC, mHC, iHC) and register-resident Metal Birkhoff Sinkhorn kernel (#844)", .serialized)
struct HyperConnectionTests {
    static let sampleTokens: [Int32] = [5, 19, 42, 7, 88, 13, 64, 21]

    @Test("HyperConnectionSpec and HyperConnectionConfiguration validate invariants and round-trip JSON aliases")
    func specAndConfigValidation() throws {
        try HyperConnectionSpec().validate()
        #expect(throws: BasaltError.self) { try HyperConnectionSpec(expansionRate: 0).validate() }
        #expect(throws: BasaltError.self) { try HyperConnectionSpec(sinkhornIterations: 0).validate() }
        #expect(throws: BasaltError.self) { try HyperConnectionSpec(sinkhornEps: 0).validate() }
        #expect(throws: BasaltError.self) { try HyperConnectionSpec(sinkhornEps: .nan).validate() }
        #expect(throws: BasaltError.self) { try HyperConnectionSpec().with(alphaInit: -0.1).validate() }

        let ihcJSON = """
            {"model_type": "ihc", "vocab_size": 128, "hidden_size": 32, "num_hidden_layers": 2,
             "intermediate_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2}
            """
        let decodedIHC = try JSONDecoder.json5().decode(HyperConnectionConfiguration.self, from: Data(ihcJSON.utf8)).validated()
        #expect(decodedIHC.hyperConnection.variant == .ihc)
        #expect(decodedIHC.headDim == 8)

        let roundTripped = try JSONDecoder().decode(
            HyperConnectionConfiguration.self,
            from: try JSONEncoder().encode(decodedIHC)
        )
        #expect(roundTripped == decodedIHC)

        let badHeads = """
            {"model_type": "mhc", "vocab_size": 128, "hidden_size": 32, "num_hidden_layers": 2,
             "intermediate_size": 64, "num_attention_heads": 5, "num_key_value_heads": 2}
            """
        #expect(throws: BasaltError.self) {
            _ = try JSONDecoder.json5().decode(HyperConnectionConfiguration.self, from: Data(badHeads.utf8)).validated()
        }
    }

    @Test("LLMTypeRegistry and ModelCapabilities wire mhc, hyper_connection, and ihc")
    func registryAndCapabilities() async throws {
        await MLXInferenceEngine.registerAliases()
        for type in HyperConnectionModel.modelTypes {
            #expect(await LLMTypeRegistry.shared.contains(type))
            let caps = try #require(ModelCapabilities.of(modelType: type))
            #expect(caps.allows(.batchedDecode))
            #expect(caps.allows(.promptCacheReuse))
            #expect(caps.allows(.kvQuantization))
            #expect(caps.allows(.quest))
            #expect(caps.allows(.speculativeLookup))
        }
        let created = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(TinyModel.Family.mhc.config.utf8),
            modelType: HyperConnectionModel.modelType
        )
        #expect(created is HyperConnectionModel)
    }

    @Test("Birkhoff invariants and Metal-vs-MLX-vs-CPU parity hold across n in 2...8 and fp32/bf16/fp16 (#228)",
          arguments: [2, 3, 4, 5, 6, 7, 8], [DType.float32, .bfloat16, .float16])
    func birkhoffInvariantsAndParity(n: Int, dtype: DType) {
        let count = 16
        let fp32Logits = withRandomState(MLXRandom.RandomState(seed: UInt64(8440 + n))) {
            MLXRandom.normal([count, n, n], dtype: .float32) * Float(0.8)
        }
        let input = fp32Logits.asType(dtype)
        #expect(HyperConnectionSinkhornKernel.isEligible(logits: input, iterations: 20))
        let metalOut = HyperConnectionSinkhornKernel.project(logits: input, iterations: 20, eps: 1e-6, useMetal: true)
        let mlxOut = HyperConnectionSinkhornKernel.project(logits: input, iterations: 20, eps: 1e-6, useMetal: false)
        #expect(metalOut.dtype == .float32)
        #expect(metalOut.shape == [count, n, n])
        #expect(MLX.abs(metalOut - mlxOut).max().item(Float.self) < 1e-5)

        let metalValues = metalOut.asArray(Float.self)
        let quantizedInput = input.asType(.float32).asArray(Float.self)
        let unquantizedInput = fp32Logits.asArray(Float.self)
        let nn = n * n
        let unquantTol: Float = dtype == .float32 ? 1e-5 : 2e-3
        for m in 0 ..< count {
            let slice = Array(metalValues[(m * nn) ..< ((m + 1) * nn)])
            let dev = BirkhoffMath.rowColDeviations(matrix: slice, n: n)
            #expect(dev.minEntry >= 0)
            #expect(dev.maxRowError < 1e-5)
            #expect(dev.maxColError < 1e-5)

            let cpuFromQuant = BirkhoffMath.sinkhornCPU(
                logits: Array(quantizedInput[(m * nn) ..< ((m + 1) * nn)]),
                n: n, iterations: 20, eps: 1e-6
            )
            let cpuFromFP32 = BirkhoffMath.sinkhornCPU(
                logits: Array(unquantizedInput[(m * nn) ..< ((m + 1) * nn)]),
                n: n, iterations: 20, eps: 1e-6
            )
            let maxDiffQuant = zip(slice, cpuFromQuant).map { Swift.abs($0 - $1) }.max() ?? 0
            let maxDiffFP32 = zip(slice, cpuFromFP32).map { Swift.abs($0 - $1) }.max() ?? 0
            #expect(maxDiffQuant < 1e-5)
            #expect(maxDiffFP32 < unquantTol)
        }
    }

    @Test("Expansion rate n = 12 falls back cleanly to pure-MLX Sinkhorn and matches CPU reference (#228)")
    func largeExpansionFallbackParity() {
        let n = 12
        let logits = withRandomState(MLXRandom.RandomState(seed: 84412)) {
            MLXRandom.normal([3, n, n], dtype: .float32) * Float(0.5)
        }
        #expect(!HyperConnectionSinkhornKernel.isEligible(logits: logits, iterations: 20))
        let out = HyperConnectionSinkhornKernel.project(logits: logits, iterations: 20, eps: 1e-6, useMetal: true)
        let values = out.asArray(Float.self)
        let raw = logits.asArray(Float.self)
        let nn = n * n
        for m in 0 ..< 3 {
            let slice = Array(values[(m * nn) ..< ((m + 1) * nn)])
            let dev = BirkhoffMath.rowColDeviations(matrix: slice, n: n)
            #expect(dev.minEntry >= 0)
            #expect(dev.maxRowError < 1e-5)
            #expect(dev.maxColError < 1e-5)
            let cpu = BirkhoffMath.sinkhornCPU(
                logits: Array(raw[(m * nn) ..< ((m + 1) * nn)]),
                n: n, iterations: 20, eps: 1e-6
            )
            #expect((zip(slice, cpu).map { Swift.abs($0 - $1) }.max() ?? 0) < 1e-5)
        }
    }

    @Test("Sinkhorn handles CPU default device, n = 1, constant-space size < 8, empty tensors, and Int32 iteration overflow")
    func sinkhornEdgeGuardsAndCPUFallback() {
        let small = MLXArray([Float(0.25), -0.5, 0.75, 0.1], [1, 2, 2])
        let gpuOut = HyperConnectionSinkhornKernel.project(logits: small, iterations: 12, eps: 1e-6, useMetal: true)
        let cpuOut = Device.withDefaultDevice(.cpu) {
            #expect(!HyperConnectionSinkhornKernel.isEligible(logits: small, iterations: 12))
            return HyperConnectionSinkhornKernel.project(logits: small, iterations: 12, eps: 1e-6, useMetal: true)
        }
        #expect(MLX.abs(gpuOut - cpuOut).max().item(Float.self) < 1e-5)

        let one = HyperConnectionSinkhornKernel.project(logits: MLXArray([Float(1.5)], [1, 1, 1]), iterations: 5)
        #expect(Swift.abs(one.item(Float.self) - 1.0) < 1e-5)

        let empty = HyperConnectionSinkhornKernel.project(logits: MLXArray.zeros([0, 4, 4], dtype: .float32))
        #expect(empty.shape == [0, 4, 4])

        #expect(!HyperConnectionSinkhornKernel.isEligible(logits: small, iterations: Int(Int32.max) + 1))
    }

    @Test("64-layer composite Amax Gain stays <= 1 + 1e-4 for mHC, == 1.0 for iHC, and explodes (> 10) for unconstrained HC")
    func sixtyFourLayerCompositeAmaxGain() {
        let n = 4
        let layers = 64
        let logits = withRandomState(MLXRandom.RandomState(seed: 84464)) {
            MLXRandom.normal([layers, n, n], dtype: .float32) * Float(0.6)
        }
        let projected = HyperConnectionSinkhornKernel.project(logits: logits, iterations: 20, eps: 1e-6, useMetal: true)
            .asArray(Float.self)
        let nn = n * n
        let mhcMats = (0 ..< layers).map { i in Array(projected[(i * nn) ..< ((i + 1) * nn)]) }
        let mhcGain = BirkhoffMath.amaxGain(matrix: BirkhoffMath.compose(matrices: mhcMats, n: n), n: n)
        #expect(mhcGain <= 1.0 + 1e-4)

        let ihcMats = (0 ..< layers).map { _ in BirkhoffMath.identity(n: n) }
        let ihcGain = BirkhoffMath.amaxGain(matrix: BirkhoffMath.compose(matrices: ihcMats, n: n), n: n)
        #expect(ihcGain == 1.0)

        let unconstrained = (MLXArray.eye(n, dtype: .float32).reshaped(1, n, n) + logits).asArray(Float.self)
        let hcMats = (0 ..< layers).map { i in Array(unconstrained[(i * nn) ..< ((i + 1) * nn)]) }
        let hcGain = BirkhoffMath.amaxGain(matrix: BirkhoffMath.compose(matrices: hcMats, n: n), n: n)
        #expect(hcGain > 10.0)
    }

    @Test("Listing B.8 monograph gate mode (softmax/softplus static) and iHC zero-Sinkhorn path match closed form")
    func monographListingB8AndIHC() {
        let spec = HyperConnectionSpec(
            variant: .mhc, expansionRate: 4, sinkhornIterations: 8,
            gateMode: .monographSoftmaxSoftplus, collapseMode: .mean
        ).with(dynamic: false)
        let layer = HyperConnectionSublayer(hiddenSize: 16, spec: spec)
        #expect(layer.phiWeight == nil)
        let streams = withRandomState(MLXRandom.RandomState(seed: 84408)) {
            MLXRandom.normal([2, 3, 4, 16], dtype: .float32)
        }
        let gates = layer.computeGates(streams)
        let expectedPre = softmax(layer.biasPre.asType(.float32), axis: -1)
        let expectedPost = MLXNN.softplus(layer.biasPost.asType(.float32))
        #expect(MLX.abs(gates.hPre - expectedPre).max().item(Float.self) < 1e-6)
        #expect(MLX.abs(gates.hPost - expectedPost).max().item(Float.self) < 1e-6)
        #expect(gates.hRes != nil)

        let ihcLayer = HyperConnectionSublayer(
            hiddenSize: 16,
            spec: HyperConnectionSpec(variant: .ihc, expansionRate: 4)
        )
        #expect(ihcLayer.alphaRes == nil)
        #expect(ihcLayer.biasRes == nil)
        let ihcGates = ihcLayer.computeGates(streams)
        #expect(ihcGates.hRes == nil)
        let eye = ihcLayer.resolvedResidualMatrix(for: streams)
        #expect(MLX.abs(eye - MLXArray.eye(4, dtype: .float32)).max().item(Float.self) == 0)
    }
}

extension HyperConnectionTests {
    static func makeSeededModel(
        variant: HyperConnectionVariant,
        gateMode: HyperConnectionGateMode = .sigmoid,
        collapseMode: HyperConnectionCollapseMode = .mean,
        dynamic: Bool = true,
        seed: UInt64 = 84400
    ) throws -> HyperConnectionModel {
        let spec = HyperConnectionSpec(
            variant: variant,
            expansionRate: 4,
            sinkhornIterations: 20,
            sinkhornEps: 1e-6,
            gateMode: gateMode,
            collapseMode: collapseMode
        ).with(dynamic: dynamic, alphaInit: 0.1)
        var config = HyperConnectionConfiguration(
            vocabSize: 128,
            hiddenSize: 32,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            hyperConnection: spec
        )
        config.intermediateSize = 64
        try config.validateModelConfiguration()
        let model = withRandomState(MLXRandom.RandomState(seed: seed)) {
            let m = HyperConnectionModel(config)
            let suffixes = [".phi_weight", ".bias_pre", ".bias_post", ".bias_res"]
            let matching = m.parameters().flattened().filter { key, _ in suffixes.contains { key.hasSuffix($0) } }
            let seeded = matching.map { ($0.0, MLXRandom.normal($0.1.shape, dtype: .float32) * Float(0.25)) }
            m.update(parameters: ModuleParameters.unflattened(seeded))
            return m
        }
        eval(model)
        return model
    }

    @Test("Full HyperConnectionModel matches independent plain-Swift CPU reference across mHC, iHC, and HC (#228)",
          arguments: [HyperConnectionVariant.mhc, .ihc, .hc])
    func fullModelMatchesPlainSwiftCPUReference(variant: HyperConnectionVariant) throws {
        let model = try Self.makeSeededModel(variant: variant, seed: 84401)
        let input = MLXArray(Self.sampleTokens)[.newAxis]
        let logits = model(input, cache: nil).asType(.float32)
        let cpuFlat = HyperConnectionCPUReference.forward(model, tokens: Self.sampleTokens.map(Int.init))
        let cpuLogits = MLXArray(cpuFlat).reshaped(1, Self.sampleTokens.count, model.vocabularySize)
        #expect(MLX.abs(logits - cpuLogits).max().item(Float.self) < 1e-4)
        #expect(argMax(logits, axis: -1).asArray(Int32.self) == argMax(cpuLogits, axis: -1).asArray(Int32.self))

        model.setForcePureMLX(true)
        let pureLogits = model(input, cache: nil).asType(.float32)
        model.setForcePureMLX(false)
        #expect(MLX.abs(logits - pureLogits).max().item(Float.self) < 1e-5)
        #expect(argMax(logits, axis: -1).asArray(Int32.self) == argMax(pureLogits, axis: -1).asArray(Int32.self))
    }

    @Test("Monograph softmax/softplus gate mode, static mode, and sum collapse match plain-Swift CPU reference (#228)")
    func monographGateAndSumCollapseMatchCPUReference() throws {
        let model = try Self.makeSeededModel(
            variant: .mhc,
            gateMode: .monographSoftmaxSoftplus,
            collapseMode: .sum,
            dynamic: false,
            seed: 84402
        )
        let input = MLXArray(Self.sampleTokens)[.newAxis]
        let logits = model(input, cache: nil).asType(.float32)
        let cpuFlat = HyperConnectionCPUReference.forward(model, tokens: Self.sampleTokens.map(Int.init))
        let cpuLogits = MLXArray(cpuFlat).reshaped(1, Self.sampleTokens.count, model.vocabularySize)
        #expect(MLX.abs(logits - cpuLogits).max().item(Float.self) < 1e-4)
        #expect(argMax(logits, axis: -1).asArray(Int32.self) == argMax(cpuLogits, axis: -1).asArray(Int32.self))
    }

    @Test("Cached incremental decode and B = 4 batched decode match uncached and solo forwards (#228)")
    func cachedAndBatchedDecodeParity() throws {
        let model = try #require(try TinyModel.make(.mhc) as? HyperConnectionModel)
        let full = model(MLXArray(Self.sampleTokens)[.newAxis], cache: nil).asType(.float32)
        let cache = try model.newCache(parameters: nil)
        let stepped = concatenated(
            Self.sampleTokens.map { model(MLXArray([$0])[.newAxis], cache: cache).asType(.float32) },
            axis: 1
        )
        #expect(MLX.abs(stepped - full).max().item(Float.self) < 1e-4)
        #expect(argMax(stepped, axis: -1).asArray(Int32.self) == argMax(full, axis: -1).asArray(Int32.self))

        let batchSeqs: [[Int32]] = [
            [5, 19, 42, 7, 88, 13],
            [11, 22, 33, 44, 55, 66],
            [9, 18, 27, 36, 45, 54],
            [3, 14, 15, 92, 65, 35],
        ]
        let batchInput = MLXArray(batchSeqs.flatMap { $0 }).reshaped(4, 6)
        let batchLogits = model(batchInput, cache: nil).asType(.float32)
        for row in 0 ..< 4 {
            let solo = model(MLXArray(batchSeqs[row])[.newAxis], cache: nil).asType(.float32)[0]
            #expect(MLX.abs(batchLogits[row] - solo).max().item(Float.self) < 1e-4)
            #expect(argMax(batchLogits[row], axis: -1).asArray(Int32.self) == argMax(solo, axis: -1).asArray(Int32.self))
        }

        let prompts: [[Int32]] = [
            [5, 19, 42, 7],
            [11, 22, 33, 44, 55],
            [9, 18, 27, 36, 45, 54],
            [3, 14, 15, 92, 65, 35, 89],
        ]
        let nextSteps: [[Int32]] = [[12, 34, 56], [21, 43, 65], [7, 8, 9], [10, 20, 30]]
        let soloCaches = try prompts.map { _ in try model.newCache(parameters: nil) }
        for row in 0 ..< 4 {
            _ = model(MLXArray(prompts[row])[.newAxis], cache: soloCaches[row])
        }
        let batchCaches = (0 ..< model.configuration.numHiddenLayers).map { _ in BatchKVCache() }
        for layer in 0 ..< model.configuration.numHiddenLayers {
            let simpleRows = soloCaches.compactMap { $0[layer] as? KVCacheSimple }
            #expect(simpleRows.count == 4)
            batchCaches[layer].append(simpleRows)
        }
        for step in 0 ..< 3 {
            let stepTokens = MLXArray((0 ..< 4).map { nextSteps[$0][step] }).reshaped(4, 1)
            let bOut = model(stepTokens, cache: batchCaches).asType(.float32)
            for row in 0 ..< 4 {
                let sOut = model(MLXArray([nextSteps[row][step]])[.newAxis], cache: soloCaches[row]).asType(.float32)[0]
                #expect(MLX.abs(bOut[row] - sOut).max().item(Float.self) < 1e-4)
                #expect(argMax(bOut[row], axis: -1).asArray(Int32.self) == argMax(sOut, axis: -1).asArray(Int32.self))
            }
        }
    }
}

extension HyperConnectionTests {
    @Test("ForwardProbe read-only is bit-identical, steering preserves collapse(streams') == edited for mean and sum, and headMask applies")
    func forwardProbeParityAndSteering() throws {
        for mode in HyperConnectionCollapseMode.allCases {
            let model = try Self.makeSeededModel(variant: .mhc, collapseMode: mode, seed: 84410)
            let input = MLXArray(Self.sampleTokens)[.newAxis]
            let baseline = model(input, cache: nil).asType(.float32)

            let readOnly = CaptureSteerProbe()
            let probed = model.forwardWithProbe(input, cache: nil, probe: readOnly).asType(.float32)
            #expect(MLX.abs(probed - baseline).max().item(Float.self) == 0)
            #expect(readOnly.capturedResiduals.count == model.configuration.numHiddenLayers)
            #expect(readOnly.capturedResiduals[0]?.shape == [1, Self.sampleTokens.count, model.configuration.hiddenSize])

            let deltaVals = (0 ..< model.configuration.hiddenSize).map { Float($0 + 1) * 0.05 }
            let delta = MLXArray(deltaVals).reshaped(1, 1, model.configuration.hiddenSize)
            let streams = withRandomState(MLXRandom.RandomState(seed: 84411)) {
                MLXRandom.normal([1, 4, 4, model.configuration.hiddenSize], dtype: .float32)
            }
            let collapsed = HyperConnectionSublayer.collapseStreams(streams, mode: mode)
            let edited = collapsed + delta
            let distributed = HyperConnectionSublayer.distributeProbeDelta(
                streams: streams, collapsed: collapsed, edited: edited, mode: mode
            )
            let recollapsed = HyperConnectionSublayer.collapseStreams(distributed, mode: mode)
            #expect(MLX.abs(recollapsed - edited).max().item(Float.self) < 1e-5)

            let steerProbe = CaptureSteerProbe(steerLayer: 0, steerDelta: delta)
            let steeredLogits = model.forwardWithProbe(input, cache: nil, probe: steerProbe).asType(.float32)
            let cpuSteered = MLXArray(
                HyperConnectionCPUReference.forward(
                    model, tokens: Self.sampleTokens.map(Int.init), steerLayer: 0, steerDelta: deltaVals
                )
            ).reshaped(1, Self.sampleTokens.count, model.vocabularySize)
            #expect(MLX.abs(steeredLogits - cpuSteered).max().item(Float.self) < 1e-4)
            #expect(MLX.abs(steeredLogits - baseline).max().item(Float.self) > 1e-3)

            let headMask = MLXArray([Float(0), 1, 1, 1])
            let maskedProbe = CaptureSteerProbe(maskLayer: 0, mask: headMask)
            let maskedLogits = model.forwardWithProbe(input, cache: nil, probe: maskedProbe).asType(.float32)
            #expect(MLX.abs(maskedLogits - baseline).max().item(Float.self) > 1e-4)
        }
    }

    @Test("LoRA attachment and SuperAdapterComposer adapt attention/MLP projections while leaving manifold @ParameterInfo gates untouched")
    func loraAndSuperAdapterLeaveGatesUntouched() throws {
        let model = try #require(try TinyModel.make(.mhc) as? HyperConnectionModel)
        let linearPaths = model.namedModules().compactMap { path, mod in mod is Linear ? path : nil }
        #expect(!linearPaths.isEmpty)
        #expect(linearPaths.allSatisfy { !$0.contains("hc_attn") && !$0.contains("hc_mlp") })

        let layerLinearKeys = Set(model.loraLayers.flatMap { $0.namedModules() }.compactMap { $0.1 is Linear ? $0.0 : nil })
        #expect(layerLinearKeys == [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        ])

        let gateBefore = Self.gateSnapshot(of: model)
        #expect(gateBefore.count == model.configuration.numHiddenLayers * 2 * 7)

        let loraConfig = try MLXInferenceEngine.attachLoRA(to: model, rank: 2)
        let attachedKeys = Set(loraConfig.loraParameters.keys ?? [])
        #expect(!attachedKeys.isEmpty)
        #expect(attachedKeys.contains("self_attn.q_proj"))
        #expect(attachedKeys.allSatisfy { !$0.contains("hc_attn") && !$0.contains("hc_mlp") })

        let trainableKeys = model.trainableParameters().flattened().map(\.0)
        #expect(!trainableKeys.isEmpty)
        #expect(trainableKeys.allSatisfy { !$0.contains("hc_attn") && !$0.contains("hc_mlp") })
        Self.expectGatesUnchanged(model: model, snapshot: gateBefore)

        try Self.verifySuperAdapterParity()
    }
}

extension HyperConnectionTests {
    private static func gateSnapshot(of model: HyperConnectionModel) -> [String: MLXArray] {
        Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().filter { key, _ in
                key.contains(".hc_attn.") || key.contains(".hc_mlp.")
            }
        )
    }

    private static func expectGatesUnchanged(model: HyperConnectionModel, snapshot: [String: MLXArray]) {
        let after = gateSnapshot(of: model)
        #expect(Set(after.keys) == Set(snapshot.keys))
        for (key, beforeTensor) in snapshot {
            if let afterTensor = after[key] {
                #expect(MLX.abs(afterTensor - beforeTensor).max().item(Float.self) == 0)
            }
        }
    }

    private static func verifySuperAdapterParity() throws {
        let baseModel = try #require(try TinyModel.make(.mhc) as? HyperConnectionModel)
        let gateBefore = gateSnapshot(of: baseModel)
        let targets = ["self_attn.q_proj", "self_attn.v_proj", "mlp.down_proj"]
        let pairsA = makeLoRAPairs(for: baseModel, suffixes: targets, rank: 2, seed: 84421)
        let pairsB = makeLoRAPairs(for: baseModel, suffixes: targets, rank: 2, seed: 84422)
        let dirA = try writeMLXAdapter(pairs: pairsA, rank: 2, scale: 1.5, numLayers: baseModel.loraLayers.count)
        let dirB = try writeMLXAdapter(pairs: pairsB, rank: 2, scale: 2.0, numLayers: baseModel.loraLayers.count)
        defer {
            try? FileManager.default.removeItem(at: dirA)
            try? FileManager.default.removeItem(at: dirB)
        }
        let spec = SuperAdapterSpec(sources: [
            .init(name: "alpha", directory: dirA, weight: 0.5),
            .init(name: "beta", directory: dirB, weight: 0.5),
        ])
        let composed = try SuperAdapterComposer.compose(spec, for: baseModel)
        #expect(composed.configuration.loraParameters.rank == 4)

        let adapted = try #require(try TinyModel.make(.mhc) as? HyperConnectionModel)
        try adapted.load(adapter: composed)
        Self.expectGatesUnchanged(model: adapted, snapshot: gateBefore)

        let merged = try denseMerged(pairsA: pairsA, scaleA: 0.75, pairsB: pairsB, scaleB: 1.0)
        let input = MLXArray(Self.sampleTokens)[.newAxis]
        let adaptedLogits = adapted(input, cache: nil).asType(.float32)
        let mergedLogits = merged(input, cache: nil).asType(.float32)
        let baseLogits = baseModel(input, cache: nil).asType(.float32)
        #expect(MLX.abs(adaptedLogits - mergedLogits).max().item(Float.self) < 1e-4)
        #expect(argMax(adaptedLogits, axis: -1).asArray(Int32.self) == argMax(mergedLogits, axis: -1).asArray(Int32.self))
        #expect(MLX.abs(adaptedLogits - baseLogits).max().item(Float.self) > 1e-4)
    }

    private static func makeLoRAPairs(
        for model: Module,
        suffixes: [String],
        rank: Int,
        seed: UInt64
    ) -> [TestLoRAPair] {
        let linears = model.namedModules()
            .compactMap { path, mod in (mod as? Linear).map { (path, $0) } }
            .filter { path, _ in suffixes.contains { path.hasSuffix(".\($0)") } }
        return withRandomState(MLXRandom.RandomState(seed: seed)) {
            linears.map { path, linear in
                let (outputs, inputs) = linear.shape
                return TestLoRAPair(
                    path: path,
                    a: MLXRandom.normal([rank, inputs]) * Float(0.05),
                    b: MLXRandom.normal([outputs, rank]) * Float(0.05)
                )
            }
        }
    }

    private static func writeMLXAdapter(
        pairs: [TestLoRAPair],
        rank: Int,
        scale: Float,
        numLayers: Int
    ) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appending(path: "mhc-lora-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let keys = Set(pairs.compactMap { PEFTLoRAReader.layerRelative($0.path) }).sorted()
        let config = LoRAConfiguration(
            numLayers: numLayers,
            fineTuneType: .lora,
            loraParameters: .init(rank: rank, scale: scale, keys: keys)
        )
        try JSONEncoder().encode(config).write(to: dir.appending(component: "adapter_config.json"))
        var arrays: [String: MLXArray] = [:]
        for pair in pairs {
            arrays["\(pair.path).lora_a"] = pair.a.T
            arrays["\(pair.path).lora_b"] = pair.b.T
        }
        try MLX.save(arrays: arrays, url: dir.appending(component: "adapters.safetensors"))
        return dir
    }

    private static func denseMerged(
        pairsA: [TestLoRAPair],
        scaleA: Float,
        pairsB: [TestLoRAPair],
        scaleB: Float
    ) throws -> HyperConnectionModel {
        let model = try #require(try TinyModel.make(.mhc) as? HyperConnectionModel)
        var deltas: [String: MLXArray] = [:]
        for pair in pairsA {
            deltas[pair.path] = scaleA * matmul(pair.b, pair.a)
        }
        for pair in pairsB {
            let d = scaleB * matmul(pair.b, pair.a)
            deltas[pair.path] = deltas[pair.path].map { $0 + d } ?? d
        }
        let linears = Dictionary(
            model.namedModules().compactMap { path, mod in (mod as? Linear).map { (path, $0) } },
            uniquingKeysWith: { first, _ in first }
        )
        let updated = deltas.compactMap { path, delta -> (String, MLXArray)? in
            guard let linear = linears[path] else { return nil }
            return ("\(path).weight", linear.weight + delta.asType(linear.weight.dtype))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
        return model
    }
}

extension HyperConnectionTests {
    @Test("mHC Sinkhorn CustomFunction VJP differentiates HyperConnectionModel even with train(false) and handles ±Inf/float64")
    func sinkhornCustomVJPAndEdgeInputs() throws {
        let model = try #require(try TinyModel.make(.mhc) as? HyperConnectionModel)
        #expect(!model.training)
        let tokens = MLXArray(Self.sampleTokens)[.newAxis]
        let (lossMetal, gradsMetal) = valueAndGrad(model: model) { m, arrays in
            [m(arrays[0], cache: nil).asType(.float32).square().mean()]
        }(model, [tokens])

        model.train(true)
        let (lossPure, gradsPure) = valueAndGrad(model: model) { m, arrays in
            [m(arrays[0], cache: nil).asType(.float32).square().mean()]
        }(model, [tokens])
        model.train(false)

        eval(lossMetal, lossPure)
        #expect(lossMetal[0].item(Float.self).isFinite)
        #expect(Swift.abs(lossMetal[0].item(Float.self) - lossPure[0].item(Float.self)) < 1e-4)

        let flatMetal = Dictionary(uniqueKeysWithValues: gradsMetal.flattened())
        let flatPure = Dictionary(uniqueKeysWithValues: gradsPure.flattened())
        let gateKeys = flatMetal.keys.filter { $0.contains("hc_attn") || $0.contains("hc_mlp") }
        #expect(!gateKeys.isEmpty)
        for key in gateKeys {
            let gm = try #require(flatMetal[key])
            let gp = try #require(flatPure[key])
            eval(gm, gp)
            #expect(gm.asArray(Float.self).allSatisfy { $0.isFinite })
            #expect(MLX.abs(gm - gp).max().item(Float.self) < 1e-4)
        }

        let infLogits = MLXArray([Float.infinity, 0.0, -Float.infinity, -Float.infinity], [1, 2, 2])
        let infMetal = HyperConnectionSinkhornKernel.project(logits: infLogits, iterations: 10, eps: 1e-6, useMetal: true)
        let infPure = HyperConnectionSinkhornKernel.project(logits: infLogits, iterations: 10, eps: 1e-6, useMetal: false)
        eval(infMetal, infPure)
        #expect(infMetal.asArray(Float.self).allSatisfy { $0.isFinite })
        #expect(MLX.abs(infMetal - infPure).max().item(Float.self) < 1e-5)

        let f64Logits = MLXArray([Float(0.2), -0.3, 0.4, 0.1], [1, 2, 2]).asType(.float64, stream: .cpu)
        let f64Out = HyperConnectionSinkhornKernel.project(logits: f64Logits, iterations: 10, eps: 1e-6, useMetal: true)
        eval(f64Out)
        #expect(f64Out.shape == [1, 2, 2] && f64Out.asArray(Float.self).allSatisfy { $0.isFinite })
    }
}
