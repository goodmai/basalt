import Foundation
import MLX
import MLXNN
import BasaltCore

final class Zonos3MiniCache {
    var position: Int = 0
    var kv: [Int: (keys: MLXArray, values: MLXArray)] = [:]
    var gdnState: [Int: MLXArray] = [:]
    var convState: [Int: MLXArray] = [:]
}

final class Zonos3WeightHolder: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(_ shape: [Int], scale: Float = 0.02, ones: Bool = false) {
        self._weight.wrappedValue = ones
            ? MLXArray.ones(shape, dtype: .float32)
            : MLXRandom.normal(shape, dtype: .float32) * scale
        super.init()
    }
}

final class Zonos3GDNMixer: Module {
    let heads: Int
    let headDim: Int
    let kernel: Int
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ParameterInfo(key: "out_weight") var outWeight: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ModuleInfo(key: "wo") var wo: Linear

    init(_ c: Zonos3MiniConfig.LM) {
        self.heads = c.gdnHeads
        self.headDim = c.gdnHeadDim
        self.kernel = c.gdnKernel
        let inner = c.gdnHeads * c.gdnHeadDim
        self._inProj.wrappedValue = Linear(c.dim, 4 * inner + 2 * c.gdnHeads, bias: false)
        self._conv.wrappedValue = Conv1d(
            inputChannels: 3 * inner, outputChannels: 3 * inner,
            kernelSize: c.gdnKernel, groups: 3 * inner, bias: false
        )
        self._outWeight.wrappedValue = MLXArray.ones([c.gdnHeadDim], dtype: .float32)
        self._aLog.wrappedValue = MLXArray.zeros([c.gdnHeads], dtype: .float32)
        self._dtBias.wrappedValue = MLXArray.zeros([c.gdnHeads], dtype: .float32)
        self._wo.wrappedValue = Linear(inner, c.dim, bias: false)
        super.init()
    }

    func forward(_ x: MLXArray, layer: Int, cache: Zonos3MiniCache) -> MLXArray {
        let n = x.dim(0)
        let inner = heads * headDim
        let channels = 3 * inner
        let proj = inProj(x)
        let qkv = proj[0..., ..<channels]
        let z = proj[0..., channels ..< (4 * inner)]
        let aa = proj[0..., (4 * inner) ..< (4 * inner + heads)]
        let b = proj[0..., (4 * inner + heads)...]
        let history = cache.convState[layer] ?? MLXArray.zeros([kernel - 1, channels], dtype: x.dtype)
        let sequence = concatenated([history, qkv], axis: 0)
        cache.convState[layer] = sequence[(sequence.dim(0) - (kernel - 1))...]
        let convOut = MLXNN.silu(conv(sequence.expandedDimensions(axis: 0)).squeezed(axis: 0))
        let qRaw = convOut[0..., ..<inner].reshaped(n, heads, headDim).asType(.float32)
        let kRaw = convOut[0..., inner ..< (2 * inner)].reshaped(n, heads, headDim).asType(.float32)
        let v = convOut[0..., (2 * inner)...].reshaped(n, heads, headDim).asType(.float32)
        let q = qRaw * rsqrt(square(qRaw).sum(axis: -1, keepDims: true) + 1e-6)
        let k = kRaw * rsqrt(square(kRaw).sum(axis: -1, keepDims: true) + 1e-6)
        let g = -exp(aLog.asType(.float32)).reshaped(1, heads) * MLXNN.softplus(aa.asType(.float32) + dtBias.asType(.float32))
        let beta = sigmoid(b.asType(.float32))
        let (recOut, nextState) = Self.recurrentGatedDeltaRule(
            q: q, k: k, v: v, g: g, beta: beta, initialState: cache.gdnState[layer]
        )
        cache.gdnState[layer] = nextState
        let normed = MLXFast.rmsNorm(recOut, weight: outWeight.asType(.float32), eps: 1e-5)
        let gated = (normed * MLXNN.silu(z.reshaped(n, heads, headDim).asType(.float32)))
            .asType(x.dtype)
            .reshaped(n, inner)
        return wo(gated)
    }

    static func recurrentGatedDeltaRule(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, initialState: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let n = q.dim(0), heads = q.dim(1), hd = q.dim(2)
        let qScaled = q * (1.0 / Float(hd).squareRoot())
        var state = initialState?.asType(.float32) ?? MLXArray.zeros([heads, hd, hd], dtype: .float32)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(n)
        for t in 0 ..< n {
            let qt = qScaled[t], kt = k[t], vt = v[t]
            let decay = exp(g[t]).reshaped(heads, 1, 1)
            state = state * decay
            let kvMem = (state * kt.reshaped(heads, hd, 1)).sum(axis: -2)
            let delta = (vt - kvMem) * beta[t].reshaped(heads, 1)
            state = state + matmul(kt.reshaped(heads, hd, 1), delta.reshaped(heads, 1, hd))
            let outT = (state * qt.reshaped(heads, hd, 1)).sum(axis: -2)
            outputs.append(outT)
            if t > 0, t.isMultiple(of: 32) { eval(state) }
        }
        return (stacked(outputs, axis: 0), state)
    }
}

final class Zonos3AttentionMixer: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let dim: Int
    let ropeTheta: Float
    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wkv") var wkv: Zonos3WeightHolder
    @ModuleInfo(key: "wo") var wo: Linear
    @ParameterInfo(key: "temp") var temp: MLXArray
    @ModuleInfo(key: "gater") var gater: Linear

    init(_ c: Zonos3MiniConfig.LM) {
        self.nHeads = c.nHeads
        self.nKVHeads = c.nKVHeads
        self.headDim = c.headDim
        self.dim = c.dim
        self.ropeTheta = c.ropeTheta
        self._wq.wrappedValue = Linear(c.dim, c.nHeads * c.headDim, bias: false)
        self._wkv.wrappedValue = Zonos3WeightHolder([2, c.nKVHeads * c.headDim, c.dim])
        self._wo.wrappedValue = Linear(c.nHeads * c.headDim, c.dim, bias: false)
        self._temp.wrappedValue = MLXArray.ones([1, c.nHeads, 1], dtype: .float32)
        self._gater.wrappedValue = Linear(c.dim, c.nHeads, bias: false)
        super.init()
    }

    func forward(_ x: MLXArray, layer: Int, cache: Zonos3MiniCache) -> MLXArray {
        let n = x.dim(0)
        let kvWidth = nKVHeads * headDim
        let gate = sigmoid(gater(x))
        var q = wq(x).reshaped(n, nHeads, headDim)
        let wkvFlat = wkv.weight.reshaped(2 * kvWidth, dim).asType(x.dtype)
        let kv = matmul(x, wkvFlat.transposed(1, 0))
        var k = kv[0..., ..<kvWidth].reshaped(n, nKVHeads, headDim)
        let v = kv[0..., kvWidth...].reshaped(n, nKVHeads, headDim)
        let ones = MLXArray.ones([headDim], dtype: q.dtype)
        q = MLXFast.rmsNorm(q, weight: ones, eps: 1e-6) * abs(temp.asType(q.dtype)).reshaped(1, nHeads, 1)
        k = MLXFast.rmsNorm(k, weight: ones, eps: 1e-6)
        let positions = MLXArray((cache.position ..< (cache.position + n)).map(Float.init))
        q = Self.ropeInterleaved(q, positions: positions, theta: ropeTheta)
        k = Self.ropeInterleaved(k, positions: positions, theta: ropeTheta)
        let fullK: MLXArray
        let fullV: MLXArray
        if let prev = cache.kv[layer] {
            fullK = concatenated([prev.keys, k], axis: 0)
            fullV = concatenated([prev.values, v], axis: 0)
        } else {
            fullK = k
            fullV = v
        }
        cache.kv[layer] = (fullK, fullV)
        let qBHL = q.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let kBHL = fullK.asType(q.dtype).transposed(1, 0, 2).expandedDimensions(axis: 0)
        let vBHL = fullV.asType(q.dtype).transposed(1, 0, 2).expandedDimensions(axis: 0)
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if n <= 1 {
            maskMode = .none
        } else if cache.position == 0 {
            maskMode = .causal
        } else {
            let total = fullK.dim(0)
            let qIdx = MLXArray((cache.position ..< (cache.position + n)).map(Int32.init)).reshaped(n, 1)
            let kIdx = MLXArray((0 ..< total).map(Int32.init)).reshaped(1, total)
            let m = which(kIdx .<= qIdx, MLXArray(Float(0)), MLXArray(-Float.infinity)).asType(q.dtype)
            maskMode = .array(m.reshaped(1, 1, n, total))
        }
        let attn = MLXFast.scaledDotProductAttention(
            queries: qBHL, keys: kBHL, values: vBHL,
            scale: 1.0 / Float(headDim).squareRoot(), mask: maskMode
        ).squeezed(axis: 0).transposed(1, 0, 2)
        let gated = (attn * gate.expandedDimensions(axis: -1)).reshaped(n, nHeads * headDim)
        return wo(gated)
    }

    static func ropeInterleaved(_ x: MLXArray, positions: MLXArray, theta: Float) -> MLXArray {
        let hd = x.dim(-1)
        let half = hd / 2
        let freqs = (0 ..< half).map { idx in pow(theta, -Float(2 * idx) / Float(hd)) }
        let angles = matmul(positions.reshaped(-1, 1), MLXArray(freqs).reshaped(1, half))
        let cosA = cos(angles).expandedDimensions(axis: 1)
        let sinA = sin(angles).expandedDimensions(axis: 1)
        let pairs = x.asType(.float32).reshaped(x.dim(0), x.dim(1), half, 2)
        let a = pairs[0..., 0..., 0..., 0]
        let b = pairs[0..., 0..., 0..., 1]
        let rot = stacked([a * cosA - b * sinA, a * sinA + b * cosA], axis: -1)
        return rot.reshaped(x.shape).asType(x.dtype)
    }
}

final class Zonos3DenseFFN: Module {
    let ffnHidden: Int
    let dim: Int
    @ModuleInfo(key: "w_in") var wIn: Zonos3WeightHolder
    @ModuleInfo(key: "w_out") var wOut: Linear

    init(_ c: Zonos3MiniConfig.LM) {
        self.ffnHidden = c.ffnHidden
        self.dim = c.dim
        self._wIn.wrappedValue = Zonos3WeightHolder([2, c.ffnHidden, c.dim])
        self._wOut.wrappedValue = Linear(c.ffnHidden, c.dim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let wFlat = wIn.weight.reshaped(2 * ffnHidden, dim).asType(x.dtype)
        let flat = matmul(x, wFlat.transposed(1, 0))
        return wOut(flat[0..., ..<ffnHidden] * MLXNN.silu(flat[0..., ffnHidden...]))
    }
}

final class Zonos3MoEExperts: Module {
    @ParameterInfo(key: "gate_up_proj") var gateUpProj: MLXArray
    @ParameterInfo(key: "down_proj") var downProj: MLXArray

    init(_ c: Zonos3MiniConfig.LM) {
        self._gateUpProj.wrappedValue = MLXRandom.normal([c.nExperts, 2 * c.moeHidden, c.moeRoutedDim], dtype: .float32) * 0.02
        self._downProj.wrappedValue = MLXRandom.normal([c.nExperts, c.moeRoutedDim, c.moeHidden], dtype: .float32) * 0.02
        super.init()
    }
}

final class Zonos3MoERouter: Module {
    @ModuleInfo(key: "router_linear") var routerLinear: Linear
    @ModuleInfo(key: "rmsnorm_eda") var rmsnormEda: Zonos3WeightHolder
    @ParameterInfo(key: "balancing_biases") var balancingBiases: MLXArray

    init(_ c: Zonos3MiniConfig.LM) {
        self._routerLinear.wrappedValue = Linear(c.dim, c.nExperts, bias: false)
        self._rmsnormEda.wrappedValue = Zonos3WeightHolder([c.dim], ones: true)
        self._balancingBiases.wrappedValue = MLXArray.zeros([c.nExperts], dtype: .float32)
        super.init()
    }
}

final class Zonos3LatentMoE: Module {
    let moeHidden: Int
    let moeTopK: Int
    let normEps: Float
    @ModuleInfo(key: "experts") var experts: Zonos3MoEExperts
    @ModuleInfo(key: "latent_down") var latentDown: Linear
    @ModuleInfo(key: "latent_up") var latentUp: Linear
    @ModuleInfo(key: "router") var router: Zonos3MoERouter

    init(_ c: Zonos3MiniConfig.LM) {
        self.moeHidden = c.moeHidden
        self.moeTopK = c.moeTopK
        self.normEps = c.normEps
        self._experts.wrappedValue = Zonos3MoEExperts(c)
        self._latentDown.wrappedValue = Linear(c.dim, c.moeRoutedDim, bias: false)
        self._latentUp.wrappedValue = Linear(c.moeRoutedDim, c.dim, bias: false)
        self._router.wrappedValue = Zonos3MoERouter(c)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = MLXFast.rmsNorm(x, weight: router.rmsnormEda.weight.asType(x.dtype), eps: normEps)
        let logits = router.routerLinear(normed).asType(.float32)
        let biased = logits - router.balancingBiases.asType(.float32).reshaped(1, -1)
        let sortedIdx = argSort(-biased, axis: -1)
        let choices = sortedIdx[0..., ..<moeTopK]
        let chosenLogits = takeAlong(logits, choices, axis: -1)
        let weights = softmax(chosenLogits, axis: -1)
        let latent = latentDown(x)
        let n = latent.dim(0), rDim = latent.dim(1)
        var chunks: [MLXArray] = []
        for start in stride(from: 0, to: n, by: 16) {
            let end = min(n, start + 16)
            let m = end - start
            let subChoices = choices[start ..< end].reshaped(m * moeTopK)
            let subLatent = repeated(latent[start ..< end].expandedDimensions(axis: 1), count: moeTopK, axis: 1)
                .reshaped(m * moeTopK, 1, rDim)
            let guW = experts.gateUpProj[subChoices].asType(latent.dtype).transposed(0, 2, 1)
            let gu = matmul(subLatent, guW).squeezed(axis: 1)
            let act = (MLXNN.silu(gu[0..., ..<moeHidden]) * gu[0..., moeHidden...]).expandedDimensions(axis: 1)
            let dnW = experts.downProj[subChoices].asType(latent.dtype).transposed(0, 2, 1)
            let out = matmul(act, dnW).squeezed(axis: 1).reshaped(m, moeTopK, rDim)
            let weighted = (out.asType(.float32) * weights[start ..< end].expandedDimensions(axis: -1)).asType(latent.dtype)
            chunks.append(weighted.sum(axis: 1))
        }
        return latentUp(concatenated(chunks, axis: 0))
    }
}

final class Zonos3DecoderLayer: Module {
    let usesGDN: Bool
    let isDenseFFN: Bool
    @ModuleInfo(key: "attention_norm") var attentionNorm: Zonos3WeightHolder
    @ModuleInfo(key: "ffn_norm") var ffnNorm: Zonos3WeightHolder
    @ModuleInfo(key: "attention") var attention: Module
    @ModuleInfo(key: "feed_forward") var feedForward: Module

    init(_ c: Zonos3MiniConfig.LM, layer: Int) {
        self.usesGDN = c.usesGDN(layer: layer)
        self.isDenseFFN = (layer == 0)
        self._attentionNorm.wrappedValue = Zonos3WeightHolder([c.dim], ones: true)
        self._ffnNorm.wrappedValue = Zonos3WeightHolder([c.dim], ones: true)
        self._attention.wrappedValue = usesGDN ? Zonos3GDNMixer(c) : Zonos3AttentionMixer(c)
        self._feedForward.wrappedValue = isDenseFFN ? Zonos3DenseFFN(c) : Zonos3LatentMoE(c)
        super.init()
    }
}

final class Zonos3MultiEmbedder: Module {
    @ModuleInfo(key: "embedders") var embedders: [Embedding]

    init(_ c: Zonos3MiniConfig.LM) {
        let counts = Array(repeating: c.audioVocab, count: c.nCodebooks) + [c.textRows]
        self._embedders.wrappedValue = counts.map { Embedding(embeddingCount: $0, dimensions: c.dim) }
        super.init()
    }
}

struct Zonos3Conditioning {
    let speakerRow: Int?
    let speakerEmbedding: MLXArray?
    let conditions: [(row: Int, bucket: Int)]

    init(
        speakerRow: Int? = nil,
        speakerEmbedding: MLXArray? = nil,
        conditions: [(row: Int, bucket: Int)] = []
    ) {
        self.speakerRow = speakerRow
        self.speakerEmbedding = speakerEmbedding
        self.conditions = conditions
    }
}

final class Zonos3MiniLMModule: Module {
    let config: Zonos3MiniConfig.LM
    @ModuleInfo(key: "multi_embedder") var multiEmbedder: Zonos3MultiEmbedder
    @ModuleInfo(key: "layers") var layers: [Zonos3DecoderLayer]
    @ModuleInfo(key: "out_norm") var outNorm: Zonos3WeightHolder
    @ModuleInfo(key: "multi_output") var multiOutput: Linear
    @ParameterInfo(key: "block_attnres_queries") var blockAttnResQueries: MLXArray
    @ModuleInfo(key: "speaker_projection") var speakerProjection: Linear
    @ModuleInfo(key: "condition_embeddings") var conditionEmbeddings: [Embedding]

    init(_ c: Zonos3MiniConfig.LM) {
        self.config = c
        self._multiEmbedder.wrappedValue = Zonos3MultiEmbedder(c)
        self._layers.wrappedValue = (0 ..< c.nLayers).map { Zonos3DecoderLayer(c, layer: $0) }
        self._outNorm.wrappedValue = Zonos3WeightHolder([c.dim], ones: true)
        self._multiOutput.wrappedValue = Linear(c.dim, c.nCodebooks * c.audioVocab, bias: false)
        self._blockAttnResQueries.wrappedValue = MLXArray.zeros([2 * c.nLayers + 1, c.dim], dtype: .float32)
        self._speakerProjection.wrappedValue = Linear(c.speakerEmbeddingDim, c.dim, bias: true)
        self._conditionEmbeddings.wrappedValue = c.conditionBucketCounts.map { Embedding(embeddingCount: $0 + 1, dimensions: c.dim) }
        super.init()
    }

    static func fullAttnResRoute(sources: [MLXArray], query: MLXArray, eps: Float) -> MLXArray {
        let bank = stacked(sources, axis: 0)
        let ones = MLXArray.ones([bank.dim(-1)], dtype: bank.dtype)
        let keys = MLXFast.rmsNorm(bank, weight: ones, eps: eps)
        let scores = (keys.asType(.float32) * query.asType(.float32).reshaped(1, 1, -1)).sum(axis: -1)
        let weights = softmax(scores, axis: 0).asType(bank.dtype).expandedDimensions(axis: -1)
        return (bank * weights).sum(axis: 0)
    }

    func forward(
        ids: MLXArray,
        cache: Zonos3MiniCache,
        conditioning: Zonos3Conditioning = Zonos3Conditioning(),
        logitsToKeep: Int = 0
    ) -> MLXArray {
        let lo = cache.position
        let nRows = ids.dim(0)
        var x = multiEmbedder.embedders[0](ids[0..., 0])
        for cb in 1 ... config.nCodebooks {
            x = x + multiEmbedder.embedders[cb](ids[0..., cb])
        }
        x = replaceConditionedRows(x, lo: lo, nRows: nRows, conditioning: conditioning)
        let ones = MLXArray.ones([config.dim], dtype: x.dtype)
        x = MLXFast.rmsNorm(x, weight: ones, eps: config.normEps)
        var sources: [MLXArray] = [x]
        sources.reserveCapacity(2 * config.nLayers + 1)
        for (i, block) in layers.enumerated() {
            let mixedA = Self.fullAttnResRoute(sources: sources, query: blockAttnResQueries[2 * i], eps: config.normEps)
            let attnIn = MLXFast.rmsNorm(mixedA, weight: block.attentionNorm.weight.asType(x.dtype), eps: config.normEps)
            if let gdn = block.attention as? Zonos3GDNMixer {
                sources.append(gdn.forward(attnIn, layer: i, cache: cache))
            } else if let attn = block.attention as? Zonos3AttentionMixer {
                sources.append(attn.forward(attnIn, layer: i, cache: cache))
            }
            let mixedB = Self.fullAttnResRoute(sources: sources, query: blockAttnResQueries[2 * i + 1], eps: config.normEps)
            let ffnIn = MLXFast.rmsNorm(mixedB, weight: block.ffnNorm.weight.asType(x.dtype), eps: config.normEps)
            if let dense = block.feedForward as? Zonos3DenseFFN {
                sources.append(dense(ffnIn))
            } else if let moe = block.feedForward as? Zonos3LatentMoE {
                sources.append(moe(ffnIn))
            }
        }
        let finalMixed = Self.fullAttnResRoute(
            sources: sources, query: blockAttnResQueries[2 * config.nLayers], eps: config.normEps
        )
        var normalized = MLXFast.rmsNorm(finalMixed, weight: outNorm.weight.asType(x.dtype), eps: config.normEps)
        if logitsToKeep > 0, logitsToKeep < normalized.dim(0) {
            normalized = normalized[(normalized.dim(0) - logitsToKeep)...]
        }
        let rawLogits = multiOutput(normalized).reshaped(normalized.dim(0), config.nCodebooks, config.audioVocab)
        cache.position = lo + nRows
        let cap = config.lossSoftcap
        return cap * tanh(rawLogits.asType(.float32) / cap)
    }

    private func replaceConditionedRows(
        _ x: MLXArray,
        lo: Int,
        nRows: Int,
        conditioning: Zonos3Conditioning
    ) -> MLXArray {
        let hi = lo + nRows
        var updated = x
        if let row = conditioning.speakerRow, let emb = conditioning.speakerEmbedding, lo <= row, row < hi {
            let projDType = (speakerProjection as? QuantizedLinear)?.scales.dtype ?? speakerProjection.weight.dtype
            let safeDType = projDType.isFloatingPoint ? projDType : emb.dtype
            let proj = speakerProjection(emb.asType(safeDType).reshaped(1, -1)).asType(x.dtype)
            updated = Self.replaceSingleRow(updated, index: row - lo, replacement: proj)
        }
        for (table, entry) in conditioning.conditions.sorted(by: { $0.row < $1.row }).enumerated()
        where table < conditionEmbeddings.count && lo <= entry.row && entry.row < hi {
            let cond = conditionEmbeddings[table](MLXArray([Int32(entry.bucket)])).asType(x.dtype)
            updated = Self.replaceSingleRow(updated, index: entry.row - lo, replacement: cond)
        }
        return updated
    }

    private static func replaceSingleRow(_ matrix: MLXArray, index: Int, replacement: MLXArray) -> MLXArray {
        let total = matrix.dim(0)
        if total == 1 && index == 0 { return replacement }
        var parts: [MLXArray] = []
        if index > 0 { parts.append(matrix[..<index]) }
        parts.append(replacement)
        if index + 1 < total { parts.append(matrix[(index + 1)...]) }
        return concatenated(parts, axis: 0)
    }
}
