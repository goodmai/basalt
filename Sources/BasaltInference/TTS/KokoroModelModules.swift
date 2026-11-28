import Foundation
import MLX
import MLXNN

final class KokoroBiLSTM: Module {
    @ParameterInfo(key: "weight_ih_l0") var weightIH: MLXArray
    @ParameterInfo(key: "weight_hh_l0") var weightHH: MLXArray
    @ParameterInfo(key: "bias_ih_l0") var biasIH: MLXArray
    @ParameterInfo(key: "bias_hh_l0") var biasHH: MLXArray
    @ParameterInfo(key: "weight_ih_l0_reverse") var weightIHRev: MLXArray
    @ParameterInfo(key: "weight_hh_l0_reverse") var weightHHRev: MLXArray
    @ParameterInfo(key: "bias_ih_l0_reverse") var biasIHRev: MLXArray
    @ParameterInfo(key: "bias_hh_l0_reverse") var biasHHRev: MLXArray
    let hiddenSize: Int

    init(inputSize: Int, hiddenSize: Int) {
        self.hiddenSize = hiddenSize
        let gates = hiddenSize * 4
        self._weightIH.wrappedValue = MLXArray.zeros([gates, inputSize], dtype: .float32)
        self._weightHH.wrappedValue = MLXArray.zeros([gates, hiddenSize], dtype: .float32)
        self._biasIH.wrappedValue = MLXArray.zeros([gates], dtype: .float32)
        self._biasHH.wrappedValue = MLXArray.zeros([gates], dtype: .float32)
        self._weightIHRev.wrappedValue = MLXArray.zeros([gates, inputSize], dtype: .float32)
        self._weightHHRev.wrappedValue = MLXArray.zeros([gates, hiddenSize], dtype: .float32)
        self._biasIHRev.wrappedValue = MLXArray.zeros([gates], dtype: .float32)
        self._biasHHRev.wrappedValue = MLXArray.zeros([gates], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let fwd = runDirection(x, wIH: weightIH, wHH: weightHH, bIH: biasIH, bHH: biasHH, reverse: false)
        let rev = runDirection(x, wIH: weightIHRev, wHH: weightHHRev, bIH: biasIHRev, bHH: biasHHRev, reverse: true)
        return concatenated([fwd, rev], axis: -1)
    }

    private func runDirection(
        _ x: MLXArray,
        wIH: MLXArray,
        wHH: MLXArray,
        bIH: MLXArray,
        bHH: MLXArray,
        reverse: Bool
    ) -> MLXArray {
        let batch = x.dim(0)
        let steps = x.dim(1)
        let projected = matmul(x, wIH.transposed()) + (bIH + bHH)
        let wHHT = wHH.transposed()
        var h = MLXArray.zeros([batch, hiddenSize], dtype: x.dtype)
        var c = MLXArray.zeros([batch, hiddenSize], dtype: x.dtype)
        var outputs = [MLXArray]()
        outputs.reserveCapacity(steps)
        let indices: [Int] = reverse ? Array((0 ..< steps).reversed()) : Array(0 ..< steps)
        for t in indices {
            let gates = projected[0..., t, 0...] + matmul(h, wHHT)
            let parts = split(gates, parts: 4, axis: -1)
            let i = sigmoid(parts[0])
            let f = sigmoid(parts[1])
            let g = tanh(parts[2])
            let o = sigmoid(parts[3])
            c = f * c + i * g
            h = o * tanh(c)
            outputs.append(h)
        }
        if reverse { outputs.reverse() }
        return stacked(outputs, axis: 1)
    }
}

final class KokoroLayerNorm1d: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray
    @ParameterInfo(key: "beta") var beta: MLXArray
    let eps: Float = 1e-5

    init(channels: Int) {
        self._gamma.wrappedValue = MLXArray.ones([channels], dtype: .float32)
        self._beta.wrappedValue = MLXArray.zeros([channels], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = x.variance(axis: -1, keepDims: true)
        return ((x - mean) * rsqrt(variance + eps)) * gamma + beta
    }
}

final class KokoroAdaIN1d: Module {
    @ModuleInfo(key: "fc") var fc: Linear
    let channels: Int

    init(styleDim: Int, channels: Int) {
        self.channels = channels
        self._fc.wrappedValue = Linear(styleDim, channels * 2)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, style: MLXArray) -> MLXArray {
        let mean = x.mean(axis: 1, keepDims: true)
        let variance = x.variance(axis: 1, keepDims: true)
        let normed = (x - mean) * rsqrt(variance + Float(1e-5))
        let h = fc(style).expandedDimensions(axis: 1)
        let gamma = h[0..., 0..., ..<channels]
        let beta = h[0..., 0..., channels...]
        return (1.0 + gamma) * normed + beta
    }
}

final class KokoroAdaLayerNorm: Module {
    @ModuleInfo(key: "fc") var fc: Linear
    let channels: Int

    init(styleDim: Int, channels: Int) {
        self.channels = channels
        self._fc.wrappedValue = Linear(styleDim, channels * 2)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, style: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = x.variance(axis: -1, keepDims: true)
        let normed = (x - mean) * rsqrt(variance + Float(1e-5))
        let h = fc(style).expandedDimensions(axis: 1)
        let gamma = h[0..., 0..., ..<channels]
        let beta = h[0..., 0..., channels...]
        return (1.0 + gamma) * normed + beta
    }
}

final class KokoroAdainResBlk1d: Module {
    @ModuleInfo(key: "norm1") var norm1: KokoroAdaIN1d
    @ModuleInfo(key: "norm2") var norm2: KokoroAdaIN1d
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "conv1x1") var conv1x1: Conv1d?
    @ModuleInfo(key: "pool") var pool: ConvTransposed1d?
    let upsample: Bool
    let dimIn: Int

    init(dimIn: Int, dimOut: Int, styleDim: Int = 128, upsample: Bool = false) {
        self.upsample = upsample
        self.dimIn = dimIn
        self._norm1.wrappedValue = KokoroAdaIN1d(styleDim: styleDim, channels: dimIn)
        self._norm2.wrappedValue = KokoroAdaIN1d(styleDim: styleDim, channels: dimOut)
        self._conv1.wrappedValue = Conv1d(inputChannels: dimIn, outputChannels: dimOut, kernelSize: 3, padding: 1)
        self._conv2.wrappedValue = Conv1d(inputChannels: dimOut, outputChannels: dimOut, kernelSize: 3, padding: 1)
        if dimIn != dimOut {
            self._conv1x1.wrappedValue = Conv1d(inputChannels: dimIn, outputChannels: dimOut, kernelSize: 1, bias: false)
        }
        if upsample {
            self._pool.wrappedValue = ConvTransposed1d(
                inputChannels: 1,
                outputChannels: dimIn,
                kernelSize: 3,
                stride: 2,
                padding: 1,
                outputPadding: 1
            )
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, style: MLXArray) -> MLXArray {
        let residual = shortcut(x)
        var h = leakyRelu(norm1(x, style: style), negativeSlope: 0.2)
        if let pool {
            h = depthwiseConvTranspose2x(h, pool: pool)
        }
        h = conv1(h)
        h = leakyRelu(norm2(h, style: style), negativeSlope: 0.2)
        h = conv2(h)
        return (h + residual) * Float(1.0 / sqrt(2.0))
    }

    private func shortcut(_ x: MLXArray) -> MLXArray {
        var h = x
        if upsample {
            h = repeated(h, count: 2, axis: 1)
        }
        if let conv1x1 {
            h = conv1x1(h)
        }
        return h
    }

    private func depthwiseConvTranspose2x(_ x: MLXArray, pool: ConvTransposed1d) -> MLXArray {
        let batch = x.dim(0)
        let steps = x.dim(1)
        let ch = x.dim(2)
        let w = pool.weight
        let k0 = w[0..., 0, 0].reshaped(1, 1, ch)
        let k1 = w[0..., 1, 0].reshaped(1, 1, ch)
        let k2 = w[0..., 2, 0].reshaped(1, 1, ch)
        let zero = MLXArray.zeros([batch, 1, ch], dtype: x.dtype)
        let xNext = concatenated([x[0..., 1..., 0...], zero], axis: 1)
        let even = x * k1
        let odd = x * k2 + xNext * k0
        var out = stacked([even, odd], axis: 2).reshaped(batch, steps * 2, ch)
        if let bias = pool.bias {
            out = out + bias
        }
        return out
    }
}

final class KokoroALBERTEmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo(key: "position_embeddings") var positionEmbeddings: Embedding
    @ModuleInfo(key: "token_type_embeddings") var tokenTypeEmbeddings: Embedding
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

    init(vocabSize: Int = 178, embeddingSize: Int = 128, maxPosition: Int = 512) {
        self._wordEmbeddings.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: embeddingSize)
        self._positionEmbeddings.wrappedValue = Embedding(embeddingCount: maxPosition, dimensions: embeddingSize)
        self._tokenTypeEmbeddings.wrappedValue = Embedding(embeddingCount: 2, dimensions: embeddingSize)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: embeddingSize, eps: 1e-12)
        super.init()
    }

    func callAsFunction(_ inputIDs: MLXArray) -> MLXArray {
        let seqLen = inputIDs.dim(1)
        let positions = MLXArray(Int32(0) ..< Int32(seqLen)).reshaped(1, seqLen)
        let tokenTypes = MLXArray.zeros(like: inputIDs)
        let sum = wordEmbeddings(inputIDs) + positionEmbeddings(positions) + tokenTypeEmbeddings(tokenTypes)
        return layerNorm(sum)
    }
}

final class KokoroALBERTAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm
    let numHeads: Int
    let headDim: Int

    init(hiddenSize: Int = 768, numHeads: Int = 12) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self._query.wrappedValue = Linear(hiddenSize, hiddenSize)
        self._key.wrappedValue = Linear(hiddenSize, hiddenSize)
        self._value.wrappedValue = Linear(hiddenSize, hiddenSize)
        self._dense.wrappedValue = Linear(hiddenSize, hiddenSize)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-12)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let steps = x.dim(1)
        let q = query(x).reshaped(batch, steps, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = key(x).reshaped(batch, steps, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = value(x).reshaped(batch, steps, numHeads, headDim).transposed(0, 2, 1, 3)
        let scale = Float(1.0 / sqrt(Double(headDim)))
        let attn = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        let merged = attn.transposed(0, 2, 1, 3).reshaped(batch, steps, numHeads * headDim)
        return layerNorm(dense(merged) + x)
    }
}

final class KokoroALBERTLayer: Module {
    @ModuleInfo(key: "attention") var attention: KokoroALBERTAttention
    @ModuleInfo(key: "ffn") var ffn: Linear
    @ModuleInfo(key: "ffn_output") var ffnOutput: Linear
    @ModuleInfo(key: "full_layer_layer_norm") var fullLayerNorm: LayerNorm

    init(hiddenSize: Int = 768, intermediateSize: Int = 2048, numHeads: Int = 12) {
        self._attention.wrappedValue = KokoroALBERTAttention(hiddenSize: hiddenSize, numHeads: numHeads)
        self._ffn.wrappedValue = Linear(hiddenSize, intermediateSize)
        self._ffnOutput.wrappedValue = Linear(intermediateSize, hiddenSize)
        self._fullLayerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-12)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let attnOut = attention(x)
        let ffOut = ffnOutput(gelu(ffn(attnOut)))
        return fullLayerNorm(ffOut + attnOut)
    }
}

final class KokoroALBERTEncoder: Module {
    @ModuleInfo(key: "embedding_hidden_mapping_in") var embeddingMapping: Linear
    @ModuleInfo(key: "layer") var layer: KokoroALBERTLayer
    let numHiddenLayers: Int

    init(embeddingSize: Int = 128, hiddenSize: Int = 768, intermediateSize: Int = 2048, numHeads: Int = 12, numHiddenLayers: Int = 12) {
        self.numHiddenLayers = numHiddenLayers
        self._embeddingMapping.wrappedValue = Linear(embeddingSize, hiddenSize)
        self._layer.wrappedValue = KokoroALBERTLayer(hiddenSize: hiddenSize, intermediateSize: intermediateSize, numHeads: numHeads)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = embeddingMapping(x)
        for _ in 0 ..< numHiddenLayers {
            h = layer(h)
        }
        return h
    }
}

final class KokoroALBERT: Module {
    @ModuleInfo(key: "embeddings") var embeddings: KokoroALBERTEmbeddings
    @ModuleInfo(key: "encoder") var encoder: KokoroALBERTEncoder
    @ModuleInfo(key: "pooler") var pooler: Linear

    init(vocabSize: Int = 178, hiddenSize: Int = 768, intermediateSize: Int = 2048, numHeads: Int = 12, numLayers: Int = 12) {
        self._embeddings.wrappedValue = KokoroALBERTEmbeddings(vocabSize: vocabSize)
        self._encoder.wrappedValue = KokoroALBERTEncoder(
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            numHeads: numHeads,
            numHiddenLayers: numLayers
        )
        self._pooler.wrappedValue = Linear(hiddenSize, hiddenSize)
        super.init()
    }

    func callAsFunction(_ inputIDs: MLXArray) -> MLXArray {
        encoder(embeddings(inputIDs))
    }
}

final class KokoroTextEncoder: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "cnn_conv") var cnnConvs: [Conv1d]
    @ModuleInfo(key: "cnn_norm") var cnnNorms: [KokoroLayerNorm1d]
    @ModuleInfo(key: "lstm") var lstm: KokoroBiLSTM

    init(vocabSize: Int = 178, channels: Int = 512, kernelSize: Int = 5, numLayers: Int = 3) {
        self._embedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: channels)
        let pad = (kernelSize - 1) / 2
        self._cnnConvs.wrappedValue = (0 ..< numLayers).map { _ in
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernelSize, padding: pad)
        }
        self._cnnNorms.wrappedValue = (0 ..< numLayers).map { _ in KokoroLayerNorm1d(channels: channels) }
        self._lstm.wrappedValue = KokoroBiLSTM(inputSize: channels, hiddenSize: channels / 2)
        super.init()
    }

    func callAsFunction(_ inputIDs: MLXArray) -> MLXArray {
        var x = embedding(inputIDs)
        for index in cnnConvs.indices {
            x = leakyRelu(cnnNorms[index](cnnConvs[index](x)), negativeSlope: 0.2)
        }
        return lstm(x)
    }
}

final class KokoroDurationEncoder: Module {
    @ModuleInfo(key: "lstms") var lstms: [KokoroBiLSTM]
    @ModuleInfo(key: "norms") var norms: [KokoroAdaLayerNorm]

    init(dModel: Int = 512, styleDim: Int = 128, numLayers: Int = 3) {
        self._lstms.wrappedValue = (0 ..< numLayers).map { _ in
            KokoroBiLSTM(inputSize: dModel + styleDim, hiddenSize: dModel / 2)
        }
        self._norms.wrappedValue = (0 ..< numLayers).map { _ in
            KokoroAdaLayerNorm(styleDim: styleDim, channels: dModel)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, style: MLXArray) -> MLXArray {
        let steps = x.dim(1)
        let sExpanded = broadcast(style.expandedDimensions(axis: 1), to: [x.dim(0), steps, style.dim(-1)])
        var h = concatenated([x, sExpanded], axis: -1)
        for index in lstms.indices {
            h = lstms[index](h)
            h = norms[index](h, style: style)
            h = concatenated([h, sExpanded], axis: -1)
        }
        return h
    }
}

final class KokoroProsodyPredictor: Module {
    @ModuleInfo(key: "text_encoder") var textEncoder: KokoroDurationEncoder
    @ModuleInfo(key: "lstm") var lstm: KokoroBiLSTM
    @ModuleInfo(key: "duration_proj") var durationProj: Linear
    @ModuleInfo(key: "shared") var shared: KokoroBiLSTM
    @ModuleInfo(key: "F0") var f0Blocks: [KokoroAdainResBlk1d]
    @ModuleInfo(key: "N") var nBlocks: [KokoroAdainResBlk1d]
    @ModuleInfo(key: "F0_proj") var f0Proj: Conv1d
    @ModuleInfo(key: "N_proj") var nProj: Conv1d

    init(dModel: Int = 512, styleDim: Int = 128, maxDur: Int = 50, numLayers: Int = 3) {
        self._textEncoder.wrappedValue = KokoroDurationEncoder(dModel: dModel, styleDim: styleDim, numLayers: numLayers)
        self._lstm.wrappedValue = KokoroBiLSTM(inputSize: dModel + styleDim, hiddenSize: dModel / 2)
        self._durationProj.wrappedValue = Linear(dModel, maxDur)
        self._shared.wrappedValue = KokoroBiLSTM(inputSize: dModel + styleDim, hiddenSize: dModel / 2)
        self._f0Blocks.wrappedValue = [
            KokoroAdainResBlk1d(dimIn: dModel, dimOut: dModel, styleDim: styleDim, upsample: false),
            KokoroAdainResBlk1d(dimIn: dModel, dimOut: dModel / 2, styleDim: styleDim, upsample: true),
            KokoroAdainResBlk1d(dimIn: dModel / 2, dimOut: dModel / 2, styleDim: styleDim, upsample: false),
        ]
        self._nBlocks.wrappedValue = [
            KokoroAdainResBlk1d(dimIn: dModel, dimOut: dModel, styleDim: styleDim, upsample: false),
            KokoroAdainResBlk1d(dimIn: dModel, dimOut: dModel / 2, styleDim: styleDim, upsample: true),
            KokoroAdainResBlk1d(dimIn: dModel / 2, dimOut: dModel / 2, styleDim: styleDim, upsample: false),
        ]
        self._f0Proj.wrappedValue = Conv1d(inputChannels: dModel / 2, outputChannels: 1, kernelSize: 1)
        self._nProj.wrappedValue = Conv1d(inputChannels: dModel / 2, outputChannels: 1, kernelSize: 1)
        super.init()
    }

    func predictF0N(_ en: MLXArray, style: MLXArray) -> (f0: MLXArray, n: MLXArray) {
        let sharedOut = shared(en)
        var f0 = sharedOut
        for block in f0Blocks { f0 = block(f0, style: style) }
        var n = sharedOut
        for block in nBlocks { n = block(n, style: style) }
        return (f0Proj(f0).squeezed(axis: -1), nProj(n).squeezed(axis: -1))
    }
}
