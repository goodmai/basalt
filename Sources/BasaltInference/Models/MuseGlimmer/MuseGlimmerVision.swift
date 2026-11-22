import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import BasaltCore

package final class MuseGlimmerPatchEmbedder: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "position_embedding_table") var positionEmbeddingTable: Embedding
    let side: Int

    package init(_ config: MuseGlimmerVisionConfiguration) {
        let patchDim = config.patchTemporal * 3 * config.patchSize * config.patchSize
        self._patchEmbedding.wrappedValue = Linear(patchDim, config.hiddenSize, bias: false)
        self._positionEmbeddingTable.wrappedValue = Embedding(
            embeddingCount: config.positionEmbeddingHeight * config.positionEmbeddingWidth,
            dimensions: config.hiddenSize
        )
        self.side = config.positionEmbeddingHeight
        super.init()
    }

    package func callAsFunction(_ pixelValues: MLXArray, grid: [THW]) -> MLXArray {
        let embeddings = patchEmbedding(pixelValues)
        let positions = MuseGlimmerVisionMath.interpolatedPositions(grid: grid, side: side) {
            positionEmbeddingTable($0)
        }.asType(embeddings.dtype)
        return embeddings + positions
    }
}

package final class MuseGlimmerVisionAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "proj") var outputProj: Linear

    let heads: Int
    let scale: Float

    package init(_ config: MuseGlimmerVisionConfiguration) {
        self.heads = config.attentionHeads
        let headDim = config.hiddenSize / config.attentionHeads
        self.scale = pow(Float(headDim), -0.5)
        self._queryProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        self._keyProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        self._valueProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        self._outputProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        super.init()
    }

    package func callAsFunction(_ hiddenStates: MLXArray, splitPoints: [Int], cos: MLXArray, sin: MLXArray) -> MLXArray {
        let length = hiddenStates.dim(0)
        var q = queryProj(hiddenStates).reshaped(1, length, heads, -1)
        var k = keyProj(hiddenStates).reshaped(1, length, heads, -1)
        let v = valueProj(hiddenStates).reshaped(1, length, heads, -1)

        let cosExp = expandedDimensions(expandedDimensions(cos, axis: 0), axis: 2)
        let sinExp = expandedDimensions(expandedDimensions(sin, axis: 0), axis: 2)
        q = (q.asType(.float32) * cosExp + MuseGlimmerVisionMath.rotateHalf(q).asType(.float32) * sinExp).asType(q.dtype)
        k = (k.asType(.float32) * cosExp + MuseGlimmerVisionMath.rotateHalf(k).asType(.float32) * sinExp).asType(k.dtype)

        let queries = q.transposed(0, 2, 1, 3)
        let keys = k.transposed(0, 2, 1, 3)
        let values = v.transposed(0, 2, 1, 3)

        let output: MLXArray
        if splitPoints.isEmpty {
            output = MuseGlimmerVisionMath.fusedSDPA(queries, keys, values, scale: scale)
        } else {
            let qs = split(queries, indices: splitPoints, axis: 2)
            let ks = split(keys, indices: splitPoints, axis: 2)
            let vs = split(values, indices: splitPoints, axis: 2)
            output = concatenated(
                (0 ..< qs.count).map { MuseGlimmerVisionMath.fusedSDPA(qs[$0], ks[$0], vs[$0], scale: scale) },
                axis: 2
            )
        }
        return outputProj(output.transposed(0, 2, 1, 3).reshaped(length, -1))
    }
}

package final class MuseGlimmerVisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    package init(_ config: MuseGlimmerVisionConfiguration) {
        self._fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        self._fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        super.init()
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(MLXNN.gelu(fc1(x)))
    }
}

package final class MuseGlimmerVisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attention: MuseGlimmerVisionAttention
    @ModuleInfo var mlp: MuseGlimmerVisionMLP

    package init(_ config: MuseGlimmerVisionConfiguration) {
        self._norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._attention.wrappedValue = MuseGlimmerVisionAttention(config)
        self._mlp.wrappedValue = MuseGlimmerVisionMLP(config)
        super.init()
    }

    package func callAsFunction(_ x: MLXArray, splitPoints: [Int], cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = x + attention(norm1(x), splitPoints: splitPoints, cos: cos, sin: sin)
        return h + mlp(norm2(h))
    }
}

package final class MuseGlimmerVisionTower: Module {
    @ModuleInfo(key: "patch_embedder") var patchEmbedder: MuseGlimmerPatchEmbedder
    @ModuleInfo(key: "ln_pre") var lnPre: LayerNorm
    @ModuleInfo var layers: [MuseGlimmerVisionBlock]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm

    let config: MuseGlimmerVisionConfiguration
    private let ropeSpatialDim: Int

    package init(_ config: MuseGlimmerVisionConfiguration) {
        self.config = config
        self._patchEmbedder.wrappedValue = MuseGlimmerPatchEmbedder(config)
        self._lnPre.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in MuseGlimmerVisionBlock(config) }
        self._lnPost.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.ropeSpatialDim = (config.hiddenSize / config.attentionHeads) / 2
        super.init()
    }

    private func rotary(_ positionIds: MLXArray) -> (MLXArray, MLXArray) {
        let exponent = MLXArray(stride(from: 0, to: ropeSpatialDim, by: 2)).asType(.float32) / Float(ropeSpatialDim)
        let inverseFreq = 1.0 / pow(MLXArray(config.ropeTheta), exponent)
        let freqW = expandedDimensions(positionIds[0..., 0].asType(.float32), axis: -1) * expandedDimensions(inverseFreq, axis: 0)
        let freqH = expandedDimensions(positionIds[0..., 1].asType(.float32), axis: -1) * expandedDimensions(inverseFreq, axis: 0)
        let freqs = concatenated([freqW, freqH, freqW, freqH], axis: -1)
        return (cos(freqs), sin(freqs))
    }

    package func callAsFunction(_ pixelValues: MLXArray, grid: [THW]) -> MLXArray {
        let fullSplit = Array(MuseGlimmerVisionMath.cuSeqlens(grid).dropFirst().dropLast())
        let (windowIndex, windowBounds) = MuseGlimmerVisionMath.windowIndex(grid, windowSize: config.positionEmbeddingHeight)
        let windowSplit = Array(windowBounds.dropFirst().dropLast())

        var hidden = lnPre(patchEmbedder(pixelValues, grid: grid))
        var positions = MuseGlimmerVisionMath.positionIds(grid)
        if let windowIndex {
            hidden = hidden[windowIndex]
            positions = positions[windowIndex]
        }
        let (cosVal, sinVal) = rotary(positions)
        for (index, block) in layers.enumerated() {
            let splitPoints = config.layerTypes[index] == "full_attention" ? fullSplit : windowSplit
            hidden = block(hidden, splitPoints: splitPoints, cos: cosVal, sin: sinVal)
        }
        if let windowIndex {
            hidden = hidden[argSort(windowIndex, axis: 0)]
        }
        return MuseGlimmerVisionMath.pixelShuffle(lnPost(hidden), grid: grid, mergeSize: config.mergeSize)
    }
}

package final class MuseGlimmerVisionAdapter: Module, UnaryLayer {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    package init(_ config: MuseGlimmerConfiguration) {
        self._fc1.wrappedValue = Linear(config.outHiddenSize, config.projectorHiddenSize, bias: false)
        self._fc2.wrappedValue = Linear(config.projectorHiddenSize, config.projectorHiddenSize, bias: false)
        super.init()
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXNN.gelu(fc2(MLXNN.gelu(fc1(x))))
    }
}

public final class MuseGlimmerVLM: Module, VLMModel, KVCacheDimensionProvider, LoRAModel {
    @ModuleInfo(key: "language_model") public var languageModel: MuseGlimmerTextModel
    @ModuleInfo(key: "vision_tower") var visionTower: MuseGlimmerVisionTower
    @ModuleInfo(key: "vision_adapter") var visionAdapter: MuseGlimmerVisionAdapter
    @ModuleInfo(key: "vision_projection") var visionProjection: Linear
    @ModuleInfo(key: "perception_emb_norm") var perceptionEmbNorm: MuseGlimmerScalelessRMSNorm

    public let config: MuseGlimmerConfiguration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.loraLayers }
    public var toolCallFormat: ToolCallFormat? { .atem }
    public var reasoningConfig: ReasoningConfig? { languageModel.reasoningConfig }

    public init(_ config: MuseGlimmerConfiguration) {
        self.config = config
        self._languageModel.wrappedValue = MuseGlimmerTextModel(config.textConfiguration)
        self._visionTower.wrappedValue = MuseGlimmerVisionTower(config.visionConfiguration)
        self._visionAdapter.wrappedValue = MuseGlimmerVisionAdapter(config)
        self._visionProjection.wrappedValue = Linear(config.projectorHiddenSize, config.textConfiguration.hiddenSize, bias: false)
        self._perceptionEmbNorm.wrappedValue = MuseGlimmerScalelessRMSNorm(eps: config.textConfiguration.rmsNormEps)
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        try languageModel.newCache(parameters: parameters)
    }

    private func inputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?, grid: [THW]) throws -> MLXArray {
        let ids = inputIds.ndim == 1 ? inputIds.expandedDimensions(axis: 0) : inputIds
        var embeddings = languageModel.model.embedNorm(languageModel.model.embedTokens(ids))
        if embeddings.ndim == 2 { embeddings = embeddings.expandedDimensions(axis: 0) }
        guard let pixelValues else { return embeddings }

        let dtype = (visionTower.patchEmbedder.patchEmbedding as? QuantizedLinear)?.scales.dtype
            ?? visionTower.patchEmbedder.patchEmbedding.weight.dtype
        let projected = perceptionEmbNorm(visionProjection(visionAdapter(visionTower(pixelValues.asType(dtype), grid: grid))))
            .asType(embeddings.dtype)

        let positions = ids.asArray(Int.self).enumerated().compactMap { index, id in
            (id == config.imageTokenId || id == config.videoTokenId) ? Int32(index) : nil
        }
        guard positions.count == projected.dim(0) else {
            throw VLMError.processing("Image features and image tokens do not match: tokens=\(positions.count), features=\(projected.dim(0))")
        }
        guard !positions.isEmpty else { return embeddings }
        embeddings[0..., MLXArray(positions), 0...] = projected.expandedDimensions(axis: 0)
        return embeddings
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], state: LMOutput.State?, prefill: PrefillParameters) throws -> PrepareResult {
        let converted = cache.compactMap { $0 as KVCache }
        let tokens = input.text.tokens.ndim == 1 ? input.text.tokens.expandedDimensions(axis: 0) : input.text.tokens
        let embeddings = try inputEmbeddings(inputIds: tokens, pixelValues: input.image?.pixels, grid: input.image?.frames ?? [])
        let totalPositions = embeddings.dim(1)
        let processed = try prefill.forEachChunk(total: totalPositions) { range in
            _ = languageModel.forward(inputs: nil, cache: converted, inputEmbedding: embeddings[0..., range, 0...])
            asyncEval(cache)
        }
        if processed > 0 { eval(cache) }
        let logits = languageModel.forward(inputs: nil, cache: converted, inputEmbedding: embeddings[0..., processed..., 0...])
        prefill.progress?(totalPositions, totalPositions)
        return .logits(LMOutput(logits: logits))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray](minimumCapacity: weights.count)
        for (rawKey, value) in weights where !rawKey.contains("rotary_emb.inv_freq") {
            var key = rawKey
            if key.hasPrefix("model.language_model.") {
                key = "language_model.model." + key.dropFirst("model.language_model.".count)
            } else if key.hasPrefix("model.") {
                key = String(key.dropFirst("model.".count))
            } else if key.hasPrefix("lm_head.") {
                key = "language_model." + key
            }
            result[key] = value
        }
        return result
    }
}
