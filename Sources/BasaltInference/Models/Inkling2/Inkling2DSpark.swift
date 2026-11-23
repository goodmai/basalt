import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class Inkling2DSparkDraftModel: Module, StatefulMTPDrafterModel {
    let configuration: Inkling2DSparkConfiguration
    let vocabulary: Int
    let softcap: Float
    var maximumBlockSize: Int? { configuration.blockSize }
    let requiresSharedTargetKV = false
    let requiresPromptPrefill = true
    let requiresGreedySampling = true
    let adaptsDraftDepth = false

    @ModuleInfo var layers: [Inkling2DSparkLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "markov_head") var markovHead: Inkling2DSparkMarkovHead
    @ParameterInfo(key: "fuse_weight") var fuseWeight: MLXArray
    @ParameterInfo(key: "mask_embed") var maskEmbedding: MLXArray

    init(_ configuration: Inkling2DSparkConfiguration, target: Inkling2Configuration) {
        self.configuration = configuration
        (vocabulary, softcap) = (target.vocabSize, target.finalLogitSoftcap)
        _layers.wrappedValue = (0 ..< configuration.hiddenLayers).map {
            Inkling2DSparkLayer(configuration, isLocal: configuration.localLayers.contains($0))
        }
        _norm.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _markovHead.wrappedValue = Inkling2DSparkMarkovHead(configuration, vocabulary: target.vocabSize)
        _fuseWeight.wrappedValue = MLXArray.zeros([configuration.hiddenLayers + 1, configuration.targetLayerIDs.count])
        _maskEmbedding.wrappedValue = MLXArray.zeros([configuration.hiddenSize])
        super.init()
    }

    func makeState(parameters _: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: layers.map { _ in KVCacheSimple() })
    }

    func prepareDrafterState(target _: any LanguageModel, input: MTPPrefillInput, state: inout MTPDrafterState, sampler _: any LogitSampler) {
        appendContext(input.targetHidden, state: &state)
        state.carryHidden = input.targetHidden[0..., (-1)..., 0...]
    }

    func draftBlock(target: any LanguageModel, input: MTPDraftInput, sampler: any LogitSampler) -> MLXArray {
        var state = makeState(parameters: nil)
        return draftBlock(target: target, input: input, state: &state, sampler: sampler)
    }

    func draftBlock(target: any LanguageModel, input: MTPDraftInput, state: inout MTPDrafterState, sampler: any LogitSampler) -> MLXArray {
        guard let target = target as? Inkling2Model else { preconditionFailure("Inkling 2 DSpark drafts for an Inkling 2 target") }
        let (drafts, block) = (input.blockSize - 1, configuration.blockSize)
        precondition((1 ..< block).contains(drafts), "Inkling 2 DSpark drafts 1 to \(block - 1)")
        let anchor = input.lastToken.reshaped(-1, 1)
        let rows = anchor.dim(0)
        let anchorEmbedding = target.embedding(anchor)
        let masks = broadcast(maskEmbedding.asType(anchorEmbedding.dtype).reshaped(1, 1, -1), to: [rows, block - 1, configuration.hiddenSize])
        var x = concatenated([anchorEmbedding, masks], axis: 1)
        for (layer, cache) in zip(layers, state.cache) { x = layer(x, cache: cache) }
        for cache in state.cache { cache.trim(block) }
        let hidden = norm(x)[0..., 1 ... drafts]
        let baseLogits = target.vocabularyLogits(hidden)
        let context = markovHead.context(hidden: hidden, features: mix(input.lastHidden, row: layers.count, markovHead.featureNorm))
        var markovState = MLXArray.zeros([rows, 1, markovHead.rank], dtype: hidden.dtype)
        var previous = anchor
        var proposals = [MLXArray]()
        for index in 0 ..< drafts {
            let (next, correction) = markovHead.step(state: markovState, previous: previous, context: context[0..., index ..< (index + 1)])
            markovState = next
            let logits = (baseLogits[0..., index ..< (index + 1)] + correction).asType(.float32)
            let capped = tanh(logits / softcap) * softcap
            previous = sampler.sample(logits: capped[0..., 0, ..<target.vocabularySize]).reshaped(-1, 1)
            proposals.append(previous)
        }
        return concatenated(proposals, axis: 1)
    }

    func commitDrafterState(target _: any LanguageModel, input: MTPCommitInput, state: inout MTPDrafterState, sampler _: any LogitSampler) {
        let committed = input.acceptedCount + 1
        precondition(input.targetHidden.dim(1) >= committed, "the target did not emit every committed feature")
        appendContext(input.targetHidden[0..., ..<committed], state: &state)
    }

    private func appendContext(_ features: MLXArray, state: inout MTPDrafterState) {
        for (index, (layer, cache)) in zip(layers, state.cache).enumerated() {
            layer.attention.appendContext(mix(features, row: index, layer.featureNorm), cache: cache)
        }
        state.nextPosition += features.dim(1)
    }

    private func mix(_ features: MLXArray, row: Int, _ norm: RMSNorm) -> MLXArray {
        let taps = configuration.targetLayerIDs.count
        let shape = Array(features.shape.dropLast()) + [taps, configuration.hiddenSize]
        let alpha = softmax(fuseWeight[row].asType(.float32), axis: -1).reshaped(taps, 1)
        let mixed = (features.reshaped(shape).asType(.float32) * alpha).sum(axis: -2)
        return MLXFast.rmsNorm(mixed, weight: norm.weight.asType(.float32), eps: norm.eps).asType(features.dtype)
    }
}

extension Inkling2DSparkDraftModel {
    static func load(url: URL, directory: URL, target: Inkling2Model) throws -> Inkling2DSparkDraftModel {
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let configuration = try JSONDecoder().decode(Inkling2DSparkConfiguration.self, from: data).validated(for: target.config)
        let model = Inkling2DSparkDraftModel(configuration, target: target.config)
        var weights = try MLX.loadArrays(url: url)
        guard let projection = weights.removeValue(forKey: "markov_head.hidden_out.weight"),
              let bias = weights.removeValue(forKey: "markov_head.hidden_out.bias")
        else { throw MTPSidecarError.malformed("\(url.lastPathComponent) has no markov_head.hidden_out") }
        (weights["markov_head.fold.weight"], weights["markov_head.fold.bias"]) = folded(target.lmHead, projection: projection, bias: bias)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    static func folded(_ head: Linear, projection: MLXArray, bias: MLXArray) -> (MLXArray, MLXArray) {
        var (foldWeights, foldBiases) = ([MLXArray](), [MLXArray]())
        let count = head.weight.dim(0)
        for start in stride(from: 0, to: count, by: 16384) {
            let rows = denseRows(head, start ..< min(start + 16384, count)).asType(.float32)
            foldWeights.append(matmul(rows, projection.asType(.float32)).asType(.bfloat16))
            foldBiases.append(matmul(rows, bias.asType(.float32)).asType(.bfloat16))
            eval(foldWeights.last!, foldBiases.last!)
        }
        return (concatenated(foldWeights), concatenated(foldBiases))
    }

    private static func denseRows(_ head: Linear, _ rows: Range<Int>) -> MLXArray {
        guard let quantized = head as? QuantizedLinear else { return head.weight[rows] }
        return dequantized(
            quantized.weight[rows], scales: quantized.scales[rows], biases: quantized.biases?[rows], groupSize: quantized.groupSize,
            bits: quantized.bits, mode: quantized.mode, globalScale: quantized.globalScale)
    }
}
