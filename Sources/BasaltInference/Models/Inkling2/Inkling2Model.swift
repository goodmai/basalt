import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

final class Inkling2TextModel: Module {
    let config: Inkling2Configuration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_norm") var embedNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [Inkling2DecoderLayer]
    @ModuleInfo(key: "final_input") var finalInput: Inkling2Query
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "message_input") var messageInput: Inkling2MessageInput?

    init(_ config: Inkling2Configuration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _embedNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layers.wrappedValue = (0 ..< config.layers).map { Inkling2DecoderLayer(config, index: $0) }
        _finalInput.wrappedValue = Inkling2Query(rows: 1, dims: config.hiddenSize)
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if let loop = config.loop {
            _messageInput.wrappedValue = Inkling2MessageInput(looped: loop.coreEnd - loop.coreStart, dims: config.hiddenSize)
        }
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        forward(tokens, cache: cache).hidden
    }

    func forward(_ tokens: MLXArray, cache: [KVCache]?, capturing: Set<Int> = []) -> (hidden: MLXArray, captured: [MLXArray]) {
        let embedded = embedNorm(embedTokens(tokens))
        var captured = [MLXArray]()
        var visit = 0
        func run(_ range: Range<Int>, _ hidden: MLXArray, _ blocks: MLXArray?, bank: (blocks: MLXArray, partial: MLXArray)?) -> (MLXArray, MLXArray?) {
            var (h, b) = (hidden, blocks)
            for (offset, index) in range.enumerated() {
                let messages = bank.flatMap { bank in messageInput.map { $0(bank, layer: offset) } }
                let attentionCache = cache?[2 * visit]
                let convCache = cache?[2 * visit + 1] as? ArraysCache
                if capturing.contains(visit) { captured.append(h.asType(embedded.dtype)) }
                (h, b) = layers[index](h, blocks: b, cache: attentionCache, conv: convCache, messages: messages)
                visit += 1
            }
            return (h, b)
        }
        var (hidden, blocks): (MLXArray, MLXArray?)
        if let loop = config.loop {
            let prelude = run(0 ..< loop.coreStart, embedded, nil, bank: nil)
            (hidden, blocks) = prelude
            for pass in 0 ..< loop.numPasses {
                var bank: (blocks: MLXArray, partial: MLXArray)?
                if pass > 0, let current = blocks {
                    let skip = (prelude.1?.dim(0) ?? 0) + 1
                    bank = (current[skip...], Inkling2Ops.unitRMS(hidden))
                    (hidden, blocks) = prelude
                }
                (hidden, blocks) = run(loop.coreStart ..< loop.coreEnd, hidden, blocks, bank: bank)
            }
            (hidden, blocks) = run(loop.coreEnd ..< config.layers, hidden, blocks, bank: nil)
        } else {
            (hidden, blocks) = run(0 ..< config.layers, embedded, nil, bank: nil)
        }
        let readout = finalInput(blocks!, Inkling2Ops.unitRMS(hidden))[.ellipsis, 0, 0...]
        let final = norm(readout).asType(embedded.dtype)
        if capturing.contains(visit) { captured.append(final) }
        return (final, captured)
    }
}

final class Inkling2Inner: Module {
    @ModuleInfo(key: "language_model") var languageModel: Inkling2TextModel

    init(_ config: Inkling2Configuration) {
        _languageModel.wrappedValue = Inkling2TextModel(config)
    }
}

public final class Inkling2Model: Module, LLMModel, LoRAModel {
    public static let modelTypes = ["inkling2_loop", "inkling2"]

    public let vocabularySize: Int
    let config: Inkling2Configuration
    @ModuleInfo(key: "model") var model: Inkling2Inner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ config: Inkling2Configuration) {
        self.config = config
        vocabularySize = config.unpaddedVocabSize
        _model.wrappedValue = Inkling2Inner(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    public convenience init(configurationData data: Data) throws {
        self.init(try JSONDecoder().decode(Inkling2Configuration.self, from: data).validated())
    }

    public var loraLayers: [Module] { model.languageModel.layers }

    public var toolCallFormat: ToolCallFormat? { .inkling }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(model.languageModel(inputs, cache: cache))
    }

    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        guard state?[mtpEmitFlagKey] ?? false else { return LMOutput(logits: self(input.tokens, cache: cache)) }
        let (hidden, features) = draftFeatures(input.tokens, cache: cache)
        var state = state ?? LMOutput.State()
        state[mtpLastHiddenStatesKey] = features
        state[mtpSharedKVStatesKey] = [:]
        state[mtpSharedKVOffsetsKey] = [:]
        state[mtpSharedKVSourceIndicesKey] = [:]
        return LMOutput(logits: logits(hidden), state: state)
    }

    func draftFeatures(_ tokens: MLXArray, cache: [KVCache]?) -> (hidden: MLXArray, features: MLXArray) {
        let taps = Set(config.draftTaps ?? [config.visitOrder.count])
        let length = tokens.dim(-1)
        guard length > Self.draftChunk else {
            let output = model.languageModel.forward(tokens, cache: cache, capturing: taps)
            return (output.hidden, concatenated(output.captured, axis: -1))
        }
        var features = [MLXArray]()
        var last = tokens
        for start in stride(from: 0, to: length, by: Self.draftChunk) {
            let output = model.languageModel.forward(tokens[.ellipsis, start ..< min(start + Self.draftChunk, length)], cache: cache, capturing: taps)
            features.append(concatenated(output.captured, axis: -1))
            last = output.hidden[0..., (-1)..., 0...]
            eval([features.last!, last] + (cache ?? []).flatMap { $0.innerState() })
        }
        return (last, concatenated(features, axis: 1))
    }

    static let draftChunk = 512

    func logits(_ hidden: MLXArray) -> MLXArray {
        let capped = tanh(vocabularyLogits(hidden).asType(.float32) / config.finalLogitSoftcap) * config.finalLogitSoftcap
        return capped[.ellipsis, ..<config.unpaddedVocabSize]
    }

    func embedding(_ tokens: MLXArray) -> MLXArray {
        model.languageModel.embedNorm(model.languageModel.embedTokens(tokens))
    }

    func vocabularyLogits(_ hidden: MLXArray) -> MLXArray {
        lmHead(hidden)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        let local = Set(config.localLayers)
        return config.visitOrder.flatMap { index -> [any KVCache] in
            let attention: any KVCache = local.contains(index) ? Inkling2WindowKVCache(window: config.slidingWindow) : KVCacheSimple()
            return [attention, ConvolutionStatesCache(slots: 4, stateLength: config.shortConvKernel - 1)]
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let mapped = Inkling2Checkpoint.sanitized(weights, config: config)
        FlintModel.quantizeModulesMatchingPackedWeights(in: self, weights: mapped)
        return mapped
    }
}
