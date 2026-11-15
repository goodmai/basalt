import Foundation
import MLX
import MLXLMCommon
import BasaltCore

extension MLXInferenceEngine {
    static let presenceContextSize = 1024

    func sampling(for request: GenerationRequest, breakers: Set<Int> = []) -> (
        GenerateParameters, GenerationComponents, shaping: (@Sendable () -> EntropyDynamicTemperature)?,
        exclusion: (@Sendable () -> ExcludeTopChoices)?
    ) {
        let temperature = Float(request.temperature ?? samplingDefaults.temperature ?? 0.7)
        var params = GenerateParameters(
            maxTokens: request.maxTokens ?? 1024,
            temperature: temperature,
            topP: Float(request.topP ?? samplingDefaults.topP ?? 0.9)
        )
        params.kvBits = modelFacts.capabilities.kvBits(request.kvPrecision ?? configuration.kvPrecision)
        params.kvScheme = modelFacts.capabilities.kvScheme(request.kvPrecision ?? configuration.kvPrecision)
        params.quantizedKVStart = configuration.quantizedKVStart
        params.repetitionPenalty = request.repetitionPenalty.map(Float.init) ?? configuration.repetitionPenalty
            ?? samplingDefaults.repetitionPenalty.map(Float.init)
        if let topK = request.topK ?? configuration.topK ?? samplingDefaults.topK { params.topK = topK }
        if let effectiveMinP = request.minP.map(Float.init) ?? configuration.minP ?? profile.defaultMinP {
            params.minP = effectiveMinP
        }
        params.seed = configuration.seed

        let components = BaseSamplingStages(request: request, breakers: breakers).map { stages in
            GenerationComponents(logitProcessorFactory: { stages.build() })
        } ?? GenerationComponents()

        var exclusion: (@Sendable () -> ExcludeTopChoices)?
        if let probability = request.xtcProbability.map(Float.init), probability > 0, temperature > 0 {
            let threshold = request.xtcThreshold.map(Float.init) ?? ExcludeTopChoices.defaultThreshold
            let seed = configuration.seed
            exclusion = { ExcludeTopChoices(threshold: threshold, probability: probability, seed: seed) }
        }
        var shaping: (@Sendable () -> EntropyDynamicTemperature)?
        if request.dynamicTemperature == true, temperature > 0 {
            let (topP, minP) = (params.topP, params.minP)
            shaping = { EntropyDynamicTemperature(baseTemperature: temperature, topP: topP, minP: minP) }
            params.topP = 1
            params.minP = 0
        }
        return (params, components, shaping, exclusion)
    }

    func attachConfidenceProbe(
        to components: GenerationComponents, for request: GenerationRequest
    ) -> (GenerationComponents, ConfidenceProbe?) {
        guard request.recordConfidence || configuration.recordConfidence else { return (components, nil) }
        let probe = ConfidenceProbe()
        return (components.appendingBasaltProcessor { ConfidenceProbeRecorder(probe: probe) }, probe)
    }

    func sequenceBreakerIds(for request: GenerationRequest) async -> Set<Int> {
        guard (request.dryMultiplier ?? 0) > 0, let container else { return [] }
        let strings = request.drySequenceBreakers ?? DRYPenalty.defaultSequenceBreakers
        if let sequenceBreakers, sequenceBreakers.strings == strings { return sequenceBreakers.ids }
        let epoch = modelEpoch
        let ids = await container.perform { DRYPenalty.breakerIds(strings, tokenizer: $0.tokenizer) }
        if epoch == modelEpoch { sequenceBreakers = (strings, ids) }
        return ids
    }
}

private struct BaseSamplingStages: Sendable {
    let presence: Float?
    let dryMultiplier: Float?
    let dryBase: Float
    let dryAllowedLength: Int
    let breakers: Set<Int>
    let probe: LogitProbe?

    init?(request: GenerationRequest, breakers: Set<Int>) {
        let presence = request.presencePenalty.map { Float($0) }
        self.presence = presence != 0 ? presence : nil
        let multiplier = request.dryMultiplier.map { Float($0) }
        self.dryMultiplier = (multiplier ?? 0) > 0 ? multiplier : nil
        self.dryBase = request.dryBase.map { Float($0) } ?? DRYPenalty.defaultBase
        self.dryAllowedLength = request.dryAllowedLength ?? DRYPenalty.defaultAllowedLength
        self.breakers = breakers
        self.probe = request.logitProbe
        if self.presence == nil, self.dryMultiplier == nil, self.probe == nil { return nil }
    }

    func build() -> any LogitProcessor {
        var chain = LogitProcessorChain(capacity: 3)
        if let presence {
            chain.appendStage(.presence(GeneratedTokensPresencePenalty(penalty: presence, window: MLXInferenceEngine.presenceContextSize)))
        }
        if let dryMultiplier {
            chain.appendStage(.dry(DRYPenalty(multiplier: dryMultiplier, base: dryBase, allowedLength: dryAllowedLength, breakers: breakers)))
        }
        if let probe {
            chain.appendStage(.logitProbe(LogitProbeRecorder(probe: probe)))
        }
        return chain.stages.count == 1 ? chain.stages[0].erased : chain
    }
}

struct GeneratedTokensPresencePenalty: LogitProcessor {
    var context: PresencePenaltyContext

    init(penalty: Float, window: Int) {
        context = PresencePenaltyContext(presencePenalty: penalty, presenceContextSize: window)
    }

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        context.process(logits: logits)
    }

    mutating func didSample(token: MLXArray) {
        context.didSample(token: token)
    }
}

struct LogitProbeRecorder: LogitProcessor {
    let probe: LogitProbe

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        let row = logits.reshaped(-1).asType(.float32)
        let logprobs = row - logSumExp(row)
        let ids = argSort(-logprobs)[..<min(probe.topK, row.dim(0))]
        probe.record(ids: ids.asArray(Int.self), logprobs: logprobs[ids].asArray(Float.self))
        return logits
    }

    mutating func didSample(token: MLXArray) {
        probe.sampled(token.item(Int.self))
    }
}

struct ConfidenceProbeRecorder: LogitProcessor {
    let probe: ConfidenceProbe

    static func maxProbability(logits: MLXArray) -> Double? {
        statistics(logits: logits)?.probability
    }

    mutating func prompt(_ prompt: MLXArray) {}

    static func statistics(logits: MLXArray) -> (probability: Double, entropy: Double, margin: Double)? {
        FusedLogitConfidence.statistics(logits: logits)
    }

    static func summarize(logits: MLXArray) -> ResponseConfidence? {
        FusedLogitConfidence.summarize(logits)
    }

    func process(logits: MLXArray) -> MLXArray {
        if let stats = Self.statistics(logits: logits) {
            probe.record(stats.probability, entropy: stats.entropy, margin: stats.margin)
        }
        return logits
    }

    mutating func didSample(token: MLXArray) {}
}

struct PromptAnchoredProcessor: LogitProcessor, ForcedTokenSource {
    var chain: LogitProcessorChain
    let prompt: MLXArray

    init(base: any LogitProcessor, prompt: MLXArray) {
        self.chain = (base as? LogitProcessorChain) ?? LogitProcessorChain(single: base)
        self.prompt = prompt
    }

    init(chain: LogitProcessorChain, prompt: MLXArray) {
        self.chain = chain
        self.prompt = prompt
    }

    func copy() -> PromptAnchoredProcessor {
        PromptAnchoredProcessor(chain: chain.copy(), prompt: prompt)
    }

    mutating func prompt(_ tail: MLXArray) {
        chain.prompt(prompt)
    }

    func process(logits: MLXArray) -> MLXArray {
        chain.process(logits: logits)
    }

    mutating func didSample(token: MLXArray) {
        chain.didSample(token: token)
    }

    mutating func takeForcedTokens(limit: Int) -> [Int] {
        chain.takeForcedTokens(limit: limit)
    }

    mutating func didSampleForced(_ token: MLXArray) {
        chain.didSampleForced(token)
    }
}

extension GenerationComponents {
    func appendingBasaltProcessor<P: LogitStageConvertible & SendableMetatype>(
        _ factory: @escaping @Sendable () -> P
    ) -> GenerationComponents {
        let existingFactory = logitProcessorFactory
        return GenerationComponents(
            logitProcessorFactory: {
                guard let existing = existingFactory?() else { return factory() }
                var chain = (existing as? LogitProcessorChain) ?? LogitProcessorChain(single: existing)
                chain.appendStage(factory().logitStage)
                return chain
            },
            parameterValidator: parameterValidator)
    }

    func appendingBasaltProcessor(
        _ factory: @escaping @Sendable () -> any LogitProcessor
    ) -> GenerationComponents {
        let existingFactory = logitProcessorFactory
        return GenerationComponents(
            logitProcessorFactory: {
                guard let existing = existingFactory?() else { return factory() }
                var chain = (existing as? LogitProcessorChain) ?? LogitProcessorChain(single: existing)
                chain.append(factory())
                return chain
            },
            parameterValidator: parameterValidator)
    }

    func makeProcessor(parameters: GenerateParameters) -> (any LogitProcessor)? {
        switch (parameters.processor(), logitProcessorFactory?()) {
        case (nil, nil):
            return nil
        case (let penalty?, nil):
            return penalty
        case (nil, let custom?):
            return custom
        case (let penalty?, let custom?):
            var chain = (custom as? LogitProcessorChain) ?? LogitProcessorChain(single: custom)
            chain.prepend(penalty)
            return chain
        }
    }
}
