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
        params.kvBits = modelFacts.capabilities.kvBits(request.kvPrecision ?? kvPrecision)
        params.quantizedKVStart = quantizedKVStart
        params.repetitionPenalty = request.repetitionPenalty.map(Float.init) ?? repetitionPenalty
            ?? samplingDefaults.repetitionPenalty.map(Float.init)
        if let topK = request.topK ?? topK ?? samplingDefaults.topK { params.topK = topK }
        if let effectiveMinP = request.minP.map(Float.init) ?? minP ?? profile.defaultMinP {
            params.minP = effectiveMinP
        }
        params.seed = seed

        var components = GenerationComponents()
        if let presencePenalty = request.presencePenalty.map(Float.init), presencePenalty != 0 {
            components = components.appendingBasaltProcessor {
                GeneratedTokensPresencePenalty(penalty: presencePenalty, window: Self.presenceContextSize)
            }
        }
        if let multiplier = request.dryMultiplier.map(Float.init), multiplier > 0 {
            let base = request.dryBase.map(Float.init) ?? DRYPenalty.defaultBase
            let allowedLength = request.dryAllowedLength ?? DRYPenalty.defaultAllowedLength
            components = components.appendingBasaltProcessor {
                DRYPenalty(multiplier: multiplier, base: base, allowedLength: allowedLength, breakers: breakers)
            }
        }
        var exclusion: (@Sendable () -> ExcludeTopChoices)?
        if let probability = request.xtcProbability.map(Float.init), probability > 0, temperature > 0 {
            let threshold = request.xtcThreshold.map(Float.init) ?? ExcludeTopChoices.defaultThreshold
            let seed = seed
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

struct PromptAnchoredProcessor: LogitProcessor, ForcedTokenSource {
    var base: any LogitProcessor
    let prompt: MLXArray

    mutating func prompt(_ tail: MLXArray) {
        base.prompt(prompt)
    }

    func process(logits: MLXArray) -> MLXArray {
        base.process(logits: logits)
    }

    mutating func didSample(token: MLXArray) {
        base.didSample(token: token)
    }

    mutating func takeForcedTokens(limit: Int) -> [Int] {
        if var source = base as? ForcedTokenSource {
            let tokens = source.takeForcedTokens(limit: limit)
            base = source
            return tokens
        }
        return []
    }

    mutating func didSampleForced(_ token: MLXArray) {
        if var source = base as? ForcedTokenSource {
            source.didSampleForced(token)
            base = source
        } else {
            base.didSample(token: token)
        }
    }
}

struct BasaltChainedLogitProcessor: LogitProcessor, ForcedTokenSource {
    var processors: [any LogitProcessor]

    init(processors: [any LogitProcessor]) {
        self.processors = processors
    }

    mutating func prompt(_ prompt: MLXArray) {
        for index in processors.indices {
            processors[index].prompt(prompt)
        }
    }

    func process(logits: MLXArray) -> MLXArray {
        processors.reduce(logits) { $1.process(logits: $0) }
    }

    mutating func didSample(token: MLXArray) {
        for index in processors.indices {
            processors[index].didSample(token: token)
        }
    }

    mutating func takeForcedTokens(limit: Int) -> [Int] {
        var result: [Int] = []
        for index in processors.indices where result.count < limit {
            if var source = processors[index] as? ForcedTokenSource {
                result.append(contentsOf: source.takeForcedTokens(limit: limit - result.count))
                processors[index] = source
            }
        }
        return result
    }

    mutating func didSampleForced(_ token: MLXArray) {
        for index in processors.indices {
            if var source = processors[index] as? ForcedTokenSource {
                source.didSampleForced(token)
                processors[index] = source
            } else {
                processors[index].didSample(token: token)
            }
        }
    }
}

extension GenerationComponents {
    func appendingBasaltProcessor(
        _ factory: @escaping @Sendable () -> any LogitProcessor
    ) -> GenerationComponents {
        let existingFactory = logitProcessorFactory
        return GenerationComponents(
            logitProcessorFactory: {
                guard let existing = existingFactory?() else {
                    return factory()
                }
                if let chained = existing as? BasaltChainedLogitProcessor {
                    return BasaltChainedLogitProcessor(processors: chained.processors + [factory()])
                }
                return BasaltChainedLogitProcessor(processors: [existing, factory()])
            },
            parameterValidator: parameterValidator)
    }

    func makeProcessor(parameters: GenerateParameters) -> (any LogitProcessor)? {
        let penalty = parameters.processor()
        let custom = logitProcessorFactory?()
        switch (penalty, custom) {
        case (nil, nil):
            return nil
        case (let p?, nil):
            return p
        case (nil, let c?):
            return c
        case (let p?, let c?):
            if let chained = c as? BasaltChainedLogitProcessor {
                return BasaltChainedLogitProcessor(processors: [p] + chained.processors)
            }
            return BasaltChainedLogitProcessor(processors: [p, c])
        }
    }
}
