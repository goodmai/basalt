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
        params.quantizedKVStart = configuration.quantizedKVStart
        params.repetitionPenalty = request.repetitionPenalty.map(Float.init) ?? configuration.repetitionPenalty
            ?? samplingDefaults.repetitionPenalty.map(Float.init)
        if let topK = request.topK ?? configuration.topK ?? samplingDefaults.topK { params.topK = topK }
        if let effectiveMinP = request.minP.map(Float.init) ?? configuration.minP ?? profile.defaultMinP {
            params.minP = effectiveMinP
        }
        params.seed = configuration.seed

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
        if let probe = request.logitProbe {
            components = components.appendingBasaltProcessor { LogitProbeRecorder(probe: probe) }
        }
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
        let row = logits.reshaped(-1).asType(.float32)
        guard NonFiniteLogitGuard.isUsable(row) else { return nil }
        let maxLogprob = (row - logSumExp(row)).max().item(Float.self)
        let prob = Double(exp(maxLogprob))
        return prob.isFinite ? min(max(prob, 0.0), 1.0) : nil
    }

    mutating func prompt(_ prompt: MLXArray) {}

    static func statistics(logits: MLXArray) -> (probability: Double, entropy: Double, margin: Double)? {
        let row = logits.reshaped(-1).asType(.float32)
        guard NonFiniteLogitGuard.isUsable(row) else { return nil }
        let logprobs = row - logSumExp(row)
        let best = top(logprobs, k: min(2, row.dim(0)))
        let entropy = -(exp(logprobs) * logprobs).sum()
        let values = concatenated([best.max().reshaped(1), best.min().reshaped(1), entropy.reshaped(1)]).asArray(Float.self)
        let (first, second) = (Double(exp(values[0])), Double(exp(values[1])))
        guard first.isFinite, second.isFinite, values[2].isFinite else { return nil }
        return (min(max(first, 0), 1), Double(values[2]), max(0, first - second))
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
