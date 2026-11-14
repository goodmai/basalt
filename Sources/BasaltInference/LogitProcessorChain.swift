import MLX
import MLXLMCommon

protocol LogitStageConvertible: LogitProcessor {
    var logitStage: LogitStage { get }
}

extension PenaltyProcessor: LogitStageConvertible {
    var logitStage: LogitStage { .penalty(self) }
}

extension RepetitionContext: LogitStageConvertible {
    var logitStage: LogitStage { .repetition(self) }
}

extension GeneratedTokensPresencePenalty: LogitStageConvertible {
    var logitStage: LogitStage { .presence(self) }
}

extension DRYPenalty: LogitStageConvertible {
    var logitStage: LogitStage { .dry(self) }
}

extension LogitProbeRecorder: LogitStageConvertible {
    var logitStage: LogitStage { .logitProbe(self) }
}

extension NoRepeatNGram: LogitStageConvertible {
    var logitStage: LogitStage { .noRepeatNGram(self) }
}

extension ConfidenceProbeRecorder: LogitStageConvertible {
    var logitStage: LogitStage { .confidence(self) }
}

extension JSONConstraintProcessor: LogitStageConvertible {
    var logitStage: LogitStage { .jsonConstraint(self) }
}

extension ExcludeTopChoices: LogitStageConvertible {
    var logitStage: LogitStage { .excludeTopChoices(self) }
}

extension EntropyDynamicTemperature: LogitStageConvertible {
    var logitStage: LogitStage { .dynamicTemperature(self) }
}

extension NonFiniteLogitGuard: LogitStageConvertible {
    var logitStage: LogitStage { .nonFiniteGuard(self) }
}

enum LogitStage {
    case penalty(PenaltyProcessor)
    case repetition(RepetitionContext)
    case presence(GeneratedTokensPresencePenalty)
    case dry(DRYPenalty)
    case logitProbe(LogitProbeRecorder)
    case noRepeatNGram(NoRepeatNGram)
    case confidence(ConfidenceProbeRecorder)
    case jsonConstraint(JSONConstraintProcessor)
    case excludeTopChoices(ExcludeTopChoices)
    case dynamicTemperature(EntropyDynamicTemperature)
    case nonFiniteGuard(NonFiniteLogitGuard)
    case other(any LogitProcessor)

    init(_ processor: any LogitProcessor) {
        if let stage = (processor as? any LogitStageConvertible)?.logitStage {
            self = stage
        } else {
            self = .other(processor)
        }
    }

    var carriesForcedTokens: Bool {
        switch self {
        case .jsonConstraint: true
        case .other(let value): value is any ForcedTokenSource
        case .penalty, .repetition, .presence, .dry, .logitProbe, .noRepeatNGram, .confidence, .excludeTopChoices, .dynamicTemperature, .nonFiniteGuard:
            false
        }
    }

    var erased: any LogitProcessor {
        switch self {
        case .penalty(let value): value
        case .repetition(let value): value
        case .presence(let value): value
        case .dry(let value): value
        case .logitProbe(let value): value
        case .noRepeatNGram(let value): value
        case .confidence(let value): value
        case .jsonConstraint(let value): value
        case .excludeTopChoices(let value): value
        case .dynamicTemperature(let value): value
        case .nonFiniteGuard(let value): value
        case .other(let value): value
        }
    }

    func copied() -> LogitStage {
        switch self {
        case .dry(let value): .dry(value.copy())
        case .other(let value): .other(value.copy())
        case .penalty, .repetition, .presence, .logitProbe, .noRepeatNGram, .confidence, .jsonConstraint, .excludeTopChoices, .dynamicTemperature, .nonFiniteGuard:
            self
        }
    }

    mutating func prompt(_ tokens: MLXArray) {
        switch self {
        case .penalty(var value):
            value.prompt(tokens)
            self = .penalty(value)
        case .repetition(var value):
            value.prompt(tokens)
            self = .repetition(value)
        case .dry(var value):
            value.prompt(tokens)
            self = .dry(value)
        case .noRepeatNGram(var value):
            value.prompt(tokens)
            self = .noRepeatNGram(value)
        case .other(var value):
            value.prompt(tokens)
            self = .other(value)
        case .presence, .logitProbe, .confidence, .jsonConstraint, .excludeTopChoices, .dynamicTemperature, .nonFiniteGuard:
            break
        }
    }

    func process(logits: MLXArray) -> MLXArray {
        switch self {
        case .penalty(let value): value.process(logits: logits)
        case .repetition(let value): value.process(logits: logits)
        case .presence(let value): value.process(logits: logits)
        case .dry(let value): value.process(logits: logits)
        case .logitProbe(let value): value.process(logits: logits)
        case .noRepeatNGram(let value): value.process(logits: logits)
        case .confidence(let value): value.process(logits: logits)
        case .jsonConstraint(let value): value.process(logits: logits)
        case .excludeTopChoices(let value): value.process(logits: logits)
        case .dynamicTemperature(let value): value.process(logits: logits)
        case .nonFiniteGuard(let value): value.process(logits: logits)
        case .other(let value): value.process(logits: logits)
        }
    }

    mutating func didSample(token: MLXArray) {
        switch self {
        case .penalty(var value):
            value.didSample(token: token)
            self = .penalty(value)
        case .repetition(var value):
            value.didSample(token: token)
            self = .repetition(value)
        case .presence(var value):
            value.didSample(token: token)
            self = .presence(value)
        case .dry(var value):
            value.didSample(token: token)
            self = .dry(value)
        case .logitProbe(var value):
            value.didSample(token: token)
            self = .logitProbe(value)
        case .noRepeatNGram(var value):
            value.didSample(token: token)
            self = .noRepeatNGram(value)
        case .jsonConstraint(var value):
            value.didSample(token: token)
            self = .jsonConstraint(value)
        case .dynamicTemperature(var value):
            value.didSample(token: token)
            self = .dynamicTemperature(value)
        case .nonFiniteGuard(var value):
            value.didSample(token: token)
            self = .nonFiniteGuard(value)
        case .other(var value):
            value.didSample(token: token)
            self = .other(value)
        case .confidence, .excludeTopChoices:
            break
        }
    }

    mutating func takeForcedTokens(limit: Int) -> [Int] {
        switch self {
        case .jsonConstraint(var value):
            let tokens = value.takeForcedTokens(limit: limit)
            self = .jsonConstraint(value)
            return tokens
        case .other(let value):
            guard var source = value as? any ForcedTokenSource else { return [] }
            let tokens = source.takeForcedTokens(limit: limit)
            self = .other(source)
            return tokens
        case .penalty, .repetition, .presence, .dry, .logitProbe, .noRepeatNGram, .confidence, .excludeTopChoices, .dynamicTemperature, .nonFiniteGuard:
            return []
        }
    }

    mutating func didSampleForced(_ token: MLXArray) {
        switch self {
        case .jsonConstraint(var value):
            value.didSampleForced(token)
            self = .jsonConstraint(value)
        case .other(let value):
            if var source = value as? any ForcedTokenSource {
                source.didSampleForced(token)
                self = .other(source)
            } else {
                var copy = value
                copy.didSample(token: token)
                self = .other(copy)
            }
        case .penalty, .repetition, .presence, .dry, .logitProbe, .noRepeatNGram, .confidence, .excludeTopChoices, .dynamicTemperature, .nonFiniteGuard:
            didSample(token: token)
        }
    }
}

typealias BasaltChainedLogitProcessor = LogitProcessorChain

struct LogitProcessorChain: LogitProcessor, ForcedTokenSource {
    var stages: ContiguousArray<LogitStage>
    var hasForcedSource = false

    var processors: [any LogitProcessor] {
        stages.map(\.erased)
    }

    init(capacity: Int = 0) {
        stages = ContiguousArray()
        if capacity > 0 { stages.reserveCapacity(capacity) }
    }

    init(single processor: any LogitProcessor) {
        stages = ContiguousArray()
        stages.reserveCapacity(2)
        append(processor)
    }

    init(processors: [any LogitProcessor]) {
        stages = ContiguousArray()
        stages.reserveCapacity(processors.count)
        for processor in processors { append(processor) }
    }

    mutating func appendStage(_ stage: LogitStage) {
        stages.append(stage)
        hasForcedSource = hasForcedSource || stage.carriesForcedTokens
    }

    mutating func append(_ processor: any LogitProcessor) {
        if let nested = processor as? LogitProcessorChain {
            stages.append(contentsOf: nested.stages)
            hasForcedSource = hasForcedSource || nested.hasForcedSource
        } else {
            appendStage(LogitStage(processor))
        }
    }

    mutating func prepend(_ processor: any LogitProcessor) {
        if let nested = processor as? LogitProcessorChain {
            var combined = nested.stages
            combined.append(contentsOf: stages)
            stages = combined
            hasForcedSource = hasForcedSource || nested.hasForcedSource
        } else {
            let stage = LogitStage(processor)
            stages.insert(stage, at: 0)
            hasForcedSource = hasForcedSource || stage.carriesForcedTokens
        }
    }

    func copy() -> LogitProcessorChain {
        var duplicate = LogitProcessorChain(capacity: stages.count)
        for stage in stages { duplicate.stages.append(stage.copied()) }
        duplicate.hasForcedSource = hasForcedSource
        return duplicate
    }

    mutating func prompt(_ prompt: MLXArray) {
        for index in stages.indices { stages[index].prompt(prompt) }
    }

    func process(logits: MLXArray) -> MLXArray {
        var current = logits
        for index in stages.indices { current = stages[index].process(logits: current) }
        return current
    }

    mutating func didSample(token: MLXArray) {
        for index in stages.indices { stages[index].didSample(token: token) }
    }

    mutating func takeForcedTokens(limit: Int) -> [Int] {
        guard hasForcedSource, limit > 0 else { return [] }
        var result: [Int] = []
        for index in stages.indices where result.count < limit {
            result.append(contentsOf: stages[index].takeForcedTokens(limit: limit - result.count))
        }
        return result
    }

    mutating func didSampleForced(_ token: MLXArray) {
        for index in stages.indices { stages[index].didSampleForced(token) }
    }
}
