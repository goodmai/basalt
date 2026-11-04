import Foundation
import MLX
import MLXLMCommon
import Tokenizers
import Jinja
import BasaltCore

extension MLXInferenceEngine {
    static let thinkingOpeners = ["<think>", "<|channel>thought", "<|channel|>thought", "[THINK]"]
    static let thinkingClosers = ["</think>", "<channel|>", "<|channel|>", "<|turn|>", "[/THINK]"]
    static let strayThinkingClosers = ["</think>", "<channel|>", "[/THINK]"]
    static let prefillThinkingClosers = ["</think>", "<channel|>", "<|channel|>", "[/THINK]"]
    static let thinkingMarkerHoldback = (thinkingOpeners + thinkingClosers).map(\.count).max()! - 1

    static func firstMarker(_ markers: [String], in text: some StringProtocol) -> Range<String.Index>? {
        markers.compactMap { text.range(of: $0) }.min { $0.lowerBound < $1.lowerBound }
    }

    static func opensThinking(_ prefill: String) -> Bool {
        let start = prefill.drop(while: \.isWhitespace)
        return thinkingOpeners.contains { start.hasPrefix($0) }
    }

    static func endsInsideThinking(_ rendered: String, assistantPrefix: String = "") -> Bool {
        answerAfterThinking(rendered, assistantPrefix: assistantPrefix) == nil
    }

    static func tailOpensThinking(_ rendered: String) -> Bool {
        var tail = Substring(rendered)
        while tail.last?.isWhitespace == true { tail.removeLast() }
        return thinkingOpeners.contains { tail.hasSuffix($0) }
    }

    static func guardedPrefill(_ prefill: String, after renderedTail: String, controlTokens: ControlTokenGuard) -> String {
        var inside = tailOpensThinking(renderedTail)
        var rest = Substring(prefill)
        if inside, let opener = thinkingOpeners.first(where: { rest.drop(while: \.isWhitespace).hasPrefix($0) }) {
            rest = rest.drop(while: \.isWhitespace).dropFirst(opener.count).drop(while: \.isNewline)
        }
        var kept = ""
        while true {
            if inside {
                guard let closer = firstMarker(prefillThinkingClosers, in: rest) else { return kept + controlTokens.neutralize(String(rest)) }
                kept += controlTokens.neutralize(String(rest[..<closer.lowerBound])) + rest[closer]
                rest = rest[closer.upperBound...]
                inside = false
            } else {
                let start = rest.drop(while: \.isWhitespace)
                guard let opener = thinkingOpeners.first(where: { start.hasPrefix($0) }) else { return kept + controlTokens.neutralize(String(rest)) }
                kept += rest[..<start.startIndex] + opener
                rest = start.dropFirst(opener.count)
                inside = true
            }
        }
    }

    static func constraintPhase(thinkingDisabled: Bool, endsInsideThinking: Bool, prefillAnswered: Bool) -> JSONConstraintProcessor.Phase {
        thinkingDisabled || prefillAnswered ? .constrained : endsInsideThinking ? .reasoning : .awaitingAnswer
    }

    static func answerAfterThinking(_ rendered: String, assistantPrefix: String = "") -> Substring? {
        var inside = tailOpensThinking(rendered)
        var rest = Substring(assistantPrefix)
        while true {
            if inside {
                guard let closer = firstMarker(thinkingClosers, in: rest) else { return nil }
                rest = rest[closer.upperBound...]
                inside = false
            } else {
                let start = rest.drop(while: \.isWhitespace)
                guard let opener = thinkingOpeners.first(where: { start.hasPrefix($0) }) else { return start }
                rest = start.dropFirst(opener.count)
                inside = true
            }
        }
    }
    static let liquidReasoning = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>", promptStrategy: .alwaysOn, isSpecialToken: true,
        implicitEndDelimiters: ["<|tool_call_start|>"], budgetTransition: QwenReasoningProtocol.qwen3.budgetTransition)

    static func reasoningProtocol(for profile: any ModelProfile) -> ReasoningConfig? {
        switch profile {
        case is QwenProfile:   QwenReasoningProtocol.tagged
        case is LiquidProfile: liquidReasoning
        default:               nil
        }
    }

    static func budgetTransition(for reasoning: ReasoningConfig, profile: any ModelProfile) -> ReasoningBudgetTransition? {
        if let declared = reasoning.budgetTransition { return declared }
        guard profile is QwenProfile, reasoning.endDelimiter == QwenReasoningProtocol.qwen3.endDelimiter else {
            return nil
        }
        return QwenReasoningProtocol.qwen3.budgetTransition
    }

    static func applyingThinkingBudget(
        _ budget: Int, to components: GenerationComponents, parameters: GenerateParameters,
        reasoning: ReasoningConfig?, tokenizer: any MLXLMCommon.Tokenizer, profile: any ModelProfile
    ) throws(BasaltError) -> GenerationComponents {
        guard let reasoning = reasoning ?? reasoningProtocol(for: profile) else {
            return components
        }
        guard let transition = budgetTransition(for: reasoning, profile: profile) else {
            BasaltLogger(module: "MLXInferenceEngine").info(
                "thinking_budget ignored: \(type(of: profile)) declares no budget transition for \(reasoning.endDelimiter)")
            return components
        }
        let answerReserve = GenerationRequest.thinkingAnswerReserve(maxTokens: parameters.maxTokens ?? 0)
        func applying(_ budget: Int) throws -> GenerationComponents {
            let budgeted = try components.applyingThinkingBudget(
                ThinkingBudgetConfiguration(
                    maximumTokenCount: budget, minimumAnswerTokenCount: answerReserve, transitionOverride: transition),
                reasoning: reasoning, tokenizer: tokenizer)
            try budgeted.validate(parameters: parameters)
            return budgeted
        }
        do {
            do {
                return try applying(budget)
            } catch ThinkingBudgetError.insufficientGenerationTokenLimit(let required, let actual)
                where budget >= required - actual {
                return try applying(budget - (required - actual))
            }
        } catch {
            throw .invalidRequestStructure(details: "thinking_budget: \(error.localizedDescription)")
        }
    }
}
