import Foundation
import MLX
import MLXLMCommon
import BasaltCore

struct RenderedPrompt {
    var input: LMInput
    var makeInput: @Sendable (String) -> UserInput
    var template: String?
    var anchorTurns: [ChatMessageInput]
    var context: [String: any Sendable]?
    var policy: ReasoningPolicy
    var prefill: String?
}

struct PromptInput {
    var input: LMInput
    var tokens: [Int]
    var renderedCount: Int
    var carriesMoreThanTokens: Bool
    var echoedPrefix: String?
    var endsWithThinking: Bool
    var prefixAnswer: String
}

struct CacheClaim {
    var leaseID: Int
    var reuse: (cache: PromptCache, reusedTokens: Int)?
    var mtpPrefix: MTPPrefix?
    var fork: Int?
}

extension MLXInferenceEngine {
    typealias Draft = (makeInput: @Sendable (String) -> UserInput, template: String?, lastText: String)

    func render(_ request: GenerationRequest, tools: [ToolSpec]?, container: ModelContainer) async throws -> RenderedPrompt {
        let continued = request.messages.flatMap(ChatMessageInput.splittingPrefill)
        let prefill = continued.flatMap { $0.prefill.isEmpty ? nil : $0.prefill }
        let requestedPolicy = Self.reasoningPolicy(request: request, defaultReasoningEffort: reasoningEffort)
        let policy = prefill.map { Self.opensThinking($0) ? requestedPolicy : .disabled } ?? requestedPolicy
        let turns = (continued?.turns ?? request.messages).flatMap {
            $0.isEmpty ? nil : profile.turns(controlTokens.neutralize($0), for: policy)
        }
        let templateContext = profile.templateContext(for: policy)
        let context = templateContext.isEmpty ? nil : templateContext
        var draft: Draft
        var anchorTurns: [ChatMessageInput] = []
        if turns == nil, request.prompt.contains("<|turn|>") || request.prompt.contains("<|channel|>") {
            draft = ({ UserInput(prompt: $0) }, "raw", request.prompt)
        } else {
            anchorTurns = turns ?? profile.turns(controlTokens.neutralize([ChatMessageInput(role: .user, text: request.prompt)]), for: policy)
            draft = Self.chatInput(anchorTurns, normalisation: "native", tools: tools, context: context)
        }
        try faults.check(.tokenize)
        let input: LMInput
        do {
            input = try await container.prepare(input: draft.makeInput(draft.lastText))
        } catch where turns != nil && Self.isTemplateRefusal(error) {
            (input, draft, anchorTurns) = try await renderingFallbacks(turns ?? [], refusal: error, tools: tools, context: context, container: container)
        }
        return RenderedPrompt(
            input: input, makeInput: draft.makeInput, template: draft.template, anchorTurns: anchorTurns, context: context,
            policy: policy, prefill: prefill)
    }

    func renderingFallbacks(
        _ turns: [ChatMessageInput], refusal native: any Error, tools: [ToolSpec]?, context: [String: any Sendable]?, container: ModelContainer
    ) async throws -> (LMInput, Draft, [ChatMessageInput]) {
        var refusal = native
        var previous = turns
        for (step, messages) in Self.templateFallbacks(turns) where messages != previous {
            previous = messages
            let draft: Draft = Self.chatInput(messages, normalisation: step, tools: tools, context: context)
            do {
                let input = try await container.prepare(input: draft.makeInput(draft.lastText))
                logger.info("Chat template refused native turns (\(Self.briefly(native))); rendered with \(step) turns")
                return (input, draft, messages)
            } catch where Self.isTemplateRefusal(error) {
                refusal = error
            }
        }
        throw BasaltError.invalidRequestStructure(
            details: "Chat template rejected the conversation: \(Self.briefly(native)); after normalising: \(Self.briefly(refusal))")
    }

    func budgeted(
        _ components: GenerationComponents, _ request: GenerationRequest, _ policy: ReasoningPolicy, params: GenerateParameters,
        container: ModelContainer
    ) async throws -> GenerationComponents {
        guard let budget = request.thinkingBudget, policy != .disabled else { return components }
        let profile = profile
        return try await container.perform { context in
            try Self.applyingThinkingBudget(
                budget, to: components, parameters: params,
                reasoning: context.configuration.reasoningConfig, tokenizer: context.tokenizer, profile: profile)
        }
    }

    func promptInput(_ rendered: RenderedPrompt, request: GenerationRequest, container: ModelContainer) async throws -> PromptInput {
        logger.trace("Input prepared. Starting MLX generation...")
        let carries = Self.carriesMoreThanTokens(rendered.input)
        if carries, rendered.prefill != nil {
            throw BasaltError.invalidRequestStructure(details: "a trailing assistant message cannot be continued on a prompt that carries media")
        }
        let echoedPrefix = carries || rendered.prefill != nil ? nil : request.assistantPrefix.flatMap { $0.isEmpty ? nil : $0 }
        let tailIds = Array(rendered.input.text.tokens.asArray(Int.self).suffix(16))
        let renderedTail = await container.perform { $0.tokenizer.decode(tokenIds: tailIds, skipSpecialTokens: false) }
        let controlTokens = controlTokens
        let assistantPrefix = rendered.prefill.map { Self.guardedPrefill($0, after: renderedTail, controlTokens: controlTokens) } ?? echoedPrefix
        var input = rendered.input
        if let assistantPrefix {
            let ids = await container.perform { $0.tokenizer.encode(text: assistantPrefix, addSpecialTokens: false) }
            input = Self.appending(ids, to: rendered.input)
        }
        let answer = Self.answerAfterThinking(renderedTail, assistantPrefix: assistantPrefix ?? "")
        return PromptInput(
            input: input, tokens: input.text.tokens.asArray(Int.self), renderedCount: rendered.input.text.tokens.dim(-1),
            carriesMoreThanTokens: carries, echoedPrefix: echoedPrefix, endsWithThinking: answer == nil, prefixAnswer: String(answer ?? ""))
    }

    func checkedConstraint(
        _ request: GenerationRequest, _ settings: GenerationPlan.Settings, _ prompt: PromptInput, container: ModelContainer
    ) async throws -> OutputConstraint? {
        let constraint = try await outputConstraint(
            for: request, allowsFastForward: GenerationPlan.plainPrompt(request, settings, carriesMoreThanTokens: prompt.carriesMoreThanTokens))
        guard let constraint, !prompt.prefixAnswer.isEmpty else { return constraint }
        let prefixAnswer = prompt.prefixAnswer
        let ids = await container.perform { $0.tokenizer.encode(text: prefixAnswer, addSpecialTokens: false) }
        guard constraint.session.accepts(ids) else {
            throw BasaltError.invalidRequestStructure(details: "the assistant prefill is not a valid start of the requested JSON output")
        }
        return constraint
    }

    func claimCache(_ namespace: String?, drafts: Bool, tokens: [Int]) throws -> CacheClaim {
        var lease: PromptCacheLease? = if let namespace { promptCaches.lease(namespace, for: tokens) } else { nil }
        var storedPrefix: MTPPrefix?
        if drafts, let namespace {
            let leased = switch lease {
            case let held?: held.reusedTokens
            case nil: 0
            }
            let prefix = Array(tokens.prefix(leased))
            storedPrefix = mtpPrefixes.last { $0.namespace == namespace && $0.tokens == prefix }?.prefix
            if leased > 0, storedPrefix == nil { lease = nil }
        }
        let fork = namespace.map { promptCaches.storedPrefix(of: tokens, in: $0) }
        let reuse: (cache: PromptCache, reusedTokens: Int)? = switch lease {
        case let held?: (held.cache, held.reusedTokens)
        case nil: nil
        }
        try faults.check(.anchorOffsets)
        return CacheClaim(leaseID: hold(lease), reuse: reuse, mtpPrefix: storedPrefix, fork: fork)
    }

    static func shaped(
        _ components: GenerationComponents, _ constraint: OutputConstraint?, _ policy: ReasoningPolicy, _ prompt: PromptInput,
        shaping: (@Sendable () -> EntropyDynamicTemperature)?, exclusion: (@Sendable () -> ExcludeTopChoices)?
    ) -> GenerationComponents {
        let masked = constraining(
            components, with: constraint,
            phase: constraintPhase(
                thinkingDisabled: policy == .disabled, endsInsideThinking: prompt.endsWithThinking, prefillAnswered: !prompt.prefixAnswer.isEmpty))
        let excluded = exclusion.map { masked.appendingBasaltProcessor($0) } ?? masked
        return shaping.map { excluded.appendingBasaltProcessor($0) } ?? excluded
    }

    func route(
        _ options: inout LaunchOptions, _ plan: GenerationPlan, jumpForward: Bool, mtpPrefix: MTPPrefix?, container: ModelContainer, epoch: ModelEpoch
    ) async {
        let expertRouting = (plan.speculativeLookup || jumpForward) ? await routesExperts(container, epoch: epoch) : true
        options.batch = plan.batchable ? await batchedDecoder(container, epoch: epoch) : nil
        options.lookup = LookupOptions(
            speculative: plan.speculativeLookup, recycling: speculativeRecycling, jumpForward: jumpForward, routesExperts: expertRouting)
        (options.kvZones, options.kvHeadMask, options.questTopK, options.kvPrecision) = (plan.kvZones, plan.kvHeadMask, plan.questTopK, plan.kvPrecision)
        (options.mtp, options.mtpBlockSize, options.mtpPrefix) = (plan.drafts, mtpBlockSize, mtpPrefix)
    }

    func launching(_ input: LMInput, _ prompt: CachedPrompt, options: LaunchOptions, container: ModelContainer) async throws -> GenerationLaunch {
        try faults.check(.launch)
        return try await container.performEvaluating(nonSendable: (input, prompt)) { context, values in
            try Self.launch(input: values.0, prompt: values.1, options: options, context: context)
        }
    }
}
