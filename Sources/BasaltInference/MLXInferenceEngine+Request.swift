import Foundation
import CoreImage
import MLX
import MLXLMCommon
import MLXVLM
import BasaltCore

struct RenderedPrompt {
    var input: LMInput
    var makeInput: @Sendable (String) -> UserInput
    var template: String?
    var anchorTurns: [ChatMessageInput]
    var context: [String: any Sendable]?
    var policy: ReasoningPolicy
    var prefill: String?
    var images: [String: CIImage]
}

struct PromptInput {
    var input: LMInput
    var tokens: [Int]
    var renderedCount: Int
    var carriesMoreThanTokens: Bool
    var echoedPrefix: String?
    var endsWithThinking: Bool
    var prefixAnswer: String
    var media: PromptMedia
}

struct RelayRoute {
    var namespace: String?
    var leaseID: Int
    var epoch: ModelEpoch
    var container: ModelContainer
    var alarm: NonFiniteLogitAlarm
    var startsThinking: Bool
}

struct CacheClaim {
    var leaseID: Int
    var reuse: (cache: PromptCache, reusedTokens: Int)?
    var mtpPrefix: MTPPrefix?
    var fork: Int?
}

extension MLXInferenceEngine {
    typealias Draft = (makeInput: @Sendable (String) -> UserInput, template: String?, lastText: String)

    func render(
        _ request: GenerationRequest, tools: [ToolSpec]?, images: [String: CIImage], container: ModelContainer
    ) async throws -> RenderedPrompt {
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
            draft = Self.chatInput(anchorTurns, normalisation: "native", tools: tools, context: context, images: images)
        }
        try faults.check(.tokenize)
        let input: LMInput
        do {
            input = try await container.prepare(input: draft.makeInput(draft.lastText))
        } catch where turns != nil && Self.isTemplateRefusal(error) {
            (input, draft, anchorTurns) = try await renderingFallbacks(
                turns ?? [], refusal: error, tools: tools, context: context, images: images, container: container)
        }
        return RenderedPrompt(
            input: input, makeInput: draft.makeInput, template: draft.template, anchorTurns: anchorTurns, context: context,
            policy: policy, prefill: prefill, images: images)
    }

    func renderingFallbacks(
        _ turns: [ChatMessageInput], refusal native: any Error, tools: [ToolSpec]?, context: [String: any Sendable]?,
        images: [String: CIImage], container: ModelContainer
    ) async throws -> (LMInput, Draft, [ChatMessageInput]) {
        var refusal = native
        var previous = turns
        for (step, messages) in Self.templateFallbacks(turns) where messages != previous {
            previous = messages
            let draft: Draft = Self.chatInput(messages, normalisation: step, tools: tools, context: context, images: images)
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

    func promptInput(_ rendered: RenderedPrompt, request: GenerationRequest, container: ModelContainer, epoch: ModelEpoch) async throws -> PromptInput {
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
        let tokens = input.text.tokens.asArray(Int.self)
        let snapshotsCarryMedia = carries ? await cacheIsHybrid(container, epoch: epoch) : false
        return PromptInput(
            input: input, tokens: tokens, renderedCount: rendered.input.text.tokens.dim(-1),
            carriesMoreThanTokens: carries, echoedPrefix: echoedPrefix, endsWithThinking: answer == nil, prefixAnswer: String(answer ?? ""),
            media: .of(request.messages, carriesMoreThanTokens: carries, tokens: tokens, imageToken: snapshotsCarryMedia ? modelFacts.imageToken : nil))
    }

    func checkImages(_ messages: [ChatMessageInput]?, container: ModelContainer) async throws(BasaltError) -> [String: CIImage] {
        guard let messages, messages.contains(where: { !$0.images.isEmpty }) else { return [:] }
        try ImageInput.checkLimits(messages.flatMap(\.images))
        let (name, takesImages) = await container.perform { ($0.configuration.name, $0.model is any VLMModel) }
        guard takesImages else {
            let model = ModelCache.canonicalRepoId(from: modelFacts.description.isEmpty ? name : modelFacts.description)
            throw .invalidRequestStructure(details: "\(model) does not take images")
        }
        return try Self.decodedImages(messages)
    }

    func checkedConstraint(
        _ request: GenerationRequest, _ settings: GenerationPlan.Settings, _ prompt: PromptInput, container: ModelContainer
    ) async throws -> OutputConstraint? {
        let constraint = try await outputConstraint(
            for: request,
            allowsFastForward: prompt.prefixAnswer.isEmpty && GenerationPlan.plainPrompt(request, settings, carriesMoreThanTokens: prompt.carriesMoreThanTokens))
        guard let constraint, !prompt.prefixAnswer.isEmpty else { return constraint }
        let prefixAnswer = prompt.prefixAnswer
        let ids = await container.perform { $0.tokenizer.encode(text: prefixAnswer, addSpecialTokens: false) }
        guard constraint.session.accepts(ids) else {
            throw BasaltError.invalidRequestStructure(details: "the assistant prefill is not a valid start of the requested JSON output")
        }
        return constraint
    }

    func claimCache(_ namespace: String?, drafts: Bool, tokens: [Int], mediaEnd: Int) throws -> CacheClaim {
        var lease: PromptCacheLease? = if let namespace { promptCaches.lease(namespace, for: tokens) } else { nil }
        let leased = switch lease {
        case let held?: held.reusedTokens
        case nil: 0
        }
        if leased < mediaEnd { lease = nil }
        var storedPrefix: MTPPrefix?
        if drafts, let namespace {
            let prefix = Array(tokens.prefix(leased))
            storedPrefix = mtpPrefixes.last { $0.namespace == namespace && $0.tokens == prefix }?.prefix
            if leased > 0, storedPrefix == nil { lease = nil }
        }
        let fork = mediaEnd > 0 ? nil : namespace.map { promptCaches.storedPrefix(of: tokens, in: $0) }
        let reuse: (cache: PromptCache, reusedTokens: Int)? = switch lease {
        case let held?: (held.cache, held.reusedTokens)
        case nil: nil
        }
        try faults.check(.anchorOffsets)
        return CacheClaim(leaseID: hold(lease), reuse: reuse, mtpPrefix: storedPrefix, fork: fork)
    }
}

extension MLXInferenceEngine {
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
        options.pacer = prefillPacer
    }

    func launching(_ input: LMInput, _ prompt: CachedPrompt, options: LaunchOptions, container: ModelContainer) async throws -> GenerationLaunch {
        try faults.check(.launch)
        return try await container.performEvaluating(nonSendable: (input, prompt)) { context, values in
            try Self.launch(input: values.0, prompt: values.1, options: options, context: context)
        }
    }

    var planSettings: GenerationPlan.Settings {
        GenerationPlan.Settings(
            kvPrecision: kvPrecision, kvZones: kvZones, attendsThroughZones: modelAttendsThroughZones, prefillsInChunks: modelPrefillsInChunks,
            kvHeadMask: kvHeadMask, questTopK: questApplies ? questTopK : nil, speculativeLookup: speculativeLookup,
            capabilities: modelFacts.capabilities, adapter: adapterNamespace)
    }

    func checkedSampling(_ request: GenerationRequest) async throws(BasaltError) -> (
        GenerateParameters, GenerationComponents, shaping: (@Sendable () -> EntropyDynamicTemperature)?,
        exclusion: (@Sendable () -> ExcludeTopChoices)?
    ) {
        if request.usesExperimentalSamplers, !experimentalSamplers {
            throw .invalidRequestStructure(
                details: "dry_* and xtc_* are experimental (DRY can turn a detectable loop into one with corrupted facts on low-bit models); start the server with --experimental-samplers to use them")
        }
        return sampling(for: request, breakers: await sequenceBreakerIds(for: request))
    }

    func boundary(_ namespace: String?, _ prompt: PromptInput, _ rendered: RenderedPrompt, container: ModelContainer, epoch: ModelEpoch) async throws -> Int {
        try faults.check(.snapshotBoundary)
        guard namespace != nil else { return prompt.tokens.count - 1 }
        return await snapshotBoundary(prompt.renderedCount, container: container, epoch: epoch, template: rendered.template, rendered.makeInput)
    }

    func restoreIfNeeded(_ namespace: String?, drafts: Bool, tokens: [Int], container: ModelContainer, epoch: ModelEpoch) async throws {
        guard let namespace, !drafts else { return }
        try faults.check(.restoreFromDisk)
        await restoreFromDisk(namespace, prompt: tokens, container: container, epoch: epoch)
    }

    func relayStream(
        _ launch: GenerationLaunch, _ request: GenerationRequest, _ prompt: PromptInput, _ constraint: OutputConstraint?, _ plan: GenerationPlan,
        route: RelayRoute
    ) -> AsyncStream<StreamChunk> {
        let relay = GenerationRelay(GenerationRelay.Setup(
            request: request, promptTokens: prompt.tokens.count, reusedTokens: launch.reusedTokens.rawValue, alarm: route.alarm,
            controlTokens: controlTokens.tokens, startsThinking: route.startsThinking,
            scansToolCallText: profile.writesToolCallsAsText && request.tools?.isEmpty == false,
            echoedPrefix: prompt.echoedPrefix, forcedTool: constraint?.toolName, prefixAnswer: prompt.prefixAnswer,
            guardsRepetition: constraint == nil, draftSource: plan.draftSource))
        let inputs = RelayInputs(
            stream: launch.stream, task: launch.task, prefillSeconds: launch.prefillSeconds,
            promptTokens: prompt.tokens, namespace: route.namespace, leaseID: route.leaseID, epoch: route.epoch, constraint: constraint,
            container: route.container, faults: faults)
        return AsyncStream<StreamChunk> { continuation in
            let task = Task { @concurrent in
                await self.relay(relay, inputs, into: continuation)
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
