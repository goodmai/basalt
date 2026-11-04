import Foundation
import MLX
import MLXLLM
// Registers MLXVLM's factory trampoline (Gemma 4, Qwen3.5-VL, …). The registry
// finds it via NSClassFromString, so the module must actually be linked in.
import MLXVLM
import MLXLMCommon
import MLXHuggingFace
import MLXGuidedGeneration
import Tokenizers
import Hub
import Jinja
import BasaltCore

extension Duration {
    var inSeconds: Double {
        Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}

// MARK: — Real implementation

public actor MLXInferenceEngine: InferenceEngine {

    private(set) var state = EngineState.unloaded(0)
    let logger = BasaltLogger(module: "MLXInferenceEngine")

    /// Chat-template knob for models with a reasoning budget (Qwen3.8: xhigh | medium | low).
    /// Left nil the template picks its own default — for Qwen3.8 that is `xhigh`, which
    /// can spend the whole token budget thinking and never emit an answer.
    private let reasoningEffort: String?

    let kvPrecision: KVPrecision?
    let quantizedKVStart: Int
    let kvZones: KVZones?
    let wiresWorkingSet: Bool
    var probesModel = false
    var wiredTicket: WiredMemoryTicket?
    let kvHeadSplit: Double?
    let questTopK: Int?
    let repetitionPenalty: Float?
    let topK: Int?
    let minP: Float?
    let seed: UInt64?
    let gpuCacheLimit: Int?

    var promptCaches: PromptCacheTree
    private(set) var leases: [Int: PromptCacheLeaseSlot] = [:]
    private(set) var launches: [Int: GenerationLaunch] = [:]
    var faults = FaultPlan()
    var experiments = (epoch: ModelEpoch(0), registry: OverlayRegistry())
    private var leaseCount = 0
    private(set) var mtpPrefixes: [(namespace: String, tokens: [Int], prefix: MTPPrefix)] = []
    private var promptCacheFullBudget: ByteCount
    private(set) var appliedMemoryDirective = MemoryDirective.relaxed
    private var kvMemoryBudget: Int64?
    private let configuredPromptCacheBytes: ByteCount?
    private let hybridAnchors: Int
    var templateSuffixes: [String: Int] = [:]
    var grammarVocabulary: Result<GrammarVocabulary, ConstraintUnavailable>?
    var sequenceBreakers: (strings: [String], ids: Set<Int>)?
    var grammarPool = GrammarPool()
    private let cacheBlendRatio: Float?
    private(set) var cacheBlend: CacheBlendStore?
    let speculativeLookup: Bool
    private let speculativeRecycling: Bool
    let mtp: Bool
    private let mtpBlockSize: Int
    private let experimentalSamplers: Bool
    let grammarJumpForward: Bool
    public var kvMemoryBudgetBytes: Int64? { kvMemoryBudget }
    var promptCacheDisk: PromptCacheDiskStore?
    let batchSlots: Int
    var batchDecoder: BatchedDecoder?

    public init(
        reasoningEffort: String? = nil,
        gpuCacheLimit: Int? = 512 << 20,
        promptCacheBytes: Int? = nil,
        kvPrecision: KVPrecision? = nil,
        quantizedKVStart: Int = KVCacheGeometry.defaultQuantizedKVStart,
        kvZones: KVZones? = nil,
        kvHeadSplit: Double? = nil,
        questTopK: Int? = nil,
        repetitionPenalty: Float? = nil,
        topK: Int? = nil,
        minP: Float? = nil,
        seed: UInt64? = nil,
        cacheBlendRatio: Float? = nil,
        speculativeLookup: Bool = false,
        speculativeRecycling: Bool = false,
        mtp: Bool = false,
        mtpBlockSize: Int = 4,
        experimentalSamplers: Bool = false,
        grammarJumpForward: Bool = false,
        promptCacheDisk: PromptCacheDiskStore? = nil,
        hybridAnchors: Int = 8,
        batchSlots: Int = 1,
        wiresWorkingSet: Bool = false
    ) {
        self.wiresWorkingSet = wiresWorkingSet
        self.hybridAnchors = hybridAnchors
        self.batchSlots = max(1, batchSlots)
        self.promptCacheDisk = promptCacheDisk
        self.speculativeLookup = speculativeLookup
        self.speculativeRecycling = speculativeRecycling
        self.mtp = mtp
        self.mtpBlockSize = max(2, mtpBlockSize)
        self.experimentalSamplers = experimentalSamplers
        self.grammarJumpForward = grammarJumpForward
        self.reasoningEffort = reasoningEffort
        self.gpuCacheLimit = gpuCacheLimit
        self.kvPrecision = kvPrecision
        self.quantizedKVStart = quantizedKVStart
        self.kvZones = kvZones
        self.kvHeadSplit = kvHeadSplit
        self.questTopK = questTopK
        self.repetitionPenalty = repetitionPenalty
        self.topK = topK
        self.minP = minP
        self.seed = seed
        self.configuredPromptCacheBytes = promptCacheBytes.map { ByteCount($0) }
        self.cacheBlendRatio = cacheBlendRatio
        self.promptCaches = PromptCacheTree(byteBudget: ByteCount(promptCacheBytes ?? 0), anchorLimit: max(1, hybridAnchors))
        self.promptCacheFullBudget = ByteCount(promptCacheBytes ?? 0)
    }

    public func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk> {
        guard let container else {
            throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first")
        }
        let epoch = modelEpoch

        if request.usesExperimentalSamplers, !experimentalSamplers {
            throw .invalidRequestStructure(
                details: "dry_* and xtc_* are experimental (DRY can turn a detectable loop into one with corrupted facts on low-bit models); start the server with --experimental-samplers to use them")
        }
        let (params, unbudgeted, shaping, exclusion) = sampling(for: request, breakers: await sequenceBreakerIds(for: request))
        let settings = GenerationPlan.Settings(
            kvPrecision: kvPrecision, kvZones: kvZones, attendsThroughZones: modelAttendsThroughZones, prefillsInChunks: modelPrefillsInChunks,
            kvHeadMask: kvHeadMask, questTopK: questApplies ? questTopK : nil, speculativeLookup: speculativeLookup,
            capabilities: modelFacts.capabilities)

        let toolSpecs = Self.toolSpecs(request.tools)
        let scansToolCallText = profile.writesToolCallsAsText && request.tools?.isEmpty == false
        var makeInput: @Sendable (String) -> UserInput
        var template: String?
        let continued = request.messages.flatMap(ChatMessageInput.splittingPrefill)
        let prefill = continued.flatMap { $0.prefill.isEmpty ? nil : $0.prefill }
        let requestedPolicy = Self.reasoningPolicy(request: request, defaultReasoningEffort: reasoningEffort)
        let policy = prefill.map { Self.opensThinking($0) ? requestedPolicy : .disabled } ?? requestedPolicy
        let turns = (continued?.turns ?? request.messages).flatMap {
            $0.isEmpty ? nil : profile.turns(controlTokens.neutralize($0), for: policy)
        }
        var lastText: String
        var anchorTurns: [ChatMessageInput] = []
        let templateContext = profile.templateContext(for: policy)
        let context = templateContext.isEmpty ? nil : templateContext
        if turns == nil, request.prompt.contains("<|turn|>") || request.prompt.contains("<|channel|>") {
            makeInput = { UserInput(prompt: $0) }
            template = "raw"
            lastText = request.prompt
        } else {
            anchorTurns = turns ?? profile.turns(controlTokens.neutralize([ChatMessageInput(role: .user, text: request.prompt)]), for: policy)
            (makeInput, template, lastText) = Self.chatInput(
                anchorTurns, normalisation: "native", tools: toolSpecs, context: context)
        }

        do {
            let components: GenerationComponents
            if let budget = request.thinkingBudget, policy != .disabled {
                let profile = profile
                components = try await container.perform { context in
                    try Self.applyingThinkingBudget(
                        budget, to: unbudgeted, parameters: params,
                        reasoning: context.configuration.reasoningConfig, tokenizer: context.tokenizer, profile: profile)
                }
            } else {
                components = unbudgeted
            }
            try faults.check(.tokenize)
            let rendered: LMInput
            do {
                rendered = try await container.prepare(input: makeInput(lastText))
            } catch where turns != nil && Self.isTemplateRefusal(error) {
                let nativeRefusal = error
                var refusal = error
                var prepared: LMInput?
                var previous = turns
                for (step, messages) in Self.templateFallbacks(turns ?? []) where prepared == nil && messages != previous {
                    previous = messages
                    anchorTurns = messages
                    (makeInput, template, lastText) = Self.chatInput(
                        messages, normalisation: step, tools: toolSpecs, context: context)
                    do {
                        prepared = try await container.prepare(input: makeInput(lastText))
                        logger.info("Chat template refused native turns (\(Self.briefly(nativeRefusal))); rendered with \(step) turns")
                    } catch where Self.isTemplateRefusal(error) {
                        refusal = error
                    }
                }
                guard let prepared else {
                    throw BasaltError.invalidRequestStructure(
                        details: "Chat template rejected the conversation: \(Self.briefly(nativeRefusal)); after normalising: \(Self.briefly(refusal))")
                }
                rendered = prepared
            }
            logger.trace("Input prepared. Starting MLX generation...")
            let renderedCount = rendered.text.tokens.dim(-1)
            let carriesMoreThanTokens = Self.carriesMoreThanTokens(rendered)
            if carriesMoreThanTokens, prefill != nil {
                throw BasaltError.invalidRequestStructure(
                    details: "a trailing assistant message cannot be continued on a prompt that carries media")
            }
            let echoedPrefix = carriesMoreThanTokens || prefill != nil ? nil : request.assistantPrefix.flatMap { $0.isEmpty ? nil : $0 }
            let tailIds = Array(rendered.text.tokens.asArray(Int.self).suffix(16))
            let renderedTail = await container.perform { $0.tokenizer.decode(tokenIds: tailIds, skipSpecialTokens: false) }
            let controlTokens = controlTokens
            let assistantPrefix = prefill.map { Self.guardedPrefill($0, after: renderedTail, controlTokens: controlTokens) } ?? echoedPrefix
            let lmInput: LMInput
            if let assistantPrefix {
                let ids = await container.perform { $0.tokenizer.encode(text: assistantPrefix, addSpecialTokens: false) }
                lmInput = Self.appending(ids, to: rendered)
            } else {
                lmInput = rendered
            }
            let promptTokens = lmInput.text.tokens.asArray(Int.self)
            let answerAfterThinking = Self.answerAfterThinking(renderedTail, assistantPrefix: assistantPrefix ?? "")
            let promptEndsWithThinking = answerAfterThinking == nil
            let prefixAnswer = String(answerAfterThinking ?? "")
            let namespace = GenerationPlan.cacheNamespace(
                request, settings, cacheEnabled: promptCaches.byteBudget > 0 && !modelModified, carriesMoreThanTokens: carriesMoreThanTokens, sameEpoch: epoch == modelEpoch)
            let constraint = try await outputConstraint(
                for: request, allowsFastForward: GenerationPlan.plainPrompt(request, settings, carriesMoreThanTokens: carriesMoreThanTokens))
            if let constraint, !prefixAnswer.isEmpty {
                let ids = await container.perform { $0.tokenizer.encode(text: prefixAnswer, addSpecialTokens: false) }
                guard constraint.session.accepts(ids) else {
                    throw BasaltError.invalidRequestStructure(
                        details: "the assistant prefill is not a valid start of the requested JSON output")
                }
            }
            try faults.check(.snapshotBoundary)
            let boundary = namespace == nil
                ? promptTokens.count - 1
                : await snapshotBoundary(renderedCount, container: container, epoch: epoch, template: template, makeInput)
            let plan = GenerationPlan.make(request, settings, GenerationPlan.Facts(
                namespace: namespace, promptTokens: promptTokens.count, carriesMoreThanTokens: carriesMoreThanTokens,
                constrained: constraint != nil, hasMTPDrafter: hasMTPDrafter, blendAvailable: cacheBlend != nil,
                modelIsHybrid: modelIsHybrid, modelHasRecurrentState: modelHasRecurrentState))
            if let namespace, !plan.drafts {
                try faults.check(.restoreFromDisk)
                await restoreFromDisk(namespace, prompt: promptTokens, container: container, epoch: epoch)
            }
            var lease: PromptCacheLease? = if let namespace { promptCaches.lease(namespace, for: promptTokens) } else { nil }
            var storedPrefix: MTPPrefix?
            if plan.drafts, let namespace {
                let leased = switch lease {
                case let held?: held.reusedTokens
                case nil: 0
                }
                let prefix = Array(promptTokens.prefix(leased))
                storedPrefix = mtpPrefixes.last { $0.namespace == namespace && $0.tokens == prefix }?.prefix
                if leased > 0, storedPrefix == nil { lease = nil }
            }
            let mtpPrefix = storedPrefix
            let fork = namespace.map { promptCaches.storedPrefix(of: promptTokens, in: $0) }
            let reuse: (cache: PromptCache, reusedTokens: Int)? = switch lease {
            case let held?: (held.cache, held.reusedTokens)
            case nil: nil
            }
            try faults.check(.anchorOffsets)
            let leaseID = hold(lease)
            let anchors = plan.findsAnchors
                ? await anchorOffsets(
                    anchorTurns, AnchorSearch(prompt: promptTokens, reused: reuse?.reusedTokens ?? 0, boundary: boundary, limit: hybridAnchors),
                    container: container, tools: toolSpecs, context: context)
                : []
            let blending = plan.blends && epoch == modelEpoch ? namespace.flatMap { space in cacheBlend.map { ($0, space) } } : nil
            let masked = Self.constraining(
                components, with: constraint,
                phase: Self.constraintPhase(
                    thinkingDisabled: policy == .disabled, endsInsideThinking: promptEndsWithThinking,
                    prefillAnswered: !prefixAnswer.isEmpty))
            let excluded = exclusion.map { masked.appendingBasaltProcessor($0) } ?? masked
            let (constrained, alarm) = NonFiniteLogitGuard.guarding(
                shaping.map { excluded.appendingBasaltProcessor($0) } ?? excluded, temperature: params.temperature)
            let forcedTool = constraint?.toolName
            let jumpForward = constraint?.fastForward == true
            let expertRouting = (plan.speculativeLookup || jumpForward) ? await routesExperts(container, epoch: epoch) : true
            let batch = plan.batchable ? await batchedDecoder(container, epoch: epoch) : nil
            let options = LaunchOptions(
                params: params, components: constrained, tools: forcedTool == nil ? toolSpecs : nil, keepsCache: namespace != nil,
                batch: batch, lookup: LookupOptions(
                    speculative: plan.speculativeLookup, recycling: speculativeRecycling, jumpForward: jumpForward, routesExperts: expertRouting),
                kvZones: plan.kvZones, kvHeadMask: plan.kvHeadMask, questTopK: plan.questTopK, kvPrecision: plan.kvPrecision,
                mtp: plan.drafts, mtpBlockSize: mtpBlockSize, mtpPrefix: mtpPrefix)
            let cuts = (fork.map { [$0] } ?? []) + anchors
            let launch: GenerationLaunch
            do {
                try faults.check(.launch)
                launch = try await container.performEvaluating(nonSendable: (lmInput, reuse)) { context, values in
                    let prompt = CachedPrompt(reuse: values.1, cuts: cuts, boundary: boundary, blend: blending)
                    return try Self.launch(input: values.0, prompt: prompt, options: options, context: context)
                }
            } catch {
                keep(tokens: promptTokens, namespace: namespace, releasing: leaseID, epoch: epoch)
                throw error
            }
            launches[leaseID] = launch
            let relay = GenerationRelay(GenerationRelay.Setup(
                request: request, promptTokens: promptTokens.count, reusedTokens: launch.reusedTokens.rawValue, alarm: alarm,
                controlTokens: controlTokens.tokens, startsThinking: promptEndsWithThinking, scansToolCallText: scansToolCallText,
                echoedPrefix: echoedPrefix, forcedTool: forcedTool, prefixAnswer: prefixAnswer,
                guardsRepetition: constraint == nil, draftSource: plan.draftSource))
            let inputs = RelayInputs(
                stream: launch.stream, task: launch.task, prefillSeconds: launch.prefillSeconds,
                promptTokens: promptTokens, namespace: namespace, leaseID: leaseID, epoch: epoch, constraint: constraint,
                container: container, faults: faults)

            return AsyncStream<StreamChunk> { continuation in
                let task = Task { @concurrent in
                    await self.relay(relay, inputs, into: continuation)
                }

                continuation.onTermination = { _ in
                    task.cancel()
                }
            }
        } catch let err as BasaltError {
            throw err
        } catch {
            throw error is CancellationError ? .cancelled : .inferenceHardwareFailure(reason: error.localizedDescription)
        }
    }

    static func drainCompletionInfo(_ generations: inout AsyncStream<Generation>.Iterator) async -> GenerateCompletionInfo? {
        var info: GenerateCompletionInfo?
        while let generation = await generations.next() {
            if case .info(let completion) = generation { info = completion }
        }
        return info
    }

    static func settle(_ task: Task<Void, Never>?, cancelled: Bool) async {
        if cancelled { task?.cancel() }
        await task?.value
    }
}

extension MLXInferenceEngine {
    func record(_ event: EngineEvent) { state = .reduce(state, event) }

    public var hybridAnchorUsage: (count: Int, bytes: Int64)? {
        guard modelIsHybrid == true else { return nil }
        let usage = promptCaches.anchorUsage
        return (usage.count, usage.bytes.rawValue)
    }

    public func apply(_ directive: MemoryDirective) async -> MemoryLimits? {
        if directive != appliedMemoryDirective {
            if let gpuCacheLimit { Memory.cacheLimit = Self.cacheLimit(gpuCacheLimit, under: directive) }
            if directive > appliedMemoryDirective, directive.clearsCache { Memory.clearCache() }
            let evictedBefore = promptCaches.evictions
            promptCaches.rebudget(ByteCount(Int64(Double(promptCacheFullBudget.rawValue) * directive.promptCacheShare)))
            logger.info("Memory directive \(appliedMemoryDirective) → \(directive): prompt cache \(promptCaches.byteBudget.rawValue >> 20) MB, \(promptCaches.evictions - evictedBefore) nodes evicted")
            appliedMemoryDirective = directive
        }
        return MemoryLimits(
            cacheLimitBytes: gpuCacheLimit.map { Self.cacheLimit($0, under: appliedMemoryDirective) },
            promptCacheBudgetBytes: Int(promptCaches.byteBudget.rawValue), promptCacheBytes: Int(promptCaches.byteCount.rawValue),
            promptCacheEvictions: promptCaches.evictions)
    }

    static func cacheLimit(_ relaxed: Int, under directive: MemoryDirective) -> Int {
        Int(Double(relaxed) * directive.cacheLimitShare)
    }

    @discardableResult
    func attach(_ container: ModelContainer, facts: ModelFacts, epoch: ModelEpoch) -> Bool {
        state = .reduce(state, .loadFinished(epoch, container, facts))
        guard state.loaded?.epoch == epoch else { return false }
        batchDecoder = nil
        grammarVocabulary = nil
        sequenceBreakers = nil
        grammarPool = GrammarPool()
        let workingSet = PromptCacheBudget.gpuWorkingSet()
        let weights = ByteCount(Memory.activeMemory)
        let budget = configuredPromptCacheBytes ?? PromptCacheBudget.automatic(workingSet: workingSet, weights: weights)
        kvMemoryBudget = max(0, workingSet - weights - budget).rawValue
        let blendBytes = cacheBlendRatio == nil ? 0 : budget / 2
        promptCaches = PromptCacheTree(byteBudget: budget - blendBytes, anchorLimit: max(1, hybridAnchors))
        mtpPrefixes.removeAll()
        promptCacheFullBudget = budget - blendBytes
        appliedMemoryDirective = .relaxed
        cacheBlend = cacheBlendRatio.map { CacheBlendStore(ratio: $0, byteBudget: Int(blendBytes.rawValue)) }
        templateSuffixes = [:]
        logger.info("Prompt cache budget: \(promptCaches.byteBudget.rawValue >> 20) MB")
        return true
    }

    func unloadCurrentModel() async {
        await persistPromptCache(within: Self.modelSwitchPersistDeadline)
        state = .reduce(state, .unloaded)
        await releaseWiredMemory()
        batchDecoder = nil
        dropPromptCaches()
        cacheBlend = nil
        Memory.clearCache()
        logger.info("Previous model unloaded, GPU cache cleared.")
    }

    func dropPromptCaches() { promptCaches.removeAll(); mtpPrefixes.removeAll() }

    fileprivate func hold(_ lease: consuming PromptCacheLease?) -> Int {
        leaseCount += 1
        leases[leaseCount] = PromptCacheLeaseSlot(lease)
        return leaseCount
    }

    func keep(tokens prompt: [Int], namespace: String?, releasing lease: Int, epoch: ModelEpoch) {
        leases[lease] = nil
        let launch = launches.removeValue(forKey: lease)
        guard epoch == modelEpoch, lease > modelFacts.cacheFence, faults.point != .keep, let namespace, let launch else { return }
        let (tokens, anchors) = (Array(prompt.prefix(launch.exactTokens.rawValue)), launch.anchors)
        for anchor in anchors {
            promptCaches.store(namespace, tokens: Array(tokens.prefix(anchor.tokenCount)), cache: anchor)
        }
        if let bytes = anchors.first.flatMap(RecurrentSnapshot.init(of:))?.bytes {
            logger.info("Recurrent snapshots before the boundary at prompt tokens \(anchors.map(\.tokenCount)): \(bytes.rawValue >> 10) KB each")
        }
        if let cache = launch.cache {
            promptCaches.store(namespace, tokens: Array(tokens.prefix(cache.tokenCount)), cache: cache)
            if let mtpPrefix = launch.mtpPrefix {
                mtpPrefixes.removeAll { $0.namespace == namespace }
                mtpPrefixes.append((namespace, Array(tokens.prefix(cache.tokenCount)), mtpPrefix))
                mtpPrefixes = Array(mtpPrefixes.suffix(Self.mtpPrefixLimit))
            }
        }
    }
}
