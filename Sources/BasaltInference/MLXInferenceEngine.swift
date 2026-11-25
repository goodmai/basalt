import Foundation
import MLX
import MLXLLM
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

public actor MLXInferenceEngine: InferenceEngine {

    private(set) var state = EngineState.unloaded(0)
    let logger = BasaltLogger(module: "MLXInferenceEngine")

    let configuration: EngineConfiguration
    var wiredTicket: WiredMemoryTicket?

    var promptCaches: PromptCacheTree
    private(set) var leases: [Int: PromptCacheLeaseSlot] = [:]
    private(set) var launches: [Int: GenerationLaunch] = [:]
    var faults = FaultPlan()
    var experiments = (epoch: ModelEpoch(0), registry: OverlayRegistry())
    private var leaseCount = 0
    private(set) var mtpPrefixes: [(namespace: String, tokens: [Int], prefix: MTPPrefix)] = []
    private var promptCacheFullBudget: ByteCount
    private(set) var appliedMemoryDirective = MemoryDirective.relaxed
    private(set) var prefillPacer: PrefillPacer?
    private var kvBudget: ByteCount?
    var templateSuffixes: [String: Int] = [:]
    var grammarVocabulary: Result<GrammarVocabulary, ConstraintUnavailable>?
    var sequenceBreakers: (strings: [String], ids: Set<Int>)?
    var grammarPool = GrammarPool()
    private(set) var cacheBlend: CacheBlendStore?
    public var kvMemoryBudget: ByteCount? { kvBudget }
    var promptCacheDisk: PromptCacheDiskStore?
    let telemetry: TelemetryTracker?
    var batchDecoder: BatchedDecoder?

    public init(
        _ configuration: EngineConfiguration = .standard,
        promptCacheDisk: PromptCacheDiskStore? = nil,
        telemetry: TelemetryTracker? = nil
    ) {
        self.configuration = configuration.normalized
        self.promptCacheDisk = promptCacheDisk
        self.telemetry = telemetry
        promptCaches = PromptCacheTree(byteBudget: ByteCount(configuration.promptCacheBytes ?? 0), anchorLimit: max(1, configuration.hybridAnchors))
        promptCacheFullBudget = ByteCount(configuration.promptCacheBytes ?? 0)
    }

    func checkAdapter(_ requested: String?) throws(BasaltError) {
        guard !modelFacts.switchingAdapter else { throw .inferenceError("an adapter switch is in progress; try again when it finishes") }
        guard requested == activeAdapter else { throw .inferenceError("adapter \(requested ?? "none") was asked for while \(activeAdapter ?? "the base model") is active") }
    }

    public func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk> {
        guard let container else {
            throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first")
        }
        var request = request
        if request.logprobs == true, request.logitProbe == nil {
            request.logitProbe = LogitProbe(topK: max(64, request.topLogprobs ?? 5))
        }
        try checkAdapter(request.adapter)
        let epoch = modelEpoch
        let (params, unbudgeted, shaping, exclusion, confidenceProbe) = try await checkedSampling(request)
        let settings = planSettings
        let toolSpecs = Self.toolSpecs(request.tools)
        let images = try await checkImages(request.messages, container: container)
        do {
            let rendered = try await render(request, tools: toolSpecs, images: images, container: container)
            let components = try await budgeted(unbudgeted, request, rendered.policy, params: params, container: container)
            let prompt = try await promptInput(rendered, request: request, container: container, epoch: epoch)
            let namespace = GenerationPlan.cacheNamespace(
                request, settings, cacheEnabled: promptCaches.byteBudget > 0 && !modelModified,
                media: prompt.media, sameEpoch: epoch == modelEpoch)
            let constraint = try await checkedConstraint(request, settings, prompt, container: container)
            let boundary = max(prompt.media.end, try await boundary(namespace, prompt, rendered, container: container, epoch: epoch))
            let plan = GenerationPlan.make(request, settings, GenerationPlan.Facts(
                namespace: namespace, promptTokens: prompt.tokens.count, carriesMoreThanTokens: prompt.carriesMoreThanTokens,
                constrained: constraint != nil, jumpsForward: constraint?.fastForward == true, hasMTPDrafter: hasMTPDrafter,
                blendAvailable: cacheBlend != nil,
                modelIsHybrid: modelIsHybrid, modelHasRecurrentState: modelHasRecurrentState))
            try await restoreIfNeeded(namespace, drafts: plan.drafts, tokens: prompt.tokens, container: container, epoch: epoch)
            let claim = try claimCache(namespace, drafts: plan.drafts, tokens: prompt.tokens, mediaEnd: prompt.media.end)
            let search = AnchorSearch(prompt: prompt.tokens, reused: claim.reuse?.reusedTokens ?? 0, boundary: boundary, limit: configuration.hybridAnchors)
            let anchors = plan.findsAnchors
                ? await anchorOffsets(rendered, search, container: container, tools: toolSpecs) : []
            let (constrained, alarm) = NonFiniteLogitGuard.guarding(
                Self.shaped(components, constraint, rendered.policy, prompt, shaping: shaping, exclusion: exclusion),
                temperature: params.temperature)
            var options = LaunchOptions(
                params: params, components: constrained, tools: constraint?.toolName == nil ? toolSpecs : nil, keepsCache: namespace != nil)
            await route(&options, plan, jumpForward: constraint?.fastForward == true, mtpPrefix: claim.mtpPrefix, container: container, epoch: epoch)
            let blend = plan.blends && epoch == modelEpoch ? namespace.flatMap { space in cacheBlend.map { ($0, space) } } : nil
            let cached = CachedPrompt(reuse: claim.reuse, cuts: (claim.fork.map { [$0] } ?? []) + anchors, boundary: boundary, blend: blend)
            let launch: GenerationLaunch
            do {
                launch = try await launching(prompt.input, cached, options: options, container: container)
            } catch {
                keep(tokens: prompt.tokens, namespace: namespace, releasing: claim.leaseID, epoch: epoch)
                throw error
            }
            launches[claim.leaseID] = launch
            let route = RelayRoute(
                namespace: namespace, leaseID: claim.leaseID, epoch: epoch, container: container, alarm: alarm,
                startsThinking: Self.relayStartsThinking(prompt, constrained: constraint != nil, policy: rendered.policy),
                confidenceProbe: confidenceProbe)
            return relayStream(launch, request, prompt, constraint, plan, route: route)
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
            if let gpuCacheLimit = configuration.gpuCacheLimit { Memory.cacheLimit = Self.cacheLimit(gpuCacheLimit, under: directive) }
            if directive > appliedMemoryDirective, directive.clearsCache { Memory.clearCache() }
            let evictedBefore = promptCaches.evictions
            promptCaches.rebudget(ByteCount(Int64(Double(promptCacheFullBudget.rawValue) * directive.promptCacheShare)))
            logger.info("Memory directive \(appliedMemoryDirective) → \(directive): prompt cache \(promptCaches.byteBudget.rawValue >> 20) MB, \(promptCaches.evictions - evictedBefore) nodes evicted")
            appliedMemoryDirective = directive
        }
        let limits = MemoryLimits(
            cacheLimitBytes: configuration.gpuCacheLimit.map { Self.cacheLimit($0, under: appliedMemoryDirective) },
            promptCacheBudgetBytes: Int(promptCaches.byteBudget.rawValue), promptCacheBytes: Int(promptCaches.byteCount.rawValue),
            promptCacheEvictions: promptCaches.evictions)
        if let telemetry, let delta = promptCaches.takeTelemetryDelta() { await telemetry.recordCache(delta) }
        return limits
    }

    public func pace(with pacer: PrefillPacer) async {
        prefillPacer = pacer
    }

    static func cacheLimit(_ relaxed: Int, under directive: MemoryDirective) -> Int {
        Int(Double(relaxed) * directive.cacheLimitShare)
    }

    @discardableResult
    func attach(_ container: ModelContainer, facts: ModelFacts, epoch: ModelEpoch, warming: Bool = false) -> Bool {
        state = .reduce(state, .loadFinished(epoch, container, facts, warming: warming))
        guard state.attached?.epoch == epoch else { return false }
        batchDecoder = nil
        grammarVocabulary = nil
        sequenceBreakers = nil
        grammarPool = GrammarPool()
        let workingSet = PromptCacheBudget.gpuWorkingSet()
        let weights = ByteCount(Memory.activeMemory) + facts.expertReserve
        let budget = configuration.promptCacheBytes.map { ByteCount($0) } ?? PromptCacheBudget.automatic(workingSet: workingSet, weights: weights)
        kvBudget = max(0, workingSet - weights - budget)
        let blendBytes = configuration.cacheBlendRatio == nil ? 0 : budget / 2
        promptCaches = PromptCacheTree(byteBudget: budget - blendBytes, anchorLimit: max(1, configuration.hybridAnchors))
        mtpPrefixes.removeAll()
        promptCacheFullBudget = budget - blendBytes
        appliedMemoryDirective = .relaxed
        cacheBlend = configuration.cacheBlendRatio.map { CacheBlendStore(ratio: $0, byteBudget: Int(blendBytes.rawValue)) }
        templateSuffixes = [:]
        logger.info("Prompt cache budget: \(promptCaches.byteBudget.rawValue >> 20) MB")
        return true
    }

    public func unload() async { await unloadCurrentModel() }

    public func releaseCaches() async {
        dropPromptCaches()
        Memory.clearCache()
        if let telemetry, let delta = promptCaches.takeTelemetryDelta() { await telemetry.recordCache(delta) }
    }

    func drainPromptCacheTelemetry() async -> CacheTelemetry? {
        guard let delta = promptCaches.takeTelemetryDelta() else { return nil }
        guard let telemetry else { return delta }
        await telemetry.recordCache(delta)
        return nil
    }

    func unloadCurrentModel() async {
        await persistPromptCache(within: Self.modelSwitchPersistDeadline)
        state = .reduce(state, .unloaded)
        batchDecoder = nil
        dropPromptCaches()
        cacheBlend = nil
        await releaseWiredMemory()
        Memory.clearCache()
        if let telemetry, let delta = promptCaches.takeTelemetryDelta() { await telemetry.recordCache(delta) }
        logger.info("Previous model unloaded, GPU cache cleared.")
    }

    func dropPromptCaches() { promptCaches.removeAll(); mtpPrefixes.removeAll() }

    func hold(_ lease: consuming PromptCacheLease?) -> Int {
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
            let keptTokens = Array(tokens.prefix(cache.tokenCount))
            promptCaches.store(namespace, tokens: keptTokens, cache: cache, drainSource: true)
            if let mtpPrefix = launch.mtpPrefix {
                mtpPrefixes.removeAll { $0.namespace == namespace }
                mtpPrefixes.append((namespace, keptTokens, mtpPrefix))
                mtpPrefixes = Array(mtpPrefixes.suffix(Self.mtpPrefixLimit))
            }
        }
    }
}
