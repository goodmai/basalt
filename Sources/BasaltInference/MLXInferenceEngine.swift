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
    let reasoningEffort: String?

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
    let speculativeRecycling: Bool
    let mtp: Bool
    let mtpBlockSize: Int
    let experimentalSamplers: Bool
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
        let (params, unbudgeted, shaping, exclusion) = try await checkedSampling(request)
        let settings = planSettings
        let toolSpecs = Self.toolSpecs(request.tools)
        do {
            let rendered = try await render(request, tools: toolSpecs, container: container)
            let components = try await budgeted(unbudgeted, request, rendered.policy, params: params, container: container)
            let prompt = try await promptInput(rendered, request: request, container: container)
            let namespace = GenerationPlan.cacheNamespace(
                request, settings, cacheEnabled: promptCaches.byteBudget > 0 && !modelModified,
                carriesMoreThanTokens: prompt.carriesMoreThanTokens, sameEpoch: epoch == modelEpoch)
            let constraint = try await checkedConstraint(request, settings, prompt, container: container)
            let boundary = try await boundary(namespace, prompt, rendered, container: container, epoch: epoch)
            let plan = GenerationPlan.make(request, settings, GenerationPlan.Facts(
                namespace: namespace, promptTokens: prompt.tokens.count, carriesMoreThanTokens: prompt.carriesMoreThanTokens,
                constrained: constraint != nil, hasMTPDrafter: hasMTPDrafter, blendAvailable: cacheBlend != nil,
                modelIsHybrid: modelIsHybrid, modelHasRecurrentState: modelHasRecurrentState))
            try await restoreIfNeeded(namespace, drafts: plan.drafts, tokens: prompt.tokens, container: container, epoch: epoch)
            let claim = try claimCache(namespace, drafts: plan.drafts, tokens: prompt.tokens)
            let search = AnchorSearch(prompt: prompt.tokens, reused: claim.reuse?.reusedTokens ?? 0, boundary: boundary, limit: hybridAnchors)
            let anchors = plan.findsAnchors
                ? await anchorOffsets(rendered.anchorTurns, search, container: container, tools: toolSpecs, context: rendered.context) : []
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
            let route = RelayRoute(namespace: namespace, leaseID: claim.leaseID, epoch: epoch, container: container, alarm: alarm)
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
            promptCaches.store(namespace, tokens: Array(tokens.prefix(cache.tokenCount)), cache: cache)
            if let mtpPrefix = launch.mtpPrefix {
                mtpPrefixes.removeAll { $0.namespace == namespace }
                mtpPrefixes.append((namespace, Array(tokens.prefix(cache.tokenCount)), mtpPrefix))
                mtpPrefixes = Array(mtpPrefixes.suffix(Self.mtpPrefixLimit))
            }
        }
    }
}
