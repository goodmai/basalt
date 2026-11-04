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

private extension Duration {
    var inSeconds: Double {
        Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}

struct TokenizerBridge: MLXLMCommon.Tokenizer {
    private let upstream: any Tokenizers.Tokenizer
    init(_ upstream: any Tokenizers.Tokenizer) { self.upstream = upstream }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }
    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }
    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        } catch {
            throw error
        }
    }
}

struct DefaultTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        TokenizerBridge(try await MLXInferenceEngine.upstreamTokenizer(at: directory))
    }
}

// MARK: — Real implementation

public actor MLXInferenceEngine: InferenceEngine {

    private(set) var container: ModelContainer?

    /// Resolved from the loaded model's config.json. Supplies per-family sampling
    /// defaults that the caller did not override.
    private var profile: any ModelProfile = GenericProfile()
    private var samplingDefaults = SamplingDefaults()
    private let logger = BasaltLogger(module: "MLXInferenceEngine")

    /// Chat-template knob for models with a reasoning budget (Qwen3.8: xhigh | medium | low).
    /// Left nil the template picks its own default — for Qwen3.8 that is `xhigh`, which
    /// can spend the whole token budget thinking and never emit an answer.
    private let reasoningEffort: String?

    /// Bits for KV-cache quantization (4 or 8). nil keeps the cache in full
    /// precision. The cache grows linearly with context, so on long prompts it
    /// competes with the weights for unified memory.
    private let kvPrecision: KVPrecision?

    /// Token index at which the quantized cache kicks in. Quantizing from token 0
    /// costs accuracy on short prompts for no memory benefit.
    private let quantizedKVStart: Int
    private let kvZones: KVZones?
    private let wiresWorkingSet: Bool
    private var wiredTicket: WiredMemoryTicket?
    private let kvHeadSplit: Double?
    private(set) var kvHeadMask: KVHeadMask?
    private let questTopK: Int?
    private(set) var questApplies = false

    /// Multiplicative penalty on recently emitted tokens. Left nil, nothing damps
    /// a degenerate loop: a 2-bit model answered "127 + 458" by repeating the
    /// question until it hit the ceiling.
    private let repetitionPenalty: Float?

    /// Truncates the sampling tail. `topK` keeps the k most likely tokens; `minP`
    /// drops anything below a fraction of the top token's probability. Both are
    /// stricter than top-p on a peaked distribution, which is what a recalled
    /// constant looks like: at temperature 0.7 a model wrote 2.5 where the
    /// Colebrook relation needs 2.51 — a value you cannot approximate.
    private let topK: Int?
    private let minP: Float?

    /// Seeds the sampler so a non-greedy run can be repeated. Without it, two
    /// identical requests give different answers and a single run is not evidence.
    private let seed: UInt64?

    /// MLX buffer-cache ceiling in bytes. nil leaves MLX's own default.
    private let gpuCacheLimit: Int?

    private var promptCaches: PromptCacheTree
    private var leases: [Int: PromptCacheLeaseSlot] = [:]
    private var launches: [Int: GenerationLaunch] = [:]
    private var leaseCount = 0
    private(set) var mtpPrefixes: [(namespace: String, tokens: [Int], prefix: MTPPrefix)] = []
    private var promptCacheFullBudget: ByteCount
    private(set) var appliedMemoryDirective = MemoryDirective.relaxed
    private var kvMemoryBudget: Int64?
    private var controlTokens = ControlTokenGuard.none

    public var kvMemoryBudgetBytes: Int64? { kvMemoryBudget }
    public var hybridAnchorUsage: (count: Int, bytes: Int64)? {
        guard modelIsHybrid == true else { return nil }
        let usage = promptCaches.anchorUsage
        return (usage.count, usage.bytes.rawValue)
    }
    public var supportsKVZones: Bool { modelAttendsThroughZones }
    public var supportsKVEviction: Bool { modelPrefillsInChunks }
    private var modelAttendsThroughZones = false
    private var modelPrefillsInChunks = false
    private(set) var valueRotationUnavailable = false

    public func promptTokenCount(for request: GenerationRequest) async -> Int? {
        guard let container else { return nil }
        let text = request.messages.map { ChatMessageInput.plainText($0) } ?? request.prompt
        return await container.perform { $0.tokenizer.encode(text: text, addSpecialTokens: false).count }
    }
    private let configuredPromptCacheBytes: ByteCount?
    private let hybridAnchors: Int
    private var modelEpoch = 0
    private var modelIsHybrid: Bool?
    private var modelHasRecurrentState: Bool?
    private var modelRoutesExperts: Bool?
    private var templateSuffixes: [String: Int] = [:]
    private var grammarVocabulary: Result<GrammarVocabulary, ConstraintUnavailable>?
    private var sequenceBreakers: (strings: [String], ids: Set<Int>)?
    private(set) var idleGrammars: [String: [GrammarConstraint]] = [:]
    private var idleGrammarRecency: [String] = []
    private let cacheBlendRatio: Float?
    private(set) var cacheBlend: CacheBlendStore?
    private let speculativeLookup: Bool
    private let speculativeRecycling: Bool
    private let mtp: Bool
    private let mtpBlockSize: Int
    private let experimentalSamplers: Bool
    private var hasMTPDrafter = false
    private let grammarJumpForward: Bool
    private(set) var promptCacheDisk: PromptCacheDiskStore?
    private var modelFingerprint: String?
    private var modelDescription = ""
    private var modelCacheLayout: [String]?
    private var modelSupportsMultiTokenForward: Bool?
    private let batchSlots: Int
    private var batchDecoder: BatchedDecoder?

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

    func probeKVHeads(directory: URL) async {
        kvHeadMask = nil
        guard let ratio = kvHeadSplit, let container else { return }
        guard (try? await container.perform({ try ZonedKVCache.modelAttendsThroughCache($0.model) })) == true else {
            logger.warn("--kv-head-split does not apply to this model: its attention does not go through the cache")
            return
        }
        do {
            let scores = try await KVHeadClassifier.loadOrScore(directory: directory, container: container)
            guard KVHeadMask.isConcentrated(scores) else {
                logger.warn("--kv-head-split does not apply to this model: no small set of heads carries its long-range recall (the median head's ablation score is above \(KVHeadMask.concentration) of the top one), so dropping any of them loses facts; the KV cache stays dense")
                return
            }
            kvHeadMask = KVHeadMask.classify(scores: scores, ratio: ratio)
            let kept = kvHeadMask?.retrieval.map(\.count).reduce(0, +) ?? 0
            logger.info("KV head split: \(kept) of \(scores.map(\.count).reduce(0, +)) KV heads keep the whole context; the rest keep \(HeadSplitKVCache.sinkTokens) sink and \(HeadSplitKVCache.windowTokens) recent tokens")
        } catch {
            logger.warn("--kv-head-split: scoring the heads failed (\(error.localizedDescription)); the KV cache stays dense")
        }
    }

    func probeQuest() async {
        questApplies = false
        guard let questTopK, let container else { return }
        do {
            guard try await container.perform({ try QuestKVCache.modelAttendsThroughCache($0.model) }) else {
                logger.warn("--quest-top-k does not apply to this model: its full-attention layers read the whole cache through cache.update (Gemma 4 does), so they cannot attend to a selection of blocks; attention stays dense")
                return
            }
        } catch {
            logger.warn("--quest-top-k: probing the model failed (\(error.localizedDescription)); attention stays dense")
            return
        }
        questApplies = true
        logger.info("Quest: a decode step attends to \(QuestKVCache.sinkTokens) sink tokens, the last \(QuestKVCache.windowTokens) or more, and the \(questTopK) best \(QuestKVCache.blockTokens)-token blocks between them")
    }

    public static func renewsResidency() {
        setenv("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS", "1000", 0)
    }

    func wireWorkingSet() async {
        await releaseWiredMemory()
        guard wiresWorkingSet else { return }
        let ticket = WiredMemoryTicket(size: Int(PromptCacheBudget.gpuWorkingSet().rawValue), policy: WiredMaxPolicy(), kind: .active)
        let applied = await ticket.start()
        wiredTicket = ticket
        logger.info("Wired memory limit: \(applied >> 20) MB, Metal's recommended working set, while the model is loaded (--no-wired-limit leaves the system default)")
    }

    func releaseWiredMemory() async {
        guard let ticket = wiredTicket else { return }
        wiredTicket = nil
        _ = await ticket.end()
    }

    var wiredLimitBytes: Int? { wiredTicket?.size }

    func probeKVZones() async {
        guard kvZones != nil, let container else { return }
        let precision = kvPrecision ?? .q8
        let probed = (try? await container.perform { try ZonedKVCache.probe($0.model, precision: precision) }) ?? (attendsThroughCache: false, rotatesValues: false)
        modelAttendsThroughZones = probed.attendsThroughCache
        if let zones = kvZones, zones.evictedMiddleFraction > 0, modelAttendsThroughZones {
            do {
                modelPrefillsInChunks = try await container.perform {
                    try ZonedKVCache.prefillsWithinWindow($0.model, zones: zones, precision: precision, prefill: GenerateParameters().prefill)
                }
                if !modelPrefillsInChunks {
                    logger.warn("--kv-evict does not apply to this model: it prefills a prompt longer than the recent window in one pass, so its zoned cache keeps every token and is sized that way.")
                }
            } catch {
                logger.warn("--kv-evict is off for this model: its prefill probe failed (\(error)), so its zoned cache keeps every token and is sized that way.")
            }
        }
        valueRotationUnavailable = modelAttendsThroughZones && precision.rotatesValues && !probed.rotatesValues
        if !modelAttendsThroughZones {
            logger.warn("--kv-zones does not apply to this model: its attention reads the whole cache through update(), or its head dimension has no quantization group. Its quantized requests use uniform --kv-bits, k8v4 as 8-bit\((kvZones?.evictedMiddleFraction ?? 0) > 0 ? ", and --kv-evict keeps every token" : "").")
        } else if valueRotationUnavailable {
            logger.warn("--kv-bits \(precision) cannot rotate this model's value heads: the Hadamard transform is its own inverse only for head dimensions 2^k and 12·2^k. It runs as k8v4.")
        }
    }

    func attach(
        _ container: ModelContainer, profile: any ModelProfile = GenericProfile(),
        samplingDefaults: SamplingDefaults = SamplingDefaults(), controlTokens: ControlTokenGuard = .none,
        fingerprint: String? = nil, modelDescription: String = ""
    ) {
        self.container = container
        modelFingerprint = fingerprint
        self.modelDescription = modelDescription
        modelCacheLayout = nil
        modelSupportsMultiTokenForward = nil
        batchDecoder = nil
        self.controlTokens = controlTokens
        grammarVocabulary = nil
        sequenceBreakers = nil
        idleGrammars = [:]
        idleGrammarRecency = []
        self.profile = profile
        self.samplingDefaults = samplingDefaults
        let workingSet = PromptCacheBudget.gpuWorkingSet()
        let weights = ByteCount(Memory.activeMemory)
        let budget = configuredPromptCacheBytes ?? PromptCacheBudget.automatic(workingSet: workingSet, weights: weights)
        kvMemoryBudget = max(0, workingSet - weights - budget).rawValue
        modelAttendsThroughZones = false
        modelPrefillsInChunks = false
        let blendBytes = cacheBlendRatio == nil ? 0 : budget / 2
        promptCaches = PromptCacheTree(byteBudget: budget - blendBytes, anchorLimit: max(1, hybridAnchors))
        mtpPrefixes.removeAll()
        promptCacheFullBudget = budget - blendBytes
        appliedMemoryDirective = .relaxed
        cacheBlend = cacheBlendRatio.map { CacheBlendStore(ratio: $0, byteBudget: Int(blendBytes.rawValue)) }
        modelEpoch += 1
        modelIsHybrid = nil
        modelHasRecurrentState = nil
        modelRoutesExperts = nil
        templateSuffixes = [:]
        logger.info("Prompt cache budget: \(promptCaches.byteBudget.rawValue >> 20) MB")
    }

    private static let aliasRegistration = Task {
        let registry = LLMTypeRegistry.shared
        for type in ["qwen3_8", "qwen3.8", "qwen3_6", "qwen3.6"] {
            await registry.registerModelType(type) { data in
                MLXLLM.Qwen35Model(try JSONDecoder.json5().decode(MLXLLM.Qwen35Configuration.self, from: data))
            }
        }
        for type in ["qwen3_8_moe", "qwen3.8_moe", "qwen3_6_moe"] {
            await registry.registerModelType(type) { data in
                MLXLLM.Qwen35MoEModel(try JSONDecoder.json5().decode(MLXLLM.Qwen35Configuration.self, from: data))
            }
        }
        for type in ["minimax_m2", "minimax-m2"] {
            await registry.registerModelType(type) { data in
                MiniMaxModel(try JSONDecoder.json5().decode(MiniMaxConfiguration.self, from: data))
            }
        }
        await registry.registerModelType(PrismHadamard.modelType) { data in
            try PrismHadamard.model(configuration: data)
        }
    }

    public static func registerAliases() async {
        await aliasRegistration.value
    }

    // MARK: — Load

    public func load(modelPath: String) async throws(BasaltError) {
        logger.trace("Loading model from path: \(modelPath)")
        await Self.registerAliases()
        guard modelPath.hasPrefix("/") || modelPath.hasPrefix(".") else {
            throw .modelNotFound(
                identifier: "'\(modelPath)' not found in local cache. Run: basalt models download \(modelPath)"
            )
        }

        let directory = URL(fileURLWithPath: modelPath)
        guard ModelCache.hasUsableConfig(at: directory) else {
            logger.error("Refusing to load \(modelPath): no readable config.json; the current model stays loaded")
            throw .modelNotFound(identifier: modelPath)
        }

        do {
            if container != nil { await unloadCurrentModel() }
            if let gpuCacheLimit { Memory.cacheLimit = gpuCacheLimit }
            let loaded = try await loadModelContainer(from: directory, using: DefaultTokenizerLoader())
            try await loaded.perform { try PrismHadamard.prepareLoaded($0.model) }
            if mtp { await loadMTPDrafter(for: loaded, directory: directory, modelPath: modelPath) }
            attach(loaded,
                   profile: ModelProfileRegistry.profile(forModelAt: modelPath),
                   samplingDefaults: SamplingDefaults(generationConfig: ModelCache.generationConfig(at: directory)),
                   controlTokens: ControlTokenGuard(modelDirectory: directory),
                   fingerprint: promptCacheDisk == nil ? nil : PromptCacheDiskStore.fingerprint(modelDirectory: directory),
                   modelDescription: modelPath)
            await probeKVZones()
            await wireWorkingSet()
            await probeKVHeads(directory: directory)
            await probeQuest()
            logger.info("Sampling defaults from generation_config.json: \(samplingDefaults)")
            for line in ModelCache.config(at: directory).map(architectureLines(config:)) ?? [] { logger.info(line) }
            logger.info("Model loaded and initialized successfully.")
        } catch {
            throw await Self.loadFailure(error, modelPath: modelPath)
        }
    }

    func unloadCurrentModel() async {
        await persistPromptCache(within: Self.modelSwitchPersistDeadline)
        container = nil
        await releaseWiredMemory()
        batchDecoder = nil
        hasMTPDrafter = false
        promptCaches.removeAll()
        mtpPrefixes.removeAll()
        cacheBlend = nil
        modelEpoch += 1
        Memory.clearCache()
        logger.info("Previous model unloaded, GPU cache cleared.")
    }


    static func loadFailure(_ error: any Error, modelPath: String) async -> BasaltError {
        switch error as? ModelFactoryError {
        case .configurationDecodingError?:
            return .modelConfigurationInvalid(path: modelPath, reason: error.localizedDescription)
        case .unsupportedModelType(let modelType)? where await !VLMTypeRegistry.shared.contains(modelType):
            return .modelConfigurationInvalid(path: modelPath, reason: error.localizedDescription)
        default:
            return .weightsCorrupted(path: modelPath, reason: error.localizedDescription)
        }
    }

    static func upstreamTokenizer(at directory: URL) async throws -> any Tokenizers.Tokenizer {
        let configuration = LanguageModelConfigurationFromHub(modelFolder: directory)
        guard let tokenizerConfig = try await configuration.tokenizerConfig else {
            throw Tokenizers.TokenizerError.missingConfig
        }
        return try Tokenizers.PreTrainedTokenizer(
            tokenizerConfig: compatibleChatTemplate(in: tokenizerConfig),
            tokenizerData: try await configuration.tokenizerData)
    }

    static func compatibleChatTemplate(in tokenizerConfig: Config) -> Config {
        guard let template = tokenizerConfig.chatTemplate.string(), var entries = tokenizerConfig.dictionary() else {
            return tokenizerConfig
        }
        let compatible = ChatTemplateCompatibility.compatible(template)
        guard compatible != template else { return tokenizerConfig }
        entries["chat_template"] = Config(compatible)
        return Config(entries)
    }

    // MARK: — Generate

    public func generate(request: GenerationRequest) async throws(BasaltError) -> GenerationResponse {
        let stream = try await generateStream(request: request)
        var textChunks: [String] = []
        var reasoningChunks: [String] = []
        var toolCalls: [ToolCallOutput] = []
        var finalResponse: GenerationResponse?

        for await chunk in stream {
            switch chunk {
            case .text(let t):
                textChunks.append(t)
            case .reasoning(let r):
                reasoningChunks.append(r)
            case .metadata(let m):
                finalResponse = m
            case .toolCall(let call):
                toolCalls.append(call)
            }
        }

        guard let finalResponse else {
            throw .inferenceHardwareFailure(reason: "Stream finished without metadata")
        }

        let fullReasoning = reasoningChunks.joined()
        return GenerationResponse(
            generatedText: textChunks.joined(),
            reasoningText: fullReasoning.isEmpty ? nil : fullReasoning,
            promptTokens: finalResponse.promptTokens,
            cachedPromptTokens: finalResponse.cachedPromptTokens,
            completionTokens: finalResponse.completionTokens,
            tokensPerSecond: finalResponse.tokensPerSecond,
            generationTime: finalResponse.generationTime,
            timeToFirstToken: finalResponse.timeToFirstToken,
            memory: finalResponse.memory,
            finishReason: finalResponse.finishReason,
            toolCalls: toolCalls.isEmpty ? nil : toolCalls,
            reasoningTokens: finalResponse.reasoningTokens,
            draftTokens: finalResponse.draftTokens,
            acceptedDraftTokens: finalResponse.acceptedDraftTokens,
            forcedTokens: finalResponse.forcedTokens,
            modelCalls: finalResponse.modelCalls,
            toolCallRepairs: finalResponse.toolCallRepairs,
            toolCallRejections: finalResponse.toolCallRejections
        )
    }

    static let presenceContextSize = 1024

    struct OutputConstraint: Sendable {
        let session: GrammarSession
        let vocabulary: GrammarVocabulary
        let toolName: String?
        let fastForward: Bool
    }

    static let grammarSchemaLimit = 16
    static let idleGrammarsPerSchema = 4

    func outputConstraint(for request: GenerationRequest, allowsFastForward: Bool = true) async throws(BasaltError) -> OutputConstraint? {
        let forcedTool = request.toolChoice?.forcedTool.flatMap { name in request.tools?.first { $0.name == name } }
        let format = forcedTool.map { $0.parametersJSON.map { .jsonSchema($0) } ?? .jsonObject } ?? request.responseFormat
        guard let format else { return nil }
        func unconstrained(_ reason: String) throws(BasaltError) -> OutputConstraint? {
            guard let forcedTool else { throw .invalidRequestStructure(details: "response_format: \(reason)") }
            logger.info("tool_choice \(forcedTool.name): arguments left unconstrained, \(reason)")
            return nil
        }
        let schema: String
        do { schema = try format.prepared().schema } catch { return try unconstrained(error.message) }
        let epoch = modelEpoch
        let vocabulary: GrammarVocabulary
        switch await loadedGrammarVocabulary() {
        case .success(let loaded): vocabulary = loaded
        case .failure(let error): return try unconstrained(error.reason)
        }
        guard !vocabulary.stopTokens.isEmpty else { return try unconstrained("the model declares no stop token to end the document") }
        let fastForwardRequested = allowsFastForward && (request.grammarJumpForward ?? grammarJumpForward)
        let fastForward: Bool
        if fastForwardRequested, let container {
            fastForward = await supportsMultiTokenForward(container, epoch: epoch)
        } else {
            fastForward = false
        }
        let poolKey = schema + (fastForward ? ":ff" : "")
        let grammar: GrammarConstraint
        if epoch == modelEpoch, let idle = idleGrammars[poolKey]?.popLast() {
            grammar = idle
        } else {
            do {
                grammar = try GrammarConstraint(
                    tokenizer: vocabulary.tokenizer, jsonSchema: schema,
                    fastForward: fastForward, hostTokenizer: fastForward ? vocabulary.hostTokenizer : nil)
            } catch GrammarError.invalidJSONSchema(let message) {
                return try unconstrained(message)
            } catch {
                return try unconstrained("\(error)")
            }
        }
        return OutputConstraint(
            session: GrammarSession(schema: schema, grammar: grammar, poolKey: poolKey), vocabulary: vocabulary,
            toolName: forcedTool?.name, fastForward: fastForward)
    }

    func recycle(_ session: GrammarSession, epoch: Int) {
        guard epoch == modelEpoch, let grammar = session.rewound() else { return }
        idleGrammarRecency.removeAll { $0 == session.poolKey }
        idleGrammarRecency.append(session.poolKey)
        if idleGrammarRecency.count > Self.grammarSchemaLimit { idleGrammars[idleGrammarRecency.removeFirst()] = nil }
        guard idleGrammars[session.poolKey, default: []].count < Self.idleGrammarsPerSchema else { return }
        idleGrammars[session.poolKey, default: []].append(grammar)
    }

    struct ConstraintUnavailable: Error {
        let reason: String
    }

    func loadedGrammarVocabulary() async -> Result<GrammarVocabulary, ConstraintUnavailable> {
        if let grammarVocabulary { return grammarVocabulary }
        guard let container else { return .failure(ConstraintUnavailable(reason: "no model is loaded")) }
        let epoch = modelEpoch
        let loaded: Result<GrammarVocabulary, ConstraintUnavailable> = await container.perform { context in
            do { return .success(try GrammarVocabulary(context: context)) } catch {
                return .failure(ConstraintUnavailable(reason: "constrained decoding is unavailable for this model: \(error)"))
            }
        }
        if epoch == modelEpoch { grammarVocabulary = loaded }
        return loaded
    }

    static func batchStopTokens(_ context: ModelContext) -> Set<Int> {
        stopTokenIds(context).union(context.tokenizer.unknownTokenId.map { [$0] } ?? [])
    }

    static func stopTokenIds(_ context: ModelContext) -> Set<Int> {
        var ids = context.configuration.eosTokenIds
        if let eos = context.tokenizer.eosTokenId { ids.insert(eos) }
        for token in context.configuration.extraEOSTokens {
            if let id = context.tokenizer.convertTokenToId(token) { ids.insert(id) }
        }
        return ids
    }

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
        params.kvBits = (request.kvPrecision ?? kvPrecision)?.uniform.keyBits
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

    public func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk> {
        guard let container else {
            throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first")
        }
        let epoch = modelEpoch

        let requestedMaxTokens = request.maxTokens ?? 1024
        if request.usesExperimentalSamplers, !experimentalSamplers {
            throw .invalidRequestStructure(
                details: "dry_* and xtc_* are experimental (DRY can turn a detectable loop into one with corrupted facts on low-bit models); start the server with --experimental-samplers to use them")
        }
        let (params, unbudgeted, shaping, exclusion) = sampling(for: request, breakers: await sequenceBreakerIds(for: request))
        let zones = modelAttendsThroughZones ? (modelPrefillsInChunks ? kvZones : kvZones?.evicting(0)) : nil
        let headMask = zones == nil && request.kvPrecision == nil && kvPrecision == nil ? kvHeadMask : nil
        let questBlocks = zones == nil && request.kvPrecision == nil && kvPrecision == nil && headMask == nil && questApplies ? questTopK : nil
        let zonedPrecision = zones == nil ? nil : request.kvPrecision ?? kvPrecision

        let toolSpecs = Self.toolSpecs(request.tools)
        let declaredTools = Set(request.tools?.map(\.name) ?? [])
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
            let namespace = carriesMoreThanTokens || epoch != modelEpoch ? nil : promptCacheNamespace(for: request)
            let allowsFastForward = params.kvBits == nil && zonedPrecision == nil && !carriesMoreThanTokens
            let constraint = try await outputConstraint(for: request, allowsFastForward: allowsFastForward)
            if let constraint, !prefixAnswer.isEmpty {
                let ids = await container.perform { $0.tokenizer.encode(text: prefixAnswer, addSpecialTokens: false) }
                guard constraint.session.accepts(ids) else {
                    throw BasaltError.invalidRequestStructure(
                        details: "the assistant prefill is not a valid start of the requested JSON output")
                }
            }
            let speculativeLookup = request.speculativeLookup ?? speculativeLookup
            let drafts = constraint == nil && !speculativeLookup && zones == nil && params.kvBits == nil
                && promptTokens.count <= Self.mtpPromptLimit && hasMTPDrafter
            if let namespace, !drafts {
                await restoreFromDisk(namespace, prompt: promptTokens, container: container, epoch: epoch)
            }
            var lease: PromptCacheLease? = if let namespace { promptCaches.lease(namespace, for: promptTokens) } else { nil }
            var storedPrefix: MTPPrefix?
            if drafts, let namespace {
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
            let leaseID = hold(lease)
            let boundary = namespace == nil
                ? promptTokens.count - 1
                : await snapshotBoundary(renderedCount, container: container, epoch: epoch, template: template, makeInput)
            let anchors = namespace != nil && !drafts && modelHasRecurrentState == true
                ? await anchorOffsets(
                    anchorTurns, prompt: promptTokens, reused: reuse?.reusedTokens ?? 0, boundary: boundary,
                    container: container, tools: toolSpecs, context: context)
                : []
            let blend = epoch == modelEpoch && modelIsHybrid == false ? cacheBlend : nil
            let masked = Self.constraining(
                components, with: constraint,
                phase: Self.constraintPhase(
                    thinkingDisabled: policy == .disabled, endsInsideThinking: promptEndsWithThinking,
                    prefillAnswered: !prefixAnswer.isEmpty))
            let excluded = exclusion.map { masked.appendingBasaltProcessor($0) } ?? masked
            let constrained = shaping.map { excluded.appendingBasaltProcessor($0) } ?? excluded
            let forcedTool = constraint?.toolName
            let jumpForward = constraint?.fastForward == true
            let expertRouting = (speculativeLookup || jumpForward) ? await routesExperts(container, epoch: epoch) : true
            let blending = namespace.flatMap { space in blend.map { ($0, space) } }
            let batchable = constraint == nil && request.toolChoice?.forcedTool == nil && !speculativeLookup && !drafts
                && blending == nil && params.kvBits == nil && zonedPrecision == nil && !carriesMoreThanTokens
            let batch = batchable ? await batchedDecoder(container, epoch: epoch) : nil
            let options = LaunchOptions(
                params: params, components: constrained, tools: forcedTool == nil ? toolSpecs : nil, keepsCache: namespace != nil,
                batch: batch, lookup: LookupOptions(
                    speculative: speculativeLookup, recycling: speculativeRecycling, jumpForward: jumpForward, routesExperts: expertRouting),
                kvZones: zones, kvHeadMask: headMask, questTopK: questBlocks, kvPrecision: zonedPrecision,
                mtp: drafts, mtpBlockSize: mtpBlockSize, mtpPrefix: mtpPrefix)
            let cuts = (fork.map { [$0] } ?? []) + anchors
            let launch: GenerationLaunch
            do {
                launch = try await container.performSending(nonSendable: (lmInput, reuse)) { context, values in
                    let prompt = CachedPrompt(reuse: values.1, cuts: cuts, boundary: boundary, blend: blending)
                    return try ModelEvaluationLock.lock(for: context.model).withLock {
                        try Self.launch(input: values.0, prompt: prompt, options: options, context: context)
                    }
                }
            } catch {
                keep(tokens: promptTokens, namespace: namespace, releasing: leaseID, epoch: epoch)
                throw error
            }
            let (mlxStream, mlxTask, reusedTokens, prefillSeconds) = (launch.stream, launch.task, launch.reusedTokens, launch.prefillSeconds)
            launches[leaseID] = launch

            return AsyncStream<StreamChunk> { continuation in
                let task = Task.detached {
                    let clock = ContinuousClock()
                    let startTime = clock.now
                    var lastInfo: GenerateCompletionInfo?
                    var tokenCount = 0
                    var firstTokenAt: ContinuousClock.Instant?
                    var splitter = ThinkingSplitter(startsThinking: promptEndsWithThinking)
                    var gate = ToolCallGate(request: request)
                    if let echoedPrefix { continuation.yield(.text(echoedPrefix)) }

                    var toolCallScanner = scansToolCallText ? NemotronToolCallScanner() : nil
                    var reasoningText = ""
                    var forcedToolArguments = prefixAnswer
                    func emitReasoning(_ text: String) {
                        reasoningText += text
                        continuation.yield(.reasoning(text))
                    }
                    func admit(_ chunks: [StreamChunk]) {
                        for chunk in gate.admit(chunks) { continuation.yield(chunk) }
                    }
                    func emitText(_ text: String) {
                        guard forcedTool == nil else { forcedToolArguments += text; return }
                        guard var scanner = toolCallScanner else { continuation.yield(.text(text)); return }
                        admit(scanner.feed(text))
                        toolCallScanner = scanner
                    }

                    func route(_ segments: [ThinkingSplitter.Segment]) {
                        for segment in segments {
                            switch segment {
                            case .reasoning(let text): emitReasoning(text)
                            case .answer(let text): emitText(text)
                            }
                        }
                    }

                    var stoppedAfterCall = false
                    var repetition = constraint == nil ? RepetitionGuard() : nil
                    var stoppedOnRepetition = false
                    var toolCallRepairs = 0
                    var toolCallRejections = 0
                    var generations = mlxStream.makeAsyncIterator()
                    func stopAfterCall() async {
                        if let task = mlxTask {
                            task.cancel()
                            lastInfo = await Self.drainCompletionInfo(&generations) ?? lastInfo
                        } else {
                            stoppedAfterCall = true
                        }
                    }
                    tokens: while let generation = await generations.next() {
                        if Task.isCancelled { 
                            break 
                        }
                        
                        tokenCount += 1
                        
                        switch generation {
                        case .chunk(let text):
                            if firstTokenAt == nil { firstTokenAt = clock.now }
                            route(splitter.feed(text))
                            if gate.isSaturated { await stopAfterCall(); break tokens }
                            if repetition?.shouldStop(after: text) == true {
                                stoppedOnRepetition = true
                                await stopAfterCall()
                                break tokens
                            }
                        case .info(let info):
                            lastInfo = info
                        case .toolCall(let call):
                            route(splitter.feed("", final: true))
                            if let chunk = gate.admit(Self.toolCallOutput(call)) { continuation.yield(chunk) }
                            if gate.isSaturated { await stopAfterCall(); break tokens }
                        case .rejectedToolCall(let rejection):
                            route(splitter.feed("", final: true))
                            let chunk = Self.resolveRejectedToolCall(
                                reason: rejection.reason.rawValue, toolName: rejection.toolName,
                                callID: rejection.callID, rawText: rejection.rawTextPreview,
                                truncated: rejection.isPreviewTruncated, declaredTools: declaredTools)
                            if case .toolCall = chunk {
                                toolCallRepairs += 1
                            } else {
                                toolCallRejections += 1
                                BasaltLogger(module: "MLXInferenceEngine").warn("Tool call rejected: \(rejection.reason.rawValue)")
                            }
                            if let chunk = gate.admit(chunk) { continuation.yield(chunk) }
                            if gate.isSaturated { await stopAfterCall(); break tokens }
                        @unknown default:
                            break
                        }
                    }
                    
                    route(splitter.feed("", final: true))
                    admit(toolCallScanner?.finish() ?? [])
                    let arguments = forcedToolArguments.trimmingCharacters(in: .whitespacesAndNewlines)
                    if let forcedTool, (try? JSONSerialization.jsonObject(with: Data(arguments.utf8))) is [String: Any] {
                        admit([.toolCall(ToolCallOutput(
                            id: "call_\(UUID().uuidString.prefix(24))", name: forcedTool, arguments: arguments))])
                    } else if forcedTool != nil {
                        let continued = String(forcedToolArguments.dropFirst(prefixAnswer.count))
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                        if !continued.isEmpty { continuation.yield(.text(continued)) }
                    }
                    let completed = !(Task.isCancelled || stoppedAfterCall)
                    await Self.settle(mlxTask, cancelled: !completed)
                    if completed, let session = constraint?.session { await self.recycle(session, epoch: epoch) }
                    await self.keep(tokens: promptTokens, namespace: namespace, releasing: leaseID, epoch: epoch)
                    if !Task.isCancelled {
                        let generationTime = (clock.now - startTime).inSeconds + prefillSeconds
                        let timeToFirstToken = (firstTokenAt.map { ($0 - startTime).inSeconds } ?? 0) + prefillSeconds
                        let mem = Memory.snapshot()
                        let reasoning = reasoningText
                        let reasoningTokens = reasoning.isEmpty ? nil : await container.perform {
                            $0.tokenizer.encode(text: reasoning, addSpecialTokens: false).count
                        }

                        let metadata = GenerationResponse(
                            generatedText: "", 
                            reasoningText: nil,
                            promptTokens: lastInfo.map { reusedTokens.rawValue + $0.promptTokenCount } ?? promptTokens.count,
                            cachedPromptTokens: reusedTokens.rawValue,
                            completionTokens: lastInfo?.generationTokenCount ?? tokenCount,
                            tokensPerSecond: lastInfo?.tokensPerSecond ?? 0,
                            generationTime: generationTime,
                            timeToFirstToken: timeToFirstToken,
                            memory: .init(
                                peakBytes: mem.peakMemory,
                                activeBytes: mem.activeMemory,
                                cacheBytes: mem.cacheMemory
                            ),
                            finishReason: stoppedOnRepetition ? .repetition
                                : (lastInfo?.generationTokenCount ?? 0) >= requestedMaxTokens ? .length : .stop,
                            reasoningTokens: reasoningTokens,
                            draftTokens: lastInfo?.speculativeDecodingTelemetry?.draftTokenCount,
                            acceptedDraftTokens: lastInfo?.speculativeDecodingTelemetry?.acceptedDraftTokenCount,
                            forcedTokens: constraint?.fastForward == true ? constraint?.session.totalForcedTokens : nil,
                            modelCalls: lastInfo?.speculativeDecodingTelemetry?.targetModelCallCount,
                            toolCallRepairs: toolCallRepairs,
                            toolCallRejections: toolCallRejections
                        )
                        continuation.yield(.metadata(metadata))
                    }
                    continuation.finish()
                }

                continuation.onTermination = { _ in
                    task.cancel()
                }
            }
        } catch let err as BasaltError {
            throw err
        } catch {
            throw .inferenceHardwareFailure(reason: error.localizedDescription)
        }
    }

    public var isLoaded: Bool { container != nil }

    static func constraining(
        _ components: GenerationComponents, with constraint: OutputConstraint?, phase: JSONConstraintProcessor.Phase
    ) -> GenerationComponents {
        guard let constraint else { return components }
        return components.appendingBasaltProcessor {
            JSONConstraintProcessor(session: constraint.session, vocabulary: constraint.vocabulary, phase: phase)
        }
    }

    static func carriesMoreThanTokens(_ input: LMInput) -> Bool {
        input.image != nil || input.video != nil || input.audio != nil
            || input.text.mask.map { !all($0).item(Bool.self) } ?? false
    }

    static func appending(_ ids: [Int], to input: LMInput) -> LMInput {
        let tokens = input.text.tokens
        let extra = MLXArray(ids.map(Int32.init)).asType(tokens.dtype)
            .reshaped(Array(tokens.shape.dropLast()) + [ids.count])
        return LMInput(tokens: concatenated([tokens, extra], axis: -1))
    }

    struct GenerationLaunch {
        let stream: AsyncStream<Generation>
        let task: Task<Void, Never>?
        let cache: PromptCache?
        var anchors: [PromptCache] = []
        let reusedTokens: TokenCount
        var exactTokens = TokenCount(Int.max)
        var prefillSeconds: TimeInterval = 0
        var mtpPrefix: MTPPrefix?
    }

    static let mtpPromptLimit = 8192
    static let mtpPrefixLimit = 4

    static func handOff<T: TokenIteratorProtocol>(
        promptTokenCount: Int, modelConfiguration: ModelConfiguration, tokenizer: any MLXLMCommon.Tokenizer,
        iterator: consuming T, tools: [ToolSpec]?, toolCallPolicy: ToolCallPolicy, since started: ContinuousClock.Instant = .now
    ) -> (AsyncStream<Generation>, Task<Void, Never>, TimeInterval) {
        Stream.defaultStream.synchronize()
        let prefill = (ContinuousClock.now - started).inSeconds
        let (stream, task) = generateTask(
            promptTokenCount: promptTokenCount, modelConfiguration: modelConfiguration, tokenizer: tokenizer,
            iterator: iterator, tools: tools, toolCallPolicy: toolCallPolicy)
        return (stream, task, prefill)
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

    private func batchedDecoder(_ container: ModelContainer, epoch: Int) async -> BatchedDecoder? {
        guard batchSlots > 1, epoch == modelEpoch, appliedMemoryDirective < .tight else { return nil }
        if let batchDecoder { return batchDecoder }
        let slots = batchSlots
        let made = await container.perform { BatchedDecoder(model: $0.model, slots: slots) }
        guard epoch == modelEpoch else { return nil }
        if let batchDecoder { return batchDecoder }
        batchDecoder = made
        return made
    }

    var batchedRows: Int { batchDecoder?.admittedRows ?? 0 }

    private func routesExperts(_ container: ModelContainer, epoch: Int) async -> Bool {
        if epoch == modelEpoch, let known = modelRoutesExperts { return known }
        let routes = await container.perform { PromptLookupTokenIterator.routesExperts($0.model) }
        if epoch == modelEpoch {
            modelRoutesExperts = routes
        }
        return routes
    }

    private func supportsMultiTokenForward(_ container: ModelContainer, epoch: Int) async -> Bool {
        if epoch == modelEpoch, let known = modelSupportsMultiTokenForward { return known }
        let supports = await container.perform { context in
            let cache = try? context.model.newCache(parameters: nil)
            return cache.map { PromptLookupTokenIterator.supports($0, routesExperts: PromptLookupTokenIterator.routesExperts(context.model)) } ?? false
        }
        if epoch == modelEpoch {
            modelSupportsMultiTokenForward = supports
        }
        return supports
    }

    private func snapshotBoundary(
        _ count: Int, container: ModelContainer, epoch: Int, template: String?,
        _ makeInput: @Sendable (String) -> UserInput
    ) async -> Int {
        let hybrid: Bool
        if let known = modelIsHybrid {
            hybrid = known
        } else {
            let kinds = (try? await container.perform { context -> (hybrid: Bool, recurrent: Bool) in
                let cache = try context.model.newCache(parameters: nil)
                return (PromptCache(cache).isHybrid, cache.contains { $0 is MambaCache })
            }) ?? (hybrid: false, recurrent: false)
            hybrid = kinds.hybrid
            if epoch == modelEpoch {
                modelIsHybrid = kinds.hybrid
                modelHasRecurrentState = kinds.recurrent
            }
        }
        guard hybrid else { return count - 1 }
        if let template, let suffix = templateSuffixes[template] {
            return count - suffix
        }
        let suffix = await HybridPrefixSnapshot.templateSuffixLength(container, makeInput)
        if let template, epoch == modelEpoch {
            if templateSuffixes.count >= 64 {
                templateSuffixes.removeAll()
            }
            templateSuffixes[template] = suffix
        }
        return count - suffix
    }

    private func anchorOffsets(
        _ turns: [ChatMessageInput], prompt: [Int], reused: Int, boundary: Int,
        container: ModelContainer, tools: [ToolSpec]?, context: [String: any Sendable]?
    ) async -> [Int] {
        var found: [Int] = []
        let spacing = HybridPrefixSnapshot.minAnchorSpacing
        guard boundary - reused >= 2 * spacing else { return found }
        for prefix in HybridPrefixSnapshot.anchorPrefixes(turns) where found.count < hybridAnchors {
            guard let rendered = try? await container.prepare(
                input: UserInput(chat: Self.chat(prefix, lastText: prefix.last?.text ?? ""), tools: tools, additionalContext: context))
            else { continue }
            let offset = zip(prompt, rendered.text.tokens.asArray(Int.self)).prefix { $0 == $1 }.count
            guard offset - reused >= spacing else { break }
            if boundary - offset >= spacing, found.allSatisfy({ abs($0 - offset) >= spacing }) {
                found.append(offset)
            }
        }
        return found
    }

    private func promptCacheNamespace(for request: GenerationRequest) -> String? {
        guard promptCaches.byteBudget > 0, (request.kvPrecision ?? kvPrecision) == nil else { return nil }
        return request.cacheKey
    }

    static let modelSwitchPersistDeadline = Duration.seconds(10)

    private func cacheLayout(_ container: ModelContainer, epoch: Int) async -> [String]? {
        if let modelCacheLayout { return modelCacheLayout }
        let layout = try? await container.perform { try $0.model.newCache(parameters: nil).map(PromptCacheDiskStore.layoutName) }
        if epoch == modelEpoch {
            modelCacheLayout = layout
        }
        return layout
    }

    private func restoreFromDisk(_ namespace: String, prompt: [Int], container: ModelContainer, epoch: Int) async {
        guard promptCacheDisk != nil, modelFingerprint != nil,
              promptCaches.storedPrefix(of: prompt, in: namespace) < prompt.count - 1,
              let layout = await cacheLayout(container, epoch: epoch),
              epoch == modelEpoch, let fingerprint = modelFingerprint,
              let hit = promptCacheDisk?.longestPrefix(
                fingerprint: fingerprint, namespace: namespace, of: Array(prompt.dropLast()),
                longerThan: promptCaches.storedPrefix(of: prompt, in: namespace), maxBytes: promptCaches.byteBudget)
        else { return }
        let started = ContinuousClock.now
        do {
            guard let cache = try promptCacheDisk?.load(hit.entry, layout: layout) else { return }
            promptCaches.store(namespace, tokens: Array(hit.entry.tokens.prefix(hit.usable)), cache: cache)
            let seconds = (ContinuousClock.now - started).inSeconds
            logger.info(String(
                format: "Restored %d of %d stored prompt tokens from disk: %.1f MB in %.3f s (%.2f GB/s)",
                hit.usable, hit.entry.tokens.count, Double(hit.entry.bytes) / 1e6, seconds,
                Double(hit.entry.bytes) / 1e9 / max(seconds, 1e-9)))
        } catch {
            let removed = promptCacheDisk?.discardIfBroken(hit.entry, after: error) == true
            logger.warn("Prompt-cache file \(hit.entry.name) could not be restored (\(removed ? "removed" : "kept, the error may be transient")): \(error)")
        }
    }

    public func persistPromptCache(within deadline: Duration) async {
        guard promptCacheDisk != nil, let fingerprint = modelFingerprint else { return }
        let started = ContinuousClock.now
        var written: Set<String> = []
        var skipped = 0
        for path in promptCaches.storedPathsByRecency() {
            guard let lease = promptCaches.lease(path.namespace, for: path.tokens + [0]) else { continue }
            let tokens = Array(path.tokens.prefix(lease.reusedTokens))
            guard !tokens.isEmpty else { continue }
            let alreadyOnDisk = promptCacheDisk?.has(fingerprint: fingerprint, namespace: path.namespace, tokens: tokens) == true
            let expectedWrite = alreadyOnDisk ? 0 : Double(PromptCacheDiskStore.estimatedBytes(of: lease.cache, tokens: tokens.count))
                / PromptCacheDiskStore.assumedWriteBytesPerSecond
            guard ContinuousClock.now - started + .seconds(expectedWrite) <= deadline else {
                skipped += 1
                continue
            }
            if let name = promptCacheDisk?.store(
                fingerprint: fingerprint, model: modelDescription, namespace: path.namespace,
                tokens: tokens, cache: lease.cache, keeping: written) {
                written.insert(name)
            }
        }
        logger.info("Persisted \(written.count) prompt-cache paths in \(String(format: "%.2f", (ContinuousClock.now - started).inSeconds)) s"
            + (skipped > 0 ? "; \(skipped) paths skipped that could not be written within the \(deadline) deadline" : ""))
    }

    private func keep(tokens prompt: [Int], namespace: String?, releasing lease: Int, epoch: Int) {
        leases[lease] = nil
        let launch = launches.removeValue(forKey: lease)
        guard epoch == modelEpoch, let namespace, let launch else { return }
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

extension MLXInferenceEngine {
    func useMTPDrafter(in container: ModelContainer, _ make: @Sendable (ModelContext) -> (any MTPDrafterModel)?) async {
        await container.update { $0.mtpDrafter = make($0) }
        hasMTPDrafter = await container.perform { $0.mtpDrafter != nil }
    }

    func loadMTPDrafter(for loaded: ModelContainer, directory: URL, modelPath: String) async {
        let logger = self.logger, speculativeLookup = self.speculativeLookup
        await useMTPDrafter(in: loaded) { context in
            do {
                guard let drafter = try MTPSidecar.load(directory: directory, target: context.model) else {
                    logger.warn("--mtp: \(modelPath) has no MTP head this build can use (Qwen3.5-family target plus mtp.safetensors); decoding stays plain")
                    return nil
                }
                logger.info("MTP head loaded from \(MTPSidecar.file(in: directory)?.path ?? modelPath); requests without constraints, lookup or --kv-bits and prompts up to \(Self.mtpPromptLimit) tokens draft with it")
                if speculativeLookup {
                    logger.warn("--mtp with --speculative-lookup: lookup wins, so the MTP head drafts only for requests that set speculative_lookup to false")
                }
                return drafter
            } catch {
                logger.warn("--mtp: the MTP head of \(modelPath) did not load (\(error.localizedDescription)); decoding stays plain")
                return nil
            }
        }
    }

    func readPromptCaches<R: Sendable>(_ read: @Sendable (PromptCacheTree) throws -> R) rethrows -> R {
        try read(promptCaches)
    }

    fileprivate func hold(_ lease: consuming PromptCacheLease?) -> Int {
        leaseCount += 1
        leases[leaseCount] = PromptCacheLeaseSlot(lease)
        return leaseCount
    }

    static func launch(input: LMInput, prompt: CachedPrompt, options: LaunchOptions, context: ModelContext) throws -> GenerationLaunch {
        let start = GenerationLauncher(
            input: input, params: options.params, components: options.components, tools: options.tools, context: context, started: .now)
        if options.mtp, let drafter = context.mtpDrafter, !carriesMoreThanTokens(input) {
            return try start.mtp(
                MTPRequest(drafter: drafter, blockSize: options.mtpBlockSize, prefix: options.mtpPrefix), prompt: prompt,
                keepsCache: options.keepsCache)
        }
        guard options.keepsCache else {
            let plan = FreshCache.plan(
                kvBitsSet: options.params.kvBits != nil, headMask: options.kvHeadMask, questTopK: options.questTopK,
                zones: options.kvZones, precision: options.kvPrecision)
            return try start.fresh(plan, batch: options.batch, lookup: options.lookup)
        }
        return try start.cached(prompt, batch: options.batch, lookup: options.lookup)
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
