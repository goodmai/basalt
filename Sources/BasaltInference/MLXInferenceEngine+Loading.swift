import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import Tokenizers
import Hub
import BasaltCore

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

extension MLXInferenceEngine {
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

    public var isLoaded: Bool { container != nil }

    static func imageToken(in config: [String: Any]?) -> Int? {
        (config?["image_token_id"] ?? config?["image_token_index"]) as? Int
    }

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
        let modelType = ModelCache.config(at: directory)?["model_type"] as? String
        try ModelCapabilities.refuseUnverifiedKVBits(configuration.kvPrecision, modelType: modelType, path: modelPath)

        if container != nil { await unloadCurrentModel() }
        do { try faults.check(.unload) } catch { throw await Self.loadFailure(error, modelPath: modelPath) }
        record(.loadStarted(path: modelPath))
        let epoch = modelEpoch
        do {
            try faults.check(.load)
            if let gpuCacheLimit = configuration.gpuCacheLimit { Memory.cacheLimit = gpuCacheLimit }
            let loaded = try await loadContainer(from: directory)
            try await loaded.perform { try PrismHadamard.prepareLoaded($0.model); SmallMQuantizedLinear.installIfEnabled(in: $0.model) }
            try await probe(loaded, directory: directory, modelPath: modelPath)
            let drafts = configuration.mtp ? await loadMTPDrafter(for: loaded, directory: directory, modelPath: modelPath) : false
            let capabilities = ModelCapabilities.of(modelType: modelType)
            let facts = ModelFacts(
                profile: ModelProfileRegistry.profile(forModelAt: modelPath),
                samplingDefaults: SamplingDefaults(generationConfig: ModelCache.generationConfig(at: directory)),
                controlTokens: ControlTokenGuard(modelDirectory: directory),
                fingerprint: promptCacheDisk == nil ? nil : PromptCacheDiskStore.fingerprint(modelDirectory: directory),
                description: modelPath, hasMTPDrafter: drafts, expertReserve: await Self.expertReserve(loaded), capabilities: capabilities ?? .conservative,
                imageToken: Self.imageToken(in: ModelCache.config(at: directory)))
            guard attach(loaded, facts: facts, epoch: epoch) else { return }
            if capabilities == nil { logger.warn(ModelCapabilities.unverifiedWarning(modelType: modelType)) }
            await probeKVZones()
            await wireWorkingSet()
            await probeKVHeads(directory: directory)
            await probeQuest()
            logger.info("Sampling defaults from generation_config.json: \(samplingDefaults)")
            for line in ModelCache.config(at: directory).map(architectureLines(config:)) ?? [] { logger.info(line) }
            logger.info("Model loaded and initialized successfully.")
        } catch {
            let failure = await Self.loadFailure(error, modelPath: modelPath)
            record(.loadFailed(epoch, failure))
            throw failure
        }
    }

    func attach(
        _ container: ModelContainer, profile: any ModelProfile = GenericProfile(),
        samplingDefaults: SamplingDefaults = SamplingDefaults(), controlTokens: ControlTokenGuard = .none,
        fingerprint: String? = nil, modelDescription: String = ""
    ) {
        record(.loadStarted(path: modelDescription))
        attach(container, facts: ModelFacts(
            profile: profile, samplingDefaults: samplingDefaults, controlTokens: controlTokens,
            fingerprint: fingerprint, description: modelDescription), epoch: modelEpoch)
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
}

extension MLXInferenceEngine {
    static let mtpPromptLimit = 8192
    static let mtpPrefixLimit = 4

    @discardableResult
    func useMTPDrafter(in container: ModelContainer, _ make: @Sendable (ModelContext) -> (any MTPDrafterModel)?) async -> Bool {
        let epoch = modelEpoch
        await container.update { $0.mtpDrafter = make($0) }
        let drafts = await container.perform { $0.mtpDrafter != nil }
        record(.learned(epoch) { $0.hasMTPDrafter = drafts })
        return drafts
    }

    func loadMTPDrafter(for loaded: ModelContainer, directory: URL, modelPath: String) async -> Bool {
        let logger = self.logger, speculativeLookup = configuration.speculativeLookup
        return await useMTPDrafter(in: loaded) { context in
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
}

extension MLXInferenceEngine {
    public var supportsKVZones: Bool { modelAttendsThroughZones }

    func probeKVZones() async {
        guard let zones = configuration.kvZones, modelFacts.capabilities.allows(.kvZones), let container else { return }
        let epoch = modelEpoch
        let precision = configuration.kvPrecision ?? .q8
        let probed = (try? await container.perform { try ZonedKVCache.probe($0.model, precision: precision) })
            ?? (attendsThroughCache: false, rotatesValues: false, readsThroughUpdate: false)
        let attends = probed.attendsThroughCache
        let evictable = zones.evictedMiddleFraction > 0 && (attends || probed.readsThroughUpdate)
        let chunks = evictable ? await prefillsWithinWindow(zones: zones, precision: precision, container: container) : false
        let rotationUnavailable = attends && precision.rotatesValues && !probed.rotatesValues
        let throughUpdate = !attends && chunks
        record(.learned(epoch) {
            ($0.attendsThroughZones, $0.prefillsInChunks, $0.valueRotationUnavailable, $0.evictsThroughUpdate) =
                (attends, attends && chunks, rotationUnavailable, throughUpdate)
        })
        if throughUpdate {
            logger.info("--kv-evict on this model: its attention reads the whole cache through update() (Gemma 4), so its full-attention layers keep \(zones.sinkTokens) sink tokens, the last \(zones.windowTokens) and \(Int(((1 - zones.evictedMiddleFraction) * 100).rounded()))% of the tokens between them in fp16; --kv-zones and --kv-bits do not apply to its cache")
        } else if !attends {
            logger.warn("--kv-zones does not apply to this model: its attention reads the whole cache through update(), or its head dimension has no quantization group. Its quantized requests use uniform --kv-bits, k8v4 as 8-bit\(zones.evictedMiddleFraction > 0 ? ", and --kv-evict keeps every token" : "").")
        } else if rotationUnavailable {
            logger.warn("--kv-bits \(precision) cannot rotate this model's value heads: the Hadamard transform is its own inverse only for head dimensions 2^k and 12·2^k. It runs as k8v4.")
        }
    }

    private func prefillsWithinWindow(zones: KVZones, precision: KVPrecision, container: ModelContainer) async -> Bool {
        do {
            let chunks = try await container.perform {
                try ZonedKVCache.prefillsWithinWindow($0.model, zones: zones, precision: precision, prefill: GenerateParameters().prefill)
            }
            if !chunks {
                logger.warn("--kv-evict does not apply to this model: it prefills a prompt longer than the recent window in one pass, so its zoned cache keeps every token and is sized that way.")
            }
            return chunks
        } catch {
            logger.warn("--kv-evict is off for this model: its prefill probe failed (\(error)), so its zoned cache keeps every token and is sized that way.")
            return false
        }
    }

    func probeKVHeads(directory: URL) async {
        guard let ratio = configuration.kvHeadSplit, let container else { return }
        guard modelFacts.capabilities.allows(.kvHeadSplit) else {
            logger.warn("--kv-head-split is not verified on this model family, so the KV cache stays dense")
            return
        }
        let epoch = modelEpoch
        guard (try? await container.perform({ try ZonedKVCache.modelAttendsThroughCache($0.model) })) == true else {
            logger.warn("--kv-head-split does not apply to this model: its attention does not go through the cache")
            return
        }
        do {
            let started = ContinuousClock.now
            let selection = try await KVHeadClassifier.loadOrSelect(directory: directory, container: container, ratio: ratio)
            let measured = "answers kept \(selection.answersKept), mean KL \(selection.meanKL) nats per token against the dense cache (limit \(KVHeadSelection.maxMeanKL)), chosen or read in \(ContinuousClock.now - started)"
            guard selection.keepsFidelity else {
                logger.warn("--kv-head-split \(ratio) does not apply to this model: the \(selection.keptHeads) KV heads that best keep its output still lose the calibration passkeys (\(measured)); the KV cache stays dense")
                return
            }
            record(.learned(epoch) { $0.kvHeadMask = selection.mask })
            logger.info("KV head split: \(selection.keptHeads) KV heads chosen by output fidelity keep the whole context, the rest keep \(HeadSplitKVCache.sinkTokens) sink and \(HeadSplitKVCache.windowTokens) recent tokens (\(measured))")
        } catch {
            logger.warn("--kv-head-split: choosing the heads failed (\(error.localizedDescription)); the KV cache stays dense")
        }
    }

    func probeQuest() async {
        guard let questTopK = configuration.questTopK, modelFacts.capabilities.allows(.quest), let container else { return }
        let epoch = modelEpoch
        do {
            guard try await container.perform({ try QuestKVCache.modelAttendsThroughCache($0.model) }) else {
                logger.warn("--quest-top-k does not apply to this model: its full-attention layers read the whole cache through cache.update (Gemma 4 does), so they cannot attend to a selection of blocks; attention stays dense")
                return
            }
        } catch {
            logger.warn("--quest-top-k: probing the model failed (\(error.localizedDescription)); attention stays dense")
            return
        }
        record(.learned(epoch) { $0.questApplies = true })
        logger.info("Quest: a decode step attends to \(QuestKVCache.sinkTokens) sink tokens, the last \(QuestKVCache.windowTokens) or more, and the \(questTopK) best \(QuestKVCache.blockTokens)-token blocks between them")
    }

    func routesExperts(_ container: ModelContainer, epoch: ModelEpoch) async -> Bool {
        if epoch == modelEpoch, let known = modelFacts.routesExperts { return known }
        let routes = await container.perform { PromptLookupTokenIterator.routesExperts($0.model) }
        record(.learned(epoch) { $0.routesExperts = routes })
        return routes
    }

    func supportsMultiTokenForward(_ container: ModelContainer, epoch: ModelEpoch) async -> Bool {
        if epoch == modelEpoch, let known = modelFacts.supportsMultiTokenForward { return known }
        let supports = await container.perform { context in
            let cache = try? context.model.newCache(parameters: nil)
            return cache.map { PromptLookupTokenIterator.supports($0, routesExperts: PromptLookupTokenIterator.routesExperts(context.model)) } ?? false
        }
        record(.learned(epoch) { $0.supportsMultiTokenForward = supports })
        return supports
    }

    func cacheLayout(_ container: ModelContainer, epoch: ModelEpoch) async -> [String]? {
        if let known = modelFacts.cacheLayout { return known }
        let layout = try? await container.perform { try $0.model.newCache(parameters: nil).map(PromptCacheDiskStore.layoutName) }
        record(.learned(epoch) { $0.cacheLayout = layout })
        return layout
    }
}

extension MLXInferenceEngine {
    public static func renewsResidency() {
        setenv("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS", "1000", 0)
    }

    func wireWorkingSet() async {
        await releaseWiredMemory()
        guard configuration.wiresWorkingSet else { return }
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
}
