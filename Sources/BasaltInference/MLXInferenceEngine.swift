import Foundation
import Synchronization
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

// MARK: — Real implementation

/// Actor-изолированный MLX inference engine.
/// Все Metal/MLX вызовы выполняются из одного контекста — нет гонок на GPU command queue.
public actor MLXInferenceEngine: InferenceEngine {

    private(set) var container: ModelContainer?

    /// Resolved from the loaded model's config.json. Supplies per-family sampling
    /// defaults that the caller did not override.
    private var profile: any ModelProfile = GenericProfile()
    private var samplingDefaults = SamplingDefaults()
    private let logger = GemLogger(module: "MLXInferenceEngine")

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

    private(set) var promptCaches: PromptCacheTree
    private var kvMemoryBudget: Int64?
    private var controlTokens = ControlTokenGuard.none

    public var kvMemoryBudgetBytes: Int64? { kvMemoryBudget }
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
    private let configuredPromptCacheBytes: Int?
    private var modelEpoch = 0
    private var modelIsHybrid: Bool?
    private var modelRoutesExperts: Bool?
    private var templateSuffixes: [String: Int] = [:]
    private var grammarVocabulary: Result<GrammarVocabulary, ConstraintUnavailable>?
    private(set) var idleGrammars: [String: [GrammarConstraint]] = [:]
    private var idleGrammarRecency: [String] = []
    private let cacheBlendRatio: Float?
    private(set) var cacheBlend: CacheBlendStore?
    private let speculativeLookup: Bool
    private let speculativeRecycling: Bool
    private(set) var promptCacheDisk: PromptCacheDiskStore?
    private var modelFingerprint: String?
    private var modelDescription = ""
    private var modelCacheLayout: [String]?

    public init(
        reasoningEffort: String? = nil,
        gpuCacheLimit: Int? = 512 << 20,
        promptCacheBytes: Int? = nil,
        kvPrecision: KVPrecision? = nil,
        quantizedKVStart: Int = KVCacheGeometry.defaultQuantizedKVStart,
        kvZones: KVZones? = nil,
        repetitionPenalty: Float? = nil,
        topK: Int? = nil,
        minP: Float? = nil,
        seed: UInt64? = nil,
        cacheBlendRatio: Float? = nil,
        speculativeLookup: Bool = false,
        speculativeRecycling: Bool = false,
        promptCacheDisk: PromptCacheDiskStore? = nil
    ) {
        self.promptCacheDisk = promptCacheDisk
        self.speculativeLookup = speculativeLookup
        self.speculativeRecycling = speculativeRecycling
        self.reasoningEffort = reasoningEffort
        self.gpuCacheLimit = gpuCacheLimit
        self.kvPrecision = kvPrecision
        self.quantizedKVStart = quantizedKVStart
        self.kvZones = kvZones
        self.repetitionPenalty = repetitionPenalty
        self.topK = topK
        self.minP = minP
        self.seed = seed
        self.configuredPromptCacheBytes = promptCacheBytes
        self.cacheBlendRatio = cacheBlendRatio
        self.promptCaches = PromptCacheTree(byteBudget: promptCacheBytes ?? 0)
    }

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
        self.controlTokens = controlTokens
        grammarVocabulary = nil
        idleGrammars = [:]
        idleGrammarRecency = []
        self.profile = profile
        self.samplingDefaults = samplingDefaults
        let workingSet = PromptCacheBudget.gpuWorkingSet()
        let weights = Memory.activeMemory
        let budget = configuredPromptCacheBytes ?? PromptCacheBudget.automatic(workingSet: workingSet, weights: weights)
        kvMemoryBudget = Int64(max(0, workingSet - weights - budget))
        modelAttendsThroughZones = false
        modelPrefillsInChunks = false
        let blendBytes = cacheBlendRatio == nil ? 0 : budget / 2
        promptCaches = PromptCacheTree(byteBudget: budget - blendBytes)
        cacheBlend = cacheBlendRatio.map { CacheBlendStore(ratio: $0, byteBudget: blendBytes) }
        modelEpoch += 1
        modelIsHybrid = nil
        modelRoutesExperts = nil
        templateSuffixes = [:]
        logger.info("Prompt cache budget: \(promptCaches.byteBudget >> 20) MB")
    }

    /// `registerAliases` is a static method, so it is *not* actor-isolated and two
    /// concurrent loads could both observe `false` and register twice. Claim the
    /// flag atomically instead of check-then-set on a bare Bool.
    private static let aliasesRegistered = Mutex<Bool>(false)

    /// Register architecture aliases (Qwen 3.8, Qwen 3.6, MiniMax M2) in LLMTypeRegistry.
    public static func registerAliases() async {
        let alreadyDone = aliasesRegistered.withLock { done -> Bool in
            defer { done = true }
            return done
        }
        guard !alreadyDone else { return }

        await LLMTypeRegistry.shared.registerModelType("qwen3_8") { data in
            let config = try JSONDecoder.json5().decode(MLXLLM.Qwen35Configuration.self, from: data)
            return MLXLLM.Qwen35Model(config)
        }
        await LLMTypeRegistry.shared.registerModelType("qwen3.8") { data in
            let config = try JSONDecoder.json5().decode(MLXLLM.Qwen35Configuration.self, from: data)
            return MLXLLM.Qwen35Model(config)
        }
        await LLMTypeRegistry.shared.registerModelType("qwen3_8_moe") { data in
            let config = try JSONDecoder.json5().decode(MLXLLM.Qwen35Configuration.self, from: data)
            return MLXLLM.Qwen35MoEModel(config)
        }
        await LLMTypeRegistry.shared.registerModelType("qwen3.8_moe") { data in
            let config = try JSONDecoder.json5().decode(MLXLLM.Qwen35Configuration.self, from: data)
            return MLXLLM.Qwen35MoEModel(config)
        }
        await LLMTypeRegistry.shared.registerModelType("qwen3_6") { data in
            let config = try JSONDecoder.json5().decode(MLXLLM.Qwen35Configuration.self, from: data)
            return MLXLLM.Qwen35Model(config)
        }
        await LLMTypeRegistry.shared.registerModelType("qwen3.6") { data in
            let config = try JSONDecoder.json5().decode(MLXLLM.Qwen35Configuration.self, from: data)
            return MLXLLM.Qwen35Model(config)
        }
        await LLMTypeRegistry.shared.registerModelType("qwen3_6_moe") { data in
            let config = try JSONDecoder.json5().decode(MLXLLM.Qwen35Configuration.self, from: data)
            return MLXLLM.Qwen35MoEModel(config)
        }
        await LLMTypeRegistry.shared.registerModelType("minimax_m2") { data in
            let config = try JSONDecoder.json5().decode(MiniMaxConfiguration.self, from: data)
            return MiniMaxModel(config)
        }
        await LLMTypeRegistry.shared.registerModelType("minimax-m2") { data in
            let config = try JSONDecoder.json5().decode(MiniMaxConfiguration.self, from: data)
            return MiniMaxModel(config)
        }
    }

    // MARK: — Load

    public func load(modelPath: String) async throws(BasaltError) {
        logger.trace("Loading model from path: \(modelPath)")
        await Self.registerAliases()
        // ServeCommand всегда передаёт абсолютный локальный путь после разрешения кэша.
        // Если путь не начинается с "/" — это нераспознанный формат.
        guard modelPath.hasPrefix("/") || modelPath.hasPrefix(".") else {
            // HF repo ID без локального кэша — сообщаем пользователю
            throw .modelNotFound(
                identifier: "'\(modelPath)' не найден в локальном кэше. " +
                "Сначала загрузите: Gem models download \(modelPath)"
            )
        }

        let directory = URL(fileURLWithPath: modelPath)
        guard ModelCache.hasUsableConfig(at: directory) else {
            logger.error("Refusing to load \(modelPath): no readable config.json; the current model stays loaded")
            throw .modelNotFound(identifier: modelPath)
        }

        do {
            // Free the old model BEFORE loading the new one to avoid a peak-memory spike
            // where both models occupy GPU/unified memory simultaneously.
            // Setting container to nil drops all ARC references to the ModelContainer,
            // releasing the underlying MLXArray tensors and Metal buffers.
            if container != nil {
                await persistPromptCache(within: Self.modelSwitchPersistDeadline)
                container = nil
                promptCaches.removeAll()
                cacheBlend = nil
                modelEpoch += 1
                Memory.clearCache()   // flush MLX's internal memory allocator cache
                logger.info("Previous model unloaded, GPU cache cleared.")
            }

            // Unbounded buffer cache competes with model weights for unified memory
            // and pushes large models (27B @ 4-bit ≈ 16 GB) into swap on 24 GB Macs.
            // ponytail: fixed 512 MB cap; make it a --gpu-cache flag if a workload needs more.
            if let gpuCacheLimit { Memory.cacheLimit = gpuCacheLimit }

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

            attach(try await loadModelContainer(from: directory, using: DefaultTokenizerLoader()),
                   profile: ModelProfileRegistry.profile(forModelAt: modelPath),
                   samplingDefaults: SamplingDefaults(generationConfig: ModelCache.generationConfig(at: directory)),
                   controlTokens: ControlTokenGuard(modelDirectory: directory),
                   fingerprint: promptCacheDisk == nil ? nil : PromptCacheDiskStore.fingerprint(modelDirectory: directory),
                   modelDescription: modelPath)
            await probeKVZones()
            logger.info("Sampling defaults from generation_config.json: \(samplingDefaults)")
            
            // Dynamic config inspection
            if let json = ModelCache.config(at: directory) {
                
                let arch = (json["architectures"] as? [String])?.first ?? "Unknown"
                let modelType = json["model_type"] as? String ?? "Unknown"
                
                let textConfig = json["text_config"] as? [String: Any]
                let numExperts = json["num_experts"] as? Int ?? textConfig?["num_experts"] as? Int
                let isMoE = (numExperts ?? 0) > 0
                
                let qConfig = json["quantization_config"] as? [String: Any] ?? json["quantization"] as? [String: Any]
                let bits = qConfig?["bits"] as? Int
                let groupSize = qConfig?["group_size"] as? Int
                let qMode = qConfig?["mode"] as? String
                
                logger.info("--- Model Architecture ---")
                logger.info("Type: \(modelType) (\(arch))")
                if isMoE {
                    logger.info("Topology: Sparse MoE (Experts: \(numExperts ?? 0))")
                } else {
                    logger.info("Topology: Dense")
                }
                if let b = bits {
                    let modeStr = qMode.map { " [\($0)]" } ?? ""
                    logger.info("Quantization: \(b)-bit (Group Size: \(groupSize ?? -1))\(modeStr)")
                } else if let qDict = json["quantization"] as? [String: Any], !qDict.isEmpty {
                    logger.info("Quantization: Dynamic Quantization (DQ - per-layer, \(qDict.count) layers)")
                } else {
                    logger.info("Quantization: None (FP16/BF16 Base)")
                }
                logger.info("--------------------------")
            }
            logger.info("Model loaded and initialized successfully.")
        } catch {
            throw await Self.loadFailure(error, modelPath: modelPath)
        }
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

    // MARK: — Template Context
    public static func buildTemplateContext(
        request: GenerationRequest,
        defaultReasoningEffort: String? = nil,
        profile: (any ModelProfile)? = nil
    ) -> [String: any Sendable]? {
        // ReasoningPolicy(rawValue:) maps "off"/"none" to disabled; each profile maps level names
        // onto its template (Qwen: "high" → "xhigh"), GenericProfile passes them through.
        let policy = reasoningPolicy(request: request, defaultReasoningEffort: defaultReasoningEffort)
        let context = (profile ?? GenericProfile()).templateContext(for: policy)
        return context.isEmpty ? nil : context
    }

    static func reasoningPolicy(request: GenerationRequest, defaultReasoningEffort: String?) -> ReasoningPolicy {
        request.enableThinking == false
            ? .disabled
            : ReasoningPolicy(rawValue: request.reasoningEffort ?? defaultReasoningEffort) ?? .modelDefault
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
            acceptedDraftTokens: finalResponse.acceptedDraftTokens
        )
    }

    // MARK: — Tools

    static func chat(_ messages: [ChatMessageInput], lastText: String) -> [Chat.Message] {
        messages.enumerated().map { index, message in
            let text = index == messages.count - 1 ? lastText : message.text
            switch message.role {
            case .system:    return .system(text)
            case .user:      return .user(text)
            case .assistant: return .assistant(text, toolCalls: message.toolCalls?.map(toolCall))
            case .tool:      return .tool(text, id: message.toolCallId, name: message.toolName)
            }
        }
    }

    static func chatInput(
        _ messages: [ChatMessageInput], normalisation: String, tools: [ToolSpec]?, context: [String: any Sendable]?
    ) -> (@Sendable (String) -> UserInput, String?, String) {
        let last = messages[messages.count - 1]
        return (
            { UserInput(chat: Self.chat(messages, lastText: $0), tools: tools, additionalContext: context) },
            HybridPrefixSnapshot.templateKey([tools ?? [], context ?? [:], last.role.rawValue, last.toolCalls != nil, normalisation]),
            last.text
        )
    }

    static func templateFallbacks(_ turns: [ChatMessageInput]) -> [(String, [ChatMessageInput])] {
        [("leadingSystem", ChatMessageInput.leadingSystem(turns)), ("alternating", ChatMessageInput.alternating(turns))]
    }

    static func briefly(_ error: any Error, limit: Int = 300) -> String {
        let text = String(describing: error)
        return text.count > limit ? String(text.prefix(limit)) + "…" : text
    }

    static func isTemplateRefusal(_ error: any Error) -> Bool {
        if error is TemplateException { return true }
        if case JinjaError.runtime = error { return true }
        return false
    }

    static let thinkingOpeners = ["<think>", "<|channel>thought", "<|channel|>thought", "[THINK]"]
    static let thinkingClosers = ["</think>", "<channel|>", "<|channel|>", "<|turn|>", "[/THINK]"]
    static let strayThinkingClosers = ["</think>", "<channel|>", "[/THINK]"]
    static let thinkingMarkerHoldback = (thinkingOpeners + thinkingClosers).map(\.count).max()! - 1

    static func firstMarker(_ markers: [String], in text: some StringProtocol) -> Range<String.Index>? {
        markers.compactMap { text.range(of: $0) }.min { $0.lowerBound < $1.lowerBound }
    }

    static func endsInsideThinking(_ rendered: String, assistantPrefix: String = "") -> Bool {
        var tail = Substring(rendered)
        while tail.last?.isWhitespace == true { tail.removeLast() }
        var inside = thinkingOpeners.contains { tail.hasSuffix($0) }
        var rest = Substring(assistantPrefix)
        while true {
            if inside {
                guard let closer = firstMarker(thinkingClosers, in: rest) else { return true }
                rest = rest[closer.upperBound...]
                inside = false
            } else {
                let start = rest.drop(while: \.isWhitespace)
                guard let opener = thinkingOpeners.first(where: { start.hasPrefix($0) }) else { return false }
                rest = start.dropFirst(opener.count)
                inside = true
            }
        }
    }

    static func toolCall(_ call: ToolCallOutput) -> ToolCall {
        let arguments = (try? JSONDecoder().decode([String: JSONValue].self, from: Data(call.arguments.utf8)))
            ?? ["input": .string(call.arguments)]
        return ToolCall(function: .init(name: call.name, arguments: arguments), id: call.id.isEmpty ? nil : call.id)
    }

    /// OpenAI-style function specs for the chat template's tool section.
    static func toolSpecs(_ tools: [ToolDefinition]?) -> [ToolSpec]? {
        guard let tools, !tools.isEmpty else { return nil }
        return tools.map { tool in
            var function: [String: any Sendable] = ["name": tool.name]
            if let description = tool.description { function["description"] = description }
            if let json = tool.parametersJSON,
               let schema = try? JSONSerialization.jsonObject(with: Data(json.utf8)) {
                function["parameters"] = sendable(schema)
            }
            return ["type": "function", "function": function]
        }
    }

    /// JSONSerialization output (NSDictionary/NSArray/NSNumber) as plain Sendable values the
    /// Jinja renderer understands.
    private static func sendable(_ value: Any) -> any Sendable {
        switch value {
        case let dict as [String: Any]: return dict.mapValues { sendable($0) }
        case let array as [Any]:        return array.map { sendable($0) }
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() { return number.boolValue }
            return number.doubleValue.rounded() == number.doubleValue ? number.intValue : number.doubleValue
        case let string as String:      return string
        default:                        return String(describing: value)
        }
    }

    /// What a rejected tool call becomes: quantized models often emit almost-JSON, so a malformed
    /// call to a declared tool is repaired into a real call; anything else becomes a visible note
    /// instead of a silently empty turn.
    static func resolveRejectedToolCall(
        reason: String, toolName: String?, callID: String?,
        rawText: String, truncated: Bool, declaredTools: Set<String>
    ) -> StreamChunk {
        if reason == "malformed_syntax",
           let call = ToolCallRepairer.repairToolCall(
               name: toolName, callID: callID, rawText: rawText, truncated: truncated, declaredTools: declaredTools) {
            return .toolCall(call)
        }
        let tool = toolName.map { " to \($0)" } ?? ""
        return .text("\n[tool call\(tool) rejected: \(reason)]")
    }

    static func toolCallOutput(_ call: ToolCall) -> ToolCallOutput {
        let arguments = (try? JSONEncoder().encode(call.function.arguments)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        return ToolCallOutput(id: "call_\(UUID().uuidString.prefix(24))", name: call.function.name, arguments: arguments)
    }

    static let presenceContextSize = 1024

    struct OutputConstraint: Sendable {
        let session: GrammarSession
        let vocabulary: GrammarVocabulary
        let toolName: String?
    }

    static let grammarSchemaLimit = 16
    static let idleGrammarsPerSchema = 4

    func outputConstraint(for request: GenerationRequest) async throws(BasaltError) -> OutputConstraint? {
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
        let grammar: GrammarConstraint
        if epoch == modelEpoch, let idle = idleGrammars[schema]?.popLast() {
            grammar = idle
        } else {
            do {
                grammar = try GrammarConstraint(tokenizer: vocabulary.tokenizer, jsonSchema: schema)
            } catch GrammarError.invalidJSONSchema(let message) {
                return try unconstrained(message)
            } catch {
                return try unconstrained("\(error)")
            }
        }
        return OutputConstraint(
            session: GrammarSession(schema: schema, grammar: grammar), vocabulary: vocabulary, toolName: forcedTool?.name)
    }

    func recycle(_ session: GrammarSession, epoch: Int) {
        guard epoch == modelEpoch, let grammar = session.rewound() else { return }
        idleGrammarRecency.removeAll { $0 == session.schema }
        idleGrammarRecency.append(session.schema)
        if idleGrammarRecency.count > Self.grammarSchemaLimit { idleGrammars[idleGrammarRecency.removeFirst()] = nil }
        guard idleGrammars[session.schema, default: []].count < Self.idleGrammarsPerSchema else { return }
        idleGrammars[session.schema, default: []].append(grammar)
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

    static func stopTokenIds(_ context: ModelContext) -> Set<Int> {
        var ids = context.configuration.eosTokenIds
        if let eos = context.tokenizer.eosTokenId { ids.insert(eos) }
        for token in context.configuration.extraEOSTokens {
            if let id = context.tokenizer.convertTokenToId(token) { ids.insert(id) }
        }
        return ids
    }

    func sampling(for request: GenerationRequest) -> (GenerateParameters, GenerationComponents, shaping: (@Sendable () -> EntropyDynamicTemperature)?) {
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
            components = components.appendingLogitProcessor {
                GeneratedTokensPresencePenalty(penalty: presencePenalty, window: Self.presenceContextSize)
            }
        }
        var shaping: (@Sendable () -> EntropyDynamicTemperature)?
        if request.dynamicTemperature == true, temperature > 0 {
            let (topP, minP) = (params.topP, params.minP)
            shaping = { EntropyDynamicTemperature(baseTemperature: temperature, topP: topP, minP: minP) }
            params.topP = 1
            params.minP = 0
        }
        return (params, components, shaping)
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
            GemLogger(module: "MLXInferenceEngine").info(
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

    public func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk> {
        guard let container else {
            throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first")
        }
        let epoch = modelEpoch

        let requestedMaxTokens = request.maxTokens ?? 1024
        let (params, unbudgeted, shaping) = sampling(for: request)
        let constraint = try await outputConstraint(for: request)

        let toolSpecs = Self.toolSpecs(request.tools)
        let declaredTools = Set(request.tools?.map(\.name) ?? [])
        let scansToolCallText = profile.writesToolCallsAsText && request.tools?.isEmpty == false
        var makeInput: @Sendable (String) -> UserInput
        var template: String?
        let policy = Self.reasoningPolicy(request: request, defaultReasoningEffort: reasoningEffort)
        let turns = request.messages.flatMap { $0.isEmpty ? nil : profile.turns(controlTokens.neutralize($0), for: policy) }
        var lastText: String
        let context = Self.buildTemplateContext(
            request: request,
            defaultReasoningEffort: reasoningEffort,
            profile: profile
        )
        if turns == nil, request.prompt.contains("<|turn|>") || request.prompt.contains("<|channel|>") {
            makeInput = { UserInput(prompt: $0) }
            template = "raw"
            lastText = request.prompt
        } else {
            (makeInput, template, lastText) = Self.chatInput(
                turns ?? profile.turns(controlTokens.neutralize([ChatMessageInput(role: .user, text: request.prompt)]), for: policy),
                normalisation: "native", tools: toolSpecs, context: context)
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
            let assistantPrefix = carriesMoreThanTokens ? nil : request.assistantPrefix.flatMap { $0.isEmpty ? nil : $0 }
            let lmInput: LMInput
            if let assistantPrefix {
                let ids = await container.perform { $0.tokenizer.encode(text: assistantPrefix, addSpecialTokens: false) }
                lmInput = Self.appending(ids, to: rendered)
            } else {
                lmInput = rendered
            }
            let promptTokens = lmInput.text.tokens.asArray(Int.self)
            let namespace = carriesMoreThanTokens || epoch != modelEpoch ? nil : promptCacheNamespace(for: request)
            if let namespace {
                await restoreFromDisk(namespace, prompt: promptTokens, container: container, epoch: epoch)
            }
            let lease = namespace.flatMap { promptCaches.lease($0, for: promptTokens) }
            let boundary = namespace == nil
                ? promptTokens.count - 1
                : await snapshotBoundary(renderedCount, container: container, epoch: epoch, template: template, makeInput)
            let blend = epoch == modelEpoch && modelIsHybrid == false ? cacheBlend : nil
            let renderedTail = await container.perform {
                $0.tokenizer.decode(tokenIds: Array(promptTokens.prefix(renderedCount).suffix(16)), skipSpecialTokens: false)
            }
            let promptEndsWithThinking = Self.endsInsideThinking(renderedTail, assistantPrefix: assistantPrefix ?? "")
            let masked = Self.constraining(
                components, with: constraint,
                phase: policy == .disabled ? .constrained : promptEndsWithThinking ? .reasoning : .awaitingAnswer)
            let constrained = shaping.map { masked.appendingLogitProcessor($0) } ?? masked
            let forcedTool = constraint?.toolName
            let speculativeLookup = request.speculativeLookup ?? speculativeLookup
            let expertRouting = speculativeLookup ? await routesExperts(container, epoch: epoch) : true
            let zones = modelAttendsThroughZones ? (modelPrefillsInChunks ? kvZones : kvZones?.evicting(0)) : nil
            let zonedPrecision = zones == nil ? nil : request.kvPrecision ?? kvPrecision
            let launch: GenerationLaunch
            do {
                launch = try await container.perform(nonSendable: lmInput) { context, input in
                    try Self.launch(
                        input: input, lease: lease, keepsCache: namespace != nil, boundary: boundary,
                        blend: namespace.flatMap { space in blend.map { ($0, space) } },
                        params: params, components: constrained, tools: forcedTool == nil ? toolSpecs : nil,
                        context: context, speculativeLookup: speculativeLookup, speculativeRecycling: speculativeRecycling,
                        routesExperts: expertRouting, kvZones: zones, kvPrecision: zonedPrecision)
                }
            } catch {
                keep(nil, tokens: promptTokens, namespace: namespace, releasing: lease, epoch: epoch)
                throw error
            }
            let mlxStream = launch.stream

            return AsyncStream<StreamChunk> { continuation in
                let task = Task.detached {
                    let clock = ContinuousClock()
                    let startTime = clock.now
                    var lastInfo: GenerateCompletionInfo?
                    var tokenCount = 0
                    // Stamped on the first emitted chunk. Was hardcoded to 0, so every
                    // benchmark reported TTFT 0.000 with zero variance — a constant
                    // masquerading as a measurement, hiding all prefill cost.
                    var firstTokenAt: ContinuousClock.Instant?
                    
                    var buffer = ""
                    var isThinking = promptEndsWithThinking
                    var gate = ToolCallGate(request: request)
                    if let assistantPrefix { continuation.yield(.text(assistantPrefix)) }

                    var toolCallScanner = scansToolCallText ? NemotronToolCallScanner() : nil
                    var reasoningText = ""
                    var forcedToolArguments = ""
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

                    var trimsAnswerStart = false
                    func emitAnswer(_ text: String) {
                        guard trimsAnswerStart else { emitText(text); return }
                        let visible = String(text.drop(while: \.isNewline))
                        guard !visible.isEmpty else { return }
                        trimsAnswerStart = false
                        emitText(visible)
                    }

                    func processBuffer(final: Bool) {
                        while !buffer.isEmpty {
                            let boundary = isThinking
                                ? Self.firstMarker(Self.thinkingClosers, in: buffer)
                                : [Self.firstMarker(Self.thinkingOpeners, in: buffer), Self.firstMarker(Self.strayThinkingClosers, in: buffer)]
                                    .compactMap { $0 }.min { $0.lowerBound < $1.lowerBound }
                            guard let boundary else {
                                let held = final ? 0 : Self.thinkingMarkerHoldback
                                guard buffer.count > held else { break }
                                let cut = buffer.index(buffer.endIndex, offsetBy: -held)
                                let piece = String(buffer[..<cut])
                                buffer.removeSubrange(..<cut)
                                if isThinking { emitReasoning(piece) } else { emitAnswer(piece) }
                                break
                            }
                            let leading = String(buffer[..<boundary.lowerBound])
                            let opensThinking = !isThinking && Self.thinkingOpeners.contains(String(buffer[boundary]))
                            if !leading.isEmpty {
                                if isThinking { emitReasoning(leading) } else { emitAnswer(leading) }
                            }
                            buffer.removeSubrange(..<boundary.upperBound)
                            if buffer.hasPrefix("\n") { buffer.removeFirst() }
                            if isThinking, !opensThinking { trimsAnswerStart = true }
                            isThinking = opensThinking
                        }
                    }
                    
                    var stoppedAfterCall = false
                    var repetition = constraint == nil ? RepetitionGuard() : nil
                    var stoppedOnRepetition = false
                    var generations = mlxStream.makeAsyncIterator()
                    func stopAfterCall() async {
                        if let task = launch.task {
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
                            buffer.append(text)
                            processBuffer(final: false)
                            if gate.isSaturated { await stopAfterCall(); break tokens }
                            if repetition?.shouldStop(after: text) == true {
                                stoppedOnRepetition = true
                                await stopAfterCall()
                                break tokens
                            }
                        case .info(let info):
                            lastInfo = info
                        case .toolCall(let call):
                            processBuffer(final: true)
                            if let chunk = gate.admit(Self.toolCallOutput(call)) { continuation.yield(chunk) }
                            if gate.isSaturated { await stopAfterCall(); break tokens }
                        case .rejectedToolCall(let rejection):
                            processBuffer(final: true)
                            let chunk = Self.resolveRejectedToolCall(
                                reason: rejection.reason.rawValue, toolName: rejection.toolName,
                                callID: rejection.callID, rawText: rejection.rawTextPreview,
                                truncated: rejection.isPreviewTruncated, declaredTools: declaredTools)
                            if case .text = chunk {
                                GemLogger(module: "MLXInferenceEngine").warn("Tool call rejected: \(rejection.reason.rawValue)")
                            }
                            if let chunk = gate.admit(chunk) { continuation.yield(chunk) }
                            if gate.isSaturated { await stopAfterCall(); break tokens }
                        @unknown default:
                            break
                        }
                    }
                    
                    processBuffer(final: true)
                    admit(toolCallScanner?.finish() ?? [])
                    let arguments = forcedToolArguments.trimmingCharacters(in: .whitespacesAndNewlines)
                    if let forcedTool, (try? JSONSerialization.jsonObject(with: Data(arguments.utf8))) is [String: Any] {
                        admit([.toolCall(ToolCallOutput(
                            id: "call_\(UUID().uuidString.prefix(24))", name: forcedTool, arguments: arguments))])
                    } else if forcedTool != nil, !arguments.isEmpty {
                        continuation.yield(.text(arguments))
                    }
                    let completed = !(Task.isCancelled || stoppedAfterCall)
                    await Self.settle(launch, cancelled: !completed)
                    if completed, let session = constraint?.session { await self.recycle(session, epoch: epoch) }
                    await self.keep(launch.cache, tokens: Array(promptTokens.prefix(launch.exactTokens)), namespace: namespace, releasing: lease, epoch: epoch)

                    // End of stream - send metadata
                    if !Task.isCancelled {
                        let generationTime = (clock.now - startTime).inSeconds
                        let timeToFirstToken = firstTokenAt.map { ($0 - startTime).inSeconds } ?? 0
                        let mem = Memory.snapshot()
                        let reasoning = reasoningText
                        let reasoningTokens = reasoning.isEmpty ? nil : await container.perform {
                            $0.tokenizer.encode(text: reasoning, addSpecialTokens: false).count
                        }

                        let metadata = GenerationResponse(
                            generatedText: "", 
                            reasoningText: nil,
                            promptTokens: lastInfo.map { launch.reusedTokens + $0.promptTokenCount } ?? promptTokens.count,
                            cachedPromptTokens: launch.reusedTokens,
                            completionTokens: lastInfo?.generationTokenCount ?? tokenCount,
                            tokensPerSecond: lastInfo?.tokensPerSecond ?? 0,
                            generationTime: generationTime,
                            timeToFirstToken: timeToFirstToken,
                            memory: .init(
                                peakBytes: mem.peakMemory,
                                activeBytes: mem.activeMemory,
                                cacheBytes: mem.cacheMemory
                            ),
                            // Was hardcoded to .stop, so a generation cut off at the
                            // token ceiling reported a clean finish and truncation went
                            // unnoticed by every caller.
                            finishReason: stoppedOnRepetition ? .repetition
                                : (lastInfo?.generationTokenCount ?? 0) >= requestedMaxTokens ? .length : .stop,
                            reasoningTokens: reasoningTokens,
                            draftTokens: lastInfo?.speculativeDecodingTelemetry?.draftTokenCount,
                            acceptedDraftTokens: lastInfo?.speculativeDecodingTelemetry?.acceptedDraftTokenCount
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
        return components.appendingLogitProcessor {
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

    struct GenerationLaunch: Sendable {
        let stream: AsyncStream<Generation>
        let task: Task<Void, Never>?
        let cache: PromptCache?
        let reusedTokens: Int
        var exactTokens = Int.max
    }

    static func drainCompletionInfo(_ generations: inout AsyncStream<Generation>.Iterator) async -> GenerateCompletionInfo? {
        var info: GenerateCompletionInfo?
        while let generation = await generations.next() {
            if case .info(let completion) = generation { info = completion }
        }
        return info
    }

    static func settle(_ launch: GenerationLaunch, cancelled: Bool) async {
        if cancelled { launch.task?.cancel() }
        await launch.task?.value
    }

    private func routesExperts(_ container: ModelContainer, epoch: Int) async -> Bool {
        if epoch == modelEpoch, let known = modelRoutesExperts { return known }
        let routes = await container.perform { PromptLookupTokenIterator.routesExperts($0.model) }
        if epoch == modelEpoch {
            modelRoutesExperts = routes
        }
        return routes
    }

    private func snapshotBoundary(
        _ count: Int, container: ModelContainer, epoch: Int, template: String?,
        _ makeInput: @Sendable (String) -> UserInput
    ) async -> Int {
        let hybrid: Bool
        if let known = modelIsHybrid {
            hybrid = known
        } else {
            hybrid = (try? await container.perform { PromptCache(try $0.model.newCache(parameters: nil)).isHybrid })
                ?? false
            if epoch == modelEpoch {
                modelIsHybrid = hybrid
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
            guard ContinuousClock.now - started < deadline else {
                skipped += 1
                continue
            }
            guard let lease = promptCaches.lease(path.namespace, for: path.tokens + [0]) else { continue }
            let expectedWrite = Double(PromptCacheDiskStore.estimatedBytes(of: lease.cache, tokens: lease.reusedTokens))
                / PromptCacheDiskStore.assumedWriteBytesPerSecond
            guard ContinuousClock.now - started + .seconds(expectedWrite) <= deadline else {
                promptCaches.release(lease)
                skipped += 1
                continue
            }
            if lease.reusedTokens > 0, let name = promptCacheDisk?.store(
                fingerprint: fingerprint, model: modelDescription, namespace: path.namespace,
                tokens: Array(path.tokens.prefix(lease.reusedTokens)), cache: lease.cache, keeping: written) {
                written.insert(name)
            }
            promptCaches.release(lease)
        }
        logger.info("Persisted \(written.count) prompt-cache paths in \(String(format: "%.2f", (ContinuousClock.now - started).inSeconds)) s"
            + (skipped > 0 ? "; \(skipped) paths skipped that could not be written within the \(deadline) deadline" : ""))
    }

    private func keep(
        _ cache: PromptCache?, tokens: [Int], namespace: String?, releasing lease: PromptCacheLease?, epoch: Int
    ) {
        guard epoch == modelEpoch, let namespace else { return }
        if let lease {
            promptCaches.release(lease)
        }
        if let cache {
            promptCaches.store(namespace, tokens: Array(tokens.prefix(cache.tokenCount)), cache: cache)
        }
    }

    static func zonedCache(
        params: GenerateParameters, zones: KVZones?, precision: KVPrecision?, model: any LanguageModel
    ) throws -> (cache: [any KVCache], params: GenerateParameters)? {
        guard let zones, let precision else { return nil }
        var exact = params
        exact.kvBits = nil
        return (ZonedKVCache.replacingFullAttention(try model.newCache(parameters: exact), zones: zones, precision: precision), exact)
    }

    static func launch(
        input: LMInput, lease: PromptCacheLease?, keepsCache: Bool, boundary: Int,
        blend: (store: CacheBlendStore, namespace: String)? = nil,
        params: GenerateParameters, components: GenerationComponents,
        tools: [ToolSpec]?, context: ModelContext, speculativeLookup: Bool = false, speculativeRecycling: Bool = false,
        routesExperts: Bool = true, kvZones: KVZones? = nil,
        kvPrecision: KVPrecision? = nil
    ) throws -> GenerationLaunch {
        guard keepsCache else {
            if let zoned = try zonedCache(params: params, zones: kvZones, precision: kvPrecision, model: context.model) {
                let stream = try MLXLMCommon.generate(
                    input: input, cache: zoned.cache, parameters: zoned.params, context: context, components: components, tools: tools)
                return GenerationLaunch(stream: stream, task: nil, cache: nil, reusedTokens: 0)
            }
            let fresh = speculativeLookup && params.kvBits == nil && !carriesMoreThanTokens(input)
                ? try context.model.newCache(parameters: params) : []
            guard PromptLookupTokenIterator.supports(fresh, routesExperts: routesExperts) else {
                let stream = try MLXLMCommon.generate(
                    input: input, parameters: params, context: context, components: components, tools: tools)
                return GenerationLaunch(stream: stream, task: nil, cache: nil, reusedTokens: 0)
            }
            try components.validate(parameters: params)
            let iterator = try PromptLookupTokenIterator(
                input: input, context: input.text.tokens.asArray(Int.self), model: context.model, cache: fresh,
                processor: components.logitProcessor(parameters: params), sampler: params.sampler(),
                prefill: params.prefill, maxTokens: params.maxTokens, recyclesTokens: speculativeRecycling)
            let (stream, task) = generateTask(
                promptTokenCount: input.text.tokens.dim(-1), modelConfiguration: context.configuration,
                tokenizer: context.tokenizer, iterator: iterator, tools: tools, toolCallPolicy: params.toolCallPolicy)
            return GenerationLaunch(stream: stream, task: task, cache: nil, reusedTokens: 0)
        }
        try components.validate(parameters: params)
        var cache = try lease?.cache ?? PromptCache(context.model.newCache(parameters: params))
        let reused = lease?.reusedTokens ?? 0
        let blended = cache.isHybrid ? nil : blend.flatMap {
            $0.store.extend(
                cache, prompt: input.text.tokens.asArray(Int.self), from: reused, namespace: $0.namespace,
                model: context.model, tokenizer: context.tokenizer, prefillStep: params.prefill.stepSize ?? 512)
        }
        try Task.checkCancellation()
        if let blended {
            cache = blended.cache
        }
        if let blended, blended.reused > 0 {
            GemLogger(module: "CacheBlend").info(
                "Blended \(blended.tokens) prompt tokens: \(blended.reused) from cached chunks, \(blended.recomputed) of them recomputed")
        }
        let head = try blended.map { (kept: $0.cache, start: reused + $0.tokens, state: $0.cache.state) }
            ?? HybridPrefixSnapshot.prefillHead(
                input.text.tokens, into: cache, reused: reused, boundary: boundary, model: context.model, prefill: params.prefill)
        let tail = LMInput(tokens: input.text.tokens[.ellipsis, head.start...])
        let processor = components.logitProcessor(parameters: params).map {
            PromptAnchoredProcessor(base: $0, prompt: input.text.tokens)
        }
        func launching(_ iterator: some TokenIteratorProtocol) -> GenerationLaunch {
            let keepable = iterator.state == nil || cache.isHybrid
            let (stream, task) = generateTask(
                promptTokenCount: input.text.tokens.dim(-1) - reused, modelConfiguration: context.configuration,
                tokenizer: context.tokenizer, iterator: iterator, tools: tools, toolCallPolicy: params.toolCallPolicy)
            return GenerationLaunch(
                stream: stream, task: task, cache: keepable ? head.kept : nil, reusedTokens: reused,
                exactTokens: blended?.exactPrefix ?? Int.max)
        }
        if speculativeLookup, PromptLookupTokenIterator.supports(cache.layers, routesExperts: routesExperts) {
            return launching(try PromptLookupTokenIterator(
                input: tail, context: input.text.tokens.asArray(Int.self), model: context.model, cache: cache.layers,
                state: head.state, processor: processor, sampler: params.sampler(), prefill: params.prefill,
                maxTokens: params.maxTokens, recyclesTokens: speculativeRecycling))
        }
        return launching(try TokenIterator(
            input: tail, model: context.model, cache: cache.layers, state: head.state,
            processor: processor, sampler: params.sampler(), prefill: params.prefill, maxTokens: params.maxTokens))
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

struct PromptAnchoredProcessor: LogitProcessor {
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
}
