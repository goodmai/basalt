import Foundation
import Synchronization
import MLX
import MLXLLM
// Registers MLXVLM's factory trampoline (Gemma 4, Qwen3.5-VL, …). The registry
// finds it via NSClassFromString, so the module must actually be linked in.
import MLXVLM
import MLXLMCommon
import MLXHuggingFace
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
    private let logger = GemLogger(module: "MLXInferenceEngine")

    /// Chat-template knob for models with a reasoning budget (Qwen3.8: xhigh | medium | low).
    /// Left nil the template picks its own default — for Qwen3.8 that is `xhigh`, which
    /// can spend the whole token budget thinking and never emit an answer.
    private let reasoningEffort: String?

    /// Bits for KV-cache quantization (4 or 8). nil keeps the cache in full
    /// precision. The cache grows linearly with context, so on long prompts it
    /// competes with the weights for unified memory.
    private let kvBits: Int?

    /// Token index at which the quantized cache kicks in. Quantizing from token 0
    /// costs accuracy on short prompts for no memory benefit.
    private let quantizedKVStart: Int

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
    private let configuredPromptCacheBytes: Int?
    private var modelEpoch = 0
    private var modelIsHybrid: Bool?
    private var templateSuffixes: [String: Int] = [:]
    private let cacheBlendRatio: Float?
    private(set) var cacheBlend: CacheBlendStore?

    public init(
        reasoningEffort: String? = nil,
        gpuCacheLimit: Int? = 512 << 20,
        promptCacheBytes: Int? = nil,
        kvBits: Int? = nil,
        quantizedKVStart: Int = 512,
        repetitionPenalty: Float? = nil,
        topK: Int? = nil,
        minP: Float? = nil,
        seed: UInt64? = nil,
        cacheBlendRatio: Float? = nil
    ) {
        self.reasoningEffort = reasoningEffort
        self.gpuCacheLimit = gpuCacheLimit
        self.kvBits = kvBits
        self.quantizedKVStart = quantizedKVStart
        self.repetitionPenalty = repetitionPenalty
        self.topK = topK
        self.minP = minP
        self.seed = seed
        self.configuredPromptCacheBytes = promptCacheBytes
        self.cacheBlendRatio = cacheBlendRatio
        self.promptCaches = PromptCacheTree(byteBudget: promptCacheBytes ?? 0)
    }

    func attach(_ container: ModelContainer) {
        self.container = container
        let budget = configuredPromptCacheBytes ?? PromptCacheBudget.automatic(
            workingSet: PromptCacheBudget.gpuWorkingSet(), weights: Memory.activeMemory)
        let blendBytes = cacheBlendRatio == nil ? 0 : budget / 2
        promptCaches = PromptCacheTree(byteBudget: budget - blendBytes)
        cacheBlend = cacheBlendRatio.map { CacheBlendStore(ratio: $0, byteBudget: blendBytes) }
        modelEpoch += 1
        modelIsHybrid = nil
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

            attach(try await loadModelContainer(from: directory, using: DefaultTokenizerLoader()))
            profile = ModelProfileRegistry.profile(forModelAt: modelPath)
            
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
        let compatible = ChatTemplateCompatibility.renamingReservedLoopVariables(template)
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
            toolCalls: toolCalls.isEmpty ? nil : toolCalls
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

    func sampling(for request: GenerationRequest) -> (GenerateParameters, GenerationComponents) {
        let temperature = Float(request.temperature ?? 0.7)
        var params = GenerateParameters(
            maxTokens: request.maxTokens ?? 1024,
            temperature: temperature,
            topP: Float(request.topP ?? 0.9)
        )
        params.kvBits = kvBits
        params.quantizedKVStart = quantizedKVStart
        params.repetitionPenalty = repetitionPenalty
        if let topK { params.topK = topK }
        if let effectiveMinP = request.minP.map(Float.init) ?? minP ?? profile.defaultMinP {
            params.minP = effectiveMinP
        }
        params.seed = seed

        var components = GenerationComponents()
        if request.dynamicTemperature == true, temperature > 0 {
            let (topP, minP) = (params.topP, params.minP)
            params.topP = 1
            params.minP = 0
            components = components.appendingLogitProcessor {
                EntropyDynamicTemperature(baseTemperature: temperature, topP: topP, minP: minP)
            }
        }
        return (params, components)
    }

    public func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk> {
        guard let container else {
            throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first")
        }
        let epoch = modelEpoch

        let requestedMaxTokens = request.maxTokens ?? 1024
        let (params, components) = sampling(for: request)

        let toolSpecs = Self.toolSpecs(request.tools)
        let declaredTools = Set(request.tools?.map(\.name) ?? [])
        let scansToolCallText = profile.writesToolCallsAsText && request.tools?.isEmpty == false
        var makeInput: @Sendable (String) -> UserInput
        var template: String?
        let policy = Self.reasoningPolicy(request: request, defaultReasoningEffort: reasoningEffort)
        let turns = request.messages.flatMap { $0.isEmpty ? nil : profile.turns($0, for: policy) }
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
                turns ?? profile.turns([ChatMessageInput(role: .user, text: request.prompt)], for: policy),
                normalisation: "native", tools: toolSpecs, context: context)
        }

        do {
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
            let carriesMoreThanTokens = rendered.image != nil || rendered.video != nil
                || rendered.audio != nil || rendered.text.mask != nil
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
            let lease = namespace.flatMap { promptCaches.lease($0, for: promptTokens) }
            let boundary = namespace == nil
                ? promptTokens.count - 1
                : await snapshotBoundary(renderedCount, container: container, epoch: epoch, template: template, makeInput)
            let blend = epoch == modelEpoch && modelIsHybrid == false ? cacheBlend : nil
            let launch: GenerationLaunch
            do {
                launch = try await container.perform(nonSendable: lmInput) { context, input in
                    try Self.launch(
                        input: input, lease: lease, keepsCache: namespace != nil, boundary: boundary,
                        blend: namespace.flatMap { space in blend.map { ($0, space) } },
                        params: params, components: components, tools: toolSpecs, context: context)
                }
            } catch {
                keep(nil, tokens: promptTokens, namespace: namespace, releasing: lease, epoch: epoch)
                throw error
            }
            let mlxStream = launch.stream
            
            let promptEndsWithThinking = request.prompt.hasSuffix("<|channel>thought\n")
                || request.prompt.hasSuffix("<|channel>thought")
                || request.prompt.hasSuffix("<|channel|>thought\n")
                || request.prompt.hasSuffix("<|channel|>thought")
                || request.prompt.hasSuffix("<think>\n")
                || request.prompt.hasSuffix("<think>")

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
                    func admit(_ chunks: [StreamChunk]) {
                        for chunk in gate.admit(chunks) { continuation.yield(chunk) }
                    }
                    func emitText(_ text: String) {
                        guard var scanner = toolCallScanner else { continuation.yield(.text(text)); return }
                        admit(scanner.feed(text))
                        toolCallScanner = scanner
                    }

                    func processBuffer(final: Bool) {
                        while !buffer.isEmpty {
                            if !isThinking {
                                if let startRange = buffer.range(of: "<|channel|>thought") ?? buffer.range(of: "<think>") ?? buffer.range(of: "<|think|>") {
                                    let leading = String(buffer[..<startRange.lowerBound])
                                    if !leading.isEmpty {
                                        emitText(leading)
                                    }
                                    buffer.removeSubrange(..<startRange.upperBound)
                                    if buffer.hasPrefix("\n") { buffer.removeFirst() }
                                    isThinking = true
                                } else if let endRange = buffer.range(of: "<channel|>") ?? buffer.range(of: "</think>") {
                                    let leading = String(buffer[..<endRange.lowerBound])
                                    if !leading.isEmpty {
                                        emitText(leading)
                                    }
                                    buffer.removeSubrange(..<endRange.upperBound)
                                    if buffer.hasPrefix("\n") { buffer.removeFirst() }
                                } else {
                                    if !final && buffer.count > 15 {
                                        let safeEndIndex = buffer.index(buffer.endIndex, offsetBy: -15)
                                        let textToYield = String(buffer[..<safeEndIndex])
                                        buffer.removeSubrange(..<safeEndIndex)
                                        emitText(textToYield)
                                    } else if final {
                                        emitText(buffer)
                                        buffer = ""
                                    }
                                    break
                                }
                            } else {
                                if let endRange = buffer.range(of: "<|channel|>") ?? buffer.range(of: "<channel|>") ?? buffer.range(of: "</think>") ?? buffer.range(of: "<|turn|>") {
                                    let reasoning = String(buffer[..<endRange.lowerBound])
                                    if !reasoning.isEmpty {
                                        continuation.yield(.reasoning(reasoning))
                                    }
                                    buffer.removeSubrange(..<endRange.upperBound)
                                    if buffer.hasPrefix("\n") { buffer.removeFirst() }
                                    isThinking = false
                                } else {
                                    if !final && buffer.count > 15 {
                                        let safeEndIndex = buffer.index(buffer.endIndex, offsetBy: -15)
                                        let reasoningToYield = String(buffer[..<safeEndIndex])
                                        buffer.removeSubrange(..<safeEndIndex)
                                        continuation.yield(.reasoning(reasoningToYield))
                                    } else if final {
                                        if !buffer.isEmpty {
                                            continuation.yield(.reasoning(buffer))
                                            buffer = ""
                                        }
                                    }
                                    break
                                }
                            }
                        }
                    }
                    
                    var stoppedAfterCall = false
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
                    await Self.settle(launch, cancelled: Task.isCancelled || stoppedAfterCall)
                    await self.keep(launch.cache, tokens: Array(promptTokens.prefix(launch.exactTokens)), namespace: namespace, releasing: lease, epoch: epoch)

                    // End of stream - send metadata
                    if !Task.isCancelled {
                        let generationTime = (clock.now - startTime).inSeconds
                        let timeToFirstToken = firstTokenAt.map { ($0 - startTime).inSeconds } ?? 0
                        let mem = Memory.snapshot()

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
                            finishReason: (lastInfo?.generationTokenCount ?? 0) >= requestedMaxTokens
                                ? .length : .stop
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
        guard promptCaches.byteBudget > 0, kvBits == nil else { return nil }
        return request.cacheKey
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

    static func launch(
        input: LMInput, lease: PromptCacheLease?, keepsCache: Bool, boundary: Int,
        blend: (store: CacheBlendStore, namespace: String)? = nil,
        params: GenerateParameters, components: GenerationComponents,
        tools: [ToolSpec]?, context: ModelContext
    ) throws -> GenerationLaunch {
        guard keepsCache else {
            let stream = try MLXLMCommon.generate(
                input: input, parameters: params, context: context, components: components, tools: tools)
            return GenerationLaunch(stream: stream, task: nil, cache: nil, reusedTokens: 0)
        }
        try components.validate(parameters: params)
        var cache = try lease?.cache ?? PromptCache(context.model.newCache(parameters: params))
        let reused = lease?.reusedTokens ?? 0
        let blended = cache.isHybrid ? nil : blend.flatMap {
            $0.store.extend(
                cache, prompt: input.text.tokens.asArray(Int.self), from: reused, namespace: $0.namespace,
                model: context.model, tokenizer: context.tokenizer, prefillStep: params.prefill.stepSize ?? 512)
        }
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
        let iterator = try TokenIterator(
            input: tail, model: context.model, cache: cache.layers, state: head.state,
            processor: components.logitProcessor(parameters: params).map {
                PromptAnchoredProcessor(base: $0, prompt: input.text.tokens)
            },
            sampler: params.sampler(), prefill: params.prefill, maxTokens: params.maxTokens)
        let keepable = iterator.state == nil || cache.isHybrid
        let (stream, task) = generateTask(
            promptTokenCount: input.text.tokens.dim(-1) - reused, modelConfiguration: context.configuration,
            tokenizer: context.tokenizer, iterator: iterator, tools: tools, toolCallPolicy: params.toolCallPolicy)
        return GenerationLaunch(
            stream: stream, task: task, cache: keepable ? head.kept : nil, reusedTokens: reused,
            exactTokens: blended?.exactPrefix ?? Int.max)
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
