import Foundation
import MLX
import MLXLLM
// Registers MLXVLM's factory trampoline (Gemma 4, Qwen3.5-VL, …). The registry
// finds it via NSClassFromString, so the module must actually be linked in.
import MLXVLM
import MLXLMCommon
import MLXHuggingFace
import Tokenizers

private extension Duration {
    var inSeconds: Double {
        Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}

// MARK: — Protocol

public protocol InferenceEngine: Sendable {
    func load(modelPath: String) async throws(GemError)
    func generate(request: GenerationRequest) async throws(GemError) -> GenerationResponse
    func generateStream(request: GenerationRequest) async throws(GemError) -> AsyncStream<StreamChunk>
    /// async getter — compatible with actor-isolated implementations.
    var isLoaded: Bool { get async }
}

// MARK: — Real implementation

/// Actor-изолированный MLX inference engine.
/// Все Metal/MLX вызовы выполняются из одного контекста — нет гонок на GPU command queue.
public actor MLXInferenceEngine: InferenceEngine {

    private var container: ModelContainer?
    private let logger = GemLogger(module: "MLXInferenceEngine")

    /// Chat-template knob for models with a reasoning budget (Qwen3.8: xhigh | medium | low).
    /// Left nil the template picks its own default — for Qwen3.8 that is `xhigh`, which
    /// can spend the whole token budget thinking and never emit an answer.
    private let reasoningEffort: String?

    public init(reasoningEffort: String? = nil) {
        self.reasoningEffort = reasoningEffort
    }

    nonisolated(unsafe) private static var aliasesRegistered = false

    /// Register architecture aliases (Qwen 3.8, Qwen 3.6, MiniMax M2) in LLMTypeRegistry.
    public static func registerAliases() async {
        guard !aliasesRegistered else { return }
        aliasesRegistered = true

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

    public func load(modelPath: String) async throws(GemError) {
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
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw .modelNotFound(identifier: modelPath)
        }

        do {
            // Free the old model BEFORE loading the new one to avoid a peak-memory spike
            // where both models occupy GPU/unified memory simultaneously.
            // Setting container to nil drops all ARC references to the ModelContainer,
            // releasing the underlying MLXArray tensors and Metal buffers.
            if container != nil {
                container = nil
                Memory.clearCache()   // flush MLX's internal memory allocator cache
                logger.info("Previous model unloaded, GPU cache cleared.")
            }

            // Unbounded buffer cache competes with model weights for unified memory
            // and pushes large models (27B @ 4-bit ≈ 16 GB) into swap on 24 GB Macs.
            // ponytail: fixed 512 MB cap; make it a --gpu-cache flag if a workload needs more.
            Memory.cacheLimit = 512 << 20

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
                    let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
                    return TokenizerBridge(upstream)
                }
            }

            container = try await loadModelContainer(
                from: directory,
                using: DefaultTokenizerLoader()
            )
            
            // Dynamic config inspection
            let configURL = directory.appendingPathComponent("config.json")
            if let data = try? Data(contentsOf: configURL),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                
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
            throw .weightsCorrupted(path: modelPath, reason: error.localizedDescription)
        }
    }

    // MARK: — Generate

    public func generate(request: GenerationRequest) async throws(GemError) -> GenerationResponse {
        let stream = try await generateStream(request: request)
        var textChunks: [String] = []
        var reasoningChunks: [String] = []
        var finalResponse: GenerationResponse?

        for await chunk in stream {
            switch chunk {
            case .text(let t):
                textChunks.append(t)
            case .reasoning(let r):
                reasoningChunks.append(r)
            case .metadata(let m):
                finalResponse = m
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
            completionTokens: finalResponse.completionTokens,
            tokensPerSecond: finalResponse.tokensPerSecond,
            generationTime: finalResponse.generationTime,
            timeToFirstToken: finalResponse.timeToFirstToken,
            memory: finalResponse.memory,
            finishReason: finalResponse.finishReason
        )
    }

    public func generateStream(request: GenerationRequest) async throws(GemError) -> AsyncStream<StreamChunk> {
        guard let container else {
            throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first")
        }

        let requestedMaxTokens = request.maxTokens ?? 1024
        let params = GenerateParameters(
            maxTokens: requestedMaxTokens,
            temperature: Float(request.temperature ?? 0.7),
            topP: Float(request.topP ?? 0.9)
        )

        let userInput: UserInput
        if request.prompt.contains("<|turn|>") || request.prompt.contains("<|channel|>") {
            userInput = UserInput(prompt: request.prompt)
        } else {
            // Qwen3.x templates open a `<think>` block on every turn and default
            // reasoning_effort to `xhigh`, so an unqualified request spends its whole
            // token budget reasoning and gets cut off mid-answer. `none` reaches the
            // template's harder switch, which emits an already-closed think block and
            // makes the model answer directly.
            var context: [String: any Sendable] = [:]
            switch reasoningEffort {
            case "none":
                context["enable_thinking"] = false
            case .some(let effort):
                context["reasoning_effort"] = effort
            case nil:
                break
            }
            userInput = UserInput(
                chat: [.user(request.prompt)],
                additionalContext: context.isEmpty ? nil : context
            )
        }

        do {
            let lmInput = try await container.prepare(input: userInput)
            logger.trace("Input prepared. Starting MLX generation...")
            let mlxStream = try await container.generate(input: lmInput, parameters: params)
            
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
                    
                    var buffer = ""
                    var isThinking = promptEndsWithThinking

                    func processBuffer(final: Bool) {
                        while !buffer.isEmpty {
                            if !isThinking {
                                if let startRange = buffer.range(of: "<|channel|>thought") ?? buffer.range(of: "<think>") ?? buffer.range(of: "<|think|>") {
                                    let leading = String(buffer[..<startRange.lowerBound])
                                    if !leading.isEmpty {
                                        continuation.yield(.text(leading))
                                    }
                                    buffer.removeSubrange(..<startRange.upperBound)
                                    if buffer.hasPrefix("\n") { buffer.removeFirst() }
                                    isThinking = true
                                } else if let endRange = buffer.range(of: "<channel|>") ?? buffer.range(of: "</think>") {
                                    let leading = String(buffer[..<endRange.lowerBound])
                                    if !leading.isEmpty {
                                        continuation.yield(.text(leading))
                                    }
                                    buffer.removeSubrange(..<endRange.upperBound)
                                    if buffer.hasPrefix("\n") { buffer.removeFirst() }
                                } else {
                                    if !final && buffer.count > 15 {
                                        let safeEndIndex = buffer.index(buffer.endIndex, offsetBy: -15)
                                        let textToYield = String(buffer[..<safeEndIndex])
                                        buffer.removeSubrange(..<safeEndIndex)
                                        continuation.yield(.text(textToYield))
                                    } else if final {
                                        continuation.yield(.text(buffer))
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
                    
                    for await generation in mlxStream {
                        if Task.isCancelled { 
                            break 
                        }
                        
                        tokenCount += 1
                        
                        switch generation {
                        case .chunk(let text):
                            buffer.append(text)
                            processBuffer(final: false)
                        case .info(let info):
                            lastInfo = info
                        case .toolCall:
                            break
                        }
                    }
                    
                    // Flush any remaining text in buffer at EOF
                    processBuffer(final: true)

                    // End of stream - send metadata
                    if !Task.isCancelled {
                        let generationTime = (clock.now - startTime).inSeconds
                        let timeToFirstToken = 0.0
                        let mem = Memory.snapshot()

                        let metadata = GenerationResponse(
                            generatedText: "", 
                            reasoningText: nil,
                            promptTokens: lastInfo?.promptTokenCount ?? 0,
                            completionTokens: lastInfo?.generationTokenCount ?? 0,
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
        } catch let err as GemError {
            throw err
        } catch {
            throw .inferenceHardwareFailure(reason: error.localizedDescription)
        }
    }

    // MARK: — State

    public var isLoaded: Bool { container != nil }
}
