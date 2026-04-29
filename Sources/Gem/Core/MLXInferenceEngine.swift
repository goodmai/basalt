import Foundation
import MLX
import MLXLLM
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

    public init() {}

    // MARK: — Load

    public func load(modelPath: String) async throws(GemError) {
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
            // #huggingFaceTokenizerLoader() — MLXHuggingFace macro, разворачивается в
            // TokenizerLoader использующий Tokenizers.AutoTokenizer.from(pretrained:)
            container = try await loadModelContainer(
                from: directory,
                using: #huggingFaceTokenizerLoader()
            )
        } catch {
            throw .weightsCorrupted(path: modelPath, reason: error.localizedDescription)
        }
    }

    // MARK: — Generate

    public func generate(request: GenerationRequest) async throws(GemError) -> GenerationResponse {
        let stream = try await generateStream(request: request)
        var chunks: [String] = []
        var finalResponse: GenerationResponse?

        for await chunk in stream {
            switch chunk {
            case .text(let t):
                chunks.append(t)
            case .metadata(let m):
                finalResponse = m
            }
        }

        guard let finalResponse else {
            throw .inferenceHardwareFailure(reason: "Stream finished without metadata")
        }

        return GenerationResponse(
            generatedText: chunks.joined(),
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

        let params = GenerateParameters(
            maxTokens: request.maxTokens ?? 1024,
            temperature: Float(request.temperature ?? 0.7),
            topP: Float(request.topP ?? 0.9)
        )

        let userInput = UserInput(chat: [.user(request.prompt)])

        do {
            let lmInput = try await container.prepare(input: userInput)
            let mlxStream = try await container.generate(input: lmInput, parameters: params)
            
            return AsyncStream<StreamChunk> { continuation in
                let task = Task {
                    let clock = ContinuousClock()
                    let startTime = clock.now
                    var firstTokenTime: ContinuousClock.Instant?
                    var lastInfo: GenerateCompletionInfo?
                    
                    for await generation in mlxStream {
                        if Task.isCancelled { break }

                        if firstTokenTime == nil {
                            firstTokenTime = clock.now
                        }
                        
                        switch generation {
                        case .chunk(let text):
                            continuation.yield(.text(text))
                        case .info(let info):
                            lastInfo = info
                        case .toolCall:
                            break
                        }
                    }
                    
                    // End of stream - send metadata
                    if !Task.isCancelled {
                        let generationTime = (clock.now - startTime).inSeconds
                        let timeToFirstToken = ((firstTokenTime ?? clock.now) - startTime).inSeconds
                        let mem = Memory.snapshot()

                        let metadata = GenerationResponse(
                            generatedText: "", 
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
                            finishReason: .stop
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
