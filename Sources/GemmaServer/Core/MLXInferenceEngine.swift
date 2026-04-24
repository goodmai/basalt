import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import Tokenizers

// MARK: — Protocol

public protocol InferenceEngine: Sendable {
    func load(modelPath: String) async throws(GemmaServerError)
    func generate(request: GenerationRequest) async throws(GemmaServerError) -> GenerationResponse
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

    public func load(modelPath: String) async throws(GemmaServerError) {
        // ServeCommand всегда передаёт абсолютный локальный путь после разрешения кэша.
        // Если путь не начинается с "/" — это нераспознанный формат.
        guard modelPath.hasPrefix("/") || modelPath.hasPrefix(".") else {
            // HF repo ID без локального кэша — сообщаем пользователю
            throw .modelNotFound(
                identifier: "'\(modelPath)' не найден в локальном кэше. " +
                "Сначала загрузите: GemmaServer models download \(modelPath)"
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

    public func generate(request: GenerationRequest) async throws(GemmaServerError) -> GenerationResponse {
        guard let container else {
            throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first")
        }

        let params = GenerateParameters(
            maxTokens: request.maxTokens ?? 1024,
            temperature: Float(request.temperature ?? 0.7),
            topP: Float(request.topP ?? 0.9)
        )

        // Используем chat-формат чтобы применить chat template модели корректно
        let userInput = UserInput(chat: [.user(request.prompt)])

        do {
            // prepare() конвертирует UserInput → LMInput через processor модели
            let lmInput = try await container.prepare(input: userInput)

            let clock = ContinuousClock()
            let startTime = clock.now

            // generate() возвращает AsyncStream<Generation> (v3.31.3 API)
            let stream = try await container.generate(input: lmInput, parameters: params)

            var outputText = ""
            var completionInfo: GenerateCompletionInfo?

            for await generation in stream {
                switch generation {
                case .chunk(let text):
                    outputText += text
                case .info(let info):
                    completionInfo = info
                case .toolCall:
                    break   // не используем tool calling в этом пути
                }
            }

            let duration = clock.now - startTime
            let generationTime = Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18

            return GenerationResponse(
                generatedText: outputText,
                promptTokens: completionInfo?.promptTokenCount ?? 0,
                completionTokens: completionInfo?.generationTokenCount ?? 0,
                tokensPerSecond: completionInfo?.tokensPerSecond ?? 0,
                generationTime: generationTime,
                finishReason: .stop
            )
        } catch let err as GemmaServerError {
            throw err
        } catch {
            throw .inferenceHardwareFailure(reason: error.localizedDescription)
        }
    }

    // MARK: — State

    public var isLoaded: Bool { container != nil }
}
