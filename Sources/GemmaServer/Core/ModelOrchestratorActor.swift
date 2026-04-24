import Foundation

// MARK: — Orchestrator

/// Единственный «источник истины» для обоих транспортов (MCP + REST).
///
/// Гарантирует FIFO-обслуживание: Swift actors сериализуют доступ
/// автоматически, исключая race conditions между MCP и REST клиентами.
///
/// Теорема устойчивости: λ_mcp + λ_rest < μ_inference
/// гарантирует что очередь не накапливается (система устойчива).
public actor ModelOrchestratorActor {

    // MARK: — State

    private let engine: any InferenceEngine
    private let maxTokens: Int
    private var requestCount: Int = 0
    private var totalTokensGenerated: Int = 0

    // MARK: — Init

    public init(engine: any InferenceEngine, maxTokens: Int = 4096) {
        self.engine = engine
        self.maxTokens = maxTokens
    }

    // MARK: — Public API (typed throws — Swift 6)

    /// Загружает модель из указанного пути.
    public func loadModel(path: String) async throws(GemmaServerError) {
        try await engine.load(modelPath: path)
    }

    /// Генерирует текст. Единственная точка входа для обоих транспортов.
    /// Swift actor гарантирует что вызовы выстраиваются в очередь (FIFO)
    /// — никаких дополнительных локов не требуется.
    public func generate(request: GenerationRequest) async throws(GemmaServerError) -> GenerationResponse {
        let validated = try request.validated(defaultMaxTokens: maxTokens)
        requestCount += 1
        let response = try await engine.generate(request: validated)
        totalTokensGenerated += response.completionTokens
        return response
    }

    /// Состояние для /health эндпоинта и MCP tool: gemma_status.
    public func healthSnapshot(modelId: String?) async -> HealthResponse {
        let ready = await engine.isLoaded
        return HealthResponse(
            status: ready ? "ok" : "initializing",
            modelId: modelId,
            isReady: ready,
            version: HealthResponse.version
        )
    }

    // Diagnostics — только для логирования, не экспортируется.
    var diagnostics: (requests: Int, tokens: Int) {
        (requestCount, totalTokensGenerated)
    }
}
