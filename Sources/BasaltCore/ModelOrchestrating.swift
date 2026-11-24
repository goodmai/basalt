import Foundation

public struct OrchestratorKVOptions: Sendable {
    let precision: KVPrecision?
    let adaptive: Bool
    let zones: KVZones?

    public init(precision: KVPrecision? = nil, adaptive: Bool = false, zones: KVZones? = nil) {
        self.precision = precision
        self.adaptive = adaptive
        self.zones = zones
    }
}

public protocol ModelOrchestrating: Actor {
    nonisolated var gate: GenerationGate { get }
    nonisolated var sessionLog: SessionLogWriter? { get }
    nonisolated var telemetry: TelemetryTracker { get }
    var currentModelId: String { get }
    var modelPath: String { get }
    var modelInfo: String { get }
    var isLoaded: Bool { get async }
    var adapterNames: [String] { get async }
    var ropeStatus: ModelRoPEStatus { get async }
    var transcribesAudio: Bool { get async }
    var readsDocuments: Bool { get }

    func configureCloudFallback(
        _ fallback: (any CloudFallback)?, arbiter: HybridArbiter, freeMemory: @escaping @Sendable () -> Int64)
    func setFreeMemoryProbe(_ probe: @escaping @Sendable () -> Int64)
    func setModelId(_ id: String)
    func setAdaptiveThinkingBudget(_ enabled: Bool)
    func setDocumentTranscriber(_ transcriber: (any DocumentTranscriber)?, imageFallback: Bool)
    func setSpeechFallback(_ fallback: (any AudioTranscriber)?)
    func loadModel(path: String) async throws(BasaltError)
    func unloadModel() async
    func switchModel(to modelId: String) async throws(BasaltError)
    func loadAdapters(_ directories: [String: URL]) async throws(BasaltError)
    func shutdown(deadline: Duration) async
    func healthSnapshot(modelId: String?) async -> HealthResponse
    func telemetrySnapshot() async -> ServerTelemetrySnapshot
    func cachedModels() -> [CachedModelInfo]
    func admissionRefusal(for request: GenerationRequest) async -> BasaltError?
    func effectiveMaxTokens(for request: GenerationRequest) async -> Int
    func coordinateMemory(with coordinator: MemoryCoordinator) async
    func coordinateThermal(with source: @escaping @Sendable () -> ThermalLevel) async
    func generate(request: GenerationRequest, modelId: String?) async throws(BasaltError) -> GenerationResponse
    func generateStream(request: GenerationRequest, modelId: String?) async throws(BasaltError) -> AsyncStream<StreamChunk>
    @concurrent nonisolated func transcribingImages(
        in request: GenerationRequest, modelId: String?
    ) async throws(BasaltError) -> (request: GenerationRequest, images: Int)
    @concurrent nonisolated func ocr(_ request: OCRRequest) async throws(BasaltError) -> OCRResult
    func transcribe(_ request: TranscriptionRequest) async throws(BasaltError) -> Transcription
    func research<T: Sendable>(
        _ body: @Sendable (any ModelResearch) async throws(BasaltError) -> T
    ) async throws(BasaltError) -> T
    func preemptibly<T: Sendable>(
        _ body: @Sendable (any ModelLab, StepLease) async throws(BasaltError) -> T
    ) async throws(BasaltError) -> T
    func exclusively<T: Sendable>(
        unloadingModel unloads: Bool, _ body: @escaping @Sendable () async throws(BasaltError) -> T
    ) async throws(BasaltError) -> (value: T, swap: Duration)
    func sharedImagePromptEmbedding(_ prompt: String, dimension: Int) async throws -> [Float]?
}

extension ModelOrchestrating {
    public var ropeStatus: ModelRoPEStatus {
        get async { .empty }
    }

    public func sharedImagePromptEmbedding(_ prompt: String, dimension: Int) async throws -> [Float]? {
        nil
    }
    public func setCloudFallback(
        _ fallback: (any CloudFallback)?, arbiter: HybridArbiter = HybridArbiter(),
        freeMemory: @escaping @Sendable () -> Int64 = { SystemProfiler.availableSystemMemory() }
    ) {
        configureCloudFallback(fallback, arbiter: arbiter, freeMemory: freeMemory)
    }

    public func setDocumentTranscriber(_ transcriber: (any DocumentTranscriber)?) {
        setDocumentTranscriber(transcriber, imageFallback: false)
    }

    public func shutdown() async {
        await shutdown(deadline: .seconds(5))
    }

    public func coordinateThermal() async {
        await coordinateThermal(with: ThermalLevel.current)
    }

    public func generate(request: GenerationRequest) async throws(BasaltError) -> GenerationResponse {
        try await generate(request: request, modelId: nil)
    }

    public func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk> {
        try await generateStream(request: request, modelId: nil)
    }

    @concurrent
    public nonisolated func transcribingImages(
        in request: GenerationRequest
    ) async throws(BasaltError) -> (request: GenerationRequest, images: Int) {
        try await transcribingImages(in: request, modelId: nil)
    }
}
