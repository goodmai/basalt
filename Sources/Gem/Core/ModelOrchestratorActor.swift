import Foundation
import os

// MARK: — Orchestrator

/// Single source of truth for both transports (MCP + REST).
///
/// Swift actor serialises all access (FIFO) — no explicit locking needed.
/// λ_mcp + λ_rest < μ_inference guarantees the queue stays bounded.
public actor ModelOrchestratorActor {

    // MARK: — State

    private let engine: any InferenceEngine
    private let maxTokens: Int
    private var requestCount: Int = 0
    private var totalTokensGenerated: Int = 0
    private var modelSizeMB: Int = 4000
    private let generationTimeoutSeconds: UInt64 = 300  // 5 minutes

    private let logger = GemLogger(module: "ModelOrchestrator")

    /// Filesystem path of the loaded model.
    private var currentModelPath: String = ""

    /// HuggingFace repo ID of the loaded model, e.g. "mlx-community/Qwen3.5-4B-4bit".
    /// Empty string means nothing has been loaded yet.
    public private(set) var currentModelId: String = ""

    // MARK: — Init

    public init(engine: any InferenceEngine, maxTokens: Int = 4096) {
        self.engine = engine
        self.maxTokens = maxTokens
    }

    // MARK: — Model loading

    /// Load a model from a filesystem path.
    /// After loading, call `setModelId(_:)` to record the HuggingFace repo ID.
    public func loadModel(path: String) async throws(GemError) {
        self.currentModelPath = path
        let size = try? calculateDirectorySize(url: URL(fileURLWithPath: path))
        if let size {
            self.modelSizeMB = Int(size / 1_048_576)
            logger.info("Model size: \(self.modelSizeMB) MB")
        }
        try await engine.load(modelPath: path)
    }

    /// Record the HuggingFace repo ID that was just loaded.
    /// Call this once after `loadModel` completes.
    public func setModelId(_ id: String) {
        self.currentModelId = id
    }

    // MARK: — Model switching

    /// Hot-swap the loaded model. Resolves `modelId` from the local HuggingFace cache.
    /// Throws `.modelNotCached` if the model hasn't been downloaded yet.
    /// Because this is an actor method it serialises with all in-flight generate calls —
    /// any concurrent request will queue and run after the switch completes.
    public func switchModel(to modelId: String) async throws(GemError) {
        guard modelId != currentModelId else {
            logger.trace("switchModel: already loaded '\(modelId)'")
            return
        }

        let path = ModelCache.cacheDir(for: modelId).path
        guard FileManager.default.fileExists(atPath: path) else {
            logger.error("switchModel: model not in cache: \(modelId)")
            throw GemError.modelNotCached(identifier: modelId)
        }

        logger.info("Switching model: '\(currentModelId)' → '\(modelId)'")
        try await loadModel(path: path)
        self.currentModelId = modelId
        logger.info("Model switch complete: '\(modelId)'")
    }

    /// Info string for logging / health checks.
    public var modelInfo: String {
        currentModelId.isEmpty
            ? URL(fileURLWithPath: currentModelPath).lastPathComponent
            : currentModelId
    }

    // MARK: — Generate (non-streaming)

    /// Generate a completion. If `modelId` differs from the currently loaded model,
    /// the model is switched automatically before generation.
    public func generate(
        request: GenerationRequest,
        modelId: String? = nil
    ) async throws(GemError) -> GenerationResponse {
        try await autoSwitch(to: modelId)

        logger.trace("generate: \(request.prompt.prefix(60))…")
        let validated = try capped(request)
        requestCount += 1

        do {
            let response = try await withTimeout(seconds: generationTimeoutSeconds) {
                try await self.engine.generate(request: validated)
            }
            totalTokensGenerated += response.completionTokens
            return response
        } catch let error as GemError {
            throw error
        } catch {
            throw GemError.inferenceError("Generation error: \(error)")
        }
    }

    // MARK: — Generate (streaming)

    /// Returns a filtered AsyncStream. Strips `<think>…</think>` blocks.
    /// If `modelId` differs from the currently loaded model, switches first.
    public func generateStream(
        request: GenerationRequest,
        modelId: String? = nil
    ) async throws(GemError) -> AsyncStream<StreamChunk> {
        try await autoSwitch(to: modelId)

        logger.trace("generateStream start")
        let validated = try capped(request)
        requestCount += 1

        let stream: AsyncStream<StreamChunk>
        do {
            stream = try await withTimeout(seconds: generationTimeoutSeconds) {
                try await self.engine.generateStream(request: validated)
            }
        } catch let error as GemError {
            throw error
        } catch {
            throw GemError.inferenceError("Stream setup error: \(error)")
        }

        return AsyncStream<StreamChunk> { continuation in
            let task = Task.detached {
                for await chunk in stream {
                    switch chunk {
                    case .text(let t):
                        continuation.yield(.text(t))
                    case .reasoning(let r):
                        continuation.yield(.reasoning(r))
                    case .metadata(let m):
                        continuation.yield(.metadata(m))
                    }
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: — Health

    public func healthSnapshot(modelId: String?) async -> HealthResponse {
        let ready = await engine.isLoaded
        return HealthResponse(
            status: ready ? "ok" : "initializing",
            modelId: modelId ?? (currentModelId.isEmpty ? nil : currentModelId),
            isReady: ready,
            version: HealthResponse.version
        )
    }

    // MARK: — Model catalogue

    /// Returns all models currently in the local HuggingFace cache.
    public func cachedModels() -> [CachedModelInfo] {
        let root = ModelCache.root
        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: root.path) else {
            return []
        }

        return entries
            .filter { $0.hasPrefix("models--") }
            .sorted()
            .compactMap { dir -> CachedModelInfo? in
                let repoId = dir
                    .replacingOccurrences(of: "models--", with: "")
                    .replacingOccurrences(of: "--", with: "/")
                let url  = root.appendingPathComponent(dir)
                let size = directorySize(at: url)
                return CachedModelInfo(
                    id: repoId,
                    sizeBytes: size,
                    isLoaded: repoId == currentModelId
                )
            }
    }

    // MARK: — Diagnostics

    var diagnostics: (requests: Int, tokens: Int) {
        (requestCount, totalTokensGenerated)
    }

    // MARK: — Private helpers

    /// Switch model if the requested ID differs from the loaded one.
    /// Ignores nil, empty, and generic placeholder IDs like "gemm".
    private func autoSwitch(to modelId: String?) async throws(GemError) {
        guard let id = modelId, !id.isEmpty, id != "gemm" else { return }
        if await engine.isLoaded { return }
        guard id != currentModelId else { return }
        try await switchModel(to: id)
    }

    private func capped(_ request: GenerationRequest) throws(GemError) -> GenerationRequest {
        let maxT = calculateDynamicMaxTokens()
        var v = try request.validated(defaultMaxTokens: maxT)
        if let req = request.maxTokens, req > maxT {
            logger.warn("Token request (\(req)) > budget (\(maxT)). Capping.")
            v = GenerationRequest(prompt: v.prompt, maxTokens: maxT, temperature: v.temperature)
        }
        return v
    }

    private func calculateDynamicMaxTokens() -> Int {
        max(maxTokens, TokenBudgetCalculator.calculateMaxTokensForSystem(modelSizeMB: modelSizeMB))
    }

    private func calculateDirectorySize(url: URL) throws -> UInt64 {
        var total: UInt64 = 0
        guard let e = FileManager.default.enumerator(at: url,
                        includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        for case let f as URL in e {
            if let s = try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                total += UInt64(s)
            }
        }
        return total
    }

    private func directorySize(at url: URL) -> Int64 {
        guard let e = FileManager.default.enumerator(at: url,
                        includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let f as URL in e {
            total += (try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize)
                        .map { Int64($0) } ?? 0
        }
        return total
    }

    private func withTimeout<T: Sendable>(
        seconds: UInt64,
        operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        let ns = seconds * 1_000_000_000
        return try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask { try await operation() }
            group.addTask {
                try await Task.sleep(nanoseconds: ns)
                try Task.checkCancellation()
                throw GemError.inferenceError("Timeout after \(seconds)s")
            }
            guard let result = try await group.next() else {
                throw GemError.inferenceError("Operation produced no result")
            }
            group.cancelAll()
            return result
        }
    }
}

// MARK: — CachedModelInfo DTO

public struct CachedModelInfo: Sendable {
    public let id: String
    public let sizeBytes: Int64
    public let isLoaded: Bool

    public var sizeFormatted: String {
        switch sizeBytes {
        case 0..<1_048_576:    return String(format: "%.1f KB", Double(sizeBytes) / 1_024)
        case 0..<1_073_741_824: return String(format: "%.1f MB", Double(sizeBytes) / 1_048_576)
        default:               return String(format: "%.2f GB", Double(sizeBytes) / 1_073_741_824)
        }
    }
}
