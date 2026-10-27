import Foundation
import os

// MARK: — Orchestrator

public actor ModelOrchestratorActor {

    // MARK: — State

    private let engine: any InferenceEngine
    private let maxTokens: Int
    private var modelSizeMB: Int = 4000
    private let kvBits: Int?
    private var kvGeometry = KVCacheGeometry(parameterProxy: 4)
    /// Floor for the generation watchdog. The real budget scales with the token
    /// count — see `timeout(for:)`.
    private let generationTimeoutSeconds: UInt64 = 300  // 5 minutes

    private let logger = GemLogger(module: "ModelOrchestrator")

    /// Filesystem path of the loaded model.
    private var currentModelPath: String = ""

    /// HuggingFace repo ID of the loaded model, e.g. "mlx-community/Qwen3.5-4B-4bit".
    /// Empty string means nothing has been loaded yet.
    public private(set) var currentModelId: String = ""

    public let gate: GenerationGate
    public let sessionLog: SessionLogWriter?
    private var pendingSwitch: (modelId: String, task: Task<Void, Error>)?
    private var cloudFallback: (any CloudFallback)?
    private var arbiter = HybridArbiter()
    private var freeMemory: @Sendable () -> Int64 = { SystemProfiler.availableSystemMemory() }

    // MARK: — Init

    /// The serving process's counters; injectable so tests do not share global state.
    public let telemetry: TelemetryTracker

    public init(
        engine: any InferenceEngine, maxTokens: Int = 65_536, kvBits: Int? = nil, telemetry: TelemetryTracker = .shared,
        gate: GenerationGate = GenerationGate(), sessionLog: SessionLogWriter? = nil
    ) {
        self.gate = gate
        self.sessionLog = sessionLog
        self.engine = engine
        self.maxTokens = maxTokens
        self.kvBits = kvBits
        self.telemetry = telemetry
    }

    public func setCloudFallback(
        _ fallback: (any CloudFallback)?, arbiter: HybridArbiter = HybridArbiter(),
        freeMemory: @escaping @Sendable () -> Int64 = { SystemProfiler.availableSystemMemory() }
    ) {
        self.cloudFallback = fallback
        self.arbiter = arbiter
        self.freeMemory = freeMemory
    }

    func cloudFallbackReason(for request: GenerationRequest) async -> CloudFallbackReason? {
        guard let cloudFallback, request.isServiceRequest != true,
              let assessment = await cloudFallback.assess(request)
        else { return nil }
        let promptTokens = Self.estimatedTokens(Self.budgetedText(request))
        let required = kvGeometry.bytes(tokens: promptTokens + (request.maxTokens ?? 0), kvBits: kvBits)
        let localTPS = await telemetry.snapshot().tokensPerSecond
        let local = HybridCandidate(
            capability: 1 - min(max(assessment.capabilityGap, 0), 1),
            degradation: arbiter.localDegradation(contextTokens: promptTokens),
            latencySeconds: localTPS > 0 ? Double(request.maxTokens ?? maxTokens) / localTPS : 0,
            cost: 0)
        let cloud = HybridCandidate(
            capability: 1, degradation: 0, latencySeconds: assessment.latencySeconds, cost: assessment.estimatedCost)
        return arbiter.cloudReason(
            local: local, localFits: required <= (await kvBudgetBytes()),
            cloud: cloud, remainingBudget: assessment.remainingBudget)
    }

    // MARK: — Model loading

    public func loadModel(path: String) async throws(BasaltError) {
        do {
            try await gate.exclusively { try await self.performLoad(path: path) }
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .inferenceError("Model load error: \(error)")
        }
    }

    private func performLoad(path: String) async throws(BasaltError) {
        let size = try? calculateDirectorySize(url: URL(fileURLWithPath: path))
        try await engine.load(modelPath: path)
        currentModelPath = path
        if let size {
            modelSizeMB = Int(size / 1_048_576)
            logger.info("Model size: \(self.modelSizeMB) MB")
        }
        kvGeometry = KVCacheGeometry.forModel(
            path, parameters: ModelDefinition.inferTotalParameters(id: path, name: "", ramMB: Int64(modelSizeMB))
        )
        // After the load: a failed load must not report the failed model's size.
        await self.telemetry.setModelMemory(mb: self.modelSizeMB)
    }

    /// Record the HuggingFace repo ID that was just loaded.
    /// Call this once after `loadModel` completes.
    public func setModelId(_ id: String) {
        self.currentModelId = id
    }

    // MARK: — Model switching

    public func switchModel(to modelId: String) async throws(BasaltError) {
        if let pending = pendingSwitch {
            guard pending.modelId == modelId else {
                throw .modelSwitching(from: currentModelId, to: pending.modelId)
            }
            try await Self.awaitSwitch(pending.task)
            return
        }
        guard modelId != currentModelId else {
            logger.trace("switchModel: already loaded '\(modelId)'")
            return
        }

        // Same resolution `serve` uses, so a repo with 4bit/8bit subfolders switches too.
        let path = (ModelCache.resolve(repoId: modelId) ?? ModelCache.cacheDir(for: modelId)).path
        guard FileManager.default.fileExists(atPath: path) else {
            logger.error("switchModel: model not in cache: \(modelId)")
            throw BasaltError.modelNotCached(identifier: modelId)
        }

        logger.info("Switching model: '\(currentModelId)' → '\(modelId)'")
        let switchStarted = ContinuousClock.now
        let previousModel = currentModelId
        let task = Task {
            defer { self.pendingSwitch = nil }
            try await self.gate.exclusively { try await self.performLoad(path: path) }
            self.currentModelId = modelId
        }
        pendingSwitch = (modelId, task)
        try await Self.awaitSwitch(task)
        await sessionLog?.logModelSwitch(from: previousModel, to: modelId, duration: ContinuousClock.now - switchStarted)
        logger.info("Model switch complete: '\(modelId)'")
    }

    private static func awaitSwitch(_ task: Task<Void, Error>) async throws(BasaltError) {
        do {
            try await task.value
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .inferenceError("Model switch error: \(error)")
        }
    }

    private func admitted(_ priority: RequestPriority) async throws(BasaltError) -> Int {
        do {
            return try await gate.admit(priority: priority)
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .inferenceError("Cancelled while queued")
        }
    }

    public func admissionRefusal(for request: GenerationRequest) async -> BasaltError? {
        if await cloudFallbackReason(for: request) != nil { return nil }
        if let refusal = await gate.refusal(for: request.priority) { return refusal }
        let (room, fitting, promptTokens) = await generationRoom(for: request)
        return room > 0 ? nil : .contextLengthExceeded(maxTokens: fitting, requested: promptTokens + 1)
    }

    public func shutdown(deadline: Duration = .seconds(5)) async {
        await gate.shutdown(deadline: deadline)
        await sessionLog?.close()
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
    ) async throws(BasaltError) -> GenerationResponse {
        guard let sessionLog else { return try await performGenerate(request: request, modelId: modelId).response }
        let id = Self.newRequestId()
        await sessionLog.logRequest(request, id: id, origin: request.origin, modelRequested: modelId)
        await sessionLog.flush()
        do {
            let (response, queueWait) = try await performGenerate(request: request, modelId: modelId)
            await sessionLog.logResponse(request, id: id, status: "ok", result: SessionLogResult(response), queueWait: queueWait)
            return response
        } catch {
            await sessionLog.logResponse(request, id: id, status: Self.logStatus(error), result: SessionLogResult(), queueWait: .zero,
                                         error: error.localizedDescription)
            throw error
        }
    }

    static func newRequestId() -> String {
        "req_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(24)
    }

    static func logStatus(_ error: BasaltError) -> String {
        if case .inferenceError(let message) = error {
            if message.hasPrefix("Timeout after") { return "timeout" }
            if message == "Cancelled while queued" { return "cancelled" }
        }
        return Task.isCancelled ? "cancelled" : "error"
    }

    private func performGenerate(
        request: GenerationRequest, modelId: String?
    ) async throws(BasaltError) -> (response: GenerationResponse, queueWait: Duration) {
        if let cloudFallback, let reason = await cloudFallbackReason(for: request) {
            do {
                logger.info("Serving from the cloud: \(reason.rawValue)")
                return (try await cloudFallback.generate(request), .zero)
            } catch {
                logger.warn("Cloud fallback (\(reason.rawValue)) failed, serving locally: \(error)")
            }
        }
        try await autoSwitch(to: modelId)

        logger.trace("generate: \(request.prompt.prefix(60))…")
        let validated = try await capped(request)
        let queued = ContinuousClock.now
        let ticket = try await admitted(request.priority)
        let queueWait = ContinuousClock.now - queued
        let seconds = timeout(for: validated)
        let work = Task { try await self.withTimeout(seconds: seconds) { try await self.engine.generate(request: validated) } }
        await gate.onCancel(ticket) { work.cancel() }

        await self.telemetry.requestStarted()
        do {
            let response = try await withTaskCancellationHandler { try await work.value } onCancel: { work.cancel() }
            await gate.release(ticket)
            await self.telemetry.requestFinished(with: response)
            return (response, queueWait)
        } catch let error as BasaltError {
            await gate.release(ticket)
            await self.telemetry.requestFinished()
            throw error
        } catch {
            await gate.release(ticket)
            await self.telemetry.requestFinished()
            throw BasaltError.inferenceError("Generation error: \(error)")
        }
    }

    // MARK: — Generate (streaming)

    /// Returns a filtered AsyncStream. Strips `<think>…</think>` blocks.
    /// If `modelId` differs from the currently loaded model, switches first.
    public func generateStream(
        request: GenerationRequest,
        modelId: String? = nil
    ) async throws(BasaltError) -> AsyncStream<StreamChunk> {
        guard let sessionLog else { return try await performStream(request: request, modelId: modelId).stream }
        let id = Self.newRequestId()
        await sessionLog.logRequest(request, id: id, origin: request.origin, modelRequested: modelId)
        await sessionLog.flush()
        let started: (stream: AsyncStream<StreamChunk>, queueWait: Duration)
        do {
            started = try await performStream(request: request, modelId: modelId)
        } catch {
            await sessionLog.logResponse(request, id: id, status: Self.logStatus(error), result: SessionLogResult(), queueWait: .zero,
                                         error: error.localizedDescription)
            throw error
        }
        let source = started.stream
        let queueWait = started.queueWait
        return AsyncStream { continuation in
            let task = Task {
                var result = SessionLogResult()
                for await chunk in source {
                    result.absorb(chunk)
                    continuation.yield(chunk)
                }
                let status = Task.isCancelled ? "cancelled" : result.metadata?.finishReason == .error ? "error" : "ok"
                await sessionLog.logResponse(request, id: id, status: status, result: result, queueWait: queueWait)
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func performStream(
        request: GenerationRequest, modelId: String?
    ) async throws(BasaltError) -> (stream: AsyncStream<StreamChunk>, queueWait: Duration) {
        if let cloudFallback, let reason = await cloudFallbackReason(for: request) {
            do {
                logger.info("Streaming from the cloud: \(reason.rawValue)")
                return (Self.endingWithErrorMetadata(try await cloudFallback.generateStream(request)), .zero)
            } catch {
                logger.warn("Cloud fallback (\(reason.rawValue)) failed to start, serving locally: \(error)")
            }
        }
        try await autoSwitch(to: modelId)

        logger.trace("generateStream start")
        let validated = try await capped(request)
        let queued = ContinuousClock.now
        let ticket = try await admitted(request.priority)
        let queueWait = ContinuousClock.now - queued

        await self.telemetry.requestStarted()
        let stream: AsyncStream<StreamChunk>
        do {
            let seconds = timeout(for: validated)
            let setup = Task { try await self.withTimeout(seconds: seconds) { try await self.engine.generateStream(request: validated) } }
            await gate.onCancel(ticket) { setup.cancel() }
            stream = try await withTaskCancellationHandler { try await setup.value } onCancel: { setup.cancel() }
        } catch let error as BasaltError {
            await gate.release(ticket)
            await self.telemetry.requestFinished()
            throw error
        } catch {
            await gate.release(ticket)
            await self.telemetry.requestFinished()
            throw BasaltError.inferenceError("Stream setup error: \(error)")
        }

        let telemetry = self.telemetry
        let gate = self.gate
        return (AsyncStream<StreamChunk> { continuation in
            let task = Task.detached {
                var result: GenerationResponse?
                for await chunk in stream {
                    switch chunk {
                    case .text(let t):
                        continuation.yield(.text(t))
                    case .reasoning(let r):
                        continuation.yield(.reasoning(r))
                    case .metadata(let m):
                        result = m
                        continuation.yield(.metadata(m))
                    case .toolCall(let call):
                        continuation.yield(.toolCall(call))
                    }
                }
                await gate.release(ticket)
                await telemetry.requestFinished(with: result)
                continuation.finish()
            }
            Task { await gate.onCancel(ticket) { task.cancel() } }
            continuation.onTermination = { _ in task.cancel() }
        }, queueWait)
    }

    static func endingWithErrorMetadata(_ source: AsyncThrowingStream<StreamChunk, Error>) -> AsyncStream<StreamChunk> {
        AsyncStream { continuation in
            let task = Task {
                do {
                    for try await chunk in source { continuation.yield(chunk) }
                } catch {
                    GemLogger(module: "ModelOrchestrator").warn("Cloud stream failed mid-way: \(error)")
                    continuation.yield(.metadata(GenerationResponse(
                        generatedText: "", promptTokens: 0, completionTokens: 0, tokensPerSecond: 0,
                        generationTime: 0, timeToFirstToken: 0,
                        memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0), finishReason: .error)))
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

    // MARK: — Telemetry

    public func telemetrySnapshot() async -> ServerTelemetrySnapshot {
        await telemetry.snapshot()
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

    // MARK: — Private helpers

    /// Switch model if the requested ID differs from the loaded one.
    /// Ignores nil, empty, and generic placeholder IDs like "basalt".
    private func autoSwitch(to modelId: String?) async throws(BasaltError) {
        guard let id = modelId, !id.isEmpty, id != "basalt" else { return }
        if await engine.isLoaded { return }
        guard id != currentModelId else { return }
        try await switchModel(to: id)
    }

    private func timeout(for request: GenerationRequest) -> UInt64 {
        Self.watchdogSeconds(
            maxTokens: request.maxTokens ?? 0, promptTokens: Self.estimatedTokens(Self.budgetedText(request)),
            floor: generationTimeoutSeconds
        )
    }

    static func budgetedText(_ request: GenerationRequest) -> String {
        guard let messages = request.messages, !messages.isEmpty else { return request.prompt }
        return ChatMessageInput.plainText(messages)
    }

    static func watchdogSeconds(maxTokens: Int, promptTokens: Int, floor: UInt64 = 300) -> UInt64 {
        let decodeSeconds = UInt64(max(0, maxTokens)) / 4
        let prefillSeconds = UInt64(max(0, promptTokens)) / 200
        return max(floor, decodeSeconds + prefillSeconds)
    }

    public func effectiveMaxTokens(for request: GenerationRequest) async -> Int {
        let room = await generationRoom(for: request).room
        return max(0, min(request.maxTokens ?? maxTokens, maxTokens, room))
    }

    func kvBudgetBytes() async -> Int64 {
        if let engineBudget = await engine.kvMemoryBudgetBytes { return engineBudget }
        return Int64(Double(freeMemory()) * TokenBudgetCalculator.safetyMargin)
    }

    func generationRoom(for request: GenerationRequest) async -> (room: Int, fitting: Int, promptTokens: Int) {
        let promptTokens = await engine.promptTokenCount(for: request) ?? Self.estimatedTokens(Self.budgetedText(request))
        let fitting = kvGeometry.maxTokens(fitting: await kvBudgetBytes(), kvBits: kvBits)
        return (fitting - promptTokens, fitting, promptTokens)
    }

    private func capped(_ request: GenerationRequest) async throws(BasaltError) -> GenerationRequest {
        let (room, fitting, promptTokens) = await generationRoom(for: request)
        guard room > 0 else {
            throw .contextLengthExceeded(maxTokens: fitting, requested: promptTokens + 1)
        }
        let cap = min(maxTokens, room)
        var v = try request.validated(defaultMaxTokens: cap)
        if let req = request.maxTokens, req > cap {
            logger.warn("Token request (\(req)) > budget (\(cap): --max-tokens \(maxTokens), memory room \(room)). Capping.")
            v = v.withMaxTokens(cap)
        }
        return v
    }

    public func setFreeMemoryProbe(_ probe: @escaping @Sendable () -> Int64) {
        freeMemory = probe
    }

    static func estimatedTokens(_ prompt: String) -> Int {
        prompt.utf8.count / 4
    }

    private func calculateDirectorySize(url: URL) throws -> UInt64 {
        UInt64(max(0, Self.resolvedDirectorySize(at: url)))
    }

    /// Sums file sizes, following symlinks.
    ///
    /// The standard HuggingFace cache stores weights in `blobs/` and fills
    /// `snapshots/` with symlinks, so measuring the link itself reports a few
    /// bytes per shard: a 16 GB model came out as 6 MB, and every downstream
    /// memory estimate and token budget was computed from that.
    static func resolvedDirectorySize(at url: URL) -> Int64 {
        guard let e = FileManager.default.enumerator(at: url,
                        includingPropertiesForKeys: [.fileSizeKey],
                        options: []) else { return 0 }
        var total: Int64 = 0
        for case let f as URL in e {
            let target = f.resolvingSymlinksInPath()
            if let s = try? target.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                total += Int64(s)
            }
        }
        return total
    }

    private func directorySize(at url: URL) -> Int64 {
        Self.resolvedDirectorySize(at: url)
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
                throw BasaltError.inferenceError("Timeout after \(seconds)s")
            }
            guard let result = try await group.next() else {
                throw BasaltError.inferenceError("Operation produced no result")
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
