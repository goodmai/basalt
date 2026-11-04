import Foundation
import os

public actor ModelOrchestratorActor {
    let engine: any InferenceEngine
    private var kv: KVPlanner
    private var reservations = KVReservations()
    private var modelSizeMB: Int = 4000
    private let logger = BasaltLogger(module: "ModelOrchestrator")
    private var currentModelPath: String = ""
    public private(set) var currentModelId: String = ""
    nonisolated public let gate: GenerationGate
    public let sessionLog: SessionLogWriter?
    public let telemetry: TelemetryTracker
    private var pendingSwitch: (modelId: String, task: Task<Void, Error>)?
    private var shutdownTask: Task<Void, Never>?
    private var cloudFallback: (any CloudFallback)?
    private var arbiter = HybridArbiter()
    private var freeMemory: @Sendable () -> Int64 = { SystemProfiler.availableSystemMemory() }
    private var memory: MemoryCoordinator?
    private var memoryTask: Task<Void, Never>?
    private var memoryLimits: MemoryLimits?
    private var memoryRefusals = 0

    public init(
        engine: any InferenceEngine, maxTokens: Int = 65_536, kvPrecision: KVPrecision? = nil, adaptiveKVPrecision: Bool = false,
        kvZones: KVZones? = nil, telemetry: TelemetryTracker = .shared,
        gate: GenerationGate = GenerationGate(), sessionLog: SessionLogWriter? = nil
    ) {
        self.gate = gate
        self.sessionLog = sessionLog
        self.engine = engine
        self.kv = KVPlanner(
            maxTokens: maxTokens,
            precisionLadder: adaptiveKVPrecision ? KVCacheGeometry.adaptiveKVPrecisionLadder : [kvPrecision],
            zones: kvZones)
        self.telemetry = telemetry
    }

    var kvReservations: [Int: Int64] { reservations.bytes }

    var routingModelKey: String {
        "\(modelInfo)@\(URL(fileURLWithPath: currentModelPath).lastPathComponent)"
    }

    public var modelInfo: String {
        currentModelId.isEmpty
            ? URL(fileURLWithPath: currentModelPath).lastPathComponent
            : currentModelId
    }

    public func setCloudFallback(
        _ fallback: (any CloudFallback)?, arbiter: HybridArbiter = HybridArbiter(),
        freeMemory: @escaping @Sendable () -> Int64 = { SystemProfiler.availableSystemMemory() }
    ) {
        self.cloudFallback = fallback
        self.arbiter = arbiter
        self.freeMemory = freeMemory
    }

    public func setFreeMemoryProbe(_ probe: @escaping @Sendable () -> Int64) {
        freeMemory = probe
    }

    public func setModelId(_ id: String) {
        self.currentModelId = id
    }
}

extension ModelOrchestratorActor {
    public func loadModel(path: String) async throws(BasaltError) {
        do {
            try await gate.exclusively { try await self.performLoad(path: path) }
        } catch {
            throw Self.basaltError(error, "Model load error")
        }
    }

    private func performLoad(path: String) async throws(BasaltError) {
        let size = UInt64(max(0, Self.resolvedDirectorySize(at: URL(fileURLWithPath: path))))
        try await engine.load(modelPath: path)
        currentModelPath = path
        modelSizeMB = Int(size / 1_048_576)
        logger.info("Model size: \(self.modelSizeMB) MB")
        kv.geometry = KVCacheGeometry.forModel(
            path, parameters: ModelDefinition.inferTotalParameters(id: path, name: "", ramMB: Int64(modelSizeMB))
        )
        kv.takesZones = await engine.supportsKVZones
        kv.takesEviction = await engine.supportsKVEviction
        kv.takesQuantization = await engine.supportsKVQuantization
        await self.telemetry.setModelMemory(mb: self.modelSizeMB)
        await telemetry.setModel(URL(fileURLWithPath: path).lastPathComponent)
    }

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
        } catch {
            throw basaltError(error, "Model switch error")
        }
    }

    private func autoSwitch(to modelId: String?) async throws(BasaltError) {
        guard let id = modelId, !id.isEmpty, id != "basalt" else { return }
        if await engine.isLoaded { return }
        guard id != currentModelId else { return }
        try await switchModel(to: id)
    }

    public func shutdown(deadline: Duration = .seconds(5)) async {
        memoryTask?.cancel()
        let task: Task<Void, Never>
        if let existing = shutdownTask {
            task = existing
        } else {
            task = Task {
                await withTaskCancellationShield {
                    await self.gate.shutdown(deadline: deadline)
                    await self.engine.persistPromptCache(within: deadline)
                    await self.sessionLog?.close()
                }
            }
            shutdownTask = task
        }
        await withTaskCancellationShield { await task.value }
    }

    public func healthSnapshot(modelId: String?) async -> HealthResponse {
        let ready = await engine.isLoaded
        return HealthResponse(
            status: ready ? "ok" : "initializing",
            modelId: modelId ?? (currentModelId.isEmpty ? nil : currentModelId),
            isReady: ready,
            version: HealthResponse.version
        )
    }

    public func telemetrySnapshot() async -> ServerTelemetrySnapshot {
        var snapshot = await telemetry.snapshot()
        let anchors = await engine.hybridAnchorUsage
        (snapshot.hybridAnchors, snapshot.hybridAnchorBytes) = (anchors?.count, anchors?.bytes)
        snapshot.activeDeltas = await engine.activeDeltas
        if let memory {
            snapshot.memory = MemoryTelemetry(state: await memory.state, limits: memoryLimits, refusedAdmissions: memoryRefusals)
        }
        snapshot.routingBias = await cloudFallback?.routingBias
        return snapshot
    }

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
                return CachedModelInfo(
                    id: repoId,
                    sizeBytes: Self.resolvedDirectorySize(at: root.appendingPathComponent(dir)),
                    isLoaded: repoId == currentModelId
                )
            }
    }
}

extension ModelOrchestratorActor {
    func cloudFallbackReason(for request: GenerationRequest) async -> CloudFallbackReason? {
        guard let cloudFallback, request.isServiceRequest != true,
              let assessment = await cloudFallback.assess(request, localModel: routingModelKey)
        else { return nil }
        let promptTokens = Self.estimatedTokens(Self.budgetedText(request))
        let wanted = promptTokens + (request.maxTokens ?? 0)
        let budget = await kvBudgetBytes()
        let fitting = kv.fit(promptTokens: promptTokens, wantedTokens: wanted, budget: budget).fitting
        let tokensPerSecond = await telemetry.snapshot().tokensPerSecond
        let local = LocalEstimate(
            promptTokens: promptTokens, wantedTokens: wanted, fittingTokens: fitting,
            decodeTokens: request.maxTokens ?? kv.maxTokens, tokensPerSecond: tokensPerSecond)
        return arbiter.cloudReason(assessment, local: local)
    }

    public func admissionRefusal(for request: GenerationRequest) async -> BasaltError? {
        let directive = await memory?.sample().directive ?? .relaxed
        if await cloudFallbackReason(for: request) != nil { return nil }
        if let refusal = await gate.refusal(for: request.priority) { return refusal }
        let room = await generationRoom(for: request)
        return refusal(room, under: directive)
    }

    public func effectiveMaxTokens(for request: GenerationRequest) async -> Int {
        let room = await generationRoom(for: request)
        return kv.effectiveMaxTokens(requested: request.maxTokens, room: room)
    }

    public func coordinateMemory(with coordinator: MemoryCoordinator) async {
        memoryTask?.cancel()
        memory = coordinator
        await coordinator.sample()
        await applyMemoryDirective()
        memoryTask = Task {
            async let sampling: Void = coordinator.run()
            for await state in coordinator.changes {
                logger.info("Memory pressure \(state.sample.pressure.rawValue), \(state.sample.availableBytes >> 20) MB available: directive \(state.directive)")
                guard await gate.inFlight == 0 else { continue }
                await applyMemoryDirective()
            }
            await sampling
        }
    }

    private func applyMemoryDirective() async {
        guard let memory else { return }
        let directive = await memory.state.directive
        if let limits = await engine.apply(directive) { memoryLimits = limits }
    }

    private func refusal(_ room: KVRoom, under directive: MemoryDirective) -> BasaltError? {
        let refusal = room.refusal(under: directive)
        if case .memoryPressure? = refusal {
            memoryRefusals += 1
            logger.warn("Refused with 503: critical memory pressure, and this request needs \(room.promptTokens) prompt tokens with room for \(max(0, room.room)) more; retry in \(MemoryDirective.retryAfterSeconds) s.")
        }
        return refusal
    }

    func kvBudgetBytes() async -> Int64 {
        let budget = if let engineBudget = await engine.kvMemoryBudgetBytes {
            engineBudget
        } else {
            Int64(Double(freeMemory()) * TokenBudgetCalculator.safetyMargin)
        }
        guard let state = await memory?.state, state.directive.limitsAdmission else { return budget }
        return min(budget, Int64(Double(state.sample.availableBytes) * TokenBudgetCalculator.safetyMargin))
    }

    private func generationRoom(for request: GenerationRequest) async -> KVRoom {
        let promptTokens = await engine.promptTokenCount(for: request) ?? Self.estimatedTokens(Self.budgetedText(request))
        let budget = await kvBudgetBytes()
        return kv.room(promptTokens: promptTokens, requestedMaxTokens: request.maxTokens, budget: budget)
    }

    private func admit(
        _ request: GenerationRequest
    ) async throws(BasaltError) -> (request: GenerationRequest, ticket: Int, queueWait: Duration) {
        let (validated, promptTokens) = try await capped(request)
        let queued = ContinuousClock.now
        let ticket = try await admitted(request.priority)
        try await reserveKV(ticket, promptTokens: promptTokens, for: validated)
        await applyMemoryDirective()
        return (validated, ticket, ContinuousClock.now - queued)
    }

    private func capped(_ request: GenerationRequest) async throws(BasaltError) -> (request: GenerationRequest, promptTokens: Int) {
        let directive = await memory?.sample().directive ?? .relaxed
        let room = await generationRoom(for: request)
        if let refusal = refusal(room, under: directive) { throw refusal }
        let validated = try kv.validated(request, in: room)
        let cap = kv.cap(for: room)
        if let kvPrecision = room.kvPrecision, kv.isAdaptive {
            logger.info("KV cache quantized \(kvPrecision) for this request: fp16 cannot hold \(room.promptTokens) prompt + \(request.maxTokens ?? cap) tokens.")
        }
        if let wanted = request.maxTokens, wanted > cap {
            logger.warn("Token request (\(wanted)) > budget (\(cap): --max-tokens \(kv.maxTokens), memory room \(room.room)). Capping.")
        }
        return (validated, room.promptTokens)
    }

    func admitted(_ priority: RequestPriority) async throws(BasaltError) -> Int {
        do {
            return try await gate.admit(priority: priority)
        } catch {
            throw error as? BasaltError ?? .inferenceError("Cancelled while queued")
        }
    }

    func reserveKV(_ ticket: Int, promptTokens: Int, for request: GenerationRequest) async throws(BasaltError) {
        let budget = await kvBudgetBytes()
        let bytes = kv.reservationBytes(promptTokens: promptTokens, for: request)
        guard reservations.reserve(bytes, for: ticket, within: budget) else {
            logger.warn("Refused with 503: this request needs \(bytes >> 20) MB of KV cache and \(reservations.bytes.count) running requests hold \(reservations.held >> 20) MB of the \(budget >> 20) MB room; retry when one finishes.")
            await gate.release(ticket)
            throw .serverBusy(retryAfterSeconds: 1)
        }
    }

    func finish(_ ticket: Int) async {
        reservations.release(ticket)
        await applyMemoryDirective()
        await gate.release(ticket)
    }
}

extension ModelOrchestratorActor {
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
}

extension ModelOrchestratorActor {
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
        let (validated, ticket, queueWait) = try await admit(request)
        defer {
            await withTaskCancellationShield {
                await finish(ticket)
            }
        }
        let seconds = Self.timeout(for: validated)
        let work = Task { try await self.withTimeout(seconds: seconds) { try await self.engine.generate(request: validated) } }
        await gate.onCancel(ticket) { work.cancel() }
        await telemetry.requestStarted()
        do {
            let response = try await withTaskCancellationHandler { try await work.value } onCancel: { work.cancel() }
            await telemetry.requestFinished(with: response)
            await cloudFallback?.recordLocalOutcome(request, response: response, localModel: routingModelKey)
            return (response, queueWait)
        } catch {
            await telemetry.requestFinished()
            throw Self.basaltError(error, "Generation error")
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
        let (validated, ticket, queueWait) = try await admit(request)
        await telemetry.requestStarted()
        let stream: AsyncStream<StreamChunk>
        do {
            let seconds = Self.timeout(for: validated)
            let setup = Task { try await self.withTimeout(seconds: seconds) { try await self.engine.generateStream(request: validated) } }
            await gate.onCancel(ticket) { setup.cancel() }
            stream = try await withTaskCancellationHandler { try await setup.value } onCancel: { setup.cancel() }
        } catch {
            await finish(ticket)
            await telemetry.requestFinished()
            throw Self.basaltError(error, "Stream setup error")
        }
        return (relay(stream, ticket: ticket, request: request), queueWait)
    }

    private func relay(_ stream: AsyncStream<StreamChunk>, ticket: Int, request: GenerationRequest) -> AsyncStream<StreamChunk> {
        let telemetry = self.telemetry
        let gate = self.gate
        let cloudFallback = self.cloudFallback
        let localModel = routingModelKey
        return AsyncStream<StreamChunk> { continuation in
            let task = Task.detached {
                var result: GenerationResponse?
                for await chunk in stream {
                    if case .metadata(let metadata) = chunk { result = metadata }
                    continuation.yield(chunk)
                }
                await self.finish(ticket)
                await telemetry.requestFinished(with: result)
                if let result { await cloudFallback?.recordLocalOutcome(request, response: result, localModel: localModel) }
                continuation.finish()
            }
            Task { await gate.onCancel(ticket) { task.cancel() } }
            continuation.onTermination = { _ in task.cancel() }
        }
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
