import Foundation
import os

public actor ModelOrchestratorActor {
    let engine: any InferenceEngine
    private var state: OrchestratorState
    let logger = BasaltLogger(module: "ModelOrchestrator")
    nonisolated public let gate: GenerationGate
    nonisolated public let sessionLog: SessionLogWriter?
    public nonisolated let telemetry: TelemetryTracker
    private var shutdownTask: Task<Void, Never>?
    private var cloudFallback: (any CloudFallback)?
    private var arbiter = HybridArbiter()
    private var freeMemory: @Sendable () -> Int64 = { SystemProfiler.availableSystemMemory() }
    private var memory: MemoryCoordinator?
    private var memoryTask: Task<Void, Never>?
    private var thermal = ThermalLevel.current
    private var pacesThermally = false
    private var adaptiveThinkingBudget = false
    var media = MediaFallbacks()

    public struct KVOptions: Sendable {
        let precision: KVPrecision?
        let adaptive: Bool
        let zones: KVZones?

        public init(precision: KVPrecision? = nil, adaptive: Bool = false, zones: KVZones? = nil) {
            self.precision = precision
            self.adaptive = adaptive
            self.zones = zones
        }
    }

    public init(
        engine: any InferenceEngine, maxTokens: Int = 65_536, kv: KVOptions = KVOptions(), telemetry: TelemetryTracker = .shared,
        gate: GenerationGate = GenerationGate(), sessionLog: SessionLogWriter? = nil
    ) {
        self.gate = gate
        self.sessionLog = sessionLog
        self.engine = engine
        self.state = OrchestratorState(kv: KVPlanner(
            maxTokens: maxTokens,
            precisionLadder: kv.adaptive ? KVCacheGeometry.adaptiveKVPrecisionLadder : [kv.precision],
            zones: kv.zones))
        self.telemetry = telemetry
    }

    private var kv: KVPlanner { state.kv }

    func markTraining(_ running: Bool) throws(BasaltError) {
        guard !running || !state.training else { throw .invalidRequestStructure(details: "a distillation is already running on this server; wait for it to finish or cancel it before starting another") }
        state = .reduce(state, running ? .trainingStarted : .trainingEnded)
    }

    public var currentModelId: String { state.modelId }
    public var modelPath: String { state.modelPath }

    var kvReservations: [Int: ByteCount] { state.reservations.bytes }

    var loaded: (path: String, epoch: Int) { (state.modelPath, state.modelEpoch) }

    var routingModelKey: String {
        "\(modelInfo)@\(URL(fileURLWithPath: state.modelPath).lastPathComponent)"
    }

    public var modelInfo: String {
        currentModelId.isEmpty
            ? URL(fileURLWithPath: state.modelPath).lastPathComponent
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
        state = .reduce(state, .named(id))
    }

    public func setAdaptiveThinkingBudget(_ enabled: Bool) {
        adaptiveThinkingBudget = enabled
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

    func performUnload() async -> (path: String, modelId: String) {
        let model = (state.modelPath, state.modelId)
        await engine.unload()
        state = .reduce(state, .unloaded)
        await telemetry.setModelMemory(mb: 0)
        return model
    }

    func performLoad(path: String, modelId: String? = nil) async throws(BasaltError) {
        let sizeMB = Int(UInt64(max(0, Self.resolvedDirectorySize(at: URL(fileURLWithPath: path)))) / 1_048_576)
        try await engine.load(modelPath: path)
        let geometry = KVCacheGeometry.forModel(
            path, parameters: ModelDefinition.inferTotalParameters(id: path, name: "", ramMB: Int64(sizeMB))
        )
        let takes = await (
            zones: engine.supportsKVZones, eviction: engine.supportsKVEviction, quantization: engine.supportsKVQuantization,
            fullPrecisionEviction: engine.evictsKVInFullPrecision)
        state = .reduce(state, .loaded(path: path, modelId: modelId, geometry: geometry, takes: takes))
        logger.info("Model size: \(sizeMB) MB")
        await telemetry.setModelMemory(mb: sizeMB)
        await telemetry.setModel(URL(fileURLWithPath: path).lastPathComponent)
    }

    public func switchModel(to modelId: String) async throws(BasaltError) {
        if let pending = state.pendingSwitch {
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
            defer { self.state = .reduce(self.state, .switchEnded(modelId)) }
            try await self.gate.exclusively { try await self.performLoad(path: path, modelId: modelId) }
        }
        state = .reduce(state, .switchStarted(modelId, task))
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

    func autoSwitch(to modelId: String?) async throws(BasaltError) {
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
        snapshot.experts = await engine.expertTelemetry
        if let memory {
            snapshot.memory = MemoryTelemetry(state: await memory.state, limits: state.memoryLimits, refusedAdmissions: state.memoryRefusals)
        }
        snapshot.thermal = thermalTelemetry
        snapshot.routingBias = await cloudFallback?.routingBias
        if let deferrals = await cloudFallback?.cascadeDeferrals, deferrals > 0 {
            snapshot.cascadeDeferrals = deferrals
        }
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
        guard let cloudFallback, request.isServiceRequest != true, request.adapter == nil, request.origin != OCRRecipe.origin, request.origin.source != PipelineCore.perceivedSource,
              let assessment = await cloudFallback.assess(request, localModel: routingModelKey)
        else { return nil }
        let promptTokens = Self.estimatedTokens(Self.budgetedText(request))
        let wanted = promptTokens + (request.maxTokens ?? 0)
        let budget = await kvBudget()
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
        guard memory === coordinator else { return }
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

    public func coordinateThermal(with source: @escaping @Sendable () -> ThermalLevel = ThermalLevel.current) async {
        (thermal, pacesThermally) = (source, true)
        await engine.pace(with: PrefillPacer(directive: { ThermalDirective(source()) }))
    }

    var thermalTelemetry: ThermalTelemetry {
        let level = thermal()
        return ThermalTelemetry(level: level, directive: pacesThermally ? ThermalDirective(level) : .normal)
    }

    private func applyMemoryDirective() async {
        guard let memory else { return }
        let directive = await memory.state.directive
        if let limits = await engine.apply(directive) { state = .reduce(state, .limitsApplied(limits)) }
    }

    private func refusal(_ room: KVRoom, under directive: MemoryDirective) -> BasaltError? {
        let heat = thermalTelemetry
        if let refusal = heat.directive.refusal(promptTokens: room.promptTokens) {
            logger.warn("Refused with 503: thermal state \(heat.level.rawValue), and this request needs \(room.promptTokens) prompt tokens; retry in \(ThermalDirective.retryAfterSeconds) s.")
            return refusal
        }
        let refusal = room.refusal(under: directive)
        if case .memoryPressure? = refusal {
            state = .reduce(state, .refusedForMemory)
            logger.warn("Refused with 503: critical memory pressure, and this request needs \(room.promptTokens) prompt tokens with room for \(max(0, room.room)) more; retry in \(MemoryDirective.retryAfterSeconds) s.")
        }
        return refusal
    }

    func kvBudget() async -> ByteCount {
        let budget = if let engineBudget = await engine.kvMemoryBudget {
            engineBudget
        } else {
            ByteCount(Int64(Double(freeMemory()) * TokenBudgetCalculator.safetyMargin))
        }
        guard let state = await memory?.state, state.directive.limitsAdmission else { return budget }
        return min(budget, ByteCount(Int64(Double(state.sample.availableBytes) * TokenBudgetCalculator.safetyMargin)))
    }

    private func generationRoom(for request: GenerationRequest) async -> KVRoom {
        let promptTokens = await engine.promptTokenCount(for: request) ?? Self.estimatedTokens(Self.budgetedText(request))
        let budget = await kvBudget()
        return kv.room(promptTokens: promptTokens, requestedMaxTokens: request.maxTokens, budget: budget)
    }

    private func admit(
        _ request: GenerationRequest
    ) async throws(BasaltError) -> (request: GenerationRequest, ticket: Int, queueWait: Duration) {
        await gate.arrive()
        defer { await gate.arrived() }
        var plan = try await capped(request)
        let queued = ContinuousClock.now
        let ticket = try await admitted(request)
        if plan.epoch != state.modelEpoch {
            do {
                plan = try await capped(request)
            } catch {
                await gate.release(ticket)
                throw error
            }
        }
        try await reserveKV(ticket, promptTokens: plan.promptTokens, for: plan.request, planned: plan.epoch)
        await applyMemoryDirective()
        return (plan.request, ticket, ContinuousClock.now - queued)
    }

    private func capped(
        _ request: GenerationRequest
    ) async throws(BasaltError) -> (request: GenerationRequest, promptTokens: Int, epoch: Int) {
        let epoch = state.modelEpoch
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
        return (budgetedByDifficulty(validated), room.promptTokens, epoch)
    }

    private func budgetedByDifficulty(_ request: GenerationRequest) -> GenerationRequest {
        guard adaptiveThinkingBudget, let (difficulty, budget) = ThinkingDifficulty.adaptiveBudget(for: request) else { return request }
        logger.info("Adaptive thinking budget: \(difficulty.rawValue) prompt, thinking_budget \(budget)")
        var budgeted = request
        budgeted.thinkingBudget = budget
        return budgeted
    }

    func admitted(_ priority: RequestPriority) async throws(BasaltError) -> Int {
        do {
            return try await gate.admit(priority: priority)
        } catch {
            throw error as? BasaltError ?? .cancelled
        }
    }

    func reserveKV(_ ticket: Int, promptTokens: Int, for request: GenerationRequest, planned epoch: Int) async throws(BasaltError) {
        let budget = await kvBudget()
        let bytes = kv.reservationBytes(promptTokens: promptTokens, for: request)
        state = .reduce(state, .reserved(ticket: ticket, bytes: bytes, budget: budget, epoch: epoch))
        guard state.reservations.bytes[ticket] == nil else { return }
        let reservations = state.reservations
        logger.warn("Refused with 503: this request needs \(bytes.rawValue >> 20) MB of KV cache and \(reservations.bytes.count) running requests hold \(reservations.held.rawValue >> 20) MB of the \(budget.rawValue >> 20) MB room; retry when one finishes.")
        await gate.release(ticket)
        throw .serverBusy(retryAfterSeconds: 1)
    }

    func finish(_ ticket: Int) async {
        state = .reduce(state, .released(ticket))
        await applyMemoryDirective()
        await gate.release(ticket)
    }
}

extension ModelOrchestratorActor {
    public func generate(
        request: GenerationRequest,
        modelId: String? = nil
    ) async throws(BasaltError) -> GenerationResponse {
        let request = await telemetry.sampled(request)
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
        let request = await telemetry.sampled(request)
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
            let task = Task { @concurrent in
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
