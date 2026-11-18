import Foundation

public struct RoutedImageSynthesizer: ImageSynthesizer {
    public struct Entry: Sendable {
        public var alias: String
        public var synthesizer: any ImageSynthesizer

        public init(alias: String, synthesizer: any ImageSynthesizer) {
            self.alias = alias
            self.synthesizer = synthesizer
        }
    }

    public struct AuxStages: Sendable {
        public var sharedEncoderModel: String?
        public var upsamplerModel: String?
        public var upsampler: (@Sendable (String) async throws -> String)?
        public var decoderModel: String?
        public var judgeModel: String?
        public var candidateJudge: (@Sendable (String, [ImageCompletedArtifact]) async throws -> Int)?
        public var defaultCandidates: Int?
        public var defaultTeacacheThreshold: Double?
        public var defaultPreviewEverySteps: Int?

        public init(
            sharedEncoderModel: String? = nil,
            upsamplerModel: String? = nil,
            upsampler: (@Sendable (String) async throws -> String)? = nil,
            judgeModel: String? = nil,
            candidateJudge: (@Sendable (String, [ImageCompletedArtifact]) async throws -> Int)? = nil
        ) {
            self.sharedEncoderModel = sharedEncoderModel
            self.upsamplerModel = upsamplerModel
            self.upsampler = upsampler
            self.decoderModel = nil
            self.judgeModel = judgeModel
            self.candidateJudge = candidateJudge
            self.defaultCandidates = nil
            self.defaultTeacacheThreshold = nil
            self.defaultPreviewEverySteps = nil
        }
    }

    public let entries: [Entry]
    public let onGPUBusy: ImageOnGPUBusy
    public let isGPUBusy: @Sendable () async -> Bool
    public var aux: AuxStages

    public init(
        entries: [Entry],
        onGPUBusy: ImageOnGPUBusy = .queue,
        isGPUBusy: @escaping @Sendable () async -> Bool = { false },
        aux: AuxStages = AuxStages()
    ) {
        self.entries = entries
        self.onGPUBusy = onGPUBusy
        self.isGPUBusy = isGPUBusy
        self.aux = aux
    }

    public var sharedEncoderModel: String? {
        get { aux.sharedEncoderModel }
        set { aux.sharedEncoderModel = newValue }
    }

    public var upsamplerModel: String? {
        get { aux.upsamplerModel }
        set { aux.upsamplerModel = newValue }
    }

    public var upsampler: (@Sendable (String) async throws -> String)? {
        get { aux.upsampler }
        set { aux.upsampler = newValue }
    }

    public var judgeModel: String? {
        get { aux.judgeModel }
        set { aux.judgeModel = newValue }
    }

    public var decoderModel: String? {
        get { aux.decoderModel }
        set { aux.decoderModel = newValue }
    }

    public var candidateJudge: (@Sendable (String, [ImageCompletedArtifact]) async throws -> Int)? {
        get { aux.candidateJudge }
        set { aux.candidateJudge = newValue }
    }

    public var modelIdentifier: String {
        entries.first?.synthesizer.modelIdentifier ?? "routed-image"
    }

    public var descriptor: ImageModelDescriptor {
        entries.first?.synthesizer.descriptor ?? ImageModelDescriptor.zImageTurbo()
    }

    public var allDescriptors: [ImageModelDescriptor] {
        var result: [ImageModelDescriptor] = []
        var seen: Set<String> = []
        for entry in entries where seen.insert(entry.synthesizer.descriptor.id).inserted {
            result.append(entry.synthesizer.descriptor)
        }
        return result
    }

    public func resolveRoute(_ request: ImageGenerationRequest) async -> ImageRouteDecision {
        guard let primary = entries.first else {
            return decorated(ImageRouteDecision(selectedModel: request.model))
        }
        if onGPUBusy == .fallback, entries.count > 1, await isGPUBusy() {
            let fallback = entries.dropFirst().first {
                $0.synthesizer.descriptor.defaultCompute != .mlx
            } ?? entries[1]
            let decision = ImageRouteDecision(
                selectedModel: fallback.synthesizer.modelIdentifier,
                compute: fallback.synthesizer.descriptor.defaultCompute,
                fallbackFrom: primary.synthesizer.modelIdentifier
            )
            return decorated(decision)
        }
        let selected = entries.first {
            $0.alias == request.model || $0.synthesizer.modelIdentifier == request.model
        } ?? primary
        let decision = ImageRouteDecision(
            selectedModel: selected.synthesizer.modelIdentifier,
            compute: selected.synthesizer.descriptor.defaultCompute
        )
        return decorated(decision)
    }

    public func generate(_ request: ImageGenerationRequest) async throws -> AsyncThrowingStream<ImageGenerationEvent, any Error> {
        guard !entries.isEmpty else {
            throw BasaltError.invalidRequestStructure(details: "image: no image synthesizer is available in chain")
        }
        try request.validate()
        let enriched = try await prepareRequest(request)
        let decision = await resolveRoute(enriched)
        let chain = orderedChain(for: decision)
        if enriched.config.candidates > 1, let judge = aux.candidateJudge {
            let winner = try await generateJudgedWinner(request: enriched, chain: chain, judge: judge)
            return AsyncThrowingStream { continuation in
                continuation.yield(.completed(winner))
                continuation.finish()
            }
        }
        return Self.streamWithFallback(request: enriched, chain: chain, decoderModel: aux.decoderModel)
    }
}

extension RoutedImageSynthesizer {
    private func decorated(_ base: ImageRouteDecision) -> ImageRouteDecision {
        var decision = base
        decision.sharedEncoderModel = aux.sharedEncoderModel
        decision.upsamplerModel = aux.upsamplerModel
        decision.decoderModel = aux.decoderModel
        decision.judgeModel = aux.judgeModel
        return decision
    }

    private func prepareRequest(_ request: ImageGenerationRequest) async throws -> ImageGenerationRequest {
        var enriched = request
        if enriched.config.candidates == 1, let defaultCandidates = aux.defaultCandidates, defaultCandidates > 1 {
            enriched.config.candidates = defaultCandidates
        }
        if enriched.config.teacacheThreshold == 0, let threshold = aux.defaultTeacacheThreshold, threshold > 0 {
            enriched.config.teacacheThreshold = threshold
        }
        if enriched.config.previewEverySteps == 0, let preview = aux.defaultPreviewEverySteps, preview > 0 {
            enriched.config.previewEverySteps = preview
        }
        if enriched.config.upsampledPrompt == nil, let upsampler = aux.upsampler {
            let expanded = try await upsampler(enriched.prompt).trimmingCharacters(in: .whitespacesAndNewlines)
            if !expanded.isEmpty {
                enriched.config.upsampledPrompt = expanded
            }
        }
        return enriched
    }

    private func orderedChain(for decision: ImageRouteDecision) -> [Entry] {
        guard let startIndex = entries.firstIndex(where: { $0.synthesizer.modelIdentifier == decision.selectedModel }) else {
            return entries
        }
        var ordered = [entries[startIndex]]
        for (idx, entry) in entries.enumerated() where idx != startIndex {
            ordered.append(entry)
        }
        return ordered
    }

    private func generateJudgedWinner(
        request: ImageGenerationRequest,
        chain: [Entry],
        judge: @Sendable (String, [ImageCompletedArtifact]) async throws -> Int
    ) async throws -> ImageCompletedArtifact {
        let baseSeed = request.config.seed ?? 42
        var artifacts: [ImageCompletedArtifact] = []
        artifacts.reserveCapacity(request.config.candidates)
        for index in 0 ..< request.config.candidates {
            var single = request
            single.config.candidates = 1
            single.stream = false
            single.config.previewEverySteps = 0
            single.config.seed = baseSeed &+ UInt64(index)
            let artifact = try await Self.collectSingleArtifact(request: single, chain: chain, decoderModel: aux.decoderModel)
            artifacts.append(artifact)
        }
        guard !artifacts.isEmpty else {
            throw BasaltError.modelInferenceError(details: "image: candidate generation produced no artifacts")
        }
        let chosen = try await judge(request.effectivePrompt, artifacts)
        let clamped = min(max(0, chosen), artifacts.count - 1)
        return artifacts[clamped]
    }

    private static func collectSingleArtifact(
        request: ImageGenerationRequest,
        chain: [Entry],
        decoderModel: String?
    ) async throws -> ImageCompletedArtifact {
        let stream = streamWithFallback(request: request, chain: chain, decoderModel: decoderModel)
        var completed: ImageCompletedArtifact?
        for try await event in stream {
            if case .completed(let artifact) = event {
                completed = artifact
            }
        }
        guard let completed else {
            throw BasaltError.modelInferenceError(details: "image: synthesizer finished without a completed artifact")
        }
        return completed
    }

    private static func streamWithFallback(
        request: ImageGenerationRequest,
        chain: [Entry],
        decoderModel: String?
    ) -> AsyncThrowingStream<ImageGenerationEvent, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                var lastError: (any Error)?
                for (index, entry) in chain.enumerated() {
                    var emittedFrame = false
                    do {
                        let inner = try await entry.synthesizer.generate(request)
                        for try await event in inner {
                            emittedFrame = true
                            continuation.yield(annotated(event, upsampledPrompt: request.config.upsampledPrompt, decoderModel: decoderModel))
                        }
                        continuation.finish()
                        return
                    } catch {
                        lastError = error
                        if emittedFrame || index == chain.count - 1 {
                            continuation.finish(throwing: error)
                            return
                        }
                    }
                }
                continuation.finish(throwing: lastError ?? BasaltError.modelInferenceError(details: "image: chain exhausted"))
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private static func annotated(
        _ event: ImageGenerationEvent,
        upsampledPrompt: String?,
        decoderModel: String?
    ) -> ImageGenerationEvent {
        guard case .completed(var artifact) = event else { return event }
        if artifact.upsampledPrompt == nil {
            artifact.upsampledPrompt = upsampledPrompt
        }
        if artifact.decoderModel == nil {
            artifact.decoderModel = decoderModel
        }
        return .completed(artifact)
    }
}

public actor OnDemandImageSynthesizer: ImageSynthesizer {
    public nonisolated let modelIdentifier: String
    public nonisolated let path: String
    public nonisolated let descriptor: ImageModelDescriptor
    public nonisolated let idle: Duration
    public nonisolated let phases: AsyncStream<OnDemandCore.Phase>
    private let announce: AsyncStream<OnDemandCore.Phase>.Continuation
    private let loader: @Sendable () async throws -> any ImageSynthesizer
    private let sleep: @Sendable (Duration) async throws -> Void
    private var core = OnDemandCore()
    private var current: (any ImageSynthesizer)?
    private var transition: Task<Result<Void, BasaltError>, Never>?
    private var timer: Task<Void, Never>?
    public private(set) var lastUsed: Date?

    public init(
        modelIdentifier: String,
        path: String,
        descriptor: ImageModelDescriptor,
        idle: Duration,
        loader: @escaping @Sendable () async throws -> any ImageSynthesizer,
        sleep: @escaping @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
    ) {
        self.modelIdentifier = modelIdentifier
        self.path = path
        self.descriptor = descriptor
        self.idle = idle
        self.loader = loader
        self.sleep = sleep
        (phases, announce) = AsyncStream.makeStream(bufferingPolicy: .bufferingNewest(1))
    }

    public var phase: OnDemandCore.Phase { core.phase }

    public nonisolated func resolveRoute(_ request: ImageGenerationRequest) async -> ImageRouteDecision {
        _ = request
        return ImageRouteDecision(selectedModel: modelIdentifier, compute: descriptor.defaultCompute)
    }

    public func generate(_ request: ImageGenerationRequest) async throws -> AsyncThrowingStream<ImageGenerationEvent, any Error> {
        let engine = try await acquire()
        let inner: AsyncThrowingStream<ImageGenerationEvent, any Error>
        do {
            inner = try await engine.generate(request)
        } catch {
            release()
            throw error
        }
        return Self.forwardStream(inner: inner, owner: self)
    }

    nonisolated private static func forwardStream(
        inner: AsyncThrowingStream<ImageGenerationEvent, any Error>,
        owner: OnDemandImageSynthesizer
    ) -> AsyncThrowingStream<ImageGenerationEvent, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                defer {
                    await withTaskCancellationShield {
                        await owner.release()
                    }
                }
                do {
                    for try await event in inner {
                        continuation.yield(event)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public func unload() async {
        timer?.cancel()
        timer = nil
        _ = await transition?.value
        current = nil
        core.unloaded()
        announce.yield(core.phase)
    }
}

extension OnDemandImageSynthesizer {
    private func acquire() async throws(BasaltError) -> any ImageSynthesizer {
        while true {
            timer?.cancel()
            timer = nil
            switch core.enter() {
            case .ready:
                if let current {
                    lastUsed = Date()
                    return current
                }
                core.unloaded()
            case .wait:
                _ = await transition?.value
            case .load:
                try await performLoad()
            }
        }
    }

    private func performLoad() async throws(BasaltError) {
        let loader = self.loader
        let loadTask = Task { () async -> Result<any ImageSynthesizer, BasaltError> in
            do {
                return .success(try await loader())
            } catch let error as BasaltError {
                return .failure(error)
            } catch {
                return .failure(.modelInferenceError(details: error.localizedDescription))
            }
        }
        let gateTask = Task { () async -> Result<Void, BasaltError> in
            (await loadTask.value).map { _ in () }
        }
        transition = gateTask
        let outcome = await loadTask.value
        _ = await gateTask.value
        transition = nil
        switch outcome {
        case .success(let engine):
            current = engine
            core.loaded(true)
            announce.yield(core.phase)
        case .failure(let error):
            core.loaded(false)
            announce.yield(core.phase)
            throw error
        }
        if Task.isCancelled {
            schedule(core.generation)
            throw .cancelled
        }
    }

    private func release() {
        lastUsed = Date()
        guard let generation = core.leave() else { return }
        schedule(generation)
    }

    private func schedule(_ generation: Int) {
        timer?.cancel()
        timer = Task { [sleep, idle] in
            guard (try? await sleep(idle)) != nil else { return }
            await self.expire(generation)
        }
    }

    private func expire(_ generation: Int) async {
        guard core.expire(generation) else { return }
        current = nil
        core.unloaded()
        announce.yield(core.phase)
    }
}
