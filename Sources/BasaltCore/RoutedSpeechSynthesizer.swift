import Foundation

public struct RoutedSpeechSynthesizer: SpeechSynthesizer {
    public struct Tier: Sendable {
        public var candidate: SpeechCandidate
        public var synthesizer: (any SpeechSynthesizer)?

        public init(candidate: SpeechCandidate, synthesizer: (any SpeechSynthesizer)?) {
            self.candidate = SpeechCandidate(
                id: candidate.id,
                model: candidate.model,
                compute: candidate.compute,
                isAvailable: candidate.isAvailable && synthesizer != nil,
                estimatedTTFAMs: candidate.estimatedTTFAMs
            )
            self.synthesizer = synthesizer
        }
    }

    public let tiers: [Tier]
    public let onGPUBusy: SpeechOnGPUBusy
    public let maxTTFAMs: Int?
    public let defaults: ComposeSpec.Speak?
    private let gpuBusyProbe: @Sendable () async -> Bool
    private let memoryPressureProbe: @Sendable () async -> Bool

    public init(
        tiers: [Tier],
        onGPUBusy: SpeechOnGPUBusy = .fallbackANE,
        maxTTFAMs: Int? = nil,
        defaults: ComposeSpec.Speak? = nil,
        gpuBusyProbe: @escaping @Sendable () async -> Bool = { false },
        memoryPressureProbe: @escaping @Sendable () async -> Bool = { false }
    ) {
        self.tiers = tiers
        self.onGPUBusy = onGPUBusy
        self.maxTTFAMs = maxTTFAMs
        self.defaults = defaults
        self.gpuBusyProbe = gpuBusyProbe
        self.memoryPressureProbe = memoryPressureProbe
    }

    public var voices: [VoiceSpec] {
        get async {
            var collected: [VoiceSpec] = []
            var seen: Set<String> = []
            for tier in tiers {
                guard let synthesizer = tier.synthesizer else { continue }
                for voice in await synthesizer.voices where seen.insert("\(voice.model):\(voice.id)").inserted {
                    collected.append(voice)
                }
            }
            return collected
        }
    }

    public func resolveRoute(_ request: SpeechRequest) async -> SpeechRouteDecision {
        let busy = await gpuBusyProbe()
        let pressured = await memoryPressureProbe()
        let input = SpeechRouteInput(
            chain: tiers.map(\.candidate),
            gpuBusy: busy,
            onGPUBusy: onGPUBusy,
            memoryPressure: pressured,
            preferANE: request.preferANE,
            maxTTFAMs: maxTTFAMs
        )
        return SpeechFallbackRouter.route(input)
    }

    public func synthesize(_ request: SpeechRequest) async throws(BasaltError) -> AsyncThrowingStream<AudioChunk, any Error> {
        let effective = applyingDefaults(to: request)
        let decision = await resolveRoute(effective)
        if let matched = tiers.first(where: { $0.candidate.id == decision.selected.id }), let backend = matched.synthesizer {
            return try await backend.synthesize(effective)
        }
        for tier in tiers {
            if let backend = tier.synthesizer {
                return try await backend.synthesize(effective)
            }
        }
        throw .invalidRequestStructure(details: "speech: no available speech synthesizer in fallback chain")
    }

    private func applyingDefaults(to request: SpeechRequest) -> SpeechRequest {
        guard let defaults else { return request }
        var merged = request
        if merged.voice == "alloy", let voice = defaults.voice, !voice.isEmpty {
            merged.voice = voice
        }
        if merged.language == nil { merged.language = defaults.language }
        if merged.exaggeration == nil { merged.exaggeration = defaults.exaggeration }
        if merged.referenceAudio == nil { merged.referenceAudio = defaults.referenceAudio }
        return merged
    }

    public func unload() async {
        for tier in tiers {
            await tier.synthesizer?.unload()
        }
    }
}

public actor OnDemandSpeechSynthesizer: SpeechSynthesizer {
    public struct Descriptor: Sendable {
        public var modelID: String
        public var path: String
        public var candidate: SpeechCandidate
        public var declaredVoices: [VoiceSpec]
        public var idle: Duration

        public init(
            modelID: String,
            path: String,
            candidate: SpeechCandidate,
            declaredVoices: [VoiceSpec],
            idle: Duration
        ) {
            (self.modelID, self.path, self.candidate, self.declaredVoices, self.idle) =
                (modelID, path, candidate, declaredVoices, idle)
        }
    }

    public nonisolated let descriptor: Descriptor
    public nonisolated let phases: AsyncStream<OnDemandCore.Phase>
    private let announce: AsyncStream<OnDemandCore.Phase>.Continuation
    private let loader: @Sendable () async throws -> any SpeechSynthesizer
    private let sleep: @Sendable (Duration) async throws -> Void
    private var core = OnDemandCore()
    private var current: (any SpeechSynthesizer)?
    private var transition: Task<Result<Void, BasaltError>, Never>?
    private var timer: Task<Void, Never>?
    private var unloadEpoch = 0
    public private(set) var lastUsed: Date?

    deinit {
        timer?.cancel()
        announce.finish()
    }

    public init(
        descriptor: Descriptor,
        loader: @escaping @Sendable () async throws -> any SpeechSynthesizer,
        sleep: @escaping @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
    ) {
        self.descriptor = descriptor
        self.loader = loader
        self.sleep = sleep
        (phases, announce) = AsyncStream.makeStream(bufferingPolicy: .bufferingNewest(1))
    }

    public var phase: OnDemandCore.Phase { core.phase }

    public var voices: [VoiceSpec] {
        get async {
            if let current { return await current.voices }
            return descriptor.declaredVoices
        }
    }

    public func resolveRoute(_ request: SpeechRequest) async -> SpeechRouteDecision {
        SpeechRouteDecision(selected: descriptor.candidate)
    }

    public func synthesize(_ request: SpeechRequest) async throws(BasaltError) -> AsyncThrowingStream<AudioChunk, any Error> {
        let engine = try await acquire()
        let inner: AsyncThrowingStream<AudioChunk, any Error>
        do {
            inner = try await engine.synthesize(request)
        } catch {
            release()
            throw error
        }
        return Self.forwardStream(inner: inner, owner: self)
    }

    nonisolated private static func forwardStream(
        inner: AsyncThrowingStream<AudioChunk, any Error>,
        owner: OnDemandSpeechSynthesizer
    ) -> AsyncThrowingStream<AudioChunk, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                defer {
                    await withTaskCancellationShield {
                        await owner.release()
                    }
                }
                do {
                    for try await chunk in inner {
                        continuation.yield(chunk)
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
        unloadEpoch &+= 1
        while true {
            timer?.cancel()
            timer = nil
            if let transition {
                _ = await transition.value
                continue
            }
            guard core.beginUnload() else { return }
            announce.yield(core.phase)
            let engine = current
            current = nil
            let task = Task { () async -> Result<Void, BasaltError> in
                if let engine {
                    await withTaskCancellationShield { await engine.unload() }
                }
                return .success(())
            }
            transition = task
            _ = await task.value
            transition = nil
            core.unloaded()
            announce.yield(core.phase)
            return
        }
    }
}

extension OnDemandSpeechSynthesizer {
    private func acquire() async throws(BasaltError) -> any SpeechSynthesizer {
        while true {
            if Task.isCancelled { throw .cancelled }
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
        let epoch = unloadEpoch
        let loader = self.loader
        let task = Task { () async -> Result<Void, BasaltError> in
            let outcome: Result<any SpeechSynthesizer, BasaltError>
            do {
                outcome = .success(try await loader())
            } catch let error as BasaltError {
                outcome = .failure(error)
            } catch {
                outcome = .failure(.modelInferenceError(details: error.localizedDescription))
            }
            await self.finishLoad(outcome, epoch: epoch)
            return outcome.map { _ in () }
        }
        transition = task
        let result = await task.value
        try result.get()
        if epoch != unloadEpoch || Task.isCancelled {
            if core.phase == .loaded, core.users == 0 { schedule(core.generation) }
            throw .cancelled
        }
    }

    private func finishLoad(_ outcome: Result<any SpeechSynthesizer, BasaltError>, epoch: Int) async {
        if case .success(let engine) = outcome, epoch != unloadEpoch {
            await withTaskCancellationShield { await engine.unload() }
            transition = nil
            core.loaded(false)
            announce.yield(core.phase)
            return
        }
        transition = nil
        switch outcome {
        case .success(let engine):
            current = engine
            core.loaded(true)
        case .failure:
            core.loaded(false)
        }
        announce.yield(core.phase)
    }

    private func release() {
        lastUsed = Date()
        guard let generation = core.leave() else { return }
        schedule(generation)
    }

    private func schedule(_ generation: Int) {
        timer?.cancel()
        timer = Task { [weak self, sleep, idle = descriptor.idle] in
            guard (try? await sleep(idle)) != nil else { return }
            await self?.expire(generation)
        }
    }

    private func expire(_ generation: Int) async {
        guard core.expire(generation) else { return }
        announce.yield(core.phase)
        let engine = current
        current = nil
        let task = Task { () async -> Result<Void, BasaltError> in
            if let engine {
                await withTaskCancellationShield { await engine.unload() }
            }
            return .success(())
        }
        transition = task
        _ = await task.value
        transition = nil
        core.unloaded()
        announce.yield(core.phase)
    }
}
