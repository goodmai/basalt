import Foundation

public struct OnDemandCore: Sendable, Equatable {
    public enum Phase: Sendable, Equatable { case unloaded, loading, loaded, unloading }
    public enum Entry: Sendable, Equatable { case ready, load, wait }

    public private(set) var phase = Phase.unloaded
    public private(set) var users = 0
    public private(set) var generation = 0
    public private(set) var loadEpoch = 0

    public init() {}

    public mutating func enter() -> Entry {
        switch phase {
        case .loaded:
            users += 1
            generation += 1
            return .ready
        case .unloaded:
            phase = .loading
            return .load
        case .loading, .unloading:
            return .wait
        }
    }

    public mutating func loaded(_ succeeded: Bool) {
        phase = succeeded ? .loaded : .unloaded
        generation += 1
        if succeeded { loadEpoch += 1 }
    }

    public mutating func leave(epoch: Int? = nil) -> Int? {
        if let epoch, epoch != loadEpoch { return nil }
        users = max(0, users - 1)
        generation += 1
        return users == 0 ? generation : nil
    }

    public mutating func expire(_ scheduled: Int) -> Bool {
        guard phase == .loaded, users == 0, scheduled == generation else { return false }
        phase = .unloading
        return true
    }

    public mutating func beginUnload() -> Bool {
        guard phase == .loaded else { return false }
        phase = .unloading
        users = 0
        generation += 1
        loadEpoch += 1
        return true
    }

    public mutating func unloaded() {
        phase = .unloaded
        users = 0
        generation += 1
    }

    static func idleWaiter<C: Clock>(
        _ clock: C
    ) -> @Sendable (Duration) -> @Sendable () async throws -> Void where C.Duration == Duration {
        { duration in
            let deadline = clock.now.advanced(by: duration)
            return { try await clock.sleep(until: deadline, tolerance: nil) }
        }
    }
}

public actor OnDemandModel {
    public nonisolated let model: any ModelOrchestrating
    public nonisolated let path: String
    public nonisolated let idle: Duration
    public private(set) var adapters: [String: URL]
    public nonisolated let phases: AsyncStream<OnDemandCore.Phase>
    private let announce: AsyncStream<OnDemandCore.Phase>.Continuation
    private let makeIdleWait: @Sendable (Duration) -> @Sendable () async throws -> Void
    private var core = OnDemandCore()
    private var transition: Task<Result<Void, BasaltError>, Never>?
    private var timer: Task<Void, Never>?
    private var unloadEpoch = 0
    public private(set) var lastUsed: Date?

    public init(
        model: any ModelOrchestrating, path: String, idle: Duration,
        adapters: [String: URL] = [:],
        sleep: @escaping @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
    ) {
        (self.model, self.path, self.idle, self.adapters) = (model, path, idle, adapters)
        self.makeIdleWait = { duration in { try await sleep(duration) } }
        (phases, announce) = AsyncStream.makeStream(bufferingPolicy: .bufferingNewest(1))
    }

    public init<C: Clock>(
        model: any ModelOrchestrating, path: String, idle: Duration,
        adapters: [String: URL] = [:],
        clock: C
    ) where C.Duration == Duration {
        (self.model, self.path, self.idle, self.adapters) = (model, path, idle, adapters)
        self.makeIdleWait = OnDemandCore.idleWaiter(clock)
        (phases, announce) = AsyncStream.makeStream(bufferingPolicy: .bufferingNewest(1))
    }

    deinit {
        timer?.cancel()
        announce.finish()
    }

    public var phase: OnDemandCore.Phase { core.phase }

    public func setAdapters(_ next: [String: URL]) async throws {
        adapters = next
        if core.phase == .loaded {
            try await model.loadAdapters(next)
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
            let task = Task { [model] () async -> Result<Void, BasaltError> in
                await withTaskCancellationShield { await model.unloadModel() }
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

    public func acquire() async throws(BasaltError) -> Duration {
        let started = ContinuousClock.now
        var loads = false
        while true {
            if Task.isCancelled { throw .cancelled }
            timer?.cancel()
            timer = nil
            switch core.enter() {
            case .ready:
                lastUsed = Date()
                return loads ? ContinuousClock.now - started : .zero
            case .wait:
                _ = await transition?.value
            case .load:
                loads = true
                try await performLoad()
            }
        }
    }

    private func performLoad() async throws(BasaltError) {
        let epoch = unloadEpoch
        let (model, path, adapters) = (model, path, adapters)
        let task = Task { () -> Result<Void, BasaltError> in
            let outcome = await Self.loadBaseAndAdapters(model: model, path: path, adapters: adapters)
            await self.finishLoad(outcome, epoch: epoch)
            return outcome
        }
        transition = task
        let result = await task.value
        try result.get()
        if epoch != unloadEpoch || Task.isCancelled {
            if core.phase == .loaded, core.users == 0 { schedule(core.generation) }
            throw .cancelled
        }
    }

    private static func loadBaseAndAdapters(
        model: any ModelOrchestrating, path: String, adapters: [String: URL]
    ) async -> Result<Void, BasaltError> {
        var loadedBase = false
        do throws(BasaltError) {
            try await model.loadModel(path: path)
            loadedBase = true
            if !adapters.isEmpty { try await model.loadAdapters(adapters) }
            return .success(())
        } catch {
            if loadedBase {
                await withTaskCancellationShield { await model.unloadModel() }
            }
            return .failure(error)
        }
    }

    private func finishLoad(_ outcome: Result<Void, BasaltError>, epoch: Int) async {
        if (try? outcome.get()) != nil, epoch != unloadEpoch {
            await withTaskCancellationShield { await model.unloadModel() }
            transition = nil
            core.loaded(false)
            announce.yield(core.phase)
            return
        }
        transition = nil
        core.loaded((try? outcome.get()) != nil)
        announce.yield(core.phase)
    }

    public func release() {
        lastUsed = Date()
        guard let generation = core.leave() else { return }
        schedule(generation)
    }

    private func schedule(_ generation: Int) {
        timer?.cancel()
        let wait = makeIdleWait(idle)
        timer = Task { [weak self] in
            guard (try? await wait()) != nil else { return }
            await self?.expire(generation)
        }
    }

    func expire(_ generation: Int) async {
        guard core.expire(generation) else { return }
        let task = Task { [model] () async -> Result<Void, BasaltError> in
            await withTaskCancellationShield { await model.unloadModel() }
            return .success(())
        }
        transition = task
        _ = await task.value
        transition = nil
        core.unloaded()
        announce.yield(core.phase)
    }
}
