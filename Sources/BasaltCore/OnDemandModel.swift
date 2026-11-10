import Foundation

public struct OnDemandCore: Sendable, Equatable {
    public enum Phase: Sendable, Equatable { case unloaded, loading, loaded, unloading }
    public enum Entry: Sendable, Equatable { case ready, load, wait }

    public private(set) var phase = Phase.unloaded
    public private(set) var users = 0
    public private(set) var generation = 0

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
    }

    public mutating func leave() -> Int? {
        users = max(0, users - 1)
        generation += 1
        return users == 0 ? generation : nil
    }

    public mutating func expire(_ scheduled: Int) -> Bool {
        guard phase == .loaded, users == 0, scheduled == generation else { return false }
        phase = .unloading
        return true
    }

    public mutating func unloaded() {
        phase = .unloaded
        generation += 1
    }
}

public actor OnDemandModel {
    public nonisolated let model: ModelOrchestratorActor
    public nonisolated let path: String
    public nonisolated let idle: Duration
    public nonisolated let phases: AsyncStream<OnDemandCore.Phase>
    private let announce: AsyncStream<OnDemandCore.Phase>.Continuation
    private let sleep: @Sendable (Duration) async throws -> Void
    private var core = OnDemandCore()
    private var transition: Task<Result<Void, BasaltError>, Never>?
    private var timer: Task<Void, Never>?

    public init(
        model: ModelOrchestratorActor, path: String, idle: Duration,
        sleep: @escaping @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
    ) {
        (self.model, self.path, self.idle, self.sleep) = (model, path, idle, sleep)
        (phases, announce) = AsyncStream.makeStream(bufferingPolicy: .bufferingNewest(1))
    }

    public var phase: OnDemandCore.Phase { core.phase }

    public func acquire() async throws(BasaltError) -> Duration {
        let started = ContinuousClock.now
        var loads = false
        while true {
            timer?.cancel()
            timer = nil
            switch core.enter() {
            case .ready:
                return loads ? ContinuousClock.now - started : .zero
            case .wait:
                _ = await transition?.value
            case .load:
                loads = true
                let task = Task { [model, path] () async -> Result<Void, BasaltError> in
                    do throws(BasaltError) {
                        try await model.loadModel(path: path)
                        return .success(())
                    } catch {
                        return .failure(error)
                    }
                }
                transition = task
                let result = await task.value
                transition = nil
                core.loaded((try? result.get()) != nil)
                announce.yield(core.phase)
                try result.get()
                if Task.isCancelled {
                    schedule(core.generation)
                    throw .cancelled
                }
            }
        }
    }

    public func release() {
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

    func expire(_ generation: Int) async {
        guard core.expire(generation) else { return }
        let task = Task { [model] () async -> Result<Void, BasaltError> in
            await model.unloadModel()
            return .success(())
        }
        transition = task
        _ = await task.value
        transition = nil
        core.unloaded()
        announce.yield(core.phase)
    }
}
