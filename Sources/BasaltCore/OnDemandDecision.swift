import Foundation

public actor OnDemandDecision: DecisionEngine {
    public nonisolated let modelID: String
    public nonisolated let path: String
    public nonisolated let idle: Duration
    public nonisolated let runsBesideDecode: Bool
    public nonisolated let phases: AsyncStream<OnDemandCore.Phase>
    private let announce: AsyncStream<OnDemandCore.Phase>.Continuation
    private let loader: @Sendable () async throws -> any DecisionEngine
    private let sleep: @Sendable (Duration) async throws -> Void
    private var core = OnDemandCore()
    private var current: (any DecisionEngine)?
    private var transition: Task<Result<Void, BasaltError>, Never>?
    private var timer: Task<Void, Never>?
    public private(set) var lastUsed: Date?

    public init(
        modelID: String, path: String, idle: Duration, runsBesideDecode: Bool = false,
        loader: @escaping @Sendable () async throws -> any DecisionEngine,
        sleep: @escaping @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
    ) {
        (self.modelID, self.path, self.idle, self.runsBesideDecode) = (modelID, path, idle, runsBesideDecode)
        (self.loader, self.sleep) = (loader, sleep)
        (phases, announce) = AsyncStream.makeStream(bufferingPolicy: .bufferingNewest(1))
    }

    public var phase: OnDemandCore.Phase { core.phase }

    public func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        let engine = try await acquire()
        defer { release() }
        return try await engine.decide(request)
    }

    public func unload() async {
        timer?.cancel()
        timer = nil
        _ = await transition?.value
        if let engine = current {
            current = nil
            await engine.unload()
        }
        core.unloaded()
        announce.yield(core.phase)
    }

    public func acquire() async throws(BasaltError) -> any DecisionEngine {
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
        let loadTask = Task { () async -> Result<any DecisionEngine, BasaltError> in
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

    public func release() {
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

    func expire(_ generation: Int) async {
        guard core.expire(generation) else { return }
        let engine = current
        current = nil
        let task = Task { () async -> Result<Void, BasaltError> in
            await engine?.unload()
            return .success(())
        }
        transition = task
        _ = await task.value
        transition = nil
        core.unloaded()
        announce.yield(core.phase)
    }
}
