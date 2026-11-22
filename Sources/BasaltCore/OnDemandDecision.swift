import Foundation

public actor OnDemandDecision: DecisionEngine {
    private enum Transition {
        case load(Task<Result<any DecisionEngine, BasaltError>, Never>)
        case unload(Task<Void, Never>)

        func wait() async {
            switch self {
            case .load(let task): _ = await task.value
            case .unload(let task): await task.value
            }
        }
    }

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
    private var transition: Transition?
    private var timer: Task<Void, Never>?
    private var unloadEpoch = 0
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

    deinit {
        timer?.cancel()
        announce.finish()
    }

    public var phase: OnDemandCore.Phase { core.phase }

    public func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        let engine = try await acquire()
        defer { release() }
        return try await engine.decide(request)
    }

    public func unload() async {
        unloadEpoch &+= 1
        while true {
            timer?.cancel()
            timer = nil
            if let transition {
                await transition.wait()
                continue
            }
            guard core.beginUnload() else { return }
            announce.yield(core.phase)
            let engine = current
            current = nil
            let task = Task {
                await withTaskCancellationShield { await engine?.unload() }
                self.finishUnload()
            }
            transition = .unload(task)
            await task.value
            return
        }
    }

    public func acquire() async throws(BasaltError) -> any DecisionEngine {
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
                await transition?.wait()
            case .load:
                try await performLoad()
            }
        }
    }

    private func performLoad() async throws(BasaltError) {
        let epoch = unloadEpoch
        let loader = self.loader
        let task = Task { () -> Result<any DecisionEngine, BasaltError> in
            let outcome = await Self.runLoader(loader)
            await self.finishLoad(outcome, epoch: epoch)
            return outcome
        }
        transition = .load(task)
        let outcome = await task.value
        _ = try outcome.get()
        if epoch != unloadEpoch || Task.isCancelled {
            if core.phase == .loaded, core.users == 0 { schedule(core.generation) }
            throw .cancelled
        }
    }

    private static func runLoader(
        _ loader: @Sendable () async throws -> any DecisionEngine
    ) async -> Result<any DecisionEngine, BasaltError> {
        do {
            return .success(try await loader())
        } catch let error as BasaltError {
            return .failure(error)
        } catch is CancellationError {
            return .failure(.cancelled)
        } catch {
            return .failure(.modelInferenceError(details: error.localizedDescription))
        }
    }

    private func finishLoad(_ outcome: Result<any DecisionEngine, BasaltError>, epoch: Int) async {
        switch outcome {
        case .success(let engine):
            if epoch != unloadEpoch {
                await withTaskCancellationShield { await engine.unload() }
                transition = nil
                core.loaded(false)
            } else {
                current = engine
                transition = nil
                core.loaded(true)
            }
        case .failure:
            transition = nil
            core.loaded(false)
        }
        announce.yield(core.phase)
    }

    private func finishUnload() {
        transition = nil
        core.unloaded()
        announce.yield(core.phase)
    }

    public func release() {
        lastUsed = Date()
        guard let generation = core.leave() else { return }
        schedule(generation)
    }

    private func schedule(_ generation: Int) {
        timer?.cancel()
        timer = Task { [weak self, sleep, idle] in
            guard (try? await sleep(idle)) != nil else { return }
            await self?.expire(generation)
        }
    }

    func expire(_ generation: Int) async {
        guard core.expire(generation) else { return }
        announce.yield(core.phase)
        let engine = current
        current = nil
        let task = Task {
            await withTaskCancellationShield { await engine?.unload() }
            self.finishUnload()
        }
        transition = .unload(task)
        await task.value
    }
}
