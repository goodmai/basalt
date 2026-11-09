import Foundation

public enum RequestPriority: Int, Comparable, Sendable, CaseIterable {
    case interactive
    case toolValidation
    case standard
    case background

    public static func < (lhs: RequestPriority, rhs: RequestPriority) -> Bool { lhs.rawValue < rhs.rawValue }
}

public actor GenerationGate {
    nonisolated public let maxConcurrent: Int
    nonisolated public let maxQueue: Int

    private var running: [Int: @Sendable () -> Void] = [:]
    private var queue: [(id: Int, priority: RequestPriority, continuation: CheckedContinuation<Void, Error>)] = []
    private var cancelledBeforeQueued: Set<Int> = []
    private(set) var arriving = 0
    private var exclusiveHeld = false
    private var exclusiveQueue: [ExclusiveWaiter] = []
    private var queueTurn = false
    private var closed = false
    private var aborted = false
    private var nextId = 0
    private(set) var drainPolls = 0
    private let now: @Sendable () -> Duration
    private let sleep: @Sendable (Duration) async throws -> Void

    public init(
        maxConcurrent: Int = 1,
        maxQueue: Int = 8,
        now: @escaping @Sendable () -> Duration = { [origin = ContinuousClock.now] in ContinuousClock.now - origin },
        sleep: @escaping @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
    ) {
        self.maxConcurrent = max(1, maxConcurrent)
        self.maxQueue = max(0, maxQueue)
        self.now = now
        self.sleep = sleep
    }

    public var inFlight: Int { running.count }
    public var queued: Int { queue.count }
    public var exclusiveWaiting: Int { exclusiveQueue.count }
    public var refusal: BasaltError? { refusal(for: .standard) }

    public func refusal(for priority: RequestPriority) -> BasaltError? {
        if closed { return .serverBusy(retryAfterSeconds: nil) }
        return canRun || hasPlace(for: priority) ? nil : .serverBusy(retryAfterSeconds: 1)
    }

    private func hasPlace(for priority: RequestPriority) -> Bool {
        queue.count < maxQueue || (queue.last.map { $0.priority > priority } ?? false)
    }

    public func admit(priority: RequestPriority = .standard) async throws -> Int {
        guard !closed else { throw BasaltError.serverBusy(retryAfterSeconds: nil) }
        nextId += 1
        let id = nextId
        if canRun {
            running[id] = {}
            queueTurn = false
            return id
        }
        guard hasPlace(for: priority) else { throw BasaltError.serverBusy(retryAfterSeconds: 1) }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                if cancelledBeforeQueued.remove(id) != nil {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                if queue.count >= maxQueue {
                    queue.removeLast().continuation.resume(throwing: BasaltError.serverBusy(retryAfterSeconds: 1))
                }
                let index = queue.firstIndex { $0.priority > priority } ?? queue.endIndex
                queue.insert((id, priority, continuation), at: index)
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
        return id
    }

    public func arrive() {
        arriving += 1
    }

    public func arrived() {
        arriving = max(0, arriving - 1)
    }

    public func onCancel(_ id: Int, _ cancel: @escaping @Sendable () -> Void) {
        guard running[id] != nil else { return }
        if aborted {
            cancel()
            return
        }
        running[id] = cancel
    }

    public func release(_ id: Int) {
        guard running.removeValue(forKey: id) != nil else { return }
        promote()
        assert(checkInvariants().isEmpty)
    }

    func checkInvariants(idle: Bool = false) -> [String] {
        var problems: [String] = []
        if running.count > maxConcurrent { problems.append("\(running.count) running for \(maxConcurrent) slots") }
        if queue.count > maxQueue { problems.append("\(queue.count) queued for \(maxQueue) places") }
        if exclusiveHeld, !running.isEmpty { problems.append("held exclusively while \(running.count) run") }
        if canRun, !queue.isEmpty { problems.append("\(queue.count) wait while a slot is free") }
        if idle, !running.isEmpty || !queue.isEmpty || exclusiveHeld || !exclusiveQueue.isEmpty || arriving > 0 {
            problems.append("not idle: \(running.count) running, \(queue.count) queued, \(arriving) arriving, exclusive \(exclusiveHeld), \(exclusiveQueue.count) waiting for it")
        }
        return problems
    }

    public func exclusively<T: Sendable>(_ body: @Sendable () async throws -> T) async rethrows -> T {
        await holdExclusively()
        defer {
            exclusiveHeld = false
            promote()
        }
        return try await body()
    }

    public func admit(after body: @Sendable () async throws -> Void) async throws -> Int {
        if exclusiveHeld || !running.isEmpty, exclusiveQueue.count >= maxQueue { throw BasaltError.serverBusy(retryAfterSeconds: 1) }
        await holdExclusively()
        defer {
            exclusiveHeld = false
            promote()
        }
        guard !closed else { throw BasaltError.serverBusy(retryAfterSeconds: nil) }
        try Task.checkCancellation()
        try await body()
        nextId += 1
        running[nextId] = {}
        return nextId
    }

    private func holdExclusively() async {
        if exclusiveHeld || !running.isEmpty {
            await withCheckedContinuation { exclusiveQueue.append(.plain($0)) }
        } else {
            exclusiveHeld = true
            queueTurn = true
        }
    }

    public func idle(within limit: Duration) async -> Bool {
        let end = now() + limit
        while !running.isEmpty || !queue.isEmpty || arriving > 0 || exclusiveHeld || !exclusiveQueue.isEmpty {
            guard now() < end, (try? await sleep(.milliseconds(20))) != nil else { return false }
        }
        return true
    }

    public func shutdown(deadline: Duration) async {
        closed = true
        let waiting = queue
        queue.removeAll()
        for waiter in waiting { waiter.continuation.resume(throwing: BasaltError.serverBusy(retryAfterSeconds: nil)) }
        failBoundedWaiters()
        let end = now() + deadline
        await withTaskCancellationShield {
            while !running.isEmpty, now() < end {
                drainPolls += 1
                try? await sleep(.milliseconds(20))
            }
        }
        aborted = true
        for cancel in running.values { cancel() }
    }

    private var canRun: Bool {
        !exclusiveHeld && (exclusiveQueue.isEmpty || queueTurn) && running.count < maxConcurrent
    }

    private func promote() {
        if !exclusiveHeld, running.isEmpty, !exclusiveQueue.isEmpty, !queueTurn || queue.isEmpty {
            exclusiveHeld = true
            queueTurn = true
            exclusiveQueue.removeFirst().resume()
            return
        }
        var granted = false
        while canRun, !queue.isEmpty {
            let next = queue.removeFirst()
            running[next.id] = {}
            next.continuation.resume()
            granted = true
        }
        if granted { queueTurn = false }
    }

    private func cancelWaiter(_ id: Int) {
        if let index = queue.firstIndex(where: { $0.id == id }) {
            queue.remove(at: index).continuation.resume(throwing: CancellationError())
        } else if running[id] == nil {
            cancelledBeforeQueued.insert(id)
        }
    }
}

enum ExclusiveWaiter {
    case plain(CheckedContinuation<Void, Never>)
    case bounded(Int, CheckedContinuation<Void, Error>)

    var boundedId: Int? {
        guard case .bounded(let id, _) = self else { return nil }
        return id
    }

    func resume() {
        switch self {
        case .plain(let continuation): continuation.resume()
        case .bounded(_, let continuation): continuation.resume()
        }
    }

    func fail(_ error: any Error) {
        guard case .bounded(_, let continuation) = self else { return }
        continuation.resume(throwing: error)
    }
}

extension GenerationGate {
    public func boundedExclusively<T: Sendable>(_ body: @Sendable () async throws -> T) async throws -> T {
        guard !closed else { throw BasaltError.serverBusy(retryAfterSeconds: nil) }
        if exclusiveHeld || !running.isEmpty {
            guard exclusiveQueue.count < maxQueue else { throw BasaltError.serverBusy(retryAfterSeconds: 1) }
            try await waitExclusively()
        } else {
            exclusiveHeld = true
            queueTurn = true
        }
        defer {
            exclusiveHeld = false
            promote()
        }
        guard !closed else { throw BasaltError.serverBusy(retryAfterSeconds: nil) }
        try Task.checkCancellation()
        return try await body()
    }

    private func waitExclusively() async throws {
        nextId += 1
        let id = nextId
        defer { cancelledBeforeQueued.remove(id) }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                if cancelledBeforeQueued.remove(id) != nil {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                exclusiveQueue.append(.bounded(id, continuation))
            }
        } onCancel: {
            Task { await self.cancelExclusive(id) }
        }
    }

    private func cancelExclusive(_ id: Int) {
        guard let index = exclusiveQueue.firstIndex(where: { $0.boundedId == id }) else {
            cancelledBeforeQueued.insert(id)
            return
        }
        exclusiveQueue.remove(at: index).fail(CancellationError())
        promote()
    }

    func failBoundedWaiters() {
        let bounded = exclusiveQueue.filter { $0.boundedId != nil }
        exclusiveQueue.removeAll { $0.boundedId != nil }
        for waiter in bounded { waiter.fail(BasaltError.serverBusy(retryAfterSeconds: nil)) }
    }
}
