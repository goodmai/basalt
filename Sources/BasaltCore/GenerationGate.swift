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
    private var exclusiveHeld = false
    private var exclusiveQueue: [CheckedContinuation<Void, Never>] = []
    private var closed = false
    private var nextId = 0
    private(set) var drainPolls = 0

    public init(maxConcurrent: Int = 1, maxQueue: Int = 8) {
        self.maxConcurrent = max(1, maxConcurrent)
        self.maxQueue = max(0, maxQueue)
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

    public func onCancel(_ id: Int, _ cancel: @escaping @Sendable () -> Void) {
        guard running[id] != nil else { return }
        running[id] = cancel
    }

    public func release(_ id: Int) {
        guard running.removeValue(forKey: id) != nil else { return }
        promote()
    }

    public func exclusively<T: Sendable>(_ body: @Sendable () async throws -> T) async rethrows -> T {
        if exclusiveHeld || !running.isEmpty {
            await withCheckedContinuation { exclusiveQueue.append($0) }
        } else {
            exclusiveHeld = true
        }
        defer {
            exclusiveHeld = false
            promote()
        }
        return try await body()
    }

    public func shutdown(deadline: Duration) async {
        closed = true
        let waiting = queue
        queue.removeAll()
        for waiter in waiting { waiter.continuation.resume(throwing: BasaltError.serverBusy(retryAfterSeconds: nil)) }
        let clock = ContinuousClock()
        let end = clock.now + deadline
        await withTaskCancellationShield {
            while !running.isEmpty, clock.now < end {
                drainPolls += 1
                try? await Task.sleep(for: .milliseconds(20))
            }
        }
        for cancel in running.values { cancel() }
    }

    private var canRun: Bool {
        !exclusiveHeld && exclusiveQueue.isEmpty && running.count < maxConcurrent
    }

    private func promote() {
        if !exclusiveHeld, running.isEmpty, !exclusiveQueue.isEmpty {
            exclusiveHeld = true
            exclusiveQueue.removeFirst().resume()
            return
        }
        while canRun, !queue.isEmpty {
            let next = queue.removeFirst()
            running[next.id] = {}
            next.continuation.resume()
        }
    }

    private func cancelWaiter(_ id: Int) {
        if let index = queue.firstIndex(where: { $0.id == id }) {
            queue.remove(at: index).continuation.resume(throwing: CancellationError())
        } else if running[id] == nil {
            cancelledBeforeQueued.insert(id)
        }
    }
}
