import Foundation

public actor GenerationGate {
    nonisolated public let maxConcurrent: Int
    nonisolated public let maxQueue: Int

    private var running: [Int: @Sendable () -> Void] = [:]
    private var queue: [(id: Int, continuation: CheckedContinuation<Void, Error>)] = []
    private var cancelledBeforeQueued: Set<Int> = []
    private var exclusiveHeld = false
    private var exclusiveQueue: [CheckedContinuation<Void, Never>] = []
    private var closed = false
    private var nextId = 0

    public init(maxConcurrent: Int = 1, maxQueue: Int = 8) {
        self.maxConcurrent = max(1, maxConcurrent)
        self.maxQueue = max(0, maxQueue)
    }

    public var inFlight: Int { running.count }
    public var queued: Int { queue.count }
    public var refusal: BasaltError? {
        if closed { return .serverBusy(retryAfterSeconds: nil) }
        return canRun || queue.count < maxQueue ? nil : .serverBusy(retryAfterSeconds: 1)
    }

    public func admit() async throws -> Int {
        guard !closed else { throw BasaltError.serverBusy(retryAfterSeconds: nil) }
        nextId += 1
        let id = nextId
        if canRun {
            running[id] = {}
            return id
        }
        guard queue.count < maxQueue else { throw BasaltError.serverBusy(retryAfterSeconds: 1) }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                if cancelledBeforeQueued.remove(id) != nil {
                    continuation.resume(throwing: CancellationError())
                } else {
                    queue.append((id, continuation))
                }
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
        while !running.isEmpty, clock.now < end {
            try? await Task.sleep(for: .milliseconds(20))
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
