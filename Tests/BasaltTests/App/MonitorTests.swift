import BasaltCore
import Foundation
import Synchronization
import Testing
@testable import BasaltAppCore

@Suite("Mac app Monitor tab: polls /api/v1/stats on an injected clock, stops within one interval of disappearing, shows a problem banner instead of frozen numbers (#55, #434 phase 1)")
@MainActor
struct MonitorTests {
    final class Box<Value: Sendable>: Sendable {
        let value: Mutex<Value>
        init(_ value: Value) { self.value = Mutex(value) }
    }

    actor Ticks {
        private var waiting: [CheckedContinuation<Void, any Error>] = []
        private var sleepers: [CheckedContinuation<Void, Never>] = []

        func sleep() async throws {
            try await withCheckedThrowingContinuation { waiting.append($0); for s in sleepers { s.resume() }; sleepers.removeAll() }
        }

        func waitForSleeper() async {
            guard waiting.isEmpty else { return }
            await withCheckedContinuation { sleepers.append($0) }
        }

        func tick() {
            let all = waiting
            waiting.removeAll()
            for continuation in all { continuation.resume() }
        }

        func cancelAll() {
            let all = waiting
            waiting.removeAll()
            for continuation in all { continuation.resume(throwing: CancellationError()) }
        }
    }

    nonisolated static func snapshot(_ requests: Int) -> ServerTelemetrySnapshot {
        var snapshot = ServerTelemetrySnapshot()
        snapshot.totalRequests = requests
        return snapshot
    }

    @Test("Each interval fetches once; values replace the last; disappearing stops the loop and later results are ignored")
    func polls() async {
        let (calls, ticks) = (Box(0), Ticks())
        let store = Store(MonitorState(), reduce: MonitorReducer.reduce, handle: MonitorReducer.handler(
            stats: { Self.snapshot(calls.value.withLock { $0 += 1; return $0 }) }, every: .seconds(1), sleep: { _ in try await ticks.sleep() }))
        store.send(.appeared)
        await ticks.waitForSleeper()
        #expect(store.state.snapshot?.totalRequests == 1)
        await ticks.tick()
        await ticks.waitForSleeper()
        #expect(store.state.snapshot?.totalRequests == 2)
        store.send(.disappeared)
        await ticks.cancelAll()
        await store.settle()
        #expect(calls.value.withLock { $0 } == 2 && !store.state.polling)
        store.send(.received(Self.snapshot(99)))
        #expect(store.state.snapshot?.totalRequests == 2)
    }

    @Test("An unreachable server shows the problem on the first poll, and the next good answer clears it")
    func banner() async {
        let (fails, ticks) = (Box(true), Ticks())
        let store = Store(MonitorState(), reduce: MonitorReducer.reduce, handle: MonitorReducer.handler(
            stats: {
                if fails.value.withLock({ $0 }) { throw EngineError.transport("Could not connect to the server.") }
                return Self.snapshot(5)
            }, every: .seconds(1), sleep: { _ in try await ticks.sleep() }))
        store.send(.appeared)
        await ticks.waitForSleeper()
        #expect(store.state.problem?.contains("Could not connect") == true && store.state.snapshot == nil)
        fails.value.withLock { $0 = false }
        await ticks.tick()
        await ticks.waitForSleeper()
        #expect(store.state.problem == nil && store.state.snapshot?.totalRequests == 5)
        store.send(.disappeared)
        await ticks.cancelAll()
        await store.settle()
    }
}
