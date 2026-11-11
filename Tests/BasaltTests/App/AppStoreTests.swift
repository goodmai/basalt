import Foundation
import Testing
@testable import BasaltAppCore

@Suite("Mac app core: the generic store runs the reducer, executes effects as tasks by id, cancels them and feeds their actions back (#434 phase 1)")
@MainActor
struct AppStoreTests {
    enum Action: Sendable, Equatable { case begin(Int), tick(Int), halt }
    enum Effect: Sendable, Equatable { case count(Int), cancel }

    static func store(gate: AsyncStream<Void>? = nil) -> Store<[Int], Action, Effect> {
        Store([]) { state, action in
            switch action {
            case .begin(let limit): return [.count(limit)]
            case .tick(let value):
                state.append(value)
                return []
            case .halt: return [.cancel]
            }
        } handle: { effect in
            switch effect {
            case .cancel: return .cancel(id: "count")
            case .count(let limit):
                return .run(id: "count") { send in
                    for value in 1 ... limit {
                        await send(.tick(value))
                        if let gate { for await _ in gate { break } }
                        if Task.isCancelled { return }
                    }
                }
            }
        }
    }

    @Test("An effect's actions reach the reducer in order, and the task is forgotten when it ends")
    func runs() async {
        let store = Self.store()
        store.send(.begin(3))
        #expect(store.running == ["count"])
        await store.settle()
        #expect(store.state == [1, 2, 3] && store.running.isEmpty)
    }

    @Test("Cancel stops the task with that id; starting the same id again replaces the running one")
    func cancels() async {
        let (gate, open) = AsyncStream<Void>.makeStream()
        let store = Self.store(gate: gate)
        store.send(.begin(100))
        store.send(.halt)
        #expect(store.running.isEmpty)
        open.finish()
        await store.settle()
        #expect(store.state.count <= 1)
        let replaced = Self.store()
        replaced.send(.begin(2))
        replaced.send(.begin(2))
        await replaced.settle()
        #expect(replaced.running.isEmpty && replaced.state.suffix(2) == [1, 2])
    }
}
