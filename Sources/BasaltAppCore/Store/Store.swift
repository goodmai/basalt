import Foundation
import Observation

public enum EffectRun<Action: Sendable>: Sendable {
    case none
    case run(id: String, @Sendable (_ send: @escaping @Sendable (Action) async -> Void) async -> Void)
    case cancel(id: String)
}

@Observable @MainActor
public final class Store<State: Sendable, Action: Sendable, Effect: Sendable> {
    public private(set) var state: State
    @ObservationIgnored private let reduce: @Sendable (inout State, Action) -> [Effect]
    @ObservationIgnored private let handle: @Sendable (Effect) -> EffectRun<Action>
    @ObservationIgnored private var tasks: [String: (token: Int, task: Task<Void, Never>)] = [:]
    @ObservationIgnored private var issued = 0

    public init(
        _ state: State, reduce: @escaping @Sendable (inout State, Action) -> [Effect],
        handle: @escaping @Sendable (Effect) -> EffectRun<Action>
    ) {
        (self.state, self.reduce, self.handle) = (state, reduce, handle)
    }

    public func send(_ action: Action) {
        for effect in reduce(&state, action) {
            switch handle(effect) {
            case .none: break
            case .cancel(let id): tasks.removeValue(forKey: id)?.task.cancel()
            case .run(let id, let operation): start(id, operation)
            }
        }
    }

    private func start(_ id: String, _ operation: @escaping @Sendable (@escaping @Sendable (Action) async -> Void) async -> Void) {
        tasks.removeValue(forKey: id)?.task.cancel()
        issued += 1
        let token = issued
        let task = Task { [weak self] in
            await operation { [weak self] action in await self?.send(action) }
            self?.finish(id, token)
        }
        tasks[id] = (token, task)
    }

    private func finish(_ id: String, _ token: Int) {
        if tasks[id]?.token == token { tasks[id] = nil }
    }

    var running: Set<String> { Set(tasks.keys) }

    func settle() async {
        while let task = tasks.values.first?.task { await task.value }
    }
}
