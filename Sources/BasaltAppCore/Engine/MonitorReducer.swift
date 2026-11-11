import BasaltCore
import Foundation

public struct MonitorState: Sendable, Equatable {
    public var snapshot: ServerTelemetrySnapshot?
    public var problem: String?
    public var polling = false

    public init() {}
}

public enum MonitorAction: Sendable, Equatable {
    case appeared
    case disappeared
    case received(ServerTelemetrySnapshot)
    case failed(String)
}

public enum MonitorEffect: Sendable, Equatable {
    case poll
    case stop
}

public enum MonitorReducer {
    public static let pollID = "monitor-poll"

    public static func reduce(_ state: inout MonitorState, _ action: MonitorAction) -> [MonitorEffect] {
        switch action {
        case .appeared:
            guard !state.polling else { return [] }
            state.polling = true
            return [.poll]
        case .disappeared:
            guard state.polling else { return [] }
            state.polling = false
            return [.stop]
        case .received(let snapshot):
            guard state.polling else { return [] }
            (state.snapshot, state.problem) = (snapshot, nil)
            return []
        case .failed(let message):
            guard state.polling else { return [] }
            state.problem = message
            return []
        }
    }

    public static func handler(
        stats: @escaping @Sendable () async throws -> ServerTelemetrySnapshot, every interval: Duration,
        sleep: @escaping @Sendable (Duration) async throws -> Void
    ) -> @Sendable (MonitorEffect) -> EffectRun<MonitorAction> {
        { effect in
            switch effect {
            case .stop: return .cancel(id: pollID)
            case .poll:
                return .run(id: pollID) { send in
                    repeat {
                        do {
                            await send(.received(try await stats()))
                        } catch {
                            await send(.failed((error as? EngineError).map(String.init(describing:)) ?? error.localizedDescription))
                        }
                    } while (try? await sleep(interval)) != nil && !Task.isCancelled
                }
            }
        }
    }
}
