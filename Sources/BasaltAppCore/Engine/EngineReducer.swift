import Foundation

public struct EngineState: Sendable, Equatable {
    public enum Phase: Sendable, Equatable {
        case stopped
        case spawning
        case loading(port: Int)
        case ready(port: Int)
        case backingOff(attempt: Int)
        case failed(log: [String])
    }

    public var phase = Phase.stopped
    public var epoch = 0
    public var crashes: [Duration] = []

    public init() {}

    public var isIdle: Bool {
        switch phase {
        case .stopped, .failed: true
        case .spawning, .loading, .ready, .backingOff: false
        }
    }

    public var port: Int? {
        switch phase {
        case .loading(let port), .ready(let port): port
        case .stopped, .spawning, .backingOff, .failed: nil
        }
    }
}

public enum EngineEvent: Sendable, Equatable {
    case start
    case stop
    case handshake(epoch: Int, port: Int)
    case ready(epoch: Int)
    case exited(epoch: Int, at: Duration, log: [String])
    case retry(epoch: Int)
}

public enum EngineEffect: Sendable, Equatable {
    case spawn(epoch: Int)
    case kill(epoch: Int)
    case probe(epoch: Int, port: Int)
    case schedule(epoch: Int, after: Duration)
    case writeRecord(port: Int)
    case removeRecord
}

public enum EngineReducer {
    public static let crashWindow = Duration.seconds(60)
    public static let crashLimit = 3
    public static let longestBackoff = 30

    public static func backoff(attempt: Int) -> Duration {
        .seconds(min(longestBackoff, 1 << min(max(attempt - 1, 0), 10)))
    }

    public static func reduce(_ state: inout EngineState, _ event: EngineEvent) -> [EngineEffect] {
        switch event {
        case .start:
            return state.isIdle ? spawn(&state) : []
        case .stop:
            guard state.phase != .stopped else { return [] }
            let running = state.epoch
            (state.phase, state.epoch, state.crashes) = (.stopped, state.epoch + 1, [])
            return [.kill(epoch: running), .removeRecord]
        case .handshake(let epoch, let port):
            guard epoch == state.epoch, state.phase == .spawning else { return [] }
            state.phase = .loading(port: port)
            return [.writeRecord(port: port), .probe(epoch: epoch, port: port)]
        case .ready(let epoch):
            guard epoch == state.epoch, case .loading(let port) = state.phase else { return [] }
            state.phase = .ready(port: port)
            return []
        case .exited(let epoch, let at, let log):
            guard epoch == state.epoch, state.phase != .stopped else { return [] }
            return crashed(&state, at: at, log: log)
        case .retry(let epoch):
            guard epoch == state.epoch, case .backingOff = state.phase else { return [] }
            return spawn(&state)
        }
    }

    static func crashed(_ state: inout EngineState, at: Duration, log: [String]) -> [EngineEffect] {
        state.crashes = state.crashes.filter { at - $0 < crashWindow } + [at]
        guard state.crashes.count < crashLimit else {
            (state.phase, state.crashes) = (.failed(log: log), [])
            return [.removeRecord]
        }
        state.phase = .backingOff(attempt: state.crashes.count)
        return [.removeRecord, .schedule(epoch: state.epoch, after: backoff(attempt: state.crashes.count))]
    }

    static func spawn(_ state: inout EngineState) -> [EngineEffect] {
        (state.phase, state.epoch) = (.spawning, state.epoch + 1)
        return [.spawn(epoch: state.epoch)]
    }
}
