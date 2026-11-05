import Foundation
import Synchronization

public enum ThermalLevel: String, Sendable, Codable, CaseIterable {
    case nominal, fair, serious, critical

    public init(_ state: ProcessInfo.ThermalState) {
        switch state {
        case .nominal: self = .nominal
        case .fair: self = .fair
        case .serious: self = .serious
        case .critical: self = .critical
        @unknown default: self = .nominal
        }
    }

    public static let current: @Sendable () -> ThermalLevel = { ThermalLevel(ProcessInfo.processInfo.thermalState) }
}

public enum ThermalDirective: String, Sendable, Codable {
    case normal, paced, refusing

    public static let pacedStepSize = 256
    public static let pauseShare = 0.25
    public static let retryAfterSeconds = 60

    public init(_ level: ThermalLevel) {
        switch level {
        case .nominal, .fair: self = .normal
        case .serious: self = .paced
        case .critical: self = .refusing
        }
    }

    public var prefillStepSize: Int? { self == .normal ? nil : Self.pacedStepSize }
    public var pauseShare: Double { self == .normal ? 0 : Self.pauseShare }

    public func refusal(promptTokens: Int) -> BasaltError? {
        guard self == .refusing, promptTokens >= MemoryDirective.longContextTokens else { return nil }
        return .serverBusy(retryAfterSeconds: Self.retryAfterSeconds)
    }
}

public struct ThermalTelemetry: Codable, Sendable, Equatable {
    public let level: ThermalLevel
    public let directive: ThermalDirective

    var formattedStats: String { "Thermal: \(level.rawValue), \(directive.rawValue)" }
}

public struct PrefillPacer: Sendable {
    public let directive: @Sendable () -> ThermalDirective
    let now: @Sendable () -> ContinuousClock.Instant
    let pause: @Sendable (Duration) -> Void

    public init(
        directive: @escaping @Sendable () -> ThermalDirective,
        now: @escaping @Sendable () -> ContinuousClock.Instant = { .now },
        pause: @escaping @Sendable (Duration) -> Void = { Thread.sleep(forTimeInterval: $0 / .seconds(1)) }
    ) {
        self.directive = directive
        self.now = now
        self.pause = pause
    }

    public func stepSize(under requested: Int?) -> Int? {
        guard let paced = directive().prefillStepSize else { return requested }
        return min(requested ?? paced, paced)
    }

    public func progress(chaining next: (@Sendable (Int, Int) -> Void)?) -> @Sendable (Int, Int) -> Void {
        let (directive, now, pause) = (directive, now, pause)
        let chunkStart = Mutex(now())
        return { processed, total in
            next?(processed, total)
            let share = directive().pauseShare
            if share > 0, processed < total { pause(chunkStart.withLock { now() - $0 } * share) }
            chunkStart.withLock { $0 = now() }
        }
    }
}
