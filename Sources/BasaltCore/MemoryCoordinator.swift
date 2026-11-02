import Foundation

public enum MemoryPressure: String, Sendable, Codable {
    case normal, warning, critical

    public init(kernelLevel: Int32) {
        switch kernelLevel {
        case 4...: self = .critical
        case 2...: self = .warning
        default: self = .normal
        }
    }
}

public struct MemorySample: Sendable, Equatable {
    public let pressure: MemoryPressure
    public let availableBytes: Int64

    public init(pressure: MemoryPressure, availableBytes: Int64) {
        self.pressure = pressure
        self.availableBytes = availableBytes
    }
}

public protocol MemoryPressureSource: Sendable {
    func sample() -> MemorySample
}

public struct SystemMemoryPressure: MemoryPressureSource {
    public init() {}

    public func sample() -> MemorySample {
        var level: Int32 = 0
        var size = MemoryLayout<Int32>.size
        let read = sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &size, nil, 0) == 0
        return MemorySample(
            pressure: read ? MemoryPressure(kernelLevel: level) : .normal,
            availableBytes: SystemProfiler.availableSystemMemory())
    }
}

public enum MemoryDirective: Int, Sendable, Codable, Comparable {
    case relaxed, restoring, tight, critical

    public static let longContextTokens = 8_192
    public static let retryAfterSeconds = 30

    public var cacheLimitShare: Double {
        switch self {
        case .relaxed: 1
        case .restoring: 0.5
        case .tight: 0.25
        case .critical: 0.125
        }
    }

    public var promptCacheShare: Double {
        switch self {
        case .relaxed: 1
        case .restoring: 0.5
        case .tight: 0.25
        case .critical: 0
        }
    }

    public var clearsCache: Bool { self >= .tight }
    public var limitsAdmission: Bool { self == .critical }

    public static func < (lhs: MemoryDirective, rhs: MemoryDirective) -> Bool { lhs.rawValue < rhs.rawValue }
}

public struct MemoryState: Sendable, Equatable {
    public let sample: MemorySample
    public let directive: MemoryDirective
}

public struct MemoryLimits: Sendable, Equatable, Codable {
    public let cacheLimitBytes: Int?
    public let promptCacheBudgetBytes: Int
    public let promptCacheEvictions: Int

    public init(cacheLimitBytes: Int?, promptCacheBudgetBytes: Int, promptCacheEvictions: Int) {
        self.cacheLimitBytes = cacheLimitBytes
        self.promptCacheBudgetBytes = promptCacheBudgetBytes
        self.promptCacheEvictions = promptCacheEvictions
    }
}

public actor MemoryCoordinator {
    public static let restoreAfter: Duration = .seconds(30)
    public static let interval: Duration = .seconds(1)

    public nonisolated let changes: AsyncStream<MemoryState>
    private let continuation: AsyncStream<MemoryState>.Continuation
    private let source: any MemoryPressureSource
    private let now: @Sendable () -> Duration
    private let sleep: @Sendable (Duration) async throws -> Void
    public private(set) var state = MemoryState(sample: MemorySample(pressure: .normal, availableBytes: 0), directive: .relaxed)
    private var calmSince: Duration?

    public init(
        source: some MemoryPressureSource = SystemMemoryPressure(),
        now: @escaping @Sendable () -> Duration = { [origin = ContinuousClock.now] in ContinuousClock.now - origin },
        sleep: @escaping @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
    ) {
        (changes, continuation) = AsyncStream.makeStream(bufferingPolicy: .bufferingNewest(1))
        self.source = source
        self.now = now
        self.sleep = sleep
    }

    @discardableResult
    public func sample() -> MemoryState {
        let sample = source.sample()
        let directive = directive(after: sample.pressure, at: now())
        let changed = directive != state.directive
        state = MemoryState(sample: sample, directive: directive)
        if changed { continuation.yield(state) }
        return state
    }

    public func run() async {
        defer { continuation.finish() }
        while true {
            do {
                try await sleep(Self.interval)
            } catch {
                return
            }
            sample()
        }
    }

    private func directive(after pressure: MemoryPressure, at time: Duration) -> MemoryDirective {
        let current = state.directive
        switch pressure {
        case .critical:
            calmSince = nil
            return .critical
        case .warning:
            calmSince = nil
            return max(current, .tight)
        case .normal:
            guard let eased = MemoryDirective(rawValue: current.rawValue - 1) else { return current }
            guard let since = calmSince else {
                calmSince = time
                return current
            }
            guard time - since >= Self.restoreAfter else { return current }
            calmSince = time
            return eased
        }
    }
}
