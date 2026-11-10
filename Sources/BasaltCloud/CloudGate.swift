import Foundation
import BasaltCore

public struct CloudGateCore: Sendable, Equatable, Codable {
    public enum Refusal: Sendable, Equatable {
        case disabled(String)
        case paused(until: Date)
        case budget(spent: Double, limit: Double)

        public var reason: String {
            switch self {
            case .disabled(let why): "disabled until restart: \(why)"
            case .paused(let until): "rate limited until \(ISO8601DateFormatter().string(from: until))"
            case .budget(let spent, let limit): String(format: "daily budget spent: $%.4f of $%.4f", spent, limit)
            }
        }
    }

    public static let defaultPause: TimeInterval = 60
    public static let total = "*"

    public var day: String
    public var spent: [String: Double] = [:]
    public var limits: [String: Double] = [:]
    public var disabled: [String: String] = [:]
    public var pausedUntil: [String: Date] = [:]

    enum CodingKeys: String, CodingKey { case day, spent }

    public init(day: String, limits: [String: Double] = [:]) {
        (self.day, self.limits) = (day, limits)
    }

    public static func day(of date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withFullDate]
        return formatter.string(from: date)
    }

    public mutating func roll(to today: String) {
        guard today != day else { return }
        (day, spent) = (today, [:])
    }

    public func refusal(_ model: String, now: Date) -> Refusal? {
        if let why = disabled[model] { return .disabled(why) }
        if let until = pausedUntil[model], until > now { return .paused(until: until) }
        for key in [model, Self.total] {
            if let limit = limits[key], spent[key, default: 0] >= limit { return .budget(spent: spent[key, default: 0], limit: limit) }
        }
        return nil
    }

    public mutating func record(_ model: String, cost: Double) {
        guard cost.isFinite, cost > 0 else { return }
        spent[model, default: 0] += cost
        spent[Self.total, default: 0] += cost
    }

    public mutating func failed(_ model: String, _ error: BasaltError, now: Date) {
        switch error {
        case .authenticationFailed(let details): disabled[model] = details
        case .serverBusy(let seconds): pausedUntil[model] = now.addingTimeInterval(TimeInterval(seconds ?? Int(Self.defaultPause)))
        default: break
        }
    }
}

public actor CloudGate {
    private var core: CloudGateCore
    private let storage: URL?
    private let now: @Sendable () -> Date

    public init(limits: [String: Double], storage: URL?, now: @escaping @Sendable () -> Date = { Date() }) {
        let today = CloudGateCore.day(of: now())
        var core = CloudGateCore(day: today, limits: limits)
        if let storage, let data = try? Data(contentsOf: storage), let saved = try? JSONDecoder().decode(CloudGateCore.self, from: data), saved.day == today {
            core.spent = saved.spent
        }
        (self.core, self.storage, self.now) = (core, storage, now)
    }

    public func refusal(_ model: String) -> CloudGateCore.Refusal? {
        core.roll(to: CloudGateCore.day(of: now()))
        return core.refusal(model, now: now())
    }

    public func record(_ model: String, cost: Double) {
        core.roll(to: CloudGateCore.day(of: now()))
        core.record(model, cost: cost)
        guard let storage, let data = try? JSONEncoder().encode(core) else { return }
        try? FileManager.default.createDirectory(at: storage.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: storage, options: .atomic)
    }

    public func failed(_ model: String, _ error: BasaltError) {
        core.failed(model, error, now: now())
    }

    public func spent(_ model: String) -> Double { core.spent[model, default: 0] }
}

public struct GuardedDecisionEngine: DecisionEngine {
    public let inner: any DecisionEngine
    public let gate: CloudGate
    public var modelID: String { inner.modelID }
    public var runsBesideDecode: Bool { inner.runsBesideDecode }
    public var onDevice: (any DecisionEngine)? { inner.onDevice }

    public init(_ inner: any DecisionEngine, gate: CloudGate) {
        (self.inner, self.gate) = (inner, gate)
    }

    public func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        if let refusal = await gate.refusal(modelID) { throw .modelInferenceError(details: "cloud decisions: \(refusal.reason)") }
        do {
            let response = try await inner.decide(request)
            await gate.record(modelID, cost: response.usage.cost ?? 0)
            return response
        } catch {
            await gate.failed(modelID, error)
            throw error
        }
    }
}
