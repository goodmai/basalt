import Foundation

public struct ModelRoPEStatus: Sendable, Equatable, Codable {
    public let factor: Double?
    public let nativeContext: Int?
    public let effectiveContext: Int?
    public let loadWarnings: [String]

    public static let empty = ModelRoPEStatus(
        factor: nil,
        nativeContext: nil,
        effectiveContext: nil,
        loadWarnings: []
    )

    public init(
        factor: Double? = nil,
        nativeContext: Int? = nil,
        effectiveContext: Int? = nil,
        loadWarnings: [String] = []
    ) {
        self.factor = factor
        self.nativeContext = nativeContext
        self.effectiveContext = effectiveContext
        self.loadWarnings = loadWarnings
    }

    public var ropeLabel: String? {
        guard let factor, factor > 1, factor.isFinite else { return nil }
        return "yarn ×\(Self.formatFactor(factor))"
    }

    public func contextOverflowWarning(promptTokens: Int, maxTokens: Int) -> String? {
        guard let limit = effectiveContext, limit > 0 else { return nil }
        let requested = promptTokens + max(0, maxTokens)
        guard requested > limit else { return nil }
        let mode = ropeLabel ?? "stock RoPE"
        return "Requested context \(promptTokens) prompt + \(max(0, maxTokens)) max_tokens = \(requested) exceeds effective native context \(limit) (\(mode))"
    }

    static func formatFactor(_ value: Double) -> String {
        let rounded = value.rounded()
        if abs(value - rounded) < 1e-9, rounded >= 1, rounded <= 1_000_000 {
            return String(Int(rounded))
        }
        return String(format: "%g", value)
    }
}

extension ModelOrchestratorActor {
    public var ropeStatus: ModelRoPEStatus {
        get async { await engine.ropeStatus }
    }
}
