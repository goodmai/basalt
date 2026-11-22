import Foundation

public struct InklingProfile: ModelProfile {
    public static let namedEfforts: Set<String> = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]

    public let name = "inkling2"

    public init() {}

    public func templateContext(for reasoning: ReasoningPolicy) -> [String: any Sendable] {
        switch reasoning {
        case .modelDefault:
            return [:]
        case .disabled:
            return ["reasoning_effort": "none"]
        case .effort(let level):
            if Self.namedEfforts.contains(level) { return ["reasoning_effort": level] }
            guard let number = Double(level), (0 ... 0.99).contains(number) else { return [:] }
            return ["reasoning_effort": number]
        }
    }

    public var thinkingMarkers: ThinkingMarkers { .none }

    public var defaultMaxTokens: Int { 32_768 }

    public var defaultMinP: Float? { nil }
}
