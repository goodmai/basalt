import Foundation

public struct ConditionalIntervention: Sendable, Codable, Equatable {
    public var action: Intervention
    public var trigger: InterventionTrigger
    public var window: Int?

    public static let windowTokens = 1...4096

    public init(action: Intervention, trigger: InterventionTrigger, window: Int? = nil) {
        self.action = action
        self.trigger = trigger
        self.window = window
    }
}

public enum InterventionTrigger: Sendable, Codable, Equatable {
    case projection(layer: Int, vector: [Float], above: Float)
    case expertRouted(layer: Int, expert: Int, gateAbove: Float)

    public var layer: Int {
        switch self {
        case .projection(let layer, _, _), .expertRouted(let layer, _, _): layer
        }
    }
}

extension Intervention {
    public var layer: Int {
        switch self {
        case .steer(let layer, _, _), .ablateHead(let layer, _), .ablateExpert(let layer, _): layer
        case .conditional(let rule): rule.action.layer
        }
    }
}

public struct TriggerGate: Sendable, Equatable {
    public let window: Int?
    public private(set) var tokens = 0
    public private(set) var first: Int?
    public private(set) var fired: [Int] = []
    public private(set) var applied = 0

    public init(window: Int?) {
        self.window = window
    }

    public mutating func step(_ conditions: [Bool]) -> [Bool] {
        let start = tokens
        tokens += conditions.count
        let active = conditions.enumerated().map { offset, condition in
            let position = start + offset
            if condition {
                fired.append(position)
                if first == nil { first = position }
            }
            guard let window else { return condition }
            return first.map { (position - $0) < window } ?? false
        }
        applied += active.filter { $0 }.count
        return active
    }

    public var rate: Float? { tokens > 0 ? Float(applied) / Float(tokens) : nil }
}
