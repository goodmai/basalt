import Foundation

public struct BailingHybridProfile: ModelProfile {
    public let name = "bailing_hybrid"

    public init() {}

    public func templateContext(for reasoning: ReasoningPolicy) -> [String: any Sendable] {
        switch reasoning {
        case .modelDefault:
            return [:]
        case .disabled:
            return ["enable_thinking": false]
        case .effort(let level):
            return ["enable_thinking": level != "none"]
        }
    }

    public func turns(_ turns: [ChatMessageInput], for reasoning: ReasoningPolicy) -> [ChatMessageInput] {
        let directive: String
        switch reasoning {
        case .modelDefault:
            return turns
        case .disabled:
            directive = "detailed thinking off"
        case .effort(let level):
            directive = level == "none" ? "detailed thinking off" : "detailed thinking on"
        }
        guard let first = turns.first, first.role == .system else {
            return [ChatMessageInput(role: .system, text: directive)] + turns
        }
        if first.text.contains("detailed thinking on") || first.text.contains("detailed thinking off") {
            return turns
        }
        return [ChatMessageInput(role: .system, text: first.text + "\n" + directive)] + turns.dropFirst()
    }

    public var thinkingMarkers: ThinkingMarkers {
        ThinkingMarkers(open: ["<think>"], close: ["</think>"])
    }

    public var defaultMaxTokens: Int { 16_384 }

    public var defaultMinP: Float? { nil }
}
