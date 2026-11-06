import Foundation

public enum ThinkingDifficulty: String, Sendable, CaseIterable {
    case easy, medium, hard

    static let markers = [
        "prove", "proof", "derive", "step by step", "algorithm", "complexity", "optimi", "theorem",
        "debug", "refactor", "implement", "```", "\\frac", "integral", "puzzle", "edge case",
    ]

    public init(text: String) {
        let lowered = text.lowercased()
        let words = lowered.split(whereSeparator: \.isWhitespace).count
        let score = Self.markers.count { lowered.contains($0) } + (words >= 200 ? 2 : words >= 50 ? 1 : 0)
        self = score >= 3 ? .hard : score >= 1 ? .medium : .easy
    }

    public init(request: GenerationRequest) {
        guard request.tools?.isEmpty ?? true else {
            self = .hard
            return
        }
        self.init(text: request.messages?.last { $0.role == .user }?.text ?? request.prompt)
    }

    public var thinkingBudget: Int? {
        switch self {
        case .easy: 512
        case .medium: 2048
        case .hard: nil
        }
    }

    public static func adaptiveBudget(for request: GenerationRequest) -> (difficulty: ThinkingDifficulty, budget: Int)? {
        guard request.thinkingBudget == nil, request.reasoningEffort == nil, !request.requestDisablesThinking,
              let maxTokens = request.maxTokens else { return nil }
        let difficulty = ThinkingDifficulty(request: request)
        let overhead = GenerationRequest.thinkingAnswerReserve(maxTokens: maxTokens) + GenerationRequest.thinkingTransitionHeadroom
        guard let budget = difficulty.thinkingBudget, budget + overhead <= maxTokens else { return nil }
        return (difficulty, budget)
    }
}
