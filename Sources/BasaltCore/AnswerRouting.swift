import Foundation

public struct AnswerRoute: Sendable, Equatable {
    public enum Modality: String, Sendable, Equatable, CaseIterable {
        case documents, audio, text
    }

    public struct When: Sendable, Equatable {
        public var modality: Modality?
        public var decide: [String: String] = [:]
        public var promptTokensOver: Int?
        public var escalated: Bool?
        public var costLeft: Bool?

        public init(modality: Modality? = nil, decide: [String: String] = [:], promptTokensOver: Int? = nil, escalated: Bool? = nil, costLeft: Bool? = nil) {
            (self.modality, self.decide, self.promptTokensOver, self.escalated, self.costLeft) = (modality, decide, promptTokensOver, escalated, costLeft)
        }
    }

    public var when: When
    public var model: String
    public var fallback: String?

    public init(when: When = When(), model: String, fallback: String? = nil) {
        (self.when, self.model, self.fallback) = (when, model, fallback)
    }
}

public struct AnswerSituation: Sendable, Equatable {
    public static let bytesPerToken = 4

    public var modalities: Set<AnswerRoute.Modality>
    public var labels: [String: String]
    public var promptTokens: Int
    public var escalated: Bool
    public var costLeft: Set<String>

    public init(modalities: Set<AnswerRoute.Modality>, labels: [String: String], promptTokens: Int, escalated: Bool, costLeft: Set<String>) {
        (self.modalities, self.labels, self.promptTokens, self.escalated, self.costLeft) = (modalities, labels, promptTokens, escalated, costLeft)
    }

    public var perceived: Bool { modalities.contains(.documents) || modalities.contains(.audio) }

    public static func estimatedTokens(_ prompt: String) -> Int { (prompt.utf8.count + bytesPerToken - 1) / bytesPerToken }
}

public enum AnswerRouting {
    public struct Choice: Sendable, Equatable {
        public let rule: Int
        public let chain: [String]
    }

    public static func choose(_ routes: [AnswerRoute], _ situation: AnswerSituation) -> Choice? {
        guard let rule = routes.firstIndex(where: { matches($0, situation) }) else { return nil }
        let route = routes[rule]
        return Choice(rule: rule, chain: [route.model] + (route.fallback.map { [$0] } ?? []))
    }

    public static func matches(_ route: AnswerRoute, _ situation: AnswerSituation) -> Bool {
        let when = route.when
        if let modality = when.modality, !situation.modalities.contains(modality) { return false }
        if when.decide.contains(where: { situation.labels[$0.key] != $0.value }) { return false }
        if let over = when.promptTokensOver, situation.promptTokens <= over { return false }
        if let escalated = when.escalated, escalated != situation.escalated { return false }
        if let costLeft = when.costLeft, costLeft != situation.costLeft.contains(route.model) { return false }
        return true
    }
}
