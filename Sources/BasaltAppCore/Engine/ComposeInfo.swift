import Foundation

public struct ComposeInfo: Sendable, Equatable, Decodable {
    public struct Model: Sendable, Equatable, Decodable {
        public var name: String
        public var source: String
        public var keep: String
        public var state: String
        public var spentToday: Double?
        public var limit: Double?
        public var refusal: String?

        enum CodingKeys: String, CodingKey {
            case name, source, keep, state, limit, refusal
            case spentToday = "spent_today"
        }

        public var isCloud: Bool { source.hasPrefix("cloud:") }
    }

    public var name: String
    public var pipelines: [String]
    public var models: [Model]
    public var spentToday: Double

    enum CodingKeys: String, CodingKey {
        case name, pipelines, models
        case spentToday = "spent_today"
    }
}

public enum EngineChoice {
    public static let modelKey = "model"
    public static let composeKey = "compose"

    public static func arguments(model: String, compose: String?) -> [String] {
        guard let compose = compose?.trimmingCharacters(in: .whitespaces), !compose.isEmpty else { return ["serve", "--model", model] }
        return ["serve", "--compose", compose]
    }
}
