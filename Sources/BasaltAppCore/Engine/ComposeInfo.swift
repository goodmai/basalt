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
    public static let defaultChatModel = "goodmai/gemma-4-12B-it-qat-4bit"

    public static func arguments(model: String, compose: String?) -> [String] {
        guard let compose = compose?.trimmingCharacters(in: .whitespaces), !compose.isEmpty else { return ["serve", "--model", model] }
        return ["serve", "--compose", compose]
    }

    private static let chatFlags: [String: [String]] = [
        "": [], "chat": [], "mtp": ["--mtp"], "moe-stream": ["--experimental-expert-streaming"],
    ]
    private static let sidecarFlags = [
        "rerank": "--rerank-model", "embed": "--embedding-model",
        "ocr": "--pipeline-ocr-model", "asr": "--pipeline-audio-model", "decide": "--decision-model",
    ]

    public static func launchURL(_ url: URL, fallbackChatModel: String = defaultChatModel) -> (model: String, arguments: [String])? {
        guard url.scheme?.lowercased() == "basalt", url.host(percentEncoded: false)?.lowercased() == "run" else { return nil }
        let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        guard let repo = items.first(where: { $0.name == "model" })?.value, LibraryReducer.isRepoID(repo) else { return nil }
        let role = items.first(where: { $0.name == "role" })?.value?.lowercased() ?? "chat"
        if let extras = chatFlags[role] { return (repo, ["serve", "--model", repo] + extras) }
        guard let flag = sidecarFlags[role] else { return nil }
        let chat = LibraryReducer.isRepoID(fallbackChatModel) ? fallbackChatModel : defaultChatModel
        return (chat, ["serve", "--model", chat, flag, repo])
    }
}
