import BasaltCore
import Foundation

public struct HubModelSuggestion: Codable, Sendable, Equatable, Identifiable {
    public var id: String
    public var downloads: Int
    public var likes: Int
    public var quantization: String
    public var parameterSize: String

    enum CodingKeys: String, CodingKey {
        case id
        case downloads
        case likes
        case quantization
        case parameterSize = "parameter_size"
    }

    public init(id: String, downloads: Int = 0, likes: Int = 0, quantization: String = "unknown", parameterSize: String = "?") {
        self.id = id
        self.downloads = downloads
        self.likes = likes
        self.quantization = quantization
        self.parameterSize = parameterSize
    }
}

public struct PublishDraft: Sendable, Equatable {
    public var sourceId: String
    public var sourceBytes: Int64
    public var repoId: String
    public var isPrivate: Bool = false
    public var commitMessage: String

    public init(sourceId: String, sourceBytes: Int64) {
        self.sourceId = sourceId
        self.sourceBytes = sourceBytes
        let tail = sourceId.split(separator: "/").last.map(String.init) ?? "model"
        self.repoId = sourceId.hasPrefix("local/") ? "" : sourceId
        self.commitMessage = "Upload \(tail) via Basalt"
    }

    public var isValid: Bool {
        LibraryReducer.isRepoID(repoId.trimmingCharacters(in: .whitespaces))
    }
}

public struct PublishJobState: Sendable, Equatable, Identifiable {
    public var id: String { repoId }
    public var sourceId: String
    public var repoId: String
    public var done: Int64 = 0
    public var total: Int64 = 0
    public var bytesPerSecond: Int64 = 0
    public var etaSeconds: Int?
    public var active: Bool = true
    public var publishedURL: String?
    public var error: String?

    public init(sourceId: String, repoId: String) {
        self.sourceId = sourceId
        self.repoId = repoId
    }

    public var fraction: Double {
        total > 0 ? min(1.0, Double(done) / Double(total)) : 0
    }
}

public enum HubTransferFormatting {
    public static func rateAndETA(bytesPerSecond: Int64, etaSeconds: Int?) -> String? {
        guard bytesPerSecond > 0 else { return nil }
        let speed = "\(ByteCountFormatter.string(fromByteCount: bytesPerSecond, countStyle: .file))/s"
        guard let eta = etaSeconds, eta > 0 else { return speed }
        return eta >= 60 ? "\(speed) · \(eta / 60)m \(eta % 60)s" : "\(speed) · \(eta)s"
    }
}

extension LibraryReducer {
    static func reduceHubAndPublish(_ state: inout LibraryState, _ action: LibraryAction) -> [LibraryEffect] {
        switch action {
        case .searchQueryChanged, .hubSuggestionsLoaded:
            return reduceHubSearch(&state, action)
        default:
            return reducePublish(&state, action)
        }
    }

    private static func reduceHubSearch(_ state: inout LibraryState, _ action: LibraryAction) -> [LibraryEffect] {
        switch action {
        case .searchQueryChanged(let raw):
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            state.searchQuery = raw
            guard trimmed.count >= 2, !isRepoID(trimmed) else {
                state.hubSuggestions = []
                return [.cancelHubSearch]
            }
            return [.searchHub(trimmed)]
        case .hubSuggestionsLoaded(let query, let items):
            guard state.searchQuery.trimmingCharacters(in: .whitespacesAndNewlines) == query else { return [] }
            state.hubSuggestions = items
            return []
        default:
            return []
        }
    }

    private static func reducePublish(_ state: inout LibraryState, _ action: LibraryAction) -> [LibraryEffect] {
        switch action {
        case .openPublish(let model):
            state.publishDraft = PublishDraft(sourceId: model.id, sourceBytes: model.sizeBytes)
            return []
        case .updatePublishDraft(let draft):
            state.publishDraft = draft
            return []
        case .closePublish:
            state.publishDraft = nil
            return []
        case .startPublish:
            return beginPublish(&state)
        case .publishProgressed(let repoId, let telemetry):
            applyPublishProgress(&state, repoId: repoId, telemetry: telemetry)
            return []
        case .publishCompleted(let repoId, let result):
            applyPublishCompletion(&state, repoId: repoId, result: result)
            return []
        case .publishFailed(let repoId, let message):
            state.publishes[repoId]?.active = false
            state.publishes[repoId]?.error = message
            return []
        case .cancelPublish(let repoId):
            state.publishes[repoId] = nil
            return [.cancelPublish(repoId)]
        default:
            return []
        }
    }

    private static func beginPublish(_ state: inout LibraryState) -> [LibraryEffect] {
        guard let draft = state.publishDraft else { return [] }
        let targetRepo = draft.repoId.trimmingCharacters(in: .whitespaces)
        guard isRepoID(targetRepo) else {
            state.problem = "\(targetRepo) is not a Hugging Face repo id such as org/model-name"
            return []
        }
        let message = draft.commitMessage.trimmingCharacters(in: .whitespacesAndNewlines)
        let req = HubPublishRequest(
            sourceId: draft.sourceId,
            repoId: targetRepo,
            isPrivate: draft.isPrivate,
            commitMessage: message.isEmpty ? "Upload \(targetRepo) via Basalt" : draft.commitMessage
        )
        state.publishDraft = nil
        state.problem = nil
        state.publishes[targetRepo] = PublishJobState(sourceId: draft.sourceId, repoId: targetRepo)
        return [.publish(req)]
    }

    private static func applyPublishProgress(_ state: inout LibraryState, repoId: String, telemetry: HubTransferTelemetry) {
        state.publishes[repoId]?.done = telemetry.done
        state.publishes[repoId]?.total = telemetry.total
        state.publishes[repoId]?.bytesPerSecond = telemetry.bytesPerSecond
        state.publishes[repoId]?.etaSeconds = telemetry.etaSeconds
    }

    private static func applyPublishCompletion(_ state: inout LibraryState, repoId: String, result: HubPublishResult) {
        let prevTotal = state.publishes[repoId]?.total ?? 0
        state.publishes[repoId]?.active = false
        state.publishes[repoId]?.done = result.totalBytes
        state.publishes[repoId]?.total = max(prevTotal, result.totalBytes)
        state.publishes[repoId]?.publishedURL = result.url
    }
}
