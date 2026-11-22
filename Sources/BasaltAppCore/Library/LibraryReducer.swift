import BasaltCore
import Foundation

public struct LibraryModel: Sendable, Equatable, Identifiable {
    public let id: String
    public var sizeBytes: Int64
    public var isLoaded: Bool

    public init(id: String, sizeBytes: Int64, isLoaded: Bool) { (self.id, self.sizeBytes, self.isLoaded) = (id, sizeBytes, isLoaded) }
}

public struct ModelFit: Sendable, Equatable {
    public var fits: Bool
    public var level: String
    public var requiredBytes: Int64
    public var availableBytes: Int64
    public var warning: String?

    public init(fits: Bool, level: String, requiredBytes: Int64, availableBytes: Int64, warning: String? = nil) {
        (self.fits, self.level, self.requiredBytes, self.availableBytes, self.warning) = (fits, level, requiredBytes, availableBytes, warning)
    }
}

public struct Download: Sendable, Equatable {
    public var done: Int64 = 0
    public var total: Int64 = 0
    public var bytesPerSecond: Int64 = 0
    public var etaSeconds: Int?
    public var active = false
    public var error: String?

    public init() {}

    public var fraction: Double { total > 0 ? Double(done) / Double(total) : 0 }
}

public struct LibraryState: Sendable, Equatable {
    public var models: [LibraryModel] = []
    public var fits: [String: ModelFit] = [:]
    public var downloads: [String: Download] = [:]
    public var conversions: [String: ConversionJobState] = [:]
    public var publishes: [String: PublishJobState] = [:]
    public var convertDraft: ConversionDraft?
    public var publishDraft: PublishDraft?
    public var searchQuery: String = ""
    public var hubSuggestions: [HubModelSuggestion] = []
    public var loading: String?
    public var problem: String?

    public init(pending: [String] = []) { downloads = Dictionary(uniqueKeysWithValues: pending.map { ($0, Download()) }) }

    public func canLoad(_ id: String) -> Bool { loading == nil && fits[id]?.fits == true && models.contains { $0.id == id && !$0.isLoaded } }
}

public enum LibraryAction: Sendable {
    case appeared
    case listed([LibraryModel])
    case fitted(String, ModelFit)
    case load(String)
    case loaded(String)
    case download(String)
    case progressed(String, done: Int64, total: Int64, bytesPerSecond: Int64 = 0, etaSeconds: Int? = nil)
    case downloaded(String)
    case downloadFailed(String, String)
    case cancelDownload(String)
    case delete(String)
    case openConvert(LibraryModel)
    case updateConvertDraft(ConversionDraft)
    case closeConvert
    case conversionPlanned(ModelConversionPlan)
    case startConversion
    case conversionProgressed(String, ModelConversionProgress)
    case conversionCompleted(String, ModelConversionResult)
    case conversionFailed(String, String)
    case cancelConversion(String)
    case searchQueryChanged(String)
    case hubSuggestionsLoaded(String, [HubModelSuggestion])
    case openPublish(LibraryModel)
    case updatePublishDraft(PublishDraft)
    case closePublish
    case startPublish
    case publishProgressed(String, HubTransferTelemetry)
    case publishCompleted(String, HubPublishResult)
    case publishFailed(String, String)
    case cancelPublish(String)
    case listFailed(String)
    case failed(String)
}

public enum LibraryEffect: Sendable, Equatable {
    case list
    case fit(String)
    case load(String)
    case download(String)
    case cancelDownload(String)
    case delete(String)
    case savePending([String])
    case planConversion(source: String, target: QuantizationTarget, budgetGB: Double?, pvTune: Bool)
    case convert(ModelConversionRequest)
    case cancelConversion(String)
    case searchHub(String)
    case cancelHubSearch
    case publish(HubPublishRequest)
    case cancelPublish(String)
}

public enum LibraryReducer {
    public static func isRepoID(_ id: String) -> Bool {
        let parts = id.split(separator: "/", omittingEmptySubsequences: false)
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        return parts.count == 2 && parts.allSatisfy { part in
            !part.isEmpty && part != "." && part != ".." && part.unicodeScalars.allSatisfy(allowed.contains)
        }
    }

    public static func reduce(_ state: inout LibraryState, _ action: LibraryAction) -> [LibraryEffect] {
        switch action {
        case .appeared:
            return [.list] + resume(&state)
        case .listed(let models):
            state.models = models
            return models.map(\.id).filter { state.fits[$0] == nil }.map(LibraryEffect.fit)
        case .fitted(let id, let fit):
            state.fits[id] = fit
            return []
        case .load(let id):
            return load(&state, id)
        case .loaded:
            state.loading = nil
            return [.list]
        case .download(let id):
            return download(&state, id)
        case .progressed, .downloaded, .downloadFailed, .cancelDownload:
            return transfer(&state, action)
        case .delete(let id):
            return delete(&state, id)
        case .openConvert, .updateConvertDraft, .closeConvert, .conversionPlanned,
             .startConversion, .conversionProgressed, .conversionCompleted,
             .conversionFailed, .cancelConversion:
            return reduceConversion(&state, action)
        case .searchQueryChanged, .hubSuggestionsLoaded, .openPublish, .updatePublishDraft,
             .closePublish, .startPublish, .publishProgressed, .publishCompleted,
             .publishFailed, .cancelPublish:
            return reduceHubAndPublish(&state, action)
        case .listFailed(let message):
            (state.problem, state.loading) = (message, nil)
            return []
        case .failed(let message):
            (state.problem, state.loading) = (message, nil)
            return [.list]
        }
    }

    static func pending(_ state: LibraryState) -> [String] { state.downloads.keys.sorted() }

    static func transfer(_ state: inout LibraryState, _ action: LibraryAction) -> [LibraryEffect] {
        switch action {
        case .progressed(let id, let done, let total, let bps, let eta):
            state.downloads[id]?.done = done
            state.downloads[id]?.total = total
            state.downloads[id]?.bytesPerSecond = bps
            state.downloads[id]?.etaSeconds = eta
            return []
        case .downloaded(let id):
            state.downloads[id] = nil
            return [.savePending(pending(state)), .list]
        case .downloadFailed(let id, let message):
            state.downloads[id]?.active = false
            state.downloads[id]?.error = message
            return []
        case .cancelDownload(let id):
            state.downloads[id] = nil
            return [.cancelDownload(id), .savePending(pending(state))]
        default:
            return []
        }
    }

    static func resume(_ state: inout LibraryState) -> [LibraryEffect] {
        let idle = state.downloads.filter { !$0.value.active && $0.value.error == nil }.keys.sorted()
        for id in idle { state.downloads[id]?.active = true }
        return idle.map(LibraryEffect.download)
    }

    static func load(_ state: inout LibraryState, _ id: String) -> [LibraryEffect] {
        guard let fit = state.fits[id], fit.fits else {
            state.problem = "\(id) does not fit in this Mac's memory" + (state.fits[id]?.warning.map { ": \($0)" } ?? "") + ", so it is not loaded"
            return []
        }
        guard state.canLoad(id) else { return [] }
        (state.loading, state.problem) = (id, nil)
        return [.load(id)]
    }

    static func download(_ state: inout LibraryState, _ id: String) -> [LibraryEffect] {
        guard isRepoID(id) else {
            state.problem = "\(id) is not a Hugging Face repo id such as mlx-community/gemma-4-12B-it-qat-4bit"
            return []
        }
        guard state.downloads[id]?.active != true else { return [] }
        var download = Download()
        download.active = true
        (state.downloads[id], state.problem, state.hubSuggestions, state.searchQuery) = (download, nil, [], "")
        return [.download(id), .fit(id), .savePending(pending(state))]
    }

    static func delete(_ state: inout LibraryState, _ id: String) -> [LibraryEffect] {
        guard isRepoID(id), state.models.contains(where: { $0.id == id && !$0.isLoaded }) else {
            state.problem = "\(id) is loaded or not in the library, so it is not deleted"
            return []
        }
        state.problem = nil
        return [.delete(id)]
    }
}
