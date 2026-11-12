import BasaltCore
import Foundation
import Synchronization

public struct LibraryClient: Sendable {
    public typealias Progress = @Sendable (Int64, Int64) -> Void
    public typealias Downloader = @Sendable (String, @escaping Progress) async throws -> Void

    public var models: @Sendable () async throws -> [LibraryModel]
    public var fit: @Sendable (String) async throws -> ModelFit
    public var load: @Sendable (String) async throws -> Void
    public var download: Downloader
    public var delete: @Sendable (String) throws -> Void

    public init(
        models: @escaping @Sendable () async throws -> [LibraryModel], fit: @escaping @Sendable (String) async throws -> ModelFit,
        load: @escaping @Sendable (String) async throws -> Void,
        download: @escaping Downloader,
        delete: @escaping @Sendable (String) throws -> Void
    ) {
        (self.models, self.fit, self.load, self.download, self.delete) = (models, fit, load, download, delete)
    }
}

extension LibraryClient {
    package static func live(_ endpoint: EngineEndpoint, session: URLSession = .engine, hub: HuggingFaceHub = HuggingFaceHub(), cache: URL = ModelCache.root) -> LibraryClient {
        LibraryClient(
            models: {
                let (data, response) = try await session.data(for: endpoint.request("v1/models"))
                try EngineClient.check(response, data)
                return try Self.models(data)
            },
            fit: { id in
                var components = URLComponents(url: endpoint.base.appending(path: "v1/models/preview"), resolvingAgainstBaseURL: false)
                components?.queryItems = [URLQueryItem(name: "model_id", value: id)]
                var request = endpoint.request("v1/models/preview")
                request.url = components?.url
                let (data, response) = try await session.data(for: request)
                try EngineClient.check(response, data)
                return try Self.fit(data)
            },
            load: { id in
                let body = try JSONSerialization.data(withJSONObject: ["model": id])
                var request = endpoint.request("v1/models/load", method: "POST", body: body)
                request.timeoutInterval = 600
                let (data, response) = try await session.data(for: request)
                try EngineClient.check(response, data)
            },
            download: { id, progress in
                let files = Mutex<[String: (Int64, Int64)]>([:])
                _ = try await hub.download(repoId: id) { file, done, total in
                    let sum = files.withLock { files in
                        files[file] = (done, total)
                        return files.values.reduce((Int64(0), Int64(0))) { ($0.0 + $1.0, $0.1 + $1.1) }
                    }
                    progress(sum.0, sum.1)
                }
            },
            delete: { id in
                guard LibraryReducer.isRepoID(id) else { throw EngineError.malformed("not a repo id: \(id)") }
                try FileManager.default.removeItem(at: cache.appending(path: "models--" + id.replacingOccurrences(of: "/", with: "--")))
            })
    }

    static func models(_ data: Data) throws -> [LibraryModel] {
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let entries = root?["data"] as? [[String: Any]] ?? []
        return entries.compactMap { entry in
            guard let id = entry["display_name"] as? String, LibraryReducer.isRepoID(id) else { return nil }
            return LibraryModel(id: id, sizeBytes: (entry["size_bytes"] as? NSNumber)?.int64Value ?? 0, isLoaded: entry["is_loaded"] as? Bool ?? false)
        }
    }

    static func fit(_ data: Data) throws -> ModelFit {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any], let fits = json["fits"] as? Bool else {
            throw EngineError.malformed("a model preview without fits")
        }
        func bytes(_ key: String) -> Int64 { (json[key] as? NSNumber)?.int64Value ?? 0 }
        return ModelFit(fits: fits, level: json["fit_level"] as? String ?? "", requiredBytes: bytes("required_bytes"), availableBytes: bytes("available_bytes"), warning: json["warning"] as? String)
    }
}

public struct LibraryEnvironment: Sendable {
    public var client: @Sendable () async -> LibraryClient?
    public var savePending: @Sendable ([String]) -> Void
    public var chose: @Sendable (String) async -> Void

    public init(client: @escaping @Sendable () async -> LibraryClient?, savePending: @escaping @Sendable ([String]) -> Void, chose: @escaping @Sendable (String) async -> Void = { _ in }) {
        (self.client, self.savePending, self.chose) = (client, savePending, chose)
    }
}

public enum LibraryEffects {
    public static func handler(_ environment: LibraryEnvironment) -> @Sendable (LibraryEffect) -> EffectRun<LibraryAction> {
        { effect in
            switch effect {
            case .savePending(let ids): return .run(id: "pending") { _ in environment.savePending(ids) }
            case .cancelDownload(let id): return .cancel(id: "download:\(id)")
            case .download(let id): return .run(id: "download:\(id)") { send in await download(id, environment, send) }
            case .list, .fit, .load, .delete: return .run(id: Self.id(effect)) { send in await call(effect, environment, send) }
            }
        }
    }

    static func describe(_ error: any Error) -> String { (error as? LocalizedError)?.errorDescription ?? String(describing: error) }

    static func id(_ effect: LibraryEffect) -> String {
        switch effect {
        case .fit(let id): "fit:\(id)"
        case .load: "load"
        case .delete(let id): "delete:\(id)"
        default: "list"
        }
    }

    static func download(_ id: String, _ environment: LibraryEnvironment, _ send: @escaping @Sendable (LibraryAction) async -> Void) async {
        guard let client = await environment.client() else { return await send(.downloadFailed(id, "the engine is not running")) }
        let (updates, progress) = AsyncStream.makeStream(of: (Int64, Int64).self, bufferingPolicy: .bufferingNewest(1))
        let relay = Task { for await (done, total) in updates { await send(.progressed(id, done: done, total: total)) } }
        defer { relay.cancel() }
        do {
            try await client.download(id) { done, total in progress.yield((done, total)) }
            progress.finish()
            await relay.value
            guard !Task.isCancelled else { return }
            await send(.downloaded(id))
        } catch {
            guard !Task.isCancelled else { return }
            await send(.downloadFailed(id, Self.describe(error)))
        }
    }

    static func call(_ effect: LibraryEffect, _ environment: LibraryEnvironment, _ send: @escaping @Sendable (LibraryAction) async -> Void) async {
        guard let client = await environment.client() else { return }
        do {
            switch effect {
            case .list: await send(.listed(try await client.models()))
            case .fit(let id): await send(.fitted(id, try await client.fit(id)))
            case .load(let id):
                try await client.load(id)
                await environment.chose(id)
                await send(.loaded(id))
            case .delete(let id):
                try client.delete(id)
                await send(.listed(try await client.models()))
            default: break
            }
        } catch {
            await send(.failed(Self.describe(error)))
        }
    }
}
