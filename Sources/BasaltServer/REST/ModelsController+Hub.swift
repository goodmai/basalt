import BasaltCore
import Foundation
import HTTPTypes
import Hummingbird
import Synchronization

package struct ModelHubOperations: Sendable {
    package typealias SearchClosure = @Sendable (String, Int) async throws -> [HFModelInfo]
    package typealias PublishClosure = @Sendable (URL, HubPublishRequest, @escaping HuggingFaceHub.Progress) async throws -> HubPublishResult

    package var cacheRoot: URL
    package var converter: (any ModelConversionService)?
    package var conversionCoordinator: ConversionCoordinator
    package var search: SearchClosure = { query, limit in
        try await HuggingFaceHub().searchCatalog(query: query, limit: limit)
    }
    package var publish: PublishClosure = { dir, req, onFile in
        let opts = HubPublishOptions(repoId: req.repoId, isPrivate: req.isPrivate, commitMessage: req.commitMessage)
        return try await HuggingFaceHub().publishDirectory(at: dir, options: opts, onFile: onFile)
    }
    package var download: @Sendable (String, @escaping HuggingFaceHub.Progress) async throws -> URL

    package init(
        cacheRoot: URL = ModelCache.root,
        converter: (any ModelConversionService)? = nil,
        conversionCoordinator: ConversionCoordinator = ConversionCoordinator(),
        download: @escaping @Sendable (String, @escaping HuggingFaceHub.Progress) async throws -> URL
    ) {
        self.cacheRoot = cacheRoot
        self.converter = converter
        self.conversionCoordinator = conversionCoordinator
        self.download = download
    }

    package func withCatalogAndPublish(
        search: @escaping SearchClosure,
        publish: @escaping PublishClosure
    ) -> ModelHubOperations {
        var copy = self
        copy.search = search
        copy.publish = publish
        return copy
    }

    package static func live(
        hub: HuggingFaceHub = HuggingFaceHub(),
        cacheRoot: URL = ModelCache.root,
        converter: (any ModelConversionService)? = nil
    ) -> ModelHubOperations {
        ModelHubOperations(cacheRoot: cacheRoot, converter: converter) { repoId, onFile in
            try await hub.download(repoId: repoId, onFile: onFile)
        }
        .withCatalogAndPublish(
            search: { query, limit in
                try await hub.searchCatalog(query: query, limit: limit)
            },
            publish: { dir, req, onFile in
                let opts = HubPublishOptions(repoId: req.repoId, isPrivate: req.isPrivate, commitMessage: req.commitMessage)
                return try await hub.publishDirectory(at: dir, options: opts, onFile: onFile)
            }
        )
    }
}

struct ModelsHubController: Sendable {
    static let bodyLimit = 64 * 1024

    struct DownloadRequest: Decodable {
        var model: String
        var stream: Bool?
    }

    struct ProgressEvent: Encodable {
        let model: String
        let done: Int64
        let total: Int64
        let bytesPerSecond: Int64
        let etaSeconds: Int?

        enum CodingKeys: String, CodingKey {
            case model
            case done
            case total
            case bytesPerSecond = "bytes_per_second"
            case etaSeconds = "eta_seconds"
        }
    }

    let orchestrator: any ModelOrchestrating
    let operations: ModelHubOperations

    func add(to group: RouterGroup<BasicRequestContext>) {
        group.get("/models/hub", use: searchHubModels)
        group.post("/models/download", use: downloadModel)
        group.post("/models/publish", use: publishModel)
        group.delete("/models/{id}", use: deleteModel)
        group.delete("/models/{id}/{name}", use: deleteModel)
    }

    @Sendable
    func downloadModel(request: Request, context: BasicRequestContext) async throws -> Response {
        let buffer = try await request.body.collect(upTo: Self.bodyLimit)
        let data = Data(buffer.readableBytesView)
        guard let dto = try? JSONDecoder().decode(DownloadRequest.self, from: data), !dto.model.isEmpty else {
            return ProviderErrorResponse.response(
                status: .badRequest, message: "Request body must be JSON with a non-empty \"model\" field.", shape: .openAI)
        }
        guard let hfId = ModelsController.restModel(from: dto.model).repoId else {
            return ProviderErrorResponse.response(status: .badRequest, message: ModelsController.repoIdRequired, shape: .openAI)
        }
        if dto.stream == false {
            return await downloadJSON(hfId)
        }
        return streamDownload(hfId)
    }

    private func downloadJSON(_ hfId: String) async -> Response {
        do {
            _ = try await operations.download(hfId) { _, _, _ in }
            return Self.json([
                "id": ModelsController.claudeLocalId(from: hfId),
                "display_name": hfId,
                "object": "model",
                "status": "downloaded",
            ])
        } catch {
            return ProviderErrorResponse.response(
                status: .badGateway, message: (error as? LocalizedError)?.errorDescription ?? error.localizedDescription, shape: .openAI)
        }
    }

    func streamDownload(_ hfId: String) -> Response {
        let operations = operations
        let body = AsyncStream<ByteBuffer> { continuation in
            let task = Task {
                let throttle = Mutex(HubProgressThrottle())
                let encoder = JSONEncoder()
                do {
                    _ = try await operations.download(hfId) { file, done, total in
                        let nowNs = DispatchTime.now().uptimeNanoseconds
                        let telemetry = throttle.withLock { state in
                            state.record(file: file, done: done, total: total, at: nowNs)
                        }
                        if let telemetry,
                           let payload = try? encoder.encode(
                               ProgressEvent(
                                   model: hfId,
                                   done: telemetry.done,
                                   total: telemetry.total,
                                   bytesPerSecond: telemetry.bytesPerSecond,
                                   etaSeconds: telemetry.etaSeconds
                               )
                           )
                        {
                            continuation.yield(ByteBuffer(string: "event: progress\ndata: \(String(decoding: payload, as: UTF8.self))\n\n"))
                        }
                    }
                    let doneData = (try? JSONSerialization.data(withJSONObject: ["model": hfId, "status": "downloaded"])) ?? Data()
                    continuation.yield(ByteBuffer(string: "event: done\ndata: \(String(decoding: doneData, as: UTF8.self))\n\n"))
                } catch {
                    let msg = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
                    let errData = (try? JSONSerialization.data(withJSONObject: ["error": ["message": msg]])) ?? Data()
                    continuation.yield(ByteBuffer(string: "event: error\ndata: \(String(decoding: errData, as: UTF8.self))\n\n"))
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        return Response(status: .ok, headers: [.contentType: "text/event-stream", .cacheControl: "no-cache"], body: ResponseBody(asyncSequence: body))
    }

    @Sendable
    func deleteModel(request: Request, context: BasicRequestContext) async throws -> Response {
        let first = context.parameters.get("id")?.removingPercentEncoding ?? ""
        let raw = context.parameters.get("name").map { "\(first)/\($0.removingPercentEncoding ?? $0)" } ?? first
        guard let hfId = ModelsController.restModel(from: raw).repoId else {
            return ProviderErrorResponse.response(status: .badRequest, message: ModelsController.repoIdRequired, shape: .openAI)
        }
        let loaded = ModelsController.hfRepoId(from: await orchestrator.currentModelId)
        if await orchestrator.isLoaded, loaded == hfId {
            return ProviderErrorResponse.response(status: .conflict, message: "\(hfId) is currently loaded and cannot be deleted.", shape: .openAI)
        }
        let folder = operations.cacheRoot.appending(path: "models--" + hfId.replacingOccurrences(of: "/", with: "--"))
        guard FileManager.default.fileExists(atPath: folder.path) else {
            return ProviderErrorResponse.response(status: .notFound, message: "\(hfId) is not in the local Hugging Face cache.", shape: .openAI)
        }
        do {
            try FileManager.default.removeItem(at: folder)
        } catch {
            return ProviderErrorResponse.response(status: .internalServerError, message: error.localizedDescription, shape: .openAI)
        }
        return Self.json([
            "id": ModelsController.claudeLocalId(from: hfId),
            "display_name": hfId,
            "object": "model",
            "deleted": true,
        ])
    }

    static func json(_ body: [String: Any]) -> Response {
        let data = (try? JSONSerialization.data(withJSONObject: body)) ?? Data()
        return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: ByteBuffer(bytes: data)))
    }
}
