import BasaltCore
import Foundation
import HTTPTypes
import Hummingbird
import Synchronization

package struct ModelHubOperations: Sendable {
    package var cacheRoot: URL
    package var download: @Sendable (String, @escaping HuggingFaceHub.Progress) async throws -> URL
    package var converter: (any ModelConversionService)?
    package var conversionCoordinator: ConversionCoordinator

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

    package static func live(
        hub: HuggingFaceHub = HuggingFaceHub(),
        cacheRoot: URL = ModelCache.root,
        converter: (any ModelConversionService)? = nil
    ) -> ModelHubOperations {
        ModelHubOperations(cacheRoot: cacheRoot, converter: converter) { repoId, onFile in
            try await hub.download(repoId: repoId, onFile: onFile)
        }
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
    }

    let orchestrator: any ModelOrchestrating
    let operations: ModelHubOperations

    func add(to group: RouterGroup<BasicRequestContext>) {
        group.post("/models/download", use: downloadModel)
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

    private func streamDownload(_ hfId: String) -> Response {
        let operations = operations
        let body = AsyncStream<ByteBuffer> { continuation in
            let task = Task {
                let files = Mutex<[String: (Int64, Int64)]>([:])
                let encoder = JSONEncoder()
                do {
                    _ = try await operations.download(hfId) { file, done, total in
                        let sum = files.withLock { state -> (Int64, Int64) in
                            state[file] = (done, total)
                            return state.values.reduce((Int64(0), Int64(0))) { ($0.0 + $1.0, $0.1 + $1.1) }
                        }
                        if let payload = try? encoder.encode(ProgressEvent(model: hfId, done: sum.0, total: sum.1)) {
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

    private static func json(_ body: [String: Any]) -> Response {
        let data = (try? JSONSerialization.data(withJSONObject: body)) ?? Data()
        return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: ByteBuffer(bytes: data)))
    }
}
