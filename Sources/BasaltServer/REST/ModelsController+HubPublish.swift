import BasaltCore
import Foundation
import HTTPTypes
import Hummingbird
import Synchronization

extension ModelsHubController {
    struct PublishBody: Decodable {
        var source: String
        var repo: String
        var `private`: Bool?
        var message: String?
        var stream: Bool?
    }

    @Sendable
    func searchHubModels(request: Request, context: BasicRequestContext) async throws -> Response {
        let query = request.uri.queryParameters.get("q").map { String($0) } ?? ""
        let rawLimit = request.uri.queryParameters.get("limit").flatMap { Int($0) } ?? 12
        let limit = min(max(rawLimit, 1), 50)
        do {
            let items = try await operations.search(query, limit)
            let rows: [[String: Any]] = items.map { item in
                [
                    "id": item.id,
                    "downloads": item.downloads ?? 0,
                    "likes": item.likes ?? 0,
                    "quantization": item.quantization,
                    "parameter_size": item.parameterSize,
                ]
            }
            return Self.json(["object": "list", "data": rows])
        } catch {
            let msg = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            return ProviderErrorResponse.response(status: .badGateway, message: msg, shape: .openAI)
        }
    }

    @Sendable
    func publishModel(request: Request, context: BasicRequestContext) async throws -> Response {
        let buffer = try await request.body.collect(upTo: Self.bodyLimit)
        let data = Data(buffer.readableBytesView)
        guard let dto = try? JSONDecoder().decode(PublishBody.self, from: data),
              !dto.source.isEmpty, !dto.repo.isEmpty else {
            return ProviderErrorResponse.response(
                status: .badRequest,
                message: "Request body must include non-empty \"source\" and \"repo\" fields.",
                shape: .openAI
            )
        }
        guard let sourceRepoId = ModelsController.restModel(from: dto.source).repoId,
              let targetRepoId = ModelsController.restModel(from: dto.repo).repoId else {
            return ProviderErrorResponse.response(status: .badRequest, message: ModelsController.repoIdRequired, shape: .openAI)
        }
        guard let sourceDir = Self.resolveCachedSnapshot(for: sourceRepoId, in: operations.cacheRoot) else {
            return ProviderErrorResponse.response(
                status: .notFound,
                message: "\(sourceRepoId) is not in the local Hugging Face cache.",
                shape: .openAI
            )
        }
        let publishReq = HubPublishRequest(
            sourceId: sourceRepoId,
            repoId: targetRepoId,
            isPrivate: dto.private ?? false,
            commitMessage: dto.message ?? "Upload \(targetRepoId) via Basalt"
        )
        if dto.stream == false {
            return await publishJSON(sourceDir: sourceDir, req: publishReq)
        }
        return streamPublish(sourceDir: sourceDir, req: publishReq)
    }

    private func publishJSON(sourceDir: URL, req: HubPublishRequest) async -> Response {
        do {
            let result = try await operations.publish(sourceDir, req) { _, _, _ in }
            return Self.json([
                "repo": result.repoId,
                "url": result.url,
                "uploaded_files": result.uploadedFiles,
                "total_bytes": result.totalBytes,
                "status": "published",
            ])
        } catch {
            let msg = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            return ProviderErrorResponse.response(status: .badGateway, message: msg, shape: .openAI)
        }
    }

    private func streamPublish(sourceDir: URL, req: HubPublishRequest) -> Response {
        let operations = operations
        let body = AsyncStream<ByteBuffer> { continuation in
            let task = Task {
                let throttle = Mutex(HubProgressThrottle())
                let encoder = JSONEncoder()
                do {
                    let result = try await operations.publish(sourceDir, req) { file, done, total in
                        let nowNs = DispatchTime.now().uptimeNanoseconds
                        let telemetry = throttle.withLock { state in
                            state.record(file: file, done: done, total: total, at: nowNs)
                        }
                        if let telemetry,
                           let payload = try? encoder.encode(
                               ProgressEvent(
                                   model: req.repoId,
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
                    let doneObj: [String: Any] = [
                        "repo": result.repoId,
                        "url": result.url,
                        "uploaded_files": result.uploadedFiles,
                        "total_bytes": result.totalBytes,
                        "status": "published",
                    ]
                    let doneData = (try? JSONSerialization.data(withJSONObject: doneObj)) ?? Data()
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

    static func resolveCachedSnapshot(for repoId: String, in cacheRoot: URL) -> URL? {
        let sanitized = repoId.replacingOccurrences(of: "/", with: "--")
        let modelRoot = cacheRoot.appending(path: "models--\(sanitized)")
        let refsMain = modelRoot.appending(path: "refs/main")
        if let hash = try? String(contentsOf: refsMain, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
           !hash.isEmpty
        {
            let snap = modelRoot.appending(path: "snapshots/\(hash)")
            if FileManager.default.fileExists(atPath: snap.appending(path: "config.json").path) {
                return snap
            }
        }
        let mainSnap = modelRoot.appending(path: "snapshots/main")
        if FileManager.default.fileExists(atPath: mainSnap.appending(path: "config.json").path) {
            return mainSnap
        }
        let snapsDir = modelRoot.appending(path: "snapshots")
        if let entries = try? FileManager.default.contentsOfDirectory(at: snapsDir, includingPropertiesForKeys: nil) {
            for entry in entries.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                if FileManager.default.fileExists(atPath: entry.appending(path: "config.json").path) {
                    return entry
                }
            }
        }
        return nil
    }
}
