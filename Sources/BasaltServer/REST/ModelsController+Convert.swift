import BasaltCore
import Foundation
import HTTPTypes
import Hummingbird

package actor ConversionCoordinator {
    private var activeJobId: UUID?

    package init() {}

    func tryBegin() -> UUID? {
        guard activeJobId == nil else { return nil }
        let id = UUID()
        activeJobId = id
        return id
    }

    func finish(_ id: UUID) {
        if activeJobId == id {
            activeJobId = nil
        }
    }
}

struct ModelsConvertController: Sendable {
    static let bodyLimit = 256 * 1024

    let operations: ModelHubOperations

    func add(to group: RouterGroup<BasicRequestContext>) {
        group.get("/models/convert/plan", use: planConversion)
        group.post("/models/convert", use: convertModel)
    }

    static let validRecipes: Set<String> = ["rtn", "sinq", "a-sinq", "moe-rw"]

    private static func isSupportedRecipe(_ raw: String?) -> Bool {
        guard let raw, !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return true }
        return validRecipes.contains(raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
    }

    @Sendable
    func planConversion(request: Request, context: BasicRequestContext) async throws -> Response {
        guard let converter = operations.converter else {
            return ProviderErrorResponse.response(
                status: .serviceUnavailable, message: "Model conversion service is not configured.", shape: .openAI)
        }
        guard let rawSource = request.uri.queryParameters.get("source"), !rawSource.isEmpty else {
            return ProviderErrorResponse.response(
                status: .badRequest, message: "Query parameter 'source' is required.", shape: .openAI)
        }
        if let rawRecipe = request.uri.queryParameters.get("recipe").map({ String($0) }),
           !Self.isSupportedRecipe(rawRecipe) {
            return ProviderErrorResponse.response(
                status: .badRequest, message: "Unsupported recipe '\(rawRecipe)'. Expected rtn, sinq, a-sinq, or moe-rw.", shape: .openAI)
        }
        let rawTarget = request.uri.queryParameters.get("target").map { String($0) } ?? "macaq:affine:4:64"
        let target: QuantizationTarget
        do {
            target = try QuantizationTarget.parse(rawTarget)
        } catch {
            return ProviderErrorResponse.response(status: .badRequest, message: error.localizedDescription, shape: .openAI)
        }
        let budgetGB = request.uri.queryParameters.get("budget_gb").flatMap { Double($0) }
        let pvTune = request.uri.queryParameters.get("pv_tune").map { $0 == "true" || $0 == "1" } ?? false
        guard let sourceId = ModelsController.restModel(from: String(rawSource)).repoId else {
            return ProviderErrorResponse.response(
                status: .badRequest, message: ModelsController.repoIdRequired, shape: .openAI)
        }
        do {
            let plan = try await converter.plan(
                source: sourceId, target: target, budgetGB: budgetGB, pvTune: pvTune, cacheRoot: operations.cacheRoot)
            let data = try JSONEncoder().encode(plan)
            return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: ByteBuffer(bytes: data)))
        } catch let err as BasaltError {
            let status: HTTPResponse.Status = if case .modelNotFound = err { .notFound } else { .unprocessableContent }
            return ProviderErrorResponse.response(status: status, message: ModelsController.publicMessage(for: err), shape: .openAI)
        } catch {
            return ProviderErrorResponse.response(status: .unprocessableContent, message: ModelsController.publicMessage(for: error), shape: .openAI)
        }
    }

    @Sendable
    func convertModel(request: Request, context: BasicRequestContext) async throws -> Response {
        guard let converter = operations.converter else {
            return ProviderErrorResponse.response(
                status: .serviceUnavailable, message: "Model conversion service is not configured.", shape: .openAI)
        }
        let buffer = try await request.body.collect(upTo: Self.bodyLimit)
        let data = Data(buffer.readableBytesView)
        let req: ModelConversionRequest
        do {
            req = try JSONDecoder().decode(ModelConversionRequest.self, from: data)
        } catch {
            return ProviderErrorResponse.response(
                status: .badRequest, message: "Invalid conversion request JSON: \(error.localizedDescription)", shape: .openAI)
        }
        guard ModelsController.isRepoId(req.outputId) else {
            return ProviderErrorResponse.response(
                status: .badRequest, message: "output_id must be a valid repo id (org/name), e.g. local/model-4bit.", shape: .openAI)
        }
        guard Self.isSupportedRecipe(req.recipe) else {
            return ProviderErrorResponse.response(
                status: .badRequest, message: "Unsupported recipe '\(req.recipe ?? "")'. Expected rtn, sinq, a-sinq, or moe-rw.", shape: .openAI)
        }
        guard let sourceId = ModelsController.restModel(from: req.source).repoId else {
            return ProviderErrorResponse.response(
                status: .badRequest, message: ModelsController.repoIdRequired, shape: .openAI)
        }
        var normalized = req
        normalized.source = sourceId
        if let rawTeacher = req.pvTeacher, !rawTeacher.isEmpty {
            guard let teacherId = ModelsController.restModel(from: rawTeacher).repoId else {
                return ProviderErrorResponse.response(
                    status: .badRequest, message: ModelsController.repoIdRequired, shape: .openAI)
            }
            normalized.pvTeacher = teacherId
        }
        if let rawAdapter = req.adapter, !rawAdapter.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return ProviderErrorResponse.response(
                status: .badRequest, message: "Local adapter paths are not permitted over REST; use `basalt models convert --adapter`.", shape: .openAI)
        }
        guard let jobId = await operations.conversionCoordinator.tryBegin() else {
            return ProviderErrorResponse.response(
                status: .conflict, message: "Another model conversion is currently running.", shape: .openAI)
        }
        if !normalized.stream {
            return await runJSON(normalized, jobId: jobId, converter: converter)
        }
        return runStream(normalized, jobId: jobId, converter: converter)
    }

    private func runJSON(
        _ req: ModelConversionRequest,
        jobId: UUID,
        converter: any ModelConversionService
    ) async -> Response {
        let coordinator = operations.conversionCoordinator
        let cacheRoot = operations.cacheRoot
        defer {
            await withTaskCancellationShield {
                await coordinator.finish(jobId)
            }
        }
        do {
            let result = try await converter.convert(req, cacheRoot: cacheRoot) { _ in }
            let encoded = try JSONEncoder().encode(result)
            return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: ByteBuffer(bytes: encoded)))
        } catch {
            let msg = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            return ProviderErrorResponse.response(status: .unprocessableContent, message: msg, shape: .openAI)
        }
    }

    private func runStream(
        _ req: ModelConversionRequest,
        jobId: UUID,
        converter: any ModelConversionService
    ) -> Response {
        let coordinator = operations.conversionCoordinator
        let cacheRoot = operations.cacheRoot
        let body = AsyncStream<ByteBuffer> { continuation in
            let task = Task {
                defer {
                    await withTaskCancellationShield {
                        await coordinator.finish(jobId)
                    }
                }
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.withoutEscapingSlashes]
                do {
                    let result = try await converter.convert(req, cacheRoot: cacheRoot) { progress in
                        if let payload = try? encoder.encode(progress) {
                            continuation.yield(ByteBuffer(string: "event: progress\ndata: \(String(decoding: payload, as: UTF8.self))\n\n"))
                        }
                    }
                    if let doneData = try? encoder.encode(result) {
                        continuation.yield(ByteBuffer(string: "event: done\ndata: \(String(decoding: doneData, as: UTF8.self))\n\n"))
                    }
                } catch {
                    let msg = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
                    let errObj: [String: Any] = ["message": msg, "error": ["message": msg]]
                    let errData = (try? JSONSerialization.data(withJSONObject: errObj)) ?? Data()
                    continuation.yield(ByteBuffer(string: "event: error\ndata: \(String(decoding: errData, as: UTF8.self))\n\n"))
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        return Response(
            status: .ok,
            headers: [.contentType: "text/event-stream", .cacheControl: "no-cache"],
            body: ResponseBody(asyncSequence: body))
    }
}
