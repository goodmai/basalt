import Foundation
import Hummingbird
import BasaltCore

struct GenerateController: Sendable {

    let orchestrator: any ModelOrchestrating
    let modelId: String?
    private let logger = BasaltLogger(module: "GenerateController")

    @Sendable
    func generate(request: Request, context: BasicRequestContext) async throws -> Response {
        logger.trace("Handling POST /api/v1/generate")

        let dto: GenerationRequest
        do {
            let buffer = try await request.body.collect(upTo: 1024 * 1024)
            let data = Data(buffer.readableBytesView)
            dto = PromptCacheKey.scoped(try JSONDecoder().decode(GenerationRequest.self, from: data), headers: request.headers).routed(by: request.headers).withOrigin(RequestOrigin(source: "generate", endpoint: "/api/v1/generate", stream: false))
        } catch let httpError as any HTTPResponseError where httpError.status == .contentTooLarge {
            return ProviderErrorResponse.response(status: .contentTooLarge, message: "Request body too large.", path: "/api/v1/generate")
        } catch {
            return errorResponse(status: .badRequest, message: "Invalid JSON: \(error.localizedDescription)", code: 400)
        }

        do {
            _ = try dto.validated()
            let result = try await orchestrator.generate(request: dto)
            return try jsonResponse(status: .ok, body: result)
        } catch {
            return ProviderErrorResponse.response(for: error, path: "/api/v1/generate")
        }
    }

    @Sendable
    func generateStream(request: Request, context: BasicRequestContext) async throws -> Response {
        logger.trace("Handling POST /api/v1/generate/stream")

        let dto: GenerationRequest
        do {
            let buffer = try await request.body.collect(upTo: 1024 * 1024)
            let data = Data(buffer.readableBytesView)
            dto = PromptCacheKey.scoped(try JSONDecoder().decode(GenerationRequest.self, from: data), headers: request.headers).routed(by: request.headers).withOrigin(RequestOrigin(source: "generate", endpoint: "/api/v1/generate/stream", stream: true))
        } catch let httpError as any HTTPResponseError where httpError.status == .contentTooLarge {
            return ProviderErrorResponse.response(status: .contentTooLarge, message: "Request body too large.", path: "/api/v1/generate/stream")
        } catch {
            return errorResponse(status: .badRequest, message: "Invalid JSON: \(error.localizedDescription)", code: 400)
        }
        do {
            _ = try dto.validated()
        } catch {
            return ProviderErrorResponse.response(for: error, path: "/api/v1/generate/stream")
        }
        if let refusal = await orchestrator.admissionRefusal(for: dto) {
            return ProviderErrorResponse.response(for: refusal, path: "/api/v1/generate/stream")
        }
        let stream: AsyncStream<StreamChunk>
        do {
            stream = try await orchestrator.generateStream(request: dto)
        } catch {
            return ProviderErrorResponse.response(for: error, path: "/api/v1/generate/stream")
        }

        let response = Response(
            status: .ok,
            headers: [
                .contentType: "text/event-stream",
                .cacheControl: "no-cache",
                .connection: "keep-alive"
            ]
        )
        let body = ResponseBody(asyncSequence: sseStream(stream: stream))
        return Response(status: response.status, headers: response.headers, body: body)
    }

    private func sseStream(stream: AsyncStream<StreamChunk>) -> AsyncStream<ByteBuffer> {
        AsyncStream { continuation in
            let task = Task {
                var failed = false
                for await chunk in stream {
                    if case .metadata(let meta) = chunk, meta.finishReason == .error {
                        failed = true
                        var errorBuffer = ByteBuffer()
                        errorBuffer.writeString("event: error\ndata: {\"error\":\"generation failed\"}\n\n")
                        continuation.yield(errorBuffer)
                        continue
                    }
                    let event = formatSSE(chunk: chunk)
                    var buffer = ByteBuffer()
                    buffer.writeString(event)
                    continuation.yield(buffer)
                }
                if !failed && !Task.isCancelled {
                    var doneBuffer = ByteBuffer()
                    doneBuffer.writeString("data: [DONE]\n\n")
                    continuation.yield(doneBuffer)
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func formatSSE(chunk: StreamChunk) -> String {
        guard let data = try? JSONEncoder().encode(chunk),
              let json = String(data: data, encoding: .utf8) else {
            return "data: {\"error\": \"encoding failed\"}\n\n"
        }
        return "data: \(json)\n\n"
    }

    // MARK: — GET /api/v1/health

    @Sendable
    func health(request: Request, context: BasicRequestContext) async throws -> Response {
        let snapshot = await orchestrator.healthSnapshot(modelId: modelId)
        return try jsonResponse(status: .ok, body: snapshot)
    }

    /// GET /api/v1/stats — the serving process's telemetry (same snapshot as `serve --stats` and MCP status).
    @Sendable
    func stats(request: Request, context: BasicRequestContext) async throws -> Response {
        try jsonResponse(status: .ok, body: await orchestrator.telemetrySnapshot())
    }

    // MARK: — Helpers

    private func jsonResponse<T: Encodable>(status: HTTPResponse.Status, body: T) throws -> Response {
        let data = try JSONEncoder().encode(body)
        var buffer = ByteBuffer()
        buffer.writeBytes(data)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: buffer)
        )
    }

    private func errorResponse(status: HTTPResponse.Status, message: String, code: Int) -> Response {
        ProviderErrorResponse.response(status: status, message: message, shape: .native)
    }
}
