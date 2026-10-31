import Foundation
import Hummingbird
import BasaltCore

// MARK: — Route handlers for REST / A2A interface

struct GenerateController: Sendable {

    let orchestrator: ModelOrchestratorActor
    let modelId: String?
    private let logger = BasaltLogger(module: "GenerateController")

    // MARK: — POST /api/v1/generate

    @Sendable
    func generate(request: Request, context: BasicRequestContext) async throws -> Response {
        logger.trace("Handling POST /api/v1/generate")

        let dto: GenerationRequest
        do {
            let buffer = try await request.body.collect(upTo: 1024 * 1024)
            let data = Data(buffer.readableBytesView)
            dto = PromptCacheKey.scoped(try JSONDecoder().decode(GenerationRequest.self, from: data)).routed(by: request.headers).withOrigin(RequestOrigin(source: "generate", endpoint: "/api/v1/generate", stream: false))
        } catch {
            return errorResponse(status: .badRequest, message: "Invalid JSON: \(error.localizedDescription)", code: 400)
        }

        do {
            let result = try await orchestrator.generate(request: dto)
            return try jsonResponse(status: .ok, body: result)
        } catch {
            return ProviderErrorResponse.response(for: error, path: "/api/v1/generate")
        }
    }

    // MARK: — POST /api/v1/generate/stream

    @Sendable
    func generateStream(request: Request, context: BasicRequestContext) async throws -> Response {
        logger.trace("Handling POST /api/v1/generate/stream")
        
        let dto: GenerationRequest
        do {
            let buffer = try await request.body.collect(upTo: 1024 * 1024)
            let data = Data(buffer.readableBytesView)
            dto = PromptCacheKey.scoped(try JSONDecoder().decode(GenerationRequest.self, from: data)).routed(by: request.headers).withOrigin(RequestOrigin(source: "generate", endpoint: "/api/v1/generate/stream", stream: true))
        } catch {
            return errorResponse(status: .badRequest, message: "Invalid JSON: \(error.localizedDescription)", code: 400)
        }
        if let refusal = await orchestrator.admissionRefusal(for: dto) {
            return ProviderErrorResponse.response(for: refusal, path: "/api/v1/generate/stream")
        }

        let response = Response(
            status: .ok,
            headers: [
                .contentType: "text/event-stream",
                .cacheControl: "no-cache",
                .connection: "keep-alive"
            ]
        )
        let body = ResponseBody(asyncSequence: sseStream(dto: dto))
        return Response(status: response.status, headers: response.headers, body: body)
    }

    private func sseStream(dto: GenerationRequest) -> AsyncStream<ByteBuffer> {
        AsyncStream { continuation in
            let task = Task {
                do {
                    let stream = try await orchestrator.generateStream(request: dto)
                    for await chunk in stream {
                        let event = formatSSE(chunk: chunk)
                        var buffer = ByteBuffer()
                        buffer.writeString(event)
                        continuation.yield(buffer)
                    }
                    var doneBuffer = ByteBuffer()
                    doneBuffer.writeString("data: [DONE]\n\n")
                    continuation.yield(doneBuffer)
                    continuation.finish()
                } catch {
                    let errPayload: [String: Any] = ["error": ModelsController.publicMessage(for: error)]
                    if let json = try? JSONSerialization.data(withJSONObject: errPayload),
                       let str = String(data: json, encoding: .utf8) {
                        var errorBuffer = ByteBuffer()
                        errorBuffer.writeString("event: error\n")
                        errorBuffer.writeString("data: \(str)\n\n")
                        continuation.yield(errorBuffer)
                    }
                    continuation.finish()
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
    
    private func formatSSE(chunk: StreamChunk) -> String {
        // Convert chunk to JSON
        guard let data = try? JSONEncoder().encode(chunk),
              let json = String(data: data, encoding: .utf8) else {
            return "data: {\"error\": \"encoding failed\"}\n\n"
        }
        
        // SSE format: "data: <json>\n\n"
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
