import Foundation
import Hummingbird

// MARK: — Route handlers for REST / A2A interface

struct GenerateController: Sendable {

    let orchestrator: ModelOrchestratorActor
    let modelId: String?
    private let logger = GemLogger(module: "GenerateController")

    // MARK: — POST /api/v1/generate

    @Sendable
    func generate(request: Request, context: BasicRequestContext) async throws -> Response {
        logger.trace("Handling POST /api/v1/generate")

        let dto: GenerationRequest
        do {
            let buffer = try await request.body.collect(upTo: 1024 * 1024)
            let data = Data(buffer.readableBytesView)
            dto = try JSONDecoder().decode(GenerationRequest.self, from: data)
        } catch {
            return errorResponse(status: .badRequest, message: "Invalid JSON: \(error.localizedDescription)", code: 400)
        }

        do {
            let result = try await orchestrator.generate(request: dto)
            return try jsonResponse(status: .ok, body: result)
        } catch let err as GemError {
            return errorResponse(
                status: HTTPResponse.Status(code: err.httpStatus),
                message: err.errorDescription ?? err.localizedDescription,
                code: err.httpStatus
            )
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
            dto = try JSONDecoder().decode(GenerationRequest.self, from: data)
        } catch {
            return errorResponse(status: .badRequest, message: "Invalid JSON: \(error.localizedDescription)", code: 400)
        }
        
        // Create Server-Sent Events (SSE) response
        let response = Response(
            status: .ok,
            headers: [
                .contentType: "text/event-stream",
                .cacheControl: "no-cache",
                .connection: "keep-alive"
            ]
        )
        
        // Create streaming body
        let body = ResponseBody(asyncSequence: sseStream(dto: dto))
        
        return Response(
            status: response.status,
            headers: response.headers,
            body: body
        )
    }
    
    // MARK: — SSE Stream Helper
    
    private func sseStream(dto: GenerationRequest) -> AsyncStream<ByteBuffer> {
        AsyncStream { continuation in
            Task {
                do {
                    let stream = try await orchestrator.generateStream(request: dto)
                    
                    for await chunk in stream {
                        // Format as Server-Sent Event
                        let event = formatSSE(chunk: chunk)
                        var buffer = ByteBuffer()
                        buffer.writeString(event)
                        continuation.yield(buffer)
                    }
                    
                    // Send final [DONE] message
                    var doneBuffer = ByteBuffer()
                    doneBuffer.writeString("data: [DONE]\n\n")
                    continuation.yield(doneBuffer)
                    
                    continuation.finish()
                } catch {
                    // Send error event
                    var errorBuffer = ByteBuffer()
                    let errorJSON = "{\"error\": \"\(error.localizedDescription)\"}"
                    errorBuffer.writeString("event: error\n")
                    errorBuffer.writeString("data: \(errorJSON)\n\n")
                    continuation.yield(errorBuffer)
                    continuation.finish()
                }
            }
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
        let envelope = ErrorResponse(error: message, code: code)
        let data = (try? JSONEncoder().encode(envelope)) ?? Data()
        var buffer = ByteBuffer()
        buffer.writeBytes(data)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: buffer)
        )
    }
}

// HTTPResponse.Status from raw code
private extension HTTPResponse.Status {
    init(code: Int) {
        self = HTTPResponse.Status(code: code, reasonPhrase: "")
    }
}
