import Foundation
import Hummingbird
import HummingbirdAuth

// MARK: — Route handlers for REST / A2A interface

struct GenerateController: Sendable {

    let orchestrator: ModelOrchestratorActor
    let modelId: String?

    // MARK: — POST /api/v1/generate

    @Sendable
    func generate(request: Request, context: GemmaRequestContext) async throws -> Response {
        // Require authentication
        _ = try context.requireIdentity()

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
        } catch let err as GemmaServerError {
            return errorResponse(
                status: HTTPResponse.Status(code: err.httpStatus),
                message: err.errorDescription ?? err.localizedDescription,
                code: err.httpStatus
            )
        }
    }

    // MARK: — GET /api/v1/health

    @Sendable
    func health(request: Request, context: GemmaRequestContext) async throws -> Response {
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
