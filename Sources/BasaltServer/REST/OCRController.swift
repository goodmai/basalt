import Foundation
import Hummingbird
import BasaltCore

struct OCRController: Sendable {
    static let path = "/v1/ocr"

    let orchestrator: ModelOrchestratorActor

    struct Body: Decodable {
        let file: String
        let pages: String?
        let task: String?
        let format: String?
        let model: String?
        let resolution: String?
    }

    @Sendable
    func ocr(request: Request, context: BasicRequestContext) async throws -> Response {
        do {
            let data = Data(try await request.body.collect(upTo: ChatPipeline.bodyLimit).readableBytesView)
            guard let body = try? JSONDecoder().decode(Body.self, from: data) else {
                throw BasaltError.invalidRequestStructure(
                    details: "an OCR request is JSON with file (base64 or a data: URI of a PDF or image) and optional pages, task, format, model and resolution")
            }
            let ocr = try OCRRequest(
                document: OCRDocument(encoded: body.file), pages: body.pages, task: body.task, format: body.format, model: body.model,
                resolution: body.resolution)
            var buffer = ByteBuffer()
            buffer.writeBytes(try JSONEncoder().encode(try await orchestrator.ocr(ocr)))
            return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buffer))
        } catch {
            return ProviderErrorResponse.response(for: error, path: Self.path)
        }
    }
}
