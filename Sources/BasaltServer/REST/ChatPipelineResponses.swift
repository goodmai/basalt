import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

protocol EventStreamEncoder: Sendable {
    mutating func opening() -> [String]
    mutating func frames(for chunk: StreamChunk) -> [String]
    mutating func closing() -> [String]
}

extension EventStreamEncoder {
    static func json(_ object: [String: Any]) -> String? {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    func response(streaming chunks: AsyncStream<StreamChunk>) -> Response {
        let body = AsyncStream<ByteBuffer> { continuation in
            let task = Task {
                var encoder = self
                func yield(_ frames: [String]) {
                    for frame in frames { continuation.yield(ByteBuffer(string: frame)) }
                }
                yield(encoder.opening())
                for await chunk in chunks { yield(encoder.frames(for: chunk)) }
                if !Task.isCancelled { yield(encoder.closing()) }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        return Response(
            status: .ok,
            headers: [.contentType: "text/event-stream", .cacheControl: "no-cache", .connection: "keep-alive"],
            body: ResponseBody(asyncSequence: body)
        )
    }
}

extension ChatPipeline {
    static var imageFallbackHeader: HTTPField.Name { HTTPField.Name("x-basalt-image-fallback")! }

    func respond(
        to prepared: Prepared, stream: Bool,
        streamed: (AsyncStream<StreamChunk>) -> Response, completed: (GenerationResponse) throws -> Data
    ) async -> Response {
        do {
            let (request, transcribed) = try await orchestrator.transcribingImages(in: prepared.request, modelId: prepared.requestedModel)
            var response: Response
            if stream {
                response = streamed(try await orchestrator.generateStream(request: request, modelId: prepared.requestedModel))
            } else {
                let result = try await orchestrator.generate(request: request, modelId: prepared.requestedModel)
                var buffer = ByteBuffer()
                buffer.writeBytes(Self.marked(try completed(result), imagesTranscribed: transcribed))
                response = Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buffer))
            }
            if transcribed > 0 { response.headers[Self.imageFallbackHeader] = "ocr; images=\(transcribed)" }
            return response
        } catch {
            return ProviderErrorResponse.response(for: error, path: endpoint)
        }
    }

    static func marked(_ body: Data, imagesTranscribed: Int) -> Data {
        guard imagesTranscribed > 0, var object = object(in: body) else { return body }
        object["basalt"] = ["image_fallback": "ocr", "images_transcribed": imagesTranscribed]
        return (try? JSONSerialization.data(withJSONObject: object)) ?? body
    }
}
