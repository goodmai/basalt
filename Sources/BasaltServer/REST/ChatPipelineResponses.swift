import Foundation
import Hummingbird
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
    func respond(
        to prepared: Prepared, stream: Bool,
        streamed: (AsyncStream<StreamChunk>) -> Response, completed: (GenerationResponse) throws -> Data
    ) async -> Response {
        do {
            if stream {
                return streamed(try await orchestrator.generateStream(request: prepared.request, modelId: prepared.requestedModel))
            }
            let result = try await orchestrator.generate(request: prepared.request, modelId: prepared.requestedModel)
            var buffer = ByteBuffer()
            buffer.writeBytes(try completed(result))
            return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buffer))
        } catch {
            return ProviderErrorResponse.response(for: error, path: endpoint)
        }
    }
}
