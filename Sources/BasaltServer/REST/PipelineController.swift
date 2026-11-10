import Foundation
import Hummingbird
import BasaltCore

struct PipelineController: Sendable {
    static let path = "/v1/pipeline"
    static let bodyLimit = ChatPipeline.bodyLimit + (AudioClip.maxBytes + 2) / 3 * 4

    let resolve: @Sendable (String) async -> PipelineRunner?

    init(runner: PipelineRunner) { resolve = { _ in runner } }

    init(registry: PipelineRegistry, chat: ModelOrchestratorActor) {
        resolve = { name in await registry.system(name).map { PipelineRunner(chat: chat, decisions: $0.decisions, stages: $0.stages) } }
    }

    struct Body: Decodable {
        struct Attachment: Decodable {
            let kind: String
            let data: String?
            let path: String?
            let format: String?
        }

        let attachments: [Attachment]?
        let text: String?
        let questions: [String: SystemOneQuestion]?
        let task: String?
        let pages: String?
    }

    static func request(from data: Data, imageDirectories: [URL], defaultQuestions: [String: SystemOneQuestion] = [:]) throws(BasaltError) -> PipelineRequest {
        guard let body = try? JSONDecoder().decode(Body.self, from: data) else {
            throw .invalidRequestStructure(
                details: "a pipeline request is JSON with attachments [{kind: pdf|image|audio, data: base64, format?}], text, questions {id: {type, instructions, criteria?}}, task and pages")
        }
        var attachments: [PipelineAttachment] = []
        for attachment in body.attachments ?? [] {
            switch (attachment.data, attachment.path) {
            case (let encoded?, nil):
                attachments.append(try PipelineAttachment(kind: attachment.kind, encoded: encoded, format: attachment.format))
            case (nil, let path?):
                let file = try MCPImages.read(path, allowedIn: imageDirectories)
                attachments.append(try PipelineAttachment(kind: attachment.kind, data: file.data, format: attachment.format ?? file.url.pathExtension))
            default:
                throw .invalidRequestStructure(details: "pipeline: each attachment needs exactly one of data (base64) and path (MCP, inside a serve --image-dir directory)")
            }
        }
        return try PipelineRequest(attachments: attachments, text: body.text, questions: body.questions ?? defaultQuestions, task: body.task, pages: body.pages)
    }

    @Sendable
    func pipeline(request: Request, context: BasicRequestContext) async throws -> Response {
        let name = context.parameters.get("name") ?? ComposeSpec.defaultPipeline
        guard let runner = await resolve(name) else {
            return ProviderErrorResponse.response(status: .notFound, message: "no pipeline named \(name)", shape: .openAI)
        }
        do {
            let data = Data(try await request.body.collect(upTo: Self.bodyLimit).readableBytesView)
            let parsed = try Self.request(from: data, imageDirectories: [], defaultQuestions: runner.stages.questions)
            if let missing = await runner.missingStage(parsed) {
                return ProviderErrorResponse.response(status: .notFound, message: missing, shape: .openAI)
            }
            let response = try await runner.run(
                parsed, origin: RequestOrigin(source: "pipeline", endpoint: Self.path, stream: false),
                sessionId: GenerationRequest.sessionId(in: request.headers))
            var buffer = ByteBuffer()
            buffer.writeBytes(try JSONEncoder().encode(response))
            return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buffer))
        } catch {
            return ProviderErrorResponse.response(for: error, path: Self.path)
        }
    }
}
