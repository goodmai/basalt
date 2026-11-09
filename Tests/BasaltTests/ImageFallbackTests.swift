import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer
@testable import BasaltCLI

struct ScriptedDocumentTranscriber: DocumentTranscriber {
    let text: String

    func transcribe(_ image: ImageInput) async throws(BasaltError) -> DocumentMarkdown {
        DocumentMarkdown([DocumentRegion(.paragraph(text), box: .page)])
    }
}

@Suite("Image fallback: serve --image-fallback ocr reads images for a text-only model instead of refusing them (#403)")
struct ImageFallbackTests {
    static func question() throws -> GenerationRequest {
        GenerationRequest(
            prompt: "", maxTokens: 4, temperature: 0,
            messages: [ChatMessageInput(role: .user, text: "what does it say?", images: [try ImageInput(data: Data("png".utf8), mime: "image/png")])])
    }

    @Test("On a text-only tiny model with the fallback on, the image turns into an OCR text block and generation runs")
    func textOnlyModelReadsTheImage() async throws {
        let orchestrator = ModelOrchestratorActor(engine: try await TinyModel.engine(.qwen3), telemetry: TelemetryTracker())
        await orchestrator.setDocumentTranscriber(ScriptedDocumentTranscriber(text: "INVOICE 42"), imageFallback: true)
        let (request, images) = try await orchestrator.transcribingImages(in: Self.question())
        #expect(images == 1)
        let message = try #require(request.messages?.first)
        #expect(message.images.isEmpty)
        #expect(message.text == "what does it say?\n\n[image 1, OCR]\nINVOICE 42")
        let response = try await orchestrator.generate(request: request)
        #expect(response.completionTokens > 0)
    }

    @Test("With the fallback off the request is left alone and the text-only model still refuses the image with a 400",
          arguments: [false, true])
    func refusesWithFallbackOff(transcriberSet: Bool) async throws {
        let orchestrator = ModelOrchestratorActor(engine: try await TinyModel.engine(.qwen3), telemetry: TelemetryTracker())
        if transcriberSet { await orchestrator.setDocumentTranscriber(ScriptedDocumentTranscriber(text: "unused")) }
        let (request, images) = try await orchestrator.transcribingImages(in: Self.question())
        #expect(images == 0)
        #expect(try request.messages == Self.question().messages)
        let error = await #expect(throws: BasaltError.self) { try await orchestrator.generate(request: request) }
        #expect(error?.httpStatus == 400)
        #expect(error?.localizedDescription.contains("does not take images") == true)
    }

    @Test("A model that takes images keeps them, and a request without images is untouched, with the fallback on")
    func visionModelKeepsImages() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        await orchestrator.setDocumentTranscriber(ScriptedDocumentTranscriber(text: "x"), imageFallback: true)
        #expect(try await orchestrator.transcribingImages(in: Self.question()).images == 0)
        await engine.setMockTakesImages(false)
        #expect(try await orchestrator.transcribingImages(in: GenerationRequest(prompt: "hi")).images == 0)
        #expect(try await orchestrator.transcribingImages(in: Self.question()).images == 1)
    }

    private func chat(fallback: Bool, stream: Bool) async throws -> (headers: HTTPFields, body: String, sent: GenerationRequest?) {
        let engine = MockInferenceEngine()
        await engine.setMockTakesImages(false)
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "mock")
        await orchestrator.setDocumentTranscriber(ScriptedDocumentTranscriber(text: "RECEIPT"), imageFallback: fallback)
        let app = Application(router: RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "mock")).makeRouter())
        let json = ImageInputTests.openAI([ImageInputTests.imageURL(ImageInputTests.pngURI)]).dropLast() + #", "stream": \#(stream)}"#
        let (headers, body) = try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/chat/completions", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: String(json))
            ) { ($0.headers, String(buffer: $0.body)) }
        }
        return (headers, body, await engine.lastRequest)
    }

    @Test("REST: the engine gets the OCR text, and the response says so in X-Basalt-Image-Fallback and a basalt field")
    func restMarksTheResponse() async throws {
        let result = try await chat(fallback: true, stream: false)
        #expect(result.headers[ChatPipeline.imageFallbackHeader] == "ocr; images=1")
        let object = try #require(try JSONSerialization.jsonObject(with: Data(result.body.utf8)) as? [String: Any])
        let basalt = try #require(object["basalt"] as? [String: Any])
        #expect(basalt["image_fallback"] as? String == "ocr" && basalt["images_transcribed"] as? Int == 1)
        #expect(object["choices"] is [[String: Any]])
        let message = try #require(result.sent?.messages?.first)
        #expect(message.images.isEmpty && message.text == "what is this?\n\n[image 1, OCR]\nRECEIPT")
    }

    @Test("REST streaming carries the header; with the fallback off there is no header or field and the image goes to the engine")
    func restStreamAndOff() async throws {
        let streamed = try await chat(fallback: true, stream: true)
        #expect(streamed.headers[ChatPipeline.imageFallbackHeader] == "ocr; images=1")
        #expect(streamed.sent?.messages?.first?.images.isEmpty == true)
        let off = try await chat(fallback: false, stream: false)
        #expect(off.headers[ChatPipeline.imageFallbackHeader] == nil)
        #expect(!off.body.contains("image_fallback"))
        #expect(off.sent?.messages?.first?.images.count == 1)
    }

    @Test("The JSON marker leaves a body alone when no image was read or the body is not a JSON object")
    func markerOnlyWhenRead() {
        let body = Data(#"{"id":"x"}"#.utf8)
        #expect(ChatPipeline.marked(body, imagesTranscribed: 0) == body)
        #expect(ChatPipeline.marked(Data("[1]".utf8), imagesTranscribed: 2) == Data("[1]".utf8))
    }

    @Test("serve --image-fallback takes ocr, refuses anything else, and is off by default")
    func serveFlag() throws {
        #expect(try ServeCommand.parse(["--image-fallback", "ocr"]).experimental.imageFallback == .ocr)
        #expect(try ServeCommand.parse([]).experimental.imageFallback == nil)
        #expect(throws: (any Error).self) { try ServeCommand.parse(["--image-fallback", "vlm"]) }
    }
}
