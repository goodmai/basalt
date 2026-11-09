import Testing
import Foundation
import CoreGraphics
import CoreText
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltServer
@testable import BasaltSpeech
@testable import BasaltCLI

enum VisionFixture {
    static let lines: [(text: String, x: CGFloat, top: CGFloat, size: CGFloat)] = [
        ("Quarterly Report", 100, 120, 56),
        ("The basalt server reads scanned pages on device.", 100, 240, 32),
        ("1. First item here", 120, 400, 32), ("2. Second item here", 120, 450, 32), ("3. Third item here", 120, 500, 32),
        ("Name", 100, 650, 32), ("Count", 500, 650, 32),
        ("Apples", 100, 710, 32), ("12", 500, 710, 32),
        ("Pears", 100, 770, 32), ("7", 500, 770, 32),
        ("Closing paragraph after the table.", 100, 900, 32),
    ]

    static func page(width: Int = 1200, height: Int = 1100) throws -> Data {
        let context = try #require(OCRImage.canvas(width: width, height: height))
        for line in lines {
            let font = CTFontCreateWithName("Helvetica" as CFString, line.size, nil)
            let text = NSAttributedString(string: line.text, attributes: [NSAttributedString.Key(kCTFontAttributeName as String): font])
            context.textPosition = CGPoint(x: line.x, y: CGFloat(height) - line.top)
            CTLineDraw(CTLineCreateWithAttributedString(text), context)
        }
        return try OCRImage.png(try #require(context.makeImage()))
    }

    static func image() throws -> ImageInput {
        try ImageInput(data: try page(), mime: "image/png")
    }
}

@Suite("Apple Vision OCR: VisionDocumentTranscriber on a rendered page, /v1/ocr, basalt ocr and basalt_ocr with apple-vision (#403)")
struct VisionOCRTests {
    @Test("RecognizeDocumentsRequest reads a rendered page into a title, a paragraph, a numbered list, a table and the closing paragraph, in order")
    func readsRenderedPage() async throws {
        let page = try await VisionDocumentTranscriber().transcribe(try VisionFixture.image())
        let markdown = page.markdown
        let expected = ["# Quarterly Report", "The basalt server reads scanned pages on device.", "1. First item here\n2. Second item here\n3. Third item here",
                        "| Apples | 12 |", "Closing paragraph after the table."]
        let positions = try expected.map { try #require(markdown.range(of: $0)?.lowerBound, "missing \($0) in \(markdown)") }
        #expect(positions == positions.sorted())
        #expect(page.blocks.map(\.type) == ["title", "text", "list", "table", "text"])
        #expect(page.blocks.allSatisfy { 0 <= $0.bbox.x1 && $0.bbox.x1 < $0.bbox.x2 && $0.bbox.x2 <= 1 && 0 <= $0.bbox.y1 && $0.bbox.y1 < $0.bbox.y2 && $0.bbox.y2 <= 1 })
        #expect(try #require(page.blocks.first).bbox.y1 < 0.2)
    }

    @Test("The RecognizeTextRequest fallback returns one line region per text line with a top-left box")
    func lineFallback() async throws {
        let regions = try await VisionDocumentTranscriber.lineRegions(in: try VisionFixture.page())
        let closing = try #require(regions.first { $0.element == .paragraph("Closing paragraph after the table.") })
        #expect(closing.box.y1 > 0.7 && closing.box.y2 < 0.9)
        #expect(regions.contains { $0.element == .paragraph("Quarterly Report") && $0.box.y1 < 0.15 })
        #expect(regions.allSatisfy { $0.box.y1 < $0.box.y2 })
        #expect(VisionDocumentTranscriber.isOrdered(.decimal) && VisionDocumentTranscriber.isOrdered(.lowercaseLatin))
        #expect(!VisionDocumentTranscriber.isOrdered(.bullet) && !VisionDocumentTranscriber.isOrdered(nil))
    }

    private func post(_ body: [String: Any], transcriber: Bool = true) async throws -> (status: HTTPResponse.Status, json: [String: Any], requests: Int) {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        if transcriber { await orchestrator.setDocumentTranscriber(VisionDocumentTranscriber()) }
        let app = Application(router: RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "org/mock-model")).makeRouter())
        let data = try JSONSerialization.data(withJSONObject: body)
        let (status, json) = try await app.test(.router) { client in
            try await client.execute(uri: "/v1/ocr", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(bytes: data)) {
                ($0.status, (try? JSONSerialization.jsonObject(with: Data(buffer: $0.body)) as? [String: Any]) ?? [:])
            }
        }
        return (status, json, await engine.getRequests().count)
    }

    @Test("/v1/ocr with model apple-vision reads the page with no model loaded and returns the usual page shape with zero tokens")
    func restAppleVision() async throws {
        let response = try await post(["file": try VisionFixture.page().base64EncodedString(), "model": "apple-vision", "format": "json"])
        #expect(response.status == .ok)
        #expect(response.json["model"] as? String == "apple-vision" && response.json["family"] as? String == "Apple Vision")
        let page = try #require((response.json["pages"] as? [[String: Any]])?.first)
        #expect(page["page"] as? Int == 1 && page["tokens"] as? Int == 0 && page["ms"] is Int)
        #expect((page["markdown"] as? String)?.contains("Closing paragraph after the table.") == true)
        #expect((page["blocks"] as? [[String: Any]])?.contains { $0["type"] as? String == "table" && ($0["bbox"] as? [Double])?.count == 4 } == true)
        #expect(response.requests == 0)
    }

    @Test("/v1/ocr with apple-vision refuses formula and chart tasks, and a process without the transcriber, with 400")
    func restRefusals() async throws {
        let file = try VisionFixture.page().base64EncodedString()
        for task in ["formula", "chart"] {
            #expect(try await post(["file": file, "model": "apple-vision", "task": task]).status == .badRequest)
        }
        #expect(try await post(["file": file, "model": "apple-vision"], transcriber: false).status == .badRequest)
    }

    @Test("basalt ocr --model apple-vision reads a PDF page without the model cache")
    func cliAppleVision() async throws {
        let command = try OCRCommand.parse(["scan.pdf", "--model", "apple-vision"])
        #expect(command.model == OCRRequest.appleVision)
        let request = try OCRRequest(
            document: OCRDocument(data: OCRFixture.pdf(pages: ["Hello from Vision"])), pages: nil, task: nil, format: "json", model: command.model)
        let result = try await OCRCommand.readWithVision(request)
        #expect(result.model == "apple-vision" && result.family == "Apple Vision")
        #expect(result.pages.first?.markdown.contains("Hello from Vision") == true)
        #expect(try OCRCommand.render(result, as: .markdown).contains("Hello from Vision"))
    }

    @Test("basalt_ocr lists model apple-vision, reads with it, and refuses any other model name")
    func mcpAppleVision() async throws {
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine(), telemetry: TelemetryTracker())
        await orchestrator.setDocumentTranscriber(VisionDocumentTranscriber())
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: orchestrator) { capturer.capture($0) }
        await server.dispatch(#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#)
        #expect(capturer.lines.last?.contains("apple-vision") == true)
        let png = try VisionFixture.page().base64EncodedString()
        await server.dispatch(#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"basalt_ocr","arguments":{"file":"\#(png)","model":"apple-vision"}}}"#)
        #expect(capturer.lines.last?.contains("Closing paragraph after the table.") == true)
        await server.dispatch(#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"basalt_ocr","arguments":{"file":"\#(png)","model":"org/other"}}}"#)
        #expect(capturer.lines.last?.contains("model must be apple-vision") == true)
    }
}
