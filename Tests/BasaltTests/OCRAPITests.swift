import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltServer

@Suite("OCR: /v1/ocr and basalt_ocr on a stub engine (#400)")
struct OCRAPITests {
    static let layout = """
    <|box_start|>100 100 900 200<|box_end|><|ref_start|>title<|ref_end|><|rotate_up|>
    <|box_start|>100 300 900 500<|box_end|><|ref_start|>table<|ref_end|><|rotate_up|>
    <|box_start|>100 600 500 900<|box_end|><|ref_start|>image<|ref_end|><|rotate_up|>
    """

    static func modelDirectory(_ name: String, modelType: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("ocr-\(UUID().uuidString)").appendingPathComponent(name)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(#"{"model_type":"\#(modelType)"}"#.utf8).write(to: directory.appendingPathComponent("config.json"))
        return directory
    }

    static func orchestrator(_ name: String, modelType: String, engine: MockInferenceEngine) async throws -> ModelOrchestratorActor {
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: try modelDirectory(name, modelType: modelType).path)
        return orchestrator
    }

    static func minerUResponder(_ request: GenerationRequest) -> String {
        switch request.prompt {
        case "\nLayout Detection:": layout
        case "\nTable Recognition:": "<fcel>A<fcel>B<nl>"
        default: "Report"
        }
    }

    private func post(_ orchestrator: ModelOrchestratorActor, _ body: [String: Any]) async throws -> (status: HTTPResponse.Status, json: [String: Any]) {
        let app = Application(router: RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "org/mock-model")).makeRouter())
        let data = try JSONSerialization.data(withJSONObject: body)
        return try await app.test(.router) { client in
            try await client.execute(uri: "/v1/ocr", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(bytes: data)) {
                ($0.status, (try? JSONSerialization.jsonObject(with: Data(buffer: $0.body)) as? [String: Any]) ?? [:])
            }
        }
    }

    private func pages(_ json: [String: Any]) -> [[String: Any]] {
        json["pages"] as? [[String: Any]] ?? []
    }

    @Test("GLM-OCR reads each selected PDF page with its task prompt and returns markdown, blocks, tokens and ms per page")
    func glmPDF() async throws {
        let engine = MockInferenceEngine()
        await engine.setResponder { _ in "| a | b |" }
        let orchestrator = try await Self.orchestrator("GLM-OCR-bf16", modelType: "glm_ocr", engine: engine)
        let pdf = OCRFixture.pdf(pages: ["one", "two", "three"]).base64EncodedString()
        let response = try await post(orchestrator, ["file": pdf, "pages": "2-3", "task": "table", "format": "json"])
        #expect(response.status == .ok)
        #expect(response.json["family"] as? String == "GLM-OCR" && response.json["task"] as? String == "table")
        let read = pages(response.json)
        #expect(read.map { $0["page"] as? Int } == [2, 3])
        #expect(read.first?["markdown"] as? String == "| a | b |")
        #expect(read.first?["tokens"] as? Int == 10)
        #expect(read.first?["ms"] is Int)
        let block = (read.first?["blocks"] as? [[String: Any]])?.first
        #expect(block?["type"] as? String == "table" && block?["bbox"] as? [Double] == [0, 0, 1, 1] && block?["text"] as? String == "| a | b |")
        let requests = await engine.getRequests()
        #expect(requests.count == 2)
        #expect(requests.allSatisfy { $0.prompt == "Table Recognition:" && $0.temperature == 0 && $0.messages?.first?.images.count == 1 })
        #expect(try OCRFixture.size(of: try #require(requests.first?.messages?.first?.images.first)) == (1700, 2200))
    }

    @Test("Markdown, the default format, leaves blocks out; an image is one page sent as it came")
    func markdownImage() async throws {
        let engine = MockInferenceEngine()
        await engine.setResponder { _ in "Hello" }
        let orchestrator = try await Self.orchestrator("GLM-OCR-4bit", modelType: "glm_ocr", engine: engine)
        let png = try OCRFixture.png(width: 8, height: 4)
        let response = try await post(orchestrator, ["file": "data:image/png;base64," + png.base64EncodedString()])
        #expect(response.status == .ok)
        #expect(pages(response.json).first?["markdown"] as? String == "Hello")
        #expect(pages(response.json).first?["blocks"] == nil)
        #expect(await engine.getRequests().first?.messages?.first?.images.first?.data == png)
        #expect(await engine.getRequests().first?.prompt == "Text Recognition:")
    }

    @Test("MinerU2.5 detects the layout on a 1036x1036 page, recognizes each text and table crop, skips images, and assembles the page")
    func minerUDocument() async throws {
        let engine = MockInferenceEngine()
        await engine.setResponder(Self.minerUResponder)
        let orchestrator = try await Self.orchestrator("MinerU2.5-Pro-2605-1.2B", modelType: "qwen2_vl", engine: engine)
        let png = try OCRFixture.png(width: 400, height: 600)
        let response = try await post(orchestrator, ["file": png.base64EncodedString(), "format": "json"])
        #expect(response.status == .ok)
        let page = try #require(pages(response.json).first)
        #expect(page["markdown"] as? String == "Report\n\n<table><tr><td>A</td><td>B</td></tr></table>")
        #expect(page["tokens"] as? Int == 30)
        let blocks = page["blocks"] as? [[String: Any]] ?? []
        #expect(blocks.map { $0["type"] as? String } == ["title", "table", "image"])
        #expect(blocks.last?["text"] as? String == "")
        #expect(blocks[1]["bbox"] as? [Double] == [0.1, 0.3, 0.9, 0.5])
        let requests = await engine.getRequests()
        #expect(requests.map(\.prompt) == ["\nLayout Detection:", "\nText Recognition:", "\nTable Recognition:"])
        #expect(try OCRFixture.size(of: try #require(requests.first?.messages?.first?.images.first)) == (1036, 1036))
        #expect(try OCRFixture.size(of: try #require(requests.last?.messages?.first?.images.first)) == (320, 120))
        #expect(requests[0].presencePenalty == nil && requests[2].presencePenalty == 1)
    }

    @Test("A loaded model without an OCR recipe is refused with 400 before any page reaches the engine",
          arguments: [("Llama-mock", "llama"), ("Qwen2-VL-2B-Instruct", "qwen2_vl")])
    func refusesWithoutRecipe(_ name: String, _ modelType: String) async throws {
        let engine = MockInferenceEngine()
        let orchestrator = try await Self.orchestrator(name, modelType: modelType, engine: engine)
        let response = try await post(orchestrator, ["file": try OCRFixture.png(width: 2, height: 2).base64EncodedString()])
        #expect(response.status == .badRequest)
        #expect(((response.json["error"] as? [String: Any])?["message"] as? String)?.contains("no OCR recipe") == true)
        #expect(await engine.getRequests().isEmpty)
    }

    @Test("Bad requests are refused with 400 before the engine: no file, a text file, an unknown task, a chart for GLM-OCR, a page past the end")
    func refusesBadRequests() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = try await Self.orchestrator("GLM-OCR", modelType: "glm_ocr", engine: engine)
        let png = try OCRFixture.png(width: 2, height: 2).base64EncodedString()
        let pdf = OCRFixture.pdf(pages: ["one"]).base64EncodedString()
        let bodies: [[String: Any]] = [
            ["pages": "1"], ["file": Data("plain text".utf8).base64EncodedString()], ["file": png, "task": "poem"],
            ["file": png, "task": "chart"], ["file": pdf, "pages": "2"], ["file": png, "format": "html"], ["file": png, "pages": "1-99"],
        ]
        for body in bodies {
            #expect(try await post(orchestrator, body).status == .badRequest)
        }
        #expect(await engine.getRequests().isEmpty)
    }

    @Test("An OCR page never goes to the cloud fallback: its prompt only means something to the local OCR model")
    func staysLocal() async throws {
        let engine = MockInferenceEngine()
        try await engine.load(modelPath: "local")
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        let cloud = StubCloud(CloudAssessment(capabilityGap: 0.9, estimatedCost: 0.01, remainingBudget: 10, latencySeconds: 1))
        await orchestrator.setCloudFallback(cloud, freeMemory: { 64 << 30 })
        let image = try ImageInput(data: try OCRFixture.png(width: 2, height: 2), mime: "image/png")
        let page = OCRRecipe(family: .glmOCR).request(prompt: "Text Recognition:", image: image, stage: .recognition)
        #expect(await orchestrator.cloudFallbackReason(for: page) == nil)
        #expect(await orchestrator.cloudFallbackReason(for: GenerationRequest(prompt: "Text Recognition:")) == .capability)
    }

    @Test("DeepSeek-OCR over /v1/ocr and basalt_ocr: the resolution and grounding reach the engine as the OCR mode, grounding tags become blocks, and a resolution for another family or an unknown one is a 400")
    func deepSeekResolution() async throws {
        let engine = MockInferenceEngine()
        await engine.setResponder { _ in "<|ref|>title<|/ref|><|det|>[[0, 0, 999, 99]]<|/det|>\n# Title\n" }
        let orchestrator = try await Self.orchestrator("DeepSeek-OCR-8bit", modelType: "deepseekocr", engine: engine)
        let png = try OCRFixture.png(width: 8, height: 4).base64EncodedString()
        let response = try await post(orchestrator, ["file": png, "resolution": "small", "format": "json"])
        #expect(response.status == .ok && response.json["family"] as? String == "DeepSeek-OCR")
        let page = try #require(pages(response.json).first)
        #expect(page["markdown"] as? String == "# Title")
        #expect((page["blocks"] as? [[String: Any]])?.first?["type"] as? String == "title")
        let server = MCPServer(orchestrator: orchestrator) { _ in }
        await server.dispatch(#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"basalt_ocr","arguments":{"file":"\#(png)","resolution":"tiny","task":"text"}}}"#)
        let requests = await engine.getRequests()
        #expect(requests.map(\.ocrMode) == [OCRMode(resolution: .small, grounding: true), OCRMode(resolution: .tiny, grounding: false)])
        #expect(requests.first?.prompt == "Convert the document to markdown.")
        #expect(try await post(orchestrator, ["file": png, "resolution": "huge"]).status == .badRequest)
        let glm = try await Self.orchestrator("GLM-OCR", modelType: "glm_ocr", engine: engine)
        #expect(try await post(glm, ["file": png, "resolution": "base"]).status == .badRequest)
        #expect(await engine.getRequests().count == 2)
    }

    @Test("basalt_ocr is listed, reads base64 or a path inside --image-dir, and refuses a path outside it")
    func mcpTool() async throws {
        let engine = MockInferenceEngine()
        await engine.setResponder { _ in "From MCP" }
        let orchestrator = try await Self.orchestrator("GLM-OCR", modelType: "glm_ocr", engine: engine)
        let inside = FileManager.default.temporaryDirectory.appendingPathComponent("ocr-mcp-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: inside, withIntermediateDirectories: true)
        let pdfPath = inside.appendingPathComponent("doc.pdf")
        try OCRFixture.pdf(pages: ["one", "two"]).write(to: pdfPath)
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: orchestrator) { capturer.capture($0) }.withImageDirectories([inside])
        #expect(MCPTool(name: "basalt_ocr") == .media(.ocr))
        await server.dispatch(#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#)
        #expect(capturer.lines.last?.contains("basalt_ocr") == true)
        let png = try OCRFixture.png(width: 2, height: 2).base64EncodedString()
        await server.dispatch(#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"basalt_ocr","arguments":{"file":"\#(png)"}}}"#)
        #expect(capturer.lines.last?.contains("From MCP") == true)
        await server.dispatch(#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"basalt_ocr","arguments":{"path":"\#(pdfPath.path)","pages":"2","format":"json"}}}"#)
        #expect(capturer.lines.last?.contains(#"\"page\":2"#) == true)
        #expect(capturer.lines.last?.contains("blocks") == true)
        await server.dispatch(#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"basalt_ocr","arguments":{"path":"/etc/hosts"}}}"#)
        #expect(capturer.lines.last?.contains("--image-dir") == true)
        await server.dispatch(#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"basalt_ocr","arguments":{}}}"#)
        #expect(capturer.lines.last?.contains("\"error\"") == true)
        #expect(await engine.getRequests().count == 2)
    }
}
