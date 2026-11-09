import Foundation
import BasaltCore

enum MediaTool: Sendable, Equatable {
    case transcribe, ocr, decide, pipeline
}

extension MCPServer {
    func ocr(_ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        let (file, path) = (try MCPArguments.string("file", in: args), try MCPArguments.string("path", in: args))
        let (pages, task, format) = (
            try MCPArguments.string("pages", in: args), try MCPArguments.string("task", in: args), try MCPArguments.string("format", in: args))
        do throws(BasaltError) {
            let document = switch (file, path) {
            case (let file?, nil): try OCRDocument(encoded: file)
            case (nil, let path?): try OCRDocument(data: MCPImages.read(path, allowedIn: imageDirectories).data)
            default: throw BasaltError.invalidRequestStructure(details: "send exactly one of file (base64) and path (inside a serve --image-dir directory)")
            }
            return .json(try await orchestrator.ocr(OCRRequest(document: document, pages: pages, task: task, format: format, model: nil)))
        } catch {
            throw MCPFailure(error)
        }
    }

    func pipeline(_ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        guard let args, JSONSerialization.isValidJSONObject(args), let data = try? JSONSerialization.data(withJSONObject: args) else {
            throw .invalidParams("basalt_pipeline needs an object of attachments, text, questions, task and pages")
        }
        do throws(BasaltError) {
            let request = try PipelineController.request(from: data, imageDirectories: imageDirectories)
            let runner = PipelineRunner(chat: orchestrator, decisions: decisions, stages: pipeline)
            return .json(try await runner.run(request, origin: RequestOrigin(source: "mcp", endpoint: "basalt_pipeline", stream: false), sessionId: nil))
        } catch {
            throw MCPFailure(error)
        }
    }
}

extension MCPToolCatalog {
    static let ocr = MCPToolDescriptor(
        name: "basalt_ocr",
        description: "Read a PDF or an image with the loaded OCR model (GLM-OCR or MinerU2.5) and return Markdown per page, with blocks {type, bbox, text} when format is json. PDF pages render at 200 DPI; a request reads at most \(OCRPageSelection.maxPages) pages.",
        inputSchema: ResearchTool.object([
            "file": ["type": "string", "description": "The PDF or image, base64-encoded or as a data: URI"],
            "path": ["type": "string", "description": "Instead of file: a PDF or image inside a directory given with serve --image-dir"],
            "pages": ["type": "string", "description": "PDF pages to read, such as 1-3,5 (default: all)"],
            "task": ["type": "string", "enum": OCRTask.allCases.map(\.rawValue), "description": "document (default), text, table, formula or chart"],
            "format": ["type": "string", "enum": OCRFormat.allCases.map(\.rawValue), "description": "markdown (default) or json, which adds blocks"],
        ]))

    static let pipeline = MCPToolDescriptor(
        name: "basalt_pipeline",
        description: "One call for perceive → decide → answer: pdf and image attachments are read by the OCR model (serve --pipeline-ocr-model) and audio is transcribed; typed questions go to System One (serve --decision-model), and only the questions it abstains on, plus a free-form task, go to the chat model. Returns the perceived text, answers per question with answered_by decide or answer, the System 2 reply to the task, and a route receipt with model, ms, tokens, confidence and escalation per stage.",
        inputSchema: ResearchTool.object([
            "attachments": ["type": "array", "description": "Items {kind: pdf|image|audio, data: base64, format?} or {kind, path} with a path inside a serve --image-dir directory; audio needs format, such as wav"],
            "text": ["type": "string", "description": "Client text added to the perceived text"],
            "questions": ["type": "object", "description": "Map of question ID → {type: noul|choice|score, instructions, criteria}, as for basalt_systemone_decide"],
            "task": ["type": "string", "description": "A free-form task for the chat model, with the perceived text as context"],
            "pages": ["type": "string", "description": "PDF pages to read, such as 1-3,5 (default: all)"],
        ]))
}
