import Foundation
import BasaltCore

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
}
