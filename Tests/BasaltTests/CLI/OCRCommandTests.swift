import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltCLI

@Suite("basalt ocr: arguments and output (#400)")
struct OCRCommandTests {
    static let result = OCRResult(
        model: "mlx-community/GLM-OCR-bf16", family: OCRRecipe.Family.glmOCR.rawValue, task: .document,
        pages: [
            OCRPageResult(page: 1, markdown: "# One", blocks: [OCRBlock(type: "text", bbox: .page, text: "# One")], tokens: 3, ms: 5),
            OCRPageResult(page: 2, markdown: "Two", blocks: nil, tokens: 1, ms: 2),
        ],
        format: .json)

    @Test("basalt ocr takes a file, --model, --pages, --task and --format, defaulting to document and markdown")
    func parsesArguments() throws {
        let command = try OCRCommand.parse(["scan.pdf", "--model", "mlx-community/GLM-OCR-bf16", "--pages", "1-3"])
        #expect(command.file == "scan.pdf" && command.model == "mlx-community/GLM-OCR-bf16" && command.pages == "1-3")
        #expect(command.task == "document" && command.format == "markdown")
        #expect(throws: (any Error).self) { try OCRCommand.parse(["scan.pdf"]) }
    }

    @Test("Markdown output marks each page of a multi-page document; json output carries the pages with their blocks")
    func rendersOutput() throws {
        #expect(try OCRCommand.render(Self.result, as: .markdown) == "<!-- page 1 -->\n\n# One\n\n<!-- page 2 -->\n\nTwo")
        let json = try OCRCommand.render(Self.result, as: .json)
        let object = try #require(try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        #expect(object["family"] as? String == "GLM-OCR")
        #expect((object["pages"] as? [[String: Any]])?.first?["blocks"] is [[String: Any]])
    }
}
