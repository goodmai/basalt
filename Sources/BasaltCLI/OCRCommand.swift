import ArgumentParser
import BasaltCore
import BasaltInference
import Foundation

struct OCRCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ocr",
        abstract: "Read a PDF or an image to Markdown with GLM-OCR or MinerU2.5"
    )

    @Argument(help: "PDF or image file (PNG, JPEG, WebP, HEIC)")
    var file: String

    @Option(name: .long, help: "OCR model repo id from the local cache, or a local model directory")
    var model: String

    @Option(name: .long, help: "PDF pages to read, such as 1-3,5 (default: all, at most \(OCRPageSelection.maxPages))")
    var pages: String?

    @Option(name: .long, help: "document, text, table, formula or chart")
    var task = OCRTask.document.rawValue

    @Option(name: .long, help: "markdown, or json with blocks {type, bbox, text}")
    var format = OCRFormat.markdown.rawValue

    mutating func run() async throws {
        guard let data = FileManager.default.contents(atPath: (file as NSString).expandingTildeInPath) else {
            throw ValidationError("Cannot read \(file)")
        }
        let request = try OCRRequest(document: OCRDocument(data: data), pages: pages, task: task, format: format, model: nil)
        guard let modelPath = FileManager.default.fileExists(atPath: model) ? model : ModelCache.resolve(repoId: model)?.path else {
            throw ValidationError("Model '\(model)' is not in the local cache; run `basalt models download \(model)` first")
        }
        let orchestrator = ModelOrchestratorActor(engine: MLXInferenceEngine())
        defer { await orchestrator.shutdown() }
        try await orchestrator.loadModel(path: modelPath)
        await orchestrator.setModelId(model)
        let result = try await orchestrator.ocr(request)
        print(try Self.render(result, as: request.format))
    }

    static func render(_ result: OCRResult, as format: OCRFormat) throws -> String {
        switch format {
        case .markdown:
            return result.pages.map { result.pages.count > 1 ? "<!-- page \($0.page) -->\n\n\($0.markdown)" : $0.markdown }.joined(separator: "\n\n")
        case .json:
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            return String(decoding: try encoder.encode(result), as: UTF8.self)
        }
    }
}
