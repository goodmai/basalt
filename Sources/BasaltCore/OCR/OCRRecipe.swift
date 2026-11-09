import Foundation

public struct OCRRecipe: Sendable, Equatable {
    public enum Family: String, Sendable {
        case glmOCR = "GLM-OCR"
        case minerU = "MinerU2.5"
        case paddleOCR = "PaddleOCR-VL"
        case deepSeekOCR = "DeepSeek-OCR"
    }

    public enum Step: Sendable, Equatable {
        case recognize(prompt: String, blockType: String)
        case layout(prompt: String, side: Int)
    }

    public enum Stage: Sendable, Equatable {
        case layout, recognition
    }

    public static let pageDPI = 200.0
    public static let maxTokens = 8192
    public static let origin = RequestOrigin(source: "ocr", endpoint: "/v1/ocr", stream: false)

    public let family: Family
    public var resolution: OCRResolution?

    public init(family: Family, resolution: OCRResolution? = nil) {
        (self.family, self.resolution) = (family, resolution)
    }

    public static func of(modelType: String?, hints: [String]) -> OCRRecipe? {
        switch modelType?.lowercased() {
        case "glm_ocr": OCRRecipe(family: .glmOCR)
        case "paddleocr_vl": OCRRecipe(family: .paddleOCR)
        case "deepseekocr": OCRRecipe(family: .deepSeekOCR)
        case "qwen2_vl" where hints.contains(where: { $0.lowercased().contains("mineru") }): OCRRecipe(family: .minerU)
        default: nil
        }
    }

    public func step(for task: OCRTask) throws(BasaltError) -> Step {
        switch family {
        case .glmOCR: try Self.glmStep(task)
        case .minerU: Self.minerUStep(task)
        case .paddleOCR: Self.paddleStep(task)
        case .deepSeekOCR: try Self.deepSeekStep(task)
        }
    }

    public func with(resolution: OCRResolution?) throws(BasaltError) -> OCRRecipe {
        guard resolution == nil || family == .deepSeekOCR else {
            throw .invalidRequestStructure(details: "\(family.rawValue) has no resolution modes; leave resolution out (DeepSeek-OCR takes \(OCRResolution.allCases.map(\.rawValue).joined(separator: ", ")))")
        }
        return OCRRecipe(family: family, resolution: resolution)
    }

    static func deepSeekStep(_ task: OCRTask) throws(BasaltError) -> Step {
        switch task {
        case .document: .recognize(prompt: DeepSeekGrounding.prefix + "Convert the document to markdown.", blockType: "text")
        case .text: .recognize(prompt: "Free OCR.", blockType: "text")
        case .chart: .recognize(prompt: "Parse the figure.", blockType: "chart")
        case .table, .formula:
            throw .invalidRequestStructure(details: "DeepSeek-OCR has no table or formula prompt; task document converts the tables and formulas on a page")
        }
    }

    static func paddleStep(_ task: OCRTask) -> Step {
        switch task {
        case .document, .text: .recognize(prompt: "OCR:", blockType: "text")
        case .table: .recognize(prompt: "Table Recognition:", blockType: "table")
        case .formula: .recognize(prompt: "Formula Recognition:", blockType: "equation")
        case .chart: .recognize(prompt: "Chart Recognition:", blockType: "chart")
        }
    }

    static func glmStep(_ task: OCRTask) throws(BasaltError) -> Step {
        switch task {
        case .document, .text: .recognize(prompt: "Text Recognition:", blockType: "text")
        case .table: .recognize(prompt: "Table Recognition:", blockType: "table")
        case .formula: .recognize(prompt: "Formula Recognition:", blockType: "equation")
        case .chart: throw .invalidRequestStructure(details: "GLM-OCR has no chart prompt (its SDK skips charts); use task document, text, table or formula")
        }
    }

    static func minerUStep(_ task: OCRTask) -> Step {
        switch task {
        case .document: .layout(prompt: MinerULayout.layoutPrompt, side: MinerULayout.layoutSide)
        case .text: .recognize(prompt: MinerULayout.textPrompt, blockType: "text")
        case .table: .recognize(prompt: MinerULayout.tablePrompt, blockType: "table")
        case .formula: .recognize(prompt: MinerULayout.formulaPrompt, blockType: "equation")
        case .chart: .recognize(prompt: MinerULayout.imageAnalysisPrompt, blockType: "chart")
        }
    }

    public func request(prompt: String, image: ImageInput, stage: Stage) -> GenerationRequest {
        let grounded = family == .deepSeekOCR && prompt.hasPrefix(DeepSeekGrounding.prefix)
        let text = grounded ? String(prompt.dropFirst(DeepSeekGrounding.prefix.count)) : prompt
        var request = GenerationRequest(
            prompt: text, maxTokens: Self.maxTokens, temperature: 0, topP: nil,
            messages: [ChatMessageInput(role: .user, text: text, images: [image])])
        switch (family, stage) {
        case (.glmOCR, _): request.repetitionPenalty = 1.1
        case (.minerU, .recognition): request.presencePenalty = 1
        case (.deepSeekOCR, _): request.ocrMode = OCRMode(resolution: resolution ?? .gundam, grounding: grounded)
        case (.minerU, .layout), (.paddleOCR, _): break
        }
        return request.withOrigin(Self.origin)
    }

    public func finished(_ output: String, blockType: String) -> String {
        let text = output.trimmingCharacters(in: .whitespacesAndNewlines)
        switch (family, blockType) {
        case (.minerU, "table"), (.paddleOCR, "table"): return OTSLTable.html(text)
        case (.minerU, "equation"): return MinerULayout.bracketed(text)
        default: return text
        }
    }

    public func read(_ output: String, blockType: String) -> (markdown: String, blocks: [OCRBlock]) {
        guard family == .deepSeekOCR else {
            let text = finished(output, blockType: blockType)
            return (text, [OCRBlock(type: blockType, bbox: .page, text: text)])
        }
        return DeepSeekGrounding.parse(output, blockType: blockType)
    }
}
