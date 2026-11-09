import Foundation

public struct OCRRecipe: Sendable, Equatable {
    public enum Family: String, Sendable {
        case glmOCR = "GLM-OCR"
        case minerU = "MinerU2.5"
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

    public static func of(modelType: String?, hints: [String]) -> OCRRecipe? {
        switch modelType?.lowercased() {
        case "glm_ocr": OCRRecipe(family: .glmOCR)
        case "qwen2_vl" where hints.contains(where: { $0.lowercased().contains("mineru") }): OCRRecipe(family: .minerU)
        default: nil
        }
    }

    public func step(for task: OCRTask) throws(BasaltError) -> Step {
        switch family {
        case .glmOCR: try Self.glmStep(task)
        case .minerU: Self.minerUStep(task)
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
        var request = GenerationRequest(
            prompt: prompt, maxTokens: Self.maxTokens, temperature: 0, topP: nil,
            messages: [ChatMessageInput(role: .user, text: prompt, images: [image])])
        switch (family, stage) {
        case (.glmOCR, _): request.repetitionPenalty = 1.1
        case (.minerU, .recognition): request.presencePenalty = 1
        case (.minerU, .layout): break
        }
        return request.withOrigin(Self.origin)
    }

    public func finished(_ output: String, blockType: String) -> String {
        let text = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard family == .minerU else { return text }
        switch blockType {
        case "table": return OTSLTable.html(text)
        case "equation": return MinerULayout.bracketed(text)
        default: return text
        }
    }
}
