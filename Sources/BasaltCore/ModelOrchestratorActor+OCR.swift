import CoreGraphics
import Foundation

extension ModelOrchestratorActor {
    @concurrent
    public nonisolated func ocr(_ request: OCRRequest) async throws(BasaltError) -> OCRResult {
        let (recipe, epoch) = try await ocrRecipe(switchingTo: request.model)
        let step = try recipe.step(for: request.task)
        var pages: [OCRPageResult] = []
        for page in try request.document.pages(request.pages, dpi: OCRRecipe.pageDPI) {
            let started = ContinuousClock.now
            let read = switch step {
            case .recognize(let prompt, let type): try await recognize(page.image, prompt: prompt, type: type, recipe: recipe, epoch: epoch)
            case .layout(let prompt, let side): try await parseLayout(page.image, prompt: prompt, side: side, recipe: recipe, epoch: epoch)
            }
            let elapsed = (ContinuousClock.now - started).components
            let ms = Int(elapsed.seconds * 1000 + elapsed.attoseconds / 1_000_000_000_000_000)
            pages.append(OCRPageResult(page: page.number, markdown: read.markdown, blocks: read.blocks, tokens: read.tokens, ms: ms))
        }
        return OCRResult(model: await modelInfo, family: recipe.family, task: request.task, pages: pages, format: request.format)
    }

    func ocrRecipe(switchingTo model: String?) async throws(BasaltError) -> (recipe: OCRRecipe, epoch: Int) {
        try await autoSwitch(to: model)
        let modelType = ModelCache.config(at: URL(fileURLWithPath: loaded.path))?["model_type"] as? String
        guard let recipe = OCRRecipe.of(modelType: modelType, hints: [loaded.path, currentModelId]) else {
            throw .invalidRequestStructure(
                details: "the loaded model \(modelInfo) (model_type \(modelType ?? "unknown")) has no OCR recipe; load GLM-OCR (glm_ocr) or MinerU2.5 (qwen2_vl)")
        }
        return (recipe, loaded.epoch)
    }

    nonisolated func generated(_ request: GenerationRequest, epoch: Int) async throws(BasaltError) -> GenerationResponse {
        let response = try await generate(request: request)
        guard await loaded.epoch == epoch else { throw .serverBusy(retryAfterSeconds: 1) }
        return response
    }

    nonisolated func recognize(
        _ image: ImageInput, prompt: String, type: String, recipe: OCRRecipe, epoch: Int
    ) async throws(BasaltError) -> (markdown: String, blocks: [OCRBlock], tokens: Int) {
        let response = try await generated(recipe.request(prompt: prompt, image: image, stage: .recognition), epoch: epoch)
        let text = recipe.finished(response.generatedText, blockType: type)
        return (text, [OCRBlock(type: type, bbox: .page, text: text)], response.completionTokens)
    }

    nonisolated func parseLayout(
        _ page: ImageInput, prompt: String, side: Int, recipe: OCRRecipe, epoch: Int
    ) async throws(BasaltError) -> (markdown: String, blocks: [OCRBlock], tokens: Int) {
        let image = try OCRImage.decode(page.data)
        guard let downsampled = OCRImage.resized(image, width: side, height: side) else {
            throw .inferenceError("cannot downsample the page for layout detection")
        }
        let layout = try await generated(recipe.request(prompt: prompt, image: try OCRImage.pngInput(downsampled), stage: .layout), epoch: epoch)
        var tokens = layout.completionTokens
        var recognized: [(block: MinerULayout.Block, text: String)] = []
        for block in MinerULayout(parsing: layout.generatedText).blocks {
            guard let prompt = MinerULayout.prompt(forBlock: block.type), let crop = Self.crop(image, block) else {
                recognized.append((block, ""))
                continue
            }
            let response = try await generated(recipe.request(prompt: prompt, image: try OCRImage.pngInput(crop), stage: .recognition), epoch: epoch)
            tokens += response.completionTokens
            recognized.append((block, recipe.finished(response.generatedText, blockType: block.type)))
        }
        let assembled = MinerULayout.assemble(recognized)
        return (assembled.markdown, assembled.blocks, tokens)
    }

    nonisolated static func crop(_ page: CGImage, _ block: MinerULayout.Block) -> CGImage? {
        guard let crop = OCRImage.crop(page, to: block.box),
              let upright = OCRImage.rotated(crop, counterclockwise: block.angle ?? 0) else { return nil }
        return OCRImage.fitted(upright, minEdge: MinerULayout.minEdge, maxRatio: MinerULayout.maxEdgeRatio)
    }
}
