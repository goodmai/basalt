import Foundation
import MLX
import MLXLMCommon
import BasaltCore

extension MLXInferenceEngine {
    func loadVisionTower(_ loaded: ModelContainer, epoch: ModelEpoch) async {
        guard let path = configuration.visionTowerBundle else { return }
        do {
            let tower = try await CoreAIVisionTower(directory: URL(fileURLWithPath: path))
            let fits = await loaded.perform { context in (context.model as? PaddleOCRVL).map { tower.fits($0.visual, frames: []) } ?? false }
            guard fits else {
                logger.warn("--experimental-ane-vision-tower: \(path) does not fit the loaded model (PaddleOCR-VL only); its vision tower stays on the GPU")
                return
            }
            record(.learned(epoch) { $0.visionTower = tower })
            logger.info("Vision tower on the Neural Engine from \(path), buckets of \(tower.bundle.functions.map(\.tokens)) patches")
        } catch {
            logger.warn("--experimental-ane-vision-tower: \(error.localizedDescription); the vision tower stays on the GPU")
        }
    }

    func encodingImages(_ input: LMInput, container: ModelContainer) async throws -> LMInput {
        guard let tower = modelFacts.visionTower, input.image != nil else { return input }
        return try await container.performSending(nonSendable: input) { context, input in
            guard let model = context.model as? PaddleOCRVL, let image = input.image,
                  let features = try await tower.features(image, visual: model.visual) else { return input }
            eval(features)
            return LMInput(text: input.text, image: LMInput.ProcessedImage(pixels: features, frames: image.frames))
        }
    }
}
