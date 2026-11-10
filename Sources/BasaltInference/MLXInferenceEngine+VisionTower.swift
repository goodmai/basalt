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
        guard let tower = modelFacts.visionTower, let image = input.image, await tower.admission.enter() else { return input }
        defer { await tower.admission.leave() }
        let epoch = modelEpoch
        let encoded = try await container.performEvaluating(nonSendable: image) { context, image in
            try (context.model as? PaddleOCRVL).flatMap { try tower.inputs(image, visual: $0.visual) }
        }
        guard let encoded, epoch == modelEpoch else { return input }
        let outputs = try await tower.run(encoded)
        guard epoch == modelEpoch else { return input }
        return try await container.performSending(nonSendable: input) { context, input in
            let dtype = (context.model as? PaddleOCRVL)?.visual.embeddings.patchEmbedding.weight.dtype ?? .bfloat16
            let features = try CoreAIVisionTower.features(outputs, encoded: encoded, dtype: dtype)
            eval(features)
            return LMInput(text: input.text, image: LMInput.ProcessedImage(pixels: features, frames: input.image?.frames))
        }
    }
}
