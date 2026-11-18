import Foundation
import BasaltCore

public enum ImageSynthesizerLoader {
    public static func load(
        identifier: String,
        directory: URL,
        sharedTextEmbedder: (@Sendable (String) async throws -> [Float])? = nil,
        externalDecoderDirectory: URL? = nil
    ) async throws -> any ImageSynthesizer {
        try Task.checkCancellation()
        let (descriptor, _) = try ImageRunnerCore.parseDirectoryConfig(
            directory: directory,
            identifier: identifier,
            externalDecoderDirectory: externalDecoderDirectory
        )
        switch descriptor.family {
        case .fluxKlein, .flux3Klein, .fiboPerceiver:
            return try FluxKleinSynthesizer(
                directory: directory,
                identifier: identifier,
                sharedTextEmbedder: sharedTextEmbedder,
                externalDecoderDirectory: externalDecoderDirectory
            )
        case .s3DiT, .step1XReasonEdit:
            return try S3DiTSynthesizer(
                directory: directory,
                identifier: identifier,
                sharedTextEmbedder: sharedTextEmbedder,
                externalDecoderDirectory: externalDecoderDirectory
            )
        case .pixelDiT, .linearSana, .ternaryBitNet, .unifiedMoT, .audioToImage:
            return try PixelDiTSynthesizer(
                directory: directory,
                identifier: identifier,
                sharedTextEmbedder: sharedTextEmbedder,
                externalDecoderDirectory: externalDecoderDirectory
            )
        }
    }

    public static func inspect(
        identifier: String,
        directory: URL
    ) throws(BasaltError) -> ImageModelDescriptor {
        let (descriptor, _) = try ImageRunnerCore.parseDirectoryConfig(
            directory: directory,
            identifier: identifier
        )
        return descriptor
    }

    @discardableResult
    public static func probeImageSynthesizer(
        _ synthesizer: any ImageSynthesizer
    ) async throws(BasaltError) -> ImageCompletedArtifact {
        var cfg = ImageGenerationConfig(guidanceScale: 2.0, seed: 42)
        cfg.previewEverySteps = 0
        let request = try ImageGenerationRequest(
            model: synthesizer.modelIdentifier,
            prompt: "Basalt native MLX image probe",
            dimensions: ImageDimensions(width: 64, height: 64),
            steps: 2,
            outputFormat: .png,
            config: cfg
        )
        do {
            let stream = try await synthesizer.generate(request)
            var finalArtifact: ImageCompletedArtifact?
            for try await event in stream {
                if case .completed(let artifact) = event {
                    finalArtifact = artifact
                }
            }
            guard let finalArtifact, !finalArtifact.imageData.isEmpty else {
                throw BasaltError.modelProbeFailed(
                    path: synthesizer.modelIdentifier,
                    check: "image_synthesis",
                    reason: "synthesizer did not emit a non-empty completed artifact"
                )
            }
            return finalArtifact
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .modelProbeFailed(
                path: synthesizer.modelIdentifier,
                check: "image_synthesis",
                reason: error.localizedDescription
            )
        }
    }
}
