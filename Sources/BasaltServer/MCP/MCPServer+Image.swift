import Foundation
import BasaltCore

package enum MCPToolName: String, Sendable, Equatable, CaseIterable {
    case generateImage = "basalt_generate_image"
    case editImage = "basalt_edit_image"

    static func isAllowed(_ name: String, by filter: MCPToolFilter) -> Bool {
        !filter.isExplicitlyDisabled(name) && !filter.disabled.isSuperset(of: MCPToolFilter.Group.media.tools)
    }
}

package struct MCPImageToolOutput: Codable, Sendable, Equatable {
    package let assetID: String
    package let path: String
    package let url: String
    package let width: Int
    package let height: Int
    package let format: String
    package let hasAlpha: Bool
    package let seed: UInt64
    package let model: String
    package let licenseTier: String
    package let steps: Int
    package let teacacheSkipped: Int
    package let revisedPrompt: String?

    init(artifact: ImageCompletedArtifact, persisted: ImageArtifactStore.PersistedAsset) {
        self.assetID = persisted.assetID
        self.path = persisted.path
        self.url = persisted.url
        self.width = artifact.width
        self.height = artifact.height
        self.format = artifact.format.rawValue
        self.hasAlpha = artifact.hasAlpha
        self.seed = artifact.seed
        self.model = artifact.model
        self.licenseTier = artifact.licenseTier.rawValue
        self.steps = artifact.stepsExecuted
        self.teacacheSkipped = artifact.teacacheSkippedSteps
        self.revisedPrompt = artifact.upsampledPrompt
    }

    enum CodingKeys: String, CodingKey {
        case path, url, width, height, format, seed, model, steps
        case assetID = "asset_id"
        case hasAlpha = "has_alpha"
        case licenseTier = "license_tier"
        case teacacheSkipped = "teacache_skipped"
        case revisedPrompt = "revised_prompt"
    }
}

extension MCPServer {
    func generateImageTool(_ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        try await executeImageTool(args, tool: .generateImage, requiresEditInputs: false)
    }

    func editImageTool(_ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        try await executeImageTool(args, tool: .editImage, requiresEditInputs: true)
    }

    private func executeImageTool(
        _ args: [String: Any]?,
        tool: MCPToolName,
        requiresEditInputs: Bool
    ) async throws(MCPFailure) -> ToolResult {
        guard let synthesizer = imageSynthesizer else {
            throw .methodNotFound("\(tool.rawValue) needs an image model: start serve with --image-model <repo|dir> or compose imagine:")
        }
        let request = try MCPArguments.imageGeneration(
            args,
            defaultModel: synthesizer.modelIdentifier,
            defaultSteps: synthesizer.descriptor.defaultSteps,
            allowedIn: imageDirectories,
            requiresEditInputs: requiresEditInputs
        )
        do {
            let stream = try await synthesizer.generate(request)
            var finalArtifact: ImageCompletedArtifact?
            for try await event in stream {
                if case .completed(let completed) = event {
                    finalArtifact = completed
                }
            }
            guard let artifact = finalArtifact else {
                throw BasaltError.modelInferenceError(details: "image: synthesizer finished without a completed artifact")
            }
            let persisted = try ImageArtifactStore.persist(artifact, in: imageStoreDirectory)
            return .json(MCPImageToolOutput(artifact: artifact, persisted: persisted))
        } catch let error as BasaltError {
            throw MCPFailure(error)
        } catch {
            throw .internalError(error.localizedDescription)
        }
    }
}

extension MCPArguments {
    static func imageGeneration(
        _ args: [String: Any]?,
        defaultModel: String,
        defaultSteps: Int,
        allowedIn directories: [URL],
        requiresEditInputs: Bool
    ) throws(MCPFailure) -> ImageGenerationRequest {
        let prompt = try string("prompt", in: args) ?? ""
        let negSnake = try string("negative_prompt", in: args)
        let negative = if let negSnake { negSnake } else { try string("negativePrompt", in: args) }
        let model = try string("model", in: args)
        let size = try string("size", in: args)
        let steps = try integer("steps", in: args, 1 ... 100) ?? defaultSteps
        let fmtPrimary = try string("format", in: args)
        let rawFormat = if let fmtPrimary { fmtPrimary } else { try string("output_format", in: args) }
        let config = try buildImageConfig(args, allowedIn: directories, requiresEditInputs: requiresEditInputs)
        do throws(BasaltError) {
            let dims = if let size, !size.isEmpty { try ImageGenerationRequest.parseSize(size) } else { ImageDimensions() }
            let outFormat = try WireImageRequest.parseFormat(rawFormat)
            var request = try ImageGenerationRequest(
                model: (model?.isEmpty == false ? model : nil) ?? defaultModel,
                prompt: prompt,
                dimensions: dims,
                steps: steps,
                outputFormat: outFormat,
                config: config
            )
            request.negativePrompt = negative
            return request
        } catch {
            throw MCPFailure(error)
        }
    }

    private static func buildImageConfig(
        _ args: [String: Any]?,
        allowedIn directories: [URL],
        requiresEditInputs: Bool
    ) throws(MCPFailure) -> ImageGenerationConfig {
        let candidates = try integer("candidates", in: args, 1 ... 64) ?? 1
        let seedVal = try integer("seed", in: args, 0 ... Int.max).map { UInt64($0) }
        let guidance = try number("guidance_scale", in: args).map(Double.init) ?? 3.5
        let teacache = try number("teacache_threshold", in: args).map(Double.init) ?? 0.0
        let rawBg = try string("background", in: args)
        do throws(BasaltError) {
            let bg = try WireImageRequest.parseBackground(rawBg)
            var config = ImageGenerationConfig(
                guidanceScale: guidance,
                seed: seedVal,
                background: bg,
                teacacheThreshold: teacache
            )
            config.candidates = candidates
            config.referenceImages = try decodeMCPReferences(args, allowedIn: directories)
            config.maskImage = try decodeMCPMask(args, allowedIn: directories)
            if let audioB64 = args?["audio"] as? String {
                config.audioConditioning = try ImageArtifactStore.decodeBase64Payload(audioB64, field: "audio")
            }
            if requiresEditInputs, config.referenceImages.isEmpty, config.maskImage == nil, config.audioConditioning == nil {
                throw .invalidRequestStructure(details: "image: basalt_edit_image requires image/images/path/paths, mask or audio")
            }
            return config
        } catch {
            throw MCPFailure(error)
        }
    }

    private static func decodeMCPReferences(_ args: [String: Any]?, allowedIn directories: [URL]) throws(BasaltError) -> [Data] {
        var refs: [Data] = []
        if let single = args?["image"] as? String {
            refs.append(try ImageArtifactStore.decodeBase64Payload(single, field: "image"))
        }
        if let list = args?["images"] as? [String] {
            for encoded in list {
                refs.append(try ImageArtifactStore.decodeBase64Payload(encoded, field: "images"))
            }
        }
        if let path = args?["path"] as? String {
            refs.append(try MCPImages.read(path, allowedIn: directories).data)
        }
        if let paths = args?["paths"] as? [String] {
            for path in paths {
                refs.append(try MCPImages.read(path, allowedIn: directories).data)
            }
        }
        return refs
    }

    private static func decodeMCPMask(_ args: [String: Any]?, allowedIn directories: [URL]) throws(BasaltError) -> Data? {
        if let mask = args?["mask"] as? String {
            return try ImageArtifactStore.decodeBase64Payload(mask, field: "mask")
        }
        if let maskPath = args?["mask_path"] as? String {
            return try MCPImages.read(maskPath, allowedIn: directories).data
        }
        return nil
    }
}

extension MCPToolCatalog {
    static let generateImage = MCPToolDescriptor(
        name: MCPToolName.generateImage.rawValue,
        description: "Generate an image on Apple Silicon with the configured MLX image model (FLUX.2/3-klein, Z-Image-Turbo, Iris-3B, SANA-1.5, Bonsai-4B), persist it under ~/.basalt/images/<sha256>.<ext>, and return its path, URL and license/provenance metadata.",
        inputSchema: ResearchTool.object([
            "prompt": ["type": "string", "description": "Visual description of the image to synthesize"],
            "negative_prompt": ["type": "string", "description": "Optional elements to avoid"],
            "model": ["type": "string", "description": "Optional image model identifier or chain alias"],
            "size": ["type": "string", "description": "Dimensions as <width>x<height>, multiples of 16 (default 1024x1024)"],
            "steps": ["type": "integer", "minimum": 1, "maximum": 100, "description": "Denoising steps (default: model default)"],
            "format": ["type": "string", "enum": ImageOutputFormat.allCases.map(\.rawValue), "description": "Output image format: png, webp or jpeg"],
            "background": ["type": "string", "enum": ImageBackgroundMode.allCases.map(\.rawValue), "description": "Background mode: opaque, transparent or auto"],
            "seed": ["type": "integer", "minimum": 0, "description": "Deterministic RNG seed"],
            "guidance_scale": ["type": "number", "description": "Classifier-free guidance scale (default 3.5)"],
            "teacache_threshold": ["type": "number", "description": "TeaCache residual skip threshold (default 0)"],
            "candidates": ["type": "integer", "minimum": 1, "maximum": 64, "description": "Best-of-N candidate count when judge_via is configured"],
            "audio": ["type": "string", "description": "Optional base64 audio clip for audio-conditioned generation"],
        ], required: ["prompt"])
    )

    static let editImage = MCPToolDescriptor(
        name: MCPToolName.editImage.rawValue,
        description: "Edit or recompose up to 10 reference images (plus optional mask or audio conditioning) with the configured MLX image model, persist the result under ~/.basalt/images/<sha256>.<ext>, and return its path, URL and provenance metadata.",
        inputSchema: ResearchTool.object([
            "prompt": ["type": "string", "description": "Edit instruction"],
            "image": ["type": "string", "description": "Primary reference image, base64-encoded"],
            "images": ["type": "array", "items": ["type": "string"], "description": "Up to 10 reference images, base64-encoded"],
            "path": ["type": "string", "description": "Reference image path inside a serve --image-dir directory"],
            "paths": ["type": "array", "items": ["type": "string"], "description": "Reference image paths inside serve --image-dir directories"],
            "mask": ["type": "string", "description": "Optional inpainting mask image, base64-encoded"],
            "mask_path": ["type": "string", "description": "Optional mask path inside a serve --image-dir directory"],
            "audio": ["type": "string", "description": "Optional base64 audio clip for audio-conditioned editing"],
            "model": ["type": "string", "description": "Optional image model identifier"],
            "size": ["type": "string", "description": "Dimensions as <width>x<height>, multiples of 16 (default 1024x1024)"],
            "steps": ["type": "integer", "minimum": 1, "maximum": 100, "description": "Denoising steps"],
            "format": ["type": "string", "enum": ImageOutputFormat.allCases.map(\.rawValue), "description": "Output image format: png, webp or jpeg"],
            "background": ["type": "string", "enum": ImageBackgroundMode.allCases.map(\.rawValue), "description": "Background mode: opaque, transparent or auto"],
            "seed": ["type": "integer", "minimum": 0, "description": "Deterministic RNG seed"],
        ], required: ["prompt"])
    )
}
