import Foundation
import BasaltCore

extension ImagesController {
    package struct ModelsEnvelope: Codable, Sendable, Equatable {
        package let models: [ImageModelDescriptor]
    }

    struct ImagesResponseEnvelope: Codable, Sendable, Equatable {
        let created: Int
        let data: [ImageWireItem]
    }

    struct ImageWireItem: Codable, Sendable, Equatable {
        let assetID: String
        let url: String
        let b64JSON: String
        let revisedPrompt: String?
        let width: Int
        let height: Int
        let format: String
        let hasAlpha: Bool
        let seed: UInt64
        let steps: Int
        let teacacheSkipped: Int
        let referenceKVReused: Bool
        let upscaledBy: Int
        let model: String
        let licenseTier: String
        let elapsedMs: Int

        init(
            artifact: ImageCompletedArtifact,
            persisted: ImageArtifactStore.PersistedAsset,
            elapsedMs: Int
        ) {
            self.assetID = persisted.assetID
            self.url = persisted.url
            self.b64JSON = artifact.imageData.base64EncodedString()
            self.revisedPrompt = artifact.upsampledPrompt
            self.width = artifact.width
            self.height = artifact.height
            self.format = artifact.format.rawValue
            self.hasAlpha = artifact.hasAlpha
            self.seed = artifact.seed
            self.steps = artifact.stepsExecuted
            self.teacacheSkipped = artifact.teacacheSkippedSteps
            self.referenceKVReused = artifact.referenceKVReused
            self.upscaledBy = artifact.upscaledBy
            self.model = artifact.model
            self.licenseTier = artifact.licenseTier.rawValue
            self.elapsedMs = elapsedMs
        }

        enum CodingKeys: String, CodingKey {
            case url, width, height, format, seed, steps, model
            case assetID = "asset_id"
            case b64JSON = "b64_json"
            case revisedPrompt = "revised_prompt"
            case hasAlpha = "has_alpha"
            case teacacheSkipped = "teacache_skipped"
            case referenceKVReused = "reference_kv_reused"
            case upscaledBy = "upscaled_by"
            case licenseTier = "license_tier"
            case elapsedMs = "elapsed_ms"
        }
    }

    struct ImagePartialSSEEvent: Codable, Sendable, Equatable {
        let type: String
        let step: Int
        let totalSteps: Int
        let b64JSON: String
        let width: Int
        let height: Int
        let format: String

        init(frame: ImagePartialFrame) {
            self.type = "image_generation.partial_image"
            self.step = frame.step
            self.totalSteps = frame.totalSteps
            self.b64JSON = frame.imageData.base64EncodedString()
            self.width = frame.width
            self.height = frame.height
            self.format = frame.format.rawValue
        }

        enum CodingKeys: String, CodingKey {
            case type, step, width, height, format
            case totalSteps = "total_steps"
            case b64JSON = "b64_json"
        }
    }

    struct ImageCompletedSSEEvent: Codable, Sendable, Equatable {
        let type: String
        let assetID: String
        let url: String
        let b64JSON: String
        let revisedPrompt: String?
        let width: Int
        let height: Int
        let format: String
        let hasAlpha: Bool
        let seed: UInt64
        let steps: Int
        let teacacheSkipped: Int
        let referenceKVReused: Bool
        let upscaledBy: Int
        let model: String
        let licenseTier: String
        let elapsedMs: Int

        init(
            artifact: ImageCompletedArtifact,
            persisted: ImageArtifactStore.PersistedAsset,
            elapsedMs: Int
        ) {
            self.type = "image_generation.completed"
            self.assetID = persisted.assetID
            self.url = persisted.url
            self.b64JSON = artifact.imageData.base64EncodedString()
            self.revisedPrompt = artifact.upsampledPrompt
            self.width = artifact.width
            self.height = artifact.height
            self.format = artifact.format.rawValue
            self.hasAlpha = artifact.hasAlpha
            self.seed = artifact.seed
            self.steps = artifact.stepsExecuted
            self.teacacheSkipped = artifact.teacacheSkippedSteps
            self.referenceKVReused = artifact.referenceKVReused
            self.upscaledBy = artifact.upscaledBy
            self.model = artifact.model
            self.licenseTier = artifact.licenseTier.rawValue
            self.elapsedMs = elapsedMs
        }

        enum CodingKeys: String, CodingKey {
            case type, url, width, height, format, seed, steps, model
            case assetID = "asset_id"
            case b64JSON = "b64_json"
            case revisedPrompt = "revised_prompt"
            case hasAlpha = "has_alpha"
            case teacacheSkipped = "teacache_skipped"
            case referenceKVReused = "reference_kv_reused"
            case upscaledBy = "upscaled_by"
            case licenseTier = "license_tier"
            case elapsedMs = "elapsed_ms"
        }
    }
}

struct WireImageRequest: Decodable {
    var model: String?
    var prompt: String?
    var negativePrompt: String?
    var size: String?
    var width: Int?
    var height: Int?
    var steps: Int?
    var outputFormat: String?
    var format: String?
    var stream: Bool?
    var seed: UInt64?
    var guidanceScale: Double?
    var background: String?
    var teacacheThreshold: Double?
    var previewEverySteps: Int?
    var partialImages: Int?
    var candidates: Int?
    var n: Int?
    var bboxLayout: [ImageBoundingBox]?
    var vglSceneJSON: String?
    var image: String?
    var images: [String]?
    var referenceImages: [String]?
    var mask: String?
    var audio: String?
    var audioStrength: Double?
    var reuseReferenceKV: Bool?
    var upscaleFactor: Int?

    enum CodingKeys: String, CodingKey {
        case model, prompt, size, width, height, steps, format, stream, seed
        case background, candidates, n, image, images, mask, audio
        case negativePrompt = "negative_prompt"
        case outputFormat = "output_format"
        case guidanceScale = "guidance_scale"
        case teacacheThreshold = "teacache_threshold"
        case previewEverySteps = "preview_every_steps"
        case partialImages = "partial_images"
        case bboxLayout = "bbox_layout"
        case vglSceneJSON = "vgl_scene_json"
        case referenceImages = "reference_images"
        case audioStrength = "audio_strength"
        case reuseReferenceKV = "reuse_reference_kv"
        case upscaleFactor = "upscale_factor"
    }
}

extension WireImageRequest {
    static func parseGeneration(
        _ data: Data,
        defaultModel: String,
        defaultSteps: Int
    ) throws(BasaltError) -> ImageGenerationRequest {
        let wire = try decodeJSON(data)
        return try wire.buildRequest(defaultModel: defaultModel, defaultSteps: defaultSteps)
    }

    static func parseJSONEdit(
        _ data: Data,
        defaultModel: String,
        defaultSteps: Int
    ) throws(BasaltError) -> ImageGenerationRequest {
        let wire = try decodeJSON(data)
        return try wire.buildRequest(defaultModel: defaultModel, defaultSteps: defaultSteps)
    }

    private static func decodeJSON(_ data: Data) throws(BasaltError) -> WireImageRequest {
        do {
            return try JSONDecoder().decode(WireImageRequest.self, from: data)
        } catch {
            throw .invalidRequestStructure(details: "image: body must be valid JSON matching /v1/images/*")
        }
    }

    func buildRequest(defaultModel: String, defaultSteps: Int) throws(BasaltError) -> ImageGenerationRequest {
        let dimensions = try resolveDimensions()
        let resolvedFormat = try Self.parseFormat(outputFormat ?? format)
        let resolvedBackground = try Self.parseBackground(background)
        var config = ImageGenerationConfig(
            guidanceScale: guidanceScale ?? 3.5,
            seed: seed,
            background: resolvedBackground,
            teacacheThreshold: teacacheThreshold ?? 0.0,
            previewEverySteps: previewEverySteps ?? partialImages ?? 0
        )
        config.candidates = candidates ?? n ?? 1
        config.bboxLayout = bboxLayout ?? []
        config.vglSceneJSON = vglSceneJSON
        config.referenceImages = try decodeReferenceImages()
        if let mask {
            config.maskImage = try ImageArtifactStore.decodeBase64Payload(mask, field: "mask")
        }
        if let audio {
            config.audioConditioning = try ImageArtifactStore.decodeBase64Payload(audio, field: "audio")
        }
        config.audioStrength = audioStrength
        config.reuseReferenceKV = reuseReferenceKV ?? true
        config.upscaleFactor = upscaleFactor
        var request = try ImageGenerationRequest(
            model: (model?.isEmpty == false ? model : nil) ?? defaultModel,
            prompt: prompt ?? "",
            dimensions: dimensions,
            steps: steps ?? defaultSteps,
            outputFormat: resolvedFormat,
            config: config
        )
        request.negativePrompt = negativePrompt
        request.stream = stream ?? false
        return request
    }

    private func resolveDimensions() throws(BasaltError) -> ImageDimensions {
        if let size, !size.isEmpty {
            return try ImageGenerationRequest.parseSize(size)
        }
        if let width, let height {
            return ImageDimensions(width: width, height: height)
        }
        return ImageDimensions()
    }

    private func decodeReferenceImages() throws(BasaltError) -> [Data] {
        var encoded: [String] = []
        if let image { encoded.append(image) }
        if let images { encoded.append(contentsOf: images) }
        if let referenceImages { encoded.append(contentsOf: referenceImages) }
        var decoded: [Data] = []
        decoded.reserveCapacity(encoded.count)
        for item in encoded {
            decoded.append(try ImageArtifactStore.decodeBase64Payload(item, field: "image"))
        }
        return decoded
    }
}

extension WireImageRequest {
    static func parseMultipartEdit(
        _ body: Data,
        boundary: String,
        defaultModel: String,
        defaultSteps: Int
    ) throws(BasaltError) -> ImageGenerationRequest {
        let form = try MultipartForm(body, boundary: boundary, repeatable: ["image", "image[]"], limit: 24)
        let imageParts = form.values("image") + form.values("image[]")
        let sizeText = try form.text("size")
        let dims = if let sizeText, !sizeText.isEmpty { try ImageGenerationRequest.parseSize(sizeText) } else { ImageDimensions() }
        let formatField: String? = if let explicit = try form.text("output_format") {
            explicit
        } else {
            try form.text("format")
        }
        let outFormat = try parseFormat(formatField)
        let bg = try parseBackground(try form.text("background"))
        let parsedSteps = try parseIntField(form, "steps") ?? defaultSteps
        var config = ImageGenerationConfig(
            guidanceScale: try parseDoubleField(form, "guidance_scale") ?? 3.5,
            seed: try parseUInt64Field(form, "seed"),
            background: bg,
            teacacheThreshold: try parseDoubleField(form, "teacache_threshold") ?? 0.0,
            previewEverySteps: try parseIntField(form, "preview_every_steps") ?? 0
        )
        config.candidates = try parseIntField(form, "candidates") ?? 1
        config.referenceImages = imageParts.map(\.body)
        config.maskImage = form.parts["mask"]?.body
        config.audioConditioning = form.parts["audio"]?.body
        config.audioStrength = try parseDoubleField(form, "audio_strength")
        let modelField = try form.text("model")
        var request = try ImageGenerationRequest(
            model: (modelField?.isEmpty == false ? modelField : nil) ?? defaultModel,
            prompt: try form.text("prompt") ?? "",
            dimensions: dims,
            steps: parsedSteps,
            outputFormat: outFormat,
            config: config
        )
        request.negativePrompt = try form.text("negative_prompt")
        request.stream = try parseBoolField(form, "stream") ?? false
        return request
    }

    static func parseFormat(_ raw: String?) throws(BasaltError) -> ImageOutputFormat {
        guard let raw, !raw.isEmpty else { return .png }
        let normalized = raw.lowercased() == "jpg" ? "jpeg" : raw.lowercased()
        guard let format = ImageOutputFormat(rawValue: normalized) else {
            throw .invalidRequestStructure(details: "image: output_format must be png, webp or jpeg")
        }
        return format
    }

    static func parseBackground(_ raw: String?) throws(BasaltError) -> ImageBackgroundMode {
        guard let raw, !raw.isEmpty else { return .auto }
        guard let mode = ImageBackgroundMode(rawValue: raw.lowercased()) else {
            throw .invalidRequestStructure(details: "image: background must be opaque, transparent or auto")
        }
        return mode
    }

    private static func parseIntField(_ form: MultipartForm, _ name: String) throws(BasaltError) -> Int? {
        guard let raw = try form.text(name), !raw.isEmpty else { return nil }
        guard let val = Int(raw) else {
            throw .invalidRequestStructure(details: "image: form field \(name) must be an integer")
        }
        return val
    }

    private static func parseUInt64Field(_ form: MultipartForm, _ name: String) throws(BasaltError) -> UInt64? {
        guard let raw = try form.text(name), !raw.isEmpty else { return nil }
        guard let val = UInt64(raw) else {
            throw .invalidRequestStructure(details: "image: form field \(name) must be a non-negative integer")
        }
        return val
    }

    private static func parseDoubleField(_ form: MultipartForm, _ name: String) throws(BasaltError) -> Double? {
        guard let raw = try form.text(name), !raw.isEmpty else { return nil }
        guard let val = Double(raw), val.isFinite else {
            throw .invalidRequestStructure(details: "image: form field \(name) must be a finite number")
        }
        return val
    }

    private static func parseBoolField(_ form: MultipartForm, _ name: String) throws(BasaltError) -> Bool? {
        guard let raw = try form.text(name), !raw.isEmpty else { return nil }
        switch raw.lowercased() {
        case "true", "1": return true
        case "false", "0": return false
        default: throw .invalidRequestStructure(details: "image: form field \(name) must be true or false")
        }
    }
}
