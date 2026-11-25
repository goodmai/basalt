import Foundation

public struct ChatImageBox: Sendable, Equatable, Codable, Hashable {
    public var label: String
    public var x: Double
    public var y: Double
    public var width: Double
    public var height: Double

    public init(label: String, x: Double, y: Double, width: Double, height: Double) {
        self.label = label
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }
}

public struct GeneratedImagePart: Sendable, Equatable, Codable {
    public var assetID: String
    public var filePath: String?
    public var beforeFilePath: String?
    public var width: Int
    public var height: Int
    public var hasAlpha = false
    public var prompt: String
    public var revisedPrompt: String?
    public var model: String
    public var seed: UInt64?
    public var currentStep = 0
    public var totalSteps: Int
    public var teacacheSkipped = 0
    public var referenceKVReused = false
    public var upscaledBy = 1
    public var licenseTier: String?
    public var boxes: [ChatImageBox] = []
    public var previewPNG: Data?

    public init(
        assetID: String = "",
        prompt: String,
        model: String = "z-image-turbo",
        width: Int = 1024,
        height: Int = 1024,
        totalSteps: Int = 8
    ) {
        self.assetID = assetID
        self.prompt = prompt
        self.model = model
        self.width = width
        self.height = height
        self.totalSteps = totalSteps
    }

    public var persisted: GeneratedImagePart {
        var copy = self
        copy.previewPNG = nil
        return copy
    }

    public var isStreamingPreview: Bool {
        filePath == nil
    }
}

public struct ImageGenerationCall: Sendable, Equatable, Codable {
    private struct WireRequest: Encodable {
        var model: String?
        var prompt: String
        var size: String
        var steps: Int
        var background: String
        var outputFormat: String
        var previewEverySteps: Int
        var images: [String]?
        var reuseReferenceKV: Bool
        var upscaleFactor: Int?
        var stream = true

        enum CodingKeys: String, CodingKey {
            case model, prompt, size, steps, background, images, stream
            case outputFormat = "output_format"
            case previewEverySteps = "preview_every_steps"
            case reuseReferenceKV = "reuse_reference_kv"
            case upscaleFactor = "upscale_factor"
        }
    }

    public var prompt: String
    public var model: String?
    public var size: String
    public var steps: Int
    public var background = "auto"
    public var outputFormat = "png"
    public var previewEverySteps = 2
    public var reuseReferenceKV = true
    public var upscaleFactor: Int?
    public var referenceImages: [Data]
    public var referenceImagePaths: [String] = []
    public var beforeFilePath: String?

    public init(
        prompt: String,
        model: String? = nil,
        size: String = "1024x1024",
        steps: Int = 8,
        referenceImages: [Data] = [],
        beforeFilePath: String? = nil
    ) {
        self.prompt = prompt
        self.model = model
        self.size = size
        self.steps = steps
        self.referenceImages = referenceImages
        self.beforeFilePath = beforeFilePath
    }

    public var endpointPath: String {
        (referenceImages.isEmpty && referenceImagePaths.isEmpty) ? "v1/images/generations" : "v1/images/edits"
    }

    public func wireBody() throws -> Data {
        let encodedImages = referenceImages.isEmpty ? nil : referenceImages.map { $0.base64EncodedString() }
        let payload = WireRequest(
            model: model,
            prompt: prompt,
            size: size,
            steps: steps,
            background: background,
            outputFormat: outputFormat,
            previewEverySteps: previewEverySteps,
            images: encodedImages,
            reuseReferenceKV: reuseReferenceKV,
            upscaleFactor: upscaleFactor
        )
        return try JSONEncoder().encode(payload)
    }
}

public enum ImageStreamEvent: Sendable, Equatable {
    case partial(step: Int, totalSteps: Int, png: Data, width: Int, height: Int)
    case completed(GeneratedImagePart, imageBytes: Data, elapsedMs: Int)
}
