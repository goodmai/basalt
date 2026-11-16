import Foundation

public enum ImageOutputFormat: String, Sendable, Codable, Equatable, CaseIterable {
    case png
    case webp
    case jpeg

    public var contentType: String {
        switch self {
        case .png: "image/png"
        case .webp: "image/webp"
        case .jpeg: "image/jpeg"
        }
    }

    public var fileExtension: String { rawValue }
}

public enum ImageBackgroundMode: String, Sendable, Codable, Equatable, CaseIterable {
    case opaque
    case transparent
    case auto
}

public enum ImageComputeBackend: String, Sendable, Codable, Equatable, CaseIterable {
    case mlx
    case mlxANEHybrid = "mlx_ane_hybrid"
    case cloud
}

public enum ImageOnGPUBusy: String, Sendable, Codable, Equatable, CaseIterable {
    case queue
    case fallback
}

public enum ImageLicenseTier: String, Sendable, Codable, Equatable, CaseIterable {
    case permissiveCommercial = "permissive_commercial"
    case revenueCappedCommercial = "revenue_capped_commercial"
    case nonCommercialResearch = "non_commercial_research"

    public var allowsUnrestrictedCommercialUse: Bool {
        self == .permissiveCommercial
    }
}

public struct ImageDimensions: Sendable, Codable, Equatable, Hashable {
    public var width: Int
    public var height: Int

    public init(width: Int = 1_024, height: Int = 1_024) {
        self.width = width
        self.height = height
    }
}

public struct ImageBoundingBox: Sendable, Codable, Equatable, Hashable {
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

public struct ImageGenerationConfig: Sendable, Codable, Equatable {
    public var guidanceScale: Double
    public var seed: UInt64?
    public var background: ImageBackgroundMode
    public var teacacheThreshold: Double
    public var previewEverySteps: Int
    public var candidates: Int
    public var bboxLayout: [ImageBoundingBox]
    public var vglSceneJSON: String?
    public var referenceImages: [Data]
    public var maskImage: Data?
    public var audioConditioning: Data?
    public var audioStrength: Double?
    public var upsampledPrompt: String?

    public init(
        guidanceScale: Double = 3.5,
        seed: UInt64? = nil,
        background: ImageBackgroundMode = .auto,
        teacacheThreshold: Double = 0.0,
        previewEverySteps: Int = 0
    ) {
        self.guidanceScale = guidanceScale
        self.seed = seed
        self.background = background
        self.teacacheThreshold = teacacheThreshold
        self.previewEverySteps = previewEverySteps
        self.candidates = 1
        self.bboxLayout = []
        self.vglSceneJSON = nil
        self.referenceImages = []
        self.maskImage = nil
        self.audioConditioning = nil
        self.audioStrength = nil
        self.upsampledPrompt = nil
    }
}

public struct ImageGenerationRequest: Sendable, Codable, Equatable {
    public static let maxPromptCharacters = 16_384
    public static let minDimension = 64
    public static let maxDimension = 4_096
    public static let maxReferenceImages = 10

    public var model: String
    public var prompt: String
    public var negativePrompt: String?
    public var width: Int
    public var height: Int
    public var steps: Int
    public var outputFormat: ImageOutputFormat
    public var stream: Bool
    public var config: ImageGenerationConfig

    public init(
        model: String,
        prompt: String,
        dimensions: ImageDimensions = ImageDimensions(),
        steps: Int = 8,
        outputFormat: ImageOutputFormat = .png,
        config: ImageGenerationConfig = ImageGenerationConfig()
    ) throws(BasaltError) {
        self.model = model
        self.prompt = prompt
        self.negativePrompt = nil
        self.width = dimensions.width
        self.height = dimensions.height
        self.steps = steps
        self.outputFormat = outputFormat
        self.stream = false
        self.config = config
        try validate()
    }

    public var effectivePrompt: String {
        if let upsampled = config.upsampledPrompt, !upsampled.isEmpty {
            return upsampled
        }
        return prompt
    }

    public var isEditRequest: Bool {
        !config.referenceImages.isEmpty || config.maskImage != nil
    }

    public var isAudioConditioned: Bool {
        config.audioConditioning != nil
    }

    public func validate() throws(BasaltError) {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty && config.audioConditioning == nil {
            throw .invalidRequestStructure(details: "image: prompt must not be empty unless audio conditioning is provided")
        }
        if prompt.count > Self.maxPromptCharacters {
            throw .invalidRequestStructure(details: "image: prompt exceeds \(Self.maxPromptCharacters) characters")
        }
        guard (Self.minDimension ... Self.maxDimension).contains(width),
              (Self.minDimension ... Self.maxDimension).contains(height),
              width.isMultiple(of: 16), height.isMultiple(of: 16) else {
            throw .invalidRequestStructure(
                details: "image: dimensions \(width)x\(height) must be multiples of 16 within \(Self.minDimension)...\(Self.maxDimension)"
            )
        }
        guard (1 ... 100).contains(steps) else {
            throw .invalidRequestStructure(details: "image: steps must be within 1...100, got \(steps)")
        }
        guard config.referenceImages.count <= Self.maxReferenceImages else {
            throw .invalidRequestStructure(details: "image: at most \(Self.maxReferenceImages) reference images are supported")
        }
        guard (1 ... 64).contains(config.candidates) else {
            throw .invalidRequestStructure(details: "image: candidates must be within 1...64")
        }
    }

    public static func parseSize(_ size: String) throws(BasaltError) -> ImageDimensions {
        let parts = size.lowercased().split(separator: "x", omittingEmptySubsequences: false)
        guard parts.count == 2, let w = Int(parts[0]), let h = Int(parts[1]) else {
            throw .invalidRequestStructure(details: "image: size '\(size)' must have format <width>x<height>, e.g. 1024x1024")
        }
        return ImageDimensions(width: w, height: h)
    }
}

public struct ImagePartialFrame: Sendable, Codable, Equatable {
    public var step: Int
    public var totalSteps: Int
    public var imageData: Data
    public var format: ImageOutputFormat
    public var dimensions: ImageDimensions

    public init(step: Int, totalSteps: Int, imageData: Data, format: ImageOutputFormat, dimensions: ImageDimensions) {
        self.step = step
        self.totalSteps = totalSteps
        self.imageData = imageData
        self.format = format
        self.dimensions = dimensions
    }

    public var width: Int { dimensions.width }
    public var height: Int { dimensions.height }
}

public struct ImageCompletedArtifact: Sendable, Codable, Equatable {
    public var imageData: Data
    public var format: ImageOutputFormat
    public var dimensions: ImageDimensions
    public var hasAlpha: Bool
    public var seed: UInt64
    public var stepsExecuted: Int
    public var teacacheSkippedSteps: Int
    public var model: String
    public var compute: ImageComputeBackend
    public var licenseTier: ImageLicenseTier
    public var upsampledPrompt: String?
    public var elapsedMs: Int

    public init(
        imageData: Data,
        format: ImageOutputFormat,
        dimensions: ImageDimensions,
        hasAlpha: Bool = false,
        seed: UInt64 = 0,
        model: String
    ) {
        self.imageData = imageData
        self.format = format
        self.dimensions = dimensions
        self.hasAlpha = hasAlpha
        self.seed = seed
        self.stepsExecuted = 1
        self.teacacheSkippedSteps = 0
        self.model = model
        self.compute = .mlx
        self.licenseTier = .permissiveCommercial
        self.upsampledPrompt = nil
        self.elapsedMs = 0
    }

    public var width: Int { dimensions.width }
    public var height: Int { dimensions.height }
}

public enum ImageGenerationEvent: Sendable, Codable, Equatable {
    case partial(ImagePartialFrame)
    case completed(ImageCompletedArtifact)
}

public struct ImageRouteDecision: Sendable, Codable, Equatable {
    public var selectedModel: String
    public var compute: ImageComputeBackend
    public var fallbackFrom: String?
    public var sharedEncoderModel: String?
    public var upsamplerModel: String?
    public var judgeModel: String?

    public init(selectedModel: String, compute: ImageComputeBackend = .mlx, fallbackFrom: String? = nil) {
        self.selectedModel = selectedModel
        self.compute = compute
        self.fallbackFrom = fallbackFrom
        self.sharedEncoderModel = nil
        self.upsamplerModel = nil
        self.judgeModel = nil
    }
}

public protocol ImageSynthesizer: Sendable {
    var modelIdentifier: String { get }
    var descriptor: ImageModelDescriptor { get }
    func resolveRoute(_ request: ImageGenerationRequest) async -> ImageRouteDecision
    func generate(_ request: ImageGenerationRequest) async throws -> AsyncThrowingStream<ImageGenerationEvent, any Error>
}

extension ImageSynthesizer {
    public func resolveRoute(_ request: ImageGenerationRequest) async -> ImageRouteDecision {
        _ = request
        return ImageRouteDecision(selectedModel: modelIdentifier, compute: descriptor.defaultCompute)
    }
}
