import Foundation

public enum VideoContainerFormat: String, Sendable, Codable, Equatable, CaseIterable {
    case mp4
    case mov

    public var contentType: String {
        switch self {
        case .mp4: "video/mp4"
        case .mov: "video/quicktime"
        }
    }

    public var fileExtension: String { rawValue }
}

public enum VideoCodecKind: String, Sendable, Codable, Equatable, CaseIterable {
    case h264
    case hevc
}

public enum VideoAudioCodecKind: String, Sendable, Codable, Equatable, CaseIterable {
    case aac
    case linearPCM = "pcm"
}

public enum VideoExecutionPhase: String, Sendable, Codable, Equatable, CaseIterable {
    case textConditioning = "text_conditioning"
    case denoising = "denoising"
    case vaeAndAudioMuxing = "vae_and_audio_muxing"
}

public enum VideoExecutionMode: String, Sendable, Codable, Equatable, CaseIterable {
    case strictPhasedTeardown = "strict_phased"
    case hybridPhasedDiT = "hybrid_phased"
    case resident = "resident"
}

public struct VideoDimensions: Sendable, Codable, Equatable, Hashable {
    public static let defaultWidescreen = VideoDimensions(width: 832, height: 480)

    public var width: Int
    public var height: Int

    public init(width: Int = 512, height: Int = 512) {
        self.width = width
        self.height = height
    }
}

public struct VideoTemporalSpec: Sendable, Codable, Equatable, Hashable {
    public var frames: Int
    public var fps: Int
    public var steps: Int

    public init(frames: Int = 25, fps: Int = 24, steps: Int = 20) {
        self.frames = frames
        self.fps = fps
        self.steps = steps
    }
}

public struct VideoGenerationConfig: Sendable, Codable, Equatable {
    public static let supportedSampleRates: Set<Int> = [16_000, 22_050, 24_000, 44_100, 48_000]

    public var guidanceScale: Double
    public var seed: UInt64?
    public var codec: VideoCodecKind
    public var audioCodec: VideoAudioCodecKind
    public var includeAudio: Bool
    public var audioSampleRate: Int
    public var audioChannels: Int
    public var teacacheThreshold: Double
    public var staWindow: Int?
    public var previewEverySteps: Int
    public var forcePhasedTeardown: Bool

    public init(
        guidanceScale: Double = 5.0,
        seed: UInt64? = nil,
        codec: VideoCodecKind = .h264,
        includeAudio: Bool = true,
        teacacheThreshold: Double = 0.12
    ) {
        self.guidanceScale = guidanceScale
        self.seed = seed
        self.codec = codec
        self.audioCodec = .aac
        self.includeAudio = includeAudio
        self.audioSampleRate = 44_100
        self.audioChannels = 2
        self.teacacheThreshold = teacacheThreshold
        self.staWindow = nil
        self.previewEverySteps = 0
        self.forcePhasedTeardown = false
    }
}

public struct VideoGenerationRequest: Sendable, Codable, Equatable {
    public static let maxPromptCharacters = 16_384
    public static let minDimension = 64
    public static let maxDimension = 3_840
    public static let maxFrames = 257

    public var model: String
    public var prompt: String
    public var negativePrompt: String?
    public var width: Int
    public var height: Int
    public var frames: Int
    public var fps: Int
    public var steps: Int
    public var containerFormat: VideoContainerFormat
    public var stream: Bool
    public var config: VideoGenerationConfig

    public init(
        model: String,
        prompt: String,
        dimensions: VideoDimensions = VideoDimensions(),
        temporal: VideoTemporalSpec = VideoTemporalSpec(),
        containerFormat: VideoContainerFormat = .mp4,
        config: VideoGenerationConfig = VideoGenerationConfig()
    ) throws(BasaltError) {
        self.model = model
        self.prompt = prompt
        self.negativePrompt = nil
        self.width = dimensions.width
        self.height = dimensions.height
        self.frames = temporal.frames
        self.fps = temporal.fps
        self.steps = temporal.steps
        self.containerFormat = containerFormat
        self.stream = false
        self.config = config
        try validate()
    }

    public var durationSeconds: Double {
        Double(frames) / Double(max(1, fps))
    }

    public static func parseSize(_ size: String) throws(BasaltError) -> VideoDimensions {
        let parts = size.lowercased().split(separator: "x", omittingEmptySubsequences: false)
        guard parts.count == 2, let w = Int(parts[0]), let h = Int(parts[1]) else {
            throw .invalidRequestStructure(details: "video: size '\(size)' must have format <width>x<height>, e.g. 832x480")
        }
        return VideoDimensions(width: w, height: h)
    }
}

extension VideoGenerationRequest {
    public func validate() throws(BasaltError) {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw .invalidRequestStructure(details: "video: prompt must not be empty")
        }
        guard prompt.count <= Self.maxPromptCharacters else {
            throw .invalidRequestStructure(details: "video: prompt exceeds \(Self.maxPromptCharacters) characters")
        }
        guard (Self.minDimension ... Self.maxDimension).contains(width),
              (Self.minDimension ... Self.maxDimension).contains(height),
              width.isMultiple(of: 16), height.isMultiple(of: 16) else {
            throw .invalidRequestStructure(
                details: "video: dimensions \(width)x\(height) must be multiples of 16 within \(Self.minDimension)...\(Self.maxDimension)"
            )
        }
        guard (1 ... Self.maxFrames).contains(frames) else {
            throw .invalidRequestStructure(details: "video: frames must lie within 1...\(Self.maxFrames), got \(frames)")
        }
        guard (1 ... 120).contains(fps) else {
            throw .invalidRequestStructure(details: "video: fps must lie within 1...120, got \(fps)")
        }
        guard (1 ... 100).contains(steps) else {
            throw .invalidRequestStructure(details: "video: steps must lie within 1...100, got \(steps)")
        }
        try validateAudioAndSchedule()
    }

    private func validateAudioAndSchedule() throws(BasaltError) {
        guard VideoGenerationConfig.supportedSampleRates.contains(config.audioSampleRate) else {
            throw .invalidRequestStructure(details: "video: unsupported audio_sample_rate \(config.audioSampleRate)")
        }
        guard (1 ... 2).contains(config.audioChannels) else {
            throw .invalidRequestStructure(details: "video: audio_channels must be 1 or 2")
        }
        guard config.teacacheThreshold.isFinite, config.teacacheThreshold >= 0 else {
            throw .invalidRequestStructure(details: "video: teacache_threshold must be 0 or positive")
        }
        if let sta = config.staWindow, sta <= 0 {
            throw .invalidRequestStructure(details: "video: sta_window must be positive")
        }
    }
}

public struct VideoGenerationProgress: Sendable, Codable, Equatable {
    public var phase: VideoExecutionPhase
    public var step: Int
    public var totalSteps: Int
    public var fractionCompleted: Double
    public var teacacheSkipped: Bool
    public var previewPNG: Data?
    public var message: String

    public init(
        phase: VideoExecutionPhase,
        step: Int,
        totalSteps: Int,
        teacacheSkipped: Bool = false,
        previewPNG: Data? = nil,
        message: String = ""
    ) {
        self.phase = phase
        self.step = step
        self.totalSteps = max(1, totalSteps)
        let rawFrac = Double(max(0, min(step, max(1, totalSteps)))) / Double(max(1, totalSteps))
        self.fractionCompleted = switch phase {
        case .textConditioning: 0.10 * rawFrac
        case .denoising: 0.10 + 0.75 * rawFrac
        case .vaeAndAudioMuxing: 0.85 + 0.15 * rawFrac
        }
        self.teacacheSkipped = teacacheSkipped
        self.previewPNG = previewPNG
        self.message = message
    }
}

public struct VideoGenerationResult: Sendable, Codable, Equatable {
    public var videoData: Data
    public var containerFormat: VideoContainerFormat
    public var codec: VideoCodecKind
    public var audioCodec: VideoAudioCodecKind?
    public var width: Int
    public var height: Int
    public var frames: Int
    public var fps: Int
    public var durationSeconds: Double
    public var hasAudioTrack: Bool
    public var seed: UInt64
    public var stepsExecuted: Int
    public var teacacheSkippedSteps: Int
    public var executionMode: VideoExecutionMode
    public var phasesCompleted: [VideoExecutionPhase]
    public var model: String
    public var elapsedMs: Int

    public init(
        videoData: Data,
        request: VideoGenerationRequest,
        seed: UInt64,
        executionMode: VideoExecutionMode,
        phasesCompleted: [VideoExecutionPhase] = VideoExecutionPhase.allCases
    ) {
        self.videoData = videoData
        self.containerFormat = request.containerFormat
        self.codec = request.config.codec
        self.audioCodec = request.config.includeAudio ? request.config.audioCodec : nil
        self.width = request.width
        self.height = request.height
        self.frames = request.frames
        self.fps = request.fps
        self.durationSeconds = request.durationSeconds
        self.hasAudioTrack = request.config.includeAudio
        self.seed = seed
        self.stepsExecuted = request.steps
        self.teacacheSkippedSteps = 0
        self.executionMode = executionMode
        self.phasesCompleted = phasesCompleted
        self.model = request.model
        self.elapsedMs = 0
    }
}

public enum VideoGenerationEvent: Sendable, Codable, Equatable {
    case progress(VideoGenerationProgress)
    case completed(VideoGenerationResult)
}

public struct VideoStageFootprint: Sendable, Codable, Equatable {
    public static let standard24GB = VideoStageFootprint(
        textEncoderGB: 7.5,
        ditGB: 12.5,
        vaeAndVocoderGB: 2.2,
        latentScratchGB: 0.6
    )

    public var textEncoderGB: Double
    public var ditGB: Double
    public var vaeAndVocoderGB: Double
    public var latentScratchGB: Double

    public init(
        textEncoderGB: Double,
        ditGB: Double,
        vaeAndVocoderGB: Double,
        latentScratchGB: Double = 0.6
    ) {
        self.textEncoderGB = max(0.1, textEncoderGB)
        self.ditGB = max(0.1, ditGB)
        self.vaeAndVocoderGB = max(0.1, vaeAndVocoderGB)
        self.latentScratchGB = max(0.05, latentScratchGB)
    }

    public var residentTotalGB: Double {
        textEncoderGB + ditGB + vaeAndVocoderGB + latentScratchGB
    }
}

public struct VideoExecutionPlan: Sendable, Codable, Equatable {
    public var tier: ComposeMemoryTier
    public var mode: VideoExecutionMode
    public var unloadTextEncoderAfterPhase1: Bool
    public var unloadDiTAfterPhase2: Bool
    public var unloadDecodersAfterPhase3: Bool
    public var clearCacheBetweenPhases: Bool
    public var useTiledVAE: Bool
    public var vaeTemporalTileFrames: Int
    public var vaeSpatialTileSize: Int
    public var streamMoEExpertsFromSSD: Bool
    public var peakPhaseGB: Double
    public var residentTotalGB: Double
    public var fitsWithoutSwap: Bool
}

public enum VideoMemoryPlanner {
    public static func plan(
        tier: ComposeMemoryTier,
        footprint: VideoStageFootprint = .standard24GB,
        directive: MemoryDirective = .relaxed,
        forcePhased: Bool = false
    ) -> VideoExecutionPlan {
        let safeBudget = tier.workingSetGB * (directive >= .tight ? 0.72 : 0.88)
        let residentTotal = footprint.residentTotalGB
        let streamExperts = (footprint.ditGB + footprint.latentScratchGB) > safeBudget
        let activeDiT = streamExperts ? min(footprint.ditGB, max(4.2, safeBudget * 0.62)) : footprint.ditGB

        if forcePhased || directive >= .tight || tier == .gb16 || tier == .gb24 || residentTotal > safeBudget {
            if !forcePhased && directive == .relaxed && tier >= .gb48,
               (footprint.textEncoderGB + footprint.vaeAndVocoderGB + activeDiT) <= tier.workingSetGB * 0.94 {
                return makeHybridPlan(tier: tier, footprint: footprint, activeDiT: activeDiT, streamExperts: streamExperts)
            }
            return makeStrictPhasedPlan(tier: tier, footprint: footprint, activeDiT: activeDiT, streamExperts: streamExperts)
        }
        return makeResidentPlan(tier: tier, residentTotal: residentTotal)
    }

    private static func makeStrictPhasedPlan(
        tier: ComposeMemoryTier,
        footprint: VideoStageFootprint,
        activeDiT: Double,
        streamExperts: Bool
    ) -> VideoExecutionPlan {
        let phase1Peak = footprint.textEncoderGB + 0.35
        let phase2Peak = activeDiT + footprint.latentScratchGB
        let phase3Peak = footprint.vaeAndVocoderGB + footprint.latentScratchGB
        let phasedPeak = max(phase1Peak, max(phase2Peak, phase3Peak))
        return VideoExecutionPlan(
            tier: tier,
            mode: .strictPhasedTeardown,
            unloadTextEncoderAfterPhase1: true,
            unloadDiTAfterPhase2: true,
            unloadDecodersAfterPhase3: true,
            clearCacheBetweenPhases: true,
            useTiledVAE: true,
            vaeTemporalTileFrames: tier == .gb16 ? 5 : 9,
            vaeSpatialTileSize: tier == .gb16 ? 256 : 512,
            streamMoEExpertsFromSSD: streamExperts,
            peakPhaseGB: phasedPeak,
            residentTotalGB: footprint.residentTotalGB,
            fitsWithoutSwap: phasedPeak <= tier.workingSetGB
        )
    }

    private static func makeHybridPlan(
        tier: ComposeMemoryTier,
        footprint: VideoStageFootprint,
        activeDiT: Double,
        streamExperts: Bool
    ) -> VideoExecutionPlan {
        let peak = max(footprint.textEncoderGB + activeDiT, activeDiT + footprint.vaeAndVocoderGB) + footprint.latentScratchGB
        return VideoExecutionPlan(
            tier: tier,
            mode: .hybridPhasedDiT,
            unloadTextEncoderAfterPhase1: true,
            unloadDiTAfterPhase2: true,
            unloadDecodersAfterPhase3: false,
            clearCacheBetweenPhases: true,
            useTiledVAE: tier.nominalGB <= 48,
            vaeTemporalTileFrames: 9,
            vaeSpatialTileSize: 512,
            streamMoEExpertsFromSSD: streamExperts,
            peakPhaseGB: peak,
            residentTotalGB: footprint.residentTotalGB,
            fitsWithoutSwap: peak <= tier.workingSetGB
        )
    }

    private static func makeResidentPlan(tier: ComposeMemoryTier, residentTotal: Double) -> VideoExecutionPlan {
        VideoExecutionPlan(
            tier: tier,
            mode: .resident,
            unloadTextEncoderAfterPhase1: false,
            unloadDiTAfterPhase2: false,
            unloadDecodersAfterPhase3: false,
            clearCacheBetweenPhases: false,
            useTiledVAE: false,
            vaeTemporalTileFrames: 25,
            vaeSpatialTileSize: 1_024,
            streamMoEExpertsFromSSD: false,
            peakPhaseGB: residentTotal,
            residentTotalGB: residentTotal,
            fitsWithoutSwap: residentTotal <= tier.workingSetGB
        )
    }
}

extension ComposeMemoryTier: Comparable {
    public static func < (lhs: ComposeMemoryTier, rhs: ComposeMemoryTier) -> Bool {
        lhs.nominalGB < rhs.nominalGB
    }
}

public struct TeaCacheStepSchedule: Sendable, Codable, Equatable {
    public static let defaultCoefficients: [Double] = [
        498.651651,
        -283.781631,
        55.8554382,
        -3.82021401,
        0.264230861,
    ]

    public let totalSteps: Int
    public let threshold: Double
    public private(set) var accumulatedDelta: Double = 0
    public private(set) var executedSteps: [Int] = []
    public private(set) var skippedSteps: [Int] = []

    public init(totalSteps: Int, threshold: Double) {
        self.totalSteps = max(1, totalSteps)
        self.threshold = max(0.0, threshold)
    }

    public var skippingEnabled: Bool {
        threshold > 0.0 && totalSteps > 4
    }

    public mutating func shouldExecuteStep(_ step: Int, relativeL1Delta: Double) -> Bool {
        let isBoundary = step <= 0 || step >= (totalSteps - 1)
        if skippingEnabled && !isBoundary && !executedSteps.isEmpty {
            accumulatedDelta += Self.rescaledDelta(relativeL1Delta)
            if accumulatedDelta < threshold {
                skippedSteps.append(step)
                return false
            }
        }
        accumulatedDelta = 0.0
        executedSteps.append(step)
        return true
    }

    public static func rescaledDelta(_ relL1: Double, coefficients: [Double] = defaultCoefficients) -> Double {
        guard relL1.isFinite, relL1 > 0 else { return 0.0 }
        let clamped = min(relL1, 0.25)
        var poly = 0.0
        for coeff in coefficients {
            poly = poly * clamped + coeff
        }
        let positivePoly = max(0.01, poly)
        let ramp = min(1.0, relL1 / 0.12)
        let overflow = max(0.0, relL1 - 0.25)
        return positivePoly * ramp + overflow
    }

    public static func simulate(totalSteps: Int, threshold: Double, deltas: [Double]) -> TeaCacheStepSchedule {
        var schedule = TeaCacheStepSchedule(totalSteps: totalSteps, threshold: threshold)
        for step in 0 ..< max(1, totalSteps) {
            let delta = step < deltas.count ? deltas[step] : 1.0
            _ = schedule.shouldExecuteStep(step, relativeL1Delta: delta)
        }
        return schedule
    }
}

public protocol VideoSynthesizer: Sendable {
    var modelIdentifier: String { get }
    var descriptor: VideoModelDescriptor { get }
    func generate(_ request: VideoGenerationRequest) async throws -> AsyncThrowingStream<VideoGenerationEvent, any Error>
    func unload() async
}

extension VideoSynthesizer {
    public func unload() async {}
}
