import Foundation

public struct MusicModelDescriptor: Sendable, Codable, Equatable {
    public var id: String
    public var family: String
    public var licenseName: String
    public var billionParams: Double
    public var defaultSampleRate: Int
    public var defaultSteps: Int
    public var maxDurationSeconds: Double
    public var supportsLyrics: Bool
    public var minimumRecommendedBits: Int

    public init(
        id: String,
        family: String = "ace_step_1_5",
        licenseName: String = "Apache-2.0",
        billionParams: Double = 3.5,
        defaultSampleRate: Int = 24_000,
        defaultSteps: Int = 8
    ) {
        self.id = id
        self.family = family
        self.licenseName = licenseName
        self.billionParams = billionParams
        self.defaultSampleRate = defaultSampleRate
        self.defaultSteps = defaultSteps
        self.maxDurationSeconds = 240.0
        self.supportsLyrics = true
        self.minimumRecommendedBits = 4
    }

    public static func aceStep15(id: String = "ACE-Step/ACE-Step-1.5-3.5B") -> MusicModelDescriptor {
        MusicModelDescriptor(
            id: id,
            family: "ace_step_1_5",
            licenseName: "Apache-2.0",
            billionParams: 3.5,
            defaultSampleRate: 24_000,
            defaultSteps: 8
        )
    }

    public static func detect(identifier: String, configJSON: [String: Any]? = nil) -> MusicModelDescriptor {
        let modelType = (configJSON?["model_type"] as? String)?.lowercased() ?? ""
        let lower = "\(identifier.lowercased()) \(modelType)"
        var desc = aceStep15(id: identifier)
        if lower.contains("44k") || (configJSON?["sample_rate"] as? Int) == 44_100 {
            desc.defaultSampleRate = 44_100
        }
        return desc
    }
}

public struct MusicGenerationOptions: Sendable, Codable, Equatable {
    public var lyrics: String?
    public var tags: [String]
    public var bpm: Double
    public var steps: Int
    public var guidanceScale: Double
    public var seed: UInt64?

    public init(
        lyrics: String? = nil,
        tags: [String] = [],
        bpm: Double = 120.0,
        steps: Int = 8,
        guidanceScale: Double = 4.0,
        seed: UInt64? = nil
    ) {
        self.lyrics = lyrics
        self.tags = tags
        self.bpm = bpm
        self.steps = steps
        self.guidanceScale = guidanceScale
        self.seed = seed
    }
}

public struct MusicGenerationRequest: Sendable, Codable, Equatable {
    public static let defaultSampleRate = 24_000
    public static let maxPromptCharacters = 8_192
    public static let maxLyricsCharacters = 16_384
    public static let durationRange = 0.25 ... 240.0
    public static let bpmRange = 40.0 ... 240.0

    public var model: String
    public var prompt: String
    public var durationSeconds: Double
    public var responseFormat: SpeechAudioFormat
    public var streamFormat: SpeechStreamFormat
    public var options: MusicGenerationOptions

    public init(
        model: String = "ACE-Step/ACE-Step-1.5-3.5B",
        prompt: String,
        durationSeconds: Double = 4.0,
        responseFormat: SpeechAudioFormat = .wav,
        streamFormat: SpeechStreamFormat = .audio,
        options: MusicGenerationOptions = MusicGenerationOptions()
    ) throws(BasaltError) {
        self.model = model
        self.prompt = prompt
        self.durationSeconds = durationSeconds
        self.responseFormat = responseFormat
        self.streamFormat = streamFormat
        self.options = options
        try validate()
    }

    public func validate() throws(BasaltError) {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty || !options.tags.isEmpty else {
            throw .invalidRequestStructure(details: "music: prompt or tags must not be empty")
        }
        guard prompt.count <= Self.maxPromptCharacters else {
            throw .invalidRequestStructure(details: "music: prompt exceeds \(Self.maxPromptCharacters) characters")
        }
        if let lyrics = options.lyrics, lyrics.count > Self.maxLyricsCharacters {
            throw .invalidRequestStructure(details: "music: lyrics exceed \(Self.maxLyricsCharacters) characters")
        }
        guard durationSeconds.isFinite, Self.durationRange.contains(durationSeconds) else {
            throw .invalidRequestStructure(details: "music: duration_seconds must lie within 0.25...240.0")
        }
        guard options.bpm.isFinite, Self.bpmRange.contains(options.bpm) else {
            throw .invalidRequestStructure(details: "music: bpm must lie within 40...240")
        }
        guard (1 ... 64).contains(options.steps) else {
            throw .invalidRequestStructure(details: "music: steps must lie within 1...64")
        }
        guard options.guidanceScale.isFinite, options.guidanceScale >= 0 else {
            throw .invalidRequestStructure(details: "music: guidance_scale must be non-negative")
        }
        guard responseFormat.isUncompressed else {
            throw .invalidRequestStructure(details: "music: response_format must be wav, pcm or pcm_24k")
        }
    }
}

public struct MusicReceipt: Sendable, Codable, Equatable {
    public var model: String
    public var sampleRate: Int
    public var durationSeconds: Double
    public var stepsExecuted: Int
    public var elapsedMs: Double
    public var seed: UInt64

    public init(
        model: String,
        sampleRate: Int,
        durationSeconds: Double,
        stepsExecuted: Int,
        elapsedMs: Double,
        seed: UInt64
    ) {
        self.model = model
        self.sampleRate = sampleRate
        self.durationSeconds = durationSeconds
        self.stepsExecuted = stepsExecuted
        self.elapsedMs = elapsedMs
        self.seed = seed
    }

    enum CodingKeys: String, CodingKey {
        case model, seed
        case sampleRate = "sample_rate"
        case durationSeconds = "duration_seconds"
        case stepsExecuted = "steps_executed"
        case elapsedMs = "elapsed_ms"
    }
}

public protocol MusicSynthesizer: Sendable {
    var modelIdentifier: String { get }
    var descriptor: MusicModelDescriptor { get }
    func synthesize(_ request: MusicGenerationRequest) async throws -> AsyncThrowingStream<AudioChunk, any Error>
    func unload() async
}

extension MusicSynthesizer {
    public func unload() async {}
}
