import Foundation

public struct AudioClip: Sendable, Equatable {
    public static let sampleRate = 16_000.0
    public static let maxBytes = 25 * 1024 * 1024
    public static let maxSeconds = 1_800.0
    public static let fileExtensions: Set<String> = ["wav", "mp3", "mpga", "mpeg", "m4a", "mp4", "aac", "flac", "aif", "aiff", "caf"]

    public let samples: [Float]

    public init(samples: [Float]) throws(BasaltError) {
        guard !samples.isEmpty else { throw .invalidRequestStructure(details: "the audio holds no samples") }
        let seconds = Double(samples.count) / Self.sampleRate
        guard seconds <= Self.maxSeconds else { throw Self.tooLong(seconds) }
        self.samples = samples
    }

    public var seconds: Double { Double(samples.count) / Self.sampleRate }

    static func tooLong(_ seconds: Double) -> BasaltError {
        .invalidRequestStructure(details: "the audio lasts \(Int(seconds)) s, over the limit of \(Int(maxSeconds)) s per request")
    }

    public static func checkedExtension(_ format: String) throws(BasaltError) -> String {
        let format = format.lowercased()
        guard fileExtensions.contains(format) else {
            throw .invalidRequestStructure(details: "audio format '\(format.prefix(16))' is not supported; send one of \(fileExtensions.sorted().joined(separator: ", "))")
        }
        return format
    }
}

public struct TranscriptionRequest: Sendable, Equatable {
    public let audio: AudioClip
    public let language: String?
    public let temperature: Double

    public init(audio: AudioClip, language: String? = nil, temperature: Double? = nil) throws(BasaltError) {
        if let language, language.wholeMatch(of: /[A-Za-z]{2,3}(-[A-Za-z0-9]{2,4})?/) == nil {
            throw .invalidRequestStructure(details: "language must be an ISO-639-1 code such as en or en-US, got '\(language.prefix(16))'")
        }
        if let temperature, !(0 ... 1).contains(temperature) {
            throw .invalidRequestStructure(details: "temperature must be between 0 and 1, got \(temperature)")
        }
        self.audio = audio
        self.language = language?.lowercased()
        self.temperature = temperature ?? 0
    }
}

public struct Transcription: Sendable, Equatable {
    public struct Segment: Sendable, Equatable {
        public let start: Double
        public let end: Double
        public let text: String

        public init(start: Double, end: Double, text: String) {
            self.start = start
            self.end = end
            self.text = text
        }
    }

    public let text: String
    public let language: String?
    public let segments: [Segment]

    public init(text: String, language: String? = nil, segments: [Segment] = []) {
        self.text = text
        self.language = language
        self.segments = segments
    }
}

public protocol AudioTranscriber: Sendable {
    func transcribe(_ request: TranscriptionRequest) async throws(BasaltError) -> Transcription
}

extension BasaltError {
    public static func noAudioEncoder(_ model: String) -> BasaltError {
        .invalidRequestStructure(details: "\(model.isEmpty ? "the loaded model" : model) has no audio encoder, so it cannot transcribe; serve a speech model or start serve with --experimental-speech-analyzer")
    }
}
