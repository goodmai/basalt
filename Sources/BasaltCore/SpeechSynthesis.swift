import Foundation

public enum SpeechAudioFormat: String, Sendable, Codable, Equatable, CaseIterable {
    case wav
    case pcm
    case pcm24k = "pcm_24k"
    case mp3
    case opus
    case aac
    case flac

    public var contentType: String {
        switch self {
        case .wav: "audio/wav"
        case .pcm, .pcm24k: "audio/pcm"
        case .mp3: "audio/mpeg"
        case .opus: "audio/opus"
        case .aac: "audio/aac"
        case .flac: "audio/flac"
        }
    }

    public var isUncompressed: Bool {
        switch self {
        case .wav, .pcm, .pcm24k: true
        case .mp3, .opus, .aac, .flac: false
        }
    }
}

public enum SpeechStreamFormat: String, Sendable, Codable, Equatable, CaseIterable {
    case audio
    case sse
}

public enum SpeechComputeBackend: String, Sendable, Codable, Equatable, CaseIterable {
    case mlx
    case ane
    case appleSpeech = "apple_speech"
}

public enum SpeechOnGPUBusy: String, Sendable, Codable, Equatable, CaseIterable {
    case keep
    case fallbackANE = "fallback_ane"
    case fallbackAppleSpeech = "fallback_apple_speech"
}

public struct VoiceSpec: Sendable, Codable, Equatable {
    public var id: String
    public var name: String
    public var language: String
    public var model: String
    public var compute: SpeechComputeBackend

    public init(id: String, name: String, language: String, model: String, compute: SpeechComputeBackend) {
        (self.id, self.name, self.language, self.model, self.compute) = (id, name, language, model, compute)
    }
}

public struct SpeechRequest: Sendable, Codable, Equatable {
    public static let defaultSampleRate = 24_000
    public static let maxInputCharacters = 16_384
    public static let speedRange = 0.25 ... 4.0

    public var model: String
    public var input: String
    public var voice: String
    public var responseFormat: SpeechAudioFormat
    public var speed: Double
    public var streamFormat: SpeechStreamFormat
    public var language: String?
    public var instruct: String?
    public var exaggeration: Double?
    public var preferANE: Bool
    public var referenceAudio: String?

    public init(
        model: String,
        input: String,
        voice: String = "alloy",
        responseFormat: SpeechAudioFormat = .wav,
        speed: Double = 1.0,
        streamFormat: SpeechStreamFormat = .audio
    ) throws(BasaltError) {
        self.model = model
        self.input = input
        self.voice = voice
        self.responseFormat = responseFormat
        self.speed = speed
        self.streamFormat = streamFormat
        self.language = nil
        self.instruct = nil
        self.exaggeration = nil
        self.preferANE = false
        self.referenceAudio = nil
        try validate()
    }

    public mutating func configure(
        language: String? = nil,
        instruct: String? = nil,
        exaggeration: Double? = nil,
        preferANE: Bool = false,
        referenceAudio: String? = nil
    ) throws(BasaltError) {
        self.language = language
        self.instruct = instruct
        self.exaggeration = exaggeration
        self.preferANE = preferANE
        self.referenceAudio = referenceAudio
        try validate()
    }

    public func validate() throws(BasaltError) {
        guard !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, model.count <= 256 else {
            throw .invalidRequestStructure(details: "speech: model must be a non-empty name of at most 256 characters")
        }
        guard !input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw .invalidRequestStructure(details: "speech: input text cannot be empty")
        }
        guard input.count <= Self.maxInputCharacters else {
            throw .invalidRequestStructure(details: "speech: input exceeds \(Self.maxInputCharacters) characters")
        }
        guard !voice.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, voice.count <= 128 else {
            throw .invalidRequestStructure(details: "speech: voice must be a non-empty identifier of at most 128 characters")
        }
        guard speed.isFinite, Self.speedRange.contains(speed) else {
            throw .invalidRequestStructure(details: "speech: speed must lie between 0.25 and 4.0, got \(speed)")
        }
        try Self.validateConditioning(language: language, exaggeration: exaggeration, referenceAudio: referenceAudio)
    }

    private static func validateConditioning(
        language: String?,
        exaggeration: Double?,
        referenceAudio: String?
    ) throws(BasaltError) {
        if let exaggeration, !exaggeration.isFinite || !(0 ... 2).contains(exaggeration) {
            throw .invalidRequestStructure(details: "speech: exaggeration must lie between 0.0 and 2.0")
        }
        if let language, !language.isEmpty, language.wholeMatch(of: /[A-Za-z]{2,3}([-_][A-Za-z0-9]{2,8})?/) == nil {
            throw .invalidRequestStructure(details: "speech: language must be a BCP-47 / ISO-639 code such as en, en_us or ru-RU")
        }
        if let referenceAudio {
            try validateReferenceAudio(referenceAudio)
        }
    }

    public static func validateReferenceAudio(_ path: String) throws(BasaltError) {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed.count <= 1_024, !trimmed.contains("\0"), !trimmed.contains("://") else {
            throw .invalidRequestStructure(details: "speech: reference_audio must be a valid local file path")
        }
        let segments = trimmed.split(omittingEmptySubsequences: false) { $0 == "/" || $0 == "\\" }
        if segments.contains("..") {
            throw .invalidRequestStructure(details: "speech: reference_audio must not contain '..' path traversal")
        }
        let ext = URL(fileURLWithPath: trimmed).pathExtension.lowercased()
        guard ["wav", "pcm", "flac"].contains(ext) else {
            throw .invalidRequestStructure(details: "speech: reference_audio must be a local .wav, .pcm or .flac file")
        }
    }

    enum CodingKeys: String, CodingKey {
        case model, input, voice, speed, language, instruct, exaggeration
        case responseFormat = "response_format"
        case streamFormat = "stream_format"
        case preferANE = "prefer_ane"
        case referenceAudio = "reference_audio"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decodeIfPresent(String.self, forKey: .model) ?? ComposeSpec.appleSpeech
        input = try container.decode(String.self, forKey: .input)
        voice = try container.decodeIfPresent(String.self, forKey: .voice) ?? "alloy"
        responseFormat = try container.decodeIfPresent(SpeechAudioFormat.self, forKey: .responseFormat) ?? .wav
        speed = try container.decodeIfPresent(Double.self, forKey: .speed) ?? 1.0
        streamFormat = try container.decodeIfPresent(SpeechStreamFormat.self, forKey: .streamFormat) ?? .audio
        language = try container.decodeIfPresent(String.self, forKey: .language)
        instruct = try container.decodeIfPresent(String.self, forKey: .instruct)
        exaggeration = try container.decodeIfPresent(Double.self, forKey: .exaggeration)
        preferANE = try container.decodeIfPresent(Bool.self, forKey: .preferANE) ?? false
        referenceAudio = try container.decodeIfPresent(String.self, forKey: .referenceAudio)
        try validate()
    }
}

public struct AudioChunk: Sendable, Codable, Equatable {
    public var samples: [Float]
    public var sampleRate: Int
    public var isFinal: Bool

    public init(samples: [Float], sampleRate: Int = SpeechRequest.defaultSampleRate, isFinal: Bool = false) {
        (self.samples, self.sampleRate, self.isFinal) = (samples, sampleRate, isFinal)
    }

    public func resampled(to targetRate: Int) -> AudioChunk {
        guard !samples.isEmpty, sampleRate > 0, targetRate > 0, sampleRate != targetRate else {
            return AudioChunk(samples: samples, sampleRate: targetRate > 0 ? targetRate : sampleRate, isFinal: isFinal)
        }
        let outCount = max(1, Int((Double(samples.count) * Double(targetRate) / Double(sampleRate)).rounded()))
        if samples.count == 1 {
            return AudioChunk(samples: Array(repeating: samples[0], count: outCount), sampleRate: targetRate, isFinal: isFinal)
        }
        let ratio = Double(samples.count - 1) / Double(max(1, outCount - 1))
        let out = (0 ..< outCount).map { idx in
            let pos = Double(idx) * ratio
            let lo = min(samples.count - 1, Int(pos))
            let hi = min(samples.count - 1, lo + 1)
            let frac = Float(pos - Double(lo))
            return samples[lo] * (1 - frac) + samples[hi] * frac
        }
        return AudioChunk(samples: out, sampleRate: targetRate, isFinal: isFinal)
    }

    public func encodePCM16LE() -> Data {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(samples.count * 2)
        for sample in samples {
            let clean = sample.isFinite ? max(-1, min(1, sample)) : 0
            let quantized = Int16((clean * Float(Int16.max)).rounded())
            let bits = UInt16(bitPattern: quantized)
            bytes.append(UInt8(truncatingIfNeeded: bits))
            bytes.append(UInt8(truncatingIfNeeded: bits >> 8))
        }
        return Data(bytes)
    }

    public static func decodePCM16LE(
        _ data: Data,
        sampleRate: Int = SpeechRequest.defaultSampleRate,
        isFinal: Bool = false
    ) throws(BasaltError) -> AudioChunk {
        guard sampleRate > 0 else {
            throw .invalidRequestStructure(details: "PCM sample rate must be positive")
        }
        guard data.count.isMultiple(of: 2) else {
            throw .invalidRequestStructure(details: "PCM16LE byte length must be even, got \(data.count)")
        }
        var decoded: [Float] = []
        decoded.reserveCapacity(data.count / 2)
        var index = data.startIndex
        while index < data.endIndex {
            let next = data.index(after: index)
            let word = UInt16(data[index]) | (UInt16(data[next]) << 8)
            decoded.append(Float(Int16(bitPattern: word)) / Float(Int16.max))
            index = data.index(after: next)
        }
        return AudioChunk(samples: decoded, sampleRate: sampleRate, isFinal: isFinal)
    }
}

public enum WAVEncoder {
    public struct DecodedWAV: Sendable, Equatable {
        public let samples: [Float]
        public let sampleRate: Int
        public let channels: Int

        public init(samples: [Float], sampleRate: Int, channels: Int) {
            (self.samples, self.sampleRate, self.channels) = (samples, sampleRate, channels)
        }
    }

    public static func encode(samples: [Float], sampleRate: Int = SpeechRequest.defaultSampleRate) -> Data {
        let rate = UInt32(max(1, sampleRate))
        let payload = AudioChunk(samples: samples, sampleRate: Int(rate)).encodePCM16LE()
        let dataBytes = UInt32(payload.count)
        var header: [UInt8] = []
        header.reserveCapacity(44)
        func put16(_ value: UInt16) {
            header.append(UInt8(truncatingIfNeeded: value))
            header.append(UInt8(truncatingIfNeeded: value >> 8))
        }
        func put32(_ value: UInt32) {
            put16(UInt16(truncatingIfNeeded: value))
            put16(UInt16(truncatingIfNeeded: value >> 16))
        }
        header.append(contentsOf: "RIFF".utf8)
        put32(36 &+ dataBytes)
        header.append(contentsOf: "WAVEfmt ".utf8)
        put32(16)
        put16(1)
        put16(1)
        put32(rate)
        put32(rate &* 2)
        put16(2)
        put16(16)
        header.append(contentsOf: "data".utf8)
        put32(dataBytes)
        var wav = Data(header)
        wav.append(payload)
        return wav
    }

    public static func decode(_ data: Data) throws(BasaltError) -> DecodedWAV {
        let bytes = Array(data)
        guard bytes.count >= 44, String(decoding: bytes[0 ..< 4], as: UTF8.self) == "RIFF",
              String(decoding: bytes[8 ..< 12], as: UTF8.self) == "WAVE" else {
            throw .invalidRequestStructure(details: "WAV header must start with RIFF....WAVE")
        }
        func u16(_ at: Int) -> UInt16 { UInt16(bytes[at]) | (UInt16(bytes[at + 1]) << 8) }
        func u32(_ at: Int) -> UInt32 { UInt32(u16(at)) | (UInt32(u16(at + 2)) << 16) }
        var cursor = 12
        var format: (channels: Int, rate: Int, bits: Int)?
        while cursor + 8 <= bytes.count {
            let tag = String(decoding: bytes[cursor ..< cursor + 4], as: UTF8.self)
            let size = Int(u32(cursor + 4))
            let start = cursor + 8
            guard size >= 0, start + size <= bytes.count else {
                throw .invalidRequestStructure(details: "WAV chunk '\(tag)' exceeds buffer bounds")
            }
            if tag == "fmt " {
                guard size >= 16, u16(start) == 1 else {
                    throw .invalidRequestStructure(details: "WAV fmt chunk must be uncompressed PCM (format 1)")
                }
                format = (Int(u16(start + 2)), Int(u32(start + 4)), Int(u16(start + 14)))
            } else if tag == "data", let fmt = format {
                return try decodePCMData(bytes[start ..< start + size], channels: fmt.channels, rate: fmt.rate, bits: fmt.bits)
            }
            cursor = start + size + (size.isMultiple(of: 2) ? 0 : 1)
        }
        throw .invalidRequestStructure(details: "WAV buffer is missing fmt or data chunk")
    }

    private static func decodePCMData(
        _ slice: ArraySlice<UInt8>,
        channels: Int,
        rate: Int,
        bits: Int
    ) throws(BasaltError) -> DecodedWAV {
        guard channels >= 1, rate > 0, bits == 16, slice.count.isMultiple(of: channels * 2) else {
            throw .invalidRequestStructure(details: "WAV PCM data must be 16-bit with valid channel alignment")
        }
        let frameStride = channels * 2
        let frames = slice.count / frameStride
        var samples: [Float] = []
        samples.reserveCapacity(frames)
        var offset = slice.startIndex
        for _ in 0 ..< frames {
            var sum: Float = 0
            for ch in 0 ..< channels {
                let idx = offset + ch * 2
                let word = UInt16(slice[idx]) | (UInt16(slice[idx + 1]) << 8)
                sum += Float(Int16(bitPattern: word)) / Float(Int16.max)
            }
            samples.append(sum / Float(channels))
            offset += frameStride
        }
        return DecodedWAV(samples: samples, sampleRate: rate, channels: channels)
    }
}

public struct SpeechReceipt: Sendable, Codable, Equatable {
    public var model: String
    public var compute: SpeechComputeBackend
    public var fallbackFrom: String?
    public var reason: String?
    public var ttfaMs: Double
    public var audioMs: Double
    public var rtf: Double

    public init(
        model: String,
        compute: SpeechComputeBackend,
        ttfaMs: Double,
        audioMs: Double,
        rtf: Double
    ) {
        self.model = model
        self.compute = compute
        self.fallbackFrom = nil
        self.reason = nil
        self.ttfaMs = ttfaMs
        self.audioMs = audioMs
        self.rtf = rtf
    }

    public func withFallback(from: String?, reason: String?) -> SpeechReceipt {
        var copy = self
        copy.fallbackFrom = from
        copy.reason = reason
        return copy
    }

    enum CodingKeys: String, CodingKey {
        case model, compute, reason, rtf
        case fallbackFrom = "fallback_from"
        case ttfaMs = "ttfa_ms"
        case audioMs = "audio_ms"
    }
}

public struct SynthesizedSpeech: Sendable, Codable, Equatable {
    public var audio: Data
    public var contentType: String
    public var sampleRate: Int
    public var durationSeconds: Double
    public var receipt: SpeechReceipt

    public init(audio: Data, contentType: String, sampleRate: Int, durationSeconds: Double, receipt: SpeechReceipt) {
        (self.audio, self.contentType, self.sampleRate, self.durationSeconds, self.receipt) =
            (audio, contentType, sampleRate, durationSeconds, receipt)
    }

    enum CodingKeys: String, CodingKey {
        case audio, receipt
        case contentType = "content_type"
        case sampleRate = "sample_rate"
        case durationSeconds = "duration_seconds"
    }
}

public protocol SpeechSynthesizer: Sendable {
    var voices: [VoiceSpec] { get async }
    func synthesize(_ request: SpeechRequest) async throws(BasaltError) -> AsyncThrowingStream<AudioChunk, any Error>
    func resolveRoute(_ request: SpeechRequest) async -> SpeechRouteDecision
    func unload() async
}

extension SpeechSynthesizer {
    public func resolveRoute(_ request: SpeechRequest) async -> SpeechRouteDecision {
        let available = await voices
        let matched = available.first { $0.id == request.voice } ?? available.first
        let model = matched?.model ?? request.model
        let compute = matched?.compute ?? (model == ComposeSpec.appleSpeech ? .appleSpeech : .mlx)
        return SpeechRouteDecision(selected: SpeechCandidate(id: model, model: model, compute: compute))
    }

    public func unload() async {}
}

public enum SpeechFallbackReason: String, Sendable, Codable, Equatable, CaseIterable {
    case gpuBusy = "gpu_busy"
    case memoryPressure = "memory_pressure"
    case ttfaSLA = "ttfa_sla"
    case unavailableModel = "unavailable_model"
    case preferANE = "prefer_ane"
}

public struct SpeechCandidate: Sendable, Codable, Equatable {
    public static let appleSpeech = SpeechCandidate(
        id: ComposeSpec.appleSpeech,
        model: ComposeSpec.appleSpeech,
        compute: .appleSpeech,
        isAvailable: true,
        estimatedTTFAMs: 20
    )

    public var id: String
    public var model: String
    public var compute: SpeechComputeBackend
    public var isAvailable: Bool
    public var estimatedTTFAMs: Int?

    public init(
        id: String,
        model: String,
        compute: SpeechComputeBackend,
        isAvailable: Bool = true,
        estimatedTTFAMs: Int? = nil
    ) {
        (self.id, self.model, self.compute, self.isAvailable, self.estimatedTTFAMs) =
            (id, model, compute, isAvailable, estimatedTTFAMs)
    }
}

public struct SpeechRouteInput: Sendable, Equatable {
    public var chain: [SpeechCandidate]
    public var gpuBusy: Bool
    public var onGPUBusy: SpeechOnGPUBusy
    public var memoryPressure: Bool
    public var preferANE: Bool
    public var maxTTFAMs: Int?

    public init(
        chain: [SpeechCandidate],
        gpuBusy: Bool = false,
        onGPUBusy: SpeechOnGPUBusy = .fallbackANE,
        memoryPressure: Bool = false,
        preferANE: Bool = false,
        maxTTFAMs: Int? = nil
    ) {
        (self.chain, self.gpuBusy, self.onGPUBusy, self.memoryPressure, self.preferANE, self.maxTTFAMs) =
            (chain, gpuBusy, onGPUBusy, memoryPressure, preferANE, maxTTFAMs)
    }
}

public struct SpeechRouteDecision: Sendable, Equatable {
    public var selected: SpeechCandidate
    public var fallbackFrom: String?
    public var reason: SpeechFallbackReason?

    public init(selected: SpeechCandidate, fallbackFrom: String? = nil, reason: SpeechFallbackReason? = nil) {
        (self.selected, self.fallbackFrom, self.reason) = (selected, fallbackFrom, reason)
    }

    public func receipt(ttfaMs: Double, audioMs: Double, rtf: Double) -> SpeechReceipt {
        let resolvedReason = fallbackFrom != nil ? reason?.rawValue : nil
        return SpeechReceipt(model: selected.model, compute: selected.compute, ttfaMs: ttfaMs, audioMs: audioMs, rtf: rtf)
            .withFallback(from: resolvedReason != nil ? fallbackFrom : nil, reason: resolvedReason)
    }
}

public enum SpeechFallbackRouter {
    public static func route(_ input: SpeechRouteInput) -> SpeechRouteDecision {
        guard let primary = input.chain.first else {
            return SpeechRouteDecision(selected: .appleSpeech)
        }
        let hasANE = input.chain.contains { $0.isAvailable && $0.compute == .ane }
        var firstReason: SpeechFallbackReason?
        for candidate in input.chain {
            if let skip = skipReason(candidate, input: input, primary: primary, hasANE: hasANE) {
                if firstReason == nil { firstReason = skip }
                continue
            }
            if candidate.id == primary.id {
                return SpeechRouteDecision(selected: candidate)
            }
            return SpeechRouteDecision(selected: candidate, fallbackFrom: primary.model, reason: firstReason ?? .unavailableModel)
        }
        return SpeechRouteDecision(
            selected: .appleSpeech,
            fallbackFrom: primary.id == ComposeSpec.appleSpeech ? nil : primary.model,
            reason: primary.id == ComposeSpec.appleSpeech ? nil : (firstReason ?? .unavailableModel)
        )
    }

    private static func skipReason(
        _ candidate: SpeechCandidate,
        input: SpeechRouteInput,
        primary: SpeechCandidate,
        hasANE: Bool
    ) -> SpeechFallbackReason? {
        guard candidate.isAvailable else { return .unavailableModel }
        switch candidate.compute {
        case .mlx:
            if input.memoryPressure { return .memoryPressure }
            if input.preferANE, hasANE { return .preferANE }
            if input.gpuBusy, input.onGPUBusy != .keep { return .gpuBusy }
        case .ane:
            if input.gpuBusy, input.onGPUBusy == .fallbackAppleSpeech, primary.compute == .mlx { return .gpuBusy }
        case .appleSpeech:
            return nil
        }
        if let maxTTFAMs = input.maxTTFAMs, let estimate = candidate.estimatedTTFAMs, estimate > maxTTFAMs {
            return .ttfaSLA
        }
        return nil
    }
}
