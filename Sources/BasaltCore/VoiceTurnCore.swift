import Foundation

public struct VoiceTurnConfig: Sendable, Codable, Equatable {
    public static let maxSpeakers = 4

    public var vadSpeechThreshold: Float
    public var semanticTurnThreshold: Float
    public var minSpeechSeconds: Double
    public var minSilenceSeconds: Double
    public var maxThinkingPauseSeconds: Double
    public var diarizationThreshold: Float

    public init(
        vadSpeechThreshold: Float = 0.5,
        semanticTurnThreshold: Float = 0.68,
        minSpeechSeconds: Double = 0.16,
        minSilenceSeconds: Double = 0.02,
        maxThinkingPauseSeconds: Double = 2.4,
        diarizationThreshold: Float = 0.45
    ) {
        self.vadSpeechThreshold = Self.clamp01(vadSpeechThreshold, fallback: 0.5)
        self.semanticTurnThreshold = Self.clamp01(semanticTurnThreshold, fallback: 0.68)
        self.minSpeechSeconds = max(0, minSpeechSeconds.isFinite ? minSpeechSeconds : 0.16)
        self.minSilenceSeconds = max(0, minSilenceSeconds.isFinite ? minSilenceSeconds : 0.02)
        self.maxThinkingPauseSeconds = max(self.minSilenceSeconds, maxThinkingPauseSeconds.isFinite ? maxThinkingPauseSeconds : 2.4)
        self.diarizationThreshold = Self.clamp01(diarizationThreshold, fallback: 0.45)
    }

    static func clamp01(_ value: Float, fallback: Float) -> Float {
        guard value.isFinite else { return fallback }
        return min(1, max(0, value))
    }
}

public struct SpeakerSegment: Sendable, Codable, Equatable {
    public let speakerId: Int
    public let startSeconds: Double
    public let endSeconds: Double
    public let confidence: Float

    public var durationSeconds: Double {
        max(0, endSeconds - startSeconds)
    }

    public init(speakerId: Int, startSeconds: Double, endSeconds: Double, confidence: Float) {
        self.speakerId = min(VoiceTurnConfig.maxSpeakers - 1, max(0, speakerId))
        let cleanStart = max(0, startSeconds.isFinite ? startSeconds : 0)
        self.startSeconds = cleanStart
        self.endSeconds = max(cleanStart, endSeconds.isFinite ? endSeconds : cleanStart)
        self.confidence = VoiceTurnConfig.clamp01(confidence, fallback: 0)
    }
}

public enum TurnDecision: Sendable, Codable, Equatable {
    case listening
    case holdThinkingPause
    case commitTurn(speakerId: Int)
}

public enum SileroVADScorer {
    public static func score(_ samples: [Float], sampleRate: Int = 16_000) -> Float {
        guard !samples.isEmpty, sampleRate > 0 else { return 0 }
        var sumSq = 0.0
        var crossings = 0
        var validCount = 0
        var previous: Float = 0
        for sample in samples where sample.isFinite {
            let clamped = min(1, max(-1, sample))
            sumSq += Double(clamped) * Double(clamped)
            if validCount > 0, (clamped >= 0) != (previous >= 0) {
                crossings += 1
            }
            previous = clamped
            validCount += 1
        }
        guard validCount > 0 else { return 0 }
        let rms = (sumSq / Double(validCount)).squareRoot()
        if rms < 1e-4 { return 0.01 }
        let zcrHz = Double(crossings) * Double(sampleRate) / Double(max(1, 2 * validCount))
        let bandPenalty = (zcrHz > 4_200 || zcrHz < 35) ? 0.25 : 0.0
        let db = 20.0 * log10(max(1e-6, rms))
        let logit = (db + 34.0) * 0.28 - bandPenalty * 3.0
        return Float(1.0 / (1.0 + exp(-logit)))
    }
}

public enum SmartTurnScorer {
    private static let hesitationTokens: Set<String> = [
        "um", "uh", "erm", "er", "hmm", "uhm", "and", "or", "but", "because",
        "so", "if", "when", "while", "although", "though", "that", "which", "like",
        "и", "но", "или", "а", "потому", "что", "если", "когда", "эм", "э", "ну", "как",
    ]

    public static func score(transcript: String, trailingEnergyDrop: Float = 0) -> Float {
        let trimmed = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return 0.15 }
        let dropBonus = Double(VoiceTurnConfig.clamp01(trailingEnergyDrop, fallback: 0)) * 0.22
        if trimmed.hasSuffix("...") || trimmed.hasSuffix("…") || trimmed.hasSuffix(",")
            || trimmed.hasSuffix("—") || trimmed.hasSuffix("-") || trimmed.hasSuffix(":") || trimmed.hasSuffix(";") {
            return Float(min(0.35, 0.14 + dropBonus * 0.4))
        }
        let hasTerminalPunct = trimmed.hasSuffix(".") || trimmed.hasSuffix("?") || trimmed.hasSuffix("!")
        let words = trimmed.lowercased()
            .split { !$0.isLetter && !$0.isNumber && $0 != "'" }
            .map(String.init)
        if let lastWord = words.last, hesitationTokens.contains(lastWord), !hasTerminalPunct {
            return Float(max(0.08, 0.18 - dropBonus * 0.1))
        }
        if hasTerminalPunct {
            return Float(min(0.98, 0.84 + dropBonus * 0.5))
        }
        let lengthScore = min(0.28, Double(words.count) * 0.045)
        return Float(min(0.92, 0.36 + lengthScore + dropBonus))
    }
}

public struct VoiceTurnCore: Sendable, Equatable {
    public var config: VoiceTurnConfig
    public private(set) var elapsedSeconds: Double = 0
    public private(set) var speechSecondsInTurn: Double = 0
    public private(set) var trailingSilenceSeconds: Double = 0
    public private(set) var latestSemanticEOT: Float = 0
    public private(set) var speakerActivitySeconds: [Double]
    public private(set) var segments: [SpeakerSegment] = []
    private var previousFrameRMS: Float = 0

    public init(config: VoiceTurnConfig = VoiceTurnConfig()) {
        self.config = config
        self.speakerActivitySeconds = Array(repeating: 0, count: VoiceTurnConfig.maxSpeakers)
    }

    public var dominantSpeakerId: Int {
        var bestId = segments.last?.speakerId ?? 0
        var bestScore = -1.0
        for idx in 0 ..< VoiceTurnConfig.maxSpeakers where speakerActivitySeconds[idx] > bestScore && speakerActivitySeconds[idx] > 0 {
            bestScore = speakerActivitySeconds[idx]
            bestId = idx
        }
        return bestId
    }

    public mutating func resetTurn() {
        speechSecondsInTurn = 0
        trailingSilenceSeconds = 0
        latestSemanticEOT = 0
        speakerActivitySeconds = Array(repeating: 0, count: VoiceTurnConfig.maxSpeakers)
    }

    public mutating func resetSession() {
        resetTurn()
        elapsedSeconds = 0
        segments.removeAll()
        previousFrameRMS = 0
    }

    public mutating func step(
        vadProbability: Float,
        semanticEOTProbability: Float,
        speakerPosteriors: [Float] = [],
        frameDurationSeconds: Double
    ) -> TurnDecision {
        let dt = max(0, frameDurationSeconds.isFinite ? frameDurationSeconds : 0)
        let vad = VoiceTurnConfig.clamp01(vadProbability, fallback: 0)
        let eot = VoiceTurnConfig.clamp01(semanticEOTProbability, fallback: 0)
        let frameStart = elapsedSeconds
        let frameEnd = frameStart + dt
        elapsedSeconds = frameEnd
        latestSemanticEOT = eot

        let isSpeech = vad >= config.vadSpeechThreshold
        recordSortformerFrame(posteriors: speakerPosteriors, start: frameStart, end: frameEnd, dt: dt, isSpeech: isSpeech)

        if isSpeech {
            speechSecondsInTurn += dt
            trailingSilenceSeconds = 0
            return .listening
        }
        trailingSilenceSeconds += dt
        guard speechSecondsInTurn >= config.minSpeechSeconds else {
            return .listening
        }
        let semanticReady = eot >= config.semanticTurnThreshold && trailingSilenceSeconds >= config.minSilenceSeconds
        let timedOut = trailingSilenceSeconds >= config.maxThinkingPauseSeconds
        if semanticReady || timedOut {
            let committedSpeaker = dominantSpeakerId
            resetTurn()
            return .commitTurn(speakerId: committedSpeaker)
        }
        return .holdThinkingPause
    }

    public mutating func stepPCMFrame(
        _ samples: [Float],
        sampleRate: Int = 16_000,
        transcriptHint: String = "",
        speakerPosteriors: [Float] = []
    ) -> TurnDecision {
        let rate = max(1, sampleRate)
        let dt = Double(samples.count) / Double(rate)
        let vad = SileroVADScorer.score(samples, sampleRate: rate)
        let currentRMS = Self.frameRMS(samples)
        let drop: Float = previousFrameRMS > 1e-4 ? max(0, (previousFrameRMS - currentRMS) / previousFrameRMS) : 0
        if vad >= config.vadSpeechThreshold {
            previousFrameRMS = currentRMS
        }
        let eot = SmartTurnScorer.score(transcript: transcriptHint, trailingEnergyDrop: drop)
        return step(
            vadProbability: vad,
            semanticEOTProbability: eot,
            speakerPosteriors: speakerPosteriors,
            frameDurationSeconds: dt
        )
    }

    private mutating func recordSortformerFrame(
        posteriors: [Float],
        start: Double,
        end: Double,
        dt: Double,
        isSpeech: Bool
    ) {
        guard dt > 0, !posteriors.isEmpty else { return }
        var probs = InlineArray<4, Float>(repeating: 0)
        let count = min(VoiceTurnConfig.maxSpeakers, posteriors.count)
        for idx in 0 ..< count {
            probs[idx] = VoiceTurnConfig.clamp01(posteriors[idx], fallback: 0)
        }
        var bestSpeaker = 0
        var bestConfidence = probs[0]
        for idx in 1 ..< VoiceTurnConfig.maxSpeakers where probs[idx] > bestConfidence {
            bestConfidence = probs[idx]
            bestSpeaker = idx
        }
        guard bestConfidence >= config.diarizationThreshold else { return }
        let vadWeight = isSpeech ? 1.0 : 0.5
        speakerActivitySeconds[bestSpeaker] += Double(bestConfidence) * dt * vadWeight
        appendOrMergeSegment(speakerId: bestSpeaker, start: start, end: end, confidence: bestConfidence)
    }

    private mutating func appendOrMergeSegment(speakerId: Int, start: Double, end: Double, confidence: Float) {
        if let last = segments.last, last.speakerId == speakerId, abs(last.endSeconds - start) <= 1e-5 {
            let prevDur = last.durationSeconds
            let newDur = max(0, end - start)
            let totalDur = prevDur + newDur
            let blended: Float = totalDur > 0
                ? Float((Double(last.confidence) * prevDur + Double(confidence) * newDur) / totalDur)
                : confidence
            segments[segments.count - 1] = SpeakerSegment(
                speakerId: speakerId,
                startSeconds: last.startSeconds,
                endSeconds: end,
                confidence: blended
            )
        } else {
            segments.append(SpeakerSegment(speakerId: speakerId, startSeconds: start, endSeconds: end, confidence: confidence))
        }
    }

    private static func frameRMS(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        var sum = 0.0
        var count = 0
        for s in samples where s.isFinite {
            sum += Double(s) * Double(s)
            count += 1
        }
        return count > 0 ? Float((sum / Double(count)).squareRoot()) : 0
    }
}
