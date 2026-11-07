import Foundation
import AVFoundation
import Speech
import BasaltCore

public struct SpeechAnalyzerTranscriber: AudioTranscriber {
    public init() {}

    public func transcribe(_ request: TranscriptionRequest) async throws(BasaltError) -> Transcription {
        let locale = try await Self.locale(for: request.language)
        do {
            let segments = try await Self.segments(of: request.audio, in: locale)
            let text = segments.map(\.text).joined().trimmingCharacters(in: .whitespacesAndNewlines)
            return Transcription(text: text, language: locale.language.languageCode?.identifier, segments: segments)
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .inferenceError("SpeechAnalyzer failed: \(error.localizedDescription)")
        }
    }

    static func locale(for language: String?) async throws(BasaltError) -> Locale {
        let wanted = language.map { Locale(identifier: $0) } ?? Locale.current
        guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: wanted) else {
            throw .invalidRequestStructure(details: "SpeechAnalyzer does not transcribe \(wanted.identifier)")
        }
        return locale
    }

    static func segments(of audio: AudioClip, in locale: Locale) async throws -> [Transcription.Segment] {
        let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
        if let installation = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            try await installation.downloadAndInstall()
        }
        guard let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber]) else {
            throw BasaltError.inferenceError("SpeechAnalyzer offers no audio format for \(locale.identifier)")
        }
        let (input, feed) = AsyncStream<AnalyzerInput>.makeStream()
        feed.yield(AnalyzerInput(buffer: try audio.buffer(in: format)))
        feed.finish()
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        async let results = transcriber.results.reduce(into: [Transcription.Segment]()) { segments, result in
            segments.append(Transcription.Segment(
                start: result.range.start.seconds, end: result.range.end.seconds, text: String(result.text.characters)))
        }
        if let end = try await analyzer.analyzeSequence(input) {
            try await analyzer.finalizeAndFinish(through: end)
        } else {
            await analyzer.cancelAndFinishNow()
        }
        return try await results
    }
}
