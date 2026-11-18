import Foundation

extension ModelOrchestratorActor {
    public func setSpeechFallback(_ fallback: (any AudioTranscriber)?) {
        media.speech = fallback
    }

    public var transcribesAudio: Bool {
        get async { await engine.transcribesAudio }
    }

    public var readsDocuments: Bool { media.document != nil }

    public func transcribe(_ request: TranscriptionRequest) async throws(BasaltError) -> Transcription {
        let usesEngine = await engine.transcribesAudio
        let fallback = media.speech
        guard usesEngine || fallback != nil else { throw .noAudioEncoder(modelInfo) }
        let ticket = try await admitted(.standard)
        defer {
            await withTaskCancellationShield {
                await finish(ticket)
            }
        }
        if await engine.transcribesAudio { return try await engine.transcribe(request) }
        guard let speechFallback = media.speech ?? fallback else { throw .noAudioEncoder(modelInfo) }
        return try await speechFallback.transcribe(request)
    }
}
