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
        guard await engine.transcribesAudio else {
            guard let speechFallback = media.speech else { throw .noAudioEncoder(modelInfo) }
            return try await speechFallback.transcribe(request)
        }
        let ticket = try await admitted(.standard)
        defer {
            await withTaskCancellationShield {
                await finish(ticket)
            }
        }
        return try await engine.transcribe(request)
    }
}
