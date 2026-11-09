import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import BasaltCore

extension MLXInferenceEngine {
    public var transcribesAudio: Bool {
        get async {
            guard let container else { return false }
            if modelFacts.transcribesAudio { return true }
            return await container.perform { $0.model is NemotronTranscribeModel }
        }
    }

    public var takesImages: Bool {
        get async {
            guard let container else { return true }
            return await container.perform { $0.model is any VLMModel }
        }
    }

    public func transcribe(_ request: TranscriptionRequest) async throws(BasaltError) -> Transcription {
        guard let container else {
            throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first")
        }
        let name = URL(fileURLWithPath: modelDescription).lastPathComponent
        let samples = request.audio.samples
        let temperature = Float(request.temperature)
        let language = request.language
        let duration = request.audio.seconds
        do {
            let text = try await container.performEvaluating(nonSendable: ()) { context, _ in
                guard let model = context.model as? NemotronTranscribeModel else {
                    throw BasaltError.noAudioEncoder(name)
                }
                return try model.transcribe(
                    audio: samples,
                    tokenizer: context.tokenizer,
                    temperature: temperature
                )
            }
            let segments = text.isEmpty ? [] : [Transcription.Segment(start: 0, end: duration, text: text)]
            return Transcription(text: text, language: language, segments: segments)
        } catch let error as BasaltError {
            throw error
        } catch is CancellationError {
            throw .cancelled
        } catch {
            throw .inferenceHardwareFailure(reason: error.localizedDescription)
        }
    }
}
