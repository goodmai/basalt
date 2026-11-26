import MLXLMCommon

protocol SpeechRecognitionModel {
    func transcribe(audio: [Float], tokenizer: any MLXLMCommon.Tokenizer, temperature: Float, language: String?) throws -> String
}

extension NemotronTranscribeModel: SpeechRecognitionModel {
    func transcribe(audio: [Float], tokenizer: any MLXLMCommon.Tokenizer, temperature: Float, language: String?) throws -> String {
        try transcribe(audio: audio, tokenizer: tokenizer, temperature: temperature)
    }
}

extension MLXInferenceEngine {
    public static func transcribesAudio(modelType: String?) -> Bool {
        [NemotronTranscribeModel.modelType, WhisperModel.modelType].contains(modelType)
    }
}
