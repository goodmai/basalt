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
        [NemotronTranscribeModel.modelType, WhisperModel.modelType, ParakeetModel.modelType].contains(modelType)
    }

    public static func speechModelType(config: [String: Any]?) -> String? {
        guard let config else { return nil }
        return config["model_type"] as? String ?? (ParakeetConfiguration.describes(config) ? ParakeetModel.modelType : nil)
    }
}
