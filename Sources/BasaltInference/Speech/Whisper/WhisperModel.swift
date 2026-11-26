import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

final class WhisperModel: Module, LLMModel, SpeechRecognitionModel {
    static let modelType = "whisper"

    let configuration: WhisperConfiguration
    @ModuleInfo(key: "encoder") var encoder: WhisperAudioEncoder
    @ModuleInfo(key: "decoder") var decoder: WhisperTextDecoder

    init(_ configuration: WhisperConfiguration) {
        self.configuration = configuration
        _encoder.wrappedValue = WhisperAudioEncoder(configuration)
        _decoder.wrappedValue = WhisperTextDecoder(configuration)
    }

    convenience init(configurationData data: Data) throws {
        self.init(try JSONDecoder.json5().decode(WhisperConfiguration.self, from: data))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        decoder(inputs, cross: nil, cache: cache)
    }

    var loraLayers: [Module] { decoder.blocks }

    func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        decoder.blocks.map { _ in KVCacheSimple() }
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { $0.key != "alignment_heads" }
    }

    func encode(_ samples: ArraySlice<Float>) -> [WhisperCrossKV] {
        let filters = WhisperAudio.melFilters(count: configuration.melCount)
        let mel = WhisperAudio.logMel(samples, filters: filters, frames: configuration.melFrames)
        let cross = decoder.crossKeysValues(encoder(mel.expandedDimensions(axis: 0)))
        eval(cross.flatMap { [$0.keys, $0.values] })
        return cross
    }

    func transcribe(audio: [Float], tokenizer: any MLXLMCommon.Tokenizer, temperature: Float, language: String?) throws -> String {
        let tokens = WhisperSpecialTokens(vocabularySize: configuration.vocabularySize).resolved(by: tokenizer)
        var spoken = tokens.language(language, tokenizer: tokenizer)
        var texts: [String] = []
        for start in stride(from: 0, to: max(audio.count, 1), by: configuration.windowSamples) {
            try Task.checkCancellation()
            let cross = encode(audio[start ..< min(start + configuration.windowSamples, audio.count)])
            spoken = spoken ?? detectLanguage(cross: cross, tokens: tokens)
            let ids = try decode(cross: cross, prompt: tokens.prompt(language: spoken), endOfText: tokens.endOfText, temperature: temperature)
            texts.append(tokenizer.decode(tokenIds: ids, skipSpecialTokens: true).trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return texts.filter { !$0.isEmpty }.joined(separator: " ")
    }

    func detectLanguage(cross: [WhisperCrossKV], tokens: WhisperSpecialTokens) -> Int? {
        let languages = tokens.languages
        guard tokens.multilingual, !languages.isEmpty, languages.upperBound <= configuration.vocabularySize else { return nil }
        let start = MLXArray([Int32(tokens.startOfTranscript)]).reshaped(1, 1)
        let logits = decoder(start, cross: cross, cache: nil)[0, -1]
        return languages.lowerBound + argMax(logits[languages.lowerBound ..< languages.upperBound]).item(Int.self)
    }

    func decode(cross: [WhisperCrossKV], prompt: [Int], endOfText: Int, temperature: Float) throws -> [Int] {
        let cache = newCache(parameters: nil)
        let step = { (tokens: MLXArray) in
            self.pick(self.decoder(tokens, cross: cross, cache: cache)[0..., -1, 0...], endOfText: endOfText, temperature: temperature)
        }
        var current = step(MLXArray(prompt.map(Int32.init)).reshaped(1, -1))
        var generated: [Int] = []
        while true {
            try Task.checkCancellation()
            let next: MLXArray? = prompt.count + generated.count < configuration.textContext ? step(current.reshaped(1, 1)) : nil
            if let next { asyncEval(next) }
            let id = current.item(Int.self)
            guard id != endOfText else { break }
            generated.append(id)
            guard let next else { break }
            current = next
        }
        return generated
    }

    func pick(_ logits: MLXArray, endOfText: Int, temperature: Float) -> MLXArray {
        let allowed = logits[0..., ..<(endOfText + 1)].asType(.float32)
        guard temperature > 0 else { return argMax(allowed, axis: -1) }
        return categorical(allowed * (1 / temperature), axis: -1)
    }
}
