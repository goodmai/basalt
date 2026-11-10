import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import BasaltCore

final class NemotronTranscribeModel: Module, LLMModel, KVCacheDimensionProvider {
    static let modelType = "nemotron_transcribe"
    static let asrPrompt =
        "Produce a verbatim transcript of the audio. Preserve named entities, "
        + "abbreviations, numbers, dates, measurements, acronyms, and technical terms "
        + "as clearly as possible. Keep the same language as spoken and do not translate."

    let configuration: NemotronTranscribeConfiguration
    @ModuleInfo(key: "llm") var llm: NemotronHModel
    @ModuleInfo(key: "perception") var perception: NemotronPerception

    var vocabularySize: Int { llm.vocabularySize }
    var kvHeads: [Int] { llm.kvHeads }
    var loraLayers: [Module] { llm.loraLayers }

    init(_ configuration: NemotronTranscribeConfiguration) {
        self.configuration = configuration
        self._llm.wrappedValue = NemotronHModel(configuration.llmConfig)
        self._perception.wrappedValue = NemotronPerception(configuration)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try llm.newCache(parameters: parameters)
    }

    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        llm(inputs, cache: cache, inputEmbeddings: inputEmbeddings)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        llm(inputs, cache: cache, inputEmbeddings: nil)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var llmWeights = [String: MLXArray]()
        var sanitized = [String: MLXArray]()

        for (rawKey, rawValue) in weights {
            if rawKey.hasSuffix("._extra_state")
                || rawKey.hasPrefix("ctc_timestamp.")
                || rawKey.hasPrefix("llm.mtp.")
            {
                continue
            }
            var key = rawKey
            var value = rawValue
            if key.hasPrefix("backbone.") || key.hasPrefix("lm_head.") {
                key = "llm." + key
            }
            key = key.replacingOccurrences(of: "llm.model.embed_tokens.", with: "llm.backbone.embeddings.")
            key = key.replacingOccurrences(of: "llm.model.norm.", with: "llm.backbone.norm_f.")
            key = key.replacingOccurrences(of: "llm.model.layers.", with: "llm.backbone.layers.")
            key = key.replacingOccurrences(of: "._fp32_params.", with: ".")
            key = key.replacingOccurrences(of: ".ffn.net.0.", with: ".ffn.linear1.")
            key = key.replacingOccurrences(of: ".ffn.net.3.", with: ".ffn.linear2.")
            key = key.replacingOccurrences(of: ".preprocessor.featurizer.", with: ".preprocessor.")

            if key.hasPrefix("llm.") {
                if key.hasSuffix(".gate.weight") {
                    value = value.asType(.float32)
                }
                let stripped = String(key.dropFirst(4))
                llmWeights[stripped] = value
            } else {
                if key.hasSuffix(".subpixel_upsample.weight"), value.dim(-1) == 3 {
                    value = value.swappedAxes(1, 2)
                }
                sanitized[key] = value
            }
        }

        for (key, value) in llm.sanitize(weights: llmWeights) {
            sanitized["llm." + key] = value
        }
        return sanitized
    }

    func encodeAudio(_ samples: MLXArray) throws -> MLXArray {
        let embeds = try perception(samples)
        eval(embeds)
        return embeds
    }

    func spliceEmbeddings(
        ids: [Int], audioIndex: Int, audioEmbeds: MLXArray
    ) -> (tokens: [Int], tokenArray: MLXArray, embeddings: MLXArray) {
        let nAudio = audioEmbeds.dim(1)
        let expanded = Array(ids[..<audioIndex])
            + Array(repeating: configuration.padTokenId, count: nAudio)
            + Array(ids[(audioIndex + 1)...])
        let tokenArray = MLXArray(expanded.map(Int32.init))
        let baseEmbeddings = llm.embedTokens(tokenArray.expandedDimensions(axis: 0))
        var parts = [MLXArray]()
        if audioIndex > 0 {
            parts.append(baseEmbeddings[0..., ..<audioIndex, 0...])
        }
        if nAudio > 0 {
            parts.append(audioEmbeds.asType(baseEmbeddings.dtype))
        }
        let suffixStart = audioIndex + nAudio
        if suffixStart < expanded.count {
            parts.append(baseEmbeddings[0..., suffixStart..., 0...])
        }
        let spliced = concatenated(parts, axis: 1)
        eval(spliced)
        return (expanded, tokenArray, spliced[0])
    }

    func promptTokenIds(tokenizer: any MLXLMCommon.Tokenizer) throws(BasaltError) -> (ids: [Int], audioIndex: Int) {
        let tag = configuration.audioLocatorTag
        let userContent = "\(Self.asrPrompt) \(tag)"
        let turns = [
            "<|im_start|>system\n<|im_end|>\n",
            "<|im_start|>user\n\(userContent)<|im_end|>\n",
            "<|im_start|>assistant\n<think></think>",
        ]
        let ids = turns.flatMap { tokenizer.encode(text: $0, addSpecialTokens: false) }
        let placeholderId = tokenizer.convertTokenToId(tag)
            ?? tokenizer.encode(text: tag, addSpecialTokens: false).first
        guard let placeholderId, let index = ids.firstIndex(of: placeholderId) else {
            throw .modelConfigurationInvalid(
                path: Self.modelType,
                reason: "tokenizer does not contain audio locator token '\(tag)'"
            )
        }
        return (ids, index)
    }

    private func sampleToken(logits: MLXArray, temperature: Float) -> MLXArray {
        guard temperature > 0 else {
            return argMax(logits, axis: -1)
        }
        let scaled = logits.asType(.float32) * (1.0 / temperature)
        return categorical(scaled, axis: -1)
    }

    func transcribe(
        audio: [Float],
        tokenizer: any MLXLMCommon.Tokenizer,
        temperature: Float = 0,
        maxTokens: Int = 4096,
        prefillStepSize: Int = 512
    ) throws -> String {
        let minSamples = max(1, configuration.perception.preprocessor.hopLength)
        let paddedSamples = audio.count >= minSamples
            ? audio
            : audio + Array(repeating: Float(0), count: minSamples - audio.count)
        let audioEmbeds = try encodeAudio(MLXArray(paddedSamples))
        let (ids, audioIndex) = try promptTokenIds(tokenizer: tokenizer)
        let (expanded, tokens, embeddings) = spliceEmbeddings(
            ids: ids, audioIndex: audioIndex, audioEmbeds: audioEmbeds
        )
        let cache = try newCache(parameters: nil)
        var offset = 0
        let total = expanded.count
        while total - offset > 1 {
            try Task.checkCancellation()
            let step = min(prefillStepSize, total - offset - 1)
            let chunkTokens = tokens[offset ..< (offset + step)].expandedDimensions(axis: 0)
            let chunkEmb = embeddings[offset ..< (offset + step), 0...].expandedDimensions(axis: 0)
            _ = callAsFunction(chunkTokens, cache: cache, inputEmbeddings: chunkEmb)
            eval(cache.flatMap(\.state))
            offset += step
        }
        try Task.checkCancellation()
        let lastToken = tokens[offset ..< total].expandedDimensions(axis: 0)
        let lastEmb = embeddings[offset ..< total, 0...].expandedDimensions(axis: 0)
        let firstLogits = callAsFunction(lastToken, cache: cache, inputEmbeddings: lastEmb)[0..., -1, 0...]
        var current = sampleToken(logits: firstLogits, temperature: temperature)
        eval(current)

        var eosSet = Set(configuration.eosTokenIds)
        if let eosToken = tokenizer.eosToken, let eosId = tokenizer.convertTokenToId(eosToken) {
            eosSet.insert(eosId)
        }
        var generated = [Int]()
        asyncEval(current)
        while generated.count < maxTokens {
            try Task.checkCancellation()
            let nextLogits = callAsFunction(current.expandedDimensions(axis: 0), cache: cache)[0..., -1, 0...]
            let next = sampleToken(logits: nextLogits, temperature: temperature)
            asyncEval(next)
            let tokenId = current.item(Int.self)
            if eosSet.contains(tokenId) {
                break
            }
            generated.append(tokenId)
            current = next
        }
        return tokenizer.decode(tokenIds: generated, skipSpecialTokens: true)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

extension MLXInferenceEngine {
    public static func transcribesAudio(modelType: String?) -> Bool { modelType == NemotronTranscribeModel.modelType }
}
