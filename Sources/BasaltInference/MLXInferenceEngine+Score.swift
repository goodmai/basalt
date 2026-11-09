import Foundation
import CryptoKit
import MLX
import MLXLMCommon
import BasaltCore

extension MLXInferenceEngine {
    static let scoreTopK = 1...20
    static let scoreTokens = 1...32_768
    static let tokenizerProbe = "Basalt tokenizer probe: Grüße aus 東京, 3.14159 + x**2 != y; <think>ok</think>\n\tdef f(): return [1, 2]"

    public func score(_ request: ScoreRequest) async throws(BasaltError) -> ScoreReport {
        guard let container else { throw .modelNotLoaded }
        try Self.validate(request)
        return try await Self.labRun(container, "scoring") { context in try Self.score(request, model: context.model, tokenizer: context.tokenizer) }
    }

    static func validate(_ request: ScoreRequest) throws(BasaltError) {
        func refused(_ reason: String) -> BasaltError { .invalidRequestStructure(details: "score: \(reason)") }
        guard !request.messages.isEmpty, request.messages.allSatisfy({ $0["role"] != nil && $0["content"] != nil }) else {
            throw refused("messages must be a non-empty array of {\"role\": …, \"content\": …}")
        }
        guard scoreTokens.contains(request.completion.count) else { throw refused("completion must hold \(scoreTokens) token ids") }
        guard scoreTopK.contains(request.topLogprobs) else { throw refused("top_logprobs must be in \(scoreTopK)") }
    }

    static func score(_ request: ScoreRequest, model: any LanguageModel, tokenizer: any Tokenizer) throws -> ScoreReport {
        let vocabulary = 0 ..< vocabularySize(model)
        guard request.completion.allSatisfy(vocabulary.contains), request.topLogprobs <= vocabulary.count else {
            throw BasaltError.invalidRequestStructure(details: "score: completion ids and top_logprobs must be below the vocabulary size \(vocabulary.upperBound)")
        }
        let prompt = try labPrompt(messages: request.messages, tokenizer)
        guard !prompt.isEmpty else { throw BasaltError.invalidRequestStructure(details: "score: the chat template rendered the messages to no tokens") }
        let positions = topLogprobs(prompt: prompt, completion: request.completion, k: request.topLogprobs, model: model)
        return ScoreReport(tokenizer: tokenizerFingerprint(tokenizer), promptTokens: prompt.count, positions: positions)
    }

    static func topLogprobs(prompt: [Int], completion: [Int], k: Int, model: any LanguageModel) -> [TopLogprobs] {
        let tokens = prompt + completion
        let logits = model(MLXArray(tokens.map(Int32.init))[.newAxis], cache: nil)[0, (prompt.count - 1) ..< (tokens.count - 1)].asType(.float32)
        let logprobs = logits - logSumExp(logits, axis: -1, keepDims: true)
        let ids = argSort(-logprobs, axis: -1)[0..., ..<k]
        let (flatIds, flatValues) = (ids.asType(.int32).asArray(Int32.self), takeAlong(logprobs, ids, axis: -1).asArray(Float.self))
        return (0 ..< completion.count).map { row in
            let range = row * k ..< (row + 1) * k
            return TopLogprobs(ids: flatIds[range].map(Int.init), logprobs: Array(flatValues[range]))
        }
    }

    static func vocabularySize(_ model: any LanguageModel) -> Int {
        model(MLXArray([Int32(0)])[.newAxis], cache: nil).dim(-1)
    }

    static func tokenizerFingerprint(_ tokenizer: any Tokenizer) -> String {
        let ids = tokenizer.encode(text: tokenizerProbe, addSpecialTokens: false) + [tokenizer.eosTokenId ?? -1]
        return SHA256.hash(data: Data(ids.map(String.init).joined(separator: ",").utf8)).prefix(8).map { String(format: "%02x", $0) }.joined()
    }
}
