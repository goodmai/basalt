import Foundation
import CryptoKit
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

extension MLXInferenceEngine {
    static let scoreTopK = 1...20
    static let scoreTokens = 1...8192
    static let scoreChunk = 256
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
        let hasText = !(request.completionText ?? "").isEmpty
        guard scoreTokens.contains(request.completion.count) || (request.completion.isEmpty && hasText) else {
            throw refused("completion must hold \(scoreTokens) token ids (or provide completion_text)")
        }
        guard scoreTopK.contains(request.topLogprobs) else { throw refused("top_logprobs must be in \(scoreTopK)") }
    }

    static func score(_ request: ScoreRequest, model: any LanguageModel, tokenizer: any Tokenizer) throws -> ScoreReport {
        let completion: [Int]
        if !request.completion.isEmpty {
            completion = request.completion
        } else if let text = request.completionText, !text.isEmpty {
            completion = tokenizer.encode(text: text, addSpecialTokens: false)
            guard scoreTokens.contains(completion.count) else {
                throw BasaltError.invalidRequestStructure(details: "score: completion_text tokenized to \(completion.count) tokens, expected \(scoreTokens)")
            }
        } else {
            throw BasaltError.invalidRequestStructure(details: "score: completion or completion_text is required")
        }
        let vocabulary = 0 ..< vocabularySize(model)
        guard completion.allSatisfy(vocabulary.contains), request.topLogprobs <= vocabulary.count else {
            throw BasaltError.invalidRequestStructure(details: "score: completion ids and top_logprobs must be below the vocabulary size \(vocabulary.upperBound)")
        }
        let prompt = try labPrompt(messages: request.messages, tokenizer)
        guard !prompt.isEmpty else { throw BasaltError.invalidRequestStructure(details: "score: the chat template rendered the messages to no tokens") }
        let scored = try scorePositions(prompt: prompt, completion: completion, k: request.topLogprobs, model: model)
        return ScoreReport(tokenizer: tokenizerFingerprint(tokenizer), promptTokens: prompt.count, positions: scored.positions, meanNLL: scored.meanNLL)
    }

    static func topLogprobs(prompt: [Int], completion: [Int], k: Int, model: any LanguageModel, chunk: Int = scoreChunk) throws -> [TopLogprobs] {
        try scorePositions(prompt: prompt, completion: completion, k: k, model: model, chunk: chunk).positions
    }

    static func scorePositions(
        prompt: [Int], completion: [Int], k: Int, model: any LanguageModel, chunk: Int = scoreChunk
    ) throws -> (positions: [TopLogprobs], meanNLL: Double) {
        let cache = try model.newCache(parameters: nil)
        if prompt.count > 1 { _ = model(MLXArray(prompt.dropLast().map(Int32.init))[.newAxis], cache: cache) }
        let inputs = [prompt[prompt.count - 1]] + completion.dropLast()
        var positions: [TopLogprobs] = []
        positions.reserveCapacity(inputs.count)
        var nllSum = 0.0
        for start in stride(from: 0, to: inputs.count, by: chunk) {
            let end = min(start + chunk, inputs.count)
            let ids = inputs[start ..< end].map(Int32.init)
            let logits = model(MLXArray(ids)[.newAxis], cache: cache)[0].asType(.float32)
            let logprobs = logits - logSumExp(logits, axis: -1, keepDims: true)
            let targets = MLXArray(completion[start ..< end].map(Int32.init))[0..., .newAxis]
            nllSum -= Double(takeAlong(logprobs, targets, axis: -1).sum().item(Float.self))
            positions.append(contentsOf: topK(fromLogprobs: logprobs, k: k))
        }
        return (positions, nllSum / Double(max(1, completion.count)))
    }

    static func topK(_ logits: MLXArray, k: Int) -> [TopLogprobs] {
        topK(fromLogprobs: logits - logSumExp(logits, axis: -1, keepDims: true), k: k)
    }

    static func topK(fromLogprobs logprobs: MLXArray, k: Int) -> [TopLogprobs] {
        let candidates = argPartition(-logprobs, kth: k - 1, axis: -1)[0..., ..<k]
        let (ids, values) = (candidates.asType(.int32).asArray(Int32.self), takeAlong(logprobs, candidates, axis: -1).asArray(Float.self))
        return (0 ..< logprobs.dim(0)).map { row in
            let ranked = (row * k ..< (row + 1) * k).sorted { (values[$0], -ids[$0]) > (values[$1], -ids[$1]) }
            return TopLogprobs(ids: ranked.map { Int(ids[$0]) }, logprobs: ranked.map { values[$0] })
        }
    }

    static func vocabularySize(_ model: any LanguageModel) -> Int {
        let modules = (model as Module).namedModules().map { (name: $0.0.split(separator: ".").last.map(String.init) ?? "", module: $0.1) }
        let head = modules.first { $0.name == "lm_head" }?.module as? Linear
        let table = modules.first { ["embed_tokens", "embeddings"].contains($0.name) && $0.module is Embedding }?.module as? Embedding
        return head?.weight.dim(0) ?? table?.weight.dim(0) ?? model(MLXArray([Int32(0)])[.newAxis], cache: nil).dim(-1)
    }

    static func tokenizerFingerprint(_ tokenizer: any Tokenizer) -> String {
        let ids = tokenizer.encode(text: tokenizerProbe, addSpecialTokens: false) + [tokenizer.eosTokenId ?? -1]
        return SHA256.hash(data: Data(ids.map(String.init).joined(separator: ",").utf8)).prefix(8).map { String(format: "%02x", $0) }.joined()
    }
}
