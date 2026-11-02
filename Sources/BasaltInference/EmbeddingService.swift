import Foundation
import MLX
import MLXLMCommon
import MLXEmbedders
import MLXRerankers
import BasaltCore

public struct EmbeddingService: EmbeddingEngine {
    public let models: [EmbeddingModelInfo]
    private let embedder: EmbedderModelContainer?
    private let reranker: RerankerContainer?

    public init(
        embeddingModel: (id: String, directory: URL)?, rerankModel: (id: String, directory: URL)?
    ) async throws(BasaltError) {
        var models: [EmbeddingModelInfo] = []
        if let (id, directory) = embeddingModel {
            let layout = ["modules.json", "1_Pooling/config.json"].contains {
                FileManager.default.fileExists(atPath: directory.appending(path: $0).path)
            }
            if let reason = Self.embedderRefusal(
                config: ModelCache.config(at: directory) ?? [:], sentenceTransformersLayout: layout, name: id) {
                throw .modelConfigurationInvalid(path: directory.path, reason: reason)
            }
            embedder = try await Self.loading(directory) {
                try await EmbedderModelFactory.shared.loadContainer(from: directory, using: DefaultTokenizerLoader())
            }
            let readme = (try? String(contentsOf: directory.appending(component: "README.md"), encoding: .utf8)) ?? ""
            models.append(EmbeddingModelInfo(
                id: id, task: .embedding,
                truncatesDimensions: readme.localizedCaseInsensitiveContains("matryoshka") || readme.contains("MRL")))
        } else {
            embedder = nil
        }
        if let (id, directory) = rerankModel {
            reranker = try await Self.loading(directory) {
                try await RerankerModelFactory.shared.loadContainer(
                    from: directory, using: DefaultTokenizerLoader(),
                    allowUnverifiedModel: id.localizedCaseInsensitiveContains("rerank"))
            }
            models.append(EmbeddingModelInfo(id: id, task: .rerank))
        } else {
            reranker = nil
        }
        self.models = models
    }

    private static func loading<Loaded>(_ directory: URL, _ load: () async throws -> Loaded) async throws(BasaltError) -> Loaded {
        do {
            return try await load()
        } catch {
            throw .modelConfigurationInvalid(path: directory.path, reason: error.localizedDescription)
        }
    }

    static func embedderRefusal(config: [String: Any], sentenceTransformersLayout: Bool, name: String) -> String? {
        let type = config["model_type"] as? String ?? ""
        let architectures = config["architectures"] as? [String] ?? []
        switch type {
        case "bert", "roberta", "xlm-roberta", "distilbert", "nomic_bert":
            guard architectures.contains(where: { $0.hasSuffix("ForSequenceClassification") }) else { return nil }
            return "\(name) is a sequence-classification (reranker) checkpoint; pass it as --rerank-model."
        case "qwen3":
            guard !sentenceTransformersLayout, !name.localizedCaseInsensitiveContains("embed") else { return nil }
            return "\(name) is a Qwen3 chat checkpoint (no sentence-transformers pooling config and no \"embed\" in its name); basalt does not pool embeddings from a chat model."
        case "lfm2":
            guard (config["mlx"] as? [String: Any])?["head"] as? String != "embedding" else { return nil }
            return "\(name) is not an LFM2.5 embedding checkpoint (config.json has no \"mlx\": {\"head\": \"embedding\"}); chat and ColBERT checkpoints are not served as embedders."
        default:
            return "\(name) has model_type '\(type)', which is not an embedding model basalt serves: BERT, RoBERTa, XLM-RoBERTa, DistilBERT, NomicBERT, Qwen3-Embedding or LFM2.5-Embedding."
        }
    }

    public func embed(_ inputs: [EmbeddingInput], dimensions: Int?) async throws(BasaltError) -> Embeddings {
        guard let embedder else { throw .invalidRequestStructure(details: "no embedding model is loaded; start serve with --embedding-model") }
        do {
            return try await embedder.perform { context in
                let tokens = inputs.map { input in
                    switch input {
                    case .text(let text): context.tokenizer.encode(text: text, addSpecialTokens: true)
                    case .tokens(let ids): ids
                    }
                }
                let total = tokens.reduce(0) { $0 + $1.count }
                guard total <= EmbeddingLimits.maxTokens else {
                    throw BasaltError.contextLengthExceeded(maxTokens: EmbeddingLimits.maxTokens, requested: total)
                }
                let longest = tokens.map(\.count).max() ?? 0
                if let limit = context.model.maxPositionEmbeddings, longest > limit {
                    throw BasaltError.contextLengthExceeded(maxTokens: limit, requested: longest)
                }
                let vocabulary = 0 ..< context.model.vocabularySize
                guard tokens.allSatisfy({ !$0.isEmpty && $0.allSatisfy(vocabulary.contains) }) else {
                    throw BasaltError.invalidRequestStructure(details: "every input needs at least one token, each below the vocabulary size \(vocabulary.upperBound)")
                }
                var vectors = [[Float]](repeating: [], count: tokens.count)
                for batch in Self.microBatches(tokens.map(\.count)) {
                    try Task.checkCancellation()
                    let length = tokens[batch[batch.count - 1]].count
                    let ids = MLXArray(batch.flatMap { tokens[$0] + Array(repeating: 0, count: length - tokens[$0].count) })
                        .reshaped(batch.count, length)
                    let mask = MLXArray(batch.flatMap {
                        Array(repeating: Int32(1), count: tokens[$0].count) + Array(repeating: Int32(0), count: length - tokens[$0].count)
                    }).reshaped(batch.count, length)
                    var pooled = context.pooling(
                        context.model(ids, positionIds: nil, tokenTypeIds: MLXArray.zeros(like: ids), attentionMask: mask), mask: mask)
                    if let dimensions {
                        guard dimensions <= pooled.dim(1) else {
                            throw BasaltError.invalidRequestStructure(details: "dimensions \(dimensions) exceeds the model's \(pooled.dim(1))")
                        }
                        pooled = pooled[0..., ..<dimensions]
                    }
                    let width = pooled.dim(1)
                    let rows = pooled.MLXEmbedders::l2Normalized().asArray(Float.self)
                    for (row, index) in batch.enumerated() {
                        vectors[index] = Array(rows[row * width ..< (row + 1) * width])
                    }
                }
                return Embeddings(vectors: vectors, promptTokens: total)
            }
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .inferenceError("Embedding failed: \(error)")
        }
    }

    static func microBatches(_ lengths: [Int], maxTokens: Int = 8_192, maxCount: Int = 32) -> [[Int]] {
        var batches: [[Int]] = []
        for index in lengths.indices.sorted(by: { lengths[$0] < lengths[$1] }) {
            if let last = batches.last, last.count < maxCount, lengths[index] * (last.count + 1) <= maxTokens {
                batches[batches.count - 1].append(index)
            } else {
                batches.append([index])
            }
        }
        return batches
    }

    public func rerank(query: String, documents: [String]) async throws(BasaltError) -> [Double] {
        guard let reranker else { throw .invalidRequestStructure(details: "no rerank model is loaded; start serve with --rerank-model") }
        do {
            return try await reranker.scores(query: query, documents: documents, instruction: nil, options: RerankExecutionOptions())
                .results.map(\.score)
        } catch {
            throw .inferenceError("Rerank failed: \(error)")
        }
    }
}
