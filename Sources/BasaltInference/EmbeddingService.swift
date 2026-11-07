import Foundation
import MLX
import MLXLMCommon
import MLXEmbedders
import MLXRerankers
import BasaltCore

public struct EmbeddingService: EmbeddingEngine {
    public let models: [EmbeddingModelInfo]
    private let embedder: EmbedderModelContainer?
    private let isCausal: Bool
    private let poolingStrategyOverride: Pooling.Strategy?
    let reranker: RerankerContainer?

    public init(
        embeddingModel: (id: String, directory: URL)?,
        rerankModel: (id: String, directory: URL)?,
        embeddingAdapter: URL? = nil,
        isCausal: Bool? = nil
    ) async throws(BasaltError) {
        var models: [EmbeddingModelInfo] = []
        if let (id, directory) = embeddingModel {
            let layout = ["modules.json", "1_Pooling/config.json"].contains {
                FileManager.default.fileExists(atPath: directory.appending(path: $0).path)
            }
            if let reason = Self.embedderRefusal(
                config: ModelCache.config(at: directory) ?? [:], sentenceTransformersLayout: layout, name: id,
                hasAdapter: embeddingAdapter != nil) {
                throw .modelConfigurationInvalid(path: directory.path, reason: reason)
            }
            let loaded = try await Self.loading(directory) {
                try await EmbedderModelFactory.shared.loadContainer(from: directory, using: DefaultTokenizerLoader())
            }
            if let embeddingAdapter {
                try await Self.loading(embeddingAdapter) {
                    let lora = try LoRAContainer.from(directory: embeddingAdapter)
                    try await loaded.perform { context in
                        try LeJEPABackbone.applyLoRA(lora, to: context.model)
                    }
                }
            }
            let hasPoolingFile = FileManager.default.fileExists(
                atPath: directory.appending(path: "1_Pooling/config.json").path)
            let baseStrategy = await loaded.poolingStrategy
            self.isCausal = isCausal ?? (embeddingAdapter == nil && baseStrategy == .last)
            self.poolingStrategyOverride = (embeddingAdapter != nil || isCausal == false) && !hasPoolingFile && baseStrategy == .last ? .mean : nil
            embedder = loaded
            let readme = (try? String(contentsOf: directory.appending(component: "README.md"), encoding: .utf8)) ?? ""
            models.append(EmbeddingModelInfo(
                id: id, task: .embedding,
                truncatesDimensions: readme.localizedCaseInsensitiveContains("matryoshka") || readme.contains("MRL")))
        } else {
            embedder = nil
            self.isCausal = false
            self.poolingStrategyOverride = nil
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

    static func embedderRefusal(
        config: [String: Any], sentenceTransformersLayout: Bool, name: String, hasAdapter: Bool = false
    ) -> String? {
        let type = config["model_type"] as? String ?? ""
        let architectures = config["architectures"] as? [String] ?? []
        switch type {
        case "bert", "roberta", "xlm-roberta", "distilbert", "nomic_bert":
            guard architectures.contains(where: { $0.hasSuffix("ForSequenceClassification") }) else { return nil }
            return "\(name) is a sequence-classification (reranker) checkpoint; pass it as --rerank-model."
        case "qwen3":
            guard !sentenceTransformersLayout, !hasAdapter, config["pooling"] == nil,
                  !name.localizedCaseInsensitiveContains("embed") else { return nil }
            return "\(name) is a Qwen3 chat checkpoint (no sentence-transformers pooling config and no \"embed\" in its name); basalt does not pool embeddings from a chat model."
        case "lfm2":
            guard (config["mlx"] as? [String: Any])?["head"] as? String != "embedding" else { return nil }
            return "\(name) is not an LFM2.5 embedding checkpoint (config.json has no \"mlx\": {\"head\": \"embedding\"}); chat and ColBERT checkpoints are not served as embedders."
        default:
            return "\(name) has model_type '\(type)', which is not an embedding model basalt serves: BERT, RoBERTa, XLM-RoBERTa, DistilBERT, NomicBERT, Qwen3-Embedding or LFM2.5-Embedding."
        }
    }

    public func embed(_ inputs: [EmbeddingInput], dimensions: Int?, yieldingTo gate: GenerationGate?) async throws(BasaltError) -> Embeddings {
        guard let embedder else { throw .invalidRequestStructure(details: "no embedding model is loaded; start serve with --embedding-model") }
        let causal = isCausal
        let override = poolingStrategyOverride
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
                var pending = Self.microBatches(tokens.map(\.count))
                while !pending.isEmpty {
                    try Task.checkCancellation()
                    var batch = pending.removeFirst()
                    if let gate, await !gate.idle(within: Self.generationWait), batch.count > 1 {
                        pending.insert(contentsOf: batch.dropFirst().map { [$0] }, at: 0)
                        batch = [batch[0]]
                    }
                    let rows = try Self.forward(
                        context, tokens: batch.map { tokens[$0] }, dimensions: dimensions,
                        isCausal: causal, strategyOverride: override)
                    for (row, index) in batch.enumerated() { vectors[index] = rows[row] }
                }
                return Embeddings(vectors: vectors, promptTokens: total)
            }
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .inferenceError("Embedding failed: \(error)")
        }
    }

    static let generationWait = Duration.seconds(2)

    static func forward(
        _ context: EmbedderModelContext, tokens: [[Int]], dimensions: Int?,
        isCausal: Bool = false, strategyOverride: Pooling.Strategy? = nil
    ) throws -> [[Float]] {
        let length = tokens.map(\.count).max() ?? 0
        let ids = MLXArray(tokens.flatMap { $0 + Array(repeating: 0, count: length - $0.count) }).reshaped(tokens.count, length)
        let mask = MLXArray(tokens.flatMap {
            Array(repeating: Int32(1), count: $0.count) + Array(repeating: Int32(0), count: length - $0.count)
        }).reshaped(tokens.count, length)
        let output = LeJEPABackbone.encode(model: context.model, inputIds: ids, attentionMask: mask, isCausal: isCausal)
        let strategy = strategyOverride ?? context.pooling.strategy
        var pooled = LeJEPABackbone.pool(output, mask: mask, strategy: strategy)
        if let poolDim = context.pooling.dimension, poolDim < pooled.dim(1) {
            pooled = pooled[0..., ..<poolDim]
        }
        if let dimensions {
            guard dimensions <= pooled.dim(1) else {
                throw BasaltError.invalidRequestStructure(details: "dimensions \(dimensions) exceeds the model's \(pooled.dim(1))")
            }
            pooled = pooled[0..., ..<dimensions]
        }
        let width = pooled.dim(1)
        let flat = pooled.MLXEmbedders::l2Normalized().asArray(Float.self)
        return (0 ..< tokens.count).map { Array(flat[$0 * width ..< ($0 + 1) * width]) }
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

    static func rerankOptions(yieldingTo gate: GenerationGate?) -> RerankExecutionOptions {
        var options = RerankExecutionOptions()
        if let gate {
            options.beforeMicroBatch = { await !gate.idle(within: generationWait) }
        }
        return options
    }

    public func rerank(query: String, documents: [String], yieldingTo gate: GenerationGate?) async throws(BasaltError) -> [Double] {
        guard let reranker else { throw .invalidRequestStructure(details: "no rerank model is loaded; start serve with --rerank-model") }
        do {
            return try await reranker.scores(query: query, documents: documents, instruction: nil, options: Self.rerankOptions(yieldingTo: gate))
                .results.map(\.score)
        } catch {
            throw .inferenceError("Rerank failed: \(error)")
        }
    }
}
