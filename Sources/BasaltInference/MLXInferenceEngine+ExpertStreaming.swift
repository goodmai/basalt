import Foundation
import MLXLMCommon
import BasaltCore

extension MLXInferenceEngine {
    public func expertCacheStats() async -> (stats: ExpertCacheStats, capacity: ByteCount)? {
        guard let container else { return nil }
        return await container.performEvaluating(nonSendable: ()) { context, _ in
            ExpertStreamingLoader.store(in: context.model).map { ($0.stats, $0.capacity) }
        }
    }

    func loadContainer(from directory: URL) async throws -> ModelContainer {
        guard let expertStreaming = configuration.expertStreaming else { return try await loadModelContainer(from: directory, using: DefaultTokenizerLoader()) }
        let loaded = try await ExpertStreamingLoader.loadContainer(from: directory, streaming: expertStreaming, tokenizerLoader: DefaultTokenizerLoader())
        if let capacity = await loaded.perform({ ExpertStreamingLoader.store(in: $0.model)?.capacity }) {
            logger.info("Experimental expert streaming: routed experts stay in the safetensors shards and load on demand into a \(formatRAMBytes(capacity.rawValue)) LRU expert cache")
        }
        return loaded
    }
}
