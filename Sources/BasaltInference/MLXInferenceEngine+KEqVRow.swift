import Foundation
import BasaltCore
import MLXLMCommon

extension MLXInferenceEngine {
    func probeAttentionCaches(directory: URL, epoch: ModelEpoch) async {
        await probeQuest()
        await probeKEqVRow(directory: directory, epoch: epoch)
    }

    func probeKEqVRow(directory: URL, epoch: ModelEpoch) async {
        guard configuration.kvSharedRow, let container else { return }
        guard let layout = ModelCache.config(at: directory).flatMap(KEqVRowLayout.init(config:)) else {
            return logger.warn("--experimental-kv-shared-row does not apply: the model's config has no attention_k_eq_v with partial RoPE on its global layers (Gemma 4 12B, 26B-A4B and 31B have it); the KV cache stays stock")
        }
        do {
            guard try await container.perform({ try KEqVRowKVCache.modelAttendsThroughCache($0.model, layout: layout) }) else {
                return logger.warn("--experimental-kv-shared-row does not apply: the global layers did not attend through the row cache; the KV cache stays stock")
            }
        } catch {
            return logger.warn("--experimental-kv-shared-row probe failed (\(error.localizedDescription)); the KV cache stays stock")
        }
        record(.learned(epoch) { $0.kEqVRow = layout })
        let bytes = layout.globalBytesPerToken()
        logger.info("KV shared row (experimental, #645): \(layout.globalLayers) global layers store one \(layout.rowWidth)-value row per KV head instead of \(2 * layout.headDim) values, \(bytes.stock) → \(bytes.row) bytes per token; requests without a prompt-cache namespace use it")
    }
}
