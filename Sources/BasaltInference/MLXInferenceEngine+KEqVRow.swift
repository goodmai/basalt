import Foundation
import BasaltCore
import MLXLMCommon

extension MLXInferenceEngine {
    func probeAttentionCaches(directory: URL, epoch: ModelEpoch) async {
        await probeQuest()
        await probeKEqVRow(config: ModelCache.config(at: directory))
    }

    var kvSharedRowApplies: Bool { configuration.kvSharedRow && modelFacts.kEqVRow != nil }

    public var sharedKVRowLayout: KEqVRowLayout? { kvSharedRowApplies ? modelFacts.kEqVRow : nil }

    func probeKEqVRow(config: [String: Any]?) async {
        guard configuration.kvSharedRow, let container = warmingContainer else { return }
        let epoch = modelEpoch
        guard let layout = config.flatMap(KEqVRowLayout.init(config:)) else {
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
        logger.info("KV shared row (experimental, #645): \(layout.globalLayers) global layers store one \(layout.rowWidth)-value row per KV head instead of \(2 * layout.headDim) values, \(bytes.stock) → \(bytes.row) bytes per token; the in-memory prompt cache keeps rows, the disk prompt cache is skipped")
    }
}
