import BasaltCore
import Foundation
import Metal
import MLXLMCommon

struct PrefillAttentionScratch: Sendable, Equatable {
    static let budgetBytes = 256 << 20
    static let minimumStep = 64
    static let fusedHeadDims: Set<Int> = [64, 72, 80, 96, 128, 192, 256]

    let heads: Int

    func stepCap(keys: Int) -> Int? {
        let cap = max(Self.minimumStep, Self.budgetBytes / MemoryLayout<Float>.size / max(heads, 1) / max(keys, 1))
        return cap < PrefillParameters.defaultStepSize ? cap : nil
    }

    static func unfused(config: [String: Any]?, nax: Bool) -> Self? {
        let text = config?["text_config"] as? [String: Any] ?? config ?? [:]
        guard let heads = text["num_attention_heads"] as? Int, heads > 0 else { return nil }
        let fused = nax ? fusedHeadDims.union([512]) : fusedHeadDims
        var dims = ["head_dim", "global_head_dim"].compactMap { text[$0] as? Int }
        if let rank = text["kv_lora_rank"] as? Int, let rope = text["qk_rope_head_dim"] as? Int,
           (text["model_type"] as? String) == BailingHybridModel.modelType || text["layer_group_size"] is Int
        {
            if rope > 0 && !(rank + rope == 192 && rank == 128) { return Self(heads: heads) }
            dims.append(rank + rope)
        }
        return dims.contains { !fused.contains($0) } ? Self(heads: heads) : nil
    }

    static func naxAvailable(architecture: String) -> Bool {
        guard let match = architecture.firstMatch(of: /g(\d+)([a-z])$/), let generation = Int(match.1) else { return false }
        return generation >= (match.2 == "p" ? 18 : 17)
    }

    static var deviceHasNAX: Bool {
        MTLCreateSystemDefaultDevice().map { naxAvailable(architecture: $0.architecture.name) } ?? false
    }
}

extension MLXInferenceEngine {
    func prefillScratch(at directory: URL) -> PrefillAttentionScratch? {
        let scratch = PrefillAttentionScratch.unfused(config: ModelCache.config(at: directory), nax: PrefillAttentionScratch.deviceHasNAX)
        if let scratch {
            logger.info("Prefill attention runs unfused on this GPU (a head dim outside the fused kernel's set): long prompts prefill in chunks that keep \(scratch.heads)-head scores under \(PrefillAttentionScratch.budgetBytes >> 20) MB (#741)")
        }
        return scratch
    }
}
