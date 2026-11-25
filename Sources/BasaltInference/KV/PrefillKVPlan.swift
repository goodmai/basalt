import Foundation
import MLX
import MLXLMCommon

struct PrefillKVPlan: Sendable, Equatable {
    static let thresholdTokens = 1_024
    static let defaultStep = 256
    static let maxStaggerGroups = 8
    static let defaultDecodeReserve = 256
    static let maxDecodeReserve = 1_024

    let initialSteps: [Int]
    let decodeStep: Int

    var isActive: Bool { !initialSteps.isEmpty }

    static func make(
        promptTokens: Int,
        reusedTokens: Int = 0,
        simpleLayerCount: Int,
        maxTokens: Int? = nil,
        quantizesAfterPrefill: Bool = false
    ) -> Self {
        let newTokens = promptTokens - max(0, reusedTokens)
        guard simpleLayerCount > 0, newTokens >= thresholdTokens else {
            return Self(initialSteps: [], decodeStep: defaultStep)
        }
        let reserve = quantizesAfterPrefill ? 0 : min(max(0, maxTokens ?? defaultDecodeReserve), maxDecodeReserve)
        let base = newTokens + reserve
        let groups = min(simpleLayerCount, maxStaggerGroups)
        let steps = (0 ..< simpleLayerCount).map { base + ($0 % groups) * defaultStep }
        return Self(initialSteps: steps, decodeStep: groups * defaultStep)
    }

    static func simpleLayers(in layers: [any KVCache]) -> [KVCacheSimple] {
        layers.compactMap { type(of: $0) == KVCacheSimple.self ? ($0 as? KVCacheSimple) : nil }
    }

    func apply(to layers: [any KVCache]) {
        guard isActive else { return }
        for (layer, step) in zip(Self.simpleLayers(in: layers), initialSteps) {
            layer.step = step
        }
    }

    func restore(_ layers: [any KVCache]) {
        guard isActive else { return }
        for layer in Self.simpleLayers(in: layers) {
            layer.step = decodeStep
        }
    }

    func synchronizing(_ prefill: PrefillParameters) -> PrefillParameters {
        guard isActive else { return prefill }
        let previous = prefill.progress
        var synced = prefill
        synced.progress = { processed, total in
            if processed < total {
                Stream.defaultStream.synchronize()
            }
            previous?(processed, total)
        }
        return synced
    }
}

extension GenerationLauncher {
    func prefillKVPlan(layers: [any KVCache], reusedTokens: Int = 0) -> PrefillKVPlan {
        PrefillKVPlan.make(
            promptTokens: input.text.tokens.dim(-1),
            reusedTokens: reusedTokens,
            simpleLayerCount: PrefillKVPlan.simpleLayers(in: layers).count,
            maxTokens: params.maxTokens,
            quantizesAfterPrefill: params.kvBits != nil || params.kvScheme != nil)
    }

    func parameters(for plan: PrefillKVPlan) -> GenerateParameters {
        guard plan.isActive else { return params }
        var synced = params
        synced.prefill = plan.synchronizing(params.prefill)
        return synced
    }
}
