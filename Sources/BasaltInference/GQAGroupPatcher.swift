import Foundation
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

public enum GQAGroupPatcher {
    static let attentionOutputKeys = ["self_attn.o_proj", "self_attn.out_proj", "operator.out_proj"]

    struct SearchInput {
        let layers: [Int]
        let best: Int
        let direction: MLXArray
        let gap: Float
        let search: ContrastSearch
    }

    struct ProjectionTarget {
        let key: String
        let linear: Linear
        let owner: Module
    }

    public static func queryHeads(kvGroup: Int, heads: Int, kvHeads: Int) throws(BasaltError) -> Range<Int> {
        guard heads > 0, kvHeads > 0, heads.isMultiple(of: kvHeads) else {
            throw .invalidRequestStructure(details: "attention heads (\(heads)) must be positive and divisible by kvHeads (\(kvHeads))")
        }
        guard (0 ..< kvHeads).contains(kvGroup) else {
            throw .invalidRequestStructure(details: "kvGroup \(kvGroup) must be in 0..<\(kvHeads)")
        }
        let groupSize = heads / kvHeads
        return (kvGroup * groupSize) ..< ((kvGroup + 1) * groupSize)
    }

    public static func importance(baseline: Float, patched: Float, epsilon: Float = 1e-6) -> Float {
        let denom = max(Double(abs(baseline)) + Double(max(epsilon, 1e-12)), 1e-12)
        return Float((Double(patched) - Double(baseline)) / denom)
    }

    public static func recommendMode(
        heads: Int,
        kvHeads: Int,
        kurtosis: Float,
        kurtosisThreshold: Float = 50
    ) -> HeadInterventionMode {
        if kvHeads > 0, heads > kvHeads, kurtosis < kurtosisThreshold {
            return .gqaGroup
        }
        return .singleHead
    }

    static func rankGroups(_ input: SearchInput) throws -> [GQAGroupEffect] {
        var effects: [GQAGroupEffect] = []
        for layer in input.layers {
            let heads = input.search.anatomy.layers[layer].heads ?? 0
            let kvHeads = input.search.anatomy.layers[layer].kvHeads ?? heads
            guard heads > 0, kvHeads > 0, heads.isMultiple(of: kvHeads) else { continue }
            effects.append(contentsOf: try layerGroupEffects(layer: layer, heads: heads, kvHeads: kvHeads, input: input))
        }
        return effects.sorted {
            (abs($0.importance), abs($0.effect), -$0.layer, -$0.kvGroup)
                > (abs($1.importance), abs($1.effect), -$1.layer, -$1.kvGroup)
        }
    }

    static func attentionOutputTarget(in owner: Module) -> ProjectionTarget? {
        let modules = Dictionary(owner.namedModules(), uniquingKeysWith: { first, _ in first })
        return attentionOutputKeys.lazy.compactMap { key in
            (modules[key] as? Linear).map { ProjectionTarget(key: key, linear: $0, owner: owner) }
        }.first
    }

    private static func layerGroupEffects(
        layer: Int,
        heads: Int,
        kvHeads: Int,
        input: SearchInput
    ) throws -> [GQAGroupEffect] {
        if let found = MLXInferenceEngine.decoderLayer(layer, in: input.search.context.model),
           let target = attentionOutputTarget(in: found.layer),
           target.linear.shape.1.isMultiple(of: kvHeads) {
            return try patchedProjectionEffects(layer: layer, heads: heads, kvHeads: kvHeads, target: target, input: input)
        }
        return try maskedGroupEffects(layer: layer, heads: heads, kvHeads: kvHeads, input: input)
    }

    private static func patchedProjectionEffects(
        layer: Int,
        heads: Int,
        kvHeads: Int,
        target: ProjectionTarget,
        input: SearchInput
    ) throws -> [GQAGroupEffect] {
        let allPrompts = input.search.positive + input.search.negative
        let meanActivation = try meanAttentionInput(target: target, prompts: allPrompts, context: input.search.context)
        let groupWidth = target.linear.shape.1 / kvHeads
        var results: [GQAGroupEffect] = []
        for kvGroup in 0 ..< kvHeads {
            let qRange = try queryHeads(kvGroup: kvGroup, heads: heads, kvHeads: kvHeads)
            let sliceRange = (kvGroup * groupWidth) ..< ((kvGroup + 1) * groupWidth)
            let replacement = meanActivation[sliceRange]
            let patchedGap = try withGroupPatch(target: target, range: sliceRange, replacement: replacement, context: input.search.context) {
                try gapAtBest(input: input, edits: [])
            }
            results.append(GQAGroupEffect(
                layer: layer,
                kvGroup: kvGroup,
                queryHeads: Array(qRange),
                effect: input.gap - patchedGap,
                importance: importance(baseline: input.gap, patched: patchedGap)
            ))
        }
        return results
    }

    private static func maskedGroupEffects(
        layer: Int,
        heads: Int,
        kvHeads: Int,
        input: SearchInput
    ) throws -> [GQAGroupEffect] {
        var results: [GQAGroupEffect] = []
        for kvGroup in 0 ..< kvHeads {
            let qRange = try queryHeads(kvGroup: kvGroup, heads: heads, kvHeads: kvHeads)
            let edits = qRange.map { Intervention.ablateHead(layer: layer, head: $0) }
            let patchedGap = try gapAtBest(input: input, edits: edits)
            results.append(GQAGroupEffect(
                layer: layer,
                kvGroup: kvGroup,
                queryHeads: Array(qRange),
                effect: input.gap - patchedGap,
                importance: importance(baseline: input.gap, patched: patchedGap)
            ))
        }
        return results
    }

    private static func meanAttentionInput(
        target: ProjectionTarget,
        prompts: [[Int]],
        context: ModelContext
    ) throws -> MLXArray {
        var rows: [MLXArray] = []
        target.owner.update(modules: .unflattened([(target.key, MeanSequenceInputTap(target.linear, record: { rows.append($0) }) as Module)]))
        context.model.invalidateCompiledTraces()
        defer {
            target.owner.update(modules: .unflattened([(target.key, target.linear as Module)]))
            context.model.invalidateCompiledTraces()
        }
        for ids in prompts {
            _ = MLXInferenceEngine.forward(context, probe: nil, ids, cache: try context.model.newCache(parameters: nil))
            eval(rows)
        }
        guard !rows.isEmpty else {
            return MLXArray.zeros([target.linear.shape.1], dtype: .float32)
        }
        let mean = stacked(rows).mean(axis: 0)
        eval(mean)
        return mean
    }

    private static func withGroupPatch<R>(
        target: ProjectionTarget,
        range: Range<Int>,
        replacement: MLXArray,
        context: ModelContext,
        body: () throws -> R
    ) throws -> R {
        let tap = GroupSlicePatchTap(target.linear, range: range, replacement: replacement)
        target.owner.update(modules: .unflattened([(target.key, tap as Module)]))
        context.model.invalidateCompiledTraces()
        defer {
            target.owner.update(modules: .unflattened([(target.key, target.linear as Module)]))
            context.model.invalidateCompiledTraces()
        }
        return try body()
    }

    private static func gapAtBest(input: SearchInput, edits: [Intervention]) throws -> Float {
        let probeBuilder = { edits.isEmpty ? nil : InterventionProbe(edits, anatomy: input.search.anatomy) }
        let posStates = try input.search.positive.map { try captureLayerState($0, layer: input.best, probe: probeBuilder(), context: input.search.context) }
        let negStates = try input.search.negative.map { try captureLayerState($0, layer: input.best, probe: probeBuilder(), context: input.search.context) }
        let diff = stacked(posStates).mean(axis: 0) - stacked(negStates).mean(axis: 0)
        let projected = (diff * input.direction).sum()
        eval(projected)
        return projected.item(Float.self)
    }

    private static func captureLayerState(
        _ ids: [Int],
        layer: Int,
        probe: InterventionProbe?,
        context: ModelContext
    ) throws -> MLXArray {
        let contrastProbe = ContrastProbe(probe)
        _ = MLXInferenceEngine.forward(context, probe: contrastProbe, ids, cache: try context.model.newCache(parameters: nil))
        guard let state = contrastProbe.states[layer] else {
            throw BasaltError.invalidRequestStructure(details: "layer \(layer) was not reported to the forward probe")
        }
        eval(state)
        return state
    }
}

private final class MeanSequenceInputTap: Linear {
    private let original: (MLXArray) -> MLXArray
    private let record: (MLXArray) -> Void

    init(_ linear: Linear, record: @escaping (MLXArray) -> Void) {
        original = { linear($0) }
        self.record = record
        super.init(weight: linear.weight, bias: linear.bias)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        record(x[0].asType(.float32).mean(axis: 0))
        return original(x)
    }
}

private final class GroupSlicePatchTap: Linear {
    private let original: (MLXArray) -> MLXArray
    private let range: Range<Int>
    private let replacement: MLXArray

    init(_ linear: Linear, range: Range<Int>, replacement: MLXArray) {
        original = { linear($0) }
        self.range = range
        self.replacement = replacement
        super.init(weight: linear.weight, bias: linear.bias)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let total = x.dim(-1)
        guard x.ndim == 3, range.lowerBound >= 0, range.upperBound <= total else { return original(x) }
        let patch = broadcast(replacement.asType(x.dtype).reshaped(1, 1, range.count), to: [x.dim(0), x.dim(1), range.count])
        var parts: [MLXArray] = []
        if range.lowerBound > 0 { parts.append(x[0..., 0..., 0 ..< range.lowerBound]) }
        parts.append(patch)
        if range.upperBound < total { parts.append(x[0..., 0..., range.upperBound ..< total]) }
        return original(concatenated(parts, axis: 2))
    }
}
