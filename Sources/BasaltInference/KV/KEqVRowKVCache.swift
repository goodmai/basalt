import Foundation
import MLX
import MLXLMCommon
import MLXNN
import BasaltCore

final class KEqVRowKVCache: KVCacheAttentionProtocol {
    static let step = 256

    let layout: KEqVRowLayout
    private let rotated: MLXArray
    private let unrotated: MLXArray
    private let keyOrder: MLXArray
    private let outputRestore: MLXArray
    private let keyRestore: MLXArray
    private let valueUnfold: MLXArray
    private var rows: MLXArray?
    private(set) var materializedForCaller = false
    private(set) var offset = 0

    convenience init(layout: KEqVRowLayout, keyNormWeight: MLXArray) {
        let unrotated = MLXArray(layout.unrotated.map(Int32.init))
        let ones = MLXArray.ones([layout.rotated.count], dtype: .float32)
        let normW = keyNormWeight.take(unrotated, axis: 0).asType(.float32)
        let safeW = MLX.where(abs(normW) .> Float(1e-12), normW, MLXArray(Float(1.0)))
        let inverse = MLX.where(abs(normW) .> Float(1e-12), 1 / safeW, MLXArray(Float(0.0)))
        self.init(layout: layout, valueUnfold: concatenated([inverse, ones]))
    }

    init(layout: KEqVRowLayout, valueUnfold: MLXArray) {
        self.layout = layout
        let index = { (dims: [Int]) in MLXArray(dims.map(Int32.init)) }
        rotated = index(layout.rotated)
        unrotated = index(layout.unrotated)
        keyOrder = index(layout.keyOrder)
        outputRestore = index(layout.outputRestore)
        var inverseKey = [Int](repeating: 0, count: layout.headDim)
        for (position, dim) in layout.keyOrder.enumerated() { inverseKey[dim] = position }
        keyRestore = index(inverseKey)
        self.valueUnfold = valueUnfold
    }

    var requiresAttentionThroughCache: Bool { true }
    var maxSize: Int? { nil }
    var isTrimmable: Bool { true }

    func innerState() -> [MLXArray] { rows.map { [$0] } ?? [] }

    var metaState: [String] {
        get { [] }
        set {}
    }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard n > 1 else { return .none }
        if returnArray || (windowSize.map { n > $0 } ?? false) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    func emptied() -> KEqVRowKVCache {
        KEqVRowKVCache(layout: layout, valueUnfold: valueUnfold)
    }

    func copy() -> any KVCache {
        let copied = KEqVRowKVCache(layout: layout, valueUnfold: valueUnfold)
        copied.rows = rows
        copied.offset = offset
        copied.materializedForCaller = materializedForCaller
        return copied
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        let trimmed = min(offset, max(0, n))
        offset -= trimmed
        return trimmed
    }

    var nbytes: Int { rows?.nbytes ?? 0 }

    var state: [MLXArray] {
        get { rows.map { [$0[.ellipsis, ..<offset, 0...]] } ?? [] }
        set {
            rows = newValue.first
            offset = rows?.dim(2) ?? 0
        }
    }

    private func append(keys: MLXArray, values: MLXArray) -> MLXArray {
        let incoming = concatenated(
            [keys.take(rotated, axis: -1), keys.take(unrotated, axis: -1), values.take(rotated, axis: -1)], axis: -1)
        let (batch, heads, length) = (incoming.dim(0), incoming.dim(1), incoming.dim(2))
        let previous = offset
        if rows == nil || previous + length > rows!.dim(2) {
            let grow = ((length + Self.step - 1) / Self.step) * Self.step
            let fresh = MLXArray.zeros([batch, heads, grow, layout.rowWidth], dtype: incoming.dtype)
            if let current = rows {
                rows = concatenated([current[.ellipsis, ..<previous, 0...], fresh], axis: 2)
            } else {
                rows = fresh
            }
        }
        offset += length
        rows![.ellipsis, previous ..< offset, 0...] = incoming
        return rows![.ellipsis, ..<offset, 0...]
    }

    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let current = append(keys: keys, values: values)
        let out = MLXFast.scaledDotProductAttention(
            queries: queries.take(keyOrder, axis: -1),
            keys: current[.ellipsis, ..<layout.headDim],
            values: current[.ellipsis, (2 * layout.rotatedHalf)...],
            scale: scale, mask: mask)
        return (out.asType(.float32) * valueUnfold).asType(out.dtype).take(outputRestore, axis: -1)
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        materializedForCaller = true
        let current = append(keys: keys, values: values)
        let valueView = (current[.ellipsis, (2 * layout.rotatedHalf)...].asType(.float32) * valueUnfold).asType(current.dtype)
        return (current[.ellipsis, ..<layout.headDim].take(keyRestore, axis: -1), valueView.take(outputRestore, axis: -1))
    }

    static func replacingFullAttention(_ caches: [any KVCache], model: any LanguageModel, layout: KEqVRowLayout) -> [any KVCache] {
        let norms = keyNorms(model, headDim: layout.headDim)
        return caches.enumerated().map { index, cache in
            guard type(of: cache) == KVCacheSimple.self, let weight = norms[index] else { return cache }
            return KEqVRowKVCache(layout: layout, keyNormWeight: weight)
        }
    }

    static func keyNorms(_ model: any LanguageModel, headDim: Int) -> [Int: MLXArray] {
        var found: [Int: MLXArray] = [:]
        for (path, module) in model.leafModules().flattened() where path.hasSuffix(".self_attn.k_norm") {
            guard case .value(let weight)? = module.parameters()["weight"], weight.ndim == 1, weight.dim(0) == headDim,
                  let layer = path.split(separator: ".").reversed().dropFirst(2).first.flatMap({ Int($0) }) else { continue }
            found[layer] = weight
        }
        return found
    }

    static func modelAttendsThroughCache(_ model: any LanguageModel, layout: KEqVRowLayout) throws -> Bool {
        let caches = replacingFullAttention(try model.newCache(parameters: nil), model: model, layout: layout)
        let rows = caches.compactMap { $0 as? KEqVRowKVCache }
        guard rows.count == layout.globalLayers else { return false }
        _ = model(LMInput.Text(tokens: MLXArray([Int32(0)]).reshaped(1, 1)), cache: caches, state: nil)
        return rows.allSatisfy { !$0.materializedForCaller && $0.offset == 1 }
    }
}
