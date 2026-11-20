import BasaltCore
import Foundation
import MLX
import MLXLMCommon

final class KVarNKVCache: KVCacheAttentionProtocol {
    static let tileTokens = 64

    struct HeadTile {
        let key: KVarNEncodedTile
        let value: KVarNEncodedTile
    }

    let useQJL: Bool
    private var tailKeys: MLXArray?
    private var tailValues: MLXArray?
    private var tilesByHead: [[HeadTile]] = []
    private var jlMatrix: MLXArray?
    private var encodedTokenCount = 0
    private(set) var compressedSteps = 0
    private(set) var materializedForCaller = false

    init(useQJL: Bool) {
        self.useQJL = useQJL
    }

    static func replacingFullAttention(_ caches: [any KVCache], useQJL: Bool) -> [any KVCache] {
        caches.map { type(of: $0) == KVCacheSimple.self ? KVarNKVCache(useQJL: useQJL) : $0 }
    }

    static func modelAttendsThroughCache(_ model: any LanguageModel, useQJL: Bool = false) throws -> Bool {
        let caches = replacingFullAttention(try model.newCache(parameters: nil), useQJL: useQJL)
        let probes = caches.compactMap { $0 as? KVarNKVCache }
        guard !probes.isEmpty else { return false }
        _ = model(LMInput.Text(tokens: MLXArray([Int32(0)]).reshaped(1, 1)), cache: caches, state: nil)
        return probes.allSatisfy { !$0.materializedForCaller }
    }

    var offset: Int { encodedTokenCount + (tailKeys?.dim(2) ?? 0) }
    var maxSize: Int? { nil }
    var isTrimmable: Bool { false }

    func innerState() -> [MLXArray] {
        var arrays: [MLXArray] = []
        if let tailKeys { arrays.append(tailKeys) }
        if let tailValues { arrays.append(tailValues) }
        for headTiles in tilesByHead {
            for tile in headTiles {
                arrays.append(tile.key.packedCodes)
                arrays.append(tile.key.tokenScale)
                arrays.append(tile.key.channelScale)
                if let qb = tile.key.qjlBits { arrays.append(qb) }
                if let qn = tile.key.qjlNorm { arrays.append(qn) }
                arrays.append(tile.value.packedCodes)
                arrays.append(tile.value.tokenScale)
                arrays.append(tile.value.channelScale)
            }
        }
        return arrays
    }

    var state: [MLXArray] {
        get { innerState() }
        set { preconditionFailure("KVarNKVCache is never restored from a serialized prompt cache") }
    }

    var metaState: [String] {
        get { [useQJL ? "kvarn-qjl" : "kvarn", String(offset)] }
        set { preconditionFailure("KVarNKVCache is never restored from a serialized prompt cache") }
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        materializedForCaller = true
        appendTail(keys: keys, values: values)
        return (tailKeys ?? keys, tailValues ?? values)
    }

    @discardableResult
    func trim(_ n: Int) -> Int { 0 }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        n == 1 ? .none : .causal
    }

    func copy() -> any KVCache {
        let copied = KVarNKVCache(useQJL: useQJL)
        copied.tailKeys = tailKeys
        copied.tailValues = tailValues
        copied.tilesByHead = tilesByHead
        copied.jlMatrix = jlMatrix
        copied.encodedTokenCount = encodedTokenCount
        copied.compressedSteps = compressedSteps
        copied.materializedForCaller = materializedForCaller
        return copied
    }

    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        appendTail(keys: keys, values: values)
        guard let tk = tailKeys, let tv = tailValues else {
            return MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        }
        let d = queries.dim(-1)
        let fits = queries.dim(0) == 1 && d.isMultiple(of: 32) && d <= KVarNFlashDecodeMetal.maxHeadDim
            && KVarNFlashDecodeMetal.supportedDTypes.contains(queries.dtype)
            && queries.dim(1).isMultiple(of: max(1, tk.dim(1)))
        if !fits {
            return MLXFast.scaledDotProductAttention(queries: queries, keys: tk, values: tv, scale: scale, mask: mask)
        }
        if queries.dim(2) > 1 {
            if tilesByHead.isEmpty {
                return MLXFast.scaledDotProductAttention(queries: queries, keys: tk, values: tv, scale: scale, mask: mask)
            }
            compressedSteps += 1
            return decodeCompressedMultiQuery(queries: queries, scale: scale)
        }
        compressFullTiles()
        guard let firstHead = tilesByHead.first, !firstHead.isEmpty else {
            return MLXFast.scaledDotProductAttention(queries: queries, keys: tailKeys ?? tk, values: tailValues ?? tv, scale: scale, mask: mask)
        }
        compressedSteps += 1
        return decodeCompressed(queries: queries, scale: scale)
    }

    private func decodeCompressedMultiQuery(queries: MLXArray, scale: Float) -> MLXArray {
        let lQ = queries.dim(2)
        let tailLen = tailKeys?.dim(2) ?? 0
        var slices: [MLXArray] = []
        slices.reserveCapacity(lQ)
        for i in 0 ..< lQ {
            let validTail = max(0, tailLen - (lQ - 1 - i))
            let qStep = queries[0..., 0..., i ..< (i + 1), 0...]
            slices.append(decodeCompressed(queries: qStep, scale: scale, tailLimit: validTail))
        }
        return concatenated(slices, axis: 2)
    }

    private func appendTail(keys: MLXArray, values: MLXArray) {
        if let tk = tailKeys, let tv = tailValues {
            tailKeys = concatenated([tk, keys], axis: 2)
            tailValues = concatenated([tv, values], axis: 2)
        } else {
            tailKeys = keys
            tailValues = values
        }
    }

    private func compressFullTiles() {
        guard let tk = tailKeys, let tv = tailValues else { return }
        let total = tk.dim(2)
        let complete = total / Self.tileTokens
        guard complete > 0 else { return }
        let kvHeads = tk.dim(1)
        let d = tk.dim(3)
        if tilesByHead.isEmpty { tilesByHead = Array(repeating: [], count: kvHeads) }
        if useQJL, jlMatrix == nil { jlMatrix = KVarNTileCodec.makeJLMatrix(dim: d) }
        let consumed = complete * Self.tileTokens
        for tIdx in 0 ..< complete {
            let start = tIdx * Self.tileTokens
            let end = start + Self.tileTokens
            for h in 0 ..< kvHeads {
                let kSlice = tk[0, h, start ..< end, 0...]
                let vSlice = tv[0, h, start ..< end, 0...]
                let kTile = KVarNTileCodec.encodeTile(kSlice, useQJL: useQJL, jlMatrix: jlMatrix)
                let vTile = KVarNTileCodec.encodeTile(vSlice, useQJL: false)
                tilesByHead[h].append(HeadTile(key: kTile, value: vTile))
            }
        }
        encodedTokenCount += consumed
        if total > consumed {
            tailKeys = tk[0..., 0..., consumed..., 0...]
            tailValues = tv[0..., 0..., consumed..., 0...]
        } else {
            tailKeys = nil
            tailValues = nil
        }
    }

    private func decodeCompressed(queries: MLXArray, scale: Float, tailLimit: Int? = nil) -> MLXArray {
        let origType = queries.dtype
        let qHeads = queries.dim(1)
        let d = queries.dim(3)
        let kvHeads = tilesByHead.count
        let group = qHeads / kvHeads
        var headOutputs: [MLXArray] = []
        headOutputs.reserveCapacity(kvHeads)
        for h in 0 ..< kvHeads {
            let qGroup = queries[0, (h * group) ..< ((h + 1) * group), 0, 0...]
            var outs: [MLXArray] = []
            var lses: [MLXArray] = []
            for tile in tilesByHead[h] {
                let (o, lse) = KVarNFlashDecodeMetal.flashDecodeWithLSE(
                    KVarNFlashDecodeMetal.Inputs(
                        queries: qGroup, keyTile: tile.key, valueTile: tile.value,
                        jlMatrix: jlMatrix, scale: scale))
                outs.append(o.asType(.float32))
                lses.append(lse)
            }
            if let tk = tailKeys, let tv = tailValues {
                let useTail = tailLimit ?? tk.dim(2)
                if useTail > 0 {
                    let (oTail, lseTail) = Self.exactTailAttention(
                        qGroup: qGroup.asType(.float32),
                        kTail: tk[0, h, 0 ..< useTail, 0...].asType(.float32),
                        vTail: tv[0, h, 0 ..< useTail, 0...].asType(.float32), scale: scale)
                    outs.append(oTail)
                    lses.append(lseTail)
                }
            }
            let merged = Self.mergePartitions(outputs: outs, lses: lses).asType(origType)
            headOutputs.append(merged.reshaped(1, group, 1, d))
        }
        return concatenated(headOutputs, axis: 1)
    }
}

extension KVarNKVCache {
    private static func exactTailAttention(
        qGroup: MLXArray, kTail: MLXArray, vTail: MLXArray, scale: Float
    ) -> (MLXArray, MLXArray) {
        let scores = matmul(qGroup, kTail.T) * scale
        let m = scores.max(axis: -1, keepDims: true)
        let exps = exp(scores - m)
        let denom = sum(exps, axis: -1, keepDims: true)
        let lse = (m + log(maximum(denom, Float(1e-30)))).squeezed(axis: -1)
        let out = matmul(exps / maximum(denom, Float(1e-30)), vTail)
        return (out, lse)
    }

    private static func mergePartitions(outputs: [MLXArray], lses: [MLXArray]) -> MLXArray {
        guard outputs.count > 1 else { return outputs[0] }
        let stackedLSE = stacked(lses, axis: 0)
        let maxLSE = stackedLSE.max(axis: 0, keepDims: true)
        let weights = exp(stackedLSE - maxLSE)
        let normWeights = weights / maximum(sum(weights, axis: 0, keepDims: true), Float(1e-30))
        let stackedOut = stacked(outputs, axis: 0)
        return sum(stackedOut * normWeights.expandedDimensions(axis: -1), axis: 0)
    }
}
