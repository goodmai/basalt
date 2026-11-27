import Foundation
import MLX
import MLXLMCommon

final class GaugeRoPEKVCache: KVCache {
    let layout: GaugeRoPELayout
    private(set) var codec: GaugeRoPECodec?
    private var exactKeys: MLXArray?
    private var exactValues: MLXArray?
    private var codes: GaugeRoPECodes?

    init(layout: GaugeRoPELayout, codec: GaugeRoPECodec? = nil) {
        self.layout = layout
        self.codec = codec
    }

    static func replacingFullAttention(
        _ caches: [any KVCache], layout: GaugeRoPELayout, codecs: [Int: GaugeRoPECodec] = [:]
    ) -> [any KVCache] {
        caches.enumerated().map { index, cache in
            type(of: cache) == KVCacheSimple.self ? GaugeRoPEKVCache(layout: layout, codec: codecs[index]) : cache
        }
    }

    var encodedTokens: Int { codes?.joint.dim(0) ?? 0 }
    var offset: Int { encodedTokens + (exactKeys?.dim(2) ?? 0) }
    var maxSize: Int? { nil }
    var isTrimmable: Bool { false }

    func innerState() -> [MLXArray] {
        [exactKeys, exactValues, codes?.carrier, codes?.joint].compactMap { $0 }
    }

    var state: [MLXArray] {
        get { innerState() }
        set { preconditionFailure("GaugeRoPEKVCache is never restored from a serialized prompt cache") }
    }

    var metaState: [String] {
        get { ["gaugerope", String(offset)] }
        set { preconditionFailure("GaugeRoPEKVCache is never restored from a serialized prompt cache") }
    }

    @discardableResult
    func trim(_ n: Int) -> Int { 0 }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard n > 1 else { return .none }
        if returnArray || (windowSize.map { n > $0 } ?? false) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    func copy() -> any KVCache {
        let copied = GaugeRoPEKVCache(layout: layout, codec: codec)
        copied.exactKeys = exactKeys
        copied.exactValues = exactValues
        copied.codes = codes
        return copied
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if let codec { return appending(keys: keys, values: values, codec: codec) }
        let pendingKeys = exactKeys.map { concatenated([$0, keys], axis: 2) } ?? keys
        let pendingValues = exactValues.map { concatenated([$0, values], axis: 2) } ?? values
        let n = layout.calibrationTokens
        exactKeys = pendingKeys
        exactValues = pendingValues
        guard pendingKeys.dim(2) >= n else { return (pendingKeys, pendingValues) }
        let calibrated = GaugeRoPECodec.calibrate(
            keys: Self.tokenMajor(pendingKeys[0..., 0..., 0 ..< n]), values: Self.tokenMajor(pendingValues[0..., 0..., 0 ..< n]),
            layout: layout)
        exactKeys = nil
        exactValues = nil
        codec = calibrated
        return appending(keys: pendingKeys, values: pendingValues, codec: calibrated)
    }

    private func appending(keys: MLXArray, values: MLXArray, codec: GaugeRoPECodec) -> (MLXArray, MLXArray) {
        let fresh = codec.encode(keys: Self.tokenMajor(keys), values: Self.tokenMajor(values), offset: encodedTokens)
        let carrier = fresh.carrier.asType(keys.dtype)
        let joint = fresh.joint.asType(keys.dtype)
        let all = codes.map {
            GaugeRoPECodes(carrier: concatenated([$0.carrier, carrier], axis: 0), joint: concatenated([$0.joint, joint], axis: 0))
        } ?? GaugeRoPECodes(carrier: carrier, joint: joint)
        codes = all
        let decoded = codec.decode(all)
        return (Self.headMajor(decoded.keys, like: keys), Self.headMajor(decoded.values, like: values))
    }

    static func tokenMajor(_ x: MLXArray) -> MLXArray {
        x[0].transposed(1, 0, 2)
    }

    static func headMajor(_ x: MLXArray, like reference: MLXArray) -> MLXArray {
        x.transposed(1, 0, 2).expandedDimensions(axis: 0).asType(reference.dtype)
    }
}
