import BasaltCore
import Foundation
import MLX

struct WhitenedPCACodec {
    let encoder: MLXArray
    let decoder: MLXArray

    var maxRank: Int { encoder.dim(1) }

    static func fit(_ matrix: MLXArray, metric: MLXArray? = nil) -> WhitenedPCACodec {
        let x = matrix.asType(.float32)
        let dim = x.dim(1)
        let pair = metric.map { InputWhitenedSVD.computeWhiteningPair(hessian: $0) }
        let s = pair?.s ?? MLXArray.eye(dim, dtype: .float32)
        let sInv = pair?.sInv ?? MLXArray.eye(dim, dtype: .float32)
        let (_, _, vt) = MLXLinalg.svd(matmul(x, s), stream: .cpu)
        eval(vt)
        return WhitenedPCACodec(encoder: matmul(s, vt.T), decoder: matmul(vt, sInv))
    }
}

struct GaugeRoPEKV {
    let keys: MLXArray
    let values: MLXArray
}

struct GaugeRoPECodes {
    let carrier: MLXArray
    let joint: MLXArray
}

struct GaugeRoPECodec {
    let carrier: GaugeRoPECarrier
    let projection: MLXArray
    let ropeBase: Float
    let keyDim: Int
    let valueDim: Int

    var codeRank: Int { projection.dim(1) }
    var elementsPerToken: Int { carrier.carrierHeads * carrier.rotaryDims + codeRank }

    static func calibrate(keys: MLXArray, values: MLXArray, layout: GaugeRoPELayout) -> GaugeRoPECodec {
        let carrier = GaugeRoPECarrier.calibrate(keys: keys, rotaryDims: layout.rotaryDims, carrierHeads: layout.carrierRank)
        let matrix = jointMatrix(keys: keys, values: values, carrier: carrier, base: layout.ropeBase, offset: 0).matrix
        let pca = WhitenedPCACodec.fit(matrix)
        let budgeted = defaultCodeRank(
            heads: keys.dim(1), keyDim: keys.dim(2), valueDim: values.dim(2), carrierElements: carrier.carrierHeads * layout.rotaryDims)
        let rank = min(max(1, layout.codeRank ?? budgeted), pca.maxRank)
        let projection = contiguous(pca.encoder[0..., 0 ..< rank])
        eval(projection)
        return GaugeRoPECodec(
            carrier: carrier, projection: projection, ropeBase: layout.ropeBase, keyDim: keys.dim(2), valueDim: values.dim(2))
    }

    static func defaultCodeRank(heads: Int, keyDim: Int, valueDim: Int, carrierElements: Int) -> Int {
        let k8v4 = Double(heads) * (Double(keyDim) * KVCacheGeometry.bytesPerElement(kvBits: 8)
            + Double(valueDim) * KVCacheGeometry.bytesPerElement(kvBits: 4))
        let budget = KVPrecision.gaugeRoPEBudgetVsK8v4 * k8v4 / KVCacheGeometry.activationBytes
        return max(1, Int((budget - Double(carrierElements)).rounded(.down)))
    }

    static func jointMatrix(
        keys: MLXArray, values: MLXArray, carrier: GaugeRoPECarrier, base: Float, offset: Int
    ) -> (code: MLXArray, matrix: MLXArray) {
        let (n, heads, r) = (keys.dim(0), keys.dim(1), carrier.rotaryDims)
        let code = carrier.encode(keys)
        let residual = GaugeRoPERotary.apply(
            keys[0..., 0..., 0 ..< r].asType(.float32) - carrier.decode(code), rotaryDims: r, base: base, offset: offset, inverse: true)
        let matrix = concatenated(
            [
                residual.reshaped([n, heads * r]),
                keys[0..., 0..., r...].reshaped([n, heads * (keys.dim(2) - r)]).asType(.float32),
                values.reshaped([n, heads * values.dim(2)]).asType(.float32),
            ], axis: -1)
        return (code, matrix)
    }

    func encode(keys: MLXArray, values: MLXArray, offset: Int) -> GaugeRoPECodes {
        let parts = Self.jointMatrix(keys: keys, values: values, carrier: carrier, base: ropeBase, offset: offset)
        return GaugeRoPECodes(carrier: parts.code, joint: matmul(parts.matrix, projection))
    }

    func decode(_ codes: GaugeRoPECodes) -> GaugeRoPEKV {
        let n = codes.joint.dim(0)
        let (heads, r) = (carrier.numKVHeads, carrier.rotaryDims)
        let nope = keyDim - r
        let hat = matmul(codes.joint.asType(.float32), projection.T)
        let residual = GaugeRoPERotary.apply(hat[0..., 0 ..< (heads * r)].reshaped([n, heads, r]), rotaryDims: r, base: ropeBase)
        let rotated = carrier.decode(codes.carrier.asType(.float32)) + residual
        let values = hat[0..., (heads * keyDim)...].reshaped([n, heads, valueDim])
        guard nope > 0 else { return GaugeRoPEKV(keys: rotated, values: values) }
        let kNope = hat[0..., (heads * r) ..< (heads * keyDim)].reshaped([n, heads, nope])
        return GaugeRoPEKV(keys: concatenated([rotated, kNope], axis: -1), values: values)
    }
}
