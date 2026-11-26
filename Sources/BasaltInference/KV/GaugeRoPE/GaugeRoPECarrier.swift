import Foundation
import MLX

struct GaugeRoPECarrier {
    let basis: MLXArray
    let rotaryDims: Int
    let numKVHeads: Int
    let carrierHeads: Int

    static func calibrate(keys: MLXArray, rotaryDims: Int, carrierHeads: Int) -> GaugeRoPECarrier {
        let numHeads = keys.dim(1)
        let pairs = Self.pairs(keys, rotaryDims: rotaryDims)
        let gram = matmul(pairs.transposed(0, 1, 3, 2), pairs).sum(axis: 0)
        let sym = 0.5 * (gram + gram.transposed(0, 2, 1))
        let (_, eigVecs) = MLXLinalg.eigh(sym, stream: .cpu)
        let rank = min(max(1, carrierHeads), numHeads)
        let basis = eigVecs[0..., 0..., (numHeads - rank) ..< numHeads]
        eval(basis)
        return GaugeRoPECarrier(basis: basis, rotaryDims: rotaryDims, numKVHeads: numHeads, carrierHeads: rank)
    }

    static func pairs(_ x: MLXArray, rotaryDims: Int) -> MLXArray {
        let (n, rows) = (x.dim(0), x.dim(1))
        return x[0..., 0..., 0 ..< rotaryDims].asType(.float32)
            .reshaped([n, rows, 2, rotaryDims / 2]).transposed(2, 3, 0, 1)
    }

    static func unpairs(_ pairs: MLXArray, rotaryDims: Int) -> MLXArray {
        pairs.transposed(2, 3, 0, 1).reshaped([pairs.dim(2), pairs.dim(3), rotaryDims])
    }

    func encode(_ keys: MLXArray) -> MLXArray {
        Self.unpairs(matmul(Self.pairs(keys, rotaryDims: rotaryDims), basis), rotaryDims: rotaryDims)
    }

    func decode(_ latent: MLXArray) -> MLXArray {
        Self.unpairs(matmul(Self.pairs(latent, rotaryDims: rotaryDims), basis.transposed(0, 2, 1)), rotaryDims: rotaryDims)
    }
}

enum GaugeRoPERotary {
    static func apply(_ x: MLXArray, rotaryDims: Int, base: Float, offset: Int = 0, inverse: Bool = false) -> MLXArray {
        let seq = x.dim(0)
        let half = rotaryDims / 2
        let invFreq = MLX.exp(MLXArray(0 ..< half).asType(.float32) * (-2.0 * log(base) / Float(rotaryDims)))
        let positions = MLXArray(offset ..< (offset + seq)).asType(.float32).reshaped([seq, 1])
        let angles = matmul(positions, invFreq.reshaped([1, half])) * Float(inverse ? -1 : 1)
        let cosA = MLX.cos(angles).reshaped([seq, 1, half])
        let sinA = MLX.sin(angles).reshaped([seq, 1, half])
        let x0 = x[0..., 0..., 0 ..< half]
        let x1 = x[0..., 0..., half ..< rotaryDims]
        let rot = concatenated([x0 * cosA - x1 * sinA, x0 * sinA + x1 * cosA], axis: -1)
        if rotaryDims == x.dim(2) { return rot }
        return concatenated([rot, x[0..., 0..., rotaryDims...]], axis: -1)
    }
}
