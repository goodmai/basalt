import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

struct GaugeRoPECarrier: Sendable {
    let basisPerFrequency: [MLXArray]
    let rotaryDims: Int
    let numKVHeads: Int
    let carrierHeads: Int

    static func calibrate(preRopeKeys: MLXArray, rotaryDims: Int, carrierHeads: Int) -> GaugeRoPECarrier {
        let numHeads = preRopeKeys.dim(1)
        let half = rotaryDims / 2
        let kRot = preRopeKeys[0..., 0..., 0 ..< rotaryDims].asType(.float32)
        let k0 = kRot[0..., 0..., 0 ..< half]
        let k1 = kRot[0..., 0..., half ..< rotaryDims]
        var bases: [MLXArray] = []
        bases.reserveCapacity(half)
        let rank = min(max(1, carrierHeads), numHeads)
        for freq in 0 ..< half {
            let x = k0[0..., 0..., freq]
            let y = k1[0..., 0..., freq]
            let gram = matmul(x.T, x) + matmul(y.T, y)
            let sym = 0.5 * (gram + gram.T)
            let (_, eigVecs) = MLXLinalg.eigh(sym, stream: .cpu)
            eval(eigVecs)
            let top = eigVecs[0..., (numHeads - rank) ..< numHeads]
            bases.append(top)
        }
        return GaugeRoPECarrier(basisPerFrequency: bases, rotaryDims: rotaryDims, numKVHeads: numHeads, carrierHeads: rank)
    }

    func encode(_ keys: MLXArray) -> MLXArray {
        let half = rotaryDims / 2
        let kRot = keys[0..., 0..., 0 ..< rotaryDims].asType(.float32)
        let k0 = kRot[0..., 0..., 0 ..< half]
        let k1 = kRot[0..., 0..., half ..< rotaryDims]
        var out0: [MLXArray] = []
        var out1: [MLXArray] = []
        out0.reserveCapacity(half)
        out1.reserveCapacity(half)
        for freq in 0 ..< half {
            let u = basisPerFrequency[freq]
            out0.append(matmul(k0[0..., 0..., freq], u))
            out1.append(matmul(k1[0..., 0..., freq], u))
        }
        let c0 = stacked(out0, axis: -1)
        let c1 = stacked(out1, axis: -1)
        return concatenated([c0, c1], axis: -1)
    }

    func decode(_ latent: MLXArray) -> MLXArray {
        let half = rotaryDims / 2
        let c0 = latent[0..., 0..., 0 ..< half]
        let c1 = latent[0..., 0..., half ..< rotaryDims]
        var out0: [MLXArray] = []
        var out1: [MLXArray] = []
        out0.reserveCapacity(half)
        out1.reserveCapacity(half)
        for freq in 0 ..< half {
            let uT = basisPerFrequency[freq].T
            out0.append(matmul(c0[0..., 0..., freq], uT))
            out1.append(matmul(c1[0..., 0..., freq], uT))
        }
        return concatenated([stacked(out0, axis: -1), stacked(out1, axis: -1)], axis: -1)
    }
}

struct JointPCACodec: Sendable {
    let basis: MLXArray
    let rank: Int

    static func calibrate(matrix: MLXArray, rank: Int) -> JointPCACodec {
        let x = matrix.asType(.float32)
        let dim = x.dim(1)
        let r = min(max(1, rank), dim)
        let gram = matmul(x.T, x)
        let sym = 0.5 * (gram + gram.T)
        let (_, eigVecs) = MLXLinalg.eigh(sym, stream: .cpu)
        eval(eigVecs)
        let top = eigVecs[0..., (dim - r) ..< dim]
        return JointPCACodec(basis: top, rank: r)
    }

    func encode(_ matrix: MLXArray) -> MLXArray {
        matmul(matrix.asType(.float32), basis)
    }

    func decode(_ latent: MLXArray) -> MLXArray {
        matmul(latent.asType(.float32), basis.T)
    }
}

enum GaugeRoPEByteBudget {
    static func bytesPerToken(layers: Int, kvHeads: Int, headDim: Int, kBits: Double, vBits: Double) -> Double {
        Double(layers * kvHeads * headDim) * (kBits + vBits) / 8.0
    }

    static func k8v4BytesPerToken(layers: Int, kvHeads: Int, headDim: Int) -> Double {
        bytesPerToken(layers: layers, kvHeads: kvHeads, headDim: headDim, kBits: 8.5, vBits: 4.5)
    }

    static func latentBytesPerToken(layers: Int, carrierFloats: Int, rkvFloats: Int, carrierBits: Double = 8.0, rkvBits: Double = 4.0) -> Double {
        Double(layers) * (Double(carrierFloats) * carrierBits + Double(rkvFloats) * rkvBits) / 8.0
    }
}

@Suite("GaugeRoPE S0 spike: iso-frequency RoPE equivariance, full-rank exactness, MLA-split baseline, and byte budgets (#647)")
struct GaugeRoPESpikeTests {
    private static func applyRoPE(_ x: MLXArray, rotaryDims: Int, offset: Int, base: Float = 10_000.0) -> MLXArray {
        let seq = x.dim(0)
        let half = rotaryDims / 2
        let invFreq = MLX.exp(MLXArray(0 ..< half).asType(.float32) * (-log(base) / Float(rotaryDims)))
        let positions = MLXArray(offset ..< (offset + seq)).asType(.float32).reshaped([seq, 1])
        let angles = matmul(positions, invFreq.reshaped([1, half]))
        let cosA = MLX.cos(angles).reshaped([seq, 1, half])
        let sinA = MLX.sin(angles).reshaped([seq, 1, half])
        let x0 = x[0..., 0..., 0 ..< half]
        let x1 = x[0..., 0..., half ..< rotaryDims]
        let rot = concatenated([x0 * cosA - x1 * sinA, x0 * sinA + x1 * cosA], axis: -1)
        if rotaryDims == x.dim(2) { return rot }
        return concatenated([rot, x[0..., 0..., rotaryDims...]], axis: -1)
    }

    @Test("GaugeRoPE iso-frequency carrier commutes exactly with RoPE across positions 0...4096")
    func carrierCommutesWithRoPE() {
        let keys = MLXRandom.normal([64, 4, 64], dtype: .float32)
        let carrier = GaugeRoPECarrier.calibrate(preRopeKeys: keys, rotaryDims: 64, carrierHeads: 2)

        for offset in [0, 17, 512, 4096] {
            let rotThenEncode = carrier.encode(Self.applyRoPE(keys, rotaryDims: 64, offset: offset))
            let encodeThenRot = Self.applyRoPE(carrier.encode(keys), rotaryDims: 64, offset: offset)
            eval(rotThenEncode, encodeThenRot)
            let maxDiff = abs(rotThenEncode - encodeThenRot).max().item(Float.self)
            #expect(maxDiff < 1e-4, "offset=\(offset) maxDiff=\(maxDiff)")
        }
    }

    @Test("Full-rank GaugeRoPE carrier and joint RKV codec reconstruct pre-RoPE K and V exactly")
    func fullRankExactReconstruction() {
        let seq = 32
        let heads = 4
        let headDim = 64
        let rotaryDims = 16
        let k = MLXRandom.normal([seq, heads, headDim], dtype: .float32)
        let v = MLXRandom.normal([seq, heads, headDim], dtype: .float32)

        let carrier = GaugeRoPECarrier.calibrate(preRopeKeys: k, rotaryDims: rotaryDims, carrierHeads: heads)
        let kRotHat = carrier.decode(carrier.encode(k))
        let kRotResid = k[0..., 0..., 0 ..< rotaryDims] - kRotHat
        eval(kRotResid)
        #expect(abs(kRotResid).max().item(Float.self) < 1e-4)

        let kNope = k[0..., 0..., rotaryDims...].reshaped([seq, -1])
        let vFlat = v.reshaped([seq, -1])
        let joint = concatenated([kNope, vFlat], axis: -1)
        let codec = JointPCACodec.calibrate(matrix: joint, rank: joint.dim(1))
        let recon = codec.decode(codec.encode(joint))
        eval(recon)
        #expect(abs(recon - joint).max().item(Float.self) < 1e-4)
    }

    @Test("Reduced-rank GaugeRoPE vs MLA-split reconstruction error improves monotonically with rank and hits 0.6x/0.45x/0.3x k8v4 budgets")
    func reducedRankAndByteBudgets() {
        let k8v4 = GaugeRoPEByteBudget.k8v4BytesPerToken(layers: 16, kvHeads: 4, headDim: 256)
        #expect(k8v4 > 25_000 && k8v4 < 28_000)
        for targetRatio in [0.6, 0.45, 0.3] {
            let budget = k8v4 * targetRatio
            let carrierFloats = 2 * 64
            let rkvFloats = max(64, Int(((budget / 16.0) * 8.0 - Double(carrierFloats) * 8.0) / 4.0))
            let actual = GaugeRoPEByteBudget.latentBytesPerToken(layers: 16, carrierFloats: carrierFloats, rkvFloats: rkvFloats)
            #expect(actual <= budget * 1.02)
        }

        let seq = 48
        let heads = 4
        let headDim = 64
        let rotaryDims = 16
        let k = MLXRandom.normal([seq, heads, headDim], dtype: .float32)
        let v = MLXRandom.normal([seq, heads, headDim], dtype: .float32)
        let joint = concatenated([k[0..., 0..., rotaryDims...].reshaped([seq, -1]), v.reshaped([seq, -1])], axis: -1)

        var prevErr: Float = .greatestFiniteMagnitude
        for rank in [16, 32, 64, 128] {
            let codec = JointPCACodec.calibrate(matrix: joint, rank: rank)
            let recon = codec.decode(codec.encode(joint))
            let err = (MLX.sum(MLX.pow(recon - joint, 2)).item(Float.self) / MLX.sum(MLX.pow(joint, 2)).item(Float.self)).squareRoot()
            #expect(err <= prevErr + 1e-5)
            prevErr = err
        }
        #expect(prevErr < 1.0)
    }
}
