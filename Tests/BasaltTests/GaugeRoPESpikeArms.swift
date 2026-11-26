import Foundation
import MLX
import MLXLMCommon
@testable import BasaltInference

extension WhitenedPCACodec {
    func roundTrip(_ matrix: MLXArray, rank: Int, codeBits: Int?) -> MLXArray {
        let r = min(max(1, rank), maxRank)
        let code = matmul(matrix.asType(.float32), encoder[0..., 0 ..< r])
        return matmul(GaugeRoPEArms.fakeQuantized(code, bits: codeBits), decoder[0 ..< r, 0...])
    }
}

struct MLASplitArm {
    let codec: WhitenedPCACodec
    let rotaryDims: Int

    static func joint(keys: MLXArray, values: MLXArray, rotaryDims: Int) -> MLXArray {
        let n = keys.dim(0)
        return concatenated(
            [keys[0..., 0..., rotaryDims...].reshaped([n, -1]), values.reshaped([n, -1])], axis: -1
        ).asType(.float32)
    }

    static func fit(keys: MLXArray, values: MLXArray, rotaryDims: Int, metric: MLXArray? = nil) -> MLASplitArm {
        let matrix = Self.joint(keys: keys, values: values, rotaryDims: rotaryDims)
        return MLASplitArm(codec: WhitenedPCACodec.fit(matrix, metric: metric), rotaryDims: rotaryDims)
    }

    func reconstruct(keys: MLXArray, values: MLXArray, rank: Int, quantized: Bool) -> GaugeRoPEKV {
        let (n, heads, nope) = (keys.dim(0), keys.dim(1), keys.dim(2) - rotaryDims)
        let matrix = Self.joint(keys: keys, values: values, rotaryDims: rotaryDims)
        let hat = codec.roundTrip(matrix, rank: rank, codeBits: quantized ? 4 : nil)
        let kRot = GaugeRoPEArms.fakeQuantized(keys[0..., 0..., 0 ..< rotaryDims], bits: quantized ? 8 : nil)
        let kNope = hat[0..., 0 ..< (heads * nope)].reshaped([n, heads, nope])
        let vHat = hat[0..., (heads * nope)...].reshaped([n, heads, values.dim(2)])
        return GaugeRoPEKV(keys: concatenated([kRot, kNope], axis: -1), values: vHat)
    }
}

struct GaugeRoPEArm {
    let carrier: GaugeRoPECarrier
    let codec: WhitenedPCACodec

    static func joint(keys: MLXArray, values: MLXArray, carrier: GaugeRoPECarrier, quantized: Bool) -> (matrix: MLXArray, carrier: MLXArray) {
        let (n, r) = (keys.dim(0), carrier.rotaryDims)
        let carrierHat = carrier.decode(GaugeRoPEArms.fakeQuantized(carrier.encode(keys), bits: quantized ? 8 : nil))
        let residual = keys[0..., 0..., 0 ..< r].asType(.float32) - carrierHat
        let matrix = concatenated(
            [
                residual.reshaped([n, -1]),
                keys[0..., 0..., r...].reshaped([n, -1]).asType(.float32),
                values.reshaped([n, -1]).asType(.float32),
            ], axis: -1)
        return (matrix, carrierHat)
    }

    static func fit(
        keys: MLXArray, values: MLXArray, rotaryDims: Int, carrierRank: Int, metric: MLXArray? = nil, quantized: Bool = false
    ) -> GaugeRoPEArm {
        let carrier = GaugeRoPECarrier.calibrate(keys: keys, rotaryDims: rotaryDims, carrierHeads: carrierRank)
        let parts = Self.joint(keys: keys, values: values, carrier: carrier, quantized: quantized)
        return GaugeRoPEArm(carrier: carrier, codec: WhitenedPCACodec.fit(parts.matrix, metric: metric))
    }

    func reconstruct(keys: MLXArray, values: MLXArray, rank: Int, quantized: Bool) -> GaugeRoPEKV {
        let (n, heads, r) = (keys.dim(0), keys.dim(1), carrier.rotaryDims)
        let nope = keys.dim(2) - r
        let parts = Self.joint(keys: keys, values: values, carrier: carrier, quantized: quantized)
        let hat = codec.roundTrip(parts.matrix, rank: rank, codeBits: quantized ? 4 : nil)
        let residual = hat[0..., 0 ..< (heads * r)].reshaped([n, heads, r])
        let kNope = hat[0..., (heads * r) ..< (heads * (r + nope))].reshaped([n, heads, nope])
        let vHat = hat[0..., (heads * (r + nope))...].reshaped([n, heads, values.dim(2)])
        return GaugeRoPEKV(keys: concatenated([parts.carrier + residual, kNope], axis: -1), values: vHat)
    }
}

enum GaugeRoPEByteBudget {
    static let keyBits = 8.5
    static let valueBits = 4.5

    static func k8v4BytesPerToken(kvHeads: Int, headDim: Int) -> Double {
        Double(kvHeads * headDim) * (keyBits + valueBits) / 8
    }

    static func latentBytesPerToken(exactFloats: Int, codeRank: Int) -> Double {
        (Double(exactFloats) * keyBits + Double(codeRank) * valueBits) / 8
    }

    static func codeRank(budget: Double, exactFloats: Int, multiple: Int = 64) -> Int {
        let free = (budget * 8 - Double(exactFloats) * keyBits) / valueBits
        return max(0, Int(free.rounded(.down)) / multiple * multiple)
    }
}

enum GaugeRoPEArms {
    static let groupSize = 64

    static func fakeQuantized(_ x: MLXArray, bits: Int?) -> MLXArray {
        let xf = x.asType(.float32)
        guard let bits else { return xf }
        let q = MLX.quantized(xf, groupSize: groupSize, bits: bits)
        return MLX.dequantized(q.wq, scales: q.scales, biases: q.biases, groupSize: groupSize, bits: bits).asType(.float32)
    }

    static func attend(queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float) -> MLXArray {
        let (m, n) = (queries.dim(1), keys.dim(0))
        return MLXFast.scaledDotProductAttention(
            queries: queries.asType(.float32).expandedDimensions(axis: 0),
            keys: keys.asType(.float32).transposed(1, 0, 2).expandedDimensions(axis: 0),
            values: values.asType(.float32).transposed(1, 0, 2).expandedDimensions(axis: 0),
            scale: scale, mask: .array(createCausalMask(n: m, offset: n - m)))
    }

    static func relativeError(_ approx: [MLXArray], _ exact: [MLXArray]) -> Double {
        let num = zip(approx, exact).map { MLX.sum(MLX.square($0 - $1)).item(Float.self) }.reduce(0, +)
        let den = exact.map { MLX.sum(MLX.square($0)).item(Float.self) }.reduce(0, +)
        return Double(num / max(den, 1e-30)).squareRoot()
    }

    static func kvarnRoundTrip(_ x: MLXArray) -> (decoded: MLXArray, bytes: Int) {
        let tile = KVarNKVCache.tileTokens
        let n = x.dim(0)
        let full = n / tile * tile
        var bytes = 0
        var heads: [MLXArray] = []
        for h in 0 ..< x.dim(1) {
            var parts: [MLXArray] = []
            for start in stride(from: 0, to: full, by: tile) {
                let encoded = KVarNTileCodec.encodeTile(x[start ..< (start + tile), h, 0...])
                bytes += encoded.packedCodes.nbytes + encoded.tokenScale.nbytes + encoded.channelScale.nbytes
                parts.append(KVarNTileCodec.decodeTile(encoded))
            }
            if full < n {
                let tail = x[full..., h, 0...].asType(.float32)
                bytes += tail.nbytes / 2
                parts.append(tail)
            }
            heads.append(concatenated(parts, axis: 0))
        }
        return (stacked(heads, axis: 1), bytes)
    }
}

enum GaugeRoPEMetric {
    static func blockDiagonal(_ blocks: [MLXArray]) -> MLXArray {
        let total = blocks.reduce(0) { $0 + $1.dim(0) }
        let out = MLXArray.zeros([total, total], dtype: .float32)
        var at = 0
        for block in blocks {
            let size = block.dim(0)
            out[at ..< (at + size), at ..< (at + size)] = block
            at += size
        }
        return out
    }

    static func metrics(queries: MLXArray, kvHeads: Int, rotaryDims: Int, valueDims: Int) -> (mla: MLXArray, gauge: MLXArray) {
        let (group, headDim, half) = (queries.dim(0) / kvHeads, queries.dim(2), rotaryDims / 2)
        var rot: [MLXArray] = []
        var nope: [MLXArray] = []
        var trace: Float = 0
        for h in 0 ..< kvHeads {
            let q = queries[(h * group) ..< ((h + 1) * group)].reshaped([-1, headDim]).asType(.float32)
            let qn = q[0..., rotaryDims...]
            nope.append(matmul(qn.T, qn) / Float(q.dim(0)))
            let energy = MLX.mean(MLX.square(q), axis: 0)
            let pair = (energy[0 ..< half] + energy[half ..< rotaryDims]) / Float(2)
            rot.append(MLXArray.eye(rotaryDims, dtype: .float32) * concatenated([pair, pair]).reshaped([1, -1]))
            trace += MLX.sum(energy).item(Float.self)
        }
        let norm = max(trace / Float(kvHeads * headDim), 1e-30)
        let nopeBlocks = nope.map { $0 / norm }
        let value = MLXArray.eye(kvHeads * valueDims, dtype: .float32)
        return (blockDiagonal(nopeBlocks + [value]), blockDiagonal(rot.map { $0 / norm } + nopeBlocks + [value]))
    }
}
