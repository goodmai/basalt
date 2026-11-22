import BasaltCore
import Foundation
import MLX

package struct KVarNEncodedTile {
    package var packedCodes: MLXArray
    package var tokenScale: MLXArray
    package var channelScale: MLXArray
    package var qjlBits: MLXArray?
    package var qjlNorm: MLXArray?
    package var tokens: Int
    package var headDim: Int
}

package enum KVarNTileCodec {
    package static let lloydMax2Bit: [Float] = [-1.5104, -0.4528, 0.4528, 1.5104]
    package static let qjlScaleFactor: Float = Float((Double.pi / 2.0).squareRoot())

    package static func makeJLMatrix(dim: Int, seed: UInt64 = 2406) -> MLXArray {
        withRandomState(MLXRandom.RandomState(seed: seed)) {
            MLXRandom.normal([dim, dim], dtype: .float32)
        }
    }

    package static func packSignBits(_ projected: MLXArray) -> MLXArray {
        let t = projected.dim(0)
        let d = projected.dim(1)
        let neg = (projected .< Float(0.0)).asType(.uint32).reshaped(t, d / 32, 32)
        let shifts = MLXArray((0 ..< 32).map(UInt32.init)).reshaped(1, 1, 32)
        return (neg << shifts).sum(axis: -1).asType(.uint32)
    }

    package static func encodeTile(
        _ matrix: MLXArray,
        useQJL: Bool = false,
        jlMatrix: MLXArray? = nil,
        balanceSteps: Int = 4
    ) -> KVarNEncodedTile {
        let t = matrix.dim(0)
        let d = matrix.dim(1)
        precondition(d.isMultiple(of: 32))
        let ones = MLXArray.ones([d], dtype: .float32)
        let kRot = MACaQFusedKernels.blockHadamard(matrix.asType(.float32), scaleVector: ones, blockSize: 32, isInverse: false)

        var u = kRot
        var sTok = MLXArray.ones([t, 1], dtype: .float32)
        var sChan = MLXArray.ones([1, d], dtype: .float32)
        for _ in 0 ..< max(1, balanceSteps) {
            let rTok = clip(sqrt(maximum(mean(square(u), axis: 1, keepDims: true), Float(1e-10))), min: Float(1e-3), max: Float(1e3))
            u = u / rTok
            sTok = sTok * rTok
            let rChan = clip(sqrt(maximum(mean(square(u), axis: 0, keepDims: true), Float(1e-10))), min: Float(1e-3), max: Float(1e3))
            u = u / rChan
            sChan = sChan * rChan
        }

        let b0 = (u .> Float(-0.9816)).asType(.uint8)
        let b1 = (u .> Float(0.0)).asType(.uint8)
        let b2 = (u .> Float(0.9816)).asType(.uint8)
        let codes = b0 + b1 + b2
        let packed = PVCodeGrid.packCodes(codes, bits: 2)

        let table = MLXArray(lloydMax2Bit)
        let uHat = table.take(codes.asType(.int32))
        let kHat1 = (uHat * sChan) * sTok

        var qBits: MLXArray?
        var qNorm: MLXArray?
        if useQJL {
            let sJL = jlMatrix ?? makeJLMatrix(dim: d)
            let residual = kRot - kHat1
            let rNorm = sqrt(maximum(sum(square(residual), axis: 1), Float(1e-16)))
            let proj = matmul(residual, sJL.T)
            qBits = packSignBits(proj)
            qNorm = rNorm
        }
        eval(packed, sTok, sChan)
        if let qBits, let qNorm { eval(qBits, qNorm) }
        return KVarNEncodedTile(
            packedCodes: packed,
            tokenScale: sTok.squeezed(axis: 1),
            channelScale: sChan.squeezed(axis: 0),
            qjlBits: qBits,
            qjlNorm: qNorm,
            tokens: t,
            headDim: d)
    }

    package static func decodeTile(_ tile: KVarNEncodedTile, dtype: DType = .float32) -> MLXArray {
        let codes = PVCodeGrid.unpackCodes(tile.packedCodes, bits: 2).asType(.int32)
        let table = MLXArray(lloydMax2Bit)
        let kRot = (table.take(codes) * tile.channelScale.reshaped(1, tile.headDim)) * tile.tokenScale.reshaped(tile.tokens, 1)
        let ones = MLXArray.ones([tile.headDim], dtype: .float32)
        return MACaQFusedKernels.blockHadamard(kRot, scaleVector: ones, blockSize: 32, isInverse: true).asType(dtype)
    }

    package static func cpuEstimateScores(
        query: [Float],
        tile: KVarNEncodedTile,
        jlMatrixHost: [Float]? = nil
    ) -> [Float] {
        let t = tile.tokens
        let d = tile.headDim
        let ones = [Float](repeating: 1.0, count: 32)
        let qRot = BasaltMetalSubstrate.cpuHadamardBlock(query, scale: ones, blockSize: 32)
        let sChan = tile.channelScale.asArray(Float.self)
        let sTok = tile.tokenScale.asArray(Float.self)
        let qWeighted = (0 ..< d).map { qRot[$0] * sChan[$0] }
        let unpacked = PVCodeGrid.unpackCodes(tile.packedCodes, bits: 2).asType(.int32).asArray(Int32.self)

        var qJL = [Float](repeating: 0, count: d)
        if let jlMatrixHost, tile.qjlBits != nil {
            for r in 0 ..< d {
                var s: Float = 0
                for c in 0 ..< d { s += jlMatrixHost[r * d + c] * qRot[c] }
                qJL[r] = s
            }
        }
        let bitsHost = tile.qjlBits?.asArray(UInt32.self)
        let normHost = tile.qjlNorm?.asArray(Float.self)
        let wordsPerTok = d / 32

        var scores = [Float](repeating: 0, count: t)
        for tok in 0 ..< t {
            var dot1: Float = 0
            for c in 0 ..< d {
                let code = Int(unpacked[tok * d + c])
                dot1 += qWeighted[c] * lloydMax2Bit[code]
            }
            var total = dot1 * sTok[tok]
            if let bitsHost, let normHost {
                var jlSum: Float = 0
                for c in 0 ..< d {
                    let word = bitsHost[tok * wordsPerTok + (c >> 5)]
                    let neg = ((word >> UInt32(c & 31)) & 1) != 0
                    jlSum += neg ? -qJL[c] : qJL[c]
                }
                total += (qjlScaleFactor / Float(d)) * normHost[tok] * jlSum
            }
            scores[tok] = total
        }
        return scores
    }
}

package enum ChanMixCodec {
    package struct Encoded {
        package var background2Bit: MLXArray
        package var sensitiveIndices: MLXArray
        package var sensitive4Bit: MLXArray
        package var scale: MLXArray
    }

    package static func encode(_ keys: MLXArray, sensitiveCount: Int = 8) -> Encoded {
        let kf = keys.asType(.float32)
        let d = kf.dim(1)
        let kSens = min(max(1, sensitiveCount), d)
        let chanVar = mean(square(kf), axis: 0)
        let topIdx = argSort(-chanVar, axis: 0)[0 ..< kSens].asType(.int32)
        let scale = maximum(sqrt(mean(square(kf), axis: 1, keepDims: true)), Float(1e-6))
        let norm = kf / scale

        let bgCodes = clip(round((norm + Float(1.5))), min: Float(0), max: Float(3)).asType(.uint8)
        let sensVals = norm.take(topIdx, axis: 1)
        let sensCodes = clip(round(sensVals * Float(2.0) + Float(7.5)), min: Float(0), max: Float(15)).asType(.uint8)
        eval(bgCodes, topIdx, sensCodes, scale)
        return Encoded(
            background2Bit: bgCodes,
            sensitiveIndices: topIdx,
            sensitive4Bit: sensCodes,
            scale: scale.squeezed(axis: 1))
    }

    package static func decode(_ encoded: Encoded) -> MLXArray {
        let t = encoded.background2Bit.dim(0)
        var norm = encoded.background2Bit.asType(.float32) - Float(1.5)
        let sensRec = (encoded.sensitive4Bit.asType(.float32) - Float(7.5)) * Float(0.5)
        let idx2D = broadcast(encoded.sensitiveIndices.reshaped(1, -1), to: [t, encoded.sensitiveIndices.size])
        norm = putAlong(norm, idx2D, values: sensRec, axis: 1)
        return norm * encoded.scale.reshaped(t, 1)
    }
}

package enum ThinKVSegmenter {
    package enum PhasePrecision: Int, Sendable, Equatable {
        case completedThought2Bit = 2
        case activeSynthesis4Bit = 4
    }

    package static func classifyTokens(
        tokenIDs: [Int32],
        thinkOpenID: Int32,
        thinkCloseID: Int32,
        recentWindow: Int = 64
    ) -> [PhasePrecision] {
        let n = tokenIDs.count
        guard n > 0 else { return [] }
        var inThink = false
        var spans: [(start: Int, end: Int)] = []
        var curStart = 0
        for (idx, tok) in tokenIDs.enumerated() {
            if tok == thinkOpenID && !inThink {
                inThink = true
                curStart = idx
            } else if tok == thinkCloseID && inThink {
                inThink = false
                spans.append((curStart, idx))
            }
        }
        let recentStart = max(0, n - max(0, recentWindow))
        var result = [PhasePrecision](repeating: .activeSynthesis4Bit, count: n)
        for span in spans where span.end < recentStart {
            for pos in span.start ... span.end {
                result[pos] = .completedThought2Bit
            }
        }
        return result
    }
}
