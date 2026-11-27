import Foundation
import MLX
import MLXNN

package enum QTIPDifferentiableDecode {
    private static let edgeTableValues: [Int32] = QTIPCodebook.edgeCodeTableValues().map { Int32($0) }

    private static func decodeTiles(
        packed: MLXArray,
        rowScales: MLXArray,
        tlut: MLXArray,
        rate: QTIPRate
    ) -> MLXArray {
        let tileRows = packed.dim(-2)
        let tileCols = packed.dim(-1) / rate.wordsPerTile
        let leading = Array(packed.shape.dropLast())
        let words = stopGradient(packed.asType(.uint32)).reshaped(leading + [tileCols, rate.wordsPerTile])
        let k = rate.stepBits
        let wpt = rate.wordsPerTile
        var edgeSlices: [MLXArray] = []
        edgeSlices.reserveCapacity(128)
        for step in 0 ..< 128 {
            let bitPos = step * k
            let wIdx = bitPos >> 5
            let shift = bitPos & 31
            let lo = words[.ellipsis, wIdx ..< (wIdx + 1)]
            let hi = words[.ellipsis, ((wIdx + 1) % wpt) ..< (((wIdx + 1) % wpt) + 1)]
            let raw: MLXArray
            if shift == 0 {
                raw = bitwiseAnd(lo, UInt32(0xFFFF))
            } else {
                let shiftedLo = rightShift(lo, UInt32(shift))
                let shiftedHi = leftShift(hi, UInt32(32 - shift))
                raw = bitwiseAnd(bitwiseOr(shiftedLo, shiftedHi), UInt32(0xFFFF))
            }
            edgeSlices.append(raw.asType(.int32))
        }
        let edges = concatenated(edgeSlices, axis: -1)
        let codeIdx = stopGradient(MLXArray(edgeTableValues)[edges])
        let lutIdx = bitwiseAnd(codeIdx, Int32(511))
        let sx = which(bitwiseAnd(codeIdx, Int32(512)) .!= Int32(0), MLXArray(Float(-1)), MLXArray(Float(1)))
        let sy = which(bitwiseAnd(codeIdx, Int32(1024)) .!= Int32(0), MLXArray(Float(-1)), MLXArray(Float(1)))
        let tlut16 = stopGradient(tlut.asType(.float16).asType(.float32)).reshaped(512, 2)
        let pairs = tlut16[lutIdx]
        let xVals = pairs[.ellipsis, 0] * sx
        let yVals = pairs[.ellipsis, 1] * sy
        let tileFlat = stacked([xVals, yVals], axis: -1).reshaped(leading + [tileCols, 16, 16])
        let tileShapePrefix = Array(leading.dropLast())
        let matrix = tileFlat.swappedAxes(-3, -2).reshaped(tileShapePrefix + [tileRows * 16, tileCols * 16])
        let scaleExpanded = expandedDimensions(rowScales.asType(.float32), axis: -1)
        let tiledMatrix = matrix.reshaped(tileShapePrefix + [tileRows, 16, tileCols * 16])
        return (tiledMatrix * expandedDimensions(scaleExpanded, axis: -1))
            .reshaped(tileShapePrefix + [tileRows * 16, tileCols * 16])
    }

    package static func linearRotated(
        _ xRot: MLXArray,
        packed: MLXArray,
        rowScales: MLXArray,
        tlut: MLXArray,
        rate: QTIPRate,
        inFeatures: Int
    ) -> MLXArray {
        QTIPLinear.onCPUIfFloat64([xRot.dtype, packed.dtype, rowScales.dtype, tlut.dtype]) {
            precondition(xRot.ndim >= 1 && packed.ndim >= 2, "QTIPDifferentiableDecode.linearRotated requires ndim >= 1 input and ndim >= 2 packed")
            precondition(xRot.dim(-1) == inFeatures, "QTIPDifferentiableDecode.linearRotated input width mismatch")
            let tileRows = packed.ndim == 2 ? packed.dim(0) : packed.dim(0)
            let outFeatures = tileRows * 16
            let outShape = Array(xRot.shape.dropLast()) + [outFeatures]
            guard xRot.size > 0, tileRows > 0, inFeatures > 0 else {
                return MLXArray.zeros(outShape, dtype: xRot.dtype)
            }
            precondition(inFeatures.isMultiple(of: 16), "QTIPDifferentiableDecode.linearRotated requires inFeatures multiple of 16")
            let packed2D = packed.ndim == 2 ? packed : packed.reshaped(tileRows, (inFeatures / 16) * rate.wordsPerTile)
            let wDense = decodeTiles(packed: packed2D, rowScales: rowScales.reshaped(tileRows), tlut: tlut, rate: rate)
            let flatX = xRot.asType(.float32).reshaped(-1, inFeatures)
            return matmul(flatX, wDense.T).reshaped(outShape).asType(xRot.dtype)
        }
    }
}
