import BasaltCore
import Foundation
import MLX

package struct GGUFMacAqTensorReport: Sendable, Equatable {
    package var name: String
    package var type: GGMLType
    package var flips: Int
    package var lossBefore: Float
    package var lossAfter: Float

    package init(name: String, type: GGMLType, flips: Int, lossBefore: Float, lossAfter: Float) {
        self.name = name
        self.type = type
        self.flips = flips
        self.lossBefore = lossBefore
        self.lossAfter = lossAfter
    }
}

package enum GGUFMacAqRefiner {
    package static let supportedTypes: Set<GGMLType> = [.q4K, .q5K, .q6K]

    package static func unpackKQuant(
        bytes: [UInt8],
        type: GGMLType,
        rows: Int,
        cols: Int
    ) throws(BasaltError) -> MACaQGridState {
        guard supportedTypes.contains(type), cols > 0, cols.isMultiple(of: 256) else {
            throw .invalidRequestStructure(details: "GGUFMacAqRefiner requires Q4_K, Q5_K or Q6_K with cols % 256 == 0")
        }
        let blockBytes = type.block.bytes
        let totalBlocks = (rows * cols) / 256
        guard bytes.count == totalBlocks * blockBytes else {
            throw .invalidRequestStructure(details: "GGUFMacAqRefiner byte count mismatch: got \(bytes.count), expected \(totalBlocks * blockBytes)")
        }
        let total = rows * cols
        var codes = [Int32](repeating: 0, count: total)
        var scales = [Float](repeating: 0, count: total)
        var biases = [Float](repeating: 0, count: total)
        for blk in 0 ..< totalBlocks {
            let b0 = blk * blockBytes
            let v0 = blk * 256
            switch type {
            case .q4K:
                unpackQ4KBlock(bytes, b0: b0, v0: v0, codes: &codes, scales: &scales, biases: &biases)
            case .q5K:
                unpackQ5KBlock(bytes, b0: b0, v0: v0, codes: &codes, scales: &scales, biases: &biases)
            case .q6K:
                unpackQ6KBlock(bytes, b0: b0, v0: v0, codes: &codes, scales: &scales, biases: &biases)
            default:
                break
            }
        }
        let maxLevel: Int32 = type == .q4K ? 15 : (type == .q5K ? 31 : 63)
        return MACaQGridState(
            codes: MLXArray(codes, [rows, cols]),
            scaleElem: MLXArray(scales, [rows, cols]),
            biasElem: MLXArray(biases, [rows, cols]),
            levelTable: nil,
            maxLevel: maxLevel)
    }

    package static func repackKQuant(
        originalBytes: [UInt8],
        codes: [Int32],
        type: GGMLType
    ) -> [UInt8] {
        var out = originalBytes
        let blockBytes = type.block.bytes
        let totalBlocks = codes.count / 256
        for blk in 0 ..< totalBlocks {
            let b0 = blk * blockBytes
            let v0 = blk * 256
            switch type {
            case .q4K:
                repackQ4KBlock(&out, b0: b0, codes: codes, v0: v0)
            case .q5K:
                repackQ5KBlock(&out, b0: b0, codes: codes, v0: v0)
            case .q6K:
                repackQ6KBlock(&out, b0: b0, codes: codes, v0: v0)
            default:
                break
            }
        }
        return out
    }

    package static func refineTensorBytes(
        rawBytes: [UInt8],
        type: GGMLType,
        targetWeight: MLXArray,
        hessian: MLXArray,
        maxIterations: Int = 128
    ) throws(BasaltError) -> (bytes: [UInt8], flipResult: MACaQFlipResult) {
        let rows = targetWeight.dim(0)
        let cols = targetWeight.dim(1)
        var state = try unpackKQuant(bytes: rawBytes, type: type, rows: rows, cols: cols)
        let res = MACaQMath.flipRefine(
            state: &state,
            targetWeight: targetWeight.asType(.float32),
            hessian: hessian.asType(.float32),
            maxIterations: maxIterations)
        guard res.flips > 0, res.lossAfter <= res.lossBefore else {
            return (rawBytes, res)
        }
        eval(state.codes)
        let updatedCodes = state.codes.asType(.int32).asArray(Int32.self)
        let repacked = repackKQuant(originalBytes: rawBytes, codes: updatedCodes, type: type)
        return (repacked, res)
    }
}

extension GGUFMacAqRefiner {
    fileprivate static func halfAt(_ b: [UInt8], _ offset: Int) -> Float {
        let u = UInt16(b[offset]) | (UInt16(b[offset + 1]) << 8)
        return Float(Float16(bitPattern: u))
    }

    fileprivate static func scaleMinAt(_ j: Int, _ b: [UInt8], _ b0: Int) -> (Float, Float) {
        let q0 = Int(b[b0 + 4 + (j < 4 ? j : j + 4)])
        if j < 4 {
            let m0 = Int(b[b0 + 8 + j])
            return (Float(q0 & 63), Float(m0 & 63))
        }
        let hiS = Int(b[b0 + 4 + (j - 4)]) >> 6
        let hiM = Int(b[b0 + 4 + j]) >> 6
        return (Float((q0 & 0xf) | (hiS << 4)), Float((q0 >> 4) | (hiM << 4)))
    }

    fileprivate static func unpackQ4KBlock(
        _ b: [UInt8], b0: Int, v0: Int,
        codes: inout [Int32], scales: inout [Float], biases: inout [Float]
    ) {
        let (d, dmin) = (halfAt(b, b0), halfAt(b, b0 + 2))
        for group in 0 ..< 4 {
            let (s1, m1) = scaleMinAt(2 * group, b, b0)
            let (s2, m2) = scaleMinAt(2 * group + 1, b, b0)
            let (dl0, ml0) = (d * s1, -(dmin * m1))
            let (dl1, ml1) = (d * s2, -(dmin * m2))
            for l in 0 ..< 32 {
                let q = Int32(b[b0 + 16 + group * 32 + l])
                let i0 = v0 + group * 64 + l
                let i1 = i0 + 32
                codes[i0] = q & 0xf
                scales[i0] = dl0
                biases[i0] = ml0
                codes[i1] = (q >> 4) & 0xf
                scales[i1] = dl1
                biases[i1] = ml1
            }
        }
    }

    fileprivate static func repackQ4KBlock(_ out: inout [UInt8], b0: Int, codes: [Int32], v0: Int) {
        for group in 0 ..< 4 {
            for l in 0 ..< 32 {
                let c0 = UInt8(clamping: min(max(codes[v0 + group * 64 + l], 0), 15))
                let c1 = UInt8(clamping: min(max(codes[v0 + group * 64 + 32 + l], 0), 15))
                out[b0 + 16 + group * 32 + l] = c0 | (c1 << 4)
            }
        }
    }

    fileprivate static func unpackQ5KBlock(
        _ b: [UInt8], b0: Int, v0: Int,
        codes: inout [Int32], scales: inout [Float], biases: inout [Float]
    ) {
        let (d, dmin) = (halfAt(b, b0), halfAt(b, b0 + 2))
        for g in 0 ..< 8 {
            let (s, m) = scaleMinAt(g, b, b0)
            let (dl, ml) = (d * s, -(dmin * m))
            for l in 0 ..< 32 {
                let ql = (Int32(b[b0 + 48 + (g / 2) * 32 + l]) >> Int32(4 * (g & 1))) & 0xf
                let qh = (Int32(b[b0 + 16 + l]) >> Int32(g)) & 1
                let idx = v0 + g * 32 + l
                codes[idx] = ql | (qh << 4)
                scales[idx] = dl
                biases[idx] = ml
            }
        }
    }

    fileprivate static func repackQ5KBlock(_ out: inout [UInt8], b0: Int, codes: [Int32], v0: Int) {
        for off in 16 ..< 176 { out[b0 + off] = 0 }
        for g in 0 ..< 8 {
            for l in 0 ..< 32 {
                let c = UInt8(clamping: min(max(codes[v0 + g * 32 + l], 0), 31))
                out[b0 + 48 + (g / 2) * 32 + l] |= (c & 0xf) << UInt8(4 * (g & 1))
                out[b0 + 16 + l] |= ((c >> 4) & 1) << UInt8(g)
            }
        }
    }

    fileprivate static func unpackQ6KBlock(
        _ b: [UInt8], b0: Int, v0: Int,
        codes: inout [Int32], scales: inout [Float], biases: inout [Float]
    ) {
        let d = halfAt(b, b0 + 208)
        for chunk in 0 ..< 2 {
            let (ql, qh, sc) = (b0 + chunk * 64, b0 + 128 + chunk * 32, b0 + 192 + chunk * 8)
            for l in 0 ..< 32 {
                let low = Int32(b[ql + l])
                let high = Int32(b[ql + l + 32])
                let top = Int32(b[qh + l])
                let s0 = d * Float(Int8(bitPattern: b[sc + l / 16]))
                let s1 = d * Float(Int8(bitPattern: b[sc + l / 16 + 2]))
                let s2 = d * Float(Int8(bitPattern: b[sc + l / 16 + 4]))
                let s3 = d * Float(Int8(bitPattern: b[sc + l / 16 + 6]))
                let base = v0 + chunk * 128 + l
                codes[base] = (low & 0xf) | ((top & 3) << 4)
                scales[base] = s0
                biases[base] = -32.0 * s0
                codes[base + 32] = (high & 0xf) | (((top >> 2) & 3) << 4)
                scales[base + 32] = s1
                biases[base + 32] = -32.0 * s1
                codes[base + 64] = ((low >> 4) & 0xf) | (((top >> 4) & 3) << 4)
                scales[base + 64] = s2
                biases[base + 64] = -32.0 * s2
                codes[base + 96] = ((high >> 4) & 0xf) | (((top >> 6) & 3) << 4)
                scales[base + 96] = s3
                biases[base + 96] = -32.0 * s3
            }
        }
    }

    fileprivate static func repackQ6KBlock(_ out: inout [UInt8], b0: Int, codes: [Int32], v0: Int) {
        for chunk in 0 ..< 2 {
            let (ql, qh) = (b0 + chunk * 64, b0 + 128 + chunk * 32)
            for l in 0 ..< 32 {
                let base = v0 + chunk * 128 + l
                let c0 = UInt8(clamping: min(max(codes[base], 0), 63))
                let c1 = UInt8(clamping: min(max(codes[base + 32], 0), 63))
                let c2 = UInt8(clamping: min(max(codes[base + 64], 0), 63))
                let c3 = UInt8(clamping: min(max(codes[base + 96], 0), 63))
                out[ql + l] = (c0 & 0xf) | ((c2 & 0xf) << 4)
                out[ql + l + 32] = (c1 & 0xf) | ((c3 & 0xf) << 4)
                out[qh + l] = ((c0 >> 4) & 3) | (((c1 >> 4) & 3) << 2) | (((c2 >> 4) & 3) << 4) | (((c3 >> 4) & 3) << 6)
            }
        }
    }
}
