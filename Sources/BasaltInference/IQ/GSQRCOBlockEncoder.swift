import BasaltCore
import Foundation

package enum GSQRCOBlockEncoder {
    package static let encodableTypes: [GGMLType] = [
        .iq2XXS, .iq2XS, .q2K, .iq3XXS, .iq3S, .q3K, .iq4XS, .iq4NL, .q4K, .q8_0,
    ]

    package static func supports(_ type: GGMLType) -> Bool {
        encodableTypes.contains(type)
    }

    package static func encodeMatrix(_ values: [Float], rows: Int, columns: Int, type: GGMLType) -> [UInt8] {
        let (be, bb) = type.block
        precondition(columns > 0 && columns.isMultiple(of: be), "columns \(columns) must align to \(be)")
        precondition(values.count == rows * columns, "expected \(rows * columns) values, got \(values.count)")
        let blocksPerRow = columns / be
        let totalBlocks = rows * blocksPerRow
        var out = [UInt8](repeating: 0, count: totalBlocks * bb)
        for blk in 0 ..< totalBlocks {
            encodeBlock(values, valStart: blk * be, into: &out, outStart: blk * bb, type: type)
        }
        return out
    }

    private static func encodeBlock(
        _ x: [Float], valStart: Int, into out: inout [UInt8], outStart: Int, type: GGMLType
    ) {
        switch type {
        case .q8_0: encodeQ8_0(x, v0: valStart, into: &out, b0: outStart)
        case .iq4NL: encodeIQ4NL(x, v0: valStart, into: &out, b0: outStart)
        case .iq4XS: encodeIQ4XS(x, v0: valStart, into: &out, b0: outStart)
        case .q4K: encodeQ4K(x, v0: valStart, into: &out, b0: outStart)
        case .q3K: encodeQ3K(x, v0: valStart, into: &out, b0: outStart)
        case .q2K: encodeQ2K(x, v0: valStart, into: &out, b0: outStart)
        case .iq3S: encodeIQ3S(x, v0: valStart, into: &out, b0: outStart)
        case .iq3XXS: encodeIQ3XXS(x, v0: valStart, into: &out, b0: outStart)
        case .iq2XS: encodeIQ2XS(x, v0: valStart, into: &out, b0: outStart)
        case .iq2XXS: encodeIQ2XXS(x, v0: valStart, into: &out, b0: outStart)
        default: preconditionFailure("GSQRCOBlockEncoder does not support \(type.name)")
        }
    }

    static func clampInt(_ value: Float, _ lo: Int, _ hi: Int) -> Int {
        guard value.isFinite else { return lo }
        return max(lo, min(hi, Int(value.rounded())))
    }

    static func putHalf(_ value: Float, into out: inout [UInt8], at offset: Int) -> Float {
        let clamped = value.isFinite ? max(-65504.0, min(65504.0, value)) : 0.0
        let h = Float16(clamped)
        let bits = h.bitPattern
        out[offset] = UInt8(bits & 0xff)
        out[offset + 1] = UInt8(bits >> 8)
        return Float(h)
    }

    static func putU16(_ value: UInt16, into out: inout [UInt8], at offset: Int) {
        out[offset] = UInt8(value & 0xff)
        out[offset + 1] = UInt8(value >> 8)
    }

    static func putU32(_ value: UInt32, into out: inout [UInt8], at offset: Int) {
        out[offset] = UInt8(truncatingIfNeeded: value)
        out[offset + 1] = UInt8(truncatingIfNeeded: value >> 8)
        out[offset + 2] = UInt8(truncatingIfNeeded: value >> 16)
        out[offset + 3] = UInt8(truncatingIfNeeded: value >> 24)
    }

    static func maxAbs(_ x: [Float], start: Int, count: Int) -> Float {
        var m: Float = 0
        for idx in start ..< (start + count) {
            let a = abs(x[idx])
            if a > m { m = a }
        }
        return m
    }

    static func nearestIQ4NL(_ value: Float, scale: Float) -> UInt8 {
        guard scale > 0 else { return 8 }
        let target = value / scale
        var bestIdx = 0
        var bestErr = abs(target - Float(ggmlValuesIQ4NL[0]))
        for idx in 1 ..< 16 {
            let err = abs(target - Float(ggmlValuesIQ4NL[idx]))
            if err < bestErr {
                bestErr = err
                bestIdx = idx
            }
        }
        return UInt8(bestIdx)
    }
}

extension GSQRCOBlockEncoder {
    static func encodeQ8_0(_ x: [Float], v0: Int, into out: inout [UInt8], b0: Int) {
        let d = putHalf(maxAbs(x, start: v0, count: 32) / 127.0, into: &out, at: b0)
        let inv: Float = d > 0 ? (1.0 / d) : 0.0
        for j in 0 ..< 32 {
            let q = clampInt(x[v0 + j] * inv, -127, 127)
            out[b0 + 2 + j] = UInt8(bitPattern: Int8(q))
        }
    }

    static func encodeIQ4NL(_ x: [Float], v0: Int, into out: inout [UInt8], b0: Int) {
        let d = putHalf(maxAbs(x, start: v0, count: 32) / 127.0, into: &out, at: b0)
        for j in 0 ..< 16 {
            let lo = nearestIQ4NL(x[v0 + j], scale: d)
            let hi = nearestIQ4NL(x[v0 + j + 16], scale: d)
            out[b0 + 2 + j] = lo | (hi << 4)
        }
    }

    static func encodeIQ4XS(_ x: [Float], v0: Int, into out: inout [UInt8], b0: Int) {
        let d = putHalf(maxAbs(x, start: v0, count: 256) / (31.0 * 127.0), into: &out, at: b0)
        var high: UInt16 = 0
        for ib in 0 ..< 8 {
            let subMax = maxAbs(x, start: v0 + ib * 32, count: 32)
            let ls = d > 0 ? clampInt(subMax / (127.0 * d) + 32.0, 0, 63) : 32
            out[b0 + 4 + ib / 2] |= UInt8(ls & 0xf) << (4 * (ib % 2))
            high |= UInt16((ls >> 4) & 3) << (2 * ib)
            let dl = d * Float(ls - 32)
            for j in 0 ..< 16 {
                let lo = nearestIQ4NL(x[v0 + ib * 32 + j], scale: dl)
                let hi = nearestIQ4NL(x[v0 + ib * 32 + j + 16], scale: dl)
                out[b0 + 8 + ib * 16 + j] = lo | (hi << 4)
            }
        }
        putU16(high, into: &out, at: b0 + 2)
    }

    static func encodeQ4K(_ x: [Float], v0: Int, into out: inout [UInt8], b0: Int) {
        var rawScales = [Float](repeating: 0, count: 8)
        var rawOffsets = [Float](repeating: 0, count: 8)
        for g in 0 ..< 8 {
            var lo: Float = .greatestFiniteMagnitude
            var hi: Float = -.greatestFiniteMagnitude
            for l in 0 ..< 32 {
                let v = x[v0 + g * 32 + l]
                if v < lo { lo = v }
                if v > hi { hi = v }
            }
            rawOffsets[g] = max(0, -lo)
            rawScales[g] = max(0, hi + rawOffsets[g]) / 15.0
        }
        let d = putHalf((rawScales.max() ?? 0) / 63.0, into: &out, at: b0)
        let dmin = putHalf((rawOffsets.max() ?? 0) / 63.0, into: &out, at: b0 + 2)
        let sInt = rawScales.map { d > 0 ? clampInt($0 / d, 0, 63) : 0 }
        let mInt = rawOffsets.map { dmin > 0 ? clampInt($0 / dmin, 0, 63) : 0 }
        for j in 0 ..< 4 {
            out[b0 + 4 + j] = UInt8((sInt[j] & 63) | (((sInt[j + 4] >> 4) & 3) << 6))
            out[b0 + 8 + j] = UInt8((mInt[j] & 63) | (((mInt[j + 4] >> 4) & 3) << 6))
            out[b0 + 12 + j] = UInt8((sInt[j + 4] & 0xf) | ((mInt[j + 4] & 0xf) << 4))
        }
        for group in 0 ..< 4 {
            let (dl0, ml0) = (d * Float(sInt[2 * group]), dmin * Float(mInt[2 * group]))
            let (dl1, ml1) = (d * Float(sInt[2 * group + 1]), dmin * Float(mInt[2 * group + 1]))
            for l in 0 ..< 32 {
                let q0 = dl0 > 0 ? clampInt((x[v0 + group * 64 + l] + ml0) / dl0, 0, 15) : 0
                let q1 = dl1 > 0 ? clampInt((x[v0 + group * 64 + 32 + l] + ml1) / dl1, 0, 15) : 0
                out[b0 + 16 + group * 32 + l] = UInt8(q0 | (q1 << 4))
            }
        }
    }

    static func encodeQ3K(_ x: [Float], v0: Int, into out: inout [UInt8], b0: Int) {
        let subMax = (0 ..< 16).map { maxAbs(x, start: v0 + $0 * 16, count: 16) }
        let d = putHalf((subMax.max() ?? 0) / (31.0 * 4.0), into: &out, at: b0 + 108)
        var scArr = [Int](repeating: 32, count: 16)
        for s in 0 ..< 16 {
            let sc = d > 0 ? clampInt(subMax[s] / (4.0 * d) + 32.0, 0, 63) : 32
            scArr[s] = sc
            out[b0 + 96 + (s % 8)] |= UInt8(sc & 0xf) << (4 * (s / 8))
            out[b0 + 104 + (s % 4)] |= UInt8((sc >> 4) & 3) << (2 * (s / 4))
        }
        for g in 0 ..< 8 {
            for h in 0 ..< 2 {
                let dl = d * Float(scArr[2 * g + h] - 32)
                for l in 0 ..< 16 {
                    let qSigned = dl > 0 ? clampInt(x[v0 + g * 32 + h * 16 + l] / dl, -4, 3) : 0
                    let u = qSigned + 4
                    out[b0 + 32 + (g / 4) * 32 + h * 16 + l] |= UInt8(u & 3) << (2 * (g % 4))
                    out[b0 + h * 16 + l] |= UInt8((u >> 2) & 1) << g
                }
            }
        }
    }

    static func encodeQ2K(_ x: [Float], v0: Int, into out: inout [UInt8], b0: Int) {
        var rawScales = [Float](repeating: 0, count: 16)
        var rawOffsets = [Float](repeating: 0, count: 16)
        for s in 0 ..< 16 {
            var lo: Float = .greatestFiniteMagnitude
            var hi: Float = -.greatestFiniteMagnitude
            for l in 0 ..< 16 {
                let v = x[v0 + s * 16 + l]
                if v < lo { lo = v }
                if v > hi { hi = v }
            }
            rawOffsets[s] = max(0, -lo)
            rawScales[s] = max(0, hi + rawOffsets[s]) / 3.0
        }
        let d = putHalf((rawScales.max() ?? 0) / 15.0, into: &out, at: b0 + 80)
        let dmin = putHalf((rawOffsets.max() ?? 0) / 15.0, into: &out, at: b0 + 82)
        for g in 0 ..< 8 {
            for h in 0 ..< 2 {
                let s = 2 * g + h
                let sc = d > 0 ? clampInt(rawScales[s] / d, 0, 15) : 0
                let mc = dmin > 0 ? clampInt(rawOffsets[s] / dmin, 0, 15) : 0
                out[b0 + s] = UInt8(sc | (mc << 4))
                let (dl, ml) = (d * Float(sc), dmin * Float(mc))
                for l in 0 ..< 16 {
                    let q = dl > 0 ? clampInt((x[v0 + s * 16 + l] + ml) / dl, 0, 3) : 0
                    out[b0 + 16 + (g / 4) * 32 + h * 16 + l] |= UInt8(q) << (2 * (g % 4))
                }
            }
        }
    }
}

extension GSQRCOBlockEncoder {
    static func evenParitySignIndex8(_ x: [Float], start: Int) -> Int {
        var mask = 0
        var minIdx = 0
        var minMag = abs(x[start])
        for j in 0 ..< 8 {
            let v = x[start + j]
            if v < 0 { mask |= (1 << j) }
            let a = abs(v)
            if a < minMag {
                minMag = a
                minIdx = j
            }
        }
        if (mask.nonzeroBitCount & 1) != 0 { mask ^= (1 << minIdx) }
        return mask & 127
    }

    static func nearestGrid4(
        _ x: [Float], start: Int, scale: Float, grid: [UInt32]
    ) -> (idx: Int, signs: Int) {
        var signs = 0
        var mag = InlineArray<4, Float>(repeating: 0)
        let inv: Float = scale > 0 ? (1.0 / scale) : 0.0
        for j in 0 ..< 4 {
            let v = x[start + j]
            if v < 0 { signs |= (1 << j) }
            mag[j] = abs(v) * inv
        }
        guard scale > 0 else { return (0, signs) }
        var bestIdx = 0
        var bestErr: Float = .greatestFiniteMagnitude
        for idx in 0 ..< grid.count {
            let word = grid[idx]
            let e0 = mag[0] - Float(word & 0xff)
            let e1 = mag[1] - Float((word >> 8) & 0xff)
            let e2 = mag[2] - Float((word >> 16) & 0xff)
            let e3 = mag[3] - Float((word >> 24) & 0xff)
            let err = e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3
            if err < bestErr {
                bestErr = err
                bestIdx = idx
            }
        }
        return (bestIdx, signs)
    }

    static func nearestSignedGrid4(
        _ x: [Float], start: Int, scale: Float, signs: Int, grid: [UInt32]
    ) -> Int {
        guard scale > 0 else { return 0 }
        let inv = 1.0 / scale
        var target = InlineArray<4, Float>(repeating: 0)
        for j in 0 ..< 4 {
            let s: Float = ((signs >> j) & 1) != 0 ? -1.0 : 1.0
            target[j] = x[start + j] * inv * s
        }
        var bestIdx = 0
        var bestErr: Float = .greatestFiniteMagnitude
        for idx in 0 ..< grid.count {
            let word = grid[idx]
            let e0 = target[0] - Float(word & 0xff)
            let e1 = target[1] - Float((word >> 8) & 0xff)
            let e2 = target[2] - Float((word >> 16) & 0xff)
            let e3 = target[3] - Float((word >> 24) & 0xff)
            let err = e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3
            if err < bestErr {
                bestErr = err
                bestIdx = idx
            }
        }
        return bestIdx
    }

    static func nearestSignedGrid8(
        _ x: [Float], start: Int, scale: Float, signs: Int, grid: [UInt64]
    ) -> Int {
        guard scale > 0 else { return 0 }
        let inv = 1.0 / scale
        var target = InlineArray<8, Float>(repeating: 0)
        for j in 0 ..< 8 {
            let s: Float = ((signs >> j) & 1) != 0 ? -1.0 : 1.0
            target[j] = x[start + j] * inv * s
        }
        var bestIdx = 0
        var bestErr: Float = .greatestFiniteMagnitude
        for idx in 0 ..< grid.count {
            let word = grid[idx]
            var err: Float = 0
            for j in 0 ..< 8 {
                let d = target[j] - Float((word >> (8 * j)) & 0xff)
                err += d * d
            }
            if err < bestErr {
                bestErr = err
                bestIdx = idx
            }
        }
        return bestIdx
    }
}

extension GSQRCOBlockEncoder {
    static func encodeIQ3S(_ x: [Float], v0: Int, into out: inout [UInt8], b0: Int) {
        let d = putHalf(maxAbs(x, start: v0, count: 256) / (31.0 * 15.0), into: &out, at: b0)
        for ib in 0 ..< 8 {
            let subMax = maxAbs(x, start: v0 + ib * 32, count: 32)
            let sc = d > 0 ? clampInt(((subMax / (15.0 * d)) - 1.0) * 0.5, 0, 15) : 0
            out[b0 + 106 + ib / 2] |= UInt8(sc) << (4 * (ib % 2))
            let db = d * Float(1 + 2 * sc)
            var qh = 0
            for l in 0 ..< 4 {
                let first = nearestGrid4(x, start: v0 + ib * 32 + l * 8, scale: db, grid: ggmlIQ3SGrid)
                let second = nearestGrid4(x, start: v0 + ib * 32 + l * 8 + 4, scale: db, grid: ggmlIQ3SGrid)
                out[b0 + 2 + ib * 8 + 2 * l] = UInt8(first.idx & 0xff)
                out[b0 + 2 + ib * 8 + 2 * l + 1] = UInt8(second.idx & 0xff)
                if (first.idx & 256) != 0 { qh |= (1 << (2 * l)) }
                if (second.idx & 256) != 0 { qh |= (1 << (2 * l + 1)) }
                out[b0 + 74 + ib * 4 + l] = UInt8(first.signs | (second.signs << 4))
            }
            out[b0 + 66 + ib] = UInt8(qh)
        }
    }

    static func encodeIQ3XXS(_ x: [Float], v0: Int, into out: inout [UInt8], b0: Int) {
        let d = putHalf(maxAbs(x, start: v0, count: 256) / (7.75 * 62.0), into: &out, at: b0)
        for ib in 0 ..< 8 {
            let subMax = maxAbs(x, start: v0 + ib * 32, count: 32)
            let sc = d > 0 ? clampInt((subMax / (62.0 * d)) * 2.0 - 0.5, 0, 15) : 0
            let db = d * (0.5 + Float(sc)) * 0.5
            var aux = UInt32(sc) << 28
            for l in 0 ..< 4 {
                let sIdx = evenParitySignIndex8(x, start: v0 + ib * 32 + l * 8)
                let sByte = Int(ggmlSignsIQ2XS[sIdx])
                let c0 = nearestSignedGrid4(x, start: v0 + ib * 32 + l * 8, scale: db, signs: sByte & 0xf, grid: ggmlIQ3XXSGrid)
                let c1 = nearestSignedGrid4(x, start: v0 + ib * 32 + l * 8 + 4, scale: db, signs: (sByte >> 4) & 0xf, grid: ggmlIQ3XXSGrid)
                out[b0 + 2 + ib * 8 + 2 * l] = UInt8(c0)
                out[b0 + 2 + ib * 8 + 2 * l + 1] = UInt8(c1)
                aux |= UInt32(sIdx) << UInt32(7 * l)
            }
            putU32(aux, into: &out, at: b0 + 66 + 4 * ib)
        }
    }

    static func encodeIQ2XS(_ x: [Float], v0: Int, into out: inout [UInt8], b0: Int) {
        let d = putHalf(maxAbs(x, start: v0, count: 256) / (3.875 * 43.0), into: &out, at: b0)
        for ib in 0 ..< 8 {
            let maxLo = maxAbs(x, start: v0 + ib * 32, count: 16)
            let maxHi = maxAbs(x, start: v0 + ib * 32 + 16, count: 16)
            let scLo = d > 0 ? clampInt((maxLo / (43.0 * d)) * 4.0 - 0.5, 0, 15) : 0
            let scHi = d > 0 ? clampInt((maxHi / (43.0 * d)) * 4.0 - 0.5, 0, 15) : 0
            out[b0 + 66 + ib] = UInt8(scLo | (scHi << 4))
            for l in 0 ..< 4 {
                let db = d * (0.5 + Float(l < 2 ? scLo : scHi)) * 0.25
                let sIdx = evenParitySignIndex8(x, start: v0 + ib * 32 + l * 8)
                let sByte = Int(ggmlSignsIQ2XS[sIdx])
                let gIdx = nearestSignedGrid8(x, start: v0 + ib * 32 + l * 8, scale: db, signs: sByte, grid: ggmlIQ2XSGrid)
                let q = UInt16(gIdx) | (UInt16(sIdx) << 9)
                putU16(q, into: &out, at: b0 + 2 + 2 * (4 * ib + l))
            }
        }
    }

    static func encodeIQ2XXS(_ x: [Float], v0: Int, into out: inout [UInt8], b0: Int) {
        let d = putHalf(maxAbs(x, start: v0, count: 256) / (3.875 * 43.0), into: &out, at: b0)
        for ib in 0 ..< 8 {
            let subMax = maxAbs(x, start: v0 + ib * 32, count: 32)
            let sc = d > 0 ? clampInt((subMax / (43.0 * d)) * 4.0 - 0.5, 0, 15) : 0
            let db = d * (0.5 + Float(sc)) * 0.25
            var codes: UInt32 = 0
            var scaleWord = UInt32(sc) << 28
            for l in 0 ..< 4 {
                let sIdx = evenParitySignIndex8(x, start: v0 + ib * 32 + l * 8)
                let sByte = Int(ggmlSignsIQ2XS[sIdx])
                let gIdx = nearestSignedGrid8(x, start: v0 + ib * 32 + l * 8, scale: db, signs: sByte, grid: ggmlIQ2XXSGrid)
                codes |= UInt32(gIdx) << UInt32(8 * l)
                scaleWord |= UInt32(sIdx) << UInt32(7 * l)
            }
            putU32(codes, into: &out, at: b0 + 2 + 8 * ib)
            putU32(scaleWord, into: &out, at: b0 + 6 + 8 * ib)
        }
    }
}
