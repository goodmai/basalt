import Foundation

extension GGUFDequant {
    static func iq3S(_ b: UnsafeRawBufferPointer, _ y: UnsafeMutableBufferPointer<Float>) {
        let d = half(b, 0)
        for pair in 0 ..< 4 {
            let scales = Int(b[106 + pair])
            for part in 0 ..< 2 {
                let db = d * Float(1 + 2 * (part == 0 ? scales & 0xf : scales >> 4))
                let (qs, qh, signs) = (2 + pair * 16 + part * 8, Int(b[66 + pair * 2 + part]), 74 + pair * 8 + part * 4)
                for l in 0 ..< 4 {
                    let first = Int(b[qs + 2 * l]) | ((qh << (8 - 2 * l)) & 256)
                    let second = Int(b[qs + 2 * l + 1]) | ((qh << (7 - 2 * l)) & 256)
                    let (sign, out) = (Int(b[signs + l]), pair * 64 + part * 32 + l * 8)
                    for j in 0 ..< 4 {
                        y[out + j] = db * Float(gridByte(ggmlIQ3SGrid, first, j)) * GGUFDequant.sign(sign, j)
                        y[out + j + 4] = db * Float(gridByte(ggmlIQ3SGrid, second, j)) * GGUFDequant.sign(sign, j + 4)
                    }
                }
            }
        }
    }

    static func iq3XXS(_ b: UnsafeRawBufferPointer, _ y: UnsafeMutableBufferPointer<Float>) {
        let d = half(b, 0)
        for ib in 0 ..< 8 {
            let aux = u32(b, 66 + 4 * ib)
            let db = d * (0.5 + Float(aux >> 28)) * 0.5
            for l in 0 ..< 4 {
                let signs = Int(ggmlSignsIQ2XS[Int((aux >> (7 * UInt32(l))) & 127)])
                let (first, second, out) = (Int(b[2 + ib * 8 + 2 * l]), Int(b[2 + ib * 8 + 2 * l + 1]), ib * 32 + l * 8)
                for j in 0 ..< 4 {
                    y[out + j] = db * Float(gridByte(ggmlIQ3XXSGrid, first, j)) * sign(signs, j)
                    y[out + j + 4] = db * Float(gridByte(ggmlIQ3XXSGrid, second, j)) * sign(signs, j + 4)
                }
            }
        }
    }

    static func iq2S(_ b: UnsafeRawBufferPointer, _ y: UnsafeMutableBufferPointer<Float>) {
        let d = half(b, 0)
        for ib in 0 ..< 8 {
            let scales = Int(b[74 + ib])
            let (low, high) = (d * (0.5 + Float(scales & 0xf)) * 0.25, d * (0.5 + Float(scales >> 4)) * 0.25)
            let qh = Int(b[66 + ib])
            for l in 0 ..< 4 {
                let index = Int(b[2 + ib * 4 + l]) | ((qh << (8 - 2 * l)) & 0x300)
                let signs = Int(b[34 + ib * 4 + l])
                let db = l < 2 ? low : high
                for j in 0 ..< 8 { y[ib * 32 + l * 8 + j] = db * Float(gridByte(ggmlIQ2SGrid, index, j)) * sign(signs, j) }
            }
        }
    }

    static func iq2XS(_ b: UnsafeRawBufferPointer, _ y: UnsafeMutableBufferPointer<Float>) {
        let d = half(b, 0)
        for ib in 0 ..< 8 {
            let scales = Int(b[66 + ib])
            let (low, high) = (d * (0.5 + Float(scales & 0xf)) * 0.25, d * (0.5 + Float(scales >> 4)) * 0.25)
            for l in 0 ..< 4 {
                let q = u16(b, 2 + 2 * (4 * ib + l))
                let db = l < 2 ? low : high
                let signs = Int(ggmlSignsIQ2XS[q >> 9])
                for j in 0 ..< 8 { y[ib * 32 + l * 8 + j] = db * Float(gridByte(ggmlIQ2XSGrid, q & 511, j)) * sign(signs, j) }
            }
        }
    }

    static func iq2XXS(_ b: UnsafeRawBufferPointer, _ y: UnsafeMutableBufferPointer<Float>) {
        let d = half(b, 0)
        for ib in 0 ..< 8 {
            let (grid, scale) = (u32(b, 2 + 8 * ib), u32(b, 6 + 8 * ib))
            let db = d * (0.5 + Float(scale >> 28)) * 0.25
            for l in 0 ..< 4 {
                let index = Int((grid >> (8 * UInt32(l))) & 0xff)
                let signs = Int(ggmlSignsIQ2XS[Int((scale >> (7 * UInt32(l))) & 127)])
                for j in 0 ..< 8 { y[ib * 32 + l * 8 + j] = db * Float(gridByte(ggmlIQ2XXSGrid, index, j)) * sign(signs, j) }
            }
        }
    }

    static func iq1M(_ b: UnsafeRawBufferPointer, _ y: UnsafeMutableBufferPointer<Float>) {
        let sc = (0 ..< 4).map { u16(b, 48 + 2 * $0) }
        let bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)
        let d = Float(Float16(bitPattern: UInt16(bits)))
        for ib in 0 ..< 8 {
            let shift = 6 * (ib % 2)
            let dl = [d * Float(2 * ((sc[ib / 2] >> shift) & 7) + 1), d * Float(2 * ((sc[ib / 2] >> (shift + 3)) & 7) + 1)]
            let (qh0, qh1) = (Int(b[32 + ib * 2]), Int(b[33 + ib * 2]))
            let qs = (0 ..< 4).map { Int(b[ib * 4 + $0]) }
            let index = [qs[0] | ((qh0 << 8) & 0x700), qs[1] | ((qh0 << 4) & 0x700), qs[2] | ((qh1 << 8) & 0x700), qs[3] | ((qh1 << 4) & 0x700)]
            let delta: [Float] = [qh0 & 0x08, qh0 & 0x80, qh1 & 0x08, qh1 & 0x80].map { $0 != 0 ? -0.125 : 0.125 }
            for l in 0 ..< 4 {
                for j in 0 ..< 8 {
                    let value = Float(Int8(truncatingIfNeeded: gridByte(ggmlIQ1SGrid, index[l], j)))
                    y[ib * 32 + l * 8 + j] = dl[l / 2] * (value + delta[l])
                }
            }
        }
    }
}
