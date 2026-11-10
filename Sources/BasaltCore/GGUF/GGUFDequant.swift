import Foundation

enum GGUFDequant {
    static func floats(_ raw: UnsafeRawBufferPointer, type: GGMLType, count: Int) -> [Float] {
        [Float](unsafeUninitializedCapacity: count) { out, initialized in
            let (elements, bytes) = type.block
            for block in 0 ..< count / elements {
                let source = UnsafeRawBufferPointer(rebasing: raw[block * bytes ..< (block + 1) * bytes])
                let target = UnsafeMutableBufferPointer(rebasing: out[block * elements ..< (block + 1) * elements])
                decode(source, type: type, into: target)
            }
            initialized = count
        }
    }

    static func decode(_ b: UnsafeRawBufferPointer, type: GGMLType, into y: UnsafeMutableBufferPointer<Float>) {
        switch type {
        case .f32: y[0] = b.loadUnaligned(as: Float.self)
        case .f16: y[0] = half(b, 0)
        case .bf16: y[0] = Float(bitPattern: UInt32(b.loadUnaligned(as: UInt16.self)) << 16)
        case .q8_0: for j in 0 ..< 32 { y[j] = half(b, 0) * Float(Int8(bitPattern: b[2 + j])) }
        case .iq4NL: iq4NL(b, y)
        case .q2K: q2K(b, y)
        case .q4K: q4K(b, y)
        case .q6K: q6K(b, y)
        case .iq4XS: iq4XS(b, y)
        case .iq3S: iq3S(b, y)
        case .iq3XXS: iq3XXS(b, y)
        case .iq2S: iq2S(b, y)
        case .iq2XS: iq2XS(b, y)
        case .iq2XXS: iq2XXS(b, y)
        case .iq1M: iq1M(b, y)
        }
    }

    static func half(_ b: UnsafeRawBufferPointer, _ offset: Int) -> Float {
        Float(Float16(bitPattern: b.loadUnaligned(fromByteOffset: offset, as: UInt16.self)))
    }

    static func u16(_ b: UnsafeRawBufferPointer, _ offset: Int) -> Int { Int(b.loadUnaligned(fromByteOffset: offset, as: UInt16.self)) }

    static func u32(_ b: UnsafeRawBufferPointer, _ offset: Int) -> UInt32 { b.loadUnaligned(fromByteOffset: offset, as: UInt32.self) }

    static func gridByte<T: FixedWidthInteger>(_ grid: [T], _ index: Int, _ byte: Int) -> Int {
        Int(truncatingIfNeeded: grid[index] >> (8 * byte)) & 0xff
    }

    static func sign(_ signs: Int, _ bit: Int) -> Float { signs & (1 << bit) != 0 ? -1 : 1 }

    static func iq4NL(_ b: UnsafeRawBufferPointer, _ y: UnsafeMutableBufferPointer<Float>) {
        let d = half(b, 0)
        for j in 0 ..< 16 {
            y[j] = d * Float(ggmlValuesIQ4NL[Int(b[2 + j] & 0xf)])
            y[j + 16] = d * Float(ggmlValuesIQ4NL[Int(b[2 + j] >> 4)])
        }
    }

    static func q2K(_ b: UnsafeRawBufferPointer, _ y: UnsafeMutableBufferPointer<Float>) {
        let (d, dmin) = (half(b, 80), half(b, 82))
        var (scale, out) = (0, 0)
        for chunk in 0 ..< 2 {
            for shift in stride(from: 0, to: 8, by: 2) {
                for half16 in 0 ..< 2 {
                    let sc = Int(b[scale])
                    scale += 1
                    let (dl, ml) = (d * Float(sc & 0xf), dmin * Float(sc >> 4))
                    for l in 0 ..< 16 { y[out + l] = dl * Float((Int(b[16 + chunk * 32 + half16 * 16 + l]) >> shift) & 3) - ml }
                    out += 16
                }
            }
        }
    }

    static func scaleMin(_ j: Int, _ b: UnsafeRawBufferPointer) -> (Float, Float) {
        let q = { Int(b[4 + $0]) }
        if j < 4 { return (Float(q(j) & 63), Float(q(j + 4) & 63)) }
        return (Float((q(j + 4) & 0xf) | ((q(j - 4) >> 6) << 4)), Float((q(j + 4) >> 4) | ((q(j) >> 6) << 4)))
    }

    static func q4K(_ b: UnsafeRawBufferPointer, _ y: UnsafeMutableBufferPointer<Float>) {
        let (d, dmin) = (half(b, 0), half(b, 2))
        for group in 0 ..< 4 {
            let ((s1, m1), (s2, m2)) = (scaleMin(2 * group, b), scaleMin(2 * group + 1, b))
            for l in 0 ..< 32 {
                let q = Int(b[16 + group * 32 + l])
                y[group * 64 + l] = d * s1 * Float(q & 0xf) - dmin * m1
                y[group * 64 + 32 + l] = d * s2 * Float(q >> 4) - dmin * m2
            }
        }
    }

    static func q6K(_ b: UnsafeRawBufferPointer, _ y: UnsafeMutableBufferPointer<Float>) {
        let d = half(b, 208)
        for chunk in 0 ..< 2 {
            let (ql, qh, sc) = (chunk * 64, 128 + chunk * 32, 192 + chunk * 8)
            for l in 0 ..< 32 {
                let s = { Float(Int8(bitPattern: b[sc + l / 16 + $0])) }
                let low = Int(b[ql + l]), high = Int(b[ql + l + 32]), top = Int(b[qh + l])
                y[chunk * 128 + l] = d * s(0) * Float(((low & 0xf) | ((top & 3) << 4)) - 32)
                y[chunk * 128 + l + 32] = d * s(2) * Float(((high & 0xf) | (((top >> 2) & 3) << 4)) - 32)
                y[chunk * 128 + l + 64] = d * s(4) * Float(((low >> 4) | (((top >> 4) & 3) << 4)) - 32)
                y[chunk * 128 + l + 96] = d * s(6) * Float(((high >> 4) | (((top >> 6) & 3) << 4)) - 32)
            }
        }
    }

    static func iq4XS(_ b: UnsafeRawBufferPointer, _ y: UnsafeMutableBufferPointer<Float>) {
        let (d, high) = (half(b, 0), u16(b, 2))
        for ib in 0 ..< 8 {
            let ls = ((Int(b[4 + ib / 2]) >> (4 * (ib % 2))) & 0xf) | (((high >> (2 * ib)) & 3) << 4)
            let dl = d * Float(ls - 32)
            for j in 0 ..< 16 {
                let q = Int(b[8 + ib * 16 + j])
                y[ib * 32 + j] = dl * Float(ggmlValuesIQ4NL[q & 0xf])
                y[ib * 32 + j + 16] = dl * Float(ggmlValuesIQ4NL[q >> 4])
            }
        }
    }
}
