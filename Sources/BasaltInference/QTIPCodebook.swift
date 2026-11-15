import Foundation
import MLX

package enum QTIPRate: Int, CaseIterable, Sendable {
    case twoBPW = 4
    case twoHalfBPW = 5
    case threeBPW = 6

    package init?(bpw: Double) {
        switch bpw {
        case 2.0: self = .twoBPW
        case 2.5: self = .twoHalfBPW
        case 3.0: self = .threeBPW
        default: return nil
        }
    }

    package var stepBits: Int { rawValue }
    package var bpw: Double { Double(rawValue) * 0.5 }
    package var stateBits: Int { 16 - rawValue }
    package var numStates: Int { 1 << stateBits }
    package var fanout: Int { 1 << rawValue }
    package var wordsPerTile: Int { (128 * rawValue) / 32 }

    package var codebookScale: Float {
        switch self {
        case .twoBPW: 0.96824584
        case .twoHalfBPW: 0.98425098
        case .threeBPW: 0.99215674
        }
    }

    package var label: String {
        switch self {
        case .twoBPW: "2.0"
        case .twoHalfBPW: "2.5"
        case .threeBPW: "3.0"
        }
    }
}

package enum QTIPCodebook {
    static let ringCounts: [Int] = [
        4, 8, 12, 16, 20, 24, 28, 30, 32, 32, 30, 28, 26, 24, 22, 20,
        18, 16, 16, 14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 2
    ]

    @inline(__always)
    package static func perm16(_ edge: UInt16) -> UInt32 {
        var h = UInt32(edge)
        h = ((h ^ (h >> 8)) &* 0x9E37) & 0xFFFF
        h = ((h ^ (h >> 7)) &* 0xC2B5) & 0xFFFF
        h ^= (h >> 9)
        return h
    }

    @inline(__always)
    package static func edgeCodeIndex(_ edge: UInt16) -> UInt16 {
        UInt16(perm16(edge) & 2047)
    }

    package static func edgeCodeTableValues() -> [UInt16] {
        (0 ..< 65536).map { edgeCodeIndex(UInt16($0)) }
    }

    package static func baseLUTValues(rate: QTIPRate) -> [Float] {
        let sigma = Double(rate.codebookScale)
        var out: [Float] = []
        out.reserveCapacity(1024)
        var cumCells = 0
        let quad = 8
        for count in ringCounts {
            let u0 = Double(cumCells) / 512.0
            let u1 = Double(cumCells + count) / 512.0
            cumCells += count
            var rMean = 0.0
            for qr in 0 ..< quad {
                let u = min(0.9999995, u0 + (u1 - u0) * (Double(qr) + 0.5) / Double(quad))
                rMean += sigma * (-2.0 * Foundation.log(1.0 - u)).squareRoot()
            }
            rMean /= Double(quad)
            let dTheta = (Double.pi * 0.5) / Double(count)
            let arcFactor = (dTheta > 0) ? (2.0 * Foundation.sin(dTheta * 0.5) / dTheta) : 1.0
            for sector in 0 ..< count {
                let midTheta = (Double(sector) + 0.5) * dTheta
                let x = rMean * arcFactor * Foundation.cos(midTheta)
                let y = rMean * arcFactor * Foundation.sin(midTheta)
                out.append(Float(x))
                out.append(Float(y))
            }
        }
        return out
    }

    @inline(__always)
    package static func lookupCodeIndex(_ codeIdx: UInt16, tlut: [Float]) -> (Float, Float) {
        let idx = Int(codeIdx & 511)
        let sx: Float = (codeIdx & 512) != 0 ? -1.0 : 1.0
        let sy: Float = (codeIdx & 1024) != 0 ? -1.0 : 1.0
        return (tlut[2 * idx] * sx, tlut[2 * idx + 1] * sy)
    }

    @inline(__always)
    package static func lookup(edge: UInt16, tlut: [Float]) -> (Float, Float) {
        lookupCodeIndex(edgeCodeIndex(edge), tlut: tlut)
    }
}

package enum QTIPReference {
    @inline(__always)
    package static func readEdge(words: [UInt32], step: Int, rate: QTIPRate) -> UInt16 {
        let bitPos = step * rate.stepBits
        let wIdx = bitPos >> 5
        let shift = bitPos & 31
        let lo = UInt64(words[wIdx])
        let hi = UInt64(words[(wIdx + 1) % rate.wordsPerTile])
        return UInt16(((lo | (hi << 32)) >> UInt64(shift)) & 0xFFFF)
    }

    package static func decodeTile(words: [UInt32], tlut: [Float], rate: QTIPRate, scale: Float = 1.0) -> [Float] {
        var out = [Float](repeating: 0, count: 256)
        for step in 0 ..< 128 {
            let edge = readEdge(words: words, step: step, rate: rate)
            let (c0, c1) = QTIPCodebook.lookup(edge: edge, tlut: tlut)
            out[2 * step] = c0 * scale
            out[2 * step + 1] = c1 * scale
        }
        return out
    }

    static func stepDistances(w0: Float, w1: Float, tlut: [Float]) -> [Float] {
        var dist = [Float](repeating: 0, count: 2048)
        for idx in 0 ..< 512 {
            let ux = tlut[2 * idx]
            let uy = tlut[2 * idx + 1]
            let norm2 = 0.5 * (ux * ux + uy * uy)
            let dx = w0 * ux
            let dy = w1 * uy
            dist[idx] = norm2 - dx - dy
            dist[idx + 512] = norm2 + dx - dy
            dist[idx + 1024] = norm2 - dx + dy
            dist[idx + 1536] = norm2 + dx + dy
        }
        return dist
    }

    package static func encodeTile(
        tile: [Float], tlut: [Float], edgeCodes: [UInt16], rate: QTIPRate, burnInSteps: Int = 24
    ) -> (words: [UInt32], decoded: [Float]) {
        let numStates = rate.numStates
        let stateMask = numStates - 1
        let k = rate.stepBits
        let fanout = rate.fanout
        var cost = [Float](repeating: 0, count: numStates)
        var nextCost = [Float](repeating: 0, count: numStates)
        for step in (128 - burnInSteps) ..< 128 {
            let dist = stepDistances(w0: tile[2 * step], w1: tile[2 * step + 1], tlut: tlut)
            for sNext in 0 ..< numStates {
                let baseEdge = sNext << k
                var best: Float = 1e20
                for b in 0 ..< fanout {
                    let edge = baseEdge | b
                    let cand = cost[edge & stateMask] + dist[Int(edgeCodes[edge])]
                    if cand < best { best = cand }
                }
                nextCost[sNext] = best
            }
            swap(&cost, &nextCost)
        }
        var s0 = 0
        for s in 1 ..< numStates where cost[s] < cost[s0] { s0 = s }
        return pinnedViterbi(tile: tile, tlut: tlut, edgeCodes: edgeCodes, rate: rate, s0: s0)
    }

    private static func pinnedViterbi(
        tile: [Float], tlut: [Float], edgeCodes: [UInt16], rate: QTIPRate, s0: Int
    ) -> (words: [UInt32], decoded: [Float]) {
        let numStates = rate.numStates
        let stateMask = numStates - 1
        let k = rate.stepBits
        let fanout = rate.fanout
        var cost = [Float](repeating: 1e20, count: numStates)
        cost[s0] = 0
        var nextCost = [Float](repeating: 1e20, count: numStates)
        var backptr = [UInt8](repeating: 0, count: 128 * numStates)
        for step in 0 ..< 128 {
            let dist = stepDistances(w0: tile[2 * step], w1: tile[2 * step + 1], tlut: tlut)
            let rowBase = step * numStates
            for sNext in 0 ..< numStates {
                let baseEdge = sNext << k
                var best: Float = 1e20
                var bestB: UInt8 = 0
                for b in 0 ..< fanout {
                    let edge = baseEdge | b
                    let cand = cost[edge & stateMask] + dist[Int(edgeCodes[edge])]
                    if cand < best {
                        best = cand
                        bestB = UInt8(b)
                    }
                }
                nextCost[sNext] = best
                backptr[rowBase + sNext] = bestB
            }
            swap(&cost, &nextCost)
        }
        var words = [UInt32](repeating: 0, count: rate.wordsPerTile)
        var cur = s0
        for step in stride(from: 127, through: 0, by: -1) {
            let b = UInt32(backptr[step * numStates + cur])
            let bitPos = step * k
            let wIdx = bitPos >> 5
            let shift = bitPos & 31
            words[wIdx] |= (b << UInt32(shift))
            if shift + k > 32 {
                words[wIdx + 1] |= (b >> UInt32(32 - shift))
            }
            cur = ((cur << k) | Int(b)) & stateMask
        }
        return (words, decodeTile(words: words, tlut: tlut, rate: rate))
    }
}

package struct KroneckerRHT: Sendable, Equatable {
    package let dimension: Int
    package let seed: UInt64
    package let outer: Int
    package let hadamardWidth: Int
    package let hadamardScale: Float
    package let signsValues: [Float]
    package let packedSignsValues: [UInt32]
    package let qValues: [Float]?

    package var spec: FusedKroneckerRHT.Spec {
        FusedKroneckerRHT.Spec(self)
    }

    package static func decompose(_ d: Int) -> (outer: Int, hadamardWidth: Int)? {
        guard d >= 16, d.isMultiple(of: 16) else { return nil }
        let tz = d.trailingZeroBitCount
        let odd = d >> tz
        guard tz >= 4, odd <= 64 else { return nil }
        return (odd, 1 << tz)
    }

    package static func supports(_ d: Int) -> Bool {
        decompose(d) != nil
    }

    package init?(dimension: Int, seed: UInt64) {
        guard let (outer, hadamardWidth) = Self.decompose(dimension) else { return nil }
        let generated = Self.generateFactors(dimension: dimension, outer: outer, seed: seed)
        self.dimension = dimension
        self.seed = seed
        self.outer = outer
        self.hadamardWidth = hadamardWidth
        self.hadamardScale = 1.0 / Float(hadamardWidth).squareRoot()
        self.signsValues = generated.signs
        self.packedSignsValues = FusedKroneckerRHT.Spec.packSigns(generated.signs)
        self.qValues = generated.qValues
    }

    package static func generateFactors(
        dimension: Int, outer: Int, seed: UInt64
    ) -> (signs: [Float], qValues: [Float]?) {
        var state = seed ^ (UInt64(dimension) &* 0x9E37_79B9_7F4A_7C15)
        var signs = [Float](repeating: 1.0, count: dimension)
        for idx in 0 ..< dimension {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            z ^= (z >> 31)
            signs[idx] = (z & 1) == 0 ? 1.0 : -1.0
        }
        return (signs, outer > 1 ? orthogonalFactor(size: outer, seed: state) : nil)
    }

    private static func orthogonalFactor(size: Int, seed: UInt64) -> [Float] {
        var state = seed
        var mat = [Float](repeating: 0, count: size * size)
        for idx in stride(from: 0, to: mat.count, by: 2) {
            let u1 = max(1e-7, nextUniform(&state))
            let u2 = nextUniform(&state)
            let r = (-2.0 * Foundation.log(u1)).squareRoot()
            let theta = 2.0 * Double.pi * u2
            mat[idx] = Float(r * Foundation.cos(theta))
            if idx + 1 < mat.count { mat[idx + 1] = Float(r * Foundation.sin(theta)) }
        }
        let g = MLXArray(mat, [size, size])
        let (qRaw, rRaw) = MLXLinalg.qr(g, stream: .cpu)
        eval(qRaw, rRaw)
        let rArr = rRaw.asArray(Float.self)
        var diagSigns = [Float](repeating: 1, count: size)
        for idx in 0 ..< size {
            diagSigns[idx] = rArr[idx * size + idx] < 0 ? -1.0 : 1.0
        }
        let qFixed = qRaw * MLXArray(diagSigns).reshaped(1, size)
        eval(qFixed)
        return qFixed.asArray(Float.self)
    }

    private static func nextUniform(_ state: inout UInt64) -> Double {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= (z >> 31)
        return Double(z >> 11) * (1.0 / 9007199254740992.0)
    }

    package func forwardFloat32(_ x: MLXArray, signs: MLXArray, qMatrix: MLXArray?) -> MLXArray {
        let origShape = x.shape
        var y = hadamardTransform(
            (x.asType(.float32) * signs).reshaped(-1, hadamardWidth),
            scale: hadamardScale
        )
        if let qMatrix, outer > 1 {
            let batch = y.size / dimension
            y = matmul(
                y.reshaped(batch, outer, hadamardWidth).transposed(0, 2, 1),
                qMatrix.T
            ).transposed(0, 2, 1)
        }
        return y.reshaped(origShape)
    }

    package func inverseFloat32(_ y: MLXArray, signs: MLXArray, qMatrix: MLXArray?) -> MLXArray {
        let origShape = y.shape
        var z = y.asType(.float32)
        if let qMatrix, outer > 1 {
            let batch = z.size / dimension
            z = matmul(
                z.reshaped(batch, outer, hadamardWidth).transposed(0, 2, 1),
                qMatrix
            ).transposed(0, 2, 1)
        }
        z = hadamardTransform(z.reshaped(-1, hadamardWidth), scale: hadamardScale).reshaped(origShape)
        return z * signs
    }
}
