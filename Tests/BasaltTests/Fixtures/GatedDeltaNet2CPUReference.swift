import Foundation

struct GatedDeltaNet2CPUShape: Sendable, Equatable {
    let batch: Int
    let steps: Int
    let hk: Int
    let hv: Int
    let dk: Int
    let dv: Int

    init(batch: Int, steps: Int, hk: Int, hv: Int, dk: Int, dv: Int) {
        self.batch = batch
        self.steps = steps
        self.hk = hk
        self.hv = hv
        self.dk = dk
        self.dv = dv
    }
}

struct GatedDeltaNet2CPUInputs: Sendable {
    var shape: GatedDeltaNet2CPUShape
    var q: [Float]
    var k: [Float]
    var v: [Float]
    var decay: [Float]
    var erase: [Float]
    var write: [Float]
    var betaGuard: Bool
    var initialState: [Float]?
    var mask: [Bool]?

    init(
        shape: GatedDeltaNet2CPUShape,
        q: [Float],
        k: [Float],
        v: [Float],
        decay: [Float],
        erase: [Float]
    ) {
        self.shape = shape
        self.q = q
        self.k = k
        self.v = v
        self.decay = decay
        self.erase = erase
        self.write = [Float](repeating: 1, count: shape.batch * shape.steps * shape.hv * shape.dv)
        self.betaGuard = false
        self.initialState = nil
        self.mask = nil
    }

    func withWriteAndState(
        write: [Float],
        betaGuard: Bool,
        initialState: [Float]? = nil,
        mask: [Bool]? = nil
    ) -> GatedDeltaNet2CPUInputs {
        var copy = self
        copy.write = write
        copy.betaGuard = betaGuard
        copy.initialState = initialState
        copy.mask = mask
        return copy
    }
}

struct GatedDeltaNet2CPUOutput: Sendable, Equatable {
    let output: [Float]
    let state: [Float]

    init(output: [Float], state: [Float]) {
        self.output = output
        self.state = state
    }
}

enum GatedDeltaNet2CPUReference {
    static func applyBetaGuard(
        erase: [Double],
        key: [Double],
        decay: [Double]
    ) -> [Double] {
        let dk = erase.count
        var aSum = 0.0
        var sSum = 0.0
        var dMax = 0.0
        for i in 0 ..< dk {
            let k2 = key[i] * key[i]
            aSum += k2
            sSum += erase[i] * k2
            if decay[i] > dMax { dMax = decay[i] }
        }
        guard aSum > 1e-12 else { return erase }
        let clampedD = min(dMax, 1.0)
        let bMean = sSum / aSum
        var varSum = 0.0
        for i in 0 ..< dk {
            let diff = erase[i] - bMean
            varSum += diff * diff * key[i] * key[i]
        }
        let delta2 = aSum * varSum
        let d2 = clampedD * clampedD
        let lhs = d2 * delta2
        let oneMinusS = 1.0 - sSum
        let factor = max(0.0, 1.0 - d2 * oneMinusS * oneMinusS)
        let rhs = max(0.0, (1.0 - d2) * factor)
        guard lhs > rhs && lhs > 0.0 else { return erase }
        let rho = sqrt(rhs / lhs)
        return erase.map { bMean + rho * ($0 - bMean) }
    }

    static func transitionSpectralNorm(
        erase: [Double],
        key: [Double],
        decay: [Double],
        iterations: Int = 64
    ) -> Double {
        let dk = key.count
        guard dk > 0 else { return 0 }
        var mat = [Double](repeating: 0, count: dk * dk)
        for row in 0 ..< dk {
            for col in 0 ..< dk {
                let delta = row == col ? 1.0 : 0.0
                mat[row * dk + col] = (delta - key[row] * erase[col] * key[col]) * decay[col]
            }
        }
        var vec = (0 ..< dk).map { Double($0 + 1) }
        var norm = sqrt(vec.reduce(0.0) { $0 + $1 * $1 })
        for i in 0 ..< dk { vec[i] /= norm }
        var sigma = 0.0
        for _ in 0 ..< max(1, iterations) {
            var y = [Double](repeating: 0, count: dk)
            for r in 0 ..< dk {
                var acc = 0.0
                for c in 0 ..< dk { acc += mat[r * dk + c] * vec[c] }
                y[r] = acc
            }
            sigma = sqrt(y.reduce(0.0) { $0 + $1 * $1 })
            guard sigma > 1e-15 else { return 0 }
            var next = [Double](repeating: 0, count: dk)
            for c in 0 ..< dk {
                var acc = 0.0
                for r in 0 ..< dk { acc += mat[r * dk + c] * y[r] }
                next[c] = acc
            }
            norm = sqrt(next.reduce(0.0) { $0 + $1 * $1 })
            guard norm > 1e-15 else { break }
            for c in 0 ..< dk { vec[c] = next[c] / norm }
        }
        return sigma
    }

    static func forward(_ inputs: GatedDeltaNet2CPUInputs) -> GatedDeltaNet2CPUOutput {
        let s = inputs.shape
        let stateCount = s.batch * s.hv * s.dv * s.dk
        var state = (inputs.initialState ?? [Float](repeating: 0, count: stateCount)).map(Double.init)
        var output = [Float](repeating: 0, count: s.batch * s.steps * s.hv * s.dv)
        let groupSize = max(1, s.hv / max(1, s.hk))
        for b in 0 ..< s.batch {
            for t in 0 ..< s.steps {
                let active = inputs.mask.map { $0[b * s.steps + t] } ?? true
                guard active else { continue }
                for hv in 0 ..< s.hv {
                    stepHead(
                        inputs: inputs,
                        coords: HeadCoords(b: b, t: t, hv: hv, hk: hv / groupSize),
                        state: &state,
                        output: &output
                    )
                }
            }
        }
        return GatedDeltaNet2CPUOutput(output: output, state: state.map(Float.init))
    }

    private struct HeadCoords {
        let b: Int
        let t: Int
        let hv: Int
        let hk: Int
    }

    private static func stepHead(
        inputs: GatedDeltaNet2CPUInputs,
        coords: HeadCoords,
        state: inout [Double],
        output: inout [Float]
    ) {
        let s = inputs.shape
        let qkBase = ((coords.b * s.steps + coords.t) * s.hk + coords.hk) * s.dk
        let gBase = ((coords.b * s.steps + coords.t) * s.hv + coords.hv) * s.dk
        let vBase = ((coords.b * s.steps + coords.t) * s.hv + coords.hv) * s.dv
        let stateHeadBase = (coords.b * s.hv + coords.hv) * s.dv * s.dk

        let key = (0 ..< s.dk).map { Double(inputs.k[qkBase + $0]) }
        let query = (0 ..< s.dk).map { Double(inputs.q[qkBase + $0]) }
        let decay = (0 ..< s.dk).map { Double(inputs.decay[gBase + $0]) }
        let rawErase = (0 ..< s.dk).map { Double(inputs.erase[gBase + $0]) }
        let erase = inputs.betaGuard ? applyBetaGuard(erase: rawErase, key: key, decay: decay) : rawErase

        for dvIdx in 0 ..< s.dv {
            let rowBase = stateHeadBase + dvIdx * s.dk
            var kvMem = 0.0
            for dkIdx in 0 ..< s.dk {
                let decayed = state[rowBase + dkIdx] * decay[dkIdx]
                state[rowBase + dkIdx] = decayed
                kvMem += decayed * (erase[dkIdx] * key[dkIdx])
            }
            let delta = Double(inputs.write[vBase + dvIdx]) * Double(inputs.v[vBase + dvIdx]) - kvMem
            var yVal = 0.0
            for dkIdx in 0 ..< s.dk {
                let updated = state[rowBase + dkIdx] + delta * key[dkIdx]
                state[rowBase + dkIdx] = updated
                yVal += updated * query[dkIdx]
            }
            output[vBase + dvIdx] = Float(yVal)
        }
    }
}
