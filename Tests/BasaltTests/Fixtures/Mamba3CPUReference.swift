import Foundation

struct Mamba3CPUShape: Sendable, Equatable {
    let batch: Int
    let steps: Int
    let heads: Int
    let headDim: Int
    let dState: Int
    let mimoRank: Int
    var ropeFraction: Double = 1.0

    var ropeDim: Int { min(dState, max(0, 2 * (Int((Double(dState) * min(max(ropeFraction, 0.0), 1.0)).rounded(.down)) / 2))) }
    var numRopeAngles: Int { ropeDim / 2 }
    var isMIMO: Bool { mimoRank > 1 }
}

struct Mamba3CPUState: Sendable, Equatable {
    var angleState: [Float]
    var ssmState: [Float]
    var kState: [Float]
    var vState: [Float]

    init(angleState: [Float], ssmState: [Float], kState: [Float], vState: [Float]) {
        self.angleState = angleState
        self.ssmState = ssmState
        self.kState = kState
        self.vState = vState
    }

    static func zeros(shape: Mamba3CPUShape) -> Mamba3CPUState {
        let bh = shape.batch * shape.heads
        return Mamba3CPUState(
            angleState: [Float](repeating: 0, count: bh * shape.numRopeAngles),
            ssmState: [Float](repeating: 0, count: bh * shape.headDim * shape.dState),
            kState: [Float](repeating: 0, count: bh * shape.mimoRank * shape.dState),
            vState: [Float](repeating: 0, count: bh * shape.mimoRank * shape.headDim)
        )
    }
}

struct Mamba3CPUParameters: Sendable {
    var dtBias: [Float]
    var aFloor: Float
    var dSkip: [Float]
    var mimoX: [Float]?
    var mimoZ: [Float]?
    var mimoO: [Float]?
    var heavyTail: Bool

    init(
        dtBias: [Float],
        aFloor: Float = 1e-4,
        dSkip: [Float],
        mimoX: [Float]? = nil,
        mimoZ: [Float]? = nil,
        mimoO: [Float]? = nil
    ) {
        self.dtBias = dtBias
        self.aFloor = aFloor
        self.dSkip = dSkip
        self.mimoX = mimoX
        self.mimoZ = mimoZ
        self.mimoO = mimoO
        self.heavyTail = true
    }

    func withHeavyTail(_ enabled: Bool) -> Mamba3CPUParameters {
        var copy = self
        copy.heavyTail = enabled
        return copy
    }
}

struct Mamba3CPUInputs: Sendable {
    var shape: Mamba3CPUShape
    var x: [Float]
    var z: [Float]
    var b: [Float]
    var c: [Float]
    var ddDt: [Float]
    var ddA: [Float]
    var trap: [Float]
    var angles: [Float]
    var parameters: Mamba3CPUParameters
    var initialState: Mamba3CPUState?
    var mask: [Bool]?

    init(
        shape: Mamba3CPUShape,
        x: [Float],
        z: [Float],
        b: [Float],
        c: [Float],
        parameters: Mamba3CPUParameters
    ) {
        self.shape = shape
        self.x = x
        self.z = z
        self.b = b
        self.c = c
        let bth = shape.batch * shape.steps * shape.heads
        self.ddDt = [Float](repeating: 0, count: bth)
        self.ddA = [Float](repeating: 0, count: bth)
        self.trap = [Float](repeating: 0, count: bth)
        self.angles = [Float](repeating: 0, count: bth * shape.numRopeAngles)
        self.parameters = parameters
        self.initialState = nil
        self.mask = nil
    }

    func withControls(
        ddDt: [Float],
        ddA: [Float],
        trap: [Float],
        angles: [Float],
        initialState: Mamba3CPUState? = nil,
        mask: [Bool]? = nil
    ) -> Mamba3CPUInputs {
        var copy = self
        copy.ddDt = ddDt
        copy.ddA = ddA
        copy.trap = trap
        copy.angles = angles
        copy.initialState = initialState
        copy.mask = mask
        return copy
    }
}

struct Mamba3CPUOutput: Sendable, Equatable {
    let output: [Float]
    let state: Mamba3CPUState

    init(output: [Float], state: Mamba3CPUState) {
        self.output = output
        self.state = state
    }
}

enum Mamba3CPUReference {
    struct Discretization: Sendable, Equatable {
        let dt: Double
        let alpha: Double
        let beta: Double
        let gamma: Double
    }

    static func heavyTail(_ x: Double) -> Double {
        x >= 0 ? 1.0 + x : 1.0 / (1.0 - x)
    }

    static func softplus(_ x: Double) -> Double {
        x > 20.0 ? x : log1p(exp(x))
    }

    static func sigmoid(_ x: Double) -> Double {
        1.0 / (1.0 + exp(-x))
    }

    static func silu(_ x: Double) -> Double {
        x * sigmoid(x)
    }

    static func discretize(
        ddDt: Double,
        dtBias: Double,
        ddA: Double,
        aFloor: Double,
        trap: Double,
        heavyTail useHeavyTail: Bool
    ) -> Discretization {
        let dt = softplus(ddDt + dtBias)
        let aMagnitude = useHeavyTail ? heavyTail(ddA) : softplus(ddA)
        let aNeg = min(-aMagnitude, -aFloor)
        let alpha = exp(dt * aNeg)
        let lambda = sigmoid(trap)
        let beta = (1.0 - lambda) * dt * alpha
        let gamma = lambda * dt
        return Discretization(dt: dt, alpha: alpha, beta: beta, gamma: gamma)
    }

    static func applyRoPE(
        _ vector: [Double],
        cos cosValues: [Double],
        sin sinValues: [Double],
        splitHalf: Bool
    ) -> [Double] {
        let half = cosValues.count
        var out = vector
        guard half > 0 else { return out }
        if splitHalf {
            let halfState = vector.count / 2
            for i in 0 ..< half {
                let (v1, v2) = (vector[i], vector[i + halfState])
                out[i] = v1 * cosValues[i] - v2 * sinValues[i]
                out[i + halfState] = v1 * sinValues[i] + v2 * cosValues[i]
            }
        } else {
            for i in 0 ..< half {
                let (v1, v2) = (vector[2 * i], vector[2 * i + 1])
                out[2 * i] = v1 * cosValues[i] - v2 * sinValues[i]
                out[2 * i + 1] = v1 * sinValues[i] + v2 * cosValues[i]
            }
        }
        return out
    }

    static func forward(_ inputs: Mamba3CPUInputs) -> Mamba3CPUOutput {
        let s = inputs.shape
        let initState = inputs.initialState ?? Mamba3CPUState.zeros(shape: s)
        var work = WorkBuffers(
            angle: initState.angleState.map(Double.init),
            ssm: initState.ssmState.map(Double.init),
            k: initState.kState.map(Double.init),
            v: initState.vState.map(Double.init),
            out: [Float](repeating: 0, count: s.batch * s.steps * s.heads * s.headDim)
        )
        for b in 0 ..< s.batch {
            for t in 0 ..< s.steps {
                let active = inputs.mask.map { $0[b * s.steps + t] } ?? true
                guard active else { continue }
                for h in 0 ..< s.heads {
                    stepHead(inputs: inputs, b: b, t: t, h: h, work: &work)
                }
            }
        }
        return Mamba3CPUOutput(
            output: work.out,
            state: Mamba3CPUState(
                angleState: work.angle.map(Float.init),
                ssmState: work.ssm.map(Float.init),
                kState: work.k.map(Float.init),
                vState: work.v.map(Float.init)
            )
        )
    }
}

extension Mamba3CPUReference {
    private struct WorkBuffers {
        var angle: [Double]
        var ssm: [Double]
        var k: [Double]
        var v: [Double]
        var out: [Float]
    }

    private struct StepVectors {
        let kCurr: [Double]
        let qCurr: [Double]
        let vCurr: [Double]
    }

    private static func stepHead(
        inputs: Mamba3CPUInputs,
        b: Int,
        t: Int,
        h: Int,
        work: inout WorkBuffers
    ) {
        let s = inputs.shape
        let bth = (b * s.steps + t) * s.heads + h
        let bh = b * s.heads + h
        let disc = discretize(
            ddDt: Double(inputs.ddDt[bth]),
            dtBias: Double(inputs.parameters.dtBias[h]),
            ddA: Double(inputs.ddA[bth]),
            aFloor: Double(inputs.parameters.aFloor),
            trap: Double(inputs.trap[bth]),
            heavyTail: inputs.parameters.heavyTail
        )
        let (cosVals, sinVals) = advanceAngles(inputs: inputs, bh: bh, bth: bth, dt: disc.dt, angle: &work.angle)
        let vecs = buildStepVectors(inputs: inputs, h: h, bth: bth, cosVals: cosVals, sinVals: sinVals)
        updateSSM(shape: s, bh: bh, disc: disc, vecs: vecs, work: &work)
        readout(inputs: inputs, bh: bh, bth: bth, h: h, vecs: vecs, work: &work)
    }

    private static func advanceAngles(
        inputs: Mamba3CPUInputs,
        bh: Int,
        bth: Int,
        dt: Double,
        angle: inout [Double]
    ) -> ([Double], [Double]) {
        let half = inputs.shape.numRopeAngles
        let angleBase = bh * half
        let inBase = bth * half
        var cosVals = [Double](repeating: 0, count: half)
        var sinVals = [Double](repeating: 0, count: half)
        for i in 0 ..< half {
            let nextPhi = angle[angleBase + i] + Double.pi * Foundation.tanh(Double(inputs.angles[inBase + i])) * dt
            angle[angleBase + i] = nextPhi
            cosVals[i] = Foundation.cos(nextPhi)
            sinVals[i] = Foundation.sin(nextPhi)
        }
        return (cosVals, sinVals)
    }

    private static func buildStepVectors(
        inputs: Mamba3CPUInputs,
        h: Int,
        bth: Int,
        cosVals: [Double],
        sinVals: [Double]
    ) -> StepVectors {
        let s = inputs.shape
        let (p, n, r) = (s.headDim, s.dState, s.mimoRank)
        var kCurr = [Double](repeating: 0, count: r * n)
        var qCurr = [Double](repeating: 0, count: r * n)
        let bcBase = bth * r * n
        for rank in 0 ..< r {
            let off = rank * n
            let rawK = (0 ..< n).map { Double(inputs.b[bcBase + off + $0]) }
            let rawQ = (0 ..< n).map { Double(inputs.c[bcBase + off + $0]) }
            let rotK = applyRoPE(rawK, cos: cosVals, sin: sinVals, splitHalf: s.isMIMO)
            let rotQ = applyRoPE(rawQ, cos: cosVals, sin: sinVals, splitHalf: s.isMIMO)
            for j in 0 ..< n {
                kCurr[off + j] = rotK[j]
                qCurr[off + j] = rotQ[j]
            }
        }
        var vCurr = [Double](repeating: 0, count: r * p)
        let xvBase = bth * p
        if s.isMIMO, let mimoX = inputs.parameters.mimoX {
            let mBase = h * r * p
            for rank in 0 ..< r {
                for j in 0 ..< p {
                    vCurr[rank * p + j] = Double(inputs.x[xvBase + j]) * Double(mimoX[mBase + rank * p + j])
                }
            }
        } else {
            for j in 0 ..< p { vCurr[j] = Double(inputs.x[xvBase + j]) }
        }
        return StepVectors(kCurr: kCurr, qCurr: qCurr, vCurr: vCurr)
    }

    private static func updateSSM(
        shape s: Mamba3CPUShape,
        bh: Int,
        disc: Discretization,
        vecs: StepVectors,
        work: inout WorkBuffers
    ) {
        let (p, n, r) = (s.headDim, s.dState, s.mimoRank)
        let ssmBase = bh * p * n
        let kBase = bh * r * n
        let vBase = bh * r * p
        for row in 0 ..< p {
            let rowOff = ssmBase + row * n
            for col in 0 ..< n {
                var uPrev = 0.0
                var uCurr = 0.0
                for rank in 0 ..< r {
                    uPrev += work.v[vBase + rank * p + row] * work.k[kBase + rank * n + col]
                    uCurr += vecs.vCurr[rank * p + row] * vecs.kCurr[rank * n + col]
                }
                work.ssm[rowOff + col] = disc.alpha * work.ssm[rowOff + col] + disc.beta * uPrev + disc.gamma * uCurr
            }
        }
        for idx in 0 ..< (r * n) { work.k[kBase + idx] = vecs.kCurr[idx] }
        for idx in 0 ..< (r * p) { work.v[vBase + idx] = vecs.vCurr[idx] }
    }

    private static func readout(
        inputs: Mamba3CPUInputs,
        bh: Int,
        bth: Int,
        h: Int,
        vecs: StepVectors,
        work: inout WorkBuffers
    ) {
        let s = inputs.shape
        let (p, n, r) = (s.headDim, s.dState, s.mimoRank)
        let ssmBase = bh * p * n
        let xvBase = bth * p
        let dSkip = Double(inputs.parameters.dSkip[h])
        let mBase = h * r * p
        for row in 0 ..< p {
            let rowOff = ssmBase + row * n
            let zScalar = Double(inputs.z[xvBase + row])
            var acc = 0.0
            for rank in 0 ..< r {
                var dot = 0.0
                for col in 0 ..< n { dot += work.ssm[rowOff + col] * vecs.qCurr[rank * n + col] }
                let yRank = dot + dSkip * vecs.vCurr[rank * p + row]
                if s.isMIMO, let mimoZ = inputs.parameters.mimoZ, let mimoO = inputs.parameters.mimoO {
                    let zRank = zScalar * Double(mimoZ[mBase + rank * p + row])
                    acc += (yRank * silu(zRank)) * Double(mimoO[mBase + rank * p + row])
                } else {
                    acc += yRank * silu(zScalar)
                }
            }
            work.out[xvBase + row] = Float(acc)
        }
    }
}
