import Foundation
import Testing
import MLX
import MLXNN
import MLXOptimizers
@testable import BasaltCore
@testable import BasaltInference

private final class SOAPMatrixModule: Module {
    var weight: MLXArray

    init(weight: MLXArray) {
        self.weight = weight
        super.init()
    }
}

private final class SOAPMixedModule: Module {
    var oneSidedTall: MLXArray
    var twoSided: MLXArray
    var oversized: MLXArray
    var bias: MLXArray

    init(oneSidedTall: MLXArray, twoSided: MLXArray, oversized: MLXArray, bias: MLXArray) {
        self.oneSidedTall = oneSidedTall
        self.twoSided = twoSided
        self.oversized = oversized
        self.bias = bias
        super.init()
    }
}

private enum CPULinalg {
    static func matMul(_ a: [Double], _ rowsA: Int, _ colsA: Int, _ b: [Double], _ colsB: Int) -> [Double] {
        var out = Array(repeating: 0.0, count: rowsA * colsB)
        for i in 0 ..< rowsA {
            for k in 0 ..< colsA {
                let aik = a[i * colsA + k]
                for j in 0 ..< colsB {
                    out[i * colsB + j] += aik * b[k * colsB + j]
                }
            }
        }
        return out
    }

    static func transpose(_ a: [Double], rows: Int, cols: Int) -> [Double] {
        var out = Array(repeating: 0.0, count: rows * cols)
        for i in 0 ..< rows {
            for j in 0 ..< cols {
                out[j * rows + i] = a[i * cols + j]
            }
        }
        return out
    }

    static func identity(_ dim: Int) -> [Double] {
        var out = Array(repeating: 0.0, count: dim * dim)
        for i in 0 ..< dim { out[i * dim + i] = 1.0 }
        return out
    }

    static func symmetrize(_ m: [Double], dim: Int, damping: Double) -> [Double] {
        var out = Array(repeating: 0.0, count: dim * dim)
        for i in 0 ..< dim {
            for j in 0 ..< dim {
                out[i * dim + j] = 0.5 * (m[i * dim + j] + m[j * dim + i]) + (i == j ? damping : 0.0)
            }
        }
        return out
    }

    static func jacobiEigh(_ sym: [Double], dim: Int) -> [Double] {
        var a = sym
        var v = identity(dim)
        for _ in 0 ..< 60 {
            var maxOff = 0.0
            for p in 0 ..< dim {
                for q in (p + 1) ..< dim {
                    let apq = a[p * dim + q]
                    maxOff = max(maxOff, abs(apq))
                    if abs(apq) < 1e-14 { continue }
                    applyJacobiRotation(&a, &v, dim: dim, p: p, q: q)
                }
            }
            if maxOff < 1e-13 { break }
        }
        let eigvals = (0 ..< dim).map { a[$0 * dim + $0] }
        let order = (0 ..< dim).sorted { eigvals[$0] > eigvals[$1] }
        return canonicalizeColumns(reorderColumns(v, dim: dim, order: order), dim: dim)
    }

    private static func applyJacobiRotation(_ a: inout [Double], _ v: inout [Double], dim: Int, p: Int, q: Int) {
        let app = a[p * dim + p]
        let aqq = a[q * dim + q]
        let apq = a[p * dim + q]
        let tau = (aqq - app) / (2.0 * apq)
        let t = (tau >= 0 ? 1.0 : -1.0) / (abs(tau) + sqrt(1.0 + tau * tau))
        let c = 1.0 / sqrt(1.0 + t * t)
        let s = t * c
        for k in 0 ..< dim where k != p && k != q {
            let akp = a[k * dim + p]
            let akq = a[k * dim + q]
            let np = c * akp - s * akq
            let nq = s * akp + c * akq
            a[k * dim + p] = np; a[p * dim + k] = np
            a[k * dim + q] = nq; a[q * dim + k] = nq
        }
        a[p * dim + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        a[q * dim + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        a[p * dim + q] = 0.0; a[q * dim + p] = 0.0
        for k in 0 ..< dim {
            let vkp = v[k * dim + p]
            let vkq = v[k * dim + q]
            v[k * dim + p] = c * vkp - s * vkq
            v[k * dim + q] = s * vkp + c * vkq
        }
    }

    static func reorderColumns(_ mat: [Double], dim: Int, order: [Int]) -> [Double] {
        var out = Array(repeating: 0.0, count: dim * dim)
        for row in 0 ..< dim {
            for (newCol, oldCol) in order.enumerated() {
                out[row * dim + newCol] = mat[row * dim + oldCol]
            }
        }
        return out
    }

    static func canonicalizeColumns(_ mat: [Double], dim: Int) -> [Double] {
        var out = mat
        for col in 0 ..< dim {
            var bestRow = 0
            var bestAbs = -1.0
            for row in 0 ..< dim {
                let mag = abs(out[row * dim + col])
                if mag > bestAbs { bestAbs = mag; bestRow = row }
            }
            if out[bestRow * dim + col] < 0 {
                for row in 0 ..< dim { out[row * dim + col] = -out[row * dim + col] }
            }
        }
        return out
    }

    static func mgsQR(_ mat: [Double], dim: Int) -> [Double] {
        var q = Array(repeating: 0.0, count: dim * dim)
        for col in 0 ..< dim {
            var v = (0 ..< dim).map { mat[$0 * dim + col] }
            for _ in 0 ..< 2 {
                for prev in 0 ..< col {
                    var dot = 0.0
                    for row in 0 ..< dim { dot += q[row * dim + prev] * v[row] }
                    for row in 0 ..< dim { v[row] -= dot * q[row * dim + prev] }
                }
            }
            let norm = sqrt(v.reduce(0.0) { $0 + $1 * $1 })
            let inv = norm > 0 ? (1.0 / norm) : 0.0
            for row in 0 ..< dim { q[row * dim + col] = v[row] * inv }
        }
        return q
    }
}

private struct CPUSOAPReference {
    let rows: Int
    let cols: Int
    let lr: Double
    let beta1: Double
    let beta2: Double
    let shampooBeta: Double
    let eps: Double
    let weightDecay: Double
    let precondFreq: Int
    let useLeft: Bool
    let useRight: Bool
    var step = 0
    var w: [Double]
    var m: [Double]
    var v: [Double]
    var lMat: [Double]?
    var rMat: [Double]?
    var qL: [Double]?
    var qR: [Double]?

    init(
        initial: [Float],
        rows: Int,
        cols: Int,
        lr: Float,
        options: TrainingOptimizerOptions
    ) {
        self.rows = rows
        self.cols = cols
        self.lr = Double(lr)
        self.beta1 = Double(options.soapBeta1)
        self.beta2 = Double(options.soapBeta2)
        self.shampooBeta = Double(options.soapShampooBeta)
        self.eps = Double(options.soapEps)
        self.weightDecay = Double(options.weightDecay)
        self.precondFreq = options.soapPrecondFreq
        let plan = MatrixOptimizerMath.soapAxisPlan(rows: rows, cols: cols, maxPrecondDim: options.soapMaxPrecondDim)
        self.useLeft = plan.useLeft
        self.useRight = plan.useRight
        self.w = initial.map(Double.init)
        self.m = Array(repeating: 0.0, count: rows * cols)
        self.v = Array(repeating: 0.0, count: rows * cols)
        if useLeft {
            self.lMat = Array(repeating: 0.0, count: rows * rows)
            self.qL = CPULinalg.identity(rows)
        }
        if useRight {
            self.rMat = Array(repeating: 0.0, count: cols * cols)
            self.qR = CPULinalg.identity(cols)
        }
    }

    mutating func apply(gradient: [Float]) {
        step += 1
        let g = gradient.map(Double.init)
        updatePreconditioners(g)
        refreshBasesIfDue()
        for i in 0 ..< m.count { m[i] = beta1 * m[i] + (1.0 - beta1) * g[i] }
        let gPrime = projectTo(g)
        for i in 0 ..< v.count { v[i] = beta2 * v[i] + (1.0 - beta2) * gPrime[i] * gPrime[i] }
        let mPrime = projectTo(m)
        let c1 = lr / (1.0 - pow(beta1, Double(step)))
        let c2 = 1.0 / sqrt(1.0 - pow(beta2, Double(step)))
        let stepPrime = (0 ..< v.count).map { (c1 * mPrime[$0]) / (sqrt(v[$0]) * c2 + eps) }
        let stepOrig = projectFrom(stepPrime)
        for i in 0 ..< w.count {
            let decayed = weightDecay != 0 ? w[i] * (1.0 - lr * weightDecay) : w[i]
            w[i] = decayed - stepOrig[i]
        }
    }

    private mutating func updatePreconditioners(_ g: [Double]) {
        let gT = CPULinalg.transpose(g, rows: rows, cols: cols)
        if let prev = lMat {
            let ggT = CPULinalg.matMul(g, rows, cols, gT, rows)
            lMat = zip(prev, ggT).map { shampooBeta * $0 + (1.0 - shampooBeta) * $1 }
        }
        if let prev = rMat {
            let gTg = CPULinalg.matMul(gT, cols, rows, g, cols)
            rMat = zip(prev, gTg).map { shampooBeta * $0 + (1.0 - shampooBeta) * $1 }
        }
    }

    private mutating func refreshBasesIfDue() {
        guard step >= precondFreq, step % precondFreq == 0 else { return }
        let damping = Double(SOAPOptimizer.preconditionerDamping)
        if step == precondFreq {
            if let lMat {
                let nextQL = CPULinalg.jacobiEigh(CPULinalg.symmetrize(lMat, dim: rows, damping: damping), dim: rows)
                qL = nextQL
                let qLSqT = CPULinalg.transpose(nextQL.map { $0 * $0 }, rows: rows, cols: rows)
                v = CPULinalg.matMul(qLSqT, rows, rows, v, cols)
            }
            if let rMat {
                let nextQR = CPULinalg.jacobiEigh(CPULinalg.symmetrize(rMat, dim: cols, damping: damping), dim: cols)
                qR = nextQR
                let qRSq = nextQR.map { $0 * $0 }
                v = CPULinalg.matMul(v, rows, cols, qRSq, cols)
            }
            return
        }
        if let lMat, let q = qL {
            let (nextQ, order) = qrStep(precond: lMat, basis: q, dim: rows, damping: damping)
            qL = nextQ
            permuteVRows(order)
        }
        if let rMat, let q = qR {
            let (nextQ, order) = qrStep(precond: rMat, basis: q, dim: cols, damping: damping)
            qR = nextQ
            permuteVCols(order)
        }
    }

    private func qrStep(precond: [Double], basis: [Double], dim: Int, damping: Double) -> ([Double], [Int]) {
        let sym = CPULinalg.symmetrize(precond, dim: dim, damping: damping)
        let pq = CPULinalg.matMul(sym, dim, dim, basis, dim)
        let rayleigh = (0 ..< dim).map { col in
            (0 ..< dim).reduce(0.0) { $0 + basis[$1 * dim + col] * pq[$1 * dim + col] }
        }
        let order = (0 ..< dim).sorted { rayleigh[$0] > rayleigh[$1] }
        let pqSorted = CPULinalg.reorderColumns(pq, dim: dim, order: order)
        return (CPULinalg.mgsQR(pqSorted, dim: dim), order)
    }

    private mutating func permuteVRows(_ order: [Int]) {
        var next = Array(repeating: 0.0, count: rows * cols)
        for (newRow, oldRow) in order.enumerated() {
            for col in 0 ..< cols { next[newRow * cols + col] = v[oldRow * cols + col] }
        }
        v = next
    }

    private mutating func permuteVCols(_ order: [Int]) {
        var next = Array(repeating: 0.0, count: rows * cols)
        for row in 0 ..< rows {
            for (newCol, oldCol) in order.enumerated() { next[row * cols + newCol] = v[row * cols + oldCol] }
        }
        v = next
    }

    private func projectTo(_ x: [Double]) -> [Double] {
        var out = x
        if let qL { out = CPULinalg.matMul(CPULinalg.transpose(qL, rows: rows, cols: rows), rows, rows, out, cols) }
        if let qR { out = CPULinalg.matMul(out, rows, cols, qR, cols) }
        return out
    }

    private func projectFrom(_ x: [Double]) -> [Double] {
        var out = x
        if let qL { out = CPULinalg.matMul(qL, rows, rows, out, cols) }
        if let qR { out = CPULinalg.matMul(out, rows, cols, CPULinalg.transpose(qR, rows: cols, cols: cols), cols) }
        return out
    }
}

private enum SOAPTestSupport {
    static func deterministicMatrix(rows: Int, cols: Int, seed: UInt64) -> [Float] {
        var rng = SplitMix64(seed: seed)
        return (0 ..< rows * cols).map { index in
            let raw = Double(rng.next() >> 11) / Double(1 << 53)
            return Float((raw - 0.5) * 1.6 + 0.08 * sin(Double(index) * 0.7))
        }
    }

    static func orthogonalityError(_ q: MLXArray) -> Float {
        let dim = q.dim(0)
        let diff = MLX.abs(matmul(q.transposed(1, 0), q) - MLXArray.eye(dim, dtype: .float32)).max()
        eval(diff)
        return diff.item(Float.self)
    }

    static func maxAbsDiff(_ lhs: MLXArray, _ rhs: [Double]) -> Float {
        eval(lhs)
        let values = lhs.asType(.float32).asArray(Float.self)
        return zip(values, rhs).map { Swift.abs($0 - Float($1)) }.max() ?? 0
    }

    static func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let diff = MLX.abs(lhs.asType(.float32) - rhs.asType(.float32)).max()
        eval(diff)
        return diff.item(Float.self)
    }
}

@Suite("SOAP optimizer: CPU Jacobi+QR parity, orthogonality, AdamW equivalence, maxPrecondDim guard, and checkpoint resume (#846)", .serialized, .timeLimit(.minutes(2)))
struct SOAPOptimizerParityTests {
    @Test("TrainingOptimizerOptions validates bounds and MatrixOptimizerMath.soapAxisPlan enforces maxPrecondDim memory guard")
    func optionsValidationAndAxisPlan() throws {
        var valid = TrainingOptimizerOptions(kind: .soap, weightDecay: 0.01)
        try valid.validate()
        #expect(MatrixOptimizerMath.soapAxisPlan(rows: 4096, cols: 16, maxPrecondDim: 1024) == (false, true))
        #expect(MatrixOptimizerMath.soapAxisPlan(rows: 16, cols: 4096, maxPrecondDim: 1024) == (true, false))
        #expect(MatrixOptimizerMath.soapAxisPlan(rows: 64, cols: 64, maxPrecondDim: 1024) == (true, true))
        #expect(MatrixOptimizerMath.soapAxisPlan(rows: 4096, cols: 4096, maxPrecondDim: 1024) == (false, false))
        #expect(MatrixOptimizerMath.soapAxisPlan(rows: 1, cols: 1, maxPrecondDim: 1024) == (false, false))
        #expect(!MatrixOptimizerMath.isSOAPEligible(shape: [4096, 4096], maxPrecondDim: 1024))
        #expect(MatrixOptimizerMath.isSOAPEligible(shape: [4096, 16], maxPrecondDim: 1024))
        #expect(!MatrixOptimizerMath.isSOAPEligible(shape: [64], maxPrecondDim: 1024))

        valid.qkClipTau = 0
        #expect(throws: BasaltError.self) { try valid.validate() }
        valid = TrainingOptimizerOptions(kind: .soap)
        valid.soapPrecondFreq = 0
        #expect(throws: BasaltError.self) { try valid.validate() }
        valid = TrainingOptimizerOptions(kind: .soap)
        valid.soapShampooBeta = 1.0
        #expect(throws: BasaltError.self) { try valid.validate() }
        valid = TrainingOptimizerOptions(kind: .soap)
        valid.soapMaxPrecondDim = 0
        #expect(throws: BasaltError.self) { try valid.validate() }
        valid = TrainingOptimizerOptions(kind: .soap)
        valid.soapMaxPrecondDim = 5_000
        #expect(throws: BasaltError.self) { try valid.validate() }

        let minimalJSON = Data(#"{"kind":"soap"}"#.utf8)
        let decoded = try JSONDecoder().decode(TrainingOptimizerOptions.self, from: minimalJSON)
        #expect(decoded == TrainingOptimizerOptions(kind: .soap))
    }

    @Test("SOAP matches AdamW(biasCorrection: true) to < 1e-6 when eigenbases remain identity (Q = I)")
    func exactAdamWEquivalenceWhenBasisIsIdentity() {
        let dim = 4
        let initVals = SOAPTestSupport.deterministicMatrix(rows: dim, cols: dim, seed: 101)
        let soapModel = SOAPMatrixModule(weight: MLXArray(initVals, [dim, dim]))
        let adamModel = SOAPMatrixModule(weight: MLXArray(initVals, [dim, dim]))

        var options = TrainingOptimizerOptions(kind: .soap, weightDecay: 0.02)
        options.soapPrecondFreq = 100
        options.soapBeta1 = 0.9
        options.soapBeta2 = 0.99
        options.soapEps = 1e-8
        let soap = SOAPOptimizer(learningRate: 0.03, options: options, biasCorrection: true)
        let adam = AdamW(learningRate: 0.03, betas: (0.9, 0.99), eps: 1e-8, weightDecay: 0.02, biasCorrection: true)

        for step in 0 ..< 5 {
            var diag = Array(repeating: Float(0), count: dim * dim)
            for i in 0 ..< dim { diag[i * dim + i] = Float(dim - i) * (0.3 + 0.05 * Float(step)) }
            let grads = ModuleParameters.unflattened([("weight", MLXArray(diag, [dim, dim]))])
            soap.update(model: soapModel, gradients: grads)
            adam.update(model: adamModel, gradients: grads)
            eval(soapModel, soap, adamModel, adam)
        }

        #expect(SOAPTestSupport.maxAbsDiff(soapModel.weight, adamModel.weight) < 1e-6)
    }

    @Test("Two-sided SOAP matches plain-Swift CPU Jacobi+QR reference to < 1e-4 over 10 steps with orthonormal eigenbases (< 1e-5)")
    func twoSidedTenStepCPUParityAndOrthogonality() throws {
        let (rows, cols) = (4, 4)
        let initVals = SOAPTestSupport.deterministicMatrix(rows: rows, cols: cols, seed: 301)
        let model = SOAPMatrixModule(weight: MLXArray(initVals, [rows, cols]))

        var options = TrainingOptimizerOptions(kind: .soap, weightDecay: 0.01)
        options.soapPrecondFreq = 3
        options.soapMaxPrecondDim = 16
        options.soapShampooBeta = 0.9
        options.soapBeta1 = 0.9
        options.soapBeta2 = 0.95
        let lr: Float = 0.02
        let optimizer = SOAPOptimizer(learningRate: lr, options: options, biasCorrection: true)
        var cpuRef = CPUSOAPReference(initial: initVals, rows: rows, cols: cols, lr: lr, options: options)

        for step in 0 ..< 10 {
            let gradVals = SOAPTestSupport.deterministicMatrix(rows: rows, cols: cols, seed: UInt64(400 + step))
            optimizer.update(model: model, gradients: ModuleParameters.unflattened([("weight", MLXArray(gradVals, [rows, cols]))]))
            eval(model, optimizer)
            cpuRef.apply(gradient: gradVals)
            let diff = SOAPTestSupport.maxAbsDiff(model.weight, cpuRef.w)
            #expect(diff < 1e-4, "step \(step + 1) maxAbsDiff=\(diff)")
        }

        let basis = optimizer.eigenbasis(for: "weight")
        let qL = try #require(basis.qL)
        let qR = try #require(basis.qR)
        #expect(SOAPTestSupport.orthogonalityError(qL) < 1e-5)
        #expect(SOAPTestSupport.orthogonalityError(qR) < 1e-5)

        let rectModel = SOAPMatrixModule(weight: MLXArray(SOAPTestSupport.deterministicMatrix(rows: 6, cols: 4, seed: 305), [6, 4]))
        let rectOpt = SOAPOptimizer(learningRate: lr, options: options, biasCorrection: true)
        for step in 0 ..< 6 {
            let g = SOAPTestSupport.deterministicMatrix(rows: 6, cols: 4, seed: UInt64(500 + step))
            rectOpt.update(model: rectModel, gradients: ModuleParameters.unflattened([("weight", MLXArray(g, [6, 4]))]))
            eval(rectModel, rectOpt)
        }
        let rectBasis = rectOpt.eigenbasis(for: "weight")
        #expect(SOAPTestSupport.orthogonalityError(try #require(rectBasis.qL)) < 1e-5)
        #expect(SOAPTestSupport.orthogonalityError(try #require(rectBasis.qR)) < 1e-5)
    }

    @Test("One-sided left and one-sided right SOAP match CPU Jacobi+QR reference to < 1e-4 and allocate only the bounded axis preconditioner")
    func oneSidedLeftAndRightCPUParity() throws {
        for (rows, cols, expectLeft, expectRight) in [(4, 10, true, false), (10, 4, false, true)] {
            let initVals = SOAPTestSupport.deterministicMatrix(rows: rows, cols: cols, seed: UInt64(rows * 31 + cols))
            let model = SOAPMatrixModule(weight: MLXArray(initVals, [rows, cols]))
            var options = TrainingOptimizerOptions(kind: .soap, weightDecay: 0.005)
            options.soapPrecondFreq = 3
            options.soapMaxPrecondDim = 6
            let lr: Float = 0.015
            let optimizer = SOAPOptimizer(learningRate: lr, options: options, biasCorrection: true)
            var cpuRef = CPUSOAPReference(initial: initVals, rows: rows, cols: cols, lr: lr, options: options)

            for step in 0 ..< 10 {
                let gradVals = SOAPTestSupport.deterministicMatrix(rows: rows, cols: cols, seed: UInt64(700 + rows * 17 + step))
                optimizer.update(model: model, gradients: ModuleParameters.unflattened([("weight", MLXArray(gradVals, [rows, cols]))]))
                eval(model, optimizer)
                cpuRef.apply(gradient: gradVals)
            }

            #expect(SOAPTestSupport.maxAbsDiff(model.weight, cpuRef.w) < 1e-4)
            let basis = optimizer.eigenbasis(for: "weight")
            #expect((basis.qL != nil) == expectLeft)
            #expect((basis.qR != nil) == expectRight)
            if let qL = basis.qL { #expect(SOAPTestSupport.orthogonalityError(qL) < 1e-5) }
            if let qR = basis.qR { #expect(SOAPTestSupport.orthogonalityError(qR) < 1e-5) }
        }
    }

    @Test("TrainingOptimizerFactory routes SOAP-eligible matrices to SOAP, oversized and 1-D tensors to AdamW, and passes AdapterCheckpoint validation")
    func compositeRoutingAndCheckpointValidation() throws {
        var options = TrainingOptimizerOptions(kind: .soap, weightDecay: 0.01)
        options.soapPrecondFreq = 2
        options.soapMaxPrecondDim = 8
        let mixed = SOAPMixedModule(
            oneSidedTall: MLXArray(SOAPTestSupport.deterministicMatrix(rows: 16, cols: 4, seed: 11), [16, 4]),
            twoSided: MLXArray(SOAPTestSupport.deterministicMatrix(rows: 4, cols: 4, seed: 12), [4, 4]),
            oversized: MLXArray(SOAPTestSupport.deterministicMatrix(rows: 12, cols: 12, seed: 13), [12, 12]),
            bias: MLXArray(SOAPTestSupport.deterministicMatrix(rows: 6, cols: 1, seed: 14), [6])
        )
        let opt = TrainingOptimizerFactory.make(options: options, learningRate: 0.02)
        let grads = ModuleParameters.unflattened([
            ("oneSidedTall", MLXArray(SOAPTestSupport.deterministicMatrix(rows: 16, cols: 4, seed: 21), [16, 4])),
            ("twoSided", MLXArray(SOAPTestSupport.deterministicMatrix(rows: 4, cols: 4, seed: 22), [4, 4])),
            ("oversized", MLXArray(SOAPTestSupport.deterministicMatrix(rows: 12, cols: 12, seed: 23), [12, 12])),
            ("bias", MLXArray(SOAPTestSupport.deterministicMatrix(rows: 6, cols: 1, seed: 24), [6])),
        ])
        opt.update(model: mixed, gradients: grads)
        eval(mixed, opt)

        let stateShapes = Dictionary(uniqueKeysWithValues: opt.state().flattened().map { ($0.0, $0.1.shape) })
        #expect(stateShapes["states.0.oneSidedTall.r"] == [4, 4])
        #expect(stateShapes["states.0.oneSidedTall.qR"] == [4, 4])
        #expect(stateShapes["states.0.oneSidedTall.l"] == nil)
        #expect(stateShapes["states.0.twoSided.l"] == [4, 4])
        #expect(stateShapes["states.0.twoSided.r"] == [4, 4])
        #expect(stateShapes["states.1.oversized.m"] == [12, 12])
        #expect(stateShapes["states.1.bias.m"] == [6])

        let trainable = ["oneSidedTall": [16, 4], "twoSided": [4, 4], "oversized": [12, 12], "bias": [6]]
        #expect(AdapterCheckpoint.mismatch(weights: trainable, optimizer: stateShapes, trainable: trainable, kind: .soap, soapMaxPrecondDim: 8) == nil)
        #expect(AdapterCheckpoint.mismatch(weights: trainable, optimizer: stateShapes, trainable: trainable, kind: .soap, soapMaxPrecondDim: 16) != nil)
        #expect(AdapterCheckpoint.mismatch(weights: trainable, optimizer: stateShapes, trainable: trainable, kind: .adamw) != nil)
    }

    @Test("SOAP checkpoint save and restore across a precondFreq boundary resumes identically to < 1e-5")
    func checkpointSaveLoadResumeIdentity() throws {
        let scratch = FileManager.default.temporaryDirectory.appending(path: "soap-ckpt-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }

        var options = TrainingOptimizerOptions(kind: .soap, weightDecay: 0.01)
        options.soapPrecondFreq = 3
        options.soapMaxPrecondDim = 8
        let initMat = SOAPTestSupport.deterministicMatrix(rows: 6, cols: 4, seed: 801)
        let initBias = SOAPTestSupport.deterministicMatrix(rows: 6, cols: 1, seed: 802)

        let continuous = SOAPMixedModule(
            oneSidedTall: MLXArray(SOAPTestSupport.deterministicMatrix(rows: 12, cols: 4, seed: 803), [12, 4]),
            twoSided: MLXArray(initMat, [6, 4]),
            oversized: MLXArray(SOAPTestSupport.deterministicMatrix(rows: 10, cols: 10, seed: 804), [10, 10]),
            bias: MLXArray(initBias, [6])
        )
        let resumed = SOAPMixedModule(
            oneSidedTall: MLXArray(SOAPTestSupport.deterministicMatrix(rows: 12, cols: 4, seed: 803), [12, 4]),
            twoSided: MLXArray(initMat, [6, 4]),
            oversized: MLXArray(SOAPTestSupport.deterministicMatrix(rows: 10, cols: 10, seed: 804), [10, 10]),
            bias: MLXArray(initBias, [6])
        )
        let optFull = TrainingOptimizerFactory.make(options: options, learningRate: 0.02)
        var optSplit = TrainingOptimizerFactory.make(options: options, learningRate: 0.02)

        for step in 0 ..< 10 {
            let grads = ModuleParameters.unflattened([
                ("oneSidedTall", MLXArray(SOAPTestSupport.deterministicMatrix(rows: 12, cols: 4, seed: UInt64(900 + step)), [12, 4])),
                ("twoSided", MLXArray(SOAPTestSupport.deterministicMatrix(rows: 6, cols: 4, seed: UInt64(950 + step)), [6, 4])),
                ("oversized", MLXArray(SOAPTestSupport.deterministicMatrix(rows: 10, cols: 10, seed: UInt64(1000 + step)), [10, 10])),
                ("bias", MLXArray(SOAPTestSupport.deterministicMatrix(rows: 6, cols: 1, seed: UInt64(1050 + step)), [6])),
            ])
            optFull.update(model: continuous, gradients: grads)
            optSplit.update(model: resumed, gradients: grads)
            eval(continuous, optFull, resumed, optSplit)

            if step == 3 {
                let optFile = scratch.appending(component: "optimizer.safetensors")
                let saved = Dictionary(uniqueKeysWithValues: optSplit.state().flattened())
                try MLX.save(arrays: saved, url: optFile)
                let loaded = try MLX.loadArrays(url: optFile)
                optSplit = TrainingOptimizerFactory.make(options: options, learningRate: 0.02)
                optSplit.update(parameters: ModuleParameters.unflattened(loaded))
                eval(optSplit)
            }
        }

        #expect(SOAPTestSupport.maxAbsDiff(continuous.oneSidedTall, resumed.oneSidedTall) < 1e-5)
        #expect(SOAPTestSupport.maxAbsDiff(continuous.twoSided, resumed.twoSided) < 1e-5)
        #expect(SOAPTestSupport.maxAbsDiff(continuous.oversized, resumed.oversized) < 1e-5)
        #expect(SOAPTestSupport.maxAbsDiff(continuous.bias, resumed.bias) < 1e-5)
    }
}

extension SOAPOptimizerParityTests {
    @Test("SOAP doubly-stochastic variance transport conserves Frobenius energy and non-negativity across eigenbasis switch")
    func varianceTransportConservesSumAndNonNegativity() throws {
        var options = TrainingOptimizerOptions(kind: .soap, weightDecay: 0)
        options.soapPrecondFreq = 3
        options.soapMaxPrecondDim = 8
        let module = SOAPMatrixModule(weight: MLXArray(SOAPTestSupport.deterministicMatrix(rows: 4, cols: 4, seed: 1101), [4, 4]))
        let opt = SOAPOptimizer(learningRate: 0.01, options: options)
        for step in 0 ..< 2 {
            let g = MLXArray(SOAPTestSupport.deterministicMatrix(rows: 4, cols: 4, seed: UInt64(1110 + step)), [4, 4])
            opt.update(model: module, gradients: ModuleParameters.unflattened([("weight", g)]))
            eval(module, opt)
        }
        let vBefore = try #require(opt.state().flattened().first(where: { $0.0 == "weight.v" })?.1).asType(.float32).asArray(Float.self)
        let sumBefore = vBefore.reduce(0.0) { $0 + Double($1) }
        let gArr = SOAPTestSupport.deterministicMatrix(rows: 4, cols: 4, seed: 1112)
        let gSqSum = gArr.reduce(0.0) { $0 + Double($1) * Double($1) }
        opt.update(model: module, gradients: ModuleParameters.unflattened([("weight", MLXArray(gArr, [4, 4]))]))
        eval(module, opt)
        let vAfter = try #require(opt.state().flattened().first(where: { $0.0 == "weight.v" })?.1).asType(.float32).asArray(Float.self)
        let expected = Double(options.soapBeta2) * sumBefore + (1.0 - Double(options.soapBeta2)) * gSqSum
        #expect((vAfter.min() ?? -1) >= 0)
        #expect(abs(vAfter.reduce(0.0) { $0 + Double($1) } - expected) <= 1e-4 * max(1.0, expected))
    }
}
