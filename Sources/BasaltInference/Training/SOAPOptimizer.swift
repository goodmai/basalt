import Foundation
import MLX
import MLXNN
import MLXOptimizers
import BasaltCore

public final class SOAPOptimizer: Optimizer {
    public static let preconditionerDamping: Float = 1e-12

    public struct State: Updatable {
        public var step: MLXArray
        public var m: MLXArray
        public var v: MLXArray
        public var lPrecond: MLXArray?
        public var rPrecond: MLXArray?
        public var qL: MLXArray?
        public var qR: MLXArray?

        public init(
            step: MLXArray,
            m: MLXArray,
            v: MLXArray,
            lPrecond: MLXArray? = nil,
            rPrecond: MLXArray? = nil,
            qL: MLXArray? = nil
        ) {
            self.step = step
            self.m = m
            self.v = v
            self.lPrecond = lPrecond
            self.rPrecond = rPrecond
            self.qL = qL
            self.qR = nil
        }

        public func innerState() -> [MLXArray] {
            [m, v, step] + [lPrecond, rPrecond, qL, qR].compactMap { $0 }
        }
    }

    public var learningRate: Float
    public var beta1: Float
    public var beta2: Float
    public var shampooBeta: Float
    public var eps: Float
    public var weightDecay: Float
    public var precondFreq: Int
    public var maxPrecondDim: Int
    public var biasCorrection: Bool
    private var states: [String: State] = [:]

    public init(
        learningRate: Float,
        options: TrainingOptimizerOptions = TrainingOptimizerOptions(kind: .soap),
        biasCorrection: Bool = true
    ) {
        self.learningRate = learningRate
        self.beta1 = options.soapBeta1
        self.beta2 = options.soapBeta2
        self.shampooBeta = options.soapShampooBeta
        self.eps = options.soapEps
        self.weightDecay = options.weightDecay
        self.precondFreq = max(1, options.soapPrecondFreq)
        self.maxPrecondDim = max(1, options.soapMaxPrecondDim)
        self.biasCorrection = biasCorrection
    }

    public init(
        learningRate: Float,
        precondFreq: Int = TrainingOptimizerOptions.defaultSOAPPrecondFreq,
        maxPrecondDim: Int = TrainingOptimizerOptions.defaultSOAPMaxPrecondDim,
        shampooBeta: Float = TrainingOptimizerOptions.defaultSOAPShampooBeta,
        weightDecay: Float = 0.0,
        biasCorrection: Bool = true
    ) {
        self.learningRate = learningRate
        self.beta1 = TrainingOptimizerOptions.defaultSOAPBeta1
        self.beta2 = TrainingOptimizerOptions.defaultSOAPBeta2
        self.shampooBeta = shampooBeta
        self.eps = TrainingOptimizerOptions.defaultSOAPEps
        self.weightDecay = weightDecay
        self.precondFreq = max(1, precondFreq)
        self.maxPrecondDim = max(1, maxPrecondDim)
        self.biasCorrection = biasCorrection
    }

    public func eigenbasis(for key: String) -> (qL: MLXArray?, qR: MLXArray?) {
        guard let entry = states[key] else { return (nil, nil) }
        return (entry.qL, entry.qR)
    }

    public func newState(parameter: MLXArray) -> State {
        var entry = State(
            step: MLXArray(Int32(0)),
            m: MLXArray.zeros(parameter.shape, dtype: .float32),
            v: MLXArray.zeros(parameter.shape, dtype: .float32)
        )
        guard parameter.ndim == 2 else { return entry }
        let rows = parameter.dim(0)
        let cols = parameter.dim(1)
        let plan = MatrixOptimizerMath.soapAxisPlan(rows: rows, cols: cols, maxPrecondDim: maxPrecondDim)
        if plan.useLeft {
            entry.lPrecond = MLXArray.zeros([rows, rows], dtype: .float32)
            entry.qL = MLXArray.eye(rows, dtype: .float32)
        }
        if plan.useRight {
            entry.rPrecond = MLXArray.zeros([cols, cols], dtype: .float32)
            entry.qR = MLXArray.eye(cols, dtype: .float32)
        }
        return entry
    }

    public func update(model: Module, gradients: ModuleParameters) {
        let flatGrads = gradients.flattened()
        guard !flatGrads.isEmpty else { return }
        let params = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var updates: [(String, MLXArray)] = []
        updates.reserveCapacity(flatGrads.count)
        for (key, gradient) in flatGrads {
            guard let parameter = params[key] else { continue }
            let (updated, nextState) = applySingle(gradient: gradient, parameter: parameter, state: states[key])
            states[key] = nextState
            updates.append((key, updated))
        }
        if !updates.isEmpty {
            model.update(parameters: ModuleParameters.unflattened(updates))
        }
    }

    public func innerState() -> [MLXArray] {
        states.sorted { $0.key < $1.key }.flatMap { $0.value.innerState() }
    }

    public func state() -> ModuleParameters {
        guard !states.isEmpty else { return ModuleParameters() }
        var flat: [(String, MLXArray)] = []
        for (key, entry) in states.sorted(by: { $0.key < $1.key }) {
            flat.append(("\(key).m", entry.m))
            flat.append(("\(key).v", entry.v))
            flat.append(("\(key).step", entry.step))
            if let l = entry.lPrecond { flat.append(("\(key).l", l)) }
            if let r = entry.rPrecond { flat.append(("\(key).r", r)) }
            if let qL = entry.qL { flat.append(("\(key).qL", qL)) }
            if let qR = entry.qR { flat.append(("\(key).qR", qR)) }
        }
        return ModuleParameters.unflattened(flat)
    }

    public func update(parameters: ModuleParameters) {
        var grouped: [String: [String: MLXArray]] = [:]
        for (fullKey, value) in parameters.flattened() {
            guard let dot = fullKey.lastIndex(of: ".") else { continue }
            let paramKey = String(fullKey[..<dot])
            let suffix = String(fullKey[fullKey.index(after: dot)...])
            grouped[paramKey, default: [:]][suffix] = value
        }
        for (paramKey, dict) in grouped {
            guard let m = dict["m"], let v = dict["v"], let step = dict["step"] else { continue }
            var entry = State(step: step, m: m, v: v, lPrecond: dict["l"], rPrecond: dict["r"], qL: dict["qL"])
            entry.qR = dict["qR"]
            states[paramKey] = entry
        }
    }
}

extension SOAPOptimizer {
    public func applySingle(
        gradient: MLXArray,
        parameter: MLXArray,
        state: State?
    ) -> (MLXArray, State) {
        let g32 = gradient.asType(.float32)
        let p32 = parameter.asType(.float32)
        var next = state ?? newState(parameter: parameter)
        let nextStep = Int(next.step.asType(.int32).item(Int32.self)) + 1
        next.step = MLXArray(Int32(nextStep))
        updatePreconditioners(&next, gradient: g32)
        refreshEigenbasesIfDue(&next, step: nextStep)
        next.m = beta1 * next.m + (1 - beta1) * g32
        let gPrime = Self.projectToEigenbasis(g32, qL: next.qL, qR: next.qR)
        next.v = beta2 * next.v + (1 - beta2) * square(gPrime)
        let mPrime = Self.projectToEigenbasis(next.m, qL: next.qL, qR: next.qR)
        let stepPrime = adamEigenStep(mPrime: mPrime, v: next.v, step: nextStep)
        let stepOrig = Self.projectFromEigenbasis(stepPrime, qL: next.qL, qR: next.qR)
        let decayed = weightDecay != 0 ? p32 * (1 - learningRate * weightDecay) : p32
        let updated = (decayed - stepOrig).asType(parameter.dtype)
        return (updated, next)
    }

    private func updatePreconditioners(_ state: inout State, gradient: MLXArray) {
        guard gradient.ndim == 2 else { return }
        if let prevL = state.lPrecond {
            state.lPrecond = shampooBeta * prevL + (1 - shampooBeta) * matmul(gradient, gradient.transposed(1, 0))
        }
        if let prevR = state.rPrecond {
            state.rPrecond = shampooBeta * prevR + (1 - shampooBeta) * matmul(gradient.transposed(1, 0), gradient)
        }
    }

    private func refreshEigenbasesIfDue(_ state: inout State, step: Int) {
        guard step >= precondFreq, step % precondFreq == 0 else { return }
        if step == precondFreq {
            if let l = state.lPrecond {
                let qL = Self.initialEigenbasis(from: l, damping: Self.preconditionerDamping)
                state.qL = qL
                state.v = matmul(square(qL).transposed(1, 0), state.v)
            }
            if let r = state.rPrecond {
                let qR = Self.initialEigenbasis(from: r, damping: Self.preconditionerDamping)
                state.qR = qR
                state.v = matmul(state.v, square(qR))
            }
            return
        }
        if let l = state.lPrecond, let qL = state.qL {
            let updated = Self.qrSubspaceStep(
                preconditioner: l,
                basis: qL,
                secondMoment: state.v,
                axis: 0,
                damping: Self.preconditionerDamping
            )
            state.qL = updated.basis
            state.v = updated.secondMoment
        }
        if let r = state.rPrecond, let qR = state.qR {
            let updated = Self.qrSubspaceStep(
                preconditioner: r,
                basis: qR,
                secondMoment: state.v,
                axis: 1,
                damping: Self.preconditionerDamping
            )
            state.qR = updated.basis
            state.v = updated.secondMoment
        }
    }

    private func adamEigenStep(mPrime: MLXArray, v: MLXArray, step: Int) -> MLXArray {
        if biasCorrection {
            let t = Float(step)
            let c1 = learningRate / (1.0 - Foundation.pow(beta1, t))
            let c2 = Float(1.0) / (1.0 - Foundation.pow(beta2, t)).squareRoot()
            return (c1 * mPrime) / (MLX.sqrt(v) * c2 + eps)
        }
        return (learningRate * mPrime) / (MLX.sqrt(v) + eps)
    }

    public static func projectToEigenbasis(_ x: MLXArray, qL: MLXArray?, qR: MLXArray?) -> MLXArray {
        var out = x
        if let qL { out = matmul(qL.transposed(1, 0), out) }
        if let qR { out = matmul(out, qR) }
        return out
    }

    public static func projectFromEigenbasis(_ x: MLXArray, qL: MLXArray?, qR: MLXArray?) -> MLXArray {
        var out = x
        if let qL { out = matmul(qL, out) }
        if let qR { out = matmul(out, qR.transposed(1, 0)) }
        return out
    }

    public static func initialEigenbasis(from preconditioner: MLXArray, damping: Float) -> MLXArray {
        let dim = preconditioner.dim(0)
        let sym = Float(0.5) * (preconditioner + preconditioner.transposed(1, 0))
            + MLXArray(damping) * MLXArray.eye(dim, dtype: .float32)
        let (eigvals, eigvecs) = MLXLinalg.eigh(sym, stream: .cpu)
        let order = argSort(-eigvals, axis: 0)
        let ordered = take(eigvecs, order, axis: 1)
        return canonicalizeColumns(ordered)
    }

    public static func canonicalizeColumns(_ q: MLXArray) -> MLXArray {
        let pivotRows = MLX.abs(q).argMax(axis: 0, keepDims: true)
        let pivots = takeAlong(q, pivotRows, axis: 0)
        let signs = MLX.where(pivots .< Float(0), MLXArray(Float(-1)), MLXArray(Float(1)))
        return q * signs
    }

    public static func qrSubspaceStep(
        preconditioner: MLXArray,
        basis: MLXArray,
        secondMoment: MLXArray,
        axis: Int,
        damping: Float
    ) -> (basis: MLXArray, secondMoment: MLXArray) {
        let dim = preconditioner.dim(0)
        let sym = Float(0.5) * (preconditioner + preconditioner.transposed(1, 0))
            + MLXArray(damping) * MLXArray.eye(dim, dtype: .float32)
        let pq = matmul(sym, basis)
        let rayleigh = (basis * pq).sum(axis: 0)
        let order = argSort(-rayleigh, axis: 0)
        let pqSorted = take(pq, order, axis: 1)
        let permutedV = take(secondMoment, order, axis: axis)
        let (qRaw, rRaw) = MLXLinalg.qr(pqSorted, stream: .cpu)
        let diagR = MLX.diagonal(rRaw)
        let signs = MLX.where(diagR .< Float(0), MLXArray(Float(-1)), MLXArray(Float(1))).reshaped([1, dim])
        return (basis: qRaw * signs, secondMoment: permutedV)
    }
}
