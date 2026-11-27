import Foundation
import MLX
import MLXNN
import MLXOptimizers
import BasaltCore

public final class MatchedRMSMuonOptimizer: Optimizer {
    public var learningRate: Float
    public var momentum: Float
    public var weightDecay: Float
    public var nesterov: Bool
    public var nsSteps: Int
    private var momentumState: [String: MLXArray] = [:]

    public init(
        learningRate: Float,
        momentum: Float = 0.95,
        weightDecay: Float = 0.0,
        nesterov: Bool = true,
        nsSteps: Int = 5
    ) {
        self.learningRate = learningRate
        self.momentum = momentum
        self.weightDecay = weightDecay
        self.nesterov = nesterov
        self.nsSteps = max(1, nsSteps)
    }

    public static func zeropowerViaNewtonSchulz5(_ input: MLXArray, steps: Int = 5) -> MLXArray {
        precondition(input.ndim == 2, "Newton-Schulz iteration expects a 2D array")
        let (a, b, c): (Float, Float, Float) = (3.4445, -4.7750, 2.0315)
        let transposeNeeded = input.dim(-2) > input.dim(-1)
        let fp32 = input.asType(.float32)
        var x = transposeNeeded ? fp32.transposed(1, 0) : fp32
        x = x / (MLX.norm(x, keepDims: true) + Float(1e-7))
        for _ in 0 ..< steps {
            let aMat = matmul(x, x.transposed(1, 0))
            let bMat = addMM(b * aMat, aMat, aMat, alpha: c, beta: 1.0)
            x = addMM(a * x, bMat, x, alpha: 1.0, beta: 1.0)
        }
        return transposeNeeded ? x.transposed(1, 0) : x
    }

    public func applySingle(
        gradient: MLXArray,
        parameter: MLXArray,
        state: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let g32 = gradient.asType(.float32)
        let prev = state?.asType(.float32) ?? MLXArray.zeros(g32.shape, dtype: .float32)
        let p32 = parameter.asType(.float32)
        let v = momentum * prev + (1 - momentum) * g32
        var update = nesterov ? ((1 - momentum) * g32 + momentum * v) : v
        var scale: Float = 1.0
        if update.ndim >= 2 {
            let originalShape = update.shape
            let reshapeNeeded = update.ndim > 2
            if reshapeNeeded {
                update = update.reshaped([update.dim(0), -1])
            }
            let rows = update.dim(0)
            let cols = update.dim(1)
            update = Self.zeropowerViaNewtonSchulz5(update, steps: nsSteps)
            if reshapeNeeded {
                update = update.reshaped(originalShape)
            }
            scale = MatrixOptimizerMath.matchedRMSScale(rows: rows, cols: cols)
        }
        let decayed = weightDecay != 0 ? p32 * (1 - learningRate * weightDecay) : p32
        let updated = (decayed - (learningRate * scale) * update).asType(parameter.dtype)
        return (updated, v)
    }

    public func update(model: Module, gradients: ModuleParameters) {
        let flatGrads = gradients.flattened()
        guard !flatGrads.isEmpty else { return }
        let params = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var updates: [(String, MLXArray)] = []
        updates.reserveCapacity(flatGrads.count)
        for (key, gradient) in flatGrads {
            guard let parameter = params[key] else { continue }
            let (updated, nextState) = applySingle(gradient: gradient, parameter: parameter, state: momentumState[key])
            momentumState[key] = nextState
            updates.append((key, updated))
        }
        if !updates.isEmpty {
            model.update(parameters: ModuleParameters.unflattened(updates))
        }
    }

    public func innerState() -> [MLXArray] {
        momentumState.sorted { $0.key < $1.key }.map(\.value)
    }

    public func state() -> ModuleParameters {
        guard !momentumState.isEmpty else { return ModuleParameters() }
        return ModuleParameters.unflattened(momentumState.sorted { $0.key < $1.key })
    }

    public func update(parameters: ModuleParameters) {
        for (key, value) in parameters.flattened() {
            momentumState[key] = value
        }
    }
}
