import Foundation
import MLX
import BasaltCore

public struct MAOSNMFConfig: Sendable, Codable, Equatable {
    public var features: Int
    public var gamma: Float
    public var iterations: Int
    public var mmrLambda: Float
    public var ridge: Float
    public var tolerance: Float

    public init(
        features: Int,
        gamma: Float = 1.0,
        iterations: Int = 100,
        mmrLambda: Float = 0.5,
        ridge: Float = 1e-6,
        tolerance: Float = 1e-6
    ) {
        self.features = features
        self.gamma = gamma
        self.iterations = iterations
        self.mmrLambda = mmrLambda
        self.ridge = ridge
        self.tolerance = tolerance
    }
}

public struct MAOSNMFDecomposition {
    public let z: MLXArray
    public let w: MLXArray
    public let selectedSeeds: [Int]
    public let reconstructionError: Float
    public let orthogonalityDefect: Float
    public let objectiveHistory: [Float]

    public func modulate(_ activations: MLXArray, feature: Int, scale: Float) throws(BasaltError) -> MLXArray {
        let count = w.dim(1)
        guard (0 ..< count).contains(feature), scale.isFinite else {
            throw .invalidRequestStructure(details: "MAO-SNMF feature \(feature) must be in 0..<\(count) with a finite scale")
        }
        guard activations.dim(-1) == w.dim(0) else {
            throw .invalidRequestStructure(details: "MAO-SNMF activations have width \(activations.dim(-1)); dictionary expects \(w.dim(0))")
        }
        let wf = w[0..., feature]
        let normSq = MLX.maximum(MLX.square(wf).sum(), MLXArray(Float(1e-12)))
        let x = activations.asType(.float32)
        let coords = (x * wf).sum(axis: -1, keepDims: true) / normSq
        let updated = x + (scale - 1) * coords * wf
        return updated.asType(activations.dtype)
    }

    public func pin(_ activations: MLXArray, feature: Int, value: Float) throws(BasaltError) -> MLXArray {
        let count = w.dim(1)
        guard (0 ..< count).contains(feature), value.isFinite else {
            throw .invalidRequestStructure(details: "MAO-SNMF feature \(feature) must be in 0..<\(count) with a finite value")
        }
        guard activations.dim(-1) == w.dim(0) else {
            throw .invalidRequestStructure(details: "MAO-SNMF activations have width \(activations.dim(-1)); dictionary expects \(w.dim(0))")
        }
        let wf = w[0..., feature]
        let normSq = MLX.maximum(MLX.square(wf).sum(), MLXArray(Float(1e-12)))
        let x = activations.asType(.float32)
        let coords = (x * wf).sum(axis: -1, keepDims: true) / normSq
        let updated = x + (value - coords) * wf
        return updated.asType(activations.dtype)
    }
}

public enum MAOSNMFDecomposer {
    public static func decompose(_ activations: MLXArray, config: MAOSNMFConfig) throws(BasaltError) -> MAOSNMFDecomposition {
        try validate(activations, config: config)
        let a = activations.asType(.float32)
        eval(a)
        guard isFinite(a).all().item(Bool.self) else {
            throw .invalidRequestStructure(details: "MAO-SNMF activations must be finite")
        }
        var (w, seeds) = mmrInitialize(a, features: config.features, lambda: config.mmrLambda)
        var z = solveZ(a: a, w: w, ridge: config.ridge)
        let eye = MLXArray.eye(config.features, dtype: .float32)
        var history: [Float] = []
        var prevLoss = loss(a: a, z: z, w: w, eye: eye, gamma: config.gamma)
        for _ in 0 ..< config.iterations {
            w = stepW(a: a, z: z, w: w, eye: eye, gamma: config.gamma, currentLoss: prevLoss)
            z = solveZ(a: a, w: w, ridge: config.ridge)
            let current = loss(a: a, z: z, w: w, eye: eye, gamma: config.gamma)
            history.append(current)
            let relChange = abs(prevLoss - current) / max(abs(prevLoss), 1e-8)
            prevLoss = current
            if history.count >= 5, relChange < config.tolerance { break }
        }
        let recErr = MLX.sqrt(MLX.square(a - matmul(z, w.T)).sum())
        let orthErr = MLX.sqrt(MLX.square(matmul(w.T, w) - eye).sum())
        eval(z, w, recErr, orthErr)
        return MAOSNMFDecomposition(
            z: z,
            w: w,
            selectedSeeds: seeds,
            reconstructionError: recErr.item(Float.self),
            orthogonalityDefect: orthErr.item(Float.self),
            objectiveHistory: history
        )
    }

    private static func validate(_ activations: MLXArray, config: MAOSNMFConfig) throws(BasaltError) {
        guard activations.ndim == 2, activations.dim(0) > 0, activations.dim(1) > 0 else {
            throw .invalidRequestStructure(details: "MAO-SNMF needs a non-empty 2-D activation matrix [T, D]")
        }
        let d = activations.dim(1)
        guard (1 ... d).contains(config.features) else {
            throw .invalidRequestStructure(details: "MAO-SNMF features (\(config.features)) must be in 1...\(d)")
        }
        guard config.gamma.isFinite, config.gamma >= 0, config.iterations >= 1 else {
            throw .invalidRequestStructure(details: "MAO-SNMF gamma must be finite and non-negative and iterations at least 1")
        }
        guard config.mmrLambda.isFinite, (0 ... 1).contains(config.mmrLambda), config.ridge.isFinite, config.ridge > 0,
              config.tolerance.isFinite, config.tolerance >= 0 else {
            throw .invalidRequestStructure(details: "MAO-SNMF mmrLambda must be in 0...1, ridge > 0 and tolerance >= 0")
        }
    }

    static func mmrInitialize(_ a: MLXArray, features: Int, lambda: Float) -> (w: MLXArray, seeds: [Int]) {
        let d = a.dim(1)
        let energiesTensor = MLX.sqrt(MLX.square(a).sum(axis: 0))
        let unitCols = a / MLX.maximum(energiesTensor.reshaped(1, d), MLXArray(Float(1e-12)))
        let gram = matmul(unitCols.T, unitCols)
        eval(energiesTensor, gram)
        let energies = energiesTensor.asArray(Float.self)
        let simFlat = gram.asArray(Float.self)
        let maxEnergy = max(energies.max() ?? 0, 1e-12)
        var chosen: [Int] = []
        var used = Array(repeating: false, count: d)
        for _ in 0 ..< features {
            var bestIdx = 0
            var bestScore = -Float.greatestFiniteMagnitude
            for j in 0 ..< d where !used[j] {
                let relevance = energies[j] / maxEnergy
                let redundancy = chosen.map { abs(simFlat[j * d + $0]) }.max() ?? 0
                let score = chosen.isEmpty ? relevance : (lambda * relevance - (1 - lambda) * redundancy)
                if score > bestScore {
                    bestScore = score
                    bestIdx = j
                }
            }
            used[bestIdx] = true
            chosen.append(bestIdx)
        }
        var raw = Array(repeating: Float(0), count: d * features)
        for (f, seed) in chosen.enumerated() {
            var colNormSq: Float = 0
            for j in 0 ..< d {
                let val = max(0, simFlat[j * d + seed]) + (j == seed ? 1.0 : 1e-3)
                raw[j * features + f] = val
                colNormSq += val * val
            }
            let invNorm = 1.0 / max(colNormSq.squareRoot(), 1e-12)
            for j in 0 ..< d { raw[j * features + f] *= invNorm }
        }
        let w = MLXArray(raw, [d, features])
        eval(w)
        return (w, chosen)
    }

    private static func solveZ(a: MLXArray, w: MLXArray, ridge: Float) -> MLXArray {
        let f = w.dim(1)
        let gram = matmul(w.T, w) + ridge * MLXArray.eye(f, dtype: .float32)
        let rhs = matmul(a, w)
        eval(gram, rhs)
        let solved = MLXLinalg.solve(gram, rhs.T, stream: .cpu).T
        eval(solved)
        return solved
    }

    private static func stepW(
        a: MLXArray,
        z: MLXArray,
        w: MLXArray,
        eye: MLXArray,
        gamma: Float,
        currentLoss: Float
    ) -> MLXArray {
        let p = matmul(a.T, z)
        let q = matmul(z.T, z)
        let wtw = matmul(w.T, w)
        let pPos = 0.5 * (MLX.abs(p) + p)
        let pNeg = 0.5 * (MLX.abs(p) - p)
        let qPos = 0.5 * (MLX.abs(q) + q)
        let qNeg = 0.5 * (MLX.abs(q) - q)
        let eps = MLXArray(Float(1e-10))
        let numerator = pPos + matmul(w, qNeg) + (4 * gamma) * w
        let denominator = pNeg + matmul(w, qPos) + (4 * gamma) * matmul(w, wtw)
        let ratio = (numerator + eps) / (denominator + eps)
        let multHalf = MLX.maximum(w * MLX.sqrt(ratio), MLXArray(Float(0)))
        let multQuarter = MLX.maximum(w * MLX.sqrt(MLX.sqrt(ratio)), MLXArray(Float(0)))
        let grad = matmul(w, q) - p + (4 * gamma) * matmul(w, wtw - eye)
        let lip = MLX.sqrt(MLX.square(q).sum()) + (4 * gamma) * (3 * MLX.sqrt(MLX.square(wtw).sum()) + 1) + Float(1e-4)
        let pgd = MLX.maximum(w - grad / lip, MLXArray(Float(0)))
        eval(multHalf, multQuarter, pgd)
        var bestW = w
        var bestLoss = currentLoss
        for candidate in [multHalf, pgd, multQuarter] {
            let candLoss = loss(a: a, z: z, w: candidate, eye: eye, gamma: gamma)
            if candLoss.isFinite, candLoss <= bestLoss {
                bestLoss = candLoss
                bestW = candidate
            }
        }
        return bestW
    }

    private static func loss(a: MLXArray, z: MLXArray, w: MLXArray, eye: MLXArray, gamma: Float) -> Float {
        let rec = 0.5 * MLX.square(a - matmul(z, w.T)).sum()
        let orth = gamma * MLX.square(matmul(w.T, w) - eye).sum()
        let total = rec + orth
        eval(total)
        return total.item(Float.self)
    }
}
