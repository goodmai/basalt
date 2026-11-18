import Foundation
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

public enum CognitiveNullspaceExtractor {
    struct ProtectedInputSpec {
        let key: String
        let layer: Module
        let linear: Linear
        let context: ModelContext
    }

    public static func extract(
        activations: MLXArray,
        maxRank: Int? = nil,
        tolerance: Float = 1e-5
    ) throws(BasaltError) -> CognitiveNullspace {
        guard activations.ndim == 2, activations.dim(0) > 0, activations.dim(1) > 0 else {
            throw .invalidRequestStructure(details: "cognitive nullspace extraction needs a non-empty 2-D activation matrix")
        }
        if let maxRank, maxRank <= 0 {
            throw .invalidRequestStructure(details: "nullspaceMaxRank must be positive")
        }
        let a = activations.asType(.float32)
        eval(a)
        guard isFinite(a).all().item(Bool.self) else {
            throw .invalidRequestStructure(details: "cognitive anchor activations must be finite")
        }
        let (rows, dim) = (a.dim(0), a.dim(1))
        let cap = min(maxRank ?? min(rows, dim), min(rows, dim))
        let (candidates, sigmas) = rows <= dim
            ? wideSingularBasis(a, rows: rows, cap: cap, tolerance: tolerance)
            : tallSingularBasis(a, dim: dim, cap: cap, tolerance: tolerance)
        return try CognitiveNullspace(dimension: dim, basis: candidates, singularValues: sigmas)
    }

    static func residualNullspace(
        layer: Int,
        prompts: [[Int]],
        maxRank: Int?,
        context: ModelContext
    ) throws -> CognitiveNullspace {
        var states: [MLXArray] = []
        for ids in prompts {
            let probe = ContrastProbe(nil)
            _ = MLXInferenceEngine.forward(context, probe: probe, ids, cache: try context.model.newCache(parameters: nil))
            guard let state = probe.states[layer] else {
                throw BasaltError.invalidRequestStructure(details: "layer \(layer) was not reported to the forward probe")
            }
            eval(state)
            states.append(state)
        }
        let matrix = stacked(states)
        let dim = matrix.dim(1)
        let effectiveCap = min(maxRank ?? max(1, dim - 1), max(1, dim - 1))
        return try extract(activations: matrix, maxRank: effectiveCap)
    }

    static func projectedRankOneInput(
        meanInput: [Float],
        protectedPrompts: [[Int]],
        spec: ProtectedInputSpec
    ) throws -> [Float] {
        let origEnergy = meanInput.reduce(0.0) { $0 + Double($1) * Double($1) }
        guard origEnergy.isFinite, origEnergy > 0 else {
            throw BasaltError.invalidRequestStructure(details: "the mean input of \(spec.key) over the positive prompts is zero or not finite")
        }
        let captured = try collectProtectedInputs(prompts: protectedPrompts, spec: spec)
        guard !captured.isEmpty else {
            return meanInput.map { Float(Double($0) / origEnergy) }
        }
        let inputDim = spec.linear.shape.1
        let matrix = stacked(Array(captured.prefix(max(1, inputDim - 1))))
        let nullspace = try extract(activations: matrix, maxRank: max(1, inputDim - 1), tolerance: 1e-5)
        let mPerp = try nullspace.project(meanInput)
        let perpEnergy = mPerp.reduce(0.0) { $0 + Double($1) * Double($1) }
        if perpEnergy.isFinite, perpEnergy > max(1e-8 * origEnergy, 1e-12) {
            return mPerp.map { Float(Double($0) / perpEnergy) }
        }
        return meanInput.map { Float(Double($0) / origEnergy) }
    }

    private static func collectProtectedInputs(
        prompts: [[Int]],
        spec: ProtectedInputSpec
    ) throws -> [MLXArray] {
        var sequences: [MLXArray] = []
        let tap = SequenceInputTap(spec.linear) { sequences.append($0) }
        spec.layer.update(modules: .unflattened([(spec.key, tap as Module)]))
        spec.context.model.invalidateCompiledTraces()
        defer {
            spec.layer.update(modules: .unflattened([(spec.key, spec.linear as Module)]))
            spec.context.model.invalidateCompiledTraces()
        }
        for ids in prompts {
            _ = MLXInferenceEngine.forward(spec.context, probe: nil, ids, cache: try spec.context.model.newCache(parameters: nil))
            eval(sequences)
        }
        var ordered: [MLXArray] = []
        for seq in sequences where seq.dim(0) > 0 {
            ordered.append(seq[seq.dim(0) - 1])
        }
        for seq in sequences where seq.dim(0) > 1 {
            for t in 0 ..< (seq.dim(0) - 1) { ordered.append(seq[t]) }
        }
        return ordered
    }

    private static func wideSingularBasis(
        _ a: MLXArray,
        rows: Int,
        cap: Int,
        tolerance: Float
    ) -> ([[Float]], [Float]) {
        let gram = matmul(a, a.T)
        eval(gram)
        let (eigenvalues, eigenvectors) = MLXLinalg.eigh(gram, stream: .cpu)
        eval(eigenvalues, eigenvectors)
        let vals = eigenvalues.asArray(Float.self)
        let peakLambda = max(vals.last ?? 0, 0)
        guard peakLambda > 1e-18 else { return ([], []) }
        let sigmaMax = peakLambda.squareRoot()
        let cutoff = max(tolerance * sigmaMax, 1e-9)
        var vectors: [[Float]] = [], sigmas: [Float] = []
        for idx in stride(from: rows - 1, through: 0, by: -1) {
            guard vectors.count < cap, vals[idx] > 0 else { continue }
            let sigma = vals[idx].squareRoot()
            guard sigma > cutoff else { break }
            let rightVec = matmul(a.T, eigenvectors[0..., idx]) / sigma
            eval(rightVec)
            vectors.append(rightVec.asArray(Float.self))
            sigmas.append(sigma)
        }
        return (vectors, sigmas)
    }

    private static func tallSingularBasis(
        _ a: MLXArray,
        dim: Int,
        cap: Int,
        tolerance: Float
    ) -> ([[Float]], [Float]) {
        let gram = matmul(a.T, a)
        eval(gram)
        let (eigenvalues, eigenvectors) = MLXLinalg.eigh(gram, stream: .cpu)
        eval(eigenvalues, eigenvectors)
        let vals = eigenvalues.asArray(Float.self)
        let peakLambda = max(vals.last ?? 0, 0)
        guard peakLambda > 1e-18 else { return ([], []) }
        let sigmaMax = peakLambda.squareRoot()
        let cutoff = max(tolerance * sigmaMax, 1e-9)
        var vectors: [[Float]] = [], sigmas: [Float] = []
        for idx in stride(from: dim - 1, through: 0, by: -1) {
            guard vectors.count < cap, vals[idx] > 0 else { continue }
            let sigma = vals[idx].squareRoot()
            guard sigma > cutoff else { break }
            let col = eigenvectors[0..., idx]
            eval(col)
            vectors.append(col.asArray(Float.self))
            sigmas.append(sigma)
        }
        return (vectors, sigmas)
    }
}

private final class SequenceInputTap: Linear {
    private let original: (MLXArray) -> MLXArray
    private let record: (MLXArray) -> Void

    init(_ linear: Linear, record: @escaping (MLXArray) -> Void) {
        original = { linear($0) }
        self.record = record
        super.init(weight: linear.weight, bias: linear.bias)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        record(x[0].asType(.float32))
        return original(x)
    }
}
