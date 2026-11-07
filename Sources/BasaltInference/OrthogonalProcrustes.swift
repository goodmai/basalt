import Foundation
import MLX
import MLXNN
import BasaltCore

public struct OrthogonalProcrustes {
    public var q: MLXArray
    public var meanX: MLXArray
    public var meanY: MLXArray
    public var meanCosine: Float
    public var sigma: Float
    public var recallAt1: Float
    public var recallAt5: Float
    public var recallAt10: Float

    public init(q: MLXArray, meanX: MLXArray, meanY: MLXArray, meanCosine: Float, sigma: Float, recallAt1: Float = 0, recallAt5: Float = 0, recallAt10: Float = 0) {
        self.q = q
        self.meanX = meanX
        self.meanY = meanY
        self.meanCosine = meanCosine
        self.sigma = sigma
        self.recallAt1 = recallAt1
        self.recallAt5 = recallAt5
        self.recallAt10 = recallAt10
    }

    public static func fit(x: MLXArray, y: MLXArray) throws -> OrthogonalProcrustes {
        let x32 = x.asType(.float32)
        let y32 = y.asType(.float32)
        guard x32.ndim == 2, y32.ndim == 2 else {
            throw BasaltError.invalidRequestStructure(details: "OrthogonalProcrustes: inputs must be 2-D")
        }
        let n = x32.dim(0)
        guard n == y32.dim(0), n > 0 else {
            throw BasaltError.invalidRequestStructure(details: "OrthogonalProcrustes: paired matrices need the same positive row count")
        }
        let dx = x32.dim(1)
        let dy = y32.dim(1)

        let meanX = MLX.mean(x32, axis: 0)
        let meanY = MLX.mean(y32, axis: 0)
        let xc = x32 - meanX
        let yc = y32 - meanY

        let xn = MLX.sqrt(MLX.sum(xc * xc, axis: 1, keepDims: true))
        let yn = MLX.sqrt(MLX.sum(yc * yc, axis: 1, keepDims: true))
        let xhat = xc / MLX.maximum(xn, MLXArray(Float(1e-9)))
        let yhat = yc / MLX.maximum(yn, MLXArray(Float(1e-9)))

        var m = MLXArray.zeros([dx, dy], dtype: .float32)
        let chunk = 1024
        for start in stride(from: 0, to: n, by: chunk) {
            let end = min(n, start + chunk)
            let xm = xhat[start ..< end]
            let ym = yhat[start ..< end]
            m = m + MLX.matmul(xm.T, ym)
            eval(m)
        }

        let m64 = MLXArray(m.asArray(Float.self).map(Double.init), [dx, dy])
        let (u64, _, vt64) = MLXLinalg.svd(m64, stream: .cpu)
        eval(u64, vt64)
        let u = u64.asType(.float32, stream: .cpu)
        let vt = vt64.asType(.float32, stream: .cpu)

        let k = min(dx, dy)
        let uK = u[0..., 0 ..< k]
        let vtK = vt[0 ..< k, 0...]
        let q = MLX.matmul(uK, vtK, stream: .cpu)

        let aligned = MLX.matmul(xhat, q)
        let cosines = MLX.sum(aligned * yhat, axis: 1)
        eval(cosines)
        let meanCos = cosines.mean().item(Float.self)

        let sigma = try CalibratedNoiseInjector.sigma(forMeanCosine: meanCos)

        let (r1, r5, r10) = Self.recall(aligned: aligned, targets: yhat)

        return OrthogonalProcrustes(q: q, meanX: meanX, meanY: meanY, meanCosine: meanCos, sigma: sigma, recallAt1: r1, recallAt5: r5, recallAt10: r10)
    }

    public func save(to url: URL) throws {
        try Self.save(self, to: url)
    }

    public static func save(_ result: OrthogonalProcrustes, to url: URL) throws {
        let arrays: [String: MLXArray] = [
            "q": result.q,
            "mean_x": result.meanX,
            "mean_y": result.meanY,
            "mean_cosine": MLXArray(result.meanCosine),
            "sigma": MLXArray(result.sigma),
        ]
        try MLX.save(arrays: arrays, url: url)
    }

    public static func load(from url: URL) throws -> OrthogonalProcrustes {
        let arrays = try MLX.loadArrays(url: url)
        guard let q = arrays["q"], let mx = arrays["mean_x"], let my = arrays["mean_y"],
              let mc = arrays["mean_cosine"], let sig = arrays["sigma"] else {
            throw BasaltError.invalidRequestStructure(details: "OrthogonalProcrustes: missing tensors in \(url.path)")
        }
        return OrthogonalProcrustes(
            q: q, meanX: mx, meanY: my,
            meanCosine: mc.item(Float.self), sigma: sig.item(Float.self))
    }

    private static func recall(aligned: MLXArray, targets: MLXArray) -> (Float, Float, Float) {
        let n = aligned.dim(0)
        guard n > 0 else { return (0, 0, 0) }
        let sim = MLX.matmul(aligned, targets.T)
        eval(sim)
        var r1 = 0, r5 = 0, r10 = 0
        for i in 0 ..< n {
            let row = sim[i].asArray(Float.self)
            let indexed = row.enumerated().sorted { $0.element > $1.element }
            let rank = indexed.firstIndex { $0.offset == i } ?? n
            if rank < 1 { r1 += 1 }
            if rank < 5 { r5 += 1 }
            if rank < 10 { r10 += 1 }
        }
        let denom = Float(n)
        return (Float(r1) / denom, Float(r5) / denom, Float(r10) / denom)
    }
}

public enum CalibratedNoiseInjector {
    public static func sigma(forMeanCosine c: Float) throws -> Float {
        guard c > 0, c <= 1 else {
            throw BasaltError.invalidRequestStructure(details: "CalibratedNoiseInjector: mean cosine must be in (0, 1]")
        }
        return (1.0 / (c * c) - 1.0).squareRoot()
    }

    public static func perturb(_ embeddings: MLXArray, sigma: Float, seed: UInt64 = 42) -> MLXArray {
        let eps = MLXRandom.normal(embeddings.shape, dtype: .float32, key: MLXRandom.key(seed))
        let noisy = embeddings.asType(.float32) + MLXArray(sigma) * eps
        let norms = MLX.sqrt(MLX.sum(noisy * noisy, axis: -1, keepDims: true))
        return noisy / MLX.maximum(norms, MLXArray(Float(1e-9)))
    }
}
