import Foundation
import MLX
import BasaltCore

struct ProcrustesAlignment {
    var meanCosine: Float
    var recallAt1: Float
    var recallAt5: Float
    var recallAt10: Float

    var sigma: Float? { try? CalibratedNoiseInjector.sigma(forMeanCosine: meanCosine) }
}

struct OrthogonalProcrustes {
    static let recallRows = 1024

    var q: MLXArray
    var meanX: MLXArray
    var meanY: MLXArray

    static func fit(x: MLXArray, y: MLXArray) throws(BasaltError) -> OrthogonalProcrustes {
        try check(x, y)
        let meanX = x.asType(.float32).mean(axis: 0)
        let meanY = y.asType(.float32).mean(axis: 0)
        let m = matmul(normalized(x, meanX).T, normalized(y, meanY)).asType(.float64, stream: .cpu)
        let (u, _, vt) = MLXLinalg.svd(m, stream: .cpu)
        let k = min(x.dim(1), y.dim(1))
        let q = matmul(u.asType(.float32, stream: .cpu)[0..., ..<k], vt.asType(.float32, stream: .cpu)[..<k, 0...])
        eval(q, meanX, meanY)
        return OrthogonalProcrustes(q: q, meanX: meanX, meanY: meanY)
    }

    func align(_ x: MLXArray) -> MLXArray {
        matmul(Self.normalized(x, meanX), q)
    }

    func evaluate(x: MLXArray, y: MLXArray) throws(BasaltError) -> ProcrustesAlignment {
        try Self.check(x, y)
        guard x.dim(1) == q.dim(0), y.dim(1) == q.dim(1) else {
            throw .invalidRequestStructure(details: "procrustes: pairs are \(x.dim(1)) × \(y.dim(1)) wide, the rotation is \(q.dim(0)) × \(q.dim(1))")
        }
        let aligned = align(x)
        let targets = Self.normalized(y, meanY)
        let meanCosine = (aligned * targets).sum(axis: 1).mean().item(Float.self)
        var hits = [0, 0, 0]
        for start in stride(from: 0, to: x.dim(0), by: Self.recallRows) {
            let rows = start ..< min(x.dim(0), start + Self.recallRows)
            let similarity = matmul(aligned[rows], targets.T)
            let own = similarity[MLXArray(0 ..< rows.count), MLXArray(Int32(rows.lowerBound) ..< Int32(rows.upperBound))]
            let rank = (similarity .> own.expandedDimensions(axis: 1)).sum(axis: 1)
            for (index, k) in [1, 5, 10].enumerated() { hits[index] += (rank .< k).sum().item(Int.self) }
        }
        let n = Float(x.dim(0))
        return ProcrustesAlignment(meanCosine: meanCosine, recallAt1: Float(hits[0]) / n, recallAt5: Float(hits[1]) / n, recallAt10: Float(hits[2]) / n)
    }

    func save(to url: URL, alignment: ProcrustesAlignment) throws {
        var arrays = ["q": q, "mean_x": meanX, "mean_y": meanY, "mean_cosine": MLXArray(alignment.meanCosine)]
        if let sigma = alignment.sigma { arrays["sigma"] = MLXArray(sigma) }
        try MLX.save(arrays: arrays, url: url)
    }

    static func load(from url: URL) throws -> (OrthogonalProcrustes, meanCosine: Float) {
        let arrays = try MLX.loadArrays(url: url)
        guard let q = arrays["q"], let meanX = arrays["mean_x"], let meanY = arrays["mean_y"], let cosine = arrays["mean_cosine"],
              q.ndim == 2, meanX.shape == [q.dim(0)], meanY.shape == [q.dim(1)] else {
            throw BasaltError.invalidRequestStructure(details: "procrustes: \(url.path) lacks q [dx, dy], mean_x [dx], mean_y [dy] or mean_cosine")
        }
        return (OrthogonalProcrustes(q: q, meanX: meanX, meanY: meanY), cosine.item(Float.self))
    }

    static func check(_ x: MLXArray, _ y: MLXArray) throws(BasaltError) {
        guard x.ndim == 2, y.ndim == 2, x.dim(0) == y.dim(0), x.dim(0) > 0 else {
            throw .invalidRequestStructure(details: "procrustes: pairs must be two 2-D matrices with the same positive row count, got \(x.shape) and \(y.shape)")
        }
    }

    static func normalized(_ rows: MLXArray, _ mean: MLXArray) -> MLXArray {
        let centered = rows.asType(.float32) - mean
        return centered / maximum(sqrt((centered * centered).sum(axis: -1, keepDims: true)), 1e-9)
    }
}

enum CalibratedNoiseInjector {
    static func sigma(forMeanCosine c: Float) throws(BasaltError) -> Float {
        guard c > 0, c <= 1 else {
            throw .invalidRequestStructure(details: "noise calibration: the mean cosine must be in (0, 1], got \(c)")
        }
        return (1 / (c * c) - 1).squareRoot()
    }

    static func perturb(_ embeddings: MLXArray, sigma: Float, key: MLXArray? = nil) -> MLXArray {
        let noisy = embeddings.asType(.float32) + sigma * MLXRandom.normal(embeddings.shape, dtype: .float32, key: key)
        return noisy / maximum(sqrt((noisy * noisy).sum(axis: -1, keepDims: true)), 1e-9)
    }
}
