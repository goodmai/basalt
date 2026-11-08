import Foundation
import Testing
import MLX
@testable import BasaltCore
@testable import BasaltInference

@Suite("Orthogonal Procrustes: an SVD rotation between paired embeddings and calibrated noise (#387)")
struct OrthogonalProcrustesTests {
    static func orthogonal(_ side: Int, seed: UInt64) -> MLXArray {
        let (q, _) = MLXLinalg.qr(MLXRandom.normal([side, side], key: MLXRandom.key(seed)), stream: .cpu)
        return q
    }

    static func frobenius(_ a: MLXArray, _ b: MLXArray) -> Float {
        sqrt(((a - b) * (a - b)).sum()).item(Float.self)
    }

    @Test("On Y = X·Q the rotation comes back within 1e-5 Frobenius and every pair retrieves its partner at rank 1")
    func recoversRotation() throws {
        let truth = Self.orthogonal(128, seed: 1)
        let x = MLXRandom.normal([1000, 128], key: MLXRandom.key(2))
        let y = matmul(x - x.mean(axis: 0), truth)
        let fit = try OrthogonalProcrustes.fit(x: x, y: y)
        #expect(Self.frobenius(fit.q, truth) < 1e-5)
        let alignment = try fit.evaluate(x: x, y: y)
        #expect(alignment.recallAt1 == 1 && alignment.recallAt10 == 1)
        #expect(abs(alignment.meanCosine - 1) < 1e-4)
        #expect(try #require(alignment.sigma) < 0.02)
    }

    @Test("With 768-wide X and 512-wide Y, Qᵀ·Q is the 512 identity within 1e-5", arguments: [(768, 512), (512, 768)])
    func rectangular(dx: Int, dy: Int) throws {
        let fit = try OrthogonalProcrustes.fit(x: MLXRandom.normal([2000, dx], key: MLXRandom.key(3)), y: MLXRandom.normal([2000, dy], key: MLXRandom.key(4)))
        #expect(fit.q.shape == [dx, dy])
        let gram = dx >= dy ? matmul(fit.q.T, fit.q) : matmul(fit.q, fit.q.T)
        #expect(MLX.abs(gram - MLXArray.eye(min(dx, dy))).max().item(Float.self) < 1e-5)
    }

    @Test("Held-out pairs score the fit: shuffled partners lose recall, and unrelated data gives no noise calibration")
    func heldOut() throws {
        let truth = Self.orthogonal(32, seed: 5)
        let x = MLXRandom.normal([600, 32], key: MLXRandom.key(6))
        let y = matmul(x, truth) + 0.3 * MLXRandom.normal([600, 32], key: MLXRandom.key(7))
        let fit = try OrthogonalProcrustes.fit(x: x[..<400], y: y[..<400])
        let held = try fit.evaluate(x: x[400...], y: y[400...])
        #expect(held.recallAt1 > 0.9 && held.recallAt5 >= held.recallAt1 && held.recallAt10 >= held.recallAt5)
        let shuffled = try fit.evaluate(x: x[400...], y: y[400...][MLXArray((0 ..< 200).map { Int32(($0 + 1) % 200) })])
        #expect(shuffled.recallAt1 < 0.05)
        #expect(ProcrustesAlignment(meanCosine: -0.1, recallAt1: 0, recallAt5: 0, recallAt10: 0).sigma == nil)
    }

    @Test("Mismatched rows, 1-D input or a rotation of the wrong width are refused")
    func refused() throws {
        #expect(throws: BasaltError.self) { try OrthogonalProcrustes.fit(x: MLXArray.zeros([4, 3]), y: MLXArray.zeros([5, 3])) }
        #expect(throws: BasaltError.self) { try OrthogonalProcrustes.fit(x: MLXArray.zeros([4]), y: MLXArray.zeros([4])) }
        let fit = try OrthogonalProcrustes.fit(x: MLXRandom.normal([10, 3], key: MLXRandom.key(8)), y: MLXRandom.normal([10, 3], key: MLXRandom.key(9)))
        #expect(throws: BasaltError.self) { try fit.evaluate(x: MLXArray.zeros([4, 2]), y: MLXArray.zeros([4, 3])) }
    }

    @Test("Saved and loaded, the rotation and means are bit-identical; a file without them is refused")
    func roundTrip() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "procrustes-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let fit = try OrthogonalProcrustes.fit(x: MLXRandom.normal([50, 6], key: MLXRandom.key(10)), y: MLXRandom.normal([50, 4], key: MLXRandom.key(11)))
        let url = directory.appending(path: "q.safetensors")
        try fit.save(to: url, alignment: ProcrustesAlignment(meanCosine: 0.5, recallAt1: 0, recallAt5: 0, recallAt10: 0))
        let (loaded, cosine) = try OrthogonalProcrustes.load(from: url)
        #expect(loaded.q.asArray(Float.self) == fit.q.asArray(Float.self))
        #expect(loaded.meanX.asArray(Float.self) == fit.meanX.asArray(Float.self) && loaded.meanY.asArray(Float.self) == fit.meanY.asArray(Float.self))
        #expect(cosine == 0.5)
        let broken = directory.appending(path: "broken.safetensors")
        try MLX.save(arrays: ["q": fit.q, "mean_x": fit.meanY, "mean_y": fit.meanY, "mean_cosine": MLXArray(Float(0.5))], url: broken)
        #expect(throws: BasaltError.self) { try OrthogonalProcrustes.load(from: broken) }
    }

    @Test("σ = √(c⁻² − 1): 0 at c = 1, √3 at c = 0.5, refused outside (0, 1]", arguments: [(Float(1), Float(0)), (Float(0.5), Float(3).squareRoot())])
    func sigma(cosine: Float, expected: Float) throws {
        #expect(abs(try CalibratedNoiseInjector.sigma(forMeanCosine: cosine) - expected) < 1e-5)
    }

    @Test("A mean cosine of 0, below 0, above 1 or NaN is refused", arguments: [Float(0), -0.5, 1.5, .nan])
    func sigmaRefused(cosine: Float) {
        #expect(throws: BasaltError.self) { try CalibratedNoiseInjector.sigma(forMeanCosine: cosine) }
    }

    @Test("Perturbed embeddings stay unit length, follow the key, and move further from the input as σ grows")
    func perturb() {
        let y = MLXRandom.normal([256, 64], key: MLXRandom.key(12))
        let unit = y / sqrt((y * y).sum(axis: -1, keepDims: true))
        let small = CalibratedNoiseInjector.perturb(unit, sigma: 0.1, key: MLXRandom.key(13))
        let large = CalibratedNoiseInjector.perturb(unit, sigma: 3, key: MLXRandom.key(13))
        #expect(MLX.abs(sqrt((large * large).sum(axis: -1)) - 1).max().item(Float.self) < 1e-5)
        #expect(small.asArray(Float.self) == CalibratedNoiseInjector.perturb(unit, sigma: 0.1, key: MLXRandom.key(13)).asArray(Float.self))
        #expect((small * unit).sum(axis: -1).mean().item(Float.self) > (large * unit).sum(axis: -1).mean().item(Float.self))
    }
}
