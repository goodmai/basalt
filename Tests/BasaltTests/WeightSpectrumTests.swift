import Foundation
import Testing
import MLX
import MLXNN
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Weight spectrum: Marchenko–Pastur fit, Gavish–Donoho rank and power-law alpha of a dequantized matrix (#358)", .serialized)
struct WeightSpectrumTests {
    static func planted(rows: Int, columns: Int, rank: Int, seed: UInt64) -> MLXArray {
        let left = MLXRandom.normal([rows, rank], key: MLXRandom.key(seed))
        let right = MLXRandom.normal([rank, columns], key: MLXRandom.key(seed + 1))
        return matmul(left, right) + MLXRandom.normal([rows, columns], key: MLXRandom.key(seed + 2))
    }

    @Test("Gavish–Donoho returns the planted rank of a noisy rank-8 matrix, tall (Gram over two row chunks) and wide", arguments: [false, true])
    func plantedRank(wide: Bool) throws {
        let tall = Self.planted(rows: 4608, columns: 256, rank: 8, seed: 7)
        let matrix = wide ? tall.T : tall
        let spectrum = try WeightSpectrum.measure(rows: matrix.dim(0), columns: matrix.dim(1)) { matrix[$0] }
        #expect(spectrum.rank == 8, "\(spectrum)")
        #expect(spectrum.massAboveBulk > 0.8, "\(spectrum)")
        #expect(spectrum.rows == matrix.dim(0) && spectrum.columns == matrix.dim(1) && spectrum.seconds >= 0)
    }

    @Test("On a Gaussian matrix the Marchenko–Pastur fit recovers sigma² and both bulk edges, with no mass above the bulk and rank 0")
    func gaussianBulk() throws {
        let matrix = MLXRandom.normal([2048, 1024], scale: 0.5, key: MLXRandom.key(11))
        let spectrum = try WeightSpectrum.measure(rows: 2048, columns: 1024) { matrix[$0] }
        let root = 0.5.squareRoot()
        #expect(abs(spectrum.noiseVariance - 0.25) < 0.01, "\(spectrum)")
        #expect(abs(spectrum.bulkMax - 0.25 * pow(1 + root, 2)) < 0.03, "\(spectrum)")
        #expect(abs(spectrum.bulkMin - 0.25 * pow(1 - root, 2)) < 0.005, "\(spectrum)")
        #expect(spectrum.massAboveBulk < 0.01 && spectrum.rank == 0, "\(spectrum)")
    }

    @Test("The Hill estimator with a KS-chosen tail returns alpha 3 on Pareto quantiles of density exponent 3")
    func powerLawAlpha() throws {
        let count = 2000
        let values = (0 ..< count).map { pow(1 - (Double($0) + 0.5) / Double(count), -0.5) }
        let spectrum = try #require(WeightSpectrum(eigenvalues: values, rows: count, columns: count))
        let alpha = try #require(spectrum.alpha)
        #expect(abs(alpha - 3) < 0.05 && spectrum.tailSize >= 10, "\(spectrum)")
    }

    @Test("The Marchenko–Pastur median at ratio 1 is 0.6528, as Gavish and Donoho tabulate; a zero or non-finite spectrum is nil")
    func edges() {
        #expect(abs(WeightSpectrum.marchenkoPasturMedian(1) - 0.6528) < 1e-3)
        #expect(WeightSpectrum(eigenvalues: [0, 0, 0], rows: 3, columns: 3) == nil)
        #expect(WeightSpectrum(eigenvalues: [1, .nan, 2], rows: 3, columns: 3) == nil)
        #expect(WeightSpectrum(eigenvalues: [1, 2, 3], rows: 3, columns: 5)?.alpha == nil)
    }

    @Test("One expert of a 3-D stack is measured through the slice; a vector and an unsliced stack are refused")
    func stackedExpert() throws {
        let stack = stacked([Self.planted(rows: 512, columns: 256, rank: 2, seed: 3), Self.planted(rows: 512, columns: 256, rank: 5, seed: 5)])
        let view = WeightView(name: "experts.weight", array: stack, quantization: nil)
        let second = try MLXInferenceEngine.spectrum(view, ranges: [1 ..< 2, 0 ..< 512, 0 ..< 256])
        #expect(second.rank == 5 && second.rows == 512 && second.columns == 256, "\(second)")
        #expect(throws: BasaltError.self) { try MLXInferenceEngine.spectrum(view, ranges: [0 ..< 2, 0 ..< 512, 0 ..< 256]) }
        let vector = WeightView(name: "norm.weight", array: MLXArray.ones([64]), quantization: nil)
        #expect(throws: BasaltError.self) { try MLXInferenceEngine.spectrum(vector, ranges: [0 ..< 64]) }
    }

    @Test("model_tensor with spectrum on a tiny Qwen3 down_proj, 8-bit, reports its spectrum and a sliced sub-matrix's shape")
    func engineReport() async throws {
        let model = try TinyModel.make(.qwen3)
        quantize(model: model, groupSize: 32, bits: 8)
        let engine = await XRayTensorTests.engine(model)
        var request = TensorRequest(name: XRayTensorTests.downProj, maxValues: 0)
        request.spectrum = true
        let whole = try #require(try await engine.tensor(request).spectrum)
        #expect(whole.rows == 64 && whole.columns == 128 && whole.rank <= 64 && whole.noiseVariance > 0, "\(whole)")
        request.slice = [8 ..< 40, 0 ..< 96]
        let part = try #require(try await engine.tensor(request).spectrum)
        #expect(part.rows == 32 && part.columns == 96, "\(part)")
        #expect(try await engine.tensor(TensorRequest(name: XRayTensorTests.downProj, maxValues: 0)).spectrum == nil)
    }

    @Test("The MCP model_tensor call takes spectrum as a boolean, off by default")
    func mcpArgument() throws {
        var expected = TensorRequest(name: "w")
        #expect(try ResearchCall(.modelTensor, ["name": "w"]) == .tensor(expected))
        expected.spectrum = true
        #expect(try ResearchCall(.modelTensor, ["name": "w", "spectrum": true]) == .tensor(expected))
        #expect(throws: MCPFailure.self) { try ResearchCall(.modelTensor, ["name": "w", "spectrum": "yes"]) }
    }
}
