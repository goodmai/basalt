import Foundation
import Testing
import MLX
@testable import BasaltCore
@testable import BasaltInference

@Suite("Spectral statistics of sampled linear layers next to the 2-bit tensor_pack RMSE on real models (#358)", .serialized)
struct SpectralWeightRealModelTests {
    struct Row: Codable {
        var tensor: String
        var spectrum: WeightSpectrum
        var rmse: Float
        var relativeError: Float
    }

    static let env = ProcessInfo.processInfo.environment

    static var models: [String] {
        guard env["BASALT_E2E"] == "1" else { return [] }
        return (env["BASALT_E2E_SPECTRAL_MODELS"] ?? "").split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
    }

    static var layerCount: Int { env["BASALT_E2E_SPECTRAL_LAYERS"].flatMap { Int($0) } ?? 40 }

    static let metrics: [(String, @Sendable (Row) -> Double)] = [
        ("alpha", { $0.spectrum.alpha ?? .nan }),
        ("massAboveBulk", { $0.spectrum.massAboveBulk }),
        ("rankFraction", { Double($0.spectrum.rank) / Double(min($0.spectrum.rows, $0.spectrum.columns)) }),
        ("noiseVariance", { $0.spectrum.noiseVariance }),
    ]

    static func sample(_ tensors: [TensorInfo], count: Int) -> [String] {
        let linear = tensors.filter { $0.quantized && $0.shape.count == 2 && $0.name.hasSuffix(".weight") && !$0.name.contains("embed") }
        guard linear.count > count else { return linear.map(\.name) }
        return (0 ..< count).map { linear[$0 * linear.count / count].name }
    }

    static func measure(_ tensor: String, engine: MLXInferenceEngine) async throws -> Row {
        var request = TensorRequest(name: tensor, maxValues: 0)
        request.spectrum = true
        let spectrum = try #require(try await engine.tensor(request).spectrum)
        let pack = try await engine.pack(PackRequest(tensor: tensor, format: .affine(bits: 2, groupSize: 64)))
        try await engine.undo(pack.overlay.id)
        return Row(tensor: tensor, spectrum: spectrum, rmse: pack.rmse, relativeError: pack.relativeError)
    }

    static func ranks(_ values: [Double]) -> [Double] {
        let order = values.indices.sorted { values[$0] < values[$1] }
        var ranks = [Double](repeating: 0, count: values.count)
        var start = 0
        while start < order.count {
            var end = start
            while end + 1 < order.count, values[order[end + 1]] == values[order[start]] { end += 1 }
            for index in start ... end { ranks[order[index]] = Double(start + end) / 2 }
            start = end + 1
        }
        return ranks
    }

    static func spearman(_ x: [Double], _ y: [Double]) -> Double {
        let pairs = zip(x, y).filter { $0.0.isFinite && $0.1.isFinite }
        guard pairs.count > 2 else { return .nan }
        let a = ranks(pairs.map(\.0)), b = ranks(pairs.map(\.1))
        let meanA = a.reduce(0, +) / Double(a.count), meanB = b.reduce(0, +) / Double(b.count)
        let covariance = zip(a, b).reduce(0) { $0 + ($1.0 - meanA) * ($1.1 - meanB) }
        let spread = (a.reduce(0) { $0 + ($1 - meanA) * ($1 - meanA) } * b.reduce(0) { $0 + ($1 - meanB) * ($1 - meanB) }).squareRoot()
        return spread > 0 ? covariance / spread : .nan
    }

    @Test("Spearman rank correlation of each spectral statistic with the 2-bit affine RMSE over a stratified layer sample",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_SPECTRAL_MODELS"), arguments: models)
    func sweep(path: String) async throws {
        let log = try await ModelRunLog("spectral-weights", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine()
        try await log.step("load") { try await engine.load(modelPath: path) }
        let tensors = Self.sample(try await engine.tensors(matching: nil), count: Self.layerCount)
        log.record("sample", "\(tensors.count) quantized 2-D linear weights, affine 2-bit g64 pack, spectrum of the dequantized weight")
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        var rows: [Row] = []
        for tensor in tensors {
            do {
                let row = try await Self.measure(tensor, engine: engine)
                rows.append(row)
                log.record("layer", String(decoding: try encoder.encode(row), as: UTF8.self))
            } catch {
                log.record("SKIP layer", "\(tensor): \(error)")
            }
        }
        let seconds = rows.map(\.spectrum.seconds)
        log.record("cost", String(format: "%d layers, spectrum %.2f s total, %.2f s mean, %.2f s max", rows.count, seconds.reduce(0, +), seconds.reduce(0, +) / Double(max(rows.count, 1)), seconds.max() ?? 0))
        for (name, metric) in Self.metrics {
            let values = rows.map(metric)
            let rmse = Self.spearman(values, rows.map { Double($0.rmse) })
            let relative = Self.spearman(values, rows.map { Double($0.relativeError) })
            log.record("spearman", String(format: "%@ vs rmse %.3f, vs relative error %.3f (n=%d)", name, rmse, relative, rows.count))
        }
        #expect(rows.count * 2 >= tensors.count, "fewer than half of the sampled layers were measured; see the SKIP lines in \(log.url.path)")
    }
}
