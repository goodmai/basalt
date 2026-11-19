import Foundation
import MLX
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Low-Bandwidth DiLoCo Outer Nesterov & EF21 PowerSGD / QJL Compression (#567)")
struct DiLoCoAndEF21Tests {
    @Test("1-bit QJL gradient sketch is statistically unbiased across Monte Carlo seeds")
    func qjlSketchZeroBias() {
        let d = 32
        let target: [Float] = (0 ..< d).map { Float(cos(Double($0) * 0.31) * 1.4 - 0.25) }
        var meanEstimate = [Float](repeating: 0, count: d)
        let trials = 256
        for t in 0 ..< trials {
            let pkt = QJLSketchCompressor.encode(target, sketchRows: 128, seed: UInt64(10_000 + t * 97))
            #expect(pkt.wireBytes == 32)
            let dec = QJLSketchCompressor.decode(pkt)
            for i in 0 ..< d {
                meanEstimate[i] += dec[i] / Float(trials)
            }
        }
        let targetNorm = target.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        let errNorm = zip(target, meanEstimate).reduce(Float(0)) { $0 + ($1.0 - $1.1) * ($1.0 - $1.1) }.squareRoot()
        #expect(errNorm / targetNorm < 0.08)
    }

    @Test("EF21 PowerSGD residual norm contracts strictly monotonically across steps and achieves >4x compression")
    func ef21MonotonicResidualContraction() {
        let rows = 32
        let cols = 32
        let target: [Float] = (0 ..< (rows * cols)).map { idx in
            let r = Double(idx / cols)
            let c = Double(idx % cols)
            return Float(sin(r * 0.21) * cos(c * 0.19) + 0.35 * sin((r + c) * 0.43))
        }
        var ef = EF21ResidualState(dimension: rows * cols)
        var prevNorm = target.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        for _ in 0 ..< 8 {
            let step = ef.stepPowerSGD(target: target, rows: rows, cols: cols, rank: 2)
            #expect(step.residualNorm < prevNorm)
            #expect(rows * cols / step.packet.wireFloats >= 8)
            prevNorm = step.residualNorm
        }
        #expect(prevNorm < 1.0)
    }

    @Test("2-worker DiLoCo (H=8) with outer Nesterov + EF21 PowerSGD converges cleanly on TinyModel")
    func twoWorkerDiLoCoConvergence() throws {
        let (w0, w1) = try withRandomState(MLXRandom.RandomState(seed: 567)) {
            let m0 = try TinyModel.make(.qwen3)
            let m1 = try TinyModel.make(.qwen3)
            _ = try MLXInferenceEngine.attachLoRA(to: m0, layers: [0, 1], rank: 8)
            _ = try MLXInferenceEngine.attachLoRA(to: m1, layers: [0, 1], rank: 8)
            return (m0, m1)
        }
        let syncURL = FileManager.default.temporaryDirectory.appending(path: "diloco-init-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: syncURL) }
        eval(w0 as Module)
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: (w0 as Module).parameters().flattened()), url: syncURL)
        (w1 as Module).update(parameters: ModuleParameters.unflattened(try MLX.loadArrays(url: syncURL)))
        eval(w1 as Module)

        var corpus: [[Int32]] = []
        for idx in 0 ..< 4 {
            var row: [Int32] = []
            for pos in 0 ..< 12 {
                row.append(Int32((idx * 13 + pos * 5 + 7) % 200))
            }
            corpus.append(row)
        }
        let b0 = [Array(corpus[0 ..< 2])]
        let b1 = [Array(corpus[2 ..< 4])]

        let report = try DiLoCoTrainer.runSimulatedCluster(
            workerModels: [w0, w1],
            workerBatches: [b0, b1],
            evalBatch: corpus,
            schedule: DiLoCoScheduleSpec(
                outerRounds: 3,
                outerConfig: DiLoCoOuterConfig(innerSteps: 8, outerLearningRate: 0.7, nesterovMomentum: 0.85),
                innerLearningRate: 5e-3,
                powerSGDRank: 2
            )
        )
        #expect(report.finalLoss < report.initialLoss - 0.05)
        #expect(report.compressionRatio >= 2.0)
    }
}
