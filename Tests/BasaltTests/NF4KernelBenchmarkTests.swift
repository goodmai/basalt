import Foundation
import Darwin
import Testing
import MLX
import MLXLMCommon
import BasaltCore

@Suite("Native nf4 g32 quantizedMatmul against affine 4-bit g64 at equal 4.5 bpw, verify-window row counts (#357 phase c)", .serialized)
struct NF4KernelBenchmarkTests {
    typealias Bench = SmallMQuantizedMatmulBenchmarkTests

    static let rows = [1, 2, 4, 8]

    static func affineMicros(_ shape: Bench.Shape, x: MLXArray, dense: MLXArray) -> Double {
        let (wq, scales, biases) = quantized(dense, groupSize: 64, bits: 4)
        eval([wq, scales] + (biases.map { [$0] } ?? []))
        let run = { eval((0 ..< 16).map { _ in quantizedMM(x, wq, scales: scales, biases: biases, groupSize: 64, bits: 4) }) }
        run()
        return Bench.microseconds(run) / 16
    }

    static func nf4Micros(x: MLXArray, codes: MLXArray, scales: MLXArray) -> Double {
        let run = { eval((0 ..< 16).map { _ in quantizedMM(x, codes, scales: scales, biases: nil, groupSize: 32, bits: 4, mode: .nf4) }) }
        run()
        return Bench.microseconds(run) / 16
    }

    static func kernelError(x: MLXArray, codes: MLXArray, scales: MLXArray) -> Float {
        let reference = matmul(x.asType(.float32), dequantized(codes, scales: scales, biases: nil, groupSize: 32, bits: 4, mode: .nf4).asType(.float32).T)
        let kernel = quantizedMM(x, codes, scales: scales, biases: nil, groupSize: 32, bits: 4, mode: .nf4).asType(.float32)
        return (abs(kernel - reference).max() / abs(reference).max()).item(Float.self)
    }

    @Test("µs per call and relative max error at M = 1, 2, 4 and 8 on the Qwen3.8-27B projection shapes",
          .enabled(if: Bench.enabled, "set BASALT_BENCH_SMALLM=1 and run a release build"))
    func nf4AgainstAffine() throws {
        guard case .taken = try ModelRunLog.claim(ModelRunLog.gpuLock, pid: getpid()) else {
            Issue.record("\(ModelRunLog.gpuLock) is held by another run; relaunch when it is free")
            return
        }
        var lock = GPULockGuard(path: ModelRunLog.gpuLock, pid: getpid())
        defer { _ = lock.release() }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let url = root.appendingPathComponent("logs/nf4-kernel-\(Date().ISO8601Format(.iso8601.year().month().day())).log")
        Bench.log("RUN \(Date().ISO8601Format()) commit \(ModelRunLog.output("/usr/bin/git", ["-C", root.path, "rev-parse", "HEAD"])) machine \(SystemProfiler.detectChipModel()) load \(ModelRunLog.load())", to: url)
        Bench.log("shape | K x N | M | affine g64 us | nf4 g32 us | nf4 / affine | kernel rel max err", to: url)
        for shape in Bench.shapes where shape.name != "lm_head" {
            let dense = MLXRandom.normal([shape.outputs, shape.inputs], dtype: .bfloat16, scale: 0.02)
            let (codes, scales, _) = quantized(dense, groupSize: 32, bits: 4, mode: .nf4)
            eval(codes, scales)
            for m in Self.rows {
                let x = MLXRandom.normal([m, shape.inputs], dtype: .bfloat16)
                let affine = Self.affineMicros(shape, x: x, dense: dense)
                let nf4 = Self.nf4Micros(x: x, codes: codes, scales: scales)
                let error = Self.kernelError(x: x, codes: codes, scales: scales)
                Bench.log("\(shape.name) | \(shape.inputs) x \(shape.outputs) | \(m) | " + String(format: "%.1f | %.1f | %.2fx | %.2e", affine, nf4, nf4 / affine, error), to: url)
                #expect(error < 1e-2)
            }
        }
        Bench.log("END \(Date().ISO8601Format()) log \(url.path)", to: url)
    }
}
