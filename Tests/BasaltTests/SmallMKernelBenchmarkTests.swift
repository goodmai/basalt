import Foundation
import Darwin
import Testing
import MLX
import BasaltCore
@testable import BasaltInference

@Suite("Small-M 4-bit affine kernel against stock quantizedMatmul at M = 1...8 on Qwen3.8-27B shapes (#208 phase 2)", .serialized)
struct SmallMKernelBenchmarkTests {
    typealias Bench = SmallMQuantizedMatmulBenchmarkTests
    typealias Copies = [(MLXArray, MLXArray, MLXArray?)]

    static func kernelMicros(_ copies: Copies, x: MLXArray) -> Double {
        let rounds = max(1, 16 / copies.count)
        let run = {
            var outputs: [MLXArray] = []
            for _ in 0 ..< rounds {
                for (w, scales, biases) in copies {
                    outputs.append(SmallMQuantizedMatmul.matmul(x, weight: w, scales: scales, biases: biases!, groupSize: Bench.groupSize))
                }
            }
            eval(outputs)
        }
        run()
        return Bench.microseconds(run) / Double(rounds * copies.count)
    }

    static func agreement(_ copies: Copies, x: MLXArray) -> (rowsMatchOneToken: Bool, errorVsStock: Float) {
        let (w, scales, biases) = copies[0]
        let stock = { quantizedMM($0, w, scales: scales, biases: biases, groupSize: Bench.groupSize, bits: 4) }
        let kernel = SmallMQuantizedMatmul.matmul(x, weight: w, scales: scales, biases: biases!, groupSize: Bench.groupSize)
        let oneToken = concatenated((0 ..< x.dim(1)).map { stock(x[0..., $0 ..< $0 + 1]) }, axis: 1)
        let window = stock(x).asType(.float32)
        let error = abs(kernel.asType(.float32) - window).max() / abs(window).max()
        return ((oneToken .== kernel).all().item(Bool.self), error.item(Float.self))
    }

    @Test("µs per call, ratio to stock M = 1, row identity with the one-token path and difference from stock at M = 1...8, 4-bit affine g64, bf16",
          .enabled(if: Bench.enabled, "set BASALT_BENCH_SMALLM=1 and run a release build"))
    func kernelAgainstStock() throws {
        guard case .taken = try ModelRunLog.claim(ModelRunLog.gpuLock, pid: getpid()) else {
            Issue.record("\(ModelRunLog.gpuLock) is held by another run; relaunch when it is free")
            return
        }
        var lock = GPULockGuard(path: ModelRunLog.gpuLock, pid: getpid())
        defer { _ = lock.release() }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        try FileManager.default.createDirectory(at: root.appendingPathComponent("logs"), withIntermediateDirectories: true)
        let url = root.appendingPathComponent("logs/smallm-kernel-\(Date().ISO8601Format(.iso8601.year().month().day())).log")
        Bench.log("RUN \(Date().ISO8601Format()) commit \(ModelRunLog.output("/usr/bin/git", ["-C", root.path, "rev-parse", "HEAD"])) machine \(SystemProfiler.detectChipModel()) load \(ModelRunLog.load())", to: url)
        Bench.log("shape | K x N | M | stock us | kernel us | stock / stock M=1 | kernel / stock M=1 | kernel / stock | rows = stock M=1 bit for bit | rel max diff vs stock window", to: url)
        for shape in Bench.shapes where shape.name != "lm_head" {
            let (copies, _) = Bench.quantizedCopies(shape, bits: 4)
            var single = Double.nan
            for m in 1 ... 8 {
                let x = MLXRandom.normal([1, m, shape.inputs], dtype: .bfloat16)
                let stock = Bench.callMicros(copies, x: x, bits: 4)
                let kernel = Self.kernelMicros(copies, x: x)
                if m == 1 { single = stock }
                let (same, error) = Self.agreement(copies, x: x)
                Bench.log("\(shape.name) | \(shape.inputs) x \(shape.outputs) | \(m) | "
                    + String(format: "%.1f | %.1f | %.2fx | %.2fx | %.2fx | ", stock, kernel, stock / single, kernel / single, kernel / stock)
                    + "\(same) | " + String(format: "%.1e", error), to: url)
                #expect(same)
                #expect(error < 1e-2)
            }
        }
        Bench.log("END \(Date().ISO8601Format()) log \(url.path)", to: url)
    }
}
