import Foundation
import Darwin
import Testing
import MLX
import MLXNN
import BasaltCore
@testable import BasaltInference

@Suite("Prefill FFN split spike: the GPU side of one dense FFN block, 4-bit g64 on MLX, at prefill shapes (#274)", .serialized)
struct FFNSplitBenchmarkTests {
    struct Block {
        let gate: (MLXArray, MLXArray, MLXArray)
        let up: (MLXArray, MLXArray, MLXArray)
        let down: (MLXArray, MLXArray, MLXArray)

        init(hidden: Int, columns: Int) {
            func packed(_ rows: Int, _ inputs: Int) -> (MLXArray, MLXArray, MLXArray) {
                let (weight, scales, biases) = MLX.quantized(MLXRandom.normal([rows, inputs], dtype: .bfloat16) * 0.02, groupSize: 64, bits: 4)
                return (weight, scales, biases ?? MLXArray.zeros(scales.shape, dtype: scales.dtype))
            }
            (gate, up, down) = (packed(columns, hidden), packed(columns, hidden), packed(hidden, columns))
            eval(gate.0, gate.1, gate.2, up.0, up.1, up.2, down.0, down.1, down.2)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            func project(_ input: MLXArray, _ w: (MLXArray, MLXArray, MLXArray)) -> MLXArray {
                quantizedMatmul(input, w.0, scales: w.1, biases: w.2, transpose: true, groupSize: 64, bits: 4)
            }
            return project(silu(project(x, gate)) * project(x, up), down)
        }
    }

    static let enabled = ProcessInfo.processInfo.environment["BASALT_BENCH_FFN"] == "1"
    static let (hidden, intermediate) = (2560, 9728)
    static let lengths = [512, 2048]
    static let shares = [1.0, 0.7, 0.5, 0.3]
    static let samples = 9

    static func microseconds(_ body: () -> Void) -> Double {
        let clock = ContinuousClock()
        let times = (0 ..< samples).map { _ in
            let start = clock.now
            body()
            let elapsed = clock.now - start
            return Double(elapsed.components.seconds) * 1e6 + Double(elapsed.components.attoseconds) / 1e12
        }
        return times.sorted()[samples / 2]
    }

    @Test("µs per FFN block on the GPU for Qwen3-4B's FFN (2560 → 9728) at 512 and 2048 tokens, for the share of columns the GPU would keep",
          .enabled(if: enabled, "set BASALT_BENCH_FFN=1 and run a release build"))
    func gpuSide() throws {
        guard case .taken = try ModelRunLog.claim(ModelRunLog.gpuLock, pid: getpid()) else {
            Issue.record("\(ModelRunLog.gpuLock) is held by another run; relaunch when it is free")
            return
        }
        var lock = GPULockGuard(path: ModelRunLog.gpuLock, pid: getpid())
        defer { _ = lock.release() }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        try FileManager.default.createDirectory(at: root.appendingPathComponent("logs"), withIntermediateDirectories: true)
        let url = root.appendingPathComponent("logs/ffn-split-\(Date().ISO8601Format(.iso8601.year().month().day())).log")
        QuestKernelBenchmarkTests.log("RUN \(Date().ISO8601Format()) commit \(ModelRunLog.output("/usr/bin/git", ["-C", root.path, "rev-parse", "HEAD"])) machine \(SystemProfiler.detectChipModel()) load \(ModelRunLog.load())", to: url)
        QuestKernelBenchmarkTests.log("tokens | GPU share | columns | GPU us | TFLOPS", to: url)
        for share in Self.shares {
            let columns = max(64, Int((Double(Self.intermediate) * share / 64).rounded()) * 64)
            let block = Block(hidden: Self.hidden, columns: columns)
            for length in Self.lengths {
                let x = MLXRandom.normal([1, length, Self.hidden], dtype: .bfloat16)
                eval(x)
                eval(block(x))
                let us = Self.microseconds { eval(block(x)) }
                let flops = 2.0 * 3 * Double(Self.hidden * columns * length)
                QuestKernelBenchmarkTests.log(String(format: "%d | %.1f | %d | %.1f | %.2f", length, share, columns, us, flops / us / 1e6), to: url)
            }
        }
        QuestKernelBenchmarkTests.log("END \(Date().ISO8601Format()) log \(url.path)", to: url)
    }
}
