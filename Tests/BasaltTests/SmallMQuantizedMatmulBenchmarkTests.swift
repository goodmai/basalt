import Foundation
import Darwin
import Testing
import MLX
import BasaltCore

@Suite("Stock quantizedMatmul at verify-window row counts against the bandwidth bound (#208)", .serialized)
struct SmallMQuantizedMatmulBenchmarkTests {
    struct Shape: Sendable {
        let name: String
        let inputs: Int
        let outputs: Int
    }

    static let enabled = ProcessInfo.processInfo.environment["BASALT_BENCH_SMALLM"] == "1"
    static let groupSize = 64
    static let rows = [1, 2, 3, 4, 5, 6, 8, 16]
    static let cacheDefeatBytes = 256 << 20
    static let samples = 7
    static let shapes = [
        Shape(name: "lm_head", inputs: 5120, outputs: 248_320),
        Shape(name: "mlp.gate/up", inputs: 5120, outputs: 17408),
        Shape(name: "mlp.down", inputs: 17408, outputs: 5120),
        Shape(name: "linear_attn.qkv", inputs: 5120, outputs: 10240),
        Shape(name: "attn.q+gate", inputs: 5120, outputs: 12288),
        Shape(name: "attn.o/out_proj", inputs: 6144, outputs: 5120),
    ]

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

    static func bandwidth() -> Double {
        let probe = MLX.ones([1 << 28], dtype: .float32)
        eval(probe)
        eval(probe.sum())
        let micros = microseconds { eval(probe.sum()) }
        return Double(probe.nbytes) / micros / 1e3
    }

    static func quantizedCopies(_ shape: Shape, bits: Int) -> (copies: [(MLXArray, MLXArray, MLXArray?)], bytes: Int) {
        let dense = MLXRandom.normal([shape.outputs, shape.inputs], dtype: .bfloat16, scale: 0.02)
        let first = quantized(dense, groupSize: groupSize, bits: bits)
        eval([first.wq, first.scales] + (first.biases.map { [$0] } ?? []))
        let bytes = first.wq.nbytes + first.scales.nbytes + (first.biases?.nbytes ?? 0)
        var copies = [(first.wq, first.scales, first.biases)]
        for _ in 1 ..< max(1, cacheDefeatBytes / bytes) {
            let copy = (first.wq + 0, first.scales + 0, first.biases.map { $0 + 0 })
            eval([copy.0, copy.1] + (copy.2.map { [$0] } ?? []))
            copies.append(copy)
        }
        return (copies, bytes)
    }

    static func callMicros(_ copies: [(MLXArray, MLXArray, MLXArray?)], x: MLXArray, bits: Int) -> Double {
        let rounds = max(1, 16 / copies.count)
        let run = {
            var outputs: [MLXArray] = []
            for _ in 0 ..< rounds {
                for (w, scales, biases) in copies {
                    outputs.append(quantizedMM(x, w, scales: scales, biases: biases, groupSize: groupSize, bits: bits))
                }
            }
            eval(outputs)
        }
        run()
        return microseconds(run) / Double(rounds * copies.count)
    }

    static func log(_ line: String, to url: URL) {
        print(line)
        let descriptor = open(url.path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        guard descriptor >= 0 else { return }
        _ = (line + "\n").withCString { write(descriptor, $0, strlen($0)) }
        close(descriptor)
    }

    @Test("µs per call at M = 1 to 6, 8 and 16 on the Qwen3.8-27B projection shapes, 4- and 8-bit, against weight bytes / measured GB/s",
          .enabled(if: enabled, "set BASALT_BENCH_SMALLM=1 and run a release build"))
    func stockAgainstBound() throws {
        guard case .taken = try ModelRunLog.claim(ModelRunLog.gpuLock, pid: getpid()) else {
            Issue.record("\(ModelRunLog.gpuLock) is held by another run; relaunch when it is free")
            return
        }
        var lock = GPULockGuard(path: ModelRunLog.gpuLock, pid: getpid())
        defer { _ = lock.release() }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        try FileManager.default.createDirectory(at: root.appendingPathComponent("logs"), withIntermediateDirectories: true)
        let url = root.appendingPathComponent("logs/smallm-matmul-\(Date().ISO8601Format(.iso8601.year().month().day())).log")
        let gbps = Self.bandwidth()
        Self.log("RUN \(Date().ISO8601Format()) commit \(ModelRunLog.output("/usr/bin/git", ["-C", root.path, "rev-parse", "HEAD"])) machine \(SystemProfiler.detectChipModel()) load \(ModelRunLog.load()) bandwidth \(String(format: "%.1f", gbps)) GB/s", to: url)
        Self.log("bits | shape | K x N | M | us/call | bound us | % of bound | us vs M=1", to: url)
        for bits in [4, 8] {
            for shape in Self.shapes {
                let (copies, bytes) = Self.quantizedCopies(shape, bits: bits)
                let bound = Double(bytes) / gbps / 1e3
                var single = Double.nan
                for m in Self.rows {
                    let x = MLXRandom.normal([m, shape.inputs], dtype: .bfloat16)
                    let micros = Self.callMicros(copies, x: x, bits: bits)
                    if m == 1 { single = micros }
                    let line = "\(bits) | \(shape.name) | \(shape.inputs) x \(shape.outputs) | \(m) | "
                        + String(format: "%.1f | %.1f | %.0f%% | %.2fx", micros, bound, 100 * bound / micros, micros / single)
                    Self.log(line, to: url)
                    #expect(micros.isFinite && micros > 0)
                }
            }
        }
        Self.log("END \(Date().ISO8601Format()) log \(url.path)", to: url)
    }
}
