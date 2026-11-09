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

    static func denseRun(_ dense: MLXArray, x: MLXArray, mode: QuantizationMode, groupSize: Int) -> () -> Double {
        let first = quantized(dense, groupSize: groupSize, bits: 4, mode: mode)
        let bytes = first.wq.nbytes + first.scales.nbytes + (first.biases?.nbytes ?? 0)
        let copies = [(first.wq, first.scales, first.biases)] + (1 ..< max(1, Bench.cacheDefeatBytes / bytes)).map { _ in
            (first.wq + 0, first.scales + 0, first.biases.map { $0 + 0 })
        }
        eval(copies.flatMap { [$0.0, $0.1] + ($0.2.map { [$0] } ?? []) })
        let rounds = max(1, 16 / copies.count)
        let run = {
            eval((0 ..< rounds).flatMap { _ in
                copies.map { quantizedMM(x, $0.0, scales: $0.1, biases: $0.2, groupSize: groupSize, bits: 4, mode: mode) }
            })
        }
        run()
        return { Bench.microseconds(run) / Double(rounds * copies.count) }
    }

    struct Experts: Sendable {
        let name: String
        let experts: Int
        let topK: Int
        let inputs: Int
        let outputs: Int
    }

    static let experts = [
        Experts(name: "gemma4-26b-a4b gate/up", experts: 128, topK: 8, inputs: 2816, outputs: 704),
        Experts(name: "gemma4-26b-a4b down", experts: 128, topK: 8, inputs: 704, outputs: 2816),
        Experts(name: "lfm2.5-24b-a2b gate/up", experts: 64, topK: 4, inputs: 2048, outputs: 1536),
        Experts(name: "lfm2.5-24b-a2b down", experts: 64, topK: 4, inputs: 1536, outputs: 2048),
    ]

    static func gatherRun(_ shape: Experts, dense: MLXArray, mode: QuantizationMode, groupSize: Int) -> () -> Double {
        let (wq, scales, biases) = quantized(dense, groupSize: groupSize, bits: 4, mode: mode)
        eval([wq, scales] + (biases.map { [$0] } ?? []))
        let x = MLXRandom.normal([shape.topK, 1, shape.inputs], dtype: .bfloat16)
        let picks = (0 ..< 32).map { call in MLXArray((0 ..< shape.topK).map { UInt32(($0 * 37 + call * 11) % shape.experts) }) }
        eval([x] + picks)
        let run = {
            eval(picks.map {
                gatherQuantizedMM(x, wq, scales: scales, biases: biases, rhsIndices: $0, groupSize: groupSize, bits: 4, mode: mode)
            })
        }
        run()
        return { Bench.microseconds(run) / Double(picks.count) }
    }

    static func interleavedMedians(_ runs: [() -> Double], rounds: Int = 9) -> [Double] {
        var samples = Array(repeating: [Double](), count: runs.count)
        for _ in 0 ..< rounds {
            for (index, run) in runs.enumerated() { samples[index].append(run()) }
        }
        return samples.map { $0.sorted()[$0.count / 2] }
    }

    static func kernelError(x: MLXArray, codes: MLXArray, scales: MLXArray) -> Float {
        let reference = matmul(x.asType(.float32), dequantized(codes, scales: scales, biases: nil, groupSize: 32, bits: 4, mode: .nf4).asType(.float32).T)
        let kernel = quantizedMM(x, codes, scales: scales, biases: nil, groupSize: 32, bits: 4, mode: .nf4).asType(.float32)
        return (abs(kernel - reference).max() / abs(reference).max()).item(Float.self)
    }

    @Test("µs per call and relative max error at M = 1, 2, 4 and 8 on the Qwen3.8-27B projection shapes, 256 MB of weight copies per mode against the system cache, the modes interleaved",
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
                let micros = Self.interleavedMedians([
                    Self.denseRun(dense, x: x, mode: .affine, groupSize: 64),
                    Self.denseRun(dense, x: x, mode: .nf4, groupSize: 32),
                ], rounds: 3)
                let (affine, nf4) = (micros[0], micros[1])
                Memory.clearCache()
                let error = Self.kernelError(x: x, codes: codes, scales: scales)
                Bench.log("\(shape.name) | \(shape.inputs) x \(shape.outputs) | \(m) | " + String(format: "%.1f | %.1f | %.2fx | %.2e", affine, nf4, nf4 / affine, error), to: url)
                #expect(error < 1e-2)
            }
        }
        Bench.log("END \(Date().ISO8601Format()) log \(url.path)", to: url)
    }

    @Test("µs per decode-step gatherQuantizedMM (one token, top-k experts) on the Gemma 4 26B-A4B and LFM2.5-24B-A2B expert shapes, the two modes interleaved, median of 9 rounds",
          .enabled(if: Bench.enabled, "set BASALT_BENCH_SMALLM=1 and run a release build"))
    func nf4GatherAgainstAffine() throws {
        guard case .taken = try ModelRunLog.claim(ModelRunLog.gpuLock, pid: getpid()) else {
            Issue.record("\(ModelRunLog.gpuLock) is held by another run; relaunch when it is free")
            return
        }
        var lock = GPULockGuard(path: ModelRunLog.gpuLock, pid: getpid())
        defer { _ = lock.release() }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let url = root.appendingPathComponent("logs/nf4-kernel-\(Date().ISO8601Format(.iso8601.year().month().day())).log")
        Bench.log("RUN gather \(Date().ISO8601Format()) commit \(ModelRunLog.output("/usr/bin/git", ["-C", root.path, "rev-parse", "HEAD"])) machine \(SystemProfiler.detectChipModel()) load \(ModelRunLog.load())", to: url)
        Bench.log("experts | E top-k K x N | affine g64 us | nf4 g32 us | nf4 / affine", to: url)
        for shape in Self.experts {
            let dense = MLXRandom.normal([shape.experts, shape.outputs, shape.inputs], dtype: .bfloat16, scale: 0.02)
            let micros = Self.interleavedMedians([
                Self.gatherRun(shape, dense: dense, mode: .affine, groupSize: 64),
                Self.gatherRun(shape, dense: dense, mode: .nf4, groupSize: 32),
            ])
            let (affine, nf4) = (micros[0], micros[1])
            Bench.log("\(shape.name) | \(shape.experts) \(shape.topK) \(shape.inputs) x \(shape.outputs) | " + String(format: "%.1f | %.1f | %.2fx", affine, nf4, nf4 / affine), to: url)
        }
        Bench.log("END gather \(Date().ISO8601Format()) log \(url.path)", to: url)
    }
}
