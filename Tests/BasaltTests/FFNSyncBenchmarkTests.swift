import Foundation
import Darwin
import Testing
import CoreAI
import MLX
import BasaltCore
@testable import BasaltInference

@Suite("Prefill FFN split spike step 2: the MLX ↔ Core AI hand-off per layer against the GPU-only block (#274)", .serialized)
struct FFNSyncBenchmarkTests {
    static let modelPath = ProcessInfo.processInfo.environment["BASALT_BENCH_FFN_SYNC"]
    static let (hidden, intermediate, layers, length) = (2560, 9728, 36, 2048)

    static func planar(_ x: MLXArray) -> MLXArray {
        contiguous(x.reshaped([length, hidden]).transposed().asType(.float16))
    }

    static func handIn(_ x: MLXArray) -> NDArray {
        let planar = planar(x)
        eval(planar)
        let bytes = planar.asData(access: .noCopyIfContiguous).data
        var array = NDArray(shape: [1, hidden, 1, length], scalarType: .float16)
        let view = array.mutableView(as: Float16.self)
        view.withUnsafeMutablePointer { pointer, _, _ in
            bytes.withUnsafeBytes { source in _ = memcpy(pointer, source.baseAddress!, source.count) }
        }
        return array
    }

    static func handOut(_ y: NDArray) -> MLXArray {
        let planar = y.view(as: Float16.self).withUnsafePointer { pointer, _, _ in
            MLXArray(UnsafeBufferPointer(start: pointer, count: hidden * length), [hidden, length])
        }
        return planar.transposed().asType(.bfloat16).reshaped([1, length, hidden])
    }

    static func run(_ function: InferenceFunction, _ input: NDArray) async throws -> NDArray {
        var outputs = try await function.run(inputs: ["x": input])
        guard let y = outputs.remove("y")?.ndArray else { throw BasaltError.inferenceError("the FFN function returned no y") }
        return y
    }

    static func milliseconds(_ body: () async throws -> Void) async rethrows -> Double {
        let clock = ContinuousClock()
        var times: [Double] = []
        for _ in 0 ..< FFNSplitBenchmarkTests.samples {
            let start = clock.now
            try await body()
            let elapsed = clock.now - start
            times.append(Double(elapsed.components.seconds) * 1e3 + Double(elapsed.components.attoseconds) / 1e15)
        }
        return times.sorted()[times.count / 2]
    }

    @Test("Hand-in, ANE run, hand-out and one split layer at 2048 tokens, against the GPU-only 4-bit block, with the step 2 gate",
          .enabled(if: modelPath != nil, "set BASALT_BENCH_FFN_SYNC to an ffn-<share>.aimodel from #274 step 1 and run a release build"))
    func handOff() async throws {
        let path = try #require(Self.modelPath)
        guard case .taken = try ModelRunLog.claim(ModelRunLog.gpuLock, pid: getpid()) else {
            Issue.record("\(ModelRunLog.gpuLock) is held by another run; relaunch when it is free")
            return
        }
        var lock = GPULockGuard(path: ModelRunLog.gpuLock, pid: getpid())
        defer { _ = lock.release() }
        let share = Double(URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent.split(separator: "-").last ?? "") ?? 0.5
        let model = try await AIModel.specialize(contentsOf: URL(fileURLWithPath: path), options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
        let function = try #require(try model.loadFunction(named: "len\(Self.length)"))
        let columns = { (s: Double) in max(64, Int((Double(Self.intermediate) * s / 64).rounded()) * 64) }
        let (full, rest) = (FFNSplitBenchmarkTests.Block(hidden: Self.hidden, columns: Self.intermediate), FFNSplitBenchmarkTests.Block(hidden: Self.hidden, columns: columns(1 - share)))
        let x = MLXRandom.normal([1, Self.length, Self.hidden], dtype: .bfloat16)
        eval(x, full(x), rest(x))
        let input = Self.handIn(x)
        let output = try await Self.run(function, input)
        let gpu = await Self.milliseconds { eval(full(x)) }
        let inMs = await Self.milliseconds { _ = Self.handIn(x) }
        let planarMs = await Self.milliseconds { eval(Self.planar(x)) }
        let aneMs = try await Self.milliseconds { _ = try await Self.run(function, input) }
        let outMs = await Self.milliseconds { eval(Self.handOut(output)) }
        let split = try await Self.milliseconds {
            let input = Self.handIn(x)
            let partial = rest(x)
            asyncEval(partial)
            let y = try await Self.run(function, input)
            eval(partial + Self.handOut(y))
        }
        let layer = gpu / 0.6, splitLayer = layer - gpu + split
        let handOffShare = (inMs + outMs) / splitLayer
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let url = root.appendingPathComponent("logs/ffn-split-\(Date().ISO8601Format(.iso8601.year().month().day())).log")
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        QuestKernelBenchmarkTests.log("SYNC \(Date().ISO8601Format()) commit \(ModelRunLog.output("/usr/bin/git", ["-C", root.path, "rev-parse", "HEAD"])) machine \(SystemProfiler.detectChipModel()) load \(ModelRunLog.load()) model \(path)", to: url)
        QuestKernelBenchmarkTests.log(String(
            format: "tokens %d | ANE share %.2f | GPU full %.2f ms | hand-in %.2f ms (GPU transpose %.2f) | ANE run %.2f ms | hand-out %.2f ms | split layer FFN %.2f ms (%.2fx) | hand-off %.1f%% of a split layer, %.0f ms per chunk over %d layers | gate < 10%%: %@",
            Self.length, share, gpu, inMs, planarMs, aneMs, outMs, split, gpu / split, handOffShare * 100, (inMs + outMs) * Double(Self.layers), Self.layers, handOffShare < 0.1 ? "pass" : "fail"), to: url)
    }
}
