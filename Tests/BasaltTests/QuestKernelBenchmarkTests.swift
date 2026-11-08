import Foundation
import Darwin
import Testing
import MLX
import BasaltCore
@testable import BasaltInference

@Suite("Quest decode attention: dense SDPA, the gathered MLX path and the fused kernels on synthetic K/V (#280)", .serialized)
struct QuestKernelBenchmarkTests {
    struct Shape: Sendable {
        let name: String
        let queryHeads: Int
        let kvHeads: Int
        let dim: Int
    }

    static let enabled = ProcessInfo.processInfo.environment["BASALT_BENCH_QUEST"] == "1"
    static let copies = 8
    static let samples = 7
    static let contexts = [16384, 32768, 65536]
    static let topKs = [16, 32]
    static let shapes = [
        Shape(name: "Qwen3-4B", queryHeads: 32, kvHeads: 8, dim: 128),
        Shape(name: "Ornith-1.5-9B", queryHeads: 16, kvHeads: 4, dim: 256),
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

    static func caches(_ shape: Shape, context: Int, topK: Int) -> [QuestKVCache] {
        let keys = MLXRandom.normal([1, shape.kvHeads, context, shape.dim], dtype: .bfloat16)
        let values = MLXRandom.normal([1, shape.kvHeads, context, shape.dim], dtype: .bfloat16)
        return (0 ..< copies).map { copy in
            let cache = QuestKVCache(topK: topK)
            _ = cache.update(keys: keys[.ellipsis, ..<(context - 1), 0...], values: values[.ellipsis, ..<(context - 1), 0...])
            let query = MLXRandom.normal([1, shape.queryHeads, 1, shape.dim], dtype: .bfloat16, key: MLXRandom.key(UInt64(copy)))
            eval(cache.updateAndAttend(
                queries: query, keys: keys[.ellipsis, (context - 1)..., 0...], values: values[.ellipsis, (context - 1)..., 0...],
                scale: 1, mask: .none))
            eval(cache.innerState())
            return cache
        }
    }

    static func perCall(_ caches: [QuestKVCache], query: MLXArray, _ attend: (QuestKVCache, MLXArray) -> MLXArray) -> Double {
        let rounds = max(1, 32 / caches.count)
        func run() {
            var q = query
            for _ in 0 ..< rounds {
                for cache in caches {
                    q = attend(cache, q)
                }
            }
            eval(q)
        }
        run()
        return microseconds(run) / Double(rounds * caches.count)
    }

    static func dense(_ cache: QuestKVCache, _ query: MLXArray, scale: Float) -> MLXArray {
        let buffers = cache.innerState()
        return MLXFast.scaledDotProductAttention(
            queries: query, keys: buffers[0][.ellipsis, ..<cache.offset, 0...], values: buffers[1][.ellipsis, ..<cache.offset, 0...],
            scale: scale, mask: .none)
    }

    static func log(_ line: String, to url: URL) {
        print(line)
        let descriptor = open(url.path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        guard descriptor >= 0 else { return }
        _ = (line + "\n").withCString { write(descriptor, $0, strlen($0)) }
        close(descriptor)
    }

    static func row(_ shape: Shape, context: Int, topK: Int) -> String {
        let caches = Self.caches(shape, context: context, topK: topK)
        let scale = Float(1) / Float(shape.dim).squareRoot()
        let query = MLXRandom.normal([1, shape.queryHeads, 1, shape.dim], dtype: .bfloat16, key: MLXRandom.key(99))
        let gap = abs(caches[0].kernelAttention(queries: query, scale: scale) - caches[0].gatheredAttention(queries: query, scale: scale))
            .max().item(Float.self)
        let dense = perCall(caches, query: query) { Self.dense($0, $1, scale: scale) }
        let gathered = perCall(caches, query: query) { $0.gatheredAttention(queries: $1, scale: scale) }
        let fused = perCall(caches, query: query) { $0.kernelAttention(queries: $1, scale: scale) }
        #expect(fused.isFinite && fused > 0)
        return "\(shape.name) | \(context) | \(topK) | "
            + String(format: "%.1f | %.1f | %.1f | %.2fx | %.2fx | %.1e", dense, gathered, fused, dense / gathered, dense / fused, gap)
    }

    @Test("µs per decode-step attention call at 16k/32k/64k, top-k 16 and 32, on the Qwen3-4B and Ornith-1.5-9B attention shapes",
          .enabled(if: enabled, "set BASALT_BENCH_QUEST=1 and run a release build"))
    func denseGatheredFused() throws {
        guard case .taken = try ModelRunLog.claim(ModelRunLog.gpuLock, pid: getpid()) else {
            Issue.record("\(ModelRunLog.gpuLock) is held by another run; relaunch when it is free")
            return
        }
        var lock = GPULockGuard(path: ModelRunLog.gpuLock, pid: getpid())
        defer { _ = lock.release() }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        try FileManager.default.createDirectory(at: root.appendingPathComponent("logs"), withIntermediateDirectories: true)
        let url = root.appendingPathComponent("logs/quest-kernel-\(Date().ISO8601Format(.iso8601.year().month().day())).log")
        Self.log("RUN \(Date().ISO8601Format()) commit \(ModelRunLog.output("/usr/bin/git", ["-C", root.path, "rev-parse", "HEAD"])) machine \(SystemProfiler.detectChipModel()) load \(ModelRunLog.load())", to: url)
        Self.log("shape | context | top-k | dense us | gathered us | fused us | dense/gathered | dense/fused | max |fused-gathered|", to: url)
        for shape in Self.shapes {
            for context in Self.contexts {
                for topK in Self.topKs {
                    Self.log(Self.row(shape, context: context, topK: topK), to: url)
                }
            }
        }
        Self.log("END \(Date().ISO8601Format()) log \(url.path)", to: url)
    }
}
