import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("kv-evict merge spike (#352): the merge math on synthetic keys, and the batch replay against ZonedKVCache")
struct KVMergeSpikeTests {
    static func random(_ rows: Int, _ dim: Int, seed: UInt64) -> [[Float]] {
        MLXRandom.normal([rows, dim], key: MLXRandom.key(seed)).asArray(Float.self).chunked(dim)
    }

    @Test("Identical rows merged into one with bias log n attend exactly like the unmerged rows")
    func identicalRows() {
        let (a, b) = (Self.random(1, 16, seed: 1)[0], Self.random(1, 16, seed: 2)[0])
        let (va, vb) = (Self.random(1, 16, seed: 3)[0], Self.random(1, 16, seed: 4)[0])
        let query = Self.random(1, 16, seed: 5)[0]
        let unmerged = KVMergeSpike.attention(query: query, keys: [a, a, a, b], values: [va, va, va, vb], bias: [0, 0, 0, 0])
        let merged = KVMergeSpike.attention(query: query, keys: [a, b], values: [va, vb], bias: [log(3), 0])
        #expect(KVMergeSpike.relativeError(merged, unmerged) < 1e-6)
    }

    @Test("A merged group's attention mass stays within exp(|q| radius / sqrt(d)) of the members' mass, over random draws")
    func massBound() {
        for seed in 0 ..< 32 {
            let center = Self.random(1, 32, seed: UInt64(100 + seed))[0]
            let members = Self.random(4, 32, seed: UInt64(200 + seed)).map { row in zip(center, row).map { $0 + 0.05 * $1 } }
            let mean = KVMergeSpike.mean(members)
            let query = Self.random(1, 32, seed: UInt64(300 + seed))[0]
            let scale = 1 / Float(32).squareRoot()
            let scores = members.map { zip(query, $0).reduce(0) { $0 + $1.0 * $1.1 } * scale }
            let merged = Float(members.count) * exp(zip(query, mean).reduce(0) { $0 + $1.0 * $1.1 } * scale)
            let ratio = scores.map { exp($0) }.reduce(0, +) / merged
            let radius = members.map { KVMergeSpike.distance($0, mean) }.max()!
            #expect(ratio >= 1 - 1e-5 && ratio <= exp(KVMergeSpike.norm(query) * radius * scale) + 1e-5, "seed \(seed): ratio \(ratio)")
        }
    }

    @Test("Radius 0 merges nothing: the merge arm keeps exactly the rows the drop arm keeps")
    func zeroRadius() {
        let (keys, values) = (Self.random(64, 16, seed: 7), Self.random(64, 16, seed: 8))
        let drop = KVMergeSpike.compress(keys: keys, values: values, keep: 40, radius: nil)
        let merge = KVMergeSpike.compress(keys: keys, values: values, keep: 40, radius: 0)
        #expect(drop.keys == merge.keys && drop.values == merge.values && merge.merged == 0 && merge.counts.allSatisfy { $0 == 1 })
    }

    @Test("The batch replay moves the rows ZonedKVCache moves, and keeps and evicts as many at F = 0.5", arguments: [[512, 512, 512, 512] + Array(repeating: 1, count: 16), [300, 700, 1, 900, 1000, 1, 1]])
    func replayMatchesCache(appends: [Int]) {
        let zones = KVZones.standard.evicting(0.5)
        let cache = ZonedKVCache(zones: zones, precision: .k8v4)
        for (index, count) in appends.enumerated() {
            let rows = MLXRandom.normal([1, 2, count, 64], key: MLXRandom.key(UInt64(index)))
            _ = cache.update(keys: rows, values: rows)
        }
        let batches = KVMergeSpike.batches(appends: appends, zones: zones)
        let keys = MLXRandom.normal([1, 2, appends.reduce(0, +), 64], key: MLXRandom.key(9))
        let heads = KVMergeSpike.heads(keys)
        let middle = KVMergeSpike.middle(keys: heads, values: heads, batches: batches, zones: zones, radius: nil)
        #expect(batches.reduce(0) { $0 + $1.count } == cache.quantizedCount + cache.evictedCount)
        #expect(middle.keys[0].count == cache.quantizedCount && middle.droppedRows == 2 * cache.evictedCount)
    }
}

extension Array where Element == Float {
    func chunked(_ size: Int) -> [[Float]] { stride(from: 0, to: count, by: size).map { Array(self[$0 ..< Swift.min($0 + size, count)]) } }
}

@Suite("kv-evict merge spike on real models (#352 step 2): attention-output error of drop, merge with log n bias and merge without it, at equal kept rows", .serialized)
struct KVMergeRealModelTests {
    static let env = ProcessInfo.processInfo.environment
    static let models = env["BASALT_E2E"] == "1" ? (env["BASALT_E2E_KV_MERGE_MODELS"] ?? "").split(separator: ",").map(String.init) : []

    static var corpus: String {
        if let path = env["BASALT_ZONED_CORPUS"], let text = try? String(contentsOfFile: path, encoding: .utf8) { return text }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let docs = ((try? FileManager.default.contentsOfDirectory(atPath: root.appending(path: "docs").path)) ?? []).filter { $0.hasSuffix(".md") }.sorted()
        return (["README.md", "CONTRIBUTING.md"] + docs.map { "docs/" + $0 }).compactMap { try? String(contentsOf: root.appending(path: $0), encoding: .utf8) }.joined(separator: "\n\n")
    }

    static func capture(_ context: ModelContext, tokens count: Int, corpus: String) throws -> [KVRecordingCache] {
        do {
            let caches = KVRecordingCache.replacingFullAttention(try context.model.newCache(parameters: nil))
            let tokens = Array(context.tokenizer.encode(text: corpus).prefix(count)).map(Int32.init)
            var logits = MLXArray(0)
            for start in stride(from: 0, to: tokens.count, by: 512) {
                logits = context.model(MLXArray(Array(tokens[start ..< min(start + 512, tokens.count)]))[.newAxis], cache: caches)
                eval(caches)
            }
            for _ in 0 ..< 16 {
                let next = argMax(logits[0, -1], axis: -1).item(Int32.self)
                logits = context.model(MLXArray([next])[.newAxis], cache: caches)
                eval(caches)
            }
            return caches.compactMap { $0 as? KVRecordingCache }
        }
    }

    static func report(_ recorders: [KVRecordingCache], length: Int, log: ModelRunLog) {
        for fraction in [0.25, 0.5] {
            for radius: Float in [0.05, 0.1, 0.2] {
                var (perLayer, merged, dropped) = ([KVMergeSpike.Arm: [Float]](), 0, 0)
                var worst: Float = 0
                for (layer, recorder) in recorders.enumerated() {
                    let (keys, values) = recorder.rows
                    let outcome = KVMergeSpike.evaluate(
                        keys: keys, values: values, appends: recorder.appends, queries: recorder.queries, zones: KVZones.standard.evicting(fraction), radius: radius)
                    let medians = Dictionary(uniqueKeysWithValues: KVMergeSpike.Arm.allCases.map { ($0, median(outcome.errors[$0] ?? [])) })
                    for (arm, value) in medians { perLayer[arm, default: []].append(value) }
                    (merged, dropped) = (merged + outcome.merged, dropped + outcome.dropped)
                    worst = max(worst, (medians[.merge] ?? 0) / max(medians[.drop] ?? 1, 1e-9))
                    log.record("layer", String(format: "{\"tokens\":%d,\"F\":%.2f,\"r\":%.2f,\"layer\":%d,\"drop\":%.5f,\"merge\":%.5f,\"merge_no_bias\":%.5f,\"merged\":%d,\"dropped\":%d}",
                        length, fraction, radius, layer, medians[.drop] ?? 0, medians[.merge] ?? 0, medians[.mergeNoBias] ?? 0, outcome.merged, outcome.dropped))
                }
                let (a, b, c) = (median(perLayer[.drop] ?? []), median(perLayer[.merge] ?? []), median(perLayer[.mergeNoBias] ?? []))
                log.record("RESULT", String(format: "%d tokens F %.2f r %.2f: median output error drop %.4f, merge %.4f (%+.1f%%), merge without bias %.4f; merged %d of %d evicted rows (%.0f%%); worst layer merge/drop %.2f",
                    length, fraction, radius, a, b, (b / a - 1) * 100, c, merged, merged + dropped, 100 * Double(merged) / Double(max(merged + dropped, 1)), worst))
            }
        }
    }

    static func median(_ values: [Float]) -> Float { values.isEmpty ? .nan : values.sorted()[values.count / 2] }

    @Test("At F = 0.25 and 0.5 and merge radius 0.05, 0.1 and 0.2, per full-attention layer, at 8k and 16k tokens; the go gate is merge 25% under drop at the median and no layer worse",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_KV_MERGE_MODELS (Qwen3.8-27B-3bit, LFM2.5-24B-A2B-4bit); run a release build"), arguments: models)
    func spike(model: String) async throws {
        let path = NF4Run.path(model)
        let log = try await ModelRunLog("kv-merge", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        try await log.step("load") { try await engine.load(modelPath: path) }
        let container = try #require(await engine.container)
        let corpus = Self.corpus
        try await container.performSending(nonSendable: corpus) { context, corpus in
            for length in [8192, 16384] {
                let recorders = try Self.capture(context, tokens: length, corpus: corpus)
                log.record("capture", "\(length) tokens, \(recorders.count) full-attention layers recorded, \(recorders.first?.queries.count ?? 0) decode queries each")
                Self.report(recorders, length: length, log: log)
            }
        }
        await engine.unload()
    }
}
