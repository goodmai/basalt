import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Low-severity findings from the 2026-11-25 main defect hunt")
struct LowSeverityDefectTests {
    static func directory(_ name: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appending(path: "\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test("A GGUF template with non-positive linear heads, or value heads that are not a multiple of key heads, is refused instead of trapping or colliding indices",
          arguments: [(0, 4), (-1, 4), (2, 5), (4, 0)])
    func ggufHeadsAreValidated(keys: Int, values: Int) throws {
        let template = try Self.directory("gguf-template")
        defer { try? FileManager.default.removeItem(at: template) }
        let config: [String: Any] = [
            "model_type": "qwen3_5", "linear_num_key_heads": keys, "linear_num_value_heads": values,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        ]
        try JSONSerialization.data(withJSONObject: config).write(to: template.appending(component: "config.json"))
        #expect(throws: BasaltError.self) { _ = try GGUFImporter.layout(template: template) }
    }

    @Test("A weight_map entry that leaves the model directory is ignored when listing shards")
    func indexCannotLeaveDirectory() throws {
        let model = try Self.directory("shards")
        defer { try? FileManager.default.removeItem(at: model) }
        try Data().write(to: model.appending(component: "model.safetensors"))
        let index: [String: Any] = ["weight_map": ["a.weight": "../outside.safetensors"]]
        try JSONSerialization.data(withJSONObject: index).write(to: model.appending(component: "model.safetensors.index.json"))
        #expect(ModelFitAnalyzer.safetensorsShards(in: model).map(\.lastPathComponent) == ["model.safetensors"])
    }

    @Test("A cached local repo whose name contains -- lists under its own id, not a path with an extra slash")
    func cachedIdKeepsDoubleDash() {
        #expect(ModelCache.canonicalRepoId(from: "models--local--a--b") == "local/a--b")
    }

    @Test("A tiered-KV lookup reports each matched block's tier after rebalancing, not the snapshot taken before it")
    func lookupReportsRebalancedTiers() throws {
        var kv = TieredKVHierarchyCore(config: TieredKVConfig(
            chunkTokens: 4, l0CapacityBytes: 100, l1CapacityBytes: 1_000, l2CapacityBytes: 1_000, prefetchLookaheadLayers: 1))
        for hash in ["c1", "c2", "c3"] { try kv.storeChunk(prefixHash: hash, tokenCount: 4, byteSize: 80, layerCount: 4) }
        let lookup = kv.lookupLongestPrefix(chunkHashes: ["c1", "c2", "c3"])
        for block in lookup.matchedBlocks {
            #expect(block.tier == kv.block(for: block.prefixHash)?.tier, "\(block.prefixHash)")
        }
    }
}
