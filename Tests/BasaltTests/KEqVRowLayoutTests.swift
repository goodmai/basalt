import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Gemma 4 K=V global-layer row layout (#645)")
struct KEqVRowLayoutTests {
    static func config(layers: Int, globalEvery: Int, kvHeads: Int) -> [String: Any] {
        let types = (0 ..< layers).map { ($0 + 1).isMultiple(of: globalEvery) ? "full_attention" : "sliding_attention" }
        return ["text_config": [
            "attention_k_eq_v": true, "global_head_dim": 512, "head_dim": 256, "num_hidden_layers": layers,
            "num_kv_shared_layers": 0, "num_global_key_value_heads": kvHeads, "num_key_value_heads": 8, "num_attention_heads": 16, "sliding_window": 1024, "layer_types": types,
            "rope_parameters": ["full_attention": ["partial_rotary_factor": 0.25, "rope_type": "proportional"]],
        ] as [String: Any]]
    }

    @Test("Global bytes per token match the issue table for Gemma 4 12B, 26B-A4B and 31B", arguments: [
        (48, 6, 1, 16_384, 10_240), (30, 6, 2, 20_480, 12_800), (60, 6, 4, 81_920, 51_200),
    ])
    func bytes(layers: Int, every: Int, kvHeads: Int, stock: Int, row: Int) throws {
        let layout = try #require(KEqVRowLayout(config: Self.config(layers: layers, globalEvery: every, kvHeads: kvHeads)))
        #expect(layout.rowWidth == 640)
        #expect(layout.globalBytesPerToken() == (stock, row))
    }

    @Test("Rotated set is {0..63} ∪ {256..319}; key and value orders are permutations, and the output restore inverts the value order")
    func sets() throws {
        let layout = try #require(KEqVRowLayout(headDim: 512, partialRotaryFactor: 0.25, kvHeads: 1, globalLayers: 1))
        #expect(layout.rotated == Array(0 ..< 64) + Array(256 ..< 320))
        #expect(Set(layout.keyOrder) == Set(0 ..< 512) && Set(layout.valueOrder) == Set(0 ..< 512))
        #expect(layout.outputRestore.map { layout.valueOrder[$0] } == Array(0 ..< 512))
        #expect(Array(layout.keyOrder[128...]) == Array(layout.valueOrder[..<384]))
    }

    @Test("The KV geometry of a 12B-shaped config drops 384 values per global KV head per token, 6,144 bytes at fp16, with sliding layers unchanged")
    func geometry() throws {
        let config = Self.config(layers: 48, globalEvery: 6, kvHeads: 1)
        let stock = try #require(KVCacheGeometry(config: config))
        let layout = try #require(KEqVRowLayout(config: config))
        let row = stock.sharingKEqVRows(layout)
        #expect(stock.bytesPerToken(nil) - row.bytesPerToken(nil) == 6_144)
        #expect(row.windowedElementsPerToken == stock.windowedElementsPerToken && row.window == stock.window)
        #expect(KVCacheGeometry(parameterProxy: 12).sharingKEqVRows(layout) == KVCacheGeometry(parameterProxy: 12))
    }

    @Test("No layout without attention_k_eq_v, with full rotation, or with no rotation")
    func refusals() {
        var plain = Self.config(layers: 48, globalEvery: 6, kvHeads: 1)
        var text = plain["text_config"] as! [String: Any]
        text["attention_k_eq_v"] = false
        plain["text_config"] = text
        #expect(KEqVRowLayout(config: plain) == nil)
        #expect(KEqVRowLayout(config: ["model_type": "qwen3", "head_dim": 128]) == nil)
        #expect(KEqVRowLayout(headDim: 512, partialRotaryFactor: 1, kvHeads: 1, globalLayers: 1) == nil)
        #expect(KEqVRowLayout(headDim: 512, partialRotaryFactor: 0, kvHeads: 1, globalLayers: 1) == nil)
    }
}

@Suite("Gemma 4 K=V identity on the gemma4KEqV TinyModel (#645)")
struct KEqVIdentityTests {
    @Test("On global layers the cached key's unrotated dims equal k_norm.weight ⊙ value, at every position; the rotated dims match only at position 0")
    func identity() throws {
        let model = try TinyModel.make(.gemma4KEqV)
        let caches = try model.newCache(parameters: nil)
        let tokens = MLXArray([Int32(5), 19, 42, 7, 88, 13])[.newAxis]
        _ = model(LMInput.Text(tokens: tokens), cache: caches, state: nil)
        let layout = try #require(KEqVRowLayout(headDim: 64, partialRotaryFactor: 0.25, kvHeads: 1, globalLayers: 2))
        let norms = model.leafModules().flattened().filter { $0.0.hasSuffix("self_attn.k_norm") }
        var checked = 0
        for (index, cache) in caches.enumerated() {
            guard let simple = cache as? KVCacheSimple, simple.state.count == 2, simple.state[0].dim(-1) == 64 else { continue }
            let (keys, values) = (simple.state[0], simple.state[1])
            let path = "model.layers.\(index).self_attn.k_norm"
            let norm = try #require(norms.first { $0.0.hasSuffix(path) }?.1 as? RMSNorm)
            let k = keys.asType(.float32)
            let folded = values.asType(.float32) * norm.weight.asType(.float32)
            let gap = abs(k - folded)
            let unrotated = gap.take(MLXArray(layout.unrotated.map(Int32.init)), axis: -1).max().item(Float.self)
            let firstToken = gap[.ellipsis, 0, 0...].max().item(Float.self)
            let rotatedLater = gap[.ellipsis, 1..., 0...].take(MLXArray(layout.rotated.map(Int32.init)), axis: -1).max(axes: [0, 1, 2]).asArray(Float.self)
            #expect(unrotated <= 1e-6)
            #expect(firstToken <= 1e-6)
            #expect(rotatedLater.filter { $0 > 1e-4 }.count >= layout.rotated.count - 2)
            checked += 1
        }
        #expect(checked == 2)
    }
}
