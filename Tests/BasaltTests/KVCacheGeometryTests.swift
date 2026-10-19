import Testing
import Foundation
@testable import BasaltCore

@Suite("KV cache geometry")
struct KVCacheGeometryTests {

    static var qwen3_4B: [String: Any] { [
        "model_type": "qwen3", "num_hidden_layers": 36, "num_attention_heads": 32,
        "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 2560,
        "max_window_layers": 36, "use_sliding_window": false,
    ] }

    static var ornith9B: [String: Any] { [
        "model_type": "qwen3_5", "hidden_size": 4096,
        "text_config": [
            "model_type": "qwen3_5_text", "num_hidden_layers": 32, "num_attention_heads": 16,
            "num_key_value_heads": 4, "head_dim": 256, "hidden_size": 4096, "full_attention_interval": 4,
            "layer_types": (0..<32).map { ($0 + 1) % 4 == 0 ? "full_attention" : "linear_attention" },
            "linear_num_key_heads": 16, "linear_num_value_heads": 32,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
        ] as [String: Any],
    ] }

    static func dense(layers: Int, kvHeads: Int, headDim: Int = 128) -> KVCacheGeometry {
        KVCacheGeometry(config: [
            "num_hidden_layers": layers, "num_attention_heads": 40,
            "num_key_value_heads": kvHeads, "head_dim": headDim,
        ])!
    }

    @Test("Dense GQA: 2·layers·kv_heads·head_dim·2 B per token")
    func denseGQA() {
        #expect(KVCacheGeometry.kvBytesPerToken(layers: 64, kvHeads: 8, headDim: 128, bytesPerElement: 2) == 262_144)
        let qwen32B = Self.dense(layers: 64, kvHeads: 8)
        #expect(qwen32B.bytes(tokens: 1) == 262_144)
        #expect(qwen32B.bytes(tokens: 32_768) == 8_589_934_592)
        #expect(Self.dense(layers: 28, kvHeads: 4).bytes(tokens: 1) == 57_344)
    }

    @Test("Qwen3-4B config.json: 147,456 B per token, 4.62 GiB at the 33,661-token prompt of #77")
    func qwen3_4B() {
        let g = KVCacheGeometry(config: Self.qwen3_4B)!
        #expect(g.isFromConfig)
        #expect(g.bytes(tokens: 1) == 147_456)
        #expect(g.bytes(tokens: 33_661) == 4_963_516_416)
        #expect(abs(Double(g.bytes(tokens: 33_661)) / Double(1 << 30) - 4.62) < 0.01)
        #expect(g.bytes(tokens: 32_768) == 4_831_838_208)
    }

    @Test("Hybrid (Ornith-1.5-9B, text_config): KV only on the 8 full-attention layers, fixed SSM state on the 24 linear ones")
    func hybrid() {
        let g = KVCacheGeometry(config: Self.ornith9B)!
        let ssm: Int64 = 24 * (3 * 8192 * 2 + 32 * 128 * 128 * 4)
        #expect(g.fixedStateBytes == ssm)
        #expect(g.bytes(tokens: 0) == ssm)
        #expect(g.bytes(tokens: 1) - g.bytes(tokens: 0) == 2 * 8 * 4 * 256 * 2)
        #expect(g.bytes(tokens: 32_768) == 1_073_741_824 + ssm)
    }

    @Test("full_attention_interval alone marks the same layers as layer_types")
    func hybridInterval() {
        var text = Self.ornith9B["text_config"] as! [String: Any]
        text["layer_types"] = nil
        #expect(KVCacheGeometry(config: ["text_config": text]) == KVCacheGeometry(config: Self.ornith9B))
    }

    @Test("Sliding-window layers stop growing at the window")
    func slidingWindow() {
        let g = KVCacheGeometry(config: [
            "num_hidden_layers": 6, "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 256,
            "sliding_window": 512, "sliding_window_pattern": 6,
        ])!
        let perLayer: Int64 = 2 * 256 * 2
        #expect(g.bytes(tokens: 512) == 6 * perLayer * 512)
        #expect(g.bytes(tokens: 4096) == 5 * perLayer * 512 + perLayer * 4096)
        let typed = KVCacheGeometry(config: [
            "num_hidden_layers": 6, "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 256,
            "sliding_window": 512,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                            "sliding_attention", "sliding_attention", "full_attention"],
        ])
        #expect(typed == g)
        #expect(g.maxTokens(fitting: g.bytes(tokens: 4096)) == 4096)
        #expect(g.maxTokens(fitting: g.bytes(tokens: 300)) == 300)
    }

    @Test("--kv-bits: bits/8 plus a fp16 scale and bias per group of 64 elements")
    func kvBits() {
        #expect(KVCacheGeometry.bytesPerElement(kvBits: nil) == 2)
        #expect(KVCacheGeometry.bytesPerElement(kvBits: 8) == 8.5 / 8)
        #expect(KVCacheGeometry.bytesPerElement(kvBits: 4) == 4.5 / 8)
        let g = Self.dense(layers: 64, kvHeads: 8)
        #expect(g.bytes(tokens: 32_768, kvBits: 4) == 8_589_934_592 * 9 / 32)
        let ornith = KVCacheGeometry(config: Self.ornith9B)!
        #expect(ornith.bytes(tokens: 0, kvBits: 4) == ornith.fixedStateBytes)
    }

    @Test("head_dim falls back to hidden_size / num_attention_heads, kv heads to the attention heads")
    func derivedFields() {
        let g = KVCacheGeometry(config: ["num_hidden_layers": 28, "num_attention_heads": 28, "hidden_size": 3584])!
        #expect(g.bytes(tokens: 1) == 2 * 28 * 28 * 128 * 2)
    }

    @Test("Missing geometry → no config geometry; the model falls back to the parameter-count proxy")
    func fallback() throws {
        #expect(KVCacheGeometry(config: ["num_hidden_layers": 36]) == nil)
        #expect(KVCacheGeometry(config: ["num_attention_heads": 8, "head_dim": 128]) == nil)
        #expect(KVCacheGeometry(config: ["num_hidden_layers": 36, "num_attention_heads": 8]) == nil)

        let proxy = KVCacheGeometry(parameterProxy: 7)
        #expect(!proxy.isFromConfig)
        #expect(abs(Double(proxy.bytes(tokens: 4096)) / 1e9 - 0.248) < 0.001)
        #expect(KVCacheGeometry(parameterProxy: 70) == KVCacheGeometry(parameterProxy: 32))

        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(KVCacheGeometry.forModel(dir.path, parameters: 7) == proxy)
        try JSONSerialization.data(withJSONObject: Self.qwen3_4B).write(to: dir.appendingPathComponent("config.json"))
        #expect(KVCacheGeometry.forModel(dir.path, parameters: 7) == KVCacheGeometry(config: Self.qwen3_4B))
    }

    @Test("Gemma 4: full-attention layers use global_head_dim, k=v layers num_global_key_value_heads, shared layers hold no cache")
    func gemma4() {
        var config: [String: Any] = [
            "model_type": "gemma4_text", "num_hidden_layers": 35, "num_attention_heads": 8,
            "num_key_value_heads": 1, "head_dim": 256, "global_head_dim": 512, "num_kv_shared_layers": 20,
            "sliding_window": 512,
            "layer_types": (0..<35).map { ($0 + 1) % 5 == 0 ? "full_attention" : "sliding_attention" },
        ]
        let g = KVCacheGeometry(config: config)!
        #expect(g.elementsPerToken == 3 * 1 * (512 + 512))
        #expect(g.windowedElementsPerToken == 12 * 2 * 1 * 256)
        #expect(g.bytes(tokens: 4096) == 3 * 1024 * 2 * 4096 + 12 * 512 * 2 * 512)

        config["attention_k_eq_v"] = true
        config["num_global_key_value_heads"] = 4
        #expect(KVCacheGeometry(config: config)!.elementsPerToken == 3 * 4 * (512 + 512))
    }

    @Test("MLA (DeepSeek V3): MLX caches the decompressed per-head K (nope + rope) and V, not the latent")
    func mla() {
        let g = KVCacheGeometry(config: [
            "model_type": "deepseek_v3", "num_hidden_layers": 61, "num_attention_heads": 128,
            "num_key_value_heads": 128, "hidden_size": 7168, "kv_lora_rank": 512,
            "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
        ])!
        #expect(g.bytes(tokens: 1) == 61 * 128 * (192 + 128) * 2)
        #expect(g.quantizedGroupSize == 64)
    }

    @Test("Hybrids without layer_types: Jamba attn_layer_period/offset, NemotronH hybrid_override_pattern")
    func hybridPatterns() {
        let jamba = KVCacheGeometry(config: [
            "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096,
            "attn_layer_period": 8, "attn_layer_offset": 4,
        ])!
        #expect(jamba.bytes(tokens: 1) == 2 * 4 * 8 * 128 * 2)
        let nemotron = KVCacheGeometry(config: [
            "num_hidden_layers": 8, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
            "hybrid_override_pattern": "M-M*-M*-",
        ])!
        #expect(nemotron.bytes(tokens: 1) == 2 * 2 * 8 * 128 * 2)
    }

    @Test("--kv-bits leaves sliding-window layers in fp16: MLX quantizes only simple caches, not rotating ones")
    func kvBitsSkipsSlidingLayers() {
        let g = KVCacheGeometry(config: [
            "num_hidden_layers": 6, "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 96,
            "global_head_dim": 128, "sliding_window": 512, "sliding_window_pattern": 6,
        ])!
        #expect(g.quantizedGroupSize == 64)
        let windowed = Int64(5 * 2 * 96 * 2 * 512)
        let full = 2.0 * 128 * 4096
        #expect(g.bytes(tokens: 4096) == Int64(full * 2) + windowed)
        #expect(g.bytes(tokens: 4096, kvBits: 4) == Int64(full * (0.5 + 4.0 / 64)) + windowed)
        #expect(g.maxTokens(fitting: g.bytes(tokens: 4096, kvBits: 4), kvBits: 4) == 4096)
    }

    @Test("--kv-bits group falls back to 32 when head_dim is not a multiple of 64, as MLX does")
    func quantizedGroupFallback() {
        let g = KVCacheGeometry(config: ["num_hidden_layers": 2, "num_attention_heads": 4, "head_dim": 96])!
        #expect(g.quantizedGroupSize == 32)
        #expect(g.bytesPerElement(kvBits: 4) == 0.5 + 4.0 / 32)
        #expect(g.bytes(tokens: 1, kvBits: 4) == Int64(2 * 2 * 4 * 96 * (0.5 + 4.0 / 32)))
        #expect(KVCacheGeometry(config: Self.qwen3_4B)!.quantizedGroupSize == 64)
    }

    @Test("maxTokens inverts bytes: the largest prompt that fits the budget")
    func inverse() {
        let g = KVCacheGeometry(config: Self.ornith9B)!
        let budget = g.bytes(tokens: 20_000)
        #expect(g.maxTokens(fitting: budget) == 20_000)
        #expect(g.maxTokens(fitting: g.fixedStateBytes - 1) == 0)
        #expect(Self.dense(layers: 64, kvHeads: 8).maxTokens(fitting: 4 << 30) == 16_384)
    }
}
