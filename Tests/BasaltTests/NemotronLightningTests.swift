import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("NVIDIA Nemotron-3.5-Lightning-30B-A3B: Mamba-2 KV geometry, mtp.* trunk sanitization, Latent MoE, 2-block Attention+MoE MTP drafter and SSD prefetch (#521)", .serialized)
struct NemotronLightningTests {
    static let lightning30BA3BConfig = """
        {
          "architectures": ["NemotronHForCausalLM"],
          "model_type": "nemotron_h",
          "vocab_size": 131072,
          "hidden_size": 2688,
          "intermediate_size": 1856,
          "moe_intermediate_size": 1856,
          "moe_shared_expert_intermediate_size": 3712,
          "n_routed_experts": 128,
          "n_shared_experts": 1,
          "num_experts_per_tok": 6,
          "n_group": 1,
          "topk_group": 1,
          "norm_topk_prob": true,
          "routed_scaling_factor": 2.5,
          "num_attention_heads": 32,
          "num_key_value_heads": 2,
          "head_dim": 128,
          "mamba_num_heads": 64,
          "mamba_head_dim": 64,
          "ssm_state_size": 128,
          "conv_kernel": 4,
          "n_groups": 8,
          "num_hidden_layers": 52,
          "max_position_embeddings": 262144,
          "layer_norm_epsilon": 1e-5,
          "layers_block_type": [
            "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
            "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
            "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
            "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
            "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
            "mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
            "mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba", "moe"
          ],
          "num_nextn_predict_layers": 1,
          "mtp_layers_block_type": ["attention", "moe"]
        }
        """

    @Test("KVCacheGeometry sizes only the 6 attention layers and 23 Mamba-2 states of Nemotron-3.5-Lightning-30B-A3B instead of 52 full attention layers")
    func lightningKVGeometryAndArchitectureLines() throws {
        let raw = try #require(JSONSerialization.jsonObject(with: Data(Self.lightning30BA3BConfig.utf8)) as? [String: Any])
        let geometry = try #require(KVCacheGeometry(config: raw))
        #expect(geometry.elementsPerToken == Double(6 * 2 * (128 + 128)))
        #expect(geometry.fixedStateBytes > 0)
        let bytesAt262k = geometry.bytes(tokens: TokenCount(262_144)).rawValue
        let naiveFull52 = Int64(KVCacheGeometry.kvBytesPerToken(layers: 52, kvHeads: 2, headDim: 128, bytesPerElement: 2) * 262_144)
        #expect(Double(bytesAt262k) < Double(naiveFull52) * 0.25)

        let lines = architectureLines(config: raw).joined(separator: " | ")
        #expect(lines.contains("Sparse MoE (Experts: 128, top-6)"))
    }

    @Test("NemotronLightningModel normalizes Transformers 5.13 block names, strips embedded mtp.* tensors on sanitize, emits MTP hidden states and matches stock NemotronHModel logits (#228)")
    func trunkSanitizationAndParity() throws {
        let tf5Config = """
            {"model_type": "nemotron_h", "vocab_size": 256, "hidden_size": 64,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
             "mamba_num_heads": 4, "mamba_head_dim": 16, "ssm_state_size": 16,
             "conv_kernel": 4, "n_groups": 2, "intermediate_size": 128,
             "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
             "n_shared_experts": 1, "moe_shared_expert_intermediate_size": 32,
             "routed_scaling_factor": 2.5, "layer_norm_epsilon": 1e-5,
             "layer_types": ["linear_attention", "moe", "full_attention", "moe"]}
            """
        let lightning = try withRandomState(MLXRandom.RandomState(seed: 5210)) {
            try NemotronLightningModel(configurationData: Data(tf5Config.utf8))
        }
        let stock = NemotronHModel(lightning.configuration)
        try stock.update(parameters: lightning.parameters(), verify: [.all])
        eval(lightning, stock)

        var dirtyWeights = Dictionary(uniqueKeysWithValues: lightning.parameters().flattened())
        dirtyWeights["mtp.layers.0.eh_proj.weight"] = MLXArray.ones([64, 128])
        dirtyWeights["mtp.layers.1.mixer.experts.0.up_proj.weight"] = MLXArray.ones([32, 64])
        let clean = lightning.sanitize(weights: dirtyWeights)
        #expect(!clean.keys.contains { $0.hasPrefix("mtp.") })
        try lightning.update(parameters: ModuleParameters.unflattened(clean), verify: [.all])

        let tokens = MLXArray([Int32(3), 11, 27, 45, 9, 18]).reshaped(1, 6)
        var emitState = LMOutput.State()
        emitState[mtpEmitFlagKey] = true
        let out = lightning(LMInput.Text(tokens: tokens), cache: try lightning.newCache(parameters: nil), state: emitState)
        let refLogits = stock(tokens, cache: try stock.newCache(parameters: nil))
        let hidden = try #require(out.state?[mtpLastHiddenStatesKey])
        eval(out.logits, refLogits, hidden)
        #expect(hidden.shape == [1, 6, 64])
        #expect(abs(out.logits.asType(.float32) - refLogits.asType(.float32)).max().item(Float.self) < 1e-5)
    }

    @Test("NemotronLatentMoEMixer projects routed tokens through moe_latent_size and NemotronHMTPDraftModel loads mtp-4bit.safetensors sidecar and drafts valid tokens")
    func latentMoEAndMTPSidecar() throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "nemotron-mtp-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let rawCfg = TinyModel.Family.sigmoidMoe.config.replacingOccurrences(
            of: "\"tie_word_embeddings\": true",
            with: "\"tie_word_embeddings\": false, \"moe_latent_size\": 32"
        )
        try Data(rawCfg.utf8).write(to: dir.appending(component: "config.json"))
        let target = try NemotronLightningModel(configurationData: Data(rawCfg.utf8))
        eval(target)
        #expect(target.moeLatentSize == 32)

        let drafter = NemotronHMTPDraftModel(target.configuration, moeLatentSize: 32)
        eval(drafter)
        var rawMTP = Dictionary(uniqueKeysWithValues: drafter.parameters().flattened())
        for (src, dst) in [("up_proj", "fc1"), ("down_proj", "fc2")] {
            let removed = rawMTP.removeValue(forKey: "mtp.layers.1.mixer.switch_mlp.\(dst).weight")
            let stacked = try #require(removed)
            for idx in 0 ..< target.configuration.nRoutedExperts {
                rawMTP["mtp.layers.1.mixer.experts.\(idx).\(src).weight"] = stacked[idx]
            }
        }
        try save(arrays: rawMTP, metadata: ["format": "mlx"], url: dir.appending(component: "mtp-4bit.safetensors"))

        let loaded = try #require(try MTPSidecar.load(directory: dir, target: target) as? any StatefulMTPDrafterModel)
        var state = loaded.makeState(parameters: nil)
        let prompt = MLXArray([Int32(4), 12, 19, 25]).reshaped(1, 4)
        let targetHidden = MLXRandom.normal([1, 4, 64])
        let bonus = MLXArray([Int32(31)])
        loaded.prepareDrafterState(
            target: target,
            input: MTPPrefillInput(promptTokens: prompt, targetHidden: targetHidden, firstBonus: bonus),
            state: &state,
            sampler: ArgMaxSampler()
        )
        let proposed = loaded.draftBlock(
            target: target,
            input: MTPDraftInput(lastToken: bonus, lastHidden: targetHidden[0..., 3 ..< 4, 0...], sharedKV: [:], positionDeltas: nil, queryOffset: 4, blockSize: 2),
            state: &state,
            sampler: ArgMaxSampler()
        )
        eval(proposed)
        #expect(proposed.shape == [1, 1])
        let tok = proposed.item(Int.self)
        #expect(tok >= 0 && tok < target.vocabularySize)
    }

    @Test("NemotronLightningModel synchronizes backing lm_head when MLXNN.quantize replaces top-level Linear with QuantizedLinear")
    func quantizedLMHeadSynchronizationParity() throws {
        let untiedConfig = """
            {"model_type": "nemotron_h", "vocab_size": 256, "hidden_size": 64,
             "tie_word_embeddings": false,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
             "mamba_num_heads": 4, "mamba_head_dim": 16, "ssm_state_size": 16,
             "conv_kernel": 4, "n_groups": 2, "intermediate_size": 128,
             "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
             "n_shared_experts": 1, "moe_shared_expert_intermediate_size": 32,
             "routed_scaling_factor": 2.5, "layer_norm_epsilon": 1e-5,
             "layer_types": ["linear_attention", "moe", "full_attention", "moe"]}
            """
        let lightning = try withRandomState(MLXRandom.RandomState(seed: 7711)) {
            try NemotronLightningModel(configurationData: Data(untiedConfig.utf8))
        }
        let stock = withRandomState(MLXRandom.RandomState(seed: 9922)) {
            NemotronHModel(lightning.configuration)
        }
        let qFilter: (String, Module) -> Bool = { _, mod in
            guard let lin = mod as? Linear else { return false }
            return lin.weight.dim(-1).isMultiple(of: 32)
        }
        MLXNN.quantize(model: lightning, groupSize: 32, bits: 4, filter: qFilter)
        MLXNN.quantize(model: stock, groupSize: 32, bits: 4, filter: qFilter)
        try lightning.update(parameters: stock.parameters(), verify: [.all])
        eval(lightning, stock)

        let tokens = MLXArray([Int32(5), 14, 29, 63]).reshaped(1, 4)
        let directLogits = lightning(tokens, cache: try lightning.newCache(parameters: nil))
        let lmOut = lightning(LMInput.Text(tokens: tokens), cache: try lightning.newCache(parameters: nil), state: nil)
        let stockLogits = stock(tokens, cache: try stock.newCache(parameters: nil))
        eval(directLogits, lmOut.logits, stockLogits)

        let errDirect = abs(directLogits.asType(.float32) - stockLogits.asType(.float32)).max().item(Float.self)
        let errLM = abs(lmOut.logits.asType(.float32) - stockLogits.asType(.float32)).max().item(Float.self)
        #expect(errDirect < 1e-5)
        #expect(errLM < 1e-5)
    }
}
