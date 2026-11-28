import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXLLM
import MLXVLM
@testable import BasaltInference

enum TinyModel {
    enum Family: String, CaseIterable, Sendable {
        case qwen3Small, qwen3, qwen3HeadDim32, qwen35, qwen35Moe, qwen3Next, gemma3, gemma4, gemma4Moe, gemma4KEqV, gemma4Unified, inkling2Loop, qwen3Moe, sigmoidMoe, nemotronH, lfm2, lfm2Moe, prism, paddleOCR, deepSeekOCR, deepSeekOCR2, flint, ricursive, museGlimmer, museGlimmerText, ltxVideo, kandinskyCrossDiT, moeVideoDiT, bailingHybrid, ouro, llada, huginn, rwkv7, xlstm, mhc, mamba3Siso, mamba3Mimo, gdn2, tttLinear, tttMlp, titans, lact, plt, deepseekV3

        var decodesTokenByToken: Bool { self != .llada }

        func model(_ config: Data) throws -> any LanguageModel {
            let decoder = JSONDecoder.json5()
            return switch self {
            case .qwen3Small, .qwen3, .qwen3HeadDim32: Qwen3Model(try decoder.decode(Qwen3Configuration.self, from: config))
            case .qwen35, .qwen35Moe: Qwen35TextModel(try decoder.decode(Qwen35TextConfiguration.self, from: config))
            case .gemma3: Gemma3TextModel(try decoder.decode(Gemma3TextConfiguration.self, from: config))
            case .gemma4, .gemma4Moe, .gemma4KEqV: Gemma4TextModel(try decoder.decode(Gemma4TextConfiguration.self, from: config))
            case .qwen3Moe, .deepseekV3: try attentionMoEModel(config, decoder: decoder)
            case .sigmoidMoe, .nemotronH: NemotronHModel(try decoder.decode(NemotronHConfiguration.self, from: config))
            case .lfm2: LFM2Model(try decoder.decode(LFM2Configuration.self, from: config))
            case .lfm2Moe: LFM2MoEModel(try decoder.decode(LFM2MoEConfiguration.self, from: config))
            case .prism: MLXLLM.Qwen35Model(try decoder.decode(MLXLLM.Qwen35Configuration.self, from: config))
            case .flint: FlintModel(try decoder.decode(FlintConfiguration.self, from: config).validated())
            case .ricursive: RicursiveModel(try decoder.decode(RicursiveConfiguration.self, from: config).validated())
            case .museGlimmerText: MuseGlimmerTextModel(try MuseGlimmerTextConfiguration.decodeFlexible(from: config))
            case .ltxVideo, .kandinskyCrossDiT, .moeVideoDiT: try videoModel(config, decoder: decoder)
            case .gemma4Unified, .paddleOCR, .deepSeekOCR, .deepSeekOCR2, .museGlimmer, .inkling2Loop, .qwen3Next, .bailingHybrid, .ouro, .llada, .huginn, .rwkv7, .xlstm, .mhc, .mamba3Siso, .mamba3Mimo, .gdn2, .tttLinear, .tttMlp, .titans, .lact, .plt: try multimodalModel(config, decoder: decoder)
            }
        }

        private func attentionMoEModel(_ config: Data, decoder: JSONDecoder) throws -> any LanguageModel {
            if self == .deepseekV3 { return DeepseekV3Model(try decoder.decode(DeepseekV3Configuration.self, from: config)) }
            return Qwen3MoEModel(try decoder.decode(Qwen3MoEConfiguration.self, from: config))
        }

        private func videoModel(_ config: Data, decoder: JSONDecoder) throws -> any LanguageModel {
            let decoded = try decoder.decode(VideoDiTConfiguration.self, from: config)
            if self == .ltxVideo { return LTX25AVDiTModel(decoded) }
            if self == .kandinskyCrossDiT { return KandinskyCrossDiTModel(decoded) }
            return MoEVideoDiTModel(decoded)
        }

        private func multimodalModel(_ config: Data, decoder: JSONDecoder) throws -> any LanguageModel {
            if self == .gemma4Unified { return try TinyGemma4Unified.model(config) }
            if self == .inkling2Loop { return try Inkling2Model(configurationData: config) }
            if self == .qwen3Next { return Qwen3NextModel(try decoder.decode(Qwen3NextConfiguration.self, from: config)) }
            if self == .bailingHybrid { return try BailingHybridModel(configurationData: config) }
            if self == .ouro { return try OuroModel(configurationData: config) }
            if self == .llada { return try LLaDAModel(configurationData: config) }
            if self == .huginn { return try HuginnModel(configurationData: config) }
            if self == .rwkv7 { return try RWKV7Model(configurationData: config) }
            if let recurrent = try recurrentModel(config) { return recurrent }
            if let fastWeight = try fastWeightModel(config) { return fastWeight }
            if self == .plt { return try PLTModel(configurationData: config) }
            if self == .paddleOCR { return PaddleOCRVL(try decoder.decode(PaddleOCRVLConfiguration.self, from: config)) }
            if self == .deepSeekOCR { return DeepSeekOCR(try decoder.decode(DeepSeekOCRConfiguration.self, from: config).validated()) }
            if self == .museGlimmer { return MuseGlimmerVLM(try decoder.decode(MuseGlimmerConfiguration.self, from: config)) }
            return DeepSeekOCR2(try decoder.decode(DeepSeekOCR2Configuration.self, from: config).validated())
        }

        var weightScale: Float {
            switch self {
            case .qwen3Small, .gemma3: 4
            case .qwen3, .qwen3HeadDim32, .gemma4, .gemma4Moe, .gemma4KEqV, .gemma4Unified, .inkling2Loop, .qwen35, .qwen35Moe, .qwen3Next, .qwen3Moe, .sigmoidMoe, .nemotronH, .lfm2, .lfm2Moe, .prism, .paddleOCR, .deepSeekOCR, .deepSeekOCR2, .flint, .ricursive, .museGlimmer, .museGlimmerText, .ltxVideo, .kandinskyCrossDiT, .moeVideoDiT, .bailingHybrid, .ouro, .llada, .huginn, .rwkv7, .xlstm, .mhc, .mamba3Siso, .mamba3Mimo, .gdn2, .tttLinear, .tttMlp, .titans, .lact, .plt, .deepseekV3: 1
            }
        }

        var config: String {
            switch self {
            case .qwen3Small: Self.qwen3.config.replacingOccurrences(of: "\"num_hidden_layers\": 4", with: "\"num_hidden_layers\": 2")
            case .qwen3: """
                {"hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                 "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-5,
                 "vocab_size": 256, "rope_theta": 10000.0, "tie_word_embeddings": false}
                """
            case .qwen3HeadDim32: Self.qwen3.config.replacingOccurrences(of: "\"head_dim\": 16", with: "\"head_dim\": 32")
            case .qwen35: """
                {"model_type": "qwen3_5_text", "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                 "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                 "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                 "linear_key_head_dim": 16, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
                 "full_attention_interval": 2, "rms_norm_eps": 1e-5, "vocab_size": 256,
                 "tie_word_embeddings": false}
                """
            case .qwen35Moe: Self.qwen35.config.replacingOccurrences(
                of: "\"tie_word_embeddings\": false",
                with: "\"tie_word_embeddings\": false, \"num_experts\": 4, \"num_experts_per_tok\": 2, \"moe_intermediate_size\": 32, \"shared_expert_intermediate_size\": 32")
            case .qwen3Next: """
                {"model_type": "qwen3_next", "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                 "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "partial_rotary_factor": 0.25,
                 "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                 "linear_key_head_dim": 16, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
                 "full_attention_interval": 4, "num_experts": 16, "num_experts_per_tok": 4, "moe_intermediate_size": 32,
                 "shared_expert_intermediate_size": 32, "decoder_sparse_step": 1, "mlp_only_layers": [], "norm_topk_prob": true,
                 "rms_norm_eps": 1e-6, "vocab_size": 256, "rope_theta": 10000.0, "tie_word_embeddings": false}
                """
            case .gemma3: """
                {"model_type": "gemma3_text", "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                 "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 256,
                 "sliding_window": \(slidingWindow), "sliding_window_pattern": 2, "query_pre_attn_scalar": 16}
                """
            case .gemma4: """
                {"model_type": "gemma4_text", "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                 "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "global_head_dim": 32,
                 "num_kv_shared_layers": 2, "hidden_size_per_layer_input": 16, "vocab_size": 256,
                 "vocab_size_per_layer_input": 256, "sliding_window": \(slidingWindow), "sliding_window_pattern": 2,
                 "use_double_wide_mlp": false, "rms_norm_eps": 1e-6}
                """
            case .inkling2Loop: """
                {"model_type": "inkling2_loop", "text_config": {"model_type": "inkling2_loop_text", "hidden_size": 64, "num_hidden_layers": 6,
                 "vocab_size": 272, "unpadded_vocab_size": 256, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                 "d_rel": 4, "rel_extent": 8, "log_scaling_n_floor": 4, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6,
                 "qk_l2_clip_threshold": 22.627416997969522, "rel_l2_clip_threshold": 8.0, "local_layer_ids": [0, 2, 4],
                 "sconv_kernel_size": 4, "final_logit_softcapping": 30.0, "sliding_window_size": 8, "n_layers_per_attn_res_block": 2,
                 "n_routed_experts": 8, "num_experts_per_tok": 2, "n_shared_experts": 0, "intermediate_size": 32,
                 "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 8.0,
                 "loop_config": {"core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 8, "is_varloop": false}}}
                """
            case .gemma4KEqV: """
                {"model_type": "gemma4_text", "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                 "num_attention_heads": 4, "num_key_value_heads": 2, "num_global_key_value_heads": 1, "head_dim": 16,
                 "global_head_dim": 64, "attention_k_eq_v": true, "num_kv_shared_layers": 0, "vocab_size": 256,
                 "sliding_window": 8, "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
                 "rope_parameters": {"full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
                  "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}},
                 "use_double_wide_mlp": false, "rms_norm_eps": 1e-6, "tie_word_embeddings": true}
                """
            case .gemma4Unified: """
                {"model_type": "gemma4_unified", "vocab_size": 256, "image_token_id": 255, "audio_token_id": 254, "video_token_id": 253,
                 "text_config": {"model_type": "gemma4_unified_text", "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                  "num_attention_heads": 4, "num_key_value_heads": 2, "num_global_key_value_heads": 2, "head_dim": 16, "global_head_dim": 16,
                  "num_kv_shared_layers": 1, "hidden_size_per_layer_input": 16, "vocab_size": 256, "vocab_size_per_layer_input": 256,
                  "sliding_window": \(slidingWindow), "sliding_window_pattern": 2, "attention_k_eq_v": true, "use_double_wide_mlp": false,
                  "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"], "tie_word_embeddings": true},
                 "vision_config": null, "audio_config": null}
                """
            case .gemma4Moe: Self.gemma4.config.replacingOccurrences(
                of: "\"rms_norm_eps\": 1e-6}",
                with: "\"rms_norm_eps\": 1e-6, \"enable_moe_block\": true, \"num_experts\": 8, \"top_k_experts\": 2, \"moe_intermediate_size\": 32}")
            case .qwen3Moe: """
                {"model_type": "qwen3_moe", "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                 "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-5,
                 "vocab_size": 256, "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
                 "decoder_sparse_step": 1, "mlp_only_layers": [], "norm_topk_prob": true, "tie_word_embeddings": false}
                """
            case .sigmoidMoe: """
                {"model_type": "nemotron_h", "vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 4,
                 "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "mamba_num_heads": 4,
                 "mamba_head_dim": 16, "ssm_state_size": 16, "conv_kernel": 4, "n_groups": 2, "intermediate_size": 128,
                 "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32, "n_shared_experts": 1,
                 "moe_shared_expert_intermediate_size": 32, "n_group": 2, "topk_group": 1, "routed_scaling_factor": 2.5,
                 "hybrid_override_pattern": "ME*E", "layer_norm_epsilon": 1e-5, "tie_word_embeddings": true}
                """
            case .nemotronH: """
                {"model_type": "nemotron_h", "vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 4,
                 "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "mamba_num_heads": 4,
                 "mamba_head_dim": 16, "ssm_state_size": 16, "conv_kernel": 4, "n_groups": 2, "intermediate_size": 128,
                 "hybrid_override_pattern": "M*M-", "layer_norm_epsilon": 1e-5, "tie_word_embeddings": true}
                """
            case .lfm2: """
                {"model_type": "lfm2", "vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 4,
                 "num_attention_heads": 4, "num_key_value_heads": 2, "norm_eps": 1e-5, "conv_L_cache": 3,
                 "block_dim": 64, "block_ff_dim": 128, "block_multiple_of": 16,
                 "layer_types": ["conv", "full_attention", "conv", "full_attention"]}
                """
            case .lfm2Moe: """
                {"model_type": "lfm2_moe", "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128,
                 "moe_intermediate_size": 32, "num_hidden_layers": 4, "num_experts": 4, "num_experts_per_tok": 2,
                 "norm_topk_prob": true, "num_attention_heads": 4, "num_key_value_heads": 2,
                 "max_position_embeddings": 4096, "use_expert_bias": true, "num_dense_layers": 1, "norm_eps": 1e-5,
                 "conv_bias": false, "conv_L_cache": 3, "layer_types": ["conv", "full_attention", "conv", "full_attention"]}
                """
            case .prism: """
                {"model_type": "qwen3_5", "text_config": \(Self.qwen35.config)}
                """
            case .paddleOCR: """
                {"model_type": "paddleocr_vl", "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                 "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-5,
                 "vocab_size": 256, "rope_theta": 500000.0, "rope_scaling": {"mrope_section": [2, 3, 3], "type": "default"},
                 "image_token_id": 250,
                 "vision_config": {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
                  "num_channels": 3, "image_size": 56, "patch_size": 14, "layer_norm_eps": 1e-6, "spatial_merge_size": 2}}
                """
            case .deepSeekOCR: """
                {"model_type": "deepseekocr", "image_token_index": 250,
                 "language_config": {"hidden_size": 64, "num_hidden_layers": 3, "intermediate_size": 128, "moe_intermediate_size": 32,
                  "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 256, "n_routed_experts": 4, "n_shared_experts": 1,
                  "num_experts_per_tok": 2, "first_k_dense_replace": 1, "qk_nope_head_dim": 0, "topk_method": "greedy",
                  "rms_norm_eps": 1e-6, "rope_theta": 10000.0},
                 "vision_config": {"image_size": 128, "width": {
                  "sam_vit_b": {"width": 32, "layers": 2, "heads": 2, "global_attn_indexes": [1], "downsample_channels": [512, 64], "window_size": 3},
                  "clip-l-14-224": {"width": 64, "layers": 2, "heads": 4, "patch_size": 14, "image_size": 224}}},
                 "projector_config": {"input_dim": 128, "n_embed": 64, "projector_type": "linear"}}
                """
            case .deepSeekOCR2: """
                {"model_type": "deepseekocr_2", "image_token_index": 250,
                 "language_config": {"hidden_size": 64, "num_hidden_layers": 3, "intermediate_size": 128, "moe_intermediate_size": 32,
                  "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 256, "n_routed_experts": 4, "n_shared_experts": 1,
                  "num_experts_per_tok": 2, "first_k_dense_replace": 1, "qk_nope_head_dim": 0, "topk_method": "greedy",
                  "rms_norm_eps": 1e-6, "rope_theta": 10000.0},
                 "vision_config": {"image_size": 1024, "width": {
                  "sam_vit_b": {"width": 32, "layers": 2, "heads": 2, "global_attn_indexes": [1], "downsample_channels": [512, 1024], "window_size": 14},
                  "qwen2-0-5b": {"dim": 64, "layers": 2, "heads": 4, "kv_heads": 2, "intermediate_size": 128}}},
                 "projector_config": {"input_dim": 64, "n_embed": 64, "projector_type": "linear"}}
                """
            case .flint, .ricursive, .museGlimmerText, .museGlimmer, .ltxVideo, .kandinskyCrossDiT, .moeVideoDiT, .bailingHybrid, .ouro, .llada, .huginn, .rwkv7, .xlstm, .mhc, .mamba3Siso, .mamba3Mimo, .gdn2, .tttLinear, .tttMlp, .titans, .lact, .plt, .deepseekV3:
                extendedConfig
            }
        }
    }

    static let slidingWindow = 8
    static let config = Family.qwen3Small.config
    static let weightScale = Family.qwen3Small.weightScale

    static func make(_ family: Family = .qwen3Small, adding extra: String = "") throws -> any LanguageModel {
        let config = "{" + extra + family.config.dropFirst()
        let model = try withRandomState(MLXRandom.RandomState(seed: family.seed)) {
            let model = try family.model(Data(config.utf8))
            seedSpecialParameters(model, family: family)
            return model
        }
        if family.weightScale != 1 {
            model.update(parameters: model.parameters().mapValues { $0 * family.weightScale })
        }
        try model.prepare()
        eval(model)
        return model
    }

    static func container(_ family: Family = .qwen3Small, processor: any UserInputProcessor = ByteInputProcessor()) throws -> ModelContainer {
        ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-\(family.rawValue)"),
            model: try make(family),
            processor: processor,
            tokenizer: ByteTokenizer()))
    }

    static let cacheBytesPerToken = 2 * 2 * 16 * 4 * 2

    static func engine(
        _ family: Family = .qwen3Small, promptCacheBytes: Int? = nil, repetitionPenalty: Float? = nil,
        processor: any UserInputProcessor = ByteInputProcessor()
    ) async throws -> MLXInferenceEngine {
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: promptCacheBytes, repetitionPenalty: repetitionPenalty))
        await engine.attach(try container(family, processor: processor))
        return engine
    }

    static func tokens(_ prompt: String) -> [Int] {
        ByteTokenizer().encode(text: prompt, addSpecialTokens: false) + [ByteTokenizer.endOfUserTurn]
    }
}

extension TinyModel.Family {
    var extendedConfig: String {
        switch self {
        case .flint: """
            {"model_type": "flint", "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 256,
             "sliding_window": \(TinyModel.slidingWindow), "use_sliding_window": true,
             "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
             "no_rope_layer_indices": [0, 3], "mlp_only_layers": [0, 1],
             "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
             "shared_expert_intermediate_size": 32, "router_score_function": "sigmoid",
             "norm_topk_prob": true, "router_expert_bias": true, "softmax_type": "learnable",
             "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": false}
            """
        case .ricursive: """
            {"model_type": "ricursive", "hidden_size": 64, "num_hidden_layers": 5, "intermediate_size": 128,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 256,
             "sliding_window": \(TinyModel.slidingWindow),
             "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
             "use_gated_attention": true, "gate_full_precision": true, "use_valres": true,
             "ffn_clamp_threshold": 20.0, "embed_scale": 8.0, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
             "tie_word_embeddings": false}
            """
        case .museGlimmerText: """
            {"model_type": "muse_glimmer_text", "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 256,
             "sliding_window": \(TinyModel.slidingWindow), "qk_scale_factor": 3.87, "output_multiplier": 0.19611613,
             "final_logit_softcapping": 20.0, "rms_norm_eps": 1e-5, "post_norm_eps": 1e-8,
             "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
             "layer_rope_theta": [500000.0, 500000.0, 500000.0, 0.0], "tie_word_embeddings": false}
            """
        case .museGlimmer: """
            {"model_type": "muse_glimmer", "image_token_id": 254, "video_token_id": 253,
             "out_hidden_size": 64, "projector_hidden_size": 64,
             "text_config": \(Self.museGlimmerText.config),
             "vision_config": {"model_type": "muse_glimmer_vision", "hidden_size": 16, "intermediate_size": 32,
              "num_hidden_layers": 2, "num_attention_heads": 2, "patch_size": 14, "patch_temporal": 2, "merge_size": 2,
              "max_position_embeddings": 16, "pos_emb_height": 4, "pos_emb_width": 4, "layer_norm_eps": 1e-5,
              "layer_types": ["window_attention", "full_attention"]}}
            """
        case .ltxVideo: """
            {"model_type": "ltx_video", "vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2,
             "num_attention_heads": 4, "intermediate_size": 128, "video_channels": 16, "audio_channels": 8}
            """
        case .kandinskyCrossDiT: """
            {"model_type": "kandinsky_cross_dit", "vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2,
             "num_attention_heads": 4, "intermediate_size": 128, "video_channels": 16, "audio_channels": 8}
            """
        case .moeVideoDiT: """
            {"model_type": "moe_video_dit", "vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2,
             "num_attention_heads": 4, "intermediate_size": 64, "video_channels": 16, "audio_channels": 8,
             "num_experts": 4, "num_experts_per_tok": 2}
            """
        case .bailingHybrid: """
            {"model_type": "bailing_hybrid", "vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 6,
             "layer_group_size": 3, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 4,
             "head_dim": 32, "short_conv_kernel_size": 4, "no_kda_lora": true, "kda_safe_gate": true,
             "kda_lower_bound": -5.0, "q_lora_rank": 32, "kv_lora_rank": 32, "qk_nope_head_dim": 32,
             "qk_rope_head_dim": 16, "v_head_dim": 32, "rope_theta": 6000000.0, "rope_interleave": true,
             "gated_attention_proj_granularity_type": "head_wise", "num_experts": 8, "num_experts_per_tok": 2,
             "n_group": 4, "topk_group": 2, "num_shared_experts": 1, "moe_intermediate_size": 32,
             "moe_shared_expert_intermediate_size": 32, "first_k_dense_replace": 1, "routed_scaling_factor": 2.5,
             "norm_topk_prob": true, "moe_router_enable_expert_bias": true, "rms_norm_eps": 1e-6,
             "tie_word_embeddings": false}
            """
        case .plt: """
            {"model_type": "iquestpltcoder", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-6, "vocab_size": 256,
             "rope_theta": 10000.0, "plt_num_loops": 2, "plt_window_size": [8, 0], "plt_normalize_per_loop": true,
             "plt_emb_scale": 0.707, "plt_hidden_scale": 0.5, "plt_gate_use_hidden_states": true, "tie_word_embeddings": false}
            """
        case .ouro: """
            {"model_type": "ouro", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-6,
             "vocab_size": 256, "rope_theta": 10000.0, "total_ut_steps": 4, "tie_word_embeddings": false}
            """
        case .llada: """
            {"model_type": "llada", "d_model": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2, "mlp_hidden_size": 128,
             "vocab_size": 256, "embedding_size": 256, "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "mask_token_id": 255}
            """
        case .huginn: """
            {"model_type": "huginn_raven", "n_embd": 64, "n_heads": 4, "head_dim": 16, "n_layers_in_prelude": 1,
             "n_layers_in_recurrent_block": 2, "n_layers_in_coda": 1, "intermediate_size": 128, "vocab_size": 256,
             "norm_eps": 1e-6, "rope_base": 50000.0, "mean_recurrence": 3, "injection_type": "linear", "qk_bias": true,
             "bias": false, "tie_embeddings": true}
            """
        case .rwkv7: """
            {"model_type": "rwkv7", "vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 3,
             "head_dim": 16, "num_heads": 4, "decay_low_rank_dim": 16, "a_low_rank_dim": 16,
             "v_low_rank_dim": 16, "g_low_rank_dim": 16, "intermediate_size": 128, "norm_eps": 1e-5,
             "norm_first": true, "tie_word_embeddings": false}
            """
        case .xlstm: """
            {"model_type": "xlstm", "vocab_size": 256, "embedding_dim": 64, "num_blocks": 3,
             "num_heads": 4, "qk_dim_factor": 0.5, "v_dim_factor": 1.0, "ffn_proj_factor": 2.667,
             "ffn_round_up_to_multiple_of": 16, "chunk_size": 4, "gate_soft_cap": 15.0,
             "output_logit_soft_cap": 30.0, "norm_eps": 1e-6, "eps": 1e-6}
            """
        case .mhc: """
            {"model_type": "mhc", "vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2,
             "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
             "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": false,
             "hc_variant": "mhc", "hc_expansion_rate": 4, "hc_sinkhorn_iterations": 20,
             "hc_gate_mode": "sigmoid", "hc_collapse_mode": "mean", "hc_dynamic": true, "hc_alpha_init": 0.01}
            """
        case .deepseekV3: """
            {"model_type": "deepseek_v3", "vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 3,
             "intermediate_size": 128, "moe_intermediate_size": 32, "num_attention_heads": 4, "num_key_value_heads": 4,
             "kv_lora_rank": 32, "q_lora_rank": null, "qk_nope_head_dim": 24, "qk_rope_head_dim": 8, "v_head_dim": 16,
             "n_routed_experts": 8, "num_experts_per_tok": 2, "n_shared_experts": 1, "n_group": 2, "topk_group": 1,
             "scoring_func": "sigmoid", "topk_method": "noaux_tc", "norm_topk_prob": true, "routed_scaling_factor": 2.5,
             "moe_layer_freq": 1, "first_k_dense_replace": 1, "max_position_embeddings": 256, "attention_bias": false,
             "rope_theta": 10000.0, "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 64,
              "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0}, "rms_norm_eps": 1e-6}
            """
        case .mamba3Siso, .mamba3Mimo, .gdn2:
            recurrentConfig
        default:
            fastWeightConfig
        }
    }

    var seed: UInt64 {
        Self.familySeeds[self] ?? 0
    }

    private static let familySeeds: [TinyModel.Family: UInt64] = [
        .qwen3Small: 56, .qwen3: 36, .qwen3HeadDim32: 36, .qwen35: 79, .qwen35Moe: 79,
        .qwen3Next: 642, .gemma3: 103, .gemma4: 228, .gemma4Moe: 228, .qwen3Moe: 228,
        .sigmoidMoe: 228, .gemma4Unified: 274, .gemma4KEqV: 645, .inkling2Loop: 650,
        .nemotronH: 30, .lfm2: 97, .lfm2Moe: 83, .prism: 7, .paddleOCR: 400,
        .deepSeekOCR: 401, .deepSeekOCR2: 413, .flint: 518, .ricursive: 520,
        .museGlimmer: 533, .museGlimmerText: 533, .ltxVideo: 584, .kandinskyCrossDiT: 585,
        .moeVideoDiT: 586, .bailingHybrid: 672, .ouro: 820, .llada: 819, .huginn: 125,
        .rwkv7: 825, .xlstm: 843, .mhc: 844, .mamba3Siso: 8471, .mamba3Mimo: 8472,
        .gdn2: 8473, .tttLinear: 8481, .tttMlp: 8482, .titans: 8483, .lact: 8484,
        .plt: 8201, .deepseekV3: 893,
    ]
}

extension TinyModel.Family {
    func recurrentModel(_ config: Data) throws -> (any LanguageModel)? {
        switch self {
        case .xlstm: return try XLSTMModel(configurationData: config)
        case .mhc: return try HyperConnectionModel(configurationData: config)
        case .mamba3Siso, .mamba3Mimo: return try Mamba3Model(configurationData: config)
        case .gdn2: return try GatedDeltaNet2Model(configurationData: config)
        default: return nil
        }
    }

    var recurrentConfig: String {
        switch self {
        case .mamba3Siso: """
            {"model_type": "mamba3", "vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 3,
             "expand": 2, "d_state": 16, "head_dim": 16, "n_groups": 2, "mimo_rank": 1,
             "chunk_size": 8, "heavy_tail": true, "use_gated_mlp": true, "intermediate_size": 128,
             "rms_norm_eps": 1e-5, "tie_word_embeddings": false}
            """
        case .mamba3Mimo: """
            {"model_type": "mamba3", "vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 3,
             "expand": 2, "d_state": 16, "head_dim": 16, "n_groups": 2, "mimo_rank": 4,
             "chunk_size": 8, "heavy_tail": true, "use_gated_mlp": true, "intermediate_size": 128,
             "rms_norm_eps": 1e-5, "tie_word_embeddings": false}
            """
        case .gdn2: """
            {"model_type": "gdn2", "vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 4,
             "intermediate_size": 128, "num_heads": 2, "num_v_heads": 4, "head_dim": 32, "head_v_dim": 32,
             "allow_neg_eigval": true, "beta_guard": true, "use_short_conv": true, "conv_kernel_size": 4,
             "chunk_size": 8, "attention_layer_indices": [3], "num_attention_heads": 4,
             "num_key_value_heads": 2, "attention_head_dim": 16, "rms_norm_eps": 1e-5,
             "tie_word_embeddings": false}
            """
        default:
            preconditionFailure("Unexpected TinyModel.Family in recurrentConfig: \(self)")
        }
    }
}

extension TinyModel.Family {
    func fastWeightModel(_ config: Data) throws -> (any LanguageModel)? {
        switch self {
        case .tttLinear, .tttMlp: return try TTTModel(configurationData: config)
        case .titans: return try TitansModel(configurationData: config)
        case .lact: return try LaCTModel(configurationData: config)
        default: return nil
        }
    }

    var fastWeightConfig: String {
        switch self {
        case .tttLinear: """
            {"model_type": "ttt", "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128,
             "num_hidden_layers": 2, "num_attention_heads": 4, "head_dim": 16, "ttt_layer_type": "linear",
             "mini_batch_size": 4, "ttt_base_lr": 0.5, "conv_kernel": 4, "use_gate": true,
             "rms_norm_eps": 1e-6, "ttt_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": false}
            """
        case .tttMlp: """
            {"model_type": "ttt", "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128,
             "num_hidden_layers": 2, "num_attention_heads": 4, "head_dim": 16, "ttt_layer_type": "mlp",
             "mini_batch_size": 4, "ttt_base_lr": 0.5, "conv_kernel": 4, "use_gate": true,
             "ttt_mlp_expansion": 2, "rms_norm_eps": 1e-6, "ttt_norm_eps": 1e-6, "rope_theta": 10000.0,
             "tie_word_embeddings": false}
            """
        case .titans: """
            {"model_type": "titans", "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128,
             "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
             "titans_variant": "mag", "memory_depth": 2, "memory_hidden_multiplier": 2,
             "num_persistent_tokens": 2, "chunk_size": 4, "window_size": \(TinyModel.slidingWindow), "conv_kernel": 4,
             "base_lr": 0.1, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": false}
            """
        case .lact: """
            {"model_type": "lact_swiglu", "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128,
             "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
             "num_fw_heads": 4, "fw_head_dim": 16, "inter_multi": 1.0, "lact_chunk_size": 4,
             "window_size": \(TinyModel.slidingWindow), "objective": "dot_product", "update_rule": "muon",
             "memory_mode": "swiglu", "use_weight_norm": true, "muon_steps": 5, "base_lr": 0.05,
             "conv_kernel": 4, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": false}
            """
        default:
            preconditionFailure("Unexpected TinyModel.Family in fastWeightConfig: \(self)")
        }
    }
}

extension TinyModel {
    static func seedSpecialParameters(_ model: any LanguageModel, family: Family) {
        let suffixes: [String]
        let scale: Float
        switch family {
        case .sigmoidMoe:
            (suffixes, scale) = ([".gate.weight", ".e_score_correction_bias"], 1.0)
        case .flint:
            (suffixes, scale) = ([".gate.weight", ".expert_bias", ".softmax_offset"], 0.25)
        case .inkling2Loop, .deepseekV3:
            (suffixes, scale) = ([".query", ".proj", ".static_weight", "conv1d.weight", ".gate.weight", ".e_score_correction_bias"], 0.25)
        case .ricursive:
            (suffixes, scale) = ([".value_res_lambda"], 0.5)
        case .museGlimmer, .museGlimmerText:
            (suffixes, scale) = (["_layernorm.weight"], 0.2)
        case .bailingHybrid:
            (suffixes, scale) = ([".gate.weight", ".gate.expert_bias", ".A_log", ".dt_bias"], 0.25)
        case .huginn:
            (suffixes, scale) = ([".qk_bias"], 0.25)
        case .rwkv7:
            (suffixes, scale) = ([".x_r", ".x_w", ".x_k", ".x_v", ".x_a", ".x_g", ".k_k", ".k_a", ".r_k", ".lora_b.bias"], 0.25)
        case .mhc:
            (suffixes, scale) = ([".phi_weight", ".bias_pre", ".bias_post", ".bias_res"], 0.25)
        case .mamba3Siso, .mamba3Mimo:
            (suffixes, scale) = ([".dt_bias", ".A_log", ".D", ".B_bias", ".C_bias", ".mimo_x", ".mimo_z", ".mimo_o"], 0.25)
        case .gdn2:
            (suffixes, scale) = ([".A_log", ".dt_bias"], 0.25)
        case .tttLinear, .tttMlp, .titans, .lact:
            (suffixes, scale) = ([".b1_init", ".b2_init", ".ttt_norm_bias", ".ttt_lr_gate.bias", ".alpha_proj.bias", ".eta_proj.bias", ".theta_proj.bias", ".lr_proj.bias"], 0.2)
        default:
            return
        }
        let matching = model.parameters().flattened().filter { key, _ in suffixes.contains { key.hasSuffix($0) } }
        let values = matching.map { ($0.0, scale == 1 ? MLXRandom.normal($0.1.shape) : MLXRandom.normal($0.1.shape) * scale) }
        model.update(parameters: ModuleParameters.unflattened(values))
    }
}
