import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXLLM
import MLXVLM
@testable import BasaltInference

enum TinyModel {
    enum Family: String, CaseIterable, Sendable {
        case qwen3Small, qwen3, qwen3HeadDim32, qwen35, qwen35Moe, qwen3Next, gemma3, gemma4, gemma4Moe, gemma4KEqV, gemma4Unified, inkling2Loop, qwen3Moe, sigmoidMoe, nemotronH, lfm2, lfm2Moe, prism, paddleOCR, deepSeekOCR, deepSeekOCR2, flint, ricursive, museGlimmer, museGlimmerText, ltxVideo, kandinskyCrossDiT, moeVideoDiT

        func model(_ config: Data) throws -> any LanguageModel {
            let decoder = JSONDecoder.json5()
            return switch self {
            case .qwen3Small, .qwen3, .qwen3HeadDim32: Qwen3Model(try decoder.decode(Qwen3Configuration.self, from: config))
            case .qwen35, .qwen35Moe: Qwen35TextModel(try decoder.decode(Qwen35TextConfiguration.self, from: config))
            case .gemma3: Gemma3TextModel(try decoder.decode(Gemma3TextConfiguration.self, from: config))
            case .gemma4, .gemma4Moe, .gemma4KEqV: Gemma4TextModel(try decoder.decode(Gemma4TextConfiguration.self, from: config))
            case .qwen3Moe: Qwen3MoEModel(try decoder.decode(Qwen3MoEConfiguration.self, from: config))
            case .sigmoidMoe, .nemotronH: NemotronHModel(try decoder.decode(NemotronHConfiguration.self, from: config))
            case .lfm2: LFM2Model(try decoder.decode(LFM2Configuration.self, from: config))
            case .lfm2Moe: LFM2MoEModel(try decoder.decode(LFM2MoEConfiguration.self, from: config))
            case .prism: MLXLLM.Qwen35Model(try decoder.decode(MLXLLM.Qwen35Configuration.self, from: config))
            case .flint: FlintModel(try decoder.decode(FlintConfiguration.self, from: config).validated())
            case .ricursive: RicursiveModel(try decoder.decode(RicursiveConfiguration.self, from: config).validated())
            case .museGlimmerText: MuseGlimmerTextModel(try MuseGlimmerTextConfiguration.decodeFlexible(from: config))
            case .ltxVideo, .kandinskyCrossDiT, .moeVideoDiT: try videoModel(config, decoder: decoder)
            case .gemma4Unified, .paddleOCR, .deepSeekOCR, .deepSeekOCR2, .museGlimmer, .inkling2Loop, .qwen3Next: try multimodalModel(config, decoder: decoder)
            }
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
            if self == .paddleOCR { return PaddleOCRVL(try decoder.decode(PaddleOCRVLConfiguration.self, from: config)) }
            if self == .deepSeekOCR { return DeepSeekOCR(try decoder.decode(DeepSeekOCRConfiguration.self, from: config).validated()) }
            if self == .museGlimmer { return MuseGlimmerVLM(try decoder.decode(MuseGlimmerConfiguration.self, from: config)) }
            return DeepSeekOCR2(try decoder.decode(DeepSeekOCR2Configuration.self, from: config).validated())
        }

        var seed: UInt64 {
            switch self {
            case .qwen3Small: 56
            case .qwen3, .qwen3HeadDim32: 36
            case .qwen35, .qwen35Moe: 79
            case .qwen3Next: 642
            case .gemma3: 103
            case .gemma4, .gemma4Moe, .qwen3Moe, .sigmoidMoe: 228
            case .gemma4Unified: 274
            case .gemma4KEqV: 645
            case .inkling2Loop: 650
            case .nemotronH: 30
            case .lfm2: 97
            case .lfm2Moe: 83
            case .prism: 7
            case .paddleOCR: 400
            case .deepSeekOCR: 401
            case .deepSeekOCR2: 413
            case .flint: 518
            case .ricursive: 520
            case .museGlimmer, .museGlimmerText: 533
            case .ltxVideo: 584
            case .kandinskyCrossDiT: 585
            case .moeVideoDiT: 586
            }
        }

        var weightScale: Float {
            switch self {
            case .qwen3Small, .gemma3: 4
            case .qwen3, .qwen3HeadDim32, .gemma4, .gemma4Moe, .gemma4KEqV, .gemma4Unified, .inkling2Loop, .qwen35, .qwen35Moe, .qwen3Next, .qwen3Moe, .sigmoidMoe, .nemotronH, .lfm2, .lfm2Moe, .prism, .paddleOCR, .deepSeekOCR, .deepSeekOCR2, .flint, .ricursive, .museGlimmer, .museGlimmerText, .ltxVideo, .kandinskyCrossDiT, .moeVideoDiT: 1
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
            case .flint: """
                {"model_type": "flint", "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                 "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 256,
                 "sliding_window": \(slidingWindow), "use_sliding_window": true,
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
                 "sliding_window": \(slidingWindow),
                 "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
                 "use_gated_attention": true, "gate_full_precision": true, "use_valres": true,
                 "ffn_clamp_threshold": 20.0, "embed_scale": 8.0, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
                 "tie_word_embeddings": false}
                """
            case .museGlimmerText: """
                {"model_type": "muse_glimmer_text", "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                 "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 256,
                 "sliding_window": \(slidingWindow), "qk_scale_factor": 3.87, "output_multiplier": 0.19611613,
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
            if family == .sigmoidMoe {
                let router = model.parameters().flattened().filter { $0.0.hasSuffix(".gate.weight") || $0.0.hasSuffix(".e_score_correction_bias") }
                model.update(parameters: ModuleParameters.unflattened(router.map { ($0.0, MLXRandom.normal($0.1.shape)) }))
            } else if family == .flint {
                let specialized = model.parameters().flattened().filter {
                    $0.0.hasSuffix(".gate.weight") || $0.0.hasSuffix(".expert_bias") || $0.0.hasSuffix(".softmax_offset")
                }
                model.update(parameters: ModuleParameters.unflattened(specialized.map { ($0.0, MLXRandom.normal($0.1.shape) * 0.25) }))
            } else if family == .inkling2Loop {
                let zeroed = model.parameters().flattened().filter { key, _ in
                    [".query", ".proj", ".static_weight", "conv1d.weight", ".gate.weight", ".e_score_correction_bias"].contains { key.hasSuffix($0) }
                }
                model.update(parameters: ModuleParameters.unflattened(zeroed.map { ($0.0, MLXRandom.normal($0.1.shape) * 0.25) }))
            } else if family == .ricursive {
                let lambdas = model.parameters().flattened().filter { $0.0.hasSuffix(".value_res_lambda") }
                model.update(parameters: ModuleParameters.unflattened(lambdas.map { ($0.0, MLXRandom.normal($0.1.shape) * 0.5) }))
            } else if family == .museGlimmer || family == .museGlimmerText {
                let centered = model.parameters().flattened().filter {
                    $0.0.hasSuffix("_layernorm.weight")
                }
                model.update(parameters: ModuleParameters.unflattened(centered.map { ($0.0, MLXRandom.normal($0.1.shape) * 0.2) }))
            }
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
