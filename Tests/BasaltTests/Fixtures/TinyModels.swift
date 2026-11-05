import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXLLM
@testable import BasaltInference

enum TinyModel {
    enum Family: String, CaseIterable, Sendable {
        case qwen3Small, qwen3, qwen3HeadDim32, qwen35, qwen35Moe, gemma3, gemma4, gemma4Moe, qwen3Moe, sigmoidMoe, nemotronH, lfm2, lfm2Moe, prism

        func model(_ config: Data) throws -> any LanguageModel {
            let decoder = JSONDecoder.json5()
            return switch self {
            case .qwen3Small, .qwen3, .qwen3HeadDim32: Qwen3Model(try decoder.decode(Qwen3Configuration.self, from: config))
            case .qwen35, .qwen35Moe: Qwen35TextModel(try decoder.decode(Qwen35TextConfiguration.self, from: config))
            case .gemma3: Gemma3TextModel(try decoder.decode(Gemma3TextConfiguration.self, from: config))
            case .gemma4, .gemma4Moe: Gemma4TextModel(try decoder.decode(Gemma4TextConfiguration.self, from: config))
            case .qwen3Moe: Qwen3MoEModel(try decoder.decode(Qwen3MoEConfiguration.self, from: config))
            case .sigmoidMoe, .nemotronH: NemotronHModel(try decoder.decode(NemotronHConfiguration.self, from: config))
            case .lfm2: LFM2Model(try decoder.decode(LFM2Configuration.self, from: config))
            case .lfm2Moe: LFM2MoEModel(try decoder.decode(LFM2MoEConfiguration.self, from: config))
            case .prism: MLXLLM.Qwen35Model(try decoder.decode(MLXLLM.Qwen35Configuration.self, from: config))
            }
        }

        var seed: UInt64 {
            switch self {
            case .qwen3Small: 56
            case .qwen3, .qwen3HeadDim32: 36
            case .qwen35, .qwen35Moe: 79
            case .gemma3: 103
            case .gemma4, .gemma4Moe, .qwen3Moe, .sigmoidMoe: 228
            case .nemotronH: 30
            case .lfm2: 97
            case .lfm2Moe: 83
            case .prism: 7
            }
        }

        var weightScale: Float {
            switch self {
            case .qwen3Small, .gemma3: 4
            case .qwen3, .qwen3HeadDim32, .gemma4, .gemma4Moe, .qwen35, .qwen35Moe, .qwen3Moe, .sigmoidMoe, .nemotronH, .lfm2, .lfm2Moe, .prism: 1
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
        let engine = MLXInferenceEngine(promptCacheBytes: promptCacheBytes, repetitionPenalty: repetitionPenalty)
        await engine.attach(try container(family, processor: processor))
        return engine
    }

    static func tokens(_ prompt: String) -> [Int] {
        ByteTokenizer().encode(text: prompt, addSpecialTokens: false) + [ByteTokenizer.endOfUserTurn]
    }
}
