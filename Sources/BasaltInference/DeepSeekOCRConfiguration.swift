import BasaltCore
import Foundation

struct DeepSeekOCRConfiguration: Codable, Sendable {
    struct Text: Codable, Sendable {
        let hiddenSize: Int
        let layers: Int
        let intermediateSize: Int
        let moeIntermediateSize: Int
        let heads: Int
        let kvHeads: Int
        let vocabularySize: Int
        let routedExperts: Int?
        let sharedExperts: Int?
        let expertsPerToken: Int
        let firstDenseLayers: Int
        let qkNopeHeadDim: Int?
        let topkMethod: String?
        let rmsNormEps: Float?
        let ropeTheta: Float?
        let routedScalingFactor: Float?
        let scoringFunction: String?
        let normalizesTopK: Bool?
        let moeLayerFrequency: Int?
        let groups: Int?
        let topKGroups: Int?

        var eps: Float { rmsNormEps ?? 1e-6 }
        var theta: Float { ropeTheta ?? 10_000 }
        var headDim: Int { hiddenSize / heads }

        func usesExperts(_ layer: Int) -> Bool { routedExperts != nil && layer >= firstDenseLayers }

        var unsupported: [String] {
            [
                (qkNopeHeadDim ?? 0) != 0 ? "multi-head latent attention (qk_nope_head_dim \(qkNopeHeadDim ?? 0))" : nil,
                (topkMethod ?? "greedy") != "greedy" ? "expert routing \(topkMethod ?? "")" : nil,
                (scoringFunction ?? "softmax") != "softmax" ? "expert scoring \(scoringFunction ?? "")" : nil,
                normalizesTopK == true ? "normalized top-k expert weights" : nil,
                (moeLayerFrequency ?? 1) != 1 ? "experts every \(moeLayerFrequency ?? 1) layers" : nil,
                max(groups ?? 1, topKGroups ?? 1) > 1 ? "grouped expert routing (n_group \(groups ?? 1), topk_group \(topKGroups ?? 1))" : nil,
            ].compactMap { $0 }
        }

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case layers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case moeIntermediateSize = "moe_intermediate_size"
            case heads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case vocabularySize = "vocab_size"
            case routedExperts = "n_routed_experts"
            case sharedExperts = "n_shared_experts"
            case expertsPerToken = "num_experts_per_tok"
            case firstDenseLayers = "first_k_dense_replace"
            case qkNopeHeadDim = "qk_nope_head_dim"
            case topkMethod = "topk_method"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case routedScalingFactor = "routed_scaling_factor"
            case scoringFunction = "scoring_func"
            case normalizesTopK = "norm_topk_prob"
            case moeLayerFrequency = "moe_layer_freq"
            case groups = "n_group"
            case topKGroups = "topk_group"
        }
    }

    struct SAM: Codable, Sendable {
        let width: Int
        let layers: Int
        let heads: Int
        let globalAttentionLayers: [Int]
        let downsampleChannels: [Int]
        let windowSize: Int?
        let imageSize: Int?

        var window: Int { windowSize ?? 14 }
        var patchSize: Int { 16 }
        var neckChannels: Int { 256 }

        enum CodingKeys: String, CodingKey {
            case width, layers, heads
            case globalAttentionLayers = "global_attn_indexes"
            case downsampleChannels = "downsample_channels"
            case windowSize = "window_size"
            case imageSize = "image_size"
        }
    }

    struct CLIP: Codable, Sendable {
        let width: Int
        let layers: Int
        let heads: Int
        let patchSize: Int
        let imageSize: Int

        enum CodingKeys: String, CodingKey {
            case width, layers, heads
            case patchSize = "patch_size"
            case imageSize = "image_size"
        }
    }

    struct Vision: Codable, Sendable {
        struct Towers: Codable, Sendable {
            let sam: SAM
            let clip: CLIP

            enum CodingKeys: String, CodingKey {
                case sam = "sam_vit_b"
                case clip = "clip-l-14-224"
            }
        }

        let imageSize: Int
        let width: Towers

        enum CodingKeys: String, CodingKey {
            case imageSize = "image_size"
            case width
        }
    }

    struct Projector: Codable, Sendable {
        let inputDim: Int
        let embedDim: Int
        let type: String

        enum CodingKeys: String, CodingKey {
            case inputDim = "input_dim"
            case embedDim = "n_embed"
            case type = "projector_type"
        }
    }

    let text: Text
    let vision: Vision
    let projector: Projector
    let imageTokenId: Int?

    var imageToken: Int { imageTokenId ?? 128_815 }
    var sam: SAM { vision.width.sam }
    var clip: CLIP { vision.width.clip }
    var samImageSize: Int { sam.imageSize ?? vision.imageSize }

    enum CodingKeys: String, CodingKey {
        case text = "language_config"
        case vision = "vision_config"
        case projector = "projector_config"
        case imageTokenId = "image_token_index"
    }

    func validated() throws(BasaltError) -> Self {
        let reasons = text.unsupported + [
            projector.type != "linear" ? "projector \(projector.type)" : nil,
            sam.downsampleChannels.count != 2 ? "\(sam.downsampleChannels.count) SAM downsample stages" : nil,
            clip.width != sam.downsampleChannels.last ? "SAM features of \(sam.downsampleChannels.last ?? 0) channels into CLIP width \(clip.width)" : nil,
            projector.inputDim != 2 * clip.width ? "a projector input of \(projector.inputDim)" : nil,
        ].compactMap { $0 }
        guard reasons.isEmpty else {
            throw .modelConfigurationInvalid(path: "config.json", reason: "DeepSeek-OCR: this port does not run \(reasons.joined(separator: ", "))")
        }
        return self
    }
}
