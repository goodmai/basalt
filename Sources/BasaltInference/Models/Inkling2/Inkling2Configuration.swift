import Foundation

struct Inkling2Configuration: Decodable, Sendable, Equatable {
    struct Loop: Decodable, Sendable, Equatable {
        var coreStart: Int
        var coreEnd: Int
        var numPasses: Int
        var gateRank: Int
        var isVarloop = false

        enum CodingKeys: String, CodingKey {
            case coreStart = "core_start", coreEnd = "core_end", numPasses = "num_passes", gateRank = "gate_rank", isVarloop = "is_varloop"
        }
    }

    struct Invalid: Error, CustomStringConvertible {
        let description: String
    }

    var modelType: String
    var hiddenSize: Int
    var layers: Int
    var vocabSize: Int
    var unpaddedVocabSize: Int
    var heads: Int
    var kvHeads: Int
    var headDim: Int
    var relativeDim: Int
    var relativeExtent: Int
    var logScalingFloor: Int
    var logScalingAlpha: Float
    var rmsNormEps: Float
    var qkClip: Float
    var relativeClip: Float
    var localLayers: [Int]
    var shortConvKernel: Int
    var finalLogitSoftcap: Float
    var slidingWindow: Int
    var layersPerResidualBlock: Int
    var experts: Int
    var expertsPerToken: Int
    var sharedExperts: Int
    var expertWidth: Int
    var gateSoftcap: Float
    var upSoftcap: Float
    var routeScale: Float
    var loop: Loop?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", hiddenSize = "hidden_size", layers = "num_hidden_layers", vocabSize = "vocab_size"
        case unpaddedVocabSize = "unpadded_vocab_size", heads = "num_attention_heads", kvHeads = "num_key_value_heads", headDim = "head_dim"
        case relativeDim = "d_rel", relativeExtent = "rel_extent", logScalingFloor = "log_scaling_n_floor", logScalingAlpha = "log_scaling_alpha"
        case rmsNormEps = "rms_norm_eps", qkClip = "qk_l2_clip_threshold", relativeClip = "rel_l2_clip_threshold", localLayers = "local_layer_ids"
        case shortConvKernel = "sconv_kernel_size", finalLogitSoftcap = "final_logit_softcapping", slidingWindow = "sliding_window_size"
        case layersPerResidualBlock = "n_layers_per_attn_res_block", experts = "n_routed_experts", expertsPerToken = "num_experts_per_tok"
        case sharedExperts = "n_shared_experts", expertWidth = "intermediate_size", gateSoftcap = "mlp_gate_softcap", upSoftcap = "mlp_up_softcap"
        case routeScale = "route_scale", loop = "loop_config"
    }

    private enum Root: String, CodingKey { case textConfig = "text_config" }

    init(from decoder: Decoder) throws {
        let root = try decoder.container(keyedBy: Root.self)
        let c = root.contains(.textConfig) ? try root.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig) : try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        (hiddenSize, layers, vocabSize) = (try c.decode(Int.self, forKey: .hiddenSize), try c.decode(Int.self, forKey: .layers), try c.decode(Int.self, forKey: .vocabSize))
        unpaddedVocabSize = try c.decodeIfPresent(Int.self, forKey: .unpaddedVocabSize) ?? vocabSize
        (heads, kvHeads, headDim) = (try c.decode(Int.self, forKey: .heads), try c.decode(Int.self, forKey: .kvHeads), try c.decode(Int.self, forKey: .headDim))
        (relativeDim, relativeExtent) = (try c.decode(Int.self, forKey: .relativeDim), try c.decode(Int.self, forKey: .relativeExtent))
        logScalingFloor = try c.decodeIfPresent(Int.self, forKey: .logScalingFloor) ?? 8192
        logScalingAlpha = try c.decodeIfPresent(Float.self, forKey: .logScalingAlpha) ?? 0.1
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        qkClip = try c.decode(Float.self, forKey: .qkClip)
        relativeClip = try c.decode(Float.self, forKey: .relativeClip)
        localLayers = try c.decodeIfPresent([Int].self, forKey: .localLayers) ?? []
        shortConvKernel = try c.decodeIfPresent(Int.self, forKey: .shortConvKernel) ?? 4
        finalLogitSoftcap = try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcap) ?? 30
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        layersPerResidualBlock = try c.decode(Int.self, forKey: .layersPerResidualBlock)
        (experts, expertsPerToken) = (try c.decode(Int.self, forKey: .experts), try c.decode(Int.self, forKey: .expertsPerToken))
        sharedExperts = try c.decodeIfPresent(Int.self, forKey: .sharedExperts) ?? 0
        expertWidth = try c.decode(Int.self, forKey: .expertWidth)
        (gateSoftcap, upSoftcap) = (try c.decode(Float.self, forKey: .gateSoftcap), try c.decode(Float.self, forKey: .upSoftcap))
        routeScale = try c.decodeIfPresent(Float.self, forKey: .routeScale) ?? 1
        loop = try c.decodeIfPresent(Loop.self, forKey: .loop)
    }

    var globalLayers: [Int] {
        let local = Set(localLayers)
        return (0 ..< layers).filter { !local.contains($0) }
    }

    var visitOrder: [Int] {
        guard let loop else { return Array(0 ..< layers) }
        let core = Array(loop.coreStart ..< loop.coreEnd)
        return Array(0 ..< loop.coreStart) + Array(repeating: core, count: loop.numPasses).flatMap { $0 } + Array(loop.coreEnd ..< layers)
    }

    func validated() throws(Invalid) -> Inkling2Configuration {
        guard layers > 0, heads > 0, kvHeads > 0, heads.isMultiple(of: kvHeads), headDim > 0 else { throw Invalid(description: "attention shape \(heads)/\(kvHeads)×\(headDim)") }
        guard localLayers.allSatisfy({ (0 ..< layers).contains($0) }) else { throw Invalid(description: "local_layer_ids outside 0..<\(layers)") }
        guard layersPerResidualBlock > 0, layers.isMultiple(of: layersPerResidualBlock) else { throw Invalid(description: "\(layers) layers in blocks of \(layersPerResidualBlock)") }
        guard expertsPerToken > 0, expertsPerToken <= experts else { throw Invalid(description: "top-\(expertsPerToken) of \(experts) experts") }
        guard unpaddedVocabSize > 0, unpaddedVocabSize <= vocabSize else { throw Invalid(description: "unpadded vocab \(unpaddedVocabSize) over \(vocabSize)") }
        guard relativeDim > 0, relativeExtent > 0, slidingWindow > 0, shortConvKernel > 1 else { throw Invalid(description: "relative, window or conv sizes") }
        if let loop {
            guard loop.coreStart >= 0, loop.coreStart < loop.coreEnd, loop.coreEnd <= layers, loop.numPasses >= 1, loop.gateRank > 0 else {
                throw Invalid(description: "loop core \(loop.coreStart)..<\(loop.coreEnd) × \(loop.numPasses) passes in \(layers) layers")
            }
        }
        return self
    }
}
