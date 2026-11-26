import Foundation

struct ParakeetConfiguration: Decodable, Sendable {
    struct Preprocessor: Decodable, Sendable {
        let sampleRate: Int
        let normalize: String
        let windowSize: Double
        let windowStride: Double
        let window: String
        let features: Int
        let fftSize: Int
        let log: Bool?
        let frameSplicing: Int?
        let preemph: Float?

        enum CodingKeys: String, CodingKey {
            case sampleRate = "sample_rate", normalize, windowSize = "window_size", windowStride = "window_stride", window
            case features, fftSize = "n_fft", log, frameSplicing = "frame_splicing", preemph
        }

        var windowLength: Int { Int(windowSize * Double(sampleRate)) }
        var hopLength: Int { Int(windowStride * Double(sampleRate)) }
        var preemphasis: Float { preemph ?? 0.97 }
        var normalizesPerFeature: Bool { normalize == "per_feature" }

        var problems: [String] {
            let checks: [(Bool, String)] = [
                (sampleRate == 16_000, "preprocessor.sample_rate must be 16000"),
                (["per_feature", "all_features"].contains(normalize), "preprocessor.normalize must be per_feature or all_features"),
                (window == "hann", "preprocessor.window must be hann"),
                (features > 0 && hopLength > 0 && windowLength > 0, "preprocessor sizes must be positive"),
                (fftSize >= windowLength, "preprocessor.n_fft must cover the window"),
                (log ?? true, "preprocessor.log must be true"),
                ((frameSplicing ?? 1) == 1, "preprocessor.frame_splicing must be 1"),
            ]
            return checks.filter { !$0.0 }.map(\.1)
        }
    }

    struct Conformer: Decodable, Sendable {
        let featuresIn: Int
        let layers: Int
        let modelSize: Int
        let heads: Int
        let feedForwardExpansion: Int
        let subsampling: String
        let subsamplingFactor: Int
        let subsamplingChannels: Int
        let selfAttention: String
        let convKernelSize: Int
        let convNorm: String
        let useBias: Bool?
        let xscaling: Bool?
        let causalDownsampling: Bool?

        enum CodingKeys: String, CodingKey {
            case featuresIn = "feat_in", layers = "n_layers", modelSize = "d_model", heads = "n_heads"
            case feedForwardExpansion = "ff_expansion_factor", subsampling, subsamplingFactor = "subsampling_factor"
            case subsamplingChannels = "subsampling_conv_channels", selfAttention = "self_attention_model"
            case convKernelSize = "conv_kernel_size", convNorm = "conv_norm_type", useBias = "use_bias", xscaling
            case causalDownsampling = "causal_downsampling"
        }

        var bias: Bool { useBias ?? true }
        var subsamplingStages: Int { subsamplingFactor.trailingZeroBitCount }

        var problems: [String] {
            let checks: [(Bool, String)] = [
                (subsampling == "dw_striding" && !(causalDownsampling ?? false), "encoder.subsampling must be non-causal dw_striding"),
                (subsamplingFactor >= 2 && subsamplingFactor.nonzeroBitCount == 1, "encoder.subsampling_factor must be a power of two"),
                (selfAttention == "rel_pos", "encoder.self_attention_model must be rel_pos"),
                (convNorm == "batch_norm", "encoder.conv_norm_type must be batch_norm"),
                (min(featuresIn, layers, heads, feedForwardExpansion, subsamplingChannels) > 0, "encoder sizes must be positive"),
                (heads > 0 && modelSize % heads == 0 && modelSize % 2 == 0, "encoder.d_model must be even and divisible by n_heads"),
                (convKernelSize % 2 == 1, "encoder.conv_kernel_size must be odd"),
            ]
            return checks.filter { !$0.0 }.map(\.1)
        }
    }

    struct PredictionNetwork: Decodable, Sendable {
        let hidden: Int
        let layers: Int

        enum CodingKeys: String, CodingKey { case hidden = "pred_hidden", layers = "pred_rnn_layers" }
    }

    struct Predictor: Decodable, Sendable {
        let blankAsPad: Bool
        let vocabularySize: Int
        let prediction: PredictionNetwork

        enum CodingKeys: String, CodingKey { case blankAsPad = "blank_as_pad", vocabularySize = "vocab_size", prediction = "prednet" }
    }

    struct JointNetwork: Decodable, Sendable {
        let hidden: Int
        let activation: String
        let encoderHidden: Int
        let predictionHidden: Int

        enum CodingKeys: String, CodingKey {
            case hidden = "joint_hidden", activation, encoderHidden = "encoder_hidden", predictionHidden = "pred_hidden"
        }
    }

    struct Joint: Decodable, Sendable {
        let classes: Int
        let extraOutputs: Int
        let vocabulary: [String]
        let network: JointNetwork

        enum CodingKeys: String, CodingKey { case classes = "num_classes", extraOutputs = "num_extra_outputs", vocabulary, network = "jointnet" }
    }

    struct Greedy: Decodable, Sendable {
        let maxSymbols: Int?

        enum CodingKeys: String, CodingKey { case maxSymbols = "max_symbols" }
    }

    struct Decoding: Decodable, Sendable {
        let modelType: String
        let durations: [Int]
        let greedy: Greedy?

        enum CodingKeys: String, CodingKey { case modelType = "model_type", durations, greedy }
    }

    let preprocessor: Preprocessor
    let encoder: Conformer
    let decoder: Predictor
    let joint: Joint
    let decoding: Decoding

    var blank: Int { decoder.vocabularySize }
    var maxSymbols: Int { decoding.greedy?.maxSymbols ?? 10 }
    var windowSamples: Int { ParakeetModel.windowSeconds * preprocessor.sampleRate }

    var problems: [String] {
        let durations = decoding.durations
        let checks: [(Bool, String)] = [
            (encoder.featuresIn == preprocessor.features, "encoder.feat_in must equal preprocessor.features"),
            (decoder.blankAsPad, "decoder.blank_as_pad must be true"),
            (min(decoder.vocabularySize, decoder.prediction.hidden, decoder.prediction.layers, joint.network.hidden) > 0, "decoder and joint sizes must be positive"),
            (joint.classes == decoder.vocabularySize && joint.vocabulary.count == joint.classes, "joint.num_classes and vocabulary must match decoder.vocab_size"),
            (joint.network.encoderHidden == encoder.modelSize, "joint.jointnet.encoder_hidden must equal encoder.d_model"),
            (joint.network.predictionHidden == decoder.prediction.hidden, "joint.jointnet.pred_hidden must equal decoder.prednet.pred_hidden"),
            (["relu", "sigmoid", "tanh"].contains(joint.network.activation), "joint.jointnet.activation must be relu, sigmoid or tanh"),
            (decoding.modelType == "tdt" && joint.extraOutputs == durations.count, "decoding must be tdt with one joint output per duration"),
            (durations.allSatisfy { $0 >= 0 } && (durations.max() ?? 0) > 0 && maxSymbols > 0, "TDT durations must be non-negative with a positive one, and max_symbols positive"),
        ]
        return preprocessor.problems + encoder.problems + checks.filter { !$0.0 }.map(\.1)
    }

    static func decode(_ data: Data) throws -> ParakeetConfiguration {
        let configuration = try JSONDecoder().decode(ParakeetConfiguration.self, from: data)
        let problems = configuration.problems
        guard problems.isEmpty else {
            throw DecodingError.dataCorrupted(.init(codingPath: [], debugDescription: "parakeet: " + problems.joined(separator: "; ")))
        }
        return configuration
    }

    static func describes(_ config: [String: Any]) -> Bool {
        let target = config["target"] as? String ?? ""
        let defaults = config["model_defaults"] as? [String: Any]
        return config["model_type"] == nil && target.hasSuffix(".EncDecRNNTBPEModel") && defaults?["tdt_durations"] != nil
    }
}
