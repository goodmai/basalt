import Foundation
import MLXLLM
import MLXLMCommon

struct NemotronMelConfig: Codable, Sendable {
    var sampleRate: Int = 16_000
    var windowSize: Float = 0.025
    var windowStride: Float = 0.01
    var features: Int = 128
    var nFFT: Int = 512
    var preemphasis: Float = 0.97

    var winLength: Int { Int((Float(sampleRate) * windowSize).rounded()) }
    var hopLength: Int { Int((Float(sampleRate) * windowStride).rounded()) }

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case windowSize = "window_size"
        case windowStride = "window_stride"
        case features
        case nFFT = "n_fft"
        case preemphasis
    }

    init(
        sampleRate: Int = 16_000, windowSize: Float = 0.025, windowStride: Float = 0.01,
        features: Int = 128, nFFT: Int = 512, preemphasis: Float = 0.97
    ) {
        self.sampleRate = sampleRate
        self.windowSize = windowSize
        self.windowStride = windowStride
        self.features = features
        self.nFFT = nFFT
        self.preemphasis = preemphasis
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 16_000
        windowSize = try c.decodeIfPresent(Float.self, forKey: .windowSize) ?? 0.025
        windowStride = try c.decodeIfPresent(Float.self, forKey: .windowStride) ?? 0.01
        features = try c.decodeIfPresent(Int.self, forKey: .features) ?? 128
        nFFT = try c.decodeIfPresent(Int.self, forKey: .nFFT) ?? 512
        preemphasis = try c.decodeIfPresent(Float.self, forKey: .preemphasis) ?? 0.97
    }
}

struct NemotronEncoderConfig: Codable, Sendable {
    var featIn: Int = 128
    var dModel: Int = 1280
    var nHeads: Int = 10
    var nLayers: Int = 32
    var ffExpansion: Float = 4.0
    var subsamplingFactor: Int = 8
    var qkvBias: Bool = false
    var qkNorm: Bool = true
    var preBlockNorm: Bool = true
    var xscaling: Bool = false
    var ropeBase: Float = 10_000.0
    var rotaryFraction: Float = 0.5

    enum CodingKeys: String, CodingKey {
        case featIn = "feat_in"
        case dModel = "d_model"
        case nHeads = "n_heads"
        case nLayers = "n_layers"
        case ffExpansion = "ff_expansion"
        case subsamplingFactor = "subsampling_factor"
        case qkvBias = "qkv_bias"
        case qkNorm = "qk_norm"
        case preBlockNorm = "pre_block_norm"
        case xscaling
        case ropeBase = "rope_base"
        case rotaryFraction = "rotary_fraction"
    }

    init(
        dModel: Int = 1280, nHeads: Int = 10, nLayers: Int = 32,
        ffExpansion: Float = 4.0, qkNorm: Bool = true
    ) {
        self.dModel = dModel
        self.nHeads = nHeads
        self.nLayers = nLayers
        self.ffExpansion = ffExpansion
        self.qkNorm = qkNorm
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        featIn = try c.decodeIfPresent(Int.self, forKey: .featIn) ?? 128
        dModel = try c.decode(Int.self, forKey: .dModel)
        nHeads = try c.decode(Int.self, forKey: .nHeads)
        nLayers = try c.decode(Int.self, forKey: .nLayers)
        ffExpansion = try c.decodeIfPresent(Float.self, forKey: .ffExpansion) ?? 4.0
        subsamplingFactor = try c.decodeIfPresent(Int.self, forKey: .subsamplingFactor) ?? 8
        qkvBias = try c.decodeIfPresent(Bool.self, forKey: .qkvBias) ?? false
        qkNorm = try c.decodeIfPresent(Bool.self, forKey: .qkNorm) ?? false
        preBlockNorm = try c.decodeIfPresent(Bool.self, forKey: .preBlockNorm) ?? true
        xscaling = try c.decodeIfPresent(Bool.self, forKey: .xscaling) ?? false
        ropeBase = try c.decodeIfPresent(Float.self, forKey: .ropeBase) ?? 10_000.0
        rotaryFraction = try c.decodeIfPresent(Float.self, forKey: .rotaryFraction) ?? 0.5
    }
}

struct NemotronSortformerModulesConfig: Codable, Sendable {
    var numSpks: Int = 8
    var fcDModel: Int = 512
    var tfDModel: Int = 192
    var subsamplingFactor: Int = 8
    var spkcacheLen: Int = 188
    var fifoLen: Int = 40
    var spkcacheUpdatePeriod: Int = 300

    enum CodingKeys: String, CodingKey {
        case numSpks = "num_spks"
        case fcDModel = "fc_d_model"
        case tfDModel = "tf_d_model"
        case subsamplingFactor = "subsampling_factor"
        case spkcacheLen = "spkcache_len"
        case fifoLen = "fifo_len"
        case spkcacheUpdatePeriod = "spkcache_update_period"
    }

    init(
        numSpks: Int = 8, fcDModel: Int = 512, tfDModel: Int = 192,
        subsamplingFactor: Int = 8, spkcacheLen: Int = 188, fifoLen: Int = 40
    ) {
        self.numSpks = numSpks
        self.fcDModel = fcDModel
        self.tfDModel = tfDModel
        self.subsamplingFactor = subsamplingFactor
        self.spkcacheLen = spkcacheLen
        self.fifoLen = fifoLen
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        numSpks = try c.decodeIfPresent(Int.self, forKey: .numSpks) ?? 8
        fcDModel = try c.decodeIfPresent(Int.self, forKey: .fcDModel) ?? 512
        tfDModel = try c.decodeIfPresent(Int.self, forKey: .tfDModel) ?? 192
        subsamplingFactor = try c.decodeIfPresent(Int.self, forKey: .subsamplingFactor) ?? 8
        spkcacheLen = try c.decodeIfPresent(Int.self, forKey: .spkcacheLen) ?? 188
        fifoLen = try c.decodeIfPresent(Int.self, forKey: .fifoLen) ?? 40
        spkcacheUpdatePeriod = try c.decodeIfPresent(Int.self, forKey: .spkcacheUpdatePeriod) ?? 300
    }
}

struct NemotronDiarizationConfig: Codable, Sendable {
    var maxNumOfSpks: Int = 8
    var preprocessor: NemotronMelConfig = NemotronMelConfig()
    var encoder: NemotronEncoderConfig
    var sortformerModules: NemotronSortformerModulesConfig

    enum CodingKeys: String, CodingKey {
        case maxNumOfSpks = "max_num_of_spks"
        case preprocessor
        case encoder
        case sortformerModules = "sortformer_modules"
    }

    static let defaultEncoder = NemotronEncoderConfig(
        dModel: 512, nHeads: 8, nLayers: 31, qkNorm: false
    )

    init(
        maxNumOfSpks: Int = 8,
        preprocessor: NemotronMelConfig = NemotronMelConfig(),
        encoder: NemotronEncoderConfig = Self.defaultEncoder,
        sortformerModules: NemotronSortformerModulesConfig = NemotronSortformerModulesConfig()
    ) {
        self.maxNumOfSpks = maxNumOfSpks
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.sortformerModules = sortformerModules
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        maxNumOfSpks = try c.decodeIfPresent(Int.self, forKey: .maxNumOfSpks) ?? 8
        preprocessor = try c.decodeIfPresent(NemotronMelConfig.self, forKey: .preprocessor) ?? NemotronMelConfig()
        encoder = try c.decode(NemotronEncoderConfig.self, forKey: .encoder)
        var modules = try c.decodeIfPresent(
            NemotronSortformerModulesConfig.self, forKey: .sortformerModules
        ) ?? NemotronSortformerModulesConfig()
        modules.subsamplingFactor = encoder.subsamplingFactor
        modules.fcDModel = encoder.dModel
        if modules.fifoLen == 0 {
            modules.fifoLen = 40
            modules.spkcacheLen = 188
            modules.spkcacheUpdatePeriod = 300
        }
        sortformerModules = modules
    }
}

struct NemotronPEEncoderConfig: Codable, Sendable {
    var asrEncoderCfg: NemotronEncoderConfig?
    var diarizationModelCfg: NemotronDiarizationConfig
    var speakerFeatureMode: String = "thresholded"
    var speakerActivityThreshold: Float = 0.5
    var spkKernelScale: Float = 1.0
    var onlineInferenceLength: Int = 500
    var chunkLeftContext: Int = 50
    var chunkRightContext: Int = 50

    enum CodingKeys: String, CodingKey {
        case asrEncoderCfg = "asr_encoder_cfg"
        case diarizationModelCfg = "diarization_model_cfg"
        case speakerFeatureMode = "speaker_feature_mode"
        case speakerActivityThreshold = "speaker_activity_threshold"
        case spkKernelScale = "spk_kernel_scale"
        case onlineInferenceLength = "online_inference_length"
        case chunkLeftContext = "chunk_left_context"
        case chunkRightContext = "chunk_right_context"
    }

    init(
        asrEncoderCfg: NemotronEncoderConfig? = nil,
        diarizationModelCfg: NemotronDiarizationConfig = NemotronDiarizationConfig()
    ) {
        self.asrEncoderCfg = asrEncoderCfg
        self.diarizationModelCfg = diarizationModelCfg
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        asrEncoderCfg = try c.decodeIfPresent(NemotronEncoderConfig.self, forKey: .asrEncoderCfg)
        diarizationModelCfg = try c.decode(NemotronDiarizationConfig.self, forKey: .diarizationModelCfg)
        speakerFeatureMode = try c.decodeIfPresent(String.self, forKey: .speakerFeatureMode) ?? "thresholded"
        speakerActivityThreshold = try c.decodeIfPresent(Float.self, forKey: .speakerActivityThreshold) ?? 0.5
        spkKernelScale = try c.decodeIfPresent(Float.self, forKey: .spkKernelScale) ?? 1.0
        onlineInferenceLength = try c.decodeIfPresent(Int.self, forKey: .onlineInferenceLength) ?? 500
        chunkLeftContext = try c.decodeIfPresent(Int.self, forKey: .chunkLeftContext) ?? 50
        chunkRightContext = try c.decodeIfPresent(Int.self, forKey: .chunkRightContext) ?? 50
    }
}

struct NemotronPerceptionSection: Codable, Sendable {
    var outputDim: Int = 2688
    var preprocessor: NemotronMelConfig = NemotronMelConfig()
    var encoder: NemotronEncoderConfig = NemotronEncoderConfig()

    enum CodingKeys: String, CodingKey {
        case outputDim = "output_dim"
        case preprocessor
        case encoder
    }
}

struct NemotronTranscribeConfiguration: Codable, Sendable {
    var modelType: String = "nemotron_transcribe"
    var audioLocatorTag: String = "<|audio|>"
    var padTokenId: Int = 0
    var eosTokenIds: [Int] = [11]
    var perception: NemotronPerceptionSection
    var peEncoderConfig: NemotronPEEncoderConfig
    var llmConfig: NemotronHConfiguration

    var asrConfig: NemotronEncoderConfig {
        peEncoderConfig.asrEncoderCfg ?? perception.encoder
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioLocatorTag = "audio_locator_tag"
        case padTokenId = "pad_token_id"
        case eosTokenId = "eos_token_id"
        case perception
        case peEncoderConfig = "pe_encoder_config"
        case llmConfig = "llm_config"
    }

    init(
        perception: NemotronPerceptionSection,
        peEncoderConfig: NemotronPEEncoderConfig,
        llmConfig: NemotronHConfiguration,
        audioLocatorTag: String = "<|audio|>",
        padTokenId: Int = 0,
        eosTokenIds: [Int] = [11]
    ) {
        self.perception = perception
        self.peEncoderConfig = peEncoderConfig
        self.llmConfig = llmConfig
        self.audioLocatorTag = audioLocatorTag
        self.padTokenId = padTokenId
        self.eosTokenIds = eosTokenIds
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "nemotron_transcribe"
        audioLocatorTag = try c.decodeIfPresent(String.self, forKey: .audioLocatorTag) ?? "<|audio|>"
        padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 0
        if let eosArray = try? c.decode([Int].self, forKey: .eosTokenId) {
            eosTokenIds = eosArray
        } else if let eosSingle = try? c.decode(Int.self, forKey: .eosTokenId) {
            eosTokenIds = [eosSingle]
        } else {
            eosTokenIds = [11]
        }
        perception = try c.decode(NemotronPerceptionSection.self, forKey: .perception)
        peEncoderConfig = try c.decode(NemotronPEEncoderConfig.self, forKey: .peEncoderConfig)
        llmConfig = try c.decode(NemotronHConfiguration.self, forKey: .llmConfig)
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(audioLocatorTag, forKey: .audioLocatorTag)
        try c.encode(padTokenId, forKey: .padTokenId)
        try c.encode(eosTokenIds, forKey: .eosTokenId)
        try c.encode(perception, forKey: .perception)
        try c.encode(peEncoderConfig, forKey: .peEncoderConfig)
        try c.encode(llmConfig, forKey: .llmConfig)
    }
}
