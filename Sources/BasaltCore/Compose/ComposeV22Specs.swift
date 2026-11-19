import Foundation

public struct ModelSpeculativeSpec: Sendable, Codable, Equatable {
    public static let validModes: Set<String> = ["dflash", "mtp", "lookup", "draft"]
    public static let validVerifiers: Set<String> = ["spectr", "residual-race", "greedy"]

    public var mode: String
    public var drafter: String?
    public var blockSize: Int?
    public var verify: String?

    public init(mode: String, drafter: String? = nil, blockSize: Int? = nil, verify: String? = nil) {
        self.mode = mode
        self.drafter = drafter
        self.blockSize = blockSize
        self.verify = verify
    }

    enum CodingKeys: String, CodingKey {
        case mode, drafter, verify
        case blockSize = "block_size"
    }
}

public struct DiffusionTextSpec: Sendable, Codable, Equatable {
    public var dualKVCache: Bool
    public var confidenceThreshold: Double?
    public var contrastiveAlpha: Double?

    public init(dualKVCache: Bool = true, confidenceThreshold: Double? = nil, contrastiveAlpha: Double? = nil) {
        self.dualKVCache = dualKVCache
        self.confidenceThreshold = confidenceThreshold
        self.contrastiveAlpha = contrastiveAlpha
    }

    enum CodingKeys: String, CodingKey {
        case dualKVCache = "dual_kv_cache"
        case confidenceThreshold = "confidence_threshold"
        case contrastiveAlpha = "contrastive_alpha"
    }
}

public struct ImagineUpscaleSpec: Sendable, Codable, Equatable {
    public var model: String
    public var steps: Int?
    public var targetResolution: Int?

    public init(model: String, steps: Int? = nil, targetResolution: Int? = nil) {
        self.model = model
        self.steps = steps
        self.targetResolution = targetResolution
    }

    enum CodingKeys: String, CodingKey {
        case model, steps
        case targetResolution = "target_resolution"
    }
}

public struct VoiceTurnDetectionSpec: Sendable, Codable, Equatable {
    public var model: String?
    public var diarization: String?

    public init(model: String? = nil, diarization: String? = nil) {
        self.model = model
        self.diarization = diarization
    }
}

public struct VoiceDuplexSpec: Sendable, Codable, Equatable {
    public static let validModes: Set<String> = ["sts", "pipeline"]

    public var mode: String
    public var stsModel: String?
    public var denoise: String?
    public var turnDetection: VoiceTurnDetectionSpec?
    public var ttsFallback: String?
    public var aligner: String?

    public init(
        mode: String = "sts",
        stsModel: String? = nil,
        denoise: String? = nil,
        turnDetection: VoiceTurnDetectionSpec? = nil,
        ttsFallback: String? = nil,
        aligner: String? = nil
    ) {
        self.mode = mode
        self.stsModel = stsModel
        self.denoise = denoise
        self.turnDetection = turnDetection
        self.ttsFallback = ttsFallback
        self.aligner = aligner
    }

    enum CodingKeys: String, CodingKey {
        case mode, denoise, aligner
        case stsModel = "sts_model"
        case turnDetection = "turn_detection"
        case ttsFallback = "tts_fallback"
    }
}

public struct MusicComposeSpec: Sendable, Codable, Equatable {
    public var model: String
    public var steps: Int?
    public var durationSeconds: Int?

    public init(model: String, steps: Int? = nil, durationSeconds: Int? = nil) {
        self.model = model
        self.steps = steps
        self.durationSeconds = durationSeconds
    }

    enum CodingKeys: String, CodingKey {
        case model, steps
        case durationSeconds = "duration_seconds"
    }
}

public struct VideoComposeSpec: Sendable, Codable, Equatable {
    public var model: String
    public var vaeTiling: Bool?
    public var staSparse: Bool?
    public var fps: Int?
    public var frames: Int?

    public init(model: String, vaeTiling: Bool? = nil, staSparse: Bool? = nil, fps: Int? = nil, frames: Int? = nil) {
        self.model = model
        self.vaeTiling = vaeTiling
        self.staSparse = staSparse
        self.fps = fps
        self.frames = frames
    }

    enum CodingKeys: String, CodingKey {
        case model, fps, frames
        case vaeTiling = "vae_tiling"
        case staSparse = "sta_sparse"
    }
}

public struct ComposeTrainSpec: Sendable, Codable, Equatable {
    public static let validMethods: Set<String> = ["lora", "phm", "oft", "adpo", "cadpo"]

    public var method: String
    public var segments: Int?
    public var emaDecay: Double?

    public init(method: String, segments: Int? = nil, emaDecay: Double? = nil) {
        self.method = method
        self.segments = segments
        self.emaDecay = emaDecay
    }

    enum CodingKeys: String, CodingKey {
        case method, segments
        case emaDecay = "ema_decay"
    }
}

public struct ComposeClusterSpec: Sendable, Codable, Equatable {
    public static let validTopologies: Set<String> = ["auto", "ddp-zero2", "pipeline-zbh1", "diloco"]
    public static let validCompressions: Set<String> = ["none", "fp16", "qjl1", "int4"]

    public var topology: String
    public var peers: [String]
    public var compression: String?

    public init(topology: String = "auto", peers: [String] = [], compression: String? = nil) {
        self.topology = topology
        self.peers = peers
        self.compression = compression
    }
}

public enum ComposeV22MemoryPlanner {
    public static let dflashDrafterGB = 0.80
    public static let soprano80MGB = 0.16
    public static let mossFormer2GB = 0.12
    public static let seedVR2UpscalerGB = 3.90
    public static let sharedEncoderSavingsGB = 2.40

    public static func estimatePeakGB(
        spec: ComposeSpec,
        modelSizesGB: [String: Double]
    ) -> Double {
        var (residentSum, maxOnDemand) = baseModelFootprint(spec: spec, modelSizesGB: modelSizesGB)
        applyVoiceAndImagineSidecars(spec: spec, residentSum: &residentSum, maxOnDemand: &maxOnDemand)
        return spec.memory == .sequential ? max(residentSum, maxOnDemand) : (residentSum + maxOnDemand)
    }

    private static func baseModelFootprint(
        spec: ComposeSpec,
        modelSizesGB: [String: Double]
    ) -> (resident: Double, maxOnDemand: Double) {
        var residentSum = 0.0
        var maxOnDemand = 0.0
        for (name, model) in spec.models where !model.isCloud {
            let sz = modelSizesGB[name] ?? 4.0
            let effectiveKeep = model.keep ?? (spec.answerModels.contains(name) ? .resident : .onDemand)
            if effectiveKeep == .resident && spec.memory != .sequential {
                residentSum += sz
            } else {
                maxOnDemand = max(maxOnDemand, sz)
            }
            if model.speculative?.mode == "dflash",
               let referenced = model.speculative?.drafter,
               spec.models[referenced] == nil
            {
                residentSum += dflashDrafterGB
            }
        }
        return (residentSum, maxOnDemand)
    }

    private static func applyVoiceAndImagineSidecars(
        spec: ComposeSpec,
        residentSum: inout Double,
        maxOnDemand: inout Double
    ) {
        if let voice = spec.voice {
            if voice.denoise == "mossformer2-se" { residentSum += mossFormer2GB }
            if voice.ttsFallback == "soprano-80m" { residentSum += soprano80MGB }
            if voice.mode == "sts", let sts = voice.stsModel, spec.models[sts] == nil {
                maxOnDemand = max(maxOnDemand, 5.2)
            }
        }
        guard let imagine = spec.imagine else { return }
        if let share = imagine.shareTextEncoderWith ?? imagine.shareEncoder, spec.models[share] != nil {
            residentSum = max(0.5, residentSum - sharedEncoderSavingsGB)
        }
        if let up = imagine.upscale, spec.models[up.model] == nil {
            if spec.memory == .resident {
                residentSum += seedVR2UpscalerGB
            } else {
                maxOnDemand = max(maxOnDemand, seedVR2UpscalerGB)
            }
        }
    }
}

extension ComposeSpec {
    var v22ReferencedModels: Set<String> {
        var refs: Set<String> = []
        for (_, model) in models {
            if let dr = model.speculative?.drafter, models[dr] != nil { refs.insert(dr) }
        }
        if let sts = voice?.stsModel, models[sts] != nil { refs.insert(sts) }
        if let m = music?.model, models[m] != nil { refs.insert(m) }
        if let v = video?.model, models[v] != nil { refs.insert(v) }
        for (_, view) in views {
            if let share = view.imagine?.shareTextEncoderWith, models[share] != nil { refs.insert(share) }
            if let up = view.imagine?.upscale?.model, models[up] != nil { refs.insert(up) }
        }
        return refs
    }

    var v22Issues: [Issue] {
        var out: [Issue] = []
        for (name, model) in models.sorted(by: { $0.key < $1.key }) {
            out += v22ModelIssues(name: name, model: model)
        }
        out += v22VoiceIssues + v22MediaIssues + v22TrainAndClusterIssues
        return out
    }

    private func v22ModelIssues(name: String, model: Model) -> [Issue] {
        var out: [Issue] = []
        if let spec = model.speculative {
            if !ModelSpeculativeSpec.validModes.contains(spec.mode) {
                out.append(Issue("models.\(name).speculative.mode", "must be dflash, mtp, lookup or draft"))
            }
            if let bs = spec.blockSize, !(1 ... 64).contains(bs) {
                out.append(Issue("models.\(name).speculative.block_size", "must lie between 1 and 64"))
            }
            if let ver = spec.verify, !ModelSpeculativeSpec.validVerifiers.contains(ver) {
                out.append(Issue("models.\(name).speculative.verify", "must be spectr, residual-race or greedy"))
            }
        }
        if let diff = model.diffusionText {
            if let tau = diff.confidenceThreshold, !(0.1 ... 1.0).contains(tau) {
                out.append(Issue("models.\(name).diffusion_text.confidence_threshold", "must lie between 0.1 and 1.0"))
            }
            if let alpha = diff.contrastiveAlpha, !(0.0 ... 2.0).contains(alpha) {
                out.append(Issue("models.\(name).diffusion_text.contrastive_alpha", "must lie between 0.0 and 2.0"))
            }
        }
        return out
    }

    private var v22VoiceIssues: [Issue] {
        guard let voice else { return [] }
        var out: [Issue] = []
        if !VoiceDuplexSpec.validModes.contains(voice.mode) {
            out.append(Issue("voice.mode", "must be sts or pipeline"))
        }
        if voice.mode == "sts", (voice.stsModel ?? "").trimmingCharacters(in: .whitespaces).isEmpty {
            out.append(Issue("voice.sts_model", "voice.mode: sts requires sts_model"))
        }
        return out
    }

    private var v22MediaIssues: [Issue] {
        var out: [Issue] = []
        if let music, music.model.trimmingCharacters(in: .whitespaces).isEmpty {
            out.append(Issue("music.model", "music.model must not be empty"))
        }
        if let video {
            if video.model.trimmingCharacters(in: .whitespaces).isEmpty {
                out.append(Issue("video.model", "video.model must not be empty"))
            }
            if let fps = video.fps, !(1 ... 120).contains(fps) {
                out.append(Issue("video.fps", "must lie between 1 and 120"))
            }
            if let frames = video.frames, !(1 ... 961).contains(frames) {
                out.append(Issue("video.frames", "must lie between 1 and 961"))
            }
        }
        return out
    }

    private var v22TrainAndClusterIssues: [Issue] {
        var out: [Issue] = []
        if let train {
            if !ComposeTrainSpec.validMethods.contains(train.method) {
                out.append(Issue("train.method", "must be lora, phm, oft, adpo or cadpo"))
            }
            if let seg = train.segments, !(1 ... 4096).contains(seg) {
                out.append(Issue("train.segments", "must lie between 1 and 4096"))
            }
            if let ema = train.emaDecay, !(0.0 ... 0.99999).contains(ema) {
                out.append(Issue("train.ema_decay", "must lie between 0 and 0.99999"))
            }
        }
        if let cluster {
            if !ComposeClusterSpec.validTopologies.contains(cluster.topology) {
                out.append(Issue("cluster.topology", "must be auto, ddp-zero2, pipeline-zbh1 or diloco"))
            }
            if let comp = cluster.compression, !ComposeClusterSpec.validCompressions.contains(comp) {
                out.append(Issue("cluster.compression", "must be none, fp16, qjl1 or int4"))
            }
        }
        return out
    }
}
