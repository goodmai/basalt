import BasaltCore
import Foundation
import Yams

extension ComposeFile.Walk {
    static func parseVersionNode(_ node: Node?) -> (major: Int, minor: Int) {
        guard let node else { return (0, 0) }
        if let intVal = node.int {
            return (intVal, 0)
        }
        if let raw = node.scalar?.string.trimmingCharacters(in: .whitespacesAndNewlines) {
            if raw == "2.2" { return (2, 2) }
            if let iv = Int(raw) { return (iv, 0) }
        }
        if let flt = node.float, abs(flt - 2.2) < 1e-6 {
            return (2, 2)
        }
        return (0, 0)
    }

    mutating func boolean(_ node: Node?, _ path: String) -> Bool? {
        guard let node else { return nil }
        guard let value = node.bool else {
            issue(path, node, "must be true or false")
            return nil
        }
        return value
    }

    mutating func parseModelV22(_ field: [String: Node], path: String, into model: inout ComposeSpec.Model) {
        if let specNode = field["speculative"],
           let sf = fields(specNode, path: "\(path).speculative", section: "model_speculative")
        {
            model.speculative = ModelSpeculativeSpec(
                mode: text(sf["mode"], "\(path).speculative.mode") ?? "",
                drafter: text(sf["drafter"], "\(path).speculative.drafter"),
                blockSize: integer(sf["block_size"], "\(path).speculative.block_size"),
                verify: text(sf["verify"], "\(path).speculative.verify")
            )
        }
        if let diffNode = field["diffusion_text"],
           let df = fields(diffNode, path: "\(path).diffusion_text", section: "diffusion_text")
        {
            model.diffusionText = DiffusionTextSpec(
                dualKVCache: boolean(df["dual_kv_cache"], "\(path).diffusion_text.dual_kv_cache") ?? true,
                confidenceThreshold: number(df["confidence_threshold"], "\(path).diffusion_text.confidence_threshold"),
                contrastiveAlpha: number(df["contrastive_alpha"], "\(path).diffusion_text.contrastive_alpha")
            )
        }
    }

    mutating func parseImagineV22(_ field: [String: Node], path: String, into stage: inout ComposeSpec.Imagine) {
        stage.shareTextEncoderWith = text(field["share_text_encoder_with"], "\(path).share_text_encoder_with")
        stage.kvCacheReferences = boolean(field["kv_cache_references"], "\(path).kv_cache_references")
        stage.rgba = boolean(field["rgba"], "\(path).rgba")
        if let upNode = field["upscale"],
           let uf = fields(upNode, path: "\(path).upscale", section: "imagine_upscale")
        {
            stage.upscale = ImagineUpscaleSpec(
                model: text(uf["model"], "\(path).upscale.model") ?? "",
                steps: integer(uf["steps"], "\(path).upscale.steps"),
                targetResolution: integer(uf["target_resolution"], "\(path).upscale.target_resolution")
            )
        }
    }

    mutating func parseRootV22(_ top: [String: Node], into spec: inout ComposeSpec) {
        parseVoiceV22(top["voice"], into: &spec)
        parseMediaV22(top, into: &spec)
        parseTrainAndClusterV22(top, into: &spec)
    }

    private mutating func parseVoiceV22(_ node: Node?, into spec: inout ComposeSpec) {
        guard let node, let vf = fields(node, path: "voice", section: "voice_v22") else { return }
        var turnSpec: VoiceTurnDetectionSpec?
        if let tn = vf["turn_detection"],
           let tf = fields(tn, path: "voice.turn_detection", section: "voice_turn_detection")
        {
            turnSpec = VoiceTurnDetectionSpec(
                model: text(tf["model"], "voice.turn_detection.model"),
                diarization: text(tf["diarization"], "voice.turn_detection.diarization")
            )
        }
        spec.voice = VoiceDuplexSpec(
            mode: text(vf["mode"], "voice.mode") ?? "sts",
            stsModel: text(vf["sts_model"], "voice.sts_model"),
            denoise: text(vf["denoise"], "voice.denoise"),
            turnDetection: turnSpec,
            ttsFallback: text(vf["tts_fallback"], "voice.tts_fallback"),
            aligner: text(vf["aligner"], "voice.aligner")
        )
    }

    private mutating func parseMediaV22(_ top: [String: Node], into spec: inout ComposeSpec) {
        if let mn = top["music"], let mf = fields(mn, path: "music", section: "music_v22") {
            spec.music = MusicComposeSpec(
                model: text(mf["model"], "music.model") ?? "",
                steps: integer(mf["steps"], "music.steps"),
                durationSeconds: integer(mf["duration_seconds"], "music.duration_seconds")
            )
        }
        if let vn = top["video"], let vf = fields(vn, path: "video", section: "video_v22") {
            spec.video = VideoComposeSpec(
                model: text(vf["model"], "video.model") ?? "",
                vaeTiling: boolean(vf["vae_tiling"], "video.vae_tiling"),
                staSparse: boolean(vf["sta_sparse"], "video.sta_sparse"),
                fps: integer(vf["fps"], "video.fps"),
                frames: integer(vf["frames"], "video.frames")
            )
        }
    }

    private mutating func parseTrainAndClusterV22(_ top: [String: Node], into spec: inout ComposeSpec) {
        if let tn = top["train"], let tf = fields(tn, path: "train", section: "train_v22") {
            spec.train = ComposeTrainSpec(
                method: text(tf["method"], "train.method") ?? "",
                segments: integer(tf["segments"], "train.segments"),
                emaDecay: number(tf["ema_decay"], "train.ema_decay")
            )
        }
        if let cn = top["cluster"], let cf = fields(cn, path: "cluster", section: "cluster_v22") {
            spec.cluster = ComposeClusterSpec(
                topology: text(cf["topology"], "cluster.topology") ?? "auto",
                peers: list(cf["peers"], "cluster.peers"),
                compression: text(cf["compression"], "cluster.compression")
            )
        }
    }
}
