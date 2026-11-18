import Foundation

public struct VideoComposeCombination: Sendable, Equatable, Identifiable {
    struct Metadata: Sendable {
        let id: String
        let tier: ComposeMemoryTier
        let title: String
        let peakGB: Double
        let frames: Int
        let streamsExperts: Bool
    }

    public let id: String
    public let tier: ComposeMemoryTier
    public let title: String
    public let directorModel: String
    public let videoModel: String
    public let estimatedPeakGB: Double
    public let frames: Int
    public let fps: Int
    public let streamsExperts: Bool
    public let yaml: String

    init(meta: Metadata, directorModel: String, videoModel: String, yaml: String) {
        self.id = meta.id
        self.tier = meta.tier
        self.title = meta.title
        self.directorModel = directorModel
        self.videoModel = videoModel
        self.estimatedPeakGB = meta.peakGB
        self.frames = meta.frames
        self.fps = 24
        self.streamsExperts = meta.streamsExperts
        self.yaml = yaml
    }
}

extension ComposeProfileCatalog {
    public static let videoCombinations: [VideoComposeCombination] = [
        makeVideoKandinsky16GB(),
        makeVideoWan5B24GB(),
        makeVideoLingBot24GB(),
        makeVideoWanA14B48GB(),
        makeVideoLTX64GB(),
    ]

    public static var videoYAML: [String: String] {
        Dictionary(uniqueKeysWithValues: videoCombinations.map { ($0.id, $0.yaml) })
    }

    private static func makeVideoKandinsky16GB() -> VideoComposeCombination {
        let meta = VideoComposeCombination.Metadata(
            id: "video-16gb-kandinsky-lite",
            tier: .gb16,
            title: "Kandinsky 5.0 Lite Fast (16 GB)",
            peakGB: 8.9,
            frames: 49,
            streamsExperts: false
        )
        return makeVideoPreset(
            meta: meta,
            director: "mlx-community/Qwen3-4B-4bit",
            cinema: "goodmai/Kandinsky-5.0-T2V-Lite-PV-MACaQ-4bit-MXFP4-MLX"
        )
    }

    private static func makeVideoWan5B24GB() -> VideoComposeCombination {
        let meta = VideoComposeCombination.Metadata(
            id: "video-24gb-wan22-5b",
            tier: .gb24,
            title: "Wan 2.2 TI2V-5B Balanced (24 GB)",
            peakGB: 13.4,
            frames: 81,
            streamsExperts: false
        )
        return makeVideoPreset(
            meta: meta,
            director: "goodmai/gemma-4-12B-it-qat-4bit",
            cinema: "goodmai/Wan2.2-TI2V-5B-PV-MACaQ-4bit-MXFP4-MLX"
        )
    }

    private static func makeVideoLingBot24GB() -> VideoComposeCombination {
        let meta = VideoComposeCombination.Metadata(
            id: "video-24gb-lingbot-moe",
            tier: .gb24,
            title: "LingBot-Video 30B-A3B MoE (24 GB)",
            peakGB: 16.8,
            frames: 121,
            streamsExperts: true
        )
        return makeVideoPreset(
            meta: meta,
            director: "mlx-community/Qwen3.5-4B-4bit",
            cinema: "goodmai/LingBot-Video-30B-A3B-PV-MACaQ-4bit-MXFP4-MLX"
        )
    }

    private static func makeVideoWanA14B48GB() -> VideoComposeCombination {
        let meta = VideoComposeCombination.Metadata(
            id: "video-48gb-wan22-a14b",
            tier: .gb48,
            title: "Wan 2.2 T2V-A14B Cinema (48 GB)",
            peakGB: 24.6,
            frames: 121,
            streamsExperts: false
        )
        return makeVideoPreset(
            meta: meta,
            director: "goodmai/gemma-4-26B-A4B-it-qat-4bit",
            cinema: "goodmai/Wan2.2-T2V-A14B-PV-MACaQ-4bit-MXFP4-MLX"
        )
    }

    private static func makeVideoLTX64GB() -> VideoComposeCombination {
        let meta = VideoComposeCombination.Metadata(
            id: "video-64gb-ltx2-av",
            tier: .gb64,
            title: "LTX-2 19B Audio-Visual Studio (64+ GB)",
            peakGB: 26.2,
            frames: 121,
            streamsExperts: false
        )
        return makeVideoPreset(
            meta: meta,
            director: "goodmai/Qwen3.8-27B-QTIP-3.0bpw-MLX",
            cinema: "goodmai/LTX-2-19B-Distilled-PV-MACaQ-4bit-MXFP4-MLX"
        )
    }
}

extension ComposeProfileCatalog {
    fileprivate static func makeVideoPreset(
        meta: VideoComposeCombination.Metadata,
        director: String,
        cinema: String
    ) -> VideoComposeCombination {
        let resident = meta.tier.nominalGB >= 64
        let memoryMode = resident ? "auto" : "sequential"
        let keepClause = resident ? "    keep: resident" : "    keep: on-demand\n    idle_unload: 180s"
        let expSection = meta.streamsExperts
            ? "experimental:\n  expert_streaming: true\n  expert_cache_gb: 10\n"
            : ""
        let yaml = """
            version: "2.2"
            name: \(meta.id)
            memory: \(memoryMode)
            \(expSection)models:
              director:
                source: \(director)
              cinema:
                source: \(cinema)
            \(keepClause)
            video:
              model: cinema
              vae_tiling: \(resident ? "false" : "true")
              sta_sparse: true
              fps: 24
              frames: \(meta.frames)
            pipeline:
              answer: director
            """
        return VideoComposeCombination(meta: meta, directorModel: director, videoModel: cinema, yaml: yaml)
    }

    static func makeVideoStudio(id: String, p: ComposeTierPalette) -> ComposeBundledProfile {
        let spec = tierVideoSelection(for: p.tier)
        let resident = p.tier.nominalGB >= 128
        let memoryMode = resident ? "resident" : (p.keepsSidecarsResident ? "auto" : "sequential")
        let keepClause = resident ? "    keep: resident" : "    keep: on-demand\n    idle_unload: 180s"
        let expSection = spec.streamsExperts
            ? "experimental:\n  expert_streaming: true\n  expert_cache_gb: \(max(6.0, p.expertCacheGB))\n"
            : ""
        let peak = resident ? (p.chatGB + spec.modelGB) : max(p.chatGB, spec.modelGB)
        return ComposeBundledProfile(
            id: id,
            tier: p.tier,
            archetype: .videoStudio,
            estimatedResidentGB: peak,
            primaryModel: spec.model,
            capabilityBadges: spec.badges,
            yaml: """
                version: "2.2"
                name: \(id)
                memory: \(memoryMode)
                \(expSection)models:
                  director:
                    source: \(p.chat)
                  cinema:
                    source: \(spec.model)
                \(keepClause)
                video:
                  model: cinema
                  vae_tiling: \(resident ? "false" : "true")
                  sta_sparse: true
                  fps: 24
                  frames: \(spec.frames)
                pipeline:
                  answer: director
                """
        )
    }

    private struct TierVideoSelection {
        let model: String
        let modelGB: Double
        let frames: Int
        let streamsExperts: Bool
        let badges: [String]
    }

    private static func tierVideoSelection(for tier: ComposeMemoryTier) -> TierVideoSelection {
        switch tier {
        case .gb16:
            TierVideoSelection(
                model: "goodmai/Kandinsky-5.0-T2V-Lite-PV-MACaQ-4bit-MXFP4-MLX",
                modelGB: 8.9, frames: 49, streamsExperts: false,
                badges: ["Video DiT", "PV-MACaQ", "MP4"]
            )
        case .gb24:
            TierVideoSelection(
                model: "goodmai/Wan2.2-TI2V-5B-PV-MACaQ-4bit-MXFP4-MLX",
                modelGB: 13.4, frames: 81, streamsExperts: false,
                badges: ["Video DiT", "Phased UMA", "MP4"]
            )
        case .gb36:
            TierVideoSelection(
                model: "goodmai/LingBot-Video-30B-A3B-PV-MACaQ-4bit-MXFP4-MLX",
                modelGB: 16.8, frames: 121, streamsExperts: true,
                badges: ["MoE Video", "SSD Stream", "STA"]
            )
        case .gb48:
            TierVideoSelection(
                model: "goodmai/Wan2.2-T2V-A14B-PV-MACaQ-4bit-MXFP4-MLX",
                modelGB: 24.6, frames: 121, streamsExperts: false,
                badges: ["Wan 2.2 A14B", "3D-VAE", "STA"]
            )
        case .gb64, .gb128, .gb256:
            TierVideoSelection(
                model: "goodmai/LTX-2-19B-Distilled-PV-MACaQ-4bit-MXFP4-MLX",
                modelGB: 26.2, frames: 121, streamsExperts: false,
                badges: ["LTX-2 A/V", "PV-MACaQ", "AAC+HEVC"]
            )
        }
    }
}
