import Foundation

public struct V22ComposeSpecialist: Sendable, Equatable, Identifiable {
    public let id: String
    public let tier: ComposeMemoryTier
    public let archetype: ComposeProfileArchetype
    public let primaryModel: String
    public let estimatedPeakGB: Double
    public let badges: [String]
    public let yaml: String
}

extension ComposeProfileCatalog {
    public static let v22Specialists: [V22ComposeSpecialist] = [
        make24GBCodingDFlash(),
        make24GBVoiceSTSDuplex(),
        make24GBMusicACEStep(),
        make24GBVideoLingBot(),
        make36GBStudioEdit2K(),
        make64GBClusterDiLoCo(),
    ]

    public static var v22YAML: [String: String] {
        Dictionary(uniqueKeysWithValues: v22Specialists.map { ($0.id, $0.yaml) })
    }

    private static func make24GBCodingDFlash() -> V22ComposeSpecialist {
        let model = "goodmai/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-PV-MACaQ-3bit-MLX"
        return V22ComposeSpecialist(
            id: "24gb-coding-dflash",
            tier: .gb24,
            archetype: .codingAgent,
            primaryModel: model,
            estimatedPeakGB: 14.9,
            badges: ["DFlash", "dLLM Infill", "MCP"],
            yaml: """
                version: "2.2"
                name: 24gb-coding-dflash
                memory: auto
                mcp:
                  enabled: true
                  rest: true
                models:
                  coder:
                    source: \(model)
                    speculative:
                      mode: dflash
                      drafter: z-lab/Qwen3-DFlash-Drafter-4bit
                      block_size: 12
                      verify: spectr
                  infill:
                    source: goodmai/DiffuCoder-7B-MACaQ-4bit-MLX
                    keep: on-demand
                    idle_unload: 180s
                    diffusion_text:
                      dual_kv_cache: true
                      confidence_threshold: 0.92
                      contrastive_alpha: 0.35
                pipeline:
                  answer:
                    model: coder
                    routes:
                      - when:
                          prompt_tokens:
                            over: 4096
                        model: coder
                      - model: infill
                        fallback: coder
                """
        )
    }

    private static func make24GBVoiceSTSDuplex() -> V22ComposeSpecialist {
        make24GBVoiceSTSDuplex(
            id: "24gb-voice-sts-duplex",
            tier: .gb24,
            archetype: .voiceDuplex,
            primaryModel: "LiquidAI/LFM2.5-Audio-MLX-4bit",
            estimatedResidentGB: 13.8
        )
    }

    private static func make24GBVoiceSTSDuplex(id: String, tier: ComposeMemoryTier, archetype: ComposeProfileArchetype, primaryModel: String, estimatedResidentGB: Double) -> V22ComposeSpecialist {
        V22ComposeSpecialist(
            id: id,
            tier: tier,
            archetype: archetype,
            primaryModel: primaryModel,
            estimatedPeakGB: estimatedResidentGB,
            badges: ["Duplex STS", "SmartTurn", "Soprano 80M"],
            yaml: """
                version: "2.2"
                name: \(id)
                memory: auto
                models:
                  chat:
                    source: mlx-community/gemma-4-12B-it-qat-4bit
                  sts:
                    source: \(primaryModel)
                    keep: on-demand
                    idle_unload: 300s
                voice:
                  mode: sts
                  sts_model: sts
                  denoise: mossformer2-se
                  turn_detection:
                    model: smart-turn-v2
                    diarization: sortformer-4spk
                  tts_fallback: soprano-80m
                  aligner: qwen3-forced-aligner
                pipeline:
                  perceive:
                    audio: [speech-analyzer]
                  answer: chat
                """
        )
    }

    private static func make24GBMusicACEStep() -> V22ComposeSpecialist {
        let music = "mlx-community/ACE-Step-1.5-MLX-4bit"
        return V22ComposeSpecialist(
            id: "24gb-music-acestep",
            tier: .gb24,
            archetype: .creativeStudio,
            primaryModel: music,
            estimatedPeakGB: 12.4,
            badges: ["ACE-Step 1.5", "Music DiT", "Lyrics"],
            yaml: """
                version: "2.2"
                name: 24gb-music-acestep
                memory: auto
                models:
                  lyricist:
                    source: mlx-community/Qwen3.5-9B-4bit
                  composer:
                    source: \(music)
                    keep: on-demand
                    idle_unload: 240s
                music:
                  model: composer
                  steps: 24
                  duration_seconds: 30
                pipeline:
                  answer: lyricist
                """
        )
    }

    private static func make24GBVideoLingBot() -> V22ComposeSpecialist {
        let vid = "goodmai/LingBot-Video-30B-A3B-MACaQ-MXFP4-MLX"
        return V22ComposeSpecialist(
            id: "24gb-video-lingbot",
            tier: .gb24,
            archetype: .creativeStudio,
            primaryModel: vid,
            estimatedPeakGB: 17.6,
            badges: ["Video DiT", "3D-VAE Tiled", "STA"],
            yaml: """
                version: "2.2"
                name: 24gb-video-lingbot
                memory: sequential
                experimental:
                  expert_streaming: true
                  expert_cache_gb: 12
                models:
                  director:
                    source: mlx-community/Qwen3.5-4B-4bit
                  cinema:
                    source: \(vid)
                    keep: on-demand
                    idle_unload: 180s
                video:
                  model: cinema
                  vae_tiling: true
                  sta_sparse: true
                  fps: 24
                  frames: 121
                pipeline:
                  answer: director
                """
        )
    }

    private static func make36GBStudioEdit2K() -> V22ComposeSpecialist {
        let img = "goodmai/Qwen-Image-2.1-MACaQ-MXFP4-MLX"
        return V22ComposeSpecialist(
            id: "36gb-studio-edit-2k",
            tier: .gb36,
            archetype: .creativeStudio,
            primaryModel: img,
            estimatedPeakGB: 21.5,
            badges: ["Shared Encoder", "SeedVR2 2K", "RGBA"],
            yaml: """
                version: "2.2"
                name: 36gb-studio-edit-2k
                memory: auto
                models:
                  coder:
                    source: mlx-community/Qwen3-VL-8B-Instruct-4bit
                  painter:
                    source: \(img)
                    keep: on-demand
                    idle_unload: 240s
                  upscaler:
                    source: xocialize/SeedVR2-3B-int8-MLX
                    keep: on-demand
                    idle_unload: 180s
                pipeline:
                  answer: coder
                  imagine:
                    chain: [painter]
                    share_text_encoder_with: coder
                    kv_cache_references: true
                    rgba: true
                    upscale:
                      model: upscaler
                      steps: 1
                      target_resolution: 2048
                """
        )
    }

    private static func make64GBClusterDiLoCo() -> V22ComposeSpecialist {
        let model = "goodmai/Flint-40B-A2.5B-PV-MACaQ-3bit-MLX"
        return V22ComposeSpecialist(
            id: "64gb-cluster-diloco",
            tier: .gb64,
            archetype: .researchXray,
            primaryModel: model,
            estimatedPeakGB: 28.0,
            badges: ["Cluster TB5", "CA-DPO", "QJL-1"],
            yaml: """
                version: "2.2"
                name: 64gb-cluster-diloco
                memory: auto
                models:
                  frontier:
                    source: \(model)
                train:
                  method: cadpo
                  segments: 128
                  ema_decay: 0.995
                cluster:
                  topology: diloco
                  peers: [tb5://mac-studio-node2.local:9443]
                  compression: qjl1
                pipeline:
                  answer: frontier
                """
        )
    }
}
