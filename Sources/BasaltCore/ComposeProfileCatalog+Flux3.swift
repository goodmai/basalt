import Foundation

public struct Flux3ComposeCombination: Sendable, Equatable, Identifiable {
    public struct Roles: Sendable, Equatable {
        public let rewriter: String
        public let encoder: String
        public let painter: String
        public let decoder: String

        public init(rewriter: String, encoder: String, painter: String, decoder: String) {
            self.rewriter = rewriter
            self.encoder = encoder
            self.painter = painter
            self.decoder = decoder
        }
    }

    struct Metadata: Sendable {
        let id: String
        let tier: ComposeMemoryTier
        let title: String
        let peakGB: Double
        let steps: Int
    }

    public let id: String
    public let tier: ComposeMemoryTier
    public let title: String
    public let roles: Roles
    public let estimatedPeakGB: Double
    public let steps: Int
    public let yaml: String

    public var rewriterModel: String { roles.rewriter }
    public var encoderModel: String { roles.encoder }
    public var painterModel: String { roles.painter }
    public var decoderModel: String { roles.decoder }

    init(meta: Metadata, roles: Roles, yaml: String) {
        self.id = meta.id
        self.tier = meta.tier
        self.title = meta.title
        self.roles = roles
        self.estimatedPeakGB = meta.peakGB
        self.steps = meta.steps
        self.yaml = yaml
    }
}

extension ComposeProfileCatalog {
    public static let flux3Combinations: [Flux3ComposeCombination] = [
        makeFlux3Fast16GB(),
        makeFlux3Balanced24GB(),
        makeFlux3MoEVision36GB(),
        makeFlux3StudioBase64GB(),
        makeFlux3Reference128GB(),
    ]

    public static var flux3YAML: [String: String] {
        Dictionary(uniqueKeysWithValues: flux3Combinations.map { ($0.id, $0.yaml) })
    }

    private static func makeFlux3Fast16GB() -> Flux3ComposeCombination {
        let roles = Flux3ComposeCombination.Roles(
            rewriter: "mlx-community/Qwen3.5-4B-4bit",
            encoder: "mlx-community/Qwen3-VL-4B-Instruct-4bit",
            painter: "goodmai/FLUX.3-klein-MACaQ-MXFP4-MLX",
            decoder: "black-forest-labs/FLUX.2-small-decoder"
        )
        return Flux3ComposeCombination(
            meta: .init(id: "flux3-16gb-fast", tier: .gb16, title: "FLUX.3 Edge Fast (16 GB)", peakGB: 9.6, steps: 8),
            roles: roles,
            yaml: """
                version: 2
                name: flux3-16gb-fast
                memory: auto
                models:
                  rewriter:
                    source: \(roles.rewriter)
                  encoder:
                    source: \(roles.encoder)
                    keep: on-demand
                    idle_unload: 180s
                  painter:
                    source: \(roles.painter)
                    keep: on-demand
                    idle_unload: 180s
                  vae:
                    source: \(roles.decoder)
                    keep: on-demand
                    idle_unload: 180s
                pipeline:
                  answer: rewriter
                  imagine:
                    chain: [painter]
                    upsample_via: rewriter
                    share_encoder: encoder
                    decoder_via: vae
                    steps: 8
                    size: 1024x1024
                    format: png
                    teacache_threshold: 0.14
                    preview_every_steps: 2
                    on_gpu_busy: queue
                """
        )
    }

    private static func makeFlux3Balanced24GB() -> Flux3ComposeCombination {
        let roles = Flux3ComposeCombination.Roles(
            rewriter: "lmstudio-community/Qwen3.5-9B-MLX-4bit",
            encoder: "mlx-community/Qwen3-VL-4B-Instruct-4bit",
            painter: "goodmai/FLUX.3-klein-MACaQ-MXFP4-MLX",
            decoder: "unsloth/FLUX.2-VAE"
        )
        return Flux3ComposeCombination(
            meta: .init(id: "flux3-24gb-balanced", tier: .gb24, title: "FLUX.3 Balanced Sweet-Spot (24 GB)", peakGB: 12.8, steps: 16),
            roles: roles,
            yaml: """
                version: 2
                name: flux3-24gb-balanced
                memory: auto
                models:
                  rewriter:
                    source: \(roles.rewriter)
                  encoder:
                    source: \(roles.encoder)
                    keep: on-demand
                    idle_unload: 240s
                  painter:
                    source: \(roles.painter)
                    keep: on-demand
                    idle_unload: 240s
                  vae:
                    source: \(roles.decoder)
                    keep: on-demand
                    idle_unload: 240s
                pipeline:
                  answer: rewriter
                  imagine:
                    chain: [painter]
                    upsample_via: rewriter
                    share_encoder: encoder
                    decoder_via: vae
                    steps: 16
                    size: 1024x1024
                    format: png
                    teacache_threshold: 0.12
                    preview_every_steps: 4
                    on_gpu_busy: queue
                """
        )
    }
}

extension ComposeProfileCatalog {
    private static func makeFlux3MoEVision36GB() -> Flux3ComposeCombination {
        let roles = Flux3ComposeCombination.Roles(
            rewriter: "goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX",
            encoder: "mlx-community/Qwen3-VL-4B-Instruct-8bit",
            painter: "goodmai/FLUX.3-klein-MACaQ-MXFP4-MLX",
            decoder: "unsloth/FLUX.2-VAE"
        )
        return Flux3ComposeCombination(
            meta: .init(id: "flux3-36gb-moe-vision", tier: .gb36, title: "FLUX.3 MoE Scene Director (32–36 GB)", peakGB: 23.8, steps: 16),
            roles: roles,
            yaml: """
                version: 2
                name: flux3-36gb-moe-vision
                memory: auto
                models:
                  rewriter:
                    source: \(roles.rewriter)
                  encoder:
                    source: \(roles.encoder)
                    keep: on-demand
                    idle_unload: 300s
                  painter:
                    source: \(roles.painter)
                    keep: on-demand
                    idle_unload: 300s
                  fast_img:
                    source: Efficient-Large-Model/SANA-1.5-1.6B
                    keep: on-demand
                    idle_unload: 300s
                  vae:
                    source: \(roles.decoder)
                    keep: on-demand
                    idle_unload: 300s
                pipeline:
                  answer: rewriter
                  imagine:
                    chain: [painter, fast_img]
                    upsample_via: rewriter
                    share_encoder: encoder
                    decoder_via: vae
                    steps: 16
                    size: 1024x1024
                    format: png
                    teacache_threshold: 0.10
                    preview_every_steps: 4
                    on_gpu_busy: fallback
                """
        )
    }

    private static func makeFlux3StudioBase64GB() -> Flux3ComposeCombination {
        let roles = Flux3ComposeCombination.Roles(
            rewriter: "goodmai/Qwen3.8-27B-MACaQ-3bit-MLX",
            encoder: "Qwen/Qwen3-VL-4B-Instruct",
            painter: "mflux-community/flux-3-image-klein-base-mflux-q8",
            decoder: "Felldude/FLUX.2-HDR-VAE"
        )
        return Flux3ComposeCombination(
            meta: .init(id: "flux3-64gb-studio-base", tier: .gb64, title: "FLUX.3 Undistilled CFG Studio (48–64 GB)", peakGB: 31.8, steps: 36),
            roles: roles,
            yaml: """
                version: 2
                name: flux3-64gb-studio-base
                memory: auto
                models:
                  rewriter:
                    source: \(roles.rewriter)
                  encoder:
                    source: \(roles.encoder)
                    keep: resident
                  painter:
                    source: \(roles.painter)
                    keep: resident
                  vae:
                    source: \(roles.decoder)
                    keep: resident
                pipeline:
                  answer: rewriter
                  imagine:
                    chain: [painter]
                    upsample_via: rewriter
                    share_encoder: encoder
                    decoder_via: vae
                    steps: 36
                    size: 1024x1024
                    format: png
                    preview_every_steps: 6
                    on_gpu_busy: queue
                """
        )
    }

    private static func makeFlux3Reference128GB() -> Flux3ComposeCombination {
        let roles = Flux3ComposeCombination.Roles(
            rewriter: "black-forest-labs/flux-3-image-klein-pup",
            encoder: "Qwen/Qwen3-VL-4B-Instruct",
            painter: "mflux-community/flux-3-image-klein-mflux-bf16",
            decoder: "unsloth/FLUX.2-VAE"
        )
        return Flux3ComposeCombination(
            meta: .init(id: "flux3-128gb-reference", tier: .gb128, title: "FLUX.3 Native BF16 Master (128+ GB)", peakGB: 47.0, steps: 16),
            roles: roles,
            yaml: """
                version: 2
                name: flux3-128gb-reference
                memory: resident
                models:
                  rewriter:
                    source: \(roles.rewriter)
                  encoder:
                    source: \(roles.encoder)
                    keep: resident
                  painter:
                    source: \(roles.painter)
                    keep: resident
                  vae:
                    source: \(roles.decoder)
                    keep: resident
                pipeline:
                  answer: rewriter
                  imagine:
                    chain: [painter]
                    upsample_via: rewriter
                    share_encoder: encoder
                    decoder_via: vae
                    steps: 16
                    size: 1024x1024
                    format: png
                    preview_every_steps: 4
                    on_gpu_busy: queue
                """
        )
    }
}
