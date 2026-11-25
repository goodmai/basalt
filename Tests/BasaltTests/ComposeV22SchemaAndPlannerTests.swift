import Foundation
import Testing
@testable import BasaltCLI
@testable import BasaltCore

@Suite("Compose YAML v2.2 schema, hardware-calibrated specialist profiles and RAM deduplication planner (#578)")
struct ComposeV22SchemaAndPlannerTests {
    static let sampleV22YAML = """
        version: "2.2"
        name: studio-v22-full
        memory: auto
        models:
          coder:
            source: goodmai/Qwen3.8-27B-QTIP-2.5bpw-MLX
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
          painter:
            source: goodmai/Qwen-Image-2.1-MACaQ-MXFP4-MLX
            keep: on-demand
            idle_unload: 180s
        voice:
          mode: sts
          sts_model: LiquidAI/LFM2.5-Audio-MLX-4bit
          denoise: mossformer2-se
          turn_detection:
            model: smart-turn-v2
            diarization: sortformer-4spk
          tts_fallback: soprano-80m
          aligner: qwen3-forced-aligner
        music:
          model: mlx-community/ACE-Step-1.5-MLX-4bit
          steps: 24
          duration_seconds: 30
        video:
          model: goodmai/LingBot-Video-30B-A3B-MACaQ-MXFP4-MLX
          vae_tiling: true
          sta_sparse: true
          fps: 24
          frames: 121
        train:
          method: cadpo
          segments: 128
          ema_decay: 0.995
        cluster:
          topology: diloco
          peers: [tb5://mac-studio.local:9443]
          compression: qjl1
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
          imagine:
            chain: [painter]
            share_text_encoder_with: coder
            kv_cache_references: true
            rgba: true
            upscale:
              model: xocialize/SeedVR2-3B-int8-MLX
              steps: 1
              target_resolution: 2048
        """

    @Test("Full v2.2 basalt.yaml parses speculative DFlash, diffusion_text, duplex STS voice, shared encoder imagine, music, video, train and cluster blocks cleanly")
    func fullV22SpecParsesAndValidates() throws {
        let spec = try ComposeFile.parse(Self.sampleV22YAML, file: "basalt.yaml")
        #expect(spec.version == 2)
        #expect(spec.schemaMinorVersion == 2)
        #expect(spec.issues.isEmpty)
        #expect(spec.models["coder"]?.speculative?.mode == "dflash")
        #expect(spec.models["coder"]?.speculative?.blockSize == 12)
        #expect(spec.models["infill"]?.diffusionText?.dualKVCache == true)
        #expect(spec.voice?.turnDetection?.diarization == "sortformer-4spk")
        #expect(spec.imagine?.shareTextEncoderWith == "coder")
        #expect(spec.imagine?.upscale?.targetResolution == 2048)
        #expect(spec.music?.durationSeconds == 30)
        #expect(spec.video?.frames == 121)
        #expect(spec.unservedMediaModels == Set([spec.music?.model, spec.video?.model].compactMap { $0 }))
        #expect(spec.unservedMediaModels.count == 2)
        #expect(spec.train?.method == "cadpo")
        #expect(spec.cluster?.compression == "qjl1")
    }

    @Test("ComposeV22MemoryPlanner credits shared text encoder deduplication and accounts for DFlash, Soprano-80M, MossFormer2 and SeedVR2 sidecars")
    func memoryPlannerDeduplicatesSharedTextEncoder() throws {
        let withShare = try ComposeFile.load("36gb-studio-edit-2k")
        var withoutShare = withShare
        withoutShare.imagine?.shareTextEncoderWith = nil

        let sizes = ["coder": 5.5, "painter": 12.0, "upscaler": 3.9]
        let sharedPeak = ComposeV22MemoryPlanner.estimatePeakGB(spec: withShare, modelSizesGB: sizes)
        let unsharedPeak = ComposeV22MemoryPlanner.estimatePeakGB(spec: withoutShare, modelSizesGB: sizes)
        #expect(sharedPeak < unsharedPeak)
        #expect(abs((unsharedPeak - sharedPeak) - ComposeV22MemoryPlanner.sharedEncoderSavingsGB) < 1e-6)

        for item in ComposeProfileCatalog.v22Specialists {
            let resolved = try #require(ComposeProfileCatalog.resolve(id: item.id))
            #expect(resolved.id == item.id)
            let parsed = try ComposeFile.load(item.id)
            #expect(parsed.issues.isEmpty)
        }
    }

    @Test("Invalid v2.2 speculative, train and cluster values surface structured ComposeSpec issues")
    func invalidV22ValuesRejected() {
        var model = ComposeSpec.Model(source: .local("org/m"))
        model.speculative = ModelSpeculativeSpec(mode: "bad-mode", blockSize: 128, verify: "bad-verify")
        model.diffusionText = DiffusionTextSpec(dualKVCache: true, confidenceThreshold: 1.5, contrastiveAlpha: -1.0)
        var spec = ComposeSpec(name: "invalid-v22", models: ["chat": model], answer: "chat")
        spec.version = 2
        spec.voice = VoiceDuplexSpec(mode: "sts", stsModel: nil)
        spec.train = ComposeTrainSpec(method: "invalid", segments: 0, emaDecay: 1.2)
        spec.cluster = ComposeClusterSpec(topology: "ring", peers: [], compression: "bad")
        let paths = Set(spec.issues.map(\.path))
        #expect(paths.contains("models.chat.speculative.mode"))
        #expect(paths.contains("models.chat.speculative.block_size"))
        #expect(paths.contains("models.chat.diffusion_text.confidence_threshold"))
        #expect(paths.contains("voice.sts_model"))
        #expect(paths.contains("train.method"))
        #expect(paths.contains("cluster.topology"))
    }
}
