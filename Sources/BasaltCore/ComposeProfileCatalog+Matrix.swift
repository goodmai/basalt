import Foundation

public enum ComposeProfileCatalog {
    public static let profiles: [ComposeBundledProfile] = buildAllProfiles()

    public static let profilesByID: [String: ComposeBundledProfile] = Dictionary(
        uniqueKeysWithValues: profiles.map { ($0.id, $0) }
    )

    public static let legacyYAML: [String: String] = [
        "document-qa": """
            version: 1
            name: document-qa
            memory: auto
            models:
              chat:
                source: mlx-community/gemma-4-12B-it-qat-4bit
              ocr:
                source: mlx-community/PaddleOCR-VL-1.5-8bit
                keep: on-demand
                idle_unload: 120s
            pipeline:
              perceive:
                documents: [ocr, apple-vision]
              decide:
                cascade: [chat]
                abstain_below: 0.75
                min_margin: 0.1
              answer: chat
            """,
        "voice-agent": """
            version: 1
            name: voice-agent
            memory: sequential
            models:
              chat:
                source: mlx-community/gemma-4-12B-it-qat-4bit
              asr:
                source: divergentlabs/Nemotron-3.5-Transcribe-30B-A4B-MLX
                keep: sequential
            pipeline:
              perceive:
                audio: [asr, speech-analyzer]
              answer: chat
            """,
        "zonos-voice": """
            version: 1
            name: zonos-voice
            models:
              chat:
                source: mlx-community/gemma-4-12B-it-qat-4bit
              tts:
                source: Zyphra/ZONOS3-Mini-hf
                keep: on-demand
                idle_unload: 120s
            pipeline:
              perceive:
                audio: [speech-analyzer]
              answer: chat
              speak:
                chain: [tts, apple-speech]
                voice: Savannah
                language: en_us
                on_gpu_busy: fallback_apple_speech
                sentence_streaming: true
            """,
    ]

    public static var bundledYAML: [String: String] {
        var map = legacyYAML
            .merging(flux3YAML) { _, new in new }
            .merging(videoYAML) { _, new in new }
            .merging(v22YAML) { _, new in new }
        for profile in profiles {
            map[profile.id] = profile.yaml
            for alias in profile.tier.aliases {
                map["\(alias)-\(profile.archetype.rawValue)"] = profile.yaml
            }
        }
        return map
    }

    public static func resolve(id rawID: String) -> ComposeBundledProfile? {
        let clean = rawID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if let exact = profilesByID[clean] { return exact }
        if let v22 = v22Specialists.first(where: { $0.id == clean }) {
            return ComposeBundledProfile(
                id: v22.id,
                tier: v22.tier,
                archetype: v22.archetype,
                estimatedResidentGB: v22.estimatedPeakGB,
                primaryModel: v22.primaryModel,
                capabilityBadges: v22.badges,
                yaml: v22.yaml
            )
        }
        if let flux = flux3Combinations.first(where: { $0.id == clean }) {
            return ComposeBundledProfile(
                id: flux.id,
                tier: flux.tier,
                archetype: .creativeStudio,
                estimatedResidentGB: flux.estimatedPeakGB,
                primaryModel: flux.painterModel,
                capabilityBadges: ["Chat", "FLUX.3", "\(flux.steps) steps"],
                yaml: flux.yaml
            )
        }
        if let vid = videoCombinations.first(where: { $0.id == clean }) {
            return ComposeBundledProfile(
                id: vid.id,
                tier: vid.tier,
                archetype: .videoStudio,
                estimatedResidentGB: vid.estimatedPeakGB,
                primaryModel: vid.videoModel,
                capabilityBadges: ["Video DiT", "PV-MACaQ", "\(vid.frames)f"],
                yaml: vid.yaml
            )
        }
        if let legacy = legacyYAML[clean] {
            let archetype: ComposeProfileArchetype = clean == "document-qa" ? .documentOCR : .voiceZonos
            return ComposeBundledProfile(
                id: clean,
                tier: .gb24,
                archetype: archetype,
                estimatedResidentGB: 14.0,
                primaryModel: "mlx-community/Qwen3.5-9B-4bit",
                capabilityBadges: ["Chat", clean],
                yaml: legacy
            )
        }
        for tier in ComposeMemoryTier.allCases {
            for alias in tier.aliases where clean.hasPrefix("\(alias)-") {
                let suffix = clean.dropFirst(alias.count + 1)
                if let mapped = profilesByID["\(tier.rawValue)-\(suffix)"] { return mapped }
            }
        }
        return nil
    }

    public static func recommended(for ramGB: Int, primaryOnly: Bool = false) -> [ComposeBundledProfile] {
        let tier = ComposeMemoryTier.detect(ramGB: ramGB)
        return profiles(for: tier).filter { !primaryOnly || $0.isPrimary }
    }

    public static func profiles(for tier: ComposeMemoryTier) -> [ComposeBundledProfile] {
        profiles.filter { $0.tier == tier }
    }
}

struct ComposeTierPalette: Sendable {
    let tier: ComposeMemoryTier
    let chat: String
    let chatGB: Double
    let coder: String
    let coderGB: Double
    let reasoner: String
    let reasonerGB: Double
    let ocr: String
    let judge: String?
    let duplexASR: String?
    let duplexTTS: String
    let duplexVoice: String
    let imageMain: String
    let moeModel: String
    let moeGB: Double
    let moeStreams: Bool
    let expertCacheGB: Double
    let keepsSidecarsResident: Bool
}

extension ComposeProfileCatalog {
    static let palettes: [ComposeTierPalette] = [
        ComposeTierPalette(
            tier: .gb16,
            chat: "mlx-community/Qwen3-4B-4bit", chatGB: 2.4,
            coder: "LiquidAI/LFM2.5-8B-A1B-MLX-4bit", coderGB: 4.6,
            reasoner: "goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed", reasonerGB: 7.4,
            ocr: "mlx-community/PaddleOCR-VL-1.5-8bit",
            judge: nil, duplexASR: nil,
            duplexTTS: "hexgrad/Kokoro-82M", duplexVoice: "af_heart",
            imageMain: "Tongyi-MAI/Z-Image-Turbo",
            moeModel: "goodmai/LFM2.5-24B-A2B-MLX-4bit", moeGB: 6.8, moeStreams: true, expertCacheGB: 2.5,
            keepsSidecarsResident: false
        ),
        ComposeTierPalette(
            tier: .gb24,
            chat: "goodmai/gemma-4-12B-it-qat-4bit", chatGB: 7.6,
            coder: "goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native", coderGB: 11.8,
            reasoner: "goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed", reasonerGB: 7.4,
            ocr: "goodmai/DeepSeek-OCR-2-8bit",
            judge: "goodmai/OpenJev-Flash-9B-MLX-4bit", duplexASR: nil,
            duplexTTS: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", duplexVoice: "Serena",
            imageMain: "Tongyi-MAI/Z-Image-Turbo",
            moeModel: "goodmai/Qwen3.6-35B-A3B-8bit", moeGB: 13.9, moeStreams: true, expertCacheGB: 4.0,
            keepsSidecarsResident: false
        ),
        ComposeTierPalette(
            tier: .gb36,
            chat: "goodmai/gemma-4-26B-A4B-it-qat-4bit", chatGB: 14.5,
            coder: "mlx-community/Muse-Glimmer-30B-4bit", coderGB: 18.1,
            reasoner: "mlx-community/flint-oQ3e", reasonerGB: 17.6,
            ocr: "goodmai/DeepSeek-OCR-2-8bit",
            judge: "goodmai/OpenJev-Flash-9B-MLX-4bit",
            duplexASR: "divergentlabs/Nemotron-3.5-Transcribe-30B-A4B-MLX",
            duplexTTS: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", duplexVoice: "Serena",
            imageMain: "black-forest-labs/FLUX.2-klein-4B",
            moeModel: "mlx-community/flint-4bit", moeGB: 16.5, moeStreams: true, expertCacheGB: 6.0,
            keepsSidecarsResident: false
        ),
        ComposeTierPalette(
            tier: .gb48,
            chat: "goodmai/gemma-4-26B-A4B-it-qat-4bit", chatGB: 14.5,
            coder: "Jundot/Muse-Glimmer-30B-oQ4e", coderGB: 19.8,
            reasoner: "mlx-community/flint-4bit", reasonerGB: 22.0,
            ocr: "goodmai/DeepSeek-OCR-2-8bit",
            judge: "goodmai/OpenJev-Flash-9B-MLX-4bit",
            duplexASR: "divergentlabs/Nemotron-3.5-Transcribe-30B-A4B-MLX",
            duplexTTS: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", duplexVoice: "Serena",
            imageMain: "black-forest-labs/FLUX.2-klein-4B",
            moeModel: "mlx-community/Ricursive-Ripple-65B-4bit", moeGB: 24.0, moeStreams: true, expertCacheGB: 10.0,
            keepsSidecarsResident: true
        ),
        ComposeTierPalette(
            tier: .gb64,
            chat: "mlx-community/flint-4bit", chatGB: 22.0,
            coder: "goodmai/Qwen3.6-35B-A3B-8bit", coderGB: 37.7,
            reasoner: "Jundot/Muse-Glimmer-30B-oQ4e", reasonerGB: 19.8,
            ocr: "goodmai/DeepSeek-OCR-2-8bit",
            judge: "goodmai/OpenJev-Flash-9B-MLX-4bit",
            duplexASR: "divergentlabs/Nemotron-3.5-Transcribe-30B-A4B-MLX",
            duplexTTS: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", duplexVoice: "Serena",
            imageMain: "black-forest-labs/FLUX.2-klein-4B",
            moeModel: "goodmai/Qwen3.6-35B-A3B-8bit", moeGB: 37.7, moeStreams: false, expertCacheGB: 0,
            keepsSidecarsResident: true
        ),
        ComposeTierPalette(
            tier: .gb128,
            chat: "mlx-community/flint-8bit", chatGB: 42.0,
            coder: "mlx-community/Ricursive-Ripple-65B-4bit", coderGB: 36.5,
            reasoner: "meta-models/Muse-Glimmer-30B", reasonerGB: 58.0,
            ocr: "goodmai/DeepSeek-OCR-2-8bit",
            judge: "goodmai/OpenJev-Flash-9B-MLX-4bit",
            duplexASR: "divergentlabs/Nemotron-3.5-Transcribe-30B-A4B-MLX",
            duplexTTS: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", duplexVoice: "Serena",
            imageMain: "black-forest-labs/FLUX.2-klein-4B",
            moeModel: "mlx-community/Ricursive-Ripple-65B-4bit", moeGB: 36.5, moeStreams: false, expertCacheGB: 0,
            keepsSidecarsResident: true
        ),
        ComposeTierPalette(
            tier: .gb256,
            chat: "mlx-community/Ricursive-Ripple-65B-8bit", chatGB: 69.0,
            coder: "Recursive/Flint-40B-A2.5B-bf16", coderGB: 80.0,
            reasoner: "mlx-community/Ricursive-Ripple-65B-8bit", reasonerGB: 69.0,
            ocr: "goodmai/DeepSeek-OCR-2-8bit",
            judge: "goodmai/OpenJev-Flash-9B-MLX-4bit",
            duplexASR: "divergentlabs/Nemotron-3.5-Transcribe-30B-A4B-MLX",
            duplexTTS: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", duplexVoice: "Serena",
            imageMain: "black-forest-labs/FLUX.2-klein-4B",
            moeModel: "Recursive/Flint-40B-A2.5B-bf16", moeGB: 80.0, moeStreams: false, expertCacheGB: 0,
            keepsSidecarsResident: true
        ),
    ]

    private static func buildAllProfiles() -> [ComposeBundledProfile] {
        var list: [ComposeBundledProfile] = []
        list.reserveCapacity(91)
        for palette in palettes {
            for archetype in ComposeProfileArchetype.allCases {
                list.append(makeProfile(palette: palette, archetype: archetype))
            }
        }
        return list
    }
}
