import Foundation

extension ComposeProfileCatalog {
    static let translateGemma12BPack = "Bosalt/TranslateGemma-4-12B-Pilot-PV-MACaQ-4bit-MXFP4-MLX"
    static let translateGemma26BPack = "Bosalt/TranslateGemma-4-26B-A4B-v2-PV-MACaQ-4bit-MXFP4-MLX"

    private struct TranslateGemmaSpec: Sendable {
        let id: String
        let tier: ComposeMemoryTier
        let model: String
        let asr: String?
        let tts: String
        let voice: String
        let ttsResident: Bool
        let tightKV: Bool
        let peakGB: Double
        let badges: [String]
    }

    static let translateGemmaSpecialists: [V22ComposeSpecialist] = [
        translateGemma(TranslateGemmaSpec(
            id: "16gb-translate-gemma4-12b",
            tier: .gb16,
            model: translateGemma12BPack,
            asr: nil,
            tts: "hexgrad/Kokoro-82M",
            voice: "af_heart",
            ttsResident: false,
            tightKV: true,
            peakGB: 8.6,
            badges: ["Translate 12B", "PV-MACaQ", "S2S + Vision"]
        )),
        translateGemma(TranslateGemmaSpec(
            id: "24gb-translate-gemma4-12b",
            tier: .gb24,
            model: translateGemma12BPack,
            asr: nil,
            tts: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            voice: "Serena",
            ttsResident: true,
            tightKV: false,
            peakGB: 9.8,
            badges: ["Translate 12B", "Multilingual TTS", "S2S + Vision"]
        )),
        translateGemma(TranslateGemmaSpec(
            id: "24gb-translate-gemma4-26b",
            tier: .gb24,
            model: translateGemma26BPack,
            asr: nil,
            tts: "hexgrad/Kokoro-82M",
            voice: "af_heart",
            ttsResident: false,
            tightKV: true,
            peakGB: 16.4,
            badges: ["Translate 26B MoE", "PV-MACaQ", "Tight 24 GB"]
        )),
        translateGemma(TranslateGemmaSpec(
            id: "36gb-translate-gemma4-26b",
            tier: .gb36,
            model: translateGemma26BPack,
            asr: nil,
            tts: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            voice: "Serena",
            ttsResident: true,
            tightKV: false,
            peakGB: 18.2,
            badges: ["Translate 26B MoE", "Multilingual TTS", "Real-Time S2S"]
        )),
        translateGemma(TranslateGemmaSpec(
            id: "48gb-translate-gemma4-26b",
            tier: .gb48,
            model: translateGemma26BPack,
            asr: "divergentlabs/Nemotron-3.5-Transcribe-30B-A4B-MLX",
            tts: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            voice: "Serena",
            ttsResident: true,
            tightKV: false,
            peakGB: 24.8,
            badges: ["Translate 26B MoE", "Nemotron ASR", "Multilingual TTS"]
        )),
    ]

    private static func translateGemma(_ spec: TranslateGemmaSpec) -> V22ComposeSpecialist {
        let ttsKeep = spec.ttsResident
            ? "    keep: resident"
            : "    keep: on-demand\n    idle_unload: 120s"
        let asrPart = spec.asr.map { "  asr:\n    source: \($0)\n    keep: on-demand\n    idle_unload: 180s" }
        let modelsSection = [
            "  translator:\n    source: \(spec.model)",
            asrPart,
            "  tts:\n    source: \(spec.tts)\n\(ttsKeep)",
        ].compactMap { $0 }.joined(separator: "\n")
        let audioChain = spec.asr == nil ? "[speech-analyzer]" : "[asr, speech-analyzer]"
        let experimentalFlag = spec.tightKV ? "  kv_shared_row: true" : "  smallm_kernel: true"
        return V22ComposeSpecialist(
            id: spec.id,
            tier: spec.tier,
            archetype: .voiceDuplex,
            primaryModel: spec.model,
            estimatedPeakGB: spec.peakGB,
            badges: spec.badges,
            yaml: """
                version: "2.2"
                name: \(spec.id)
                memory: auto
                experimental:
                  speech_analyzer: true
                \(experimentalFlag)
                models:
                \(modelsSection)
                pipeline:
                  perceive:
                    documents: [apple-vision]
                    audio: \(audioChain)
                  answer: translator
                  speak:
                    chain: [tts, apple-speech]
                    voice: \(spec.voice)
                    on_gpu_busy: fallback_ane
                    sentence_streaming: true
                    max_ttfa_ms: 80
                """
        )
    }
}
