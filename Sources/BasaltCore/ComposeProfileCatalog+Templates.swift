import Foundation

extension ComposeProfileCatalog {
    static func makeProfile(
        palette p: ComposeTierPalette,
        archetype: ComposeProfileArchetype
    ) -> ComposeBundledProfile {
        let id = "\(p.tier.rawValue)-\(archetype.rawValue)"
        let keep = p.keepsSidecarsResident
            ? "    keep: resident"
            : "    keep: on-demand\n    idle_unload: 180s"
        switch archetype {
        case .dailyChat: return makeDailyChat(id: id, p: p)
        case .codingAgent: return makeCodingAgent(id: id, p: p)
        case .voiceZonos: return makeVoiceZonos(id: id, p: p, keep: keep)
        case .multimodalDesk: return makeMultimodalDesk(id: id, p: p, keep: keep)
        case .reasoningLab: return makeReasoningLab(id: id, p: p)
        case .documentOCR: return makeDocumentOCR(id: id, p: p, keep: keep)
        case .voiceDuplex: return makeVoiceDuplex(id: id, p: p, keep: keep)
        case .creativeStudio: return makeCreativeStudio(id: id, p: p, keep: keep)
        case .ragKnowledge: return makeRagKnowledge(id: id, p: p)
        case .hybridCloud: return makeHybridCloud(id: id, p: p, keep: keep)
        case .moeFrontier: return makeMoEFrontier(id: id, p: p)
        case .researchXray: return makeResearchXray(id: id, p: p)
        }
    }

    private static func makeDailyChat(id: String, p: ComposeTierPalette) -> ComposeBundledProfile {
        ComposeBundledProfile(
            id: id, tier: p.tier, archetype: .dailyChat,
            estimatedResidentGB: p.chatGB, primaryModel: p.chat,
            capabilityBadges: ["Chat", "Vision", "Speech"],
            yaml: """
                version: 2
                name: \(id)
                memory: auto
                models:
                  chat:
                    source: \(p.chat)
                pipeline:
                  perceive:
                    documents: [apple-vision]
                    audio: [speech-analyzer]
                  answer: chat
                """
        )
    }

    private static func makeCodingAgent(id: String, p: ComposeTierPalette) -> ComposeBundledProfile {
        let coderKeep = p.tier.nominalGB >= 128 ? "    keep: resident" : "    keep: on-demand\n    idle_unload: 600s"
        let est = p.tier.nominalGB >= 128 ? (p.chatGB + p.coderGB) : max(p.chatGB, p.coderGB)
        return ComposeBundledProfile(
            id: id, tier: p.tier, archetype: .codingAgent,
            estimatedResidentGB: est, primaryModel: p.coder,
            capabilityBadges: ["Code", "Routes", "MCP"],
            yaml: """
                version: 2
                name: \(id)
                memory: auto
                experimental:
                  smallm_kernel: true
                mcp:
                  enabled: true
                  rest: true
                models:
                  chat:
                    source: \(p.chat)
                  coder:
                    source: \(p.coder)
                \(coderKeep)
                pipeline:
                  decide:
                    cascade: [chat]
                    abstain_below: 0.7
                  answer:
                    routes:
                      - when: { escalated: true }
                        model: coder
                      - model: chat
                """
        )
    }

    private static func makeVoiceZonos(id: String, p: ComposeTierPalette, keep: String) -> ComposeBundledProfile {
        ComposeBundledProfile(
            id: id, tier: p.tier, archetype: .voiceZonos,
            estimatedResidentGB: p.chatGB + (p.keepsSidecarsResident ? 2.6 : 2.3),
            primaryModel: "Zyphra/ZONOS3-Mini-hf",
            capabilityBadges: ["TTS 44.1 kHz", "ZONOS3", "Kokoro"],
            yaml: """
                version: 2
                name: \(id)
                memory: auto
                models:
                  chat:
                    source: \(p.chat)
                  tts_zonos:
                    source: Zyphra/ZONOS3-Mini-hf
                \(keep)
                  tts_fast:
                    source: hexgrad/Kokoro-82M
                \(keep)
                pipeline:
                  perceive:
                    audio: [speech-analyzer]
                  answer: chat
                  speak:
                    chain: [tts_zonos, tts_fast, apple-speech]
                    voice: Savannah
                    language: en_us
                    on_gpu_busy: fallback_ane
                    sentence_streaming: true
                """
        )
    }

    private static func makeMultimodalDesk(id: String, p: ComposeTierPalette, keep: String) -> ComposeBundledProfile {
        ComposeBundledProfile(
            id: id, tier: p.tier, archetype: .multimodalDesk,
            estimatedResidentGB: p.chatGB + (p.keepsSidecarsResident ? 4.2 : 2.4),
            primaryModel: p.chat,
            capabilityBadges: ["OCR", "Decide", "ZONOS3"],
            yaml: """
                version: 2
                name: \(id)
                memory: auto
                models:
                  chat:
                    source: \(p.chat)
                  ocr:
                    source: \(p.ocr)
                \(keep)
                  tts:
                    source: Zyphra/ZONOS3-Mini-hf
                \(keep)
                pipeline:
                  perceive:
                    documents: [ocr, apple-vision]
                    audio: [speech-analyzer]
                  decide:
                    cascade: [chat]
                    abstain_below: 0.75
                  answer: chat
                  speak:
                    chain: [tts, apple-speech]
                    voice: Savannah
                    language: en_us
                    on_gpu_busy: fallback_apple_speech
                    sentence_streaming: true
                """
        )
    }
}

extension ComposeProfileCatalog {
    private static func makeReasoningLab(id: String, p: ComposeTierPalette) -> ComposeBundledProfile {
        ComposeBundledProfile(
            id: id, tier: p.tier, archetype: .reasoningLab,
            estimatedResidentGB: p.reasonerGB, primaryModel: p.reasoner,
            capabilityBadges: ["Reasoning", "Thinking", "Decide"],
            yaml: """
                version: 2
                name: \(id)
                memory: auto
                experimental:
                  adaptive_thinking_budget: true
                models:
                  chat:
                    source: \(p.reasoner)
                pipeline:
                  decide:
                    cascade: [chat]
                    abstain_below: 0.8
                  answer: chat
                """
        )
    }

    private static func makeDocumentOCR(id: String, p: ComposeTierPalette, keep: String) -> ComposeBundledProfile {
        ComposeBundledProfile(
            id: id, tier: p.tier, archetype: .documentOCR,
            estimatedResidentGB: p.chatGB + 2.2, primaryModel: p.ocr,
            capabilityBadges: ["OCR", "Epistemic", "PDF"],
            yaml: """
                version: 2
                name: \(id)
                memory: auto
                models:
                  chat:
                    source: \(p.chat)
                  ocr:
                    source: \(p.ocr)
                \(keep)
                pipeline:
                  perceive:
                    documents: [ocr, apple-vision]
                  decide:
                    cascade: [chat]
                    abstain_below: 0.75
                    min_margin: 0.1
                    epistemic:
                      enabled: true
                      support_threshold: 0.65
                      refutation_threshold: 0.6
                      tool_threshold: 0.7
                      on_no_evidence: abstain
                      on_conflict: report
                  answer: chat
                """
        )
    }

    private static func makeVoiceDuplex(id: String, p: ComposeTierPalette, keep: String) -> ComposeBundledProfile {
        let asrPart = p.duplexASR.map { "  asr:\n    source: \($0)\n\(keep)" }
        let modelsSection = [
            "  chat:\n    source: \(p.chat)",
            asrPart,
            "  tts:\n    source: \(p.duplexTTS)\n\(keep)",
        ].compactMap { $0 }.joined(separator: "\n")
        let audioChain = p.duplexASR == nil ? "[speech-analyzer]" : "[asr, speech-analyzer]"
        return ComposeBundledProfile(
            id: id, tier: p.tier, archetype: .voiceDuplex,
            estimatedResidentGB: p.chatGB + (p.duplexASR == nil ? 1.4 : 6.0),
            primaryModel: p.duplexTTS,
            capabilityBadges: ["Duplex", "ASR", "Low TTFA"],
            yaml: """
                version: 2
                name: \(id)
                memory: auto
                experimental:
                  speech_analyzer: true
                models:
                \(modelsSection)
                pipeline:
                  perceive:
                    audio: \(audioChain)
                  answer: chat
                  speak:
                    chain: [tts, apple-speech]
                    voice: \(p.duplexVoice)
                    on_gpu_busy: fallback_ane
                    sentence_streaming: true
                    max_ttfa_ms: 80
                """
        )
    }

    private static func makeCreativeStudio(id: String, p: ComposeTierPalette, keep: String) -> ComposeBundledProfile {
        ComposeBundledProfile(
            id: id, tier: p.tier, archetype: .creativeStudio,
            estimatedResidentGB: p.chatGB + 4.5, primaryModel: p.imageMain,
            capabilityBadges: ["Imagine", "TeaCache", "Upsample"],
            yaml: """
                version: 2
                name: \(id)
                memory: auto
                models:
                  chat:
                    source: \(p.chat)
                  artist:
                    source: \(p.imageMain)
                \(keep)
                  fast_img:
                    source: Efficient-Large-Model/SANA1.5_1.6B_1024px
                \(keep)
                pipeline:
                  answer: chat
                  imagine:
                    chain: [artist, fast_img]
                    upsample_via: chat
                    share_encoder: chat
                    steps: 8
                    size: 1024x1024
                    format: png
                    teacache_threshold: 0.15
                    preview_every_steps: 2
                    on_gpu_busy: fallback
                """
        )
    }
}

extension ComposeProfileCatalog {
    private static func makeRagKnowledge(id: String, p: ComposeTierPalette) -> ComposeBundledProfile {
        let verKeep = p.tier.nominalGB >= 128 ? "    keep: resident" : "    keep: on-demand\n    idle_unload: 600s"
        return ComposeBundledProfile(
            id: id, tier: p.tier, archetype: .ragKnowledge,
            estimatedResidentGB: max(p.chatGB, p.coderGB), primaryModel: p.chat,
            capabilityBadges: ["Belnap", "Grounded", "Routes"],
            yaml: """
                version: 2
                name: \(id)
                memory: auto
                models:
                  chat:
                    source: \(p.chat)
                  verifier:
                    source: \(p.coder)
                \(verKeep)
                pipeline:
                  perceive:
                    documents: [apple-vision]
                  decide:
                    cascade: [chat]
                    abstain_below: 0.8
                    epistemic:
                      enabled: true
                      support_threshold: 0.7
                      refutation_threshold: 0.55
                      tool_threshold: 0.65
                      on_no_evidence: escalate
                      on_conflict: report
                  answer:
                    routes:
                      - when: { escalated: true }
                        model: verifier
                      - model: chat
                """
        )
    }

    private static func makeHybridCloud(id: String, p: ComposeTierPalette, keep: String) -> ComposeBundledProfile {
        let judgePart = p.judge.map { "  judge:\n    source: \($0)\n\(keep)" }
        let modelsSection = [
            "  chat:\n    source: \(p.chat)",
            judgePart,
            "  frontier:\n    source: cloud:openrouter/anthropic/claude-sonnet-4.5\n    api_key_env: OPENROUTER_API_KEY\n    budget:\n      usd_per_day: 1",
        ].compactMap { $0 }.joined(separator: "\n")
        let cascade = p.judge == nil ? "[chat]" : "[judge, chat]"
        let escalateLine = p.judge == nil ? "" : "\n    escalate_below: 0.65"
        return ComposeBundledProfile(
            id: id, tier: p.tier, archetype: .hybridCloud,
            estimatedResidentGB: p.chatGB + (p.judge == nil ? 0 : 5.9),
            primaryModel: p.judge ?? p.chat,
            capabilityBadges: ["Judge", "Cloud Cap", "Privacy"],
            yaml: """
                version: 2
                name: \(id)
                memory: auto
                budget:
                  usd_per_day: 2
                models:
                \(modelsSection)
                pipeline:
                  decide:
                    cascade: \(cascade)\(escalateLine)
                    abstain_below: 0.78
                  answer:
                    routes:
                      - when: { escalated: true, cost_left: true }
                        model: frontier
                        fallback: chat
                      - model: chat
                """
        )
    }

    private static func makeMoEFrontier(id: String, p: ComposeTierPalette) -> ComposeBundledProfile {
        let expBlock = p.moeStreams
            ? "experimental:\n  expert_streaming: true\n  expert_cache_gb: \(p.expertCacheGB)"
            : "experimental:\n  smallm_kernel: true"
        return ComposeBundledProfile(
            id: id, tier: p.tier, archetype: .moeFrontier,
            estimatedResidentGB: p.moeGB, primaryModel: p.moeModel,
            capabilityBadges: p.moeStreams ? ["MoE", "SSD Stream", "0 Swap"] : ["MoE", "Resident", "Fast"],
            yaml: """
                version: 2
                name: \(id)
                memory: auto
                \(expBlock)
                models:
                  chat:
                    source: \(p.moeModel)
                pipeline:
                  answer: chat
                """
        )
    }

    private static func makeResearchXray(id: String, p: ComposeTierPalette) -> ComposeBundledProfile {
        ComposeBundledProfile(
            id: id, tier: p.tier, archetype: .researchXray,
            estimatedResidentGB: p.chatGB, primaryModel: p.chat,
            capabilityBadges: ["X-Ray", "Overlays", "MCP"],
            yaml: """
                version: 2
                name: \(id)
                memory: auto
                experimental:
                  telemetry_confidence_every: 1
                mcp:
                  enabled: true
                  rest: true
                  research_tools: true
                models:
                  chat:
                    source: \(p.chat)
                pipeline:
                  decide:
                    cascade: [chat]
                    abstain_below: 0.75
                  answer: chat
                """
        )
    }
}
