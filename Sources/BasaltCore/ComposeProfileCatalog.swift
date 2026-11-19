import Foundation

public enum ComposeMemoryTier: String, Sendable, Codable, Equatable, CaseIterable, Identifiable {
    case gb16 = "16gb"
    case gb24 = "24gb"
    case gb36 = "32-36gb"
    case gb48 = "48gb"
    case gb64 = "64gb"
    case gb128 = "128gb"
    case gb256 = "256gb"

    public var id: String { rawValue }

    public var nominalGB: Int {
        switch self {
        case .gb16: 16
        case .gb24: 24
        case .gb36: 36
        case .gb48: 48
        case .gb64: 64
        case .gb128: 128
        case .gb256: 256
        }
    }

    public var minimumRecommendedGB: Int {
        switch self {
        case .gb16: 16
        case .gb24: 24
        case .gb36: 32
        case .gb48: 48
        case .gb64: 64
        case .gb128: 128
        case .gb256: 192
        }
    }

    public var workingSetGB: Double {
        switch self {
        case .gb16: 11.5
        case .gb24: 17.5
        case .gb36: 26.5
        case .gb48: 36.0
        case .gb64: 48.0
        case .gb128: 96.0
        case .gb256: 192.0
        }
    }

    public var displayTitle: String {
        switch self {
        case .gb16: "16 GB"
        case .gb24: "24 GB"
        case .gb36: "32–36 GB"
        case .gb48: "48 GB"
        case .gb64: "64 GB"
        case .gb128: "128 GB"
        case .gb256: "256 GB"
        }
    }

    public var aliases: [String] {
        switch self {
        case .gb36: ["32gb", "36gb"]
        case .gb256: ["192gb", "236gb"]
        case .gb16, .gb24, .gb48, .gb64, .gb128: []
        }
    }

    public static func detect(physicalBytes: UInt64 = ProcessInfo.processInfo.physicalMemory) -> ComposeMemoryTier {
        let gb = Double(physicalBytes) / 1_073_741_824.0
        return detect(ramGB: Int(gb.rounded()))
    }

    public static func detect(ramGB: Int) -> ComposeMemoryTier {
        switch ramGB {
        case ..<20: .gb16
        case 20 ..< 30: .gb24
        case 30 ..< 42: .gb36
        case 42 ..< 56: .gb48
        case 56 ..< 96: .gb64
        case 96 ..< 180: .gb128
        default: .gb256
        }
    }
}

public enum ComposeProfileArchetype: String, Sendable, Codable, Equatable, CaseIterable, Identifiable {
    case dailyChat = "daily-chat"
    case codingAgent = "coding-agent"
    case voiceZonos = "voice-zonos"
    case multimodalDesk = "multimodal-desk"
    case reasoningLab = "reasoning-lab"
    case documentOCR = "document-ocr"
    case voiceDuplex = "voice-duplex"
    case creativeStudio = "creative-studio"
    case videoStudio = "video-studio"
    case ragKnowledge = "rag-knowledge"
    case hybridCloud = "hybrid-cloud"
    case moeFrontier = "moe-frontier"
    case researchXray = "research-xray"

    public var id: String { rawValue }

    public var isPrimary: Bool {
        switch self {
        case .dailyChat, .codingAgent, .voiceZonos, .multimodalDesk:
            true
        case .reasoningLab, .documentOCR, .voiceDuplex, .creativeStudio,
             .videoStudio, .ragKnowledge, .hybridCloud, .moeFrontier, .researchXray:
            false
        }
    }

    public var symbolName: String {
        switch self {
        case .dailyChat: "bubble.left.and.bubble.right"
        case .codingAgent: "chevron.left.forwardslash.chevron.right"
        case .voiceZonos: "waveform"
        case .multimodalDesk: "rectangle.3.group"
        case .reasoningLab: "brain.head.profile"
        case .documentOCR: "doc.text.viewfinder"
        case .voiceDuplex: "mic.and.signal.meter"
        case .creativeStudio: "paintbrush.pointed"
        case .videoStudio: "film.stack"
        case .ragKnowledge: "checkmark.shield"
        case .hybridCloud: "cloud.bolt"
        case .moeFrontier: "cpu"
        case .researchXray: "waveform.path.ecg"
        }
    }

    public var titleKey: String {
        switch self {
        case .dailyChat: "Daily Chat & Vision"
        case .codingAgent: "Coding Agent & MCP"
        case .voiceZonos: "ZONOS3 44.1 kHz Voice"
        case .multimodalDesk: "Multimodal Workstation"
        case .reasoningLab: "Deep Reasoning Lab"
        case .documentOCR: "Document OCR & Grounded QA"
        case .voiceDuplex: "Low-Latency Duplex Speech"
        case .creativeStudio: "Creative Image Studio"
        case .videoStudio: "Audio-Visual Video Studio"
        case .ragKnowledge: "Epistemic Grounded Research"
        case .hybridCloud: "Hybrid Judge + Cloud Escalation"
        case .moeFrontier: "Frontier MoE Engine"
        case .researchXray: "Model X-Ray & Surgery Lab"
        }
    }

    public var summaryKey: String {
        switch self {
        case .dailyChat: "Fast everyday chat, Apple Vision attachments, and local speech input."
        case .codingAgent: "Speculative small-M kernel, specialist code routing, and MCP tools."
        case .voiceZonos: "Zyphra ZONOS3-Mini 44.1 kHz speech with Kokoro-82M fallback."
        case .multimodalDesk: "OCR, audio perception, System One triage, chat, and ZONOS3 voice."
        case .reasoningLab: "Adaptive thinking budgets tuned for multi-step math and logic."
        case .documentOCR: "Neural vision OCR with four-valued Belnap epistemic verification."
        case .voiceDuplex: "Full-duplex transcription and sub-80 ms TTFA streaming synthesis."
        case .creativeStudio: "Native MLX image generation with prompt upsampling and TeaCache."
        case .videoStudio: "Phased UMA Video DiT synthesis with Causal 3D-VAE, TeaCache, and native MP4 muxing."
        case .ragKnowledge: "Conflict-aware epistemic gate routing hard queries to a verifier."
        case .hybridCloud: "Local System One judge saving tokens before budgeted cloud fallback."
        case .moeFrontier: "Maximum-parameter Mixture-of-Experts tuned for this memory tier."
        case .researchXray: "Live Model X-Ray, confidence telemetry, and reversible overlays."
        }
    }
}

public struct ComposeBundledProfile: Sendable, Equatable, Identifiable {
    public let id: String
    public let tier: ComposeMemoryTier
    public let archetype: ComposeProfileArchetype
    public let estimatedResidentGB: Double
    public let primaryModel: String
    public let capabilityBadges: [String]
    public let yaml: String

    public var isPrimary: Bool { archetype.isPrimary }

    public func fits(ramGB: Int) -> Bool {
        ramGB >= tier.minimumRecommendedGB
    }
}
