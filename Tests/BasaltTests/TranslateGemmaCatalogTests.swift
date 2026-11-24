import Foundation
import Testing
@testable import BasaltCLI
@testable import BasaltCore

@Suite("TranslateGemma-4 PV-MACaQ catalog entries and S2S + Vision compose profiles across 16/24/36/48 GB tiers")
struct TranslateGemmaCatalogTests {
    static func gib(_ value: Double) -> ByteCount { ByteCount(Int64(value * 1_073_741_824)) }

    struct ExpectedProfile: Sendable {
        let id: String
        let pack: String
        let tier: ComposeMemoryTier
        let tts: String
        let expectsASR: Bool
        let expectsKVSharedRow: Bool
    }

    static let expectedProfiles: [ExpectedProfile] = [
        ExpectedProfile(
            id: "16gb-translate-gemma4-12b",
            pack: ComposeProfileCatalog.translateGemma12BPack,
            tier: .gb16,
            tts: "hexgrad/Kokoro-82M",
            expectsASR: false,
            expectsKVSharedRow: true
        ),
        ExpectedProfile(
            id: "24gb-translate-gemma4-12b",
            pack: ComposeProfileCatalog.translateGemma12BPack,
            tier: .gb24,
            tts: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            expectsASR: false,
            expectsKVSharedRow: false
        ),
        ExpectedProfile(
            id: "24gb-translate-gemma4-26b",
            pack: ComposeProfileCatalog.translateGemma26BPack,
            tier: .gb24,
            tts: "hexgrad/Kokoro-82M",
            expectsASR: false,
            expectsKVSharedRow: true
        ),
        ExpectedProfile(
            id: "36gb-translate-gemma4-26b",
            pack: ComposeProfileCatalog.translateGemma26BPack,
            tier: .gb36,
            tts: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            expectsASR: false,
            expectsKVSharedRow: false
        ),
        ExpectedProfile(
            id: "48gb-translate-gemma4-26b",
            pack: ComposeProfileCatalog.translateGemma26BPack,
            tier: .gb48,
            tts: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            expectsASR: true,
            expectsKVSharedRow: false
        ),
    ]

    @Test("ModelDatabase registers both TranslateGemma-4 PV-MACaQ packs as multimodal Gemma models with accurate fit levels")
    func modelDatabaseAndFit() throws {
        let pilot = try #require(ModelDatabase.find(id: ComposeProfileCatalog.translateGemma12BPack))
        #expect(pilot.family == .gemma && pilot.modality == .multimodal && !pilot.isMoE)
        #expect(pilot.totalParameters == 12.0 && pilot.activeParameters == 12.0)
        #expect(FitScorer.fitLevel(model: pilot, available: Self.gib(12), ramFit: 0.6) == .good)
        #expect(FitScorer.fitLevel(model: pilot, available: Self.gib(18), ramFit: 0.8) == .perfect)

        let moe = try #require(ModelDatabase.find(id: ComposeProfileCatalog.translateGemma26BPack))
        #expect(moe.family == .gemma && moe.modality == .multimodal && moe.isMoE)
        #expect(moe.totalParameters == 26.0 && moe.activeParameters == 4.0)
        #expect(FitScorer.fitLevel(model: moe, available: Self.gib(18), ramFit: 0.6) == .good)
        #expect(FitScorer.fitLevel(model: moe, available: Self.gib(27), ramFit: 0.8) == .perfect)
    }

    @Test("Each TranslateGemma-4 compose profile resolves, fits its tier working set, and parses with zero ComposeSpec issues",
          arguments: expectedProfiles)
    func composeProfilesResolveAndParse(expected: ExpectedProfile) throws {
        let profile = try #require(ComposeProfileCatalog.resolve(id: expected.id))
        #expect(profile.primaryModel == expected.pack)
        #expect(profile.tier == expected.tier)
        #expect(profile.archetype == .voiceDuplex)
        #expect(profile.estimatedResidentGB <= expected.tier.workingSetGB)

        let spec = try ComposeFile.load(expected.id)
        #expect(spec.issues.isEmpty, "Profile \(expected.id) had issues: \(spec.issues)")
        #expect(spec.local("translator") == expected.pack)
        #expect(spec.local("tts") == expected.tts)
        #expect(spec.answer == "translator")
        #expect(spec.usesSpeechAnalyzer)
        #expect(spec.documents.contains(ComposeSpec.appleVision))
        #expect(spec.audio.contains(ComposeSpec.speechAnalyzer))
        #expect(spec.experimental.kvSharedRow == expected.expectsKVSharedRow)
        #expect((spec.models["asr"] != nil) == expected.expectsASR)
        #expect(spec.speak?.sentenceStreaming == true)
        #expect(spec.speak?.chain == ["tts", ComposeSpec.appleSpeech])
    }
}
