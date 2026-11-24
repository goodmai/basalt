import Testing
import Foundation
import SwiftUI
@testable import BasaltCore
@testable import BasaltCLI
@testable import BasaltAppCore
@testable import BasaltApp

@Suite("91 hardware-tiered Compose profiles (16/24/32-36/48/64/128/256 GB x 13) and RAM detection (#526, #583)")
struct ComposeProfilesMatrixTests {
    @Test("Catalog holds 91 profiles (13 per tier, 4 primary per tier), detects RAM tiers, and resolves 32gb/36gb/236gb aliases")
    func catalogCountsTiersAndAliases() {
        #expect(ComposeMemoryTier.allCases.count == 7)
        #expect(ComposeProfileArchetype.allCases.count == 13)
        #expect(ComposeProfileCatalog.profiles.count == 91)

        for tier in ComposeMemoryTier.allCases {
            let slice = ComposeProfileCatalog.profiles(for: tier)
            #expect(slice.count == 13)
            #expect(slice.filter(\.isPrimary).count == 4)
        }

        #expect(ComposeMemoryTier.detect(ramGB: 16) == .gb16)
        #expect(ComposeMemoryTier.detect(ramGB: 24) == .gb24)
        #expect(ComposeMemoryTier.detect(ramGB: 32) == .gb36)
        #expect(ComposeMemoryTier.detect(ramGB: 36) == .gb36)
        #expect(ComposeMemoryTier.detect(ramGB: 48) == .gb48)
        #expect(ComposeMemoryTier.detect(ramGB: 64) == .gb64)
        #expect(ComposeMemoryTier.detect(ramGB: 128) == .gb128)
        #expect(ComposeMemoryTier.detect(ramGB: 236) == .gb256)
        #expect(ComposeMemoryTier.detect(ramGB: 256) == .gb256)

        #expect(ComposeProfileCatalog.resolve(id: "32gb-daily-chat")?.id == "32-36gb-daily-chat")
        #expect(ComposeProfileCatalog.resolve(id: "36gb-voice-zonos")?.id == "32-36gb-voice-zonos")
        #expect(ComposeProfileCatalog.resolve(id: "236gb-moe-frontier")?.id == "256gb-moe-frontier")
        #expect(ComposeProfileCatalog.resolve(id: "192gb-coding-agent")?.id == "256gb-coding-agent")
    }

    @Test("Every one of the 91 hardware-tiered profiles, 5 FLUX.3 presets, 5 Video DiT presets, aliases, and legacy bundled profiles parses with zero ComposeSpec issues")
    func allBundledProfilesParseCleanly() throws {
        let bundledKeys = ComposeFile.bundled.keys.sorted()
        #expect(ComposeProfileCatalog.flux3Combinations.count == 5)
        #expect(ComposeProfileCatalog.videoCombinations.count == 5)
        #expect(ComposeProfileCatalog.v22Specialists.count == 14)
        #expect(bundledKeys.count == 91 + 4 * 13 + 3 + 5 + 5 + ComposeProfileCatalog.v22Specialists.count)
        for key in bundledKeys {
            let spec = try ComposeFile.load(key)
            #expect(spec.issues.isEmpty, "Bundled profile \(key) had issues: \(spec.issues)")
            #expect(!spec.models.isEmpty)
        }
        for vid in ComposeProfileCatalog.videoCombinations {
            let resolved = try #require(ComposeProfileCatalog.resolve(id: vid.id))
            #expect(resolved.archetype == .videoStudio && resolved.primaryModel.hasPrefix("goodmai/"))
        }
    }

    @Test("ComposeProfilesReducer recommends host-RAM primary profiles, filters across tiers, searches, and dispatches activation")
    func reducerRecommendsFiltersAndActivates() {
        var state = ComposeProfilesState(detectedRAMGB: 24, activeCompose: "24gb-daily-chat")
        #expect(state.detectedTier == .gb24)
        #expect(state.recommendedPrimary.map(\.id) == [
            "24gb-daily-chat",
            "24gb-coding-agent",
            "24gb-voice-zonos",
            "24gb-multimodal-desk",
        ])
        #expect(state.recommendedSecondary.count == 9)

        _ = ComposeProfilesReducer.reduce(&state, .filterChanged(.tier(.gb64)))
        #expect(state.filteredProfiles.count == 13 && state.selectedProfile?.tier == .gb64)
        #expect(state.selectedProfile?.fits(ramGB: 24) == false)

        _ = ComposeProfilesReducer.reduce(&state, .filterChanged(.all))
        _ = ComposeProfilesReducer.reduce(&state, .queryChanged("ZONOS3"))
        #expect(state.filteredProfiles.count == 14)

        let effects = ComposeProfilesReducer.reduce(&state, .activate("24gb-voice-zonos"))
        #expect(effects == [.applyProfile("24gb-voice-zonos")])
        #expect(state.applyingID == "24gb-voice-zonos")

        _ = ComposeProfilesReducer.reduce(&state, .activated("24gb-voice-zonos"))
        #expect(state.activeCompose == "24gb-voice-zonos" && state.applyingID == nil)
    }
}

@Suite("Mac app Compose Profiles browser: 2x sRGB visual verification across Light, Dark, and Russian UI (#526)", .serialized)
@MainActor
struct ComposeProfilesVisualShowcaseTests {
    @Test("Renders ComposeProfilesView with RAM auto-detected Recommended section at 2160x1400 sRGB in Light, Dark, and Russian")
    func rendersComposeProfilesShowcase() throws {
        let store = ComposeProfilesStore(
            ComposeProfilesState(
                detectedRAMGB: 24,
                filter: .recommended,
                activeCompose: "24gb-daily-chat",
                selectedID: "24gb-voice-zonos"
            ),
            reduce: ComposeProfilesReducer.reduce,
            handle: { _ in .none }
        )
        let view = ComposeProfilesView(store: store)
        let spec = VisualSceneSpec(id: VisualSceneID("compose.profiles"), width: 1080, height: 700)

        let light = try OffscreenSceneRenderer.render(view, spec: spec, variant: .lightStandard)
        let dark = try OffscreenSceneRenderer.render(view, spec: spec, variant: .darkStandard)
        let russian = try OffscreenSceneRenderer.render(view, spec: spec, variant: .darkRussian)

        #expect(light.width == 2160 && light.height == 1400 && !light.png.isEmpty)
        #expect(dark.width == 2160 && dark.height == 1400 && !dark.png.isEmpty)
        #expect(russian.width == 2160 && russian.height == 1400 && !russian.png.isEmpty)

        if let rawDir = ProcessInfo.processInfo.environment["BASALT_VISUAL_EXPORT_DIR"], !rawDir.isEmpty {
            let outDir = URL(fileURLWithPath: rawDir)
            try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
            try light.png.write(to: outDir.appending(path: "gui_526_compose_profiles__light.png"), options: .atomic)
            try dark.png.write(to: outDir.appending(path: "gui_526_compose_profiles__dark.png"), options: .atomic)
            try russian.png.write(to: outDir.appending(path: "gui_526_compose_profiles__ru.png"), options: .atomic)
        }
    }
}
