import Foundation
import Testing
@testable import BasaltCore

@Suite("Inkling 2 P4: basalt fit ranks the oQ packs and Compose resolves the agent profiles per memory tier (#650)")
struct InklingCatalogTests {
    static func gib(_ value: Double) -> ByteCount { ByteCount(Int64(value * 1_073_741_824)) }

    @Test("basalt fit knows the oQ2.5e and oQ3e packs (oQ4e waits for #677 gap 2); oQ3e is a good fit on a 24 GB Mac's 18 GiB working set and a perfect one at 32 GiB")
    func fitCatalog() throws {
        #expect(ModelDatabase.models(for: ModelFamily.inkling).map(\.quantization) == ["oQ2.5e", "oQ3e"])
        let oQ3e = try #require(ModelDatabase.find(id: "mlx-community/Inkling-2-Mini-oQ3e-mtp"))
        #expect(oQ3e.isMoE && oQ3e.task == .code)
        #expect(FitScorer.fitLevel(model: oQ3e, available: Self.gib(18), ramFit: 0.6) == .good)
        #expect(FitScorer.fitLevel(model: oQ3e, available: Self.gib(32), ramFit: 0.8) == .perfect)
    }

    @Test("Each Inkling agent profile resolves to its pack and tier, and parses with no ComposeSpec issues",
          arguments: [
            ("24gb-agent-inkling", "mlx-community/Inkling-2-Mini-oQ3e-mtp", ComposeMemoryTier.gb24),
            ("36gb-agent-inkling", "mlx-community/Inkling-2-Mini-oQ3e-mtp", .gb36),
            ("48gb-agent-inkling", "mlx-community/Inkling-2-Mini-oQ3e-mtp", .gb48),
          ])
    func composeProfiles(id: String, pack: String, tier: ComposeMemoryTier) throws {
        let profile = try #require(ComposeProfileCatalog.resolve(id: id))
        #expect(profile.primaryModel == pack && profile.tier == tier && profile.archetype == .codingAgent)
        #expect(profile.estimatedResidentGB <= Double(tier.nominalGB) * 0.7)
        #expect(profile.yaml.contains("source: \(pack)"))
    }
}
