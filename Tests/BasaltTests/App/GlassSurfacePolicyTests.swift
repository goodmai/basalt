import Testing
@testable import BasaltAppCore

@Suite("Liquid Glass surface policy: all 8 accessibility flag combinations (#475)")
struct GlassSurfacePolicyTests {
    struct FlagTriple: Sendable, Hashable, CustomStringConvertible {
        let reduceTransparency: Bool
        let reduceMotion: Bool
        let increaseContrast: Bool

        var description: String {
            "rt=\(reduceTransparency) rm=\(reduceMotion) ic=\(increaseContrast)"
        }
    }

    static let allTriples: [FlagTriple] = {
        var list: [FlagTriple] = []
        for rt in [false, true] {
            for rm in [false, true] {
                for ic in [false, true] {
                    list.append(FlagTriple(reduceTransparency: rt, reduceMotion: rm, increaseContrast: ic))
                }
            }
        }
        return list
    }()

    @Test("GlassSurfacePolicy derives usesLiquidGlass, usesMorphAnimation, and strokeOpacity across all 8 combinations",
          arguments: allTriples)
    func policyMatrix(triple: FlagTriple) {
        let policy = GlassSurfacePolicy(
            reduceTransparency: triple.reduceTransparency,
            reduceMotion: triple.reduceMotion,
            increaseContrast: triple.increaseContrast
        )
        #expect(policy.usesLiquidGlass == !triple.reduceTransparency)
        #expect(policy.usesMorphAnimation == !triple.reduceMotion)
        if triple.increaseContrast {
            #expect(policy.strokeOpacity >= 0.35)
        } else {
            #expect(policy.strokeOpacity < 0.30)
        }
        if triple.reduceTransparency {
            let translucentCounterpart = GlassSurfacePolicy(
                reduceTransparency: false,
                reduceMotion: triple.reduceMotion,
                increaseContrast: triple.increaseContrast
            )
            #expect(policy.strokeOpacity > translucentCounterpart.strokeOpacity)
        }
    }

    @Test("GlassElevation enumerates all 4 canonical elevations")
    func elevations() {
        #expect(GlassElevation.allCases == [.chromeRegular, .chromeInteractive, .subtleClear, .prominentAction])
        #expect(GlassSurfacePolicy.standard == GlassSurfacePolicy(reduceTransparency: false, reduceMotion: false, increaseContrast: false))
    }
}
