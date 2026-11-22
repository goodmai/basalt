import Foundation
import Testing
import MLX
@testable import BasaltCore
@testable import BasaltAppCore
@testable import BasaltInference

@Suite("Round 7 audit app, sampler & media regression tests (#700, #701, #702)", .serialized)
struct Round7AuditRegressionTests {
    @Test("LibraryReducer does not re-emit .fit on repeated .listed refreshes (#700)")
    func libraryReducerNoInfiniteFitLoop() {
        var state = LibraryState()
        let models = [LibraryModel(id: "org/broken", sizeBytes: 100, isLoaded: false)]
        let first = LibraryReducer.reduce(&state, .listed(models))
        #expect(first == [.fit("org/broken")])
        let second = LibraryReducer.reduce(&state, .listed(models))
        #expect(second.isEmpty)
    }

    @Test("ComposeProfilesReducer ignores stale .activated for superseded profile (#700)")
    func composeProfilesIgnoresStaleActivation() {
        var state = ComposeProfilesState()
        state.applyingID = "profile-b"
        _ = ComposeProfilesReducer.reduce(&state, .activated("profile-a"))
        #expect(state.applyingID == "profile-b")
        #expect(state.activeCompose != "profile-a")
    }

    @Test("NoRepeatNGram.copy isolates history and consecutive didSample preserves tokens (#701)")
    func noRepeatNGramCopyAndDidSample() {
        var orig = NoRepeatNGram(size: 2, window: 10, whitelist: [])
        orig.didSample(token: MLXArray(Int32(1)))
        orig.didSample(token: MLXArray(Int32(2)))
        var copy = orig.copy()
        copy.didSample(token: MLXArray(Int32(1)))
        let logits = MLXArray([Float(0), 0, 0, 0], [1, 4])
        let copyOut = copy.process(logits: logits)
        let origOut = orig.process(logits: logits)
        eval(copyOut, origOut)
        #expect(copyOut[0, 2].item(Float.self) == -Float.infinity)
        #expect(origOut[0, 2].item(Float.self) == 0)
    }

    @Test("EntropyDynamicTemperature stays finite on zero-entropy logits and tiny topP (#701)")
    func edtFiniteOnZeroEntropyAndTinyTopP() {
        let edt = EntropyDynamicTemperature(baseTemperature: 0.7, n: 0.8, theta: 0.0, topP: 1e-7, minP: nil)
        let logits = MLXArray([Float(100.0), -100.0, -100.0, -100.0], [1, 4])
        let out = edt.process(logits: logits)
        let maxVal = out.max(axis: -1)
        eval(maxVal)
        #expect(maxVal.item(Float.self).isFinite)
    }

    @Test("FusedLogitConfidence produces finite entropy on extreme finite logits (#701)")
    func fusedLogitConfidenceExtremeFiniteLogits() {
        let logits = MLXArray([Float(10.0), -Float.greatestFiniteMagnitude, 9.0, -Float.greatestFiniteMagnitude], [1, 4])
        let stats = FusedLogitConfidence.compute(logits)
        eval(stats.entropy)
        #expect(stats.entropy.item(Float.self).isFinite)
    }

    @Test("VideoDiTConfig guarantees hiddenSize is divisible by numHeads (#702)")
    func videoDiTConfigHeadDivisibility() {
        let cfg = VideoDiTConfig(hiddenSize: 1536, numHeads: 12)
        #expect(cfg.hiddenSize.isMultiple(of: cfg.numHeads))
    }

    @Test("Causal3DVAE initializes safely when spatialScale > 8 (#702)")
    func causal3DVAELargeSpatialScale() {
        let cfg = Causal3DVAEConfig(latentChannels: 4, baseChannels: 8, temporalScale: 4, spatialScale: 16)
        let vae = Causal3DVAE(cfg)
        #expect(vae.config.spatialScale == 16)
    }

    @Test("Zonos3MiniConfig.validated rejects mismatched nCodebooks and residualCodebooks (#702)")
    func zonos3MiniConfigRejectsMismatchedCodebooks() {
        var cfg = Zonos3MiniConfig()
        cfg.lm.nCodebooks = 2
        cfg.codec.residualCodebooks = 4
        #expect(throws: BasaltError.self) {
            _ = try cfg.validated()
        }
    }
}
