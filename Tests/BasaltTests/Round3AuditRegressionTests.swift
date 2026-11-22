import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltAppCore

@Suite("Round 3 audit core regression tests (#682)", .serialized)
struct Round3CoreRegressionTests {
    @Test("OrchestratorState.reduce(.unloaded) clears modelId and modelPath so reload is not skipped (#682)")
    func orchestratorUnloadClearsModelIdentity() {
        var state = OrchestratorState(kv: KVPlanner(maxTokens: 1000, precisionLadder: [nil], zones: nil))
        let geo = KVCacheGeometry(parameterProxy: 1.0)
        state = OrchestratorState.reduce(state, .loaded(path: "/tmp/m", modelId: "org/m", geometry: geo, takes: (false, false, false, false)))
        #expect(state.modelId == "org/m" && state.modelPath == "/tmp/m")
        state = OrchestratorState.reduce(state, .unloaded)
        #expect(state.modelId.isEmpty && state.modelPath.isEmpty)
    }

    @Test("PowerSGDCompressor preserves rank-deficient matrix without row doubling (#682)")
    func powerSGDRankDeficientMatrix() {
        let matrix: [Float] = [1, 1, 1, 1]
        var warmQ: [Float] = []
        let compressed = PowerSGDCompressor.compress(matrix: matrix, rows: 2, cols: 2, rank: 2, warmQ: &warmQ)
        let reconstructed = PowerSGDCompressor.decompress(compressed)
        for i in 0 ..< 4 {
            #expect(abs(reconstructed[i] - matrix[i]) < 1e-4)
        }
    }

    @Test("ToolCallRepairer handles single quotes with embedded quotes/escapes, trailing text, and keys inside strings (#682)")
    func toolCallRepairerEdgeCases() {
        let doubleInSingle = ToolCallRepairer.repairJSON("{'cmd': 'echo \"hi\"'}")
        #expect(doubleInSingle != nil)
        let escapedApostrophe = ToolCallRepairer.repairJSON(#"{'msg': 'it\'s'}"#)
        #expect(escapedApostrophe != nil)
        let trailingChatter = ToolCallRepairer.repairJSON("{\"a\": 1} Done.")
        #expect(trailingChatter == "{\"a\": 1}")
        let keyInString = ToolCallRepairer.repairJSON("{\"cmd\": \"echo a, foo: bar\", \"ok\": True,}")
        #expect(keyInString != nil)
        #expect(keyInString?.contains("echo a, foo: bar") == true)
    }

    @Test("HeterogeneousNodeProfile avoids double linkScale and guards infinite linkGbps (#682)")
    func heterogeneousNodeProfileLinkScale() {
        let cap = PeerCapability(peerId: "p1", chipName: "M4", unifiedMemoryBytes: 32 << 30, measuredTFLOPS: 20, thermalFactor: 1.0, linkGbps: 10)
        let profile = HeterogeneousNodeProfile(capability: cap, powerWatts: 30)
        #expect(abs(profile.computeRateUnitsPerSecond - cap.effectiveThroughputScore) < 1e-6)

        let infProfile = HeterogeneousNodeProfile(nodeId: "p2", effectiveTFLOPS: 10, powerWatts: 20, availableMemoryBytes: 16 << 30, linkGbps: .infinity)
        #expect(infProfile.computeRateUnitsPerSecond.isFinite)
    }

    @Test("SemanticCacheStore rejects invalid embeddings even when minSimilarity is -1.0 (#682)")
    func semanticCacheStoreRejectsInvalidEmbeddingsAtMinusOne() {
        var store = SemanticCacheStore(config: SemanticCacheConfig(minSimilarity: -1.0))
        store.store(key: "k", tenantId: "t", embedding: [1, 0], response: "ok", now: 1)
        #expect(store.lookup(tenantId: "t", embedding: [], now: 2) == nil)
        #expect(store.lookup(tenantId: "t", embedding: [1, 0, 0], now: 2) == nil)
        #expect(store.lookup(tenantId: "t", embedding: [0, 0], now: 2) == nil)
    }

    @Test("LayerStagePartitioner rejects non-positive or NaN safetyFraction (#682)")
    func layerStagePartitionerRejectsBadSafetyFraction() {
        let neg = LayerStagePartitioner.partition(layerMemoryBytes: [100, 100], peerRAMBytes: [200, 200], safetyFraction: -0.5)
        #expect(!neg.fitsSafetyCeiling && neg.stageRanges.isEmpty)
        let nan = LayerStagePartitioner.partition(layerMemoryBytes: [100, 100], peerRAMBytes: [200, 200], safetyFraction: .nan)
        #expect(!nan.fitsSafetyCeiling && nan.stageRanges.isEmpty)
    }

    @Test("JSONSchemaGate rejects non-pointer #anchor refs (#682)")
    func jsonSchemaGateRejectsNonPointerAnchor() {
        let schema = ##"{"type":"object","properties":{"a":{"$ref":"#myAnchor"}}}"##
        #expect(throws: Error.self) {
            _ = try JSONSchemaGate.prepare(schema, strict: false)
        }
    }
}

@Suite("Round 3 audit vision, speech and app regression tests (#683, #684)", .serialized)
struct Round3VisionAppRegressionTests {
    @Test("PaddleOCRVLProcessorConfiguration.resized never returns 0 width or height (#683)")
    func paddleOCRResizedNonZero() throws {
        let cfg = PaddleOCRVLProcessorConfiguration(
            minPixels: 3136,
            maxPixels: 100_352,
            patchSize: 14,
            mergeSize: 2,
            imageMean: [0.5, 0.5, 0.5],
            imageStd: [0.5, 0.5, 0.5]
        )
        let (w, h) = try cfg.resized(width: 28, height: 4000)
        #expect(w >= 28 && h >= 28)
    }

    @Test("VoiceReducer emits stopCapture when failed arrives during recording (#684)")
    func voiceReducerStopsCaptureOnFailureWhileRecording() {
        var state = VoiceSessionState()
        state.capture = .recording
        let effects = VoiceReducer.reduce(&state, .failed("tts failed"))
        #expect(state.capture == .idle)
        #expect(effects == [.stopCapture])
    }

    @Test("URLSession.engine allows more than 4 concurrent loopback connections (#684)")
    func engineSessionConnectionLimit() {
        #expect(URLSession.engine.configuration.httpMaximumConnectionsPerHost >= 16)
    }
}
