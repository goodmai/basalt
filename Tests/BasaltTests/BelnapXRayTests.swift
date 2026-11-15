import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Belnap FOUR and Shramko-Wansing SIXTEEN_3 epistemic algebras (#360)")
struct BelnapTrilatticeTests {
    @Test("BelnapState FOUR satisfies all 64-triple bilattice, De Morgan, absorption, and 2-bit packing laws")
    func fourBilatticeLaws() {
        let all = BelnapState.allCases
        #expect(all.count == 4)
        #expect(BelnapState.neither.bitPair == 0 && BelnapState.false.bitPair == 1)
        #expect(BelnapState.true.bitPair == 2 && BelnapState.both.bitPair == 3)
        for a in all {
            #expect(BelnapState(bitPair: a.bitPair) == a)
            #expect(a.truthNegated == a.negated && a.truthNegated.truthNegated == a)
            #expect(a.truthJoin(a) == a && a.truthMeet(a) == a)
            #expect(a.infoJoin(a) == a && a.infoMeet(a) == a)
            #expect(BelnapState.false.truthPrecedesOrEquals(a) && a.truthPrecedesOrEquals(.true))
            #expect(BelnapState.neither.infoPrecedesOrEquals(a) && a.infoPrecedesOrEquals(.both))
            for b in all {
                assertFourPair(a, b)
                for c in all {
                    assertFourTriple(a, b, c)
                }
            }
        }
    }

    func assertFourPair(_ a: BelnapState, _ b: BelnapState) {
        #expect(a.truthJoin(b) == b.truthJoin(a) && a.truthMeet(b) == b.truthMeet(a))
        #expect(a.infoJoin(b) == b.infoJoin(a) && a.infoMeet(b) == b.infoMeet(a))
        #expect(a.infoJoin(b) == a.knowledgeJoin(b) && a.infoMeet(b) == a.knowledgeMeet(b))
        #expect(a.infoJoin(b).bitPair == (a.bitPair | b.bitPair))
        #expect(a.infoMeet(b).bitPair == (a.bitPair & b.bitPair))
        #expect(a.truthJoin(a.truthMeet(b)) == a && a.truthMeet(a.truthJoin(b)) == a)
        #expect(a.infoJoin(a.infoMeet(b)) == a && a.infoMeet(a.infoJoin(b)) == a)
        #expect(a.truthJoin(b).truthNegated == a.truthNegated.truthMeet(b.truthNegated))
        #expect(a.truthMeet(b).truthNegated == a.truthNegated.truthJoin(b.truthNegated))
        #expect(a.infoJoin(b).truthNegated == a.truthNegated.infoJoin(b.truthNegated))
        #expect(a.infoMeet(b).truthNegated == a.truthNegated.infoMeet(b.truthNegated))
    }

    func assertFourTriple(_ a: BelnapState, _ b: BelnapState, _ c: BelnapState) {
        #expect(a.truthJoin(b).truthJoin(c) == a.truthJoin(b.truthJoin(c)))
        #expect(a.truthMeet(b).truthMeet(c) == a.truthMeet(b.truthMeet(c)))
        #expect(a.infoJoin(b).infoJoin(c) == a.infoJoin(b.infoJoin(c)))
        #expect(a.infoMeet(b).infoMeet(c) == a.infoMeet(b.infoMeet(c)))
        #expect(a.truthMeet(b.infoJoin(c)) == a.truthMeet(b).infoJoin(a.truthMeet(c)))
        #expect(a.truthMeet(b.infoMeet(c)) == a.truthMeet(b).infoMeet(a.truthMeet(c)))
        #expect(a.truthJoin(b.infoJoin(c)) == a.truthJoin(b).infoJoin(a.truthJoin(c)))
        #expect(a.truthJoin(b.infoMeet(c)) == a.truthJoin(b).infoMeet(a.truthJoin(c)))
        #expect(a.infoMeet(b.truthJoin(c)) == a.infoMeet(b).truthJoin(a.infoMeet(c)))
        #expect(a.infoMeet(b.truthMeet(c)) == a.infoMeet(b).truthMeet(a.infoMeet(c)))
        #expect(a.infoJoin(b.truthJoin(c)) == a.infoJoin(b).truthJoin(a.infoJoin(c)))
        #expect(a.infoJoin(b.truthMeet(c)) == a.infoJoin(b).truthMeet(a.infoJoin(c)))
    }

    @Test("BelnapGeneralizedState SIXTEEN_3 satisfies all 4096-triple trilattice, De Morgan, and interlacing laws")
    func sixteenTrilatticeLaws() throws {
        let all = BelnapGeneralizedState.allCases
        #expect(all.count == 16)
        #expect(BelnapGeneralizedState(rawBits: 0xFF) == .all && BelnapGeneralizedState.empty.rawBits == 0)
        #expect(BelnapGeneralizedState.neither.rawBits == 0x1 && BelnapGeneralizedState.false.rawBits == 0x2)
        #expect(BelnapGeneralizedState.true.rawBits == 0x4 && BelnapGeneralizedState.both.rawBits == 0x8)
        #expect(BelnapGeneralizedState([.true, .false]) == (BelnapGeneralizedState.true | .false))
        for single in BelnapState.allCases {
            let singleton = BelnapGeneralizedState(single)
            for candidate in BelnapState.allCases {
                #expect(singleton.contains(candidate) == (candidate == single))
            }
        }
        let truthBottom = BelnapGeneralizedState([.neither, .false])
        let truthTop = BelnapGeneralizedState([.true, .both])
        let falsityBottom = BelnapGeneralizedState([.neither, .true])
        let falsityTop = BelnapGeneralizedState([.false, .both])
        for a in all {
            let encoded = try JSONEncoder().encode(a)
            #expect(try JSONDecoder().decode(BelnapGeneralizedState.self, from: encoded) == a)
            #expect(a.truthNegated.truthNegated == a && a.falsityNegated.falsityNegated == a && a.infoNegated.infoNegated == a)
            #expect(a.truthNegated.falsityNegated == a.falsityNegated.truthNegated)
            #expect(a.infoMeet(a) == a && a.infoJoin(a) == a)
            #expect(a.truthMeet(a) == a && a.truthJoin(a) == a)
            #expect(a.falsityMeet(a) == a && a.falsityJoin(a) == a)
            #expect(BelnapGeneralizedState.empty.infoPrecedesOrEquals(a) && a.infoPrecedesOrEquals(.all))
            #expect(truthBottom.truthPrecedesOrEquals(a) && a.truthPrecedesOrEquals(truthTop))
            #expect(falsityBottom.falsityPrecedesOrEquals(a) && a.falsityPrecedesOrEquals(falsityTop))
            for b in all {
                assertSixteenPair(a, b)
                for c in all {
                    assertSixteenTriple(a, b, c)
                }
            }
        }
    }

    func assertSixteenPair(_ a: BelnapGeneralizedState, _ b: BelnapGeneralizedState) {
        #expect((a & b) == a.infoMeet(b) && (a | b) == a.infoJoin(b))
        #expect(a.infoMeet(b) == b.infoMeet(a) && a.infoJoin(b) == b.infoJoin(a))
        #expect(a.truthMeet(b) == b.truthMeet(a) && a.truthJoin(b) == b.truthJoin(a))
        #expect(a.falsityMeet(b) == b.falsityMeet(a) && a.falsityJoin(b) == b.falsityJoin(a))
        #expect(a.infoJoin(a.infoMeet(b)) == a && a.infoMeet(a.infoJoin(b)) == a)
        #expect(a.truthJoin(a.truthMeet(b)) == a && a.truthMeet(a.truthJoin(b)) == a)
        #expect(a.falsityJoin(a.falsityMeet(b)) == a && a.falsityMeet(a.falsityJoin(b)) == a)
        #expect(a.truthJoin(b).truthNegated == a.truthNegated.truthMeet(b.truthNegated))
        #expect(a.truthMeet(b).truthNegated == a.truthNegated.truthJoin(b.truthNegated))
        #expect(a.falsityJoin(b).falsityNegated == a.falsityNegated.falsityMeet(b.falsityNegated))
        #expect(a.falsityMeet(b).falsityNegated == a.falsityNegated.falsityJoin(b.falsityNegated))
        #expect(a.infoJoin(b).infoNegated == a.infoNegated.infoMeet(b.infoNegated))
        #expect(a.infoMeet(b).infoNegated == a.infoNegated.infoJoin(b.infoNegated))
        #expect(a.falsityJoin(b).truthNegated == a.truthNegated.falsityJoin(b.truthNegated))
        #expect(a.truthJoin(b).falsityNegated == a.falsityNegated.truthJoin(b.falsityNegated))
    }

    func assertSixteenTriple(
        _ a: BelnapGeneralizedState,
        _ b: BelnapGeneralizedState,
        _ c: BelnapGeneralizedState
    ) {
        #expect(a.infoJoin(b).infoJoin(c) == a.infoJoin(b.infoJoin(c)))
        #expect(a.infoMeet(b).infoMeet(c) == a.infoMeet(b.infoMeet(c)))
        #expect(a.truthJoin(b).truthJoin(c) == a.truthJoin(b.truthJoin(c)))
        #expect(a.truthMeet(b).truthMeet(c) == a.truthMeet(b.truthMeet(c)))
        #expect(a.falsityJoin(b).falsityJoin(c) == a.falsityJoin(b.falsityJoin(c)))
        #expect(a.falsityMeet(b).falsityMeet(c) == a.falsityMeet(b.falsityMeet(c)))
        #expect(a.truthMeet(b.infoJoin(c)) == a.truthMeet(b).infoJoin(a.truthMeet(c)))
        #expect(a.truthJoin(b.infoMeet(c)) == a.truthJoin(b).infoMeet(a.truthJoin(c)))
        #expect(a.falsityMeet(b.infoJoin(c)) == a.falsityMeet(b).infoJoin(a.falsityMeet(c)))
        #expect(a.falsityJoin(b.infoMeet(c)) == a.falsityJoin(b).infoMeet(a.falsityJoin(c)))
        #expect(a.truthMeet(b.falsityJoin(c)) == a.truthMeet(b).falsityJoin(a.truthMeet(c)))
        #expect(a.falsityMeet(b.truthJoin(c)) == a.falsityMeet(b).truthJoin(a.falsityMeet(c)))
    }
}

@Suite("ResponseConfidence refutation channel, ConfidenceProbe draft hooks, and Belnap classification (#360)")
struct BelnapConfidenceTests {
    @Test("ResponseConfidence classifies grounded true, underdetermined neither, draft-refuted false, and conflicted both")
    func classification() throws {
        var grounded = ResponseConfidence(min: 0.82, mean: 0.90)
        grounded.meanEntropy = 0.2
        #expect(grounded.draftRefutationRate == 0.0)
        #expect(grounded.belnapState() == .true)

        var underdetermined = ResponseConfidence(min: 0.18, mean: 0.26)
        underdetermined.meanEntropy = 3.2
        #expect(underdetermined.belnapState() == .neither)

        let refuted = ResponseConfidence(
            min: 0.12,
            mean: 0.28,
            highConfidenceDraftRejections: 4,
            totalHighConfidenceDrafts: 5
        )
        #expect(abs(refuted.draftRefutationRate - 0.8) < 1e-9)
        #expect(refuted.belnapState() == .false)

        let draftConflicted = ResponseConfidence(
            min: 0.72,
            mean: 0.86,
            highConfidenceDraftRejections: 3,
            totalHighConfidenceDrafts: 4
        )
        #expect(draftConflicted.belnapState() == .both)

        let internalDropConflicted = ResponseConfidence(min: 0.05, mean: 0.80)
        #expect(internalDropConflicted.belnapState() == .both)

        let merged = refuted.merged(with: draftConflicted, weightSelf: 1, weightOther: 1)
        #expect(merged.highConfidenceDraftRejections == 7 && merged.totalHighConfidenceDrafts == 9)

        let roundTripped = try JSONDecoder().decode(ResponseConfidence.self, from: JSONEncoder().encode(refuted))
        #expect(roundTripped == refuted)
        let legacy = try JSONDecoder().decode(ResponseConfidence.self, from: Data(#"{"min":0.2,"mean":0.6}"#.utf8))
        #expect(legacy.highConfidenceDraftRejections == 0 && legacy.totalHighConfidenceDrafts == 0)
    }

    @Test("ConfidenceProbe and ConfidenceProbeRecorder accumulate high-confidence draft rejections through processor chains")
    func probeDraftHooks() throws {
        let probe = ConfidenceProbe()
        probe.record(0.88, entropy: 0.3, margin: 0.75)
        probe.record(0.84, entropy: 0.4, margin: 0.70)
        let anchored = PromptAnchoredProcessor(
            base: ConfidenceProbeRecorder(probe: probe),
            prompt: MLXArray([Int32(1), Int32(2)])
        )
        ConfidenceProbeRecorder.recordDraftRound(through: anchored, accepted: 0, drafted: 4)
        ConfidenceProbeRecorder.recordDraftRound(through: anchored, accepted: 0, drafted: 3)
        ConfidenceProbeRecorder.recordDraftRound(through: anchored, accepted: 2, drafted: 2)
        let summary = try #require(probe.summary)
        #expect(summary.highConfidenceDraftRejections == 2)
        #expect(summary.totalHighConfidenceDrafts == 4)
        #expect(abs(summary.draftRefutationRate - 0.5) < 1e-9)
        #expect(summary.belnapState() == .both)
    }
}

@Suite("Introspector multi-run Belnap hypothesis infoJoin reducer (#360)")
struct BelnapIntrospectorTests {
    @Test("Hypothesis classifies outcomes, decodes legacy JSON, and merges runs commutatively and idempotently via infoJoin")
    func multiRunReducer() throws {
        let openH = Hypothesis(statement: "layer 2 routes math to expert 3", test: "trace_run", outcome: "open")
        let confirmedH = Hypothesis(statement: "layer 2 routes math to expert 3", test: "trace_run", outcome: "confirmed, p = 0.91")
        let refutedH = Hypothesis(statement: "layer 2 routes math to expert 3", test: "intervene ablate_expert", outcome: "refuted on holdout")

        #expect(openH.evidenceState == .neither)
        #expect(confirmedH.evidenceState == .true)
        #expect(refutedH.evidenceState == .false)
        #expect(openH.infoJoin(confirmedH).evidenceState == .true)
        #expect(confirmedH.infoJoin(openH) == openH.infoJoin(confirmedH))

        let conflicted = confirmedH.infoJoin(refutedH)
        #expect(conflicted.evidenceState == .both)
        #expect(conflicted == refutedH.infoJoin(confirmedH))
        #expect(conflicted.infoJoin(confirmedH) == conflicted)
        #expect(conflicted.infoJoin(refutedH) == conflicted)
        #expect(openH.infoJoin(confirmedH).infoJoin(refutedH) == openH.infoJoin(confirmedH.infoJoin(refutedH)))

        var reportA = IntrospectReport(goal: "study router")
        reportA.hypotheses = [confirmedH, Hypothesis(statement: "head 1 copies prefix", outcome: "open")]
        var reportB = IntrospectReport(goal: "study router")
        reportB.hypotheses = [Hypothesis(statement: "head 1 copies prefix", outcome: "confirmed"), refutedH]

        let ab = IntrospectReport.reduceHypotheses([reportA, reportB])
        let ba = IntrospectReport.reduceHypotheses([reportB, reportA])
        let aba = IntrospectReport.reduceHypotheses([reportA, reportB, reportA])
        #expect(ab == ba && ab == aba)
        #expect(ab.map(\.evidenceState) == [.true, .both])

        let legacyJSON = Data(#"{"statement":"s","test":"t","outcome":"refuted"}"#.utf8)
        let decoded = try JSONDecoder().decode(Hypothesis.self, from: legacyJSON)
        #expect(decoded.evidenceState == .false)
    }
}

@Suite("BelnapDistribution telemetry, telemetry_query MCP confidence section, and OpenMetrics export (#360)")
struct BelnapTelemetryTests {
    @Test("ConfidenceTelemetry and HealthTelemetry track BelnapDistribution and export OpenMetrics and telemetry_query JSON")
    func telemetryAndExport() async throws {
        let tracker = TelemetryTracker()
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: tracker)
        try await orchestrator.loadModel(path: "/models/chat")

        let samples: [ResponseConfidence] = [
            ResponseConfidence(min: 0.85, mean: 0.92),
            ResponseConfidence(min: 0.20, mean: 0.25),
            ResponseConfidence(min: 0.15, mean: 0.28, highConfidenceDraftRejections: 3, totalHighConfidenceDrafts: 4),
            ResponseConfidence(min: 0.78, mean: 0.88, highConfidenceDraftRejections: 3, totalHighConfidenceDrafts: 4),
        ]
        for sample in samples {
            await engine.setMockConfidence(sample)
            _ = try await orchestrator.generate(request: GenerationRequest(prompt: "Hi", maxTokens: 4).withConfidenceRecording())
        }

        let snapshot = await tracker.snapshot()
        let expected = BelnapDistribution(neither: 1, trueCount: 1, falseCount: 1, both: 1)
        #expect(snapshot.confidence?.belnap == expected)
        #expect(snapshot.health?.belnap == expected)

        let metrics = TelemetryExport.openMetrics(snapshot)
        #expect(metrics.contains("basalt_confidence_belnap_total{state=\"neither\"} 1\n"))
        #expect(metrics.contains("basalt_confidence_belnap_total{state=\"true\"} 1\n"))
        #expect(metrics.contains("basalt_confidence_belnap_total{state=\"false\"} 1\n"))
        #expect(metrics.contains("basalt_confidence_belnap_total{state=\"both\"} 1\n"))

        let lines = MCPLines()
        let server = MCPServer(orchestrator: orchestrator, writer: { lines.append($0) })
        await server.dispatch(toolCall(1, "telemetry_query", ["sections": ["confidence", "health"]]))
        let payload = try lines.toolText(0)
        let confidenceJSON = try #require(payload["confidence"] as? [String: Any])
        let belnapJSON = try #require(confidenceJSON["belnap"] as? [String: Int])
        let healthJSON = try #require(payload["health"] as? [String: Any])
        let healthBelnapJSON = try #require(healthJSON["belnap"] as? [String: Int])
        #expect(belnapJSON == ["neither": 1, "true": 1, "false": 1, "both": 1])
        #expect(healthBelnapJSON == ["neither": 1, "true": 1, "false": 1, "both": 1])

        let legacyConfidence = Data(#"{"responses":1,"min_probability":[1,0,0,0,0,0,0,0,0,0],"mean_probability":[1,0,0,0,0,0,0,0,0,0],"mean_entropy":[1,0,0,0,0,0,0,0,0,0],"min_margin":[1,0,0,0,0,0,0,0,0,0]}"#.utf8)
        let decodedLegacy = try JSONDecoder().decode(ConfidenceTelemetry.self, from: legacyConfidence)
        #expect(decodedLegacy.belnap == BelnapDistribution())
    }
}

@Suite("X-ray TraceReport late-layer flip rate and Belnap evidenceState (#360)", .serialized)
struct BelnapXRayTraceTests {
    @Test("TraceReport.evaluateEvidence classifies stable support, uncertainty, late-layer flip refutation, and high-confidence late flips")
    func evaluateEvidenceUnit() throws {
        func score(_ id: Int, _ p: Float) -> TokenScore { TokenScore(token: "t\(id)", id: id, probability: p) }
        func layer(_ idx: Int, _ id: Int, _ p: Float) -> LayerTrace {
            LayerTrace(index: idx, hiddenNorm: 1.0, lens: [score(id, p)])
        }

        let grounded = try #require(TraceReport.evaluateEvidence(
            layers: [layer(0, 9, 0.2), layer(1, 5, 0.7), layer(2, 5, 0.8), layer(3, 5, 0.9)],
            final: [score(5, 0.9)], totalLayers: 4
        ))
        #expect(grounded.lateLayerFlipRate == 0.0 && grounded.evidenceState == .true)

        let uncertain = try #require(TraceReport.evaluateEvidence(
            layers: [layer(0, 5, 0.1), layer(1, 5, 0.15), layer(2, 5, 0.18), layer(3, 5, 0.2)],
            final: [score(5, 0.2)], totalLayers: 4
        ))
        #expect(uncertain.lateLayerFlipRate == 0.0 && uncertain.evidenceState == .neither)

        let refuted = try #require(TraceReport.evaluateEvidence(
            layers: [layer(0, 2, 0.4), layer(1, 2, 0.45), layer(2, 2, 0.48), layer(3, 5, 0.3)],
            final: [score(5, 0.3)], totalLayers: 4
        ))
        #expect(refuted.lateLayerFlipRate == 1.0 && refuted.evidenceState == .false)

        let conflicted = try #require(TraceReport.evaluateEvidence(
            layers: [layer(0, 2, 0.6), layer(1, 2, 0.75), layer(2, 2, 0.8), layer(3, 5, 0.85)],
            final: [score(5, 0.85)], totalLayers: 4
        ))
        #expect(conflicted.lateLayerFlipRate == 1.0 && conflicted.evidenceState == .both)

        let withoutLens = try #require(TraceReport.evaluateEvidence(
            layers: [LayerTrace(index: 0, hiddenNorm: 1.0)],
            final: [score(5, 0.8)], totalLayers: 1
        ))
        #expect(withoutLens.lateLayerFlipRate == nil && withoutLens.evidenceState == .true)
    }

    @Test("TinyModel engine.trace populates lateLayerFlipRate, evidencePair, and evidenceState in a single forward pass")
    func tinyModelTraceEvidence() async throws {
        let engine = try await TinyModel.engine(.qwen35)
        let report = try await engine.trace(TraceRequest(prompt: "the cat sat on the mat"))
        let flipRate = try #require(report.lateLayerFlipRate)
        let pair = try #require(report.evidencePair)
        let state = try #require(report.evidenceState)
        #expect((0.0 ... 1.0).contains(flipRate))
        #expect(abs(pair.refutation - flipRate) < 1e-9)
        #expect(BelnapState.project(pair) == state)

        var noLensRequest = TraceRequest(prompt: "the cat sat on the mat", topK: 3)
        noLensRequest.logitLens = false
        let noLensReport = try await engine.trace(noLensRequest)
        #expect(noLensReport.lateLayerFlipRate == nil)
        #expect(noLensReport.evidencePair != nil && noLensReport.evidenceState != nil)
    }
}
