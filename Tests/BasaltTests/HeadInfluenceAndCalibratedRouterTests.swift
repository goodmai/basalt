import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Head Influence Score (Bouger 2026) and Temperature-Scaled Business Router Gate (#615)")
struct HeadInfluenceAndCalibratedRouterTests {
    @Test("HeadInfluenceCore fuses residual share and logit direction alignment across layers and flags prompt-injection anomalies")
    func headInfluenceScoringAndAnomalyDetection() throws {
        var layer0 = TokenLayerPathway(
            layer: 0,
            topHeads: [
                TokenHeadActivation(head: 0, norm: 9.5, attentionEntropy: 0.2),
                TokenHeadActivation(head: 1, norm: 0.5, attentionEntropy: 1.5),
            ]
        )
        layer0.lensTopTokenId = 99
        layer0.lensTopProbability = 0.90

        var layer1 = TokenLayerPathway(
            layer: 1,
            topHeads: [
                TokenHeadActivation(head: 2, norm: 3.0, attentionEntropy: 0.6),
                TokenHeadActivation(head: 3, norm: 2.0, attentionEntropy: 0.8),
            ]
        )
        layer1.lensTopTokenId = 42
        layer1.lensTopProbability = 0.88

        let enriched = HeadInfluenceCore.enrich(pathways: [layer1, layer0], sampledId: 42, finalProbability: 0.90)
        #expect(enriched.map(\.layer) == [0, 1])

        let h0 = try #require(enriched[0].topHeads.first)
        #expect(h0.head == 0)
        #expect(try #require(h0.residualShare) > 0.9)
        #expect(try #require(h0.logitAlignment) < 0.2)
        #expect(try #require(h0.influenceScore) > 0.5)
        #expect(enriched[1].layerInfluence != nil)

        var tok = OutputTokenTelemetry(index: 0, token: "ok", tokenId: 42, logprob: -1.8, entropy: 3.2, margin: 0.1)
        tok.layers = enriched
        let report = try #require(HeadInfluenceCore.summarizeNetwork(tokens: [tok], topK: 3))
        #expect(report.layers.count == 2)
        #expect(report.topInfluentialHeads.count == 3)
        #expect(report.injectionAnomalyScore >= 0.65)
        #expect(report.anomalyFlagged)

        let encoded = try JSONEncoder().encode(report)
        let decoded = try JSONDecoder().decode(NetworkInfluenceReport.self, from: encoded)
        #expect(decoded == report)
    }

    @Test("HeadInfluenceCore handles empty pathways, zero norms, and grounded low-anomaly tokens cleanly")
    func headInfluenceEdgeCasesAndGroundedSignal() throws {
        #expect(HeadInfluenceCore.enrich(pathways: [], sampledId: 1, finalProbability: 0.8).isEmpty)
        #expect(HeadInfluenceCore.summarizeNetwork(tokens: []) == nil)

        let zeroLayer = TokenLayerPathway(
            layer: 0,
            topHeads: [
                TokenHeadActivation(head: 0, norm: 0.0),
                TokenHeadActivation(head: 1, norm: 0.0),
            ]
        )
        let enriched = HeadInfluenceCore.enrich(pathways: [zeroLayer], sampledId: 7, finalProbability: 0.95)
        let head = try #require(enriched.first?.topHeads.first)
        #expect(abs((head.residualShare ?? 0) - 0.5) < 1e-4)
        #expect((head.logitAlignment ?? 0) > 0.8)

        var groundedTok = OutputTokenTelemetry(index: 0, token: "Paris", tokenId: 7, logprob: -0.05, entropy: 0.1, margin: 0.92)
        groundedTok.layers = enriched
        let summary = AttributedTokenTraceSummary(tokens: [groundedTok])
        let net = try #require(summary.networkInfluence)
        #expect(!net.anomalyFlagged)
        #expect(net.injectionAnomalyScore < 0.5)
    }

    @Test("MLXInferenceEngine.attributeTokens populates per-head Score d'influence and summary network_influence on TinyModel qwen3")
    func tinyModelTokenAttributionPopulatesInfluenceScores() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        var req = AttributedTokenTraceRequest(prompt: "hello", maxTokens: 2, temperature: 0.0, topLogprobs: 3, topHeads: 2)
        req.includeAttentionEntropy = true
        let report = try await engine.attributeTokens(req)
        #expect(!report.tokens.isEmpty)
        let first = try #require(report.tokens.first)
        let pathways = try #require(first.layers)
        #expect(!pathways.isEmpty)
        let head = try #require(pathways.first?.topHeads.first)
        #expect(head.residualShare != nil)
        #expect(head.logitAlignment != nil)
        #expect(head.influenceScore != nil)
        #expect(pathways.first?.layerInfluence != nil)
        let net = try #require(report.summary.networkInfluence)
        #expect(net.layers.count == pathways.count)
        #expect(!net.topInfluentialHeads.isEmpty)
        #expect(net.injectionAnomalyScore >= 0.0 && net.injectionAnomalyScore <= 1.0)
    }

    @Test("CalibratedRouterGate fits Temperature Scaling to reduce ECE and routes fast vs reasoning vs symbolicVerified by domain and TTFT")
    func calibratedRouterTemperatureScalingAndTierEscalation() throws {
        var samples: [CalibrationSample] = []
        for idx in 0 ..< 40 {
            let correct = idx % 5 != 0
            let scores = correct ? [6.0, 0.0, -2.0] : [5.0, 0.0, -2.0]
            let gold = correct ? 0 : 1
            samples.append(CalibrationSample(
                stateHash: "s-\(idx)", questionID: "q-\(idx)", kind: .choice, rawScores: scores, goldIndex: gold
            ))
        }
        let (gate, fit) = CalibratedRouterGate.calibrate(samples: samples, holdoutFraction: 0.25)
        #expect(fit.eceAfter < fit.eceBefore)
        #expect(gate.config.temperature > 1.0)

        let domains: [RouterDomainClass] = [.general, .coding, .medical]
        let fastDecision = try gate.route(rawDomainLogits: [12.0, 0.0, -4.0], domains: domains, estimatedFastTTFTSeconds: 0.35)
        #expect(fastDecision.domain == .general)
        #expect(fastDecision.tier == .fastLocal)
        #expect(!fastDecision.escalatedForRegulation)

        let slowTTFTDecision = try gate.route(rawDomainLogits: [12.0, 0.0, -4.0], domains: domains, estimatedFastTTFTSeconds: 1.5)
        #expect(slowTTFTDecision.tier == .reasoning)

        let regulatedLowConf = try gate.route(rawDomainLogits: [0.2, 0.0, 1.5], domains: domains, estimatedFastTTFTSeconds: 0.2)
        #expect(regulatedLowConf.domain == .medical)
        #expect(regulatedLowConf.tier == .symbolicVerified)
        #expect(regulatedLowConf.escalatedForRegulation)

        let regulatedHighConf = try gate.route(rawDomainLogits: [-8.0, -8.0, 15.0], domains: domains, estimatedFastTTFTSeconds: 0.2)
        #expect(regulatedHighConf.domain == .medical)
        #expect(regulatedHighConf.tier == .reasoning)
        #expect(regulatedHighConf.escalatedForRegulation)

        let candFast = gate.hybridCandidate(for: fastDecision, localLatencySeconds: 0.35)
        let candSlow = gate.hybridCandidate(for: fastDecision, localLatencySeconds: 1.5)
        #expect(candFast.degradation == 0.0)
        #expect(candSlow.degradation > 0.0)

        let arbiter = HybridArbiter(capabilityWeight: 1.0, latencyWeight: 0.2, costWeight: 0.5)
        let candSym = gate.hybridCandidate(for: regulatedLowConf, localLatencySeconds: 0.4)
        let cloud = HybridCandidate(capability: 0.98, degradation: 0.0, latencySeconds: 1.0, cost: 0.01)
        #expect(arbiter.cloudReason(local: candSym, localFits: true, cloud: cloud, remainingBudget: 10.0) == .capability)
    }

    @Test("CalibratedRouterGate rejects invalid configuration and malformed logit inputs")
    func calibratedRouterValidation() throws {
        var badTemp = CalibratedRouterConfig(temperature: 0.0)
        #expect(throws: BasaltError.self) { try badTemp.validate() }
        badTemp.temperature = 1.0
        badTemp.fastConfidenceThreshold = 1.5
        #expect(throws: BasaltError.self) { try badTemp.validate() }
        badTemp.fastConfidenceThreshold = 0.8
        badTemp.regulatedConfidenceThreshold = -0.1
        #expect(throws: BasaltError.self) { try badTemp.validate() }
        badTemp.regulatedConfidenceThreshold = 0.9
        badTemp.maxFastTTFTSeconds = 0.0
        #expect(throws: BasaltError.self) { try badTemp.validate() }

        let gate = CalibratedRouterGate()
        #expect(throws: BasaltError.self) {
            _ = try gate.route(rawDomainLogits: [1.0, 2.0], domains: [.general])
        }
        #expect(throws: BasaltError.self) {
            _ = try gate.route(rawDomainLogits: [.nan], domains: [.general])
        }
    }
}
