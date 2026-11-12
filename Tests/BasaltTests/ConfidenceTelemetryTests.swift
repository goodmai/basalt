import Testing
import Foundation
import MLX
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCLI

@Suite("X-ray telemetry phase 3: per-token entropy and top-two margin next to the top-token probability, sampled every Nth generation, aggregated into deciles (#450)")
struct ConfidenceTelemetryTests {
    @Test("Logit statistics give the top probability, the entropy in nats and the gap to the runner-up")
    func statistics() throws {
        let uniform = try #require(ConfidenceProbeRecorder.statistics(logits: MLXArray([Float](repeating: 0, count: 4))))
        #expect(abs(uniform.probability - 0.25) < 1e-5 && abs(uniform.entropy - log(4)) < 1e-4 && uniform.margin < 1e-6)
        let peaked = try #require(ConfidenceProbeRecorder.statistics(logits: MLXArray([10, 0, 0, 0] as [Float])))
        #expect(peaked.probability > 0.99 && peaked.entropy < 0.01 && peaked.margin > 0.99)
        #expect(ConfidenceProbeRecorder.statistics(logits: MLXArray([Float.nan, 0])) == nil)
    }

    @Test("A probe summarizes entropy and margin, merging keeps the weighted mean and the smallest margin, and old JSON still decodes")
    func summary() throws {
        let probe = ConfidenceProbe()
        probe.record(0.9, entropy: 0.4, margin: 0.8)
        probe.record(0.5, entropy: 1.2, margin: 0.1)
        let summary = try #require(probe.summary)
        #expect(summary.min == 0.5 && abs((summary.meanEntropy ?? 0) - 0.8) < 1e-9 && summary.minMargin == 0.1)
        var other = ResponseConfidence(min: 0.7, mean: 0.7)
        (other.meanEntropy, other.minMargin) = (2.0, 0.05)
        let merged = summary.merged(with: other, weightSelf: 1, weightOther: 1)
        #expect(abs((merged.meanEntropy ?? 0) - 1.4) < 1e-9 && merged.minMargin == 0.05)
        #expect(try JSONDecoder().decode(ResponseConfidence.self, from: Data(#"{"min":0.2,"mean":0.6}"#.utf8)).meanEntropy == nil)
    }

    @Test("Every Nth generation is measured, a request that asks is always measured, and deciles count what came back")
    func sampling() async throws {
        let telemetry = TelemetryTracker()
        await telemetry.sampleConfidence(every: 3)
        var flags: [Bool] = []
        for _ in 0 ..< 6 { flags.append(await telemetry.sampled(GenerationRequest(prompt: "x", maxTokens: 1)).recordConfidence) }
        #expect(flags == [false, false, true, false, false, true])
        #expect(await telemetry.sampled(GenerationRequest(prompt: "x", maxTokens: 1).withConfidenceRecording()).recordConfidence)
        var aggregate = ConfidenceTelemetry()
        var measured = ResponseConfidence(min: 0.35, mean: 0.81)
        (measured.meanEntropy, measured.minMargin) = (1.3, 0.92)
        aggregate.record(measured)
        #expect(aggregate.responses == 1 && aggregate.minProbability[3] == 1 && aggregate.meanProbability[8] == 1 && aggregate.meanEntropy[2] == 1 && aggregate.minMargin[9] == 1)
        var snapshot = ServerTelemetrySnapshot()
        snapshot.confidence = aggregate
        #expect(TelemetryExport.openMetrics(snapshot).contains("basalt_confidence_mean_entropy{bin=\"2\"} 1"))
        #expect(throws: (any Error).self) { _ = try ServeCommand.parse(["--telemetry-confidence-every", "-1"]) }
    }
}
