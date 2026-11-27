import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("MoE load-balance math computes exact analytical metrics, collapse alerts, and finite edge cases (#859)")
struct MoELoadBalanceTelemetryTests {
    @Test("Uniform expert distribution achieves maximal normalized entropy, zero inequality, and no collapse alert")
    func uniformDistribution() {
        let metrics = MoELoadBalanceMath.evaluate(counts: [100, 100, 100, 100])
        #expect(metrics.expertCount == 4)
        #expect(metrics.totalDispatches == 400)
        #expect(abs(metrics.routerEntropy - Foundation.log(4.0)) < 1e-12)
        #expect(abs(metrics.normalizedRouterEntropy - 1.0) < 1e-12)
        #expect(abs(metrics.cvSquared - 0.0) < 1e-12)
        #expect(abs(metrics.gini - 0.0) < 1e-12)
        #expect(abs(metrics.maxToMeanRatio - 1.0) < 1e-12)
        #expect(metrics.deadExpertCount == 0)
        #expect(abs(metrics.deadExpertFraction - 0.0) < 1e-12)
        #expect(metrics.collapseAlert == false)
    }

    @Test("Complete single-expert collapse reaches zero entropy, maximum CV squared, Gini 0.75, max-to-mean 4, and triggers collapse alert")
    func completeCollapse() {
        let metrics = MoELoadBalanceMath.evaluate(counts: [400, 0, 0, 0])
        #expect(metrics.expertCount == 4)
        #expect(metrics.totalDispatches == 400)
        #expect(abs(metrics.routerEntropy - 0.0) < 1e-12)
        #expect(abs(metrics.normalizedRouterEntropy - 0.0) < 1e-12)
        #expect(abs(metrics.cvSquared - 3.0) < 1e-12)
        #expect(abs(metrics.gini - 0.75) < 1e-12)
        #expect(abs(metrics.maxToMeanRatio - 4.0) < 1e-12)
        #expect(metrics.deadExpertCount == 3)
        #expect(abs(metrics.deadExpertFraction - 0.75) < 1e-12)
        #expect(metrics.collapseAlert == true)
    }

    @Test("Bimodal distribution hits half normalized entropy, unit CV squared, Gini 0.5, max-to-mean 2, and two dead experts")
    func bimodalDistribution() {
        let metrics = MoELoadBalanceMath.evaluate(counts: [200, 200, 0, 0])
        #expect(metrics.expertCount == 4)
        #expect(metrics.totalDispatches == 400)
        #expect(abs(metrics.routerEntropy - Foundation.log(2.0)) < 1e-12)
        #expect(abs(metrics.normalizedRouterEntropy - 0.5) < 1e-12)
        #expect(abs(metrics.cvSquared - 1.0) < 1e-12)
        #expect(abs(metrics.gini - 0.5) < 1e-12)
        #expect(abs(metrics.maxToMeanRatio - 2.0) < 1e-12)
        #expect(metrics.deadExpertCount == 2)
        #expect(abs(metrics.deadExpertFraction - 0.5) < 1e-12)
        #expect(metrics.collapseAlert == true)
    }

    @Test("Zipf-like skewed histogram matches independent Double reference formulas including pairwise Gini")
    func zipfSkewedReference() {
        let counts = [80, 40, 20, 10, 5, 5, 0, 0]
        let metrics = MoELoadBalanceMath.evaluate(counts: counts)
        let values = counts.map(Double.init)
        let expertCount = Double(values.count)
        let total = values.reduce(0.0, +)
        let mean = total / expertCount
        let expectedEntropy = values.reduce(0.0) { acc, value in
            guard value > 0 else { return acc }
            let probability = value / total
            return acc - probability * Foundation.log(probability)
        }
        let expectedNormalized = expectedEntropy / Foundation.log(expertCount)
        let secondMoment = values.reduce(0.0) { $0 + $1 * $1 } / expertCount
        let expectedCVSquared = secondMoment / (mean * mean) - 1.0
        var pairwiseDiffSum = 0.0
        for left in values {
            for right in values {
                pairwiseDiffSum += abs(left - right)
            }
        }
        let expectedGini = pairwiseDiffSum / (2.0 * expertCount * total)
        let expectedMaxToMean = (values.max() ?? 0.0) / mean

        #expect(metrics.expertCount == 8)
        #expect(metrics.totalDispatches == 160)
        #expect(abs(metrics.routerEntropy - expectedEntropy) < 1e-12)
        #expect(abs(metrics.normalizedRouterEntropy - expectedNormalized) < 1e-12)
        #expect(abs(metrics.cvSquared - expectedCVSquared) < 1e-12)
        #expect(abs(metrics.gini - expectedGini) < 1e-12)
        #expect(abs(metrics.maxToMeanRatio - expectedMaxToMean) < 1e-12)
        #expect(metrics.deadExpertCount == 2)
        #expect(abs(metrics.deadExpertFraction - 0.25) < 1e-12)
        #expect(metrics.collapseAlert == true)
    }

    @Test("Collapse alert triggers independently on entropy, dead-expert fraction, or max-to-mean ratio")
    func collapseAlertClauses() {
        let healthy = MoELoadBalanceMath.evaluate(counts: [30, 28, 22, 20])
        #expect(healthy.collapseAlert == false)

        let entropyOnly = MoELoadBalanceMath.evaluate(
            counts: [30, 28, 22, 20],
            collapseEntropyThreshold: 0.999,
            collapseDeadFractionThreshold: 1.0
        )
        #expect(entropyOnly.collapseAlert == true)

        let deadOnly = MoELoadBalanceMath.evaluate(counts: [100, 100, 100, 0])
        #expect(deadOnly.normalizedRouterEntropy > 0.5)
        #expect(deadOnly.maxToMeanRatio < 2.0)
        #expect(deadOnly.deadExpertFraction == 0.25)
        #expect(deadOnly.collapseAlert == true)
        let deadRelaxed = MoELoadBalanceMath.evaluate(
            counts: [100, 100, 100, 0],
            collapseDeadFractionThreshold: 0.3
        )
        #expect(deadRelaxed.collapseAlert == false)

        let skewOnly = MoELoadBalanceMath.evaluate(counts: [100, 34, 33, 33])
        #expect(skewOnly.normalizedRouterEntropy > 0.5)
        #expect(skewOnly.deadExpertCount == 0)
        #expect(abs(skewOnly.maxToMeanRatio - 2.0) < 1e-12)
        #expect(skewOnly.collapseAlert == true)
    }
}

@Suite("MoE load-balance handles edge cases, sparse dispatch maps, Codable round-trip, and OpenMetrics (#859)")
struct MoELoadBalanceEdgeAndExportTests {
    @Test("Empty and all-zero histograms stay strictly finite with zero ratios and no false collapse alert")
    func emptyAndAllZeroCounts() {
        let empty = MoELoadBalanceMath.evaluate(counts: [])
        #expect(empty.expertCount == 0)
        #expect(empty.totalDispatches == 0)
        #expect(empty.routerEntropy == 0.0)
        #expect(empty.normalizedRouterEntropy == 0.0)
        #expect(empty.cvSquared == 0.0)
        #expect(empty.gini == 0.0)
        #expect(empty.maxToMeanRatio == 0.0)
        #expect(empty.deadExpertCount == 0)
        #expect(empty.deadExpertFraction == 0.0)
        #expect(empty.collapseAlert == false)

        let zeros = MoELoadBalanceMath.evaluate(counts: [0, 0, 0, 0])
        #expect(zeros.expertCount == 4)
        #expect(zeros.totalDispatches == 0)
        #expect(zeros.deadExpertCount == 4)
        #expect(zeros.deadExpertFraction == 1.0)
        #expect(zeros.routerEntropy == 0.0)
        #expect(zeros.normalizedRouterEntropy == 0.0)
        #expect(zeros.cvSquared == 0.0)
        #expect(zeros.gini == 0.0)
        #expect(zeros.maxToMeanRatio == 0.0)
        #expect(zeros.collapseAlert == false)
    }

    @Test("Single-expert and negative-count edge cases stay finite and clamp negative counts to zero")
    func singleExpertAndNegativeCounts() {
        let single = MoELoadBalanceMath.evaluate(counts: [42])
        #expect(single.expertCount == 1)
        #expect(single.totalDispatches == 42)
        #expect(single.routerEntropy == 0.0)
        #expect(single.normalizedRouterEntropy == 1.0)
        #expect(single.cvSquared == 0.0)
        #expect(single.gini == 0.0)
        #expect(single.maxToMeanRatio == 1.0)
        #expect(single.deadExpertCount == 0)
        #expect(single.deadExpertFraction == 0.0)
        #expect(single.collapseAlert == false)

        let clamped = MoELoadBalanceMath.evaluate(counts: [-10, 200, -5, 200])
        #expect(clamped == MoELoadBalanceMath.evaluate(counts: [0, 200, 0, 200]))

        let allNegative = MoELoadBalanceMath.evaluate(counts: [-3, -1])
        #expect(allNegative == MoELoadBalanceMath.evaluate(counts: [0, 0]))
    }

    @Test("Sparse dispatch map conversion matches dense evaluation and ignores out-of-range or non-positive entries")
    func sparseDispatchMap() {
        let dense = MoELoadBalanceMath.evaluate(counts: [80, 40, 20, 10, 5, 5, 0, 0])
        let map: [Int: Int] = [-1: 99, 0: 80, 1: 40, 2: 20, 3: 10, 4: 5, 5: 5, 6: -3, 8: 77]
        #expect(MoELoadBalanceMath.evaluate(dispatchMap: map, expertCount: 8) == dense)
        #expect(MoELoadBalanceMath.evaluate(dispatchMap: [0: 10], expertCount: 0) == MoELoadBalanceMath.evaluate(counts: []))
    }

    @Test("MoELoadBalanceMetrics round-trips through JSON Codable with snake_case keys")
    func codableRoundTrip() throws {
        let collapsed = MoELoadBalanceMath.evaluate(counts: [400, 0, 0, 0])
        let encoded = try JSONEncoder().encode(collapsed)
        let json = try #require(String(data: encoded, encoding: .utf8))
        #expect(json.contains("\"expert_count\""))
        #expect(json.contains("\"total_dispatches\""))
        #expect(json.contains("\"router_entropy\""))
        #expect(json.contains("\"normalized_router_entropy\""))
        #expect(json.contains("\"cv_squared\""))
        #expect(json.contains("\"gini\""))
        #expect(json.contains("\"max_to_mean_ratio\""))
        #expect(json.contains("\"dead_expert_count\""))
        #expect(json.contains("\"dead_expert_fraction\""))
        #expect(json.contains("\"collapse_alert\""))
        let decoded = try JSONDecoder().decode(MoELoadBalanceMetrics.self, from: encoded)
        #expect(decoded == collapsed)
    }

    @Test("TelemetryExport renders all five OpenMetrics gauges sorted by layer index")
    func openMetricsFormatting() {
        let uniform = MoELoadBalanceMath.evaluate(counts: [100, 100, 100, 100])
        let collapsed = MoELoadBalanceMath.evaluate(counts: [400, 0, 0, 0])
        let text = TelemetryExport.renderMoELoadBalance([(layer: 2, metrics: collapsed), (layer: 0, metrics: uniform)])
        #expect(text.contains("# TYPE basalt_expert_layer_router_entropy gauge\n"))
        #expect(text.contains("# TYPE basalt_expert_layer_cv_squared gauge\n"))
        #expect(text.contains("# TYPE basalt_expert_layer_gini gauge\n"))
        #expect(text.contains("# TYPE basalt_expert_layer_max_to_mean gauge\n"))
        #expect(text.contains("# TYPE basalt_expert_layer_dead_experts gauge\n"))
        #expect(text.contains("basalt_expert_layer_router_entropy{layer=\"2\"} 0\n"))
        #expect(text.contains("basalt_expert_layer_cv_squared{layer=\"0\"} 0\n"))
        #expect(text.contains("basalt_expert_layer_cv_squared{layer=\"2\"} 3\n"))
        #expect(text.contains("basalt_expert_layer_gini{layer=\"0\"} 0\n"))
        #expect(text.contains("basalt_expert_layer_gini{layer=\"2\"} 0.75\n"))
        #expect(text.contains("basalt_expert_layer_max_to_mean{layer=\"0\"} 1\n"))
        #expect(text.contains("basalt_expert_layer_max_to_mean{layer=\"2\"} 4\n"))
        #expect(text.contains("basalt_expert_layer_dead_experts{layer=\"0\"} 0\n"))
        #expect(text.contains("basalt_expert_layer_dead_experts{layer=\"2\"} 3\n"))
        #expect(TelemetryExport.renderMoELoadBalance([]).isEmpty)

        var routing = ExpertRouting()
        routing.record(module: 0, [0, 1, 2, 3])
        routing.record(module: 2, [0, 0, 0, 0])
        let paired = TelemetryExport.renderMoELoadBalance(
            routing.layers(),
            metricsByLayer: [0: uniform, 2: collapsed]
        )
        #expect(paired == text)
    }
}

@Suite("Live Qwen3 MoE tiny model router dispatches evaluate cleanly through MoELoadBalanceMath and OpenMetrics (#859)", .serialized)
struct MoELoadBalanceLiveModelTests {
    private static func captureQwen3MoERouting(maxTokens: Int) async throws -> (routing: ExpertRouting, tokens: Int) {
        let container = try TinyModel.container(.qwen3Moe)
        return try await container.perform { context in
            let promptText = "the cat sat on the mat"
            let prompt = context.tokenizer.encode(text: promptText, addSpecialTokens: true)
            let request = TokenTraceRequest(prompt: promptText, maxTokens: maxTokens)
            let recorder = TokenTraceRecorder(request, readout: nil)
            _ = try MLXInferenceEngine.tracedGreedy(prompt, maxTokens: maxTokens, recorder: recorder, context: context)
            let layout = try #require(recorder.layout)
            let (floats, ints) = try #require(recorder.table())
            let table = TokenTraceTable(
                layout: layout,
                tokens: floats.dim(0),
                floats: floats.asArray(Float.self),
                ints: ints.asArray(Int32.self)
            )
            var routing = ExpertRouting()
            for layer in table.layers {
                guard let rows = table.routed(layer: layer) else { continue }
                for row in rows {
                    routing.record(module: layer, row.ids.map(Int.init))
                }
            }
            return (routing: routing, tokens: table.tokens)
        }
    }

    @Test("Real Qwen3 MoE router selections across all four layers produce finite load-balance metrics and OpenMetrics gauges")
    func liveQwen3MoELoadBalance() async throws {
        let observed = try await Self.captureQwen3MoERouting(maxTokens: 8)
        let expertCount = 8
        let topK = 2
        let layers = observed.routing.layers()
        #expect(layers.map(\.layer) == [0, 1, 2, 3])

        var metricsByLayer: [Int: MoELoadBalanceMetrics] = [:]
        for layerSummary in layers {
            let counts = try #require(observed.routing.counts[layerSummary.layer])
            let metrics = MoELoadBalanceMath.evaluate(dispatchMap: counts, expertCount: expertCount)
            metricsByLayer[layerSummary.layer] = metrics
            #expect(metrics.expertCount == expertCount)
            #expect(metrics.totalDispatches == observed.tokens * topK)
            #expect(metrics.routerEntropy.isFinite)
            #expect(metrics.routerEntropy >= 0.0)
            #expect((0.0 ... 1.0).contains(metrics.normalizedRouterEntropy))
            #expect(metrics.cvSquared.isFinite)
            #expect(metrics.cvSquared >= 0.0)
            #expect((0.0 ... 1.0).contains(metrics.gini))
            #expect((1.0 ... Double(expertCount)).contains(metrics.maxToMeanRatio))
            #expect((0 ..< expertCount).contains(metrics.deadExpertCount))
            #expect(abs(metrics.deadExpertFraction - Double(metrics.deadExpertCount) / Double(expertCount)) < 1e-12)
        }

        let exported = TelemetryExport.renderMoELoadBalance(layers, metricsByLayer: metricsByLayer)
        for layerIndex in [0, 1, 2, 3] {
            #expect(exported.contains("basalt_expert_layer_router_entropy{layer=\"\(layerIndex)\"}"))
            #expect(exported.contains("basalt_expert_layer_cv_squared{layer=\"\(layerIndex)\"}"))
            #expect(exported.contains("basalt_expert_layer_gini{layer=\"\(layerIndex)\"}"))
            #expect(exported.contains("basalt_expert_layer_max_to_mean{layer=\"\(layerIndex)\"}"))
            #expect(exported.contains("basalt_expert_layer_dead_experts{layer=\"\(layerIndex)\"}"))
        }
    }
}
