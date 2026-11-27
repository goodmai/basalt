import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("DepthDiagnosticsMath CPU oracle, report aggregation, and MCP argument parsing (#856)")
struct DepthDiagnosticsMathOracleTests {
    @Test(
        "Effective rank returns 0 for degenerate spectra, 1 for rank-1, and k for k equal singular values",
        arguments: [1, 2, 4, 8, 16]
    )
    func effectiveRankUniformAndDegenerate(k: Int) {
        #expect(DepthDiagnosticsMath.effectiveRank(singularValues: []) == 0.0)
        #expect(DepthDiagnosticsMath.effectiveRank(singularValues: [0.0, 0.0, -3.0, .nan, .infinity]) == 0.0)
        #expect(abs(DepthDiagnosticsMath.effectiveRank(singularValues: [5.0]) - 1.0) < 1e-12)
        #expect(abs(DepthDiagnosticsMath.effectiveRank(singularValues: [5.0, 0.0, -1.0]) - 1.0) < 1e-12)
        let uniform = Array(repeating: 3.5, count: k)
        #expect(abs(DepthDiagnosticsMath.effectiveRank(singularValues: uniform) - Double(k)) < 1e-12)
        let huge = Array(repeating: 1e150, count: k)
        let tiny = Array(repeating: 1e-150, count: k)
        #expect(abs(DepthDiagnosticsMath.effectiveRank(singularValues: huge) - Double(k)) < 1e-12)
        #expect(abs(DepthDiagnosticsMath.effectiveRank(singularValues: tiny) - Double(k)) < 1e-12)
    }

    @Test("Effective rank matches analytic Shannon entropy exponential and decreases with faster spectrum decay")
    func effectiveRankDecayAndOracle() {
        let spectrum = [4.0, 2.0, 1.0, 1.0]
        let probs = [0.5, 0.25, 0.125, 0.125]
        let expectedEntropy = -probs.reduce(0.0) { $0 + $1 * Foundation.log($1) }
        let expectedRank = Foundation.exp(expectedEntropy)
        #expect(abs(DepthDiagnosticsMath.effectiveRank(singularValues: spectrum) - expectedRank) < 1e-12)

        let flat = DepthDiagnosticsMath.effectiveRank(singularValues: [1.0, 1.0, 1.0, 1.0])
        let mild = DepthDiagnosticsMath.effectiveRank(singularValues: [1.0, 0.5, 0.25, 0.125])
        let steep = DepthDiagnosticsMath.effectiveRank(singularValues: [1.0, 0.05, 0.01, 0.001])
        let rankOne = DepthDiagnosticsMath.effectiveRank(singularValues: [1.0, 0.0, 0.0, 0.0])
        #expect(flat > mild && mild > steep && steep > rankOne)
        #expect(abs(rankOne - 1.0) < 1e-12)
    }

    @Test("Residual norm growth and attention-to-MLP ratio handle nominal, identity, zero, and non-finite inputs")
    func residualGrowthAndAttnToMLPRatio() {
        #expect(abs(DepthDiagnosticsMath.residualNormGrowth(normIn: 2.0, normOut: 5.0) - 2.5) < 1e-12)
        #expect(abs(DepthDiagnosticsMath.residualNormGrowth(normIn: 4.0, normOut: 4.0) - 1.0) < 1e-12)
        #expect(DepthDiagnosticsMath.residualNormGrowth(normIn: 0.0, normOut: 0.0) == 0.0)
        #expect(abs(DepthDiagnosticsMath.residualNormGrowth(normIn: 0.0, normOut: 1.0, eps: 1e-6) - 1e6) < 1e-6)
        #expect(DepthDiagnosticsMath.residualNormGrowth(normIn: .nan, normOut: 2.0) == 0.0)

        #expect(abs(DepthDiagnosticsMath.attnToMLPRatio(attnNorm: 3.0, mlpNorm: 6.0) - 0.5) < 1e-12)
        #expect(DepthDiagnosticsMath.attnToMLPRatio(attnNorm: 0.0, mlpNorm: 0.0) == 0.0)
        #expect(abs(DepthDiagnosticsMath.attnToMLPRatio(attnNorm: 2.0, mlpNorm: 0.0, eps: 1e-6) - 2e6) < 1e-6)
        #expect(DepthDiagnosticsMath.attnToMLPRatio(attnNorm: 1.0, mlpNorm: .infinity) == 0.0)
    }

    @Test("Softcap saturation fraction at softcap 30.0 and threshold 0.95 matches |tanh(z/30)| >= 0.95")
    func softcapSaturationOracle() {
        let softcap = 30.0
        let threshold = 0.95
        let mixed = [0.0, 20.0, 54.0, 56.0, -60.0, -.infinity, .nan]
        let frac = DepthDiagnosticsMath.softcapSaturationFraction(
            logits: mixed, softcap: softcap, thresholdFraction: threshold
        )
        #expect(abs(frac - 0.4) < 1e-12)
        #expect(DepthDiagnosticsMath.softcapSaturationFraction(logits: [0.0, 10.0, -25.0], softcap: softcap, thresholdFraction: threshold) == 0.0)
        #expect(DepthDiagnosticsMath.softcapSaturationFraction(logits: [60.0, -80.0, 120.0], softcap: softcap, thresholdFraction: threshold) == 1.0)
        #expect(DepthDiagnosticsMath.softcapSaturationFraction(logits: [], softcap: softcap, thresholdFraction: threshold) == 0.0)
        #expect(DepthDiagnosticsMath.softcapSaturationFraction(logits: [60.0], softcap: 0.0, thresholdFraction: threshold) == 0.0)
    }

    @Test("DepthDiagnosticsReport aggregates summary fields and round-trips through JSON")
    func reportAggregationAndCodableRoundTrip() throws {
        var layer0 = LayerDepthMetrics(layerIndex: 0, residualNormIn: 2.0, residualNormOut: 3.0, preSoftmaxSMax: [4.0, 6.0], effectiveRank: 8.0)
        layer0.setSublayerNorms(attention: 1.5, mlp: 3.0)
        layer0.setSoftcap(value: 30.0, saturationRate: 0.0)
        var layer1 = LayerDepthMetrics(layerIndex: 1, residualNormIn: 3.0, residualNormOut: 6.0, preSoftmaxSMax: [10.0, 8.0], effectiveRank: 5.5)
        layer1.setSublayerNorms(attention: 2.0, mlp: 1.0)
        layer1.setSoftcap(value: 30.0, saturationRate: 0.25)

        let report = DepthDiagnosticsReport(promptTokenCount: 12, layerCount: 2, layers: [layer0, layer1])
        #expect(report.maxSMaxOverall == 10.0)
        #expect(report.maxSMaxLayer == 1)
        #expect(abs(report.meanResidualGrowth - 1.75) < 1e-12)
        #expect(report.minEffectiveRank == 5.5)
        #expect(layer0.attnToMLPRatio == 0.5 && layer1.attnToMLPRatio == 2.0)

        let encoded = try JSONEncoder().encode(report)
        let decoded = try JSONDecoder().decode(DepthDiagnosticsReport.self, from: encoded)
        #expect(decoded == report)
    }

    @Test("MCP depth_diagnostics descriptor schema and argument parser validate inputs")
    func mcpToolSchemaAndParsing() throws {
        let tool = MCPToolCatalog.depthDiagnosticsTool
        #expect(tool.name == "depth_diagnostics")
        let schema = try #require(tool.inputSchema.rawValue as? [String: Any])
        let props = try #require(schema["properties"] as? [String: Any])
        #expect(props.keys.contains("prompt") && props.keys.contains("includeRank"))
        #expect(props.keys.contains("softcapThresholdFraction") && props.keys.contains("maxSequenceLength"))
        #expect(props.keys.contains("layers"))

        let defaults = try MCPToolCatalog.parseDepthDiagnosticsRequest(["prompt": "hello"])
        #expect(defaults == DepthDiagnosticsRequest(prompt: "hello"))

        let custom = try MCPToolCatalog.parseDepthDiagnosticsRequest([
            "prompt": "hello",
            "includeRank": false,
            "softcapThresholdFraction": 0.9,
            "maxSequenceLength": 128,
            "layers": [0, 2],
        ])
        #expect(custom.includeRank == false && custom.softcapThresholdFraction == 0.9)
        #expect(custom.maxSequenceLength == 128 && custom.layers == [0, 2])

        #expect(throws: MCPFailure.self) { try MCPToolCatalog.parseDepthDiagnosticsRequest([:]) }
        #expect(throws: MCPFailure.self) { try MCPToolCatalog.parseDepthDiagnosticsRequest(["prompt": ""]) }
        #expect(throws: MCPFailure.self) { try MCPToolCatalog.parseDepthDiagnosticsRequest(["prompt": "ok", "softcapThresholdFraction": 0.0]) }
        #expect(throws: MCPFailure.self) { try MCPToolCatalog.parseDepthDiagnosticsRequest(["prompt": "ok", "softcapThresholdFraction": 1.5]) }
        #expect(throws: MCPFailure.self) { try MCPToolCatalog.parseDepthDiagnosticsRequest(["prompt": "ok", "layers": [Int]()]) }
        #expect(throws: MCPFailure.self) { try MCPToolCatalog.parseDepthDiagnosticsRequest(["prompt": "ok", "maxSequenceLength": 0]) }
        #expect(throws: BasaltError.self) { try DepthDiagnosticsRequest(prompt: "ok", layers: [5]).validate(layerCount: 4) }
    }
}

@Suite("TinyModel live depth diagnostics measurement and engine integration (#856)", .serialized)
struct DepthDiagnosticsLiveModelTests {
    @Test("measureDepthDiagnostics on TinyModel.make(.qwen3) reports finite growth, positive S_max, rank >= 1, and zero weight mutation")
    func tinyQwen3EndToEndAndZeroMutation() throws {
        let model = try TinyModel.make(.qwen3)
        let beforeWeights = snapshotWeights(of: model)
        let tokens = MLXArray(Array(0 ..< 16).map(Int32.init)).reshaped(1, -1)

        let report = try MLXInferenceEngine.measureDepthDiagnostics(
            model: model,
            tokens: tokens,
            includeRank: true,
            softcapThresholdFraction: 0.95,
            layerFilter: nil
        )

        assertWeightsUnchanged(before: beforeWeights, after: snapshotWeights(of: model))
        #expect(report.promptTokenCount == 16)
        #expect(report.layerCount == 4)
        #expect(report.layers.map(\.layerIndex) == [0, 1, 2, 3])
        #expect(report.maxSMaxOverall > 0)
        #expect(report.maxSMaxOverall.isFinite)
        #expect((0 ..< 4).contains(report.maxSMaxLayer))
        #expect(report.meanResidualGrowth > 0)
        #expect(report.meanResidualGrowth.isFinite)
        let minRank = try #require(report.minEffectiveRank)
        #expect(minRank >= 1.0)
        #expect(minRank <= 16.0)
        for layer in report.layers {
            try assertValidQwen3Layer(layer, maxRank: 16.0)
        }
    }

    private func assertWeightsUnchanged(
        before: [(name: String, values: [Float])],
        after: [(name: String, values: [Float])]
    ) {
        #expect(before.count == after.count)
        for (lhs, rhs) in zip(before, after) {
            #expect(lhs.name == rhs.name)
            #expect(lhs.values == rhs.values)
        }
    }

    private func assertValidQwen3Layer(_ layer: LayerDepthMetrics, maxRank: Double) throws {
        #expect(layer.residualNormIn > 0)
        #expect(layer.residualNormIn.isFinite)
        #expect(layer.residualNormOut > 0)
        #expect(layer.residualNormOut.isFinite)
        #expect(layer.residualNormGrowth > 0)
        #expect(layer.residualNormGrowth.isFinite)
        #expect(layer.preSoftmaxSMax.count == 4)
        #expect(layer.preSoftmaxSMax.allSatisfy { $0.isFinite && $0 > 0 })
        #expect(layer.maxPreSoftmaxLogit > 0)
        #expect(layer.softcapValue == nil)
        #expect(layer.softcapSaturationRate == nil)
        let attnNorm = try #require(layer.attentionContributionNorm)
        let mlpNorm = try #require(layer.mlpContributionNorm)
        let ratio = try #require(layer.attnToMLPRatio)
        #expect(attnNorm > 0)
        #expect(mlpNorm > 0)
        #expect(ratio > 0)
        #expect(ratio.isFinite)
        let rank = try #require(layer.effectiveRank)
        #expect(rank >= 1.0)
        #expect(rank <= maxRank)
    }

    @Test("measureDepthDiagnostics supports 1D token input, layer filtering, and includeRank=false")
    func layerFilteringAndRankDisabled() throws {
        let model = try TinyModel.make(.qwen3)
        let tokens = MLXArray(Array(0 ..< 12).map(Int32.init))
        let report = try MLXInferenceEngine.measureDepthDiagnostics(
            model: model,
            tokens: tokens,
            includeRank: false,
            softcapThresholdFraction: 0.95,
            layerFilter: [3, 1]
        )
        #expect(report.promptTokenCount == 12)
        #expect(report.layerCount == 4)
        #expect(report.layers.map(\.layerIndex) == [1, 3])
        #expect(report.minEffectiveRank == nil)
        #expect(report.layers.allSatisfy { $0.effectiveRank == nil && $0.preSoftmaxSMax.count == 4 })
        #expect(throws: BasaltError.self) { try MLXInferenceEngine.measureDepthDiagnostics(model: model, tokens: tokens, layerFilter: [99]) }
    }

    @Test("MLXInferenceEngine.runDepthDiagnostics runs on TinyModel.engine(.qwen3) with maxSequenceLength cap and leaves no overlays")
    func engineDepthDiagnosticsEndToEnd() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let request = DepthDiagnosticsRequest(
            prompt: "The quick brown fox jumps over the lazy dog",
            includeRank: true,
            softcapThresholdFraction: 0.95,
            maxSequenceLength: 8,
            layers: [0, 2]
        )
        let report = try await engine.runDepthDiagnostics(request)
        #expect(report.promptTokenCount > 0 && report.promptTokenCount <= 8)
        #expect(report.layerCount == 4)
        #expect(report.layers.map(\.layerIndex) == [0, 2])
        #expect(report.layers.allSatisfy { ($0.effectiveRank ?? 0) >= 1.0 && $0.maxPreSoftmaxLogit > 0 })
        #expect(await engine.overlays().isEmpty)

        let badRequest = DepthDiagnosticsRequest(prompt: "hello", layers: [99])
        await #expect(throws: BasaltError.self) { try await engine.runDepthDiagnostics(badRequest) }
    }

    private func snapshotWeights(of model: any LanguageModel) -> [(name: String, values: [Float])] {
        model.parameters().flattened()
            .sorted { $0.0 < $1.0 }
            .map { ($0.0, $0.1.asType(.float32).asArray(Float.self)) }
    }
}
