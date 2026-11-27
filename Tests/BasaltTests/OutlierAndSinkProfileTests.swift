import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Outlier kurtosis, crest factor, dead neurons, attention sink mass, and risk tiers (#858)")
struct OutlierAndSinkMathTests {
    @Test("Excess kurtosis is near 0 for Gaussian-like samples, >> 50 for a sharp spike, and 0 for constant or empty inputs")
    func excessKurtosisOracle() {
        let gaussian = Self.deterministicGaussianSamples(count: 8192)
        let gaussianKurtosis = OutlierAndSinkMath.excessKurtosis(gaussian)
        #expect(Swift.abs(gaussianKurtosis) < 0.3)

        var spiky = Array(repeating: 0.0, count: 1000)
        spiky[0] = 100.0
        #expect(OutlierAndSinkMath.excessKurtosis(spiky) > 50.0)

        #expect(OutlierAndSinkMath.excessKurtosis([]) == 0.0)
        #expect(OutlierAndSinkMath.excessKurtosis([4.2]) == 0.0)
        #expect(OutlierAndSinkMath.excessKurtosis(Array(repeating: 3.14, count: 64)) == 0.0)
    }

    @Test("Crest factor is 1.0 for nonzero constants, sqrt(N) for one spike in N, and 0 for empty or zero inputs",
          arguments: [4, 16, 64, 256])
    func crestFactorOracle(length: Int) {
        #expect(OutlierAndSinkMath.crestFactor([]) == 0.0)
        #expect(OutlierAndSinkMath.crestFactor(Array(repeating: 0.0, count: length)) == 0.0)

        let constant = Array(repeating: -4.5, count: length)
        #expect(Swift.abs(OutlierAndSinkMath.crestFactor(constant) - 1.0) < 1e-12)

        var spike = Array(repeating: 0.0, count: length)
        spike[length / 2] = 7.0
        let expected = Double(length).squareRoot()
        #expect(Swift.abs(OutlierAndSinkMath.crestFactor(spike) - expected) < 1e-10)
    }

    @Test("Dead-neuron fraction measures |x| < threshold and returns 0 on empty input")
    func deadNeuronFractionOracle() {
        #expect(OutlierAndSinkMath.deadNeuronFraction([]) == 0.0)
        let mixed = [0.0, -5e-4, 5e-4, 1e-3, -0.5]
        #expect(Swift.abs(OutlierAndSinkMath.deadNeuronFraction(mixed, threshold: 1e-3) - 0.6) < 1e-12)
        #expect(OutlierAndSinkMath.deadNeuronFraction([0.0, 0.0, 0.0], threshold: 1e-3) == 1.0)
        #expect(OutlierAndSinkMath.deadNeuronFraction([1.0, -2.0, 0.01], threshold: 1e-3) == 0.0)
    }

    @Test("Attention sink mass skips trivial prefix queries when longer than sinkCount and falls back for short rows")
    func attentionSinkMassOracle() {
        let empty = OutlierAndSinkMath.attentionSinkMass(weights: [], sinkCount: 1)
        #expect(empty.mean == 0.0 && empty.max == 0.0)

        let rows: [[Double]] = [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.4, 0.3, 0.3],
        ]
        let sink1 = OutlierAndSinkMath.attentionSinkMass(weights: rows, sinkCount: 1)
        #expect(Swift.abs(sink1.mean - 0.6) < 1e-12)
        #expect(Swift.abs(sink1.max - 0.8) < 1e-12)

        let sink2 = OutlierAndSinkMath.attentionSinkMass(weights: rows, sinkCount: 2)
        #expect(Swift.abs(sink2.mean - 0.7) < 1e-12)
        #expect(Swift.abs(sink2.max - 0.7) < 1e-12)

        let shortSeq = OutlierAndSinkMath.attentionSinkMass(weights: [[1.0]], sinkCount: 1)
        #expect(Swift.abs(shortSeq.mean - 1.0) < 1e-12)
        #expect(Swift.abs(shortSeq.max - 1.0) < 1e-12)
    }

    @Test("Quantizability risk tier classifies safe, watch, and sensitive thresholds")
    func riskTierClassification() {
        #expect(OutlierAndSinkMath.classifyQuantizabilityRisk(excessKurtosis: 3.0, crestFactor: 4.0) == .safe)
        #expect(OutlierAndSinkMath.classifyQuantizabilityRisk(excessKurtosis: 10.0, crestFactor: 3.0) == .watch)
        #expect(OutlierAndSinkMath.classifyQuantizabilityRisk(excessKurtosis: 2.0, crestFactor: 8.0) == .watch)
        #expect(OutlierAndSinkMath.classifyQuantizabilityRisk(excessKurtosis: 50.0, crestFactor: 2.0) == .sensitive)
        #expect(OutlierAndSinkMath.classifyQuantizabilityRisk(excessKurtosis: 2.0, crestFactor: 15.0) == .sensitive)
        #expect(OutlierAndSinkMath.classifyQuantizabilityRisk(excessKurtosis: .infinity, crestFactor: 1.0) == .sensitive)
    }

    @Test("Parallel moment merge matches single-pass excess kurtosis and crest factor across unequal batches")
    func pebayMomentMergeParity() {
        var all = Self.deterministicGaussianSamples(count: 300).map { $0 + 5.0 }
        all[17] = 45.0
        all[211] = -38.0
        let batchA = OutlierMomentAccumulator.from(values: Array(all[0 ..< 70]))
        let batchB = OutlierMomentAccumulator.from(values: Array(all[70 ..< 190]))
        let batchC = OutlierMomentAccumulator.from(values: Array(all[190 ..< 300]))
        var merged = batchA
        merged.merge(batchB)
        merged.merge(batchC)

        #expect(Swift.abs(merged.excessKurtosis - OutlierAndSinkMath.excessKurtosis(all)) < 1e-9)
        #expect(Swift.abs(merged.crestFactor - OutlierAndSinkMath.crestFactor(all)) < 1e-12)
    }

    @Test("outlier_profile MCP descriptor and argument parser accept valid args and reject malformed params")
    func mcpDescriptorAndParser() throws {
        let descriptor = MCPToolCatalog.outlierProfileDescriptor
        #expect(descriptor.name == "outlier_profile")
        let schema = try #require(descriptor.inputSchema.rawValue as? [String: Any])
        let properties = try #require(schema["properties"] as? [String: Any])
        #expect(properties.keys.contains("prompts"))
        #expect(properties.keys.contains("sinkTokenCount"))
        #expect(properties.keys.contains("deadNeuronThreshold"))
        #expect(properties.keys.contains("layers"))

        let parsed = try OutlierProfileRequest(mcp: [
            "prompts": ["hello world", "second prompt"],
            "sinkTokenCount": 2,
            "deadNeuronThreshold": 0.01,
            "layers": [0, 2],
        ])
        #expect(parsed.prompts == ["hello world", "second prompt"])
        #expect(parsed.sinkTokenCount == 2)
        #expect(Swift.abs(parsed.deadNeuronThreshold - 0.01) < 1e-12)
        #expect(parsed.layers == [0, 2])

        let defaults = try OutlierProfileRequest(mcp: ["prompts": ["single"]])
        #expect(defaults.sinkTokenCount == 1 && defaults.deadNeuronThreshold == 1e-3 && defaults.layers == nil)

        let missing = #expect(throws: MCPFailure.self) { _ = try OutlierProfileRequest(mcp: [:]) }
        #expect(missing?.code == -32602)
        #expect(throws: MCPFailure.self) { _ = try OutlierProfileRequest(mcp: ["prompts": [String]()]) }
        #expect(throws: MCPFailure.self) { _ = try OutlierProfileRequest(mcp: ["prompts": [""]]) }
        #expect(throws: MCPFailure.self) { _ = try OutlierProfileRequest(mcp: ["prompts": ["a"], "sinkTokenCount": 0]) }
        #expect(throws: MCPFailure.self) { _ = try OutlierProfileRequest(mcp: ["prompts": ["a"], "deadNeuronThreshold": -0.1]) }
        #expect(throws: MCPFailure.self) { _ = try OutlierProfileRequest(mcp: ["prompts": ["a"], "layers": [Int]()]) }
    }

    private static func deterministicGaussianSamples(count: Int) -> [Double] {
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        func nextUniform() -> Double {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            z = z ^ (z >> 31)
            return (Double(z >> 11) + 0.5) / Double(UInt64(1) << 53)
        }
        var samples: [Double] = []
        samples.reserveCapacity(count)
        for _ in 0 ..< count {
            let radius = (-2.0 * Foundation::log(nextUniform())).squareRoot()
            samples.append(radius * Foundation::cos(2.0 * Double.pi * nextUniform()))
        }
        return samples
    }
}

@Suite("TinyModel outlier profile capture, oracle parity, and zero weight mutation (#858)", .serialized)
struct OutlierProfileTinyModelTests {
    @Test("measureOutlierProfile on TinyModel.make(.qwen3) returns finite metrics, bounded sink mass and dead neurons, and leaves weights untouched")
    func tinyQwen3DirectCaptureAndZeroWeightMutation() throws {
        let model = try TinyModel.make(.qwen3)
        let beforeWeights = Self.snapshotWeights(of: model)
        let batchA = MLXArray([Int32(10), 20, 30, 40, 50, 60]).reshaped(1, -1)
        let batchB = MLXArray([Int32(5), 15, 25, 35, 45, 55, 65, 75]).reshaped(1, -1)

        let report = try MLXInferenceEngine.measureOutlierProfile(
            model: model,
            tokenBatches: [batchA, batchB],
            sinkTokenCount: 1,
            deadNeuronThreshold: 1e-3,
            layerFilter: nil
        )
        #expect(report.promptCount == 2)
        #expect(report.totalTokens == 14)
        #expect(report.layerCount == 4)
        #expect(report.layers.map(\.layerIndex) == [0, 1, 2, 3])
        #expect(report.maxKurtosisLayer != nil && report.maxCrestFactorLayer != nil)

        for layer in report.layers {
            #expect(layer.excessKurtosis.isFinite)
            #expect(layer.crestFactor.isFinite && layer.crestFactor >= 1.0 - 1e-5)
            #expect(layer.maxAbsActivation.isFinite && layer.maxAbsActivation > 0)
            #expect(layer.rmsActivation.isFinite && layer.rmsActivation > 0)
            let dead = try #require(layer.deadNeuronFraction)
            #expect((0.0 ... 1.0).contains(dead))
            let sink = try #require(layer.meanAttentionSinkMass)
            #expect((0.0 ... 1.0).contains(sink))
            #expect(layer.heads.map(\.headIndex) == [0, 1, 2, 3])
            for head in layer.heads {
                #expect((0.0 ... 1.0).contains(head.meanSinkMass))
                #expect((0.0 ... 1.0).contains(head.maxSinkMass))
                #expect(head.maxSinkMass >= head.meanSinkMass - 1e-6)
            }
        }
        #expect(Self.snapshotWeights(of: model) == beforeWeights)
    }

    @Test("measureOutlierProfile respects layerFilter and matches CPU Double oracle on Qwen3 residuals")
    func tinyQwen3LayerFilterAndOracleParity() throws {
        let model = try TinyModel.make(.qwen3)
        let tokens = MLXArray([Int32(12), 24, 36, 48, 60, 72]).reshaped(1, -1)
        let report = try MLXInferenceEngine.measureOutlierProfile(
            model: model,
            tokenBatches: [tokens],
            sinkTokenCount: 2,
            deadNeuronThreshold: 1e-3,
            layerFilter: Set([1, 3])
        )
        #expect(report.layerCount == 4)
        #expect(report.layers.map(\.layerIndex) == [1, 3])

        let probe = QKResidualProbe()
        _ = QKClipController.forward(model, inputs: tokens, probe: probe)
        let rawResidual = try #require(probe.residuals[1]).asType(.float32)
        eval(rawResidual)
        let values = rawResidual.asArray(Float.self).map(Double.init)

        let measured = try #require(report.layers.first { $0.layerIndex == 1 })
        let oracleKurtosis = OutlierAndSinkMath.excessKurtosis(values)
        let oracleCrest = OutlierAndSinkMath.crestFactor(values)
        #expect(Swift.abs(measured.excessKurtosis - oracleKurtosis) < 1e-3)
        #expect(Swift.abs(measured.crestFactor - oracleCrest) < 1e-4)
    }

    @Test("MLXInferenceEngine.runOutlierProfile runs end-to-end on TinyModel.engine(.qwen3), encodes cleanly, and rejects invalid requests")
    func tinyQwen3EngineEntryPoint() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let request = OutlierProfileRequest(
            prompts: ["the quick brown fox", "jumps over the lazy dog"],
            sinkTokenCount: 2,
            deadNeuronThreshold: 1e-3,
            layers: [0, 2]
        )
        let report = try await engine.runOutlierProfile(request)
        #expect(report.promptCount == 2)
        #expect(report.layerCount == 4)
        #expect(report.layers.map(\.layerIndex) == [0, 2])
        #expect(await engine.overlays().isEmpty)

        let encoded = try JSONEncoder().encode(report)
        let decoded = try JSONDecoder().decode(OutlierProfileReport.self, from: encoded)
        #expect(decoded == report)

        await #expect(throws: BasaltError.self) { _ = try await engine.runOutlierProfile(OutlierProfileRequest(prompts: [])) }
        await #expect(throws: BasaltError.self) { _ = try await engine.runOutlierProfile(OutlierProfileRequest(prompts: ["hi"], layers: [99])) }
        await #expect(throws: BasaltError.self) { _ = try await engine.runOutlierProfile(OutlierProfileRequest(prompts: ["hi"], sinkTokenCount: 0)) }
    }

    private static func snapshotWeights(of model: any LanguageModel) -> [String: [Float]] {
        var snapshot: [String: [Float]] = [:]
        for (key, value) in model.parameters().flattened() {
            snapshot[key] = value.asType(.float32).asArray(Float.self)
        }
        return snapshot
    }
}
