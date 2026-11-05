import Testing
import Foundation
@testable import BasaltCore

@Suite("Roofline calibration against this machine's measured profile (#272)", .serialized)
struct RooflineCalibrationTests {
    static let enabled = ProcessInfo.processInfo.environment["BASALT_ROOFLINE_CALIBRATION"] == "1"
    static let tolerance = 0.25
    static let decodeModels = ["mlx-community/gemma-4-26B-A4B-it-qat-4bit", "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly"]
    static let prefillModels = decodeModels + ["LiquidAI/LFM2.5-24B-A2B-MLX-4bit"]

    static func definition(_ repo: String) -> ModelDefinition {
        ModelDefinition(id: repo, name: "", family: .qwen, task: .chat, modality: .text,
                        spec: .init(ramMB: 0, contextWindow: 4096, quantization: repo, quality: 0))
    }

    @Test("Decode estimate within 25% of the measured tok/s", .enabled(if: enabled, "set BASALT_ROOFLINE_CALIBRATION=1"),
          arguments: decodeModels)
    func decode(repo: String) async throws {
        let resources = await SystemProfiler().detectResources()
        let measured = try #require(try HardwareBenchmarkProfileStore.loadChecked().measurement(for: repo, currentChip: resources.chipModel),
                                    "no measurement of \(repo) on \(resources.chipModel) in \(HardwareBenchmarkProfileStore.defaultProfilePath)")
        let model = Self.definition(repo)
        let estimate = FitScorer.decodeSpeed(model: model, bits: FitScorer.parseBits(from: repo),
                                             contextTokens: (measured.promptTokens ?? 0) + 128,
                                             resources: resources, profile: HardwareBenchmarkProfile()).tps
        let ratio = estimate / measured.measuredTPS
        print(String(format: "decode %@ on %@: estimate %.1f tok/s, measured %.1f tok/s, ratio %.3f",
                     repo, resources.chipModel, estimate, measured.measuredTPS, ratio))
        #expect(abs(ratio - 1) <= Self.tolerance)
    }

    @Test("Prefill estimate from the other models within 25% of the measured prompt tok/s",
          .enabled(if: enabled, "set BASALT_ROOFLINE_CALIBRATION=1"), arguments: prefillModels)
    func prefill(repo: String) async throws {
        let resources = await SystemProfiler().detectResources()
        var profile = try HardwareBenchmarkProfileStore.loadChecked()
        let measured = try #require(profile.measurement(for: repo, currentChip: resources.chipModel)?.prefillTokensPerSecond,
                                    "no prefill measurement of \(repo) with at least \(BenchmarkMeasurement.minimumPrefillTokens) prompt tokens")
        profile.measurements = profile.measurements.filter { $0.value.modelId.lowercased() != repo.lowercased() }
        let estimate = try #require(FitScorer.prefillSpeed(model: Self.definition(repo), resources: resources, profile: profile),
                                    "no other model with a prefill measurement on \(resources.chipModel)")
        let ratio = estimate.tps / measured
        print(String(format: "prefill %@ on %@: estimate %.0f tok/s, measured %.0f tok/s, ratio %.3f",
                     repo, resources.chipModel, estimate.tps, measured, ratio))
        #expect(!estimate.measured)
        #expect(abs(ratio - 1) <= Self.tolerance)
    }
}
