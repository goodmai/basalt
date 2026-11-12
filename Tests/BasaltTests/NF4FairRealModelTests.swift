import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("NF4 g32 against affine 4-bit g64, both converted from the same bf16 checkpoint at 4.5 bits per weight: decode, perplexity and golden-logit KL between the arms (#357)", .serialized)
struct NF4FairRealModelTests {
    static let models = NF4Run.models("BASALT_E2E_NF4_FAIR_MODELS")
    static let output = URL(fileURLWithPath: NF4Run.env["BASALT_E2E_NF4_OUT"] ?? "/private/tmp/basalt-nf4")
    static let arms: [(label: String, precision: WeightConverter.Precision)] = [("affine-g64", .affine(bits: 4, groupSize: 64)), ("nf4-g32", .nf4(groupSize: 32))]

    static func converted(_ source: String, _ arm: (label: String, precision: WeightConverter.Precision), log: ModelRunLog) async throws -> String {
        let directory = output.appendingPathComponent(log.model.replacingOccurrences(of: "/", with: "--") + "-" + arm.label)
        if !FileManager.default.fileExists(atPath: directory.path) {
            let summary = try await log.step("convert \(arm.label)") {
                try WeightConverter(precision: arm.precision).run(source: URL(fileURLWithPath: source), output: directory)
            }
            log.record("convert", "\(directory.path): \(summary.outputTensors) tensors, \(summary.bytesBefore) → \(summary.bytesAfter) bytes")
        }
        let inspection = await ModelInspector().inspect(modelIdOrPath: directory.path)
        log.record("inspect \(arm.label)", "intact \(inspection.intact), \(inspection.totalSizeFormatted), \(inspection.bitWidthDescription)")
        try #require(inspection.intact)
        return directory.path
    }

    @Test("Equal bits per weight from bf16: nf4 g32 perplexity is no worse than affine g64 and its decode reaches 0.95x",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_NF4_FAIR_MODELS to bf16 checkpoints"), arguments: models)
    func fair(entry: String) async throws {
        let source = NF4Run.path(entry)
        let log = try await ModelRunLog("nf4-fair", modelPath: source)
        defer { log.releaseGPU() }
        try #require(NF4Run.sourceBits(source) == nil, "\(entry) is quantized; pass a bf16 checkpoint")
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 1 << 30))
        var measured: [NF4Run.Arm] = []
        for arm in Self.arms {
            measured.append(try await NF4Run.measure(engine, path: try await Self.converted(source, arm, log: log), arm: arm.label, log: log))
        }
        let (affine, nf4) = (measured[0], measured[1])
        let kl = GoldenFixture(revision: "affine-g64", tolerance: .infinity, prompts: affine.traces).compare(nf4.traces)
        log.record("RESULT", "nf4 g32 against affine g64 from bf16: decode \(nf4.speed) / \(affine.speed) = \(nf4.speed / affine.speed)x, perplexity \(affine.perplexity) → \(nf4.perplexity), golden \(kl)")
        #expect(nf4.perplexity <= affine.perplexity, "perplexity \(nf4.perplexity) against \(affine.perplexity)")
        #expect(nf4.speed >= 0.95 * affine.speed, "decode \(nf4.speed) against \(affine.speed) tok/s")
    }
}
