import ArgumentParser
import Foundation
import MLX
import SwiftUI
import Testing
@testable import BasaltApp
@testable import BasaltAppCore
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference

@Suite("Domain-Aware QAT STE & Post-Quantization Quality, Fairness, and Leakage Audit (#616)")
struct QATAndQuantizationAuditTests {
    @Test("QuantizationAuditCore passes healthy quantization and blocks perplexity, accuracy, fairness disparity, and leakage regressions")
    func quantizationAuditGateEvaluation() throws {
        let healthy = QuantizationAuditInput(
            baselinePerplexity: 4.0,
            quantizedPerplexity: 4.3,
            baselineAccuracy: 0.90,
            quantizedAccuracy: 0.89,
            fairnessGroups: [
                FairnessGroupObservation(group: "group_a", baselineScore: 0.90, quantizedScore: 0.89),
                FairnessGroupObservation(group: "group_b", baselineScore: 0.88, quantizedScore: 0.86),
            ],
            leakageProbes: [
                LeakageProbeObservation(promptId: "p1", memberLogprob: -2.0, nonMemberLogprob: -2.02),
            ]
        )
        let ok = try QuantizationAuditCore.evaluate(healthy)
        #expect(ok.passed)
        #expect(ok.failures.isEmpty)
        #expect(ok.perplexityDeltaRatio < 0.15)
        #expect(ok.maxGroupDisparity < 0.08)

        let bad = QuantizationAuditInput(
            baselinePerplexity: 4.0,
            quantizedPerplexity: 5.5,
            baselineAccuracy: 0.90,
            quantizedAccuracy: 0.80,
            fairnessGroups: [
                FairnessGroupObservation(group: "group_a", baselineScore: 0.92, quantizedScore: 0.90),
                FairnessGroupObservation(group: "group_b", baselineScore: 0.88, quantizedScore: 0.70),
            ],
            leakageProbes: [
                LeakageProbeObservation(promptId: "canary", memberLogprob: -0.1, nonMemberLogprob: -3.0),
            ]
        )
        let failed = try QuantizationAuditCore.evaluate(bad)
        #expect(!failed.passed)
        #expect(failed.failures.count == 4)
        #expect(failed.maxGroupDisparity > 0.15)
        #expect(failed.dispersionDisparity > 0.15)
        #expect(failed.leakageSignal > 0.5)

        let roundTrip = try JSONDecoder().decode(QuantizationAuditVerdict.self, from: JSONEncoder().encode(failed))
        #expect(roundTrip == failed)
    }

    @Test("QuantizationAuditCore validates thresholds and rejects invalid inputs")
    func auditValidation() throws {
        let validInput = QuantizationAuditInput(
            baselinePerplexity: 3.0, quantizedPerplexity: 3.1, baselineAccuracy: 0.9, quantizedAccuracy: 0.89
        )
        #expect(throws: BasaltError.self) {
            _ = try QuantizationAuditCore.evaluate(validInput, thresholds: .init(maxPerplexityDeltaRatio: -0.1))
        }
        #expect(throws: BasaltError.self) {
            _ = try QuantizationAuditCore.evaluate(validInput, thresholds: .init(maxAccuracyDrop: 1.5))
        }
        #expect(throws: BasaltError.self) {
            _ = try QuantizationAuditCore.evaluate(validInput, thresholds: .init(maxGroupDisparity: -0.2))
        }
        #expect(throws: BasaltError.self) {
            _ = try QuantizationAuditCore.evaluate(validInput, thresholds: .init(maxLeakageSignal: -1.0))
        }
        #expect(throws: BasaltError.self) {
            _ = try QuantizationAuditCore.evaluate(
                QuantizationAuditInput(baselinePerplexity: 0, quantizedPerplexity: 3.0, baselineAccuracy: 0.9, quantizedAccuracy: 0.9)
            )
        }
        #expect(throws: BasaltError.self) {
            _ = try QuantizationAuditCore.evaluate(
                QuantizationAuditInput(baselinePerplexity: 3.0, quantizedPerplexity: 3.0, baselineAccuracy: 1.2, quantizedAccuracy: 0.9)
            )
        }
    }

    @Test("QATBudgetConfig allocates more QAT budget on domain-specialized checkpoints than generic ones (Pillet et al. 2026)")
    func qatDomainArbitrageSchedule() throws {
        let domainCfg = QATBudgetConfig(bits: 4, groupSize: 64, totalStepBudget: 100, isDomainSpecializedCheckpoint: true)
        let genericCfg = QATBudgetConfig(bits: 4, groupSize: 64, totalStepBudget: 100, isDomainSpecializedCheckpoint: false)
        let dSched = try domainCfg.recommendedSchedule()
        let gSched = try genericCfg.recommendedSchedule()
        #expect(dSched.qatSteps > gSched.qatSteps)
        #expect(dSched.domainAdaptationSteps < gSched.domainAdaptationSteps)
        #expect(dSched.domainAdaptationSteps + dSched.qatSteps == 100)

        #expect(throws: BasaltError.self) { _ = try QATBudgetConfig(bits: 5).recommendedSchedule() }
        #expect(throws: BasaltError.self) { _ = try QATBudgetConfig(groupSize: 16).recommendedSchedule() }
        #expect(throws: BasaltError.self) { _ = try QATBudgetConfig(totalStepBudget: 1).recommendedSchedule() }
    }

    @Test("QATFakeQuantizer STE gradient decreases 4-bit fake-quantized MSE loss over training steps")
    func qatFakeQuantizerSTEDecreasesLoss() throws {
        MLXRandom.seed(42)
        let trueW = MLXRandom.normal([16, 64]).asType(.float32) * 0.25
        let x = MLXRandom.normal([32, 64]).asType(.float32)
        let target = matmul(x, trueW.transposed())
        var w = trueW + MLXRandom.normal([16, 64]).asType(.float32) * 0.15
        eval(trueW, x, target, w)

        let snr = QATFakeQuantizer.quantizationSQNRdB(
            original: trueW, quantized: QATFakeQuantizer.fakeQuantize(trueW, bits: 4, groupSize: 64)
        )
        #expect(snr > 14.0)

        var firstLoss: Float = 0
        var lastLoss: Float = 0
        for step in 0 ..< 40 {
            let out = QATFakeQuantizer.qatStep(weight: w, input: x, target: target, bits: 4, groupSize: 64, learningRate: 0.1)
            w = out.updatedWeight
            if step == 0 { firstLoss = out.loss }
            lastLoss = out.loss
        }
        #expect(lastLoss < firstLoss * 0.4)
    }

    @MainActor
    @Test("Universal Converter parity: CLI flags, Codable request/result, and ConvertModelSheet Light/Dark/RU with Localizable.strings")
    func universalConverterQATAuditParity() throws {
        let cmd = try ConvertSubcommand.parse([
            "--to", "macaq:mxfp4", "--qat-refine", "--audit", "audit.json", "--out", "/tmp/out", "org/model",
        ])
        #expect(cmd.qatRefine)
        #expect(cmd.auditSuitePath == "audit.json")

        var req = ModelConversionRequest(source: "org/m", outputId: "local/m-4bit", target: .macaqMXFP4)
        req.qatRefine = true
        req.auditSuitePath = "audit.json"
        let decReq = try JSONDecoder().decode(ModelConversionRequest.self, from: JSONEncoder().encode(req))
        #expect(decReq.qatRefine && decReq.auditSuitePath == "audit.json")

        let verdict = QuantizationAuditVerdict(
            perplexityDeltaRatio: 0.04, accuracyDrop: 0.01,
            maxGroupDisparity: 0.02, dispersionDisparity: 0.02, leakageSignal: 0.01, failures: []
        )
        let convRes = ModelConversionResult(
            outputId: "local/m-4bit", bytesBefore: 1000, bytesAfter: 300, shards: 1, audit: verdict
        )
        let decRes = try JSONDecoder().decode(ModelConversionResult.self, from: JSONEncoder().encode(convRes))
        #expect(decRes.audit == verdict)

        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let en = try #require(
            NSDictionary(contentsOf: repoRoot.appending(path: "Localization/en.lproj/Localizable.strings")) as? [String: String])
        let ru = try #require(
            NSDictionary(contentsOf: repoRoot.appending(path: "Localization/ru.lproj/Localizable.strings")) as? [String: String])
        #expect(Set(en.keys) == Set(ru.keys))
        for key in ["QAT Fake-Quant Refinement (Domain STE)", "Post-Quantization Quality & Fairness Audit"] {
            #expect(en[key]?.isEmpty == false && ru[key]?.isEmpty == false)
        }

        var state = LibraryState()
        let model = LibraryModel(id: "google/gemma-4-12B-it", sizeBytes: 24_000_000_000, isLoaded: false)
        state.models = [model]
        var draft = ConversionDraft(sourceId: model.id, sourceBytes: model.sizeBytes)
        draft.qatRefine = true
        draft.runPostQuantAudit = true
        let draftReq = draft.toRequest()
        #expect(draftReq.qatRefine && draftReq.auditSuitePath == "default")
        state.convertDraft = draft
        let env = LibraryEnvironment(client: { nil }, savePending: { _ in })
        let store = LibraryStore(state, reduce: LibraryReducer.reduce, handle: LibraryEffects.handler(env))

        let outDir = ProcessInfo.processInfo.environment["BASALT_GUI_OUT"]
        for (name, variant) in [("light", TraitVariant.lightStandard), ("dark", .darkStandard), ("dark-ru_RU", .darkRussian)] {
            let bmp = try OffscreenSceneRenderer.render(
                ConvertModelSheet(store: store), spec: .init(id: .libraryMixed, width: 580, height: 690), variant: variant)
            #expect(bmp.width == 1_160 && bmp.height == 1_380 && !bmp.png.isEmpty)
            if let outDir {
                let dirURL = URL(fileURLWithPath: outDir)
                try FileManager.default.createDirectory(at: dirURL, withIntermediateDirectories: true)
                try bmp.png.write(to: dirURL.appending(path: "gui_616_convert_qat_audit__\(name).png"))
            }
        }
    }
}
