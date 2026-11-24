import Foundation
import Testing
@testable import BasaltBench
@testable import BasaltCore

@Suite("MT leaderboard submission pipeline: HF model-index YAML, WMT24++/FLORES+ bundle, raw predictions, and WMT .txt (#734)")
struct MTLeaderboardSubmissionTests {
    struct SampleScores {
        let chrf: Double
        let comet: Double
        let metricX: Double
        let fidelity: Double
    }

    static func makeSample(
        id: String,
        model: String,
        pair: String,
        dataset: String,
        answer: String,
        scores: SampleScores
    ) -> BenchmarkSampleRecord {
        var sample = BenchmarkSampleRecord(
            id: id,
            adapter: "basalt:translation",
            model: model,
            prompt: "Run `basalt serve` for segment \(id).",
            answer: answer,
            score: scores.chrf / 100.0
        )
        sample.dataset = dataset
        sample.latencyMs = 120.0
        sample.passed = true
        sample.metadata = [
            "lang_pair": pair,
            "reference": "Reference translation for \(id).",
            "base_model": "google/gemma-4-12b-it",
            "predictions_repo": "goodmai/translategemma4-eval-results",
            "sacrebleu_signature": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.4.3",
            "chrf_pp": String(format: "%.2f", scores.chrf),
            "comet": String(format: "%.4f", scores.comet),
            "metricx24": String(format: "%.2f", scores.metricX),
            "tech_fidelity": String(format: "%.1f", scores.fidelity),
        ]
        return sample
    }

    static func makeArtifact(model: String, samples: [BenchmarkSampleRecord], suite: String) -> BenchmarkRunArtifact {
        let manifest = BenchmarkManifest(
            runID: "mt-dryrun-001",
            composeName: model,
            suite: suite,
            seed: 42,
            temperature: 0.0
        )
        let provenance = BasaltProvenance(
            gitCommit: "8dd686b",
            composeSHA256: BasaltProvenance.sha256Hex("mt-leaderboard"),
            chip: "M4 Pro",
            unifiedMemoryGB: 24,
            timestampISO8601: "2026-11-24T20:00:00Z"
        )
        let card = BenchmarkScorecard.evaluate(
            metrics: ["chrf_pp": 52.0, "comet": 0.85, "metricx24": 3.4, "tech_fidelity": 99.0],
            gates: nil,
            samples: samples
        )
        return BenchmarkRunArtifact(manifest: manifest, provenance: provenance, scorecard: card, samples: samples)
    }
}

extension MTLeaderboardSubmissionTests {
    @Test("End-to-end dry run on TranslateGemma-4-12B-Pilot validates HF model-index YAML, predictions.jsonl, and sanitized WMT .txt")
    func dryRunTranslateGemma12BPilotWMT24() throws {
        let samples = [
            Self.makeSample(
                id: "wmt-ja-1",
                model: "Bosalt/TranslateGemma-4-12B-Pilot",
                pair: "en-ja_JP",
                dataset: "google/wmt24pp",
                answer: "line one\nline two",
                scores: SampleScores(chrf: 34.5, comet: 0.8712, metricX: 2.95, fidelity: 100.0)
            ),
            Self.makeSample(
                id: "wmt-ja-2",
                model: "Bosalt/TranslateGemma-4-12B-Pilot",
                pair: "en-ja_JP",
                dataset: "google/wmt24pp",
                answer: "セグメント2 `basalt serve`",
                scores: SampleScores(chrf: 36.5, comet: 0.8812, metricX: 2.55, fidelity: 100.0)
            ),
            Self.makeSample(
                id: "wmt-ru-1",
                model: "Bosalt/TranslateGemma-4-12B-Pilot",
                pair: "en-ru_RU",
                dataset: "google/wmt24pp",
                answer: "Запустите `basalt serve` для сегмента.",
                scores: SampleScores(chrf: 48.9, comet: 0.8214, metricX: 4.10, fidelity: 98.5)
            ),
        ]
        let artifact = Self.makeArtifact(model: "Bosalt/TranslateGemma-4-12B-Pilot", samples: samples, suite: "wmt24pp")
        let submit = BenchmarkSubmitSpec(target: .mtLeaderboard, apiKeyEnv: "HF_TOKEN", subset: "google/wmt24pp", split: "test")
        let outDir = FileManager.default.temporaryDirectory.appendingPathComponent("mt-734-12b-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: outDir) }

        let plan = try BenchmarkSubmitter.exportDryRun(target: .mtLeaderboard, artifact: artifact, submit: submit, outputDirectory: outDir)
        #expect(plan.isValid)
        #expect(plan.requiredEnvVar == "HF_TOKEN")

        let readme = try String(contentsOf: outDir.appendingPathComponent("README.md"), encoding: .utf8)
        #expect(HFModelIndexRenderer.validateModelCard(readme).isEmpty)
        #expect(readme.contains("model-index:"))
        #expect(readme.contains("WMT24++"))
        #expect(readme.contains("google/wmt24pp"))
        #expect(readme.contains("MetricX-24") && readme.contains("COMET") && readme.contains("chrF++") && readme.contains("tech_fidelity"))
        #expect(readme.contains("nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.4.3"))

        let jaWmt = try String(contentsOf: outDir.appendingPathComponent("wmt/en-ja_JP.txt"), encoding: .utf8)
        let jaLines = jaWmt.split(separator: "\n")
        #expect(jaLines.count == 2)
        #expect(jaLines[0] == "line one line two")

        let predsRaw = try String(contentsOf: outDir.appendingPathComponent("predictions.jsonl"), encoding: .utf8)
        let predLines = predsRaw.split(separator: "\n")
        #expect(predLines.count == 3)
        let firstObj = try #require(JSONSerialization.jsonObject(with: Data(predLines[0].utf8)) as? [String: Any])
        #expect(firstObj["lang_pair"] as? String == "en-ja_JP")
        #expect((firstObj["reference"] as? String)?.isEmpty == false)
        #expect(firstObj["metricx24"] as? Double != nil)

        let summaryData = try Data(contentsOf: outDir.appendingPathComponent("eval_summary.json"))
        let decodedSpec = try JSONDecoder().decode(MTModelCardSpec.self, from: summaryData)
        #expect(decodedSpec.results.count == 2)
        #expect(decodedSpec.languages == ["en", "ja", "ru"])
    }

    @Test("FLORES+ submission target reuses pipeline for TranslateGemma-4-26B-A4B across all 6 UN languages")
    func floresPlusSubmissionForTranslateGemma26B() throws {
        let pairs = ["en-ar_EG", "en-zh_CN", "en-fr_FR", "en-ru_RU", "en-es_MX", "en-ja_JP"]
        let samples = pairs.enumerated().map { idx, pair in
            Self.makeSample(
                id: "flores-\(idx)",
                model: "Bosalt/TranslateGemma-4-26B-A4B-v2",
                pair: pair,
                dataset: "openlanguagedata/flores_plus",
                answer: "Translated \(pair) output",
                scores: SampleScores(chrf: 45.0 + Double(idx), comet: 0.82 + Double(idx) * 0.01, metricX: 3.5, fidelity: 100.0)
            )
        }
        let artifact = Self.makeArtifact(model: "Bosalt/TranslateGemma-4-26B-A4B-v2", samples: samples, suite: "flores_plus")
        let submit = BenchmarkSubmitSpec(target: .mtLeaderboard, subset: "flores_plus", split: "devtest")
        let plan = SubmissionFormatter.build(target: .mtLeaderboard, artifact: artifact, submit: submit)
        #expect(plan.isValid)
        let readmeFile = try #require(plan.files.first { $0.relativePath == "README.md" })
        let readme = String(decoding: readmeFile.data, as: UTF8.self)
        #expect(readme.contains("FLORES+") && readme.contains("openlanguagedata/flores_plus") && readme.contains("devtest"))
        for pair in pairs {
            #expect(plan.files.contains { $0.relativePath == "wmt/\(pair).txt" })
        }
        #expect(TranslationBenchmarkAdapter().defaultSubmissionTarget == .mtLeaderboard)
    }

    @Test("HFModelIndexRenderer validator rejects malformed frontmatter, empty model names, and non-finite metric values")
    func modelCardValidatorRejectsMalformedCards() {
        #expect(!HFModelIndexRenderer.validateModelCard("# No frontmatter").isEmpty)
        let badSpec = MTModelCardSpec(
            modelName: "",
            decoding: MTDecodingInfo(),
            results: [
                MTBenchmarkResult(
                    datasetName: "WMT24++",
                    datasetType: "google/wmt24pp",
                    datasetConfig: "en-ru_RU",
                    split: "test",
                    metrics: [MTBenchmarkMetric(name: "COMET", type: "comet", value: .nan)]
                ),
            ]
        )
        let rendered = HFModelIndexRenderer.renderModelCard(badSpec)
        let errors = HFModelIndexRenderer.validateModelCard(rendered)
        #expect(errors.count >= 2)
    }
}
