import Foundation

public struct MTBenchmarkMetric: Sendable, Codable, Equatable {
    public let name: String
    public let type: String
    public let value: Double
    public let verified: Bool

    public init(name: String, type: String, value: Double, verified: Bool = false) {
        self.name = name
        self.type = type
        self.value = value
        self.verified = verified
    }
}

public struct MTBenchmarkResult: Sendable, Codable, Equatable {
    public let datasetName: String
    public let datasetType: String
    public let datasetConfig: String
    public let split: String
    public let metrics: [MTBenchmarkMetric]

    public init(
        datasetName: String,
        datasetType: String,
        datasetConfig: String,
        split: String,
        metrics: [MTBenchmarkMetric]
    ) {
        self.datasetName = datasetName
        self.datasetType = datasetType
        self.datasetConfig = datasetConfig
        self.split = split
        self.metrics = metrics
    }
}

public struct MTModelCardSpec: Sendable, Codable, Equatable {
    public let modelName: String
    public let baseModel: String?
    public let license: String
    public let languages: [String]
    public let decoding: MTDecodingInfo
    public let results: [MTBenchmarkResult]

    public init(
        modelName: String,
        baseModel: String? = nil,
        license: String = "gemma",
        languages: [String] = ["en"],
        decoding: MTDecodingInfo,
        results: [MTBenchmarkResult]
    ) {
        self.modelName = modelName
        self.baseModel = baseModel
        self.license = license
        self.languages = languages
        self.decoding = decoding
        self.results = results
    }
}

public struct MTDecodingInfo: Sendable, Codable, Equatable {
    public let temperature: Double
    public let seed: UInt64
    public let gitCommit: String
    public let hardware: String
    public let predictionsRepo: String?
    public let sacrebleuSignature: String?

    public init(
        temperature: Double = 0.0,
        seed: UInt64 = 42,
        gitCommit: String = "",
        hardware: String = "",
        predictionsRepo: String? = nil,
        sacrebleuSignature: String? = nil
    ) {
        self.temperature = temperature
        self.seed = seed
        self.gitCommit = gitCommit
        self.hardware = hardware
        self.predictionsRepo = predictionsRepo
        self.sacrebleuSignature = sacrebleuSignature
    }
}

public enum HFModelIndexRenderer {
    public static func renderModelCard(_ spec: MTModelCardSpec) -> String {
        var lines: [String] = ["---"]
        lines.append("license: \(yamlScalar(spec.license))")
        if let base = spec.baseModel, !base.isEmpty {
            lines.append("base_model: \(yamlScalar(base))")
        }
        lines.append("pipeline_tag: translation")
        lines.append("language:")
        for lang in spec.languages {
            lines.append("  - \(yamlScalar(lang))")
        }
        lines.append("tags:")
        lines.append("  - translation")
        lines.append("  - basalt")
        lines.append("  - mlx")
        lines.append("model-index:")
        lines.append("  - name: \(yamlScalar(spec.modelName))")
        lines.append("    results:")
        for result in spec.results {
            lines.append("      - task:")
            lines.append("          type: translation")
            lines.append("          name: Machine Translation")
            lines.append("        dataset:")
            lines.append("          name: \(yamlScalar(result.datasetName))")
            lines.append("          type: \(yamlScalar(result.datasetType))")
            lines.append("          config: \(yamlScalar(result.datasetConfig))")
            lines.append("          split: \(yamlScalar(result.split))")
            lines.append("        metrics:")
            for metric in result.metrics {
                lines.append("          - name: \(yamlScalar(metric.name))")
                lines.append("            type: \(yamlScalar(metric.type))")
                lines.append("            value: \(formatValue(metric.value))")
                lines.append("            verified: \(metric.verified ? "true" : "false")")
            }
        }
        lines.append("---")
        return (lines + renderMarkdownBody(spec)).joined(separator: "\n") + "\n"
    }

    private static func renderMarkdownBody(_ spec: MTModelCardSpec) -> [String] {
        var lines: [String] = [
            "",
            "# \(spec.modelName)",
            "",
            "## Evaluation Provenance",
            "- Decoding: greedy (`temperature=\(formatValue(spec.decoding.temperature))`, `seed=\(spec.decoding.seed)`)",
        ]
        if !spec.decoding.hardware.isEmpty {
            lines.append("- Hardware: \(spec.decoding.hardware)")
        }
        if !spec.decoding.gitCommit.isEmpty {
            lines.append("- Basalt commit: `\(spec.decoding.gitCommit)`")
        }
        if let sig = spec.decoding.sacrebleuSignature, !sig.isEmpty {
            lines.append("- sacrebleu signature: `\(sig)`")
        }
        if let repo = spec.decoding.predictionsRepo, !repo.isEmpty {
            lines.append("- Raw predictions dataset: `\(repo)` (`predictions.jsonl`, `wmt/*.txt`)")
        }
        lines += [
            "",
            "## Benchmark Results",
            "| Dataset | Config | Split | Metrics |",
            "|---|---|---|---|",
        ]
        for result in spec.results {
            let summary = result.metrics.map { "\($0.name): \(formatValue($0.value))" }.joined(separator: ", ")
            lines.append("| \(result.datasetName) | `\(result.datasetConfig)` | `\(result.split)` | \(summary) |")
        }
        return lines
    }

    public static func validateModelCard(_ markdown: String) -> [String] {
        let lines = markdown.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        guard lines.first == "---",
              let endIdx = lines.dropFirst().firstIndex(of: "---") else {
            return ["README.md: missing YAML frontmatter delimiters ('---')"]
        }
        let frontmatter = lines[1 ..< endIdx]
        var errors: [String] = []
        let requiredTokens = ["model-index:", "  - name:", "    results:", "          type: translation", "        dataset:", "        metrics:"]
        for token in requiredTokens where !frontmatter.contains(where: { $0.hasPrefix(token) }) {
            errors.append("README.md: missing required HF model-index field '\(token.trimmingCharacters(in: .whitespaces))'")
        }
        if frontmatter.contains(where: { $0.hasPrefix("  - name: \"\"") || $0.trimmingCharacters(in: .whitespaces) == "- name:" }) {
            errors.append("README.md: model-index name must not be empty")
        }
        for line in frontmatter where line.trimmingCharacters(in: .whitespaces).hasPrefix("value:") {
            let raw = line.split(separator: ":", maxSplits: 1).last?.trimmingCharacters(in: .whitespaces) ?? ""
            guard let val = Double(raw), val.isFinite else {
                errors.append("README.md: non-finite metric value '\(raw)' in model-index")
                continue
            }
        }
        return errors
    }

    static func yamlScalar(_ raw: String) -> String {
        let cleaned = raw
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        return "\"\(cleaned)\""
    }

    static func formatValue(_ value: Double) -> String {
        guard value.isFinite else { return "nan" }
        return String(format: "%.4f", value)
    }
}

extension SubmissionFormatter {
    static func buildMTLeaderboard(_ artifact: BenchmarkRunArtifact, submit: BenchmarkSubmitSpec?) -> SubmissionPlan {
        let modelName = artifact.samples.first?.model.isEmpty == false
            ? artifact.samples[0].model
            : artifact.manifest.composeName
        let datasetInfo = resolveMTDataset(artifact: artifact, submit: submit)
        let grouped = groupSamplesByPair(artifact.samples)
        let results = buildMTResults(grouped: grouped, artifact: artifact, datasetInfo: datasetInfo)
        let languages = extractMTLanguages(pairs: grouped.map(\.0))
        let decoding = MTDecodingInfo(
            temperature: artifact.manifest.temperature,
            seed: artifact.manifest.seed,
            gitCommit: artifact.provenance.gitCommit,
            hardware: "\(artifact.provenance.chip) (\(artifact.provenance.unifiedMemoryGB) GB)",
            predictionsRepo: submit?.endpoint ?? artifact.samples.first?.metadata["predictions_repo"],
            sacrebleuSignature: artifact.samples.first?.metadata["sacrebleu_signature"]
        )
        let spec = MTModelCardSpec(
            modelName: modelName,
            baseModel: artifact.samples.first?.metadata["base_model"],
            languages: languages,
            decoding: decoding,
            results: results
        )
        let modelCard = HFModelIndexRenderer.renderModelCard(spec)
        var errors = HFModelIndexRenderer.validateModelCard(modelCard)
        let predRows = artifact.samples.map { predictionRow($0, fallbackDataset: datasetInfo.type) }
        var files: [SubmissionFile] = [
            SubmissionFile(relativePath: "README.md", data: Data(modelCard.utf8), contentType: "text/markdown"),
            SubmissionFile(relativePath: "eval_summary.json", data: encodeCodable(spec)),
            SubmissionFile(relativePath: "predictions.jsonl", data: jsonLines(predRows), contentType: "application/x-ndjson"),
            SubmissionFile(relativePath: "submission_checklist.md", data: Data(mtChecklist(modelName: modelName, pairs: grouped.map(\.0)).utf8), contentType: "text/markdown"),
        ]
        for (pair, pairSamples) in grouped {
            let hypLines = pairSamples.map { sanitizeWMTLine($0.answer) }
            if hypLines.count != pairSamples.count {
                errors.append("mt-leaderboard: line count mismatch for pair \(pair)")
            }
            let payload = hypLines.joined(separator: "\n") + (hypLines.isEmpty ? "" : "\n")
            files.append(SubmissionFile(relativePath: "wmt/\(safeFilename(pair)).txt", data: Data(payload.utf8), contentType: "text/plain"))
        }
        return SubmissionPlan(
            target: .mtLeaderboard,
            files: files,
            cliCommand: ["basalt", "models", "publish", ".", "--repo", modelName],
            httpEndpoint: submit?.endpoint ?? "https://huggingface.co/api/models",
            requiredEnvVar: submit?.apiKeyEnv ?? "HF_TOKEN",
            validationErrors: errors
        )
    }

    private static func resolveMTDataset(
        artifact: BenchmarkRunArtifact,
        submit: BenchmarkSubmitSpec?
    ) -> (name: String, type: String, split: String) {
        let raw = (submit?.subset ?? artifact.samples.first?.dataset ?? "basalt:translation").lowercased()
        let split = submit?.split ?? "test"
        if raw.contains("wmt24") {
            return ("WMT24++", "google/wmt24pp", split)
        }
        if raw.contains("flores") {
            return ("FLORES+", "openlanguagedata/flores_plus", submit?.split ?? "devtest")
        }
        return ("basalt:translation", "basalt:translation", split)
    }
}

extension SubmissionFormatter {
    private static func groupSamplesByPair(_ samples: [BenchmarkSampleRecord]) -> [(String, [BenchmarkSampleRecord])] {
        var buckets: [String: [BenchmarkSampleRecord]] = [:]
        for sample in samples {
            let pair: String
            if let explicit = sample.metadata["lang_pair"], !explicit.isEmpty {
                pair = explicit
            } else if let target = sample.metadata["target_lang"], !target.isEmpty {
                pair = "en-\(target)"
            } else {
                pair = "all"
            }
            buckets[pair, default: []].append(sample)
        }
        return buckets.keys.sorted().map { ($0, buckets[$0] ?? []) }
    }

    private static func buildMTResults(
        grouped: [(String, [BenchmarkSampleRecord])],
        artifact: BenchmarkRunArtifact,
        datasetInfo: (name: String, type: String, split: String)
    ) -> [MTBenchmarkResult] {
        var results: [MTBenchmarkResult] = []
        for (pair, items) in grouped {
            let metrics = aggregateMTMetrics(from: items, fallback: artifact.scorecard.metrics)
            if !metrics.isEmpty {
                results.append(MTBenchmarkResult(
                    datasetName: datasetInfo.name,
                    datasetType: datasetInfo.type,
                    datasetConfig: pair,
                    split: datasetInfo.split,
                    metrics: metrics
                ))
            }
        }
        if results.isEmpty {
            let fallbackMetrics = aggregateMTMetrics(from: [], fallback: artifact.scorecard.metrics)
            results.append(MTBenchmarkResult(
                datasetName: datasetInfo.name,
                datasetType: datasetInfo.type,
                datasetConfig: "overall",
                split: datasetInfo.split,
                metrics: fallbackMetrics.isEmpty
                    ? [MTBenchmarkMetric(name: "accuracy", type: "accuracy", value: artifact.scorecard.metrics["accuracy"] ?? 0)]
                    : fallbackMetrics
            ))
        }
        return results
    }

    private static func aggregateMTMetrics(
        from samples: [BenchmarkSampleRecord],
        fallback: [String: Double]
    ) -> [MTBenchmarkMetric] {
        let defs: [(key: String, name: String, type: String)] = [
            ("chrf_pp", "chrF++", "chrf"),
            ("comet", "COMET", "comet"),
            ("metricx24", "MetricX-24", "metricx24"),
            ("tech_fidelity", "tech_fidelity", "tech_fidelity"),
        ]
        var out: [MTBenchmarkMetric] = []
        for def in defs {
            let vals = samples.compactMap { $0.metadata[def.key].flatMap(Double.init) }.filter(\.isFinite)
            if !vals.isEmpty {
                let mean = vals.reduce(0, +) / Double(vals.count)
                out.append(MTBenchmarkMetric(name: def.name, type: def.type, value: mean))
            } else if let global = fallback[def.key], global.isFinite {
                out.append(MTBenchmarkMetric(name: def.name, type: def.type, value: global))
            }
        }
        return out
    }

    private static func extractMTLanguages(pairs: [String]) -> [String] {
        var langs: Set<String> = ["en"]
        for pair in pairs {
            for part in pair.split(separator: "-") {
                let code = part.split(separator: "_").first.map(String.init) ?? String(part)
                if code.count == 2 || code.count == 3, code != "all" {
                    langs.insert(code.lowercased())
                }
            }
        }
        return langs.sorted()
    }

    private static func predictionRow(_ sample: BenchmarkSampleRecord, fallbackDataset: String) -> [String: Any] {
        let pair = sample.metadata["lang_pair"]
            ?? sample.metadata["target_lang"].map { "en-\($0)" }
            ?? "all"
        var row: [String: Any] = [
            "id": sample.id,
            "lang_pair": pair,
            "dataset": sample.dataset ?? fallbackDataset,
            "source": sample.prompt,
            "reference": sample.metadata["reference"] ?? "",
            "hypothesis": sample.answer,
            "latency_ms": sample.latencyMs,
            "passed": sample.passed,
        ]
        for key in ["chrf_pp", "comet", "metricx24", "tech_fidelity"] {
            if let raw = sample.metadata[key], let val = Double(raw), val.isFinite {
                row[key] = val
            }
        }
        return row
    }

    private static func sanitizeWMTLine(_ raw: String) -> String {
        raw.replacingOccurrences(of: "\r\n", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }

    private static func mtChecklist(modelName: String, pairs: [String]) -> String {
        let pairList = pairs.joined(separator: ", ")
        return """
        # MT Leaderboard Submission Checklist — \(modelName)

        1. Verify `README.md` YAML `model-index` frontmatter passes HF Hub schema validation.
        2. Upload merged BF16 safetensors (`scripts/mt_leaderboard/merge_and_package.py`) and MLX sibling checkpoints via `basalt models publish`.
        3. Upload `predictions.jsonl` and `eval_summary.json` to the companion Hugging Face evaluation dataset repo.
        4. Submit `wmt/<pair>.txt` one-segment-per-line outputs (pairs: \(pairList)) to WMT24++ / Open MT / WMT shared task trackers.
        """
    }
}
