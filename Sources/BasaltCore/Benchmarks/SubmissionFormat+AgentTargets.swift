import Foundation

extension SubmissionFormatter {
    static func buildLiveCodeBench(_ artifact: BenchmarkRunArtifact, submit: BenchmarkSubmitSpec?) -> SubmissionPlan {
        let outputs: [[String: Any]] = artifact.samples.map { sample in
            let codes = sample.codeCompletions.isEmpty ? [sample.answer] : sample.codeCompletions
            return ["question_id": sample.id, "code_list": codes]
        }
        let passAt1 = Double(artifact.samples.filter(\.passed).count) / Double(max(1, artifact.samples.count))
        let evalAll: [String: Any] = [
            "pass@1": artifact.scorecard.metrics["accuracy"] ?? passAt1,
            "total": artifact.samples.count,
            "compose_sha256": artifact.provenance.composeSHA256,
            "scored_results": Dictionary(uniqueKeysWithValues: artifact.samples.map { ($0.id, [$0.passed]) }),
        ]
        return SubmissionPlan(
            target: .liveCodeBench,
            files: [
                SubmissionFile(relativePath: "custom_outputs.json", data: jsonData(outputs)),
                SubmissionFile(relativePath: "eval_all.json", data: jsonData(evalAll)),
            ],
            cliCommand: ["python3", "-m", "lcb_runner.runner.custom_evaluator", "--custom_output_file", "custom_outputs.json"],
            httpEndpoint: submit?.endpoint,
            requiredEnvVar: submit?.apiKeyEnv
        )
    }

    static func buildBFCL(_ artifact: BenchmarkRunArtifact, submit: BenchmarkSubmitSpec?) -> SubmissionPlan {
        let grouped = Dictionary(grouping: artifact.samples) { sample in
            let raw = sample.category?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return raw.isEmpty ? "simple" : safeFilename(raw)
        }
        var files: [SubmissionFile] = []
        for category in grouped.keys.sorted() {
            let entries: [[String: Any]] = (grouped[category] ?? []).map { sample in
                let calls: [[String: String]] = sample.toolCalls.isEmpty
                    ? [[sample.model: sample.answer]]
                    : sample.toolCalls.map { [$0.name: $0.arguments] }
                return [
                    "id": sample.id,
                    "result": calls,
                    "input_token_count": sample.promptTokens,
                    "output_token_count": sample.completionTokens,
                    "latency": sample.latencyMs / 1000.0,
                ]
            }
            files.append(SubmissionFile(
                relativePath: "BFCL_v3_\(category)_result.json",
                data: jsonLines(entries),
                contentType: "application/x-ndjson"
            ))
        }
        let acc = artifact.scorecard.metrics["accuracy"] ?? (Double(artifact.scorecard.passedSamples) / Double(max(1, artifact.scorecard.totalSamples)))
        let p50 = artifact.scorecard.metrics["p50_ms"] ?? 0
        let csvHeader = "Model,Overall Acc,Total Cost ($),Latency p50 (ms),Compose SHA256\n"
        let csvRow = String(
            format: "%@,%.4f,%.4f,%.1f,%@\n",
            artifact.manifest.composeName, acc * 100.0, artifact.scorecard.totalCostUSD, p50, artifact.provenance.composeSHA256
        )
        files.append(SubmissionFile(relativePath: "data_overall.csv", data: Data((csvHeader + csvRow).utf8), contentType: "text/csv"))
        return SubmissionPlan(
            target: .bfcl, files: files,
            cliCommand: ["bfcl", "evaluate", "--result-dir", "."],
            httpEndpoint: submit?.endpoint, requiredEnvVar: submit?.apiKeyEnv
        )
    }

    static func buildInspectAI(_ artifact: BenchmarkRunArtifact, submit: BenchmarkSubmitSpec?) -> SubmissionPlan {
        let header: [String: Any] = [
            "version": 2,
            "status": artifact.scorecard.passed ? "success" : "completed",
            "eval": [
                "run_id": artifact.manifest.runID,
                "task": submit?.subset ?? artifact.manifest.suite,
                "model": artifact.manifest.composeName,
                "compose_sha256": artifact.provenance.composeSHA256,
                "created": artifact.provenance.timestampISO8601,
            ],
        ]
        let summaries: [[String: Any]] = artifact.samples.map { sample in
            [
                "id": sample.id,
                "epoch": 1,
                "input": sample.prompt,
                "target": sample.expected ?? "",
                "scores": ["accuracy": ["value": sample.score]],
            ]
        }
        let results: [String: Any] = [
            "scores": [["name": "basalt_scorer", "metrics": artifact.scorecard.metrics]],
            "stats": [
                "input_tokens": artifact.scorecard.totalPromptTokens,
                "output_tokens": artifact.scorecard.totalCompletionTokens,
                "cost_usd": artifact.scorecard.totalCostUSD,
            ],
        ]
        var files = [
            SubmissionFile(relativePath: "header.json", data: jsonData(header)),
            SubmissionFile(relativePath: "summaries.json", data: jsonData(summaries)),
            SubmissionFile(relativePath: "results.json", data: jsonData(results)),
        ]
        for sample in artifact.samples {
            let item: [String: Any] = [
                "id": sample.id,
                "epoch": 1,
                "input": sample.prompt,
                "output": ["model": sample.model, "completion": sample.answer],
                "scores": ["accuracy": ["value": sample.score, "passed": sample.passed]],
                "metadata": sample.metadata,
            ]
            files.append(SubmissionFile(relativePath: "samples/\(safeFilename(sample.id)).json", data: jsonData(item)))
        }
        return SubmissionPlan(
            target: .inspectAI, files: files,
            cliCommand: ["inspect", "view", "bundle", "--log-dir", "."],
            httpEndpoint: submit?.endpoint, requiredEnvVar: submit?.apiKeyEnv
        )
    }
}

extension SubmissionFormatter {
    static func buildTauBench(_ artifact: BenchmarkRunArtifact, submit: BenchmarkSubmitSpec?) -> SubmissionPlan {
        let items: [[String: Any]] = artifact.samples.map { sample in
            let traj = sample.trajectory.isEmpty
                ? [["role": "user", "content": sample.prompt], ["role": "assistant", "content": sample.answer]]
                : sample.trajectory.map { ["role": $0.role, "content": $0.content] }
            return [
                "task_id": sample.id,
                "reward": sample.passed ? 1.0 : 0.0,
                "info": [
                    "model": sample.model,
                    "latency_ms": sample.latencyMs,
                    "cost_usd": sample.costUSD,
                ],
                "traj": traj,
            ]
        }
        return SubmissionPlan(
            target: .tauBench,
            files: [SubmissionFile(relativePath: "results.json", data: jsonData(items))],
            cliCommand: ["tau-bench", "view", "--results", "results.json"],
            httpEndpoint: submit?.endpoint,
            requiredEnvVar: submit?.apiKeyEnv
        )
    }

    static func buildGAIA(_ artifact: BenchmarkRunArtifact, submit: BenchmarkSubmitSpec?) -> SubmissionPlan {
        let splitName = submit?.split == "test" ? "test.jsonl" : "validation.jsonl"
        let rows: [[String: Any]] = artifact.samples.map { sample in
            [
                "task_id": sample.id,
                "model_answer": normalizeGAIAAnswer(sample.answer),
                "reasoning_trace": sample.reasoningTrace ?? sample.answer,
            ]
        }
        return SubmissionPlan(
            target: .gaia,
            files: [SubmissionFile(relativePath: splitName, data: jsonLines(rows), contentType: "application/x-ndjson")],
            cliCommand: ["hf", "spaces", "submit", "--dataset", "gaia-benchmark", splitName],
            httpEndpoint: submit?.endpoint,
            requiredEnvVar: submit?.apiKeyEnv ?? "HF_TOKEN"
        )
    }

    static func buildWeaveOTel(_ artifact: BenchmarkRunArtifact, submit: BenchmarkSubmitSpec?) -> SubmissionPlan {
        let peakGB = artifact.scorecard.metrics["peak_gb"] ?? 0
        let spans: [[String: Any]] = artifact.samples.map { sample in
            [
                "name": "gen_ai.evaluation.sample",
                "attributes": [
                    "gen_ai.request.model": sample.model,
                    "gen_ai.evaluation.sample_id": sample.id,
                    "gen_ai.evaluation.score": sample.score,
                    "gen_ai.usage.input_tokens": sample.promptTokens,
                    "gen_ai.usage.output_tokens": sample.completionTokens,
                    "basalt.latency_ms": sample.latencyMs,
                    "basalt.ttft_ms": sample.ttftMs ?? 0,
                    "basalt.decode_tps": sample.decodeTPS ?? (artifact.scorecard.metrics["decode_tps"] ?? 0),
                    "basalt.peak_gb": peakGB,
                    "basalt.cost_usd": sample.costUSD,
                ],
            ]
        }
        let resourcePayload: [String: Any] = [
            "resourceSpans": [[
                "resource": [
                    "attributes": [
                        "service.name": "basalt",
                        "basalt.compose.name": artifact.manifest.composeName,
                        "basalt.compose.sha256": artifact.provenance.composeSHA256,
                        "basalt.git.commit": artifact.provenance.gitCommit,
                        "apple.chip": artifact.provenance.chip,
                        "apple.unified_memory_gb": artifact.provenance.unifiedMemoryGB,
                    ],
                ],
                "scopeSpans": [[
                    "scope": ["name": "basalt.bench", "version": "1.0.0"],
                    "spans": spans,
                ]],
            ]],
        ]
        return SubmissionPlan(
            target: .weaveOTel,
            files: [SubmissionFile(relativePath: "otel_eval_export.json", data: jsonData(resourcePayload))],
            cliCommand: ["basalt", "bench", "submit", "--target", "weave-otel"],
            httpEndpoint: submit?.endpoint ?? "https://trace.wandb.ai/otel/v1/traces",
            requiredEnvVar: submit?.apiKeyEnv ?? "WANDB_API_KEY"
        )
    }
}
