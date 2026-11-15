import Foundation

extension SubmissionFormatter {
    static func buildARCPrize(_ artifact: BenchmarkRunArtifact, submit: BenchmarkSubmitSpec?) -> SubmissionPlan {
        var errors: [String] = []
        var submission: [String: [[String: [[Int]]]]] = [:]
        for sample in artifact.samples {
            guard !sample.gridAttempts.isEmpty else {
                errors.append("task \(sample.id): missing ARC grid attempts")
                continue
            }
            let first = sample.gridAttempts[0]
            let second = sample.gridAttempts.count > 1 ? sample.gridAttempts[1] : first
            errors += ARCGridValidator.validate(first, label: "\(sample.id).attempt_1")
            errors += ARCGridValidator.validate(second, label: "\(sample.id).attempt_2")
            submission[sample.id] = [["attempt_1": first, "attempt_2": second]]
        }
        let count = max(1, artifact.samples.count)
        let attestation: [String: Any] = [
            "compose_sha256": artifact.provenance.composeSHA256,
            "git_commit": artifact.provenance.gitCommit,
            "chip": artifact.provenance.chip,
            "unified_memory_gb": artifact.provenance.unifiedMemoryGB,
            "tasks_evaluated": artifact.samples.count,
            "total_cost_usd": artifact.scorecard.totalCostUSD,
            "cost_per_task_usd": artifact.scorecard.totalCostUSD / Double(count),
            "wall_clock_seconds": artifact.provenance.wallClockSeconds,
        ]
        return SubmissionPlan(
            target: .arcPrize,
            files: [
                SubmissionFile(relativePath: "submission.json", data: jsonData(submission)),
                SubmissionFile(relativePath: "cost_attestation.json", data: jsonData(attestation)),
            ],
            cliCommand: ["basalt", "bench", "submit", "--target", "arc-prize"],
            httpEndpoint: submit?.endpoint ?? "https://arcprize.org/api/v1/submissions",
            requiredEnvVar: submit?.apiKeyEnv ?? "ARC_PRIZE_API_KEY",
            validationErrors: errors
        )
    }

    static func buildTerminalBench(_ artifact: BenchmarkRunArtifact, submit: BenchmarkSubmitSpec?) -> SubmissionPlan {
        let resolved = artifact.samples.filter(\.passed).map(\.id)
        let unresolved = artifact.samples.filter { !$0.passed }.map(\.id)
        let results: [String: Any] = [
            "run_id": artifact.manifest.runID,
            "dataset": submit?.subset ?? "terminal-bench-core",
            "accuracy": artifact.scorecard.metrics["accuracy"] ?? (Double(resolved.count) / Double(max(1, artifact.samples.count))),
            "n_resolved": resolved.count,
            "n_unresolved": unresolved.count,
            "resolved_ids": resolved,
            "unresolved_ids": unresolved,
        ]
        let metadata: [String: Any] = [
            "agent_name": "basalt-compose",
            "compose_name": artifact.manifest.composeName,
            "compose_sha256": artifact.provenance.composeSHA256,
            "git_commit": artifact.provenance.gitCommit,
            "chip": artifact.provenance.chip,
            "unified_memory_gb": artifact.provenance.unifiedMemoryGB,
            "slot_digests": artifact.provenance.slotDigests,
            "timestamp": artifact.provenance.timestampISO8601,
        ]
        var files = [
            SubmissionFile(relativePath: "results.json", data: jsonData(results)),
            SubmissionFile(relativePath: "agent_metadata.json", data: jsonData(metadata)),
        ]
        for sample in artifact.samples {
            let steps = sample.trajectory.isEmpty
                ? [["role": "user", "content": sample.prompt], ["role": "assistant", "content": sample.answer]]
                : sample.trajectory.map { ["role": $0.role, "content": $0.content] }
            let atif: [String: Any] = [
                "schema_version": "ATIF-v1.4",
                "task_id": sample.id,
                "model": sample.model,
                "steps": steps,
                "passed": sample.passed,
                "latency_ms": sample.latencyMs,
            ]
            files.append(SubmissionFile(relativePath: "trajectories/\(safeFilename(sample.id)).atif.json", data: jsonData(atif)))
        }
        return SubmissionPlan(
            target: .terminalBench, files: files,
            cliCommand: ["harbor", "submit", "--bundle", "."],
            httpEndpoint: submit?.endpoint, requiredEnvVar: submit?.apiKeyEnv
        )
    }
}

extension SubmissionFormatter {
    static func buildSWEBench(_ artifact: BenchmarkRunArtifact, submit: BenchmarkSubmitSpec?) -> SubmissionPlan {
        var preds: [String: [String: String]] = [:]
        var files: [SubmissionFile] = []
        var errors: [String] = []
        for sample in artifact.samples {
            if sample.id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                errors.append("swe-bench sample has blank instance_id")
            }
            let patch = sample.modelPatch ?? sample.answer
            preds[sample.id] = [
                "instance_id": sample.id,
                "model_name_or_path": sample.model,
                "model_patch": patch,
            ]
            let traj: [String: Any] = [
                "instance_id": sample.id,
                "model": sample.model,
                "patch": patch,
                "latency_ms": sample.latencyMs,
                "cost_usd": sample.costUSD,
            ]
            files.append(SubmissionFile(relativePath: "trajs/\(safeFilename(sample.id)).json", data: jsonData(traj)))
        }
        let yaml = [
            "name: \(artifact.manifest.composeName)",
            "run_id: \(artifact.manifest.runID)",
            "compose_sha256: \(artifact.provenance.composeSHA256)",
            "hardware: \(artifact.provenance.chip) (\(artifact.provenance.unifiedMemoryGB) GB)",
            "oss: true",
            "timestamp: \(artifact.provenance.timestampISO8601)",
        ].joined(separator: "\n") + "\n"
        files.insert(SubmissionFile(relativePath: "metadata.yaml", data: Data(yaml.utf8), contentType: "application/x-yaml"), at: 0)
        files.insert(SubmissionFile(relativePath: "preds.json", data: jsonData(preds)), at: 0)
        let subset = submit?.subset ?? "swe-bench_verified"
        let split = submit?.split ?? "test"
        return SubmissionPlan(
            target: .sweBench, files: files,
            cliCommand: ["sb-cli", "submit", subset, split, "--predictions_path", "preds.json", "--run_id", artifact.manifest.runID],
            httpEndpoint: submit?.endpoint, requiredEnvVar: submit?.apiKeyEnv ?? "SWEBENCH_API_KEY",
            validationErrors: errors
        )
    }

    static func buildHFLeaderboard(_ artifact: BenchmarkRunArtifact, submit: BenchmarkSubmitSpec?) -> SubmissionPlan {
        let taskName = submit?.subset ?? artifact.manifest.suite
        let results: [String: Any] = [
            "results": [taskName: artifact.scorecard.metrics],
            "configs": [taskName: ["task": taskName, "temperature": artifact.manifest.temperature, "seed": artifact.manifest.seed]],
            "versions": [taskName: 1],
            "git_hash": artifact.provenance.gitCommit,
            "compose_sha256": artifact.provenance.composeSHA256,
        ]
        let sampleRows: [[String: Any]] = artifact.samples.enumerated().map { index, sample in
            [
                "doc_id": index,
                "id": sample.id,
                "doc": ["prompt": sample.prompt],
                "target": sample.expected ?? "",
                "resps": [[sample.answer]],
                "filtered_resps": [sample.answer],
                "exact_match": sample.passed ? 1.0 : 0.0,
                "acc": sample.score,
            ]
        }
        let repo = submit?.subset ?? "open-llm-leaderboard/submissions"
        return SubmissionPlan(
            target: .hfLeaderboard,
            files: [
                SubmissionFile(relativePath: "results.json", data: jsonData(results)),
                SubmissionFile(relativePath: "samples.jsonl", data: jsonLines(sampleRows), contentType: "application/x-ndjson"),
            ],
            cliCommand: ["hf", "upload", repo, "."],
            httpEndpoint: submit?.endpoint,
            requiredEnvVar: submit?.apiKeyEnv ?? "HF_TOKEN"
        )
    }
}
