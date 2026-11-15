import Foundation

public enum BenchmarkSecretGuard {
    static let rawTokenPrefixes = [
        "sk-", "hf_", "ghp_", "gho_", "ghu_", "ghs_", "ghr_",
        "github_pat_", "glpat-", "xoxb-", "xoxp-", "xoxa-", "xoxs-",
        "AIza", "AKIA", "ASIA", "wandb_", "Bearer ", "bearer ",
    ]

    public static func looksLikeSecret(_ value: String) -> Bool {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        if rawTokenPrefixes.contains(where: { trimmed.hasPrefix($0) }) { return true }
        if SecretRedactor.redact(trimmed) != trimmed { return true }
        if trimmed.contains(":") || trimmed.contains("/") || trimmed.contains(where: \.isWhitespace) { return true }
        return false
    }

    public static func apiKeyEnvIssue(_ env: String, path: String) -> ComposeSpec.Issue? {
        if looksLikeSecret(env) {
            return ComposeSpec.Issue(path, "looks like a raw secret token; pass an uppercase environment variable name via api_key_env")
        }
        if !ComposeSpec.isEnvName(env) {
            return ComposeSpec.Issue(path, "api_key_env must be a valid POSIX uppercase environment variable identifier ([A-Z_][A-Z0-9_]*)")
        }
        return nil
    }
}

extension ComposeSpec {
    var benchmarkModels: Set<String> {
        guard let benchmarks else { return [] }
        var slots: Set<String> = []
        for suite in benchmarks.suites.values {
            for task in suite.tasks where task.target.hasPrefix("model:") {
                let slot = String(task.target.dropFirst("model:".count))
                if models[slot] != nil { slots.insert(slot) }
            }
        }
        return slots
    }

    var benchmarkIssues: [Issue] {
        guard let benchmarks else { return [] }
        var issues: [Issue] = []
        if version < 2 { issues.append(Issue("benchmarks", "needs version: 2")) }
        if let temp = benchmarks.defaults?.temperature, !(0 ... 2).contains(temp) || !temp.isFinite {
            issues.append(Issue("benchmarks.defaults.temperature", "must lie between 0 and 2"))
        }
        for (name, suite) in benchmarks.suites.sorted(by: { $0.key < $1.key }) {
            issues += suiteIssues(name: name, suite: suite)
        }
        return issues
    }

    func suiteIssues(name: String, suite: BenchmarkSuiteSpec) -> [Issue] {
        let path = "benchmarks.suites.\(name)"
        var issues: [Issue] = []
        if name.isEmpty || !name.allSatisfy({ $0.isLowercase || $0.isNumber || $0 == "-" || $0 == "_" }) {
            issues.append(Issue(path, "a benchmark suite name uses lowercase letters, digits, - and _"))
        }
        if let usd = suite.budget?.usdMax, !(usd > 0) || !usd.isFinite {
            issues.append(Issue("\(path).budget.usd_max", "must be positive"))
        }
        if let minutes = suite.budget?.maxMinutes, !(minutes > 0) || !minutes.isFinite {
            issues.append(Issue("\(path).budget.max_minutes", "must be positive"))
        }
        if suite.tasks.isEmpty {
            issues.append(Issue(path, "a benchmark suite needs at least one task"))
        }
        for (index, task) in suite.tasks.enumerated() {
            issues += taskIssues(task, path: "\(path).tasks[\(index)]")
        }
        return issues
    }

    func taskIssues(_ task: BenchmarkTaskSpec, path: String) -> [Issue] {
        var issues: [Issue] = []
        if !BenchmarkAdapterID.known.contains(task.adapter) {
            let allowed = BenchmarkAdapterID.known.sorted().joined(separator: ", ")
            issues.append(Issue("\(path).adapter", "unknown benchmark adapter \(task.adapter); expected one of \(allowed)"))
        }
        issues += targetIssues(task.target, path: "\(path).target")
        if let gates = task.gates { issues += gateIssues(gates, path: "\(path).gates") }
        if let submit = task.submit { issues += submitIssues(submit, path: "\(path).submit") }
        return issues
    }

    func targetIssues(_ target: String, path: String) -> [Issue] {
        if target == "pipeline" { return [] }
        if target.hasPrefix("pipeline:") {
            let name = String(target.dropFirst("pipeline:".count))
            if name != Self.defaultPipeline, pipelines[name] == nil {
                return [Issue(path, "pipeline \(name) is not declared under pipelines")]
            }
            return []
        }
        if target.hasPrefix("model:") {
            let slot = String(target.dropFirst("model:".count))
            if models[slot] == nil {
                return [Issue(path, "model \(slot) is not declared under models")]
            }
            return []
        }
        return [Issue(path, "target must be pipeline, pipeline:<name> or model:<slot>")]
    }

    func gateIssues(_ gates: BenchmarkGatesSpec, path: String) -> [Issue] {
        var issues: [Issue] = []
        for (key, value) in [
            ("min_decode_tps", gates.minDecodeTPS), ("max_peak_gb", gates.maxPeakGB),
            ("max_p50_ms", gates.maxP50Ms), ("max_ppl", gates.maxPPL),
        ] {
            if let value, !(value > 0) || !value.isFinite {
                issues.append(Issue("\(path).\(key)", "must be positive"))
            }
        }
        for (key, value) in [("min_accuracy", gates.minAccuracy), ("max_ece", gates.maxECE)] {
            if let value, !(0 ... 1).contains(value) || !value.isFinite {
                issues.append(Issue("\(path).\(key)", "must lie between 0 and 1"))
            }
        }
        return issues
    }

    func submitIssues(_ submit: BenchmarkSubmitSpec, path: String) -> [Issue] {
        var issues: [Issue] = []
        if let env = submit.apiKeyEnv, let issue = BenchmarkSecretGuard.apiKeyEnvIssue(env, path: "\(path).api_key_env") {
            issues.append(issue)
        }
        if let endpoint = submit.endpoint, endpoint.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            issues.append(Issue("\(path).endpoint", "endpoint must not be blank"))
        }
        return issues
    }

    public var benchmarkLines: [String] {
        guard let benchmarks, !benchmarks.suites.isEmpty else { return [] }
        return benchmarks.suites.keys.sorted().compactMap { name in
            guard let suite = benchmarks.suites[name] else { return nil }
            let summary = suite.tasks.map { "\($0.adapter) -> \($0.target)" }.joined(separator: ", ")
            var caps: [String] = []
            if let usd = suite.budget?.usdMax { caps.append(String(format: "$%.2f max", usd)) }
            if let minutes = suite.budget?.maxMinutes { caps.append(String(format: "%.0f min max", minutes)) }
            let capSuffix = caps.isEmpty ? "" : " (\(caps.joined(separator: ", ")))"
            return "  benchmark suite \(name): \(suite.tasks.count) task(s) [\(summary)]\(capSuffix)"
        }
    }
}
