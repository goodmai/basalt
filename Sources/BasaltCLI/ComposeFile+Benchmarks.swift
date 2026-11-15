import Foundation
import BasaltCore
import Yams

extension ComposeFile.Walk {
    mutating func benchmarks(_ node: Node?, into spec: inout ComposeSpec) {
        guard let node else { return }
        lines["benchmarks"] = node.mark?.line
        guard let mapping = node.mapping else {
            issue("benchmarks", node, "must be a mapping")
            return
        }
        let explicitSuites = mapping["suites"] != nil
        var out = ComposeBenchmarksSpec()
        for (key, value) in mapping {
            parseBenchmarkTopEntry(key: key, value: value, explicitSuites: explicitSuites, into: &out)
        }
        spec.benchmarks = out
    }

    mutating func parseBenchmarkTopEntry(
        key: Node, value: Node, explicitSuites: Bool, into out: inout ComposeBenchmarksSpec
    ) {
        guard key.scalar != nil, let name = key.string, !name.isEmpty else {
            issue("benchmarks", key, "keys are plain names")
            return
        }
        let child = "benchmarks.\(name)"
        lines[child] = key.mark?.line
        if ComposeFile.secretKeys.contains(name.lowercased()) {
            issue(child, key, "secrets never go in a compose file; name the environment variable that holds it with api_key_env")
            return
        }
        if name == "defaults" {
            out.defaults = benchmarkDefaults(value, path: child)
        } else if name == "suites" {
            guard value.mapping != nil else {
                issue(child, value, "must be a mapping of suite name to task list or suite")
                return
            }
            for (suiteName, suiteNode) in named(value, path: child) {
                if let parsed = benchmarkSuite(suiteNode, path: "benchmarks.suites.\(suiteName)") {
                    out.suites[suiteName] = parsed
                }
            }
        } else if explicitSuites {
            issue(child, key, "unknown key; put suites under benchmarks.suites")
        } else {
            lines["benchmarks.suites.\(name)"] = key.mark?.line
            if let parsed = benchmarkSuite(value, path: "benchmarks.suites.\(name)") {
                out.suites[name] = parsed
            }
        }
    }

    mutating func benchmarkDefaults(_ node: Node, path: String) -> BenchmarkDefaultsSpec? {
        guard let field = fields(node, path: path, section: "benchmark_defaults") else { return nil }
        var out = BenchmarkDefaultsSpec()
        if let rawSeed = integer(field["seed"], "\(path).seed") {
            if rawSeed < 0 {
                issue("\(path).seed", field["seed"]!, "must be non-negative")
            } else {
                out.seed = UInt64(rawSeed)
            }
        }
        out.temperature = number(field["temperature"], "\(path).temperature")
        out.gpuLock = flag(field["gpu_lock"], "\(path).gpu_lock")
        out.outputDir = text(field["output_dir"], "\(path).output_dir")
        return out
    }

    mutating func benchmarkSuite(_ node: Node, path: String) -> BenchmarkSuiteSpec? {
        lines[path] = lines[path] ?? node.mark?.line
        if let seq = node.sequence {
            let tasks = Array(seq).enumerated().compactMap { index, item in
                benchmarkTask(item, path: "\(path).tasks[\(index)]")
            }
            return BenchmarkSuiteSpec(tasks: tasks)
        }
        guard node.mapping != nil else {
            issue(path, node, "must be a list of tasks or { profiles, budget, tasks }")
            return nil
        }
        guard let field = fields(node, path: path, section: "benchmark_suite") else { return nil }
        let profileNames = field["profiles"].map { list($0, "\(path).profiles") } ?? []
        for name in profileNames where !declared.contains(name) {
            issue("\(path).profiles", field["profiles"]!, "\(name) is not declared under profiles")
        }
        if !profileNames.isEmpty, !profileNames.contains(where: active.contains) { return nil }
        let suiteBudget = field["budget"].flatMap { benchmarkBudget($0, path: "\(path).budget") }
        guard let tasksNode = field["tasks"], let seq = tasksNode.sequence else {
            issue("\(path).tasks", field["tasks"] ?? node, "a benchmark suite mapping needs tasks: [...]")
            return nil
        }
        let tasks = Array(seq).enumerated().compactMap { index, item in
            benchmarkTask(item, path: "\(path).tasks[\(index)]")
        }
        return BenchmarkSuiteSpec(profiles: profileNames, budget: suiteBudget, tasks: tasks)
    }

    mutating func benchmarkBudget(_ node: Node, path: String) -> BenchmarkBudgetSpec? {
        guard let field = fields(node, path: path, section: "benchmark_budget") else { return nil }
        return BenchmarkBudgetSpec(
            usdMax: number(field["usd_max"], "\(path).usd_max"),
            maxMinutes: number(field["max_minutes"], "\(path).max_minutes")
        )
    }
}

extension ComposeFile.Walk {
    mutating func benchmarkTask(_ node: Node, path: String) -> BenchmarkTaskSpec? {
        lines[path] = node.mark?.line
        guard let field = fields(node, path: path, section: "benchmark_task") else { return nil }
        guard let adapterNode = field["adapter"], let adapter = text(adapterNode, "\(path).adapter") else {
            issue("\(path).adapter", node, "a benchmark task needs adapter")
            return nil
        }
        let dataset = text(field["dataset"], "\(path).dataset")
        let target = field["target"] != nil ? (text(field["target"], "\(path).target") ?? "") : "pipeline"
        let params = field["params"].map { benchmarkParams($0, path: "\(path).params") } ?? [:]
        let gates = field["gates"].flatMap { benchmarkGates($0, path: "\(path).gates") }
        let submit = field["submit"].flatMap { benchmarkSubmit($0, path: "\(path).submit") }
        return BenchmarkTaskSpec(
            adapter: adapter, dataset: dataset, target: target,
            params: params, gates: gates, submit: submit
        )
    }

    mutating func benchmarkParams(_ node: Node, path: String) -> [String: ComposeScalar] {
        lines[path] = node.mark?.line
        guard let mapping = node.mapping else {
            issue(path, node, "params must be a mapping of scalar values")
            return [:]
        }
        var out: [String: ComposeScalar] = [:]
        for (key, value) in mapping {
            guard key.scalar != nil, let name = key.string, !name.isEmpty else {
                issue(path, key, "keys are plain names")
                continue
            }
            let child = "\(path).\(name)"
            lines[child] = key.mark?.line
            if ComposeFile.secretKeys.contains(name.lowercased()) {
                issue(child, key, "secrets never go in a compose file; name the environment variable that holds it with api_key_env")
                continue
            }
            if let scalar = benchmarkScalar(value) {
                out[name] = scalar
            } else {
                issue(child, value, "param value must be a string, number or boolean")
            }
        }
        return out
    }

    func benchmarkScalar(_ node: Node) -> ComposeScalar? {
        guard node.mapping == nil, node.sequence == nil, let raw = node.scalar?.string else { return nil }
        if node.scalar?.style == .plain {
            if raw == "true" { return .bool(true) }
            if raw == "false" { return .bool(false) }
            if let whole = node.int { return .int(whole) }
            if let real = node.float { return .double(real) }
        }
        return .string(raw)
    }

    mutating func benchmarkGates(_ node: Node, path: String) -> BenchmarkGatesSpec? {
        guard let field = fields(node, path: path, section: "benchmark_gates") else { return nil }
        return BenchmarkGatesSpec(
            minDecodeTPS: number(field["min_decode_tps"], "\(path).min_decode_tps"),
            maxPeakGB: number(field["max_peak_gb"], "\(path).max_peak_gb"),
            minAccuracy: number(field["min_accuracy"], "\(path).min_accuracy"),
            maxP50Ms: number(field["max_p50_ms"], "\(path).max_p50_ms"),
            maxPPL: number(field["max_ppl"], "\(path).max_ppl"),
            maxECE: number(field["max_ece"], "\(path).max_ece")
        )
    }

    mutating func benchmarkSubmit(_ node: Node, path: String) -> BenchmarkSubmitSpec? {
        guard let field = fields(node, path: path, section: "benchmark_submit") else { return nil }
        guard let targetNode = field["target"], let rawTarget = text(targetNode, "\(path).target") else {
            issue("\(path).target", node, "submit needs target")
            return nil
        }
        guard let target = SubmissionTarget(rawValue: rawTarget) else {
            let allowed = SubmissionTarget.allCases.map(\.rawValue).joined(separator: ", ")
            issue("\(path).target", targetNode, "unknown submission target \(rawTarget); expected one of \(allowed)")
            return nil
        }
        return BenchmarkSubmitSpec(
            target: target,
            apiKeyEnv: text(field["api_key_env"], "\(path).api_key_env"),
            endpoint: text(field["endpoint"], "\(path).endpoint"),
            subset: text(field["subset"], "\(path).subset"),
            split: text(field["split"], "\(path).split")
        )
    }
}
