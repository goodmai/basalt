import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference

extension Benchmark {
    enum Suite: String, ExpressibleByArgument, CaseIterable {
        case faeScaling = "fae-scaling"
    }

    struct FaeScalingPlan {
        let fitLevel: FitLevel
        let measured: [Int]
        let notMeasured: [String]
    }

    func say(_ step: String, _ detail: String) {
        var load = [Double](repeating: 0, count: 3)
        let average = getloadavg(&load, 3) == 3 ? load[0] : .infinity
        print("\(Date().ISO8601Format()) \(model) \(step) \(detail) " + String(format: "load %.1f", average))
        fflush(stdout)
    }

    func faeScalingPlan(path: String) async throws -> FaeScalingPlan {
        let report = await ModelInspector().inspect(modelIdOrPath: path)
        guard report.intact else {
            say("FAIL inspect", report.errors.joined(separator: "; "))
            throw ExitCode.failure
        }
        say("inspect", "intact, \(report.totalShards) shards, \(report.totalSizeFormatted), \(report.bitWidthDescription), machine \(SystemProfiler.detectChipModel()) \(ProcessInfo.processInfo.physicalMemory >> 30) GB")
        let modelContext = ContextScalingSuite.modelContext(at: URL(fileURLWithPath: path))
        let evaluator = MemoryEvaluator()
        let zones: KVZones? = kvZones ? .standard : nil
        let fitLevel = await evaluator.evaluate(modelId: model, kvPrecision: kvBits, kvZones: zones).fitLevel
        var measured: [Int] = []
        var notMeasured: [String] = []
        for context in ContextScalingSuite.faeContexts {
            let total = context + tokens
            let fit = await evaluator.evaluate(modelId: model, contextLength: total, kvPrecision: kvBits, kvZones: zones)
            if let maxContext, context > maxContext {
                notMeasured.append("\(context) (above --max-context \(maxContext))")
            } else if let modelContext, total > modelContext {
                notMeasured.append("\(context) (the model's context is \(modelContext))")
            } else if !fit.fitsInMemory {
                notMeasured.append("\(context) (fit check: \(fit.fitLevel.rawValue), \(fit.requiredRAMFormatted) against \(fit.memoryBudgetFormatted))")
            } else {
                measured.append(context)
            }
        }
        say("plan", "fit \(fitLevel.rawValue) model context \(modelContext.map(String.init) ?? "unknown") measure \(measured) not measured \(notMeasured)")
        return FaeScalingPlan(fitLevel: fitLevel, measured: measured, notMeasured: notMeasured)
    }

    func runFaeScaling(_ orchestrator: ModelOrchestratorActor, plan: FaeScalingPlan) async throws {
        let suite = ContextScalingSuite(runs: iterations, tokens: tokens, temperature: temperature)
        say("config", "runs \(suite.runs) tokens \(suite.tokens) temperature \(suite.temperature)")
        let result = try await suite.measure(orchestrator, contexts: plan.measured, log: say)
        let report = ContextScalingReport(
            model: ModelCache.canonicalRepoId(from: model), chip: SystemProfiler.detectChipModel(), fitLevel: plan.fitLevel.rawValue,
            rows: result.rows, notMeasured: plan.notMeasured + result.failed.map { "\($0) (failed, see log)" })
        try export(report)
        if !result.failed.isEmpty || result.rows.isEmpty { throw ExitCode.failure }
    }

    func export(_ report: ContextScalingReport) throws {
        let directory = URL(fileURLWithPath: "logs")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let day = Date().ISO8601Format(.iso8601.year().month().day())
        let base = directory.appending(component: "fae-scaling-\(day)-\(report.model.replacingOccurrences(of: "/", with: "--"))")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: base.appendingPathExtension("json"))
        try Data(report.markdown.utf8).write(to: base.appendingPathExtension("md"))
        print("\n" + report.markdown)
        say("export", "\(base.path).json \(base.path).md")
    }
}
