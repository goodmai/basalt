import Foundation

public struct ComposeDiff: Sendable, Equatable, Codable {
    public var added: [String] = []
    public var removed: [String] = []
    public var reloaded: [String] = []
    public var kept: [String] = []
    public var pipelines: [String] = []
    public var budgets: [String] = []
    public var rekeyed: [String] = []
    public var restart: String?

    public init() {}

    public static func between(_ old: ComposeSpec, _ new: ComposeSpec) -> ComposeDiff {
        var diff = ComposeDiff()
        diff.pipelines = new.pipelineNames
        for name in Set(old.models.keys).union(new.models.keys).sorted() {
            switch (old.models[name], new.models[name]) {
            case (nil, _?): diff.added.append(name)
            case (_?, nil): diff.removed.append(name)
            case (let before?, let after?):
                needsReload(before, after) ? diff.reloaded.append(name) : diff.kept.append(name)
                if after.isCloud, before.budget != after.budget { diff.budgets.append(name) }
                if after.isCloud, before.apiKeyEnv != after.apiKeyEnv { diff.rekeyed.append(name) }
            case (nil, nil): break
            }
        }
        if old.budget != new.budget { diff.budgets.append("total") }
        diff.restart = restartReason(old, new, touched: diff.added + diff.removed + diff.reloaded)
        return diff
    }

    static func needsReload(_ before: ComposeSpec.Model, _ after: ComposeSpec.Model) -> Bool {
        before.source != after.source || before.keep != after.keep || before.idleUnload != after.idleUnload
    }

    static func restartReason(_ old: ComposeSpec, _ new: ComposeSpec, touched: [String]) -> String? {
        if old.name != new.name { return "the file names another system (\(new.name), running \(old.name))" }
        if old.answer != new.answer || touched.contains(old.answer) { return "the chat model changes; restart with compose down and compose up" }
        if old.effectiveMemory != new.effectiveMemory { return "the memory mode changes from \(old.effectiveMemory.rawValue) to \(new.effectiveMemory.rawValue)" }
        let sequential = [old, new].flatMap { spec in [spec.ocr, spec.speech].compactMap { $0 }.filter { spec.models[$0]?.keep == .sequential } }
        if let swapped = sequential.first(where: touched.contains) { return "\(swapped) swaps per request (keep: sequential), which is set up at start" }
        if old.decider != new.decider { return "the local decision model changes from \(old.decider ?? "none") to \(new.decider ?? "none")" }
        return nil
    }

    public var lines: [String] {
        var lines = [("load", added), ("reload", reloaded), ("unload", removed), ("keep", kept), ("budget", budgets), ("key", rekeyed)].compactMap { verb, names in
            names.isEmpty ? nil : "  \(verb): \(names.joined(separator: ", "))"
        }
        lines.append("  pipelines: \(pipelines.joined(separator: ", "))")
        return lines + (restart.map { ["  needs a restart: \($0)"] } ?? [])
    }
}

extension ComposeSpec {
    public var decider: String? {
        guard let last = cascade.last, models[last]?.isCloud == false else { return nil }
        return last == answer ? "self" : last
    }
}
