import Foundation

public struct ComposeDiff: Sendable, Equatable, Codable {
    public var added: [String] = []
    public var removed: [String] = []
    public var reloaded: [String] = []
    public var kept: [String] = []
    public var pipelines: [String] = []
    public var budgets: [String] = []
    public var rekeyed: [String] = []
    public var chat: String?
    public var decider: String?
    public var restart: String?

    public init() {}

    public static func between(_ old: ComposeSpec, _ new: ComposeSpec) -> ComposeDiff {
        var diff = ComposeDiff()
        diff.pipelines = new.pipelineNames
        if old.answer != new.answer || reloads(old, new, new.answer) { diff.chat = new.answer }
        if old.decider != new.decider || new.deciderModel.map({ reloads(old, new, $0) }) == true { diff.decider = new.decider ?? "none" }
        for name in Set(old.models.keys).union(new.models.keys).sorted() { diff.classify(name, old, new) }
        if old.budget != new.budget { diff.budgets.append("total") }
        diff.restart = restartReason(old, new, touched: diff.added + diff.removed + diff.reloaded)
        return diff
    }

    mutating func classify(_ name: String, _ old: ComposeSpec, _ new: ComposeSpec) {
        switch (old.roleModels.contains(name) ? nil : old.models[name], new.roleModels.contains(name) ? nil : new.models[name]) {
        case (nil, _?): added.append(name)
        case (_?, nil): removed.append(name)
        case (let before?, let after?):
            Self.needsReload(before, after) ? reloaded.append(name) : kept.append(name)
            if after.isCloud, before.budget != after.budget { budgets.append(name) }
            if after.isCloud, before.apiKeyEnv != after.apiKeyEnv { rekeyed.append(name) }
        case (nil, nil):
            if (name == new.answer && chat == nil) || (name == new.deciderModel && decider == nil) { kept.append(name) }
        }
    }

    static func reloads(_ old: ComposeSpec, _ new: ComposeSpec, _ name: String) -> Bool {
        guard let before = old.models[name], let after = new.models[name] else { return false }
        return needsReload(before, after)
    }

    static func needsReload(_ before: ComposeSpec.Model, _ after: ComposeSpec.Model) -> Bool {
        before.source != after.source || before.keep != after.keep || before.idleUnload != after.idleUnload
    }

    static func restartReason(_ old: ComposeSpec, _ new: ComposeSpec, touched: [String]) -> String? {
        if old.name != new.name { return "the file names another system (\(new.name), running \(old.name))" }
        if old.effectiveMemory != new.effectiveMemory { return "the memory mode changes from \(old.effectiveMemory.rawValue) to \(new.effectiveMemory.rawValue)" }
        let sequential = [old, new].flatMap { spec in [spec.ocr, spec.speech].compactMap { $0 }.filter { spec.models[$0]?.keep == .sequential } }
        if let swapped = sequential.first(where: touched.contains) { return "\(swapped) swaps per request (keep: sequential), which is set up at start" }
        return nil
    }

    public var lines: [String] {
        var lines = [("load", added), ("reload", reloaded), ("unload", removed), ("keep", kept), ("budget", budgets), ("key", rekeyed)].compactMap { verb, names in
            names.isEmpty ? nil : "  \(verb): \(names.joined(separator: ", "))"
        }
        lines += [("chat", chat), ("decide", decider)].compactMap { verb, name in name.map { "  \(verb): \($0)" } }
        lines.append("  pipelines: \(pipelines.joined(separator: ", "))")
        return lines + (restart.map { ["  needs a restart: \($0)"] } ?? [])
    }
}

extension ComposeSpec {
    public var decider: String? {
        guard let last = cascade.last, models[last]?.isCloud == false else { return nil }
        return last == answer ? "self" : last
    }

    public var deciderModel: String? { decider == "self" ? nil : decider }

    var roleModels: [String] { [answer] + (deciderModel.map { [$0] } ?? []) }
}

extension ComposeDiff {
    public enum Order: Sendable, Equatable {
        case loadFirst, unloadFirst
    }

    public func admission(
        previous: ComposeSpec, next: ComposeSpec, sizes: ComposeSpec.Sizes, budgetBytes: Int64, power: PowerState = .desktop
    ) -> Result<Order, Refusal> {
        let (verdict, _) = next.plan(sizes, budgetBytes: budgetBytes, power: power)
        if let refusal = verdict.refusal { return .failure(Refusal(reason: refusal)) }
        let roles = (chat == nil ? [] : [previous.answer]) + (decider == nil ? [] : previous.deciderModel.map { [$0] } ?? [])
        let replaced = (removed + reloaded + roles).compactMap { previous.local($0).flatMap(sizes.bytes) }.reduce(0, +)
        return .success(verdict.residentBytes + replaced > budgetBytes ? .unloadFirst : .loadFirst)
    }

    public struct Refusal: Error, Sendable, Equatable {
        public let reason: String
    }
}
