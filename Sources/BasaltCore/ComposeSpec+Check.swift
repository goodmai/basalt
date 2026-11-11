import Foundation

extension ComposeSpec {
    public struct Issue: Sendable, Equatable {
        public var path: String
        public var message: String

        public init(_ path: String, _ message: String) {
            (self.path, self.message) = (path, message)
        }
    }

    public var issues: [Issue] {
        var issues: [Issue] = []
        if !ComposeSchema.versions.contains(version) { issues.append(Issue("version", "version 1 or 2 is supported")) }
        if name.trimmingCharacters(in: .whitespaces).isEmpty { issues.append(Issue("name", "a compose file needs a name")) }
        for (pipeline, view) in views {
            let prefix = pipeline == Self.defaultPipeline ? "pipeline" : "pipelines.\(pipeline)"
            issues += (view.answerIssues + view.perceiveIssues + view.cascadeIssues).map { issue in
                Issue(issue.path.hasPrefix("pipeline.") ? prefix + issue.path.dropFirst("pipeline".count) : issue.path, issue.message)
            }
        }
        issues += pipelineIssues + models.sorted { $0.key < $1.key }.flatMap { modelIssues($0.key, $0.value) }
        for (name, _) in models.sorted(by: { $0.key < $1.key }) where !roles.keys.contains(name) {
            issues.append(Issue("models.\(name)", "declared but not used by any pipeline's perceive, decide or answer"))
        }
        return issues.reduce(into: [Issue]()) { unique, issue in if !unique.contains(issue) { unique.append(issue) } }
    }

    var pipelineRoles: [String: Set<String>] {
        var roles: [String: Set<String>] = [:]
        for (role, names) in [("answer", answerModels), ("documents", documents), ("audio", audio), ("cascade", cascade)] {
            for name in names where models[name] != nil { roles[name, default: []].insert(role) }
        }
        return roles
    }

    var roles: [String: Set<String>] {
        views.map(\.spec.pipelineRoles).reduce(into: [:]) { union, roles in union.merge(roles) { $0.union($1) } }
    }

    var pipelineIssues: [Issue] {
        var issues: [Issue] = []
        let deciders = Set([answer, cascade.last].compactMap { $0 })
        for (pipeline, view) in views where pipeline != Self.defaultPipeline {
            let path = "pipelines.\(pipeline)"
            if !pipeline.allSatisfy({ $0.isLowercase || $0.isNumber || $0 == "-" }) { issues.append(Issue(path, "a pipeline name uses lowercase letters, digits and -")) }
            if [view.ocr, view.speech].contains(where: { $0.flatMap { models[$0]?.keep } == .sequential }) {
                issues.append(Issue("\(path).perceive", "sequential perceivers swap models around the default pipeline only; keep them resident or on-demand here"))
            }
            if let local = view.cascade.last, !deciders.contains(local) {
                issues.append(Issue("\(path).decide.cascade", "every pipeline shares the default pipeline's local decision model (\(deciders.sorted().joined(separator: " or ")))"))
            }
        }
        return issues
    }

    var answerIssues: [Issue] {
        guard let model = models[answer] else { return [Issue("pipeline.answer", "\(answer) is not a model declared under models")] }
        if model.isCloud { return [Issue("pipeline.answer", "the default answer model is local: perceived text never leaves the machine, so routes end with a local model")] }
        var issues: [Issue] = []
        if let keep = model.keep, keep != .resident {
            issues.append(Issue("models.\(answer).keep", "the answer model stays resident; set memory: sequential to unload it around perceivers that do not fit beside it"))
        }
        if let last = routes.last, last.when != AnswerRoute.When() {
            issues.append(Issue("pipeline.answer.routes", "the last route takes every request left: give it no when"))
        }
        for name in answerModels where models[name] == nil { issues.append(Issue("pipeline.answer.routes", "\(name) is not a declared model")) }
        if let total = budget, !(total > 0) { issues.append(Issue("budget", "usd_per_day must be positive")) }
        return issues
    }

    var perceiveIssues: [Issue] {
        var issues: [Issue] = []
        for (key, names, builtin) in [("documents", documents, Self.appleVision), ("audio", audio, Self.speechAnalyzer)] {
            let declared = names.filter { models[$0] != nil }
            for name in names where name != builtin && models[name] == nil {
                issues.append(Issue("pipeline.perceive.\(key)", "\(name) is neither a declared model nor \(builtin)"))
            }
            if declared.count > 1 { issues.append(Issue("pipeline.perceive.\(key)", "one model per modality; \(declared.joined(separator: ", ")) are several")) }
            for name in declared where models[name]?.isCloud == true {
                issues.append(Issue("pipeline.perceive.\(key)", "\(name) is a cloud model; perceived documents and audio never leave the machine"))
            }
        }
        let keeps = Set([ocr, speech].compactMap { $0 }.compactMap { models[$0]?.keep })
        if keeps.contains(.sequential), keeps.contains(.onDemand) {
            issues.append(Issue("pipeline.perceive", "perceivers are either all sequential or each resident or on-demand"))
        }
        if keeps.contains(.onDemand), memory == .sequential {
            issues.append(Issue("memory", "an on-demand perceiver may be loaded at any time, so the set must fit resident; use memory: auto or resident, or keep: sequential"))
        }
        return issues
    }

    var cascadeIssues: [Issue] {
        guard !cascade.isEmpty else {
            return escalateBelow == nil ? [] : [Issue("pipeline.decide.escalate_below", "needs a cascade of a cloud model and a local one")]
        }
        var issues: [Issue] = []
        for name in cascade where models[name] == nil { issues.append(Issue("pipeline.decide.cascade", "\(name) is not a declared model")) }
        if Set(cascade).count != cascade.count { issues.append(Issue("pipeline.decide.cascade", "a model appears twice")) }
        let cloud = cascade.map { models[$0]?.isCloud == true }
        if cloud.last != false {
            issues.append(Issue("pipeline.decide.cascade", "a cascade ends with a local model, which takes unsure questions, failed calls and perceived text"))
        } else if cloud.dropLast().contains(false) || cascade.count > 4 {
            issues.append(Issue("pipeline.decide.cascade", "a cascade is up to three cloud models followed by one local model"))
        }
        if escalateBelow != nil, cascade.count < 2 { issues.append(Issue("pipeline.decide.escalate_below", "applies only to a cascade of more than one model")) }
        for (key, value) in [("escalate_below", escalateBelow), ("abstain_below", abstainBelow), ("min_margin", minMargin)] {
            if let value, !(0 ... 1).contains(value) { issues.append(Issue("pipeline.decide.\(key)", "must lie between 0 and 1")) }
        }
        return issues
    }

    func modelIssues(_ name: String, _ model: Model) -> [Issue] {
        let path = "models.\(name)"
        let perceiver = views.contains { $0.spec.ocr == name || $0.spec.speech == name }
        let routed = views.contains { $0.spec.answer != name && $0.spec.answerModels.contains(name) }
        var issues = keepIssues(path, model, perceiver: perceiver, answer: routed)
        let answering = views.contains { $0.spec.answer == name }
        if !model.isCloud, (roles[name]?.subtracting(answering ? ["cascade"] : []).count ?? 0) > 1 { issues.append(Issue(path, "one model serves one role; the answer model may also decide")) }
        if model.budget != nil, !model.isCloud { issues.append(Issue("\(path).budget", "applies only to cloud models")) }
        if let budget = model.budget, !(budget > 0) { issues.append(Issue("\(path).budget", "usd_per_day must be positive")) }
        switch model.source {
        case .local(let source):
            if source.isEmpty { issues.append(Issue("\(path).source", "is empty")) }
            if model.apiKeyEnv != nil || model.data != nil { issues.append(Issue(path, "api_key_env and data apply only to cloud models")) }
        case .cloud(let provider, let id):
            issues += cloudIssues(path, model, provider: provider, id: id)
        }
        return issues
    }

    func keepIssues(_ path: String, _ model: Model, perceiver: Bool, answer: Bool) -> [Issue] {
        var issues: [Issue] = []
        let allowed = perceiver || (answer && model.keep == .onDemand)
        if let keep = model.keep, keep != .resident, !allowed { issues.append(Issue("\(path).keep", "\(keep.rawValue) applies only to a perceiver, or on-demand to a routed answer model")) }
        if model.idleUnload != nil, model.keep != .onDemand { issues.append(Issue("\(path).idle_unload", "applies only to keep: on-demand")) }
        if let idle = model.idleUnload, idle <= 0 { issues.append(Issue("\(path).idle_unload", "must be a positive duration")) }
        return issues
    }

    func cloudIssues(_ path: String, _ model: Model, provider: String, id: String) -> [Issue] {
        var issues: [Issue] = []
        if !Source.providers.contains(provider) { issues.append(Issue("\(path).source", "cloud provider \(provider) is not supported; use cloud:openrouter/<model>")) }
        if id.isEmpty { issues.append(Issue("\(path).source", "names no model after cloud:\(provider)/")) }
        if model.keep != nil || model.idleUnload != nil { issues.append(Issue(path, "keep and idle_unload apply only to local models")) }
        if model.apiKeyEnv.map(Self.isEnvName) != true { issues.append(Issue("\(path).api_key_env", "a cloud model names the environment variable that holds its key, such as OPENROUTER_API_KEY")) }
        if let data = model.data, data != Self.noPerceived { issues.append(Issue("\(path).data", "only no-perceived is supported: OCR text and transcripts never leave the machine")) }
        return issues
    }

    static func isEnvName(_ name: String) -> Bool {
        guard let first = name.unicodeScalars.first, first == "_" || ("A" ... "Z").contains(first) else { return false }
        return name.unicodeScalars.allSatisfy { $0 == "_" || ("A" ... "Z").contains($0) || ("0" ... "9").contains($0) }
    }
}
