import Foundation

extension ComposeSpec {
    public var effectiveMemory: PipelineMemory {
        let keeps = [ocr, speech].compactMap { $0.flatMap { models[$0]?.keep } }
        return keeps.contains(.sequential) ? .sequential : keeps.contains(.onDemand) ? .resident : memory
    }

    public var serveArguments: [String] {
        var arguments = Self.modelFlag(local(answer) ?? answer)
        for (name, model, idle) in [(ocr, "--pipeline-ocr-model", "--pipeline-ocr-idle-unload"), (speech, "--pipeline-audio-model", "--pipeline-audio-idle-unload")] {
            guard let name, let source = local(name) else { continue }
            arguments += [model, source]
            if models[name]?.keep == .onDemand { arguments += [idle, String(models[name]?.idleUnload ?? Self.defaultIdleUnload)] }
        }
        if audio.contains(Self.speechAnalyzer) { arguments.append("--experimental-speech-analyzer") }
        arguments += ["--pipeline-memory", effectiveMemory.rawValue] + decideArguments
        for (flag, value) in [("--pipeline-abstain-below", abstainBelow), ("--pipeline-min-margin", minMargin)] {
            if let value { arguments += [flag, String(value)] }
        }
        return arguments
    }

    var decideArguments: [String] {
        func local(_ name: String) -> String { name == answer ? "self" : (self.local(name) ?? name) }
        guard !cascade.contains(where: { models[$0]?.keep == .onDemand }) else { return [] }
        switch cascade.count {
        case 0: return []
        case 1: return ["--decision-model", local(cascade[0])]
        default:
            guard case .cloud(let provider, let model)? = models[cascade[0]]?.source, let last = cascade.last else { return [] }
            return ["--decision-model", provider == "gemini" ? "cloud:gemini/\(model)" : "cloud:\(model)", "--decision-escalate", local(last),
                    "--decision-api-key-env", models[cascade[0]]?.apiKeyEnv ?? "OPENROUTER_API_KEY"]
                + (escalateBelow.map { ["--decision-escalate-below", String($0)] } ?? [])
        }
    }

    static func modelFlag(_ source: String) -> [String] {
        source.hasPrefix("/") || source.hasPrefix(".") || source.hasPrefix("~")
            ? ["--model-path", (source as NSString).expandingTildeInPath] : ["--model", source]
    }

    public struct Sizes: Sendable {
        public var bytes: @Sendable (String) -> Int64?
        public var onNeuralEngine: @Sendable (String) -> Bool

        public init(bytes: @escaping @Sendable (String) -> Int64?, onNeuralEngine: @escaping @Sendable (String) -> Bool) {
            (self.bytes, self.onNeuralEngine) = (bytes, onNeuralEngine)
        }
    }

    public func workloads(_ sizes: Sizes) -> (workloads: [Workload], unknown: [String]) {
        var unknown: [String] = []
        func bytes(_ name: String) -> Int64 {
            guard let source = local(name), let size = sizes.bytes(source) else {
                unknown.append(name)
                return 0
            }
            return size
        }
        var workloads = [Workload(name: answer, role: "chat", unit: .gpu, kind: .decode, residentBytes: bytes(answer))]
        for name in [ocr, speech].compactMap({ $0 }) {
            let keep: KeepPolicy = models[name]?.keep == .sequential ? .sequential : .resident
            workloads.append(Workload(name: name, role: Workload.perceiveRole, unit: .gpu, kind: .encoder, residentBytes: bytes(name), keep: keep))
        }
        var routed: [String] = []
        for view in views.map(\.spec) {
            for name in view.answerModels where name != answer && local(name) != nil && !routed.contains(name) { routed.append(name) }
        }
        for name in routed {
            workloads.append(Workload(name: name, role: "answer", unit: .gpu, kind: .decode, residentBytes: bytes(name)))
        }
        for decider in deciderModels.sorted() {
            guard let source = local(decider) else { continue }
            workloads.append(Workload(name: decider, role: "decide", unit: sizes.onNeuralEngine(source) ? .ane : .gpu, kind: .encoder, residentBytes: bytes(decider)))
        }
        return (workloads, unknown)
    }

    public func plan(_ sizes: Sizes, budgetBytes: Int64, power: PowerState = .desktop) -> (verdict: PlanVerdict, unknown: [String]) {
        let (workloads, unknown) = workloads(sizes)
        return (ComputePlan.admit(workloads, budgetBytes: budgetBytes, power: power, memory: effectiveMemory), unknown)
    }

    public var cloudLines: [String] {
        (budget.map { [String(format: "  cloud spend at most $%.2f a day in total", $0)] } ?? []) + models.sorted { $0.key < $1.key }.compactMap { name, model in
            guard case .cloud(let provider, let id) = model.source else { return nil }
            let budget = model.budget.map { String(format: "; at most $%.2f a day", $0) } ?? ""
            return "  \(name) cloud:\(provider)/\(id): no local memory; key from $\(model.apiKeyEnv ?? "?")\(budget); documents and transcripts stay local"
        }
    }

    public var onDemandLines: [String] {
        models.keys.sorted().filter { models[$0]?.keep == .onDemand }.map {
            "  \($0) on demand: loaded by the first request that needs it, unloaded after \(models[$0]?.idleUnload ?? Self.defaultIdleUnload) s idle; counted as resident above"
        }
    }
}
