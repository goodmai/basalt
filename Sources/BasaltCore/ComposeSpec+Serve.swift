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
        switch cascade.count {
        case 1: return ["--decision-model", local(cascade[0])]
        case 2:
            guard case .cloud(_, let model)? = models[cascade[0]]?.source else { return [] }
            return ["--decision-model", "cloud:\(model)", "--decision-escalate", local(cascade[1]),
                    "--decision-api-key-env", models[cascade[0]]?.apiKeyEnv ?? "OPENROUTER_API_KEY"]
                + (escalateBelow.map { ["--decision-escalate-below", String($0)] } ?? [])
        default: return []
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
        if let decider = cascade.last, decider != answer, let source = local(decider) {
            workloads.append(Workload(name: decider, role: "decide", unit: sizes.onNeuralEngine(source) ? .ane : .gpu, kind: .encoder, residentBytes: bytes(decider)))
        }
        return (workloads, unknown)
    }

    public func plan(_ sizes: Sizes, budgetBytes: Int64, power: PowerState = .desktop) -> (verdict: PlanVerdict, unknown: [String]) {
        let (workloads, unknown) = workloads(sizes)
        return (ComputePlan.admit(workloads, budgetBytes: budgetBytes, power: power, memory: effectiveMemory), unknown)
    }

    public var cloudLines: [String] {
        models.sorted { $0.key < $1.key }.compactMap { name, model in
            guard case .cloud(let provider, let id) = model.source else { return nil }
            return "  \(name) cloud:\(provider)/\(id): no local memory; key from $\(model.apiKeyEnv ?? "?"); documents and transcripts stay local"
        }
    }

    public var onDemandLines: [String] {
        [ocr, speech].compactMap { $0 }.filter { models[$0]?.keep == .onDemand }.map {
            "  \($0) on demand: loaded by the first request that needs it, unloaded after \(models[$0]?.idleUnload ?? Self.defaultIdleUnload) s idle; counted as resident above"
        }
    }
}
