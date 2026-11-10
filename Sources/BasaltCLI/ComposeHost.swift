import Foundation
import ArgumentParser
import BasaltCore
import BasaltCloud
import BasaltServer

enum ComposeHeld: Sendable {
    case resident(ModelOrchestratorActor)
    case onDemand(OnDemandModel)
    case cloud(String)
}

struct ComposeStart: Sendable {
    let chat: ModelOrchestratorActor
    let decider: (any DecisionEngine)?
    let decisions: (any DecisionEngine)?
    let cloudGate: CloudGate
    let startup: PipelineStages
    let besideDecode: Bool
}

actor ComposeHost {
    let serve: ServeCommand
    let chat: ModelOrchestratorActor
    let decider: (any DecisionEngine)?
    let cloudGate: CloudGate
    let startup: PipelineStages
    let besideDecode: Bool
    let registry: PipelineRegistry
    private(set) var spec: ComposeSpec
    private(set) var held: [String: ComposeHeld] = [:]
    private var applying = false

    init(serve: ServeCommand, spec: ComposeSpec, start: ComposeStart) {
        (self.serve, self.spec, self.chat, self.decider, self.cloudGate) = (serve, spec, start.chat, start.decider, start.cloudGate)
        (self.startup, self.besideDecode) = (start.startup, start.besideDecode)
        registry = PipelineRegistry([:])
    }

    func start(defaultDecisions: (any DecisionEngine)?) async throws {
        held = startupHeld()
        for name in spec.models.keys.sorted() where held[name] == nil { held[name] = try await load(name, in: spec) }
        await registry.replace(try await systems(for: spec, held: held, defaultDecisions: defaultDecisions))
    }

    func startupHeld() -> [String: ComposeHeld] {
        var held: [String: ComposeHeld] = [spec.answer: .resident(chat)]
        for (name, perceiver, orchestrator) in [(spec.ocr, PipelineCore.Perceiver.ocrModel, startup.ocr), (spec.speech, .speechModel, startup.audio)] {
            guard let name, let orchestrator else { continue }
            held[name] = startup.onDemand[perceiver].map(ComposeHeld.onDemand) ?? .resident(orchestrator)
        }
        return held
    }

    func load(_ name: String, in spec: ComposeSpec) async throws -> ComposeHeld {
        guard let model = spec.models[name] else { throw ExitCode.failure }
        switch model.source {
        case .cloud(_, let id): return .cloud(id)
        case .local(let source):
            let path = try await serve.resolvedPath(source, quant: nil)
            if spec.views.contains(where: { $0.spec.ocr == name }), let refusal = ServeCommand.ocrRefusal(source, path: path) { throw BasaltError.invalidRequestStructure(details: refusal) }
            if spec.views.contains(where: { $0.spec.speech == name }), let refusal = ServeCommand.audioRefusal(source, path: path) { throw BasaltError.invalidRequestStructure(details: refusal) }
            let onDemand = model.keep == .onDemand
            guard let orchestrator = try await serve.loadPerceiver(source, path: path, role: name, loads: !onDemand, onDemand: onDemand) else { throw ExitCode.failure }
            return onDemand ? .onDemand(OnDemandModel(model: orchestrator, path: path, idle: .seconds(model.idleUnload ?? ComposeSpec.defaultIdleUnload))) : .resident(orchestrator)
        }
    }

    func apply(_ yaml: String) async throws(BasaltError) -> ComposeDiff {
        guard !applying else { throw .serverBusy(retryAfterSeconds: 5) }
        applying = true
        defer { applying = false }
        let next: ComposeSpec
        do {
            next = try ComposeFile.parse(yaml, file: "compose apply")
        } catch {
            throw .invalidRequestStructure(details: error.description)
        }
        let diff = ComposeDiff.between(spec, next)
        if let restart = diff.restart { throw .invalidRequestStructure(details: "compose apply: \(restart)") }
        var fresh: [String: ComposeHeld] = [:]
        do {
            for name in diff.added + diff.reloaded { fresh[name] = try await load(name, in: next) }
            let nextHeld = held.filter { diff.kept.contains($0.key) }.merging(fresh) { $1 }
            let built = try await systems(for: next, held: nextHeld, defaultDecisions: nil)
            let previous = held
            (held, spec) = (nextHeld, next)
            await registry.replace(built)
            for name in diff.removed + diff.reloaded { await Self.unload(previous[name]) }
        } catch {
            for model in fresh.values { await Self.unload(model) }
            throw error as? BasaltError ?? .invalidRequestStructure(details: "compose apply: \(error.localizedDescription)")
        }
        return diff
    }

    static func unload(_ held: ComposeHeld?) async {
        switch held {
        case .resident(let orchestrator): await orchestrator.unloadModel()
        case .onDemand(let model): await model.model.unloadModel()
        case .cloud, nil: break
        }
    }
}
