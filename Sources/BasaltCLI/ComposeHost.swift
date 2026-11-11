import Foundation
import ArgumentParser
import BasaltCore
import BasaltCloud
import BasaltServer

enum ComposeHeld: Sendable {
    case resident(ModelOrchestratorActor)
    case onDemand(OnDemandModel)
    case cloud(String)
    case decision(any DecisionEngine)
}

struct ComposeStart: Sendable {
    let chat: ModelOrchestratorActor
    let decider: (any DecisionEngine)?
    let decisions: (any DecisionEngine)?
    let cloudGate: CloudGate
    let startup: PipelineStages
    let besideDecode: Bool
    var admission = ComposeAdmission.machine
}

struct ComposeAdmission: Sendable {
    var sizes: ComposeSpec.Sizes
    var budget: @Sendable () async -> (bytes: Int64, power: PowerState)

    static let machine = ComposeAdmission(sizes: ComposeCommand.sizes) { (await SystemProfiler().detectResources().gpuMemory, PowerSource.current()) }
}

actor ComposeHost {
    let services: ComposeServices
    let chat: ModelOrchestratorActor
    let startDecider: (any DecisionEngine)?
    var selfDecider: (any DecisionEngine)?
    let cloudGate: CloudGate
    let startup: PipelineStages
    let besideDecode: Bool
    let admission: ComposeAdmission
    let registry: PipelineRegistry
    private(set) var spec: ComposeSpec
    private(set) var held: [String: ComposeHeld] = [:]
    private var applying = false

    init(services: ComposeServices, spec: ComposeSpec, start: ComposeStart) {
        (self.services, self.spec, self.chat, self.cloudGate) = (services, spec, start.chat, start.cloudGate)
        (startDecider, selfDecider) = spec.decider == "self" ? (nil, start.decider) : (start.decider, nil)
        (self.startup, self.besideDecode, self.admission) = (start.startup, start.besideDecode, start.admission)
        registry = PipelineRegistry([:])
    }

    func start(defaultDecisions: (any DecisionEngine)?) async throws {
        held = startupHeld()
        for name in spec.models.keys.sorted() where held[name] == nil { held[name] = try await load(name, in: spec) }
        try await ensureSelfDecider(spec)
        await registry.replace(try await systems(for: spec, held: held, defaultDecisions: defaultDecisions))
    }

    func startupHeld() -> [String: ComposeHeld] {
        var held: [String: ComposeHeld] = [spec.answer: .resident(chat)]
        if let name = spec.deciderModel, let startDecider { held[name] = .decision(startDecider) }
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
            if spec.deciderModels.contains(name) { return .decision(try await services.decider(source)) }
            let path = try await services.resolve(source)
            if spec.views.contains(where: { $0.spec.ocr == name }), let refusal = ServeCommand.ocrRefusal(source, path: path) { throw BasaltError.invalidRequestStructure(details: refusal) }
            if spec.views.contains(where: { $0.spec.speech == name }), let refusal = ServeCommand.audioRefusal(source, path: path) { throw BasaltError.invalidRequestStructure(details: refusal) }
            let onDemand = model.keep == .onDemand
            guard let orchestrator = try await services.load(source, path, name, onDemand) else { throw ExitCode.failure }
            return onDemand ? .onDemand(OnDemandModel(model: orchestrator, path: path, idle: .seconds(model.idleUnload ?? ComposeSpec.defaultIdleUnload))) : .resident(orchestrator)
        }
    }

    func apply(_ request: ComposeApply) async throws(BasaltError) -> ComposeDiff {
        guard !applying else { throw .serverBusy(retryAfterSeconds: 5) }
        applying = true
        defer { applying = false }
        let next = try Self.parse(request)
        let diff = ComposeDiff.between(spec, next)
        if let restart = diff.restart { throw .invalidRequestStructure(details: "compose apply: \(restart)") }
        let order = try await admit(diff, next)
        let replaced = diff.removed + diff.reloaded
        let chatPath = await chat.modelPath
        if order == .unloadFirst {
            for name in replaced { await Self.unload(held[name]) }
            if diff.chat != nil { await chat.unloadModel() }
        }
        var fresh: [String: ComposeHeld] = [:]
        do {
            for name in diff.added + diff.reloaded { fresh[name] = try await load(name, in: next) }
            try await switchChat(diff, next)
            try await ensureSelfDecider(next)
            let nextHeld = held.filter { diff.kept.contains($0.key) }.merging(fresh) { $1 }.merging([next.answer: .resident(chat)]) { $1 }
            let built = try await systems(for: next, held: nextHeld, defaultDecisions: nil)
            let previous = held
            (held, spec) = (nextHeld, next)
            await registry.replace(built)
            await cloudGate.update(limits: services.cloudLimits(next), rekeyed: Set(diff.rekeyed.compactMap { Self.cloudID(next.models[$0]) }))
            if order == .loadFirst { for name in replaced { await Self.unload(previous[name]) } }
        } catch {
            for model in fresh.values { await Self.unload(model) }
            if diff.chat != nil { await restoreChat(chatPath) }
            if order == .unloadFirst { await restore(replaced) }
            throw error as? BasaltError ?? .invalidRequestStructure(details: "compose apply: \(error.localizedDescription)")
        }
        return diff
    }

    static func parse(_ request: ComposeApply) throws(BasaltError) -> ComposeSpec {
        var context = ComposeContext.machine
        (context.base, context.variables, context.profiles) = (request.base.map { URL(fileURLWithPath: $0) }, request.variables ?? [:], request.profiles)
        do {
            return try ComposeFile.parse(request.yaml, file: "compose apply", context: context)
        } catch {
            throw .invalidRequestStructure(details: error.description)
        }
    }

    func admit(_ diff: ComposeDiff, _ next: ComposeSpec) async throws(BasaltError) -> ComposeDiff.Order {
        let (bytes, power) = await admission.budget()
        switch diff.admission(previous: spec, next: next, sizes: admission.sizes, budgetBytes: bytes, power: power) {
        case .success(let order): return order
        case .failure(let refusal): throw .invalidRequestStructure(details: "compose apply: the new system does not fit, nothing changed: \(refusal.reason)")
        }
    }

    func restore(_ names: [String]) async {
        for name in names { if let model = try? await load(name, in: spec) { held[name] = model } }
        if let built = try? await systems(for: spec, held: held, defaultDecisions: nil) { await registry.replace(built) }
    }

    static func cloudID(_ model: ComposeSpec.Model?) -> String? {
        guard case .cloud(_, let id)? = model?.source else { return nil }
        return id
    }

    static func unload(_ held: ComposeHeld?) async {
        switch held {
        case .resident(let orchestrator): await orchestrator.unloadModel()
        case .onDemand(let model): await model.model.unloadModel()
        case .cloud, .decision, nil: break
        }
    }
}
