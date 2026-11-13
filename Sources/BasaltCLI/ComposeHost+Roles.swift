import Foundation
import BasaltCore
import BasaltInference
import BasaltServer

extension ComposeHost {
    func loadDecider(_ name: String, source: String, model: ComposeSpec.Model) async throws -> ComposeHeld {
        let onDemand = model.keep == .onDemand
        await events.emit(onDemand ? "registered" : "loading", model: name, source)
        guard onDemand else {
            let loaded = try await services.decider(source)
            await events.emit("ready", model: name, source)
            return .decision(loaded)
        }
        let path = try await services.resolve(source)
        let ane = CoreAIDecisionEngine.isBundle(URL(fileURLWithPath: path))
        let decider = services.decider
        return .onDemandDecision(OnDemandDecision(
            modelID: source, path: path, idle: .seconds(model.idleUnload ?? ComposeSpec.defaultIdleUnload),
            runsBesideDecode: ane, loader: { try await decider(source) }))
    }
    func switchChat(_ diff: ComposeDiff, _ next: ComposeSpec) async throws {
        guard diff.chat != nil else { return }
        try await chat.loadModel(path: try await services.resolve(next.local(next.answer) ?? next.answer))
    }

    func ensureSelfDecider(_ spec: ComposeSpec) async throws {
        guard selfDecider == nil, spec.views.contains(where: { $0.spec.cascade.contains(spec.answer) }) else { return }
        selfDecider = try await services.decider("self")
    }

    func localDeciders(_ view: ComposeSpec, of spec: ComposeSpec, held: [String: ComposeHeld]) -> [String: any DecisionEngine] {
        view.cascade.reduce(into: [:]) { deciders, name in
            if name == spec.answer, let selfDecider { deciders[name] = selfDecider }
            if case .decision(let engine)? = held[name] { deciders[name] = engine }
            if case .onDemandDecision(let engine)? = held[name] { deciders[name] = engine }
        }
    }

    func restoreChat(_ path: String) async {
        let (loaded, current) = (await chat.isLoaded, await chat.modelPath)
        guard !loaded || current != path else { return }
        try? await chat.loadModel(path: path)
    }
}
