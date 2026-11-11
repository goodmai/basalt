import Foundation
import BasaltCore
import BasaltServer

extension ComposeHost {
    func switchChat(_ diff: ComposeDiff, _ next: ComposeSpec) async throws {
        guard diff.chat != nil else { return }
        try await chat.loadModel(path: try await services.resolve(next.local(next.answer) ?? next.answer))
    }

    func ensureSelfDecider(_ spec: ComposeSpec) async throws {
        guard selfDecider == nil, spec.views.contains(where: { $0.spec.cascade.last == spec.answer }) else { return }
        selfDecider = try await services.decider("self")
    }

    func localDecider(_ view: ComposeSpec, of spec: ComposeSpec, held: [String: ComposeHeld]) -> (any DecisionEngine)? {
        guard let name = view.cascade.last else { return nil }
        if name == spec.answer { return selfDecider }
        guard case .decision(let engine)? = held[name] else { return nil }
        return engine
    }

    func restoreChat(_ path: String) async {
        let (loaded, current) = (await chat.isLoaded, await chat.modelPath)
        guard !loaded || current != path else { return }
        try? await chat.loadModel(path: path)
    }
}
