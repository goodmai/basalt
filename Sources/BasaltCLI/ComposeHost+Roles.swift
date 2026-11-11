import Foundation
import BasaltCore
import BasaltServer

extension ComposeHost {
    func swapRoles(_ diff: ComposeDiff, _ next: ComposeSpec) async throws -> (held: [String: ComposeHeld], decider: (any DecisionEngine)?) {
        if diff.chat != nil { try await chat.loadModel(path: try await services.resolve(next.local(next.answer) ?? next.answer)) }
        var decider = self.decider
        if diff.decider != nil {
            decider = nil
            if let name = next.decider { decider = try await services.decider(name == "self" ? name : next.local(name) ?? name) }
        }
        var held: [String: ComposeHeld] = [next.answer: .resident(chat)]
        if let name = next.deciderModel, let decider { held[name] = .decision(decider) }
        return (held, decider)
    }

    func restoreChat(_ path: String) async {
        let (loaded, current) = (await chat.isLoaded, await chat.modelPath)
        guard !loaded || current != path else { return }
        try? await chat.loadModel(path: path)
    }
}
