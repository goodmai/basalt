import Foundation
import BasaltCore

extension ComposeHost {
    func registerComposeAdapters(
        for spec: ComposeSpec,
        held: [String: ComposeHeld],
        home: URL = FileManager.default.homeDirectoryForCurrentUser
    ) async throws {
        guard !spec.adapters.isEmpty else { return }
        for (modelName, heldEntry) in held {
            let directories = spec.adapters(forBase: modelName, home: home)
            guard !directories.isEmpty else { continue }
            switch heldEntry {
            case .resident(let orchestrator):
                try await orchestrator.loadAdapters(directories)
            case .onDemand(let onDemand):
                if await onDemand.model.isLoaded {
                    try await onDemand.model.loadAdapters(directories)
                }
            case .cloud, .decision, .onDemandDecision, .speaker, .onDemandSpeaker:
                break
            }
        }
    }
}
