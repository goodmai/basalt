import Foundation
import ServiceLifecycle
import BasaltCore

struct GenerationDrainService: Service {
    let orchestrator: ModelOrchestratorActor
    let deadline: Duration

    func run() async throws {
        try? await gracefulShutdown()
        await orchestrator.shutdown(deadline: deadline)
    }
}
