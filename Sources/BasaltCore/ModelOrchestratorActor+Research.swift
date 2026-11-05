import Foundation

extension ModelOrchestratorActor {
    public func research<T: Sendable>(
        _ body: @Sendable (any ModelResearch) async throws(BasaltError) -> T
    ) async throws(BasaltError) -> T {
        guard await engine.isLoaded else { throw .modelNotLoaded }
        guard let model = engine as? any ModelResearch else {
            throw .invalidRequestStructure(details: "the inference engine of this server has no research tools")
        }
        let ticket = try await admitted(.background, adapter: nil)
        defer {
            await withTaskCancellationShield {
                await gate.release(ticket)
            }
        }
        return try await body(model)
    }
}
