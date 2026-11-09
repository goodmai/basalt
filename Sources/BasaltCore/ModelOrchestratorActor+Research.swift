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

    public func preemptibly<T: Sendable>(
        _ body: @Sendable (any ModelLab, StepLease) async throws(BasaltError) -> T
    ) async throws(BasaltError) -> T {
        guard await engine.isLoaded else { throw .modelNotLoaded }
        guard let lab = engine as? any ModelLab else {
            throw .invalidRequestStructure(details: "the inference engine of this server has no lab tools")
        }
        let gate = gate
        return try await body(lab, StepLease(enter: { () async throws(BasaltError) -> Int in try await self.stepTicket() }, leave: { await gate.release($0) }))
    }

    func stepTicket() async throws(BasaltError) -> Int {
        while await !gate.idle(within: .seconds(1)) {
            guard !Task.isCancelled else { throw .cancelled }
        }
        return try await admitted(.background, adapter: nil)
    }
}
