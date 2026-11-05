import Foundation

extension ModelOrchestratorActor {
    public var adapterNames: [String] {
        get async { await engine.adapterNames }
    }

    public func loadAdapters(_ directories: [String: URL]) async throws(BasaltError) {
        let engine = engine
        do {
            try await gate.exclusively { try await engine.loadAdapters(directories) }
        } catch {
            throw Self.basaltError(error, "Adapter load error")
        }
    }

    func admitted(_ request: GenerationRequest) async throws(BasaltError) -> Int {
        let names = await engine.adapterNames
        if let name = request.adapter, !names.contains(name) {
            let served = names.isEmpty ? "none (serve --adapter name=directory adds one)" : names.joined(separator: ", ")
            throw .invalidRequestStructure(details: "unknown adapter \(name.prefix(64)); this server serves \(served)")
        }
        return try await admitted(request.priority, adapter: request.adapter)
    }

    func admitted(_ priority: RequestPriority, adapter: String?) async throws(BasaltError) -> Int {
        let ticket = try await admitted(priority)
        guard await engine.activeAdapter != adapter else { return ticket }
        await gate.release(ticket)
        let engine = engine
        do {
            return try await gate.admit(after: { try await engine.useAdapter(adapter) })
        } catch {
            throw Self.basaltError(error, "Adapter switch error")
        }
    }
}
