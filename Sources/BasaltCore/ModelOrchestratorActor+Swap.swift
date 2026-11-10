import Foundation

extension ModelOrchestratorActor {
    public func unloadModel() async {
        await gate.exclusively { _ = await self.performUnload() }
    }

    public func exclusively<T: Sendable>(
        unloadingModel unloads: Bool, _ body: @escaping @Sendable () async throws(BasaltError) -> T
    ) async throws(BasaltError) -> (value: T, swap: Duration) {
        do {
            return try await gate.boundedExclusively { try await self.swapped(unloads, body) }
        } catch {
            throw Self.basaltError(error, "Model swap error")
        }
    }

    func swapped<T: Sendable>(_ unloads: Bool, _ body: @Sendable () async throws(BasaltError) -> T) async throws(BasaltError) -> (value: T, swap: Duration) {
        let loaded = await engine.isLoaded
        guard unloads, loaded else {
            if loaded { await engine.releaseCaches() }
            return (try await body(), .zero)
        }
        let clock = ContinuousClock()
        let unloading = clock.now
        let model = await performUnload()
        let unloaded = clock.now - unloading
        let result: Result<T, BasaltError>
        do throws(BasaltError) {
            result = .success(try await body())
        } catch {
            result = .failure(error)
        }
        let reloading = clock.now
        try await withTaskCancellationShield { await reload(model) }.get()
        return (try result.get(), unloaded + (clock.now - reloading))
    }

    func reload(_ model: (path: String, modelId: String), attempts: Int = 2) async -> Result<Void, BasaltError> {
        var failure = BasaltError.modelNotLoaded
        for attempt in 1 ... attempts {
            do throws(BasaltError) {
                try await performLoad(path: model.path, modelId: model.modelId)
                return .success(())
            } catch {
                failure = error
                logger.error("Reloading \(model.modelId.isEmpty ? model.path : model.modelId) after a pipeline swap failed (attempt \(attempt) of \(attempts)): \(error.localizedDescription)")
            }
        }
        logger.error("The chat model is not loaded: every request fails until it is loaded again with POST /v1/models/load or a restart")
        return .failure(failure)
    }
}
