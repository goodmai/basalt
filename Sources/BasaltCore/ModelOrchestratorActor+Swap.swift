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
        guard unloads else { return (try await body(), .zero) }
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
        let reload: Result<Void, BasaltError> = await withTaskCancellationShield {
            do throws(BasaltError) {
                try await performLoad(path: model.path, modelId: model.modelId)
                return .success(())
            } catch {
                return .failure(error)
            }
        }
        try reload.get()
        return (try result.get(), unloaded + (clock.now - reloading))
    }
}
