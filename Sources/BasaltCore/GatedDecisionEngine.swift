import Foundation

public struct GatedDecisionEngine: DecisionEngine {
    public let inner: any DecisionEngine
    let gate: GenerationGate

    public var modelID: String { inner.modelID }

    public static func serving(_ engine: (any DecisionEngine)?, gate: GenerationGate) -> (any DecisionEngine)? {
        guard let engine else { return nil }
        return engine.runsBesideDecode ? engine : GatedDecisionEngine(inner: engine, gate: gate)
    }

    public func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        do {
            return try await gate.exclusively { [inner] in try await inner.decide(request) }
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .inferenceError("systemone: \(error.localizedDescription)")
        }
    }
}
