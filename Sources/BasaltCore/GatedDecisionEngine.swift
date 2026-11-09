import Foundation

public struct GatedDecisionEngine: DecisionEngine {
    public let inner: any DecisionEngine
    let gate: GenerationGate

    public var modelID: String { inner.modelID }

    public static func serving(_ engine: (any DecisionEngine)?, gate: GenerationGate, besideDecode: Bool = true) -> (any DecisionEngine)? {
        guard let engine else { return nil }
        guard !(engine is GatedDecisionEngine), !(besideDecode && engine.runsBesideDecode) else { return engine }
        return GatedDecisionEngine(inner: engine, gate: gate)
    }

    public func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        do {
            return try await gate.boundedExclusively { [inner] in try await inner.decide(request) }
        } catch let error as BasaltError {
            throw error
        } catch is CancellationError {
            throw .inferenceError("systemone: the request was cancelled while waiting for the generation gate")
        } catch {
            throw .inferenceError("systemone: \(error.localizedDescription)")
        }
    }
}
