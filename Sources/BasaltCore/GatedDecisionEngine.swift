import Foundation

public struct GatedDecisionEngine: DecisionEngine {
    public let inner: any DecisionEngine
    let gate: GenerationGate

    public var modelID: String { inner.modelID }
    public var acceptsMedia: Bool { inner.acceptsMedia }
    public var sharesPerceived: Bool { inner.sharesPerceived }
    public var onDevice: (any DecisionEngine)? { inner.onDevice.map { GatedDecisionEngine(inner: $0, gate: gate) } }

    public static func serving(_ engine: (any DecisionEngine)?, gate: GenerationGate, besideDecode: Bool = true) -> (any DecisionEngine)? {
        guard let engine else { return nil }
        guard !(engine is GatedDecisionEngine), !(engine is CascadeDecisionEngine), !(besideDecode && engine.runsBesideDecode) else { return engine }
        return GatedDecisionEngine(inner: engine, gate: gate)
    }

    public func unload() async {
        await inner.unload()
    }

    public func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        do {
            return try await gate.boundedExclusively { [inner] in try await inner.decide(request) }
        } catch let error as BasaltError {
            throw error
        } catch is CancellationError {
            throw .cancelled
        } catch {
            throw .inferenceError("systemone: \(error.localizedDescription)")
        }
    }
}
